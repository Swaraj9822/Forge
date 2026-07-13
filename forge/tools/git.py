"""The git operations tool.

This module implements the built-in ``git`` tool (Requirement 9). The tool lets
the Model inspect history and manage changes by dispatching a *fixed* set of git
operations through the ``git`` binary, run inside the Workspace.

Supported operations (Req 9.1)
------------------------------
The tool dispatches **exactly** the operations enumerated in
:data:`SUPPORTED_OPERATIONS`::

    status, diff, log, show, add, commit, branch, checkout, stash

Any other operation name is rejected with an "unsupported" result and the
``git`` binary is never invoked for it (Req 9.4, Property 16). The supported set
is defined as a :class:`frozenset` so dispatch is a single membership test.

Execution model
---------------
Operations run as ``git <operation> [args...]`` via :mod:`subprocess` with a
*list* argv (never ``shell=True``), so caller-supplied extra ``args`` cannot be
interpreted by a shell. The child runs with ``cwd`` set to the Workspace root
(:attr:`ToolContext.workspace_root`) so it operates on the Workspace repository
(Req 9.2).

Behavioral contracts
--------------------
* **Not a repository (Req 9.3).** Before running the requested operation the
  tool runs ``git rev-parse --is-inside-work-tree``; when that reports the
  Workspace is not inside a work tree (or git's stderr names a missing
  repository) the tool returns a "not a git repository" result with
  ``meta={"not_a_repo": True}``.
* **Non-zero exit (Req 9.5).** A supported operation that exits non-zero yields
  an error result that includes the captured stderr in ``error`` and the exit
  code in ``meta={"exit_code": n}``.
* **Output cap (Req 9.6, Property 12).** Output is capped at
  ``ctx.config.output_cap_chars`` (falling back to 30,000) characters; when the
  output exceeds the cap it is truncated and ``meta["truncated"]`` is set.
* **Decoding.** stdout/stderr are decoded as UTF-8 with ``errors="replace"`` so
  binary-ish diff output never raises.
* **Missing git binary.** ``git`` is assumed to be on PATH (per the spec's
  assumptions); a :class:`FileNotFoundError` is nonetheless handled defensively
  and surfaced as an error result rather than propagating.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

from forge.tools.base import ToolContext, ToolResult

__all__ = ["GitTool", "SUPPORTED_OPERATIONS"]

# The exact set of git operations the tool dispatches (Req 9.1). Any name
# outside this set is rejected as unsupported (Req 9.4).
SUPPORTED_OPERATIONS: frozenset[str] = frozenset(
    {
        "status",
        "diff",
        "log",
        "show",
        "add",
        "commit",
        "branch",
        "checkout",
        "stash",
    }
)

# Fallback output cap when no config is available on the context (Req 9.6).
_DEFAULT_OUTPUT_CAP = 30_000


@dataclass
class GitTool:
    """The ``git`` tool implementing the :class:`Tool` protocol.

    Dispatches exactly :data:`SUPPORTED_OPERATIONS` through the ``git`` binary
    inside the Workspace and returns the command output as a
    :class:`ToolResult`.
    """

    name: str = "git"
    description: str = (
        "Run a git operation in the workspace repository. The 'operation' must "
        "be one of: status, diff, log, show, add, commit, branch, checkout, "
        "stash. Optional 'args' is a list of additional string arguments passed "
        "to the git subcommand (for example operation='log' with "
        "args=['--oneline', '-n', '5']). Returns the command output; on a "
        "non-zero exit the exit code and error output are included."
    )
    # Git is conservatively non-read-only here; the approval policy applies a
    # finer-grained per-operation rule (status/diff/log/show/branch are
    # treated as read-only; add/commit/checkout/stash require approval).
    read_only: bool = False
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": (
                        "The git operation to run. One of: status, diff, log, "
                        "show, add, commit, branch, checkout, stash."
                    ),
                    "enum": sorted(SUPPORTED_OPERATIONS),
                },
                "args": {
                    "type": "array",
                    "description": (
                        "Optional additional string arguments passed to the "
                        "git subcommand."
                    ),
                    "items": {"type": "string"},
                },
            },
            "required": ["operation"],
        }
    )

    # -- validation (shape/type checks only) ---------------------------------

    def validate(self, args: dict) -> str | None:
        """Validate argument *shape* only.

        Ensures ``operation`` is present and a string and, when ``args`` is
        supplied, that it is a list of strings. Whether the operation is one of
        the supported names is intentionally *not* checked here: an unsupported
        operation is reported by :meth:`run` as a descriptive "unsupported"
        :class:`ToolResult` (Req 9.4) rather than a validation failure.
        """
        if not isinstance(args, dict):
            return "Arguments must be an object."

        operation = args.get("operation")
        if operation is None:
            return "Missing required argument 'operation'."
        if not isinstance(operation, str):
            return "Argument 'operation' must be a string."

        extra = args.get("args")
        if extra is not None:
            if not isinstance(extra, list):
                return "Argument 'args' must be a list of strings."
            for index, item in enumerate(extra):
                if not isinstance(item, str):
                    return f"Argument 'args' item at index {index} must be a string."

        return None

    # -- execution -----------------------------------------------------------

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Dispatch the requested git operation and return a :class:`ToolResult`."""
        operation = args["operation"]
        extra_args = list(args.get("args") or [])

        # Unsupported operation -> never invoke git (Req 9.4, Property 16).
        if operation not in SUPPORTED_OPERATIONS:
            return ToolResult(
                ok=False,
                content="",
                error=f"unsupported operation '{operation}'",
                meta={"unsupported": True},
            )

        cwd = str(ctx.workspace_root)

        # Not-a-repo detection before running the operation (Req 9.3).
        repo_error = self._check_is_repo(cwd)
        if repo_error is not None:
            return repo_error

        # Run `git <operation> [args...]` with a list argv (no shell).
        try:
            completed = subprocess.run(
                ["git", operation, *extra_args],
                cwd=cwd,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            return ToolResult(
                ok=False,
                content="",
                error="git executable not found on PATH",
                meta={"git_not_found": True},
            )

        stdout = completed.stdout.decode("utf-8", errors="replace")
        stderr = completed.stderr.decode("utf-8", errors="replace")
        exit_code = completed.returncode

        # Non-zero exit -> error result with exit code + captured stderr (Req 9.5).
        if exit_code != 0:
            error_text = stderr.strip() or stdout.strip() or (
                f"git {operation} exited with code {exit_code}"
            )
            capped_error, _ = self._cap(error_text, ctx)
            return ToolResult(
                ok=False,
                content="",
                error=capped_error,
                meta={"exit_code": exit_code},
            )

        # Success: cap the combined output (Req 9.6).
        output = stdout
        if stderr:
            # Some git operations (e.g. push-like) write informational text to
            # stderr even on success; include it after stdout.
            output = output + stderr if output else stderr

        capped, truncated = self._cap(output, ctx)
        meta: dict = {"exit_code": exit_code}
        if truncated:
            meta["truncated"] = True
        return ToolResult(ok=True, content=capped, error=None, meta=meta)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _check_is_repo(cwd: str) -> ToolResult | None:
        """Return a "not a git repository" result, or ``None`` when inside one.

        Runs ``git rev-parse --is-inside-work-tree``. A non-zero exit (git
        prints ``not a git repository`` to stderr) or a non-``true`` answer
        means the Workspace is not a git repository (Req 9.3).
        """
        try:
            probe = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=cwd,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            return ToolResult(
                ok=False,
                content="",
                error="git executable not found on PATH",
                meta={"git_not_found": True},
            )

        answer = probe.stdout.decode("utf-8", errors="replace").strip().lower()
        if probe.returncode != 0 or answer != "true":
            return ToolResult(
                ok=False,
                content="",
                error="not a git repository",
                meta={"not_a_repo": True},
            )
        return None

    @staticmethod
    def _cap(text: str, ctx: ToolContext) -> tuple[str, bool]:
        """Cap ``text`` to the configured output limit (Req 9.6, Property 12).

        Returns the (possibly truncated) text and whether truncation occurred.
        """
        cap = _DEFAULT_OUTPUT_CAP
        config = getattr(ctx, "config", None)
        config_cap = getattr(config, "output_cap_chars", None)
        if isinstance(config_cap, int) and config_cap >= 0:
            cap = config_cap

        if len(text) > cap:
            return text[:cap], True
        return text, False
