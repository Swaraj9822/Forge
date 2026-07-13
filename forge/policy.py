"""Autonomy modes, the approval policy, and the safe shell-command matcher.

The policy is a pure decision layer: given an autonomy mode and a tool call it
decides whether the call is (a) forbidden outright, (b) auto-approved, or (c)
requires user approval. The actual prompting is delegated to an Approver so the
policy stays offline and property-testable.

Components
----------
* :class:`AutonomyMode` - the three autonomy levels (autopilot, supervised,
  readonly).
* :class:`Decision` - the outcome of an approval request.
* :class:`ShellMatcher` - the safe shell-command allowlist matcher (rejects any
  compound/metacharacter command).
* :class:`ApprovalPolicy` - pure classification: ``is_forbidden`` /
  ``requires_approval`` predicates given (name, args, read_only).
* :class:`Approver` - the protocol a user-facing prompt must implement.
* :class:`AutoApprover` / :class:`DenyMutationsApprover` - non-interactive
  approvers used by the autopilot and headless paths.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


__all__ = [
    "AutonomyMode",
    "Decision",
    "ShellMatcher",
    "ApprovalPolicy",
    "Approver",
    "AutoApprover",
    "DenyMutationsApprover",
]


class AutonomyMode(str, Enum):
    """The three autonomy levels Forge recognizes (Phase 2, Feature B).

    * ``AUTOPILOT`` - run everything, no prompts (today's behavior).
    * ``SUPERVISED`` - prompt before mutating tools.
    * ``READONLY`` - forbid mutating tools outright (no prompt).
    """

    AUTOPILOT = "autopilot"
    SUPERVISED = "supervised"
    READONLY = "readonly"


class Decision(str, Enum):
    """The outcome of an :class:`Approver` request.

    * ``APPROVE`` - the call may proceed this once.
    * ``DENY`` - the call is refused for this attempt.
    * ``APPROVE_ALWAYS`` - approve and remember the action for the session
      (the approver itself keeps the session-scoped set; the executor just
      records and proceeds).
    """

    APPROVE = "approve"
    DENY = "deny"
    APPROVE_ALWAYS = "approve_always"


# Shell metacharacters that make an allowlist match unsafe: any of these means
# the command is compound / redirected / substituted and must NOT be
# auto-approved on a bare argv[0] match.
_SHELL_METACHARS = (";", "&", "|", "`", "$(", ">", "<", "\n")


@dataclass(frozen=True)
class ShellMatcher:
    """Decides whether a shell command is safe to auto-approve via allowlist.

    A command is *allowlisted* iff it is a single, non-compound invocation whose
    program (argv[0]) appears in :attr:`allowlist`. Any shell metacharacter, an
    unparseable command, or an empty argv returns ``False`` - the approval
    policy then requires explicit user approval for the call.
    """

    allowlist: tuple[str, ...] = ()

    def is_allowlisted(self, command: str) -> bool:
        """Return ``True`` iff ``command`` is safe to auto-approve.

        Safety rules:

        * Any shell metacharacter (``;``, ``&``, ``|``, backtick, ``$(``, ``>``,
          ``<``, newline, ``&&``, ``||``) makes the command unsafe.
        * A command that does not parse (e.g. unterminated quotes) is unsafe.
        * An empty parsed argv is unsafe.
        * Otherwise the command's leading tokens must match an allowlist entry.
          An entry is matched by *token sequence*, not substring: a single-token
          entry (``"git"``) matches any invocation of that program (``git`` with
          any args), while a multi-token entry (``"git status"``) matches only
          when the command's leading tokens are exactly that sequence. This lets
          an allowlist be as broad (``"git"``) or as narrow (``"git status"``)
          as the operator wants, so a bare program need not implicitly approve
          its every destructive subcommand.
        """

        if any(tok in command for tok in _SHELL_METACHARS):
            return False
        # `&&` and `||` are not directly matched above; reject any `&&` or `||`
        # that is not a substring of one of the single-char metachars.
        if "&&" in command or "||" in command:
            return False
        try:
            argv = shlex.split(command, posix=True)
        except ValueError:
            return False
        if not argv:
            return False
        for entry in self.allowlist:
            try:
                entry_tokens = shlex.split(entry, posix=True)
            except ValueError:
                continue
            if not entry_tokens:
                continue
            if argv[: len(entry_tokens)] == entry_tokens:
                return True
        return False


# Git subcommands that are read-only (inspect history / working tree without
# modifying the repository). Anything NOT in this set — including any future
# addition such as ``push``/``reset``/``clean`` — is treated as mutating and
# therefore requires approval in non-autopilot modes. Classifying by an
# explicit read-only allowlist (rather than a mutating denylist) makes the
# policy fail *closed*: an unrecognized operation is gated, not waved through.
_GIT_READONLY = frozenset({"status", "diff", "log", "show", "branch"})

# Flags that write output to a file — a side effect even for an otherwise
# read-only git op (e.g. ``git diff --output=FILE`` / ``-o FILE``). Their
# presence means the invocation is not side-effect-free and must not be
# auto-classified read-only.
_GIT_FILE_OUTPUT_FLAGS = frozenset({"-o", "--output"})

# Read-only "listing" flags for ``git branch``. ``branch`` is side-effect-free
# only when every argument is one of these (or an ``=value`` form). A bare
# positional argument to ``branch`` names a branch to CREATE, and flags like
# ``-d``/``-D``/``-m``/``-M``/``-c``/``-f``/``--set-upstream-to`` delete, rename,
# copy, force, or re-point refs — all mutating. Anything outside this allowlist
# therefore makes ``branch`` mutating, so classification fails closed: an
# unrecognized or positional argument is gated, not waved through as read-only.
_GIT_BRANCH_READONLY_FLAGS = frozenset({
    "-a", "--all", "-r", "--remotes", "-v", "-vv", "--verbose",
    "-l", "--list", "--show-current", "--contains", "--no-contains",
    "--merged", "--no-merged", "--points-at", "--sort", "--format",
    "--color", "--no-color", "--column", "--no-column",
    "-i", "--ignore-case", "--abbrev", "--no-abbrev",
})


@dataclass(frozen=True)
class ApprovalPolicy:
    """Pure classification of a tool call against an :class:`AutonomyMode`.

    The policy is intentionally offline: it never prompts, never blocks, never
    performs I/O. The actual prompting is the :class:`Approver`'s job; this
    class just decides *whether* a prompt is needed and *whether* a call is
    forbidden outright.

    Two predicates are exposed (rather than a single ``classify`` that returns
    a tagged union): ``is_forbidden`` is checked first, ``requires_approval``
    second. Read-only tools and read-only git subcommands never require
    approval in any mode. Mutating tools require approval in ``SUPERVISED``
    mode and are outright forbidden in ``READONLY`` mode.
    """

    mode: AutonomyMode = AutonomyMode.AUTOPILOT
    shell: ShellMatcher = field(default_factory=ShellMatcher)

    def is_forbidden(self, name: str, args: dict, *, read_only: bool) -> bool:
        """Return ``True`` when the call must be refused without prompting.

        Only ``READONLY`` mode forbids anything; a read-only tool or a
        read-only git subcommand (``status``/``diff``/``log``/``show``/
        ``branch``) is always permitted regardless of mode.
        """

        if self.mode is AutonomyMode.READONLY:
            return not read_only and not self._is_git_readonly(name, args)
        return False

    def requires_approval(
        self, name: str, args: dict, *, read_only: bool
    ) -> bool:
        """Return ``True`` when the call needs an :class:`Approver` prompt.

        Autopilot never prompts. Read-only tools and read-only git subcommands
        never prompt. A shell command is auto-approved only when its program
        is in the configured allowlist; otherwise it needs approval. Every
        other mutating tool (write/edit/non-allowlisted-shell) needs approval
        in supervised mode.
        """

        if self.mode is AutonomyMode.AUTOPILOT:
            return False
        if read_only or self._is_git_readonly(name, args):
            return False
        if name == "shell":
            return not self.shell.is_allowlisted(str(args.get("command", "")))
        # write / edit / mutating git / unknown mutating tools:
        return True

    @staticmethod
    def _is_git_readonly(name: str, args: dict) -> bool:
        """Return ``True`` when ``name='git'`` and the call is side-effect-free.

        Gating is decided by an explicit read-only allowlist of operations
        (``status``/``diff``/``log``/``show``/``branch``) *and* an inspection of
        the operation's extra ``args``: a read-only op still counts as read-only
        only when its arguments introduce no side effects. Specifically:

        * an output-redirection flag (``-o``/``--output``) writes a file, so any
          op carrying one is treated as mutating; and
        * ``branch`` is read-only only when every argument is a known listing
          flag — a positional argument (creates a branch) or a mutating flag
          (``-d``/``-D``/``-m``/``-f``/``--set-upstream-to`` …) makes it mutating.

        Every other operation — the mutating ones (``add``/``commit``/
        ``checkout``/``stash``) and any unrecognized/future op (``push``/
        ``reset``/``clean``) — is treated as needing approval, so the policy
        fails closed rather than open.
        """

        if name != "git":
            return False
        op = args.get("operation")
        if not (isinstance(op, str) and op in _GIT_READONLY):
            return False
        extra = args.get("args")
        tokens = [t for t in extra if isinstance(t, str)] if isinstance(extra, list) else []
        return ApprovalPolicy._git_args_side_effect_free(op, tokens)

    @staticmethod
    def _git_args_side_effect_free(op: str, tokens: list[str]) -> bool:
        """Return ``True`` when ``tokens`` keep a read-only git ``op`` read-only.

        Fails closed: an output-redirection flag on any op, or (for ``branch``)
        any argument outside :data:`_GIT_BRANCH_READONLY_FLAGS`, marks the call
        as having side effects so it is no longer auto-classified read-only.
        """

        for tok in tokens:
            base = tok.split("=", 1)[0]
            if base in _GIT_FILE_OUTPUT_FLAGS:
                return False
        if op == "branch":
            for tok in tokens:
                base = tok.split("=", 1)[0]
                if base not in _GIT_BRANCH_READONLY_FLAGS:
                    return False
        return True


@runtime_checkable
class Approver(Protocol):
    """Asks the user (or a policy) to approve a gated tool call.

    A structural protocol: any object exposing a matching ``request`` method
    (the interactive :class:`~forge.repl.Repl`, :class:`AutoApprover`,
    :class:`DenyMutationsApprover`) satisfies it without explicit subclassing.
    Implementations must return one of the :class:`Decision` values. They may
    use the ``preview`` (a best-effort string the executor computes via the
    tool's ``preview`` hook) to help the user decide; the protocol itself
    does not interpret the preview.
    """

    def request(self, name: str, args: dict, preview: str | None) -> Decision:
        ...


class AutoApprover:
    """Approves every gated call (used to make the autopilot behavior explicit).

    Returning ``APPROVE`` here does not change behavior compared to omitting an
    approver entirely; the class exists so the headless and bootstrap paths
    can wire a real ``Approver`` implementation rather than threading
    ``None`` checks through the executor.
    """

    def request(self, name: str, args: dict, preview: str | None) -> Decision:
        return Decision.APPROVE


class DenyMutationsApprover:
    """Non-interactive approver that denies anything requiring approval.

    Used by headless runs unless ``--yes`` is passed. Combined with the
    policy this means a headless supervised run refuses mutations rather than
    blocking on a prompt that can never be answered. A read-only tool is
    never routed through this approver (the policy classifies it as not
    requiring approval), so it passes through unchanged.
    """

    def request(self, name: str, args: dict, preview: str | None) -> Decision:
        return Decision.DENY
