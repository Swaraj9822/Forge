"""In-house codebase search engine (the ``search`` tool).

The search tool provides two modes the Model uses to locate code:

* **Content mode** - compile a regular expression (``pattern``) with :mod:`re`
  and walk the Workspace yielding every matching line as a
  ``(path, 1-based line number, matching line)`` triple. The number of matches
  is bounded by ``Config.search_result_limit`` (default 100) and each matching
  line is capped at ``Config.search_line_cap`` characters (default 500), each
  with a truncation flag.
* **Glob mode** - return the set of Workspace-relative paths matching a
  file-name ``glob`` pattern using :mod:`pathlib` globbing semantics.

The engine is self-contained (no external search binary, per the requirements'
"built-in regular-expression and glob engine" assumption). It stays within the
Workspace boundary by rooting the walk/glob at ``ctx.workspace_root`` and never
following directory symlinks (which could escape the root). Binary or otherwise
undecodable files are skipped gracefully, and a small set of noise directories
(``.git``, ``node_modules``, ``__pycache__``, ``.venv``) are pruned from the
content walk for practicality.

Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6 (Properties 13, 14, 15).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from forge.tools.base import ToolContext, ToolResult

__all__ = ["SearchTool"]

# Default caps applied when ``ctx.config`` does not supply them. These mirror
# the documented defaults on :class:`forge.config.Config`.
_DEFAULT_RESULT_LIMIT = 100
_DEFAULT_LINE_CAP = 500

# Directories pruned from the content walk for practicality. Kept intentionally
# small; this is a convenience filter, not a security boundary.
_NOISE_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv"})


class SearchTool:
    """Codebase search by content (regex) or by file-name glob.

    Mode selection: an explicit ``mode`` argument (``"content"`` or ``"glob"``)
    takes precedence; otherwise the mode is inferred from which of ``pattern``
    or ``glob`` is supplied.
    """

    name = "search"
    description = (
        "Search the workspace by content or by file name. Provide 'pattern' "
        "(a regular expression) to search file contents and get matching "
        "paths with 1-based line numbers and lines, or provide 'glob' (a "
        "file-name glob like '**/*.py') to list matching file paths. Results "
        "are scoped to the workspace; content results are capped at 100 "
        "matches and each line at 500 characters."
    )
    read_only = True
    parameters: dict = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["content", "glob"],
                "description": (
                    "Search mode. Optional; inferred from 'pattern'/'glob' "
                    "when omitted."
                ),
            },
            "pattern": {
                "type": "string",
                "description": "Regular expression to match against file contents.",
            },
            "glob": {
                "type": "string",
                "description": "File-name glob pattern (pathlib semantics).",
            },
        },
    }

    # -- validation ----------------------------------------------------------

    def validate(self, args: dict) -> str | None:
        """Return ``None`` when ``args`` select a usable mode, else an error.

        Validation is limited to argument shape (a mode can be determined and
        the required argument for that mode is present). An invalid regular
        expression is *not* a validation error: it is reported at run time as
        an "invalid pattern" result per Requirement 8.5.
        """
        mode = args.get("mode")
        pattern = args.get("pattern")
        glob = args.get("glob")

        if mode is not None and mode not in ("content", "glob"):
            return f"Unknown search mode '{mode}'; expected 'content' or 'glob'."

        if mode == "content":
            if not _is_nonempty_str(pattern):
                return "Content search requires a non-empty 'pattern' string."
            return None
        if mode == "glob":
            if not _is_nonempty_str(glob):
                return "Glob search requires a non-empty 'glob' string."
            return None

        # Mode not given: infer from which argument is present.
        has_pattern = _is_nonempty_str(pattern)
        has_glob = _is_nonempty_str(glob)
        if has_pattern and has_glob:
            return (
                "Provide exactly one of 'pattern' or 'glob' (or set 'mode' to "
                "disambiguate)."
            )
        if not has_pattern and not has_glob:
            return "Search requires either a 'pattern' or a 'glob' argument."
        return None

    # -- execution -----------------------------------------------------------

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Run a content or glob search rooted at the Workspace."""
        mode = self._resolve_mode(args)
        root = Path(os.path.realpath(ctx.workspace_root))
        if mode == "glob":
            return self._run_glob(str(args["glob"]), root)
        return self._run_content(str(args["pattern"]), root, ctx)

    @staticmethod
    def _resolve_mode(args: dict) -> str:
        mode = args.get("mode")
        if mode in ("content", "glob"):
            return mode
        return "glob" if _is_nonempty_str(args.get("glob")) else "content"

    # -- content mode --------------------------------------------------------

    def _run_content(self, pattern: str, root: Path, ctx: ToolContext) -> ToolResult:
        try:
            regex = re.compile(pattern)
        except re.error as err:
            return ToolResult(
                ok=False,
                content="",
                error="invalid pattern",
                meta={"invalid_pattern": True, "detail": str(err)},
            )

        result_limit = _config_int(ctx.config, "search_result_limit", _DEFAULT_RESULT_LIMIT)
        line_cap = _config_int(ctx.config, "search_line_cap", _DEFAULT_LINE_CAP)

        results: list[dict[str, Any]] = []
        truncated = False
        any_line_truncated = False

        for file_path in self._walk_files(root):
            if len(results) >= result_limit:
                truncated = True
                break

            text = _read_text(file_path)
            if text is None:  # binary / undecodable: skip gracefully
                continue

            rel = file_path.relative_to(root).as_posix()
            for line_no, line in enumerate(text.splitlines(), start=1):
                if not regex.search(line):
                    continue
                if len(results) >= result_limit:
                    truncated = True
                    break

                line_truncated = len(line) > line_cap
                if line_truncated:
                    line = line[:line_cap]
                    any_line_truncated = True

                results.append(
                    {
                        "path": rel,
                        "line": line_no,
                        "text": line,
                        "line_truncated": line_truncated,
                    }
                )
            if truncated:
                break

        if not results:
            return ToolResult(
                ok=True,
                content="No matches found.",
                meta={"mode": "content", "matches": 0, "results": [], "truncated": False},
            )

        rendered_lines = [f"{r['path']}:{r['line']}: {r['text']}" for r in results]
        if truncated:
            rendered_lines.append(
                f"... results truncated at {result_limit} matches."
            )
        content = "\n".join(rendered_lines)

        return ToolResult(
            ok=True,
            content=content,
            meta={
                "mode": "content",
                "matches": len(results),
                "results": results,
                "truncated": truncated,
                "line_truncated": any_line_truncated,
            },
        )

    def _walk_files(self, root: Path):
        """Yield files under ``root``, pruning noise dirs and not following
        directory symlinks (so the walk cannot escape the Workspace)."""
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            # Prune noise directories in place so os.walk skips them.
            dirnames[:] = [d for d in dirnames if d not in _NOISE_DIRS]
            for name in filenames:
                yield Path(dirpath) / name

    # -- glob mode -----------------------------------------------------------

    def _run_glob(self, pattern: str, root: Path) -> ToolResult:
        matches: list[str] = []
        for match in sorted(root.glob(pattern)):
            try:
                rel = match.relative_to(root).as_posix()
            except ValueError:  # escaped the root (shouldn't happen): skip
                continue
            matches.append(rel)

        if not matches:
            return ToolResult(
                ok=True,
                content="No matches found.",
                meta={"mode": "glob", "matches": 0, "results": []},
            )

        return ToolResult(
            ok=True,
            content="\n".join(matches),
            meta={"mode": "glob", "matches": len(matches), "results": matches},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value != ""


def _config_int(config: Any, attr: str, fallback: int) -> int:
    """Read an integer setting from ``config`` falling back to a default."""
    value = getattr(config, attr, None)
    if isinstance(value, int) and value > 0:
        return value
    return fallback


def _read_text(path: Path) -> str | None:
    """Return the UTF-8 decoded contents of ``path`` or ``None`` if it is not
    valid UTF-8 (binary) or cannot be read."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None
