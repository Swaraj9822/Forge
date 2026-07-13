"""Filesystem tools: ``read``, ``write``, and ``edit``.

This module hosts the workspace-scoped filesystem tools. It contains the
:class:`ReadTool` (task 7.1), :class:`WriteTool` (task 8.1), and
:class:`EditTool` (task 9.1) which share the
:class:`~forge.tools.base.Tool` protocol and the
:func:`~forge.tools.paths.resolve_in_workspace` path-scoping helper.

ReadTool (Requirement 5)
------------------------
The read tool returns the UTF-8 text contents of a workspace file. It supports
an optional inclusive 1-based line range and enforces the documented read cap
(2,000 lines or 1 MB). Its behavior, by acceptance criterion:

* 5.1 - return the file contents when the path exists and decodes as UTF-8.
* 5.2 - when a line range is supplied, return only ``start``..``end`` inclusive.
* 5.3 - a non-existent path (or a non-file) yields a not-found result.
* 5.4 - a path resolving outside the Workspace yields an out-of-scope result
  (delegated to :func:`resolve_in_workspace`); no read happens outside the
  Workspace.
* 5.5 - an invalid line range (start < 1, end beyond the last line, or
  start > end) yields an invalid-range result.
* 5.6 - a file that is not valid UTF-8 (decode failure or NUL bytes present)
  yields a binary result that *excludes* the raw contents.
* 5.7 - contents exceeding the configured cap are truncated to the limit and
  flagged ``truncated`` in ``meta``.

Line-range semantics: Requirement 5.2 vs 5.5
--------------------------------------------
The design narrative describes a range as "bounded from line 1 to the last line
of the file", which on its own could be read as silently clamping an
out-of-range end to the last line. Requirement 5.5, however, is explicit that a
range whose ``end`` exceeds the last line is *invalid*. This implementation
follows Requirement 5.5 (and Property 8): an ``end_line`` greater than the last
line of the file is rejected as an invalid range rather than clamped. The only
"bounding" applied is supplying sensible defaults when one endpoint is omitted
(``start`` defaults to 1, ``end`` defaults to the last line); any explicitly
supplied endpoint outside ``[1, last_line]`` (with ``start <= end``) is invalid.

EditTool — robust edit modes (Feature I)
----------------------------------------
The edit tool supports three modes:

1. **replace** (default): Replace the unique target string. Zero occurrences
   returns "target not found"; more than one returns "ambiguous".
2. **anchored**: Replace the target that appears between ``after`` and ``before``
   anchors, disambiguating otherwise ambiguous targets.
3. **line_range**: Replace lines ``start_line``..``end_line`` (1-based inclusive)
   with the replacement text, no target matching needed.
"""

from __future__ import annotations

import difflib
import os
import tempfile
from pathlib import Path
from typing import Any

from forge.tools.base import Tool, ToolContext, ToolResult
from forge.tools.paths import OutOfWorkspaceError, resolve_in_workspace

__all__ = ["ReadTool", "WriteTool", "EditTool"]

# Documented read-cap defaults, used when ``ctx.config`` does not supply them
# (e.g. config is ``None`` in tests or early wiring). See the requirements
# "Default Configuration Values" table and ``forge.config.Config``.
DEFAULT_READ_MAX_LINES = 2_000
DEFAULT_READ_MAX_BYTES = 1_000_000


def _read_limits(ctx: ToolContext) -> tuple[int, int]:
    """Resolve (max_lines, max_bytes) from ``ctx.config`` or fall back.

    ``ctx.config`` is loosely typed and may be ``None``; when present it is a
    :class:`forge.config.Config` carrying ``read_max_lines`` / ``read_max_bytes``.
    """

    config = getattr(ctx, "config", None)
    max_lines = getattr(config, "read_max_lines", None)
    max_bytes = getattr(config, "read_max_bytes", None)
    if not isinstance(max_lines, int) or max_lines <= 0:
        max_lines = DEFAULT_READ_MAX_LINES
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        max_bytes = DEFAULT_READ_MAX_BYTES
    return max_lines, max_bytes


def _is_int(value: Any) -> bool:
    """Return True for genuine integers, rejecting ``bool`` (an ``int`` subclass)."""

    return isinstance(value, int) and not isinstance(value, bool)


def _unified_diff(path: str, old: str, new: str) -> str:
    """Render ``old`` vs ``new`` as a unified diff string.

    The from/to labels mirror ``git diff`` (``a/<path>`` / ``b/<path>``) so
    previews look familiar in the terminal. When the diff is empty (no
    textual change) the literal string ``"(no textual change)"`` is returned
    so the caller can distinguish "no change" from "preview failed".
    """

    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    text = "".join(diff)
    return text or "(no textual change)"


def _cap_content(
    lines: list[str], max_lines: int, max_bytes: int
) -> tuple[str, bool]:
    """Cap ``lines`` to ``max_lines`` and the joined text to ``max_bytes``.

    Returns the (possibly truncated) content and a flag indicating whether any
    truncation occurred. Byte truncation is applied on a UTF-8 boundary so the
    returned text always decodes cleanly.
    """

    truncated = False

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True

    content = "".join(lines)
    encoded = content.encode("utf-8")
    if len(encoded) > max_bytes:
        # Truncate to the byte budget, then drop any partial trailing
        # multi-byte sequence so the result is valid UTF-8.
        content = encoded[:max_bytes].decode("utf-8", errors="ignore")
        truncated = True

    return content, truncated


class ReadTool:
    """The ``read`` tool: return the UTF-8 contents of a workspace file.

    Implements the :class:`~forge.tools.base.Tool` protocol.
    """

    name = "read"
    description = (
        "Read a UTF-8 text file within the workspace. Optionally returns only "
        "an inclusive 1-based line range via 'start_line' and 'end_line'. "
        "Output is capped at the configured maximum lines or bytes and flagged "
        "as truncated when the cap is reached."
    )
    read_only = True
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative or absolute path to the file to read.",
            },
            "start_line": {
                "type": "integer",
                "description": "Optional 1-based first line to return (inclusive).",
            },
            "end_line": {
                "type": "integer",
                "description": "Optional 1-based last line to return (inclusive).",
            },
        },
        "required": ["path"],
    }

    def validate(self, args: dict) -> str | None:
        """Type/shape validation only (Req 5 inputs).

        Ensures ``path`` is present and a string and that any supplied
        ``start_line``/``end_line`` are integers. Semantic range checks (start
        below 1, end beyond the last line, start greater than end) are deferred
        to :meth:`run` so they can return an invalid-range :class:`ToolResult`
        (Req 5.5).
        """

        if not isinstance(args, dict):
            return "Arguments must be an object."

        path = args.get("path")
        if path is None:
            return "Missing required argument 'path'."
        if not isinstance(path, str):
            return "Argument 'path' must be a string."

        start_line = args.get("start_line")
        if start_line is not None and not _is_int(start_line):
            return "Argument 'start_line' must be an integer."

        end_line = args.get("end_line")
        if end_line is not None and not _is_int(end_line):
            return "Argument 'end_line' must be an integer."

        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Read the file, applying scoping, decoding, range, and cap rules."""

        path_arg = args["path"]

        # 5.4 - reject paths that resolve outside the Workspace; no read occurs.
        try:
            resolved = resolve_in_workspace(path_arg, ctx.workspace_root)
        except OutOfWorkspaceError as exc:
            return ToolResult(
                ok=False,
                content="",
                error=f"Path is out of scope: {exc.candidate}",
                meta={"out_of_scope": True},
            )

        # 5.3 - the path must exist and be a regular file.
        if not resolved.is_file():
            return ToolResult(
                ok=False,
                content="",
                error=f"File not found: {path_arg}",
                meta={"not_found": True},
            )

        # Read raw bytes defensively (permission / IO errors -> error result).
        try:
            raw = resolved.read_bytes()
        except OSError as exc:
            return ToolResult(
                ok=False,
                content="",
                error=f"Could not read file '{path_arg}': {exc}",
                meta={"io_error": True},
            )

        # 5.6 - binary detection: NUL bytes or a failed UTF-8 decode. The raw
        # contents are deliberately excluded from the result.
        if b"\x00" in raw:
            return self._binary_result(path_arg)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return self._binary_result(path_arg)

        # Split preserving line endings so a slice reconstructs the original
        # bytes for the selected lines.
        lines = text.splitlines(keepends=True)
        last_line = len(lines)

        start_line = args.get("start_line")
        end_line = args.get("end_line")
        range_supplied = start_line is not None or end_line is not None

        if range_supplied:
            # Supply sensible defaults for an omitted endpoint, then validate.
            effective_start = start_line if start_line is not None else 1
            effective_end = end_line if end_line is not None else last_line

            # 5.5 - invalid range: start below 1, end beyond the last line, or
            # start greater than end.
            if (
                effective_start < 1
                or effective_end > last_line
                or effective_start > effective_end
            ):
                return ToolResult(
                    ok=False,
                    content="",
                    error=(
                        "Invalid line range "
                        f"[{effective_start}, {effective_end}] for a file with "
                        f"{last_line} line(s)."
                    ),
                    meta={"invalid_range": True},
                )

            # 5.2 - return exactly lines start..end inclusive.
            selected = lines[effective_start - 1 : effective_end]
        else:
            selected = lines

        # 5.7 - cap output at the configured maximum lines or bytes.
        max_lines, max_bytes = _read_limits(ctx)
        content, truncated = _cap_content(selected, max_lines, max_bytes)

        meta: dict = {}
        if truncated:
            meta["truncated"] = True

        return ToolResult(ok=True, content=content, error=None, meta=meta)

    @staticmethod
    def _binary_result(path_arg: str) -> ToolResult:
        """Build the 5.6 binary result, excluding the raw file contents."""

        return ToolResult(
            ok=False,
            content="",
            error=f"File appears to be binary (not valid UTF-8): {path_arg}",
            meta={"binary": True},
        )


# Static assertion that ``ReadTool`` satisfies the Tool protocol shape. Kept as
# a module-level reference (not executed work) for type checkers / readers.
_READ_TOOL_IS_A_TOOL: type[Tool] = ReadTool  # type: ignore[assignment]


class WriteTool:
    """The ``write`` tool: write content to a workspace file.

    Implements the :class:`~forge.tools.base.Tool` protocol.

    Behavior, by acceptance criterion (Requirement 6):

    * 6.1 - write the content to the path, replacing any existing content, and
      report the count of *bytes* written (the length of the UTF-8 encoded
      content) in the result content message and in ``meta["bytes_written"]``.
    * 6.2 - create all missing parent directories along the path before writing.
    * 6.6 - a path resolving outside the Workspace yields an out-of-scope result
      (delegated to :func:`resolve_in_workspace`) and leaves the filesystem
      unchanged.
    * 6.8 - a filesystem error (insufficient permissions, I/O failure) yields a
      result describing the failure and the affected path and leaves the
      filesystem unchanged.

    Atomicity (Req 6.8, design error-handling)
    -------------------------------------------
    The write is performed by writing the UTF-8 encoded content to a temporary
    file in the *same* directory as the target, then atomically replacing the
    target via :func:`os.replace`. Because ``os.replace`` is atomic on the same
    filesystem, a partial or failed write never leaves a half-written target:
    either the previous content remains or the full new content is present. The
    temp file is cleaned up on any failure so no stray artifacts remain.
    """

    name = "write"
    description = (
        "Write text content to a file within the workspace, replacing any "
        "existing content. Missing parent directories are created. Returns the "
        "path and the count of bytes (UTF-8 encoded) written."
    )
    read_only = False
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative or absolute path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "The full text content to write to the file.",
            },
        },
        "required": ["path", "content"],
    }

    def validate(self, args: dict) -> str | None:
        """Type/shape validation only (Req 6 inputs).

        Ensures both ``path`` and ``content`` are present and are strings.
        Returns an error string on a missing or wrongly typed argument, else
        ``None``.
        """

        if not isinstance(args, dict):
            return "Arguments must be an object."

        path = args.get("path")
        if path is None:
            return "Missing required argument 'path'."
        if not isinstance(path, str):
            return "Argument 'path' must be a string."

        content = args.get("content")
        if content is None:
            return "Missing required argument 'content'."
        if not isinstance(content, str):
            return "Argument 'content' must be a string."

        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Write the content atomically, applying scoping and error rules."""

        path_arg = args["path"]
        content = args["content"]

        # 6.6 - reject paths that resolve outside the Workspace; the filesystem
        # is left unchanged because no write is attempted.
        try:
            resolved = resolve_in_workspace(path_arg, ctx.workspace_root)
        except OutOfWorkspaceError as exc:
            return ToolResult(
                ok=False,
                content="",
                error=f"Path is out of scope: {exc.candidate}",
                meta={"out_of_scope": True},
            )

        # Capture the existing text (if any) for the diff in meta. A failed or
        # binary read is treated as no existing content; the diff will then
        # show the full new content, which is correct.
        old_text = ""
        if resolved.is_file():
            try:
                old_text = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                old_text = ""

        encoded = content.encode("utf-8")
        byte_count = len(encoded)

        # 6.2 - create all missing parent directories before writing.
        # 6.1/6.8 - write atomically via a temp file in the same directory + os.replace
        # so a partial/failed write leaves the filesystem unchanged.
        parent = resolved.parent
        tmp_path: str | None = None
        try:
            parent.mkdir(parents=True, exist_ok=True)

            # Create the temp file in the SAME directory so os.replace is atomic
            # (same filesystem). delete=False so we can replace it ourselves.
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".{resolved.name}.", suffix=".tmp", dir=str(parent)
            )
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)

            os.replace(tmp_path, resolved)
            tmp_path = None  # replaced successfully; nothing to clean up
        except OSError as exc:
            # 6.8 - filesystem error: describe the failure + path, leave the
            # filesystem unchanged (the target is untouched by os.replace).
            return ToolResult(
                ok=False,
                content="",
                error=f"Could not write file '{path_arg}': {exc}",
                meta={"io_error": True},
            )
        finally:
            # Clean up the temp file if the replace did not consume it.
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # 6.1 - success: confirm the path and the count of bytes written.
        # Include the unified diff in meta so the REPL / autopilot can
        # surface it under the `show_diffs` config without recomputing.
        diff_text = _unified_diff(str(resolved), old_text, content)
        return ToolResult(
            ok=True,
            content=f"Wrote {byte_count} byte(s) to {path_arg}.",
            error=None,
            meta={
                "bytes_written": byte_count,
                "path": str(resolved),
                "diff": diff_text,
            },
        )

    def preview(self, args: dict, ctx: ToolContext) -> str | None:
        """Return a unified diff of the proposed write without mutating.

        Scoped by :func:`resolve_in_workspace`; out-of-scope paths return
        ``None`` (the tool will report out-of-scope at run time). The preview is
        best-effort: an existing file that is binary (contains NUL bytes or is
        not valid UTF-8) or cannot be read returns ``None`` rather than raising,
        mirroring :class:`ReadTool`'s binary detection.
        """

        path_arg = args.get("path")
        new = str(args.get("content", ""))
        if not isinstance(path_arg, str):
            return None
        try:
            resolved = resolve_in_workspace(path_arg, ctx.workspace_root)
        except OutOfWorkspaceError:
            return None
        old = ""
        if resolved.is_file():
            try:
                raw = resolved.read_bytes()
            except OSError:
                return None
            # NUL bytes mark the file as binary; a textual diff would be
            # meaningless, so skip the preview (the run path handles the write).
            if b"\x00" in raw:
                return None
            # Decode via read_text so newline handling matches the pre-existing
            # behavior (universal newlines); a decode failure means binary.
            try:
                old = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return None
        return _unified_diff(str(resolved), old, new)


# Static assertion that ``WriteTool`` satisfies the Tool protocol shape.
_WRITE_TOOL_IS_A_TOOL: type[Tool] = WriteTool  # type: ignore[assignment]


class EditTool:
    """The ``edit`` tool: replace text in a workspace file with multiple modes.

    Implements the :class:`~forge.tools.base.Tool` protocol.

    Supports three edit modes:

    1. **replace** (default): Replace the unique target string. Zero occurrences
       returns "target not found"; more than one returns "ambiguous".
    2. **anchored**: Replace the target that appears between ``after`` and
       ``before`` anchors, disambiguating otherwise ambiguous targets.
    3. **line_range**: Replace lines ``start_line``..``end_line`` (1-based
       inclusive) with the replacement text, no target matching needed.

    Mode selection is inferred from which args are present; an explicit ``mode``
    takes precedence. Conflicting args yield a validation error.

    All modes preserve the existing atomic-write behavior and produce unified
    diffs in meta.
    """

    name = "edit"
    description = (
        "Replace text in a file within the workspace. Supports three modes: "
        "'replace' (default, unique target), 'anchored' (target between "
        "before/after markers), and 'line_range' (replace specific lines). "
        "The file is left unchanged on error."
    )
    read_only = False
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative or absolute path to the file to edit.",
            },
            "target": {
                "type": "string",
                "description": (
                    "The exact string to replace (for 'replace' and 'anchored' modes)."
                ),
            },
            "replacement": {
                "type": "string",
                "description": "The string to substitute for the target occurrence.",
            },
            "mode": {
                "type": "string",
                "enum": ["replace", "anchored", "line_range"],
                "description": (
                    "Edit mode. Inferred from args if omitted: 'line_range' when "
                    "start_line/end_line given, 'anchored' when after/before given, "
                    "else 'replace'."
                ),
            },
            "after": {
                "type": "string",
                "description": (
                    "Anchor text: the target must appear after this string (anchored mode)."
                ),
            },
            "before": {
                "type": "string",
                "description": (
                    "Anchor text: the target must appear before this string (anchored mode)."
                ),
            },
            "start_line": {
                "type": "integer",
                "description": (
                    "1-based first line to replace (inclusive, line_range mode)."
                ),
            },
            "end_line": {
                "type": "integer",
                "description": (
                    "1-based last line to replace (inclusive, line_range mode)."
                ),
            },
        },
        "required": ["path", "replacement"],
    }

    def validate(self, args: dict) -> str | None:
        """Validate arguments and detect the edit mode."""

        if not isinstance(args, dict):
            return "Arguments must be an object."

        path = args.get("path")
        if path is None:
            return "Missing required argument 'path'."
        if not isinstance(path, str):
            return "Argument 'path' must be a string."

        replacement = args.get("replacement")
        if replacement is None:
            return "Missing required argument 'replacement'."
        if not isinstance(replacement, str):
            return "Argument 'replacement' must be a string."

        mode = args.get("mode")
        if mode is not None:
            if not isinstance(mode, str) or mode not in (
                "replace",
                "anchored",
                "line_range",
            ):
                return "Argument 'mode' must be one of: replace, anchored, line_range."

        # Detect mode from args if not explicit
        if mode is None:
            has_start = args.get("start_line") is not None
            has_end = args.get("end_line") is not None
            has_after = args.get("after") is not None
            has_before = args.get("before") is not None

            if has_start or has_end:
                mode = "line_range"
            elif has_after or has_before:
                mode = "anchored"
            else:
                mode = "replace"

        # Mode-specific validation
        if mode == "replace":
            target = args.get("target")
            if target is None:
                return "Missing required argument 'target' for replace mode."
            if not isinstance(target, str):
                return "Argument 'target' must be a string."
            # Reject line_range/anchored args in replace mode
            if args.get("start_line") is not None or args.get("end_line") is not None:
                return "Cannot use start_line/end_line in replace mode."
            if args.get("after") is not None or args.get("before") is not None:
                return "Cannot use after/before in replace mode."

        elif mode == "anchored":
            target = args.get("target")
            if target is None:
                return "Missing required argument 'target' for anchored mode."
            if not isinstance(target, str):
                return "Argument 'target' must be a string."
            after = args.get("after")
            before = args.get("before")
            if after is None and before is None:
                return "Anchored mode requires at least one of 'after' or 'before'."
            if after is not None and not isinstance(after, str):
                return "Argument 'after' must be a string."
            if before is not None and not isinstance(before, str):
                return "Argument 'before' must be a string."
            # Reject line_range args in anchored mode
            if args.get("start_line") is not None or args.get("end_line") is not None:
                return "Cannot use start_line/end_line in anchored mode."

        elif mode == "line_range":
            start_line = args.get("start_line")
            end_line = args.get("end_line")
            if start_line is None and end_line is None:
                return "Line range mode requires at least one of 'start_line' or 'end_line'."
            if start_line is not None and not _is_int(start_line):
                return "Argument 'start_line' must be an integer."
            if end_line is not None and not _is_int(end_line):
                return "Argument 'end_line' must be an integer."
            # Reject target/anchored args in line_range mode
            if args.get("target") is not None:
                return "Cannot use target in line_range mode."
            if args.get("after") is not None or args.get("before") is not None:
                return "Cannot use after/before in line_range mode."

        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Run the edit with the detected mode."""

        path_arg = args["path"]
        replacement = args["replacement"]

        # Resolve and validate path
        try:
            resolved = resolve_in_workspace(path_arg, ctx.workspace_root)
        except OutOfWorkspaceError as exc:
            return ToolResult(
                ok=False,
                content="",
                error=f"Path is out of scope: {exc.candidate}",
                meta={"out_of_scope": True},
            )

        if not resolved.is_file():
            return ToolResult(
                ok=False,
                content="",
                error=f"File not found: {path_arg}",
                meta={"not_found": True},
            )

        # Read the file
        try:
            raw = resolved.read_bytes()
        except OSError as exc:
            return ToolResult(
                ok=False,
                content="",
                error=f"Could not read file '{path_arg}': {exc}",
                meta={"io_error": True},
            )

        if b"\x00" in raw:
            return ToolResult(
                ok=False,
                content="",
                error=f"File appears to be binary (not valid UTF-8): {path_arg}",
                meta={"binary": True},
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                ok=False,
                content="",
                error=f"File appears to be binary (not valid UTF-8): {path_arg}",
                meta={"binary": True},
            )

        # Detect mode
        mode = self._detect_mode(args)

        if mode == "line_range":
            return self._run_line_range(args, text, resolved, path_arg)
        elif mode == "anchored":
            return self._run_anchored(args, text, resolved, path_arg)
        else:
            return self._run_replace(args, text, resolved, path_arg)

    def _detect_mode(self, args: dict) -> str:
        """Detect the edit mode from args."""
        mode = args.get("mode")
        if mode in ("replace", "anchored", "line_range"):
            return mode

        has_start = args.get("start_line") is not None
        has_end = args.get("end_line") is not None
        has_after = args.get("after") is not None
        has_before = args.get("before") is not None

        if has_start or has_end:
            return "line_range"
        if has_after or has_before:
            return "anchored"
        return "replace"

    def _run_replace(
        self, args: dict, text: str, resolved: Path, path_arg: str
    ) -> ToolResult:
        """Run replace mode (original behavior)."""
        target = args["target"]
        occurrences = text.count(target)

        if occurrences == 0:
            return ToolResult(
                ok=False,
                content="",
                error="target not found",
                meta={"not_found_target": True},
            )

        if occurrences > 1:
            return ToolResult(
                ok=False,
                content="",
                error=f"target is ambiguous ({occurrences} occurrences)",
                meta={"ambiguous": True},
            )

        new_text = text.replace(target, args["replacement"], 1)
        return self._write_atomic(new_text, text, resolved, path_arg, replaced=1)

    def _run_anchored(
        self, args: dict, text: str, resolved: Path, path_arg: str
    ) -> ToolResult:
        """Run anchored mode: replace target between after/before anchors."""
        target = args["target"]
        after = args.get("after")
        before = args.get("before")
        replacement = args["replacement"]

        # Find the anchor window
        start_pos = 0
        end_pos = len(text)

        if after:
            after_idx = text.find(after)
            if after_idx == -1:
                return ToolResult(
                    ok=False,
                    content="",
                    error=f"After anchor not found: {after!r}",
                    meta={"anchor_not_found": True},
                )
            start_pos = after_idx + len(after)

        if before:
            before_idx = text.find(before, start_pos)
            if before_idx == -1:
                return ToolResult(
                    ok=False,
                    content="",
                    error=f"Before anchor not found: {before!r}",
                    meta={"anchor_not_found": True},
                )
            end_pos = before_idx

        # Extract the window
        window = text[start_pos:end_pos]
        occurrences = window.count(target)

        if occurrences == 0:
            return ToolResult(
                ok=False,
                content="",
                error="target not found within anchor window",
                meta={"not_found_target": True},
            )

        if occurrences > 1:
            return ToolResult(
                ok=False,
                content="",
                error=f"target is ambiguous within anchor window ({occurrences} occurrences)",
                meta={"ambiguous": True},
            )

        # Replace within the window, then reconstruct the full text
        new_window = window.replace(target, replacement, 1)
        new_text = text[:start_pos] + new_window + text[end_pos:]
        return self._write_atomic(new_text, text, resolved, path_arg, replaced=1)

    def _run_line_range(
        self, args: dict, text: str, resolved: Path, path_arg: str
    ) -> ToolResult:
        """Run line_range mode: replace lines start_line..end_line."""
        lines = text.splitlines(keepends=True)
        last_line = len(lines)

        start_line = args.get("start_line")
        end_line = args.get("end_line")
        replacement = args["replacement"]

        # Defaults
        effective_start = start_line if start_line is not None else 1
        effective_end = end_line if end_line is not None else last_line

        # Validate range
        if (
            effective_start < 1
            or effective_end > last_line
            or effective_start > effective_end
        ):
            return ToolResult(
                ok=False,
                content="",
                error=(
                    f"Invalid line range [{effective_start}, {effective_end}] "
                    f"for a file with {last_line} line(s)."
                ),
                meta={"invalid_range": True},
            )

        # Build new text: lines before + replacement + lines after
        before_lines = lines[: effective_start - 1]
        after_lines = lines[effective_end:]

        # Ensure replacement ends with newline if the original range did
        replacement_text = replacement
        if not replacement_text.endswith("\n") and after_lines:
            replacement_text += "\n"

        new_text = "".join(before_lines) + replacement_text + "".join(after_lines)
        return self._write_atomic(new_text, text, resolved, path_arg, replaced=1)

    def _write_atomic(
        self,
        new_text: str,
        old_text: str,
        resolved: Path,
        path_arg: str,
        replaced: int = 0,
    ) -> ToolResult:
        """Write new_text to the file atomically and return the result."""
        encoded = new_text.encode("utf-8")
        parent = resolved.parent
        tmp_path: str | None = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".{resolved.name}.", suffix=".tmp", dir=str(parent)
            )
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)

            os.replace(tmp_path, resolved)
            tmp_path = None
        except OSError as exc:
            return ToolResult(
                ok=False,
                content="",
                error=f"Could not write file '{path_arg}': {exc}",
                meta={"io_error": True},
            )
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        diff_text = _unified_diff(str(resolved), old_text, new_text)
        meta: dict = {"path": str(resolved), "diff": diff_text}
        if replaced:
            meta["replaced"] = replaced
        return ToolResult(
            ok=True,
            content=f"Successfully edited {path_arg}.",
            error=None,
            meta=meta,
        )

    def preview(self, args: dict, ctx: ToolContext) -> str | None:
        """Return a unified diff of the proposed edit without mutating.

        Best-effort: returns None for out-of-scope / not-found / ambiguous /
        binary cases.
        """

        path_arg = args.get("path")
        if not isinstance(path_arg, str):
            return None

        try:
            resolved = resolve_in_workspace(path_arg, ctx.workspace_root)
        except OutOfWorkspaceError:
            return None

        if not resolved.is_file():
            return None

        try:
            text = resolved.read_bytes()
        except OSError:
            return None

        if b"\x00" in text:
            return None

        try:
            decoded = text.decode("utf-8")
        except UnicodeDecodeError:
            return None

        # Try to compute the new text
        mode = self._detect_mode(args)
        replacement = args.get("replacement", "")

        if mode == "line_range":
            lines = decoded.splitlines(keepends=True)
            last_line = len(lines)
            start_line = args.get("start_line")
            end_line = args.get("end_line")
            effective_start = start_line if start_line is not None else 1
            effective_end = end_line if end_line is not None else last_line

            if (
                effective_start < 1
                or effective_end > last_line
                or effective_start > effective_end
            ):
                return None

            before_lines = lines[: effective_start - 1]
            after_lines = lines[effective_end:]
            replacement_text = replacement
            if not replacement_text.endswith("\n") and after_lines:
                replacement_text += "\n"
            new_text = "".join(before_lines) + replacement_text + "".join(after_lines)

        elif mode == "anchored":
            target = args.get("target", "")
            after = args.get("after")
            before = args.get("before")

            start_pos = 0
            end_pos = len(decoded)

            if after:
                after_idx = decoded.find(after)
                if after_idx == -1:
                    return None
                start_pos = after_idx + len(after)

            if before:
                before_idx = decoded.find(before, start_pos)
                if before_idx == -1:
                    return None
                end_pos = before_idx

            window = decoded[start_pos:end_pos]
            if window.count(target) != 1:
                return None

            new_window = window.replace(target, replacement, 1)
            new_text = decoded[:start_pos] + new_window + decoded[end_pos:]

        else:  # replace
            target = args.get("target", "")
            if decoded.count(target) != 1:
                return None
            new_text = decoded.replace(target, replacement, 1)

        return _unified_diff(str(resolved), decoded, new_text)


# Static assertion that ``EditTool`` satisfies the Tool protocol shape.
_EDIT_TOOL_IS_A_TOOL: type[Tool] = EditTool  # type: ignore[assignment]
