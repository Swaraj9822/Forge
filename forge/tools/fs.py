"""Filesystem tools: ``read`` and ``write`` (and, later, ``edit``).

This module hosts the workspace-scoped filesystem tools. It contains the
:class:`ReadTool` (task 7.1) and :class:`WriteTool` (task 8.1); the ``edit``
tool (task 9.1) will be added here later as a separate class that shares the
same :class:`~forge.tools.base.Tool` protocol and the
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
"""

from __future__ import annotations

import os
import tempfile
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
        return ToolResult(
            ok=True,
            content=f"Wrote {byte_count} byte(s) to {path_arg}.",
            error=None,
            meta={"bytes_written": byte_count, "path": str(resolved)},
        )


# Static assertion that ``WriteTool`` satisfies the Tool protocol shape.
_WRITE_TOOL_IS_A_TOOL: type[Tool] = WriteTool  # type: ignore[assignment]


class EditTool:
    """The ``edit`` tool: replace a unique target string in a workspace file.

    Implements the :class:`~forge.tools.base.Tool` protocol.

    Behavior, by acceptance criterion (Requirement 6):

    * 6.3 - when ``target`` occurs *exactly once* in the file, replace that
      single occurrence with ``replacement``, write the file back, and return a
      success result confirming the change.
    * 6.4 - when ``target`` occurs *zero* times in an existing file, return a
      "target not found" result (``meta["not_found_target"]``) and leave the
      file byte-for-byte unchanged.
    * 6.5 - when ``target`` occurs *more than once*, return an "ambiguous"
      result (``meta["ambiguous"]``) reporting the occurrence count and leave
      the file byte-for-byte unchanged.
    * 6.6 - a path resolving outside the Workspace yields an out-of-scope result
      (delegated to :func:`resolve_in_workspace`) and leaves the filesystem
      unchanged.
    * 6.7 - a path that does not exist yields a not-found result
      (``meta["not_found"]``) and leaves the filesystem unchanged.
    * 6.8 - a filesystem error (insufficient permissions, I/O failure) yields a
      result describing the failure and the affected path and leaves the
      filesystem unchanged.

    Distinct not-found meta keys (Req 6.7 vs 6.4)
    ---------------------------------------------
    The two "not found" cases use distinct ``meta`` keys so callers/tests can
    tell them apart: a *missing file path* (Req 6.7) sets
    ``meta["not_found"]``; a *target string absent from an existing file*
    (Req 6.4) sets ``meta["not_found_target"]``.

    Atomicity (Req 6.8, design error-handling)
    -------------------------------------------
    Like :class:`WriteTool`, the edited content is written to a temporary file
    in the *same* directory as the target and then atomically swapped in via
    :func:`os.replace`. A partial or failed write therefore never leaves a
    half-written file: either the original content remains or the full edited
    content is present. The temp file is cleaned up on any failure.
    """

    name = "edit"
    description = (
        "Replace an exact target string with a replacement string in a file "
        "within the workspace. The target must occur exactly once: zero "
        "occurrences returns 'target not found' and more than one returns "
        "'ambiguous', and in both cases the file is left unchanged."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative or absolute path to the file to edit.",
            },
            "target": {
                "type": "string",
                "description": "The exact string to replace; must occur exactly once.",
            },
            "replacement": {
                "type": "string",
                "description": "The string to substitute for the target occurrence.",
            },
        },
        "required": ["path", "target", "replacement"],
    }

    def validate(self, args: dict) -> str | None:
        """Type/shape validation only (Req 6 inputs).

        Ensures ``path``, ``target``, and ``replacement`` are all present and
        are strings. Returns an error string on a missing or wrongly typed
        argument, else ``None``. Uniqueness of the target is a runtime concern
        handled in :meth:`run` (Req 6.3/6.4/6.5).
        """

        if not isinstance(args, dict):
            return "Arguments must be an object."

        path = args.get("path")
        if path is None:
            return "Missing required argument 'path'."
        if not isinstance(path, str):
            return "Argument 'path' must be a string."

        target = args.get("target")
        if target is None:
            return "Missing required argument 'target'."
        if not isinstance(target, str):
            return "Argument 'target' must be a string."

        replacement = args.get("replacement")
        if replacement is None:
            return "Missing required argument 'replacement'."
        if not isinstance(replacement, str):
            return "Argument 'replacement' must be a string."

        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Replace the unique target, applying scoping, uniqueness, and error rules."""

        path_arg = args["path"]
        target = args["target"]
        replacement = args["replacement"]

        # 6.6 - reject paths that resolve outside the Workspace; the filesystem
        # is left unchanged because no read or write is attempted.
        try:
            resolved = resolve_in_workspace(path_arg, ctx.workspace_root)
        except OutOfWorkspaceError as exc:
            return ToolResult(
                ok=False,
                content="",
                error=f"Path is out of scope: {exc.candidate}",
                meta={"out_of_scope": True},
            )

        # 6.7 - the path must exist and be a regular file; otherwise not found.
        if not resolved.is_file():
            return ToolResult(
                ok=False,
                content="",
                error=f"File not found: {path_arg}",
                meta={"not_found": True},
            )

        # Read the existing content as UTF-8. Binary/decode errors and IO
        # failures yield error results that leave the file unchanged.
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

        # Count occurrences of the (non-overlapping) target string.
        occurrences = text.count(target)

        # 6.4 - zero occurrences: target not found; file left unchanged.
        if occurrences == 0:
            return ToolResult(
                ok=False,
                content="",
                error="target not found",
                meta={"not_found_target": True},
            )

        # 6.5 - more than one occurrence: ambiguous; file left unchanged.
        if occurrences > 1:
            return ToolResult(
                ok=False,
                content="",
                error=f"target is ambiguous ({occurrences} occurrences)",
                meta={"ambiguous": True},
            )

        # 6.3 - exactly one occurrence: replace it and write the file back.
        new_text = text.replace(target, replacement, 1)
        encoded = new_text.encode("utf-8")

        # Write atomically via a temp file in the same directory + os.replace so
        # a partial/failed write leaves the file byte-for-byte unchanged (6.8).
        parent = resolved.parent
        tmp_path: str | None = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".{resolved.name}.", suffix=".tmp", dir=str(parent)
            )
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)

            os.replace(tmp_path, resolved)
            tmp_path = None  # replaced successfully; nothing to clean up
        except OSError as exc:
            # 6.8 - filesystem error: describe the failure + path, leave the
            # file unchanged (the target is untouched by os.replace).
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

        # 6.3 - success: confirm the change.
        return ToolResult(
            ok=True,
            content=f"Replaced 1 occurrence of the target in {path_arg}.",
            error=None,
            meta={"replaced": 1, "path": str(resolved)},
        )


# Static assertion that ``EditTool`` satisfies the Tool protocol shape.
_EDIT_TOOL_IS_A_TOOL: type[Tool] = EditTool  # type: ignore[assignment]
