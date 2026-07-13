"""Unit tests for the read tool's error paths (task 7.5).

These plain ``pytest`` tests cover the three non-range failure modes of
:class:`forge.tools.fs.ReadTool`:

* **Not found (Req 5.3)** - reading a path that does not exist (or that exists
  but is not a regular file, e.g. a directory) returns an unsuccessful result
  flagged ``meta["not_found"]`` and produces no content.
* **Out of scope (Req 5.4)** - reading a path that resolves outside the
  Workspace returns an unsuccessful result flagged ``meta["out_of_scope"]``;
  no read of the outside file happens, so its contents never leak into the
  result.
* **Binary (Req 5.6)** - reading a file that is not valid UTF-8 (NUL bytes
  present, or an invalid UTF-8 byte sequence) returns an unsuccessful result
  flagged ``meta["binary"]`` whose ``content`` EXCLUDES the raw file bytes.

Per the environment note, each test builds its workspace with ``tempfile``
(``tmp_path`` is unreliable on this host) and cleans it up afterwards, using
only safe, non-reserved filenames.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from forge.interrupt import InterruptController
from forge.tools.base import ToolContext
from forge.tools.fs import ReadTool


def _make_workspace() -> Path:
    """Create an isolated, realpath-normalized workspace directory."""
    return Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_read_err_")))


def _ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace_root=workspace, interrupt=InterruptController())


# --- 5.3 not found ---------------------------------------------------------


def test_read_missing_path_returns_not_found() -> None:
    """Req 5.3: reading a path that does not exist returns a not-found result."""
    workspace = _make_workspace()
    try:
        result = ReadTool().run({"path": "missing.txt"}, _ctx(workspace))

        assert result.ok is False
        assert result.meta.get("not_found") is True
        assert result.content == ""
        assert result.error is not None
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_read_directory_path_returns_not_found() -> None:
    """Req 5.3: a path that exists but is not a regular file is not-found.

    The read tool requires a regular file (``Path.is_file()``); a directory is
    therefore reported as not found rather than read.
    """
    workspace = _make_workspace()
    try:
        sub = workspace / "subdir"
        sub.mkdir()

        result = ReadTool().run({"path": "subdir"}, _ctx(workspace))

        assert result.ok is False
        assert result.meta.get("not_found") is True
        assert result.content == ""
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


# --- 5.4 out of scope ------------------------------------------------------


def test_read_out_of_scope_is_rejected_and_does_not_leak_contents() -> None:
    """Req 5.4: a path resolving outside the Workspace is rejected, not read."""
    parent = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_read_scope_")))
    try:
        workspace = parent / "workspace"
        workspace.mkdir()

        # A file OUTSIDE the workspace with a recognizable secret content.
        outside = parent / "outside.txt"
        secret = "TOP-SECRET-OUTSIDE-CONTENT"
        outside.write_text(secret, encoding="utf-8")

        # "../outside.txt" resolves out of the workspace root.
        result = ReadTool().run({"path": "../outside.txt"}, _ctx(workspace))

        assert result.ok is False
        assert result.meta.get("out_of_scope") is True
        # The outside file's contents must never appear in the result.
        assert secret not in result.content
        assert result.content == ""
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def test_read_absolute_out_of_scope_is_rejected() -> None:
    """Req 5.4: an absolute path outside the Workspace is also rejected."""
    parent = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_read_scope_")))
    try:
        workspace = parent / "workspace"
        workspace.mkdir()

        outside = parent / "outside.txt"
        outside.write_text("data", encoding="utf-8")

        # Pass the absolute path to the outside file directly.
        result = ReadTool().run({"path": str(outside)}, _ctx(workspace))

        assert result.ok is False
        assert result.meta.get("out_of_scope") is True
        assert result.content == ""
    finally:
        shutil.rmtree(parent, ignore_errors=True)


# --- 5.6 binary ------------------------------------------------------------


def test_read_file_with_nul_byte_is_binary_and_excludes_contents() -> None:
    """Req 5.6: a file containing NUL bytes is reported binary, contents excluded."""
    workspace = _make_workspace()
    try:
        file_path = workspace / "data.bin"
        # Printable text surrounding a NUL byte: the NUL forces binary
        # classification, and the surrounding text must NOT appear in content.
        raw = b"readable-prefix\x00readable-suffix"
        file_path.write_bytes(raw)

        result = ReadTool().run({"path": "data.bin"}, _ctx(workspace))

        assert result.ok is False
        assert result.meta.get("binary") is True
        # The raw contents are excluded from the result.
        assert result.content == ""
        assert "readable-prefix" not in result.content
        assert "readable-suffix" not in result.content
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_read_invalid_utf8_is_binary_and_excludes_contents() -> None:
    """Req 5.6: a file that fails UTF-8 decoding is reported binary."""
    workspace = _make_workspace()
    try:
        file_path = workspace / "invalid.dat"
        # 0xFF / 0xFE are never valid UTF-8 lead bytes, so decoding fails.
        raw = b"\xff\xfe\xff\xfe\xff"
        file_path.write_bytes(raw)

        result = ReadTool().run({"path": "invalid.dat"}, _ctx(workspace))

        assert result.ok is False
        assert result.meta.get("binary") is True
        assert result.content == ""
        assert result.error is not None
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
