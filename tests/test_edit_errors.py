"""Unit tests for the edit tool's error paths (task 9.3).

These plain ``pytest`` tests cover the three failure modes of
:class:`forge.tools.fs.EditTool` that must leave the filesystem unchanged:

* **Not found (Req 6.7)** - editing a path that does not exist returns an
  unsuccessful result flagged ``meta["not_found"]`` and creates no file.
* **Out of scope (Req 6.6)** - editing a path that resolves outside the
  Workspace returns an unsuccessful result flagged ``meta["out_of_scope"]`` and
  leaves the outside file untouched.
* **Filesystem error (Req 6.8)** - when the write-back fails after a unique
  match is found, the result is flagged ``meta["io_error"]``, the original file
  is byte-for-byte unchanged (atomicity), and no stray temp files remain.

Per the environment note, each test builds its workspace with ``tempfile``
(``tmp_path`` is unreliable on this host) and cleans it up afterwards, and uses
only safe, non-reserved filenames.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import forge.tools.fs as fs
from forge.interrupt import InterruptController
from forge.tools.base import ToolContext
from forge.tools.fs import EditTool


def _make_workspace() -> Path:
    """Create an isolated workspace directory (realpath-normalized)."""
    return Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_edit_err_")))


def _ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace_root=workspace, interrupt=InterruptController())


def test_edit_not_found_file_leaves_filesystem_unchanged() -> None:
    """Req 6.7: editing a missing path returns not-found and creates no file."""
    workspace = _make_workspace()
    try:
        ctx = _ctx(workspace)

        result = EditTool().run(
            {"path": "missing.txt", "target": "a", "replacement": "b"}, ctx
        )

        assert result.ok is False
        assert result.meta.get("not_found") is True
        # No file was created as a side effect.
        assert not (workspace / "missing.txt").exists()
        assert list(workspace.iterdir()) == []
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_edit_out_of_scope_leaves_outside_file_unchanged() -> None:
    """Req 6.6: a path resolving outside the Workspace is rejected, file untouched."""
    parent = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_edit_scope_")))
    try:
        workspace = parent / "workspace"
        workspace.mkdir()

        # A file that lives OUTSIDE the workspace, with known content.
        outside = parent / "outside.txt"
        original = "secret target here"
        outside.write_text(original, encoding="utf-8")

        ctx = _ctx(workspace)

        # "../outside.txt" resolves out of the workspace root.
        result = EditTool().run(
            {"path": "../outside.txt", "target": "target", "replacement": "REPLACED"},
            ctx,
        )

        assert result.ok is False
        assert result.meta.get("out_of_scope") is True
        # The outside file is left byte-for-byte unchanged.
        assert outside.read_text(encoding="utf-8") == original
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def test_edit_filesystem_error_is_atomic_and_leaves_no_temp_files(monkeypatch) -> None:
    """Req 6.8: a write-back failure leaves the original file unchanged and clean."""
    workspace = _make_workspace()
    try:
        ctx = _ctx(workspace)

        # A file containing EXACTLY ONE occurrence of the target, so the edit
        # proceeds to the write-back stage where the failure is injected.
        target_file = workspace / "doc.txt"
        original = "alpha UNIQUE beta\nsecond line\n"
        target_file.write_text(original, encoding="utf-8")
        original_bytes = target_file.read_bytes()

        # Force an OSError during the atomic swap. The temp file is written
        # first, then os.replace fails - exercising the cleanup path.
        def _boom(src, dst):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(fs.os, "replace", _boom)

        result = EditTool().run(
            {"path": "doc.txt", "target": "UNIQUE", "replacement": "CHANGED"}, ctx
        )

        assert result.ok is False
        assert result.meta.get("io_error") is True

        # Atomicity: the original file content is byte-for-byte unchanged.
        assert target_file.read_bytes() == original_bytes

        # No stray temp files remain in the directory (only the original).
        remaining = sorted(p.name for p in workspace.iterdir())
        assert remaining == ["doc.txt"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


# ---------------------------------------------------------------------------
# Additional error-path coverage (task 9.3)
#
# The three tests above exercise the baseline contracts: a missing path
# (Req 6.7), a ``../`` traversal that escapes the Workspace (Req 6.6), and a
# write-back ``os.replace`` failure (Req 6.8). The tests below extend that
# coverage to the remaining distinct branches of ``EditTool.run`` -- an
# *absolute* out-of-scope path, a path that exists but is a *directory*, an I/O
# failure while *reading* the file, and an I/O failure while *creating the temp
# file* -- each of which must also leave the filesystem byte-for-byte unchanged.
# ---------------------------------------------------------------------------


def test_edit_absolute_path_outside_workspace_is_out_of_scope() -> None:
    """Req 6.6: an absolute path into another directory is rejected, file untouched."""
    workspace = _make_workspace()
    outside = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_edit_abs_")))
    try:
        # A file with a unique target living OUTSIDE the workspace.
        outside_file = outside / "secret.txt"
        original = "do not touch UNIQUE here"
        outside_file.write_text(original, encoding="utf-8")

        ctx = _ctx(workspace)

        # An absolute path pointing outside the workspace root.
        result = EditTool().run(
            {"path": str(outside_file), "target": "UNIQUE", "replacement": "REPLACED"},
            ctx,
        )

        assert result.ok is False
        assert result.meta.get("out_of_scope") is True
        # The outside file is left byte-for-byte unchanged.
        assert outside_file.read_text(encoding="utf-8") == original
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(outside, ignore_errors=True)


def test_edit_directory_path_is_not_found_and_unchanged() -> None:
    """Req 6.7: a path that exists but is a directory is not-found; nothing changes."""
    workspace = _make_workspace()
    try:
        ctx = _ctx(workspace)

        # The target path exists but is a directory, not a regular file.
        subdir = workspace / "subdir"
        subdir.mkdir()

        result = EditTool().run(
            {"path": "subdir", "target": "a", "replacement": "b"}, ctx
        )

        assert result.ok is False
        assert result.meta.get("not_found") is True
        # The directory is untouched and no file was created in its place.
        assert subdir.is_dir()
        assert sorted(p.name for p in workspace.iterdir()) == ["subdir"]
        assert list(subdir.iterdir()) == []
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_edit_read_failure_is_io_error_and_leaves_file_unchanged(monkeypatch) -> None:
    """Req 6.8: an I/O failure while reading yields io-error; file left unchanged.

    The failure is injected *before* any write occurs, so the original content
    must remain byte-for-byte intact. ``Path.read_text`` (used for the
    verification) opens the file directly and does not route through the patched
    ``Path.read_bytes``, so it still reflects the true on-disk content.
    """
    workspace = _make_workspace()
    try:
        ctx = _ctx(workspace)

        target_file = workspace / "doc.txt"
        original = "alpha UNIQUE beta\nsecond line\n"
        target_file.write_text(original, encoding="utf-8")

        real_read_bytes = Path.read_bytes

        def _boom(self):
            # Fail only for the subject file; let everything else read normally.
            if self.name == "doc.txt":
                raise OSError("simulated read failure")
            return real_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", _boom)

        result = EditTool().run(
            {"path": "doc.txt", "target": "UNIQUE", "replacement": "CHANGED"}, ctx
        )

        assert result.ok is False
        assert result.meta.get("io_error") is True

        # The file is byte-for-byte unchanged and no stray temp files remain.
        assert target_file.read_text(encoding="utf-8") == original
        assert sorted(p.name for p in workspace.iterdir()) == ["doc.txt"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_edit_tempfile_creation_failure_is_atomic_and_leaves_no_temp(monkeypatch) -> None:
    """Req 6.8: a failure creating the temp file leaves the original file intact.

    Unlike the ``os.replace`` failure case above, here the write-back fails at
    the very first step (``tempfile.mkstemp``), so no temp file is ever created.
    The original file content must remain byte-for-byte unchanged and the
    directory must contain only the original file.
    """
    workspace = _make_workspace()
    try:
        ctx = _ctx(workspace)

        target_file = workspace / "doc.txt"
        original = "alpha UNIQUE beta\nsecond line\n"
        target_file.write_text(original, encoding="utf-8")
        original_bytes = target_file.read_bytes()

        def _boom(*args, **kwargs):
            raise OSError("simulated mkstemp failure")

        monkeypatch.setattr(fs.tempfile, "mkstemp", _boom)

        result = EditTool().run(
            {"path": "doc.txt", "target": "UNIQUE", "replacement": "CHANGED"}, ctx
        )

        assert result.ok is False
        assert result.meta.get("io_error") is True

        # Atomicity: the original content is byte-for-byte unchanged.
        assert target_file.read_bytes() == original_bytes
        # No temp file was created (mkstemp failed before producing one).
        assert sorted(p.name for p in workspace.iterdir()) == ["doc.txt"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
