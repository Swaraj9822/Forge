"""Unit tests for the write tool's error paths (task 8.3).

These cover the two failure contracts of :class:`forge.tools.fs.WriteTool` that
must leave the filesystem unchanged:

* Req 6.6 - a path resolving outside the Workspace yields an out-of-scope
  result (``meta["out_of_scope"]``) and no file is created outside the
  Workspace.
* Req 6.8 - a filesystem error (e.g. a parent path that is an existing file, or
  an I/O failure during the atomic replace) yields an io-error result
  (``meta["io_error"]``) and leaves the filesystem byte-for-byte unchanged with
  no stray temp files left behind.

Per the environment note, ``tmp_path`` is unreliable on this host, so each test
builds an isolated workspace via ``tempfile.mkdtemp`` and cleans it up in a
``finally`` block. Path segments use a safe lowercase alphabet that avoids
OS-reserved characters and Windows-reserved device names.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from forge.interrupt import InterruptController
from forge.tools import fs
from forge.tools.base import ToolContext
from forge.tools.fs import WriteTool


def _make_ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace_root=workspace, interrupt=InterruptController())


def _list_dir(path: Path) -> list[str]:
    """Sorted names in ``path`` (used to assert no stray temp files remain)."""
    return sorted(p.name for p in path.iterdir())


# --- Req 6.6: out-of-scope rejection -----------------------------------------


def test_write_absolute_path_outside_workspace_is_out_of_scope() -> None:
    """An absolute path into a different directory is rejected; nothing written."""
    workspace = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_ws_")))
    outside = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_outside_")))
    try:
        ctx = _make_ctx(workspace)
        target = outside / "escape.txt"

        result = WriteTool().run(
            {"path": str(target), "content": "should not be written"}, ctx
        )

        assert result.ok is False
        assert result.meta.get("out_of_scope") is True
        # Filesystem unchanged: no file created at the outside location, and the
        # outside directory remains empty.
        assert not target.exists()
        assert _list_dir(outside) == []
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(outside, ignore_errors=True)


def test_write_relative_traversal_path_is_out_of_scope() -> None:
    """A ``../`` path that escapes the workspace is rejected; nothing written."""
    parent = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_parent_")))
    workspace = parent / "ws"
    workspace.mkdir()
    try:
        ctx = _make_ctx(workspace)

        result = WriteTool().run(
            {"path": "../escape.txt", "content": "should not be written"}, ctx
        )

        assert result.ok is False
        assert result.meta.get("out_of_scope") is True
        # The traversal target (in the parent, outside the workspace) was not
        # created.
        assert not (parent / "escape.txt").exists()
        assert _list_dir(parent) == ["ws"]
    finally:
        shutil.rmtree(parent, ignore_errors=True)


# --- Req 6.8: filesystem error (parent is an existing file) ------------------


def test_write_parent_is_existing_file_is_io_error_and_unchanged() -> None:
    """Writing under a path whose parent is a regular file fails atomically.

    Creating ``blocker/child.txt`` requires ``blocker`` to be a directory, but
    here ``blocker`` is an existing regular file, so ``mkdir`` (or the open)
    raises ``OSError``. The tool must report an io-error and leave the filesystem
    unchanged: ``blocker`` keeps its original content and ``blocker/child.txt``
    does not exist.
    """
    workspace = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_ws_")))
    try:
        ctx = _make_ctx(workspace)
        blocker = workspace / "blocker"
        blocker.write_text("original blocker content", encoding="utf-8")

        result = WriteTool().run(
            {"path": "blocker/child.txt", "content": "new content"}, ctx
        )

        assert result.ok is False
        assert result.meta.get("io_error") is True
        # Filesystem unchanged: the blocker file still holds its original bytes
        # and is still a regular file (not turned into a directory), and the
        # nested target was never created.
        assert blocker.is_file()
        assert blocker.read_text(encoding="utf-8") == "original blocker content"
        # Only the blocker file exists in the workspace (no stray temp files).
        assert _list_dir(workspace) == ["blocker"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


# --- Req 6.8: filesystem error during the atomic replace ---------------------


def test_write_replace_failure_preserves_existing_file_and_leaves_no_temp(
    monkeypatch,
) -> None:
    """If ``os.replace`` fails, an existing target keeps its original content.

    The write is performed by writing a temp file in the same directory and then
    atomically replacing the target via ``os.replace``. By forcing ``os.replace``
    to raise ``OSError`` we exercise the failure branch after the temp file has
    been written: the original file content must remain intact (atomicity) and
    the temp file must be cleaned up so no stray artifacts remain.
    """
    workspace = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_ws_")))
    try:
        ctx = _make_ctx(workspace)
        target = workspace / "existing.txt"
        target.write_text("original content", encoding="utf-8")

        def boom(src, dst):  # noqa: ANN001 - signature mirrors os.replace
            raise OSError("simulated replace failure")

        monkeypatch.setattr(fs.os, "replace", boom)

        result = WriteTool().run(
            {"path": "existing.txt", "content": "replacement content"}, ctx
        )

        assert result.ok is False
        assert result.meta.get("io_error") is True
        # Atomicity (6.8): the original file content is untouched.
        assert target.read_text(encoding="utf-8") == "original content"
        # No stray temp files left behind: only the original file remains.
        assert _list_dir(workspace) == ["existing.txt"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_write_replace_failure_creates_no_target_and_leaves_no_temp(
    monkeypatch,
) -> None:
    """If ``os.replace`` fails for a new path, the target is never created.

    Same forced failure as above but with no pre-existing target: the new file
    must not appear and the temp file must be cleaned up, leaving the workspace
    empty.
    """
    workspace = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_ws_")))
    try:
        ctx = _make_ctx(workspace)

        def boom(src, dst):  # noqa: ANN001 - signature mirrors os.replace
            raise OSError("simulated replace failure")

        monkeypatch.setattr(fs.os, "replace", boom)

        result = WriteTool().run(
            {"path": "brandnew.txt", "content": "content"}, ctx
        )

        assert result.ok is False
        assert result.meta.get("io_error") is True
        # The target was never created and no stray temp file remains.
        assert not (workspace / "brandnew.txt").exists()
        assert _list_dir(workspace) == []
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


# --- Req 6.6: out-of-scope leaves an existing outside file unchanged ----------


def test_write_out_of_scope_leaves_existing_outside_file_unchanged() -> None:
    """An out-of-scope write must not modify a file that already exists outside.

    The earlier out-of-scope tests assert nothing new is *created*; this one
    closes the "leave the file system unchanged" half of Req 6.6 for the case
    where the target already exists outside the workspace: its bytes must be
    untouched. The result is also checked to be a descriptive failure (empty
    content, populated error) rather than a silent no-op.
    """
    workspace = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_ws_")))
    outside = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_outside_")))
    try:
        ctx = _make_ctx(workspace)
        target = outside / "preexisting.txt"
        target.write_text("untouched original", encoding="utf-8")

        result = WriteTool().run(
            {"path": str(target), "content": "should not overwrite"}, ctx
        )

        assert result.ok is False
        assert result.meta.get("out_of_scope") is True
        # Descriptive failure: no content, a populated error message.
        assert result.content == ""
        assert result.error
        # 6.6 - the existing outside file keeps its original bytes, and the
        # outside directory gains no stray files.
        assert target.read_text(encoding="utf-8") == "untouched original"
        assert _list_dir(outside) == ["preexisting.txt"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(outside, ignore_errors=True)


# --- Req 6.8: target path is an existing directory ---------------------------


def test_write_target_is_existing_directory_is_io_error_and_unchanged() -> None:
    """Writing to a path that is an existing directory fails atomically.

    ``os.replace`` cannot replace a directory with the temp file, so it raises
    ``OSError`` (e.g. ``IsADirectoryError`` on POSIX, ``PermissionError`` on
    Windows). Unlike the monkeypatched ``os.replace`` tests below, this exercises
    a *real* filesystem error. The tool must report an io-error, leave the
    directory (and its contents) intact, and clean up the temp file so no stray
    artifacts remain.
    """
    workspace = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_ws_")))
    try:
        ctx = _make_ctx(workspace)
        target_dir = workspace / "adir"
        target_dir.mkdir()
        # A marker inside the directory lets us prove it was left untouched.
        (target_dir / "keep.txt").write_text("keep me", encoding="utf-8")

        result = WriteTool().run({"path": "adir", "content": "content"}, ctx)

        assert result.ok is False
        assert result.meta.get("io_error") is True
        assert result.error
        # 6.8 - the directory is still a directory with its contents intact.
        assert target_dir.is_dir()
        assert (target_dir / "keep.txt").read_text(encoding="utf-8") == "keep me"
        # No stray temp files left in the workspace: only the directory remains.
        assert _list_dir(workspace) == ["adir"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


# --- Req 6.8: filesystem error BEFORE the atomic replace ---------------------


def test_write_tempfile_creation_failure_preserves_existing_and_leaves_no_temp(
    monkeypatch,
) -> None:
    """A failure creating the temp file is reported and changes nothing.

    The ``os.replace`` tests cover failure *after* the temp file is written
    (atomicity of the swap). This covers the complementary branch: the temp
    file creation itself fails (``tempfile.mkstemp`` raising ``OSError``), so no
    temp file is ever produced and the existing target keeps its original
    content. The tool must still report an io-error and leave the workspace
    holding only the original file.
    """
    workspace = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_ws_")))
    try:
        ctx = _make_ctx(workspace)
        target = workspace / "existing.txt"
        target.write_text("original content", encoding="utf-8")

        def boom(*args, **kwargs):  # noqa: ANN002, ANN003 - mirrors mkstemp
            raise OSError("simulated temp-file creation failure")

        monkeypatch.setattr(fs.tempfile, "mkstemp", boom)

        result = WriteTool().run(
            {"path": "existing.txt", "content": "replacement content"}, ctx
        )

        assert result.ok is False
        assert result.meta.get("io_error") is True
        assert result.error
        # 6.8 - the original file content is untouched and no temp file leaked.
        assert target.read_text(encoding="utf-8") == "original content"
        assert _list_dir(workspace) == ["existing.txt"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
