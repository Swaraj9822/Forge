"""Tests for the unified-diff preview hook (Phase 2, Feature C).

The ``preview`` methods on :class:`WriteTool` and :class:`EditTool` produce
unified diffs without mutating the filesystem. They power the interactive
approval prompt (``[approve] ... [diff]``) and the autopilot ``show_diffs``
path.
"""

from __future__ import annotations

from pathlib import Path

from forge.interrupt import InterruptController
from forge.tools.base import ToolContext
from forge.tools.fs import EditTool, WriteTool


def _ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace_root=workspace, interrupt=InterruptController())


# --------------------------------------------------------------------------- #
# WriteTool.preview
# --------------------------------------------------------------------------- #


def test_write_preview_new_file(tmp_path: Path) -> None:
    """Preview on a new file shows all-new lines (no source)."""

    target = tmp_path / "new.txt"
    tool = WriteTool()
    preview = tool.preview({"path": str(target), "content": "hello\nworld\n"}, _ctx(tmp_path))
    assert preview is not None
    # Unified diff must label the file and show +lines.
    assert f"a/{target}" in preview
    assert f"b/{target}" in preview
    assert "+hello" in preview
    assert "+world" in preview


def test_write_preview_existing_file(tmp_path: Path) -> None:
    """Preview on an existing file shows the changed lines."""

    target = tmp_path / "existing.txt"
    target.write_text("old1\nold2\nkeep\n", encoding="utf-8")
    tool = WriteTool()
    preview = tool.preview(
        {"path": str(target), "content": "new1\nold2\nkeep\n"}, _ctx(tmp_path)
    )
    assert preview is not None
    assert "-old1" in preview
    assert "+new1" in preview
    # Untouched lines are in the context block, not the change block.
    assert " old2" in preview or "old2" in preview


def test_write_preview_no_change(tmp_path: Path) -> None:
    """Preview when content matches the file is the literal no-change marker."""

    target = tmp_path / "same.txt"
    target.write_text("stable\n", encoding="utf-8")
    tool = WriteTool()
    preview = tool.preview(
        {"path": str(target), "content": "stable\n"}, _ctx(tmp_path)
    )
    assert preview == "(no textual change)"


def test_write_preview_out_of_scope_returns_none(tmp_path: Path) -> None:
    """A path that escapes the workspace returns ``None`` (best-effort preview)."""

    outside = tmp_path.parent / "outside.txt"
    tool = WriteTool()
    preview = tool.preview(
        {"path": str(outside), "content": "x"}, _ctx(tmp_path)
    )
    assert preview is None


def test_write_preview_binary_existing_returns_none(tmp_path: Path) -> None:
    """A binary existing file yields ``None`` (the run path will report it)."""

    target = tmp_path / "blob.bin"
    target.write_bytes(b"\xff\xfe\xfd not valid utf-8")
    tool = WriteTool()
    preview = tool.preview(
        {"path": str(target), "content": "still text"}, _ctx(tmp_path)
    )
    assert preview is None


def test_write_run_includes_diff_in_meta(tmp_path: Path) -> None:
    """A successful write attaches the unified diff to ``meta["diff"]``."""

    target = tmp_path / "out.txt"
    tool = WriteTool()
    result = tool.run(
        {"path": str(target), "content": "hi\n"}, _ctx(tmp_path)
    )
    assert result.ok is True
    diff = result.meta.get("diff")
    assert isinstance(diff, str)
    assert "+hi" in diff


# --------------------------------------------------------------------------- #
# EditTool.preview
# --------------------------------------------------------------------------- #


def test_edit_preview_reflects_single_replacement(tmp_path: Path) -> None:
    """Preview shows the replacement, not the original target string."""

    target = tmp_path / "file.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    tool = EditTool()
    preview = tool.preview(
        {
            "path": str(target),
            "target": "beta",
            "replacement": "BETA",
        },
        _ctx(tmp_path),
    )
    assert preview is not None
    assert "-beta" in preview
    assert "+BETA" in preview


def test_edit_preview_zero_occurrences_returns_none(tmp_path: Path) -> None:
    """A target that does not exist yields ``None`` (run will report not_found)."""

    target = tmp_path / "file.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    tool = EditTool()
    preview = tool.preview(
        {"path": str(target), "target": "missing", "replacement": "x"}, _ctx(tmp_path)
    )
    assert preview is None


def test_edit_preview_ambiguous_returns_none(tmp_path: Path) -> None:
    """A target that occurs more than once yields ``None`` (run will report ambiguous)."""

    target = tmp_path / "file.txt"
    target.write_text("a a a\n", encoding="utf-8")
    tool = EditTool()
    preview = tool.preview(
        {"path": str(target), "target": "a", "replacement": "b"}, _ctx(tmp_path)
    )
    assert preview is None


def test_edit_preview_out_of_scope_returns_none(tmp_path: Path) -> None:
    """Out-of-scope edit paths return ``None``."""

    outside = tmp_path.parent / "outside.txt"
    tool = EditTool()
    preview = tool.preview(
        {"path": str(outside), "target": "x", "replacement": "y"}, _ctx(tmp_path)
    )
    assert preview is None


def test_edit_run_includes_diff_in_meta(tmp_path: Path) -> None:
    """A successful edit attaches the unified diff to ``meta["diff"]``."""

    target = tmp_path / "file.txt"
    target.write_text("a\nb\n", encoding="utf-8")
    tool = EditTool()
    result = tool.run(
        {"path": str(target), "target": "a", "replacement": "A"}, _ctx(tmp_path)
    )
    assert result.ok is True
    diff = result.meta.get("diff")
    assert isinstance(diff, str)
    assert "-a" in diff
    assert "+A" in diff
