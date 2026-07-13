"""Tests for EditTool's robust edit modes (Feature I): anchored + line_range.

The default 'replace' mode behavior is covered by test_edit_uniqueness.py and
test_edit_errors.py; these tests focus on the new modes and mode selection.
"""

from __future__ import annotations

from pathlib import Path

from forge.interrupt import InterruptController
from forge.tools.base import ToolContext
from forge.tools.fs import EditTool


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_root=tmp_path, interrupt=InterruptController())


# --------------------------------------------------------------------------- #
# line_range mode
# --------------------------------------------------------------------------- #


def test_line_range_replaces_selected_lines(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("one\ntwo\nthree\n", encoding="utf-8")
    result = EditTool().run(
        {"path": str(f), "start_line": 2, "end_line": 2, "replacement": "TWO"},
        _ctx(tmp_path),
    )
    assert result.ok is True
    assert f.read_text(encoding="utf-8") == "one\nTWO\nthree\n"


def test_line_range_multi_line_block(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")
    result = EditTool().run(
        {"path": str(f), "start_line": 2, "end_line": 3, "replacement": "X\nY"},
        _ctx(tmp_path),
    )
    assert result.ok is True
    assert f.read_text(encoding="utf-8") == "a\nX\nY\nd\n"


def test_line_range_invalid_range_rejected(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a\nb\n", encoding="utf-8")
    result = EditTool().run(
        {"path": str(f), "start_line": 1, "end_line": 99, "replacement": "x"},
        _ctx(tmp_path),
    )
    assert result.ok is False
    assert result.meta.get("invalid_range") is True
    # File unchanged.
    assert f.read_text(encoding="utf-8") == "a\nb\n"


# --------------------------------------------------------------------------- #
# anchored mode
# --------------------------------------------------------------------------- #


def test_anchored_disambiguates_repeated_target(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("x = 1\nmarker\nx = 1\n", encoding="utf-8")
    # 'x = 1' occurs twice; the 'after' anchor selects the second one.
    result = EditTool().run(
        {
            "path": str(f),
            "after": "marker",
            "target": "x = 1",
            "replacement": "x = 2",
        },
        _ctx(tmp_path),
    )
    assert result.ok is True
    assert f.read_text(encoding="utf-8") == "x = 1\nmarker\nx = 2\n"


def test_anchored_ambiguous_within_window(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("a\na\n", encoding="utf-8")
    result = EditTool().run(
        {"path": str(f), "after": "", "target": "a", "replacement": "b"},
        _ctx(tmp_path),
    )
    assert result.ok is False
    assert result.meta.get("ambiguous") is True


def test_anchored_missing_anchor(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("hello world\n", encoding="utf-8")
    result = EditTool().run(
        {
            "path": str(f),
            "after": "NOSUCHANCHOR",
            "target": "world",
            "replacement": "there",
        },
        _ctx(tmp_path),
    )
    assert result.ok is False
    assert result.meta.get("anchor_not_found") is True
    assert f.read_text(encoding="utf-8") == "hello world\n"


# --------------------------------------------------------------------------- #
# mode selection / validation
# --------------------------------------------------------------------------- #


def test_conflicting_mode_args_rejected() -> None:
    tool = EditTool()
    # target + start_line -> line_range mode rejects target.
    err = tool.validate(
        {"path": "f", "replacement": "x", "target": "t", "start_line": 1}
    )
    assert err is not None


def test_explicit_mode_takes_precedence() -> None:
    tool = EditTool()
    # Explicit replace mode with start_line present is a conflict.
    err = tool.validate(
        {"path": "f", "replacement": "x", "target": "t", "mode": "replace",
         "start_line": 1}
    )
    assert err is not None


def test_default_replace_mode_unchanged(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("alpha beta\n", encoding="utf-8")
    result = EditTool().run(
        {"path": str(f), "target": "beta", "replacement": "GAMMA"},
        _ctx(tmp_path),
    )
    assert result.ok is True
    assert f.read_text(encoding="utf-8") == "alpha GAMMA\n"


def test_anchored_requires_an_anchor() -> None:
    tool = EditTool()
    err = tool.validate(
        {"path": "f", "replacement": "x", "target": "t", "mode": "anchored"}
    )
    assert err is not None
