"""Tests for richer tool-call/result rendering in the plain-text UI."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from forge.ui import Ui, describe_tool, summarize_result


@dataclass
class _Result:
    ok: bool
    content: str = ""
    error: str | None = None
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# describe_tool
# --------------------------------------------------------------------------- #


def test_describe_read_with_range():
    assert describe_tool("read", {"path": "a/b.py", "start_line": 1, "end_line": 9}) == "a/b.py:1-9"


def test_describe_read_no_range():
    assert describe_tool("read", {"path": "a/b.py"}) == "a/b.py"


def test_describe_shell():
    assert describe_tool("shell", {"command": "pytest -q"}) == "$ pytest -q"


def test_describe_search_pattern():
    assert describe_tool("search", {"pattern": "TODO"}) == '"TODO"'


def test_describe_git_with_args():
    assert describe_tool("git", {"operation": "commit", "args": ["-m", "x"]}) == "commit -m x"


def test_describe_write_path():
    assert describe_tool("write", {"path": "out.txt", "content": "hi"}) == "out.txt"


def test_describe_none_when_no_args():
    assert describe_tool("repo_index", {}) is None


def test_describe_handles_non_dict():
    assert describe_tool("read", None) is None


def test_describe_clips_long_detail():
    long = "x" * 200
    out = describe_tool("remember", {"text": long})
    assert out is not None and len(out) <= 72 and out.endswith("...")


# --------------------------------------------------------------------------- #
# summarize_result
# --------------------------------------------------------------------------- #


def test_summary_read_lines():
    assert summarize_result("read", _Result(ok=True, content="a\nb\nc")) == "3 lines"


def test_summary_single_line():
    assert summarize_result("search", _Result(ok=True, content="only")) == "1 line"


def test_summary_write_bytes():
    r = _Result(ok=True, content="", meta={"bytes_written": 1200})
    assert summarize_result("write", r) == "wrote 1200 bytes"


def test_summary_truncated_flag():
    r = _Result(ok=True, content="x\ny", meta={"truncated": True})
    assert "truncated" in summarize_result("read", r)


def test_summary_error_returns_message():
    r = _Result(ok=False, error="file not found")
    assert summarize_result("read", r) == "file not found"


def test_summary_planning():
    assert summarize_result("planning", _Result(ok=True, content="")) == "plan updated"


# --------------------------------------------------------------------------- #
# Ui plain rendering
# --------------------------------------------------------------------------- #


def test_tool_call_plain_includes_prefix_and_detail():
    ui = Ui(io.StringIO(), color=False, spinner=False)
    assert ui.tool_call("read", "a/b.py") == "\n[tool: read] a/b.py"


def test_tool_call_plain_no_detail_matches_legacy():
    ui = Ui(io.StringIO(), color=False, spinner=False)
    # Same shape as the legacy announcement so existing consumers still match.
    assert ui.tool_call("read", None) == "\n[tool: read]"


def test_tool_result_line_plain():
    ui = Ui(io.StringIO(), color=False, spinner=False)
    assert ui.tool_result_line("42 lines", "ok") == "    -> 42 lines"
