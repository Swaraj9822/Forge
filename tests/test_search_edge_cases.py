"""Example-based unit tests for the search tool's edge cases.

These cover two error/empty paths from Requirement 8 that the property tests do
not target directly:

* **No matches** (Requirement 8.3) - a content or glob search that finds nothing
  returns a successful ``ToolResult`` that reports zero matches and indicates no
  matches were found, rather than an error.
* **Invalid regular expression** (Requirement 8.5) - a content search whose
  pattern is not a valid regular expression returns a ``ToolResult`` describing
  the pattern as invalid (a run-time result, not a validation failure).

Each test uses pytest's ``tmp_path`` fixture directly for a real, isolated
workspace; these are concrete examples (not Hypothesis ``@given`` tests), so the
function-scoped fixture is safe to use here.
"""

from __future__ import annotations

import os
from pathlib import Path

from forge.config import Config
from forge.interrupt import InterruptController
from forge.tools.base import ToolContext
from forge.tools.search import SearchTool


def _ctx(tmp_path: Path) -> ToolContext:
    """A ToolContext rooted at a realpath'd ``tmp_path`` with default caps."""
    root = Path(os.path.realpath(tmp_path))
    return ToolContext(
        workspace_root=root,
        interrupt=InterruptController(),
        config=Config(),
    )


# ---------------------------------------------------------------------------
# Requirement 8.3: no-matches result
# ---------------------------------------------------------------------------


def test_content_search_with_no_matches_reports_no_matches(tmp_path: Path) -> None:
    """Content search finding nothing succeeds and reports zero matches.

    Validates: Requirements 8.3
    """
    (tmp_path / "file.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    result = SearchTool().run(
        {"mode": "content", "pattern": "this-string-is-absent"}, _ctx(tmp_path)
    )

    assert result.ok is True
    assert result.error is None
    assert result.content == "No matches found."
    assert result.meta["mode"] == "content"
    assert result.meta["matches"] == 0
    assert result.meta["results"] == []
    assert result.meta["truncated"] is False


def test_glob_search_with_no_matches_reports_no_matches(tmp_path: Path) -> None:
    """Glob search finding nothing succeeds and reports zero matches.

    Validates: Requirements 8.3
    """
    (tmp_path / "file.txt").write_text("contents", encoding="utf-8")

    result = SearchTool().run(
        {"mode": "glob", "glob": "*.no-such-extension"}, _ctx(tmp_path)
    )

    assert result.ok is True
    assert result.error is None
    assert result.content == "No matches found."
    assert result.meta["mode"] == "glob"
    assert result.meta["matches"] == 0
    assert result.meta["results"] == []


# ---------------------------------------------------------------------------
# Requirement 8.5: invalid-regex error result
# ---------------------------------------------------------------------------


def test_content_search_with_invalid_regex_returns_invalid_pattern(
    tmp_path: Path,
) -> None:
    """An invalid regex yields an error result describing the pattern as invalid.

    Validates: Requirements 8.5
    """
    (tmp_path / "file.txt").write_text("anything", encoding="utf-8")

    # An unclosed character class is not a valid regular expression.
    result = SearchTool().run({"mode": "content", "pattern": "["}, _ctx(tmp_path))

    assert result.ok is False
    assert result.error == "invalid pattern"
    assert result.meta.get("invalid_pattern") is True
    # A detail string describing the regex error is included for the Model.
    assert isinstance(result.meta.get("detail"), str)
    assert result.meta["detail"] != ""


def test_invalid_regex_is_a_runtime_result_not_a_validation_error(
    tmp_path: Path,
) -> None:
    """Regex validity is enforced at run time, not by ``validate``.

    ``validate`` only checks argument shape, so a syntactically-invalid pattern
    passes validation and is reported by ``run`` instead (Requirement 8.5).
    """
    tool = SearchTool()

    # Shape is valid: a mode of "content" with a non-empty pattern string.
    assert tool.validate({"mode": "content", "pattern": "("}) is None

    result = tool.run({"mode": "content", "pattern": "("}, _ctx(tmp_path))
    assert result.ok is False
    assert result.meta.get("invalid_pattern") is True
