"""Property-based tests for the codebase search tool (``SearchTool``).

This module hosts three Hypothesis properties from the Forge design's
"Correctness Properties" section, each tagged with the required
``# Feature: forge, Property {n}: ...`` comment and configured for at least 100
iterations (``@settings(max_examples=10)``):

* **Property 13: Search match correctness** (Validates: Requirements 8.1) -
  every content-search result names a real Workspace file, a 1-based line
  number, and a line that actually contains a regex match at that location.
* **Property 14: Search result and line caps** (Validates: Requirements 8.4,
  8.6) - content search returns at most the configured result limit (flagging
  truncation when exceeded) and each returned line is at most the configured
  line cap (flagging line truncation when exceeded).
* **Property 15: Glob correctness** (Validates: Requirements 8.2) - glob search
  returns exactly the set of Workspace paths matching the glob.

Each generated example builds its own throwaway workspace with
``tempfile.TemporaryDirectory`` (created and cleaned up inside the test body).
This mirrors the convention used by ``tests/test_read_line_range.py`` and avoids
combining a function-scoped pytest ``tmp_path`` fixture with Hypothesis
``@given`` while still giving every example a fresh per-example workspace.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from string import ascii_lowercase

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.config import Config
from forge.interrupt import InterruptController
from forge.tools.base import ToolContext
from forge.tools.search import SearchTool

# Windows reserves a handful of device names regardless of extension; a random
# lowercase-letter stem could land on one of the letters-only reserved names, so
# they are filtered out of generated file names to keep file creation portable.
_RESERVED_STEMS = frozenset({"con", "prn", "aux", "nul"})

# A file-name stem of lowercase ASCII letters (never a reserved device name) and
# a small set of plausible extensions. Kept short so workspaces stay tiny.
_stem = st.text(alphabet=ascii_lowercase, min_size=1, max_size=6).filter(
    lambda s: s not in _RESERVED_STEMS
)
_extension = st.sampled_from([".txt", ".py", ".md"])
_file_name = st.builds(lambda stem, ext: stem + ext, _stem, _extension)


def _make_ctx(root: Path) -> ToolContext:
    """Build a ToolContext rooted at ``root`` with the default Config caps."""
    return ToolContext(
        workspace_root=root,
        interrupt=InterruptController(),
        config=Config(),
    )


def _write_file(root: Path, rel: str, content: str) -> None:
    """Write ``content`` to ``root/rel`` creating parents, with no newline
    translation so the bytes on disk match ``content`` exactly (the search
    engine reads bytes and decodes UTF-8)."""
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="")


# ---------------------------------------------------------------------------
# Property 13: Search match correctness
# ---------------------------------------------------------------------------

# Line text drawn from a tiny alphabet that overlaps the candidate patterns so
# matches occur frequently; no newline characters (lines are joined with "\n")
# and no NUL (which would make the file look binary and be skipped).
_line_text = st.text(alphabet="ab xyz", min_size=0, max_size=24)
_file_content = st.builds(
    lambda lines: "\n".join(lines),
    st.lists(_line_text, min_size=0, max_size=8),
)
# Literal patterns that are also valid regular expressions; chosen to overlap
# the line alphabet so content matches are common.
_content_pattern = st.sampled_from(["a", "b", "ab", "ba", "x", "y", "aa"])


@settings(max_examples=10, deadline=None)
@given(
    files=st.dictionaries(_file_name, _file_content, min_size=1, max_size=4),
    pattern=_content_pattern,
)
def test_content_search_results_are_real_matches(
    files: dict[str, str], pattern: str
) -> None:
    """Every content-search result points at a genuine regex match.

    # Feature: forge, Property 13: Search match correctness

    Validates: Requirements 8.1
    """
    regex = re.compile(pattern)

    with tempfile.TemporaryDirectory(prefix="forge_search13_") as raw_dir:
        root = Path(os.path.realpath(raw_dir))
        for name, content in files.items():
            _write_file(root, name, content)

        result = SearchTool().run({"mode": "content", "pattern": pattern}, _make_ctx(root))

        assert result.ok is True
        assert result.meta["mode"] == "content"

        results = result.meta["results"]
        # Reported match count matches the number of structured results.
        assert result.meta["matches"] == len(results)

        for entry in results:
            rel = entry["path"]
            file_path = root / rel
            # Names a real file inside the workspace.
            assert file_path.is_file()
            assert file_path.resolve().is_relative_to(root)

            # Names a 1-based line number within that file.
            file_lines = file_path.read_text(encoding="utf-8").splitlines()
            line_no = entry["line"]
            assert isinstance(line_no, int)
            assert 1 <= line_no <= len(file_lines)

            # The original line at that location actually contains a match.
            original_line = file_lines[line_no - 1]
            assert regex.search(original_line) is not None


# ---------------------------------------------------------------------------
# Property 14: Search result and line caps
# ---------------------------------------------------------------------------

# A fixed single-character token present on every generated line, so the number
# of matching lines equals the number of lines and is easy to reason about.
_CAP_TOKEN = "M"


@st.composite
def _capped_lines(draw: st.DrawFn) -> list[str]:
    """Draw a list of lines that each contain ``_CAP_TOKEN``.

    The number of lines straddles the 100-match result limit and individual
    line lengths straddle the 500-character line cap, so a single property
    exercises both truncation behaviours.
    """
    num_lines = draw(st.integers(min_value=1, max_value=205))
    lines: list[str] = []
    for _ in range(num_lines):
        # Pad length straddles the 500-char cap (token + up to 600 filler).
        pad = draw(st.integers(min_value=0, max_value=600))
        lines.append(_CAP_TOKEN + ("a" * pad))
    return lines


@settings(max_examples=10, deadline=None)
@given(lines=_capped_lines())
def test_content_search_respects_result_and_line_caps(lines: list[str]) -> None:
    """Content search caps match count and line length and flags truncation.

    # Feature: forge, Property 14: Search result and line caps

    Validates: Requirements 8.4, 8.6
    """
    config = Config()
    result_limit = config.search_result_limit  # documented default: 100
    line_cap = config.search_line_cap          # documented default: 500

    with tempfile.TemporaryDirectory(prefix="forge_search14_") as raw_dir:
        root = Path(os.path.realpath(raw_dir))
        # A single file keeps the matching-line count exactly len(lines).
        _write_file(root, "haystack.txt", "\n".join(lines))

        ctx = ToolContext(
            workspace_root=root,
            interrupt=InterruptController(),
            config=config,
        )
        result = SearchTool().run({"mode": "content", "pattern": _CAP_TOKEN}, ctx)

        assert result.ok is True
        results = result.meta["results"]

        # Result cap (Req 8.4): at most ``result_limit`` matches, with the
        # truncated flag set exactly when there were more matching lines.
        assert result.meta["matches"] <= result_limit
        assert len(results) <= result_limit
        assert result.meta["truncated"] is (len(lines) > result_limit)

        # Returned results correspond to the first lines of the single file, in
        # order, so each can be checked against its original line.
        any_line_truncated = False
        for offset, entry in enumerate(results):
            assert entry["line"] == offset + 1
            original_line = lines[offset]
            expected_truncated = len(original_line) > line_cap

            # Line cap (Req 8.6): every returned line is within the cap and the
            # per-result flag matches whether the original exceeded it.
            assert len(entry["text"]) <= line_cap
            assert entry["line_truncated"] is expected_truncated
            if expected_truncated:
                assert entry["text"] == original_line[:line_cap]
                assert len(entry["text"]) == line_cap
            else:
                assert entry["text"] == original_line
            any_line_truncated = any_line_truncated or expected_truncated

        # The aggregate line-truncation flag reflects the individual results.
        assert result.meta.get("line_truncated", False) is any_line_truncated


# ---------------------------------------------------------------------------
# Property 15: Glob correctness
# ---------------------------------------------------------------------------

# A curated set of globs spanning top-level, recursive, extension, and
# subdirectory patterns, plus a deliberate non-matching pattern.
_glob_pattern = st.sampled_from(
    [
        "*",
        "*.py",
        "*.txt",
        "*.md",
        "**/*",
        "**/*.py",
        "**/*.txt",
        "sub/*",
        "sub/*.py",
        "*.rs",
        "zzznomatch*",
    ]
)


@settings(max_examples=10, deadline=None)
@given(
    root_files=st.dictionaries(_file_name, st.just(""), max_size=4),
    sub_files=st.dictionaries(_file_name, st.just(""), max_size=4),
    pattern=_glob_pattern,
)
def test_glob_returns_exactly_matching_workspace_paths(
    root_files: dict[str, str],
    sub_files: dict[str, str],
    pattern: str,
) -> None:
    """Glob search returns exactly the set of paths matching the glob.

    # Feature: forge, Property 15: Glob correctness

    Validates: Requirements 8.2
    """
    with tempfile.TemporaryDirectory(prefix="forge_search15_") as raw_dir:
        root = Path(os.path.realpath(raw_dir))
        for name, content in root_files.items():
            _write_file(root, name, content)
        for name, content in sub_files.items():
            _write_file(root, f"sub/{name}", content)

        result = SearchTool().run({"mode": "glob", "glob": pattern}, _make_ctx(root))

        # The Workspace-relative paths matching the glob under pathlib
        # semantics (the design's definition of "matching the glob").
        expected = {
            p.relative_to(root).as_posix() for p in root.glob(pattern)
        }

        assert result.ok is True
        assert result.meta["mode"] == "glob"
        assert set(result.meta["results"]) == expected
        assert result.meta["matches"] == len(expected)
