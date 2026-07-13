"""Property-based test for read line-range slicing.

# Feature: forge, Property 7: Read line-range slice

Property 7 (Validates: Requirements 5.2): For any text file and any VALID line
range ``[start, end]`` (``1 <= start <= end <= last_line``), the ``read`` tool
returns exactly the lines from ``start`` through ``end`` inclusive and no
others.

The :class:`~forge.tools.fs.ReadTool` splits the decoded text with
``str.splitlines(keepends=True)`` and slices ``[start-1:end]``, so the returned
content preserves the original line endings and equals the concatenation of
exactly those original lines. The expected value is computed the SAME way so the
property is exact regardless of whether the file has a trailing newline.

Each generated example uses its own ``tempfile.TemporaryDirectory`` workspace
(created and cleaned up inside the test body). This avoids combining a
function-scoped pytest ``tmp_path`` fixture with Hypothesis ``@given`` and
sidesteps the host's ``tmp_path`` issues. Generated file content stays modest
(short lines, few of them) so the read cap (2,000 lines / 1 MB) never trips and
truncation cannot interfere with the slice comparison.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.interrupt import InterruptController
from forge.tools.base import ToolContext
from forge.tools.fs import ReadTool

# A "line" of text with NO embedded newline characters, so joining a list with
# "\n" yields exactly that many lines. Carriage returns are excluded so the
# universal-newline splitting in ``splitlines`` cannot introduce extra splits;
# NUL is excluded so the tool does not classify the file as binary (Req 5.6);
# surrogates are excluded so the content always encodes as valid UTF-8.
_line_text = st.text(
    alphabet=st.characters(
        blacklist_characters="\n\r\x00",
        blacklist_categories=("Cs",),
    ),
    min_size=0,
    max_size=40,
)

# Between 1 and 50 lines keeps the file well under the read cap so truncation
# never interferes with the slice being compared.
_lines = st.lists(_line_text, min_size=1, max_size=50)


@st.composite
def _file_and_range(draw: st.DrawFn) -> tuple[str, int, int]:
    """Draw (file_content, start_line, end_line) with a VALID 1-based range.

    The content is the lines joined with "\\n"; a trailing newline is added on
    roughly half of examples so both the "final line terminated" and
    "unterminated final line" shapes are exercised. The range satisfies
    ``1 <= start <= end <= number_of_lines``.
    """
    lines = draw(_lines)
    content = "\n".join(lines)
    if draw(st.booleans()):
        content += "\n"

    # The number of lines the implementation will actually see, computed with
    # the same splitting it uses, so a valid range here is valid there too.
    # A single empty line with no trailing newline ("") yields zero lines via
    # ``splitlines``; append a newline so the file always has >= 1 line and a
    # valid 1-based range exists.
    if not content.splitlines(keepends=True):
        content += "\n"
    line_count = len(content.splitlines(keepends=True))
    start = draw(st.integers(min_value=1, max_value=line_count))
    end = draw(st.integers(min_value=start, max_value=line_count))
    return content, start, end


@settings(max_examples=10)
@given(data=_file_and_range())
def test_read_returns_exactly_the_requested_line_range(
    data: tuple[str, int, int],
) -> None:
    """A valid range returns exactly lines start..end inclusive.

    Validates: Requirements 5.2
    """
    content, start, end = data

    with tempfile.TemporaryDirectory(prefix="forge_readrange_") as raw_dir:
        # realpath so comparisons stay stable on platforms whose temp dir
        # contains symlinks or short (8.3) names.
        workspace_root = Path(os.path.realpath(raw_dir))
        # Safe lowercase-letter filename (never a Windows-reserved name).
        file_path = workspace_root / "sample.txt"
        file_path.write_text(content, encoding="utf-8", newline="")

        ctx = ToolContext(
            workspace_root=workspace_root,
            interrupt=InterruptController(),
        )

        result = ReadTool().run(
            {"path": str(file_path), "start_line": start, "end_line": end},
            ctx,
        )

        # Expected slice computed with the SAME keepends semantics the tool uses.
        expected = "".join(content.splitlines(keepends=True)[start - 1 : end])

        assert result.ok is True
        assert result.error is None
        assert result.content == expected
        # Modest content => never truncated, so the slice is exact.
        assert result.meta.get("truncated") is not True
