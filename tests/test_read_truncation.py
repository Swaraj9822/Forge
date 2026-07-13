"""Property-based test for the read tool's truncation cap.

# Feature: forge, Property 9: Read truncation cap

Property 9 (Validates: Requirements 5.7): For any file content, the ``read``
tool returns at most ``read_max_lines`` lines and at most ``read_max_bytes``
bytes, and flags the result as ``truncated`` whenever the file exceeds EITHER
bound.

How the implementation caps (mirrored exactly below)
----------------------------------------------------
:func:`forge.tools.fs._cap_content` truncates in two stages:

1. **Line cap** - the decoded ``str.splitlines(keepends=True)`` list is sliced
   to the first ``max_lines`` lines. If the original had more lines than the
   cap, the result is flagged truncated.
2. **Byte cap** - the (already line-capped) joined text is UTF-8 encoded and,
   if longer than ``max_bytes``, truncated to that byte budget on a UTF-8
   boundary (a trailing partial multi-byte sequence is dropped). If byte
   truncation applies, the result is flagged truncated.

Because the byte cap is applied to the *line-capped* text, the "exceeds either
bound" condition for the truncated flag is computed here the SAME way the
implementation decides it:

    expected_truncated = (original_lines > max_lines) OR
                         (bytes(first max_lines lines) > max_bytes)

Small caps (``max_lines`` in 1..20, ``max_bytes`` in 1..200) are generated so
the files needed to cross the caps stay tiny and the test stays fast, while
still exercising "under both", "over lines", "over bytes", and "over both".

Each example uses its own ``tempfile.TemporaryDirectory`` workspace created and
cleaned up inside the test body (avoiding the host's pytest ``tmp_path``
issues), and a safe lowercase filename that is never a Windows-reserved name.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.config import Config
from forge.interrupt import InterruptController
from forge.tools.base import ToolContext
from forge.tools.fs import ReadTool

# A "line" of text with NO embedded newline characters so joining lines with
# "\n" yields exactly that many lines. Carriage returns are excluded so the
# universal-newline splitting in ``splitlines`` cannot introduce extra splits;
# NUL is excluded so the tool does not classify the file as binary (Req 5.6);
# surrogates (category "Cs") are excluded so the content always encodes as
# valid UTF-8. A small share of multi-byte characters is reachable so the
# byte-boundary truncation path is exercised, with byte length always computed
# via ``encode`` rather than assuming bytes == chars.
_line_text = st.text(
    alphabet=st.characters(
        blacklist_characters="\n\r\x00",
        blacklist_categories=("Cs",),
    ),
    min_size=0,
    max_size=30,
)

# 0..30 lines keeps files tiny while still able to exceed a 1..20 line cap.
_lines = st.lists(_line_text, min_size=0, max_size=30)


@st.composite
def _content_and_caps(draw: st.DrawFn) -> tuple[str, int, int]:
    """Draw (file_content, max_lines, max_bytes) with small caps.

    The content is the drawn lines joined with "\\n"; a trailing newline is
    added on roughly half of examples so both the terminated and unterminated
    final-line shapes are exercised. Caps are intentionally small so generated
    content readily lands under, over-by-lines, over-by-bytes, or over-by-both.
    """
    lines = draw(_lines)
    content = "\n".join(lines)
    if draw(st.booleans()):
        content += "\n"

    max_lines = draw(st.integers(min_value=1, max_value=20))
    max_bytes = draw(st.integers(min_value=1, max_value=200))
    return content, max_lines, max_bytes


@settings(max_examples=10)
@given(data=_content_and_caps())
def test_read_caps_lines_and_bytes_and_flags_truncation(
    data: tuple[str, int, int],
) -> None:
    """Read output honors both caps and flags truncation iff a bound is crossed.

    Validates: Requirements 5.7
    """
    content, max_lines, max_bytes = data

    with tempfile.TemporaryDirectory(prefix="forge_readtrunc_") as raw_dir:
        # realpath so comparisons stay stable on platforms whose temp dir
        # contains symlinks or short (8.3) names.
        workspace_root = Path(os.path.realpath(raw_dir))
        # Safe lowercase-letter filename (never a Windows-reserved name).
        file_path = workspace_root / "sample.txt"
        file_path.write_text(content, encoding="utf-8", newline="")

        # Small caps are supplied via a real Config on the ToolContext so the
        # tool reads them from ``ctx.config.read_max_lines/read_max_bytes``.
        config = Config(read_max_lines=max_lines, read_max_bytes=max_bytes)
        ctx = ToolContext(
            workspace_root=workspace_root,
            interrupt=InterruptController(),
            config=config,
        )

        result = ReadTool().run({"path": str(file_path)}, ctx)

        # The read itself must succeed; truncation is a flag, not a failure.
        assert result.ok is True
        assert result.error is None

        # Cap 1: at most max_lines lines in the returned content (counted with
        # the same keepends semantics the tool uses).
        returned_lines = result.content.splitlines(keepends=True)
        assert len(returned_lines) <= max_lines

        # Cap 2: at most max_bytes bytes in the returned content.
        assert len(result.content.encode("utf-8")) <= max_bytes

        # Expected truncation, computed exactly as the implementation decides:
        # the original exceeded the line cap, OR the line-capped text still
        # exceeded the byte cap.
        original_lines = content.splitlines(keepends=True)
        line_capped_text = "".join(original_lines[:max_lines])
        expected_truncated = (len(original_lines) > max_lines) or (
            len(line_capped_text.encode("utf-8")) > max_bytes
        )

        if expected_truncated:
            assert result.meta.get("truncated") is True
        else:
            # No bound crossed: the result must not be flagged truncated, and
            # the returned content is exactly the original file content.
            assert result.meta.get("truncated") is not True
            assert result.content == content


def test_read_truncation_flagged_when_line_cap_exceeded() -> None:
    """Example: a file with more lines than the cap is truncated by lines.

    Validates: Requirements 5.7
    """
    with tempfile.TemporaryDirectory(prefix="forge_readtrunc_ex_") as raw_dir:
        workspace_root = Path(os.path.realpath(raw_dir))
        file_path = workspace_root / "many.txt"
        # 10 short lines, cap at 3 lines.
        content = "\n".join(f"line{i}" for i in range(10))
        file_path.write_text(content, encoding="utf-8", newline="")

        config = Config(read_max_lines=3, read_max_bytes=1_000_000)
        ctx = ToolContext(
            workspace_root=workspace_root,
            interrupt=InterruptController(),
            config=config,
        )
        result = ReadTool().run({"path": str(file_path)}, ctx)

        assert result.ok is True
        assert result.meta.get("truncated") is True
        assert len(result.content.splitlines(keepends=True)) == 3


def test_read_truncation_flagged_when_byte_cap_exceeded() -> None:
    """Example: a single long line over the byte cap is truncated by bytes.

    Validates: Requirements 5.7
    """
    with tempfile.TemporaryDirectory(prefix="forge_readtrunc_ex_") as raw_dir:
        workspace_root = Path(os.path.realpath(raw_dir))
        file_path = workspace_root / "long.txt"
        content = "a" * 500  # one line, 500 bytes
        file_path.write_text(content, encoding="utf-8", newline="")

        config = Config(read_max_lines=2_000, read_max_bytes=100)
        ctx = ToolContext(
            workspace_root=workspace_root,
            interrupt=InterruptController(),
            config=config,
        )
        result = ReadTool().run({"path": str(file_path)}, ctx)

        assert result.ok is True
        assert result.meta.get("truncated") is True
        assert len(result.content.encode("utf-8")) <= 100


def test_read_not_truncated_when_under_both_caps() -> None:
    """Example: content under both caps is returned whole and not flagged.

    Validates: Requirements 5.7
    """
    with tempfile.TemporaryDirectory(prefix="forge_readtrunc_ex_") as raw_dir:
        workspace_root = Path(os.path.realpath(raw_dir))
        file_path = workspace_root / "small.txt"
        content = "hello\nworld"
        file_path.write_text(content, encoding="utf-8", newline="")

        config = Config(read_max_lines=10, read_max_bytes=100)
        ctx = ToolContext(
            workspace_root=workspace_root,
            interrupt=InterruptController(),
            config=config,
        )
        result = ReadTool().run({"path": str(file_path)}, ctx)

        assert result.ok is True
        assert result.meta.get("truncated") is not True
        assert result.content == content


def test_read_truncated_at_default_2000_line_cap() -> None:
    """Example: the documented default 2,000-line cap truncates a longer file.

    The property above exercises the cap logic with small configured caps for
    speed. This example pins the *documented* default line cap (Requirement 5.7
    and ``DEFAULT_READ_MAX_LINES``) by reading a file with more than 2,000 lines
    using a ToolContext with no config, so the tool falls back to its defaults.

    Validates: Requirements 5.7
    """
    with tempfile.TemporaryDirectory(prefix="forge_readtrunc_def_") as raw_dir:
        workspace_root = Path(os.path.realpath(raw_dir))
        file_path = workspace_root / "big.txt"
        # 2,500 short lines: comfortably over the 2,000-line default and well
        # under the 1 MB byte default, so the LINE cap is what trips.
        content = "".join(f"line{i}\n" for i in range(2_500))
        file_path.write_text(content, encoding="utf-8", newline="")

        # No config => the tool uses DEFAULT_READ_MAX_LINES / DEFAULT_READ_MAX_BYTES.
        ctx = ToolContext(
            workspace_root=workspace_root,
            interrupt=InterruptController(),
        )
        result = ReadTool().run({"path": str(file_path)}, ctx)

        assert result.ok is True
        assert result.error is None
        assert result.meta.get("truncated") is True
        assert len(result.content.splitlines(keepends=True)) <= 2_000


def test_read_truncated_at_default_1mb_byte_cap() -> None:
    """Example: the documented default 1 MB byte cap truncates a large file.

    A single line larger than 1 MB stays under the 2,000-line cap, so the BYTE
    cap is the bound that trips, pinning ``DEFAULT_READ_MAX_BYTES``.

    Validates: Requirements 5.7
    """
    with tempfile.TemporaryDirectory(prefix="forge_readtrunc_def_") as raw_dir:
        workspace_root = Path(os.path.realpath(raw_dir))
        file_path = workspace_root / "huge.txt"
        # One line of ~1.5 MB of ASCII: one line (under the line cap) but well
        # over the 1,000,000-byte default cap.
        content = "a" * 1_500_000
        file_path.write_text(content, encoding="utf-8", newline="")

        ctx = ToolContext(
            workspace_root=workspace_root,
            interrupt=InterruptController(),
        )
        result = ReadTool().run({"path": str(file_path)}, ctx)

        assert result.ok is True
        assert result.error is None
        assert result.meta.get("truncated") is True
        assert len(result.content.encode("utf-8")) <= 1_000_000


def test_read_default_caps_not_truncated_for_modest_file() -> None:
    """Example: a modest file under both default caps is returned whole.

    Validates: Requirements 5.7
    """
    with tempfile.TemporaryDirectory(prefix="forge_readtrunc_def_") as raw_dir:
        workspace_root = Path(os.path.realpath(raw_dir))
        file_path = workspace_root / "modest.txt"
        content = "".join(f"line{i}\n" for i in range(100))
        file_path.write_text(content, encoding="utf-8", newline="")

        ctx = ToolContext(
            workspace_root=workspace_root,
            interrupt=InterruptController(),
        )
        result = ReadTool().run({"path": str(file_path)}, ctx)

        assert result.ok is True
        assert result.meta.get("truncated") is not True
        assert result.content == content
