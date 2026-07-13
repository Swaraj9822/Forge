"""Property-based test for the read tool's invalid line-range handling.

# Feature: forge, Property 8: Invalid line range rejected

Property 8 (Validates: Requirements 5.5): For any line range whose start is
below 1, whose end exceeds the file's last line, or whose start exceeds its
end, the read tool returns an "invalid range" Tool_Result (``ok`` is ``False``
and ``meta["invalid_range"]`` is ``True``).

Because the read tool supplies defaults for an omitted endpoint (start -> 1,
end -> last line), an omitted endpoint can mask an invalid one. To force the
invalidity reliably this test always supplies BOTH ``start_line`` and
``end_line``.

Each Hypothesis example writes its own file under a per-example temporary
directory created with :mod:`tempfile` (the pytest ``tmp_path`` fixture is not
combined with ``@given`` here, both to avoid Hypothesis's function-scoped
fixture warning and because ``tmp_path`` is unreliable on this host). The
directory is removed in a ``finally`` block after each example.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from forge.interrupt import InterruptController
from forge.tools.base import ToolContext
from forge.tools.fs import ReadTool

# A fixed, non-reserved file name reused per example.
_FILE_NAME = "sample.txt"


@st.composite
def _file_size_and_invalid_range(draw: st.DrawFn) -> tuple[int, int, int]:
    """Draw ``(n_lines, start, end)`` where ``(start, end)`` is invalid for a
    file of ``n_lines`` lines.

    The three invalidity modes from Requirement 5.5 are covered:

    * mode "start_below_1": ``start <= 0`` (with ``end`` within ``[1, n]``)
    * mode "end_beyond_last": ``end > n`` (with ``start`` within ``[1, n]``)
    * mode "start_gt_end": ``start > end`` (both within ``[1, n]``)
    """

    # Mode "start_gt_end" needs at least two lines to place start > end with
    # both endpoints inside [1, n]; allow it only when n >= 2.
    n = draw(st.integers(min_value=1, max_value=50))

    modes = ["start_below_1", "end_beyond_last"]
    if n >= 2:
        modes.append("start_gt_end")
    mode = draw(st.sampled_from(modes))

    if mode == "start_below_1":
        start = draw(st.integers(min_value=-20, max_value=0))
        end = draw(st.integers(min_value=1, max_value=n))
    elif mode == "end_beyond_last":
        start = draw(st.integers(min_value=1, max_value=n))
        end = draw(st.integers(min_value=n + 1, max_value=n + 20))
    else:  # start_gt_end
        start = draw(st.integers(min_value=2, max_value=n))
        end = draw(st.integers(min_value=1, max_value=start - 1))

    return n, start, end


@settings(max_examples=10)
@given(case=_file_size_and_invalid_range())
def test_invalid_line_range_rejected(case: tuple[int, int, int]) -> None:
    """An invalid (start, end) range yields an invalid-range Tool_Result."""
    n, start, end = case

    # Guard: the generated pair must actually be invalid per Requirement 5.5
    # (start < 1 OR end > n OR start > end). The strategy guarantees this, but
    # asserting it via ``assume`` documents the contract and discards any
    # accidentally-valid pair rather than mis-testing it.
    assume(start < 1 or end > n or start > end)

    workspace = Path(tempfile.mkdtemp(prefix="forge_read_range_"))
    try:
        file_path = workspace / _FILE_NAME
        file_path.write_text(
            "".join(f"line {i}\n" for i in range(1, n + 1)),
            encoding="utf-8",
        )

        ctx = ToolContext(
            workspace_root=workspace,
            interrupt=InterruptController(),
        )

        result = ReadTool().run(
            {"path": _FILE_NAME, "start_line": start, "end_line": end},
            ctx,
        )

        assert result.ok is False
        assert result.meta.get("invalid_range") is True
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
