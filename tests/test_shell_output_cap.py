"""Property test for the shell tool's combined-output character cap.

Property 12 (shell portion) from the design: *for any* command output, the
shell tool returns at most the configured cap of characters and flags the
result as truncated whenever the rendered output exceeds that cap
(Requirement 7.5 -- the documented default cap is 30,000 characters).

Strategy
--------
We drive a deterministic, exactly-known amount of output through the *real*
shell tool by writing a file of ``n`` ASCII bytes into the workspace root and
dumping it with the platform's file-dump built-in (``type`` on Windows,
``cat`` on POSIX). Using a no-space filename relative to the workspace ``cwd``
keeps the command robust across both shells with no quoting hazards, and the
dump built-ins emit the file's bytes verbatim (no added newline), so the
captured stdout is exactly ``"a" * n``.

The tool wraps stdout/stderr/exit-code into a rendered string and then applies
the cap, so the reference "full" (un-truncated) output is the pure, deterministic
``_render("a" * n, "", 0)``. We vary both the output length and the cap so that
examples straddle the cap in both directions, exercising the truncated and the
non-truncated branches. This is a white-box property test: it reuses the tool's
own ``_render`` helper to compute the expected un-truncated rendering.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from forge.interrupt import InterruptController
from forge.tools.base import ToolContext
from forge.tools.shell import ShellTool, _render

IS_WINDOWS = sys.platform == "win32" or os.name == "nt"


def _dump_command(filename: str) -> str:
    """Return a shell command that writes ``filename``'s bytes to stdout."""
    # `type` (cmd) and `cat` (sh) both emit the file's contents verbatim.
    return f"type {filename}" if IS_WINDOWS else f"cat {filename}"


# Feature: forge, Property 12: Shell and git output char cap
@settings(
    max_examples=10,
    deadline=None,  # subprocess spawn latency varies; a per-example deadline is flaky
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    n=st.integers(min_value=0, max_value=500),
    cap=st.integers(min_value=0, max_value=600),
)
def test_shell_output_is_capped_and_flags_truncation(tmp_path, n, cap):
    """The shell tool never returns more than ``cap`` chars and flags
    truncation exactly when the full rendered output exceeds the cap.

    **Validates: Requirements 7.5**
    """
    # Write exactly n ASCII bytes (no newlines -> no platform translation).
    data = tmp_path / "data.txt"
    data.write_bytes(b"a" * n)

    config = SimpleNamespace(shell_timeout_s=120, output_cap_chars=cap)
    ctx = ToolContext(
        workspace_root=tmp_path,
        interrupt=InterruptController(),
        config=config,
    )
    tool = ShellTool()

    result = tool.run({"command": _dump_command("data.txt")}, ctx)

    # Sanity: dumping an existing file is a successful (exit 0) command, so the
    # reference rendering uses stdout="a"*n, empty stderr, and exit code 0.
    assert result.meta.get("exit_code") == 0

    full = _render("a" * n, "", 0)
    exceeds_cap = len(full) > cap

    # Core cap guarantee: the returned content never exceeds the cap (Req 7.5).
    assert len(result.content) <= cap

    if exceeds_cap:
        # Output beyond the cap is truncated and flagged.
        assert result.meta.get("truncated") is True
        assert result.content == full[:cap]
    else:
        # Output within the cap is returned whole, with no truncation flag.
        assert result.meta.get("truncated") is None
        assert result.content == full


# --------------------------------------------------------------------------- #
# Bounded-drain memory safety (safety hardening)
# --------------------------------------------------------------------------- #


def test_drain_ceiling_scales_and_floors() -> None:
    """The per-stream drain ceiling scales 2x with the cap but never drops
    below the floor, so retained bytes always exceed the presentation cap."""
    from forge.tools.shell import _DRAIN_CEILING_FLOOR, _drain_ceiling

    # Tiny/zero caps still retain the floor's worth of bytes.
    assert _drain_ceiling(0) == _DRAIN_CEILING_FLOOR
    assert _drain_ceiling(10) == _DRAIN_CEILING_FLOOR
    # A large cap scales to 2x so truncation detection still trips.
    assert _drain_ceiling(_DRAIN_CEILING_FLOOR) == 2 * _DRAIN_CEILING_FLOOR
    # Non-int inputs degrade to the floor rather than raising.
    assert _drain_ceiling(None) == _DRAIN_CEILING_FLOOR  # type: ignore[arg-type]


def test_drain_bounds_retained_bytes_but_reads_to_eof() -> None:
    """``_drain`` retains at most ``byte_ceiling`` bytes yet still consumes the
    whole stream (so the child's pipe never blocks), bounding memory for a
    runaway command."""
    import io

    from forge.tools.shell import _drain

    class _CountingStream(io.BytesIO):
        """BytesIO that records how many bytes were read before close."""

        def __init__(self, data: bytes) -> None:
            super().__init__(data)
            self.read_total = 0

        def read(self, size: int = -1) -> bytes:
            chunk = super().read(size)
            self.read_total += len(chunk)
            return chunk

    # 100 KB of data with a 4 KB ceiling: draining reads all of it but keeps
    # only a bounded prefix.
    payload = b"x" * 100_000
    ceiling = 4_096
    stream = _CountingStream(payload)
    sink: list[bytes] = []

    _drain(stream, sink, ceiling)

    retained = b"".join(sink)
    # Memory is bounded: we retained on the order of the ceiling (one final
    # 4 KB chunk may push us just over it), never the full 100 KB.
    assert len(retained) < 2 * ceiling
    assert len(retained) >= ceiling
    # The whole stream was consumed to EOF (the pipe would not have blocked).
    assert stream.read_total == len(payload)


def test_drain_without_ceiling_retains_everything() -> None:
    """With no ceiling, ``_drain`` keeps its existing behavior (retain all)."""
    import io

    from forge.tools.shell import _drain

    payload = b"y" * 20_000
    sink: list[bytes] = []
    _drain(io.BytesIO(payload), sink)
    assert b"".join(sink) == payload
