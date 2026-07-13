"""Property test for the Verification_Runner's output-cap path.

Property 5 from the auto-verification-loop design: *for any* captured combined
output and any non-negative output cap, the :class:`VerificationResult`'s
``output`` length never exceeds the cap, and the result is flagged ``truncated``
if and only if the original rendered combined output length exceeded the cap
(Requirements 4.6, 7.2).

Strategy
--------
``VerificationRunner.run`` reuses the shell core's ``_render`` + ``_cap``
helpers: it renders the captured ``stdout`` / ``stderr`` / ``exit_code`` into a
single combined string, caps that string to ``output_cap`` characters, and sets
``truncated`` when the rendered string exceeded the cap. The property is a
statement about *that cap path*, so we drive it deterministically and
cross-platform by feeding the runner a controlled
:class:`~forge.tools.shell.CommandExecution` (patching the shared
``execute_command`` the runner calls). This exercises the runner's real capping
and truncation-flagging logic over 100+ generated examples -- including
non-ASCII output and caps that land exactly at and just over the rendered
length -- without depending on flaky cross-platform process output sizing.

Two real-process example tests complement the property by driving an actual
Verify_Command through the runner with the cap sitting exactly at, and just
under, the rendered output length.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import forge.verification as verification_module
from forge.interrupt import InterruptController
from forge.tools.shell import CommandExecution, _render
from forge.verification import VerificationRunner

IS_WINDOWS = sys.platform == "win32" or os.name == "nt"


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #


@st.composite
def execution_and_cap(draw):
    """Draw a non-spawn-error execution plus an ``output_cap`` straddling it.

    The captured ``stdout`` / ``stderr`` include non-ASCII text. The execution
    flavor is either a completed run (``exit_code`` 0 or non-zero) or a timeout
    (``exit_code`` ``None``); ``spawn_error`` is always ``None`` so the runner
    takes the rendered/capped path rather than the ``start_error`` short-circuit.

    The cap is drawn from a range around the rendered combined-output length so
    examples land below, exactly at, and just over the cap boundary.
    """
    stdout = draw(st.text(max_size=300))
    stderr = draw(st.text(max_size=300))
    timed_out = draw(st.booleans())
    if timed_out:
        exit_code = None
    else:
        exit_code = draw(
            st.one_of(st.just(0), st.integers(min_value=1, max_value=255))
        )

    rendered_len = len(_render(stdout, stderr, exit_code))
    cap = draw(
        st.one_of(
            st.integers(min_value=0, max_value=rendered_len + 10),
            st.just(rendered_len),  # exactly at the cap -> not truncated
            st.just(max(rendered_len - 1, 0)),  # just over the cap -> truncated
            st.just(rendered_len + 1),  # just under the cap -> not truncated
        )
    )
    return stdout, stderr, exit_code, timed_out, cap


# --------------------------------------------------------------------------- #
# Property 5: Output capping truncates to the configured cap (Req 4.6, 7.2)
# --------------------------------------------------------------------------- #


# Feature: auto-verification-loop, Property 5: Output capping truncates to the configured cap
@settings(max_examples=200, deadline=None)
@given(payload=execution_and_cap())
def test_runner_output_never_exceeds_cap_and_flags_truncation(payload):
    """The runner's output is capped and ``truncated`` iff the render exceeded it.

    For any captured combined output and any non-negative cap, the
    :class:`VerificationResult` ``output`` length never exceeds the cap, and
    ``truncated`` is ``True`` exactly when the rendered combined output was
    longer than the cap.

    Validates: Requirements 4.6, 7.2
    """
    stdout, stderr, exit_code, timed_out, cap = payload

    execution = CommandExecution(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
        interrupted=False,
        spawn_error=None,
    )

    runner = VerificationRunner(Path("."), InterruptController())
    with mock.patch.object(
        verification_module, "execute_command", return_value=execution
    ):
        result = runner.run("verify", timeout_s=120, output_cap=cap)

    rendered_len = len(_render(stdout, stderr, exit_code))

    # Core cap guarantee: output never exceeds the configured cap.
    assert len(result.output) <= cap
    # Truncation is flagged iff the rendered combined output exceeded the cap.
    assert result.truncated == (rendered_len > cap)


# --------------------------------------------------------------------------- #
# Real-process examples: cap exactly at and just over the rendered output
# --------------------------------------------------------------------------- #


def _dump_command(filename: str) -> str:
    """Return a shell command that writes ``filename``'s bytes to stdout."""
    return f"type {filename}" if IS_WINDOWS else f"cat {filename}"


@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(n=st.integers(min_value=0, max_value=400))
def test_real_command_cap_boundaries(tmp_path, n):
    """A real Verify_Command, capped exactly at and just over its render length.

    Dumping a known ``n``-byte file yields a deterministic rendered output, so
    we can place the cap exactly at the rendered length (no truncation) and one
    character below it (truncation by exactly one char) and assert both the
    length bound and the truncation flag against the live runner.

    Validates: Requirements 4.6, 7.2
    """
    data = tmp_path / "data.txt"
    data.write_bytes(b"a" * n)
    command = _dump_command("data.txt")

    runner = VerificationRunner(tmp_path, InterruptController())

    # The dump succeeds (exit 0), so the rendered reference uses stdout="a"*n.
    full = _render("a" * n, "", 0)
    rendered_len = len(full)

    # Cap exactly at the rendered length: returned whole, not truncated.
    at_cap = runner.run(command, timeout_s=120, output_cap=rendered_len)
    assert at_cap.outcome == "passed"
    assert len(at_cap.output) <= rendered_len
    assert at_cap.truncated is False
    assert at_cap.output == full

    # Cap one char under the rendered length: truncated by exactly one char.
    just_over = runner.run(command, timeout_s=120, output_cap=rendered_len - 1)
    assert len(just_over.output) <= rendered_len - 1
    assert just_over.truncated is True
    assert just_over.output == full[: rendered_len - 1]
