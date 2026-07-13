"""Property-based test for Verification_Feedback formatting.

# Feature: auto-verification-loop, Property 7: Verification_Feedback includes the failure details

The ``format_feedback`` renderer is pure and offline: it maps a command string
and a non-passing :class:`VerificationResult` to deterministic feedback text. No
network or process I/O is involved.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.verification import (
    FAILED,
    START_ERROR,
    TIMED_OUT,
    VerificationResult,
    format_feedback,
)

# Commands and outputs span ordinary text, whitespace-only, empty, and
# non-ASCII content so the renderer is exercised across the input space.
TEXT = st.text(min_size=0, max_size=200)

# A non-passing outcome status: passed never produces feedback.
NON_PASSING = st.sampled_from([FAILED, TIMED_OUT, START_ERROR])

# Exit codes are either absent (None -> "unavailable") or arbitrary integers.
EXIT_CODES = st.one_of(st.none(), st.integers(min_value=-256, max_value=256))


@st.composite
def feedback_cases(draw: st.DrawFn):
    """Generate (command, VerificationResult) for a non-passing result."""

    command = draw(TEXT)
    result = VerificationResult(
        outcome=draw(NON_PASSING),
        exit_code=draw(EXIT_CODES),
        output=draw(TEXT),
        truncated=draw(st.booleans()),
    )
    return command, result


@settings(max_examples=200)
@given(feedback_cases())
def test_feedback_includes_failure_details(case) -> None:
    """For any non-passing VerificationResult and any command, the formatted
    feedback includes the command, the outcome status, the exit code (or an
    explicit ``unavailable`` marker when the exit code is None), and the
    captured output, and indicates truncation exactly when the result is
    truncated.

    Validates: Requirements 7.1, 7.2
    """

    command, result = case
    feedback = format_feedback(command, result)

    # Req 7.1: the Verify_Command appears in the feedback.
    assert f"Command: {command}" in feedback

    # Req 7.1: the classified outcome status appears.
    assert f"Status: {result.outcome}" in feedback

    # Req 7.1: the exit code appears when available, else an explicit marker.
    if result.exit_code is None:
        assert "Exit code: unavailable" in feedback
    else:
        assert f"Exit code: {result.exit_code}" in feedback

    # Req 7.1: the captured combined output is included, rendered last and
    # verbatim, so the feedback ends with it.
    assert feedback.endswith(result.output)

    # Req 7.2: the output header indicates truncation exactly when truncated.
    # Strip the verbatim output from the end (robust to newlines within the
    # command or output) so the header line is checked at its real position
    # rather than anywhere the random output text might coincidentally match.
    head = feedback[: len(feedback) - len(result.output)]
    if result.truncated:
        assert head.endswith("Output (truncated):\n")
    else:
        assert head.endswith("Output:\n")
        assert not head.endswith("Output (truncated):\n")
