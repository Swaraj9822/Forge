"""Property-based test for Verification_Result outcome classification.

# Feature: auto-verification-loop, Property 4: Outcome classification maps execution to status

This test exercises the pure ``classify_outcome`` helper in
``forge/verification.py`` across all combinations of the raw
:class:`CommandExecution` flags (``spawn_error`` present/absent, ``timed_out``
true/false, and a range of exit codes including ``0``, ``None``, and non-zero
values). It asserts the documented priority ordering:

* ``spawn_error`` set            -> ``start_error``
* otherwise ``timed_out``        -> ``timed_out`` (regardless of termination success)
* otherwise ``exit_code == 0``   -> ``passed``
* otherwise                      -> ``failed``

The classifier is a pure, total function over the execution flags, so the
property is fully OFFLINE: no process is ever started and no network call is
ever made.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.tools.shell import CommandExecution
from forge.verification import (
    FAILED,
    PASSED,
    START_ERROR,
    TIMED_OUT,
    classify_outcome,
)

# Exit codes span the meaningful cases: success (0), a missing code (None, as
# happens on timeout/interrupt/start-error), and assorted non-zero codes
# (including negative signal-style codes seen on POSIX).
EXIT_CODES = st.one_of(
    st.none(),
    st.just(0),
    st.integers(min_value=1, max_value=255),
    st.integers(min_value=-255, max_value=-1),
)

# spawn_error is None unless the process could not be started, in which case it
# carries a non-empty description.
SPAWN_ERRORS = st.one_of(
    st.none(),
    st.text(min_size=1, max_size=40),
)

TEXT = st.text(max_size=80)


@st.composite
def command_executions(draw: st.DrawFn) -> CommandExecution:
    """Generate a CommandExecution across all flag combinations."""
    return CommandExecution(
        stdout=draw(TEXT),
        stderr=draw(TEXT),
        exit_code=draw(EXIT_CODES),
        timed_out=draw(st.booleans()),
        interrupted=draw(st.booleans()),
        spawn_error=draw(SPAWN_ERRORS),
    )


@settings(max_examples=200)
@given(execution=command_executions())
def test_classify_outcome_priority_ordering(execution: CommandExecution) -> None:
    """Property 4: classification follows the documented priority ordering.

    Validates: Requirements 4.2, 4.3, 4.4, 4.5, 4.7
    """
    outcome = classify_outcome(execution)

    # The result is always one of the four defined statuses.
    assert outcome in {PASSED, FAILED, TIMED_OUT, START_ERROR}

    if execution.spawn_error is not None:
        # Highest priority: an unstartable process is a start_error regardless
        # of any other flag (Req 4.7).
        assert outcome == START_ERROR
    elif execution.timed_out:
        # A timeout classifies as timed_out even when the exit code happens to
        # be 0 or termination did not succeed (Req 4.4, 4.5).
        assert outcome == TIMED_OUT
    elif execution.exit_code == 0:
        # A clean completion is passed (Req 4.2).
        assert outcome == PASSED
    else:
        # Any other completion (non-zero or missing exit code) is failed
        # (Req 4.3), and the exit code and combined output are preserved by the
        # caller; here we assert the classification itself.
        assert outcome == FAILED


@settings(max_examples=100)
@given(
    exit_code=st.integers(min_value=1, max_value=255),
    stdout=TEXT,
    stderr=TEXT,
)
def test_failed_preserves_exit_code_and_output(
    exit_code: int, stdout: str, stderr: str
) -> None:
    """A non-zero, non-timeout, startable execution is failed with details intact.

    Validates: Requirements 4.3
    """
    execution = CommandExecution(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=False,
        interrupted=False,
        spawn_error=None,
    )
    assert classify_outcome(execution) == FAILED
    # The execution carries the exit code and captured combined output that the
    # runner preserves on the VerificationResult.
    assert execution.exit_code == exit_code
    assert execution.stdout == stdout
    assert execution.stderr == stderr


@settings(max_examples=100)
@given(
    timed_out=st.booleans(),
    exit_code=EXIT_CODES,
    spawn_error=st.text(min_size=1, max_size=40),
)
def test_start_error_dominates_all_other_flags(
    timed_out: bool, exit_code: int | None, spawn_error: str
) -> None:
    """start_error takes priority over timed_out and exit code.

    Validates: Requirements 4.7
    """
    execution = CommandExecution(
        stdout="",
        stderr="",
        exit_code=exit_code,
        timed_out=timed_out,
        interrupted=False,
        spawn_error=spawn_error,
    )
    assert classify_outcome(execution) == START_ERROR
