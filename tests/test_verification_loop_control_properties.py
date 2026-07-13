"""Property-based tests for the bounded Verification_Phase correction loop.

# Feature: auto-verification-loop, Property 6: The correction loop is bounded and terminates correctly

These properties exercise the pure loop-control predicate
:func:`forge.verification.should_run_correction` by driving a faithful model of
the :class:`~forge.verification.VerificationCoordinator` loop over generated
sequences of Verification_Results. The model mirrors the design's control flow
(see design.md "Control flow within the phase" and
``VerificationCoordinator.run``): the initial Verify_Command runs once
unconditionally when the gate passes, then while ``should_run_correction`` holds
the coordinator appends feedback, runs a correction turn, and re-runs the
Verify_Command.

Across all generated inputs the model must satisfy Property 6:

* (a) a gated-in phase runs the initial Verify_Command exactly once before any
  Correction_Iteration;
* (b) at most ``Max_Correction_Iterations`` Correction_Iterations are performed;
* (c) zero Correction_Iterations when ``Max_Correction_Iterations`` is 0;
* (d) the phase stops immediately on the first ``passed`` result;
* (e) a ``start_error`` is non-correctable;
* (f) once an Interrupt has halted the phase, no further Correction_Iteration
  begins.

Validates: Requirements 5.1, 5.2, 5.4, 5.5, 5.6, 5.7, 6.1, 6.3, 8.4
"""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.verification import (
    FAILED,
    PASSED,
    START_ERROR,
    TIMED_OUT,
    should_run_correction,
)

# Every Verification_Result outcome status the runner can produce.
ALL_OUTCOMES = [PASSED, FAILED, TIMED_OUT, START_ERROR]
# The correctable subset that keeps the loop going.
CORRECTABLE = {FAILED, TIMED_OUT}


@dataclass
class LoopTrace:
    """The observable record of one simulated Verification_Phase.

    Attributes
    ----------
    events:
        The ordered sequence of phase events, each ``"verify"`` (a
        Verify_Command run) or ``"correction"`` (a correction turn). Used to
        assert the initial-verify-before-any-correction ordering.
    verify_runs:
        How many times the Verify_Command was executed.
    corrections:
        How many Correction_Iterations completed.
    final_outcome:
        The outcome status of the last Verify_Command run.
    interrupted:
        Whether an Interrupt halted the phase.
    """

    events: list[str]
    verify_runs: int
    corrections: int
    final_outcome: str
    interrupted: bool


def simulate_phase(
    outcomes: list[str],
    max_iterations: int,
    interrupt_at: int | None,
) -> LoopTrace:
    """Model the coordinator's bounded loop driven by ``should_run_correction``.

    ``outcomes[0]`` is the result of the initial Verify_Command; ``outcomes[i]``
    is the result of the Verify_Command re-run after the ``i``-th
    Correction_Iteration. ``interrupt_at`` (when not ``None``) is the index of
    the Correction_Iteration during whose correction turn an Interrupt fires,
    halting the phase before that iteration completes (so no re-verify happens
    and no new iteration begins) -- mirroring the coordinator breaking on
    ``TurnResult.interrupted``.
    """
    events: list[str] = ["verify"]  # initial Verify_Command runs once (Req 5.7)
    verify_runs = 1
    corrections = 0
    interrupted = False
    latest = outcomes[0]

    while should_run_correction(latest, corrections, max_iterations, interrupted):
        # An Interrupt during this correction turn halts the phase before the
        # iteration completes and before any re-verify (Req 8.2, 8.4).
        if interrupt_at is not None and interrupt_at == corrections:
            interrupted = True
            break
        # Append feedback + run the correction turn (Req 5.3).
        events.append("correction")
        corrections += 1
        # Re-run the Verify_Command (Req 5.3).
        events.append("verify")
        verify_runs += 1
        latest = outcomes[corrections]

    return LoopTrace(
        events=events,
        verify_runs=verify_runs,
        corrections=corrections,
        final_outcome=latest,
        interrupted=interrupted,
    )


@st.composite
def phase_inputs(draw: st.DrawFn) -> tuple[list[str], int, int | None]:
    """Generate ``(outcomes, max_iterations, interrupt_at)`` for a phase.

    Edge cases from the prework are covered: ``max_iterations`` of 0 and larger
    values, every outcome status (including missing-exit-code ``timed_out`` /
    ``start_error``), and an optional Interrupt at any reachable iteration. The
    outcome sequence is always long enough (``max_iterations + 1``) to cover the
    initial verify plus one re-verify per possible Correction_Iteration.
    """
    max_iterations = draw(st.integers(min_value=0, max_value=8))
    outcomes = draw(
        st.lists(
            st.sampled_from(ALL_OUTCOMES),
            min_size=max_iterations + 1,
            max_size=max_iterations + 1,
        )
    )
    interrupt_at = draw(
        st.one_of(st.none(), st.integers(min_value=0, max_value=max_iterations))
    )
    return outcomes, max_iterations, interrupt_at


@settings(max_examples=300, deadline=None)
@given(data=phase_inputs())
def test_correction_loop_is_bounded_and_terminates_correctly(
    data: tuple[list[str], int, int | None],
) -> None:
    """Property 6: the bounded correction loop terminates correctly.

    Validates: Requirements 5.1, 5.2, 5.4, 5.5, 5.6, 5.7, 6.1, 6.3, 8.4
    """
    outcomes, max_iterations, interrupt_at = data
    trace = simulate_phase(outcomes, max_iterations, interrupt_at)

    # (a) The initial Verify_Command runs exactly once before any correction
    #     (Req 5.7): the first event is a verify, and exactly one verify
    #     precedes the first correction.
    assert trace.events[0] == "verify"
    if "correction" in trace.events:
        first_correction = trace.events.index("correction")
        assert trace.events[:first_correction].count("verify") == 1
    # The verify/correction interleaving is exactly verify,(correction,verify)*,
    # optionally with a trailing dangling correction-attempt suppressed: every
    # completed correction is bracketed by a preceding and following verify, so
    # verify_runs is always one more than the completed Correction_Iterations.
    assert trace.verify_runs == trace.corrections + 1

    # (b) At most Max_Correction_Iterations Correction_Iterations (Req 6.1, 6.3).
    assert trace.corrections <= max_iterations

    # (c) Zero Correction_Iterations when Max_Correction_Iterations is 0
    #     (Req 5.6): the initial verify still ran exactly once.
    if max_iterations == 0:
        assert trace.corrections == 0
        assert trace.verify_runs == 1

    # (d) Stop immediately upon the first passing result (Req 5.1, 5.5): a
    #     passed result is never followed by a Correction_Iteration, so passed
    #     can only ever appear as the final outcome.
    consumed = outcomes[: trace.corrections + 1]
    assert PASSED not in consumed[:-1]
    if trace.final_outcome == PASSED:
        assert trace.corrections == consumed.index(PASSED)

    # (e) start_error is non-correctable (Req 8.4 rationale): it likewise never
    #     triggers a Correction_Iteration, so it can only appear as the final
    #     outcome.
    assert START_ERROR not in consumed[:-1]

    # (f) Once an Interrupt halts the phase, no further Correction_Iteration
    #     begins (Req 8.4). When the loop stopped while still correctable and
    #     under the cap, the only valid reason is an interrupt.
    if (
        trace.final_outcome in CORRECTABLE
        and trace.corrections < max_iterations
    ):
        assert trace.interrupted

    # The phase always terminates: control reaching here means the loop ended.
    # Termination reasons are mutually exhaustive -- passed, non-correctable
    # start_error, cap reached, or interrupted.
    terminated_cleanly = (
        trace.final_outcome == PASSED
        or trace.final_outcome == START_ERROR
        or trace.corrections == max_iterations
        or trace.interrupted
    )
    assert terminated_cleanly


@settings(max_examples=200, deadline=None)
@given(
    latest=st.sampled_from(ALL_OUTCOMES),
    completed=st.integers(min_value=0, max_value=10),
    max_iterations=st.integers(min_value=0, max_value=10),
    interrupted=st.booleans(),
)
def test_should_run_correction_predicate_holds(
    latest: str,
    completed: int,
    max_iterations: int,
    interrupted: bool,
) -> None:
    """The loop-control predicate is exactly the bounded, correctable, not
    interrupted, under-cap conjunction the loop relies on.

    Validates: Requirements 5.2, 5.4, 6.3, 8.4
    """
    result = should_run_correction(latest, completed, max_iterations, interrupted)
    expected = (
        (not interrupted)
        and latest in CORRECTABLE
        and completed < max_iterations
    )
    assert result is expected

    # An interrupt always suppresses correction (Req 8.4).
    if interrupted:
        assert result is False
    # passed / start_error are never correctable (Req 5.1, 8.4 rationale).
    if latest in {PASSED, START_ERROR}:
        assert result is False
    # A zero cap never permits a Correction_Iteration (Req 5.6).
    if max_iterations == 0:
        assert result is False
