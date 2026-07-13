"""Interrupt-timing integration tests for the Verification_Phase (task 8.4).

These tests exercise the *real* interrupt path of the Verification_Phase with
live processes and the shared :class:`~forge.interrupt.InterruptController`,
rather than the pure decision logic covered by the property/unit suites. They
assert the two timing guarantees of Requirement 8:

* **Req 8.1** -- a long-running Verify_Command is terminated within ~1 second of
  an Interrupt: the process tree is torn down and control returns to the prompt
  well before the multi-second command would have finished.
* **Req 8.2** -- an Interrupt during a correction Turn halts the
  Verification_Phase within ~1 second and begins no new Correction_Iteration.

Test 1 drives a real :class:`~forge.verification.VerificationRunner` on a
long sleeping command and trips the shared interrupt from a background thread
shortly after the run starts, asserting the runner returns promptly (proving the
process tree was terminated by the shell core's sub-second interrupt polling).

Test 2 drives a :class:`~forge.verification.VerificationCoordinator` with a
scripted runner that fails quickly and a mock ``AgentLoop`` whose ``run_turn``
simulates a Ctrl-C during the correction Turn (returns
``TurnResult(interrupted=True)`` after a brief sleep). It asserts the coordinator
halts promptly, performs zero Correction_Iterations, and flags the phase
interrupted.

The sleep command branches on the platform exactly like
``tests/test_execute_command.py`` (``ping -n`` on Windows, ``sleep`` on POSIX) so
the tests run on both. Timing assertions are kept generous enough to avoid
flakiness while still proving prompt termination.

Validates: Requirements 8.1, 8.2
"""

from __future__ import annotations

import os
import sys
import threading
import time

from forge.agent import TurnResult
from forge.config import VerificationConfig
from forge.interrupt import InterruptController
from forge.session import Session, Usage
from forge.usage import UsageSummary
from forge.verification import (
    VerificationCoordinator,
    VerificationResult,
    VerificationRunner,
)

IS_WINDOWS = sys.platform == "win32" or os.name == "nt"


def _sleep_command(seconds: int) -> str:
    """A command that blocks for roughly ``seconds`` seconds on either OS."""
    if IS_WINDOWS:
        # No portable `sleep` on Windows; `ping` waits ~1s between echoes and
        # does not depend on console stdin (unlike `timeout`).
        return f"ping -n {seconds + 1} 127.0.0.1"
    return f"sleep {seconds}"


def _zero_usage() -> UsageSummary:
    """A fully zeroed, cost-unavailable usage summary for fake turn results."""
    return UsageSummary(
        turn_input_tokens=0,
        turn_output_tokens=0,
        cumulative_input_tokens=0,
        cumulative_output_tokens=0,
        turn_cost=None,
        cumulative_cost=None,
        cost_available=False,
    )


def _make_session() -> Session:
    return Session(
        id="verif-interrupt-timing",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
        messages=[],
        todos=[],
        usage=Usage(input_tokens=0, output_tokens=0, estimated_cost=None),
    )


class RecordingSessionStore:
    """Captures saved sessions without touching disk."""

    def __init__(self) -> None:
        self.saved: list[Session] = []

    def save(self, session: Session) -> None:
        self.saved.append(session)


# --------------------------------------------------------------------------- #
# Test 1: Interrupt during a live Verify_Command (Req 8.1)
# --------------------------------------------------------------------------- #


def test_runner_terminates_long_command_within_a_second_of_interrupt(tmp_path):
    """A tripped Interrupt tears down a long Verify_Command process tree fast.

    The runner is started on a 5-second sleep; a background thread trips the
    shared interrupt ~0.3s later. The shell core polls the interrupt at a
    sub-second interval and terminates the process tree, so ``run`` returns well
    under the 5-second command duration -- proving control returns to the prompt
    within ~1 second of the Interrupt (Req 8.1).

    Validates: Requirements 8.1
    """
    interrupt = InterruptController()
    runner = VerificationRunner(tmp_path, interrupt)

    # Trip the interrupt shortly after the command starts, from another thread,
    # mimicking a user pressing Ctrl-C mid-run.
    tripper = threading.Timer(0.3, interrupt.trip)
    tripper.start()
    try:
        start = time.monotonic()
        result = runner.run(_sleep_command(5), timeout_s=30, output_cap=30_000)
        elapsed = time.monotonic() - start
    finally:
        tripper.cancel()

    # The command would run ~5s; termination must be prompt (interrupt tripped
    # at ~0.3s, observed within ~1s, so comfortably under 2.5s) proving the
    # process tree was torn down rather than run to completion.
    assert elapsed < 2.5
    # The runner still returns a structured result (control returns to the
    # prompt); it did not time out (the timeout was 30s, far longer than this).
    assert isinstance(result, VerificationResult)
    assert result.outcome != "timed_out"


# --------------------------------------------------------------------------- #
# Test 2: Interrupt during a correction Turn (Req 8.2)
# --------------------------------------------------------------------------- #


class _QuickFailRunner:
    """A scripted runner that returns a failing result quickly, no real I/O.

    Records how many times ``run`` was invoked so the test can assert no second
    Verify_Command ran after the interrupted correction Turn.
    """

    def __init__(self) -> None:
        self.calls = 0

    def run(self, command, *, timeout_s, output_cap) -> VerificationResult:
        self.calls += 1
        return VerificationResult(
            outcome="failed",
            exit_code=1,
            output="boom",
            truncated=False,
        )


class _InterruptingAgentLoop:
    """A mock ``AgentLoop`` whose ``run_turn`` simulates a Ctrl-C mid-turn.

    It sleeps briefly (modeling work in progress) and then returns a
    ``TurnResult`` flagged interrupted, exactly as the real loop reports a turn
    halted by the shared interrupt. Records the number of ``run_turn`` calls.
    """

    def __init__(self, delay_s: float = 0.3) -> None:
        self._delay_s = delay_s
        self.calls = 0

    def run_turn(self, session: Session, user_text: str) -> TurnResult:
        self.calls += 1
        time.sleep(self._delay_s)
        return TurnResult(usage=_zero_usage(), interrupted=True)


def test_interrupt_during_correction_turn_halts_phase_promptly():
    """An Interrupt during the correction Turn halts the phase and starts no
    new Correction_Iteration (Req 8.2).

    The gate passes (command present, trigger ``always``, turn ok), the initial
    Verify_Command fails quickly, and the correction Turn is interrupted. The
    coordinator must halt promptly, flag the phase interrupted, perform zero
    Correction_Iterations, and re-run no further Verify_Command.

    Validates: Requirements 8.2
    """
    config = VerificationConfig(
        command="run-tests",
        max_correction_iterations=3,
        trigger="always",
        timeout_s=30,
        output_cap_chars=30_000,
    )
    runner = _QuickFailRunner()
    agent_loop = _InterruptingAgentLoop(delay_s=0.3)
    store = RecordingSessionStore()
    interrupt = InterruptController()
    coordinator = VerificationCoordinator(
        config=config,
        runner=runner,
        agent_loop=agent_loop,
        session_store=store,
        interrupt=interrupt,
    )

    turn_result = TurnResult(usage=_zero_usage(), mutated_files=True)

    start = time.monotonic()
    phase = coordinator.run(_make_session(), turn_result)
    elapsed = time.monotonic() - start

    # Halted promptly: only the ~0.3s correction turn elapsed, no further work.
    assert elapsed < 1.5
    # The phase ran, was halted by the interrupt, and began no new iteration.
    assert phase.ran is True
    assert phase.interrupted is True
    assert phase.iterations_performed == 0
    # Exactly one correction Turn was attempted and no second Verify_Command ran
    # (the initial verify is the only runner call).
    assert agent_loop.calls == 1
    assert runner.calls == 1
    # The interrupted phase still persisted the session (retention is
    # all-or-nothing -- the in-progress write completes before returning).
    assert store.saved
