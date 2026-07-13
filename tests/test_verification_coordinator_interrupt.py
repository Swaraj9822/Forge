"""Interrupt-retention unit test for the Verification_Coordinator (task 8.3).

This test drives :meth:`forge.verification.VerificationCoordinator.run` with a
scripted runner and a mock ``AgentLoop`` to verify the coordinator's behavior
when a user Interrupt halts the Verification_Phase mid-correction-turn.

Scenario: the gate passes (a Verify_Command is configured, ``trigger="always"``,
and the original turn completed normally). The scripted runner reports a
``failed`` Verification_Result, so the coordinator begins a Correction_Iteration.
The mock ``AgentLoop.run_turn`` mimics the real loop -- it appends a
Verification_Feedback ``user`` message to ``session.messages`` (which the real
``run_turn`` would also persist) -- and then returns a ``TurnResult`` with
``interrupted=True``.

The test asserts that the coordinator:

* begins no NEW Correction_Iteration past the interrupted one -- the runner is
  invoked only for the initial verify (no re-verify) and ``run_turn`` is called
  exactly once, so ``iterations_performed == 0`` (the interrupted correction
  turn is not counted as completed) (Req 8.4);
* leaves the previously appended Verification_Feedback message intact in
  ``session.messages`` (Req 8.3);
* reports ``VerificationPhaseResult.interrupted is True`` (Req 8.2);
* persists the Session via ``SessionStore.save`` before returning, and the
  persisted Session retains both the feedback message and the appended
  ``VerificationRecord`` (Req 8.3).

Collaborators are in-process fakes (no network, no disk, no real subprocess).

Validates: Requirements 8.2, 8.3, 8.4
"""

from __future__ import annotations

from forge.agent import TurnResult
from forge.config import VerificationConfig
from forge.interrupt import InterruptController
from forge.session import Message, Session, Usage
from forge.usage import UsageSummary
from forge.verification import (
    VerificationCoordinator,
    VerificationResult,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class ScriptedRunner:
    """Returns scripted :class:`VerificationResult`s, one per ``run`` call.

    Records how many times it was invoked so the test can assert that no
    Verify_Command re-run (i.e. no new Correction_Iteration) happened after the
    interrupted correction turn. Once the scripted results are exhausted the
    last result is repeated, so an unexpected extra call is still observable via
    ``calls`` without raising.
    """

    def __init__(self, results: list[VerificationResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def run(
        self, command: str, *, timeout_s: int, output_cap: int
    ) -> VerificationResult:
        self.calls += 1
        if len(self._results) > 1:
            return self._results.pop(0)
        return self._results[0]


class InterruptingAgentLoop:
    """Mock ``AgentLoop`` whose ``run_turn`` appends feedback then interrupts.

    This mirrors the real :meth:`forge.agent.AgentLoop.run_turn`, which appends
    the Verification_Feedback ``user`` message to the session (and persists it)
    before running the turn. Here the turn is reported as interrupted so the
    coordinator must halt the phase and begin no new iteration, while the
    appended feedback message remains intact.
    """

    def __init__(self, usage: UsageSummary) -> None:
        self._usage = usage
        self.run_turn_calls: list[str] = []

    def run_turn(self, session: Session, user_text: str) -> TurnResult:
        self.run_turn_calls.append(user_text)
        # Mimic real run_turn: append (and "persist") the feedback message.
        session.messages.append(Message(role="user", text=user_text))
        return TurnResult(usage=self._usage, interrupted=True)


class RecordingSessionStore:
    """Captures saved sessions without touching disk."""

    def __init__(self) -> None:
        self.saved: list[Session] = []

    def save(self, session: Session) -> None:
        self.saved.append(session)


def _usage() -> UsageSummary:
    """A minimal, cost-unavailable usage summary for a turn."""

    return UsageSummary(
        turn_input_tokens=1,
        turn_output_tokens=1,
        cumulative_input_tokens=1,
        cumulative_output_tokens=1,
        turn_cost=None,
        cumulative_cost=None,
        cost_available=False,
    )


def _make_session() -> Session:
    return Session(
        id="session-under-test",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
        messages=[],
        todos=[],
        usage=Usage(input_tokens=0, output_tokens=0, estimated_cost=None),
    )


# --------------------------------------------------------------------------- #
# Test
# --------------------------------------------------------------------------- #


def test_interrupt_mid_correction_retains_feedback_and_persists() -> None:
    """An interrupt during the correction turn halts the phase without loss.

    The gate passes and the initial Verify_Command fails, so the coordinator
    starts a Correction_Iteration. The correction turn appends a feedback
    message and reports ``interrupted=True``. The coordinator must halt: begin
    no new iteration (no re-verify), retain the appended feedback, report the
    phase as interrupted, and persist the Session (with its VerificationRecord)
    before returning.

    Validates: Requirements 8.2, 8.3, 8.4
    """

    config = VerificationConfig(
        command="pytest -q",
        max_correction_iterations=3,
        trigger="always",
        timeout_s=120,
        output_cap_chars=30_000,
    )
    failing = VerificationResult(
        outcome="failed",
        exit_code=1,
        output="1 failed, 0 passed",
        truncated=False,
    )
    runner = ScriptedRunner([failing])
    agent_loop = InterruptingAgentLoop(_usage())
    store = RecordingSessionStore()
    coordinator = VerificationCoordinator(
        config=config,
        runner=runner,
        agent_loop=agent_loop,
        session_store=store,
        interrupt=InterruptController(),
    )

    session = _make_session()
    turn_result = TurnResult(usage=_usage(), mutated_files=True)

    result = coordinator.run(session, turn_result)

    # The phase ran and was halted by the interrupt (Req 8.2).
    assert result.ran is True
    assert result.interrupted is True

    # No NEW Correction_Iteration began past the interrupted one (Req 8.4):
    # the initial verify ran once with no re-verify, and the correction turn
    # was attempted exactly once. The interrupted correction turn is not
    # counted as completed.
    assert runner.calls == 1
    assert len(agent_loop.run_turn_calls) == 1
    assert result.iterations_performed == 0

    # The previously appended Verification_Feedback message is intact in the
    # live session (Req 8.3).
    feedback_text = agent_loop.run_turn_calls[0]
    feedback_messages = [
        m for m in session.messages if m.role == "user" and m.text == feedback_text
    ]
    assert len(feedback_messages) == 1
    assert feedback_text.startswith("The verification command failed.")

    # The Session was persisted before returning, and the persisted Session
    # retains the feedback message and the appended VerificationRecord (Req 8.3).
    assert len(store.saved) >= 1
    persisted = store.saved[-1]
    assert any(
        m.role == "user" and m.text == feedback_text for m in persisted.messages
    )
    assert len(persisted.verification_records) == 1
    record = persisted.verification_records[0]
    assert record.command == config.command
    assert record.outcome == "failed"
    assert record.iterations == 0
