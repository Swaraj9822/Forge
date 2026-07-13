"""Opt-in equivalence integration test for the Verification_Phase (task 11.2).

This suite proves the feature is strictly opt-in end-to-end: with no
Verify_Command configured, a completed turn returns to the prompt with *no*
process started, *no* Verification_Feedback appended, and identical end-of-turn
rendering and persistence to today's behavior (Req 2.1, 2.2, 2.3).

The proof is driven through a real :class:`~forge.verification.VerificationCoordinator`
constructed with the default :class:`~forge.config.VerificationConfig` (so
``command is None``), wired to:

* a *spy* runner whose ``run`` records every invocation -- so we can assert the
  Verify_Command process is never started (Req 2.1);
* a *mock* :class:`~forge.agent.AgentLoop` whose ``run_turn`` records every
  invocation -- so we can assert no correction turn runs and thus no feedback
  message is appended (Req 2.2);
* a *recording* :class:`~forge.session.SessionStore` whose ``save`` records
  every call -- so we can assert the gate returns before any persistence; and
* a real :class:`~forge.interrupt.InterruptController`.

A best-effort bootstrap-level check additionally proves the coordinator is wired
unconditionally by :func:`forge.app.bootstrap` (even when the command is absent)
and short-circuits at its gate.

Validates: Requirements 2.1, 2.2, 2.3
"""

from __future__ import annotations

from io import StringIO

from forge.config import Config, VerificationConfig
from forge.interrupt import InterruptController
from forge.session import Session, Usage
from forge.usage import UsageSummary
from forge.verification import VerificationCoordinator


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class SpyRunner:
    """A stand-in :class:`~forge.verification.VerificationRunner`.

    Records every ``run`` call. When the feature is unconfigured the coordinator
    must never call this, so ``calls`` staying empty proves no Verify_Command
    process was started (Req 2.1).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, command, *, timeout_s, output_cap):  # pragma: no cover - guard
        self.calls.append((command, {"timeout_s": timeout_s, "output_cap": output_cap}))
        raise AssertionError(
            "VerificationRunner.run must not be called when no command is configured"
        )


class MockAgentLoop:
    """A stand-in :class:`~forge.agent.AgentLoop` recording ``run_turn`` calls.

    A correction turn is the only thing that would append a Verification_Feedback
    message, so ``calls`` staying empty proves no feedback was appended (Req 2.2).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Session, str]] = []

    def run_turn(self, session, user_text):  # pragma: no cover - guard
        self.calls.append((session, user_text))
        raise AssertionError(
            "AgentLoop.run_turn must not be called when no command is configured"
        )


class RecordingSessionStore:
    """Captures saved sessions without touching disk."""

    def __init__(self) -> None:
        self.saved: list[Session] = []

    def save(self, session: Session) -> None:
        self.saved.append(session)


class FakeTurnResult:
    """A completed turn result: not interrupted, no error, files mutated.

    Mirrors the fields :meth:`VerificationCoordinator.run` reads from a real
    :class:`~forge.agent.TurnResult` (``usage``, ``interrupted``, ``error``,
    ``mutated_files``). ``mutated_files=True`` is the strongest case: even with
    a file mutation, an absent command keeps the gate closed.
    """

    def __init__(
        self,
        usage: UsageSummary,
        *,
        interrupted: bool = False,
        error: str | None = None,
        mutated_files: bool = True,
    ) -> None:
        self.usage = usage
        self.interrupted = interrupted
        self.error = error
        self.mutated_files = mutated_files


def _usage() -> UsageSummary:
    """A representative per-turn usage summary carried through the phase."""
    return UsageSummary(
        turn_input_tokens=10,
        turn_output_tokens=5,
        cumulative_input_tokens=100,
        cumulative_output_tokens=50,
        turn_cost=0.25,
        cumulative_cost=1.50,
        cost_available=True,
    )


def _session() -> Session:
    return Session(
        id="optin-session",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
        messages=[],
        todos=[],
        usage=Usage(input_tokens=100, output_tokens=50, estimated_cost=1.50),
    )


def _coordinator(store: RecordingSessionStore) -> tuple[
    VerificationCoordinator, SpyRunner, MockAgentLoop
]:
    """Build a coordinator with an absent Verify_Command (the default config)."""
    runner = SpyRunner()
    agent_loop = MockAgentLoop()
    coordinator = VerificationCoordinator(
        config=VerificationConfig(),  # command is None => feature disabled
        runner=runner,
        agent_loop=agent_loop,
        session_store=store,
        interrupt=InterruptController(),
        renderer=None,
    )
    return coordinator, runner, agent_loop


# --------------------------------------------------------------------------- #
# Coordinator-level opt-in equivalence (the core proof) — Req 2.1, 2.2, 2.3
# --------------------------------------------------------------------------- #


def test_unconfigured_phase_does_not_run_or_touch_session() -> None:
    """An absent command short-circuits the gate: no process, no feedback, no save.

    Validates: Requirements 2.1, 2.2, 2.3
    """

    store = RecordingSessionStore()
    coordinator, runner, agent_loop = _coordinator(store)
    usage = _usage()
    turn_result = FakeTurnResult(usage, mutated_files=True)
    session = _session()
    messages_before = list(session.messages)
    records_before = list(session.verification_records)

    phase = coordinator.run(session, turn_result)

    # The phase did not run (Req 2.1).
    assert phase.ran is False
    assert phase.final_result is None
    assert phase.iterations_performed == 0
    assert phase.cap_reached is False
    assert phase.interrupted is False

    # No Verify_Command process was started (Req 2.1).
    assert runner.calls == []

    # No correction turn ran, so no Verification_Feedback was appended (Req 2.2).
    assert agent_loop.calls == []
    assert session.messages == messages_before == []

    # No VerificationRecord was appended (Req 2.3 — persistence unchanged).
    assert session.verification_records == records_before == []

    # The returned usage is exactly the original turn's usage (Req 2.3).
    assert phase.usage == usage
    assert phase.usage is turn_result.usage

    # The coordinator did not persist the session: the gate returned before any
    # SessionStore.save (Req 2.3).
    assert store.saved == []


def test_unconfigured_gate_false_regardless_of_trigger_or_mutation() -> None:
    """With no command, the gate is closed for every trigger / mutation combo.

    Covers ``on_file_change`` with and without a File_Mutation and ``always``;
    in every case the phase does not run, starts no process, and appends nothing.

    Validates: Requirements 2.1, 2.2, 2.3
    """

    for trigger in ("on_file_change", "always"):
        for mutated in (False, True):
            store = RecordingSessionStore()
            runner = SpyRunner()
            agent_loop = MockAgentLoop()
            coordinator = VerificationCoordinator(
                config=VerificationConfig(command=None, trigger=trigger),
                runner=runner,
                agent_loop=agent_loop,
                session_store=store,
                interrupt=InterruptController(),
                renderer=None,
            )
            session = _session()
            turn_result = FakeTurnResult(_usage(), mutated_files=mutated)

            phase = coordinator.run(session, turn_result)

            assert phase.ran is False, (trigger, mutated)
            assert runner.calls == [], (trigger, mutated)
            assert agent_loop.calls == [], (trigger, mutated)
            assert session.messages == [], (trigger, mutated)
            assert session.verification_records == [], (trigger, mutated)
            assert store.saved == [], (trigger, mutated)


# --------------------------------------------------------------------------- #
# Bootstrap-level wiring check (best-effort end-to-end) — Req 2.1, 2.2, 2.3
# --------------------------------------------------------------------------- #


def test_bootstrap_wires_coordinator_that_short_circuits_when_unconfigured() -> None:
    """bootstrap wires the coordinator unconditionally; it gates out when unconfigured.

    The default Config carries a default VerificationConfig (command None), so
    the coordinator is wired onto the Repl yet returns ``ran=False`` for a
    completed turn — proving the unconfigured path is unchanged (Req 2.1, 2.3).

    Validates: Requirements 2.1, 2.2, 2.3
    """

    from forge import app as app_module

    config = Config(project="my-project", region="us-central1")
    assert config.verification.command is None

    app = app_module.bootstrap(
        config=config,
        skip_adc_check=True,
        input_func=lambda _prompt="": "",
        out=StringIO(),
    )

    # The coordinator is wired onto the Repl and the App (Req 2.1 wiring).
    assert app.verification_coordinator is not None
    assert app.repl.verification_coordinator is app.verification_coordinator

    # Driving it for a completed turn short-circuits at the gate (Req 2.1-2.3).
    session = app.session
    messages_before = list(session.messages)
    records_before = list(session.verification_records)
    turn_result = FakeTurnResult(_usage(), mutated_files=True)

    phase = app.verification_coordinator.run(session, turn_result)

    assert phase.ran is False
    assert phase.final_result is None
    assert phase.iterations_performed == 0
    assert phase.usage is turn_result.usage
    # No feedback appended and no verification record persisted (Req 2.2, 2.3).
    assert session.messages == messages_before
    assert session.verification_records == records_before

    app.close()
