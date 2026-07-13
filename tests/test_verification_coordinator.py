"""Example/unit tests for the Verification_Coordinator loop (task 8.2).

These tests drive :meth:`forge.verification.VerificationCoordinator.run` with a
scripted runner and a mock :class:`~forge.agent.AgentLoop` to assert the phase's
*ordering* and *bounds* without any real process execution, model call, or disk
I/O. They exercise the live coordinator algorithm -- the gate, the bounded
self-correction loop, and the cap/record/persist tail -- against in-process
fakes.

The collaborators mirror the construction style of
``tests/test_file_mutation_tracking.py``: a real
:class:`~forge.interrupt.InterruptController` (never tripped here), a recording
``SessionStore`` with a ``save(session)`` method, and a
:class:`~forge.config.VerificationConfig` with the command set and
``trigger="always"`` so the gate always passes for a completed turn.

Both the scripted runner and the mock loop append to a single shared ``events``
list so the relative ordering of Verify_Command runs (``"verify"``) and
correction turns (``"run_turn"``) is directly observable:

* the initial Verify_Command runs before any correction (first event
  ``"verify"`` -- Req 5.7);
* each Correction_Iteration is feedback -> ``run_turn`` -> re-verify, i.e. the
  per-correction event order is ``"run_turn"`` then ``"verify"`` (Req 5.3, 7.3);
* an always-failing command with ``max=N`` performs exactly ``N`` corrections
  and ends ``cap_reached=True`` with the last failing result and
  ``iterations_performed == N`` (Req 6.1, 6.2, 6.3);
* a fail-then-pass sequence stops on the first ``passed`` (Req 5.5);
* ``max == 0`` performs zero corrections but runs the initial verify once
  (Req 5.6, 5.7);
* the renderer's ``on_verification_cap_reached`` is called with the final
  result and the iteration count when the cap is hit (Req 6.2).

Validates: Requirements 5.3, 5.5, 6.1, 6.2, 7.3
"""

from __future__ import annotations

from forge.agent import TurnResult
from forge.config import VerificationConfig
from forge.interrupt import InterruptController
from forge.session import Session, Usage
from forge.usage import UsageSummary
from forge.verification import (
    VerificationCoordinator,
    VerificationResult,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


def _usage(turn_in: int = 1, turn_out: int = 1) -> UsageSummary:
    """A minimal, cost-unavailable :class:`UsageSummary` for turns."""

    return UsageSummary(
        turn_input_tokens=turn_in,
        turn_output_tokens=turn_out,
        cumulative_input_tokens=turn_in,
        cumulative_output_tokens=turn_out,
        turn_cost=None,
        cumulative_cost=None,
        cost_available=False,
    )


def _failed(exit_code: int = 1) -> VerificationResult:
    return VerificationResult(
        outcome="failed", exit_code=exit_code, output="boom", truncated=False
    )


def _passed() -> VerificationResult:
    return VerificationResult(
        outcome="passed", exit_code=0, output="ok", truncated=False
    )


class ScriptedRunner:
    """A Verify_Command runner returning a pre-scripted sequence of results.

    ``results`` is consumed in order; once exhausted the final result repeats,
    so an "always-failing" runner is expressed as a single failing result. Each
    ``run`` call appends ``"verify"`` to the shared ``events`` list and records
    its own invocation count.
    """

    def __init__(self, results: list[VerificationResult], events: list[str]) -> None:
        self._results = list(results)
        self._events = events
        self.calls = 0

    def run(
        self, command: str, *, timeout_s: int, output_cap: int
    ) -> VerificationResult:
        self._events.append("verify")
        index = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[index]


class MockAgentLoop:
    """A mock Agent_Loop whose ``run_turn`` records calls and never fails.

    Each call appends ``"run_turn"`` to the shared ``events`` list, records the
    feedback ``user_text`` it received, and returns a clean
    :class:`~forge.agent.TurnResult` (not interrupted, no error) so the
    coordinator proceeds to re-verify.
    """

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.feedbacks: list[str] = []

    def run_turn(self, session: Session, user_text: str) -> TurnResult:
        self._events.append("run_turn")
        self.feedbacks.append(user_text)
        return TurnResult(usage=_usage(), interrupted=False, error=None)


class RecordingSessionStore:
    """Captures saved sessions without touching disk."""

    def __init__(self) -> None:
        self.saved: list[Session] = []

    def save(self, session: Session) -> None:
        self.saved.append(session)


class RecordingRenderer:
    """A renderer spy recording every coordinator hook invocation."""

    def __init__(self) -> None:
        self.starts: list[str] = []
        self.results: list[VerificationResult] = []
        self.iterations: list[tuple[int, int]] = []
        self.cap_reached: list[tuple[VerificationResult, int]] = []

    def on_verification_start(self, command: str) -> None:
        self.starts.append(command)

    def on_verification_result(self, result: VerificationResult) -> None:
        self.results.append(result)

    def on_correction_iteration(self, iteration: int, max_iterations: int) -> None:
        self.iterations.append((iteration, max_iterations))

    def on_verification_cap_reached(
        self, result: VerificationResult, iterations: int
    ) -> None:
        self.cap_reached.append((result, iterations))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_session() -> Session:
    return Session(
        id="session-under-test",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
        messages=[],
        todos=[],
        usage=Usage(input_tokens=0, output_tokens=0, estimated_cost=None),
    )


def _make_coordinator(
    results: list[VerificationResult],
    *,
    max_iterations: int,
    events: list[str],
    renderer: RecordingRenderer | None = None,
) -> tuple[VerificationCoordinator, ScriptedRunner, MockAgentLoop, RecordingSessionStore]:
    config = VerificationConfig(
        command="run-tests",
        max_correction_iterations=max_iterations,
        trigger="always",
        timeout_s=120,
        output_cap_chars=30_000,
    )
    runner = ScriptedRunner(results, events)
    agent_loop = MockAgentLoop(events)
    store = RecordingSessionStore()
    coordinator = VerificationCoordinator(
        config=config,
        runner=runner,
        agent_loop=agent_loop,
        session_store=store,
        interrupt=InterruptController(),
        renderer=renderer,
    )
    return coordinator, runner, agent_loop, store


def _completed_turn() -> TurnResult:
    """A turn that completed normally and mutated files (gate passes)."""

    return TurnResult(
        usage=_usage(), interrupted=False, error=None, mutated_files=True
    )


# --------------------------------------------------------------------------- #
# Initial verify ordering (Req 5.7)
# --------------------------------------------------------------------------- #


def test_initial_verify_runs_before_any_correction() -> None:
    """The first recorded event is a Verify_Command run, never a correction.

    Validates: Requirements 5.7
    """

    events: list[str] = []
    coordinator, _runner, _loop, _store = _make_coordinator(
        [_failed()], max_iterations=2, events=events
    )

    coordinator.run(_make_session(), _completed_turn())

    assert events[0] == "verify"


# --------------------------------------------------------------------------- #
# Per-correction ordering: verify -> run_turn -> verify (Req 5.3, 7.3) #
# --------------------------------------------------------------------------- #


def test_correction_ordering_is_verify_runturn_reverify() -> None:
    """Each correction feeds back then re-verifies: verify, run_turn, verify ...

    With an always-failing command and ``max=2`` the full event sequence is the
    initial verify followed by two (run_turn, verify) correction cycles.

    Validates: Requirements 5.3, 7.3
    """

    events: list[str] = []
    coordinator, runner, loop, _store = _make_coordinator(
        [_failed()], max_iterations=2, events=events
    )

    coordinator.run(_make_session(), _completed_turn())

    assert events == ["verify", "run_turn", "verify", "run_turn", "verify"]
    # A correction turn always precedes its re-verify (feedback then re-run).
    assert loop.feedbacks  # feedback was synthesized and passed to run_turn
    assert runner.calls == 3  # initial + one re-verify per correction


# --------------------------------------------------------------------------- #
# Always-failing reaches the cap with exactly N corrections (Req 6.1, 6.3) #
# --------------------------------------------------------------------------- #


def test_always_failing_performs_exactly_max_corrections_and_caps() -> None:
    """An always-failing command with ``max=N`` does exactly ``N`` corrections.

    The phase ends ``cap_reached=True`` carrying the final failing result and
    ``iterations_performed == N``.

    Validates: Requirements 6.1, 6.3
    """

    for n in (1, 3, 5):
        events: list[str] = []
        final = _failed(exit_code=7)
        coordinator, _runner, loop, store = _make_coordinator(
            [final], max_iterations=n, events=events
        )

        phase = coordinator.run(_make_session(), _completed_turn())

        assert phase.ran is True
        assert phase.iterations_performed == n
        assert phase.cap_reached is True
        assert phase.final_result == final
        assert phase.final_result.outcome == "failed"
        # Exactly N correction turns were run.
        assert events.count("run_turn") == n
        # Initial verify plus one re-verify per correction.
        assert events.count("verify") == n + 1
        # The session was persisted before returning.
        assert store.saved and store.saved[-1].verification_records[-1].cap_reached


# --------------------------------------------------------------------------- #
# Fail-then-pass stops on the first pass (Req 5.5) #
# --------------------------------------------------------------------------- #


def test_fail_then_pass_stops_on_first_pass() -> None:
    """A fail-then-pass sequence performs one correction and ends passed.

    Validates: Requirements 5.5
    """

    events: list[str] = []
    coordinator, runner, _loop, _store = _make_coordinator(
        [_failed(), _passed()], max_iterations=3, events=events
    )

    phase = coordinator.run(_make_session(), _completed_turn())

    assert events == ["verify", "run_turn", "verify"]
    assert phase.iterations_performed == 1
    assert phase.cap_reached is False
    assert phase.final_result is not None
    assert phase.final_result.outcome == "passed"
    assert runner.calls == 2


# --------------------------------------------------------------------------- #
# max == 0 performs zero corrections but runs the initial verify once #
# (Req 5.6, 5.7) #
# --------------------------------------------------------------------------- #


def test_max_zero_runs_initial_verify_once_with_no_corrections() -> None:
    """``max == 0`` runs the Verify_Command once and performs no corrections.

    With a failing outcome, not interrupted, and ``completed(0) >= max(0)``, the
    phase reports ``cap_reached=True`` while having run the initial verify
    exactly once and never invoking the Agent_Loop.

    Validates: Requirements 5.6, 5.7
    """

    events: list[str] = []
    coordinator, runner, loop, _store = _make_coordinator(
        [_failed()], max_iterations=0, events=events
    )

    phase = coordinator.run(_make_session(), _completed_turn())

    assert events == ["verify"]
    assert runner.calls == 1
    assert loop.feedbacks == []  # the Agent_Loop was never driven
    assert phase.iterations_performed == 0
    assert phase.cap_reached is True


# --------------------------------------------------------------------------- #
# Cap-reached surfacing through the renderer (Req 6.2) #
# --------------------------------------------------------------------------- #


def test_renderer_records_cap_reached_with_final_result_and_count() -> None:
    """``on_verification_cap_reached`` fires once with the final result + count.

    Validates: Requirements 6.2
    """

    events: list[str] = []
    renderer = RecordingRenderer()
    final = _failed(exit_code=2)
    coordinator, _runner, _loop, _store = _make_coordinator(
        [final], max_iterations=2, events=events, renderer=renderer
    )

    phase = coordinator.run(_make_session(), _completed_turn())

    assert len(renderer.cap_reached) == 1
    recorded_result, recorded_iterations = renderer.cap_reached[0]
    assert recorded_result == final
    assert recorded_iterations == phase.iterations_performed == 2
    # The correction-iteration indicator surfaced n/max for each correction.
    assert renderer.iterations == [(1, 2), (2, 2)]


def test_renderer_no_cap_reached_when_phase_passes() -> None:
    """When verification ends passed, the cap-reached hook never fires.

    Validates: Requirements 5.5, 6.2
    """

    events: list[str] = []
    renderer = RecordingRenderer()
    coordinator, _runner, _loop, _store = _make_coordinator(
        [_failed(), _passed()], max_iterations=3, events=events, renderer=renderer
    )

    coordinator.run(_make_session(), _completed_turn())

    assert renderer.cap_reached == []
