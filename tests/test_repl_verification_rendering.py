"""Unit tests for the REPL's Verification_Phase rendering and integration.

These tests cover two distinct surfaces of :class:`forge.repl.Repl`:

* the four :class:`forge.verification.VerificationRenderer` hooks the
  coordinator drives during a phase, asserted by calling each hook directly on
  a Repl wired with a captured :class:`io.StringIO` stream (Req 9.1-9.5); and
* :meth:`Repl.run_once`'s post-turn integration with a wired
  ``verification_coordinator``: when the phase ran, the aggregated phase usage
  replaces the bare turn usage in the ``[usage]`` line; and when no coordinator
  is wired (or the phase did not run), end-of-turn rendering and the usage line
  are exactly as they are without verification (Req 2.3).

The harness mirrors ``tests/test_repl_rendering.py``: a ``FakeAgentLoop`` whose
``run_turn`` drives renderer hooks and returns a crafted
:class:`~forge.agent.TurnResult`, a scripted ``input_func`` so no real TTY is
needed, and an injected :class:`io.StringIO` ``out`` capturing rendered output.

Covers Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 2.3.
"""

from __future__ import annotations

import io
from typing import Callable

from forge.agent import TurnResult
from forge.repl import Repl
from forge.session import Session, TodoItem, Usage
from forge.usage import UsageSummary
from forge.verification import VerificationPhaseResult, VerificationResult


# --------------------------------------------------------------------------- #
# Test doubles / helpers
# --------------------------------------------------------------------------- #


def make_usage(
    *,
    turn_input: int = 10,
    turn_output: int = 20,
    cumulative_input: int = 30,
    cumulative_output: int = 40,
    cost_available: bool = True,
) -> UsageSummary:
    """Build a :class:`UsageSummary` with caller-distinguishable token counts."""

    return UsageSummary(
        turn_input_tokens=turn_input,
        turn_output_tokens=turn_output,
        cumulative_input_tokens=cumulative_input,
        cumulative_output_tokens=cumulative_output,
        turn_cost=0.0012 if cost_available else None,
        cumulative_cost=0.0034 if cost_available else None,
        cost_available=cost_available,
    )


def make_turn_result(
    *,
    usage: UsageSummary | None = None,
    error: str | None = None,
    interrupted: bool = False,
    mutated_files: bool = False,
) -> TurnResult:
    """Build a :class:`TurnResult` with a representative usage summary."""

    return TurnResult(
        usage=usage if usage is not None else make_usage(),
        compaction=None,
        error=error,
        interrupted=interrupted,
        mutated_files=mutated_files,
    )


def make_result(outcome: str) -> VerificationResult:
    """Build a :class:`VerificationResult` carrying ``outcome``."""

    return VerificationResult(
        outcome=outcome,
        exit_code=0 if outcome == "passed" else 1,
        output="some captured output",
        truncated=False,
    )


class FakeAgentLoop:
    """A stand-in :class:`~forge.agent.AgentLoop` driven by a scripted callback."""

    def __init__(
        self, on_run: Callable[[Repl, Session, str], TurnResult]
    ) -> None:
        self.renderer = None  # Repl wires itself in when None.
        self._on_run = on_run
        self.calls: list[str] = []

    def run_turn(self, session: Session, user_text: str) -> TurnResult:
        self.calls.append(user_text)
        assert self.renderer is not None  # Repl must have installed itself.
        return self._on_run(self.renderer, session, user_text)


class ScriptedInput:
    """An ``input_func`` returning queued lines; raises ``EOFError`` when empty."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


class StubCoordinator:
    """A stub ``verification_coordinator`` returning a canned phase result.

    Records the ``(session, turn_result)`` it was called with so tests can
    assert it was (or was not) invoked, and returns the
    :class:`VerificationPhaseResult` it was constructed with.
    """

    def __init__(self, phase_result: VerificationPhaseResult) -> None:
        self._phase_result = phase_result
        self.calls: list[tuple[Session, TurnResult]] = []

    def run(
        self, session: Session, turn_result: TurnResult
    ) -> VerificationPhaseResult:
        self.calls.append((session, turn_result))
        return self._phase_result


def make_session(todos: list[TodoItem] | None = None) -> Session:
    """Build a minimal in-memory :class:`Session` for the REPL to drive."""

    return Session(
        id="test-session",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
        messages=[],
        todos=list(todos) if todos else [],
        usage=Usage(input_tokens=0, output_tokens=0, estimated_cost=None),
    )


def make_repl(
    on_run: Callable[[Repl, Session, str], TurnResult],
    lines: list[str],
    *,
    verification_coordinator: StubCoordinator | None = None,
) -> tuple[Repl, io.StringIO, FakeAgentLoop, Session]:
    """Wire a Repl with a fake loop, scripted input, and a captured stream."""

    loop = FakeAgentLoop(on_run)
    session = make_session()
    out = io.StringIO()
    reader = ScriptedInput(lines)
    repl = Repl(
        loop,
        session,
        input_func=reader,
        out=out,
        verification_coordinator=verification_coordinator,
    )
    return repl, out, loop, session


def make_bare_repl() -> tuple[Repl, io.StringIO]:
    """Build a Repl purely to exercise its VerificationRenderer hooks directly.

    The agent loop is never driven here, so a trivial callback suffices; the
    point is a captured ``out`` stream against which hook output is asserted.
    """

    def on_run(renderer: Repl, session: Session, text: str) -> TurnResult:
        raise AssertionError("hook tests must not drive the agent loop")

    loop = FakeAgentLoop(on_run)
    session = make_session()
    out = io.StringIO()
    repl = Repl(loop, session, input_func=ScriptedInput([]), out=out)
    return repl, out


# --------------------------------------------------------------------------- #
# Req 9.1 — running indicator
# --------------------------------------------------------------------------- #


def test_on_verification_start_renders_running_line() -> None:
    """``on_verification_start`` announces the running Verify_Command (Req 9.1)."""

    repl, out = make_bare_repl()
    repl.on_verification_start("pytest -q")

    assert out.getvalue() == "[verify] running: pytest -q\n"


# --------------------------------------------------------------------------- #
# Req 9.2 — passed indicator
# --------------------------------------------------------------------------- #


def test_on_verification_result_passed_renders_passed_line() -> None:
    """A passing result renders ``[verify] passed`` (Req 9.2)."""

    repl, out = make_bare_repl()
    repl.on_verification_result(make_result("passed"))

    assert out.getvalue() == "[verify] passed\n"


# --------------------------------------------------------------------------- #
# Req 9.3 — failed indicator carries the outcome status
# --------------------------------------------------------------------------- #


def test_on_verification_result_failed_renders_failed_line() -> None:
    """A failed result renders ``[verify] failed (failed)`` (Req 9.3)."""

    repl, out = make_bare_repl()
    repl.on_verification_result(make_result("failed"))

    assert out.getvalue() == "[verify] failed (failed)\n"


def test_on_verification_result_timed_out_renders_status() -> None:
    """A timed-out result carries its status in the failed line (Req 9.3)."""

    repl, out = make_bare_repl()
    repl.on_verification_result(make_result("timed_out"))

    assert out.getvalue() == "[verify] failed (timed_out)\n"


def test_on_verification_result_start_error_renders_status() -> None:
    """A start_error result carries its status in the failed line (Req 9.3)."""

    repl, out = make_bare_repl()
    repl.on_verification_result(make_result("start_error"))

    assert out.getvalue() == "[verify] failed (start_error)\n"


# --------------------------------------------------------------------------- #
# Req 9.4 — correction-iteration indicator
# --------------------------------------------------------------------------- #


def test_on_correction_iteration_renders_n_of_max() -> None:
    """``on_correction_iteration`` renders ``<n>/<max>`` (Req 9.4)."""

    repl, out = make_bare_repl()
    repl.on_correction_iteration(2, 5)

    assert out.getvalue() == "[verify] correction iteration 2/5\n"


# --------------------------------------------------------------------------- #
# Req 9.5 — cap-reached indicator
# --------------------------------------------------------------------------- #


def test_on_verification_cap_reached_renders_final_status() -> None:
    """The cap-reached notice reports iterations and final status (Req 9.5)."""

    repl, out = make_bare_repl()
    repl.on_verification_cap_reached(make_result("failed"), 3)

    assert out.getvalue() == (
        "[verify] iteration cap reached (3); final status: failed\n"
    )


# --------------------------------------------------------------------------- #
# Req 10 (integration) — run_once renders aggregated phase usage when ran
# --------------------------------------------------------------------------- #


def test_run_once_renders_aggregated_phase_usage_when_phase_ran() -> None:
    """When the phase ran, ``run_once`` renders the aggregated usage (Req 10).

    The stub coordinator reports ``ran=True`` with a usage summary carrying
    distinctive token counts that differ from the turn's own usage, so the
    ``[usage]`` line must reflect the aggregated phase totals, not the bare turn
    totals.
    """

    turn_usage = make_usage(
        turn_input=10, turn_output=20, cumulative_input=30, cumulative_output=40
    )
    aggregated = make_usage(
        turn_input=111,
        turn_output=222,
        cumulative_input=333,
        cumulative_output=444,
    )
    phase = VerificationPhaseResult(
        ran=True,
        final_result=make_result("passed"),
        iterations_performed=1,
        cap_reached=False,
        interrupted=False,
        usage=aggregated,
    )
    coordinator = StubCoordinator(phase)

    def on_run(renderer: Repl, session: Session, text: str) -> TurnResult:
        renderer.on_text("did the work")
        return make_turn_result(usage=turn_usage, mutated_files=True)

    repl, out, _loop, session = make_repl(
        on_run, ["go"], verification_coordinator=coordinator
    )
    repl.run_once()

    rendered = out.getvalue()
    # The coordinator was invoked with the session and the turn result.
    assert len(coordinator.calls) == 1
    assert coordinator.calls[0][0] is session
    # The aggregated phase usage drives the [usage] line, not the turn usage.
    assert "turn: 111 in / 222 out" in rendered
    assert "session: 333 in / 444 out" in rendered
    # The bare turn counts must NOT appear in the usage line.
    assert "turn: 10 in / 20 out" not in rendered


# --------------------------------------------------------------------------- #
# Req 2.3 — opt-in equivalence: absent / not-run leaves rendering as today
# --------------------------------------------------------------------------- #


def test_run_once_without_coordinator_renders_turn_usage_and_no_verify_lines() -> None:
    """With no coordinator, end-of-turn output is exactly as today (Req 2.3)."""

    turn_usage = make_usage(
        turn_input=10, turn_output=20, cumulative_input=30, cumulative_output=40
    )

    def on_run(renderer: Repl, session: Session, text: str) -> TurnResult:
        renderer.on_text("plain answer")
        return make_turn_result(usage=turn_usage)

    repl, out, _loop, _session = make_repl(
        on_run, ["hi"], verification_coordinator=None
    )
    repl.run_once()

    rendered = out.getvalue()
    # No verification indicators are surfaced.
    assert "[verify]" not in rendered
    # The usage line reflects the turn's own usage.
    assert "turn: 10 in / 20 out" in rendered
    assert "session: 30 in / 40 out" in rendered


def test_run_once_with_not_run_phase_matches_no_coordinator_output() -> None:
    """A coordinator reporting ``ran=False`` yields identical output (Req 2.3).

    The end-of-turn rendering with a coordinator that did not run the phase must
    byte-for-byte match the rendering produced with no coordinator at all, and
    must surface no ``[verify]`` lines while reflecting the turn's own usage.
    """

    turn_usage = make_usage(
        turn_input=10, turn_output=20, cumulative_input=30, cumulative_output=40
    )

    def on_run(renderer: Repl, session: Session, text: str) -> TurnResult:
        renderer.on_text("plain answer")
        return make_turn_result(usage=turn_usage)

    # Baseline: no coordinator wired.
    baseline_repl, baseline_out, _loop, _session = make_repl(
        on_run, ["hi"], verification_coordinator=None
    )
    baseline_repl.run_once()
    baseline = baseline_out.getvalue()

    # A coordinator that reports the phase did not run, carrying turn usage.
    not_run_phase = VerificationPhaseResult(
        ran=False,
        final_result=None,
        iterations_performed=0,
        cap_reached=False,
        interrupted=False,
        usage=turn_usage,
    )
    coordinator = StubCoordinator(not_run_phase)
    coord_repl, coord_out, _loop2, _session2 = make_repl(
        on_run, ["hi"], verification_coordinator=coordinator
    )
    coord_repl.run_once()
    with_coord = coord_out.getvalue()

    # Opt-in equivalence: identical output, no verify lines, turn usage shown.
    assert with_coord == baseline
    assert "[verify]" not in with_coord
    assert "turn: 10 in / 20 out" in with_coord
