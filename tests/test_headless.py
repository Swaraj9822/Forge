"""Tests for the non-interactive headless run path."""

from __future__ import annotations

import json
from io import StringIO

from forge.agent import TurnResult
from forge.headless import EXIT_INTERRUPTED, EXIT_OK, EXIT_TURN_ERROR, EXIT_VERIFICATION_FAILED, run_headless
from forge.session import Session, TodoItem
from forge.usage import UsageSummary


def _usage(input_tokens: int = 10, output_tokens: int = 20) -> UsageSummary:
    return UsageSummary(
        turn_input_tokens=input_tokens,
        turn_output_tokens=output_tokens,
        cumulative_input_tokens=input_tokens,
        cumulative_output_tokens=output_tokens,
        turn_cost=None,
        cumulative_cost=None,
        cost_available=False,
    )


def _session(todos: list[TodoItem] | None = None) -> Session:
    return Session(
        id="session-1",
        created_at="t1",
        updated_at="t2",
        todos=list(todos) if todos else [],
    )


class _FakeAgentLoop:
    """AgentLoop stand-in that streams scripted text and returns a TurnResult."""

    def __init__(self, result: TurnResult, response_text: str = "hello") -> None:
        self.renderer = None
        self._result = result
        self._response_text = response_text

    def run_turn(self, session: Session, prompt: str) -> TurnResult:
        if self.renderer is not None:
            on_text = getattr(self.renderer, "on_text", None)
            if callable(on_text):
                on_text(self._response_text)
        return self._result


class _FakeVerificationPhase:
    def __init__(self, ran: bool, outcome: str | None = None) -> None:
        self.ran = ran
        self.iterations_performed = 1 if ran else 0
        self.cap_reached = False
        self.final_result = _FakeResult(outcome) if outcome is not None else None
        self.usage = _usage(input_tokens=30, output_tokens=40)


class _FakeResult:
    def __init__(self, outcome: str) -> None:
        self.outcome = outcome


class _FakeVerificationCoordinator:
    def __init__(self, phase: _FakeVerificationPhase | None) -> None:
        self._phase = phase
        self.renderer = None

    def set_renderer(self, renderer) -> None:
        self.renderer = renderer

    def run(self, session: Session, turn_result: TurnResult):
        return self._phase


def test_text_mode_streams_and_prints_usage() -> None:
    """Text mode echoes the streamed text and prints a usage footer."""
    result = TurnResult(usage=_usage())
    agent = _FakeAgentLoop(result, response_text="streamed text")
    session = _session()
    out = StringIO()

    code = run_headless(agent, session, None, "prompt", output="text", out=out)

    assert code == EXIT_OK
    text = out.getvalue()
    assert "streamed text" in text
    assert "[usage]" in text
    assert "turn: 10 in / 20 out" in text


def test_json_mode_emits_single_object() -> None:
    """JSON mode emits one parseable object with the documented keys."""
    result = TurnResult(usage=_usage())
    agent = _FakeAgentLoop(result, response_text="json response")
    session = _session(todos=[TodoItem(id="1", text="todo", status="pending")])
    out = StringIO()

    code = run_headless(agent, session, None, "prompt", output="json", out=out)

    assert code == EXIT_OK
    payload = json.loads(out.getvalue())
    assert payload["ok"] is True
    assert payload["response"] == "json response"
    assert payload["error"] is None
    assert payload["interrupted"] is False
    assert payload["mutated_files"] is False
    assert payload["session_id"] == "session-1"
    assert payload["verification"] is None
    assert payload["todos"] == [{"id": "1", "text": "todo", "status": "pending"}]
    assert payload["usage"]["turn_input_tokens"] == 10


def test_json_mode_not_polluted_by_streaming() -> None:
    """JSON stdout contains no streamed fragments before the JSON object."""
    result = TurnResult(usage=_usage())
    agent = _FakeAgentLoop(result, response_text="fragment")
    out = StringIO()

    run_headless(agent, _session(), None, "prompt", output="json", out=out)

    raw = out.getvalue()
    assert raw.startswith("{")
    assert "fragment" not in raw.replace("\"response\": \"fragment\"", "")


def test_exit_code_on_turn_error() -> None:
    """A TurnResult with error yields exit code 2 and populates the error key."""
    result = TurnResult(usage=_usage(), error="boom")
    agent = _FakeAgentLoop(result)
    out = StringIO()

    code = run_headless(agent, _session(), None, "prompt", output="json", out=out)
    payload = json.loads(out.getvalue())

    assert code == EXIT_TURN_ERROR
    assert payload["ok"] is False
    assert payload["error"] == "boom"


def test_exit_code_on_interrupt() -> None:
    """An interrupted turn yields exit code 3."""
    result = TurnResult(usage=_usage(), interrupted=True)
    agent = _FakeAgentLoop(result)
    out = StringIO()

    code = run_headless(agent, _session(), None, "prompt", output="json", out=out)
    payload = json.loads(out.getvalue())

    assert code == EXIT_INTERRUPTED
    assert payload["ok"] is False
    assert payload["interrupted"] is True


def test_verification_failure_exit_code() -> None:
    """A failing verification phase yields exit code 4 and a verification object."""
    result = TurnResult(usage=_usage())
    agent = _FakeAgentLoop(result, response_text="first turn text")
    phase = _FakeVerificationPhase(ran=True, outcome="failed")
    coordinator = _FakeVerificationCoordinator(phase)
    out = StringIO()

    code = run_headless(agent, _session(), coordinator, "prompt", output="json", out=out)
    payload = json.loads(out.getvalue())

    assert code == EXIT_VERIFICATION_FAILED
    assert payload["ok"] is False
    assert payload["verification"] == {
        "ran": True,
        "outcome": "failed",
        "iterations": 1,
        "cap_reached": False,
    }


def test_response_snapshot_excludes_correction_turns() -> None:
    """The reported response is the first turn's text only."""
    result = TurnResult(usage=_usage())
    agent = _FakeAgentLoop(result, response_text="first turn text")
    phase = _FakeVerificationPhase(ran=True, outcome="passed")
    coordinator = _FakeVerificationCoordinator(phase)
    out = StringIO()

    run_headless(agent, _session(), coordinator, "prompt", output="json", out=out)
    payload = json.loads(out.getvalue())

    assert payload["response"] == "first turn text"
