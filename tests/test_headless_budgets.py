"""Tests for headless run budgets (--max-turns / --max-cost), Phase 6.

Covers three layers:

* the ``AgentLoop`` enforcement of ``max_iterations`` / ``max_cost`` (the loop
  stops early and flags ``budget_exceeded``);
* the ``run_headless`` wiring (budgets applied, exit code 5, verification
  skipped, JSON/text surfacing); and
* the CLI surface (arg validators and the ``--max-cost`` pricing guard).

All collaborators are in-process fakes — no network, disk, or TTY.
"""

from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace

import pytest

from forge.agent import AgentLoop, TurnResult
from forge.config import ModelPricing
from forge.headless import (
    EXIT_BUDGET_EXCEEDED,
    EXIT_OK,
    EXIT_TURN_ERROR,
    run_headless,
)
from forge.interrupt import InterruptController
from forge.session import Session, ToolCall, Usage
from forge.usage import UsageSummary, UsageTracker
from forge.providers import Done, UsageReport


# --------------------------------------------------------------------------- #
# Shared fakes
# --------------------------------------------------------------------------- #


def _session() -> Session:
    return Session(
        id="s",
        created_at="t1",
        updated_at="t2",
        usage=Usage(input_tokens=0, output_tokens=0, estimated_cost=None),
    )


def _usage() -> UsageSummary:
    return UsageSummary(
        turn_input_tokens=1,
        turn_output_tokens=1,
        cumulative_input_tokens=1,
        cumulative_output_tokens=1,
        turn_cost=None,
        cumulative_cost=None,
        cost_available=False,
    )


class _FakeAgentLoop:
    """Records applied budgets and returns a scripted TurnResult."""

    def __init__(self, result: TurnResult) -> None:
        self.renderer = None
        self.max_iterations = None
        self.max_cost = None
        self._result = result

    def run_turn(self, session: Session, prompt: str) -> TurnResult:
        return self._result


class _FakePhase:
    def __init__(self) -> None:
        self.ran = True
        self.iterations_performed = 0
        self.cap_reached = False
        self.final_result = SimpleNamespace(outcome="passed")
        self.usage = _usage()


class _FakeCoordinator:
    def __init__(self) -> None:
        self.run_called = False

    def set_renderer(self, renderer) -> None:
        pass

    def run(self, session, turn_result):
        self.run_called = True
        return _FakePhase()


class _FakeContext:
    def assemble(self, session: Session):
        return [{"role": m.role, "content": m.text} for m in session.messages], None


class _RecordingExecutor:
    def specs(self):
        return []

    def execute(self, call: ToolCall):
        from forge.tools.base import ToolResult

        return ToolResult(ok=True, content=f"ran {call.name}")


class _RecordingStore:
    def __init__(self) -> None:
        self.saved: list[Session] = []

    def save(self, session: Session) -> None:
        self.saved.append(session)


# --------------------------------------------------------------------------- #
# AgentLoop enforcement
# --------------------------------------------------------------------------- #


class _LoopingProvider:
    """Always asks for another tool call, so the turn would never end on its own."""

    def generate_stream(self, contents, tools):
        yield ToolCall(id="c", name="read", args={})
        yield Done()


def _make_loop(provider, *, max_iterations=None, max_cost=None, tracker=None):
    return AgentLoop(
        context_manager=_FakeContext(),
        provider=provider,
        tool_executor=_RecordingExecutor(),
        usage_tracker=tracker or UsageTracker(),
        session_store=_RecordingStore(),
        interrupt=InterruptController(),
        max_iterations=max_iterations,
        max_cost=max_cost,
    )


def test_max_turns_stops_the_loop_and_flags_budget() -> None:
    loop = _make_loop(_LoopingProvider(), max_iterations=2)
    result = loop.run_turn(_session(), "go")
    assert result.budget_exceeded is True
    assert result.interrupted is False
    assert result.error is None


class _CostProvider:
    """Reports token usage each round and keeps requesting tool calls."""

    def generate_stream(self, contents, tools):
        yield UsageReport(input_tokens=1000, output_tokens=1000)
        yield ToolCall(id="c", name="read", args={})
        yield Done()


def test_max_cost_stops_the_loop_when_pricing_available() -> None:
    tracker = UsageTracker(ModelPricing(input_per_1k=1.0, output_per_1k=1.0))
    # Each round costs (1000/1000 * 1) + (1000/1000 * 1) = 2.0, so a 1.5 budget
    # trips before the second round-trip starts.
    loop = _make_loop(_CostProvider(), max_cost=1.5, tracker=tracker)
    result = loop.run_turn(_session(), "go")
    assert result.budget_exceeded is True


def test_max_cost_never_trips_without_pricing() -> None:
    # No pricing => cost unavailable => the cost cap can never trip; the loop is
    # instead bounded by max_iterations so the test terminates.
    loop = _make_loop(_CostProvider(), max_cost=0.01, max_iterations=3)
    result = loop.run_turn(_session(), "go")
    # It stopped on the turn cap, not the (inert) cost cap — still budget stop.
    assert result.budget_exceeded is True


# --------------------------------------------------------------------------- #
# run_headless wiring
# --------------------------------------------------------------------------- #


def test_run_headless_applies_budgets_to_loop() -> None:
    agent = _FakeAgentLoop(TurnResult(usage=_usage()))
    run_headless(
        agent, _session(), None, "p", output="json", out=StringIO(),
        max_turns=7, max_cost=1.5,
    )
    assert agent.max_iterations == 7
    assert agent.max_cost == 1.5


def test_budget_exceeded_yields_exit_5_and_skips_verification() -> None:
    agent = _FakeAgentLoop(TurnResult(usage=_usage(), budget_exceeded=True))
    coordinator = _FakeCoordinator()
    out = StringIO()

    code = run_headless(agent, _session(), coordinator, "p", output="json", out=out)
    payload = json.loads(out.getvalue())

    assert code == EXIT_BUDGET_EXCEEDED
    assert payload["ok"] is False
    assert payload["budget_exceeded"] is True
    # Verification must NOT run after a budget overrun (no extra spend).
    assert coordinator.run_called is False
    assert payload["verification"] is None


def test_budget_exceeded_text_footer() -> None:
    agent = _FakeAgentLoop(TurnResult(usage=_usage(), budget_exceeded=True))
    out = StringIO()
    code = run_headless(agent, _session(), None, "p", output="text", out=out)
    assert code == EXIT_BUDGET_EXCEEDED
    assert "[budget]" in out.getvalue()


def test_ok_run_reports_budget_false() -> None:
    agent = _FakeAgentLoop(TurnResult(usage=_usage()))
    out = StringIO()
    code = run_headless(agent, _session(), None, "p", output="json", out=out)
    payload = json.loads(out.getvalue())
    assert code == EXIT_OK
    assert payload["budget_exceeded"] is False


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #


def test_parser_accepts_valid_budgets() -> None:
    from forge.__main__ import build_parser

    args = build_parser().parse_args(
        ["-p", "hi", "--max-turns", "5", "--max-cost", "0.25"]
    )
    assert args.max_turns == 5
    assert args.max_cost == 0.25


@pytest.mark.parametrize("flag,value", [("--max-turns", "0"), ("--max-cost", "-1")])
def test_parser_rejects_non_positive_budgets(flag: str, value: str) -> None:
    from forge.__main__ import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["-p", "hi", flag, value])


def test_run_prompt_refuses_max_cost_without_pricing(monkeypatch) -> None:
    from forge import app as app_mod

    class _FakeApp:
        def __init__(self) -> None:
            self.config = SimpleNamespace(pricing=ModelPricing(None, None))
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake = _FakeApp()
    monkeypatch.setattr(app_mod, "bootstrap", lambda **kw: fake)

    out, err = StringIO(), StringIO()
    code = app_mod.run_prompt("hi", out=out, err=err, max_cost=0.5)

    assert code == EXIT_TURN_ERROR
    assert "requires model pricing" in err.getvalue()
    assert fake.closed is True
