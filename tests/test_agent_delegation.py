"""Tests for AgentLoop's subagent-usage aggregation and max_iterations cap."""

from __future__ import annotations

from forge.agent import AgentLoop
from forge.config import Config
from forge.interrupt import InterruptController
from forge.session import Session, ToolCall
from forge.tools.base import ToolResult
from forge.usage import UsageTracker
from forge.vertex import Done


class _FakeContextManager:
    def assemble(self, session):
        return [], None


class _FakeSessionStore:
    def save(self, session) -> None:
        pass


class _FakeExecutor:
    """Minimal executor: returns a scripted result per call name."""

    def __init__(self, results: dict[str, ToolResult]) -> None:
        self._results = results
        self._registry: dict = {}
        self.calls: list[str] = []

    def is_exposed(self, name: str) -> bool:
        return True

    def specs(self):
        return []

    def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call.name)
        return self._results.get(call.name, ToolResult(ok=True, content="ok"))


def _session() -> Session:
    return Session(id="s", created_at="t", updated_at="t")


def _loop(provider, executor, *, max_iterations=None) -> AgentLoop:
    return AgentLoop(
        context_manager=_FakeContextManager(),
        provider=provider,
        tool_executor=executor,
        usage_tracker=UsageTracker(Config()),
        session_store=_FakeSessionStore(),
        interrupt=InterruptController(),
        max_iterations=max_iterations,
    )


def test_subagent_usage_is_aggregated_into_parent_turn() -> None:
    """A delegate result's subagent_usage is folded into the parent turn usage."""

    class _DelegateOnceProvider:
        def __init__(self) -> None:
            self.calls = 0

        def generate_stream(self, contents, tools):
            self.calls += 1
            if self.calls == 1:
                yield ToolCall(id="d1", name="delegate", args={"task": "x"})
                yield Done()
            else:
                yield Done()

    executor = _FakeExecutor(
        {
            "delegate": ToolResult(
                ok=True,
                content="sub answer",
                meta={"subagent_usage": {"input_tokens": 100, "output_tokens": 50}},
            )
        }
    )
    loop = _loop(_DelegateOnceProvider(), executor)

    result = loop.run_turn(_session(), "delegate please")

    # The parent turn's usage includes the delegated tokens even though the
    # parent model itself reported no usage.
    assert result.usage.turn_input_tokens == 100
    assert result.usage.turn_output_tokens == 50


def test_max_iterations_caps_model_round_trips() -> None:
    """A finite max_iterations bounds the number of model round-trips."""

    class _AlwaysToolProvider:
        def __init__(self) -> None:
            self.calls = 0

        def generate_stream(self, contents, tools):
            self.calls += 1
            yield ToolCall(id=f"c{self.calls}", name="noop", args={})
            yield Done()

    provider = _AlwaysToolProvider()
    executor = _FakeExecutor({})
    loop = _loop(provider, executor, max_iterations=2)

    # Without the cap this provider would loop forever (it always emits a call).
    loop.run_turn(_session(), "go")

    assert provider.calls == 2
    assert executor.calls == ["noop", "noop"]


def test_unbounded_by_default() -> None:
    """max_iterations=None (default) does not truncate a normal turn."""

    class _StopsProvider:
        def __init__(self) -> None:
            self.calls = 0

        def generate_stream(self, contents, tools):
            self.calls += 1
            if self.calls == 1:
                yield ToolCall(id="c1", name="noop", args={})
                yield Done()
            else:
                yield Done()

    provider = _StopsProvider()
    loop = _loop(provider, _FakeExecutor({}))
    loop.run_turn(_session(), "go")
    # One tool round-trip + one final round-trip that stops.
    assert provider.calls == 2
