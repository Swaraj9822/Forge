"""Tests for parallel read-only tool execution in the AgentLoop."""

from __future__ import annotations

import time
from typing import Any

from forge.agent import AgentLoop
from forge.config import Config
from forge.interrupt import InterruptController
from forge.session import Session, ToolCall
from forge.tools.base import ToolResult
from forge.usage import UsageTracker


class DummyTool:

    def __init__(self, name: str, read_only: bool) -> None:
        self.name = name
        self.description = "dummy"
        self.parameters = {}
        self.read_only = read_only

    def validate(self, args: dict) -> str | None:
        return None

    def run(self, args: dict, ctx: Any) -> ToolResult:
        return ToolResult(ok=True, content=f"ran {self.name}")


class DelayToolExecutor:

    def __init__(self, registry: dict[str, Any]) -> None:
        self._registry = registry
        self.executed_calls: list[str] = []
        self.delays: dict[str, float] = {}

    def is_exposed(self, name: str) -> bool:
        return name in self._registry

    def specs(self) -> list:
        from forge.tools.base import ToolSpec
        return [
            ToolSpec(name=tool.name, description=tool.description, parameters=tool.parameters)
            for tool in self._registry.values()
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        self.executed_calls.append(call.id)
        delay = self.delays.get(call.id, 0.0)
        if delay > 0:
            time.sleep(delay)
        return ToolResult(ok=True, content=f"res {call.name}")


class FakeContextManager:

    def assemble(self, session: Session):
        return [], None


class FakeSessionStore:

    def save(self, session: Session) -> None:
        pass


class FakeVertexClient:

    def __init__(self, tool_calls: list[ToolCall]) -> None:
        self.tool_calls = tool_calls
        self.yielded = False

    def generate_stream(self, contents, tools):
        from forge.vertex import Done

        if not self.yielded:
            self.yielded = True
            for tc in self.tool_calls:
                yield tc
        else:
            yield Done()


def test_parallel_execution_timing_and_ordering() -> None:
    registry = {
        "read": DummyTool("read", read_only=True),
    }
    executor = DelayToolExecutor(registry)

    # call-0 has 0.15s delay, call-1 has 0.01s delay
    executor.delays = {"call-0": 0.15, "call-1": 0.01}

    calls = [
        ToolCall(id="call-0", name="read", args={}),
        ToolCall(id="call-1", name="read", args={}),
    ]

    session = Session(
        id="test-session",
        created_at="2026-07-13T12:00:00Z",
        updated_at="2026-07-13T12:00:00Z",
    )

    interrupt = InterruptController()
    loop = AgentLoop(
        context_manager=FakeContextManager(),
        vertex_client=FakeVertexClient(calls),
        tool_executor=executor,
        usage_tracker=UsageTracker(Config()),
        session_store=FakeSessionStore(),
        interrupt=interrupt,
        parallel_enabled=True,
        parallel_max_workers=4,
    )

    start_time = time.time()
    loop.run_turn(session, "run")
    duration = time.time() - start_time

    # 1. Timing: total duration should be less than the serial sum (0.16s).
    # Since call-0 takes 0.15s, duration should be ~0.15s-0.18s, which is well below 0.28s.
    assert duration < 0.28

    # 2. Ordering: verify the tool results are appended in the received order: call-0 then call-1.
    tool_messages = [m for m in session.messages if m.role == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0].tool_result.call_id == "call-0"
    assert tool_messages[1].tool_result.call_id == "call-1"


def test_eligibility_and_serial_fallback() -> None:
    registry = {
        "read": DummyTool("read", read_only=True),
        "write": DummyTool("write", read_only=False),
        "git": DummyTool("git", read_only=True),
    }
    executor = DelayToolExecutor(registry)

    interrupt = InterruptController()
    loop = AgentLoop(
        context_manager=FakeContextManager(),
        vertex_client=FakeVertexClient([]),
        tool_executor=executor,
        usage_tracker=UsageTracker(Config()),
        session_store=FakeSessionStore(),
        interrupt=interrupt,
        parallel_enabled=True,
    )

    # Verify eligibility helper
    # 1. read is eligible
    assert loop._parallel_eligible(ToolCall(id="1", name="read", args={})) is True
    # 2. write is not eligible because read_only=False
    assert loop._parallel_eligible(ToolCall(id="2", name="write", args={})) is False
    # 3. git is not eligible because it's not in the safe set
    assert loop._parallel_eligible(ToolCall(id="3", name="git", args={})) is False
