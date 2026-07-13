"""Unit tests for the agent loop's control flow (task 20.3).

These tests drive :meth:`forge.agent.AgentLoop.run_turn` with a scripted mock
``VertexClient`` to cover the three control-flow behaviors called out in the
design and requirements:

* a model response with **no tool calls** terminates the turn and returns
  control without executing any tool (Req 1.3);
* a **multi-tool** turn executes every call, appends one Tool_Result each, and
  **continues** issuing model requests until a response carries no tool calls
  (Req 1.5); and
* an **interrupt** mid-turn halts the loop while **retaining** the completed
  messages, tool calls, and Tool_Results already appended to the session
  (Req 4.5).

Collaborators are in-process fakes (no network, no disk). The real
:class:`~forge.usage.UsageTracker` and
:class:`~forge.interrupt.InterruptController` are used.
"""

from __future__ import annotations

from forge.agent import AgentLoop
from forge.interrupt import InterruptController
from forge.session import Session, ToolCall, Usage
from forge.tools.base import ToolResult
from forge.usage import UsageTracker
from forge.vertex import Done, TextDelta, UsageReport


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeContextManager:
    """Minimal context manager: serialize messages, never compact."""

    def __init__(self) -> None:
        self.assemble_calls = 0

    def assemble(self, session: Session):
        self.assemble_calls += 1
        contents = [
            {"role": m.role, "content": m.text} for m in session.messages
        ]
        return contents, None


class ScriptedVertexClient:
    """Yields a scripted list of stream events per ``generate_stream`` call."""

    def __init__(self, responses: list[list]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate_stream(self, contents, tools):
        self.calls += 1
        events = self._responses.pop(0)
        for event in events:
            yield event


class RecordingToolExecutor:
    """Records executed call ids and returns a success result for each."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    def specs(self):
        return []

    def execute(self, call: ToolCall) -> ToolResult:
        self.executed.append(call.id)
        return ToolResult(ok=True, content=f"ran {call.name}")


class TrippingToolExecutor:
    """Executes calls but trips the interrupt after ``trip_after`` executions.

    Simulates a Ctrl-C that lands while tools are running: the executor records
    and completes each call it is given, and once ``trip_after`` calls have run
    it trips the shared interrupt so the loop observes the interrupt before the
    next call / next iteration.
    """

    def __init__(self, interrupt: InterruptController, trip_after: int = 1) -> None:
        self._interrupt = interrupt
        self._trip_after = trip_after
        self.executed: list[str] = []

    def specs(self):
        return []

    def execute(self, call: ToolCall) -> ToolResult:
        self.executed.append(call.id)
        if len(self.executed) >= self._trip_after:
            self._interrupt.trip()
        return ToolResult(ok=True, content=f"ran {call.name}")


class RecordingSessionStore:
    """Captures saved sessions without touching disk."""

    def __init__(self) -> None:
        self.saved: list[Session] = []

    def save(self, session: Session) -> None:
        self.saved.append(session)


def _make_session() -> Session:
    return Session(
        id="session-under-test",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
        messages=[],
        todos=[],
        usage=Usage(input_tokens=0, output_tokens=0, estimated_cost=None),
    )


def _make_loop(
    vertex: ScriptedVertexClient,
    executor,
    *,
    interrupt: InterruptController | None = None,
    context: FakeContextManager | None = None,
    store: RecordingSessionStore | None = None,
) -> tuple[AgentLoop, RecordingSessionStore, FakeContextManager]:
    store = store or RecordingSessionStore()
    context = context or FakeContextManager()
    loop = AgentLoop(
        context_manager=context,
        vertex_client=vertex,
        tool_executor=executor,
        usage_tracker=UsageTracker(),
        session_store=store,
        interrupt=interrupt or InterruptController(),
    )
    return loop, store, context


# --------------------------------------------------------------------------- #
# No-tool-call response terminates the turn (Req 1.3)
# --------------------------------------------------------------------------- #


def test_no_tool_call_response_terminates_turn() -> None:
    """A response with no tool calls is displayed and ends the turn without
    invoking any tool.

    Validates: Requirements 1.3
    """

    response = [TextDelta(text="hello "), TextDelta(text="world"), UsageReport(10, 5), Done()]
    vertex = ScriptedVertexClient([response])
    executor = RecordingToolExecutor()
    loop, store, context = _make_loop(vertex, executor)

    session = _make_session()
    result = loop.run_turn(session, "hi")

    # Exactly one model request and no tools executed.
    assert vertex.calls == 1
    assert executor.executed == []

    # The conversation is [user, model] and the model text was accumulated.
    assert [m.role for m in session.messages] == ["user", "model"]
    assert session.messages[0].text == "hi"
    assert session.messages[1].text == "hello world"
    assert session.messages[1].tool_calls == []

    # Clean termination with the usage recorded for the single response.
    assert result.interrupted is False
    assert result.error is None
    assert result.usage.turn_input_tokens == 10
    assert result.usage.turn_output_tokens == 5

    # The session was persisted once.
    assert store.saved == [session]


# --------------------------------------------------------------------------- #
# Multi-tool turn continues until a no-tool response (Req 1.5)
# --------------------------------------------------------------------------- #


def test_multi_tool_turn_continues_until_no_tool_calls() -> None:
    """A turn with tool calls executes each call, appends one Tool_Result per
    call, and continues the loop until the model returns no tool calls.

    Validates: Requirements 1.5
    """

    calls = [
        ToolCall(id="c0", name="read", args={}),
        ToolCall(id="c1", name="search", args={}),
    ]
    first = [TextDelta(text="let me look"), *calls, Done()]
    second = [TextDelta(text="all set"), Done()]
    vertex = ScriptedVertexClient([first, second])
    executor = RecordingToolExecutor()
    loop, store, context = _make_loop(vertex, executor)

    session = _make_session()
    result = loop.run_turn(session, "do the thing")

    # The loop continued: a second model request followed the tool results.
    assert vertex.calls == 2
    assert context.assemble_calls == 2

    # Both tools ran in received order.
    assert executor.executed == ["c0", "c1"]

    # Conversation: user, model(tool_calls), tool(c0), tool(c1), model(final).
    assert [m.role for m in session.messages] == [
        "user",
        "model",
        "tool",
        "tool",
        "model",
    ]

    # The first model message carries both tool calls; one Tool_Result each.
    assert [c.id for c in session.messages[1].tool_calls] == ["c0", "c1"]
    tool_results = [m.tool_result for m in session.messages if m.role == "tool"]
    assert [r.call_id for r in tool_results] == ["c0", "c1"]

    # The final model message (no tool calls) terminated the turn.
    assert session.messages[-1].text == "all set"
    assert session.messages[-1].tool_calls == []

    assert result.interrupted is False
    assert result.error is None
    assert store.saved == [session]


# --------------------------------------------------------------------------- #
# Interrupt retention of completed messages/results (Req 4.5)
# --------------------------------------------------------------------------- #


def test_interrupt_retains_completed_messages_and_results() -> None:
    """An interrupt that trips while tools are running halts the turn but keeps
    every message, tool call, and Tool_Result already appended to the session.

    Validates: Requirements 4.5
    """

    calls = [
        ToolCall(id="c0", name="read", args={}),
        ToolCall(id="c1", name="write", args={}),
    ]
    # One response carrying two tool calls. The executor trips the interrupt
    # after the first call, so the loop stops before running the second.
    first = [TextDelta(text="partial output"), *calls, Done()]
    vertex = ScriptedVertexClient([first])
    interrupt = InterruptController()
    executor = TrippingToolExecutor(interrupt, trip_after=1)
    loop, store, context = _make_loop(vertex, executor, interrupt=interrupt)

    session = _make_session()
    result = loop.run_turn(session, "start work")

    # The turn was reported as interrupted.
    assert result.interrupted is True
    assert result.error is None

    # Only the first tool ran; the second was skipped once the interrupt tripped.
    assert executor.executed == ["c0"]

    # Completed state is retained: the user message, the model message with
    # BOTH tool calls, and exactly the one Tool_Result that was produced.
    assert [m.role for m in session.messages] == ["user", "model", "tool"]
    assert session.messages[0].text == "start work"
    assert session.messages[1].text == "partial output"
    assert [c.id for c in session.messages[1].tool_calls] == ["c0", "c1"]

    tool_results = [m.tool_result for m in session.messages if m.role == "tool"]
    assert len(tool_results) == 1
    assert tool_results[0].call_id == "c0"

    # The interrupted turn was still persisted (state retained on disk too).
    assert store.saved == [session]

    # No second model request was started after the interrupt.
    assert vertex.calls == 1
