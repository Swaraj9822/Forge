"""Property-based test for tool-call execution order in the agent loop.

# Feature: forge, Property 3: Tool calls execute in received order

This property exercises :meth:`forge.agent.AgentLoop.run_turn` with a scripted
``VertexClient`` that yields a deterministic sequence of tool calls and a
recording ``ToolExecutor`` stub. It asserts the two guarantees of Property 3
(Requirements 1.4, 1.5):

* the executor runs the model's tool calls in *exactly* the order received, and
* the agent loop appends *exactly one* ``Tool_Result`` per call within the
  turn, in the same order.

Everything is in-process: no network, no disk. The real
:class:`~forge.usage.UsageTracker` and :class:`~forge.interrupt.InterruptController`
are used; ``ContextManager`` and ``SessionStore`` are replaced by tiny fakes so
the property stays fast and offline.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.agent import AgentLoop
from forge.interrupt import InterruptController
from forge.session import Message, Session, ToolCall, Usage
from forge.tools.base import ToolResult
from forge.usage import UsageTracker
from forge.vertex import Done, TextDelta, UsageReport


# --------------------------------------------------------------------------- #
# Lightweight fakes (no network, no disk)
# --------------------------------------------------------------------------- #


class FakeContextManager:
    """Returns a trivial wire-shape window and never reports compaction.

    The agent loop only needs ``assemble(session) -> (contents, info)``. The
    contents are irrelevant to this property (the scripted client ignores
    them), so we return a minimal serialization of the session's messages.
    """

    def assemble(self, session: Session):
        contents = [
            {"role": m.role, "content": m.text} for m in session.messages
        ]
        return contents, None


class ScriptedVertexClient:
    """Yields a pre-scripted sequence of stream events per ``generate_stream``.

    ``responses`` is a list of event lists; each call to ``generate_stream``
    consumes the next list and yields its events in order. The recorded
    ``calls`` let a test assert how many model requests the loop issued.
    """

    def __init__(self, responses: list[list]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple] = []

    def generate_stream(self, contents, tools):
        self.calls.append((contents, tools))
        # Pop the next scripted response; an unexpected extra request surfaces
        # loudly as an IndexError rather than silently looping.
        events = self._responses.pop(0)
        for event in events:
            yield event


class RecordingToolExecutor:
    """Records the order tool calls are executed and returns a success result.

    ``executed`` captures the ``id`` of every executed call in execution order
    so the test can compare it against the order the calls were received in.
    """

    def __init__(self) -> None:
        self.executed: list[str] = []

    def specs(self):
        return []

    def execute(self, call: ToolCall) -> ToolResult:
        self.executed.append(call.id)
        return ToolResult(ok=True, content=f"ran {call.name}")


class RecordingSessionStore:
    """Captures saved sessions so persistence can be asserted without disk."""

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


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #

# A list of tool names (content irrelevant to ordering; identity is tracked by
# the per-index call id). At least one call so the first response carries tools.
_tool_names = st.lists(
    st.text(
        alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
        min_size=1,
        max_size=10,
    ),
    min_size=1,
    max_size=8,
)


# --------------------------------------------------------------------------- #
# Property 3
# --------------------------------------------------------------------------- #


@settings(max_examples=10)
@given(names=_tool_names)
def test_tool_calls_execute_in_received_order(names: list[str]) -> None:
    """The executor runs tool calls in received order and the loop appends
    exactly one Tool_Result per call, in the same order.

    Validates: Requirements 1.4, 1.5
    """

    # Build the model's tool calls with stable, unique ids so ordering can be
    # verified independent of (possibly repeated) tool names.
    calls = [
        ToolCall(id=f"call-{i}", name=name, args={})
        for i, name in enumerate(names)
    ]
    received_ids = [c.id for c in calls]

    # First model response: a little text, then all the tool calls, then Done.
    # Second response: no tool calls, which terminates the turn (Req 1.3).
    first_response = [TextDelta(text="working"), *calls, Done()]
    second_response = [TextDelta(text="done"), UsageReport(2, 1), Done()]

    vertex = ScriptedVertexClient([first_response, second_response])
    executor = RecordingToolExecutor()
    store = RecordingSessionStore()

    loop = AgentLoop(
        context_manager=FakeContextManager(),
        vertex_client=vertex,
        tool_executor=executor,
        usage_tracker=UsageTracker(),
        session_store=store,
        interrupt=InterruptController(),
    )

    session = _make_session()
    result = loop.run_turn(session, "please use the tools")

    # The turn completed normally (no interrupt, no error).
    assert result.interrupted is False
    assert result.error is None

    # (1.4) Tools executed in exactly the received order.
    assert executor.executed == received_ids

    # (1.5) Exactly one Tool_Result appended per call, in received order.
    tool_results = [
        m.tool_result for m in session.messages if m.role == "tool"
    ]
    assert len(tool_results) == len(calls)
    assert [r.call_id for r in tool_results] == received_ids

    # The loop issued exactly two model requests: the tool turn plus the
    # terminating no-tool response.
    assert len(vertex.calls) == 2

    # The completed session was persisted once at the end of the turn.
    assert store.saved == [session]
