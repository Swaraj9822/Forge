"""Integration tests for the live planning-tool -> agent -> session path.

These tests close two production-wiring gaps that the existing unit tests could
not catch because they used stand-in doubles:

* **Todo sync (Req 10.3, 10.5).** The real :class:`~forge.tools.planning.PlanningTool`
  keeps its list in the shared :class:`~forge.tools.base.ToolContext` ``state``
  bag. The :class:`~forge.agent.AgentLoop` must mirror that list onto
  ``session.todos`` (via the tool result's ``meta["todos"]``) so the REPL
  renders it and the store persists it. The prior REPL tests mutated
  ``session.todos`` directly through a ``FakeAgentLoop``, so they exercised the
  renderer but never the real tool -> loop -> session wiring.
* **Usage persistence (Req 13.1, 17.2).** ``run_turn`` must mirror the
  :class:`~forge.usage.UsageTracker` cumulative tallies and estimated cost onto
  ``session.usage`` before persisting, rather than leaving the saved record at
  zero.

The model is a scripted in-process fake; everything else (the real
:class:`AgentLoop`, :class:`ToolExecutor`, :class:`PlanningTool`,
:class:`UsageTracker`, and a real :class:`Repl`) is exercised end to end. No
network, no disk, no ``tmp_path`` fixture.
"""

from __future__ import annotations

import io
from pathlib import Path

from forge.agent import AgentLoop
from forge.config import ModelPricing
from forge.interrupt import InterruptController
from forge.repl import Repl
from forge.session import Session, ToolCall, Usage
from forge.tools.base import ToolContext, ToolExecutor
from forge.tools.planning import PlanningTool
from forge.usage import UsageTracker
from forge.vertex import Done, TextDelta, UsageReport


# --------------------------------------------------------------------------- #
# In-process fakes (no network / no disk)
# --------------------------------------------------------------------------- #


class FakeContextManager:
    """Serialize messages into bare contents; never compact."""

    def assemble(self, session: Session):
        return [{"role": m.role, "content": m.text} for m in session.messages], None


class ScriptedVertexClient:
    """Yield a scripted list of stream events per ``generate_stream`` call."""

    def __init__(self, responses: list[list]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate_stream(self, contents, tools):
        self.calls += 1
        for event in self._responses.pop(0):
            yield event


class RecordingSessionStore:
    """Capture saved sessions (by reference) without touching disk."""

    def __init__(self) -> None:
        self.saved: list[Session] = []

    def save(self, session: Session) -> None:
        self.saved.append(session)


def _make_session() -> Session:
    return Session(
        id="planning-integration",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
        messages=[],
        todos=[],
        usage=Usage(input_tokens=0, output_tokens=0, estimated_cost=None),
    )


def _make_executor(interrupt: InterruptController) -> ToolExecutor:
    """Build a real executor exposing only the real planning tool.

    A shared :class:`ToolContext` with an empty ``state`` bag is supplied, just
    like ``app._build_tool_executor`` wires it; the planning tool stores its
    list there. ``workspace_root`` is unused by the planning tool.
    """
    context = ToolContext(
        workspace_root=Path.cwd(), interrupt=interrupt, state={}
    )
    return ToolExecutor(
        registry={"planning": PlanningTool()},
        enabled={"planning"},
        interrupt=interrupt,
        context=context,
    )


def _make_loop(
    vertex: ScriptedVertexClient,
    executor: ToolExecutor,
    interrupt: InterruptController,
    store: RecordingSessionStore,
    *,
    pricing: ModelPricing | None = None,
    renderer=None,
) -> AgentLoop:
    return AgentLoop(
        context_manager=FakeContextManager(),
        vertex_client=vertex,
        tool_executor=executor,
        usage_tracker=UsageTracker(pricing),
        session_store=store,
        interrupt=interrupt,
        renderer=renderer,
    )


# --------------------------------------------------------------------------- #
# Todo sync through the live loop (Req 10.3, 10.5)
# --------------------------------------------------------------------------- #


def test_planning_tool_set_syncs_session_todos() -> None:
    """A planning ``set`` executed through the loop populates ``session.todos``.

    Validates: Requirements 10.3, 10.5
    """
    planning_call = ToolCall(
        id="t0",
        name="planning",
        args={
            "op": "set",
            "items": [
                {"text": "write tests"},
                {"text": "run suite", "status": "in_progress"},
            ],
        },
    )
    first = [TextDelta(text="planning"), planning_call, UsageReport(10, 5), Done()]
    second = [TextDelta(text="done"), UsageReport(0, 0), Done()]

    interrupt = InterruptController()
    vertex = ScriptedVertexClient([first, second])
    executor = _make_executor(interrupt)
    store = RecordingSessionStore()
    loop = _make_loop(vertex, executor, interrupt, store)

    session = _make_session()
    loop.run_turn(session, "plan my work")

    # The real planning tool's list was mirrored onto the session.
    assert [t.text for t in session.todos] == ["write tests", "run suite"]
    assert [t.status for t in session.todos] == ["pending", "in_progress"]
    # Generated sequential ids when none were supplied.
    assert [t.id for t in session.todos] == ["1", "2"]
    # The session that was persisted carries the synced todos.
    assert store.saved == [session]
    assert store.saved[0].todos is session.todos


def test_planning_tool_update_status_syncs_session_todos() -> None:
    """A subsequent ``update`` through the loop reflects the new status.

    Drives two turns against the same executor (shared ToolContext), proving the
    list persists across turns (Req 10.5) and that an in-loop status update is
    mirrored onto ``session.todos`` (Req 10.2/10.3).
    """
    interrupt = InterruptController()
    executor = _make_executor(interrupt)
    store = RecordingSessionStore()
    session = _make_session()

    # Turn 1: set two items.
    set_call = ToolCall(
        id="t0",
        name="planning",
        args={"op": "set", "items": [{"id": "a", "text": "first"}, {"id": "b", "text": "second"}]},
    )
    vertex1 = ScriptedVertexClient(
        [[set_call, Done()], [TextDelta(text="ok"), Done()]]
    )
    _make_loop(vertex1, executor, interrupt, store).run_turn(session, "plan")
    assert [t.status for t in session.todos] == ["pending", "pending"]

    # Turn 2: mark item "b" completed.
    update_call = ToolCall(
        id="t1",
        name="planning",
        args={"op": "update", "id": "b", "status": "completed"},
    )
    vertex2 = ScriptedVertexClient(
        [[update_call, Done()], [TextDelta(text="done"), Done()]]
    )
    _make_loop(vertex2, executor, interrupt, store).run_turn(session, "finish second")

    by_id = {t.id: t.status for t in session.todos}
    assert by_id == {"a": "pending", "b": "completed"}


def test_planning_changes_render_through_real_repl() -> None:
    """End-to-end: a planning tool call drives a real REPL to render todos.

    Wires a real :class:`Repl` to a real :class:`AgentLoop` + real
    :class:`PlanningTool`; the only fake is the scripted model. The REPL renders
    ``session.todos`` after the turn, so a planning ``set`` executed by the loop
    must surface in the captured output (Req 10.3) — the path the prior
    ``FakeAgentLoop`` tests could not cover.
    """
    planning_call = ToolCall(
        id="t0",
        name="planning",
        args={"op": "set", "items": [{"text": "draft design", "status": "completed"}]},
    )
    vertex = ScriptedVertexClient(
        [[TextDelta(text="planning "), planning_call, Done()], [TextDelta(text="all set"), Done()]]
    )
    interrupt = InterruptController()
    executor = _make_executor(interrupt)
    store = RecordingSessionStore()
    loop = _make_loop(vertex, executor, interrupt, store)

    session = _make_session()
    out = io.StringIO()
    repl = Repl(loop, session, input_func=lambda prompt: "plan it", out=out)

    keep_going = repl.run_once()

    rendered = out.getvalue()
    assert keep_going is True
    # The tool was announced and the todo list rendered from the synced session.
    assert "[tool: planning]" in rendered
    assert "[todos]" in rendered
    assert "[x] draft design" in rendered


# --------------------------------------------------------------------------- #
# Usage persistence onto the session (Req 13.1, 17.2)
# --------------------------------------------------------------------------- #


def test_usage_is_persisted_onto_session() -> None:
    """``run_turn`` mirrors cumulative token tallies onto ``session.usage``.

    Validates: Requirements 13.1, 17.2
    """
    first = [TextDelta(text="thinking"), UsageReport(100, 40), Done()]
    interrupt = InterruptController()
    vertex = ScriptedVertexClient([first])
    executor = _make_executor(interrupt)
    store = RecordingSessionStore()
    loop = _make_loop(vertex, executor, interrupt, store)

    session = _make_session()
    assert session.usage.input_tokens == 0  # zero before the turn

    result = loop.run_turn(session, "hi")

    # The session now reflects the tracker's cumulative tallies, not zero.
    assert session.usage.input_tokens == 100
    assert session.usage.output_tokens == 40
    # And it matches the turn result's cumulative summary.
    assert session.usage.input_tokens == result.usage.cumulative_input_tokens
    assert session.usage.output_tokens == result.usage.cumulative_output_tokens
    # The persisted session carries the usage.
    assert store.saved[0].usage.input_tokens == 100


def test_usage_accumulates_across_responses_within_a_turn() -> None:
    """Per-response usage across a multi-response turn lands on ``session.usage``.

    A tool-using turn issues two model responses; both responses' tokens must be
    summed onto the session's cumulative usage (Req 17.2).
    """
    planning_call = ToolCall(
        id="t0", name="planning", args={"op": "get"}
    )
    first = [planning_call, UsageReport(30, 10), Done()]
    second = [TextDelta(text="done"), UsageReport(5, 3), Done()]

    interrupt = InterruptController()
    vertex = ScriptedVertexClient([first, second])
    executor = _make_executor(interrupt)
    store = RecordingSessionStore()
    loop = _make_loop(vertex, executor, interrupt, store)

    session = _make_session()
    loop.run_turn(session, "go")

    assert session.usage.input_tokens == 35
    assert session.usage.output_tokens == 13


def test_persisted_usage_includes_estimated_cost_when_priced() -> None:
    """When pricing is configured, the session records the estimated cost.

    Validates: Requirements 17.2, 17.4
    """
    pricing = ModelPricing(input_per_1k=1.0, output_per_1k=2.0)
    first = [TextDelta(text="x"), UsageReport(1000, 1000), Done()]

    interrupt = InterruptController()
    vertex = ScriptedVertexClient([first])
    executor = _make_executor(interrupt)
    store = RecordingSessionStore()
    loop = _make_loop(vertex, executor, interrupt, store, pricing=pricing)

    session = _make_session()
    loop.run_turn(session, "cost please")

    # 1000/1000 * 1.0 + 1000/1000 * 2.0 == 3.0
    assert session.usage.estimated_cost == 3.0


def test_persisted_usage_cost_none_when_unpriced() -> None:
    """Without pricing, the persisted estimated cost is ``None`` (Req 17.5)."""
    first = [TextDelta(text="x"), UsageReport(10, 10), Done()]

    interrupt = InterruptController()
    vertex = ScriptedVertexClient([first])
    executor = _make_executor(interrupt)
    store = RecordingSessionStore()
    loop = _make_loop(vertex, executor, interrupt, store, pricing=None)

    session = _make_session()
    loop.run_turn(session, "no price")

    assert session.usage.estimated_cost is None
