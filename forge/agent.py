"""The autonomous agent loop.

This module implements :class:`AgentLoop`, the thin orchestration layer that
drives a single turn. ``run_turn`` appends the user's message, then repeats:
assemble the Context_Window (capturing any compaction that occurred), stream a
model response (rendering text and announcing tool names), collect the tool
calls the model emitted, execute them in received order, append exactly one
Tool_Result per call, and continue until the model returns a response that
carries no tool calls. After the loop it persists the session and returns a
:class:`TurnResult` carrying the usage summary and any
:class:`~forge.context.CompactionInfo`.

Design choices
--------------
* **Dependency injection.** All collaborators (:class:`ContextManager`,
  :class:`VertexClient`, :class:`ToolExecutor`, :class:`UsageTracker`,
  :class:`SessionStore`, :class:`InterruptController`) are injected through the
  constructor so the loop is testable with scripted mocks (task 20.3).
* **Decoupled rendering.** Streaming output, tool-name announcements, and the
  compaction notice are surfaced through an optional :class:`Renderer` (a small
  ``Protocol``). The REPL (task 21.1) supplies the real renderer; the default is
  a no-op so the loop runs headless in tests. Each renderer hook is invoked
  defensively (``hasattr`` guarded) so a partial renderer is fine.
* **Interrupt ownership.** ``run_turn`` owns the turn boundary: it brackets the
  whole turn with :meth:`InterruptController.begin_turn` /
  :meth:`InterruptController.end_turn` so a Ctrl-C during the turn trips the
  shared event and reverts to an idle no-op afterward. The loop polls
  :meth:`InterruptController.check` between streamed events and between tool
  executions; on a tripped interrupt it stops promptly, retains every message
  and Tool_Result already appended to the session, and exits without starting a
  new model request (Req 4.5). The :class:`VertexClient` independently aborts
  its own stream on the same event, so generation also stops within ~1s.

Requirements: 1.2, 1.3, 1.4, 1.5, 3.2, 4.1, 4.5, 14.7.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from forge.context import CompactionInfo, ContextManager
from forge.interrupt import InterruptController
from forge.session import (
    Message,
    Session,
    SessionStore,
    TodoItem,
    ToolCall,
    ToolResultRecord,
    Usage,
)
from forge.tools.base import ToolExecutor, ToolResult
from forge.usage import UsageSummary, UsageTracker
from forge.vertex import Done, TextDelta, UsageReport, VertexClient, VertexError

__all__ = ["Renderer", "NullRenderer", "TurnResult", "AgentLoop"]


# --------------------------------------------------------------------------- #
# Rendering protocol
# --------------------------------------------------------------------------- #


@runtime_checkable
class Renderer(Protocol):
    """Optional sink for the loop's user-visible output.

    The REPL implements this to render streamed text within 200 ms per token
    (Req 3.1), announce tool names before a tool runs (Req 3.2), and render the
    "conversation context was compacted" notice (Req 14.7). Every method is
    optional: the :class:`AgentLoop` invokes each hook only if the renderer
    provides it, so a renderer may implement just the subset it needs.
    """

    def on_text(self, text: str) -> None:
        """Render a streamed fragment of model text."""
        ...

    def on_tool(self, name: str) -> None:
        """Announce that the named tool is about to be invoked (Req 3.2)."""
        ...

    def on_compaction(self, info: CompactionInfo) -> None:
        """Render the context-compaction notice (Req 14.7)."""
        ...


class NullRenderer:
    """A renderer that discards all output (the default).

    Lets the :class:`AgentLoop` run fully headless (tests, non-interactive use)
    without a real terminal renderer.
    """

    def on_text(self, text: str) -> None:  # noqa: D102 - no-op
        return None

    def on_tool(self, name: str) -> None:  # noqa: D102 - no-op
        return None

    def on_compaction(self, info: CompactionInfo) -> None:  # noqa: D102 - no-op
        return None


# --------------------------------------------------------------------------- #
# Turn result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TurnResult:
    """The product of one :meth:`AgentLoop.run_turn` call.

    Attributes:
        usage: The :class:`~forge.usage.UsageSummary` for the turn (turn and
            cumulative token tallies plus estimated cost) (Req 17.3).
        compaction: The first :class:`~forge.context.CompactionInfo` produced
            during the turn, or ``None`` if no compaction occurred. The Repl
            renders this as the compaction notice (Req 14.7).
        error: A human-readable message when a Vertex error ended the turn
            gracefully, else ``None``. Session state is preserved either way so
            the Repl can render the error and return to the prompt (Req 2.5,
            2.6, 2.8).
        interrupted: ``True`` when a user interrupt halted the turn. Completed
            messages and Tool_Results are retained in the session (Req 4.5).
        mutated_files: ``True`` when the turn produced at least one
            File_Mutation -- a :class:`ToolResult` with ``ok=True`` from a tool
            named ``write`` or ``edit``. The Verification_Phase reads this as
            the ``on_file_change`` trigger signal (Req 3.1, 3.2).
    """

    usage: UsageSummary
    compaction: CompactionInfo | None = None
    error: str | None = None
    interrupted: bool = False
    mutated_files: bool = False


# --------------------------------------------------------------------------- #
# AgentLoop
# --------------------------------------------------------------------------- #


class AgentLoop:
    """Orchestrates a single turn of the autonomous agent.

    Parameters
    ----------
    context_manager:
        Assembles the Context_Window and reports compaction (Req 14.7).
    vertex_client:
        Streams the model response as :data:`~forge.vertex.StreamEvent` values.
    tool_executor:
        Advertises the exposed tools (:meth:`ToolExecutor.specs`) and runs each
        tool call (:meth:`ToolExecutor.execute`).
    usage_tracker:
        Records per-response token usage and summarizes the turn.
    session_store:
        Persists the session when the turn completes (Req 13.1).
    interrupt:
        The shared interrupt controller; ``run_turn`` brackets the turn with
        its ``begin_turn``/``end_turn`` and polls ``check`` to stop promptly.
    renderer:
        Optional :class:`Renderer` for streamed text, tool announcements, and
        the compaction notice. Defaults to a no-op :class:`NullRenderer`.
    """

    def __init__(
        self,
        context_manager: ContextManager,
        vertex_client: VertexClient,
        tool_executor: ToolExecutor,
        usage_tracker: UsageTracker,
        session_store: SessionStore,
        interrupt: InterruptController,
        renderer: Renderer | None = None,
    ) -> None:
        self.context_manager = context_manager
        self.vertex_client = vertex_client
        self.tool_executor = tool_executor
        self.usage_tracker = usage_tracker
        self.session_store = session_store
        self.interrupt = interrupt
        self.renderer: Renderer = renderer or NullRenderer()

    # -- public API ----------------------------------------------------------

    def run_turn(self, session: Session, user_text: str) -> TurnResult:
        """Run one turn: send ``user_text`` and loop until no tool calls remain.

        Appends the user message (Req 1.2), then repeats assemble -> stream ->
        execute-tools until the model returns a response with no tool calls
        (Req 1.3, 1.5) or the turn is halted by an interrupt or a Vertex error.
        Persists the session afterward (Req 13.1) and returns a
        :class:`TurnResult`.
        """
        session.messages.append(Message(role="user", text=user_text))

        compaction: CompactionInfo | None = None
        error: str | None = None
        interrupted = False
        mutated_files = False

        # The agent loop owns the turn boundary: begin_turn clears any stale
        # interrupt and arms the SIGINT handler to trip the shared event;
        # end_turn reverts to the idle no-op once the turn is over.
        self.usage_tracker.begin_turn()
        self.interrupt.begin_turn()
        try:
            while True:
                if self.interrupt.check():
                    interrupted = True
                    break

                # (a) Assemble context; keep the FIRST compaction info to
                # surface on the result (Req 14.7).
                contents, info = self.context_manager.assemble(session)
                if info is not None and compaction is None:
                    compaction = info
                    self._render_compaction(info)

                # (b) Stream the model response.
                outcome = self._stream_response(contents)

                # (c) Append the model message with accumulated text + calls.
                session.messages.append(
                    Message(
                        role="model",
                        text=outcome.text or None,
                        tool_calls=list(outcome.tool_calls),
                    )
                )

                # A Vertex error ends the turn gracefully; state is preserved.
                if outcome.error is not None:
                    error = outcome.error
                    break

                # An interrupt observed mid-stream halts the turn (Req 4.5).
                if outcome.interrupted:
                    interrupted = True
                    break

                # (d) No tool calls -> the turn is complete (Req 1.3).
                if not outcome.tool_calls:
                    break

                # (e) Execute each tool call in received order, appending
                # exactly one Tool_Result per call (Req 1.4, 4.1).
                batch = self._execute_tool_calls(session, outcome.tool_calls)
                # A File_Mutation in any batch flags the whole turn (Req 3.1).
                mutated_files = mutated_files or batch.mutated_files
                if batch.interrupted:
                    interrupted = True
                    break
                # Results were appended; continue the loop to feed them back to
                # the model (Req 1.5).
        finally:
            self.interrupt.end_turn()

        # Persist the session once the turn ends, whatever the outcome
        # (completed, interrupted, or errored): all appended messages and
        # results are retained (Req 13.1, 4.5).
        summary = self.usage_tracker.turn_summary()
        # Mirror the tracker's cumulative (session) tallies and estimated cost
        # onto the session so the persisted record reflects real usage rather
        # than staying at zero (Req 13.1, 17.2).
        session.usage = Usage(
            input_tokens=summary.cumulative_input_tokens,
            output_tokens=summary.cumulative_output_tokens,
            estimated_cost=summary.cumulative_cost,
        )
        session.updated_at = datetime.now(timezone.utc).isoformat()
        self.session_store.save(session)

        return TurnResult(
            usage=summary,
            compaction=compaction,
            error=error,
            interrupted=interrupted,
            mutated_files=mutated_files,
        )

    # -- streaming -----------------------------------------------------------

    def _stream_response(self, contents: list[dict]) -> "_StreamOutcome":
        """Consume one model response stream into accumulated text + tool calls.

        Renders text deltas (Req 3.1) and tool-name announcements (Req 3.2),
        records per-response usage (Req 17.1), and polls the interrupt between
        events so an in-flight response stops promptly (Req 4.5). A typed
        :class:`~forge.vertex.VertexError` is captured (not raised) so the turn
        can end gracefully with the partial text retained.

        Token usage is recorded exactly once per response: ``usage_metadata``
        is reported as cumulative counts for the response and may appear on
        several streamed chunks, so only the last :class:`UsageReport` seen is
        recorded. Summing every report would inflate the turn/session totals
        (and estimated cost).
        """
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        outcome = _StreamOutcome(tool_calls=tool_calls)
        last_usage: UsageReport | None = None

        try:
            specs = self.tool_executor.specs()
            for event in self.vertex_client.generate_stream(contents, specs):
                if self.interrupt.check():
                    outcome.interrupted = True
                    break

                if isinstance(event, TextDelta):
                    text_parts.append(event.text)
                    self._render_text(event.text)
                elif isinstance(event, ToolCall):
                    tool_calls.append(event)
                    self._render_tool(event.name)
                elif isinstance(event, UsageReport):
                    # Keep the latest cumulative report; record once below.
                    last_usage = event
                elif isinstance(event, Done):
                    # Normal end of this model response.
                    pass

                if self.interrupt.check():
                    outcome.interrupted = True
                    break
        except VertexError as exc:
            # Surface the error so the turn ends without losing session state
            # (Req 2.5, 2.6, 2.8); partial text already rendered is retained.
            outcome.error = str(exc) or exc.__class__.__name__

        # Record this response's usage exactly once (Req 17.1, 17.2).
        if last_usage is not None:
            self.usage_tracker.record(
                last_usage.input_tokens, last_usage.output_tokens
            )

        outcome.text = "".join(text_parts)
        return outcome

    # -- tool execution ------------------------------------------------------

    def _execute_tool_calls(
        self, session: Session, calls: list[ToolCall]
    ) -> "_ToolBatchOutcome":
        """Execute ``calls`` in order, appending one Tool_Result per call.

        Returns a :class:`_ToolBatchOutcome` whose ``interrupted`` is ``True``
        when an interrupt halted execution (so the caller stops the turn) and
        whose ``mutated_files`` is ``True`` when at least one call was a
        File_Mutation -- a successful (``ok=True``) result from a tool named
        ``write`` or ``edit`` (Req 3.1, 3.2). The interrupt is polled before
        each call so a tripped event stops execution promptly (Req 4.5); calls
        already executed keep their appended Tool_Results.
        """
        mutated_files = False
        for call in calls:
            if self.interrupt.check():
                return _ToolBatchOutcome(
                    interrupted=True, mutated_files=mutated_files
                )
            result = self.tool_executor.execute(call)
            # A successful write/edit is a File_Mutation; flag the batch so the
            # turn surfaces it for the on_file_change verification trigger.
            if result.ok and call.name in ("write", "edit"):
                mutated_files = True
            # Mirror any todo-list state the tool reports (e.g. the planning
            # tool, which keeps its list in the shared ToolContext.state) onto
            # the session so the REPL renders it (Req 10.3) and it persists
            # across turns (Req 10.5). The tool exposes its authoritative list
            # via meta["todos"]; syncing here keeps the loop decoupled from the
            # ToolContext internals.
            self._sync_todos(session, result)
            session.messages.append(
                Message(
                    role="tool",
                    text=None,
                    tool_result=_to_record(call, result),
                )
            )
        return _ToolBatchOutcome(interrupted=False, mutated_files=mutated_files)

    @staticmethod
    def _sync_todos(session: Session, result: ToolResult) -> None:
        """Mirror a tool result's reported todo list onto ``session.todos``.

        A tool that maintains a session-scoped todo list (the planning tool)
        includes the current serialized list under ``meta["todos"]`` on every
        result. When present, rebuild ``session.todos`` from it so the session
        is the single source of truth the REPL renders and the store persists.
        Results without that key leave the existing list untouched.
        """
        todos = result.meta.get("todos")
        if not isinstance(todos, list):
            return
        session.todos = [
            TodoItem(id=t["id"], text=t["text"], status=t["status"])
            for t in todos
        ]

    # -- renderer hooks (defensively guarded) --------------------------------

    def _render_text(self, text: str) -> None:
        hook = getattr(self.renderer, "on_text", None)
        if callable(hook):
            hook(text)

    def _render_tool(self, name: str) -> None:
        hook = getattr(self.renderer, "on_tool", None)
        if callable(hook):
            hook(name)

    def _render_compaction(self, info: CompactionInfo) -> None:
        hook = getattr(self.renderer, "on_compaction", None)
        if callable(hook):
            hook(info)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


@dataclass
class _StreamOutcome:
    """Mutable accumulator for one model response stream."""

    tool_calls: list[ToolCall]
    text: str = ""
    interrupted: bool = False
    error: str | None = None


@dataclass(frozen=True)
class _ToolBatchOutcome:
    """The result of executing one batch of tool calls within a turn.

    Attributes:
        interrupted: ``True`` when an interrupt halted the batch before every
            call ran (Req 4.5).
        mutated_files: ``True`` when at least one call in the batch was a
            File_Mutation -- a successful (``ok=True``) result from the
            ``write`` or ``edit`` tool (Req 3.1, 3.2).
    """

    interrupted: bool
    mutated_files: bool


def _to_record(call: ToolCall, result: ToolResult) -> ToolResultRecord:
    """Convert a :class:`ToolResult` into a session :class:`ToolResultRecord`."""
    return ToolResultRecord(
        call_id=call.id,
        ok=result.ok,
        content=result.content,
        error=result.error,
        meta=result.meta,
    )
