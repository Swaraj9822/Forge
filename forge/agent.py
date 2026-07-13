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
from typing import Any, Protocol, runtime_checkable

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
from forge.providers import Done, TextDelta, UsageReport, Provider, ProviderError
from forge.ui import summarize_result

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

    def on_tool(self, name: str, args: dict | None = None) -> None:
        """Announce that the named tool is about to be invoked (Req 3.2).

        ``args`` carries the tool call's arguments so a renderer can describe
        *what* the tool is doing (the target path, command, pattern, etc.).
        """
        ...

    def on_compaction(self, info: CompactionInfo) -> None:
        """Render the context-compaction notice (Req 14.7)."""
        ...

    def on_tool_result(
        self,
        name: str,
        *,
        denied: bool,
        forbidden: bool,
        diff: str | None,
        ok: bool = True,
        summary: str | None = None,
    ) -> None:
        """Render a post-execution notice for a single tool result (Phase 2).

        Called by the loop after a tool result is appended to the session so a
        renderer can surface ``[denied]`` / ``[forbidden]`` notices, a concise
        success ``summary`` (e.g. ``"42 lines"``), a failure message when
        ``ok`` is ``False``, and optionally display the unified diff under the
        ``show_diffs`` config. ``denied`` and ``forbidden`` come from the
        approval policy; ``diff`` is the ``meta["diff"]`` of a successful
        write/edit when the tool provided one. The hook is optional — renderers
        that do not implement it are silently skipped.
        """
        ...


class NullRenderer:
    """A renderer that discards all output (the default).

    Lets the :class:`AgentLoop` run fully headless (tests, non-interactive use)
    without a real terminal renderer.
    """

    def on_text(self, text: str) -> None:  # noqa: D102 - no-op
        return None

    def on_tool(self, name: str, args: dict | None = None) -> None:  # noqa: D102 - no-op
        return None

    def on_compaction(self, info: CompactionInfo) -> None:  # noqa: D102 - no-op
        return None

    def on_tool_result(self, name: str, *, denied: bool, forbidden: bool,
                       diff: str | None, ok: bool = True,
                       summary: str | None = None) -> None:  # noqa: D102 - no-op
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
        budget_exceeded: ``True`` when a configured cap (``max_iterations`` /
            ``max_cost``) stopped the loop while the model still wanted to
            continue (its last response emitted tool calls). Headless runs map
            this to a dedicated exit code so a CI budget overrun is observable.
    """

    usage: UsageSummary
    compaction: CompactionInfo | None = None
    error: str | None = None
    interrupted: bool = False
    mutated_files: bool = False
    budget_exceeded: bool = False


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
    checkpoint:
        Optional :class:`~forge.checkpoint.CheckpointStore`. When supplied,
        :meth:`run_turn` brackets the turn with ``begin_turn`` / ``commit_turn``
        so :meth:`~forge.checkpoint.CheckpointStore.undo_last` reverts the
        most recent turn's file mutations.
    """

    def __init__(
        self,
        context_manager: ContextManager,
        provider: Provider | None = None,
        tool_executor: ToolExecutor | None = None,
        usage_tracker: UsageTracker | None = None,
        session_store: SessionStore | None = None,
        interrupt: InterruptController | None = None,
        renderer: Renderer | None = None,
        checkpoint: Any | None = None,
        parallel_enabled: bool = False,
        parallel_max_workers: int = 4,
        vertex_client: Provider | None = None,
        max_iterations: int | None = None,
        max_cost: float | None = None,
    ) -> None:
        self.context_manager = context_manager
        self.provider = provider or vertex_client
        self.vertex_client = self.provider
        self.tool_executor = tool_executor
        self.usage_tracker = usage_tracker
        self.session_store = session_store
        self.interrupt = interrupt
        self.checkpoint = checkpoint
        self.renderer: Renderer = renderer or NullRenderer()
        self.parallel_enabled = bool(parallel_enabled)
        self.parallel_max_workers = int(parallel_max_workers)
        # Optional cap on the number of model round-trips within a single turn.
        # None means unlimited (the interactive/main loop). Subagents pass a
        # finite value so their bounded-turn contract is actually enforced.
        self.max_iterations = max_iterations
        # Optional cumulative-cost budget (USD). None means no cost cap. Used by
        # headless runs (--max-cost) to stop before starting another model
        # round-trip once the estimated cost crosses the budget. Only effective
        # when model pricing is configured (otherwise cost is unavailable).
        self.max_cost = max_cost

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
        budget_exceeded = False
        iterations = 0

        # The agent loop owns the turn boundary: begin_turn clears any stale
        # interrupt and arms the SIGINT handler to trip the shared event;
        # end_turn reverts to the idle no-op once the turn is over.
        # Phase 2: when a checkpoint store is wired, begin/commit brackets the
        # turn body so /undo restores whole turns, not individual tool calls.
        self.usage_tracker.begin_turn()
        self.interrupt.begin_turn()
        if self.checkpoint is not None:
            self.checkpoint.begin_turn()
        try:
            while True:
                if self.interrupt.check():
                    interrupted = True
                    break

                # Bounded-turn cap (subagents / headless --max-turns). None =>
                # unlimited. Each loop iteration is one model round-trip; stop
                # once the cap is hit. Reaching this check means the previous
                # iteration emitted tool calls (otherwise the loop already broke
                # on "no tool calls"), so the model wanted to continue -> flag a
                # budget stop.
                if (
                    self.max_iterations is not None
                    and iterations >= self.max_iterations
                ):
                    budget_exceeded = True
                    break

                # Cost budget (headless --max-cost). Checked before starting the
                # next round-trip so we never begin a request that would push
                # spend past the budget. Only effective when pricing is
                # configured (cost_available); otherwise cumulative_cost is None
                # and this never trips.
                if self.max_cost is not None:
                    summary = self.usage_tracker.turn_summary()
                    if (
                        summary.cost_available
                        and summary.cumulative_cost is not None
                        and summary.cumulative_cost >= self.max_cost
                    ):
                        budget_exceeded = True
                        break
                iterations += 1

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
            if self.checkpoint is not None:
                self.checkpoint.commit_turn()

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
            budget_exceeded=budget_exceeded,
        )

    # -- streaming -----------------------------------------------------------

    def _stream_response(self, contents: list[dict]) -> "_StreamOutcome":
        """Consume one model response stream into accumulated text + tool calls.

        Renders text deltas (Req 3.1) and tool-name announcements (Req 3.2),
        record per-response usage (Req 17.1), and polls the interrupt between
        events so an in-flight response stops promptly (Req 4.5). A typed
        :class:`~forge.providers.ProviderError` is captured (not raised) so the
        turn can end gracefully with the partial text retained.

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
            stream = self.provider.generate_stream(contents, specs)

            status_hook = getattr(self.renderer, "status", None)
            first_event = None
            if status_hook is not None:
                with status_hook("Waiting for model response..."):
                    try:
                        first_event = next(stream)
                    except StopIteration:
                        pass
            else:
                try:
                    first_event = next(stream)
                except StopIteration:
                    pass

            import itertools
            events = [first_event] if first_event is not None else []

            for event in itertools.chain(events, stream):
                if self.interrupt.check():
                    outcome.interrupted = True
                    break

                if isinstance(event, TextDelta):
                    text_parts.append(event.text)
                    self._render_text(event.text)
                elif isinstance(event, ToolCall):
                    tool_calls.append(event)
                    self._render_tool(event.name, event.args)
                elif isinstance(event, UsageReport):
                    # Keep the latest cumulative report; record once below.
                    last_usage = event
                elif isinstance(event, Done):
                    # Normal end of this model response.
                    pass

                if self.interrupt.check():
                    outcome.interrupted = True
                    break
        except ProviderError as exc:
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

    def _parallel_eligible(self, call: ToolCall) -> bool:
        """Return True if the call is eligible for parallel execution."""
        is_exposed_fn = getattr(self.tool_executor, "is_exposed", None)
        if is_exposed_fn is not None:
            if not is_exposed_fn(call.name):
                return False

        get_tool = getattr(self.tool_executor, "get_tool", None)
        if get_tool is not None:
            tool = get_tool(call.name)
        else:  # pragma: no cover - legacy executors without the accessor
            registry = getattr(self.tool_executor, "_registry", None)
            tool = registry.get(call.name) if registry is not None else None
        if tool is None:
            return False

        read_only = getattr(tool, "read_only", False)
        safe_set = {"read", "search", "search_memory", "repo_index"}
        return bool(read_only) and (call.name in safe_set)

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
        lead = []
        i = 0
        while i < len(calls) and self._parallel_eligible(calls[i]):
            lead.append(calls[i])
            i += 1

        lead_results = []
        interrupted = False

        if len(lead) > 1 and self.parallel_enabled:
            if self.interrupt.check():
                return _ToolBatchOutcome(interrupted=True, mutated_files=False)

            import concurrent.futures

            futures = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.parallel_max_workers
            ) as executor:
                for call in lead:
                    futures.append(
                        executor.submit(self.tool_executor.execute, call)
                    )

                for fut in futures:
                    if self.interrupt.check():
                        interrupted = True
                        break
                    try:
                        res = fut.result()
                    except Exception as exc:
                        res = ToolResult(ok=False, content="", error=str(exc))
                    lead_results.append(res)
                    if self.interrupt.check():
                        interrupted = True
                        break
        else:
            for call in lead:
                if self.interrupt.check():
                    interrupted = True
                    break
                res = self.tool_executor.execute(call)
                lead_results.append(res)
                if self.interrupt.check():
                    interrupted = True
                    break

        for call, result in zip(lead[:len(lead_results)], lead_results):
            if result.ok and call.name in ("write", "edit"):
                mutated_files = True
            self._sync_todos(session, result)
            self._maybe_record_subagent_usage(result)
            session.messages.append(
                Message(
                    role="tool",
                    text=None,
                    tool_result=_to_record(call, result),
                )
            )
            self._render_tool_result(call, result)

        if interrupted:
            return _ToolBatchOutcome(
                interrupted=True, mutated_files=mutated_files
            )

        for call in calls[len(lead):]:
            if self.interrupt.check():
                return _ToolBatchOutcome(
                    interrupted=True, mutated_files=mutated_files
                )
            result = self.tool_executor.execute(call)
            if result.ok and call.name in ("write", "edit"):
                mutated_files = True
            self._sync_todos(session, result)
            self._maybe_record_subagent_usage(result)
            session.messages.append(
                Message(
                    role="tool",
                    text=None,
                    tool_result=_to_record(call, result),
                )
            )
            self._render_tool_result(call, result)

        return _ToolBatchOutcome(
            interrupted=False, mutated_files=mutated_files
        )

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

    def _maybe_record_subagent_usage(self, result: ToolResult) -> None:
        """Fold a subagent's token usage into this turn's usage tracker.

        The ``delegate`` tool runs a nested agent loop with its own usage
        tracker and reports the delegated token counts under
        ``meta["subagent_usage"]``. Recording them here makes the parent turn's
        (and session's) usage and cost include the delegated work, rather than
        siloing it in the subagent (Phase 5, Feature N usage aggregation).
        """
        usage = (result.meta or {}).get("subagent_usage")
        if not isinstance(usage, dict):
            return
        try:
            self.usage_tracker.record(
                int(usage.get("input_tokens", 0)),
                int(usage.get("output_tokens", 0)),
            )
        except (TypeError, ValueError):
            # Malformed usage meta must never break the turn.
            return

    # -- renderer hooks (defensively guarded) --------------------------------

    def _render_text(self, text: str) -> None:
        hook = getattr(self.renderer, "on_text", None)
        if callable(hook):
            hook(text)

    def _render_tool(self, name: str, args: dict | None = None) -> None:
        hook = getattr(self.renderer, "on_tool", None)
        if callable(hook):
            try:
                hook(name, args)
            except TypeError:
                # Renderer implements the older on_tool(name) signature only.
                hook(name)

    def _render_compaction(self, info: CompactionInfo) -> None:
        hook = getattr(self.renderer, "on_compaction", None)
        if callable(hook):
            hook(info)

    def _render_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        """Render a post-execution notice for a single tool result (Phase 2).

        Forwards ``(name, denied, forbidden, diff)`` to the renderer's
        :meth:`Renderer.on_tool_result` hook when present. ``denied`` and
        ``forbidden`` come from the result's ``meta`` (set by the executor
        when the policy refuses the call); ``diff`` is the tool-provided
        ``meta["diff"]`` of a successful write/edit, or ``None`` when the
        tool did not supply one. The hook is optional: renderers that do
        not implement it are silently skipped.
        """

        hook = getattr(self.renderer, "on_tool_result", None)
        if not callable(hook):
            return
        meta = result.meta or {}
        denied = bool(meta.get("denied", False))
        forbidden = bool(meta.get("forbidden", False))
        diff = meta.get("diff") if isinstance(meta.get("diff"), str) else None
        summary = summarize_result(call.name, result)
        try:
            hook(
                call.name,
                denied=denied,
                forbidden=forbidden,
                diff=diff,
                ok=bool(result.ok),
                summary=summary,
            )
        except TypeError:
            # Renderer implements the older on_tool_result signature (without
            # ok/summary); fall back so it still receives denial/diff info.
            hook(call.name, denied=denied, forbidden=forbidden, diff=diff)


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
