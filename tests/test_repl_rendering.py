"""Unit tests for the REPL's rendering flow.

These tests drive :class:`forge.repl.Repl` end-to-end without a real TTY or
model by:

* injecting an ``input_func`` that returns scripted lines (so no
  ``prompt_toolkit`` session is ever constructed), and
* injecting a :class:`io.StringIO` ``out`` stream so all rendered output is
  captured as a string for assertion.

A :class:`FakeAgentLoop` stands in for the real :class:`~forge.agent.AgentLoop`:
its ``run_turn`` drives the Repl's renderer hooks (``on_text``/``on_tool``/
``on_compaction``/``on_todos``) exactly as the real loop would and returns a
crafted :class:`~forge.agent.TurnResult`. The Repl installs itself as the fake
loop's ``renderer`` during construction, so the hooks the fake calls flow back
into the captured output stream — the same public contract the real loop uses.

Covers Requirements 1.1, 1.3, 3.1, 3.2, 3.3, 3.4, 10.3, 14.7.
"""

from __future__ import annotations

import io
from typing import Callable

from forge.agent import TurnResult
from forge.context import CompactionInfo
from forge.repl import PROMPT, Repl
from forge.session import Session, TodoItem, Usage
from forge.usage import UsageSummary


# --------------------------------------------------------------------------- #
# Test doubles / helpers
# --------------------------------------------------------------------------- #


def make_usage(*, cost_available: bool = True) -> UsageSummary:
    """Build a representative :class:`UsageSummary` for a turn result."""

    return UsageSummary(
        turn_input_tokens=10,
        turn_output_tokens=20,
        cumulative_input_tokens=30,
        cumulative_output_tokens=40,
        turn_cost=0.0012 if cost_available else None,
        cumulative_cost=0.0034 if cost_available else None,
        cost_available=cost_available,
    )


def make_turn_result(
    *,
    compaction: CompactionInfo | None = None,
    error: str | None = None,
    interrupted: bool = False,
) -> TurnResult:
    """Build a :class:`TurnResult` with a representative usage summary."""

    return TurnResult(
        usage=make_usage(),
        compaction=compaction,
        error=error,
        interrupted=interrupted,
    )


class FakeAgentLoop:
    """A stand-in :class:`~forge.agent.AgentLoop` driven by a scripted callback.

    The Repl installs itself as ``renderer`` during construction (it sees the
    ``renderer is None`` here and wires itself in). Each ``run_turn`` call
    invokes ``on_run(renderer, session, user_text)``; that callback uses the
    renderer hooks to emit streamed text / tool announcements / compaction
    notices / todo updates just like the real loop, then returns the
    :class:`TurnResult` to render.
    """

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
    """An ``input_func`` returning queued lines and recording prompts shown.

    Raises ``EOFError`` once the queue is exhausted, which the Repl treats as a
    graceful end-of-input (Ctrl-D) so :meth:`Repl.run` returns.
    """

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


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
    todos: list[TodoItem] | None = None,
) -> tuple[Repl, io.StringIO, ScriptedInput, FakeAgentLoop, Session]:
    """Wire a Repl with a fake loop, scripted input, and a captured stream."""

    loop = FakeAgentLoop(on_run)
    session = make_session(todos)
    out = io.StringIO()
    reader = ScriptedInput(lines)
    repl = Repl(loop, session, input_func=reader, out=out)
    return repl, out, reader, loop, session


# --------------------------------------------------------------------------- #
# Req 1.1 — prompt display on launch
# --------------------------------------------------------------------------- #


def test_prompt_displayed_on_launch() -> None:
    """The REPL displays the input prompt while waiting for input (Req 1.1)."""

    def on_run(renderer: Repl, session: Session, text: str) -> TurnResult:
        raise AssertionError("exit command must not invoke the agent loop")

    # A single "/exit" line terminates immediately without a turn.
    repl, _out, reader, loop, _session = make_repl(on_run, ["/exit"])
    repl.run()

    assert reader.prompts == [PROMPT]
    assert loop.calls == []  # Exit command never reaches the loop (Req 1.6).


# --------------------------------------------------------------------------- #
# Req 1.3 / 3.1 / 3.3 — no-tool-call response display, streaming, end indicator
# --------------------------------------------------------------------------- #


def test_no_tool_call_response_is_streamed_then_ended() -> None:
    """A plain response is streamed and followed by the end indicator.

    Covers the no-tool-call display path (Req 1.3), immediate per-fragment
    streaming (Req 3.1), and the end-of-response indicator (Req 3.3).
    """

    def on_run(renderer: Repl, session: Session, text: str) -> TurnResult:
        renderer.on_text("Hello, ")
        renderer.on_text("world.")
        return make_turn_result()

    repl, out, _reader, loop, _session = make_repl(on_run, ["hi"])
    keep_going = repl.run_once()

    rendered = out.getvalue()
    assert loop.calls == ["hi"]  # Non-blank input reached the loop (Req 1.2).
    assert keep_going is True  # Control returns to the prompt (Req 1.3).
    # Streamed fragments appear in order and contiguously (Req 3.1).
    assert "Hello, world." in rendered
    # End-of-response indicator follows the streamed text (Req 3.3).
    assert "[end of response]" in rendered
    assert rendered.index("Hello, world.") < rendered.index("[end of response]")
    # The usage summary is printed after the turn (Req 17.3).
    assert "[usage]" in rendered


def test_streaming_fragments_render_in_received_order() -> None:
    """Streamed text fragments are written in the order received (Req 3.1)."""

    fragments = ["The ", "quick ", "brown ", "fox"]

    def on_run(renderer: Repl, session: Session, text: str) -> TurnResult:
        for frag in fragments:
            renderer.on_text(frag)
        return make_turn_result()

    repl, out, _reader, _loop, _session = make_repl(on_run, ["go"])
    repl.run_once()

    rendered = out.getvalue()
    assert "The quick brown fox" in rendered
    # Strict ordering of each fragment within the output.
    positions = [rendered.index(frag) for frag in fragments]
    assert positions == sorted(positions)


# --------------------------------------------------------------------------- #
# Req 3.2 — tool-name announcement before execution
# --------------------------------------------------------------------------- #


def test_tool_name_announced_during_turn() -> None:
    """The Repl announces a tool name when the loop calls ``on_tool`` (Req 3.2)."""

    def on_run(renderer: Repl, session: Session, text: str) -> TurnResult:
        renderer.on_text("running a tool ")
        renderer.on_tool("read")
        renderer.on_text("done")
        return make_turn_result()

    repl, out, _reader, _loop, _session = make_repl(on_run, ["use a tool"])
    repl.run_once()

    rendered = out.getvalue()
    assert "[tool: read]" in rendered


# --------------------------------------------------------------------------- #
# Req 3.4 — error / interruption indicator with partial-token retention
# --------------------------------------------------------------------------- #


def test_stream_error_indicator_retains_partial_tokens() -> None:
    """An errored turn shows the error indicator and keeps partial output.

    The partial text already streamed to the terminal is retained, and an
    error indicator describing the failure is shown (Req 3.4).
    """

    def on_run(renderer: Repl, session: Session, text: str) -> TurnResult:
        renderer.on_text("partial answer before the failure")
        return make_turn_result(error="rate limit exceeded")

    repl, out, _reader, _loop, _session = make_repl(on_run, ["do it"])
    keep_going = repl.run_once()

    rendered = out.getvalue()
    # Partial tokens are NOT cleared (Req 3.4).
    assert "partial answer before the failure" in rendered
    # Error indicator describes the failure and notes retention (Req 3.4).
    assert "[error] rate limit exceeded" in rendered
    assert "partial output retained" in rendered
    # Control still returns to the prompt afterward.
    assert keep_going is True


def test_interrupt_indicator_retains_partial_tokens() -> None:
    """An interrupted turn shows the interruption indicator and keeps output.

    Validates the interruption branch of the stream error indicator (Req 3.4).
    """

    def on_run(renderer: Repl, session: Session, text: str) -> TurnResult:
        renderer.on_text("half a thought")
        return make_turn_result(interrupted=True)

    repl, out, _reader, _loop, _session = make_repl(on_run, ["go"])
    repl.run_once()

    rendered = out.getvalue()
    assert "half a thought" in rendered
    assert "[interrupted]" in rendered
    assert "partial output retained" in rendered


# --------------------------------------------------------------------------- #
# Req 10.3 — todo-list rendering when the list changes
# --------------------------------------------------------------------------- #


def test_todos_rendered_when_list_changes_after_turn() -> None:
    """A todo list changed during a turn is rendered afterward (Req 10.3)."""

    def on_run(renderer: Repl, session: Session, text: str) -> TurnResult:
        renderer.on_text("planning")
        session.todos.append(TodoItem(id="1", text="write tests", status="pending"))
        session.todos.append(
            TodoItem(id="2", text="run suite", status="in_progress")
        )
        return make_turn_result()

    repl, out, _reader, _loop, _session = make_repl(on_run, ["plan"])
    repl.run_once()

    rendered = out.getvalue()
    assert "[todos]" in rendered
    assert "[ ] write tests" in rendered
    assert "[~] run suite" in rendered


def test_todos_not_rerendered_when_unchanged() -> None:
    """An unchanged todo list is not re-rendered after a turn (Req 10.3).

    The Repl snapshots the session's todos at construction, so a turn that
    leaves them unchanged must not emit a todo block.
    """

    existing = [TodoItem(id="1", text="already here", status="completed")]

    def on_run(renderer: Repl, session: Session, text: str) -> TurnResult:
        renderer.on_text("no plan changes")
        return make_turn_result()

    repl, out, _reader, _loop, _session = make_repl(
        on_run, ["chat"], todos=existing
    )
    repl.run_once()

    rendered = out.getvalue()
    assert "[todos]" not in rendered


def test_mid_turn_todo_push_renders_once() -> None:
    """A mid-turn ``on_todos`` push renders the changed list (Req 10.3).

    The agent/tool layer may push a todo update mid-turn via ``on_todos``; it
    routes through the same change-detection used after the turn, so the list
    renders exactly once even though the post-turn pass sees the same list.
    """

    def on_run(renderer: Repl, session: Session, text: str) -> TurnResult:
        new_todos = [TodoItem(id="1", text="mid turn item", status="pending")]
        session.todos.extend(new_todos)
        renderer.on_todos(session.todos)  # mid-turn push
        return make_turn_result()

    repl, out, _reader, _loop, _session = make_repl(on_run, ["plan"])
    repl.run_once()

    rendered = out.getvalue()
    assert rendered.count("[ ] mid turn item") == 1


# --------------------------------------------------------------------------- #
# Req 14.7 — compaction notice rendering
# --------------------------------------------------------------------------- #


def test_compaction_notice_rendered_when_loop_signals() -> None:
    """The Repl renders the compaction notice when the loop calls ``on_compaction``.

    Validates the compaction-notice rendering hook (Req 14.7).
    """

    def on_run(renderer: Repl, session: Session, text: str) -> TurnResult:
        info = CompactionInfo(
            occurred=True, summary_message_count=1, dropped_message_count=2
        )
        renderer.on_compaction(info)
        renderer.on_text("after compaction")
        return make_turn_result(compaction=info)

    repl, out, _reader, _loop, _session = make_repl(on_run, ["long chat"])
    repl.run_once()

    rendered = out.getvalue()
    assert "compacted" in rendered
    assert "after compaction" in rendered


# --------------------------------------------------------------------------- #
# Req 1.7 — blank input is ignored by the loop (control-flow level)
# --------------------------------------------------------------------------- #


def test_blank_input_does_not_invoke_loop() -> None:
    """Blank input re-displays the prompt without invoking the loop (Req 1.7)."""

    def on_run(renderer: Repl, session: Session, text: str) -> TurnResult:
        raise AssertionError("blank input must not invoke the agent loop")

    repl, _out, _reader, loop, _session = make_repl(on_run, ["   "])
    keep_going = repl.run_once()

    assert keep_going is True
    assert loop.calls == []
