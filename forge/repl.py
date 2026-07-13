"""The interactive terminal REPL.

This module implements :class:`Repl`, the read-eval-print loop that owns the
user-facing prompt and drives the :class:`~forge.agent.AgentLoop`. A single
iteration reads one line of input, classifies it, and either terminates the
loop, re-displays the prompt, or runs one agent turn and renders its output.

Responsibilities (Requirements 1.1, 1.6, 1.7, 3.1-3.4, 10.3, 14.7, 17.3, 17.5)
------------------------------------------------------------------------------
* **Read input (Req 1.1).** Display an input prompt and read a line. Reading is
  injected through ``input_func`` so tests never need a real TTY; the default
  reader is a :class:`prompt_toolkit.PromptSession` constructed lazily so the
  module imports cleanly even where ``prompt_toolkit`` is unavailable.
* **Exit commands (Req 1.6).** Input that is exactly ``/exit`` or ``/quit``
  terminates the REPL without invoking the :class:`AgentLoop`. The pure helper
  :func:`is_exit_command` performs that exact-match classification (Property 1).
* **Blank input (Req 1.7).** Empty or whitespace-only input re-displays the
  prompt without invoking the :class:`AgentLoop`. The pure helper
  :func:`is_blank` performs that classification (Property 2).
* **Run a turn.** Otherwise the line is sent to :meth:`AgentLoop.run_turn`.
* **Rendering.** :class:`Repl` is the :class:`~forge.agent.Renderer` the
  :class:`AgentLoop` calls during a turn: :meth:`on_text` writes streamed text
  immediately (Req 3.1), :meth:`on_tool` announces a tool before it runs
  (Req 3.2), and :meth:`on_compaction` renders the compaction notice (Req 14.7).
* **After the turn.** Print an end-of-response indicator (Req 3.3); if the turn
  errored or was interrupted, print an error/interruption indicator while
  retaining the partial tokens already written (Req 3.4); render the todo list
  when it changed (Req 10.3); and print the usage summary, showing "cost
  unavailable" when pricing is absent (Req 17.3, 17.5).

Todo rendering on change (Req 10.3)
-----------------------------------
The :class:`AgentLoop`'s ``Renderer`` contract is ``on_text``/``on_tool``/
``on_compaction`` and the loop invokes each hook defensively (``hasattr``
guarded). :class:`Repl` additionally exposes :meth:`on_todos` so the agent/tool
layer *may* push a todo update mid-turn if it chooses, and — because the current
:class:`AgentLoop` does not surface todos mid-turn — :class:`Repl` also
re-renders the list after each turn from ``session.todos`` whenever it differs
from the previously rendered snapshot. Both paths funnel through the same
change-detection (:meth:`_render_todos_if_changed`) so the list is rendered
exactly once per change regardless of which path observed it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, TextIO

from forge.ui import Ui

from forge.agent import AgentLoop, TurnResult
from forge.context import CompactionInfo
from forge.policy import Decision
from forge.session import Session, TodoItem
from forge.usage import UsageSummary

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    # Imported under TYPE_CHECKING to document the duck-typed collaborators
    # without creating an import-time dependency. ``forge.verification`` imports
    # from ``forge.agent`` (as this module does) but not from ``forge.repl``, so
    # no real cycle exists; guarding the import keeps the coupling annotation-only
    # and avoids any ordering fragility during bootstrap.
    from forge.checkpoint import CheckpointStore
    from forge.verification import (
        VerificationCoordinator,
        VerificationRenderer,
        VerificationResult,
    )

__all__ = ["Repl", "is_exit_command", "is_blank", "PROMPT"]

#: The reserved keywords that terminate the REPL (the Exit_Command set).
EXIT_COMMANDS: frozenset[str] = frozenset({"/exit", "/quit"})

#: The reserved keyword that triggers a checkpoint undo (Phase 2, Feature C).
UNDO_COMMAND: str = "/undo"

#: The prompt string shown while waiting for user input (Req 1.1).
PROMPT: str = "forge> "

#: Glyphs used when rendering a todo item's status (Req 10.3).
_TODO_STATUS_GLYPHS: dict[str, str] = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "completed": "[x]",
}

#: Approver prompt responses (Phase 2, Feature B).
_APPROVE_RESPONSES: frozenset[str] = frozenset({"y", "yes"})
_DENY_RESPONSES: frozenset[str] = frozenset({"n", "no"})
_ALWAYS_RESPONSES: frozenset[str] = frozenset({"a", "always"})


# --------------------------------------------------------------------------- #
# Pure classification helpers (module-level so property tests can import them)
# --------------------------------------------------------------------------- #


def is_exit_command(text: str) -> bool:
    """Return ``True`` iff ``text`` is exactly an Exit_Command keyword.

    The match is exact (Property 1, Req 1.6): only the literal strings
    ``"/exit"`` and ``"/quit"`` classify as exit commands. No surrounding
    whitespace is tolerated, so ``"/exit "`` and ``" /quit"`` are *not* exit
    commands. Callers are expected to strip only the line terminator (``\\r``/
    ``\\n``) from a read line before classification, never interior or trailing
    spaces.
    """

    return text in EXIT_COMMANDS


def is_blank(text: str) -> bool:
    """Return ``True`` iff ``text`` is empty or only whitespace (Property 2).

    Such input is ignored by the REPL: the prompt is re-displayed without
    invoking the :class:`AgentLoop` (Req 1.7).
    """

    return text.strip() == ""


def _approval_target(name: str, args: dict) -> str:
    """Return the per-call target key used by the APPROVE_ALWAYS memory.

    The key normalizes a tool call down to the smallest identifier that
    distinguishes two meaningful variants of the same tool: ``shell`` uses
    its argv[0] (so ``pytest`` and ``pytest -q`` share an ``always`` answer
    when the user said ``a``); ``git`` uses the ``operation``; ``write`` /
    ``edit`` use the ``path``; everything else uses the whole arg dump.
    """

    if not isinstance(args, dict):
        return ""
    if name == "shell":
        cmd = args.get("command")
        if isinstance(cmd, str):
            stripped = cmd.lstrip()
            if stripped:
                # Use the first whitespace-delimited token of the command.
                return stripped.split(None, 1)[0]
        return ""
    if name == "git":
        op = args.get("operation")
        if isinstance(op, str):
            return op
        return ""
    if name in ("write", "edit"):
        path = args.get("path")
        if isinstance(path, str):
            return path
        return ""
    return repr(args)


def _summarize_args(args: dict) -> str:
    """Render ``args`` as a short, human-readable summary line for prompts."""

    if not isinstance(args, dict) or not args:
        return "(no arguments)"
    parts: list[str] = []
    for key, value in args.items():
        rendered = value if isinstance(value, str) else repr(value)
        if len(rendered) > 80:
            rendered = rendered[:77] + "..."
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Input reader
# --------------------------------------------------------------------------- #


class _InputFunc(Protocol):
    """Callable that displays ``prompt`` and returns one line of input."""

    def __call__(self, prompt: str) -> str:  # pragma: no cover - structural
        ...


def _make_prompt_toolkit_reader() -> _InputFunc:
    """Build the default line reader backed by ``prompt_toolkit``.

    Imported lazily so this module imports cleanly even where
    ``prompt_toolkit`` is not installed; the import only happens when a real
    interactive reader is actually needed (i.e. no ``input_func`` was injected).
    """

    from prompt_toolkit import PromptSession

    session: "PromptSession[str]" = PromptSession()

    def _read(prompt: str) -> str:
        return session.prompt(prompt)

    return _read


# --------------------------------------------------------------------------- #
# Repl
# --------------------------------------------------------------------------- #


class Repl:
    """The interactive prompt loop and the :class:`AgentLoop`'s renderer.

    Parameters
    ----------
    agent_loop:
        The :class:`~forge.agent.AgentLoop` driven by each non-exit, non-blank
        line. ``Repl`` installs itself as the loop's ``renderer`` (unless one is
        already wired) so streamed text, tool announcements, and the compaction
        notice flow back to this terminal.
    session:
        The current :class:`~forge.session.Session` passed to
        :meth:`AgentLoop.run_turn`.
    input_func:
        Optional callable ``(prompt: str) -> str`` that reads one line. Injected
        for testability; defaults to a lazily-constructed ``prompt_toolkit``
        reader so tests never need a real TTY.
    out:
        Optional output stream (anything with ``write``/``flush``); defaults to
        :data:`sys.stdout`. Writes are flushed immediately so streaming stays
        visible within the 200 ms-per-token budget (Req 3.1).
    prompt:
        The prompt string to display (defaults to :data:`PROMPT`).
    verification_coordinator:
        Optional post-turn ``VerificationCoordinator`` (duck-typed; the concrete
        type is :class:`forge.verification.VerificationCoordinator`). When wired,
        :meth:`run_once` invokes it after a turn that was neither interrupted nor
        errored, and renders the phase's aggregated usage. When ``None`` — or when
        the coordinator reports the phase did not run — rendering and persistence
        are exactly as they are without verification (Req 2.3).

    Verification rendering (Req 9)
    ------------------------------
    ``Repl`` also structurally satisfies the optional
    :class:`forge.verification.VerificationRenderer` protocol so the coordinator
    can surface phase progress to this terminal. Its four hooks
    (:meth:`on_verification_start`, :meth:`on_verification_result`,
    :meth:`on_correction_iteration`, :meth:`on_verification_cap_reached`) each
    emit a single ``[verify] ...`` line via :meth:`_writeln`.
    """

    def __init__(
        self,
        agent_loop: AgentLoop,
        session: Session,
        input_func: Callable[[str], str] | None = None,
        out: TextIO | None = None,
        prompt: str = PROMPT,
        verification_coordinator: "VerificationCoordinator | None" = None,
        checkpoint: "CheckpointStore | None" = None,
        show_diffs: bool = False,
        ui: Ui | None = None,
        commands_store: Any | None = None,
        mentions_enabled: bool = False,
        read_max_bytes: int = 1_000_000,
        workspace_root: Path | None = None,
    ) -> None:
        self.agent_loop = agent_loop
        self.session = session
        self._input_func = input_func
        self.out: TextIO = out if out is not None else sys.stdout
        self.prompt = prompt
        self.verification_coordinator = verification_coordinator
        self.checkpoint = checkpoint
        self.show_diffs = bool(show_diffs)
        self.ui = ui or Ui(self.out, color=False, spinner=False)
        self.commands_store = commands_store
        self.mentions_enabled = bool(mentions_enabled)
        self.read_max_bytes = int(read_max_bytes)
        self.workspace_root = workspace_root if workspace_root is not None else Path.cwd()
        # Session-scoped set of tool+target keys the user has answered
        # ``a`` (always-approve) for in this REPL session. APPROVE_ALWAYS
        # bookkeeping lives here, not in the executor, per the design.
        self._always_approve: set[tuple[str, str]] = set()

        # Install this Repl as the loop's renderer so its on_text/on_tool/
        # on_compaction hooks are driven during a turn. Respect an explicitly
        # pre-wired renderer if one is already present.
        from forge.agent import NullRenderer

        if getattr(agent_loop, "renderer", None) is None or isinstance(
            getattr(agent_loop, "renderer", None), NullRenderer
        ):
            agent_loop.renderer = self

        # Snapshot of the last todo list we rendered, for change detection
        # (Req 10.3). Initialized to the session's current todos so an unchanged
        # list is never re-rendered on the first turn.
        self._last_rendered_todos: list[dict] = self._snapshot_todos(
            session.todos
        )

    # -- input reading -------------------------------------------------------

    def _read_input(self) -> str:
        """Display the prompt and read one raw line of input (Req 1.1).

        Uses the injected ``input_func`` when provided, otherwise the default
        ``prompt_toolkit`` reader (constructed lazily on first use).
        """

        if self._input_func is None:
            self._input_func = _make_prompt_toolkit_reader()
        return self._input_func(self.prompt)

    # -- loop ----------------------------------------------------------------

    def run(self) -> None:
        """Run the full prompt loop until an Exit_Command or EOF.

        Repeatedly calls :meth:`run_once`; stops when it returns ``False`` (an
        exit command was entered) or when input reading raises ``EOFError`` /
        ``KeyboardInterrupt`` at the idle prompt (e.g. Ctrl-D / Ctrl-C), which
        also returns control to the shell.
        """

        while True:
            try:
                keep_going = self.run_once()
            except (EOFError, KeyboardInterrupt):
                # EOF (Ctrl-D) or an idle Ctrl-C at the prompt terminates the
                # REPL gracefully, returning control to the shell (Req 1.6).
                self._writeln("")
                return
            if not keep_going:
                return

    def run_once(self) -> bool:
        """Run a single REPL iteration.

        Reads one line, then:

        * returns ``False`` (terminate) when the line is an Exit_Command,
          *without* invoking the :class:`AgentLoop` (Req 1.6);
        * returns ``True`` (continue) without invoking the loop when the line is
          blank or whitespace-only (Req 1.7); or
        * otherwise runs one agent turn via :meth:`AgentLoop.run_turn`, renders
          its result, and returns ``True``.
        """

        raw = self._read_input()
        # Strip only the line terminator, never interior/trailing spaces, so the
        # exact-match exit classification (Property 1) is preserved.
        line = raw.rstrip("\r\n")

        if is_exit_command(line):
            return False
        if line == UNDO_COMMAND:
            self._handle_undo()
            return True

        if line == "/help" or line == "/commands":
            builtins = ["/exit", "/quit", "/undo", "/help", "/commands"]
            customs = []
            if self.commands_store is not None:
                customs = self.commands_store.names()
            self._writeln("Built-in commands:")
            self._writeln("  " + ", ".join(builtins))
            if customs:
                self._writeln("Custom commands:")
                self._writeln("  " + ", ".join(f"/{c}" for c in customs))
            return True

        if line.startswith("/"):
            parts = line.split(None, 1)
            cmd_name = parts[0][1:]
            arg_text = parts[1] if len(parts) > 1 else ""

            is_custom = False
            if self.commands_store is not None:
                is_custom = cmd_name in self.commands_store.names()

            if is_custom:
                rendered = self.commands_store.render(cmd_name, arg_text)
                if rendered is not None:
                    line = rendered
                else:
                    self._writeln(f"Error rendering command '/{cmd_name}'")
                    return True
            else:
                self._writeln(f"Unknown command: {parts[0]}")
                return True

        if is_blank(line):
            return True

        if self.mentions_enabled:
            from forge.commands import expand_mentions
            line, included, warnings = expand_mentions(
                line, self.workspace_root, max_bytes=self.read_max_bytes
            )
            if included:
                self._writeln(f"[included: {', '.join(included)}]")
            for warning in warnings:
                self._writeln(f"[warning] {warning}")

        result = self.agent_loop.run_turn(self.session, line)

        # After a turn that completed normally (not interrupted, no error), run
        # the post-turn Verification_Phase when a coordinator is wired. When the
        # phase ran, its aggregated usage (turn + all Correction_Iterations)
        # replaces the bare turn usage in the summary; otherwise rendering is
        # exactly as today (Req 2.3, 9, 10).
        usage_override: UsageSummary | None = None
        turn_ok = not (result.interrupted or result.error)
        if self.verification_coordinator is not None and turn_ok:
            phase = self.verification_coordinator.run(self.session, result)
            if phase.ran:
                usage_override = phase.usage

        self._render_turn_result(result, usage_override=usage_override)
        return True

    # -- Renderer hooks (driven by the AgentLoop during a turn) --------------

    def on_text(self, text: str) -> None:
        """Render a streamed fragment of model text immediately (Req 3.1)."""

        self._write(text)

    def on_tool(self, name: str) -> None:
        """Announce the tool about to run, before it executes (Req 3.2)."""

        announcement = self.ui.tool_announcement(name)
        self._writeln(announcement)

    def on_tool_result(self, name: str, denied: bool, forbidden: bool,
                       diff: str | None) -> None:
        """Render a post-execution notice for a single tool result (Phase 2).

        Called by the :class:`~forge.agent.AgentLoop` after each tool result is
        appended to the session. Used to surface denial / forbiddance (the
        approval policy refused the call) and — when ``show_diffs`` is enabled
        — the unified diff for a successful write/edit.
        """

        if denied:
            self._writeln(f"[denied] {name}")
        elif forbidden:
            self._writeln(f"[forbidden] {name}")
        if diff and self.show_diffs:
            self.ui.render_diff(diff)

    def status(self, message: str):
        """Return a status context manager via the UI helper."""
        return self.ui.status(message)

    def on_compaction(self, info: CompactionInfo) -> None:
        """Render the "conversation context was compacted" notice (Req 14.7)."""

        self._writeln("\n[notice] conversation context was compacted")

    def on_todos(self, todos: list[TodoItem]) -> None:
        """Render the todo list if it changed (Req 10.3).

        Exposed so the agent/tool layer may push a mid-turn todo update; routed
        through the same change-detection used after each turn so the list is
        rendered at most once per change.
        """

        self._render_todos_if_changed(todos)

    # -- Approver / undo (Phase 2) ------------------------------------------

    def request(self, name: str, args: dict, preview: str | None) -> Decision:
        """Prompt the user to approve a gated tool call (Phase 2, Feature B).

        Prints a summary line, the preview (when present), then a short prompt
        reading a single line of input. Recognized responses:

        * ``y`` / ``yes`` -> :attr:`Decision.APPROVE`
        * ``a`` / ``always`` -> :attr:`Decision.APPROVE_ALWAYS` (recorded for
          the session so the same action is not re-prompted)
        * anything else (including ``n`` / ``no`` and blank) -> :attr:`Decision.DENY`
        """

        target = _approval_target(name, args)
        if (name, target) in self._always_approve:
            return Decision.APPROVE_ALWAYS

        summary = _summarize_args(args)
        self._writeln(f"[approve] {name} wants to run: {summary}")
        if preview:
            for line in preview.rstrip("\n").splitlines():
                self._writeln(f"    {line}")
        self._write("[y/n/a] ")

        response = self._read_input().strip().lower()
        if response in _ALWAYS_RESPONSES:
            self._always_approve.add((name, target))
            return Decision.APPROVE_ALWAYS
        if response in _APPROVE_RESPONSES:
            return Decision.APPROVE
        return Decision.DENY

    def _handle_undo(self) -> None:
        """Undo the most recent committed turn (Phase 2, Feature C).

        Prints ``[undo] restored N file(s): …`` on success, or
        ``[undo] nothing to undo`` when there is no committed checkpoint group.
        A checkpoint store must be wired for the command to do anything.
        """

        if self.checkpoint is None:
            self._writeln("[undo] nothing to undo")
            return
        restored = self.checkpoint.undo_last()
        if not restored:
            self._writeln("[undo] nothing to undo")
            return
        joined = ", ".join(restored)
        self._writeln(f"[undo] restored {len(restored)} file(s): {joined}")

    # -- VerificationRenderer hooks (driven by the coordinator) --------------

    def on_verification_start(self, command: str) -> None:
        """Announce the Verify_Command about to run (Req 9.1).

        Renders ``[verify] running: <command>`` before the command executes.
        """

        self._writeln(f"[verify] running: {command}")

    def on_verification_result(self, result: "VerificationResult") -> None:
        """Render the classified Verify_Command outcome (Req 9.2, 9.3).

        A passing run renders ``[verify] passed`` (Req 9.2); any non-passing
        outcome (``failed`` / ``timed_out`` / ``start_error``) renders
        ``[verify] failed (<status>)`` carrying the outcome status (Req 9.3).
        """

        if result.outcome == "passed":
            self._writeln("[verify] passed")
        else:
            self._writeln(f"[verify] failed ({result.outcome})")

    def on_correction_iteration(self, iteration: int, max_iterations: int) -> None:
        """Announce a starting Correction_Iteration ``iteration``/``max`` (Req 9.4)."""

        self._writeln(
            f"[verify] correction iteration {iteration}/{max_iterations}"
        )

    def on_verification_cap_reached(
        self, result: "VerificationResult", iterations: int
    ) -> None:
        """Render the cap-reached notice without a passing result (Req 9.5).

        The design calls for "clearing the running indicator" before the notice;
        this terminal renderer streams output line-by-line and has no persistent
        spinner to clear, so there is nothing to erase — the cap-reached line is
        simply printed: ``[verify] iteration cap reached (<iterations>); final
        status: <status>``.
        """

        self._writeln(
            f"[verify] iteration cap reached ({iterations}); "
            f"final status: {result.outcome}"
        )

    # -- post-turn rendering -------------------------------------------------

    def _render_turn_result(
        self, result: TurnResult, usage_override: UsageSummary | None = None
    ) -> None:
        """Render everything that follows a completed :meth:`run_turn`.

        Order: end-of-response indicator (Req 3.3); an error/interruption
        indicator when applicable, leaving any partial tokens already written
        intact (Req 3.4); the todo list when it changed (Req 10.3); and the
        usage summary (Req 17.3, 17.5).

        When ``usage_override`` is provided (the Verification_Phase ran), it
        replaces ``result.usage`` in the summary so the aggregated phase usage —
        the original turn plus every Correction_Iteration — is reported instead
        of the bare turn usage (Req 10). When it is ``None``, the turn's own
        usage is rendered exactly as without verification (Req 2.3).
        """

        # End-of-response indicator (Req 3.3). A leading newline ensures it sits
        # on its own line after any streamed text that did not end with one.
        self._writeln("\n[end of response]")

        # Error / interruption indicator (Req 3.4). The partial response already
        # streamed to the terminal is deliberately NOT cleared.
        if result.interrupted:
            self._writeln("[interrupted] turn was interrupted; partial output retained")
        elif result.error:
            self._writeln(f"[error] {result.error}; partial output retained")

        # Todo list, only when it changed since the last render (Req 10.3).
        self._render_todos_if_changed(self.session.todos)

        # Usage summary (Req 17.3, 17.5), using the aggregated phase usage when
        # the Verification_Phase ran.
        usage = usage_override if usage_override is not None else result.usage
        self._render_usage(usage)

    def _render_usage(self, u: UsageSummary) -> None:
        """Print turn and cumulative token counts and estimated cost.

        When ``cost_available`` is ``False`` the cost is shown as
        "cost unavailable" (Req 17.5); otherwise the estimated turn and
        cumulative costs are shown (Req 17.3, 17.4).
        """

        turn_tokens = (
            f"turn: {u.turn_input_tokens} in / {u.turn_output_tokens} out"
        )
        cumulative_tokens = (
            f"session: {u.cumulative_input_tokens} in / "
            f"{u.cumulative_output_tokens} out"
        )

        if u.cost_available:
            cost = (
                f"cost: ${u.turn_cost:.6f} turn / "
                f"${u.cumulative_cost:.6f} session"
            )
        else:
            cost = "cost unavailable"

        self._writeln(f"[usage] {turn_tokens} | {cumulative_tokens} | {cost}")

    # -- todo rendering / change detection -----------------------------------

    def _render_todos_if_changed(self, todos: list[TodoItem]) -> None:
        """Render ``todos`` only when they differ from the last render (Req 10.3)."""

        snapshot = self._snapshot_todos(todos)
        if snapshot == self._last_rendered_todos:
            return
        self._last_rendered_todos = snapshot
        self._render_todos(todos)

    def _render_todos(self, todos: list[TodoItem]) -> None:
        """Render each todo item with its status marker (Req 10.3)."""

        if not todos:
            # An emptied list is itself a change worth reflecting.
            self._writeln("[todos] (cleared)")
            return

        self._writeln("[todos]")
        for item in todos:
            glyph = _TODO_STATUS_GLYPHS.get(item.status, "[?]")
            self._writeln(f"  {glyph} {item.text}")

    @staticmethod
    def _snapshot_todos(todos: list[TodoItem]) -> list[dict]:
        """Return a comparable snapshot of a todo list for change detection."""

        return [
            {"id": t.id, "text": t.text, "status": t.status} for t in todos
        ]

    # -- low-level output ----------------------------------------------------

    def _write(self, text: str) -> None:
        """Write ``text`` and flush so streaming stays visible (Req 3.1)."""

        self.out.write(text)
        self.out.flush()

    def _writeln(self, text: str) -> None:
        """Write ``text`` followed by a newline and flush."""

        self.out.write(text + "\n")
        self.out.flush()
