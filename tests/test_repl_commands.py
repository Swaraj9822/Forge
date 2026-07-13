"""Tests for REPL command classification order and mentions integration."""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge.agent import TurnResult
from forge.session import Session
from forge.repl import Repl
from forge.usage import UsageSummary


class SpyAgentLoop:

    def __init__(self) -> None:
        self.called_with: list[str] = []

    def run_turn(self, session: Session, line: str) -> TurnResult:
        self.called_with.append(line)
        return TurnResult(
            usage=UsageSummary(
                turn_input_tokens=10,
                turn_output_tokens=20,
                cumulative_input_tokens=10,
                cumulative_output_tokens=20,
                cost_available=True,
                turn_cost=0.01,
                cumulative_cost=0.01,
            ),
        )


def test_repl_classification_order() -> None:
    session = Session(
        id="test-session",
        created_at="2026-07-13T12:00:00Z",
        updated_at="2026-07-13T12:00:00Z",
    )

    # 1. Exit command
    loop = SpyAgentLoop()
    repl = Repl(agent_loop=loop, session=session, input_func=lambda prompt: "/exit")
    assert repl.run_once() is False
    assert not loop.called_with

    # 2. Blank command
    loop = SpyAgentLoop()
    repl = Repl(agent_loop=loop, session=session, input_func=lambda prompt: "   ")
    assert repl.run_once() is True
    assert not loop.called_with

    # 3. Help command
    loop = SpyAgentLoop()
    repl = Repl(agent_loop=loop, session=session, input_func=lambda prompt: "/help")
    assert repl.run_once() is True
    assert not loop.called_with

    # 4. Unknown command
    loop = SpyAgentLoop()
    repl = Repl(
        agent_loop=loop, session=session, input_func=lambda prompt: "/unknown"
    )
    assert repl.run_once() is True
    assert not loop.called_with


def test_repl_mentions_and_custom_commands() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()

        # Create a workspace file
        file_a = root / "a.txt"
        file_a.write_text("Hello A", encoding="utf-8")

        # Create a custom slash command
        cmd_dir = root / "commands"
        cmd_dir.mkdir()
        (cmd_dir / "greet.md").write_text(
            "Greet $ARGUMENTS with @a.txt", encoding="utf-8"
        )

        from forge.commands import SlashCommandStore

        store = SlashCommandStore([cmd_dir])

        session = Session(
            id="test-session",
            created_at="2026-07-13T12:00:00Z",
            updated_at="2026-07-13T12:00:00Z",
        )

        # Run a custom command turn
        loop = SpyAgentLoop()
        repl = Repl(
            agent_loop=loop,
            session=session,
            input_func=lambda prompt: "/greet World",
            commands_store=store,
            mentions_enabled=True,
            workspace_root=root,
        )

        assert repl.run_once() is True

        # Let's see what the spy was called with.
        # Custom command '/greet World' renders to 'Greet World with @a.txt'
        # Then, since mentions are enabled, '@a.txt' expands to the contents of a.txt.
        # So loop.called_with[0] should contain the greeting and the contents of a.txt!
        assert len(loop.called_with) == 1
        call_input = loop.called_with[0]
        assert "Greet World" in call_input
        assert "--- a.txt ---\n```\nHello A\n```" in call_input
