"""Tests for the REPL's /undo command (Phase 2, Feature C).

The /undo command is special-cased like /exit: it never invokes the agent
loop, it always routes to the CheckpointStore. These tests verify the routing,
the rendering, and that the loop is not entered.
"""

from __future__ import annotations

import io
from pathlib import Path

from forge.checkpoint import CheckpointStore
from forge.interrupt import InterruptController
from forge.repl import UNDO_COMMAND, Repl, is_exit_command
from forge.session import Session
from forge.tools.base import ToolContext


def _mk_session() -> Session:
    """Build a minimal in-memory Session (the dataclass has required fields)."""

    return Session(id="undo-test", created_at="t", updated_at="t")


class _LoopSpy:
    """A stand-in AgentLoop that records every call to run_turn."""

    def __init__(self) -> None:
        self.calls: list[tuple[Session, str]] = []

    def run_turn(self, session: Session, user_text: str):  # noqa: ANN001
        self.calls.append((session, user_text))
        raise AssertionError("run_turn must not be called for /undo")


def _repl(
    loop: _LoopSpy,
    session: Session,
    checkpoint: CheckpointStore | None,
    inputs: list[str],
) -> tuple[Repl, io.StringIO]:
    out = io.StringIO()
    inputs_iter = iter(inputs)

    def input_func(_prompt: str) -> str:
        return next(inputs_iter)

    repl = Repl(
        agent_loop=loop,  # type: ignore[arg-type]
        session=session,
        input_func=input_func,
        out=out,
        checkpoint=checkpoint,
    )
    return repl, out


def _ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace_root=workspace, interrupt=InterruptController())


def test_undo_command_is_recognized() -> None:
    """The UNDO_COMMAND constant exists and is not an exit command."""

    assert UNDO_COMMAND == "/undo"
    assert not is_exit_command(UNDO_COMMAND)


def test_undo_with_nothing_to_undo_prints_empty_message(tmp_path: Path) -> None:
    loop = _LoopSpy()
    session = _mk_session()
    repl, out = _repl(loop, session, checkpoint=None, inputs=[UNDO_COMMAND])

    keep_going = repl.run_once()
    assert keep_going is True
    assert "[undo] nothing to undo" in out.getvalue()
    # The agent loop must NOT have been entered.
    assert loop.calls == []


def test_undo_with_checkpoint_restores_files(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("original", encoding="utf-8")

    store = CheckpointStore(
        root=tmp_path, store_dir=tmp_path / "checkpoints"
    )
    store.begin_turn()
    store.snapshot_before(str(target), _ctx(tmp_path))
    store.commit_turn()

    # Simulate the file being mutated during the turn we just committed.
    target.write_text("MUTATED", encoding="utf-8")

    loop = _LoopSpy()
    session = _mk_session()
    repl, out = _repl(loop, session, checkpoint=store, inputs=[UNDO_COMMAND])

    repl.run_once()
    assert "[undo] restored 1 file" in out.getvalue()
    assert str(target.resolve()) in out.getvalue()
    # The filesystem was actually restored.
    assert target.read_text(encoding="utf-8") == "original"
    assert loop.calls == []


def test_undo_routes_to_checkpoint_store(tmp_path: Path) -> None:
    """The store's undo_last is invoked exactly once per /undo call."""

    target = tmp_path / "a.txt"
    target.write_text("v1", encoding="utf-8")
    store = CheckpointStore(
        root=tmp_path, store_dir=tmp_path / "checkpoints"
    )
    store.begin_turn()
    store.snapshot_before(str(target), _ctx(tmp_path))
    store.commit_turn()

    loop = _LoopSpy()
    session = _mk_session()
    repl, out = _repl(loop, session, checkpoint=store, inputs=[UNDO_COMMAND])
    repl.run_once()

    # The turn dir should be gone now (consumed by the undo).
    turn_dirs = [p for p in store.store_dir.iterdir() if p.is_dir()]
    assert turn_dirs == []
    assert loop.calls == []


def test_undo_does_not_invoke_agent_loop(tmp_path: Path) -> None:
    """Defensive: /undo must never reach AgentLoop.run_turn."""

    loop = _LoopSpy()
    session = _mk_session()
    repl, out = _repl(loop, session, checkpoint=None, inputs=[UNDO_COMMAND])
    repl.run_once()
    assert loop.calls == []


def test_undo_with_only_whitespace_input_is_blank_not_undo(tmp_path: Path) -> None:
    """Whitespace is NOT /undo (blank handling takes precedence)."""

    loop = _LoopSpy()
    session = _mk_session()
    repl, out = _repl(loop, session, checkpoint=None, inputs=["   "])
    keep_going = repl.run_once()
    assert keep_going is True
    # Blank input re-displays the prompt without invoking anything.
    assert "nothing to undo" not in out.getvalue()
    assert loop.calls == []


def test_undo_does_not_exit_repl(tmp_path: Path) -> None:
    """The /undo command returns True (continue), not False (exit)."""

    loop = _LoopSpy()
    session = _mk_session()
    repl, _out = _repl(loop, session, checkpoint=None, inputs=[UNDO_COMMAND])
    keep_going = repl.run_once()
    assert keep_going is True
