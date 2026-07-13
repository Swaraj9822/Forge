"""Tests for MemoryProvider (query-conditioned, budgeted, ephemeral)."""

from __future__ import annotations

from pathlib import Path

from forge.context_providers import MemoryProvider
from forge.memory import MemoryStore
from forge.session import Message, Session


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / ".forge" / "memory.jsonl", workspace_root=tmp_path)


def _session(messages: list[Message]) -> Session:
    return Session(id="s", created_at="t", updated_at="t", messages=list(messages))


def test_injects_relevant_memory(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("python debugging is done with pdb")
    provider = MemoryProvider(store, limit=5, char_budget=2000)
    session = _session([Message("user", "how do I do python debugging?")])

    segments = provider.segments(session)
    assert len(segments) == 1
    assert segments[0]["role"] == "user"
    assert MemoryProvider.HEADER in segments[0]["content"]
    assert "pdb" in segments[0]["content"]


def test_no_segment_when_no_user_text(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("python note")
    provider = MemoryProvider(store, limit=5, char_budget=2000)
    # No user message => empty query => no segment.
    assert provider.segments(_session([])) == []


def test_no_segment_when_no_relevant_memory(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("completely unrelated content")
    provider = MemoryProvider(store, limit=5, char_budget=2000)
    session = _session([Message("user", "python asyncio question")])
    assert provider.segments(session) == []


def test_budget_truncates_body(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # Long prose (spaces prevent the secret/base64 redaction from collapsing it).
    store.add("python " + "note phrase " * 30)
    store.add("python " + "other detail " * 30)
    provider = MemoryProvider(store, limit=5, char_budget=100)
    session = _session([Message("user", "python")])
    segments = provider.segments(session)
    assert len(segments) == 1
    # Content is header + a budget-bounded body with a truncation marker.
    assert "… (truncated)" in segments[0]["content"]


def test_not_persisted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("python note here")
    provider = MemoryProvider(store, limit=5, char_budget=2000)
    session = _session([Message("user", "python")])
    provider.segments(session)
    # The ephemeral segment must never be written to the session.
    assert all(
        MemoryProvider.HEADER not in (m.text or "") for m in session.messages
    )


def test_uses_latest_user_message(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("kubernetes deployment notes")
    provider = MemoryProvider(store, limit=5, char_budget=2000)
    session = _session(
        [
            Message("user", "tell me about python"),
            Message("model", "sure"),
            Message("user", "actually, kubernetes deployment"),
        ]
    )
    segments = provider.segments(session)
    assert len(segments) == 1
    assert "kubernetes" in segments[0]["content"]
