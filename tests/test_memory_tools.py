"""Tests for the memory tools: RememberTool and SearchMemoryTool."""

from __future__ import annotations

from pathlib import Path

from forge.interrupt import InterruptController
from forge.memory import MemoryStore
from forge.tools.base import ToolContext
from forge.tools.memory import RememberTool, SearchMemoryTool


def _ctx(tmp_path: Path, memory: MemoryStore | None = None) -> ToolContext:
    return ToolContext(
        workspace_root=tmp_path,
        interrupt=InterruptController(),
        memory=memory,
    )


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / ".forge" / "memory.jsonl", workspace_root=tmp_path)


# --------------------------------------------------------------------------- #
# RememberTool
# --------------------------------------------------------------------------- #


def test_remember_is_read_only() -> None:
    assert RememberTool().read_only is True


def test_remember_stores_and_returns_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    tool = RememberTool()
    result = tool.run({"text": "remember this"}, _ctx(tmp_path, store))
    assert result.ok is True
    assert result.meta.get("memory_id")
    assert store.all()[0].text == "remember this"


def test_remember_redacts_before_store(tmp_path: Path) -> None:
    store = _store(tmp_path)
    tool = RememberTool()
    tool.run({"text": "api_key=sk-secret-value"}, _ctx(tmp_path, store))
    stored = store.all()[0].text
    assert "sk-secret-value" not in stored


def test_remember_unavailable_without_store(tmp_path: Path) -> None:
    tool = RememberTool()
    result = tool.run({"text": "x"}, _ctx(tmp_path, memory=None))
    assert result.ok is False
    assert result.meta.get("unavailable") is True


def test_remember_validation() -> None:
    tool = RememberTool()
    assert tool.validate({}) is not None                       # missing text
    assert tool.validate({"text": ""}) is not None             # empty text
    assert tool.validate({"text": "x", "tags": [1]}) is not None  # non-str tag
    assert tool.validate({"text": "x", "paths": [1]}) is not None
    assert tool.validate({"text": "ok", "tags": ["a"]}) is None


# --------------------------------------------------------------------------- #
# SearchMemoryTool
# --------------------------------------------------------------------------- #


def test_search_memory_is_read_only() -> None:
    assert SearchMemoryTool().read_only is True


def test_search_memory_returns_hits(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("python debugging tips")
    store.add("java notes")
    tool = SearchMemoryTool()
    result = tool.run({"query": "python"}, _ctx(tmp_path, store))
    assert result.ok is True
    results = result.meta.get("results")
    assert len(results) == 1
    assert "python" in results[0]["text"].lower()


def test_search_memory_no_hits(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("python notes")
    tool = SearchMemoryTool()
    result = tool.run({"query": "nonexistentxyz"}, _ctx(tmp_path, store))
    assert result.ok is True
    assert result.meta.get("results") == []
    assert "No relevant memories" in result.content


def test_search_memory_unavailable_without_store(tmp_path: Path) -> None:
    tool = SearchMemoryTool()
    result = tool.run({"query": "x"}, _ctx(tmp_path, memory=None))
    assert result.ok is False
    assert result.meta.get("unavailable") is True


def test_search_memory_validation() -> None:
    tool = SearchMemoryTool()
    assert tool.validate({}) is not None                    # missing query
    assert tool.validate({"query": 1}) is not None          # non-str query
    assert tool.validate({"query": "x", "limit": 0}) is not None  # limit < 1
    assert tool.validate({"query": "x", "limit": True}) is not None  # bool
    assert tool.validate({"query": "x", "limit": 5}) is None
