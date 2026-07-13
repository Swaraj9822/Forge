"""Tests for forge.memory.MemoryStore."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from forge.memory import MemoryRecord, MemoryStore


@pytest.fixture
def tmp_store(tmp_path: Path) -> MemoryStore:
    """Return a MemoryStore rooted in a temp directory."""
    return MemoryStore(tmp_path / ".forge" / "memory.jsonl", max_records=10)


class TestMemoryRecord:
    """Tests for the MemoryRecord dataclass."""

    def test_to_dict_roundtrip(self) -> None:
        """Record round-trips through to_dict/from_dict."""
        r = MemoryRecord(
            id="abc",
            text="hello",
            tags=("a", "b"),
            paths=("x.py",),
            created_at="2025-01-01T00:00:00+00:00",
            source="model",
        )
        d = r.to_dict()
        r2 = MemoryRecord.from_dict(d)
        assert r2 == r

    def test_from_dict_tolerates_missing_fields(self) -> None:
        """from_dict tolerates missing/malformed fields."""
        r = MemoryRecord.from_dict({})
        assert r.id == ""
        assert r.text == ""
        assert r.tags == ()

    def test_from_dict_drops_non_string_tags(self) -> None:
        """Non-string tags are silently dropped."""
        r = MemoryRecord.from_dict({"tags": ["a", 123, None, "b"]})
        assert r.tags == ("a", "b")


class TestMemoryStoreAdd:
    """Tests for MemoryStore.add."""

    def test_add_returns_record(self, tmp_store: MemoryStore) -> None:
        """add() returns a MemoryRecord with a UUID."""
        rec = tmp_store.add("test memory")
        assert isinstance(rec, MemoryRecord)
        assert len(rec.id) > 0
        assert rec.text == "test memory"

    def test_add_persists_to_file(self, tmp_store: MemoryStore) -> None:
        """add() persists the record to the JSONL file."""
        tmp_store.add("memory one")
        tmp_store.add("memory two")
        records = tmp_store.all()
        assert len(records) == 2
        assert records[0].text == "memory one"
        assert records[1].text == "memory two"

    def test_add_with_tags_and_paths(self, tmp_store: MemoryStore) -> None:
        """add() stores tags and paths."""
        rec = tmp_store.add("tagged", tags=("a", "b"), paths=("x.py",))
        assert rec.tags == ("a", "b")
        assert rec.paths == ("x.py",)

    def test_add_redacts_secrets(self, tmp_store: MemoryStore) -> None:
        """add() redacts secrets before storing."""
        rec = tmp_store.add("api_key=sk-1234567890abcdef")
        assert "sk-1234567890abcdef" not in rec.text
        assert "redacted" in rec.text.lower() or "«redacted»" in rec.text


class TestMemoryStoreAll:
    """Tests for MemoryStore.all."""

    def test_all_empty_when_no_file(self, tmp_path: Path) -> None:
        """all() returns [] when the file doesn't exist."""
        store = MemoryStore(tmp_path / "nonexistent.jsonl")
        assert store.all() == []

    def test_all_tolerates_corrupt_lines(self, tmp_store: MemoryStore) -> None:
        """all() skips corrupt/empty lines."""
        # Write a mix of valid and corrupt lines
        valid = json.dumps({"id": "ok", "text": "good", "tags": [], "paths": [], "created_at": "", "source": "model"})
        corrupt_lines = [
            "",  # empty line
            "not json",  # corrupt
            "{bad json",  # corrupt
            valid,  # valid
            "x" * 1000 + "{",  # corrupt
        ]
        tmp_store._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_store._path.write_text("\n".join(corrupt_lines), encoding="utf-8")

        records = tmp_store.all()
        assert len(records) == 1
        assert records[0].id == "ok"


class TestMemoryStorePrune:
    """Tests for MemoryStore.prune."""

    def test_prune_keeps_newest(self, tmp_store: MemoryStore) -> None:
        """prune() keeps only the newest max_records."""
        for i in range(15):
            tmp_store.add(f"memory {i}")

        records = tmp_store.all()
        assert len(records) <= 10  # max_records=10

    def test_prune_no_op_when_under_limit(self, tmp_store: MemoryStore) -> None:
        """prune() is a no-op when under max_records."""
        tmp_store.add("one")
        tmp_store.add("two")
        records = tmp_store.all()
        assert len(records) == 2


class TestMemoryStoreSearch:
    """Tests for MemoryStore.search."""

    def test_search_returns_relevant(self, tmp_store: MemoryStore) -> None:
        """search() returns memories matching the query."""
        tmp_store.add("python is great")
        tmp_store.add("java is nice")
        tmp_store.add("python debugging tips")

        hits = tmp_store.search("python")
        assert len(hits) == 2
        assert all("python" in h.text.lower() for h in hits)

    def test_search_returns_empty_for_no_match(self, tmp_store: MemoryStore) -> None:
        """search() returns [] when nothing matches."""
        tmp_store.add("python is great")
        hits = tmp_store.search("xyznonexistent")
        assert hits == []
