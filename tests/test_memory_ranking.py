"""Tests for memory ranking + staleness (forge.memory.search_memories)."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from forge.memory import MemoryRecord, search_memories


def _rec(text: str, *, tags=(), paths=(), created_at: str | None = None) -> MemoryRecord:
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    return MemoryRecord(
        id=text[:8] or "id",
        text=text,
        tags=tuple(tags),
        paths=tuple(paths),
        created_at=created_at,
        source="model",
    )


def test_relevance_ordering() -> None:
    """A memory sharing more query words ranks higher."""
    a = _rec("python testing tips")          # 2 overlap with "python testing"
    b = _rec("python only")                  # 1 overlap
    c = _rec("unrelated content")            # 0 overlap -> filtered
    hits = search_memories("python testing", [b, c, a], limit=10)
    assert [h.text for h in hits] == ["python testing tips", "python only"]


def test_zero_score_is_filtered_out() -> None:
    """A memory with no keyword overlap is not returned."""
    a = _rec("python is great")
    b = _rec("java is nice")
    hits = search_memories("python", [a, b], limit=10)
    assert [h.text for h in hits] == ["python is great"]


def test_no_match_returns_empty() -> None:
    hits = search_memories("nonexistentquery", [_rec("hello world")], limit=10)
    assert hits == []


def test_tag_matches_weighted_higher() -> None:
    """A tag hit outranks a text-only hit for the same query."""
    tagged = _rec("some note", tags=("deploy",))     # tag overlap => score 2
    text_only = _rec("deploy the app now")           # text overlap => score 1
    hits = search_memories("deploy", [text_only, tagged], limit=10)
    assert hits[0] is tagged


def test_recency_breaks_ties() -> None:
    """Equal-score memories are ordered newest-first."""
    older = _rec(
        "python note",
        created_at=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
    )
    newer = _rec(
        "python note",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    hits = search_memories("python", [older, newer], limit=10)
    assert hits[0] is newer


def test_limit_is_respected() -> None:
    recs = [_rec(f"python item {i}") for i in range(10)]
    hits = search_memories("python", recs, limit=3)
    assert len(hits) == 3


def test_stale_memory_is_dropped(tmp_path: Path) -> None:
    """A memory whose file changed after creation is filtered out."""
    f = tmp_path / "code.py"
    f.write_text("x = 1\n", encoding="utf-8")
    # Memory created in the past; the file's mtime is 'now' (newer) => stale.
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    rec = _rec("python code note", paths=("code.py",), created_at=past)
    hits = search_memories("python", [rec], limit=10, workspace_root=tmp_path)
    assert hits == []


def test_missing_path_makes_memory_stale(tmp_path: Path) -> None:
    """A memory referencing a now-missing file is treated as stale."""
    rec = _rec(
        "python gone note",
        paths=("deleted.py",),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    hits = search_memories("python", [rec], limit=10, workspace_root=tmp_path)
    assert hits == []


def test_pathless_memory_never_stale(tmp_path: Path) -> None:
    """A memory with no associated paths is never filtered by staleness."""
    rec = _rec(
        "python plain note",
        created_at=(datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
    )
    hits = search_memories("python", [rec], limit=10, workspace_root=tmp_path)
    assert len(hits) == 1
