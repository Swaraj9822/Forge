"""Tests for RepoMapProvider (ephemeral, budgeted repo-map injection)."""

from __future__ import annotations

from pathlib import Path

from forge.context_providers import RepoMapProvider
from forge.repo_index import RepoIndexer
from forge.session import Message, Session


def _session() -> Session:
    return Session(
        id="s",
        created_at="t",
        updated_at="t",
        messages=[Message("user", "do something")],
    )


def test_injects_repo_map(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("def go():\n    pass\n", encoding="utf-8")
    provider = RepoMapProvider(RepoIndexer(tmp_path), char_budget=4000)
    segments = provider.segments(_session())
    assert len(segments) == 1
    assert segments[0]["role"] == "user"
    assert RepoMapProvider.HEADER in segments[0]["content"]
    assert "mod.py" in segments[0]["content"]


def test_empty_repo_no_segment(tmp_path: Path) -> None:
    provider = RepoMapProvider(RepoIndexer(tmp_path), char_budget=4000)
    assert provider.segments(_session()) == []


def test_budget_respected(tmp_path: Path) -> None:
    for i in range(40):
        (tmp_path / f"m{i}.py").write_text(
            f"def f{i}(aaaaaaaa, bbbbbbbb):\n    pass\n", encoding="utf-8"
        )
    budget = 300
    provider = RepoMapProvider(RepoIndexer(tmp_path), char_budget=budget)
    segments = provider.segments(_session())
    assert len(segments) == 1
    # The map body is budget-bounded; the header adds a small fixed prefix.
    assert len(segments[0]["content"]) <= budget + len(RepoMapProvider.HEADER) + 2
    assert "… (truncated)" in segments[0]["content"]


def test_not_persisted(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("def go():\n    pass\n", encoding="utf-8")
    provider = RepoMapProvider(RepoIndexer(tmp_path), char_budget=4000)
    session = _session()
    provider.segments(session)
    assert all(
        RepoMapProvider.HEADER not in (m.text or "") for m in session.messages
    )
