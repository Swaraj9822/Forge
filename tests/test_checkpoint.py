"""Tests for the CheckpointStore (Phase 2, Feature C).

The store snapshots workspace files before mutation, groups snapshots per
turn, restores the most recent turn on :meth:`undo_last`, and prunes to
``keep_turns``. Storage is workspace-local and repo-independent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.checkpoint import CheckpointStore
from forge.interrupt import InterruptController
from forge.tools.base import ToolContext


def _ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace_root=workspace, interrupt=InterruptController())


# --------------------------------------------------------------------------- #
# Begin / snapshot / commit
# --------------------------------------------------------------------------- #


def test_begin_creates_turn_id(tmp_path: Path) -> None:
    store = CheckpointStore(
        root=tmp_path,
        store_dir=tmp_path / "checkpoints",
    )
    assert store._turn_id is None
    store.begin_turn()
    assert store._turn_id is not None
    # begin_turn is idempotent within a turn.
    first_id = store._turn_id
    store.begin_turn()
    assert store._turn_id == first_id


def test_snapshot_records_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    store = CheckpointStore(
        root=tmp_path, store_dir=tmp_path / "checkpoints"
    )
    store.begin_turn()
    store.snapshot_before(str(target), _ctx(tmp_path))
    store.commit_turn()
    # A manifest must now exist under the store dir.
    assert (tmp_path / "checkpoints").exists()
    manifest_dirs = [
        p for p in (tmp_path / "checkpoints").iterdir() if p.is_dir()
    ]
    assert len(manifest_dirs) == 1
    assert (manifest_dirs[0] / "manifest.json").is_file()


def test_snapshot_records_absence(tmp_path: Path) -> None:
    """A new file is recorded as 'did not exist' so undo can delete it."""

    new_path = tmp_path / "new.txt"
    assert not new_path.exists()
    store = CheckpointStore(
        root=tmp_path, store_dir=tmp_path / "checkpoints"
    )
    store.begin_turn()
    store.snapshot_before(str(new_path), _ctx(tmp_path))
    store.commit_turn()
    # Manifest exists.
    manifest_dirs = [
        p for p in (tmp_path / "checkpoints").iterdir() if p.is_dir()
    ]
    assert len(manifest_dirs) == 1


def test_snapshot_idempotent_per_turn(tmp_path: Path) -> None:
    """snapshot_before for the same path within a turn is recorded once."""

    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    store = CheckpointStore(
        root=tmp_path, store_dir=tmp_path / "checkpoints"
    )
    store.begin_turn()
    store.snapshot_before(str(target), _ctx(tmp_path))
    store.snapshot_before(str(target), _ctx(tmp_path))
    store.snapshot_before(str(target), _ctx(tmp_path))
    store.commit_turn()
    manifest_dirs = [
        p for p in (tmp_path / "checkpoints").iterdir() if p.is_dir()
    ]
    assert len(manifest_dirs) == 1
    # Read the manifest and assert exactly one entry.
    import json

    manifest = json.loads((manifest_dirs[0] / "manifest.json").read_text("utf-8"))
    assert len(manifest["entries"]) == 1


def test_snapshot_skips_out_of_scope_paths(tmp_path: Path) -> None:
    """Paths that escape the workspace are silently skipped (no entry)."""

    outside = tmp_path.parent / "outside.txt"
    store = CheckpointStore(
        root=tmp_path, store_dir=tmp_path / "checkpoints"
    )
    store.begin_turn()
    store.snapshot_before(str(outside), _ctx(tmp_path))
    store.commit_turn()
    # No turn dir should have been written (no entries).
    assert not (tmp_path / "checkpoints").exists() or not any(
        (tmp_path / "checkpoints").iterdir()
    )


def test_snapshot_skips_oversized_files(tmp_path: Path) -> None:
    """A file larger than max_bytes is recorded as 'skipped' not snapshotted."""

    target = tmp_path / "big.bin"
    target.write_bytes(b"x" * 100)
    store = CheckpointStore(
        root=tmp_path,
        store_dir=tmp_path / "checkpoints",
        max_bytes=10,
    )
    store.begin_turn()
    store.snapshot_before(str(target), _ctx(tmp_path))
    store.commit_turn()
    import json

    manifest_dirs = [
        p for p in (tmp_path / "checkpoints").iterdir() if p.is_dir()
    ]
    assert len(manifest_dirs) == 1
    manifest = json.loads((manifest_dirs[0] / "manifest.json").read_text("utf-8"))
    entry = manifest["entries"][0]
    assert entry["snapshot"] is None
    assert "too large" in entry.get("error", "")


# --------------------------------------------------------------------------- #
# Undo
# --------------------------------------------------------------------------- #


def test_undo_last_restores_existing_file_bytes(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("original", encoding="utf-8")

    store = CheckpointStore(
        root=tmp_path, store_dir=tmp_path / "checkpoints"
    )
    store.begin_turn()
    store.snapshot_before(str(target), _ctx(tmp_path))
    store.commit_turn()

    # Mutate the file.
    target.write_text("MUTATED", encoding="utf-8")

    # Undo restores the original bytes.
    restored = store.undo_last()
    assert restored == [str(target.resolve())]
    assert target.read_text(encoding="utf-8") == "original"


def test_undo_last_deletes_newly_created_file(tmp_path: Path) -> None:
    new_path = tmp_path / "new.txt"
    assert not new_path.exists()

    store = CheckpointStore(
        root=tmp_path, store_dir=tmp_path / "checkpoints"
    )
    store.begin_turn()
    store.snapshot_before(str(new_path), _ctx(tmp_path))
    store.commit_turn()

    # Simulate the tool creating the file.
    new_path.write_text("created", encoding="utf-8")
    assert new_path.exists()

    restored = store.undo_last()
    assert restored == [str(new_path.resolve())]
    assert not new_path.exists()


def test_undo_last_empty_store_returns_empty_list(tmp_path: Path) -> None:
    store = CheckpointStore(
        root=tmp_path, store_dir=tmp_path / "checkpoints"
    )
    # No turn has been committed.
    assert store.undo_last() == []


def test_undo_last_only_reverts_newest_turn(tmp_path: Path) -> None:
    """A second undo reverts the previous turn, not the same one twice."""

    target = tmp_path / "a.txt"
    target.write_text("v1", encoding="utf-8")

    store = CheckpointStore(
        root=tmp_path, store_dir=tmp_path / "checkpoints"
    )

    # Turn 1: snapshot v1, mutate to v2.
    store.begin_turn()
    store.snapshot_before(str(target), _ctx(tmp_path))
    store.commit_turn()
    target.write_text("v2", encoding="utf-8")

    # Turn 2: snapshot v2, mutate to v3.
    store.begin_turn()
    store.snapshot_before(str(target), _ctx(tmp_path))
    store.commit_turn()
    target.write_text("v3", encoding="utf-8")

    # First undo -> v2.
    store.undo_last()
    assert target.read_text(encoding="utf-8") == "v2"
    # Second undo -> v1.
    store.undo_last()
    assert target.read_text(encoding="utf-8") == "v1"
    # Third undo -> nothing left.
    assert store.undo_last() == []


# --------------------------------------------------------------------------- #
# Pruning
# --------------------------------------------------------------------------- #


def test_keep_turns_prunes_oldest(tmp_path: Path) -> None:
    """Beyond keep_turns, the oldest turn groups are pruned."""

    target = tmp_path / "a.txt"
    target.write_text("v", encoding="utf-8")

    store = CheckpointStore(
        root=tmp_path,
        store_dir=tmp_path / "checkpoints",
        keep_turns=2,
    )

    # Commit three turns (each turn snapshots a different write so the bytes
    # differ but the path is the same -- that's fine, we only care about the
    # group count).
    for i in range(3):
        store.begin_turn()
        store.snapshot_before(str(target), _ctx(tmp_path))
        target.write_text(f"v{i}", encoding="utf-8")
        store.commit_turn()

    # Only two turn dirs should remain.
    turn_dirs = sorted(
        p for p in (tmp_path / "checkpoints").iterdir() if p.is_dir()
    )
    assert len(turn_dirs) == 2


def test_keep_turns_zero_keeps_nothing(tmp_path: Path) -> None:
    """keep_turns=0 prunes every committed turn immediately on commit."""

    target = tmp_path / "a.txt"
    target.write_text("v", encoding="utf-8")
    store = CheckpointStore(
        root=tmp_path,
        store_dir=tmp_path / "checkpoints",
        keep_turns=0,
    )
    store.begin_turn()
    store.snapshot_before(str(target), _ctx(tmp_path))
    store.commit_turn()
    # Either the dir is empty/missing or has no turn dirs.
    if (tmp_path / "checkpoints").exists():
        turn_dirs = [
            p for p in (tmp_path / "checkpoints").iterdir() if p.is_dir()
        ]
        assert turn_dirs == []


# --------------------------------------------------------------------------- #
# Round-trip: multiple files in one turn
# --------------------------------------------------------------------------- #


def test_multiple_files_in_one_turn(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("A1", encoding="utf-8")
    b.write_text("B1", encoding="utf-8")

    store = CheckpointStore(
        root=tmp_path, store_dir=tmp_path / "checkpoints"
    )
    store.begin_turn()
    store.snapshot_before(str(a), _ctx(tmp_path))
    store.snapshot_before(str(b), _ctx(tmp_path))
    store.commit_turn()

    a.write_text("A2", encoding="utf-8")
    b.write_text("B2", encoding="utf-8")

    restored = sorted(store.undo_last())
    assert restored == sorted([str(a.resolve()), str(b.resolve())])
    assert a.read_text(encoding="utf-8") == "A1"
    assert b.read_text(encoding="utf-8") == "B1"
