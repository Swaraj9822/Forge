"""Per-turn file checkpoints so a bad turn can be reverted with /undo.

Before a write/edit mutates a workspace file, the executor asks the store to
snapshot the file's current bytes (or record that it did not exist). Snapshots
are grouped per turn; :meth:`CheckpointStore.undo_last` restores the most
recent turn's files to their pre-turn state. Storage is workspace-local and
repo-independent (no git required).

Atomicity
---------
The store reuses :mod:`tempfile` + :func:`os.replace` (the same atomic-write
discipline :class:`~forge.session.SessionStore` uses) so a partial write to a
checkpoint manifest or a snapshot blob never leaves a half-written file behind.

Size cap
--------
A single file larger than ``read_max_bytes`` (the same cap the read tool uses)
is recorded as "not checkpointable" rather than copied; ``/undo`` then warns
that the file cannot be restored. This bounds the worst-case snapshot cost.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.tools.paths import OutOfWorkspaceError, resolve_in_workspace

__all__ = ["CheckpointStore", "CheckpointError"]


class CheckpointError(Exception):
    """Raised when a checkpoint operation cannot be completed safely.

    Used for unrecoverable I/O failures during snapshot/undo; ordinary
    per-file errors (e.g. out-of-scope path, file too large) are reported
    inline in the manifest so /undo can warn rather than crash.
    """


# Manifest schema for a single checkpoint group (one turn).
_MANIFEST_VERSION = 1

# Default cap when no config is available on the context; mirrors the read
# tool's documented default so the store behaves sensibly in tests.
_DEFAULT_MAX_BYTES = 1_000_000


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    """Write ``payload`` to ``target`` atomically (tempfile + os.replace).

    When ``target`` already exists its permission bits are copied onto the temp
    file before the replace, so restoring a file via ``/undo`` preserves its
    mode (mkstemp otherwise creates the temp file 0600 and os.replace keeps
    that mode).
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        if target.exists():
            try:
                shutil.copymode(target, tmp_path)
            except OSError:
                pass
        os.replace(tmp_path, target)
    except Exception:
        # Best-effort cleanup of the temp file on any failure.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _safe_unlink(path: Path) -> None:
    """Best-effort unlink; missing files are silently ignored."""

    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


@dataclass
class CheckpointStore:
    """Workspace-local, repo-independent per-turn file checkpoint store.

    Parameters
    ----------
    root:
        The workspace root. Used to scope ``snapshot_before`` calls via
        :func:`resolve_in_workspace`.
    store_dir:
        The directory holding per-turn checkpoint groups (e.g.
        ``workspace/.forge/checkpoints``). Created on first use.
    keep_turns:
        Maximum number of turn groups retained on disk; the oldest are pruned
        by :meth:`commit_turn`. Defaults to 10.
    max_bytes:
        Maximum size of a single file the store is willing to snapshot. Files
        larger than this are recorded as "skipped" so :meth:`undo_last` can
        warn that the file cannot be restored.
    """

    root: Path
    store_dir: Path
    keep_turns: int = 10
    max_bytes: int = _DEFAULT_MAX_BYTES

    # -- runtime state (per turn) -------------------------------------------
    # ``_turn_id`` is None until begin_turn is called. ``_captured`` deduplicates
    # snapshot captures within a single turn so a file that is touched by
    # multiple write/edit calls in one turn is only recorded once.
    _turn_id: str | None = field(default=None, init=False, repr=False)
    _turn_seq: int = field(default=0, init=False, repr=False)
    _captured: set[str] = field(default_factory=set, init=False, repr=False)
    _current_entries: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    # -- lifecycle ----------------------------------------------------------

    def begin_turn(self) -> None:
        """Start a new checkpoint group for the current turn.

        Idempotent within a turn: calling :meth:`begin_turn` twice in the
        same turn is a no-op (the existing group is kept) so the agent loop
        can call it defensively.
        """

        if self._turn_id is not None:
            return
        # Turn ids must be unique AND sort chronologically by name (undo/prune
        # rely on lexical ordering). A millisecond timestamp alone collides for
        # turns committed within the same millisecond (headless / verification-
        # correction turns can be sub-millisecond apart), which would make one
        # turn's snapshots overwrite another's and break multi-level undo. Use a
        # zero-padded nanosecond timestamp (orders correctly across sessions)
        # plus a per-store sequence counter (guarantees uniqueness within a
        # process even if the clock does not advance between two turns).
        self._turn_seq += 1
        self._turn_id = f"{time.time_ns():020d}-{self._turn_seq:06d}"
        self._captured = set()
        self._current_entries = []
        # Ensure the store directory exists before any snapshot lands.
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def snapshot_before(self, path_arg: Any, ctx: Any) -> None:
        """Record ``path``s pre-mutation state once per turn.

        ``path_arg`` may be a string (the ``args["path"]`` from a tool call)
        or ``None`` (e.g. a programmatic call with no path). Out-of-scope,
        already-captured, or oversized files are recorded in the manifest
        (so ``/undo`` can warn) rather than raising, so the tool itself still
        runs.

        ``ctx`` is duck-typed: only ``ctx.workspace_root`` is consulted. The
        :class:`~forge.tools.base.ToolContext` provides this naturally.
        """

        if self._turn_id is None:
            # Defensive: if the agent loop forgets to begin_turn, start one
            # now so a checkpoint is always captured.
            self.begin_turn()
        if not isinstance(path_arg, str) or not path_arg:
            return

        workspace_root = getattr(ctx, "workspace_root", None)
        if workspace_root is None:
            return

        try:
            resolved = resolve_in_workspace(path_arg, workspace_root)
        except OutOfWorkspaceError:
            return

        key = str(resolved)
        if key in self._captured:
            return
        self._captured.add(key)

        turn_dir = self.store_dir / self._turn_id
        turn_dir.mkdir(parents=True, exist_ok=True)

        existed = resolved.is_file()
        if not existed:
            # Snapshot the absence so /undo can delete a newly-created file.
            self._current_entries.append(
                {"path": key, "existed": False, "snapshot": None}
            )
            return

        try:
            payload = resolved.read_bytes()
        except OSError as exc:
            # Unreadable file: record the failure so /undo can warn.
            self._current_entries.append(
                {
                    "path": key,
                    "existed": True,
                    "snapshot": None,
                    "error": f"could not read for snapshot: {exc}",
                }
            )
            return

        if len(payload) > self.max_bytes:
            self._current_entries.append(
                {
                    "path": key,
                    "existed": True,
                    "snapshot": None,
                    "error": "file too large to checkpoint",
                }
            )
            return

        # Hash the snapshot filename so very long paths do not exceed the
        # filesystem limit; the manifest carries the real path.
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        snapshot_path = turn_dir / f"{digest}.blob"
        try:
            _atomic_write_bytes(snapshot_path, payload)
        except OSError as exc:
            self._current_entries.append(
                {
                    "path": key,
                    "existed": True,
                    "snapshot": None,
                    "error": f"could not write snapshot: {exc}",
                }
            )
            return

        self._current_entries.append(
            {"path": key, "existed": True, "snapshot": str(snapshot_path)}
        )

    def commit_turn(self) -> None:
        """Finalize the turn's group and prune to ``keep_turns``.

        If no snapshots were captured this turn, the turn directory is not
        created (and nothing is pruned). The manifest is written atomically
        so a partial write never leaves a half-written turn group on disk.
        """

        if self._turn_id is None:
            return

        turn_dir = self.store_dir / self._turn_id

        if self._current_entries:
            manifest = {
                "version": _MANIFEST_VERSION,
                "turn_id": self._turn_id,
                "entries": list(self._current_entries),
            }
            try:
                _atomic_write_bytes(
                    turn_dir / "manifest.json",
                    json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
                )
            except OSError:
                # Could not persist the manifest: drop the (now-useless)
                # snapshot blobs and continue rather than leaving a broken
                # turn group on disk.
                self._prune_turn_dir(turn_dir)
                self._reset_turn_state()
                self._prune_old_turns()
                return

        # Reset per-turn state.
        self._reset_turn_state()

        # Enforce the retention cap.
        self._prune_old_turns()

    def undo_last(self) -> list[str]:
        """Restore the newest committed turn group.

        For each recorded entry: if the file existed pre-turn, rewrite its
        saved bytes; if it did not exist, delete the (newly-created) file.
        Files recorded with ``error`` cannot be restored and are skipped (the
        caller surfaces the warnings). Returns the list of paths touched by
        the restore (skipped entries are NOT included). Returns ``[]`` when
        there is no committed turn group to undo.
        """

        turn_dir, manifest = self._latest_committed_turn()
        if turn_dir is None or manifest is None:
            return []

        entries = manifest.get("entries", [])
        restored: list[str] = []
        for entry in entries:
            path = entry.get("path")
            existed = entry.get("existed", False)
            snapshot = entry.get("snapshot")
            if not isinstance(path, str):
                continue
            target = Path(path)
            try:
                if not existed:
                    # The file did not exist before the turn; ensure it is
                    # gone now. A pre-existing file with the same path that
                    # was not part of this turn is left untouched because the
                    # file would not have been in the manifest in that case.
                    _safe_unlink(target)
                elif isinstance(snapshot, str):
                    payload = Path(snapshot).read_bytes()
                    _atomic_write_bytes(target, payload)
                else:
                    # Record-only entry (read failure / oversize). Skip.
                    continue
                restored.append(path)
            except OSError:
                # Best-effort: continue with the rest of the turn rather than
                # abandoning the whole restore.
                continue

        # Remove the consumed turn group so a second /undo reverts the
        # previous turn, not the same one twice.
        self._prune_turn_dir(turn_dir)
        return restored

    # -- helpers ------------------------------------------------------------

    def diff_last_turn(self) -> str:
        """Return a unified diff of the most recent committed turn's changes.

        For each file the turn touched, compares the pre-turn snapshot (the
        "before" bytes captured by :meth:`snapshot_before`) against the file's
        current content (the "after"), producing a ``git diff``-style unified
        diff. A file the turn *created* (no prior snapshot) shows as an addition;
        a deleted file shows as a removal. Record-only entries (a file too large
        or unreadable to snapshot) are noted but not diffed.

        Returns ``""`` when there is no committed turn to diff. This is
        repo-independent (it uses the checkpoint snapshots, not git), so it works
        even outside a git repository. It reflects only ``write`` / ``edit`` tool
        mutations — changes made by arbitrary ``shell`` commands are not
        snapshotted and therefore not shown.
        """
        turn_dir, manifest = self._latest_committed_turn()
        if turn_dir is None or manifest is None:
            return ""

        parts: list[str] = []
        for entry in manifest.get("entries", []):
            path = entry.get("path")
            if not isinstance(path, str):
                continue
            existed = entry.get("existed", False)
            snapshot = entry.get("snapshot")

            if existed and snapshot is None:
                # Record-only entry (oversize / unreadable at snapshot time).
                parts.append(f"# {path}: changed (not captured for diff)\n")
                continue

            old = ""
            if existed and isinstance(snapshot, str):
                try:
                    old = Path(snapshot).read_bytes().decode("utf-8", "replace")
                except OSError:
                    old = ""

            new = ""
            target = Path(path)
            if target.is_file():
                try:
                    new = target.read_bytes().decode("utf-8", "replace")
                except OSError:
                    new = ""

            if old == new:
                continue

            diff = difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
            text = "".join(diff)
            if text and not text.endswith("\n"):
                text += "\n"
            if text:
                parts.append(text)

        return "".join(parts)

    def _reset_turn_state(self) -> None:
        """Clear the in-memory per-turn state after commit/undo."""

        self._turn_id = None
        self._captured = set()
        self._current_entries = []

    def _latest_committed_turn(
        self,
    ) -> tuple[Path | None, dict[str, Any] | None]:
        """Return ``(turn_dir, manifest)`` for the newest committed turn, or
        ``(None, None)`` if there is nothing to undo.

        A committed turn is any subdirectory of ``store_dir`` that has a
        ``manifest.json``. Sort key is the directory name (a zero-padded
        nanosecond timestamp plus a per-store sequence counter) so the newest
        is last in lexical order.
        """

        if not self.store_dir.exists():
            return None, None
        candidates: list[Path] = []
        for child in self.store_dir.iterdir():
            if not child.is_dir():
                continue
            if (child / "manifest.json").is_file():
                candidates.append(child)
        if not candidates:
            return None, None
        candidates.sort(key=lambda p: p.name)
        turn_dir = candidates[-1]
        try:
            data = json.loads((turn_dir / "manifest.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            # Corrupt manifest: prune it so future /undo calls do not keep
            # picking up the broken entry.
            self._prune_turn_dir(turn_dir)
            return None, None
        if not isinstance(data, dict):
            self._prune_turn_dir(turn_dir)
            return None, None
        return turn_dir, data

    def _prune_old_turns(self) -> None:
        """Remove the oldest turn groups until ``keep_turns`` remain.

        Counts every committed turn directory (the in-flight one, if any, is
        excluded because :meth:`commit_turn` resets the in-memory state
        before this method runs). A broken turn group (no manifest) is
        pruned first so it does not count against the cap.
        """

        if not self.store_dir.exists():
            return

        # Collect every turn directory, broken or committed.
        all_turns: list[tuple[Path, bool]] = []
        for child in self.store_dir.iterdir():
            if not child.is_dir():
                continue
            has_manifest = (child / "manifest.json").is_file()
            all_turns.append((child, has_manifest))

        # Drop broken (manifest-less) turn dirs first.
        for turn_dir, has_manifest in all_turns:
            if not has_manifest:
                self._prune_turn_dir(turn_dir)

        committed = sorted(
            (td for td, ok in all_turns if ok),
            key=lambda p: p.name,
        )
        excess = len(committed) - self.keep_turns
        for turn_dir in committed[: max(excess, 0)]:
            self._prune_turn_dir(turn_dir)

    @staticmethod
    def _prune_turn_dir(turn_dir: Path) -> None:
        """Best-effort recursive removal of a turn directory."""

        if not turn_dir.exists():
            return
        # Walk bottom-up so empty parents do not block the unlink.
        for root, dirs, files in os.walk(turn_dir, topdown=False):
            for name in files:
                _safe_unlink(Path(root) / name)
            for name in dirs:
                try:
                    (Path(root) / name).rmdir()
                except OSError:
                    pass
        try:
            turn_dir.rmdir()
        except OSError:
            pass
