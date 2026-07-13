"""Durable cross-session memory store (Feature F).

Provides a JSONL-backed append-only memory store with:
- Secret redaction on write
- Keyword+recency ranking for search
- Path-based staleness invalidation
- Atomic writes and corrupt-line tolerance
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MemoryRecord:
    """A single memory record stored in JSONL format."""

    id: str
    text: str
    tags: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    created_at: str = ""
    source: str = "model"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": self.id,
            "text": self.text,
            "tags": list(self.tags),
            "paths": list(self.paths),
            "created_at": self.created_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryRecord:
        """Deserialize from a dict, tolerating missing/malformed fields."""
        return cls(
            id=str(data.get("id", "")),
            text=str(data.get("text", "")),
            tags=tuple(str(t) for t in data.get("tags", []) if isinstance(t, str)),
            paths=tuple(str(p) for p in data.get("paths", []) if isinstance(p, str)),
            created_at=str(data.get("created_at", "")),
            source=str(data.get("source", "model")),
        )


# ---------------------------------------------------------------------------
# Secret redaction (pure, best-effort defense-in-depth)
# ---------------------------------------------------------------------------

# Patterns for common secrets (case-insensitive where noted)
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    # API keys, tokens, passwords, authorization values. The value runs to the
    # end of the line (not just the first whitespace-delimited token) so a
    # `Authorization: Bearer <jwt>` style value is fully redacted rather than
    # leaving the token after the scheme in the clear.
    re.compile(
        r"(?i)(api[_\-]?key|secret|token|password|authorization)\s*[:=]\s*.+"
    ),
    # AWS access key IDs
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # PEM private key blocks
    re.compile(
        r"-----BEGIN [^-]+ PRIVATE KEY-----.*?-----END [^-]+ PRIVATE KEY-----",
        re.DOTALL,
    ),
    # Long high-entropy hex strings (64+ chars, likely hashes/tokens)
    re.compile(r"\b[0-9a-fA-F]{64,}\b"),
    # Long base64 strings (80+ chars)
    re.compile(r"\b[A-Za-z0-9+/]{80,}={0,2}\b"),
]

_REDACTED_PLACEHOLDER = "«redacted»"


def redact_secrets(text: str) -> str:
    """Redact common secret patterns from text.

    This is best-effort defense-in-depth, not a security guarantee.
    Regular prose is unchanged; only matching patterns are replaced.
    """
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(_REDACTED_PLACEHOLDER, result)
    return result


# ---------------------------------------------------------------------------
# Ranking and staleness (pure, offline)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase split on non-alphanumeric boundaries."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _keyword_score(query: str, record: MemoryRecord) -> int:
    """Score a memory against a query based on keyword overlap.

    Tag matches are weighted 2x vs text matches.
    """
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0

    text_tokens = set(_tokenize(record.text))
    tag_tokens = set(t.lower() for t in record.tags)

    text_overlap = len(query_tokens & text_tokens)
    tag_overlap = len(query_tokens & tag_tokens)

    return text_overlap + tag_overlap * 2


def _parse_iso8601(s: str) -> datetime | None:
    """Best-effort parse of an ISO-8601 timestamp."""
    if not s:
        return None
    try:
        # Handle both Z and +00:00 suffixes
        s_clean = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s_clean)
    except (ValueError, TypeError):
        return None


def _check_staleness(
    record: MemoryRecord,
    workspace_root: Path | None,
) -> bool:
    """Return True if the memory is stale (file changed after creation).

    A memory is stale when:
    - It lists paths, AND
    - Any of those paths' current mtime > memory's created_at, OR
    - Any listed path no longer exists.
    """
    if not record.paths or workspace_root is None:
        return False

    mem_time = _parse_iso8601(record.created_at)
    if mem_time is None:
        return False

    for rel_path in record.paths:
        full_path = workspace_root / rel_path
        if not full_path.is_file():
            return True  # Missing file → stale
        try:
            mtime = datetime.fromtimestamp(
                full_path.stat().st_mtime, tz=timezone.utc
            )
            # Allow a 1-second tolerance for filesystem mtime resolution: only
            # a file modified more than a second AFTER the memory was created
            # marks it stale. (Adding the tolerance to the memory time — rather
            # than truncating it down — is what makes this a tolerance instead
            # of a bias toward false staleness.)
            if mtime > mem_time + timedelta(seconds=1):
                return True
        except OSError:
            return True
    return False


def search_memories(
    query: str,
    memories: list[MemoryRecord],
    *,
    limit: int = 10,
    workspace_root: Path | None = None,
) -> list[MemoryRecord]:
    """Rank memories by keyword score + recency, filtering stale entries.

    Returns the top `limit` non-stale memories, sorted by:
    1. Descending keyword score
    2. Descending recency (newer first) as tie-breaker
    """
    scored: list[tuple[int, datetime, MemoryRecord]] = []

    for record in memories:
        # Skip stale memories
        if _check_staleness(record, workspace_root):
            continue

        score = _keyword_score(query, record)
        # Only relevant memories (at least one keyword overlap) are returned.
        # A zero score means the query did not match the memory at all, so it
        # must not surface: search returns matches, not the whole store. This
        # also makes a no-match query correctly return an empty list.
        if score <= 0:
            continue
        created = _parse_iso8601(record.created_at) or datetime.min.replace(
            tzinfo=timezone.utc
        )
        scored.append((score, created, record))

    # Sort by score desc, then recency desc
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    return [r for _, _, r in scored[:limit]]


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------


class MemoryStore:
    """JSONL-backed append-only memory store with atomic writes.

    Stores memories in a single JSONL file, one record per line.
    Supports append, search with ranking, and periodic pruning.
    Tolerates corrupt/partial lines on read (skips them).
    """

    def __init__(self, path: Path, *, max_records: int = 500, workspace_root: Path | None = None) -> None:
        self._path = path
        self._max_records = max_records
        self._workspace_root = workspace_root or path.parent.parent

    @property
    def path(self) -> Path:
        return self._path

    @property
    def workspace_root(self) -> Path:
        """Return the workspace root for staleness detection."""
        return self._workspace_root

    def add(
        self,
        text: str,
        *,
        tags: tuple[str, ...] = (),
        paths: tuple[str, ...] = (),
        source: str = "model",
    ) -> MemoryRecord:
        """Redact secrets, create a record, and append it atomically.

        Prunes to max_records after appending.
        """
        redacted = redact_secrets(text)
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            text=redacted,
            tags=tags,
            paths=paths,
            created_at=datetime.now(timezone.utc).isoformat(),
            source=source,
        )

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Append atomically: write to temp file in same dir, then os.replace
        # For append, we read existing + new line, write temp, replace
        self._append_record(record)
        self.prune()
        return record

    def all(self) -> list[MemoryRecord]:
        """Read all records, tolerating corrupt/partial lines."""
        if not self._path.is_file():
            return []

        records: list[MemoryRecord] = []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        record = MemoryRecord.from_dict(data)
                        if record.id and record.text:
                            records.append(record)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # Corrupt line: skip and continue
                        continue
        except OSError:
            return []
        return records

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        workspace_root: Path | None = None,
    ) -> list[MemoryRecord]:
        """Search memories with keyword+recency ranking and staleness filtering."""
        return search_memories(
            query,
            self.all(),
            limit=limit,
            workspace_root=workspace_root,
        )

    def prune(self) -> None:
        """Keep only the newest max_records, dropping the rest."""
        records = self.all()
        if len(records) <= self._max_records:
            return

        # Sort by created_at descending (newest first), keep top N
        records.sort(
            key=lambda r: _parse_iso8601(r.created_at) or datetime.min.replace(
                tzinfo=timezone.utc
            ),
            reverse=True,
        )
        kept = records[: self._max_records]

        # Rewrite atomically
        self._write_all(kept)

    def _append_record(self, record: MemoryRecord) -> None:
        """Append a single record to the JSONL file.

        Uses a plain append (``open(..., "a")``) rather than a
        read-all/rewrite/replace cycle. Appending one line is O(1) in the store
        size and — because each record is a self-contained line — is robust
        under concurrent writers: two sessions appending at once each add their
        own line instead of racing on a whole-file rewrite where one update
        would be lost. A crash mid-line leaves a partial trailing line, which
        :meth:`all` already tolerates by skipping unparsable lines.
        """
        line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"

        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)

    def _write_all(self, records: list[MemoryRecord]) -> None:
        """Rewrite the entire JSONL file with the given records atomically."""
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{self._path.name}.", suffix=".tmp", dir=str(parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            os.replace(tmp_path, self._path)
        except OSError:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
