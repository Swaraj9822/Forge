"""Session data models and lossless JSON serialization.

This module defines the dataclasses that make up a persisted Forge Session
and the functions that serialize a ``Session`` to JSON and reconstruct an
equal ``Session`` from that JSON. The serialization is lossless: every nested
``ToolCall``, ``ToolResultRecord``, ``Message``, ``TodoItem`` and ``Usage``
round-trips back into an equal dataclass instance (not a raw ``dict``), so the
default dataclass ``__eq__`` holds after a serialize -> deserialize cycle.

Each session is stored on disk as a single JSON file named
``<session_id>.json`` (see task 4.2, ``SessionStore``).
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ToolCall:
    """A structured request emitted by the Model naming a Tool and its args.

    ``thought_signature`` carries the opaque base64-encoded signature some
    Gemini models (e.g. Gemini 3) attach to the response part that holds a
    function call. It must be preserved and sent back with the function call in
    subsequent requests, or the API rejects the call with a 400 "missing
    thought_signature" error. It is ``None`` for models/responses that do not
    emit one.
    """

    id: str
    name: str
    args: dict
    thought_signature: str | None = None


@dataclass
class ToolResultRecord:
    """The structured output returned to the Model after a Tool runs."""

    call_id: str
    ok: bool
    content: str
    error: str | None
    meta: dict


@dataclass
class Message:
    """A single message in the conversation.

    ``role`` is one of "system" | "user" | "model" | "tool".
    ``tool_calls`` is present (non-empty) on model messages.
    ``tool_result`` is present on tool messages.
    """

    role: str
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_result: ToolResultRecord | None = None


@dataclass
class TodoItem:
    """A single planning todo item.

    ``status`` is one of "pending" | "in_progress" | "completed".
    """

    id: str
    text: str
    status: str


@dataclass
class Usage:
    """Cumulative token usage and estimated cost for a session."""

    input_tokens: int
    output_tokens: int
    estimated_cost: float | None


@dataclass
class VerificationRecord:
    """A persisted record of one completed Verification_Phase.

    Captures the final outcome of a Verify_Command run and the bounded
    self-correction loop that followed it. The captured failure output is not
    duplicated here; it lives verbatim in the persisted Verification_Feedback
    messages appended by the correction turns.

    ``outcome`` is one of "passed" | "failed" | "timed_out" | "start_error".
    ``exit_code`` is present when the process ran to completion, else ``None``.
    ``iterations`` is the number of Correction_Iterations performed.
    ``cap_reached`` is ``True`` when the phase ended at the iteration cap
    without a passing result. ``truncated`` is ``True`` when the captured
    combined output exceeded the configured output cap.
    """

    command: str
    outcome: str
    exit_code: int | None
    iterations: int
    cap_reached: bool
    truncated: bool


@dataclass
class Session:
    """A persisted record of a conversation."""

    id: str
    created_at: str  # ISO-8601 UTC
    updated_at: str  # ISO-8601 UTC
    messages: list[Message] = field(default_factory=list)
    todos: list[TodoItem] = field(default_factory=list)
    usage: Usage = field(
        default_factory=lambda: Usage(
            input_tokens=0, output_tokens=0, estimated_cost=None
        )
    )
    verification_records: list[VerificationRecord] = field(default_factory=list)


@dataclass
class SessionMeta:
    """Lightweight session descriptor returned by ``SessionStore.list()``.

    Carries at least the session identifier and its creation timestamp.
    """

    id: str
    created_at: str


# --------------------------------------------------------------------------- #
# Serialization: dataclass <-> plain dict <-> JSON
#
# We convert to plain dicts explicitly (rather than dataclasses.asdict, which
# recurses through nested dataclasses but would still require an explicit
# reconstruction path) so that from_dict can rebuild the exact nested dataclass
# instances and equality is preserved on round-trip.
# --------------------------------------------------------------------------- #


def _tool_call_to_dict(call: ToolCall) -> dict:
    return {
        "id": call.id,
        "name": call.name,
        "args": call.args,
        "thought_signature": call.thought_signature,
    }


def _tool_call_from_dict(data: dict) -> ToolCall:
    return ToolCall(
        id=data["id"],
        name=data["name"],
        args=data["args"],
        thought_signature=data.get("thought_signature"),
    )


def _tool_result_to_dict(record: ToolResultRecord) -> dict:
    return {
        "call_id": record.call_id,
        "ok": record.ok,
        "content": record.content,
        "error": record.error,
        "meta": record.meta,
    }


def _tool_result_from_dict(data: dict) -> ToolResultRecord:
    return ToolResultRecord(
        call_id=data["call_id"],
        ok=data["ok"],
        content=data["content"],
        error=data["error"],
        meta=data["meta"],
    )


def _message_to_dict(message: Message) -> dict:
    return {
        "role": message.role,
        "text": message.text,
        "tool_calls": [_tool_call_to_dict(c) for c in message.tool_calls],
        "tool_result": (
            _tool_result_to_dict(message.tool_result)
            if message.tool_result is not None
            else None
        ),
    }


def _message_from_dict(data: dict) -> Message:
    tool_result = data.get("tool_result")
    return Message(
        role=data["role"],
        text=data["text"],
        tool_calls=[_tool_call_from_dict(c) for c in data.get("tool_calls", [])],
        tool_result=(
            _tool_result_from_dict(tool_result) if tool_result is not None else None
        ),
    )


def _todo_to_dict(todo: TodoItem) -> dict:
    return {"id": todo.id, "text": todo.text, "status": todo.status}


def _todo_from_dict(data: dict) -> TodoItem:
    return TodoItem(id=data["id"], text=data["text"], status=data["status"])


def _usage_to_dict(usage: Usage) -> dict:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "estimated_cost": usage.estimated_cost,
    }


def _usage_from_dict(data: dict) -> Usage:
    return Usage(
        input_tokens=data["input_tokens"],
        output_tokens=data["output_tokens"],
        estimated_cost=data["estimated_cost"],
    )


def _verification_record_to_dict(record: VerificationRecord) -> dict:
    return {
        "command": record.command,
        "outcome": record.outcome,
        "exit_code": record.exit_code,
        "iterations": record.iterations,
        "cap_reached": record.cap_reached,
        "truncated": record.truncated,
    }


def _verification_record_from_dict(data: dict) -> VerificationRecord:
    return VerificationRecord(
        command=data["command"],
        outcome=data["outcome"],
        exit_code=data["exit_code"],
        iterations=data["iterations"],
        cap_reached=data["cap_reached"],
        truncated=data["truncated"],
    )


def session_to_dict(session: Session) -> dict:
    """Convert a ``Session`` into a JSON-serializable plain ``dict``."""
    return {
        "id": session.id,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "messages": [_message_to_dict(m) for m in session.messages],
        "todos": [_todo_to_dict(t) for t in session.todos],
        "usage": _usage_to_dict(session.usage),
        "verification_records": [
            _verification_record_to_dict(r) for r in session.verification_records
        ],
    }


def session_from_dict(data: dict) -> Session:
    """Reconstruct a ``Session`` (with nested dataclasses) from a plain ``dict``."""
    return Session(
        id=data["id"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        messages=[_message_from_dict(m) for m in data.get("messages", [])],
        todos=[_todo_from_dict(t) for t in data.get("todos", [])],
        usage=_usage_from_dict(data["usage"]),
        verification_records=[
            _verification_record_from_dict(r)
            for r in data.get("verification_records", [])
        ],
    )


def session_to_json(session: Session, *, indent: int | None = 2) -> str:
    """Serialize a ``Session`` to a JSON string losslessly."""
    return json.dumps(
        session_to_dict(session), indent=indent, ensure_ascii=False, sort_keys=True
    )


def session_from_json(text: str) -> Session:
    """Deserialize a JSON string into a ``Session`` equal to the original."""
    return session_from_dict(json.loads(text))


# --------------------------------------------------------------------------- #
# Persistence: SessionStore (save / load / list / new)
#
# Each session is one JSON file named ``<session_id>.json`` in the store root.
# Saves are atomic (temp file + os.replace) and serialized per session by an
# in-process lock so concurrent writes to the same session never interleave.
# --------------------------------------------------------------------------- #


class SessionStoreError(Exception):
    """Base class for session-store errors."""


class SessionNotFoundError(SessionStoreError):
    """Raised when a requested session id does not exist on disk (Req 13.6).

    Distinct from :class:`CorruptSessionError` so the CLI layer can tell an
    unknown session apart from one whose file is present but unparseable.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"No saved session with id '{session_id}'")


class CorruptSessionError(SessionStoreError):
    """Raised when a stored session file cannot be parsed (Req 13.7).

    The on-disk bytes are left untouched; the caller may surface the affected
    session id to the user.
    """

    def __init__(self, session_id: str, *, detail: str | None = None) -> None:
        self.session_id = session_id
        self.detail = detail
        message = f"Session '{session_id}' is corrupted and could not be parsed"
        if detail:
            message += f": {detail}"
        super().__init__(message)


def _utc_now_iso() -> str:
    """Current time as an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """Persists, lists, and restores Sessions as JSON files on disk.

    Args:
        root: Directory under which ``<session_id>.json`` files are stored.
            Created lazily on first save if it does not already exist.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        # Guards access to the per-session lock map below.
        self._locks_guard = threading.Lock()
        # Per-session locks keyed by session id so concurrent saves to the same
        # session serialize while saves to different sessions stay independent.
        self._session_locks: dict[str, threading.Lock] = {}

    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_id] = lock
            return lock

    def _path_for(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def new(self) -> Session:
        """Mint a fresh in-memory Session (UUIDv4 id, UTC timestamps).

        Does not write to disk; persistence happens via :meth:`save`.
        """

        now = _utc_now_iso()
        return Session(
            id=str(uuid.uuid4()),
            created_at=now,
            updated_at=now,
            messages=[],
            todos=[],
            usage=Usage(input_tokens=0, output_tokens=0, estimated_cost=None),
        )

    def save(self, session: Session) -> None:
        """Persist ``session`` atomically (Req 13.1, 13.2, 13.3).

        Writes to a temp file in the same directory then ``os.replace``s it onto
        ``<session_id>.json``. An in-process per-session lock serializes
        concurrent saves to the same session. The root directory is created if
        absent.
        """

        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path_for(session.id)
        payload = session_to_json(session)

        with self._lock_for(session.id):
            # Temp file in the SAME directory guarantees os.replace is atomic
            # (a cross-device rename would not be).
            fd, tmp_name = _mkstemp_in(self.root, session.id)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_name, target)
            except BaseException:
                # Best-effort cleanup of the temp file on any failure so we do
                # not leave partial artifacts behind; the target is untouched.
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise

    def load(self, session_id: str) -> Session:
        """Load and return the stored Session for ``session_id``.

        Raises :class:`SessionNotFoundError` for an unknown id (Req 13.6) and
        :class:`CorruptSessionError` when the file exists but cannot be parsed,
        leaving the on-disk bytes untouched (Req 13.7).
        """

        target = self._path_for(session_id)
        if not target.exists():
            raise SessionNotFoundError(session_id)

        try:
            # Read inside the guarded block so an undecodable file (invalid
            # UTF-8) is treated as corruption too, not just unparseable JSON or
            # a wrong-shape payload. The bytes are only ever read here, never
            # rewritten, so the on-disk data is left untouched on failure.
            text = target.read_text(encoding="utf-8")
            return session_from_json(text)
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as err:
            # Parse/decode/shape failure: do not touch the file, signal
            # corruption (Req 13.7).
            raise CorruptSessionError(session_id, detail=str(err)) from err

    def list(self) -> list[SessionMeta]:
        """Return a ``SessionMeta`` (id + created_at) per stored session.

        Scans the root for ``*.json`` files. Corrupt or unreadable files are
        skipped gracefully so listing never raises (Req 13.4).
        """

        if not self.root.exists():
            return []

        metas: list[SessionMeta] = []
        for entry in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(entry.read_text(encoding="utf-8"))
                metas.append(
                    SessionMeta(id=data["id"], created_at=data["created_at"])
                )
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                # Skip corrupt/unreadable files rather than failing the listing.
                continue
        return metas


def _mkstemp_in(directory: Path, session_id: str) -> tuple[int, str]:
    """Create a uniquely-named temp file in ``directory`` for an atomic save."""
    import tempfile

    return tempfile.mkstemp(
        prefix=f".{session_id}.", suffix=".tmp", dir=str(directory)
    )
