"""Unit tests for SessionStore edge cases (task 4.4).

Covers atomic + sequential-write behavior, unknown-id load error, and
corrupt-file handling (on-disk bytes left untouched, ``CorruptSessionError``
raised, and ``list()`` skipping the corrupt file).

Validates: Requirements 13.2, 13.3, 13.6, 13.7

Note: pytest's ``tmp_path`` fixture is unusable on this host, so each test
creates its own directory with ``tempfile.mkdtemp()`` and cleans it up with
``shutil.rmtree`` in a ``finally`` block.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

import pytest

from forge.session import (
    CorruptSessionError,
    Message,
    Session,
    SessionNotFoundError,
    SessionStore,
    Usage,
    session_from_json,
)


def _make_session(store: SessionStore, *, text: str = "hello") -> Session:
    """Build an in-memory session with a single user message."""
    session = store.new()
    session.messages.append(Message(role="user", text=text))
    return session


def test_atomic_write_round_trip_leaves_single_file_and_no_temp_files() -> None:
    """save() then load() round-trips, and the atomic replace leaves exactly
    one ``<id>.json`` with no leftover temp files.

    Validates: Requirements 13.2
    """
    tmpdir = tempfile.mkdtemp()
    try:
        store = SessionStore(Path(tmpdir))
        session = _make_session(store, text="round-trip")

        store.save(session)

        # Loads back equal to what was saved.
        assert store.load(session.id) == session

        entries = os.listdir(tmpdir)
        # Exactly one persisted session file with the expected name.
        json_files = [name for name in entries if name.endswith(".json")]
        assert json_files == [f"{session.id}.json"]

        # The atomic temp file (prefix ".<id>.", suffix ".tmp") is gone.
        leftover_temps = [
            name
            for name in entries
            if name.startswith(f".{session.id}.") and name.endswith(".tmp")
        ]
        assert leftover_temps == []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_concurrent_writes_to_same_session_serialize_without_corruption() -> None:
    """Multiple threads saving the SAME session concurrently are serialized by
    the per-session lock; no exception is raised and the resulting file is
    valid JSON that loads to one of the written versions.

    Validates: Requirements 13.3
    """
    tmpdir = tempfile.mkdtemp()
    try:
        store = SessionStore(Path(tmpdir))
        base = _make_session(store, text="v0")

        # Build several distinct versions of the SAME session (same id).
        versions: list[Session] = []
        for i in range(10):
            versions.append(
                Session(
                    id=base.id,
                    created_at=base.created_at,
                    updated_at=f"2024-01-01T00:00:{i:02d}+00:00",
                    messages=[Message(role="user", text=f"v{i}")],
                    todos=[],
                    usage=Usage(input_tokens=i, output_tokens=i, estimated_cost=None),
                )
            )

        errors: list[BaseException] = []
        barrier = threading.Barrier(len(versions))

        def writer(ver: Session) -> None:
            try:
                # Maximize contention by releasing all threads together.
                barrier.wait()
                store.save(ver)
            except BaseException as exc:  # noqa: BLE001 - record for assertion
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(v,)) for v in versions]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No write raised, the serialization never interleaved.
        assert errors == []

        # The on-disk file is valid JSON (never half-written) and loads cleanly
        # to one of the exact versions that was written.
        target = Path(tmpdir) / f"{base.id}.json"
        loaded = session_from_json(target.read_text(encoding="utf-8"))
        assert loaded in versions

        # load() agrees and does not raise.
        assert store.load(base.id) == loaded

        # No leftover temp files after the storm of concurrent saves.
        leftover_temps = [
            name
            for name in os.listdir(tmpdir)
            if name.startswith(f".{base.id}.") and name.endswith(".tmp")
        ]
        assert leftover_temps == []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_load_unknown_id_raises_session_not_found() -> None:
    """Loading an id with no file raises SessionNotFoundError, distinct from
    CorruptSessionError.

    Validates: Requirements 13.6
    """
    tmpdir = tempfile.mkdtemp()
    try:
        store = SessionStore(Path(tmpdir))

        with pytest.raises(SessionNotFoundError) as excinfo:
            store.load("does-not-exist")

        # The error names the unknown session id...
        assert excinfo.value.session_id == "does-not-exist"
        # ...and is NOT a CorruptSessionError (the two cases are distinct).
        assert not isinstance(excinfo.value, CorruptSessionError)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_corrupt_file_load_raises_and_leaves_bytes_untouched() -> None:
    """A file that exists but is not parseable raises CorruptSessionError
    (naming the session), the on-disk bytes are unchanged after the failed
    load, and list() skips it without raising.

    Validates: Requirements 13.7
    """
    tmpdir = tempfile.mkdtemp()
    try:
        store = SessionStore(Path(tmpdir))

        corrupt_id = "corrupt-session"
        corrupt_path = Path(tmpdir) / f"{corrupt_id}.json"
        corrupt_bytes = b"{ this is : not valid json :: \x00\xff"
        corrupt_path.write_bytes(corrupt_bytes)

        before = corrupt_path.read_bytes()

        with pytest.raises(CorruptSessionError) as excinfo:
            store.load(corrupt_id)
        assert excinfo.value.session_id == corrupt_id

        # The failed load must not have touched the on-disk bytes.
        after = corrupt_path.read_bytes()
        assert after == before == corrupt_bytes

        # Also exercise a syntactically-valid-JSON-but-wrong-shape file to make
        # sure parse-shape failures are treated as corruption too.
        bad_shape_id = "bad-shape"
        bad_shape_path = Path(tmpdir) / f"{bad_shape_id}.json"
        bad_shape_path.write_text(json.dumps({"not": "a session"}), encoding="utf-8")
        with pytest.raises(CorruptSessionError):
            store.load(bad_shape_id)

        # And a valid session alongside the corrupt ones.
        good = _make_session(store, text="i am fine")
        store.save(good)

        # list() must not raise and must skip the corrupt files, returning only
        # the well-formed session.
        metas = store.list()
        listed_ids = {m.id for m in metas}
        assert good.id in listed_ids
        assert corrupt_id not in listed_ids
        assert bad_shape_id not in listed_ids
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
