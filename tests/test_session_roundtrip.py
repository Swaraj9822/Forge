"""Property-based test for session serialization round-trip.

# Feature: forge, Property 20: Session serialization round-trip
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.session import (
    Message,
    Session,
    SessionMeta,
    SessionStore,
    TodoItem,
    ToolCall,
    ToolResultRecord,
    Usage,
    session_from_json,
    session_to_json,
)

# --------------------------------------------------------------------------- #
# Strategies
#
# JSON only has string keys and finite numbers, so we constrain generated
# dict keys to strings and floats to finite values. Otherwise a JSON
# round-trip would coerce non-string keys to strings (breaking equality) or
# turn NaN into something that never compares equal to itself.
# --------------------------------------------------------------------------- #

# Text without surrogates/control issues so JSON round-trips losslessly.
SAFE_TEXT = st.text(
    alphabet=st.characters(
        min_codepoint=32, max_codepoint=0x10FFFF, blacklist_categories=("Cs",)
    ),
    max_size=40,
)

JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**12), max_value=10**12),
    st.floats(allow_nan=False, allow_infinity=False, width=64),
    SAFE_TEXT,
)


def json_values(max_leaves: int = 15) -> st.SearchStrategy:
    """JSON-safe values: scalars plus nested lists/dicts with string keys."""
    return st.recursive(
        JSON_SCALARS,
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(SAFE_TEXT, children, max_size=4),
        ),
        max_leaves=max_leaves,
    )


# JSON-safe dicts with string keys, used for ToolCall.args and meta dicts.
JSON_DICTS = st.dictionaries(SAFE_TEXT, json_values(), max_size=4)


tool_calls = st.builds(
    ToolCall,
    id=SAFE_TEXT,
    name=SAFE_TEXT,
    args=JSON_DICTS,
)

tool_results = st.builds(
    ToolResultRecord,
    call_id=SAFE_TEXT,
    ok=st.booleans(),
    content=SAFE_TEXT,
    error=st.one_of(st.none(), SAFE_TEXT),
    meta=JSON_DICTS,
)

messages = st.builds(
    Message,
    role=st.sampled_from(["system", "user", "model", "tool"]),
    text=st.one_of(st.none(), SAFE_TEXT),
    tool_calls=st.lists(tool_calls, max_size=4),
    tool_result=st.one_of(st.none(), tool_results),
)

todos = st.builds(
    TodoItem,
    id=SAFE_TEXT,
    text=SAFE_TEXT,
    status=st.sampled_from(["pending", "in_progress", "completed"]),
)

usages = st.builds(
    Usage,
    input_tokens=st.integers(min_value=0, max_value=10**9),
    output_tokens=st.integers(min_value=0, max_value=10**9),
    estimated_cost=st.one_of(
        st.none(), st.floats(allow_nan=False, allow_infinity=False, width=64)
    ),
)

# created_at/updated_at are ISO-ish strings; their exact format is irrelevant
# to round-tripping as long as the same value comes back.
TIMESTAMPS = st.sampled_from(
    [
        "2024-01-01T00:00:00+00:00",
        "2023-12-31T23:59:59.123456+00:00",
        "2025-06-15T12:30:00+00:00",
        "2020-02-29T08:00:00+00:00",
    ]
)


@st.composite
def sessions(draw: st.DrawFn) -> Session:
    """Compose an arbitrary Session.

    The id is a filename-safe UUID-like token so the store can save it as
    ``<id>.json``; the round-trip property does not depend on the id format.
    """
    session_id = draw(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
            min_size=1,
            max_size=36,
        )
    )
    return Session(
        id=session_id,
        created_at=draw(TIMESTAMPS),
        updated_at=draw(TIMESTAMPS),
        messages=draw(st.lists(messages, max_size=6)),
        todos=draw(st.lists(todos, max_size=6)),
        usage=draw(usages),
    )


@settings(max_examples=10, deadline=None)
@given(sessions())
def test_session_serialization_round_trip(session: Session) -> None:
    """Serialize -> deserialize yields an equal Session.

    Validates: Requirements 13.1, 13.4, 13.5
    """
    restored = session_from_json(session_to_json(session))
    assert restored == session


@settings(max_examples=10, deadline=None)
@given(sessions())
def test_saved_session_appears_in_listing(session: Session) -> None:
    """A saved session appears in the store listing with its id and
    creation timestamp.

    Validates: Requirements 13.1, 13.4, 13.5
    """
    # Use tempfile directly (not pytest tmp_path) and clean up per example.
    tmpdir = tempfile.mkdtemp()
    try:
        store = SessionStore(Path(tmpdir))
        store.save(session)

        metas = store.list()
        assert SessionMeta(id=session.id, created_at=session.created_at) in metas

        # The loaded session also equals the saved one (lossless persistence).
        assert store.load(session.id) == session
    finally:
        # Remove the temp directory and its contents.
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
