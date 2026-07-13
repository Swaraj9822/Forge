"""Property-based test for verification record / feedback round-trip.

# Feature: auto-verification-loop, Property 9: Verification records and feedback round-trip losslessly
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.session import (
    Message,
    Session,
    ToolCall,
    ToolResultRecord,
    Usage,
    VerificationRecord,
    session_from_json,
    session_to_json,
)

# --------------------------------------------------------------------------- #
# Strategies
#
# JSON only has string keys and finite numbers, so dict keys are constrained to
# strings and floats to finite values. Text spans the full non-surrogate
# Unicode range so non-ASCII verification output (e.g. tracebacks with box
# characters, accented identifiers) round-trips losslessly.
# --------------------------------------------------------------------------- #

# Text without surrogates so JSON round-trips losslessly; covers non-ASCII.
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


def json_values(max_leaves: int = 10) -> st.SearchStrategy:
    """JSON-safe values: scalars plus nested lists/dicts with string keys."""
    return st.recursive(
        JSON_SCALARS,
        lambda children: st.one_of(
            st.lists(children, max_size=3),
            st.dictionaries(SAFE_TEXT, children, max_size=3),
        ),
        max_leaves=max_leaves,
    )


JSON_DICTS = st.dictionaries(SAFE_TEXT, json_values(), max_size=3)

tool_calls = st.builds(
    ToolCall,
    id=SAFE_TEXT,
    name=SAFE_TEXT,
    args=JSON_DICTS,
    thought_signature=st.one_of(st.none(), SAFE_TEXT),
)

tool_results = st.builds(
    ToolResultRecord,
    call_id=SAFE_TEXT,
    ok=st.booleans(),
    content=SAFE_TEXT,
    error=st.one_of(st.none(), SAFE_TEXT),
    meta=JSON_DICTS,
)

# Plain (non-feedback) conversation messages.
plain_messages = st.builds(
    Message,
    role=st.sampled_from(["system", "user", "model", "tool"]),
    text=st.one_of(st.none(), SAFE_TEXT),
    tool_calls=st.lists(tool_calls, max_size=3),
    tool_result=st.one_of(st.none(), tool_results),
)


@st.composite
def feedback_messages(draw: st.DrawFn) -> Message:
    """A synthesized Verification_Feedback message.

    These are appended as the ``user_text`` of a correction turn, so they are
    ordinary user-role messages whose body renders the failing command,
    outcome, exit code, and captured (possibly truncated) combined output.
    """
    command = draw(SAFE_TEXT)
    outcome = draw(st.sampled_from(["failed", "timed_out"]))
    exit_code = draw(st.one_of(st.none(), st.integers(min_value=-256, max_value=256)))
    truncated = draw(st.booleans())
    output = draw(SAFE_TEXT)
    exit_repr = "unavailable" if exit_code is None else str(exit_code)
    trunc_marker = " (truncated)" if truncated else ""
    body = (
        "The verification command failed. Please fix the underlying problem.\n\n"
        f"Command: {command}\n"
        f"Status: {outcome}\n"
        f"Exit code: {exit_repr}\n"
        f"Output{trunc_marker}:\n"
        f"{output}"
    )
    return Message(role="user", text=body, tool_calls=[], tool_result=None)


# Mix plain and feedback messages so sessions contain interleaved feedback.
messages = st.one_of(plain_messages, feedback_messages())

verification_records = st.builds(
    VerificationRecord,
    command=SAFE_TEXT,
    outcome=st.sampled_from(["passed", "failed", "timed_out", "start_error"]),
    # None exit_code models timeouts / start errors with no completed process.
    exit_code=st.one_of(st.none(), st.integers(min_value=-256, max_value=256)),
    iterations=st.integers(min_value=0, max_value=20),
    cap_reached=st.booleans(),
    truncated=st.booleans(),
)

usages = st.builds(
    Usage,
    input_tokens=st.integers(min_value=0, max_value=10**9),
    output_tokens=st.integers(min_value=0, max_value=10**9),
    estimated_cost=st.one_of(
        st.none(), st.floats(allow_nan=False, allow_infinity=False, width=64)
    ),
)

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
    """Compose a Session carrying feedback messages and verification records.

    Record and message list sizes start at 0 so the zero / one / many cases for
    both feedback messages and verification records are all exercised.
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
        messages=draw(st.lists(messages, max_size=8)),
        todos=[],
        usage=draw(usages),
        verification_records=draw(st.lists(verification_records, max_size=6)),
    )


@settings(max_examples=200, deadline=None)
@given(sessions())
def test_verification_records_and_feedback_round_trip(session: Session) -> None:
    """Serializing a Session with feedback + records and deserializing yields
    an equal Session, preserving every feedback message, the final outcome
    status, and the iteration count.

    # Feature: auto-verification-loop, Property 9: Verification records and feedback round-trip losslessly
    Validates: Requirements 11.1, 11.2, 11.3
    """
    restored = session_from_json(session_to_json(session))

    # Whole-session equality is the lossless invariant.
    assert restored == session

    # Every feedback message is preserved verbatim and in order (Req 11.1).
    assert restored.messages == session.messages

    # Each verification record's outcome status, iteration count, and the rest
    # of its fields survive the round-trip (Req 11.2, 11.3).
    assert len(restored.verification_records) == len(session.verification_records)
    for original, recovered in zip(
        session.verification_records, restored.verification_records
    ):
        assert recovered.outcome == original.outcome
        assert recovered.iterations == original.iterations
        assert recovered.command == original.command
        assert recovered.exit_code == original.exit_code
        assert recovered.cap_reached == original.cap_reached
        assert recovered.truncated == original.truncated
