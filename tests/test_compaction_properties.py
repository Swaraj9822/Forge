"""Property-based test for context compaction bound and retention.

# Feature: forge, Property 21: Compaction bound and retention invariant

The summarization Model is mocked with a fixed-string callable summarizer so
the property is fully OFFLINE: no network call is ever made.
"""

from __future__ import annotations

import warnings

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.config import Config
from forge.context import (
    ContextManager,
    _message_to_window_dict,
    _serialize_message_text,
)
from forge.session import Message, Session, ToolCall, ToolResultRecord

# Message bodies use only lowercase letters so the marker delimiters ``<<`` and
# ``>>`` never occur inside a random body and remain reliable to search for.
BODY = st.text(
    alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    min_size=0,
    max_size=120,
)


def _marker(i: int) -> str:
    """The unique, searchable marker embedded in message ``i``'s text."""

    return f"<<m{i}>>"


@st.composite
def sessions_with_limits(draw: st.DrawFn):
    """Generate (messages, retained_recent, token_limit).

    Each message carries a unique text marker so its presence in the assembled
    window can be detected. Some model messages carry an unresolved Tool_Call
    (a pending call) so the retention of pending calls can be exercised. The
    token limit is drawn relative to the full-window estimate so both the
    "fits" and the "cannot reduce" branches are reached across examples.
    """

    n = draw(st.integers(min_value=0, max_value=10))
    messages: list[Message] = []
    for i in range(n):
        role = draw(st.sampled_from(["user", "model", "tool"]))
        body = draw(BODY)
        text = f"{_marker(i)}{body}"
        tool_calls: list[ToolCall] = []
        tool_result = None
        if role == "model" and draw(st.booleans()):
            # An unresolved Tool_Call: no Tool_Result anywhere carries this id,
            # so this message is "pending" and must be retained (Req 14.6).
            tool_calls = [ToolCall(id=f"pending-{i}", name="op", args={"i": i})]
        if role == "tool":
            # A resolved Tool_Result whose call id never collides with the
            # pending ids above, so it does not accidentally resolve them.
            tool_result = ToolResultRecord(
                call_id=f"res-{i}", ok=True, content=body, error=None, meta={}
            )
        messages.append(
            Message(
                role=role,
                text=text,
                tool_calls=tool_calls,
                tool_result=tool_result,
            )
        )

    retained = draw(st.integers(min_value=0, max_value=6))

    # Estimate the full window (estimation is limit-independent) to pick a
    # limit that spans the cannot-reduce, reducible, and no-compaction regimes.
    probe = ContextManager(Config(steering_files=[], retained_recent_messages=retained))
    sys_msgs = probe.assemble_system_messages()
    full = list(sys_msgs) + [_message_to_window_dict(m) for m in messages]
    full_est = probe.estimate_tokens(full)
    limit = draw(st.integers(min_value=1, max_value=full_est + 20))

    return messages, retained, limit


@settings(max_examples=10)
@given(sessions_with_limits())
def test_compaction_bound_and_retention(case) -> None:
    """For any Session and Token_Limit, after assembly the window either fits
    within the limit or is the smallest well-formed window (with a warning);
    in all cases the system prompt, the original task, and pending Tool_Calls
    are retained, and the most recent configured messages are retained when the
    bound allows.

    Validates: Requirements 14.1, 14.3, 14.5, 14.6, 14.8, 14.9
    """

    messages, retained, limit = case
    n = len(messages)

    config = Config(
        steering_files=[], retained_recent_messages=retained, token_limit=limit
    )
    # Mock summarizer: fixed short string keeps the property fully offline.
    manager = ContextManager(config, summarizer=lambda middle: "FIXED-SUMMARY")
    session = Session(id="s", created_at="t", updated_at="t", messages=list(messages))

    sys_msgs = manager.assemble_system_messages()
    full = list(sys_msgs) + [_message_to_window_dict(m) for m in messages]
    full_est = manager.estimate_tokens(full)

    # Req 14.1: estimation is deterministic and offline (no network).
    assert manager.estimate_tokens(full) == full_est

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        window, info = manager.assemble(session)

    # No compaction when the full window already fits (Req 14.2 negative case).
    if full_est <= limit:
        assert info is None
        assert window == full
        return

    # Compaction occurred.
    assert info is not None and info.occurred

    # The built-in system prompt (and any steering segments) are retained as
    # the leading messages, unchanged (Property 21 / Req 14.3).
    assert window[: len(sys_msgs)] == sys_msgs

    serialized = " ".join(_serialize_message_text(w) for w in window)
    window_call_ids = {
        c["id"] for w in window for c in (w.get("tool_calls") or [])
    }

    # Recompute the partition the same way the manager does so retention can be
    # checked against the generated markers.
    resolved = {m.tool_result.call_id for m in messages if m.tool_result is not None}
    pending_indices: set[int] = set()
    pending_ids: set[str] = set()
    for i, m in enumerate(messages):
        unresolved = [c for c in m.tool_calls if c.id not in resolved]
        if unresolved:
            pending_indices.add(i)
            pending_ids.update(c.id for c in unresolved)

    task_index = (
        next((i for i, m in enumerate(messages) if m.role == "user"), 0)
        if n > 0
        else None
    )
    recent = set(range(max(0, n - retained), n)) if (retained > 0 and n > 0) else set()
    protected = ({task_index} if task_index is not None else set()) | pending_indices

    # Req 14.3: the original task/instructions are retained verbatim.
    if task_index is not None:
        assert _marker(task_index) in serialized

    # Req 14.6: every pending Tool_Call is retained.
    assert pending_ids <= window_call_ids

    final_est = manager.estimate_tokens(window)
    warned = any("could not reduce" in str(w.message) for w in caught)

    if final_est <= limit:
        # Req 14.8: compaction produced a window within the limit.
        # Req 14.5: when nothing had to be dropped, the most recent configured
        # messages are all retained.
        if info.dropped_message_count == 0:
            for i in recent:
                assert _marker(i) in serialized
    else:
        # Req 14.9: the limit could not be reached, so a warning was emitted and
        # the smallest well-formed window is used — every droppable retained
        # message (recent, but neither task nor pending) has been dropped.
        assert warned
        for i in {j for j in recent if j not in protected}:
            assert _marker(i) not in serialized
