"""Unit tests for compaction triggering, the compaction notice, summary
content, and the missing-steering-file warning.

Covers Requirements 14.2 (over-limit trigger), 14.4 (decisions/outcomes
preserved in the summary), 14.7 (compaction notice surfaced), and 15.4
(missing Steering_File warning). All summarization is mocked so the tests run
fully offline.
"""

from __future__ import annotations

import pytest

from forge.config import Config
from forge.context import (
    SUMMARY_MESSAGE_PREFIX,
    CompactionInfo,
    ContextManager,
    _message_to_window_dict,
    _serialize_message_text,
    load_default_system_prompt,
)
from forge.session import Message, Session


def make_session(messages: list[Message]) -> Session:
    """Build an in-memory Session wrapping the given messages."""

    return Session(id="t", created_at="t", updated_at="t", messages=list(messages))


def serialized_window(window: list[dict]) -> str:
    """Join the searchable text of every window message."""

    return " ".join(_serialize_message_text(w) for w in window)


# --------------------------------------------------------------------------- #
# Req 14.2 / 14.7: trigger and notice
# --------------------------------------------------------------------------- #


def test_assemble_no_compaction_when_within_limit() -> None:
    """A small window stays within the limit, so no compaction runs and the
    compaction notice is absent (Req 14.2 negative, 14.7)."""

    manager = ContextManager(Config(steering_files=[], token_limit=200_000))
    session = make_session([Message("user", "hello"), Message("model", "hi there")])

    window, info = manager.assemble(session)

    assert info is None
    assert window[0]["role"] == "system"
    assert any(w.get("content") == "hello" for w in window)


def test_assemble_triggers_compaction_over_limit() -> None:
    """When the estimate exceeds the limit, compaction runs before the request
    would be sent, summarizes the middle, and fits within the limit (Req 14.2,
    14.7, 14.8)."""

    summarizer = lambda middle: "PRESERVED-DECISIONS"  # noqa: E731
    task = Message("user", "ORIGINAL-TASK")
    middle = [Message("model", "X" * 4000) for _ in range(6)]
    recents = [
        Message("user", "recent-1"),
        Message("model", "recent-2"),
        Message("user", "recent-3"),
    ]
    messages = [task] + middle + recents  # indices 0..9, recent={7,8,9}, middle=6

    probe = ContextManager(
        Config(steering_files=[], retained_recent_messages=3), summarizer=summarizer
    )
    sys_msgs = probe.assemble_system_messages()
    summary_msg = {
        "role": "user",
        "content": f"{SUMMARY_MESSAGE_PREFIX} (6 message(s) summarized)\nPRESERVED-DECISIONS",
    }
    minimal = (
        list(sys_msgs)
        + [_message_to_window_dict(task), summary_msg]
        + [_message_to_window_dict(m) for m in recents]
    )
    limit = probe.estimate_tokens(minimal) + 200

    manager = ContextManager(
        Config(steering_files=[], retained_recent_messages=3, token_limit=limit),
        summarizer=summarizer,
    )
    session = make_session(messages)

    full = list(sys_msgs) + [_message_to_window_dict(m) for m in messages]
    assert manager.estimate_tokens(full) > limit  # precondition: over the limit

    window, info = manager.assemble(session)

    assert info is not None and info.occurred
    assert info.summary_message_count == 1
    assert info.dropped_message_count == 0
    assert manager.estimate_tokens(window) <= limit

    text = serialized_window(window)
    # Req 14.3 / 14.5: task and recent messages retained.
    assert "ORIGINAL-TASK" in text
    assert "recent-1" in text and "recent-2" in text and "recent-3" in text
    # Req 14.4: the summarizer's preserved decisions appear in the summary.
    assert "PRESERVED-DECISIONS" in text
    assert any(
        str(w.get("content", "")).startswith(SUMMARY_MESSAGE_PREFIX) for w in window
    )


def test_compaction_info_is_the_surfaced_notice() -> None:
    """assemble surfaces a CompactionInfo notice when compaction happens and
    None otherwise (Req 14.7)."""

    manager = ContextManager(Config(steering_files=[], token_limit=200_000))
    _, info = manager.assemble(make_session([Message("user", "hi")]))
    assert info is None

    summarizer = lambda middle: "S"  # noqa: E731
    big = [Message("user", "TASK")] + [Message("model", "Q" * 3000) for _ in range(5)]
    # indices 0..5, retained 2 -> recent={4,5}, task=0, middle={1,2,3} (3 msgs)
    probe = ContextManager(
        Config(steering_files=[], retained_recent_messages=2), summarizer=summarizer
    )
    sys_msgs = probe.assemble_system_messages()
    summary_msg = {
        "role": "user",
        "content": f"{SUMMARY_MESSAGE_PREFIX} (3 message(s) summarized)\nS",
    }
    minimal = (
        list(sys_msgs)
        + [_message_to_window_dict(big[0]), summary_msg]
        + [_message_to_window_dict(m) for m in big[-2:]]
    )
    limit = probe.estimate_tokens(minimal) + 200

    manager2 = ContextManager(
        Config(steering_files=[], retained_recent_messages=2, token_limit=limit),
        summarizer=summarizer,
    )
    _, info2 = manager2.assemble(make_session(big))

    assert isinstance(info2, CompactionInfo)
    assert info2.occurred
    assert info2.summary_message_count == 1


# --------------------------------------------------------------------------- #
# Req 14.8 / 14.9: dropping recents, and the cannot-reduce warning
# --------------------------------------------------------------------------- #


def test_assemble_drops_recent_when_summary_insufficient() -> None:
    """When summarization alone does not fit, retained-recent messages are
    dropped oldest-first until the window fits (Req 14.8)."""

    summarizer = lambda middle: "S"  # noqa: E731
    task = Message("user", "TASK")
    middle = [Message("model", "Y" * 4000) for _ in range(5)]
    recents = [
        Message("user", "aaaaaaaaaa"),
        Message("model", "bbbbbbbbbb"),
        Message("user", "cccccccccc"),
    ]
    messages = [task] + middle + recents  # middle = 5 messages

    probe = ContextManager(
        Config(steering_files=[], retained_recent_messages=3), summarizer=summarizer
    )
    sys_msgs = probe.assemble_system_messages()
    summary_msg = {
        "role": "user",
        "content": f"{SUMMARY_MESSAGE_PREFIX} (5 message(s) summarized)\nS",
    }
    base = list(sys_msgs) + [_message_to_window_dict(task), summary_msg]
    # Limit allows only system + task + summary, forcing every recent to drop.
    limit = probe.estimate_tokens(base)

    manager = ContextManager(
        Config(steering_files=[], retained_recent_messages=3, token_limit=limit),
        summarizer=summarizer,
    )
    window, info = manager.assemble(make_session(messages))

    assert info.occurred
    assert info.dropped_message_count == 3
    assert manager.estimate_tokens(window) <= limit
    # Task is never dropped.
    assert any("TASK" in str(w.get("content", "")) for w in window)


def test_compaction_warns_when_it_cannot_reduce() -> None:
    """When even the smallest well-formed window exceeds the limit, a warning is
    emitted and the manager proceeds, still retaining the task (Req 14.9)."""

    manager = ContextManager(
        Config(steering_files=[], token_limit=1, retained_recent_messages=20),
        summarizer=lambda middle: "S",
    )
    session = make_session([Message("user", "TASK"), Message("model", "stuff")])

    with pytest.warns(UserWarning, match="could not reduce"):
        window, info = manager.assemble(session)

    assert info.occurred
    assert any("TASK" in str(w.get("content", "")) for w in window)


# --------------------------------------------------------------------------- #
# Req 14.4: summary content via a VertexClient-like summarizer (offline)
# --------------------------------------------------------------------------- #


class _Delta:
    """A TextDelta-shaped streaming event exposing ``.text``."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeVertexClient:
    """A VertexClient-like summarizer that streams fixed text deltas offline."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.calls: list[tuple] = []

    def generate_stream(self, contents, tools):
        self.calls.append((contents, tools))
        for chunk in self._chunks:
            yield _Delta(chunk)


def test_vertex_like_summarizer_used_offline() -> None:
    """A VertexClient-like summarizer is driven via generate_stream and its
    streamed text is concatenated into the summary, with no network call
    (Req 14.4)."""

    fake = _FakeVertexClient(["Decisions: ", "kept."])
    task = Message("user", "TASK")
    middle = [Message("model", "Z" * 3000) for _ in range(4)]
    recents = [Message("user", "r1"), Message("model", "r2")]
    messages = [task] + middle + recents  # middle = 4 messages

    probe = ContextManager(
        Config(steering_files=[], retained_recent_messages=2), summarizer=fake
    )
    sys_msgs = probe.assemble_system_messages()
    summary_msg = {
        "role": "user",
        "content": f"{SUMMARY_MESSAGE_PREFIX} (4 message(s) summarized)\nDecisions: kept.",
    }
    minimal = (
        list(sys_msgs)
        + [_message_to_window_dict(task), summary_msg]
        + [_message_to_window_dict(m) for m in recents]
    )
    limit = probe.estimate_tokens(minimal) + 200

    manager = ContextManager(
        Config(steering_files=[], retained_recent_messages=2, token_limit=limit),
        summarizer=fake,
    )
    window, info = manager.assemble(make_session(messages))

    assert info.occurred and info.summary_message_count == 1
    assert "Decisions: kept." in serialized_window(window)
    # The summarizer was actually invoked (locally), proving the offline path.
    assert fake.calls


# --------------------------------------------------------------------------- #
# Req 15.4: missing-steering-file warning
# --------------------------------------------------------------------------- #


def test_missing_steering_file_warns_and_is_skipped(tmp_path) -> None:
    """A configured Steering_File that does not exist triggers a warning naming
    it and is skipped, leaving only the built-in default prompt (Req 15.4)."""

    missing = tmp_path / "absent.md"
    manager = ContextManager(Config(steering_files=[str(missing)]))

    with pytest.warns(UserWarning, match="absent.md"):
        messages = manager.assemble_system_messages()

    assert len(messages) == 1
    assert messages[0]["content"] == load_default_system_prompt()


def test_missing_steering_file_among_existing_continues(tmp_path) -> None:
    """A missing Steering_File is warned about and skipped while the remaining
    existing prompts continue to be used in order (Req 15.4)."""

    present = tmp_path / "present.md"
    present.write_text("STEER-CONTENT", encoding="utf-8")
    missing = tmp_path / "gone.md"

    manager = ContextManager(Config(steering_files=[str(present), str(missing)]))

    with pytest.warns(UserWarning, match="gone.md"):
        messages = manager.assemble_system_messages()

    assert [m["content"] for m in messages] == [
        load_default_system_prompt(),
        "STEER-CONTENT",
    ]
