"""Tests for the context-provider seam and the plan-reminder provider."""

from __future__ import annotations

from forge.config import Config
from forge.context import CompactionInfo, ContextManager, ContextProvider
from forge.context_providers import PlanReminderProvider
from forge.session import Message, Session, TodoItem


def make_session(messages: list[Message] | None = None, todos: list[TodoItem] | None = None) -> Session:
    """Build an in-memory Session for context assembly tests."""
    return Session(
        id="test",
        created_at="t",
        updated_at="t",
        messages=list(messages) if messages else [],
        todos=list(todos) if todos else [],
    )


def test_no_providers_window_is_unchanged() -> None:
    """An empty providers list reproduces the no-provider assembled window."""
    config = Config(steering_files=[], token_limit=200_000)
    manager_no_providers = ContextManager(config)
    manager_empty_providers = ContextManager(config, providers=[])
    session = make_session([Message("user", "hello")])

    window_no, info_no = manager_no_providers.assemble(session)
    window_empty, info_empty = manager_empty_providers.assemble(session)

    assert info_no is None
    assert info_empty is None
    assert window_empty == window_no


def test_plan_reminder_appended_when_todos_present() -> None:
    """The plan reminder is the final message when todos exist."""
    config = Config(steering_files=[], token_limit=200_000)
    provider = PlanReminderProvider()
    manager = ContextManager(config, providers=[provider])
    todos = [
        TodoItem(id="1", text="first thing", status="in_progress"),
        TodoItem(id="2", text="second thing", status="pending"),
    ]
    session = make_session(todos=todos)

    window, _info = manager.assemble(session)

    assert window[-1]["role"] == "user"
    reminder = window[-1]["content"]
    assert PlanReminderProvider.HEADER in reminder
    assert "(in progress) first thing" in reminder
    assert "(pending) second thing" in reminder


def test_plan_reminder_absent_when_no_todos() -> None:
    """With no todos the reminder is skipped and the window is unchanged."""
    config = Config(steering_files=[], token_limit=200_000)
    provider = PlanReminderProvider()
    manager_with = ContextManager(config, providers=[provider])
    manager_without = ContextManager(config)
    session = make_session([Message("user", "hello")])

    window_with, _info = manager_with.assemble(session)
    window_without, _info = manager_without.assemble(session)

    assert window_with == window_without
    assert PlanReminderProvider.HEADER not in " ".join(
        m.get("content", "") for m in window_with
    )


def test_plan_reminder_not_persisted() -> None:
    """Provider segments are ephemeral and never written to session.messages."""
    config = Config(steering_files=[], token_limit=200_000)
    provider = PlanReminderProvider()
    manager = ContextManager(config, providers=[provider])
    session = make_session(todos=[TodoItem(id="1", text="do it", status="pending")])

    manager.assemble(session)

    assert not any(
        PlanReminderProvider.HEADER in (m.text or "") for m in session.messages
    )


def test_plan_reminder_survives_compaction() -> None:
    """The reminder is still appended after compaction occurs."""
    summarizer = lambda middle: "summary"  # noqa: E731
    config = Config(steering_files=[], token_limit=100, retained_recent_messages=0)
    provider = PlanReminderProvider()
    manager = ContextManager(config, summarizer=summarizer, providers=[provider])
    # Enough messages to force compaction against the tiny limit.
    messages = [Message("user", "task")]
    messages.extend(Message("model", "X" * 400) for _ in range(10))
    session = make_session(messages=messages, todos=[TodoItem(id="1", text="todo", status="pending")])

    window, info = manager.assemble(session)

    assert isinstance(info, CompactionInfo)
    assert info.occurred is True
    assert window[-1]["role"] == "user"
    assert PlanReminderProvider.HEADER in window[-1]["content"]


def test_budget_is_reserved_for_ephemeral_segments() -> None:
    """A large ephemeral segment forces compaction at a lower effective limit."""
    class _BulkyProvider:
        def segments(self, session: Session) -> list[dict]:
            return [{"role": "user", "content": "B" * 4000}]

    # Base window alone fits comfortably under the limit; the bulky provider
    # reservation pushes the effective limit low enough to compact.
    config = Config(steering_files=[], token_limit=500, retained_recent_messages=0)
    summarizer = lambda middle: "summary"  # noqa: E731
    manager = ContextManager(
        config, summarizer=summarizer, providers=[_BulkyProvider()]
    )
    messages = [
        Message("user", "task"),
        Message("model", "X" * 400),
        Message("user", "ok"),
    ]
    session = make_session(messages=messages)

    window, info = manager.assemble(session)

    assert info is not None
    assert info.occurred is True
    assert window[-1]["content"] == "B" * 4000


def test_provider_exception_is_swallowed() -> None:
    """A misbehaving provider must not break context assembly."""
    class _BrokenProvider(ContextProvider):
        def segments(self, session: Session) -> list[dict]:
            raise RuntimeError("provider failure")

    config = Config(steering_files=[], token_limit=200_000)
    manager = ContextManager(config, providers=[_BrokenProvider()])
    session = make_session([Message("user", "hello")])

    window, info = manager.assemble(session)

    assert info is None
    # Window identical to no-provider case.
    assert window == ContextManager(config).assemble(session)[0]
