"""Regression test: a provider error mid-turn ends the turn gracefully.

Guards against the bug where ``AgentLoop._stream_response`` caught an
undefined ``VertexError`` name, so any :class:`~forge.providers.ProviderError`
raised while streaming crashed the turn with a ``NameError`` instead of ending
it gracefully with the error surfaced on the :class:`TurnResult` (Req 2.5,
2.6, 2.8).

These fakes are fully in-process (no network, no disk) so the test runs
without a TTY or a temp directory.
"""

from __future__ import annotations

from forge.agent import AgentLoop
from forge.interrupt import InterruptController
from forge.providers import ProviderError, RateLimitError
from forge.session import Session, Usage
from forge.usage import UsageTracker


class _FakeContextManager:
    def assemble(self, session: Session):
        contents = [{"role": m.role, "content": m.text} for m in session.messages]
        return contents, None


class _RaisingProvider:
    """Provider whose stream raises after yielding nothing (or partial text)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def generate_stream(self, contents, tools):
        raise self._exc
        yield  # pragma: no cover - makes this a generator


class _RecordingStore:
    def __init__(self) -> None:
        self.saved: list[Session] = []

    def save(self, session: Session) -> None:
        self.saved.append(session)


class _NoToolExecutor:
    def specs(self):
        return []

    def execute(self, call):  # pragma: no cover - never reached
        raise AssertionError("no tool should run on a stream error")


def _make_session() -> Session:
    return Session(
        id="s",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
        usage=Usage(input_tokens=0, output_tokens=0, estimated_cost=None),
    )


def _run(exc: Exception):
    store = _RecordingStore()
    loop = AgentLoop(
        context_manager=_FakeContextManager(),
        provider=_RaisingProvider(exc),
        tool_executor=_NoToolExecutor(),
        usage_tracker=UsageTracker(),
        session_store=store,
        interrupt=InterruptController(),
    )
    session = _make_session()
    result = loop.run_turn(session, "hi")
    return result, session, store


def test_provider_error_ends_turn_gracefully() -> None:
    result, session, store = _run(ProviderError("boom"))

    # The turn ends with the error surfaced, not a raised NameError.
    assert result.error == "boom"
    assert result.interrupted is False
    # Session state is preserved: the user message and the (empty) model
    # message are retained and the session was persisted once.
    assert [m.role for m in session.messages] == ["user", "model"]
    assert session.messages[0].text == "hi"
    assert store.saved == [session]


def test_provider_error_subclass_is_also_caught() -> None:
    # A subclass (e.g. an exhausted rate-limit retry) must be handled too.
    result, _session, _store = _run(RateLimitError("slow down"))
    assert result.error == "slow down"
    assert result.interrupted is False
