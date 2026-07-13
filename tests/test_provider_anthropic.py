"""Unit tests for AnthropicProvider error, translation, and streaming paths."""

from __future__ import annotations

from types import SimpleNamespace
import pytest

from forge.config import Config
from forge.interrupt import InterruptController
from forge.providers import (
    AnthropicProvider,
    AuthorizationError,
    RateLimitError,
    RequestTimeoutError,
    TextDelta,
    UsageReport,
    Done,
)
from forge.providers.anthropic import _to_anthropic_messages, _to_anthropic_tools
from forge.session import ToolCall
from forge.tools.base import ToolSpec


class FakeAnthropicError(Exception):
    def __init__(self, status_code: int, message: str = "error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(headers={"retry-after": "10"})


def test_anthropic_message_translation() -> None:
    # 1. System extraction
    contents = [
        {"role": "system", "content": "You are a subagent."},
        {"role": "user", "content": "hello"},
        {"role": "model", "content": "hi"},
    ]
    messages, system = _to_anthropic_messages(contents)
    assert system == "You are a subagent."
    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    assert messages[1] == {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}

    # 2. Tool result correlation
    contents_with_tool = [
        {"role": "model", "content": "", "tool_calls": [{"id": "c1", "name": "read", "args": {}}]},
        {"role": "tool", "tool_result": {"call_id": "c1", "ok": True, "content": "file data"}},
    ]
    messages_tool, _ = _to_anthropic_messages(contents_with_tool)
    assert len(messages_tool) == 2
    assert messages_tool[0] == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "c1", "name": "read", "input": {}}]
    }
    assert messages_tool[1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "file data"}]
    }


def test_anthropic_tool_translation() -> None:
    specs = [ToolSpec(name="read", description="read file", parameters={"type": "object"})]
    tools = _to_anthropic_tools(specs)
    assert tools == [{
        "name": "read",
        "description": "read file",
        "input_schema": {"type": "object"},
    }]


def test_anthropic_stream_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    config = Config(provider_type="anthropic", model="claude-3-opus")
    interrupt = InterruptController()
    provider = AnthropicProvider(config, interrupt)

    fake_stream_chunks = [
        SimpleNamespace(type="message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=0))),
        SimpleNamespace(type="content_block_start", index=0, content_block=SimpleNamespace(type="text")),
        SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="hello ")),
        SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="world")),
        SimpleNamespace(type="content_block_start", index=1, content_block=SimpleNamespace(type="tool_use", id="c2", name="search")),
        SimpleNamespace(type="content_block_delta", index=1, delta=SimpleNamespace(type="input_json_delta", partial_json='{"qu')),
        SimpleNamespace(type="content_block_delta", index=1, delta=SimpleNamespace(type="input_json_delta", partial_json='ery": "test"}')),
        SimpleNamespace(type="content_block_stop", index=1),
        SimpleNamespace(type="message_delta", usage=SimpleNamespace(input_tokens=10, output_tokens=15)),
    ]

    class FakeClient:
        class FakeMessages:
            def create(self, *args, **kwargs):
                return fake_stream_chunks
        messages = FakeMessages()

    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient())

    events = list(provider.generate_stream([], []))
    assert UsageReport(10, 0) in events
    assert TextDelta("hello ") in events
    assert TextDelta("world") in events
    assert ToolCall("c2", "search", {"query": "test"}) in events
    assert UsageReport(10, 15) in events
    assert isinstance(events[-1], Done)


def test_anthropic_error_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    config = Config(provider_type="anthropic", model="claude-3-opus")
    interrupt = InterruptController()
    provider = AnthropicProvider(config, interrupt)

    # 1. 401 Unauthorized
    def raise_401(*args, **kwargs):
        raise FakeAnthropicError(401)
    
    class FakeClient1:
        class FakeMessages:
            def create(self, *args, **kwargs):
                raise_401()
        messages = FakeMessages()

    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient1())
    with pytest.raises(AuthorizationError):
        list(provider.generate_stream([], []))

    # 2. 429 Rate Limit
    def raise_429(*args, **kwargs):
        raise FakeAnthropicError(429)
    
    class FakeClient2:
        class FakeMessages:
            def create(self, *args, **kwargs):
                raise_429()
        messages = FakeMessages()

    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient2())
    # Instant backoff
    monkeypatch.setattr(provider, "_backoff", lambda attempt, retry_after=None: True)
    with pytest.raises(RateLimitError) as exc_info:
        list(provider.generate_stream([], []))
    assert exc_info.value.retry_after == 10.0
