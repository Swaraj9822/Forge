"""Unit tests for OpenAIProvider error, translation, and streaming paths."""

from __future__ import annotations

from types import SimpleNamespace
import pytest

from forge.config import Config
from forge.interrupt import InterruptController
from forge.providers import (
    OpenAIProvider,
    AuthorizationError,
    RateLimitError,
    RequestTimeoutError,
    TextDelta,
    UsageReport,
    Done,
)
from forge.providers.openai import _to_openai_messages, _to_openai_tools
from forge.session import ToolCall
from forge.tools.base import ToolSpec


class FakeOpenAIError(Exception):
    def __init__(self, status_code: int, message: str = "error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(headers={"retry-after": "5"})


def test_openai_message_translation() -> None:
    # 1. Standard messages
    contents = [
        {"role": "system", "content": "System instruction."},
        {"role": "user", "content": "hi"},
        {"role": "model", "content": "hello"},
    ]
    messages = _to_openai_messages(contents)
    assert len(messages) == 3
    assert messages[0] == {"role": "system", "content": "System instruction."}
    assert messages[1] == {"role": "user", "content": "hi"}
    assert messages[2] == {"role": "assistant", "content": "hello"}

    # 2. Tool calls and results
    contents_with_tool = [
        {"role": "model", "content": "", "tool_calls": [{"id": "c1", "name": "read", "args": {"path": "a.txt"}}]},
        {"role": "tool", "tool_result": {"call_id": "c1", "ok": True, "content": "file contents"}},
    ]
    messages_tool = _to_openai_messages(contents_with_tool)
    assert len(messages_tool) == 2
    assert messages_tool[0] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": '{"path": "a.txt"}',
                },
            }
        ],
    }
    assert messages_tool[1] == {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "read",
        "content": "file contents",
    }


def test_openai_tool_translation() -> None:
    specs = [ToolSpec(name="read", description="read file", parameters={"type": "object"})]
    tools = _to_openai_tools(specs)
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "read file",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_openai_stream_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    config = Config(provider_type="openai", model="gpt-4o")
    interrupt = InterruptController()
    provider = OpenAIProvider(config, interrupt)

    fake_stream_chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello "))], usage=None),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="world"))], usage=None),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="c1",
                                function=SimpleNamespace(name="read", arguments='{"pa'),
                            )
                        ]
                    )
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                function=SimpleNamespace(name=None, arguments='th": "a.txt"}'),
                            )
                        ]
                    )
                )
            ],
            usage=None,
        ),
        SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=15)),
    ]

    class FakeClient:
        class FakeChat:
            class FakeCompletions:
                def create(self, *args, **kwargs):
                    return fake_stream_chunks
            completions = FakeCompletions()
        chat = FakeChat()

    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient())

    events = list(provider.generate_stream([], []))
    assert TextDelta("hello ") in events
    assert TextDelta("world") in events
    assert ToolCall("c1", "read", {"path": "a.txt"}) in events
    assert UsageReport(10, 15) in events
    assert isinstance(events[-1], Done)


def test_openai_error_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    config = Config(provider_type="openai", model="gpt-4o")
    interrupt = InterruptController()
    provider = OpenAIProvider(config, interrupt)

    # 1. 401 Unauthorized
    def raise_401(*args, **kwargs):
        raise FakeOpenAIError(401)
    
    class FakeClient1:
        class FakeChat:
            class FakeCompletions:
                def create(self, *args, **kwargs):
                    raise_401()
            completions = FakeCompletions()
        chat = FakeChat()

    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient1())
    with pytest.raises(AuthorizationError):
        list(provider.generate_stream([], []))

    # 2. 429 Rate Limit
    def raise_429(*args, **kwargs):
        raise FakeOpenAIError(429)
    
    class FakeClient2:
        class FakeChat:
            class FakeCompletions:
                def create(self, *args, **kwargs):
                    raise_429()
            completions = FakeCompletions()
        chat = FakeChat()

    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient2())
    # Instant backoff
    monkeypatch.setattr(provider, "_backoff", lambda attempt, retry_after=None: True)
    with pytest.raises(RateLimitError) as exc_info:
        list(provider.generate_stream([], []))
    assert exc_info.value.retry_after == 5.0
