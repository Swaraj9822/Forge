"""Regression test: summarization must never send a request larger than the
model input limit, even when the middle region is enormous.

Reproduces the crash where compaction summarized the entire middle region in a
single request that exceeded the model's 1,048,576-token input limit.
"""

from __future__ import annotations

from forge.config import Config
from forge.context import (
    CHARS_PER_TOKEN,
    SUMMARY_INPUT_TOKEN_BUDGET,
    ContextManager,
)


class _FakeTextDelta:
    def __init__(self, text: str) -> None:
        self.text = text


class _RecordingClient:
    """VertexClient-like stub that records the size of every request it gets."""

    def __init__(self) -> None:
        self.request_char_sizes: list[int] = []

    def generate_stream(self, contents, tools):
        prompt = contents[0]["content"]
        self.request_char_sizes.append(len(prompt))
        yield _FakeTextDelta("summary")


def _big_message(i: int) -> dict:
    # Each message ~ 40k chars => ~10k estimated tokens.
    return {"role": "user", "content": f"msg-{i} " + ("x" * 40_000)}


def test_no_request_exceeds_model_input_limit():
    client = _RecordingClient()
    cm = ContextManager(Config(), summarizer=client)

    # ~150 messages * ~10k tokens = ~1.5M estimated tokens of middle region,
    # which as one request would blow past a ~1M model limit.
    middle = [_big_message(i) for i in range(150)]

    summary = cm._summarize_via_vertex(client, middle)

    assert isinstance(summary, str) and summary
    # Every request stayed within the safe budget (+ small instruction slack).
    max_allowed_chars = SUMMARY_INPUT_TOKEN_BUDGET * CHARS_PER_TOKEN + 5_000
    assert client.request_char_sizes  # requests were actually made
    assert max(client.request_char_sizes) <= max_allowed_chars
    # Chunking means more than one request was issued.
    assert len(client.request_char_sizes) > 1


def test_single_oversized_message_is_truncated():
    client = _RecordingClient()
    cm = ContextManager(Config(), summarizer=client)

    huge = {"role": "user", "content": "y" * (SUMMARY_INPUT_TOKEN_BUDGET * CHARS_PER_TOKEN * 3)}

    cm._summarize_via_vertex(client, [huge])

    max_allowed_chars = SUMMARY_INPUT_TOKEN_BUDGET * CHARS_PER_TOKEN + 5_000
    assert max(client.request_char_sizes) <= max_allowed_chars
