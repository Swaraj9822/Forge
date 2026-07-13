"""Unit tests for :class:`forge.vertex.VertexClient` error and retry paths.

These tests run fully offline. No ``google-genai`` client is constructed and no
network access occurs: the raw SDK stream is isolated behind
``VertexClient._raw_stream``, which is monkeypatched per test to return a
controllable iterator of fake chunks or to raise injected exceptions. Fake
chunks are plain ``SimpleNamespace`` objects shaped for the real
``_parse_chunk`` parser so the translation / interrupt / retry / timeout layer
is exercised end to end.

Covers:
* Authorization error surfaces as :class:`AuthorizationError` (Req 2.5).
* Rate-limit retry exhaustion raises :class:`RateLimitError` after the
  configured number of attempts (Req 2.6).
* Request timeout raises :class:`RequestTimeoutError` (Req 2.8).
* Mid-stream interruption stops generation promptly while retaining the partial
  tokens already yielded, emitting no ``Done`` (Req 3.4, 4.2).
* Mid-stream error surfaces as a :class:`VertexError` after partial tokens were
  yielded (Req 3.4).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from forge import vertex
from forge.config import Config
from forge.interrupt import InterruptController
from forge.vertex import (
    AuthorizationError,
    Done,
    RateLimitError,
    RequestTimeoutError,
    TextDelta,
    VertexClient,
    VertexError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeApiError(Exception):
    """A fake SDK/API error carrying an HTTP status code via ``code``.

    ``VertexClient._translate_and_raise`` maps the status code to a typed
    exception (401/403 -> authorization, 429 -> rate limit, 504 -> timeout).
    """

    def __init__(self, code: int, message: str | None = None) -> None:
        super().__init__(message or f"status {code}")
        self.code = code


def _text_chunk(text: str) -> SimpleNamespace:
    """Build a fake SDK chunk shaped for ``_parse_chunk`` carrying one text part."""
    part = SimpleNamespace(text=text, function_call=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate], usage_metadata=None)


def _make_client(interrupt: InterruptController, **overrides) -> VertexClient:
    """Construct a VertexClient over a Config with project/region set.

    Project/region are populated so ``ConfigMissingError`` is never raised; no
    real genai client is constructed because every test patches ``_raw_stream``.
    """
    params = {"rate_limit_retries": 3, "request_timeout_s": 60}
    params.update(overrides)
    config = Config(project="p", region="r", **params)
    return VertexClient(config, interrupt)


# ---------------------------------------------------------------------------
# 1. Authorization error (Req 2.5)
# ---------------------------------------------------------------------------


def test_authorization_error_surfaces(monkeypatch):
    interrupt = InterruptController()
    client = _make_client(interrupt)

    def fake_raw_stream(contents, tools):
        raise FakeApiError(403)

    monkeypatch.setattr(client, "_raw_stream", fake_raw_stream)

    with pytest.raises(AuthorizationError):
        list(client.generate_stream([], []))


# ---------------------------------------------------------------------------
# 2. Rate-limit retry exhaustion (Req 2.6)
# ---------------------------------------------------------------------------


def test_rate_limit_retry_exhaustion(monkeypatch):
    interrupt = InterruptController()
    client = _make_client(interrupt, rate_limit_retries=3)

    calls = {"n": 0}

    def fake_raw_stream(contents, tools):
        calls["n"] += 1
        raise FakeApiError(429)

    monkeypatch.setattr(client, "_raw_stream", fake_raw_stream)
    # Make backoff instantaneous and non-interrupting so the test is fast.
    monkeypatch.setattr(client, "_backoff", lambda attempt, retry_after=None: True)

    with pytest.raises(RateLimitError):
        list(client.generate_stream([], []))

    # One attempt per configured retry: nothing was ever emitted, so each
    # attempt re-runs until the attempts are exhausted.
    assert calls["n"] == 3


def test_rate_limit_eventually_succeeds(monkeypatch):
    """A rate limit that clears before exhaustion yields a normal stream."""
    interrupt = InterruptController()
    client = _make_client(interrupt, rate_limit_retries=3)

    calls = {"n": 0}

    def fake_raw_stream(contents, tools):
        calls["n"] += 1
        if calls["n"] < 2:
            raise FakeApiError(429)
        return iter([_text_chunk("ok")])

    monkeypatch.setattr(client, "_raw_stream", fake_raw_stream)
    monkeypatch.setattr(client, "_backoff", lambda attempt, retry_after=None: True)

    events = list(client.generate_stream([], []))

    assert calls["n"] == 2
    assert [e for e in events if isinstance(e, TextDelta)] == [TextDelta("ok")]
    assert isinstance(events[-1], Done)


# ---------------------------------------------------------------------------
# 3. Request timeout (Req 2.8)
# ---------------------------------------------------------------------------


def test_request_timeout(monkeypatch):
    interrupt = InterruptController()
    client = _make_client(interrupt, request_timeout_s=60)

    def fake_raw_stream(contents, tools):
        # An iterator that would yield chunks, but the wall-clock guard trips
        # before the first chunk is pulled.
        return iter([_text_chunk("never reached"), _text_chunk("nope")])

    monkeypatch.setattr(client, "_raw_stream", fake_raw_stream)

    # Force the wall-clock deadline to be exceeded: the first call (used to set
    # the deadline) returns 0.0; every subsequent call returns a value past the
    # deadline so the between-chunk guard raises.
    state = {"n": 0}

    def fake_monotonic():
        state["n"] += 1
        return 0.0 if state["n"] == 1 else 1_000_000.0

    monkeypatch.setattr(vertex.time, "monotonic", fake_monotonic)

    with pytest.raises(RequestTimeoutError):
        list(client.generate_stream([], []))


# ---------------------------------------------------------------------------
# 4. Mid-stream interruption (Req 3.4, 4.2)
# ---------------------------------------------------------------------------


def test_mid_stream_interrupt_retains_partial_and_aborts(monkeypatch):
    interrupt = InterruptController()
    interrupt.begin_turn()
    client = _make_client(interrupt)

    def fake_raw_stream(contents, tools):
        return iter([_text_chunk("first"), _text_chunk("second")])

    monkeypatch.setattr(client, "_raw_stream", fake_raw_stream)

    gen = client.generate_stream([], [])

    # Pull the first delta, then simulate a Ctrl-C between chunks.
    first = next(gen)
    assert first == TextDelta("first")

    interrupt.trip()

    # Generation must stop promptly without yielding the second delta or a
    # terminal Done; the already-yielded partial token is retained by us.
    start = time.monotonic()
    remaining = list(gen)
    elapsed = time.monotonic() - start

    assert remaining == []  # no further events, crucially no Done()
    assert not any(isinstance(e, Done) for e in remaining)
    assert elapsed < 1.0  # aborts within ~1 second (checks between chunks)


# ---------------------------------------------------------------------------
# 5. Mid-stream error (Req 3.4)
# ---------------------------------------------------------------------------


def test_mid_stream_error_surfaces_after_partial(monkeypatch):
    interrupt = InterruptController()
    client = _make_client(interrupt)

    def fake_raw_stream(contents, tools):
        def gen():
            yield _text_chunk("partial")
            raise RuntimeError("mid-stream boom")

        return gen()

    monkeypatch.setattr(client, "_raw_stream", fake_raw_stream)

    stream = client.generate_stream([], [])

    # The partial token is delivered before the error surfaces.
    first = next(stream)
    assert first == TextDelta("partial")

    # The mid-stream failure is translated to a typed VertexError; it is not
    # retried because an event was already emitted.
    with pytest.raises(VertexError):
        list(stream)


# ---------------------------------------------------------------------------
# 6. Rate-limit retry hint (Retry-After / RetryInfo) and backoff honoring
# ---------------------------------------------------------------------------


def test_retry_after_header_is_extracted_and_honored(monkeypatch):
    """A 429 with a Retry-After header surfaces on the error and reaches backoff."""
    interrupt = InterruptController()
    client = _make_client(interrupt, rate_limit_retries=2)

    class _Resp:
        headers = {"Retry-After": "7"}

    def fake_raw_stream(contents, tools):
        exc = FakeApiError(429)
        exc.response = _Resp()
        raise exc

    captured: dict = {}

    def fake_backoff(attempt, retry_after=None):
        captured["retry_after"] = retry_after
        return True  # don't actually wait; continue to next attempt

    monkeypatch.setattr(client, "_raw_stream", fake_raw_stream)
    monkeypatch.setattr(client, "_backoff", fake_backoff)

    with pytest.raises(RateLimitError):
        list(client.generate_stream([], []))

    # The server-advised delay was parsed from the header and passed to backoff.
    assert captured["retry_after"] == 7.0


def test_retry_info_detail_delay_is_extracted():
    """A google.rpc.RetryInfo-style 'retryDelay' in details is parsed to seconds."""
    exc = FakeApiError(429)
    exc.details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "12s"}]
    assert vertex._retry_after_seconds(exc) == 12.0


def test_no_retry_hint_returns_none():
    """A 429 with no header/detail/attribute yields no retry hint."""
    assert vertex._retry_after_seconds(FakeApiError(429)) is None


@pytest.mark.parametrize(
    "value,expected",
    [
        (7, 7.0),
        (7.5, 7.5),
        ("7s", 7.0),
        ("7.5s", 7.5),
        ("  10s ", 10.0),
        ({"seconds": 9}, 9.0),
        (SimpleNamespace(seconds=3, nanos=500_000_000), 3.5),
        (0, None),
        (-2, None),
        (True, None),
        ("nope", None),
        (None, None),
    ],
)
def test_coerce_delay_seconds_shapes(value, expected):
    assert vertex._coerce_delay_seconds(value) == expected


def test_backoff_honors_retry_after_minimum(monkeypatch):
    """The backoff wait is at least the server-advised retry_after."""
    interrupt = InterruptController()
    client = _make_client(interrupt)

    waited: dict = {}

    def fake_wait(timeout=None):
        waited["timeout"] = timeout
        return False  # not interrupted

    monkeypatch.setattr(interrupt.event, "wait", fake_wait)
    # Remove jitter randomness so the assertion is exact on the lower bound.
    monkeypatch.setattr(vertex.random, "uniform", lambda a, b: 0.0)

    # attempt=1 base delay is BACKOFF_BASE_S (1.0); retry_after=20 dominates.
    ok = client._backoff(1, retry_after=20.0)
    assert ok is True
    assert waited["timeout"] == 20.0


def test_backoff_caps_exponential_growth(monkeypatch):
    """Exponential growth is capped at BACKOFF_CAP_S (before jitter)."""
    interrupt = InterruptController()
    client = _make_client(interrupt)

    waited: dict = {}
    monkeypatch.setattr(
        interrupt.event, "wait",
        lambda timeout=None: waited.__setitem__("timeout", timeout) or False,
    )
    monkeypatch.setattr(vertex.random, "uniform", lambda a, b: 0.0)

    # A large attempt would explode without the cap; assert it is clamped.
    client._backoff(20)
    assert waited["timeout"] == vertex.BACKOFF_CAP_S
