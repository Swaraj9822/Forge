"""Base provider interface and shared data structures/exceptions for Forge."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterator, Protocol, Union

from forge.session import ToolCall
from forge.tools.base import ToolSpec

__all__ = [
    "TextDelta",
    "UsageReport",
    "Done",
    "StreamEvent",
    "Provider",
    "ProviderError",
    "CredentialsError",
    "ConfigMissingError",
    "AuthorizationError",
    "RateLimitError",
    "RequestTimeoutError",
    "BACKOFF_BASE_S",
    "BACKOFF_CAP_S",
    "BACKOFF_JITTER_FRAC",
    "wait_backoff",
]


@dataclass
class TextDelta:
    """A streamed fragment of model-generated text."""

    text: str


@dataclass
class UsageReport:
    """Per-response token usage."""

    input_tokens: int
    output_tokens: int


@dataclass
class Done:
    """Sentinel emitted when a response stream completes normally."""

    pass


StreamEvent = Union[TextDelta, ToolCall, UsageReport, Done]


class Provider(Protocol):
    """Protocol describing the model client surface that AgentLoop depends on."""

    def generate_stream(
        self, contents: list[dict], tools: list[ToolSpec]
    ) -> Iterator[StreamEvent]:
        """Stream a model response as StreamEvent values."""
        ...


# ---------------------------------------------------------------------------
# Typed exceptions
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base class for all errors surfaced by a Provider."""


class CredentialsError(ProviderError):
    """Credentials / API keys are missing or invalid."""

    DEFAULT_MESSAGE = (
        "Application Default Credentials (ADC) are unavailable. Establish them by "
        "running: gcloud auth application-default login"
    )

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE)


class ConfigMissingError(ProviderError):
    """Required configuration values (e.g. project, region) are missing."""


class AuthorizationError(ProviderError):
    """Provider rejected the request with an authorization/permission error."""


class RateLimitError(ProviderError):
    """Rate limit exceeded."""

    def __init__(
        self, message: str | None = None, *, retry_after: float | None = None
    ) -> None:
        super().__init__(message or "Rate limit exceeded.")
        self.retry_after = retry_after


class RequestTimeoutError(ProviderError):
    """Request timeout exceeded."""


# ---------------------------------------------------------------------------
# Rate-limit backoff tuning
# ---------------------------------------------------------------------------

BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0
BACKOFF_JITTER_FRAC = 0.25


def wait_backoff(attempt: int, interrupt: Any, retry_after: float | None = None) -> bool:
    """Wait before a rate-limit retry; return False if interrupted.

    Uses a capped exponential schedule with jitter.
    """
    capped = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_CAP_S)
    delay = capped + random.uniform(0.0, capped * BACKOFF_JITTER_FRAC)
    if retry_after is not None and retry_after > 0:
        delay = max(delay, float(retry_after))
    tripped = interrupt.event.wait(timeout=delay)
    return not tripped
