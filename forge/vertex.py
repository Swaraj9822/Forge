"""Shim for backwards compatibility with the original VertexClient interface.

This module re-exports types and the Vertex provider class from their new
locations in the ``forge.providers`` package so that existing imports, wiring,
and offline tests continue to work unchanged.
"""

from __future__ import annotations

import random  # Exposed for tests that monkeypatch vertex.random
import time  # Exposed for tests that monkeypatch vertex.time.monotonic

from forge.providers.base import (
    TextDelta,
    UsageReport,
    Done,
    StreamEvent,
    ProviderError as VertexError,
    CredentialsError,
    ConfigMissingError,
    AuthorizationError,
    RateLimitError,
    RequestTimeoutError,
    BACKOFF_CAP_S,
)
from forge.session import ToolCall
from forge.providers.vertex import (
    VertexProvider as VertexClient,
    _to_sdk_contents,
    _to_sdk_tools,
    _coerce_delay_seconds,
    _retry_after_seconds,
)

__all__ = [
    "TextDelta",
    "ToolCall",
    "UsageReport",
    "Done",
    "StreamEvent",
    "VertexError",
    "CredentialsError",
    "ConfigMissingError",
    "AuthorizationError",
    "RateLimitError",
    "RequestTimeoutError",
    "VertexClient",
    "_to_sdk_contents",
    "_to_sdk_tools",
    "_coerce_delay_seconds",
    "_retry_after_seconds",
    "BACKOFF_CAP_S",
]
