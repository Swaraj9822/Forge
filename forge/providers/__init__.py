"""Multi-provider client layer for Forge.

Provides the Provider factory build_provider, protocol definition, shared
exception hierarchy, and concrete implementations (Vertex, Anthropic, OpenAI).
"""

from __future__ import annotations

from forge.config import Config, ConfigError, ConfigManager
from forge.interrupt import InterruptController

from forge.providers.base import (
    TextDelta,
    UsageReport,
    Done,
    StreamEvent,
    Provider,
    ProviderError,
    CredentialsError,
    ConfigMissingError,
    AuthorizationError,
    RateLimitError,
    RequestTimeoutError,
    wait_backoff,
)
from forge.providers.vertex import VertexProvider
from forge.providers.anthropic import AnthropicProvider
from forge.providers.openai import OpenAIProvider

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
    "wait_backoff",
    "VertexProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "build_provider",
]


def build_provider(config: Config, interrupt: InterruptController) -> Provider:
    """Build a Provider instance matching config.provider_type (Req 2.4)."""
    t = config.provider_type
    if t == "vertex":
        return VertexProvider(config, interrupt)
    if t == "anthropic":
        return AnthropicProvider(config, interrupt)
    if t == "openai":
        return OpenAIProvider(config, interrupt)

    raise ConfigError(
        ConfigManager.config_path(),
        detail=f"unknown provider type {t!r}",
    )
