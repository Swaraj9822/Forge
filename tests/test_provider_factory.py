"""Tests for the provider factory build_provider."""

from __future__ import annotations

import pytest

from forge.config import Config, ConfigError
from forge.interrupt import InterruptController
from forge.providers import (
    build_provider,
    VertexProvider,
    AnthropicProvider,
    OpenAIProvider,
)


def test_build_provider_vertex() -> None:
    config = Config(project="p", region="r", provider_type="vertex")
    interrupt = InterruptController()
    provider = build_provider(config, interrupt)
    assert isinstance(provider, VertexProvider)


def test_build_provider_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    config = Config(provider_type="anthropic")
    interrupt = InterruptController()
    provider = build_provider(config, interrupt)
    assert isinstance(provider, AnthropicProvider)


def test_build_provider_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    config = Config(provider_type="openai")
    interrupt = InterruptController()
    provider = build_provider(config, interrupt)
    assert isinstance(provider, OpenAIProvider)


def test_build_provider_unknown() -> None:
    # Use config bypass to set an invalid type since it validates in Config loading,
    # but build_provider itself also validates it.
    config = Config(provider_type="unknown")
    interrupt = InterruptController()
    with pytest.raises(ConfigError) as exc_info:
        build_provider(config, interrupt)
    assert "unknown provider type" in exc_info.value.detail
