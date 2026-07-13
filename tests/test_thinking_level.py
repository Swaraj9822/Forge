"""Tests for the Gemini `thinking_level` control and thinking-token accounting.

Covers:
- config parsing/validation of `provider.thinking_level`,
- the Vertex provider wiring `thinking_level` into the request config,
- `_parse_chunk` folding `thoughts_token_count` into the output token tally so
  a thinking model's usage and cost are accurate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forge.config import Config, ConfigError, ConfigManager
from forge.interrupt import InterruptController
from forge.providers.base import UsageReport
from forge.providers.vertex import VertexProvider, _parse_chunk


# --------------------------------------------------------------------------- #
# Config parsing / validation
# --------------------------------------------------------------------------- #


def _load_raw(raw: dict) -> Config:
    return ConfigManager()._from_raw(raw)


def test_thinking_level_absent_defaults_to_none():
    cfg = _load_raw({})
    assert cfg.provider_thinking_level is None


@pytest.mark.parametrize("level", ["minimal", "low", "medium", "high"])
def test_thinking_level_valid_values(level):
    cfg = _load_raw({"provider": {"thinking_level": level}})
    assert cfg.provider_thinking_level == level


def test_thinking_level_is_normalized_to_lowercase():
    cfg = _load_raw({"provider": {"thinking_level": "  HIGH  "}})
    assert cfg.provider_thinking_level == "high"


def test_thinking_level_invalid_value_raises():
    with pytest.raises(ConfigError) as excinfo:
        _load_raw({"provider": {"thinking_level": "ultra"}})
    assert "thinking_level" in str(excinfo.value)


def test_thinking_level_non_string_raises():
    with pytest.raises(ConfigError):
        _load_raw({"provider": {"thinking_level": 5}})


# --------------------------------------------------------------------------- #
# Vertex request config wiring
# --------------------------------------------------------------------------- #


def _provider(**cfg_kwargs) -> VertexProvider:
    return VertexProvider(Config(**cfg_kwargs), InterruptController())


def test_request_config_omits_thinking_when_unset():
    provider = _provider()
    cfg = provider._build_request_config(tools=[], system_instruction=None)
    # No tools, no system instruction, no thinking level -> nothing to send.
    assert cfg is None


def test_request_config_sets_thinking_level():
    provider = _provider(provider_thinking_level="high")
    cfg = provider._build_request_config(tools=[], system_instruction="be terse")
    assert cfg is not None
    assert cfg.thinking_config is not None
    # The SDK normalizes the string into a ThinkingLevel enum whose value/name
    # reflects the configured level.
    assert "high" in str(cfg.thinking_config.thinking_level).lower()


def test_build_thinking_config_none_when_unset():
    assert _provider()._build_thinking_config() is None


# --------------------------------------------------------------------------- #
# Thinking-token accounting
# --------------------------------------------------------------------------- #


def _chunk_with_usage(prompt, candidates, thoughts):
    usage = SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
    )
    return SimpleNamespace(candidates=[], usage_metadata=usage)


def test_parse_chunk_folds_thoughts_into_output():
    chunk = _chunk_with_usage(prompt=100, candidates=40, thoughts=250)
    events = _parse_chunk(chunk)
    reports = [e for e in events if isinstance(e, UsageReport)]
    assert len(reports) == 1
    assert reports[0].input_tokens == 100
    # 40 visible output + 250 thinking tokens = 290 billed output tokens.
    assert reports[0].output_tokens == 290


def test_parse_chunk_without_thoughts_is_unchanged():
    chunk = _chunk_with_usage(prompt=100, candidates=40, thoughts=None)
    events = _parse_chunk(chunk)
    reports = [e for e in events if isinstance(e, UsageReport)]
    assert len(reports) == 1
    assert reports[0].output_tokens == 40


def test_parse_chunk_missing_thoughts_field_is_safe():
    # An older usage payload without a thoughts_token_count attribute at all.
    usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5)
    chunk = SimpleNamespace(candidates=[], usage_metadata=usage)
    reports = [e for e in _parse_chunk(chunk) if isinstance(e, UsageReport)]
    assert reports[0].output_tokens == 5
