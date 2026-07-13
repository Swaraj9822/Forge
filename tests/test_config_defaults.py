"""Property-based test for configuration defaults merging.

# Feature: forge, Property 18: Config defaults merge
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import tomli_w
from hypothesis import given, settings
from hypothesis import strategies as st

from forge.config import (
    Config,
    ConfigManager,
    DEFAULT_ENABLED_TOOLS,
    DEFAULT_LIMITS,
    DEFAULT_MODEL,
    RECOGNIZED_TOOLS,
)

# Printable ASCII text keeps generated values valid for TOML round-tripping
# (no surrogates / control characters that would break encoding).
SAFE_TEXT = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=20,
)

LIMIT_VALUES = st.integers(min_value=1, max_value=10_000_000)
PRICE_VALUES = st.floats(
    min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)


@st.composite
def partial_configs(draw: st.DrawFn) -> dict:
    """Build a partial config mapping: each setting is independently present
    (with a random valid value) or omitted, mirroring the documented TOML
    structure (limits under [limits], pricing under [pricing])."""

    document: dict = {}

    # Top-level scalars and lists.
    if draw(st.booleans()):
        document["model"] = draw(SAFE_TEXT)
    if draw(st.booleans()):
        document["project"] = draw(SAFE_TEXT)
    if draw(st.booleans()):
        document["region"] = draw(SAFE_TEXT)
    if draw(st.booleans()):
        # Only recognized tool names so the unknown-tool warning path
        # (a separate concern) does not interfere with this property.
        document["enabled_tools"] = draw(
            st.lists(st.sampled_from(RECOGNIZED_TOOLS), unique=True, max_size=len(RECOGNIZED_TOOLS))
        )
    if draw(st.booleans()):
        document["steering_files"] = draw(st.lists(SAFE_TEXT, max_size=5))

    # Numeric limits, each independently present or omitted, nested under [limits].
    limits: dict = {}
    for key in DEFAULT_LIMITS:
        if draw(st.booleans()):
            limits[key] = draw(LIMIT_VALUES)
    if limits:
        document["limits"] = limits

    # Pricing fields, each independently present or omitted, under [pricing].
    pricing: dict = {}
    if draw(st.booleans()):
        pricing["input_per_1k"] = draw(PRICE_VALUES)
    if draw(st.booleans()):
        pricing["output_per_1k"] = draw(PRICE_VALUES)
    if pricing:
        document["pricing"] = pricing

    # Context flags, each independently present or omitted, under [context].
    context: dict = {}
    if draw(st.booleans()):
        context["plan_reminder"] = draw(st.booleans())
    if draw(st.booleans()):
        context["project_memory"] = draw(st.booleans())
    if context:
        document["context"] = context

    return document


@settings(max_examples=10)
@given(partial_configs())
def test_config_defaults_merge(document: dict) -> None:
    """For any partial config, omitted settings equal documented defaults and
    present settings are preserved exactly.

    Validates: Requirements 11.4
    """

    defaults = Config()  # all documented defaults

    # Write the partial document to a temp TOML file (created/cleaned up per
    # example to stay compatible with @given).
    fd, name = tempfile.mkstemp(suffix=".toml")
    os.close(fd)
    path = Path(name)
    try:
        with open(path, "wb") as fh:
            tomli_w.dump(document, fh)
        config = ConfigManager().load(path=path)
    finally:
        path.unlink(missing_ok=True)

    limits = document.get("limits", {})
    pricing = document.get("pricing", {})

    # --- Top-level scalars and lists ---
    if "model" in document:
        assert config.model == document["model"]
    else:
        assert config.model == DEFAULT_MODEL == defaults.model

    if "project" in document:
        assert config.project == document["project"]
    else:
        assert config.project is None and defaults.project is None

    if "region" in document:
        assert config.region == document["region"]
    else:
        assert config.region is None

    if "enabled_tools" in document:
        assert config.enabled_tools == document["enabled_tools"]
    else:
        assert config.enabled_tools == DEFAULT_ENABLED_TOOLS

    if "steering_files" in document:
        assert config.steering_files == document["steering_files"]
    else:
        assert config.steering_files == defaults.steering_files == []

    # --- Numeric limits (each independent) ---
    for key, default_value in DEFAULT_LIMITS.items():
        actual = getattr(config, key)
        if key in limits:
            assert actual == limits[key]
        else:
            assert actual == default_value == getattr(defaults, key)

    # --- Pricing fields (each independent) ---
    if "input_per_1k" in pricing:
        assert config.pricing.input_per_1k == pricing["input_per_1k"]
    else:
        assert config.pricing.input_per_1k is None

    if "output_per_1k" in pricing:
        assert config.pricing.output_per_1k == pricing["output_per_1k"]
    else:
        assert config.pricing.output_per_1k is None

    # --- Context flags (each independent) ---
    context = document.get("context", {})
    if "plan_reminder" in context:
        assert config.plan_reminder == bool(context["plan_reminder"])
    else:
        assert config.plan_reminder is True

    if "project_memory" in context:
        assert config.project_memory == bool(context["project_memory"])
    else:
        assert config.project_memory is True

    # --- Policy + checkpoint (Phase 2): new fields use their dataclass
    # defaults when the partial document does not include the tables. ---
    assert config.policy_mode == "autopilot"
    assert config.shell_allowlist == ()
    assert config.show_diffs is False
    assert config.checkpoint_enabled is True
    assert config.checkpoint_keep_turns == 10

    # --- Phase 4 Ergonomics & UX defaults ---
    assert config.ui_color is False
    assert config.ui_spinner is False
    assert config.commands_dir == ".forge/commands"
    assert config.parallel_enabled is False
    assert config.parallel_max_workers == 4
    assert config.mentions_enabled is False
