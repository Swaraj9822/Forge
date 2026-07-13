"""Property-based test for Property 19: Init config round-trip.

# Feature: forge, Property 19: Init config round-trip
Validates: Requirements 12.1

Asserts that the file written by ``ConfigManager.write_default`` (the structure
emitted by ``forge init``) parses as valid TOML and loads back through
``ConfigManager.load`` to a Config equal to the documented defaults, with the
required ``project``/``region`` placeholders present.
"""

from __future__ import annotations

import tempfile
import tomllib
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.config import (
    DEFAULT_ENABLED_TOOLS,
    DEFAULT_INIT_PRICING,
    DEFAULT_LIMITS,
    DEFAULT_MODEL,
    PROJECT_PLACEHOLDER,
    REGION_PLACEHOLDER,
    ConfigManager,
)

# Windows reserved device names cannot be used as a directory or file component.
# Creating a path with any of these (case-insensitive, with or without an
# extension) raises OSError/FileNotFoundError on Windows, which is unrelated to
# the property under test. Exclude them defensively from generated components.
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def _is_filesystem_safe(name: str) -> bool:
    """True when ``name`` is not a Windows reserved device name."""
    # Reserved-name matching ignores any extension and is case-insensitive.
    stem = name.split(".", 1)[0].lower()
    return stem not in _RESERVED_NAMES


# Generate fresh, filesystem-safe subdirectory and filename components so each
# example exercises "any fresh environment" with a distinct target path. The
# alphabet is restricted to lowercase ASCII letters (so digit-bearing reserved
# names like COM1/LPT1 cannot occur), and reserved device names are filtered out.
_safe_names = st.text(
    alphabet=st.characters(whitelist_categories=("Ll",), max_codepoint=127),
    min_size=1,
    max_size=20,
).filter(_is_filesystem_safe)


@settings(max_examples=10, deadline=None)
@given(subdir=_safe_names, filename=_safe_names)
# Feature: forge, Property 19: Init config round-trip
def test_init_config_round_trip(subdir: str, filename: str) -> None:
    """write_default output parses as TOML and loads back to documented defaults."""
    with tempfile.TemporaryDirectory() as root:
        target = Path(root) / subdir / f"{filename}.toml"

        manager = ConfigManager()
        manager.write_default(target)

        # (a) The written file parses as valid TOML.
        assert target.exists()
        with open(target, "rb") as fh:
            parsed = tomllib.load(fh)
        assert isinstance(parsed, dict)

        # The required placeholders are present in the written document.
        assert parsed["project"] == PROJECT_PLACEHOLDER
        assert parsed["region"] == REGION_PLACEHOLDER

        # (b) Loading it back yields a Config equal to the documented defaults.
        config = manager.load(target)

        assert config.model == DEFAULT_MODEL
        assert config.enabled_tools == DEFAULT_ENABLED_TOOLS

        for key, default_value in DEFAULT_LIMITS.items():
            assert getattr(config, key) == default_value

        assert config.pricing.input_per_1k == DEFAULT_INIT_PRICING["input_per_1k"]
        assert config.pricing.output_per_1k == DEFAULT_INIT_PRICING["output_per_1k"]

        # The required project/region placeholders survive the round-trip.
        assert config.project == PROJECT_PLACEHOLDER
        assert config.region == REGION_PLACEHOLDER
