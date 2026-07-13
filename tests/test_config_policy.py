"""Tests for the [policy] and [checkpoint] config tables (Phase 2)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from forge.config import (
    POLICY_MODES,
    ConfigError,
    ConfigManager,
    resolve_policy_config,
)


# --------------------------------------------------------------------------- #
# resolve_policy_config pure helper
# --------------------------------------------------------------------------- #


def test_default_mode_is_autopilot() -> None:
    mode, allowlist, show_diffs = resolve_policy_config(None)
    assert mode == "autopilot"
    assert allowlist == ()
    assert show_diffs is False


def test_empty_mapping_uses_defaults() -> None:
    mode, allowlist, show_diffs = resolve_policy_config({})
    assert mode == "autopilot"
    assert allowlist == ()
    assert show_diffs is False


@pytest.mark.parametrize("mode", list(POLICY_MODES))
def test_known_modes_are_accepted(mode: str) -> None:
    resolved, _, _ = resolve_policy_config({"mode": mode})
    assert resolved == mode


def test_unknown_mode_raises_config_error() -> None:
    with pytest.raises(ConfigError) as exc:
        resolve_policy_config({"mode": "dangerous"})
    assert "policy.mode" in str(exc.value)
    assert "dangerous" in str(exc.value)


def test_allowlist_is_normalized_to_tuple_of_strings() -> None:
    _, allowlist, _ = resolve_policy_config(
        {"shell_allowlist": ["pytest", "git"]}
    )
    assert allowlist == ("pytest", "git")


def test_non_string_allowlist_entries_are_dropped() -> None:
    _, allowlist, _ = resolve_policy_config(
        {"shell_allowlist": ["pytest", 42, None, "git"]}
    )
    assert allowlist == ("pytest", "git")


def test_non_list_allowlist_raises_config_error() -> None:
    with pytest.raises(ConfigError) as exc:
        resolve_policy_config({"shell_allowlist": "pytest"})
    assert "shell_allowlist" in str(exc.value)


def test_show_diffs_flag_is_coerced_to_bool() -> None:
    _, _, show = resolve_policy_config({"show_diffs": True})
    assert show is True
    _, _, show = resolve_policy_config({"show_diffs": False})
    assert show is False
    _, _, show = resolve_policy_config({"show_diffs": 0})
    assert show is False


# --------------------------------------------------------------------------- #
# End-to-end: ConfigManager round-trip
# --------------------------------------------------------------------------- #


def test_default_config_has_autopilot_mode() -> None:
    """An absent [policy] table yields the dataclass default of autopilot."""

    manager = ConfigManager()
    config = manager.load(path=Path("/nonexistent/path/to/config.toml"))
    assert config.policy_mode == "autopilot"
    assert config.shell_allowlist == ()
    assert config.show_diffs is False


def test_load_supervised_mode_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[policy]\n'
        'mode = "supervised"\n'
        'shell_allowlist = ["pytest", "git"]\n'
        'show_diffs = true\n',
        encoding="utf-8",
    )
    config = ConfigManager().load(path)
    assert config.policy_mode == "supervised"
    assert config.shell_allowlist == ("pytest", "git")
    assert config.show_diffs is True


def test_load_readonly_mode_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[policy]\nmode = "readonly"\n', encoding="utf-8")
    config = ConfigManager().load(path)
    assert config.policy_mode == "readonly"


def test_unknown_mode_in_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[policy]\nmode = "yolo"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        ConfigManager().load(path)
    assert "policy.mode" in str(exc.value)


def test_checkpoint_table_defaults() -> None:
    """An absent [checkpoint] table yields the documented defaults."""

    config = ConfigManager().load(path=Path("/nonexistent"))
    assert config.checkpoint_enabled is True
    assert config.checkpoint_keep_turns == 10


def test_checkpoint_table_loaded(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[checkpoint]\n"
        "enabled = false\n"
        "keep_turns = 5\n",
        encoding="utf-8",
    )
    config = ConfigManager().load(path)
    assert config.checkpoint_enabled is False
    assert config.checkpoint_keep_turns == 5


def test_checkpoint_keep_turns_must_be_non_negative_int(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[checkpoint]\nkeep_turns = -1\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        ConfigManager().load(path)
    assert "checkpoint.keep_turns" in str(exc.value)


def test_write_default_emits_supervised_mode(tmp_path: Path) -> None:
    """A fresh `forge init` config opts the user into supervised mode."""

    path = tmp_path / "config.toml"
    ConfigManager().write_default(path)
    # Re-read via raw TOML so we see what was actually written.
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    assert raw["policy"]["mode"] == "supervised"
    assert raw["policy"]["show_diffs"] is True
    assert isinstance(raw["policy"]["shell_allowlist"], list)
    assert raw["checkpoint"]["enabled"] is True
    assert raw["checkpoint"]["keep_turns"] == 10


def test_write_default_round_trips(tmp_path: Path) -> None:
    """A written config loads back into an equal Config object."""

    path = tmp_path / "config.toml"
    ConfigManager().write_default(path)
    config = ConfigManager().load(path)
    assert config.policy_mode == "supervised"
    assert config.checkpoint_enabled is True
    assert config.checkpoint_keep_turns == 10
    assert "pytest" in config.shell_allowlist
    assert config.show_diffs is True
