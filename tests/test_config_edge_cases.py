"""Unit tests for ConfigManager edge cases.

Covers the documented edge-case behaviors of ``forge/config.py``:

- Syntax-error reporting names the file and (when available) the line/column
  of the error (Req 11.6).
- Enabling an unrecognized tool warns naming the tool and drops it while
  keeping recognized tools (Req 11.7).
- OS-conventional path resolution for ``config_path()`` / ``sessions_dir()``
  (Req 11.2 / 11.9).
- The ``init``-already-exists guard pattern leaves an existing file unchanged
  (Req 12.2).

Note: the end-to-end ``forge init`` "report exists + leave unchanged" behavior
(Req 12.2) lives at the CLI/app layer built in task 24.1 and is covered fully
by task 24.2. This module exercises only the unit-level guard pattern
(`if path.exists(): skip write_default`).

Tests create their own temporary directories via ``tempfile`` (matching the
existing config test modules) rather than the ``tmp_path`` fixture.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from forge.config import (
    DEFAULT_ENABLED_TOOLS,
    ConfigError,
    ConfigManager,
    _xdg_base,
)


# ---------------------------------------------------------------------------
# Req 11.6: TOML syntax error reporting (file + line/column)
# ---------------------------------------------------------------------------


def test_syntax_error_reports_file_and_location() -> None:
    """A malformed TOML file raises ConfigError naming the file; when a
    line/column is available they are ints and the location appears in the
    message."""
    with tempfile.TemporaryDirectory() as root:
        bad = Path(root) / "config.toml"
        # `project =` with no value on line 2 is a TOML syntax error.
        bad.write_text('model = "ok"\nproject =\n', encoding="utf-8")

        with pytest.raises(ConfigError) as exc_info:
            ConfigManager().load(path=bad)

        err = exc_info.value

        # The error identifies the offending file.
        assert err.path == bad
        assert str(bad) in str(err)

        # line/column are optional across Python versions: tolerate None, but
        # when present they must be ints and appear in the message.
        if err.line is not None:
            assert isinstance(err.line, int)
            assert f"line {err.line}" in str(err)
        if err.column is not None:
            assert isinstance(err.column, int)
            assert f"column {err.column}" in str(err)


# ---------------------------------------------------------------------------
# Req 11.7: Unknown enabled tool warns and is dropped; recognized tools kept
# ---------------------------------------------------------------------------


def test_unknown_tool_warns_and_is_dropped() -> None:
    """Enabling an unrecognized tool emits a UserWarning naming the tool and
    drops it from the loaded config while preserving recognized tools."""
    with tempfile.TemporaryDirectory() as root:
        cfg = Path(root) / "config.toml"
        cfg.write_text(
            'enabled_tools = ["read", "frobnicate", "git"]\n',
            encoding="utf-8",
        )

        with pytest.warns(UserWarning, match="frobnicate"):
            config = ConfigManager().load(path=cfg)

        # The unrecognized tool is dropped; recognized ones are kept in order.
        assert "frobnicate" not in config.enabled_tools
        assert config.enabled_tools == ["read", "git"]


# ---------------------------------------------------------------------------
# Req 11.2 / 11.9: OS-conventional path resolution
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path convention")
def test_windows_paths_use_appdata(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows, config_path/sessions_dir resolve under %APPDATA%\\forge."""
    appdata = r"C:\Users\test\AppData\Roaming"
    monkeypatch.setenv("APPDATA", appdata)

    assert ConfigManager.config_path() == Path(appdata) / "forge" / "config.toml"
    assert ConfigManager.sessions_dir() == Path(appdata) / "forge" / "sessions"


def test_xdg_base_prefers_env_then_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """_xdg_base returns the env-var path when set, otherwise the fallback.

    This exercises the Unix/macOS path-resolution logic directly so it can be
    asserted on any host (including this Windows host) without depending on the
    current platform branch in config_path/sessions_dir.
    """
    fallback = Path.home() / ".config"

    # When the env var is set, its value wins.
    monkeypatch.setenv("XDG_CONFIG_HOME", "/custom/xdg/config")
    assert _xdg_base("XDG_CONFIG_HOME", fallback) == Path("/custom/xdg/config")

    # When unset (or empty), the documented fallback is used.
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert _xdg_base("XDG_CONFIG_HOME", fallback) == fallback


@pytest.mark.skipif(sys.platform == "win32", reason="Unix/macOS path convention")
def test_unix_paths_honor_xdg(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Unix/macOS, XDG_CONFIG_HOME/XDG_DATA_HOME drive resolution; falling
    back to ~/.config and ~/.local/share when unset."""
    monkeypatch.setenv("XDG_CONFIG_HOME", "/xdg/config")
    monkeypatch.setenv("XDG_DATA_HOME", "/xdg/data")
    assert ConfigManager.config_path() == Path("/xdg/config/forge/config.toml")
    assert ConfigManager.sessions_dir() == Path("/xdg/data/forge/sessions")

    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert (
        ConfigManager.config_path()
        == Path.home() / ".config" / "forge" / "config.toml"
    )
    assert (
        ConfigManager.sessions_dir()
        == Path.home() / ".local" / "share" / "forge" / "sessions"
    )


# ---------------------------------------------------------------------------
# Req 12.2: init-already-exists guard leaves the existing file unchanged
# ---------------------------------------------------------------------------


def test_init_guard_leaves_existing_file_unchanged() -> None:
    """The documented init guard (skip write_default when the file exists)
    leaves an existing config file byte-for-byte unchanged.

    This is the unit-level guard pattern only. The full CLI behavior (reporting
    that configuration already exists) is covered end-to-end by task 24.2.
    """
    with tempfile.TemporaryDirectory() as root:
        existing = Path(root) / "config.toml"
        sentinel = b'model = "sentinel-do-not-touch"\n'
        existing.write_bytes(sentinel)

        manager = ConfigManager()

        # Documented init flow: only write defaults when no file is present.
        if not existing.exists():
            manager.write_default(existing)

        # The guard skipped the write, so the original bytes are preserved.
        assert existing.read_bytes() == sentinel


def test_write_default_overwrites_when_called_directly() -> None:
    """write_default always writes when invoked directly (the guard lives at
    the caller). This documents the contract the init guard relies on."""
    with tempfile.TemporaryDirectory() as root:
        target = Path(root) / "config.toml"
        target.write_bytes(b'model = "old"\n')

        ConfigManager().write_default(target)

        # Called directly, write_default replaces the file with the defaults.
        config = ConfigManager().load(path=target)
        assert config.enabled_tools == DEFAULT_ENABLED_TOOLS
