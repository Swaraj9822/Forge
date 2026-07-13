"""Unit tests for the [context] configuration table."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import tomli_w

from forge.config import ConfigManager


def test_context_unknown_keys_are_ignored() -> None:
    """Unrecognized keys under [context] are silently ignored."""
    fd, name = tempfile.mkstemp(suffix=".toml")
    os.close(fd)
    path = Path(name)
    try:
        with open(path, "wb") as fh:
            tomli_w.dump(
                {
                    "context": {
                        "plan_reminder": True,
                        "project_memory": False,
                        "unknown_key": "ignored",
                    }
                },
                fh,
            )
        config = ConfigManager().load(path=path)
        assert config.plan_reminder is True
        assert config.project_memory is False
    finally:
        path.unlink(missing_ok=True)


def test_context_values_are_coerced_to_bool() -> None:
    """Non-bool values under [context] are coerced via bool()."""
    fd, name = tempfile.mkstemp(suffix=".toml")
    os.close(fd)
    path = Path(name)
    try:
        with open(path, "wb") as fh:
            tomli_w.dump(
                {
                    "context": {
                        "plan_reminder": "yes",
                        "project_memory": 0,
                    }
                },
                fh,
            )
        config = ConfigManager().load(path=path)
        assert config.plan_reminder is True
        assert config.project_memory is False
    finally:
        path.unlink(missing_ok=True)


def test_context_false_plan_reminder_round_trips() -> None:
    """Explicitly disabling plan_reminder is preserved on load."""
    fd, name = tempfile.mkstemp(suffix=".toml")
    os.close(fd)
    path = Path(name)
    try:
        with open(path, "wb") as fh:
            tomli_w.dump({"context": {"plan_reminder": False}}, fh)
        config = ConfigManager().load(path=path)
        assert config.plan_reminder is False
        assert config.project_memory is True
    finally:
        path.unlink(missing_ok=True)
