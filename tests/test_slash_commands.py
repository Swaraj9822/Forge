"""Tests for custom slash command store and rendering."""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge.commands import SlashCommandStore


def test_slash_commands_store() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()

        # Create directories
        dir1 = root / "commands1"
        dir2 = root / "commands2"
        dir1.mkdir()
        dir2.mkdir()

        # Create some command templates
        (dir1 / "cmd1.md").write_text(
            "Hello $ARGUMENTS. $1 is nice. $2 is ok.", encoding="utf-8"
        )
        (dir2 / "cmd2.md").write_text("Command 2 body.", encoding="utf-8")
        # Override cmd1 in dir2 (should use dir1 first since it's listed first)
        (dir2 / "cmd1.md").write_text("Wrong cmd1.", encoding="utf-8")

        store = SlashCommandStore([dir1, dir2])

        # Listing names
        assert store.names() == ["cmd1", "cmd2"]

        # Rendering cmd1
        rendered = store.render("cmd1", "Alice Bob")
        assert rendered == "Hello Alice Bob. Alice is nice. Bob is ok."

        # Rendering cmd2
        rendered_cmd2 = store.render("cmd2", "some args")
        assert rendered_cmd2 == "Command 2 body."

        # Unknown command
        assert store.render("unknown", "args") is None

        # Partial arguments replacement (missing N replacement should be empty string)
        rendered_missing = store.render("cmd1", "Alice")
        assert rendered_missing == "Hello Alice. Alice is nice.  is ok."
