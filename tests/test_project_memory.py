"""Tests for project memory file auto-loading (FORGE.md / AGENTS.md)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from forge.config import Config
from forge.context import ContextManager, load_default_system_prompt


def test_project_file_loaded_after_steering() -> None:
    """A FORGE.md in the workspace is appended after configured steering files."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="forge_memory_"))
    try:
        steering = tmp_dir / "steering.md"
        steering.write_text("[steering content]", encoding="utf-8")
        project_file = tmp_dir / "FORGE.md"
        project_file.write_text("[project memory content]", encoding="utf-8")

        config = Config(steering_files=[str(steering)])
        manager = ContextManager(
            config,
            workspace_root=tmp_dir,
            project_memory_filenames=("FORGE.md", "AGENTS.md"),
        )

        messages = manager.assemble_system_messages()
        contents = [m["content"] for m in messages]

        assert contents[0] == load_default_system_prompt()
        assert contents[1] == "[steering content]"
        assert contents[2] == "[project memory content]"
        assert manager.build_system_prompt().endswith("[project memory content]")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_forge_md_precedence_over_agents_md() -> None:
    """When both files exist, FORGE.md wins and AGENTS.md is ignored."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="forge_memory_"))
    try:
        (tmp_dir / "FORGE.md").write_text("FORGE content", encoding="utf-8")
        (tmp_dir / "AGENTS.md").write_text("AGENTS content", encoding="utf-8")

        manager = ContextManager(
            Config(),
            workspace_root=tmp_dir,
            project_memory_filenames=("FORGE.md", "AGENTS.md"),
        )

        prompt = manager.build_system_prompt()
        assert "FORGE content" in prompt
        assert "AGENTS content" not in prompt
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_no_project_file_is_noop() -> None:
    """An empty workspace yields the same prompt as workspace_root=None."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="forge_memory_"))
    try:
        manager_with_root = ContextManager(
            Config(),
            workspace_root=tmp_dir,
            project_memory_filenames=("FORGE.md", "AGENTS.md"),
        )
        manager_without_root = ContextManager(Config())

        assert manager_with_root.build_system_prompt() == manager_without_root.build_system_prompt()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_unreadable_project_file_warns_and_skips() -> None:
    """Invalid UTF-8 in a project file triggers a warning and is skipped."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="forge_memory_"))
    try:
        project_file = tmp_dir / "FORGE.md"
        project_file.write_bytes(b"\xff\xfe invalid utf-8")

        manager = ContextManager(
            Config(),
            workspace_root=tmp_dir,
            project_memory_filenames=("FORGE.md", "AGENTS.md"),
        )

        with pytest.warns(UserWarning, match="Project memory file could not be read"):
            prompt = manager.build_system_prompt()

        # The unreadable bytes must not have leaked into the prompt; the
        # prompt should be identical to the default-prompt-only result.
        assert b"\xff\xfe" not in prompt.encode("utf-8")
        assert prompt == ContextManager(Config()).build_system_prompt()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
