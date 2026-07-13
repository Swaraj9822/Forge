"""Unit tests for the git tool's runtime behavior (task 12.4).

These tests exercise :class:`forge.tools.git.GitTool` directly (calling ``run``)
against the real ``git`` binary, covering:

* happy path -- a supported operation runs in the Workspace repository and the
  command output is returned (Req 9.2);
* not-a-repo -- invoking the tool where the Workspace is not a git repository
  returns a "not a git repository" result (Req 9.3);
* non-zero exit -- a supported operation that exits non-zero surfaces the exit
  code and the captured error output (Req 9.5).

A real temporary git repository is created per test via ``git init`` (plus a
local ``user.name``/``user.email`` and one commit). When ``git`` is not
available on the host these tests skip gracefully rather than fail.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge.interrupt import InterruptController
from forge.tools.base import ToolContext
from forge.tools.git import GitTool


def _git_available() -> bool:
    """Return whether a usable ``git`` binary is on PATH."""
    try:
        proc = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return False
    return proc.returncode == 0


GIT_AVAILABLE = _git_available()

pytestmark = pytest.mark.skipif(
    not GIT_AVAILABLE, reason="git binary not available on PATH"
)


def _ctx(workspace: Path, **config_kwargs) -> ToolContext:
    """Build a ToolContext rooted at ``workspace`` with a fake config."""
    config = SimpleNamespace(output_cap_chars=30_000, **config_kwargs)
    return ToolContext(
        workspace_root=workspace,
        interrupt=InterruptController(),
        config=config,
    )


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a raw git command in ``cwd`` (test setup helper)."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        check=True,
    )


def _make_repo(root: Path) -> Path:
    """Initialize a git repo at ``root`` with one committed file."""
    _run_git(root, "init")
    # Local identity so commits succeed regardless of global git config.
    _run_git(root, "config", "user.email", "forge-test@example.com")
    _run_git(root, "config", "user.name", "Forge Test")
    (root / "hello.txt").write_text("hello world\n", encoding="utf-8")
    _run_git(root, "add", "hello.txt")
    _run_git(root, "commit", "-m", "initial commit")
    return root


# -- happy path (Req 9.2) ----------------------------------------------------


def test_happy_path_returns_command_output(tmp_path):
    """A supported operation runs in the repo and returns its output."""
    repo = _make_repo(tmp_path)
    tool = GitTool()

    result = tool.run({"operation": "log", "args": ["--oneline"]}, _ctx(repo))

    assert result.ok is True
    assert result.error is None
    assert result.meta.get("exit_code") == 0
    # The committed message is part of the returned `git log --oneline` output.
    assert "initial commit" in result.content


def test_happy_path_status_succeeds_on_clean_repo(tmp_path):
    """`git status` is a supported op and succeeds on the Workspace repo."""
    repo = _make_repo(tmp_path)
    tool = GitTool()

    result = tool.run({"operation": "status"}, _ctx(repo))

    assert result.ok is True
    assert result.meta.get("exit_code") == 0
    assert result.content != ""


# -- not a repository (Req 9.3) ----------------------------------------------


def test_not_a_repo_returns_descriptive_result(tmp_path):
    """Invoking git where the Workspace is not a repo reports not-a-repo."""
    # A bare, freshly-created temp dir that has not been `git init`-ed.
    non_repo = tmp_path / "plain"
    non_repo.mkdir()
    tool = GitTool()

    result = tool.run({"operation": "status"}, _ctx(non_repo))

    assert result.ok is False
    assert result.meta.get("not_a_repo") is True
    assert "not a git repository" in (result.error or "")


# -- non-zero exit (Req 9.5) -------------------------------------------------


def test_non_zero_exit_surfaces_code_and_error_output(tmp_path):
    """A supported op that exits non-zero surfaces the code and stderr."""
    repo = _make_repo(tmp_path)
    tool = GitTool()

    # `git show` of a ref that does not exist exits non-zero and writes a
    # diagnostic to stderr -- a supported operation failing for a real reason.
    result = tool.run(
        {"operation": "show", "args": ["nonexistent-ref-xyz"]}, _ctx(repo)
    )

    assert result.ok is False
    # The exit code is reported and is genuinely non-zero (Req 9.5).
    assert result.meta.get("exit_code") not in (None, 0)
    # The captured error output is non-empty.
    assert (result.error or "") != ""


def test_non_zero_exit_on_checkout_of_missing_branch(tmp_path):
    """`git checkout` of a missing branch is a supported op that fails cleanly."""
    repo = _make_repo(tmp_path)
    tool = GitTool()

    result = tool.run(
        {"operation": "checkout", "args": ["branch-that-does-not-exist"]},
        _ctx(repo),
    )

    assert result.ok is False
    assert result.meta.get("exit_code") not in (None, 0)
    assert (result.error or "") != ""
