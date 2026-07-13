"""Unit tests for the shell tool's runtime behavior (task 10.3).

These tests exercise the :class:`forge.tools.shell.ShellTool` directly (calling
``run``) against the real platform shell, covering:

* happy path -- stdout/stderr/exit-code capture for a successful command
  (Req 7.1);
* non-zero exit -- the exit code and captured error output are surfaced
  (Req 7.2);
* timeout -- a command exceeding the configured timeout is terminated and a
  timeout result returned (Req 7.3);
* empty-command rejection -- a blank command is reported as invalid without
  spawning a process (Req 7.6);
* interrupt termination -- a tripped interrupt stops the command and yields an
  "interrupted" result (Req 7.4).

Commands are chosen to work on both Windows (``cmd.exe``) and POSIX
(``/bin/sh``); platform-specific command strings are branched on
:data:`IS_WINDOWS`. Timeout/interrupt tests use a sleeping command and short
limits so they finish quickly.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from forge.interrupt import InterruptController
from forge.tools.base import ToolContext
from forge.tools.shell import ShellTool

IS_WINDOWS = sys.platform == "win32" or os.name == "nt"


def _ctx(workspace: Path, **config_kwargs) -> ToolContext:
    """Build a ToolContext rooted at ``workspace`` with a fake config.

    ``config_kwargs`` typically carries ``shell_timeout_s`` and/or
    ``output_cap_chars``; absent limits fall back to the tool's documented
    defaults via :meth:`ShellTool._limit`.
    """
    config = SimpleNamespace(**config_kwargs) if config_kwargs else None
    return ToolContext(
        workspace_root=workspace,
        interrupt=InterruptController(),
        config=config,
    )


def _sleep_command(seconds: int) -> str:
    """A command that blocks for roughly ``seconds`` seconds on either OS."""
    if IS_WINDOWS:
        # No portable `sleep` on Windows; `ping` waits ~1s between echoes and
        # does not depend on console stdin (unlike `timeout`).
        return f"ping -n {seconds + 1} 127.0.0.1"
    return f"sleep {seconds}"


# -- happy path (Req 7.1) ----------------------------------------------------


def test_happy_path_captures_stdout_and_exit_code(tmp_path):
    """A successful command returns its stdout and a zero exit code."""
    tool = ShellTool()
    result = tool.run({"command": "echo hello"}, _ctx(tmp_path))

    assert result.ok is True
    assert result.error is None
    assert result.meta.get("exit_code") == 0
    # The rendered content surfaces the exit code and the command's stdout.
    assert "hello" in result.content
    assert "exit_code: 0" in result.content


# -- non-zero exit (Req 7.2) -------------------------------------------------


def test_non_zero_exit_surfaces_code_and_error_output(tmp_path):
    """A non-zero exit reports the exit code and the captured stderr."""
    if IS_WINDOWS:
        command = "echo boom 1>&2 & exit 7"
    else:
        command = "echo boom 1>&2; exit 7"

    tool = ShellTool()
    result = tool.run({"command": command}, _ctx(tmp_path))

    assert result.ok is False
    assert result.meta.get("exit_code") == 7
    assert "exited with code 7" in (result.error or "")
    # The captured error output is included in the result content (Req 7.2).
    assert "boom" in result.content


# -- timeout (Req 7.3) -------------------------------------------------------


def test_timeout_terminates_command_and_reports_timeout(tmp_path):
    """A command exceeding the configured timeout is terminated and flagged."""
    tool = ShellTool()
    # 1-second cap against a ~5-second sleep so the tool times out promptly.
    ctx = _ctx(tmp_path, shell_timeout_s=1, output_cap_chars=30_000)

    start = time.monotonic()
    result = tool.run({"command": _sleep_command(5)}, ctx)
    elapsed = time.monotonic() - start

    assert result.ok is False
    assert result.meta.get("timed_out") is True
    assert "timed out" in (result.error or "")
    # It must not have waited for the full 5-second command to finish.
    assert elapsed < 4.5


# -- empty-command rejection (Req 7.6) ---------------------------------------


def test_empty_command_is_rejected_as_invalid(tmp_path):
    """An empty command string is reported invalid without running anything."""
    tool = ShellTool()
    result = tool.run({"command": ""}, _ctx(tmp_path))

    assert result.ok is False
    assert result.meta.get("invalid") is True
    assert result.error == "empty/invalid command"


def test_whitespace_only_command_is_rejected_as_invalid(tmp_path):
    """A whitespace-only command is likewise reported invalid (Req 7.6)."""
    tool = ShellTool()
    result = tool.run({"command": "   \t  "}, _ctx(tmp_path))

    assert result.ok is False
    assert result.meta.get("invalid") is True
    assert result.error == "empty/invalid command"


# -- interrupt termination (Req 7.4) -----------------------------------------


def test_interrupt_terminates_command_and_reports_interrupted(tmp_path):
    """A tripped interrupt stops a running command and yields an interrupted
    result well before the command would have finished on its own."""
    tool = ShellTool()
    # Generous timeout so the interrupt -- not the timeout -- ends the command.
    ctx = _ctx(tmp_path, shell_timeout_s=30, output_cap_chars=30_000)

    # Trip the interrupt up front; the run loop polls it and stops promptly.
    ctx.interrupt.trip()

    start = time.monotonic()
    result = tool.run({"command": _sleep_command(5)}, ctx)
    elapsed = time.monotonic() - start

    assert result.ok is False
    assert result.meta.get("interrupted") is True
    assert "interrupted" in (result.error or "")
    # Stopped quickly, not after the ~5-second sleep.
    assert elapsed < 4.5
