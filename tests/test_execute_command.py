"""Unit tests for the shared shell execution core (task 1.2).

These tests exercise :func:`forge.tools.shell.execute_command` directly,
asserting the raw :class:`forge.tools.shell.CommandExecution` fields the
Verification_Runner depends on (Req 4.1): the exit code on success and failure,
the ``timed_out`` flag, and the ``spawn_error`` field when the process cannot
be started. Capping of the rendered output is applied by callers, not by
``execute_command`` itself, so the combined-output cap is covered by the
unchanged regression suites (``test_shell_behavior.py`` /
``test_shell_output_cap.py``) rather than here.

Commands are chosen to work on both Windows (``cmd.exe``) and POSIX
(``/bin/sh``); platform-specific command strings branch on :data:`IS_WINDOWS`.
The timeout test uses a sleeping command and a short limit so it finishes
quickly.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from forge.interrupt import InterruptController
from forge.tools.shell import CommandExecution, execute_command

IS_WINDOWS = sys.platform == "win32" or os.name == "nt"


def _sleep_command(seconds: int) -> str:
    """A command that blocks for roughly ``seconds`` seconds on either OS."""
    if IS_WINDOWS:
        # No portable `sleep` on Windows; `ping` waits ~1s between echoes and
        # does not depend on console stdin (unlike `timeout`).
        return f"ping -n {seconds + 1} 127.0.0.1"
    return f"sleep {seconds}"


# -- exit-0 command (Req 4.1, 4.2) -------------------------------------------


def test_exit_zero_command_captures_stdout_and_zero_exit(tmp_path):
    """A trivial successful command yields exit code 0 with no control flags."""
    execution = execute_command(
        "echo hello",
        workspace_root=tmp_path,
        interrupt=InterruptController(),
        timeout_s=30,
        output_cap=30_000,
    )

    assert isinstance(execution, CommandExecution)
    assert execution.exit_code == 0
    assert execution.timed_out is False
    assert execution.interrupted is False
    assert execution.spawn_error is None
    assert "hello" in execution.stdout


# -- non-zero exit (Req 4.3) -------------------------------------------------


def test_non_zero_exit_preserves_exit_code_and_output(tmp_path):
    """A non-zero exit preserves the exit code and the captured combined output."""
    if IS_WINDOWS:
        command = "echo boom 1>&2 & exit 7"
    else:
        command = "echo boom 1>&2; exit 7"

    execution = execute_command(
        command,
        workspace_root=tmp_path,
        interrupt=InterruptController(),
        timeout_s=30,
        output_cap=30_000,
    )

    assert execution.exit_code == 7
    assert execution.timed_out is False
    assert execution.interrupted is False
    assert execution.spawn_error is None
    assert "boom" in execution.stderr


# -- timeout (Req 4.4) -------------------------------------------------------


def test_timeout_sets_timed_out_and_terminates_promptly(tmp_path):
    """A command exceeding ``timeout_s`` is terminated and flagged timed_out."""
    start = time.monotonic()
    execution = execute_command(
        _sleep_command(5),
        workspace_root=tmp_path,
        interrupt=InterruptController(),
        timeout_s=1,
        output_cap=30_000,
    )
    elapsed = time.monotonic() - start

    assert execution.timed_out is True
    assert execution.interrupted is False
    assert execution.spawn_error is None
    # It must not have waited for the full 5-second command to finish.
    assert elapsed < 4.5


# -- interrupt (Req 4.1 interrupt path) --------------------------------------


def test_tripped_interrupt_sets_interrupted_and_stops_promptly(tmp_path):
    """A tripped interrupt stops the command and sets the interrupted flag."""
    interrupt = InterruptController()
    interrupt.trip()

    start = time.monotonic()
    execution = execute_command(
        _sleep_command(5),
        workspace_root=tmp_path,
        interrupt=interrupt,
        timeout_s=30,
        output_cap=30_000,
    )
    elapsed = time.monotonic() - start

    assert execution.interrupted is True
    assert execution.timed_out is False
    assert execution.spawn_error is None
    assert elapsed < 4.5


# -- spawn error (Req 4.7 start path) ----------------------------------------


def test_spawn_error_when_workspace_root_missing(tmp_path):
    """A non-existent workspace root makes the process unstartable, surfaced
    as a ``spawn_error`` with no exit code rather than raising."""
    missing = tmp_path / "does-not-exist"
    assert not missing.exists()

    execution = execute_command(
        "echo hello",
        workspace_root=missing,
        interrupt=InterruptController(),
        timeout_s=30,
        output_cap=30_000,
    )

    assert execution.spawn_error is not None
    assert execution.exit_code is None
    assert execution.timed_out is False
    assert execution.interrupted is False
