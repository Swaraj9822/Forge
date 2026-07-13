"""Integration tests for :class:`forge.verification.VerificationRunner` (task 7.3).

These tests run the Verify_Command as a real process through the platform
default shell, exercising the runner's reuse of the shared shell execution core
end to end (Req 4.1). They assert the three representative outcomes the runner
maps from a real execution:

* a trivial ``exit 0`` command -> outcome ``"passed"`` with exit code ``0``
  (Req 4.1, 4.2);
* a trivial non-zero exit -> outcome ``"failed"`` carrying the exit code and the
  captured combined output (Req 4.3);
* a command that sleeps past a short ``timeout_s`` -> outcome ``"timed_out"``,
  returning promptly (well under the sleep duration) which proves the process
  tree was terminated rather than waited out (Req 4.4).

Commands are chosen to work on both Windows (``cmd.exe``) and POSIX
(``/bin/sh``); platform-specific command strings branch on :data:`IS_WINDOWS`,
mirroring ``tests/test_execute_command.py``. Each test uses ``tmp_path`` as the
Workspace root and a fresh :class:`~forge.interrupt.InterruptController`.
"""

from __future__ import annotations

import os
import sys
import time

from forge.interrupt import InterruptController
from forge.verification import VerificationRunner

IS_WINDOWS = sys.platform == "win32" or os.name == "nt"


def _sleep_command(seconds: int) -> str:
    """A command that blocks for roughly ``seconds`` seconds on either OS."""
    if IS_WINDOWS:
        # No portable `sleep` on Windows; `ping` waits ~1s between echoes and
        # does not depend on console stdin (unlike `timeout`).
        return f"ping -n {seconds + 1} 127.0.0.1"
    return f"sleep {seconds}"


# -- exit-0 command -> passed (Req 4.1, 4.2) ---------------------------------


def test_exit_zero_command_passes(tmp_path):
    """A trivial successful command yields outcome ``passed`` with exit code 0."""
    runner = VerificationRunner(tmp_path, InterruptController())

    result = runner.run("echo hello", timeout_s=30, output_cap=30_000)

    assert result.outcome == "passed"
    assert result.exit_code == 0
    assert result.truncated is False


# -- non-zero exit -> failed (Req 4.3) ---------------------------------------


def test_non_zero_exit_fails_with_exit_code_and_output(tmp_path):
    """A non-zero exit yields ``failed`` with the exit code and captured output."""
    if IS_WINDOWS:
        command = "echo boom & exit 3"
    else:
        command = "echo boom; exit 3"

    runner = VerificationRunner(tmp_path, InterruptController())

    result = runner.run(command, timeout_s=30, output_cap=30_000)

    assert result.outcome == "failed"
    assert result.exit_code == 3
    assert "boom" in result.output


# -- sleeping command past a short timeout -> timed_out (Req 4.4) ------------


def test_sleeping_command_times_out_and_returns_promptly(tmp_path):
    """A command sleeping past ``timeout_s`` yields ``timed_out`` promptly.

    Returning well under the 5-second sleep duration proves the process tree
    was terminated rather than waited out.
    """
    runner = VerificationRunner(tmp_path, InterruptController())

    start = time.monotonic()
    result = runner.run(_sleep_command(5), timeout_s=1, output_cap=30_000)
    elapsed = time.monotonic() - start

    assert result.outcome == "timed_out"
    assert elapsed < 4.5
