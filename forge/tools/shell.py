"""The ``shell`` tool: run a command via the platform default shell.

The shell tool executes a single command string through the platform default
shell (``cmd.exe /C <command>`` on Windows, ``/bin/sh -c <command>`` on POSIX),
rooted at the Workspace, and returns the command's standard output, standard
error, and exit code (Req 7.1). It enforces a wall-clock timeout (Req 7.3), a
combined-output character cap (Req 7.5), rejects empty/blank commands
(Req 7.6), surfaces the exit code and error output on non-zero exit (Req 7.2),
and terminates the whole process tree on a user interrupt (Req 7.4).

Execution core
--------------
The platform-shell invocation, workspace rooting, wall-clock timeout,
process-tree termination, and sub-second interrupt polling live in the
module-level :func:`execute_command` helper, which returns a raw
:class:`CommandExecution`. :class:`ShellTool` wraps that raw execution into a
:class:`~forge.tools.base.ToolResult` (rendering, the combined-output character
cap, and the success/timeout/interrupt/spawn-error shaping). Extracting the
core lets other callers (the Verification_Runner) reuse the exact same
execution machinery without going through the Tool protocol, while
:meth:`ShellTool.run`'s observable behavior is unchanged.

Process-group / termination model
----------------------------------
The child is spawned in its own process group so the whole tree -- the shell
*and* any children it spawns -- can be terminated together:

* **POSIX:** ``start_new_session=True`` puts the child in a new session/process
  group; termination uses ``os.killpg(pgid, SIGTERM)`` then, after a short
  grace period, ``SIGKILL``.
* **Windows:** ``CREATE_NEW_PROCESS_GROUP`` lets the child form its own group;
  termination calls :meth:`Popen.terminate` and falls back to
  ``taskkill /F /T /PID <pid>`` to kill the whole tree (``/T``) forcefully
  (``/F``). ``taskkill`` is the robust way to reap ``cmd.exe`` plus its
  descendants on Windows.

Concurrency / interrupt polling
-------------------------------
:func:`execute_command` reads stdout/stderr on background threads and polls in
the calling thread roughly every :data:`InterruptController.POLL_INTERVAL_S`
seconds. Each poll checks (a) whether the process exited, (b) whether the
timeout elapsed, and (c) whether the interrupt event is tripped, so both the
timeout and the interrupt are observed within sub-second latency (satisfying
the one-second interrupt guarantee from Req 4.3).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.interrupt import InterruptController
from forge.tools.base import ToolContext, ToolResult

__all__ = ["ShellTool", "CommandExecution", "execute_command"]

# Fallback limits used when no config is supplied on the ToolContext. These
# mirror the documented defaults in forge.config (shell_timeout_s=120,
# output_cap_chars=30_000) so the tool behaves correctly even in minimal wiring.
_DEFAULT_TIMEOUT_S = 120
_DEFAULT_OUTPUT_CAP = 30_000

# Grace period between a soft terminate and a hard kill of the process tree.
_KILL_GRACE_S = 0.5


def _is_windows() -> bool:
    return sys.platform == "win32" or os.name == "nt"


@dataclass(frozen=True)
class CommandExecution:
    """The raw outcome of running a command through the platform default shell.

    Carries the captured streams and control-flow flags the callers need, but
    *not* the rendered/capped presentation: rendering and the combined-output
    character cap are applied by the caller (see :func:`_render` / :func:`_cap`)
    so each caller can flag truncation independently.

    Attributes
    ----------
    stdout / stderr:
        The decoded standard output / error captured from the command.
    exit_code:
        The process return code, or ``None`` when the command did not run to
        completion (timed out, interrupted, or could not start).
    timed_out:
        ``True`` when the wall-clock timeout elapsed and the process tree was
        terminated.
    interrupted:
        ``True`` when a user interrupt tripped during execution.
    spawn_error:
        ``None`` unless the process could not be started, in which case it
        carries a description of the failure.
    """

    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    interrupted: bool
    spawn_error: str | None = None


def execute_command(
    command: str,
    *,
    workspace_root: Path,
    interrupt: InterruptController | None,
    timeout_s: int,
    output_cap: int,
) -> CommandExecution:
    """Run ``command`` through the platform default shell rooted at the workspace.

    This is the shared shell execution core: it spawns the command in its own
    process group, drains stdout/stderr on background threads, polls for the
    wall-clock ``timeout_s`` and the ``interrupt`` event at sub-second
    intervals, and terminates the whole process tree on timeout or interrupt.
    It returns the raw :class:`CommandExecution`; callers are responsible for
    rendering and applying ``output_cap`` (via :func:`_render` / :func:`_cap`)
    so each caller can flag truncation independently.

    ``output_cap`` is accepted as part of the shared execution contract so all
    callers share one signature; the cap itself is applied at the presentation
    layer.
    """
    del output_cap  # capping is applied by callers when rendering the result.

    argv = _shell_argv(command)
    popen_kwargs = _spawn_kwargs()

    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(workspace_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **popen_kwargs,
        )
    except OSError as exc:  # pragma: no cover - environment dependent
        return CommandExecution(
            stdout="",
            stderr="",
            exit_code=None,
            timed_out=False,
            interrupted=False,
            spawn_error=str(exc),
        )

    # Drain stdout/stderr on background threads so a process that fills its
    # pipe buffers never blocks while we poll for timeout/interrupt.
    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    out_thread = threading.Thread(
        target=_drain, args=(proc.stdout, out_chunks), daemon=True
    )
    err_thread = threading.Thread(
        target=_drain, args=(proc.stderr, err_chunks), daemon=True
    )
    out_thread.start()
    err_thread.start()

    poll_interval = getattr(
        interrupt, "POLL_INTERVAL_S", InterruptController.POLL_INTERVAL_S
    )
    start = time.monotonic()
    timed_out = False
    interrupted = False

    while True:
        if proc.poll() is not None:
            break
        if interrupt is not None and interrupt.check():
            interrupted = True
            break
        if (time.monotonic() - start) >= timeout_s:
            timed_out = True
            break
        time.sleep(poll_interval)

    if timed_out or interrupted:
        _terminate_tree(proc)

    # Wait for the process and reader threads to wind down so we collect all
    # currently-available output.
    try:
        proc.wait(timeout=_KILL_GRACE_S + 1)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        _kill_tree(proc)
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    out_thread.join(timeout=1)
    err_thread.join(timeout=1)

    stdout = b"".join(out_chunks).decode("utf-8", errors="replace")
    stderr = b"".join(err_chunks).decode("utf-8", errors="replace")

    return CommandExecution(
        stdout=stdout,
        stderr=stderr,
        exit_code=proc.returncode,
        timed_out=timed_out,
        interrupted=interrupted,
        spawn_error=None,
    )


class ShellTool:
    """Run a shell command in the Workspace and capture its output.

    Implements the :class:`~forge.tools.base.Tool` protocol.
    """

    name = "shell"
    description = (
        "Run a shell command in the workspace and return its standard output, "
        "standard error, and exit code. Commands run through the platform "
        "default shell (cmd.exe on Windows, /bin/sh on Unix). Long-running "
        "commands are terminated after a timeout; large output is truncated."
    )
    read_only = False
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command line to execute.",
            }
        },
        "required": ["command"],
    }

    # -- validation ----------------------------------------------------------

    def validate(self, args: dict) -> str | None:
        """Ensure a ``command`` argument is present and is a string.

        A missing or non-string ``command`` is a validation error. An
        empty/whitespace-only string passes validation here and is rejected in
        :meth:`run` as an invalid command (Req 7.6), so the executor still runs
        the tool and the caller receives the structured "invalid" result.
        """
        if "command" not in args:
            return "Missing required argument 'command'."
        if not isinstance(args["command"], str):
            return "Argument 'command' must be a string."
        return None

    # -- execution -----------------------------------------------------------

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        command: str = args["command"]

        # Req 7.6: empty / whitespace-only command is invalid.
        if not command.strip():
            return ToolResult(
                ok=False,
                content="",
                error="empty/invalid command",
                meta={"invalid": True},
            )

        timeout_s = self._limit(ctx, "shell_timeout_s", _DEFAULT_TIMEOUT_S)
        output_cap = self._limit(ctx, "output_cap_chars", _DEFAULT_OUTPUT_CAP)

        execution = execute_command(
            command,
            workspace_root=ctx.workspace_root,
            interrupt=ctx.interrupt,
            timeout_s=timeout_s,
            output_cap=output_cap,
        )

        # Spawn failure (Req 7.1 start path).
        if execution.spawn_error is not None:  # pragma: no cover - env dependent
            return ToolResult(
                ok=False,
                content="",
                error=f"failed to start shell command: {execution.spawn_error}",
                meta={"exit_code": None, "spawn_error": True},
            )

        content, truncated = _cap(
            _render(execution.stdout, execution.stderr, execution.exit_code),
            output_cap,
        )

        # Interrupt result (Req 7.4).
        if execution.interrupted:
            meta: dict[str, Any] = {
                "interrupted": True,
                "exit_code": execution.exit_code,
            }
            if truncated:
                meta["truncated"] = True
            return ToolResult(
                ok=False,
                content=content,
                error="command interrupted",
                meta=meta,
            )

        # Timeout result (Req 7.3).
        if execution.timed_out:
            meta = {"timed_out": True, "exit_code": execution.exit_code}
            if truncated:
                meta["truncated"] = True
            return ToolResult(
                ok=False,
                content=content,
                error=f"command timed out after {timeout_s}s",
                meta=meta,
            )

        # Normal completion (Req 7.1, 7.2, 7.5).
        meta = {"exit_code": execution.exit_code}
        if truncated:
            meta["truncated"] = True

        if execution.exit_code == 0:
            return ToolResult(ok=True, content=content, meta=meta)
        return ToolResult(
            ok=False,
            content=content,
            error=f"command exited with code {execution.exit_code}",
            meta=meta,
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _limit(ctx: ToolContext, attr: str, fallback: int) -> int:
        """Read a numeric limit from ``ctx.config``; fall back when absent."""
        config = getattr(ctx, "config", None)
        if config is None:
            return fallback
        value = getattr(config, attr, None)
        if value is None:
            return fallback
        return int(value)


# -- process-tree helpers ----------------------------------------------------


def _shell_argv(command: str) -> list[str]:
    """Return the argv invoking the platform default shell for ``command``."""
    if _is_windows():
        comspec = os.environ.get("ComSpec", "cmd.exe")
        return [comspec, "/C", command]
    return ["/bin/sh", "-c", command]


def _spawn_kwargs() -> dict[str, Any]:
    """Return Popen kwargs that place the child in its own process group."""
    if _is_windows():
        # CREATE_NEW_PROCESS_GROUP lets the child form its own group so we
        # can terminate the whole tree (via taskkill /T) without touching
        # the parent process.
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    # POSIX: a new session detaches the child into its own process group
    # whose pgid equals the child pid, enabling os.killpg on the group.
    return {"start_new_session": True}


def _terminate_tree(proc: subprocess.Popen) -> None:
    """Terminate the child process tree (soft, then hard after a grace)."""
    if proc.poll() is not None:
        return
    if _is_windows():
        # Soft terminate first, then force-kill the whole tree.
        try:
            proc.terminate()
        except OSError:
            pass
        if _wait_quietly(proc, _KILL_GRACE_S):
            return
        _kill_tree(proc)
        return

    # POSIX: signal the whole process group.
    pgid = _posix_pgid(proc)
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    else:
        try:
            proc.terminate()
        except OSError:
            pass
    if _wait_quietly(proc, _KILL_GRACE_S):
        return
    _kill_tree(proc)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Forcefully kill the child process tree."""
    if proc.poll() is not None:
        return
    if _is_windows():
        # taskkill /T kills the process and its descendants; /F forces it.
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            pass
        # Best-effort direct kill as a final fallback.
        try:
            proc.kill()
        except OSError:
            pass
        return

    pgid = _posix_pgid(proc)
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def _posix_pgid(proc: subprocess.Popen) -> int | None:
    try:
        return os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return None


def _wait_quietly(proc: subprocess.Popen, timeout: float) -> bool:
    """Wait up to ``timeout`` for exit; return True if the process exited."""
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def _drain(stream: Any, sink: list[bytes]) -> None:
    """Read a binary stream to EOF, appending chunks to ``sink``."""
    if stream is None:  # pragma: no cover - defensive
        return
    try:
        for chunk in iter(lambda: stream.read(4096), b""):
            sink.append(chunk)
    except (ValueError, OSError):  # pragma: no cover - stream closed under us
        pass
    finally:
        try:
            stream.close()
        except OSError:  # pragma: no cover - defensive
            pass


def _render(stdout: str, stderr: str, exit_code: int | None) -> str:
    """Render stdout/stderr/exit code into a readable, structured string."""
    parts: list[str] = [f"exit_code: {exit_code}"]
    parts.append("stdout:")
    parts.append(stdout if stdout else "(empty)")
    parts.append("stderr:")
    parts.append(stderr if stderr else "(empty)")
    return "\n".join(parts)


def _cap(text: str, cap: int) -> tuple[str, bool]:
    """Truncate ``text`` to ``cap`` characters; return (text, truncated)."""
    if cap is not None and cap >= 0 and len(text) > cap:
        return text[:cap], True
    return text, False
