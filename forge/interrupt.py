"""Interrupt handling for Forge.

A single :class:`InterruptController` owns the process-wide ``SIGINT`` (Ctrl-C)
handler. The controller has two states:

* **idle** - while waiting at the REPL prompt, Ctrl-C is a no-op cancel of the
  input line. The interrupt event is left clear so a stray Ctrl-C between turns
  never trips an interrupt that a later turn would observe.
* **active turn** - while a turn is in progress, Ctrl-C sets a
  :class:`threading.Event`. Long-running work (model streaming and blocking
  tools) polls that event at sub-second intervals via :meth:`check` and stops
  within one second, satisfying the interrupt guarantees in requirements 4.2,
  4.3, and 4.4.

The handler logic is intentionally exposed through :meth:`handle_sigint` so it
can be exercised directly in tests without actually delivering an OS signal, and
so installation can be skipped (or fail gracefully) when Forge does not run on
the main thread.
"""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from types import FrameType
from typing import Callable, Iterator, Optional

__all__ = ["InterruptController"]


class InterruptController:
    """Owns the ``SIGINT`` handler and the shared interrupt event.

    The poll interval between chunk reads / process waits in the rest of the
    system is sub-second, so a tripped event is observed within one second.
    """

    #: Recommended sub-second polling interval (seconds) for collaborators that
    #: wait on blocking work (stream chunks, process waits). Kept below one
    #: second so an interrupt is observed within the one-second guarantee.
    POLL_INTERVAL_S: float = 0.1

    def __init__(self) -> None:
        self._event = threading.Event()
        self._active = threading.Event()
        self._previous_handler: Optional[signal.Handlers | Callable] = None
        self._installed = False
        # Optional callback invoked when Ctrl-C is pressed while idle, letting
        # the REPL cancel the current input line. Defaults to a no-op.
        self._idle_cancel: Callable[[], None] = lambda: None

    # -- handler installation ------------------------------------------------

    def install(self) -> bool:
        """Install the process ``SIGINT`` handler.

        Returns ``True`` when the handler was installed. Signal handlers can
        only be registered from the main thread; when called from another
        thread (for example under a test runner) ``signal.signal`` raises
        :class:`ValueError`. In that case installation is skipped and ``False``
        is returned so the caller can continue without a hard failure. The
        controller still functions for direct/event-based use.
        """
        try:
            self._previous_handler = signal.signal(signal.SIGINT, self.handle_sigint)
        except ValueError:
            # Not on the main thread; cannot install a signal handler.
            self._installed = False
            return False
        self._installed = True
        return True

    def uninstall(self) -> None:
        """Restore the previously installed ``SIGINT`` handler, if any."""
        if self._installed and self._previous_handler is not None:
            signal.signal(signal.SIGINT, self._previous_handler)
        self._installed = False
        self._previous_handler = None

    def set_idle_cancel(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked on Ctrl-C while idle (input-line cancel)."""
        self._idle_cancel = callback

    # -- signal handling -----------------------------------------------------

    def handle_sigint(
        self, signum: int | None = None, frame: Optional[FrameType] = None
    ) -> None:
        """Handle a ``SIGINT`` (Ctrl-C).

        During an active turn this sets the interrupt event so polling
        collaborators stop. While idle it is a no-op cancel of the input line:
        the event is left untripped and the optional idle-cancel callback runs.

        Exposed publicly so tests can trigger the handler logic directly without
        delivering a real OS signal.
        """
        if self._active.is_set():
            self._event.set()
        else:
            # Idle: no-op cancel of the input line, never trips the event.
            self._idle_cancel()

    # -- turn lifecycle ------------------------------------------------------

    def begin_turn(self) -> None:
        """Mark a turn as active so Ctrl-C trips the interrupt event.

        The event is cleared first so an interrupt left over from a prior,
        completed turn never leaks into the new one.
        """
        self._event.clear()
        self._active.set()

    def end_turn(self) -> None:
        """Mark the turn finished so Ctrl-C reverts to an idle no-op cancel."""
        self._active.clear()
        self._event.clear()

    @contextmanager
    def turn(self) -> Iterator["InterruptController"]:
        """Context manager that brackets a turn with begin/end semantics."""
        self.begin_turn()
        try:
            yield self
        finally:
            self.end_turn()

    @property
    def active(self) -> bool:
        """``True`` while a turn is in progress."""
        return self._active.is_set()

    # -- interrupt state -----------------------------------------------------

    def check(self) -> bool:
        """Return whether the interrupt event is currently tripped."""
        return self._event.is_set()

    def reset(self) -> None:
        """Clear the interrupt event."""
        self._event.clear()

    def trip(self) -> None:
        """Trip the interrupt event directly.

        Primarily useful for tests and for collaborators that detect a
        cancellation through a channel other than ``SIGINT``.
        """
        self._event.set()

    @property
    def event(self) -> threading.Event:
        """The underlying interrupt :class:`threading.Event`.

        Exposed so collaborators (the Vertex client, blocking tools, the tool
        executor) can wait on it with a sub-second timeout instead of busy
        polling.
        """
        return self._event
