"""Unit tests for :class:`forge.interrupt.InterruptController`.

These tests exercise the handler logic directly via ``handle_sigint`` without
delivering a real OS signal, covering the idle no-op behavior, event-setting
during an active turn, and the check/reset/begin/end turn semantics that
underpin the interrupt guarantees in requirements 4.2, 4.3, and 4.4.
"""

from __future__ import annotations

import threading

from forge.interrupt import InterruptController


def test_handle_sigint_is_noop_while_idle():
    """A fresh (idle) controller does not trip the event on Ctrl-C, but it
    does run a registered idle-cancel callback (input-line cancel)."""
    controller = InterruptController()
    cancelled = []
    controller.set_idle_cancel(lambda: cancelled.append(True))

    assert controller.active is False
    assert controller.check() is False

    controller.handle_sigint()

    # Idle Ctrl-C never trips the interrupt event...
    assert controller.check() is False
    # ...but it does cancel the current input line.
    assert cancelled == [True]


def test_handle_sigint_sets_event_during_turn():
    """During an active turn, Ctrl-C sets the interrupt event so pollers stop.

    Supports Req 4.2/4.3 - an interrupt during generation or tool execution
    sets the event that collaborators observe.
    """
    controller = InterruptController()
    controller.begin_turn()

    assert controller.active is True
    assert controller.check() is False

    controller.handle_sigint()

    assert controller.check() is True


def test_check_and_reset_semantics():
    """check() reflects a tripped event; reset() clears it."""
    controller = InterruptController()

    # Trip via the handler during a turn.
    controller.begin_turn()
    controller.handle_sigint()
    assert controller.check() is True
    controller.reset()
    assert controller.check() is False

    # Trip directly via trip().
    controller.trip()
    assert controller.check() is True
    controller.reset()
    assert controller.check() is False


def test_begin_turn_clears_leftover_event():
    """begin_turn() clears any leftover interrupt so it never leaks into a new
    turn; end_turn() clears both active state and the event."""
    controller = InterruptController()

    # Simulate a stray/leftover trip before a new turn starts.
    controller.trip()
    assert controller.check() is True

    controller.begin_turn()
    assert controller.check() is False
    assert controller.active is True

    # A real interrupt within the turn still trips.
    controller.handle_sigint()
    assert controller.check() is True

    controller.end_turn()
    assert controller.active is False
    assert controller.check() is False


def test_turn_context_manager_toggles_active():
    """Inside turn() the controller is active; after exit it is idle again."""
    controller = InterruptController()

    assert controller.active is False
    with controller.turn() as ctx:
        assert ctx is controller
        assert controller.active is True
    assert controller.active is False


def test_turn_context_manager_resets_active_on_exception():
    """turn() restores idle state even when the body raises."""
    controller = InterruptController()

    class Boom(Exception):
        pass

    try:
        with controller.turn():
            assert controller.active is True
            raise Boom()
    except Boom:
        pass

    assert controller.active is False
    assert controller.check() is False


def test_idle_cancel_not_called_during_turn():
    """During a turn the event is set instead of invoking the idle-cancel
    callback."""
    controller = InterruptController()
    cancelled = []
    controller.set_idle_cancel(lambda: cancelled.append(True))

    controller.begin_turn()
    controller.handle_sigint()

    assert controller.check() is True
    assert cancelled == []


def test_event_property_exposes_threading_event():
    """The event property exposes the underlying threading.Event collaborators
    wait on, and reflects the tripped state."""
    controller = InterruptController()

    assert isinstance(controller.event, threading.Event)
    assert controller.event.is_set() is False

    controller.trip()
    assert controller.event.is_set() is True
