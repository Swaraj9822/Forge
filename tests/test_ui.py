"""Tests for the optional rich-backed UI class and fallbacks."""

from __future__ import annotations

import io
from forge.ui import Ui


def test_ui_plain_fallback() -> None:
    # color=False, spinner=False, non-TTY out
    out = io.StringIO()
    ui = Ui(out, color=False, spinner=False)

    # 1. Tool announcement
    announcement = ui.tool_announcement("read")
    assert announcement == "\n[tool: read]"

    # 2. Diff rendering
    ui.render_diff("--- a\n+++ b\n+added")
    assert out.getvalue() == "--- a\n+++ b\n+added\n"


def test_ui_monkeypatched_no_rich(monkeypatch) -> None:
    # Force _RICH to False
    import forge.ui

    monkeypatch.setattr(forge.ui, "_RICH", False)

    out = io.StringIO()
    # Even if color=True, should fallback because _RICH is False
    ui = Ui(out, color=True, spinner=True)

    announcement = ui.tool_announcement("search")
    assert announcement == "\n[tool: search]"

    ui.render_diff("some diff")
    assert out.getvalue() == "some diff\n"
