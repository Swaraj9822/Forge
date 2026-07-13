"""Tests for REPL UX additions: banner, spinner-safe flow, humanized usage,
and the informational slash commands (/cost, /tools, /model, /clear)."""

from __future__ import annotations

import io
from types import SimpleNamespace

from forge.repl import Repl, _fmt_tokens
from forge.session import Session, Usage
from forge.usage import UsageSummary


def _session() -> Session:
    return Session(
        id="s",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
        messages=[],
        todos=[],
        usage=Usage(input_tokens=0, output_tokens=0, estimated_cost=None),
    )


class _Loop:
    """Minimal agent-loop stand-in; run_turn must not be called by these tests."""

    def __init__(self, **attrs):
        self.renderer = None
        for k, v in attrs.items():
            setattr(self, k, v)

    def run_turn(self, session, text):  # pragma: no cover - must not run
        raise AssertionError("slash command must not invoke the agent loop")


def _repl(lines, *, loop=None, config=None):
    loop = loop or _Loop()
    out = io.StringIO()
    reader = lambda prompt: lines.pop(0)  # noqa: E731
    repl = Repl(loop, _session(), input_func=reader, out=out, config=config)
    return repl, out


# --------------------------------------------------------------------------- #
# _fmt_tokens
# --------------------------------------------------------------------------- #


def test_fmt_tokens_small():
    assert _fmt_tokens(42) == "42"


def test_fmt_tokens_thousands():
    assert _fmt_tokens(5668) == "5.7k"


# --------------------------------------------------------------------------- #
# Usage line humanization
# --------------------------------------------------------------------------- #


def test_usage_line_humanizes_tokens():
    repl, out = _repl([])
    u = UsageSummary(
        turn_input_tokens=5668,
        turn_output_tokens=324,
        cumulative_input_tokens=12000,
        cumulative_output_tokens=999,
        turn_cost=0.0087,
        cumulative_cost=0.02,
        cost_available=True,
    )
    repl._render_usage(u)
    rendered = out.getvalue()
    assert "[usage]" in rendered
    assert "5.7k in / 324 out" in rendered
    assert "12.0k in / 999 out" in rendered
    assert "$0.008700 turn" in rendered


# --------------------------------------------------------------------------- #
# Slash commands
# --------------------------------------------------------------------------- #


def test_tools_command_lists_exposed_tools():
    loop = _Loop(tool_executor=SimpleNamespace(
        specs=lambda: [SimpleNamespace(name="read"), SimpleNamespace(name="write")]
    ))
    repl, out = _repl(["/tools"], loop=loop)
    assert repl.run_once() is True
    assert "[tools] read, write" in out.getvalue()


def test_model_command_shows_model_and_thinking():
    cfg = SimpleNamespace(
        model="gemini-3.5-flash", provider_type="vertex",
        provider_thinking_level="high", policy_mode="autopilot",
        enabled_tools=["read"],
    )
    repl, out = _repl(["/model"], config=cfg)
    assert repl.run_once() is True
    rendered = out.getvalue()
    assert "gemini-3.5-flash" in rendered
    assert "thinking: high" in rendered


def test_cost_command_reports_cumulative():
    u = UsageSummary(
        turn_input_tokens=0, turn_output_tokens=0,
        cumulative_input_tokens=2000, cumulative_output_tokens=500,
        turn_cost=None, cumulative_cost=0.05, cost_available=True,
    )
    loop = _Loop(usage_tracker=SimpleNamespace(turn_summary=lambda: u))
    repl, out = _repl(["/cost"], loop=loop)
    assert repl.run_once() is True
    rendered = out.getvalue()
    assert "[cost]" in rendered
    assert "2.0k in / 500 out" in rendered
    assert "$0.050000" in rendered


def test_clear_command_is_noop_on_non_tty():
    repl, out = _repl(["/clear"])
    assert repl.run_once() is True
    # Non-TTY StringIO: clear writes nothing.
    assert out.getvalue() == ""


def test_help_lists_new_commands():
    repl, out = _repl(["/help"])
    assert repl.run_once() is True
    rendered = out.getvalue()
    for cmd in ("/cost", "/tools", "/model", "/clear"):
        assert cmd in rendered


# --------------------------------------------------------------------------- #
# Banner
# --------------------------------------------------------------------------- #


def test_banner_printed_on_run_when_config_present():
    cfg = SimpleNamespace(
        model="gemini-3.5-flash", provider_type="vertex",
        provider_thinking_level="high", policy_mode="autopilot",
        enabled_tools=["read", "write", "edit"],
    )
    repl, out = _repl(["/exit"], config=cfg)
    repl.run()
    rendered = out.getvalue()
    assert "Forge" in rendered
    assert "gemini-3.5-flash" in rendered
    assert "autopilot" in rendered


def test_no_banner_without_config():
    repl, out = _repl(["/exit"])
    repl.run()
    assert "Forge" not in out.getvalue()
