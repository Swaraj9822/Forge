"""Tests for the post-turn Review_Phase (independent review agent).

Covers the pure decision logic (gate, verdict parsing, loop control, feedback /
prompt formatting) and the ReviewCoordinator orchestration with scripted fakes.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from forge.config import Config, ConfigError, ConfigManager, ReviewConfig
from forge.review import (
    APPROVED,
    CHANGES_REQUESTED,
    ReviewCoordinator,
    build_review_prompt,
    format_plan,
    format_review_feedback,
    original_task,
    parse_verdict,
    should_review,
    should_run_correction,
)
from forge.session import Message, Session, TodoItem, Usage
from forge.usage import UsageSummary


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "enabled,trigger,mutated,turn_ok,expected",
    [
        (True, "on_file_change", True, True, True),
        (True, "on_file_change", False, True, False),   # nothing changed
        (True, "always", False, True, True),            # always runs
        (False, "always", True, True, False),           # disabled
        (True, "on_file_change", True, False, False),    # turn not ok
    ],
)
def test_should_review(enabled, trigger, mutated, turn_ok, expected) -> None:
    assert should_review(enabled, trigger, mutated, turn_ok) is expected


@pytest.mark.parametrize(
    "verdict,completed,maxi,interrupted,expected",
    [
        (CHANGES_REQUESTED, 0, 2, False, True),
        (CHANGES_REQUESTED, 2, 2, False, False),   # cap reached
        (APPROVED, 0, 2, False, False),            # approved stops
        (CHANGES_REQUESTED, 0, 0, False, False),   # advisory-only
        (CHANGES_REQUESTED, 0, 2, True, False),    # interrupted
    ],
)
def test_should_run_correction(verdict, completed, maxi, interrupted, expected) -> None:
    assert should_run_correction(verdict, completed, maxi, interrupted) is expected


def test_parse_verdict_approve() -> None:
    result = parse_verdict("Looks good.\nVERDICT: APPROVE")
    assert result.verdict == APPROVED
    assert result.findings == []


def test_parse_verdict_changes_with_findings() -> None:
    text = (
        "VERDICT: CHANGES_REQUESTED\n"
        "FINDINGS:\n"
        "- The new function is never called from the router.\n"
        "- Missing error handling for the empty-input case.\n"
    )
    result = parse_verdict(text)
    assert result.verdict == CHANGES_REQUESTED
    assert len(result.findings) == 2
    assert "never called" in result.findings[0]


def test_parse_verdict_fails_open_to_approved() -> None:
    # No explicit verdict marker -> advisory approve (no manufactured loop).
    result = parse_verdict("Some rambling with no verdict line.")
    assert result.verdict == APPROVED


def test_parse_verdict_changes_without_findings_gets_generic() -> None:
    result = parse_verdict("VERDICT: CHANGES_REQUESTED")
    assert result.verdict == CHANGES_REQUESTED
    assert len(result.findings) == 1


def test_format_review_feedback_lists_findings() -> None:
    from forge.review import ReviewResult

    fb = format_review_feedback(
        ReviewResult(verdict=CHANGES_REQUESTED, findings=["fix a", "fix b"])
    )
    assert "- fix a" in fb and "- fix b" in fb
    assert "unrelated edits" in fb


def test_build_review_prompt_includes_task_plan_diff() -> None:
    prompt = build_review_prompt("do the thing", "- [pending] step 1", "diff here")
    assert "do the thing" in prompt
    assert "step 1" in prompt
    assert "diff here" in prompt
    assert "VERDICT: APPROVE" in prompt


def test_original_task_returns_first_user_message() -> None:
    session = Session(id="s", created_at="t", updated_at="t")
    session.messages = [
        Message(role="user", text="the original task"),
        Message(role="model", text="ok"),
        Message(role="user", text="a later correction"),
    ]
    assert original_task(session) == "the original task"


def test_format_plan() -> None:
    todos = [
        TodoItem(id="1", text="read code", status="completed"),
        TodoItem(id="2", text="fix bug", status="in_progress"),
    ]
    plan = format_plan(todos)
    assert "[completed] read code" in plan
    assert "[in_progress] fix bug" in plan
    assert format_plan([]) == ""


# --------------------------------------------------------------------------- #
# Config resolution
# --------------------------------------------------------------------------- #


def _load_raw(raw: dict) -> Config:
    return ConfigManager()._from_raw(raw)


def test_review_config_defaults_disabled() -> None:
    cfg = _load_raw({})
    assert cfg.review.enabled is False
    assert cfg.review.trigger == "on_file_change"
    assert cfg.review.max_iterations == 2


def test_review_config_parses_values() -> None:
    cfg = _load_raw(
        {"review": {"enabled": True, "trigger": "always", "max_iterations": 1,
                    "tools": ["read", "git"]}}
    )
    assert cfg.review.enabled is True
    assert cfg.review.trigger == "always"
    assert cfg.review.max_iterations == 1
    assert cfg.review.tools == ("read", "git")


@pytest.mark.parametrize(
    "review",
    [
        {"max_iterations": -1},
        {"max_iterations": "2"},
        {"trigger": "sometimes"},
        {"tools": "read"},
        {"tools": [1, 2]},
    ],
)
def test_review_config_invalid_raises(review: dict) -> None:
    with pytest.raises(ConfigError):
        _load_raw({"review": review})


# --------------------------------------------------------------------------- #
# Coordinator orchestration (with fakes)
# --------------------------------------------------------------------------- #


class _TurnResult:
    def __init__(self, *, mutated=True, interrupted=False, error=None):
        self.mutated_files = mutated
        self.interrupted = interrupted
        self.error = error
        self.usage = _usage()


def _usage() -> UsageSummary:
    return UsageSummary(
        turn_input_tokens=10,
        turn_output_tokens=5,
        cumulative_input_tokens=10,
        cumulative_output_tokens=5,
        turn_cost=None,
        cumulative_cost=None,
        cost_available=False,
    )


class _FakeTracker:
    def record(self, *_a, **_k):
        pass

    def turn_summary(self):
        return _usage()


class _FakeAgentLoop:
    """Records correction turns and yields scripted TurnResults for them."""

    def __init__(self):
        self.usage_tracker = _FakeTracker()
        self.correction_prompts: list[str] = []

    def run_turn(self, session, text):
        self.correction_prompts.append(text)
        return _TurnResult(mutated=True)


class _FakeStore:
    def __init__(self):
        self.saved = 0

    def save(self, session):
        self.saved += 1


class _FakeInterrupt:
    def check(self):
        return False


class _RecordingRenderer:
    def __init__(self):
        self.events: list[str] = []

    def on_review_start(self):
        self.events.append("start")

    def on_review_result(self, result):
        self.events.append(f"result:{result.verdict}")

    def on_review_correction(self, i, m):
        self.events.append(f"correction:{i}/{m}")

    def on_review_cap_reached(self, result, iterations):
        self.events.append("cap")


def _coordinator(monkeypatch, verdict_script, *, review_cfg, checkpoint=None):
    """Build a ReviewCoordinator whose reviewer returns scripted verdicts."""
    config = Config(review=review_cfg)
    agent_loop = _FakeAgentLoop()
    store = _FakeStore()
    renderer = _RecordingRenderer()

    coord = ReviewCoordinator(
        config,
        provider=object(),
        agent_loop=agent_loop,
        session_store=store,
        interrupt=_FakeInterrupt(),
        tool_registry={},
        workspace_root=Path("."),
        checkpoint=checkpoint,
        renderer=renderer,
    )

    # Replace the real subagent-backed reviewer with a scripted sequence.
    scripts = list(verdict_script)

    def fake_run_review(prompt):
        from forge.review import parse_verdict as _pv
        return _pv(scripts.pop(0))

    monkeypatch.setattr(coord, "_run_review", fake_run_review)
    return coord, agent_loop, store, renderer


def _session() -> Session:
    s = Session(id="s", created_at="t", updated_at="t")
    s.messages = [Message(role="user", text="implement feature X")]
    return s


def test_coordinator_skips_when_disabled(monkeypatch) -> None:
    coord, agent_loop, store, renderer = _coordinator(
        monkeypatch, ["VERDICT: APPROVE"], review_cfg=ReviewConfig(enabled=False)
    )
    phase = coord.run(_session(), _TurnResult())
    assert phase.ran is False
    assert agent_loop.correction_prompts == []
    assert store.saved == 0


def test_coordinator_approves_first_pass(monkeypatch) -> None:
    coord, agent_loop, store, renderer = _coordinator(
        monkeypatch,
        ["VERDICT: APPROVE"],
        review_cfg=ReviewConfig(enabled=True, max_iterations=2),
    )
    phase = coord.run(_session(), _TurnResult())
    assert phase.ran is True
    assert phase.final_result.verdict == APPROVED
    assert phase.iterations_performed == 0
    assert agent_loop.correction_prompts == []  # no corrections
    assert store.saved == 1
    assert "result:approved" in renderer.events


def test_coordinator_requests_changes_then_approves(monkeypatch) -> None:
    coord, agent_loop, store, renderer = _coordinator(
        monkeypatch,
        ["VERDICT: CHANGES_REQUESTED\n- wire it up", "VERDICT: APPROVE"],
        review_cfg=ReviewConfig(enabled=True, max_iterations=2),
    )
    phase = coord.run(_session(), _TurnResult())
    assert phase.ran is True
    assert phase.final_result.verdict == APPROVED
    assert phase.iterations_performed == 1
    assert len(agent_loop.correction_prompts) == 1
    assert "wire it up" in agent_loop.correction_prompts[0]
    assert phase.cap_reached is False


def test_coordinator_hits_iteration_cap(monkeypatch) -> None:
    coord, agent_loop, store, renderer = _coordinator(
        monkeypatch,
        [
            "VERDICT: CHANGES_REQUESTED\n- x",
            "VERDICT: CHANGES_REQUESTED\n- still x",
            "VERDICT: CHANGES_REQUESTED\n- yet again",
        ],
        review_cfg=ReviewConfig(enabled=True, max_iterations=2),
    )
    phase = coord.run(_session(), _TurnResult())
    assert phase.ran is True
    assert phase.final_result.verdict == CHANGES_REQUESTED
    assert phase.iterations_performed == 2
    assert phase.cap_reached is True
    assert "cap" in renderer.events
    # A ReviewRecord was persisted.
    assert store.saved == 1


def test_coordinator_advisory_only_max_zero(monkeypatch) -> None:
    coord, agent_loop, store, renderer = _coordinator(
        monkeypatch,
        ["VERDICT: CHANGES_REQUESTED\n- something"],
        review_cfg=ReviewConfig(enabled=True, max_iterations=0),
    )
    phase = coord.run(_session(), _TurnResult())
    assert phase.ran is True
    assert phase.iterations_performed == 0
    assert agent_loop.correction_prompts == []  # advisory, no correction
    assert phase.cap_reached is False


def test_coordinator_uses_checkpoint_diff(monkeypatch) -> None:
    class _FakeCheckpoint:
        def diff_last_turn(self):
            return "a/x.py b/x.py\n+added line\n"

    captured = {}

    coord, agent_loop, store, renderer = _coordinator(
        monkeypatch,
        ["VERDICT: APPROVE"],
        review_cfg=ReviewConfig(enabled=True),
        checkpoint=_FakeCheckpoint(),
    )
    # The diff builder should surface the checkpoint diff.
    assert "added line" in coord._build_diff()


# --------------------------------------------------------------------------- #
# ReviewRecord persistence
# --------------------------------------------------------------------------- #


def test_review_record_roundtrips_through_session_json() -> None:
    from forge.session import ReviewRecord, session_from_json, session_to_json

    session = Session(id="s1", created_at="t", updated_at="t")
    session.review_records.append(
        ReviewRecord(verdict="changes_requested", iterations=2, cap_reached=True, findings=3)
    )
    restored = session_from_json(session_to_json(session))
    assert restored == session
    assert restored.review_records[0].verdict == "changes_requested"
    assert restored.review_records[0].findings == 3


def test_session_without_review_records_loads() -> None:
    """Older session JSON with no review_records key still loads (defaults [])."""
    from forge.session import session_from_dict

    data = {
        "id": "s2",
        "created_at": "t",
        "updated_at": "t",
        "messages": [],
        "todos": [],
        "usage": {"input_tokens": 0, "output_tokens": 0, "estimated_cost": None},
        "verification_records": [],
        # no "review_records" key
    }
    session = session_from_dict(data)
    assert session.review_records == []


def test_coordinator_appends_review_record(monkeypatch) -> None:
    coord, agent_loop, store, renderer = _coordinator(
        monkeypatch,
        ["VERDICT: CHANGES_REQUESTED\n- fix wiring", "VERDICT: APPROVE"],
        review_cfg=ReviewConfig(enabled=True, max_iterations=2),
    )
    session = _session()
    coord.run(session, _TurnResult())
    assert len(session.review_records) == 1
    assert session.review_records[0].verdict == APPROVED
    assert session.review_records[0].iterations == 1
