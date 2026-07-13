"""Tests for the ToolExecutor's approval-policy gating (Phase 2, Feature B).

These tests verify the four guarantees documented in ``phase2.md`` §2.4:

* No policy wired -> identical to pre-Phase-2 behavior (a mutating tool runs).
* Supervised + ``AutoApprover`` -> a mutating call is approved and runs.
* Supervised + ``DenyMutationsApprover`` -> a mutating call is denied and the
  side effect (file write) does NOT happen.
* READONLY -> a mutating call is forbidden outright; the approver is never
  asked.
* Read tools are never gated in any mode.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.interrupt import InterruptController
from forge.policy import (
    ApprovalPolicy,
    Approver,
    AutoApprover,
    AutonomyMode,
    Decision,
    DenyMutationsApprover,
    ShellMatcher,
)
from forge.tools.base import ToolContext, ToolExecutor, ToolResult
from forge.tools.fs import ReadTool, WriteTool


# --------------------------------------------------------------------------- #
# Spy approver (records every request so tests can assert it was or was not
# called).
# --------------------------------------------------------------------------- #


class SpyApprover:
    """An :class:`Approver` that records each request and returns APPROVE."""

    def __init__(self, decision: Decision = Decision.APPROVE) -> None:
        self.calls: list[tuple[str, dict, str | None]] = []
        self._decision = decision

    def request(self, name: str, args: dict, preview: str | None) -> Decision:
        self.calls.append((name, dict(args), preview))
        return self._decision


class _FakeTool:
    """Minimal stand-in tool that records its calls for assertion."""

    name = "fake"
    description = "fake"
    parameters: dict = {}
    read_only = False

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def validate(self, args: dict) -> str | None:
        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult(ok=True, content="ok", meta={"ran": True})


class _ReadOnlyFakeTool(_FakeTool):
    name = "fake_ro"
    read_only = True


def _make_executor(
    *,
    tools: dict | None = None,
    enabled: set[str] | None = None,
    workspace: Path,
    policy: ApprovalPolicy | None = None,
    approver: Approver | None = None,
    checkpoint: object | None = None,
) -> ToolExecutor:
    if tools is None:
        tools = {"fake": _FakeTool()}
    if enabled is None:
        enabled = set(tools)
    interrupt = InterruptController()
    ctx = ToolContext(workspace_root=workspace, interrupt=interrupt)
    return ToolExecutor(
        registry=tools,
        enabled=enabled,
        interrupt=interrupt,
        context=ctx,
        policy=policy,
        approver=approver,
        checkpoint=checkpoint,
    )


def _make_call(name: str, args: dict | None = None):
    from forge.session import ToolCall

    return ToolCall(id="call-1", name=name, args=args or {})


# --------------------------------------------------------------------------- #
# Backward-compatibility: no policy wired means today's behavior.
# --------------------------------------------------------------------------- #


def test_no_policy_runs_mutating_tool(tmp_path: Path) -> None:
    """Regression bar: no policy wired -> the executor behaves exactly as before."""

    tool = _FakeTool()
    executor = _make_executor(
        tools={"fake": tool}, workspace=tmp_path, policy=None
    )
    result = executor.execute(_make_call("fake", {"x": 1}))
    assert result.ok is True
    assert tool.calls == [{"x": 1}]


# --------------------------------------------------------------------------- #
# Supervised + AutoApprover / DenyMutationsApprover.
# --------------------------------------------------------------------------- #


def test_supervised_with_auto_approver_runs_write(tmp_path: Path) -> None:
    """A mutating tool runs when the approver returns APPROVE."""

    target = tmp_path / "out.txt"
    write_tool = WriteTool()
    executor = _make_executor(
        tools={"write": write_tool},
        workspace=tmp_path,
        policy=ApprovalPolicy(mode=AutonomyMode.SUPERVISED),
        approver=AutoApprover(),
    )
    result = executor.execute(
        _make_call("write", {"path": str(target), "content": "hi"})
    )
    assert result.ok is True
    assert target.read_text(encoding="utf-8") == "hi"


def test_supervised_with_deny_approver_blocks_write(tmp_path: Path) -> None:
    """A denied mutating call returns meta=denied and does NOT touch the file."""

    target = tmp_path / "out.txt"
    write_tool = WriteTool()
    executor = _make_executor(
        tools={"write": write_tool},
        workspace=tmp_path,
        policy=ApprovalPolicy(mode=AutonomyMode.SUPERVISED),
        approver=DenyMutationsApprover(),
    )
    result = executor.execute(
        _make_call("write", {"path": str(target), "content": "hi"})
    )
    assert result.ok is False
    assert result.meta.get("denied") is True
    assert "was not approved" in (result.error or "")
    # File must NOT exist: the side effect was skipped.
    assert not target.exists()


def test_supervised_spy_approver_is_called_with_preview(tmp_path: Path) -> None:
    """The approver sees (name, args, preview) when a gate fires."""

    target = tmp_path / "out.txt"
    spy = SpyApprover()
    write_tool = WriteTool()
    executor = _make_executor(
        tools={"write": write_tool},
        workspace=tmp_path,
        policy=ApprovalPolicy(mode=AutonomyMode.SUPERVISED),
        approver=spy,
    )
    result = executor.execute(
        _make_call("write", {"path": str(target), "content": "hi"})
    )
    # Spy returns APPROVE -> the call runs.
    assert result.ok is True
    assert len(spy.calls) == 1
    name, args, preview = spy.calls[0]
    assert name == "write"
    assert args == {"path": str(target), "content": "hi"}
    assert isinstance(preview, str)
    # Preview must contain the new content as a +line.
    assert "+hi" in preview


# --------------------------------------------------------------------------- #
# READONLY mode: the approver is never asked.
# --------------------------------------------------------------------------- #


def test_readonly_mode_forbids_write_without_asking(tmp_path: Path) -> None:
    """READONLY forbids outright; spy approver must NOT be called."""

    target = tmp_path / "out.txt"
    spy = SpyApprover()
    write_tool = WriteTool()
    executor = _make_executor(
        tools={"write": write_tool},
        workspace=tmp_path,
        policy=ApprovalPolicy(mode=AutonomyMode.READONLY),
        approver=spy,
    )
    result = executor.execute(
        _make_call("write", {"path": str(target), "content": "hi"})
    )
    assert result.ok is False
    assert result.meta.get("forbidden") is True
    assert spy.calls == []  # the approver must NEVER be asked
    assert not target.exists()


def test_readonly_mode_allows_read(tmp_path: Path) -> None:
    """A read tool runs in READONLY mode (and never prompts)."""

    src = tmp_path / "src.txt"
    src.write_text("hello", encoding="utf-8")
    spy = SpyApprover()
    executor = _make_executor(
        tools={"read": ReadTool()},
        workspace=tmp_path,
        policy=ApprovalPolicy(mode=AutonomyMode.READONLY),
        approver=spy,
    )
    result = executor.execute(_make_call("read", {"path": str(src)}))
    assert result.ok is True
    assert result.content == "hello"
    assert spy.calls == []


def test_readonly_mode_forbids_shell(tmp_path: Path) -> None:
    """A shell call is forbidden in READONLY mode, even when allowlisted."""

    from forge.tools.shell import ShellTool

    spy = SpyApprover()
    executor = _make_executor(
        tools={"shell": ShellTool()},
        workspace=tmp_path,
        policy=ApprovalPolicy(
            mode=AutonomyMode.READONLY,
            shell=ShellMatcher(allowlist=("pytest",)),
        ),
        approver=spy,
    )
    result = executor.execute(
        _make_call("shell", {"command": "pytest"})
    )
    assert result.ok is False
    assert result.meta.get("forbidden") is True
    assert spy.calls == []


# --------------------------------------------------------------------------- #
# Read tools are NEVER gated in any mode.
# --------------------------------------------------------------------------- #


def test_read_tool_never_prompts_in_any_mode(tmp_path: Path) -> None:
    """The read tool must run in autopilot / supervised / readonly."""

    src = tmp_path / "src.txt"
    src.write_text("hello", encoding="utf-8")
    for mode in (
        AutonomyMode.AUTOPILOT,
        AutonomyMode.SUPERVISED,
        AutonomyMode.READONLY,
    ):
        spy = SpyApprover()
        executor = _make_executor(
            tools={"read": ReadTool()},
            workspace=tmp_path,
            policy=ApprovalPolicy(mode=mode),
            approver=spy,
        )
        result = executor.execute(_make_call("read", {"path": str(src)}))
        assert result.ok is True, f"mode={mode}"
        assert spy.calls == [], f"approver must not be called for read in mode={mode}"


# --------------------------------------------------------------------------- #
# Missing approver on a gated call is a safe default: DENY.
# --------------------------------------------------------------------------- #


def test_no_approver_defaults_to_deny_for_mutating_call(tmp_path: Path) -> None:
    """A path that forgot to wire an approver denies rather than hanging."""

    target = tmp_path / "out.txt"
    write_tool = WriteTool()
    executor = _make_executor(
        tools={"write": write_tool},
        workspace=tmp_path,
        policy=ApprovalPolicy(mode=AutonomyMode.SUPERVISED),
        approver=None,
    )
    result = executor.execute(
        _make_call("write", {"path": str(target), "content": "hi"})
    )
    assert result.ok is False
    assert result.meta.get("denied") is True
    assert not target.exists()


# --------------------------------------------------------------------------- #
# set_approver wiring (used by bootstrap).
# --------------------------------------------------------------------------- #


def test_set_approver_replaces_approver(tmp_path: Path) -> None:
    """set_approver swaps the approver in the live executor."""

    target = tmp_path / "out.txt"
    write_tool = WriteTool()
    executor = _make_executor(
        tools={"write": write_tool},
        workspace=tmp_path,
        policy=ApprovalPolicy(mode=AutonomyMode.SUPERVISED),
        approver=None,
    )
    # First call: no approver -> denied.
    r1 = executor.execute(
        _make_call("write", {"path": str(target), "content": "x"})
    )
    assert r1.meta.get("denied") is True

    # Wire an auto-approver after construction; next call should succeed.
    executor.set_approver(AutoApprover())
    r2 = executor.execute(
        _make_call("write", {"path": str(target), "content": "y"})
    )
    assert r2.ok is True
    assert target.read_text(encoding="utf-8") == "y"


# --------------------------------------------------------------------------- #
# Validation runs before approval: a validation-error result is never gated.
# --------------------------------------------------------------------------- #


def test_validation_error_short_circuits_before_approval(tmp_path: Path) -> None:
    """A validation error must not be reclassified as denied/forbidden."""

    spy = SpyApprover()
    write_tool = WriteTool()
    executor = _make_executor(
        tools={"write": write_tool},
        workspace=tmp_path,
        policy=ApprovalPolicy(mode=AutonomyMode.SUPERVISED),
        approver=spy,
    )
    # Missing 'content' argument -> WriteTool.validate returns an error.
    result = executor.execute(
        _make_call("write", {"path": str(tmp_path / "x")})
    )
    assert result.ok is False
    assert result.meta.get("validation_error") is True
    # The approver must not have been called: validation runs first.
    assert spy.calls == []
