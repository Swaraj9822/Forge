"""Unit tests for the delegate tool (DelegateTool)."""

from __future__ import annotations

from pathlib import Path
import pytest

from forge.config import Config
from forge.interrupt import InterruptController
from forge.session import Usage
from forge.tools.base import ToolContext
from forge.tools.subagent import DelegateTool
from forge.subagent import SubAgentResult


class FakeSubAgentRunner:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def run(self, task: str, *, workspace_root: Path) -> SubAgentResult:
        return SubAgentResult(
            text=f"delegated response for task: {task}",
            usage=Usage(input_tokens=10, output_tokens=5, estimated_cost=0.01),
        )


def test_delegate_tool_validation() -> None:
    tool = DelegateTool()

    # 1. Valid args
    assert tool.validate({"task": "explore"}) is None
    assert tool.validate({"task": "explore", "tools": ["read"], "max_turns": 2}) is None

    # 2. Invalid args
    assert tool.validate({}) is not None
    assert tool.validate({"task": "  "}) is not None
    assert tool.validate({"task": 123}) is not None
    assert tool.validate({"task": "explore", "tools": "read"}) is not None
    assert tool.validate({"task": "explore", "max_turns": -1}) is not None
    assert tool.validate({"task": "explore", "max_turns": "4"}) is not None


def test_delegate_tool_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = DelegateTool()
    config = Config()
    interrupt = InterruptController()
    
    ctx = ToolContext(
        workspace_root=tmp_path,
        interrupt=interrupt,
        config=config,
        state={},
        provider="fake-provider",
        tool_registry={"read": object(), "write": object()},
        policy="fake-policy",
        approver="fake-approver",
    )

    # Monkeypatch SubAgentRunner to use FakeSubAgentRunner
    import forge.tools.subagent
    monkeypatch.setattr(forge.tools.subagent, "SubAgentRunner", FakeSubAgentRunner)

    res = tool.run({"task": "my task"}, ctx)
    assert res.ok
    assert res.content == "delegated response for task: my task"
    assert res.meta["input_tokens"] == 10
    assert res.meta["output_tokens"] == 5
    assert res.meta["estimated_cost"] == 0.01
    # The subagent_usage marker is what the AgentLoop folds into the parent turn.
    assert res.meta["subagent_usage"] == {"input_tokens": 10, "output_tokens": 5}
