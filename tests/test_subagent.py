"""Unit tests for SubAgentRunner and subagent tool isolation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import pytest

from forge.config import Config
from forge.interrupt import InterruptController
from forge.session import ToolCall
from forge.subagent import SubAgentRunner, SubAgentContextManager
from forge.tools.base import Tool, ToolContext, ToolResult, ToolSpec


class FakeTool(Tool):
    def __init__(self, name: str, read_only: bool = True) -> None:
        self.name = name
        self.description = f"fake {name}"
        self.parameters = {"type": "object"}
        self.read_only = read_only

    def validate(self, args: dict) -> str | None:
        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True, content=f"{self.name} output")


class FakeProvider:
    def __init__(self) -> None:
        self.contents_passed = None
        self.tools_passed = None

    def generate_stream(self, contents, tools):
        self.contents_passed = contents
        self.tools_passed = tools
        # Return text delta first, then usage metadata, then done
        from forge.providers.base import TextDelta, UsageReport, Done
        return iter([
            TextDelta("subagent answer"),
            UsageReport(input_tokens=100, output_tokens=50),
            Done()
        ])


def test_subagent_context_manager_system_prompt() -> None:
    config = Config()
    cm = SubAgentContextManager(config, summarizer=None)
    prompt = cm.build_system_prompt()
    assert "You are a sub-agent helper" in prompt


def test_subagent_runner_isolation_and_restrictions(tmp_path: Path) -> None:
    config = Config(parallel_enabled=False)
    interrupt = InterruptController()
    provider = FakeProvider()
    
    registry = {
        "read": FakeTool("read"),
        "write": FakeTool("write"),
        "delegate": FakeTool("delegate"),
    }
    
    # 1. By default, subagent should only have enabled_tools intersect allowed_tools, and never "delegate"
    runner = SubAgentRunner(
        provider=provider,
        config=config,
        interrupt=interrupt,
        tool_registry=registry,
        allowed_tools={"read", "delegate"},
        max_turns=1,
    )
    
    # Check that "delegate" is strictly removed to prevent recursion
    assert "delegate" not in runner.subagent_enabled_tools
    assert "read" in runner.subagent_enabled_tools
    assert "write" not in runner.subagent_enabled_tools

    # 2. Run the runner
    res = runner.run("find something", workspace_root=tmp_path)
    
    # The provider should be invoked with the task
    assert provider.contents_passed is not None
    # Verify system prompt was passed
    system_msg = [c for c in provider.contents_passed if c.get("role") == "system"]
    assert len(system_msg) == 1
    assert "You are a sub-agent helper" in system_msg[0]["content"]

    # Verify final result text
    assert res.text == "subagent answer"
    # Verify usage was aggregated
    assert res.usage.input_tokens == 100
    assert res.usage.output_tokens == 50
