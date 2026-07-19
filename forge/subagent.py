"""Subagent execution harness for Forge.

Runs a scoped task in an isolated session with restricted tools, its own
ContextManager, and aggregated token usage.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.config import Config
from forge.context import ContextManager
from forge.interrupt import InterruptController
from forge.session import SessionStore, Usage
from forge.tools.base import Tool, ToolContext, ToolExecutor
from forge.usage import UsageTracker
from forge.agent import AgentLoop

__all__ = [
    "SubAgentResult",
    "SubAgentRunner",
    "SubAgentContextManager",
]


@dataclass(frozen=True)
class SubAgentResult:
    """The result of a subagent run."""

    text: str
    usage: Usage


class SubAgentInterruptController:
    """Wrapper around InterruptController that ignores turn boundaries.

    Prevents a subagent's run_turn lifecycle from altering the parent turn's
    active interrupt status.
    """

    def __init__(self, parent: InterruptController) -> None:
        self._parent = parent

    def begin_turn(self) -> None:
        pass

    def end_turn(self) -> None:
        pass

    def check(self) -> bool:
        return self._parent.check()

    def reset(self) -> None:
        self._parent.reset()

    def trip(self) -> None:
        self._parent.trip()

    @property
    def event(self) -> Any:
        return self._parent.event


class SubAgentContextManager(ContextManager):
    """ContextManager for subagents, overriding the default system prompt."""

    _DEFAULT_PROMPT = (
        "You are a sub-agent helper for the main coding assistant. Your job is to perform "
        "the requested scoped task, explore files/workspace as needed, and return a concise result. "
        "Do not perform changes or writes unless explicitly permitted. Return only the final outcome."
    )

    def __init__(
        self,
        config: Config,
        summarizer: Any,
        workspace_root: Path | None = None,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(
            config,
            summarizer=summarizer,
            providers=[],
            workspace_root=workspace_root,
            project_memory_filenames=(),
        )
        self._system_prompt = system_prompt or self._DEFAULT_PROMPT

    def _system_segments(self) -> list[str]:
        return [self._system_prompt]


class SubAgentRunner:
    """Orchestrates the execution of a subagent."""

    def __init__(
        self,
        provider: Any,
        config: Config,
        interrupt: InterruptController,
        *,
        tool_registry: dict[str, Tool],
        allowed_tools: set[str],
        max_turns: int = 1,
        policy: Any = None,
        approver: Any = None,
        parent_context: Any = None,
        system_prompt: str | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.interrupt = SubAgentInterruptController(interrupt)
        self.tool_registry = tool_registry
        # Subagent tools are restricted to allowed_tools and delegate is explicitly forbidden to prevent recursion
        self.subagent_enabled_tools = (set(tool_registry.keys()) & allowed_tools) - {"delegate"}
        self.max_turns = max_turns
        self.policy = policy
        self.approver = approver
        self.parent_context = parent_context
        self.system_prompt = system_prompt

    def run(self, task: str, *, workspace_root: Path) -> SubAgentResult:
        """Run a scoped task to completion in an isolated session."""
        temp_dir = tempfile.TemporaryDirectory()
        try:
            # Isolated session store
            session_store = SessionStore(Path(temp_dir.name))
            session = session_store.new()

            # Fresh ToolContext for subagent
            subagent_context = ToolContext(
                workspace_root=workspace_root,
                interrupt=self.interrupt,
                config=self.config,
                state={},
            )
            # Inherit memory store if available in parent context
            if self.parent_context and getattr(self.parent_context, "memory", None):
                subagent_context.memory = self.parent_context.memory

            # Construct subagent tool executor
            subagent_executor = ToolExecutor(
                registry=self.tool_registry,
                enabled=self.subagent_enabled_tools,
                interrupt=self.interrupt,
                context=subagent_context,
                policy=self.policy,
                approver=self.approver,
                checkpoint=None,
            )

            # Subagent specific ContextManager
            context_manager = SubAgentContextManager(
                self.config,
                summarizer=self.provider,
                workspace_root=workspace_root,
                system_prompt=self.system_prompt,
            )

            # Headless usage tracker
            usage_tracker = UsageTracker(self.config)

            # Isolated AgentLoop
            agent_loop = AgentLoop(
                context_manager=context_manager,
                provider=self.provider,
                tool_executor=subagent_executor,
                usage_tracker=usage_tracker,
                session_store=session_store,
                interrupt=self.interrupt,
                renderer=None,
                checkpoint=None,
                parallel_enabled=self.config.parallel_enabled,
                parallel_max_workers=self.config.parallel_max_workers,
                max_iterations=self.max_turns,
            )

            # Execute the turn
            turn_result = agent_loop.run_turn(session, task)

            # Extract final response from subagent
            final_text = ""
            for msg in reversed(session.messages):
                if msg.role == "model" and msg.text:
                    final_text = msg.text
                    break

            # Construct Usage result
            usage = Usage(
                input_tokens=turn_result.usage.turn_input_tokens,
                output_tokens=turn_result.usage.turn_output_tokens,
                estimated_cost=turn_result.usage.turn_cost,
            )

            return SubAgentResult(text=final_text, usage=usage)
        finally:
            temp_dir.cleanup()
