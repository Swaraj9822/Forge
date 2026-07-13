"""The delegate tool for executing subagents."""

from __future__ import annotations

from forge.tools.base import Tool, ToolContext, ToolResult
from forge.subagent import SubAgentRunner


class DelegateTool(Tool):
    """Tool that delegates a scoped task to a fresh subagent."""

    name: str = "delegate"
    description: str = (
        "Delegate a scoped task to a fresh sub-agent with its own context window "
        "and restricted tools. Returns the sub-agent's final answer."
    )
    # Read-only because it does not directly mutate the parent's workspace;
    # any sub-agent modifications are gated by their own tool executor / policy.
    read_only: bool = True
    parameters: dict = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The description of the scoped task to be performed by the sub-agent.",
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of tool names the sub-agent is allowed to use. Defaults to standard read-only tools.",
            },
            "max_turns": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional maximum number of turns for the sub-agent. Defaults to 4.",
            },
        },
        "required": ["task"],
    }

    def validate(self, args: dict) -> str | None:
        """Validate delegation arguments."""
        if "task" not in args or not isinstance(args["task"], str) or not args["task"].strip():
            return "Parameter 'task' is required and must be a non-empty string."
        if "tools" in args and not isinstance(args["tools"], list):
            return "Parameter 'tools' must be a list of tool name strings."
        if "max_turns" in args:
            if isinstance(args["max_turns"], bool) or not isinstance(args["max_turns"], int) or args["max_turns"] < 1:
                return "Parameter 'max_turns' must be an integer >= 1."
        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Run the subagent runner and return the result."""
        task = args["task"]

        default_allowed = getattr(ctx.config, "subagents_default_tools", ["read", "search", "repo_index", "search_memory"])
        tools_list = args.get("tools")
        if tools_list is None:
            allowed_tools = set(default_allowed)
        else:
            allowed_tools = set(tools_list)

        max_turns = args.get("max_turns") or getattr(ctx.config, "subagents_max_turns", 4)

        runner = SubAgentRunner(
            provider=ctx.provider,
            config=ctx.config,
            interrupt=ctx.interrupt,
            tool_registry=ctx.tool_registry,
            allowed_tools=allowed_tools,
            max_turns=max_turns,
            policy=ctx.policy,
            approver=ctx.approver,
            parent_context=ctx,
        )

        try:
            result = runner.run(task, workspace_root=ctx.workspace_root)
            meta = {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "estimated_cost": result.usage.estimated_cost,
                # Consumed by AgentLoop to fold delegated tokens into the
                # parent turn's usage/cost (Phase 5 usage aggregation).
                "subagent_usage": {
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                },
            }
            return ToolResult(ok=True, content=result.text, meta=meta)
        except Exception as exc:
            return ToolResult(ok=False, content="", error=str(exc))
