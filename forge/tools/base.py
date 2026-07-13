"""The Tool protocol and the :class:`ToolExecutor`.

This module defines the contract every built-in (and adapted MCP) tool
implements and the single executor the Agent_Loop uses to run tool calls. The
pieces are intentionally small and dependency-light so each built-in tool
(``read``/``write``/``edit``/``shell``/``search``/``git``/``planning``) and the
MCP adapters can implement the same shape.

Components
----------
* :class:`ToolResult` - the structured result returned to the Model after a
  tool runs (mirrors :class:`forge.session.ToolResultRecord`'s payload).
* :class:`Tool` - a ``Protocol`` describing the attributes and methods every
  tool provides (``name``, ``description``, ``parameters``, ``validate``,
  ``run``).
* :class:`ToolContext` - the execution context shared with a tool's ``run``;
  carries the workspace root, the interrupt controller, and optional
  config-derived state. Kept minimal but extensible for later tools.
* :class:`ToolSpec` - the model-facing description of a tool (name,
  description, JSON-schema parameters) used to advertise tools to the Model.
* :class:`ToolExecutor` - owns the ``name -> Tool`` registry, advertises the
  exposed tools via :meth:`ToolExecutor.specs`, and runs a tool call via
  :meth:`ToolExecutor.execute` with the unavailable / interrupt / validation
  guarantees required by requirements 4.1, 4.6, 4.7, 11.8 and 16.2.

Tool-exposure rule (Property 4)
-------------------------------
A tool is exposed to the Model -- and therefore runnable -- *iff* its name is
both present in the registry (a recognized built-in or an accepted MCP tool)
**and** present in the ``enabled`` set. In set terms the exposed set is

    exposed = set(registry) & enabled

This is the "recognized ∩ enabled" model from Property 4. Built-in tools are
gated by the configured ``enabled_tools``; accepted MCP tools are made
exposable by registering them and adding their names to ``enabled`` when the
MCP client wires them in (task 22.1). Any name outside the exposed set -- an
unknown tool, or a recognized-but-disabled tool -- resolves to an
"unavailable" :class:`ToolResult` with no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from forge.interrupt import InterruptController
from forge.session import ToolCall

__all__ = [
    "ToolResult",
    "Tool",
    "ToolContext",
    "ToolSpec",
    "ToolExecutor",
]


@dataclass(frozen=True)
class ToolResult:
    """The structured output returned to the Model after a Tool runs.

    ``ok`` flags success; ``content`` carries the human/model-readable output
    (or an empty string); ``error`` carries an error description when
    ``ok`` is ``False``; ``meta`` carries structured flags such as
    ``{"truncated": True}`` or ``{"unavailable": True}``.
    """

    ok: bool
    content: str
    error: str | None = None
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Tool(Protocol):
    """Contract every tool (built-in or adapted MCP) implements.

    Attributes
    ----------
    name:
        Unique identifier the Model uses to invoke the tool.
    description:
        Natural-language description advertised to the Model.
    parameters:
        JSON schema (a ``dict``) describing the tool's arguments for the Model.
    """

    name: str
    description: str
    parameters: dict

    def validate(self, args: dict) -> str | None:
        """Return ``None`` when ``args`` are valid, else an error string.

        Validation must be side-effect free: when it returns an error the
        executor reports a validation error and never calls :meth:`run`.
        """
        ...

    def run(self, args: dict, ctx: "ToolContext") -> ToolResult:
        """Execute the tool with validated ``args`` and return a result."""
        ...


@dataclass
class ToolContext:
    """Execution context shared with a tool's :meth:`Tool.run`.

    Carries the workspace root (the security boundary for file/search tools)
    and the :class:`InterruptController` so long-running tools (shell) can poll
    for interrupts. ``config`` is left loosely typed (``Any``) to avoid a hard
    import cycle with :mod:`forge.config`; it carries config-derived limits
    (timeouts, caps) for tools that need them. ``state`` is a small
    session-scoped bag later tools (e.g. planning) use to retain data across
    calls within a session. The context is deliberately minimal but
    extensible.
    """

    workspace_root: Path
    interrupt: InterruptController
    config: Any | None = None
    state: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    """The model-facing description of a tool.

    A small, serializable view of a :class:`Tool` carrying only what the Model
    needs to decide whether and how to invoke it.
    """

    name: str
    description: str
    parameters: dict


class ToolExecutor:
    """Owns the tool registry and runs tool calls with the required guarantees.

    Parameters
    ----------
    registry:
        Mapping of tool name to :class:`Tool`. Holds recognized built-in tools
        and any accepted MCP tools registered later.
    enabled:
        Names enabled by configuration (plus accepted MCP tool names). Combined
        with ``registry`` to determine the exposed set; see the module
        docstring for the exposure rule.
    interrupt:
        Shared interrupt controller, checked before and after each tool run.
    context:
        The :class:`ToolContext` passed to every :meth:`Tool.run`. Accepted
        here (rather than rebuilt per call) so wiring stays explicit and the
        same workspace/config/state is shared across calls within a session.
        When omitted a minimal context rooted at the current working directory
        is constructed, which keeps the executor usable in tests and simple
        wiring before the full app context exists.
    """

    def __init__(
        self,
        registry: dict[str, Tool],
        enabled: set[str],
        interrupt: InterruptController,
        context: ToolContext | None = None,
    ) -> None:
        self._registry = registry
        self._enabled = set(enabled)
        self._interrupt = interrupt
        self._context = context or ToolContext(
            workspace_root=Path.cwd(), interrupt=interrupt
        )

    # -- exposure ------------------------------------------------------------

    def _exposed_names(self) -> set[str]:
        """Names that are both registered and enabled (recognized ∩ enabled)."""
        return set(self._registry) & self._enabled

    def is_exposed(self, name: str) -> bool:
        """Return whether ``name`` is exposed to the Model and runnable."""
        return name in self._registry and name in self._enabled

    def specs(self) -> list[ToolSpec]:
        """Return a :class:`ToolSpec` for each exposed tool.

        Only tools that are both registered and enabled are advertised to the
        Model (Property 4). Results are sorted by name for deterministic
        ordering.
        """
        exposed = self._exposed_names()
        return [
            ToolSpec(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for name, tool in sorted(self._registry.items())
            if name in exposed
        ]

    # -- execution -----------------------------------------------------------

    def execute(self, call: ToolCall) -> ToolResult:
        """Run a single tool call and return a :class:`ToolResult`.

        Steps, in order:

        1. Resolve ``call.name`` against the exposed set. An unknown or
           disabled name yields an "unavailable" result with no side effects
           (Req 4.6, 11.8).
        2. Check the interrupt before running; a tripped interrupt yields an
           "interrupted" result without running the tool (supports task 5.4).
        3. Run ``tool.validate(args)``; a non-``None`` error string yields a
           validation-error result with no side effects (Req 4.7, Property 5).
        4. Run ``tool.run(args, ctx)``.
        5. Check the interrupt after running; a tripped interrupt yields an
           "interrupted" result.
        6. Return the tool's result.
        """
        # 1. Resolve against the exposed set (recognized ∩ enabled).
        if not self.is_exposed(call.name):
            return ToolResult(
                ok=False,
                content="",
                error=f"Tool '{call.name}' is unavailable.",
                meta={"unavailable": True},
            )

        tool = self._registry[call.name]

        # 2. Interrupt check before running -- no side effects when tripped.
        if self._interrupt.check():
            return self._interrupted_result(call.name)

        # 3. Validate; a validation error must not run the tool.
        error = tool.validate(call.args)
        if error is not None:
            return ToolResult(
                ok=False,
                content="",
                error=error,
                meta={"validation_error": True},
            )

        # 4. Run the tool.
        result = tool.run(call.args, self._context)

        # 5. Interrupt check after running.
        if self._interrupt.check():
            return self._interrupted_result(call.name)

        # 6. Return the tool's result.
        return result

    @staticmethod
    def _interrupted_result(name: str) -> ToolResult:
        return ToolResult(
            ok=False,
            content="",
            error=f"Tool '{name}' was interrupted.",
            meta={"interrupted": True},
        )
