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
from forge.policy import ApprovalPolicy, Approver, Decision
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
    read_only:
        ``True`` when the tool does not affect anything outside the session
        (the approval policy treats such tools as never-needing-approval).
        Defaults to ``False``; concrete tools set it explicitly.
    """

    name: str
    description: str
    parameters: dict
    read_only: bool

    def validate(self, args: dict) -> str | None:
        """Return ``None`` when ``args`` are valid, else an error string.

        Validation must be side-effect free: when it returns an error the
        executor reports a validation error and never calls :meth:`run`.
        """
        ...

    def run(self, args: dict, ctx: "ToolContext") -> ToolResult:
        """Execute the tool with validated ``args`` and return a result."""
        ...

    def preview(self, args: dict, ctx: "ToolContext") -> str | None:
        """Optional: return a best-effort preview of the call's effect.

        Used by the approval system to show the user what a gated call is
        about to do (e.g. a unified diff for ``write``/``edit``). The default
        implementation returns ``None``; concrete tools that have something
        useful to show override it. The executor guards calls with
        :func:`getattr` so the protocol default is safe to omit.
        """
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
    calls within a session. ``memory`` is an optional reference to the
    :class:`~forge.memory.MemoryStore` for the memory tools (Feature F).
    The context is deliberately minimal but extensible.
    """

    workspace_root: Path
    interrupt: InterruptController
    config: Any | None = None
    state: dict = field(default_factory=dict)
    memory: Any | None = None
    provider: Any | None = None
    tool_registry: dict[str, Any] = field(default_factory=dict)
    policy: Any | None = None
    approver: Any | None = None


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
    policy:
        Optional :class:`~forge.policy.ApprovalPolicy` consulted before each
        call. When ``None`` (the default) no policy gating is applied and the
        executor behaves exactly as before Phase 2.
    approver:
        Optional :class:`~forge.policy.Approver` consulted when the policy
        requires approval. When ``None`` and a prompt is needed, the call is
        denied (the safe default for any path that forgot to wire an
        approver). Auto-approve / deny non-interactive approvers are wired
        explicitly by the bootstrap / headless paths.
    checkpoint:
        Optional :class:`~forge.checkpoint.CheckpointStore` consulted before
        mutating ``write``/``edit`` calls so a later ``/undo`` can restore the
        pre-mutation state. When ``None`` no checkpoint is captured.
    """

    def __init__(
        self,
        registry: dict[str, Tool],
        enabled: set[str],
        interrupt: InterruptController,
        context: ToolContext | None = None,
        policy: ApprovalPolicy | None = None,
        approver: Approver | None = None,
        checkpoint: Any | None = None,
    ) -> None:
        self._registry = registry
        self._enabled = set(enabled)
        self._interrupt = interrupt
        self._context = context or ToolContext(
            workspace_root=Path.cwd(), interrupt=interrupt
        )
        self._policy = policy
        self._approver = approver
        self._context.approver = approver
        self._checkpoint = checkpoint

    def set_approver(self, approver: Approver | None) -> None:
        """Replace the approver (used by bootstrap after the Repl is built).

        The executor is constructed before the Repl (which implements
        :class:`Approver`) exists, so the bootstrap path wires the approver
        in afterwards via this setter. Passing ``None`` disables prompting;
        the executor will deny any call that needs approval.
        """

        self._approver = approver
        self._context.approver = approver

    def set_memory(self, memory: Any) -> None:
        """Set the memory store reference for the memory tools (Phase 3).

        The memory store is constructed during bootstrap and needs to be
        wired into the tool context so the memory tools can access it.
        """
        self._context.memory = memory

    # -- exposure ------------------------------------------------------------

    def _exposed_names(self) -> set[str]:
        """Names that are both registered and enabled (recognized ∩ enabled)."""
        return set(self._registry) & self._enabled

    def is_exposed(self, name: str) -> bool:
        """Return whether ``name`` is exposed to the Model and runnable."""
        return name in self._registry and name in self._enabled

    def get_tool(self, name: str) -> Tool | None:
        """Return the registered :class:`Tool` for ``name``, or ``None``.

        A small public accessor so collaborators (e.g. the agent loop's
        parallel-eligibility check) can inspect a registered tool's attributes
        without reaching into the private registry.
        """
        return self._registry.get(name)

    @property
    def registry(self) -> dict[str, Tool]:
        """The ``name -> Tool`` registry (read-only accessor for collaborators).

        Exposed so post-turn phases (e.g. the review agent) can build a
        subagent over the same toolset without reaching into a private field.
        """
        return self._registry

    @property
    def context(self) -> ToolContext:
        """The shared :class:`ToolContext` (read-only accessor for collaborators)."""
        return self._context

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

        # 3a. Approval-policy gating (Phase 2, Feature B). Side-effect-free:
        # forbidden / denied results never run the tool or capture a
        # checkpoint. Read-only tools pass through both checks regardless of
        # mode.
        read_only = bool(getattr(tool, "read_only", False))
        if self._policy is not None:
            if self._policy.is_forbidden(
                call.name, call.args, read_only=read_only
            ):
                return ToolResult(
                    ok=False,
                    content="",
                    error=(
                        f"Tool '{call.name}' is forbidden in "
                        f"{self._policy.mode.value} mode."
                    ),
                    meta={"forbidden": True},
                )
            if self._policy.requires_approval(
                call.name, call.args, read_only=read_only
            ):
                preview = self._preview(tool, call.args)
                if self._approver is None:
                    # No approver wired: safe default is to deny.
                    decision: Decision = Decision.DENY
                else:
                    decision = self._approver.request(
                        call.name, call.args, preview
                    )
                if decision is Decision.DENY:
                    return ToolResult(
                        ok=False,
                        content="",
                        error=(
                            f"Tool '{call.name}' was not approved."
                        ),
                        meta={"denied": True},
                    )
                # APPROVE / APPROVE_ALWAYS both proceed. APPROVE_ALWAYS
                # bookkeeping (the session-scoped memory) is owned by the
                # Approver itself; the executor does not track it.

        # 3b. Checkpoint capture (Phase 2, Feature C). Only file-mutating
        # tools that resolve into the workspace are checkpointed; out-of-scope
        # paths are silently skipped (the tool will report out-of-scope on its
        # own and there is nothing to restore).
        if self._checkpoint is not None and call.name in ("write", "edit"):
            self._checkpoint.snapshot_before(
                call.args.get("path"), self._context
            )

        # 4. Run the tool. An unexpected exception from a tool (a bug in a
        # built-in, or anything a third-party/MCP tool raises) must not abort
        # the agent loop: converting it into a structured failure result keeps
        # the turn alive, ensures the session is still persisted, and makes the
        # sequential and parallel execution paths behave identically.
        # KeyboardInterrupt / SystemExit are intentionally allowed to propagate
        # so Ctrl-C and interpreter shutdown are never swallowed.
        try:
            result = tool.run(call.args, self._context)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001 - convert to structured failure
            return ToolResult(
                ok=False,
                content="",
                error=f"Tool '{call.name}' raised an unexpected error: {exc}",
                meta={"exception": True},
            )

        # 5. Interrupt check after running.
        if self._interrupt.check():
            return self._interrupted_result(call.name)

        # 6. Return the tool's result.
        return result

    def _preview(self, tool: Tool, args: dict) -> str | None:
        """Compute a best-effort preview via the tool's optional hook.

        A tool that does not implement ``preview``, or whose hook raises, is
        treated as having no preview (the approver falls back to the args
        summary). Preview failures must never break a gated call.
        """

        hook = getattr(tool, "preview", None)
        if not callable(hook):
            return None
        try:
            value = hook(args, self._context)
        except Exception:  # noqa: BLE001 - preview must never break a call
            return None
        if isinstance(value, str):
            return value
        return None

    @staticmethod
    def _interrupted_result(name: str) -> ToolResult:
        return ToolResult(
            ok=False,
            content="",
            error=f"Tool '{name}' was interrupted.",
            meta={"interrupted": True},
        )
