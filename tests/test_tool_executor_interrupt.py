"""Unit tests for ToolExecutor interrupt handling (task 5.4).

These tests exercise the ToolExecutor's interrupt guarantees from the design
(Requirements 4.3, 4.4): an interrupt that is tripped *before* a tool runs must
stop the tool from running at all (no side effects), and an interrupt that is
tripped *during/after* a tool runs must be converted into an "interrupted"
Tool_Result after the executor's post-run check.

The implementation under test lives in ``forge/tools/base.py``:
``ToolExecutor.execute`` checks the interrupt before validating/running and
again after running, returning a ToolResult with ``meta={"interrupted": True}``
when the interrupt is tripped at either point.
"""

from __future__ import annotations

from pathlib import Path

from forge.interrupt import InterruptController
from forge.session import ToolCall
from forge.tools.base import ToolContext, ToolExecutor, ToolResult


class FakeTool:
    """A minimal Tool implementation that records when its ``run`` executes.

    ``side_effects`` is an external log appended to whenever ``run`` is called,
    letting tests assert whether the tool actually ran. An optional ``on_run``
    hook lets a test trip the interrupt from inside ``run`` to exercise the
    executor's post-run interrupt check.
    """

    def __init__(
        self,
        name: str = "fake",
        *,
        side_effects: list[str] | None = None,
        on_run=None,
    ) -> None:
        self.name = name
        self.description = "A fake tool for interrupt tests."
        self.parameters: dict = {"type": "object", "properties": {}}
        self.side_effects = side_effects if side_effects is not None else []
        self._on_run = on_run

    def validate(self, args: dict) -> str | None:
        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        if self._on_run is not None:
            self._on_run()
        self.side_effects.append("ran")
        return ToolResult(ok=True, content="done")


def _make_executor(tool: FakeTool, interrupt: InterruptController) -> ToolExecutor:
    context = ToolContext(workspace_root=Path.cwd(), interrupt=interrupt)
    return ToolExecutor(
        registry={tool.name: tool},
        enabled={tool.name},
        interrupt=interrupt,
        context=context,
    )


def test_interrupt_tripped_before_run_yields_interrupted_and_no_side_effects():
    """A pre-run tripped interrupt returns an interrupted result; tool never runs."""
    interrupt = InterruptController()
    log: list[str] = []
    tool = FakeTool(side_effects=log)
    executor = _make_executor(tool, interrupt)

    # Trip the interrupt before executing the call.
    interrupt.begin_turn()
    interrupt.trip()

    result = executor.execute(ToolCall(id="1", name="fake", args={}))

    assert result.ok is False
    assert result.meta.get("interrupted") is True or "interrupt" in (
        result.error or ""
    ).lower()
    # The tool's run must never have executed: no side effects recorded.
    assert log == []


def test_interrupt_tripped_during_run_is_converted_after_run():
    """A tool that trips the interrupt itself still runs, but the executor
    converts the post-run tripped interrupt into an interrupted result."""
    interrupt = InterruptController()
    interrupt.begin_turn()
    log: list[str] = []

    # The tool trips the interrupt from inside run(), simulating a Ctrl-C that
    # lands while the tool executes. run() still completes and returns ok.
    tool = FakeTool(side_effects=log, on_run=interrupt.trip)
    executor = _make_executor(tool, interrupt)

    result = executor.execute(ToolCall(id="2", name="fake", args={}))

    # The tool DID run (side effect recorded) ...
    assert log == ["ran"]
    # ... but the executor's after-run check converts it to interrupted.
    assert result.ok is False
    assert result.meta.get("interrupted") is True or "interrupt" in (
        result.error or ""
    ).lower()


def test_no_interrupt_runs_tool_and_returns_ok():
    """Control case: with no interrupt, a normal enabled tool runs and is ok."""
    interrupt = InterruptController()
    interrupt.begin_turn()
    log: list[str] = []
    tool = FakeTool(side_effects=log)
    executor = _make_executor(tool, interrupt)

    result = executor.execute(ToolCall(id="3", name="fake", args={}))

    assert result.ok is True
    assert log == ["ran"]
    assert result.meta.get("interrupted") is None
