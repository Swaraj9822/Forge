"""Property-based test for tool exposure and availability.

# Feature: forge, Property 4: Only enabled, recognized tools are exposed and runnable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.interrupt import InterruptController
from forge.session import ToolCall
from forge.tools.base import Tool, ToolContext, ToolExecutor, ToolResult

# A fixed pool of "recognized" tool names. These are the only names a registry
# is built from, so they stand in for the recognized built-in (plus accepted
# MCP) tools.
RECOGNIZED_POOL = ["alpha", "beta", "gamma", "delta"]

# Names that are deliberately NOT in any registry. Used to populate `enabled`
# with unrecognized names (Req 11.7/11.8) and to probe the unavailable path.
UNKNOWN_POOL = ["epsilon", "zeta", "eta", "unknown_tool"]

# The full universe of names we probe execution against: every recognized name
# plus every unknown name. Each is checked for the correct exposed/unavailable
# behavior.
NAME_UNIVERSE = RECOGNIZED_POOL + UNKNOWN_POOL


@dataclass
class RecordingTool:
    """A minimal :class:`Tool` that records a side effect when it runs.

    ``run`` appends to the shared ``log`` list so a test can assert that NO
    tool ran when a non-exposed name is invoked (the "no side effects"
    guarantee of Property 4). ``validate`` always passes so the only thing that
    can prevent ``run`` is the exposure check.
    """

    name: str
    log: list[str]
    description: str = "a recording test tool"
    parameters: dict = field(default_factory=dict)

    def validate(self, args: dict) -> str | None:
        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        self.log.append(self.name)
        return ToolResult(ok=True, content=f"ran {self.name}")


# Each `enabled` set is an arbitrary subset of recognized + unknown names, so
# enabled may contain unrecognized names (which must never be exposed) and may
# omit recognized names (which must then not be exposed).
enabled_sets = st.sets(st.sampled_from(RECOGNIZED_POOL + UNKNOWN_POOL))


@settings(max_examples=10)
@given(enabled=enabled_sets)
def test_only_enabled_recognized_tools_are_exposed_and_runnable(
    enabled: set[str],
) -> None:
    """Exposed set equals recognized ∩ enabled; non-exposed names yield an
    unavailable result with no side effects, and exposed names are runnable.

    Validates: Requirements 4.6, 11.7, 11.8
    """

    log: list[str] = []
    registry: dict[str, Tool] = {
        name: RecordingTool(name=name, log=log) for name in RECOGNIZED_POOL
    }

    interrupt = InterruptController()
    context = ToolContext(workspace_root=Path.cwd(), interrupt=interrupt)
    executor = ToolExecutor(
        registry=registry,
        enabled=enabled,
        interrupt=interrupt,
        context=context,
    )

    # --- Exposure: specs() == recognized ∩ enabled -----------------------
    expected_exposed = set(registry) & enabled
    exposed_names = {spec.name for spec in executor.specs()}
    assert exposed_names == expected_exposed
    # specs() is sorted by name for deterministic ordering.
    spec_names = [spec.name for spec in executor.specs()]
    assert spec_names == sorted(spec_names)

    # --- Invocation behavior across the full name universe ----------------
    for i, name in enumerate(NAME_UNIVERSE):
        log.clear()
        before = list(log)
        result = executor.execute(ToolCall(id=str(i), name=name, args={}))

        if name in expected_exposed:
            # Exposed tools ARE runnable: the tool ran (side effect recorded)
            # and the result is ok.
            assert result.ok is True
            assert log == [name]
        else:
            # Non-exposed names (unknown, or recognized-but-disabled) yield an
            # "unavailable" result with NO side effects.
            assert result.ok is False
            assert result.meta.get("unavailable") is True or (
                result.error is not None and "unavailable" in result.error.lower()
            )
            # No tool ran: the side-effect log is unchanged (empty).
            assert log == before == []
