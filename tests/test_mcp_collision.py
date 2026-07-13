"""Property and unit tests for MCP tool name-collision resolution.

These tests exercise the pure collision-resolution logic in
:func:`forge.mcp_client.resolve_mcp_collisions` and its integration with
:func:`forge.mcp_client.register_mcp_tools` / :class:`ToolExecutor`. No live
MCP servers are required: the collision rule is a pure function over a set of
built-in names and an ordered list of discovered ``(server, tool)`` pairs.

# Feature: forge, Property 23: MCP name-collision resolution
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.interrupt import InterruptController
from forge.mcp_client import (
    McpToolAdapter,
    register_mcp_tools,
    resolve_mcp_collisions,
)
from forge.tools.base import Tool, ToolContext, ToolExecutor, ToolResult

# The recognized built-in tool names (a superset to draw the "reserved" set
# from). A discovered MCP tool sharing any of these names must be excluded in
# favor of the built-in (Req 16.6).
BUILTIN_POOL = ["read", "write", "edit", "shell", "search", "git", "planning"]

# Server names used when generating discovered tools. Multiple servers let us
# exercise the MCP-vs-MCP first-wins tie-break.
SERVER_POOL = ["srvA", "srvB", "srvC"]

# Tool names a server might expose: a deliberate mix of names that collide with
# built-ins ("read", "write") and MCP-only names. This guarantees Hypothesis
# generates both collision kinds (MCP-vs-built-in and MCP-vs-MCP).
TOOL_POOL = ["read", "write", "fetch", "deploy", "format", "lint"]


def _reference(
    builtins: set[str],
    discovered: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Independent reference implementation of the documented rule.

    Returns ``(accepted, excluded)``: a discovered pair is excluded when its
    tool name is a built-in (built-in wins) or when the same tool name was
    already claimed by an earlier server (first server wins). Everything else
    is accepted, preserving connection order.
    """

    builtins = set(builtins)
    accepted: list[tuple[str, str]] = []
    excluded: list[tuple[str, str]] = []
    claimed: dict[str, str] = {}
    for server, tool in discovered:
        if tool in builtins:
            excluded.append((server, tool))
            continue
        if tool in claimed:
            excluded.append((server, tool))
            continue
        claimed[tool] = server
        accepted.append((server, tool))
    return accepted, excluded


builtin_names_strat = st.sets(st.sampled_from(BUILTIN_POOL))
discovered_strat = st.lists(
    st.tuples(st.sampled_from(SERVER_POOL), st.sampled_from(TOOL_POOL)),
    max_size=24,
)


@settings(max_examples=10)
@given(builtins=builtin_names_strat, discovered=discovered_strat)
def test_mcp_name_collision_resolution(
    builtins: set[str],
    discovered: list[tuple[str, str]],
) -> None:
    """The resolver keeps the built-in for every colliding name, excludes the
    conflicting MCP tool with one warning each, and keeps non-colliding MCP
    tools (first server wins on MCP-vs-MCP collisions).

    Validates: Requirements 16.6
    """

    accepted, messages = resolve_mcp_collisions(builtins, discovered)
    expected_accepted, expected_excluded = _reference(builtins, discovered)

    # Resolution matches the documented rule, preserving connection order.
    assert accepted == expected_accepted

    # Exactly one warning per excluded tool.
    assert len(messages) == len(expected_excluded)

    # No accepted tool collides with a built-in (the built-in is retained).
    assert all(tool not in set(builtins) for _, tool in accepted)

    # Every built-in-colliding discovered pair is excluded.
    for server, tool in discovered:
        if tool in set(builtins):
            assert (server, tool) not in accepted

    # Accepted tool names are unique (no two MCP servers share a name).
    accepted_names = [tool for _, tool in accepted]
    assert len(accepted_names) == len(set(accepted_names))

    # Non-colliding MCP tools remain available: the first occurrence of each
    # non-built-in tool name is accepted, owned by its first server.
    first_seen: dict[str, tuple[str, str]] = {}
    for server, tool in discovered:
        if tool in set(builtins):
            continue
        first_seen.setdefault(tool, (server, tool))
    for pair in first_seen.values():
        assert pair in accepted


def test_builtin_collision_excludes_mcp_tool() -> None:
    """A single MCP tool sharing a built-in name is excluded with a warning."""

    accepted, messages = resolve_mcp_collisions(
        {"read", "write"}, [("docs", "read")]
    )
    assert accepted == []
    assert len(messages) == 1
    assert "read" in messages[0]
    assert "docs" in messages[0]


def test_mcp_vs_mcp_first_server_wins() -> None:
    """When two servers expose the same name, the first in order wins."""

    discovered = [("srvA", "fetch"), ("srvB", "fetch")]
    accepted, messages = resolve_mcp_collisions(set(), discovered)
    assert accepted == [("srvA", "fetch")]
    assert len(messages) == 1
    # The warning names the later, excluded server and the surviving one.
    assert "srvB" in messages[0]
    assert "srvA" in messages[0]


def test_no_collisions_accepts_all_in_order() -> None:
    """With no built-in or cross-server collisions every tool is accepted."""

    discovered = [("srvA", "fetch"), ("srvA", "deploy"), ("srvB", "lint")]
    accepted, messages = resolve_mcp_collisions({"read", "write"}, discovered)
    assert accepted == discovered
    assert messages == []


@dataclass
class _FakeBuiltin:
    """A minimal built-in :class:`Tool` used to populate the executor registry."""

    name: str
    description: str = "a built-in tool"
    parameters: dict = field(default_factory=dict)

    def validate(self, args: dict) -> str | None:
        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:  # pragma: no cover
        return ToolResult(ok=True, content=f"ran {self.name}")


def test_register_keeps_builtin_and_exposes_accepted_mcp_tools() -> None:
    """End-to-end (Req 16.6): the built-in is retained for a colliding name,
    the conflicting MCP tool is excluded, and accepted MCP tools are exposed to
    the Model alongside the built-ins.
    """

    builtins = {"read", "write"}
    discovered = [
        ("srv1", "read"),   # collides with built-in -> excluded
        ("srv1", "fetch"),  # accepted
        ("srv2", "write"),  # collides with built-in -> excluded
        ("srv2", "fetch"),  # MCP-vs-MCP duplicate -> excluded (srv1 wins)
    ]
    accepted, messages = resolve_mcp_collisions(builtins, discovered)
    assert accepted == [("srv1", "fetch")]
    assert len(messages) == 3

    # Build a registry of built-in tools, then merge accepted MCP tools.
    registry: dict[str, Tool] = {name: _FakeBuiltin(name) for name in builtins}
    enabled: set[str] = set(builtins)
    adapters = [
        McpToolAdapter(
            server=server,
            name=tool,
            description="an mcp tool",
            parameters={},
            client=None,  # run() is never invoked in this test
        )
        for server, tool in accepted
    ]
    register_mcp_tools(registry, enabled, adapters)

    # Built-in tools are retained (not overwritten by an MCP adapter).
    assert isinstance(registry["read"], _FakeBuiltin)
    assert isinstance(registry["write"], _FakeBuiltin)

    # The accepted MCP tool was merged in and is exposed to the Model.
    executor = ToolExecutor(
        registry=registry,
        enabled=enabled,
        interrupt=InterruptController(),
    )
    exposed = {spec.name for spec in executor.specs()}
    assert {"read", "write", "fetch"} <= exposed
    # The excluded MCP collisions never became available under a built-in name.
    assert isinstance(registry["read"], _FakeBuiltin)
