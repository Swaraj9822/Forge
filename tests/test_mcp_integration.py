"""Integration tests for :class:`forge.mcp_client.McpClient` over stdio.

These tests launch a real, minimal MCP server as a subprocess and drive it
through the same public API Forge uses at runtime: :meth:`McpClient.connect_all`
(connect + discover), :meth:`McpClient.call` (forward a tool call), and
:meth:`McpClient.close` (teardown). They cover, against a live stdio transport:

* connect + tool discovery (Req 16.1) and exposure of discovered tools (16.2),
* tool-call forwarding and result return (Req 16.3),
* a connect failure warning that does not abort startup (Req 16.4), and
* call-time error / unreachable handling returning a failure result (Req 16.5).

The official ``mcp`` SDK is an optional/runtime dependency. When it is not
installed the whole module skips gracefully rather than failing.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

# The integration tests require the `mcp` SDK both in this process (for the
# client) and in the subprocess (for the stub server, launched with the same
# interpreter). Skip the entire module when it is unavailable.
pytest.importorskip("mcp")

from forge.config import McpServerConfig  # noqa: E402
from forge.interrupt import InterruptController  # noqa: E402
from forge.mcp_client import McpClient, McpToolAdapter  # noqa: E402
from forge.tools.base import ToolContext  # noqa: E402

# A minimal MCP server exposing three tools over stdio:
#   - echo(text): returns its argument (forwarding / round-trip),
#   - add(a, b):  returns a sum (a second discoverable tool),
#   - boom():     always raises (exercises the call-time error path, Req 16.5).
# Launched as a subprocess with the current interpreter via FastMCP's default
# stdio transport.
STUB_SERVER_SOURCE = '''\
from mcp.server.fastmcp import FastMCP

server = FastMCP("forge-stub")


@server.tool()
def echo(text: str) -> str:
    """Echo the provided text back to the caller."""
    return text


@server.tool()
def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


@server.tool()
def boom() -> str:
    """Always raise, to exercise the MCP error path."""
    raise RuntimeError("boom: intentional failure")


if __name__ == "__main__":
    # Default transport is stdio, matching what McpClient connects over.
    server.run()
'''


@pytest.fixture(scope="module")
def stub_server_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialize the stub MCP server script and return its path."""
    path = tmp_path_factory.mktemp("mcp_stub") / "stub_server.py"
    path.write_text(STUB_SERVER_SOURCE, encoding="utf-8")
    return path


@pytest.fixture
def make_client():
    """Provide a factory for McpClients and tear every one of them down."""
    clients: list[McpClient] = []

    def _make(connect_timeout_s: int = 15) -> McpClient:
        client = McpClient(connect_timeout_s=connect_timeout_s)
        clients.append(client)
        return client

    yield _make

    for client in clients:
        try:
            client.close()
        except Exception:  # pragma: no cover - best-effort teardown
            pass


def _stub_config(stub_server_path: Path, name: str = "stub") -> McpServerConfig:
    return McpServerConfig(
        name=name,
        command=sys.executable,
        args=[str(stub_server_path)],
    )


def test_connect_discovers_and_adapts_tools(
    stub_server_path: Path, make_client
) -> None:
    """connect_all connects over stdio and adapts the server's tools (16.1/16.2)."""
    client = make_client()
    tools = client.connect_all([_stub_config(stub_server_path)], builtin_names=set())

    by_name = {tool.name: tool for tool in tools}
    assert {"echo", "add", "boom"} <= set(by_name)
    for tool in tools:
        assert isinstance(tool, McpToolAdapter)
        assert tool.server == "stub"
    # Discovered schema is carried through for the Model (echo takes `text`).
    assert isinstance(by_name["echo"].parameters, dict)


def test_tool_call_forwarding_returns_result(
    stub_server_path: Path, make_client
) -> None:
    """call() forwards to the server and returns its response (Req 16.3)."""
    client = make_client()
    tools = client.connect_all([_stub_config(stub_server_path)], builtin_names=set())

    # Forward directly through the client.
    result = client.call("stub", "echo", {"text": "hello forge"})
    assert result.ok is True
    assert "hello forge" in result.content

    # And through the adapter's Tool.run, the path the executor uses.
    echo_tool = next(tool for tool in tools if tool.name == "echo")
    ctx = ToolContext(workspace_root=Path.cwd(), interrupt=InterruptController())
    adapted = echo_tool.run({"text": "via adapter"}, ctx)
    assert adapted.ok is True
    assert "via adapter" in adapted.content


def test_connect_failure_warns_and_continues(
    stub_server_path: Path, make_client
) -> None:
    """A server that cannot be launched warns and the rest still connect (16.4)."""
    client = make_client(connect_timeout_s=10)
    broken = McpServerConfig(
        name="broken",
        command="forge_nonexistent_command_zzz",
        args=[],
    )
    good = _stub_config(stub_server_path)

    with pytest.warns(UserWarning, match="broken"):
        tools = client.connect_all([broken, good], builtin_names=set())

    # The failure did not abort startup: the healthy server's tools are present.
    assert {"echo", "add", "boom"} <= {tool.name for tool in tools}


def test_call_to_failing_tool_returns_error_result(
    stub_server_path: Path, make_client
) -> None:
    """A tool that errors on the server yields a failure result (Req 16.5)."""
    client = make_client()
    client.connect_all([_stub_config(stub_server_path)], builtin_names=set())

    result = client.call("stub", "boom", {})
    assert result.ok is False
    assert result.meta.get("mcp_error") is True


def test_call_to_unknown_server_is_unreachable(
    stub_server_path: Path, make_client
) -> None:
    """Calling a server that was never connected returns a failure result (16.5)."""
    client = make_client()
    client.connect_all([_stub_config(stub_server_path)], builtin_names=set())

    result = client.call("not-connected", "echo", {"text": "x"})
    assert result.ok is False
    assert result.meta.get("mcp_error") is True


def test_call_after_close_is_unreachable(
    stub_server_path: Path, make_client
) -> None:
    """After teardown the server is unreachable and call() fails gracefully (16.5)."""
    client = make_client()
    client.connect_all([_stub_config(stub_server_path)], builtin_names=set())
    client.close()

    result = client.call("stub", "echo", {"text": "x"})
    assert result.ok is False
    assert result.meta.get("mcp_error") is True
