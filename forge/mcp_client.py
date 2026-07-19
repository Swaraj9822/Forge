"""MCP (Model Context Protocol) client for Forge.

This module connects Forge to external MCP servers, discovers the tools each
server exposes, adapts those tools to Forge's :class:`~forge.tools.base.Tool`
protocol, and forwards tool calls to the owning server -- so MCP tools live in
the same registry as the built-in tools and the rest of Forge never needs to
know whether a tool is built-in or remote. (Requirements 16.1-16.6.)

Async <-> sync bridge
---------------------
The official ``mcp`` SDK is async (anyio/asyncio): the stdio client and
``ClientSession`` are async context managers and ``call_tool`` is a coroutine.
Forge's tool execution, by contrast, is synchronous (``Tool.run`` is a plain
method). To bridge the two cleanly without leaking ``async``/``await`` into the
agent loop, :class:`McpClient` runs a single dedicated asyncio event loop on a
background daemon thread for its whole lifetime. The synchronous public methods
(:meth:`McpClient.connect_all`, :meth:`McpClient.call`, :meth:`McpClient.close`)
submit coroutines to that loop with :func:`asyncio.run_coroutine_threadsafe` and
block on the returned :class:`concurrent.futures.Future`.

Each connected server is owned by one long-lived "runner" coroutine that enters
the stdio + session context managers, signals readiness with the discovered
tools, then parks on a close event until shutdown. Keeping the enter and exit of
those context managers inside a single asyncio task avoids anyio's
"cancel scope exited in a different task" hazard and keeps the stdio session
alive for the lifetime of the client (it must stay open so later
:meth:`call` invocations can reach the server).

Name-collision resolution
--------------------------
Two kinds of collision are resolved by :func:`resolve_mcp_collisions`, a pure
function that needs no live servers (so it is unit/property testable in
isolation -- Property 23):

* **MCP vs. built-in** -- if a discovered MCP tool shares a name with a built-in
  (reserved) tool, the built-in is kept and the MCP tool is excluded, with a
  warning naming the conflicting tool and its server (Req 16.6).
* **MCP vs. MCP** -- if two MCP servers expose the same tool name, the first
  server in connection order wins and the later one is excluded with a warning.
  This is the documented deterministic tie-break rule.

App wiring (task 23.1)
----------------------
:meth:`connect_all` returns a list of adapted :class:`~forge.tools.base.Tool`
objects but does not itself mutate any executor. ``app.py`` merges them into the
:class:`~forge.tools.base.ToolExecutor` registry: for each returned tool it sets
``registry[tool.name] = tool`` and adds ``tool.name`` to the ``enabled`` set, so
the accepted MCP tools become exposed and runnable alongside the built-ins (the
executor exposes ``set(registry) & enabled``). The convenience helper
:func:`register_mcp_tools` performs exactly that merge.
"""

from __future__ import annotations

import asyncio
import threading
import warnings
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Iterable

from forge.config import McpServerConfig
from forge.tools.base import Tool, ToolContext, ToolResult

__all__ = [
    "McpClient",
    "McpToolAdapter",
    "resolve_mcp_collisions",
    "register_mcp_tools",
]

# --------------------------------------------------------------------------- #
# Defensive import of the `mcp` SDK.
#
# The module must import cleanly even when `mcp` is not installed; the hard
# dependency is only required at connect time. Guarding the import here lets the
# rest of Forge import `forge.mcp_client` (and the pure collision-resolution
# logic / the adapter type) regardless of whether the SDK is present.
# --------------------------------------------------------------------------- #
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _MCP_AVAILABLE = True
    _MCP_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised only without `mcp`
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    _MCP_AVAILABLE = False
    _MCP_IMPORT_ERROR = exc


# --------------------------------------------------------------------------- #
# Pure collision-resolution logic (Property 23) -- no live servers required.
# --------------------------------------------------------------------------- #


def resolve_mcp_collisions(
    builtin_names: Iterable[str],
    discovered: Iterable[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Resolve MCP tool-name collisions deterministically (Req 16.6).

    Parameters
    ----------
    builtin_names:
        The set of built-in / reserved tool names. A discovered MCP tool whose
        name appears here is excluded in favor of the built-in.
    discovered:
        ``(server_name, tool_name)`` pairs in connection order. Order matters:
        when two servers expose the same tool name, the earlier pair wins.

    Returns
    -------
    ``(accepted, warnings)`` where ``accepted`` is the ordered list of
    ``(server_name, tool_name)`` pairs that survive resolution and ``warnings``
    is a list of human-readable warning strings describing every exclusion (one
    per excluded tool). The function is pure: it performs no I/O and emits no
    warnings itself, so callers decide how to surface them.
    """

    builtins = set(builtin_names)
    accepted: list[tuple[str, str]] = []
    messages: list[str] = []
    claimed_by: dict[str, str] = {}

    for server_name, tool_name in discovered:
        if tool_name in builtins:
            messages.append(
                f"MCP tool '{tool_name}' from server '{server_name}' conflicts "
                f"with a built-in tool; keeping the built-in and excluding the "
                f"MCP tool."
            )
            continue
        if tool_name in claimed_by:
            messages.append(
                f"MCP tool '{tool_name}' from server '{server_name}' conflicts "
                f"with the same tool already provided by server "
                f"'{claimed_by[tool_name]}'; keeping the first and excluding "
                f"this one."
            )
            continue
        claimed_by[tool_name] = server_name
        accepted.append((server_name, tool_name))

    return accepted, messages


def register_mcp_tools(
    registry: dict[str, Tool],
    enabled: set[str],
    tools: Iterable[Tool],
) -> None:
    """Merge accepted MCP ``tools`` into an executor's ``registry``/``enabled``.

    Mutates ``registry`` (``name -> Tool``) and ``enabled`` (name set) in place
    so the MCP tools become exposed and runnable alongside the built-ins. This
    is the merge ``app.py`` performs after :meth:`McpClient.connect_all`
    (task 23.1).
    """

    for tool in tools:
        registry[tool.name] = tool
        enabled.add(tool.name)


# --------------------------------------------------------------------------- #
# Tool adapter: wraps a discovered MCP tool as a Forge Tool.
# --------------------------------------------------------------------------- #


class McpToolAdapter:
    """Adapts a discovered MCP tool to Forge's :class:`Tool` protocol.

    Exposes the MCP tool's ``name``, ``description`` and ``inputSchema`` (as
    ``parameters``) to the Model, performs a light shape check in
    :meth:`validate`, and forwards execution to the owning server through
    :meth:`McpClient.call` in :meth:`run`.
    """

    def __init__(self, server: str, name: str, description: str,
                 parameters: dict, client: "McpClient") -> None:
        self.server = server
        self.name = name
        self.description = description or ""
        self.parameters = parameters or {}
        self._client = client
        # MCP tools have unknown side-effect surface; classify conservatively
        # as non-read-only so the approval policy requires a prompt for them in
        # supervised/readonly modes.
        self.read_only = False

    def validate(self, args: dict) -> str | None:
        """Light shape check: arguments must be an object/mapping.

        The authoritative validation is performed by the MCP server against its
        own input schema; here we only reject obviously malformed argument
        payloads so the executor's validation contract still holds.
        """
        if args is None:
            return None
        if not isinstance(args, dict):
            return "MCP tool arguments must be an object."
        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Forward the call to the owning MCP server via the client."""
        return self._client.call(self.server, self.name, args or {})


# --------------------------------------------------------------------------- #
# Per-server lifecycle handle (internal).
# --------------------------------------------------------------------------- #


@dataclass
class _ServerHandle:
    """Tracks one connected server's runner task and its close signal."""

    task: "asyncio.Task"
    close_event: "asyncio.Event"


class McpClient:
    """Connects to MCP servers, discovers tools, and forwards tool calls.

    Parameters
    ----------
    connect_timeout_s:
        Per-server connect budget in seconds (default 30, matching
        :attr:`forge.config.Config.mcp_connect_timeout_s`). A server that does
        not finish connecting and listing its tools within this budget is
        skipped with a warning (Req 16.4).
    """

    def __init__(self, connect_timeout_s: int = 30, call_timeout_s: int = 60) -> None:
        self._connect_timeout_s = connect_timeout_s
        self._call_timeout_s = call_timeout_s
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # name -> live ClientSession (set once the server runner is ready).
        self._sessions: dict[str, "ClientSession"] = {}
        # name -> runner handle (task + close event) for clean shutdown.
        self._servers: dict[str, _ServerHandle] = {}
        self._closed = False

    # -- public API ----------------------------------------------------------

    def connect_all(
        self,
        servers: list[McpServerConfig],
        builtin_names: set[str] | None = None,
    ) -> list[Tool]:
        """Connect to each server, discover tools, and return adapted Tools.

        For every configured server this connects within the per-server budget,
        lists the exposed tools, and adapts each surviving tool to the
        :class:`Tool` protocol. A server that fails to connect (or times out) is
        skipped with a warning and the rest continue (Req 16.1, 16.4). Collisions
        with built-in tools, and between two MCP servers, are resolved by
        :func:`resolve_mcp_collisions` with one warning per excluded tool
        (Req 16.6).

        Returns the list of accepted, adapted MCP tools (empty when ``servers``
        is empty). Raises :class:`RuntimeError` if the ``mcp`` SDK is not
        installed and there is work to do.
        """

        if not servers:
            return []

        if not _MCP_AVAILABLE:
            raise RuntimeError(
                "The 'mcp' package is required to connect to MCP servers but is "
                "not installed."
            ) from _MCP_IMPORT_ERROR

        self._ensure_loop()

        # (server_name, mcp_tool) for every successfully discovered tool, in
        # connection order so the MCP-vs-MCP tie-break is deterministic.
        discovered: list[tuple[str, object]] = []
        for cfg in servers:
            try:
                tools = self._connect_server_sync(cfg)
            except Exception as exc:  # noqa: BLE001 - warn + continue (Req 16.4)
                warnings.warn(
                    f"Failed to connect to MCP server '{cfg.name}': {exc}",
                    stacklevel=2,
                )
                continue
            for tool in tools:
                discovered.append((cfg.name, tool))

        pairs = [(server, getattr(tool, "name")) for server, tool in discovered]
        accepted_pairs, collision_warnings = resolve_mcp_collisions(
            builtin_names or set(), pairs
        )
        for message in collision_warnings:
            warnings.warn(message, stacklevel=2)

        accepted = set(accepted_pairs)
        adapted: list[Tool] = []
        for server, tool in discovered:
            name = getattr(tool, "name")
            if (server, name) not in accepted:
                continue
            # Avoid registering the same accepted (server, name) twice.
            accepted.discard((server, name))
            adapted.append(
                McpToolAdapter(
                    server=server,
                    name=name,
                    description=getattr(tool, "description", "") or "",
                    parameters=dict(getattr(tool, "inputSchema", {}) or {}),
                    client=self,
                )
            )
        return adapted

    def call(self, server: str, tool: str, args: dict) -> ToolResult:
        """Forward a tool call to ``server`` and return the result (Req 16.3).

        On any failure -- the SDK missing, the loop gone, an unknown/closed
        server, or an error raised by the server during the call -- returns a
        failure :class:`ToolResult` flagged with ``meta={"mcp_error": True}``
        (Req 16.5) rather than raising.
        """

        if not _MCP_AVAILABLE or self._loop is None or self._closed:
            return self._error_result(
                f"MCP server '{server}' is not reachable.",
            )

        session = self._sessions.get(server)
        if session is None:
            return self._error_result(
                f"MCP server '{server}' is not connected.",
            )

        try:
            future: ConcurrentFuture = asyncio.run_coroutine_threadsafe(
                session.call_tool(tool, args or {}), self._loop
            )
            # Bound the wait so an MCP server that stops responding cannot
            # freeze the whole agent turn. On timeout we cancel the pending
            # future (best-effort; the coroutine may already be blocked in I/O)
            # and surface a structured failure rather than blocking forever.
            result = future.result(timeout=self._call_timeout_s)
        except FuturesTimeoutError:
            future.cancel()
            return self._error_result(
                f"MCP tool '{tool}' on server '{server}' timed out after "
                f"{self._call_timeout_s}s.",
            )
        except Exception as exc:  # noqa: BLE001 - surface as failure result
            return self._error_result(
                f"MCP tool '{tool}' on server '{server}' failed: {exc}",
            )

        content = _content_to_text(getattr(result, "content", None))
        if getattr(result, "isError", False):
            return ToolResult(
                ok=False,
                content=content,
                error=content or f"MCP tool '{tool}' returned an error.",
                meta={"mcp_error": True, "server": server},
            )
        return ToolResult(
            ok=True,
            content=content,
            meta={"mcp": True, "server": server},
        )

    def close(self) -> None:
        """Tear down all server connections and stop the background loop."""

        if self._loop is None:
            self._closed = True
            return

        if not self._closed:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._shutdown(), self._loop
                )
                future.result(timeout=10)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        if not self._loop.is_running():
            self._loop.close()
        self._sessions.clear()
        self._servers.clear()

    # -- event-loop management ----------------------------------------------

    def _ensure_loop(self) -> None:
        """Start the dedicated background event loop/thread if not running."""
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="forge-mcp-loop", daemon=True
        )
        self._thread.start()

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    # -- per-server connection -----------------------------------------------

    def _connect_server_sync(self, cfg: McpServerConfig) -> list[object]:
        """Synchronously connect one server within its budget; return its tools.

        Raises on failure/timeout so :meth:`connect_all` can warn and continue.
        """
        assert self._loop is not None
        future: ConcurrentFuture = asyncio.run_coroutine_threadsafe(
            self._connect_server(cfg, self._connect_timeout_s), self._loop
        )
        # A small guard beyond the in-coroutine timeout so a stuck connect can
        # never hang the caller indefinitely.
        return future.result(timeout=self._connect_timeout_s + 5)

    async def _connect_server(
        self, cfg: McpServerConfig, timeout: float
    ) -> list[object]:
        """Spawn the server runner and await readiness within ``timeout``."""
        close_event = asyncio.Event()
        ready: asyncio.Future = self._loop.create_future()  # type: ignore[union-attr]
        task = asyncio.ensure_future(
            self._server_runner(cfg, ready, close_event)
        )
        try:
            tools = await asyncio.wait_for(asyncio.shield(ready), timeout)
        except Exception:
            # Connect failed or timed out: tear the runner down and propagate.
            close_event.set()
            task.cancel()
            raise
        self._servers[cfg.name] = _ServerHandle(task=task, close_event=close_event)
        return tools

    async def _server_runner(
        self,
        cfg: McpServerConfig,
        ready: asyncio.Future,
        close_event: asyncio.Event,
    ) -> None:
        """Own one server's session lifecycle for the client's lifetime.

        Enters the stdio + session context managers, initializes the session,
        lists tools (resolving ``ready``), then parks on ``close_event`` until
        shutdown -- all inside one task so the anyio cancel scopes are entered
        and exited in the same task.
        """
        params = StdioServerParameters(
            command=cfg.command,
            args=list(cfg.args),
            env=dict(cfg.env) if cfg.env else None,
        )
        try:
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()
                listed = await session.list_tools()
                self._sessions[cfg.name] = session
                if not ready.done():
                    ready.set_result(list(listed.tools))
                await close_event.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - report to the awaiting connect
            if not ready.done():
                ready.set_exception(exc)
        finally:
            self._sessions.pop(cfg.name, None)

    async def _shutdown(self) -> None:
        """Signal every runner to close and await their completion."""
        handles = list(self._servers.values())
        for handle in handles:
            handle.close_event.set()
        tasks = [handle.task for handle in handles]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._servers.clear()
        self._sessions.clear()

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _error_result(message: str) -> ToolResult:
        return ToolResult(
            ok=False, content="", error=message, meta={"mcp_error": True}
        )


def _content_to_text(content: object) -> str:
    """Flatten an MCP tool result's content blocks into a single text string.

    Text blocks contribute their ``text``; any other block contributes its
    ``str()`` so nothing is silently dropped.
    """
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(str(block))
    return "\n".join(parts)
