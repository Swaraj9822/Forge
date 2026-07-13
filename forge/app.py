"""Application bootstrap, dependency wiring, and startup validation.

This module turns a resolved :class:`~forge.config.Config` into a fully-wired,
ready-to-run :class:`~forge.repl.Repl`. It is the single place where every
long-lived collaborator is constructed and connected together, and where the
*fatal* startup checks live:

Startup validation (resolved sequencing, see design "Startup sequencing")
--------------------------------------------------------------------------
1. **Load config.** :class:`~forge.config.ConfigManager` reads the Config_File,
   applying the documented defaults for any absent value and applying *all*
   defaults when the file is absent (Req 11.5). A missing file is therefore not
   fatal on its own; a TOML syntax error is, and surfaces as
   :class:`~forge.config.ConfigError` for the caller (``__main__``) to print.
2. **Validate required values.** Before the Vertex client is constructed, the
   required ``project`` and ``region`` are validated. When either is missing
   (absent, blank, or still the ``forge init`` placeholder), a
   :class:`StartupError` is raised directing the user to run ``forge init``
   (Req 2.4, 12.3).
3. **Check ADC.** Application Default Credentials are probed as a startup smoke
   check; when they are unavailable a :class:`StartupError` is raised whose
   message names the ``gcloud auth application-default login`` command (Req 2.3).
   When the auth library is unavailable the check is skipped — the
   :class:`~forge.vertex.VertexClient` still surfaces a
   :class:`~forge.vertex.CredentialsError` at request time.

Wiring order
------------
``ConfigManager`` -> ``SessionStore`` -> ``InterruptController`` ->
``VertexClient`` -> ``ToolExecutor`` (built-in tools + accepted MCP tools) ->
``ContextManager`` (summarized by the ``VertexClient``) -> ``UsageTracker`` ->
``AgentLoop`` -> ``Repl`` -> ``VerificationCoordinator`` (wrapping a
``VerificationRunner`` and using the ``Repl`` as its ``VerificationRenderer``).
Each component is constructed once and shared, so the same workspace root,
interrupt controller, and config flow through the whole graph. The coordinator
is wired unconditionally — even when ``verification.command`` is absent — so it
simply short-circuits at its gate, keeping the unconfigured path identical to
today with no special-casing in the ``Repl``.

The fatal startup errors are raised as :class:`StartupError` (and
:class:`~forge.config.ConfigError` for syntax problems); ``__main__`` handles
them once — printing the message and exiting non-zero — per the design's
"Startup/fatal errors" handling.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from forge.agent import AgentLoop
from forge.config import (
    PROJECT_PLACEHOLDER,
    REGION_PLACEHOLDER,
    RECOGNIZED_TOOLS,
    Config,
    ConfigManager,
)
from forge.context import ContextManager
from forge.interrupt import InterruptController
from forge.mcp_client import McpClient, register_mcp_tools
from forge.repl import Repl
from forge.session import Session, SessionStore, TodoItem
from forge.tools.base import Tool, ToolContext, ToolExecutor
from forge.tools.fs import EditTool, ReadTool, WriteTool
from forge.tools.git import GitTool
from forge.tools.planning import PlanningTool
from forge.tools.search import SearchTool
from forge.tools.shell import ShellTool
from forge.verification import VerificationCoordinator, VerificationRunner
from forge.vertex import CredentialsError, VertexClient

__all__ = [
    "StartupError",
    "App",
    "build_builtin_registry",
    "validate_required_config",
    "check_adc",
    "bootstrap",
    "main",
]


# ---------------------------------------------------------------------------
# Guarded import of google.auth for the ADC smoke check.
#
# Imported defensively so this module loads (and the rest of bootstrap works)
# even when the auth library is not installed; in that case the ADC check is
# skipped and the VertexClient surfaces a CredentialsError at request time.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import wiring
    from google.auth import default as _google_auth_default
except Exception:  # noqa: BLE001 - any import failure degrades gracefully
    _google_auth_default = None


class StartupError(Exception):
    """A fatal startup condition that must stop the process with a message.

    Carries a user-facing ``message`` (the exact text Forge should print) and
    an ``exit_code`` (non-zero) so ``__main__`` can handle every fatal startup
    error in one place: print the message to stderr and exit with the code.
    Raised for missing required configuration (Req 2.4, 12.3) and for missing
    ADC (Req 2.3).
    """

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


# ---------------------------------------------------------------------------
# Wired application handle
# ---------------------------------------------------------------------------


@dataclass
class App:
    """A fully-wired Forge application ready to run.

    Holds references to the constructed collaborators so callers (and tests)
    can inspect the wiring, run the REPL via :meth:`run`, and tear down external
    resources via :meth:`close`.
    """

    config: Config
    session: Session
    config_manager: ConfigManager
    session_store: SessionStore
    interrupt: InterruptController
    vertex_client: VertexClient
    tool_executor: ToolExecutor
    context_manager: ContextManager
    usage_tracker: "object"
    agent_loop: AgentLoop
    repl: Repl
    verification_runner: VerificationRunner
    verification_coordinator: VerificationCoordinator
    mcp_client: McpClient | None = None

    def run(self) -> None:
        """Run the REPL loop until an Exit_Command or EOF, then tear down."""
        try:
            self.interrupt.install()
            self.repl.run()
        finally:
            self.close()

    def close(self) -> None:
        """Release external resources (MCP connections, the SIGINT handler)."""
        if self.mcp_client is not None:
            try:
                self.mcp_client.close()
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
        try:
            self.interrupt.uninstall()
        except Exception:  # noqa: BLE001 - teardown is best-effort
            pass


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------


def _is_missing_required(value: str | None, placeholder: str) -> bool:
    """Return whether a required config value counts as missing.

    A value is treated as missing when it is ``None``, blank/whitespace-only, or
    still the ``forge init`` placeholder the user is expected to replace.
    """
    if value is None:
        return True
    stripped = value.strip()
    return stripped == "" or stripped == placeholder


def validate_required_config(config: Config) -> None:
    """Validate that the required ``project`` and ``region`` are configured.

    Raises :class:`StartupError` directing the user to run ``forge init`` when
    either value is absent, blank, or still the placeholder (Req 2.4, 12.3). The
    check runs *before* the Vertex client is constructed.
    """
    missing: list[str] = []
    if _is_missing_required(config.project, PROJECT_PLACEHOLDER):
        missing.append("GCP project ID")
    if _is_missing_required(config.region, REGION_PLACEHOLDER):
        missing.append("GCP region")

    if not missing:
        return

    names = " and ".join(missing)
    raise StartupError(
        f"Required configuration value(s) missing: {names}. "
        "Run 'forge init' to create a configuration file with the required "
        "placeholders, then edit it to set your GCP project ID and region."
    )


def check_adc() -> None:
    """Probe Application Default Credentials as a startup smoke check (Req 2.3).

    Raises :class:`StartupError` — whose message names the
    ``gcloud auth application-default login`` command — when ADC are
    unavailable. When the auth library is not installed the probe cannot run, so
    the check is skipped and the :class:`~forge.vertex.VertexClient` surfaces a
    :class:`~forge.vertex.CredentialsError` at request time instead.
    """
    if _google_auth_default is None:
        # Cannot determine ADC state without the auth library; defer to the
        # VertexClient's request-time credential check.
        return

    try:
        credentials, _project = _google_auth_default()
    except Exception as exc:  # noqa: BLE001 - any auth failure means no ADC
        raise StartupError(CredentialsError.DEFAULT_MESSAGE) from exc

    if credentials is None:
        raise StartupError(CredentialsError.DEFAULT_MESSAGE)


# ---------------------------------------------------------------------------
# Tool wiring
# ---------------------------------------------------------------------------


def build_builtin_registry() -> dict[str, Tool]:
    """Construct the built-in tools and return a ``name -> Tool`` registry.

    Every recognized built-in tool (read, write, edit, shell, search, git,
    planning) is instantiated. Which of them are actually *exposed* to the Model
    is decided by the executor's ``enabled`` set (sourced from
    ``config.enabled_tools``), not by this registry.
    """
    tools: list[Tool] = [
        ReadTool(),
        WriteTool(),
        EditTool(),
        ShellTool(),
        SearchTool(),
        GitTool(),
        PlanningTool(),
    ]
    return {tool.name: tool for tool in tools}


def _build_tool_executor(
    config: Config,
    interrupt: InterruptController,
    workspace_root: Path,
    initial_todos: list[TodoItem] | None = None,
) -> tuple[ToolExecutor, McpClient | None]:
    """Build the :class:`ToolExecutor` from built-in and accepted MCP tools.

    The built-in registry is populated first; ``enabled`` is seeded from
    ``config.enabled_tools`` (the executor exposes ``registry & enabled``). When
    MCP servers are configured they are connected and their accepted tools are
    merged into the registry and the ``enabled`` set so they become exposed
    alongside the built-ins (Req 11.8, 16.1, 16.2). A shared
    :class:`ToolContext` carries the workspace root, interrupt, config, and a
    session-scoped ``state`` bag (used by the planning tool).

    When ``initial_todos`` is supplied (a resumed session), the ``state`` bag is
    seeded with a copy of those items under the ``"todos"`` key so the planning
    tool continues from the restored list rather than an empty one (Req 10.5,
    13.5).
    """
    registry = build_builtin_registry()
    enabled: set[str] = set(config.enabled_tools)

    mcp_client: McpClient | None = None
    if config.mcp_servers:
        mcp_client = McpClient(connect_timeout_s=config.mcp_connect_timeout_s)
        try:
            mcp_tools = mcp_client.connect_all(
                config.mcp_servers, builtin_names=set(RECOGNIZED_TOOLS)
            )
            register_mcp_tools(registry, enabled, mcp_tools)
        except RuntimeError as exc:
            # The `mcp` SDK is not installed but servers are configured. Warn
            # and continue with the built-in tools rather than failing startup.
            warnings.warn(
                f"MCP servers are configured but could not be initialized: "
                f"{exc}. Continuing with built-in tools only.",
                stacklevel=2,
            )
            mcp_client.close()
            mcp_client = None

    # Seed the session-scoped state bag with the restored todo list (if any) so
    # the planning tool's continuity survives a `forge resume` (Req 10.5).
    state: dict = {}
    if initial_todos:
        state["todos"] = [
            TodoItem(id=t.id, text=t.text, status=t.status) for t in initial_todos
        ]

    context = ToolContext(
        workspace_root=workspace_root,
        interrupt=interrupt,
        config=config,
        state=state,
    )
    executor = ToolExecutor(
        registry=registry,
        enabled=enabled,
        interrupt=interrupt,
        context=context,
    )
    return executor, mcp_client


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap(
    *,
    config: Config | None = None,
    config_path: Path | None = None,
    session: Session | None = None,
    workspace_root: Path | None = None,
    input_func=None,
    out: TextIO | None = None,
    skip_adc_check: bool = False,
) -> App:
    """Load config, validate startup, and wire the full application graph.

    Parameters
    ----------
    config:
        A pre-resolved :class:`~forge.config.Config`. When ``None`` the config
        is loaded via :class:`~forge.config.ConfigManager` from ``config_path``
        (or the OS-conventional location). A :class:`~forge.config.ConfigError`
        from a TOML syntax error propagates to the caller (Req 11.6).
    config_path:
        Optional explicit path to the Config_File (defaults to the
        OS-conventional location).
    session:
        The :class:`~forge.session.Session` to drive. When ``None`` a fresh
        session is minted via :meth:`SessionStore.new` (the ``forge`` /
        ``forge resume`` distinction is made by the CLI in ``__main__``).
    workspace_root:
        The workspace boundary for file/search/shell/git tools (defaults to the
        current working directory).
    input_func, out:
        Optional REPL I/O injection points (forwarded to :class:`Repl`), used by
        tests so no real TTY is required.
    skip_adc_check:
        When ``True`` the ADC smoke check is skipped (used by tests and offline
        wiring). Production startup leaves this ``False``.

    Returns
    -------
    A wired :class:`App`. Raises :class:`StartupError` for missing required
    config (Req 2.4, 12.3) or missing ADC (Req 2.3).
    """
    config_manager = ConfigManager()

    # 1. Load config (defaults applied when file absent — Req 11.5).
    if config is None:
        config = config_manager.load(config_path)

    # 2. Validate required project/region BEFORE constructing the Vertex client
    #    (Req 2.4, 12.3).
    validate_required_config(config)

    # 3. ADC smoke check (Req 2.3) — names `gcloud auth application-default login`.
    if not skip_adc_check:
        check_adc()

    # 4. SessionStore rooted at the OS-conventional sessions directory.
    session_store = SessionStore(config_manager.sessions_dir())
    if session is None:
        session = session_store.new()

    # 5. Interrupt controller (the SIGINT handler is installed when the App runs).
    interrupt = InterruptController()

    # 6. Vertex client (lazily connects; no credential/network call yet).
    vertex_client = VertexClient(config, interrupt)

    # 7. Tool executor: built-in tools + accepted MCP tools. Seed the planning
    #    tool's state from any restored todos so `forge resume` keeps continuity.
    root = Path(workspace_root) if workspace_root is not None else Path.cwd()
    tool_executor, mcp_client = _build_tool_executor(
        config, interrupt, root, session.todos
    )

    # 8. Context manager, summarized by the Vertex client during compaction.
    context_manager = ContextManager(config, summarizer=vertex_client)

    # 9. Usage tracker (cost computed from config.pricing). Seed the cumulative
    #    (session) tallies from a restored session so a resumed session's totals
    #    continue rather than restarting at zero (Req 17.2).
    from forge.usage import UsageTracker

    usage_tracker = UsageTracker(config)
    usage_tracker.seed(
        session.usage.input_tokens, session.usage.output_tokens
    )

    # 10. Agent loop wiring all of the above together.
    agent_loop = AgentLoop(
        context_manager=context_manager,
        vertex_client=vertex_client,
        tool_executor=tool_executor,
        usage_tracker=usage_tracker,
        session_store=session_store,
        interrupt=interrupt,
    )

    # 11. The REPL drives the agent loop and renders its output.
    repl = Repl(
        agent_loop=agent_loop,
        session=session,
        input_func=input_func,
        out=out,
    )

    # 12. Verification phase. The runner reuses the shell execution core rooted
    #     at the same workspace and shares the interrupt controller; the
    #     coordinator shares the agent loop, session store, and interrupt, and
    #     drives the Repl as its VerificationRenderer. It is wired
    #     unconditionally — when `config.verification.command` is None it simply
    #     short-circuits at its gate, so the unconfigured path is unchanged with
    #     no special-casing in the Repl (Req 2.1, 2.2, 2.3).
    verification_runner = VerificationRunner(root, interrupt)
    verification_coordinator = VerificationCoordinator(
        config=config.verification,
        runner=verification_runner,
        agent_loop=agent_loop,
        session_store=session_store,
        interrupt=interrupt,
        renderer=repl,
    )
    repl.verification_coordinator = verification_coordinator

    return App(
        config=config,
        session=session,
        config_manager=config_manager,
        session_store=session_store,
        interrupt=interrupt,
        vertex_client=vertex_client,
        tool_executor=tool_executor,
        context_manager=context_manager,
        usage_tracker=usage_tracker,
        agent_loop=agent_loop,
        repl=repl,
        verification_runner=verification_runner,
        verification_coordinator=verification_coordinator,
        mcp_client=mcp_client,
    )


def main(
    *,
    config_path: Path | None = None,
    session: Session | None = None,
    workspace_root: Path | None = None,
    err: TextIO | None = None,
) -> int:
    """Bootstrap and run the REPL, returning a process exit code.

    Handles the fatal startup errors in one place (per the design): a
    :class:`StartupError` (missing ADC / missing required config) is printed to
    stderr and its non-zero exit code returned. ``__main__`` provides the full
    CLI dispatch (task 24.1) and may call :func:`bootstrap` directly; this entry
    is the simple "fresh REPL" path.
    """
    err = err if err is not None else sys.stderr
    try:
        app = bootstrap(
            config_path=config_path,
            session=session,
            workspace_root=workspace_root,
        )
    except StartupError as exc:
        print(exc.message, file=err)
        return exc.exit_code

    app.run()
    return 0
