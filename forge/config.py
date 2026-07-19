"""Configuration management for Forge.

Defines the configuration dataclasses (`ModelPricing`, `McpServerConfig`,
`Config`) and the `ConfigManager`, which resolves OS-conventional paths, loads
and validates the TOML config file (applying documented defaults for absent
values), and writes the default config used by `forge init`.

See the design document's "ConfigManager" and "Config TOML schema" sections.
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

# ---------------------------------------------------------------------------
# Documented defaults (see requirements "Default Configuration Values" table)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-3.1-pro-preview"

# The recognized built-in tool set. Enabled tools outside this set are dropped
# with a warning (Req 11.7). MCP tools are added to the exposed set elsewhere.
RECOGNIZED_TOOLS: tuple[str, ...] = (
    "read",
    "write",
    "edit",
    "shell",
    "search",
    "git",
    "planning",
    # Phase 3: memory and repo index tools
    "remember",
    "search_memory",
    "repo_index",
    # Phase 5: subagents delegation
    "delegate",
)

DEFAULT_ENABLED_TOOLS: list[str] = [t for t in RECOGNIZED_TOOLS if t != "delegate"]

# Documented numeric limits, grouped under the [limits] TOML table.
DEFAULT_LIMITS: dict[str, int] = {
    "token_limit": 200_000,
    "retained_recent_messages": 20,
    "request_timeout_s": 60,
    "shell_timeout_s": 120,
    "output_cap_chars": 30_000,
    "search_result_limit": 100,
    "search_line_cap": 500,
    "read_max_lines": 2_000,
    "read_max_bytes": 1_000_000,
    "rate_limit_retries": 5,
    "mcp_connect_timeout_s": 30,
}

# Allowed values for the Gemini 3 ``provider.thinking_level`` control. These map
# to the ``thinking_level`` request parameter (minimal/low/medium/high) that
# replaced ``thinking_budget`` for Gemini 3 models. ``None`` (absent) leaves the
# model's own default thinking level in place.
THINKING_LEVELS: tuple[str, ...] = ("minimal", "low", "medium", "high")

# Limits that must be strictly positive (a zero value is nonsensical: a zero
# timeout, token budget, or read cap would make the affected subsystem
# unusable). Every other limit is merely required to be a non-negative integer
# (0 is meaningful for e.g. rate_limit_retries=no-retries or output_cap=cap-all).
_POSITIVE_LIMITS: frozenset[str] = frozenset({
    "token_limit",
    "request_timeout_s",
    "shell_timeout_s",
    "mcp_connect_timeout_s",
    "read_max_lines",
    "read_max_bytes",
    "search_result_limit",
    "search_line_cap",
})

# Placeholders emitted by `forge init` for the two required values.
PROJECT_PLACEHOLDER = "REPLACE_WITH_GCP_PROJECT_ID"
REGION_PLACEHOLDER = "REPLACE_WITH_GCP_REGION"

# Pricing values written by `forge init` into the documented [pricing] table.
DEFAULT_INIT_PRICING = {"input_per_1k": 0.00125, "output_per_1k": 0.005}

_APP_DIR = "forge"
_CONFIG_FILENAME = "config.toml"
_SESSIONS_DIRNAME = "sessions"


class ConfigError(Exception):
    """Raised when the config file contains a TOML syntax error.

    Carries the offending file path plus, when available, the line and column
    of the error (extracted from :class:`tomllib.TOMLDecodeError`).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        line: int | None = None,
        column: int | None = None,
        detail: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.line = line
        self.column = column
        self.detail = detail
        location = ""
        if line is not None and column is not None:
            location = f" at line {line}, column {column}"
        elif line is not None:
            location = f" at line {line}"
        message = f"Config file {self.path}{location} contains a syntax error"
        if detail:
            message += f": {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class ModelPricing:
    """Per-1k-token pricing for the active model. ``None`` disables cost."""

    input_per_1k: float | None
    output_per_1k: float | None


@dataclass(frozen=True)
class VerificationConfig:
    """Resolved, validated configuration for the Auto-Verification Loop.

    ``command`` of ``None`` means the feature is disabled (opt-in). The other
    values carry the documented defaults applied by
    :func:`resolve_verification_config` (see the design document's
    "VerificationConfig" section).
    """

    command: str | None = None
    max_correction_iterations: int = 3
    trigger: str = "on_file_change"
    timeout_s: int = DEFAULT_LIMITS["shell_timeout_s"]
    output_cap_chars: int = DEFAULT_LIMITS["output_cap_chars"]


# Allowed values for verification.trigger.
VERIFICATION_TRIGGERS: tuple[str, ...] = ("on_file_change", "always")


def resolve_verification_config(
    raw: dict[str, Any] | None,
    *,
    shell_timeout_s: int,
    output_cap_chars: int,
) -> VerificationConfig:
    """Resolve a ``[verification]`` mapping into a :class:`VerificationConfig`.

    Pure: applies the documented defaults for absent values and validates
    present values, raising :class:`ConfigError` (naming the offending value)
    for an invalid ``max_correction_iterations`` or ``trigger`` (Req 1.1-1.8).
    """

    raw = raw or {}

    command = raw.get("command")

    if "max_correction_iterations" in raw:
        max_iters = raw["max_correction_iterations"]
        # bool is a subclass of int; reject it explicitly as a non-integer.
        if isinstance(max_iters, bool) or not isinstance(max_iters, int) or max_iters < 0:
            raise ConfigError(
                ConfigManager.config_path(),
                detail=(
                    "verification.max_correction_iterations must be an integer "
                    f">= 0, got {max_iters!r}"
                ),
            )
    else:
        max_iters = 3

    if "trigger" in raw:
        trigger = raw["trigger"]
        if trigger not in VERIFICATION_TRIGGERS:
            allowed = ", ".join(VERIFICATION_TRIGGERS)
            raise ConfigError(
                ConfigManager.config_path(),
                detail=(
                    f"verification.trigger must be one of {{{allowed}}}, "
                    f"got {trigger!r}"
                ),
            )
    else:
        trigger = "on_file_change"

    timeout_s = raw.get("timeout_s")
    if timeout_s is None:
        timeout_s = shell_timeout_s
    elif isinstance(timeout_s, bool) or not isinstance(timeout_s, int) or timeout_s < 1:
        raise ConfigError(
            ConfigManager.config_path(),
            detail=(
                "verification.timeout_s must be an integer >= 1, got "
                f"{timeout_s!r}"
            ),
        )

    return VerificationConfig(
        command=command,
        max_correction_iterations=max_iters,
        trigger=trigger,
        timeout_s=timeout_s,
        output_cap_chars=output_cap_chars,
    )


# Default read-only toolset the review agent may use to inspect the change.
DEFAULT_REVIEW_TOOLS: tuple[str, ...] = ("read", "search", "repo_index", "git")


@dataclass(frozen=True)
class ReviewConfig:
    """Resolved, validated configuration for the post-turn Review_Phase.

    ``enabled`` of ``False`` (the default) means the feature is off (opt-in).
    An independent review agent inspects a turn's implementation against the
    plan and, on a "changes requested" verdict, feeds actionable findings back
    to the coding agent within a bounded correction loop.
    """

    enabled: bool = False
    trigger: str = "on_file_change"
    max_iterations: int = 2
    tools: tuple[str, ...] = DEFAULT_REVIEW_TOOLS


# Allowed values for review.trigger (shares the verification trigger set).
REVIEW_TRIGGERS: tuple[str, ...] = ("on_file_change", "always")


def resolve_review_config(raw: dict[str, Any] | None) -> ReviewConfig:
    """Resolve a ``[review]`` mapping into a :class:`ReviewConfig`.

    Pure: applies documented defaults for absent values and validates present
    ones, raising :class:`ConfigError` (naming the offending value) for an
    invalid ``max_iterations``, ``trigger``, or ``tools`` entry.
    """

    raw = raw or {}

    enabled = bool(raw.get("enabled", False))

    if "max_iterations" in raw:
        max_iters = raw["max_iterations"]
        if isinstance(max_iters, bool) or not isinstance(max_iters, int) or max_iters < 0:
            raise ConfigError(
                ConfigManager.config_path(),
                detail=(
                    "review.max_iterations must be an integer >= 0, got "
                    f"{max_iters!r}"
                ),
            )
    else:
        max_iters = 2

    if "trigger" in raw:
        trigger = raw["trigger"]
        if trigger not in REVIEW_TRIGGERS:
            allowed = ", ".join(REVIEW_TRIGGERS)
            raise ConfigError(
                ConfigManager.config_path(),
                detail=f"review.trigger must be one of {{{allowed}}}, got {trigger!r}",
            )
    else:
        trigger = "on_file_change"

    if "tools" in raw:
        tools_raw = raw["tools"]
        if not isinstance(tools_raw, list) or not all(
            isinstance(t, str) for t in tools_raw
        ):
            raise ConfigError(
                ConfigManager.config_path(),
                detail="review.tools must be a list of tool-name strings.",
            )
        tools = tuple(tools_raw)
    else:
        tools = DEFAULT_REVIEW_TOOLS

    return ReviewConfig(
        enabled=enabled,
        trigger=trigger,
        max_iterations=max_iters,
        tools=tools,
    )


# Allowed values for the Phase 2 ``[policy]`` table's ``mode`` field.
POLICY_MODES: tuple[str, ...] = ("autopilot", "supervised", "readonly")


def resolve_policy_config(raw: dict[str, Any] | None) -> tuple[str, tuple[str, ...], bool]:
    """Resolve a ``[policy]`` mapping into ``(mode, allowlist, show_diffs)``.

    Pure: applies the documented defaults for absent values and validates
    ``mode``, raising :class:`ConfigError` (naming the offending value) for
    an unknown mode. ``shell_allowlist`` is normalized to a tuple of strings;
    non-string entries are silently dropped (an empty allowlist is valid and
    disables auto-approval for every shell command in supervised mode).
    """

    raw = raw or {}

    if "mode" in raw:
        mode = raw["mode"]
        if mode not in POLICY_MODES:
            allowed = ", ".join(POLICY_MODES)
            raise ConfigError(
                ConfigManager.config_path(),
                detail=(
                    f"policy.mode must be one of {{{allowed}}}, got {mode!r}"
                ),
            )
    else:
        mode = "autopilot"

    allowlist_raw = raw.get("shell_allowlist", [])
    if allowlist_raw is None:
        allowlist_raw = []
    if not isinstance(allowlist_raw, list):
        raise ConfigError(
            ConfigManager.config_path(),
            detail=(
                "policy.shell_allowlist must be a list of strings, got "
                f"{type(allowlist_raw).__name__}"
            ),
        )
    allowlist: list[str] = []
    for item in allowlist_raw:
        if isinstance(item, str):
            allowlist.append(item)
    show_diffs = bool(raw.get("show_diffs", False))

    return mode, tuple(allowlist), show_diffs


@dataclass(frozen=True)
class McpServerConfig:
    """Configuration for a single MCP server launched as a subprocess."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    """Fully-resolved Forge configuration with documented defaults applied."""

    model: str = DEFAULT_MODEL
    project: str | None = None
    region: str | None = None
    enabled_tools: list[str] = field(default_factory=lambda: list(DEFAULT_ENABLED_TOOLS))
    token_limit: int = DEFAULT_LIMITS["token_limit"]
    retained_recent_messages: int = DEFAULT_LIMITS["retained_recent_messages"]
    request_timeout_s: int = DEFAULT_LIMITS["request_timeout_s"]
    shell_timeout_s: int = DEFAULT_LIMITS["shell_timeout_s"]
    output_cap_chars: int = DEFAULT_LIMITS["output_cap_chars"]
    search_result_limit: int = DEFAULT_LIMITS["search_result_limit"]
    search_line_cap: int = DEFAULT_LIMITS["search_line_cap"]
    read_max_lines: int = DEFAULT_LIMITS["read_max_lines"]
    read_max_bytes: int = DEFAULT_LIMITS["read_max_bytes"]
    rate_limit_retries: int = DEFAULT_LIMITS["rate_limit_retries"]
    mcp_connect_timeout_s: int = DEFAULT_LIMITS["mcp_connect_timeout_s"]
    steering_files: list[str] = field(default_factory=list)
    mcp_servers: list[McpServerConfig] = field(default_factory=list)
    pricing: ModelPricing = field(default_factory=lambda: ModelPricing(None, None))
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    plan_reminder: bool = True
    project_memory: bool = True
    # Phase 2: autonomy / approval policy + checkpoint store.
    # The dataclass default is ``autopilot`` so an absent ``[policy]`` table
    # reproduces today's behavior and existing tests pass. A freshly
    # ``forge init``-ed config opts the user into ``supervised`` via
    # ``write_default`` below.
    policy_mode: str = "autopilot"
    shell_allowlist: tuple[str, ...] = ()
    show_diffs: bool = False
    checkpoint_enabled: bool = True
    checkpoint_keep_turns: int = 10
    # Phase 3: memory and repo map configuration.
    # Providers default OFF so absent tables reproduce today's context byte-for-byte.
    # Fresh configs opt into enabled=True via write_default.
    memory_enabled: bool = False
    memory_max_records: int = 500
    memory_inject_limit: int = 5
    memory_inject_char_budget: int = 2000
    repo_map_enabled: bool = False
    repo_map_inject: bool = False
    repo_map_char_budget: int = 4000
    # Phase 4: Ergonomics & UX configuration
    ui_color: bool = False
    ui_spinner: bool = False
    commands_dir: str = ".forge/commands"
    parallel_enabled: bool = False
    parallel_max_workers: int = 4
    mentions_enabled: bool = False
    # Phase 5: Multi-provider & Subagents
    provider_type: str = "vertex"
    provider_api_key_env: str | None = None
    provider_base_url: str | None = None
    # Amount of internal reasoning the model performs. Maps to the Gemini 3
    # ``thinking_level`` request parameter (minimal/low/medium/high). ``None``
    # leaves the model's default thinking level untouched.
    provider_thinking_level: str | None = None
    subagents_enabled: bool = False
    subagents_default_tools: list[str] = field(default_factory=lambda: ["read", "search", "repo_index", "search_memory"])
    subagents_max_turns: int = 4



def _is_windows() -> bool:
    return sys.platform == "win32" or os.name == "nt"


def _windows_appdata() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata)
    # Conventional fallback when %APPDATA% is unset.
    return Path.home() / "AppData" / "Roaming"


def _xdg_base(env_var: str, fallback: Path) -> Path:
    value = os.environ.get(env_var)
    if value:
        return Path(value)
    return fallback


def _toml_location(err: tomllib.TOMLDecodeError) -> tuple[int | None, int | None]:
    """Extract (line, column) from a TOMLDecodeError.

    Python 3.14+ exposes ``lineno``/``colno`` attributes; earlier versions only
    embed the location in the message text (e.g. ``(at line 2, column 5)``).
    """

    line = getattr(err, "lineno", None)
    column = getattr(err, "colno", None)
    if isinstance(line, int) and isinstance(column, int):
        return line, column

    match = re.search(r"at line (\d+), column (\d+)", str(err))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


class ConfigManager:
    """Loads, validates, and writes Forge configuration."""

    @staticmethod
    def config_path() -> Path:
        """Return the OS-conventional path to ``config.toml`` (Req 11.2)."""

        if _is_windows():
            return _windows_appdata() / _APP_DIR / _CONFIG_FILENAME
        base = _xdg_base("XDG_CONFIG_HOME", Path.home() / ".config")
        return base / _APP_DIR / _CONFIG_FILENAME

    @staticmethod
    def sessions_dir() -> Path:
        """Return the OS-conventional sessions directory (Req 11.9)."""

        if _is_windows():
            return _windows_appdata() / _APP_DIR / _SESSIONS_DIRNAME
        base = _xdg_base("XDG_DATA_HOME", Path.home() / ".local" / "share")
        return base / _APP_DIR / _SESSIONS_DIRNAME

    def load(self, path: Path | None = None) -> Config:
        """Load config from ``path`` (default: :meth:`config_path`).

        Applies documented defaults for absent values (Req 11.4), applies all
        defaults when the file is absent (Req 11.5), drops and warns on
        unrecognized enabled tools (Req 11.7), and raises :class:`ConfigError`
        with file path and line/column on a syntax error (Req 11.6).
        """

        target = path if path is not None else self.config_path()

        # Absent file: apply all documented defaults and continue (Req 11.5).
        if not target.exists():
            return Config()

        try:
            with open(target, "rb") as fh:
                raw: dict[str, Any] = tomllib.load(fh)
        except tomllib.TOMLDecodeError as err:
            line, column = _toml_location(err)
            raise ConfigError(
                target, line=line, column=column, detail=str(err)
            ) from err

        return self._from_raw(raw)

    def _from_raw(self, raw: dict[str, Any]) -> Config:
        """Merge a parsed TOML mapping with documented defaults into a Config."""

        limits = raw.get("limits") or {}
        pricing_raw = raw.get("pricing") or {}
        context_raw = raw.get("context") or {}

        merged_limits = {
            key: limits.get(key, default) for key, default in DEFAULT_LIMITS.items()
        }

        # Validate every limit: reject wrong types (bool/str/float), negatives,
        # and zero for limits that must be strictly positive, so a bad value
        # fails fast at load with a clear message rather than surfacing as an
        # obscure error deep in a later subsystem.
        for key, value in merged_limits.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(
                    ConfigManager.config_path(),
                    detail=f"limits.{key} must be an integer, got {value!r}",
                )
            if value < 0:
                raise ConfigError(
                    ConfigManager.config_path(),
                    detail=f"limits.{key} must be >= 0, got {value}",
                )
            if key in _POSITIVE_LIMITS and value == 0:
                raise ConfigError(
                    ConfigManager.config_path(),
                    detail=f"limits.{key} must be >= 1, got 0",
                )

        plan_reminder = bool(context_raw.get("plan_reminder", True))
        project_memory = bool(context_raw.get("project_memory", True))

        # Phase 5: provider and subagents config
        provider_raw = raw.get("provider") or {}
        provider_type = str(provider_raw.get("type", "vertex")).strip()
        if provider_type not in ("vertex", "anthropic", "openai"):
            raise ConfigError(
                ConfigManager.config_path(),
                detail=f"provider.type must be one of {{vertex, anthropic, openai}}, got {provider_type!r}",
            )
        
        provider_api_key_env = provider_raw.get("api_key_env")
        provider_base_url = provider_raw.get("base_url")

        provider_thinking_level = provider_raw.get("thinking_level")
        if provider_thinking_level is not None:
            if not isinstance(provider_thinking_level, str):
                raise ConfigError(
                    ConfigManager.config_path(),
                    detail=(
                        "provider.thinking_level must be a string, got "
                        f"{type(provider_thinking_level).__name__}"
                    ),
                )
            provider_thinking_level = provider_thinking_level.strip().lower()
            if provider_thinking_level not in THINKING_LEVELS:
                allowed = ", ".join(THINKING_LEVELS)
                raise ConfigError(
                    ConfigManager.config_path(),
                    detail=(
                        f"provider.thinking_level must be one of {{{allowed}}}, "
                        f"got {provider_raw.get('thinking_level')!r}"
                    ),
                )

        subagents_raw = raw.get("subagents") or {}
        subagents_enabled = bool(subagents_raw.get("enabled", False))
        subagents_max_turns = subagents_raw.get("max_turns", 4)
        if (
            isinstance(subagents_max_turns, bool)
            or not isinstance(subagents_max_turns, int)
            or subagents_max_turns < 1
        ):
            raise ConfigError(
                ConfigManager.config_path(),
                detail=f"subagents.max_turns must be an integer >= 1, got {subagents_max_turns!r}",
            )

        subagents_default_tools = list(subagents_raw.get("default_tools", ["read", "search", "repo_index", "search_memory"]))
        for tool_name in subagents_default_tools:
            if tool_name not in RECOGNIZED_TOOLS:
                raise ConfigError(
                    ConfigManager.config_path(),
                    detail=f"subagents.default_tools contains unrecognized tool {tool_name!r}",
                )

        enabled_tools = self._resolve_enabled_tools(raw.get("enabled_tools"), subagents_enabled)
        steering_files = list(raw.get("steering_files", []))
        mcp_servers = self._parse_mcp_servers(raw.get("mcp_servers", []))
        pricing = ModelPricing(
            input_per_1k=pricing_raw.get("input_per_1k"),
            output_per_1k=pricing_raw.get("output_per_1k"),
        )

        verification = resolve_verification_config(
            raw.get("verification"),
            shell_timeout_s=merged_limits["shell_timeout_s"],
            output_cap_chars=merged_limits["output_cap_chars"],
        )

        review = resolve_review_config(raw.get("review"))

        policy_mode, shell_allowlist, show_diffs = resolve_policy_config(
            raw.get("policy")
        )

        checkpoint_raw = raw.get("checkpoint") or {}
        checkpoint_enabled = bool(checkpoint_raw.get("enabled", True))
        checkpoint_keep = checkpoint_raw.get("keep_turns", 10)
        if (
            isinstance(checkpoint_keep, bool)
            or not isinstance(checkpoint_keep, int)
            or checkpoint_keep < 0
        ):
            raise ConfigError(
                ConfigManager.config_path(),
                detail=(
                    "checkpoint.keep_turns must be an integer >= 0, got "
                    f"{checkpoint_keep!r}"
                ),
            )

        # Phase 3: memory and repo map configuration
        memory_raw = raw.get("memory") or {}
        memory_enabled = bool(memory_raw.get("enabled", False))
        memory_max_records = memory_raw.get("max_records", 500)
        if (
            isinstance(memory_max_records, bool)
            or not isinstance(memory_max_records, int)
            or memory_max_records < 1
        ):
            raise ConfigError(
                ConfigManager.config_path(),
                detail=(
                    "memory.max_records must be an integer >= 1, got "
                    f"{memory_max_records!r}"
                ),
            )
        memory_inject_limit = memory_raw.get("inject_limit", 5)
        if (
            isinstance(memory_inject_limit, bool)
            or not isinstance(memory_inject_limit, int)
            or memory_inject_limit < 1
        ):
            raise ConfigError(
                ConfigManager.config_path(),
                detail=(
                    "memory.inject_limit must be an integer >= 1, got "
                    f"{memory_inject_limit!r}"
                ),
            )
        memory_inject_char_budget = memory_raw.get("inject_char_budget", 2000)
        if (
            isinstance(memory_inject_char_budget, bool)
            or not isinstance(memory_inject_char_budget, int)
            or memory_inject_char_budget < 100
        ):
            raise ConfigError(
                ConfigManager.config_path(),
                detail=(
                    "memory.inject_char_budget must be an integer >= 100, got "
                    f"{memory_inject_char_budget!r}"
                ),
            )

        repo_map_raw = raw.get("repo_map") or {}
        repo_map_enabled = bool(repo_map_raw.get("enabled", False))
        repo_map_inject = bool(repo_map_raw.get("inject", False))
        repo_map_char_budget = repo_map_raw.get("char_budget", 4000)
        if (
            isinstance(repo_map_char_budget, bool)
            or not isinstance(repo_map_char_budget, int)
            or repo_map_char_budget < 100
        ):
            raise ConfigError(
                ConfigManager.config_path(),
                detail=(
                    "repo_map.char_budget must be an integer >= 100, got "
                    f"{repo_map_char_budget!r}"
                ),
            )

        ui_raw = raw.get("ui") or {}
        ui_color = bool(ui_raw.get("color", False))
        ui_spinner = bool(ui_raw.get("spinner", False))

        commands_raw = raw.get("commands") or {}
        commands_dir = str(commands_raw.get("dir", ".forge/commands"))

        parallel_raw = raw.get("parallel") or {}
        parallel_enabled = bool(parallel_raw.get("enabled", False))
        parallel_max_workers = parallel_raw.get("max_workers", 4)
        if (
            isinstance(parallel_max_workers, bool)
            or not isinstance(parallel_max_workers, int)
            or parallel_max_workers < 1
        ):
            raise ConfigError(
                ConfigManager.config_path(),
                detail=(
                    "parallel.max_workers must be an integer >= 1, got "
                    f"{parallel_max_workers!r}"
                ),
            )

        mentions_raw = raw.get("mentions") or {}
        mentions_enabled = bool(mentions_raw.get("enabled", False))

        model_resolved = provider_raw.get("model") or raw.get("model") or DEFAULT_MODEL
        project_resolved = provider_raw.get("project") or raw.get("project")
        region_resolved = provider_raw.get("region") or raw.get("region")

        return Config(
            model=model_resolved,
            project=project_resolved,
            region=region_resolved,
            enabled_tools=enabled_tools,
            steering_files=steering_files,
            mcp_servers=mcp_servers,
            pricing=pricing,
            verification=verification,
            review=review,
            plan_reminder=plan_reminder,
            project_memory=project_memory,
            policy_mode=policy_mode,
            shell_allowlist=shell_allowlist,
            show_diffs=show_diffs,
            checkpoint_enabled=checkpoint_enabled,
            checkpoint_keep_turns=checkpoint_keep,
            memory_enabled=memory_enabled,
            memory_max_records=memory_max_records,
            memory_inject_limit=memory_inject_limit,
            memory_inject_char_budget=memory_inject_char_budget,
            repo_map_enabled=repo_map_enabled,
            repo_map_inject=repo_map_inject,
            repo_map_char_budget=repo_map_char_budget,
            ui_color=ui_color,
            ui_spinner=ui_spinner,
            commands_dir=commands_dir,
            parallel_enabled=parallel_enabled,
            parallel_max_workers=parallel_max_workers,
            mentions_enabled=mentions_enabled,
            provider_type=provider_type,
            provider_api_key_env=provider_api_key_env,
            provider_base_url=provider_base_url,
            provider_thinking_level=provider_thinking_level,
            subagents_enabled=subagents_enabled,
            subagents_default_tools=subagents_default_tools,
            subagents_max_turns=subagents_max_turns,
            **merged_limits,
        )

    @staticmethod
    def _resolve_enabled_tools(value: Any, subagents_enabled: bool = False) -> list[str]:
        """Apply default when absent; drop+warn unrecognized tools (Req 11.7)."""

        if value is None:
            tools = list(DEFAULT_ENABLED_TOOLS)
            if subagents_enabled and "delegate" not in tools:
                tools.append("delegate")
            return tools

        recognized: list[str] = []
        for name in value:
            # ``delegate`` is a recognized tool (it is in RECOGNIZED_TOOLS); it
            # is simply absent from DEFAULT_ENABLED_TOOLS so it stays off unless
            # explicitly listed or auto-added when subagents are enabled. It is
            # therefore resolved like any other recognized tool here — the
            # recognized-tools invariant (Property 18) must hold for it too.
            if name in RECOGNIZED_TOOLS:
                recognized.append(name)
            else:
                warnings.warn(
                    f"Unrecognized tool '{name}' in enabled_tools; ignoring it.",
                    stacklevel=2,
                )
        return recognized

    @staticmethod
    def _parse_mcp_servers(value: Any) -> list[McpServerConfig]:
        servers: list[McpServerConfig] = []
        for entry in value:
            servers.append(
                McpServerConfig(
                    name=entry["name"],
                    command=entry["command"],
                    args=list(entry.get("args", [])),
                    env=dict(entry.get("env", {})),
                )
            )
        return servers

    def write_default(self, path: Path) -> None:
        """Write the documented default config to ``path`` (Req 12.1).

        Emits the documented TOML structure with required ``project``/``region``
        placeholders. The output round-trips back through :meth:`load` to the
        documented defaults (Property 19).
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        document: dict[str, Any] = {
            "model": DEFAULT_MODEL,
            "project": PROJECT_PLACEHOLDER,
            "region": REGION_PLACEHOLDER,
            "enabled_tools": list(DEFAULT_ENABLED_TOOLS),
            "steering_files": [],
            "limits": dict(DEFAULT_LIMITS),
            "pricing": dict(DEFAULT_INIT_PRICING),
            # No MCP servers by default. A non-empty list here would be
            # connected at every startup; the placeholder example used to fail
            # to launch and warn on a fresh config. Add real servers by hand.
            "mcp_servers": [],
            "context": {"plan_reminder": True, "project_memory": True},
            # Phase 2: opt fresh configs into supervised mode (safe default)
            # and seed a small starter shell allowlist so trivial commands run
            # without a prompt. The dataclass default remains ``autopilot`` so
            # an absent ``[policy]`` table reproduces today's behavior.
            "policy": {
                "mode": "supervised",
                "shell_allowlist": [
                    "pytest",
                    "git status",
                    "git diff",
                    "git log",
                    "ls",
                    "cat",
                    "python -m pytest",
                ],
                "show_diffs": True,
            },
            "checkpoint": {"enabled": True, "keep_turns": 10},
            # Phase 3: opt fresh configs into memory and repo map features
            "memory": {
                "enabled": True,
                "max_records": 500,
                "inject_limit": 5,
                "inject_char_budget": 2000,
            },
            "repo_map": {
                "enabled": True,
                "inject": True,
                "char_budget": 4000,
            },
            # Phase 4: opt fresh configs into color terminal, spinner, mentions
            "ui": {
                "color": True,
                "spinner": True,
            },
            "commands": {
                "dir": ".forge/commands",
            },
            "parallel": {
                "enabled": False,
                "max_workers": 4,
            },
            "mentions": {
                "enabled": True,
            },
            "provider": {
                "type": "vertex",
                "model": DEFAULT_MODEL,
            },
            "subagents": {
                "enabled": False,
                "default_tools": ["read", "search", "repo_index", "search_memory"],
                "max_turns": 4,
            },
            "review": {
                "enabled": False,
                "trigger": "on_file_change",
                "max_iterations": 2,
                "tools": ["read", "search", "repo_index", "git"],
            },
        }

        with open(path, "wb") as fh:
            tomli_w.dump(document, fh)
