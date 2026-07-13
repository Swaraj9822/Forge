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
)

DEFAULT_ENABLED_TOOLS: list[str] = list(RECOGNIZED_TOOLS)

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

    return VerificationConfig(
        command=command,
        max_correction_iterations=max_iters,
        trigger=trigger,
        timeout_s=timeout_s,
        output_cap_chars=output_cap_chars,
    )


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

        merged_limits = {
            key: limits.get(key, default) for key, default in DEFAULT_LIMITS.items()
        }

        enabled_tools = self._resolve_enabled_tools(raw.get("enabled_tools"))
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

        return Config(
            model=raw.get("model", DEFAULT_MODEL),
            project=raw.get("project"),
            region=raw.get("region"),
            enabled_tools=enabled_tools,
            steering_files=steering_files,
            mcp_servers=mcp_servers,
            pricing=pricing,
            verification=verification,
            **merged_limits,
        )

    @staticmethod
    def _resolve_enabled_tools(value: Any) -> list[str]:
        """Apply default when absent; drop+warn unrecognized tools (Req 11.7)."""

        if value is None:
            return list(DEFAULT_ENABLED_TOOLS)

        recognized: list[str] = []
        for name in value:
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
        }

        with open(path, "wb") as fh:
            tomli_w.dump(document, fh)
