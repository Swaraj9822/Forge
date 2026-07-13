"""Vertex AI / Gemini provider for Forge.

This module wraps the unified ``google-genai`` SDK (initialized for Vertex AI)
behind the ``Provider`` interface.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Iterator

from forge.config import Config
from forge.interrupt import InterruptController
from forge.session import ToolCall
from forge.tools.base import ToolSpec
from forge.providers.base import (
    TextDelta,
    UsageReport,
    Done,
    StreamEvent,
    Provider,
    ProviderError,
    CredentialsError,
    ConfigMissingError,
    AuthorizationError,
    RateLimitError,
    RequestTimeoutError,
    wait_backoff,
)

# VertexError is a provider-specific alias for backwards compatibility
VertexError = ProviderError

__all__ = [
    "VertexProvider",
    "VertexError",
]


# ---------------------------------------------------------------------------
# Guarded SDK imports
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import wiring
    from google import genai as _genai
except Exception:  # noqa: BLE001 - any import failure degrades gracefully
    _genai = None

try:  # pragma: no cover - import wiring
    from google.genai import types as _genai_types
except Exception:  # noqa: BLE001
    _genai_types = None

try:  # pragma: no cover - import wiring
    from google.genai import errors as _genai_errors
except Exception:  # noqa: BLE001
    _genai_errors = None

try:  # pragma: no cover - import wiring
    from google.auth import exceptions as _google_auth_exceptions
except Exception:  # noqa: BLE001
    _google_auth_exceptions = None


# ---------------------------------------------------------------------------
# VertexProvider
# ---------------------------------------------------------------------------


class VertexProvider(Provider):
    """Streams Gemini responses over Vertex AI through the ``google-genai`` SDK.

    Parameters
    ----------
    config:
        The resolved :class:`~forge.config.Config`; supplies the model id,
        project, region, request timeout, and rate-limit retry count.
    interrupt:
        The shared :class:`~forge.interrupt.InterruptController`, polled between
        streamed chunks so generation aborts promptly on Ctrl-C.
    client:
        Optional pre-built genai client. Primarily for tests: injecting a fake
        client (or setting :attr:`client` afterwards) avoids any network or
        credential dependency. When ``None`` the client is constructed lazily on
        first use.
    """

    def __init__(
        self,
        config: Config,
        interrupt: InterruptController,
        client: Any | None = None,
    ) -> None:
        self._config = config
        self._interrupt = interrupt
        self._client = client

    # -- client construction -------------------------------------------------

    @property
    def client(self) -> Any | None:
        """The underlying genai client (``None`` until lazily constructed)."""
        return self._client

    @client.setter
    def client(self, value: Any | None) -> None:
        self._client = value

    def _validate_config(self) -> None:
        """Raise :class:`ConfigMissingError` if project/region is absent (Req 2.4)."""
        if not self._config.project:
            raise ConfigMissingError(
                "GCP project ID is missing from configuration."
            )
        if not self._config.region:
            raise ConfigMissingError(
                "GCP region is missing from configuration."
            )

    def _get_client(self) -> Any:
        """Lazily construct the genai client, mapping setup failures (Req 2.1-2.4).

        Validates that project/region are present, then constructs
        ``genai.Client(vertexai=True, project=..., location=...)`` configured
        with the request timeout. Missing ADC and other credential failures are
        translated into :class:`CredentialsError`.
        """
        if self._client is not None:
            return self._client

        self._validate_config()

        if _genai is None:  # pragma: no cover - exercised only without the SDK
            raise ProviderError(
                "The 'google-genai' package is required to contact Vertex AI but "
                "is not installed."
            )

        http_options = None
        if _genai_types is not None:
            # HttpOptions.timeout is expressed in milliseconds.
            http_options = _genai_types.HttpOptions(
                timeout=int(self._config.request_timeout_s * 1000)
            )

        try:
            kwargs: dict[str, Any] = {
                "vertexai": True,
                "project": self._config.project,
                "location": self._config.region,
            }
            if http_options is not None:
                kwargs["http_options"] = http_options
            self._client = _genai.Client(**kwargs)
        except Exception as exc:  # noqa: BLE001 - translate to typed errors
            self._translate_and_raise(exc)
        return self._client

    # -- raw stream (isolated SDK call) --------------------------------------

    def _raw_stream(
        self, contents: list[dict], tools: list[ToolSpec]
    ) -> Iterator[Any]:
        """Return the raw SDK stream iterator for ``contents``/``tools``.

        Isolated from :meth:`generate_stream` so tests can monkeypatch it (or
        inject a fake client) and drive the translation / interrupt / retry
        layer without any network access.
        """
        client = self._get_client()
        # Translate Forge's internal wire-shape message dicts into the SDK's
        # Content/Part objects and pull the system prompt out into the request
        # config's system_instruction (Gemini's `contents` only accepts
        # 'user'/'model' roles, not 'system').
        sdk_contents, system_instruction = _to_sdk_contents(contents)
        config = self._build_request_config(tools, system_instruction)
        return client.models.generate_content_stream(
            model=self._config.model,
            contents=sdk_contents,
            config=config,
        )

    def _build_request_config(
        self, tools: list[ToolSpec], system_instruction: str | None = None
    ) -> Any | None:
        """Build the SDK request config (tools + system instruction + thinking)."""
        if _genai_types is None:
            return None
        kwargs: dict[str, Any] = {}
        sdk_tools = _to_sdk_tools(tools)
        if sdk_tools is not None:
            kwargs["tools"] = sdk_tools
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        thinking_config = self._build_thinking_config()
        if thinking_config is not None:
            kwargs["thinking_config"] = thinking_config
        if not kwargs:
            return None
        try:
            return _genai_types.GenerateContentConfig(**kwargs)
        except Exception:  # noqa: BLE001 - degrade gracefully without typed config
            # A thinking_config that this SDK/model build rejects must not drop
            # tools/system_instruction: retry once without it before giving up.
            if "thinking_config" in kwargs:
                kwargs.pop("thinking_config")
                try:
                    return _genai_types.GenerateContentConfig(**kwargs)
                except Exception:  # noqa: BLE001
                    return None
            return None

    def _build_thinking_config(self) -> Any | None:
        """Build a ``ThinkingConfig`` from ``provider.thinking_level`` if set.

        Returns ``None`` when no level is configured (leaving the model's own
        default thinking behavior untouched) or when the installed SDK does not
        accept the ``thinking_level`` field, so an older SDK degrades to the
        prior behavior rather than failing the request.
        """
        level = getattr(self._config, "provider_thinking_level", None)
        if not level or _genai_types is None:
            return None
        try:
            return _genai_types.ThinkingConfig(thinking_level=level)
        except Exception:  # noqa: BLE001 - SDK too old for thinking_level
            return None

    # -- public streaming API ------------------------------------------------

    def generate_stream(
        self,
        contents: list[dict],
        tools: list[ToolSpec],
    ) -> Iterator[StreamEvent]:
        """Stream a model response as :data:`StreamEvent` values."""
        self._validate_config()

        max_attempts = max(1, int(self._config.rate_limit_retries))
        emitted = False

        for attempt in range(1, max_attempts + 1):
            try:
                for event in self._stream_once(contents, tools):
                    emitted = True
                    yield event
                return
            except RateLimitError as exc:
                # Only retry when nothing has been emitted yet and attempts
                # remain; a mid-stream rate limit cannot be safely restarted.
                if emitted or attempt >= max_attempts:
                    raise
                if not self._backoff(attempt, retry_after=exc.retry_after):
                    # Interrupted during backoff: abort quietly.
                    return

    def _stream_once(
        self, contents: list[dict], tools: list[ToolSpec]
    ) -> Iterator[StreamEvent]:
        """Run a single streaming attempt with interrupt + timeout guards."""
        deadline = time.monotonic() + self._config.request_timeout_s

        try:
            raw_stream = self._raw_stream(contents, tools)
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._translate_and_raise(exc)

        stream_iter = iter(raw_stream)
        try:
            while True:
                # Interrupt check before pulling the next chunk (Req 4.2, 3.4).
                if self._interrupt.check():
                    return

                # Wall-clock timeout guard between chunks (Req 2.8).
                if time.monotonic() > deadline:
                    raise RequestTimeoutError(
                        f"Vertex AI request exceeded "
                        f"{self._config.request_timeout_s}s."
                    )

                try:
                    chunk = next(stream_iter)
                except StopIteration:
                    break
                except ProviderError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self._translate_and_raise(exc)

                for event in _parse_chunk(chunk):
                    yield event

                # Re-check the interrupt after emitting a chunk's events so an
                # abort is observed promptly between chunks.
                if self._interrupt.check():
                    return
        finally:
            _close_stream(stream_iter)

        yield Done()

    # -- retry / backoff ------------------------------------------------------

    def _backoff(self, attempt: int, retry_after: float | None = None) -> bool:
        """Wait before a rate-limit retry; return ``False`` if interrupted."""
        return wait_backoff(attempt, self._interrupt, retry_after)

    # -- exception translation -----------------------------------------------

    def _translate_and_raise(self, exc: BaseException) -> Any:
        """Translate an SDK / auth exception into a typed :class:`ProviderError`."""
        if _is_credentials_error(exc):
            raise CredentialsError() from exc

        status = _status_code_of(exc)
        type_name = type(exc).__name__.lower()

        # Timeout / deadline-exceeded.
        if (
            "timeout" in type_name
            or "deadline" in type_name
            or status == 504
        ):
            raise RequestTimeoutError(
                f"Vertex AI request exceeded "
                f"{self._config.request_timeout_s}s."
            ) from exc

        # Rate limiting.
        if status == 429 or "resourceexhausted" in type_name:
            raise RateLimitError(
                str(exc) or "Vertex AI rate limit exceeded.",
                retry_after=_retry_after_seconds(exc),
            ) from exc

        # Authorization errors.
        if status in (401, 403) or "permissiondenied" in type_name or (
            "unauthorized" in type_name
        ):
            raise AuthorizationError(
                str(exc) or "Vertex AI authorization error."
            ) from exc

        # Unknown failure: surface as a generic ProviderError
        raise ProviderError(str(exc) or repr(exc)) from exc


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _to_sdk_tools(specs: list[ToolSpec]) -> list[Any] | None:
    """Convert :class:`ToolSpec` list into the SDK tool/function shape."""
    if not specs:
        return None
    if _genai_types is None:
        return None
    try:
        declarations = [
            _genai_types.FunctionDeclaration(
                name=spec.name,
                description=spec.description,
                parameters=spec.parameters,
            )
            for spec in specs
        ]
        return [_genai_types.Tool(function_declarations=declarations)]
    except Exception:  # noqa: BLE001 - degrade gracefully on shape mismatch
        return None


def _to_sdk_contents(
    contents: list[Any],
) -> tuple[list[Any], str | None]:
    """Translate Forge wire-shape message dicts into SDK ``Content`` objects."""
    if _genai_types is None:
        return contents, None

    types = _genai_types
    system_parts: list[str] = []
    sdk_contents: list[Any] = []
    call_names: dict[str, str] = {}
    last_was_tool = False

    for msg in contents:
        if not isinstance(msg, dict):
            sdk_contents.append(msg)
            last_was_tool = False
            continue

        role = msg.get("role")
        text = msg.get("content")
        if text is None:
            text = msg.get("text")

        if role == "system":
            if text:
                system_parts.append(str(text))
            continue

        parts: list[Any] = []

        if text:
            parts.append(types.Part(text=str(text)))

        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            name = call.get("name") or ""
            args = call.get("args") or {}
            call_id = call.get("id")
            if call_id:
                call_names[call_id] = name
            signature = call.get("thought_signature")
            try:
                fc = types.FunctionCall(name=name, args=dict(args))
                part_kwargs: dict = {"function_call": fc}
                if signature:
                    try:
                        part_kwargs["thought_signature"] = base64.b64decode(
                            signature
                        )
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    parts.append(types.Part(**part_kwargs))
                except Exception:  # noqa: BLE001
                    parts.append(types.Part(function_call=fc))
            except Exception:  # noqa: BLE001
                continue

        result = msg.get("tool_result")
        if isinstance(result, dict):
            call_id = result.get("call_id")
            name = call_names.get(call_id or "", "") or "tool"
            ok = result.get("ok", True)
            if ok:
                response: dict = {"output": result.get("content", "")}
            else:
                response = {
                    "error": result.get("error") or result.get("content", "")
                }
            try:
                parts.append(
                    types.Part.from_function_response(name=name, response=response)
                )
            except Exception:  # noqa: BLE001
                parts.append(
                    types.Part(text=f"[tool result] {result.get('content', '')}")
                )

        if not parts:
            continue

        is_tool = role == "tool"
        if is_tool and last_was_tool and sdk_contents:
            sdk_contents[-1].parts.extend(parts)
        else:
            sdk_role = "model" if role == "model" else "user"
            sdk_contents.append(types.Content(role=sdk_role, parts=parts))
        last_was_tool = is_tool

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return sdk_contents, system_instruction


def _parse_chunk(chunk: Any) -> list[StreamEvent]:
    """Extract :data:`StreamEvent` values from one streamed SDK response chunk."""
    events: list[StreamEvent] = []

    candidates = getattr(chunk, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                events.append(TextDelta(text=text))

            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                signature = getattr(part, "thought_signature", None)
                events.append(
                    _function_call_to_tool_call(function_call, signature)
                )

    usage = getattr(chunk, "usage_metadata", None)
    if usage is not None:
        input_tokens = getattr(usage, "prompt_token_count", None) or 0
        output_tokens = getattr(usage, "candidates_token_count", None) or 0
        # Thinking models report reasoning tokens separately in
        # ``thoughts_token_count``; ``candidates_token_count`` excludes them.
        # Gemini bills those thinking tokens at the output rate, so fold them
        # into the output tally to keep both the token count and the estimated
        # cost accurate (otherwise a thinking model under-reports both).
        thoughts_tokens = getattr(usage, "thoughts_token_count", None) or 0
        events.append(
            UsageReport(
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens) + int(thoughts_tokens),
            )
        )

    return events


def _function_call_to_tool_call(
    function_call: Any, thought_signature: Any = None
) -> ToolCall:
    """Adapt an SDK ``FunctionCall`` into a :class:`forge.session.ToolCall`."""
    call_id = getattr(function_call, "id", None)
    name = getattr(function_call, "name", None) or ""
    args = getattr(function_call, "args", None)
    if args is None:
        args = {}
    elif not isinstance(args, dict):
        try:
            args = dict(args)
        except Exception:  # noqa: BLE001
            args = {}
    if not call_id:
        import uuid

        call_id = f"{name}-{uuid.uuid4().hex[:8]}"

    signature: str | None = None
    if isinstance(thought_signature, (bytes, bytearray)):
        signature = base64.b64encode(bytes(thought_signature)).decode("ascii")
    elif isinstance(thought_signature, str) and thought_signature:
        signature = thought_signature

    return ToolCall(
        id=call_id, name=name, args=args, thought_signature=signature
    )


def _status_code_of(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from an exception."""
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Best-effort extraction of a server-advised retry delay (in seconds)."""
    try:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            value = None
            getter = getattr(headers, "get", None)
            if callable(getter):
                value = getter("Retry-After") or getter("retry-after")
            if value is not None:
                return float(int(str(value).strip()))
    except Exception:  # noqa: BLE001
        pass

    try:
        details = getattr(exc, "details", None)
        if isinstance(details, (list, tuple)):
            for detail in details:
                delay = None
                if isinstance(detail, dict):
                    delay = detail.get("retryDelay") or detail.get("retry_delay")
                else:
                    delay = getattr(detail, "retry_delay", None) or getattr(
                        detail, "retryDelay", None
                    )
                seconds = _coerce_delay_seconds(delay)
                if seconds is not None:
                    return seconds
    except Exception:  # noqa: BLE001
        pass

    for attr in ("retry_delay", "retry_after"):
        seconds = _coerce_delay_seconds(getattr(exc, attr, None))
        if seconds is not None:
            return seconds

    return None


def _coerce_delay_seconds(value: Any) -> float | None:
    """Coerce a retry-delay value into seconds, or ``None`` if not usable."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    if isinstance(value, str):
        text = value.strip().lower().rstrip("s").strip()
        try:
            seconds = float(text)
        except ValueError:
            return None
        return seconds if seconds > 0 else None
    if isinstance(value, dict):
        secs = value.get("seconds")
        if isinstance(secs, (int, float)):
            return float(secs) if secs > 0 else None
        return None
    secs = getattr(value, "seconds", None)
    if isinstance(secs, (int, float)):
        nanos = getattr(value, "nanos", 0) or 0
        total = float(secs) + float(nanos) / 1e9
        return total if total > 0 else None
    return None


def _is_credentials_error(exc: BaseException) -> bool:
    """Return whether ``exc`` represents an ADC / credentials failure."""
    if (
        _google_auth_exceptions is not None
        and isinstance(
            exc, getattr(_google_auth_exceptions, "GoogleAuthError", ())
        )
    ):
        return True
    type_name = type(exc).__name__.lower()
    return "credentials" in type_name or "defaultcredentials" in type_name


def _close_stream(stream_iter: Any) -> None:
    """Best-effort close of an SDK stream iterator on early exit/abort."""
    close = getattr(stream_iter, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            pass
