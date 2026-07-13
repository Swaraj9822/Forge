"""Vertex AI / Gemini client for Forge.

This module wraps the unified ``google-genai`` SDK (initialized for Vertex AI)
behind a small, testable surface. :class:`VertexClient` exposes a single
streaming method, :meth:`VertexClient.generate_stream`, which yields a tagged
union of :data:`StreamEvent` values (:class:`TextDelta`, :class:`ToolCall`,
:class:`UsageReport`, :class:`Done`) so the REPL can render text deltas,
announce tool names, and capture token usage.

Design notes
------------
* **Lazy client construction.** ``genai.Client(vertexai=True, project=...,
  location=...)`` is constructed on first use rather than in ``__init__`` so a
  missing ADC / project / region is detected at call time and a
  :class:`VertexClient` can exist without any network or credential check
  (Req 2.3, 2.4).
* **Typed exceptions.** SDK / google-api-core failures are translated into the
  module's typed exceptions (:class:`CredentialsError`,
  :class:`ConfigMissingError`, :class:`AuthorizationError`,
  :class:`RateLimitError`, :class:`RequestTimeoutError`) so the agent loop can
  render them without losing session state (Req 2.3-2.8).
* **Rate-limit retry.** A rate-limit response is retried up to
  ``config.rate_limit_retries`` attempts with exponential backoff; once the
  attempts are exhausted a :class:`RateLimitError` is raised (Req 2.6).
* **Request timeout.** The 60s (``config.request_timeout_s``) timeout is applied
  both through the SDK's ``http_options`` and a wall-clock guard between
  streamed chunks (Req 2.8).
* **Interrupt.** ``interrupt.check()`` is polled between streamed chunks so an
  in-flight generation aborts within ~1 second; already-yielded tokens are
  retained by the caller (Req 3.4, 4.2).
* **Offline testability.** Google packages are imported guardedly so this
  module imports even when ``google-genai`` is not installed. The raw SDK
  stream is isolated in :meth:`VertexClient._raw_stream`, and a client may be
  injected, so tests can drive :meth:`generate_stream` with mocked responses
  without any network access (task 16.2).
"""

from __future__ import annotations

import base64
import random
import time
from dataclasses import dataclass
from typing import Any, Iterator, Union

from forge.config import Config
from forge.interrupt import InterruptController
from forge.session import ToolCall
from forge.tools.base import ToolSpec

__all__ = [
    "TextDelta",
    "ToolCall",
    "UsageReport",
    "Done",
    "StreamEvent",
    "VertexError",
    "CredentialsError",
    "ConfigMissingError",
    "AuthorizationError",
    "RateLimitError",
    "RequestTimeoutError",
    "VertexClient",
]


# ---------------------------------------------------------------------------
# Rate-limit backoff tuning
#
# Preview Gemini models on Vertex run under Dynamic Shared Quota, so a 429 is
# usually transient capacity throttling rather than an exhausted project quota.
# A longer, capped exponential backoff (with jitter to avoid synchronized
# retries) clears these far more reliably than a sub-second schedule, and the
# server's own Retry-After/RetryInfo hint is honored when present.
# ---------------------------------------------------------------------------

#: Base delay (seconds) for the first rate-limit retry; doubles each attempt.
BACKOFF_BASE_S = 1.0
#: Upper bound (seconds) on any single backoff wait, before jitter.
BACKOFF_CAP_S = 30.0
#: Fraction of the computed delay added as uniform random jitter (0..frac).
BACKOFF_JITTER_FRAC = 0.25


# ---------------------------------------------------------------------------
# Guarded SDK imports
#
# The google packages are imported defensively so this module loads even when
# ``google-genai`` is not installed. The hard requirement is deferred to call
# time (client construction), where a clear error is raised.
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
# Stream events (tagged union)
# ---------------------------------------------------------------------------


@dataclass
class TextDelta:
    """A streamed fragment of model-generated text."""

    text: str


@dataclass
class UsageReport:
    """Per-response token usage extracted from ``usage_metadata`` (Req 17.1)."""

    input_tokens: int
    output_tokens: int


@dataclass
class Done:
    """Sentinel emitted when a response stream completes normally (Req 3.3)."""

    pass


# ``ToolCall`` (re-exported from :mod:`forge.session`) is the tool-call stream
# event so the whole agent loop shares one ``ToolCall`` type.
StreamEvent = Union[TextDelta, ToolCall, UsageReport, Done]


# ---------------------------------------------------------------------------
# Typed exceptions
# ---------------------------------------------------------------------------


class VertexError(Exception):
    """Base class for all errors surfaced by :class:`VertexClient`."""


class CredentialsError(VertexError):
    """ADC are missing or invalid when the client initializes (Req 2.3)."""

    DEFAULT_MESSAGE = (
        "Application Default Credentials (ADC) are unavailable. Establish them by "
        "running: gcloud auth application-default login"
    )

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE)


class ConfigMissingError(VertexError):
    """The GCP project ID or region is absent from configuration (Req 2.4)."""


class AuthorizationError(VertexError):
    """Vertex AI rejected the request with an authorization error (Req 2.5)."""


class RateLimitError(VertexError):
    """Rate-limit response persisted after all retries were exhausted (Req 2.6).

    ``retry_after`` carries the server-advised wait in seconds when the response
    exposed one (an HTTP ``Retry-After`` header or a ``RetryInfo``/``retryDelay``
    in the error details); it is ``None`` when the server gave no hint. The
    retry loop honors it so the backoff respects the server's pacing rather than
    relying solely on the local exponential schedule.
    """

    def __init__(
        self, message: str | None = None, *, retry_after: float | None = None
    ) -> None:
        super().__init__(message or "Vertex AI rate limit exceeded.")
        self.retry_after = retry_after


class RequestTimeoutError(VertexError):
    """The request exceeded ``config.request_timeout_s`` (Req 2.8)."""


# ---------------------------------------------------------------------------
# VertexClient
# ---------------------------------------------------------------------------


class VertexClient:
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
            raise VertexError(
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
        """Build the SDK request config (tools + system instruction)."""
        if _genai_types is None:
            return None
        kwargs: dict[str, Any] = {}
        sdk_tools = _to_sdk_tools(tools)
        if sdk_tools is not None:
            kwargs["tools"] = sdk_tools
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if not kwargs:
            return None
        try:
            return _genai_types.GenerateContentConfig(**kwargs)
        except Exception:  # noqa: BLE001 - degrade gracefully without typed config
            return None

    # -- public streaming API ------------------------------------------------

    def generate_stream(
        self,
        contents: list[dict],
        tools: list[ToolSpec],
    ) -> Iterator[StreamEvent]:
        """Stream a model response as :data:`StreamEvent` values.

        Yields :class:`TextDelta` for text fragments, :class:`ToolCall` for
        function calls, :class:`UsageReport` when ``usage_metadata`` is present,
        and a final :class:`Done` when the stream completes normally.

        Behavior:

        * A tripped interrupt between chunks stops iteration promptly (within
          ~1s); already-yielded events are retained by the caller and no
          :class:`Done` is emitted (Req 3.4, 4.2).
        * A rate-limit response is retried up to ``config.rate_limit_retries``
          attempts with a capped, jittered exponential backoff before a
          :class:`RateLimitError` is raised. When the response carried a
          server-advised ``Retry-After``/``RetryInfo`` hint it is honored (the
          wait is at least that long). Retries only occur while no event has
          been emitted yet, so a mid-stream rate limit is not retried (Req 2.6).
        * The request is bounded by ``config.request_timeout_s`` via a wall-clock
          guard between chunks (Req 2.8).
        """
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
        except VertexError:
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
                except VertexError:
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
        """Wait before a rate-limit retry; return ``False`` if interrupted.

        The wait is a capped exponential schedule with jitter:
        ``min(BACKOFF_BASE_S * 2**(attempt-1), BACKOFF_CAP_S)`` plus up to
        ``BACKOFF_JITTER_FRAC`` of that as uniform random jitter (jitter
        desynchronizes retries when several clients are throttled at once). When
        the server advised a ``retry_after`` (seconds), the wait is at least
        that long so the server's pacing is respected. The wait is performed on
        the interrupt event so a Ctrl-C ends the backoff promptly; ``False`` is
        returned when the interrupt tripped during the wait so the caller aborts.
        """
        capped = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_CAP_S)
        delay = capped + random.uniform(0.0, capped * BACKOFF_JITTER_FRAC)
        if retry_after is not None and retry_after > 0:
            delay = max(delay, float(retry_after))
        tripped = self._interrupt.event.wait(timeout=delay)
        return not tripped

    # -- exception translation -----------------------------------------------

    def _translate_and_raise(self, exc: BaseException) -> Any:
        """Translate an SDK / auth exception into a typed :class:`VertexError`.

        Detection is defensive: it matches by HTTP status code where available
        and otherwise by exception type name, so it stays robust across SDK
        versions and works even when optional google packages are absent.
        """
        # Credentials / ADC failures.
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

        # Unknown failure: surface as a generic VertexError so the agent loop
        # can render it without losing session state.
        raise VertexError(str(exc) or repr(exc)) from exc


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _to_sdk_tools(specs: list[ToolSpec]) -> list[Any] | None:
    """Convert :class:`ToolSpec` list into the SDK tool/function shape.

    Degrades gracefully to ``None`` when there are no specs or when the SDK
    ``types`` module is unavailable, so callers can omit tools entirely.
    """
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
    """Translate Forge wire-shape message dicts into SDK ``Content`` objects.

    Forge carries conversation messages internally as plain dicts shaped like
    ``{"role": ..., "content": ..., "tool_calls": [...], "tool_result": {...}}``
    (see :func:`forge.context._message_to_window_dict`). The ``google-genai``
    SDK instead expects ``contents`` to be a list of ``types.Content`` whose
    ``role`` is only ``"user"`` or ``"model"`` and whose ``parts`` hold the
    text / function-call / function-response payloads. The system prompt is not
    a content turn at all — it must be passed separately as the request's
    ``system_instruction``.

    This helper performs that translation and returns
    ``(sdk_contents, system_instruction)``:

    * ``"system"`` messages are concatenated into the returned
      ``system_instruction`` string and excluded from ``sdk_contents``;
    * ``"model"`` messages become ``role="model"`` content; every other role
      (``"user"``, ``"tool"``) becomes ``role="user"`` content, since Gemini
      only accepts those two content roles;
    * body text becomes a text ``Part``; each ``tool_calls`` entry becomes a
      ``function_call`` ``Part``; a ``tool_result`` becomes a
      ``function_response`` ``Part`` whose name is correlated back to the
      emitting call's id;
    * **consecutive ``"tool"`` messages are coalesced into a single
      ``role="user"`` content** so that all function-response parts answering a
      parallel-function-call model turn travel in one turn. Gemini requires the
      number of function-response parts in the turn following a function-call
      turn to equal the number of function-call parts; emitting one content per
      tool result splits them across turns and the API rejects the request
      ("the number of function response parts is equal to the number of
      function call parts of the function call turn"); and
    * messages that would carry no parts are skipped (Gemini rejects empty
      content).

    Degrades to returning the input unchanged when the SDK ``types`` module is
    unavailable (only reachable without ``google-genai`` installed).
    """
    if _genai_types is None:
        return contents, None

    types = _genai_types
    system_parts: list[str] = []
    sdk_contents: list[Any] = []
    # Correlate a tool_result back to the function name of the call that
    # produced it (FunctionResponse requires the function name, which the
    # tool_result wire dict does not carry).
    call_names: dict[str, str] = {}
    # Track whether the last appended Content was a coalescible tool-result
    # turn so consecutive tool results (the responses to a parallel-function-
    # call model turn) are merged into ONE user content rather than split
    # across several — which Gemini rejects (function-call/response part counts
    # must match for the turn).
    last_was_tool = False

    for msg in contents:
        if not isinstance(msg, dict):
            # Already an SDK Content / string: pass through untouched.
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
                # Echo back the thought_signature Gemini 3 requires on the
                # function-call part it originally produced.
                if signature:
                    try:
                        part_kwargs["thought_signature"] = base64.b64decode(
                            signature
                        )
                    except Exception:  # noqa: BLE001 - drop an undecodable sig
                        pass
                try:
                    parts.append(types.Part(**part_kwargs))
                except Exception:  # noqa: BLE001 - SDK may not accept the kwarg
                    parts.append(types.Part(function_call=fc))
            except Exception:  # noqa: BLE001 - skip a malformed call
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
            except Exception:  # noqa: BLE001 - fall back to a text rendering
                parts.append(
                    types.Part(text=f"[tool result] {result.get('content', '')}")
                )

        if not parts:
            continue

        is_tool = role == "tool"
        # Coalesce a run of tool-result turns into the single user content that
        # answers the preceding function-call turn (see the part-count rule
        # above). A non-tool message always starts a fresh content.
        if is_tool and last_was_tool and sdk_contents:
            sdk_contents[-1].parts.extend(parts)
        else:
            sdk_role = "model" if role == "model" else "user"
            sdk_contents.append(types.Content(role=sdk_role, parts=parts))
        last_was_tool = is_tool

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return sdk_contents, system_instruction


def _parse_chunk(chunk: Any) -> list[StreamEvent]:
    """Extract :data:`StreamEvent` values from one streamed SDK response chunk.

    Pulls text fragments and function calls from candidate parts and a
    :class:`UsageReport` from ``usage_metadata`` when present. All attribute
    access is defensive so partial/oddly-shaped chunks never raise.
    """
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
                # Gemini 3 attaches a thought_signature to the part carrying a
                # function call; capture it so it can be echoed back on the next
                # request (required, or the API rejects the call).
                signature = getattr(part, "thought_signature", None)
                events.append(
                    _function_call_to_tool_call(function_call, signature)
                )

    usage = getattr(chunk, "usage_metadata", None)
    if usage is not None:
        input_tokens = getattr(usage, "prompt_token_count", None) or 0
        output_tokens = getattr(usage, "candidates_token_count", None) or 0
        events.append(
            UsageReport(
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
            )
        )

    return events


def _function_call_to_tool_call(
    function_call: Any, thought_signature: Any = None
) -> ToolCall:
    """Adapt an SDK ``FunctionCall`` into a :class:`forge.session.ToolCall`.

    ``thought_signature`` (when present on the response part) is base64-encoded
    to a string so it round-trips through JSON session persistence and can be
    re-attached to the function-call part on the next request.
    """
    call_id = getattr(function_call, "id", None)
    name = getattr(function_call, "name", None) or ""
    args = getattr(function_call, "args", None)
    if args is None:
        args = {}
    elif not isinstance(args, dict):
        # ``args`` is normally a mapping; coerce defensively.
        try:
            args = dict(args)
        except Exception:  # noqa: BLE001
            args = {}
    if not call_id:
        # The SDK may omit an id; mint a stable-enough one from the name.
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
    """Best-effort extraction of a server-advised retry delay (in seconds).

    Vertex / google-api-core surface a retry hint in a few shapes; this checks
    them defensively and returns the first usable value, or ``None`` when no
    hint is present:

    * an HTTP ``Retry-After`` response header (integer seconds), reachable via a
      ``response.headers`` mapping on the exception;
    * a ``RetryInfo`` in the error ``details`` carrying a ``retryDelay`` like
      ``"7s"`` or ``{"seconds": 7}`` (the google.rpc.RetryInfo shape); or
    * a ``retry_delay``/``retry_after`` attribute exposing ``.seconds`` or a
      raw number.

    All access is wrapped so a malformed payload never raises.
    """

    # 1. HTTP Retry-After header.
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
    except Exception:  # noqa: BLE001 - header is best-effort
        pass

    # 2. RetryInfo in structured error details.
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
    except Exception:  # noqa: BLE001 - details shape varies across SDK versions
        pass

    # 3. A retry_delay/retry_after attribute on the exception itself.
    for attr in ("retry_delay", "retry_after"):
        seconds = _coerce_delay_seconds(getattr(exc, attr, None))
        if seconds is not None:
            return seconds

    return None


def _coerce_delay_seconds(value: Any) -> float | None:
    """Coerce a retry-delay value into seconds, or ``None`` if not usable.

    Accepts a numeric value, a duration string like ``"7s"``/``"7.5s"``, a
    mapping with a ``seconds`` key, or a protobuf ``Duration``-like object with a
    ``.seconds`` (and optional ``.nanos``) attribute.
    """
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
        except Exception:  # noqa: BLE001 - closing is best-effort
            pass
