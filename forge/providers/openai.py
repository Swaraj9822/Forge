"""OpenAI provider for Forge."""

from __future__ import annotations

import os
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
    AuthorizationError,
    RateLimitError,
    RequestTimeoutError,
    wait_backoff,
)

__all__ = [
    "OpenAIProvider",
]

try:
    import openai as _openai
except ImportError:
    _openai = None


class OpenAIProvider(Provider):
    """Streams responses from OpenAI using the ``openai`` SDK."""

    def __init__(self, config: Config, interrupt: InterruptController) -> None:
        self._config = config
        self._interrupt = interrupt
        self._client = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if _openai is None:
            raise ProviderError(
                "The 'openai' package is required to use OpenAI but is not installed."
            )

        api_key_var = self._config.provider_api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_var)
        if not api_key:
            raise CredentialsError(
                f"OpenAI API key is missing. Set the environment variable '{api_key_var}'."
            )

        base_url = self._config.provider_base_url
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": float(self._config.request_timeout_s),
        }
        if base_url:
            kwargs["base_url"] = base_url

        try:
            self._client = _openai.OpenAI(**kwargs)
        except Exception as exc:
            self._translate_and_raise(exc)
        return self._client

    def generate_stream(
        self,
        contents: list[dict],
        tools: list[ToolSpec],
    ) -> Iterator[StreamEvent]:
        # Validate credentials exist before starting the stream loop
        api_key_var = self._config.provider_api_key_env or "OPENAI_API_KEY"
        if not os.environ.get(api_key_var):
            raise CredentialsError(
                f"OpenAI API key is missing. Set the environment variable '{api_key_var}'."
            )

        max_attempts = max(1, int(self._config.rate_limit_retries))
        emitted = False

        for attempt in range(1, max_attempts + 1):
            try:
                for event in self._stream_once(contents, tools):
                    emitted = True
                    yield event
                return
            except RateLimitError as exc:
                if emitted or attempt >= max_attempts:
                    raise
                if not self._backoff(attempt, exc.retry_after):
                    return

    def _stream_once(
        self,
        contents: list[dict],
        tools: list[ToolSpec],
    ) -> Iterator[StreamEvent]:
        client = self._get_client()
        deadline = time.monotonic() + self._config.request_timeout_s

        messages = _to_openai_messages(contents)
        openai_tools = _to_openai_tools(tools)

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools
            
        # Standard OpenAI streams do not return usage metrics unless stream_options is configured
        kwargs["stream_options"] = {"include_usage": True}

        try:
            raw_stream = client.chat.completions.create(stream=True, **kwargs)
        except Exception as exc:
            self._translate_and_raise(exc)

        # Track active tool call deltas being streamed
        active_tool_calls: dict[int, dict] = {}
        input_tokens = 0
        output_tokens = 0

        stream_iter = iter(raw_stream)
        try:
            while True:
                if self._interrupt.check():
                    return

                if time.monotonic() > deadline:
                    raise RequestTimeoutError(
                        f"OpenAI request exceeded {self._config.request_timeout_s}s."
                    )

                try:
                    chunk = next(stream_iter)
                except StopIteration:
                    break
                except Exception as exc:
                    self._translate_and_raise(exc)

                # 1. Parse top-level usage metadata if present
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(usage, "completion_tokens", 0) or 0
                    yield UsageReport(input_tokens, output_tokens)

                # 2. Parse choices/deltas
                choices = getattr(chunk, "choices", None) or []
                for choice in choices:
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue

                    # Text delta
                    content = getattr(delta, "content", None)
                    if content:
                        yield TextDelta(text=content)

                    # Tool call delta
                    tool_calls = getattr(delta, "tool_calls", None) or []
                    for tc in tool_calls:
                        idx = getattr(tc, "index", None)
                        if idx is None:
                            continue

                        if idx not in active_tool_calls:
                            active_tool_calls[idx] = {
                                "id": getattr(tc, "id", None) or "",
                                "name": "",
                                "partial_arguments": [],
                            }

                        # Update fields when they stream in
                        tc_id = getattr(tc, "id", None)
                        if tc_id:
                            active_tool_calls[idx]["id"] = tc_id

                        func = getattr(tc, "function", None)
                        if func:
                            name = getattr(func, "name", None)
                            if name:
                                active_tool_calls[idx]["name"] = name
                            args = getattr(func, "arguments", None)
                            if args:
                                active_tool_calls[idx]["partial_arguments"].append(args)

                if self._interrupt.check():
                    return
        finally:
            _close_stream(stream_iter)

        # Emit accumulated tool calls
        for idx, call_info in active_tool_calls.items():
            raw_args = "".join(call_info["partial_arguments"])
            import json
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except Exception:
                args = {}
            yield ToolCall(
                id=call_info["id"] or f"call_{idx}",
                name=call_info["name"] or "tool",
                args=args,
            )

        yield Done()

    def _translate_and_raise(self, exc: BaseException) -> Any:
        type_name = type(exc).__name__.lower()

        if "timeout" in type_name or "connect" in type_name:
            raise RequestTimeoutError(
                f"OpenAI request exceeded {self._config.request_timeout_s}s."
            ) from exc

        status = getattr(exc, "status_code", None)
        if status in (401, 403):
            raise AuthorizationError(
                str(exc) or "OpenAI authorization error."
            ) from exc

        if status == 429:
            retry_after = None
            response = getattr(exc, "response", None)
            if response:
                headers = getattr(response, "headers", None)
                if headers:
                    val = headers.get("retry-after") or headers.get("Retry-After")
                    if val:
                        try:
                            retry_after = float(val)
                        except ValueError:
                            pass
            raise RateLimitError(
                str(exc) or "OpenAI rate limit exceeded.",
                retry_after=retry_after,
            ) from exc

        raise ProviderError(str(exc) or repr(exc)) from exc

    def _backoff(self, attempt: int, retry_after: float | None = None) -> bool:
        return wait_backoff(attempt, self._interrupt, retry_after)


def _to_openai_messages(contents: list[dict]) -> list[dict]:
    """Convert Forge wire-shape messages to OpenAI chat format."""
    messages: list[dict] = []
    call_names: dict[str, str] = {}

    for msg in contents:
        if not isinstance(msg, dict):
            messages.append(msg)
            continue

        role = msg.get("role")
        text = msg.get("content") or msg.get("text")

        if role == "system":
            messages.append({"role": "system", "content": str(text or "")})
            continue

        if role == "user":
            messages.append({"role": "user", "content": str(text or "")})
            continue

        if role == "model":
            tool_calls = []
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("id") or ""
                name = call.get("name") or ""
                call_names[call_id] = name
                import json
                args_str = json.dumps(call.get("args") or {})
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args_str,
                    },
                })
            item: dict[str, Any] = {
                "role": "assistant",
                "content": str(text) if text else None,
            }
            if tool_calls:
                item["tool_calls"] = tool_calls
            messages.append(item)
            continue

        if role == "tool":
            result = msg.get("tool_result")
            if isinstance(result, dict):
                call_id = result.get("call_id") or ""
                name = call_names.get(call_id, "tool")
                ok = result.get("ok", True)
                content_str = result.get("content", "")
                err = result.get("error")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": content_str if ok else (err or content_str),
                })
            continue

    return messages


def _to_openai_tools(specs: list[ToolSpec]) -> list[dict] | None:
    """Convert ToolSpec list to OpenAI tools schema."""
    if not specs:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in specs
    ]


def _close_stream(stream_iter: Any) -> None:
    """Best-effort close of the stream iterator."""
    close = getattr(stream_iter, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            pass
