"""Anthropic provider for Forge."""

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
    "AnthropicProvider",
]

# Anthropic REQUIRES an explicit max_tokens, and it caps *output* tokens (not
# context) — it must stay within the model's maximum output or the request is
# rejected. ``config.token_limit`` is the context budget (~200k) and is far too
# large to use here, so bound the output with a conservative default.
_DEFAULT_MAX_OUTPUT_TOKENS = 8192

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None


class AnthropicProvider(Provider):
    """Streams responses from Anthropic using the ``anthropic`` SDK."""

    def __init__(self, config: Config, interrupt: InterruptController) -> None:
        self._config = config
        self._interrupt = interrupt
        self._client = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if _anthropic is None:
            raise ProviderError(
                "The 'anthropic' package is required to use Anthropic but is not installed."
            )

        api_key_var = self._config.provider_api_key_env or "ANTHROPIC_API_KEY"
        api_key = os.environ.get(api_key_var)
        if not api_key:
            raise CredentialsError(
                f"Anthropic API key is missing. Set the environment variable '{api_key_var}'."
            )

        try:
            self._client = _anthropic.Anthropic(
                api_key=api_key,
                timeout=float(self._config.request_timeout_s),
            )
        except Exception as exc:
            self._translate_and_raise(exc)
        return self._client

    def generate_stream(
        self,
        contents: list[dict],
        tools: list[ToolSpec],
    ) -> Iterator[StreamEvent]:
        # Validate credentials exist before starting the stream loop
        api_key_var = self._config.provider_api_key_env or "ANTHROPIC_API_KEY"
        if not os.environ.get(api_key_var):
            raise CredentialsError(
                f"Anthropic API key is missing. Set the environment variable '{api_key_var}'."
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

        messages, system_instruction = _to_anthropic_messages(contents)
        anthropic_tools = _to_anthropic_tools(tools)

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": min(self._config.token_limit, _DEFAULT_MAX_OUTPUT_TOKENS),
            "messages": messages,
        }
        if system_instruction:
            kwargs["system"] = system_instruction
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        try:
            raw_stream = client.messages.create(stream=True, **kwargs)
        except Exception as exc:
            self._translate_and_raise(exc)

        # Track active tool use blocks being streamed
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
                        f"Anthropic request exceeded {self._config.request_timeout_s}s."
                    )

                try:
                    event = next(stream_iter)
                except StopIteration:
                    break
                except Exception as exc:
                    self._translate_and_raise(exc)

                event_type = getattr(event, "type", None)

                if event_type == "message_start":
                    msg = getattr(event, "message", None)
                    usage = getattr(msg, "usage", None)
                    if usage:
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                        output_tokens = getattr(usage, "output_tokens", 0) or 0
                        yield UsageReport(input_tokens, output_tokens)

                elif event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    idx = getattr(event, "index", 0)
                    if block and getattr(block, "type", None) == "tool_use":
                        active_tool_calls[idx] = {
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                            "partial_json": [],
                        }

                elif event_type == "content_block_delta":
                    idx = getattr(event, "index", 0)
                    delta = getattr(event, "delta", None)
                    if delta:
                        delta_type = getattr(delta, "type", None)
                        if delta_type == "text_delta":
                            yield TextDelta(text=getattr(delta, "text", ""))
                        elif delta_type == "input_json_delta":
                            if idx in active_tool_calls:
                                active_tool_calls[idx]["partial_json"].append(
                                    getattr(delta, "partial_json", "")
                                )

                elif event_type == "content_block_stop":
                    idx = getattr(event, "index", 0)
                    if idx in active_tool_calls:
                        call_info = active_tool_calls.pop(idx)
                        raw_args = "".join(call_info["partial_json"])
                        import json
                        try:
                            args = json.loads(raw_args) if raw_args.strip() else {}
                        except Exception:
                            args = {}
                        yield ToolCall(
                            id=call_info["id"],
                            name=call_info["name"],
                            args=args,
                        )

                elif event_type == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage:
                        input_tokens = getattr(usage, "input_tokens", input_tokens) or input_tokens
                        output_tokens = getattr(usage, "output_tokens", output_tokens) or output_tokens
                        yield UsageReport(input_tokens, output_tokens)

                if self._interrupt.check():
                    return
        finally:
            _close_stream(stream_iter)

        yield Done()

    def _translate_and_raise(self, exc: BaseException) -> Any:
        type_name = type(exc).__name__.lower()

        if "timeout" in type_name or "connect" in type_name:
            raise RequestTimeoutError(
                f"Anthropic request exceeded {self._config.request_timeout_s}s."
            ) from exc

        status = getattr(exc, "status_code", None)
        if status in (401, 403):
            raise AuthorizationError(
                str(exc) or "Anthropic authorization error."
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
                str(exc) or "Anthropic rate limit exceeded.",
                retry_after=retry_after,
            ) from exc

        raise ProviderError(str(exc) or repr(exc)) from exc

    def _backoff(self, attempt: int, retry_after: float | None = None) -> bool:
        return wait_backoff(attempt, self._interrupt, retry_after)


def _to_anthropic_messages(contents: list[dict]) -> tuple[list[dict], str | None]:
    """Convert Forge wire-shape messages to Anthropic messages."""
    system_parts: list[str] = []
    messages: list[dict] = []

    for msg in contents:
        if not isinstance(msg, dict):
            messages.append(msg)
            continue

        role = msg.get("role")
        text = msg.get("content") or msg.get("text")

        if role == "system":
            if text:
                system_parts.append(str(text))
            continue

        content_blocks: list[dict] = []
        if text:
            content_blocks.append({"type": "text", "text": str(text)})

        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            content_blocks.append({
                "type": "tool_use",
                "id": call.get("id") or f"call_{len(content_blocks)}",
                "name": call.get("name"),
                "input": call.get("args") or {},
            })

        result = msg.get("tool_result")
        if isinstance(result, dict):
            ok = result.get("ok", True)
            content_str = result.get("content", "")
            err = result.get("error")

            tool_res = {
                "type": "tool_result",
                "tool_use_id": result.get("call_id") or "",
                "content": content_str if ok else (err or content_str),
            }
            if not ok:
                tool_res["is_error"] = True
            content_blocks.append(tool_res)

        if not content_blocks:
            continue

        mapped_role = "assistant" if role == "model" else "user"

        if messages and messages[-1]["role"] == mapped_role:
            messages[-1]["content"].extend(content_blocks)
        else:
            messages.append({
                "role": mapped_role,
                "content": content_blocks,
            })

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return messages, system_instruction


def _to_anthropic_tools(specs: list[ToolSpec]) -> list[dict] | None:
    """Convert ToolSpec list to Anthropic tools schema."""
    if not specs:
        return None
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.parameters,
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
