"""Context assembly, system-prompt construction, and token estimation.

This module implements the ``ContextManager``, which is responsible for:

* loading the built-in default system prompt shipped as package data
  (``forge/data/system_prompt.md``) via :mod:`importlib.resources`;
* assembling the system context by placing the built-in default prompt first,
  followed by the contents of each configured steering file in the order they
  are listed in the configuration, warning about and skipping any missing
  steering file (Requirements 15.1-15.4, Property 22); and
* estimating the token count of a Context_Window with a deterministic, offline
  heuristic so compaction decisions can be made before a request is sent
  (Requirement 14.1).

The compaction algorithm (summarizing the middle region via the injected
summarizer, dropping retained-recent messages when summarization alone is not
enough, and emitting the compaction notice) is implemented here in ``compact``
and wired into ``assemble`` so that an over-limit window is compacted before it
is returned (Requirements 14.2-14.9, Property 21).

See the design document's "ContextManager" section.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from .config import Config
from .session import Message, Session

# --------------------------------------------------------------------------- #
# Token-estimation constants
# --------------------------------------------------------------------------- #

# The standard conservative approximation of ~4 characters per token used by
# the deterministic local heuristic (Req 14.1). Estimation never makes a
# network call so it is fast, offline, and reproducible.
CHARS_PER_TOKEN = 4

# A small fixed per-message overhead (in tokens) added on top of the
# character-based estimate for every message. Real tokenizers add a few tokens
# of structural framing (role markers, message delimiters) per message; this
# constant approximates that framing so multi-message windows are not
# under-counted. The exact value is a heuristic; 4 is a reasonable small
# constant in line with the 4-chars-per-token ratio.
PER_MESSAGE_OVERHEAD_TOKENS = 4

# The package and resource name of the bundled built-in default system prompt.
_DATA_PACKAGE = "forge.data"
_SYSTEM_PROMPT_RESOURCE = "system_prompt.md"

# --------------------------------------------------------------------------- #
# Context-provider seam
# --------------------------------------------------------------------------- #

# Per-project instruction files auto-loaded as system context (Feature E).
DEFAULT_PROJECT_MEMORY_FILES: tuple[str, ...] = ("FORGE.md", "AGENTS.md")


@runtime_checkable
class ContextProvider(Protocol):
    """Supplies ephemeral, non-persisted segments appended to the context window.

    A provider is consulted once per :meth:`ContextManager.assemble`. It returns
    zero or more wire-shape message dicts (``{"role": ..., "content": ...}``)
    that are appended to the *assembled window only* — they are never written to
    ``session.messages`` and therefore never persisted. Returning an empty list
    means "nothing to add this turn", which must leave the window byte-identical
    to the no-provider case.
    """

    def segments(self, session: "Session") -> list[dict]:
        ...


# --------------------------------------------------------------------------- #
# Compaction constants
# --------------------------------------------------------------------------- #

# The instruction prepended to the middle-region transcript when the summary is
# produced by a Model (VertexClient-like summarizer). It asks for a structured
# summary that preserves the decisions and outcomes the agent needs to keep
# working after the earlier turns are dropped (Req 14.3, 14.4).
SUMMARY_INSTRUCTION = (
    "You are compacting the earlier portion of a conversation between a "
    "developer and an AI coding agent so the agent can keep working without "
    "exceeding the model's context limit. Summarize the messages below into a "
    "concise, structured summary. Preserve every decision that was made, every "
    "action and tool invocation and its outcome, and any facts, file paths, or "
    "constraints needed to continue the task. Do not invent information."
)

# Prefix used to label the single synthetic message that replaces the
# summarized middle region, so it is recognizable in the assembled window.
SUMMARY_MESSAGE_PREFIX = "[Summary of earlier conversation]"

# Conservative upper bound, in *estimated* tokens, for the transcript embedded
# in a single summarization request. The middle region being compacted can be
# far larger than the model's real input limit, so it is chunked into pieces no
# larger than this budget and each piece is summarized independently before the
# partial summaries are consolidated. The value is kept well below typical model
# input maxima (e.g. ~1M tokens) so that even if the offline chars/4 heuristic
# under-counts the true token cost by several times, the real request still fits
# with headroom left for the instruction and the generated summary.
SUMMARY_INPUT_TOKEN_BUDGET = 200_000


# --------------------------------------------------------------------------- #
# Result/info types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompactionInfo:
    """Summary of a compaction event, surfaced as the compaction notice.

    Returned by :meth:`ContextManager.assemble` alongside the Context_Window
    when compaction occurred (``occurred=True``); ``None`` is returned instead
    when the window was already within the limit. The ``AgentLoop`` carries this
    up on its ``TurnResult`` and the Repl renders the "conversation context was
    compacted" notice (Req 14.7).

    Attributes:
        occurred: Whether compaction actually ran for this assembly.
        summary_message_count: Number of synthetic summary messages that
            replaced the compacted middle region.
        dropped_message_count: Number of retained-recent messages dropped
            because summarization alone could not reach the token limit.
    """

    occurred: bool
    summary_message_count: int = 0
    dropped_message_count: int = 0


@dataclass(frozen=True)
class CompactionResult:
    """The product of a compaction pass (populated by task 18.2).

    Attributes:
        messages: The compacted Context_Window as wire-shape message dicts.
        info: The :class:`CompactionInfo` describing what compaction did.
    """

    messages: list[dict] = field(default_factory=list)
    info: CompactionInfo = field(default_factory=lambda: CompactionInfo(False))


# --------------------------------------------------------------------------- #
# Built-in default system prompt loading
# --------------------------------------------------------------------------- #


def load_default_system_prompt() -> str:
    """Load the built-in default system prompt from package data.

    Reads ``forge/data/system_prompt.md`` via :mod:`importlib.resources` so the
    prompt resolves correctly whether Forge runs from a source checkout or an
    installed wheel/zip. (Requirements 15.2, 15.3)
    """

    return (
        resources.files(_DATA_PACKAGE)
        .joinpath(_SYSTEM_PROMPT_RESOURCE)
        .read_text(encoding="utf-8")
    )


# --------------------------------------------------------------------------- #
# Message serialization for token estimation
# --------------------------------------------------------------------------- #


def _serialize_message_text(message: dict) -> str:
    """Return the textual content of a wire-shape message dict for counting.

    A Context_Window message is a plain ``dict``. Its "serialized text" for the
    token heuristic is the concatenation of its textual parts:

    * the ``role`` marker,
    * the ``content``/``text`` body (when present), and
    * any tool-call text (tool name plus a JSON rendering of its arguments) and
      tool-result text (result content plus any error string).

    Non-text scalar values are stringified deterministically; argument and meta
    mappings are rendered with sorted keys so the same message always yields the
    same character count.
    """

    parts: list[str] = []

    role = message.get("role")
    if role is not None:
        parts.append(str(role))

    # Body text may be carried under "content" (wire shape) or "text" (session
    # Message shape); include whichever is present.
    for key in ("content", "text"):
        value = message.get(key)
        if value:
            parts.append(str(value))

    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            name = call.get("name")
            if name:
                parts.append(str(name))
            args = call.get("args")
            if args:
                parts.append(json.dumps(args, sort_keys=True, ensure_ascii=False))
        else:
            parts.append(str(call))

    result = message.get("tool_result")
    if isinstance(result, dict):
        content = result.get("content")
        if content:
            parts.append(str(content))
        error = result.get("error")
        if error:
            parts.append(str(error))

    return "".join(parts)


# --------------------------------------------------------------------------- #
# Session -> wire-shape conversion
# --------------------------------------------------------------------------- #


def _message_to_window_dict(message: Message) -> dict:
    """Convert a session :class:`Message` into a Context_Window dict."""

    window: dict = {"role": message.role, "content": message.text}
    if message.tool_calls:
        window["tool_calls"] = [
            {
                "id": c.id,
                "name": c.name,
                "args": c.args,
                "thought_signature": c.thought_signature,
            }
            for c in message.tool_calls
        ]
    if message.tool_result is not None:
        r = message.tool_result
        window["tool_result"] = {
            "call_id": r.call_id,
            "ok": r.ok,
            "content": r.content,
            "error": r.error,
            "meta": r.meta,
        }
    return window


# --------------------------------------------------------------------------- #
# ContextManager
# --------------------------------------------------------------------------- #


class ContextManager:
    """Assembles the Context_Window and (in task 18.2) performs compaction.

    Args:
        config: The resolved Forge configuration. Supplies ``steering_files``
            (ordered) and ``token_limit``/``retained_recent_messages`` used by
            compaction.
        summarizer: An optional summarization dependency used to summarize the
            middle region during compaction. Two shapes are supported (see
            :meth:`_summarize_middle` for the full contract):

            * a ``VertexClient``-like object exposing
              ``generate_stream(contents, tools)`` — Forge sends a bounded
              summarization request and concatenates the streamed text; or
            * a plain ``Callable[[list[dict]], str]`` that receives the middle
              messages (as wire-shape dicts) and returns the summary text.

            When ``None``, compaction falls back to a deterministic local
            placeholder summary so it still functions offline without a Model.
    """

    def __init__(
        self,
        config: Config,
        summarizer: Callable[[list[dict]], str] | object | None = None,
        providers: "list[ContextProvider] | None" = None,
        workspace_root: "Path | None" = None,
        project_memory_filenames: "tuple[str, ...]" = (),
    ) -> None:
        self.config = config
        self.summarizer = summarizer
        self.providers = list(providers) if providers else []
        self.workspace_root = workspace_root
        self.project_memory_filenames = tuple(project_memory_filenames)

    # ----- System-prompt assembly (Req 15.1-15.4, Property 22) ----- #

    def _system_segments(self) -> list[str]:
        """Return ordered system segments: default prompt then steering files.

        The built-in default prompt is always first. Each configured steering
        file's contents follow in the order listed in ``config.steering_files``.
        A configured steering file whose path does not exist produces a warning
        naming the missing file and is skipped, continuing with the rest
        (Req 15.4).
        """

        segments: list[str] = [load_default_system_prompt()]

        for raw_path in self.config.steering_files:
            path = Path(raw_path)
            if not path.exists():
                warnings.warn(
                    f"Steering file not found: {raw_path}; skipping it.",
                    stacklevel=2,
                )
                continue
            segments.append(path.read_text(encoding="utf-8"))

        # Feature E: auto-load a per-project instruction file from the workspace.
        segments.extend(self._project_memory_segments())
        return segments

    def build_system_prompt(self) -> str:
        """Return the combined system prompt text in configured order.

        The built-in default prompt comes first, followed by the contents of
        each existing configured steering file, joined by blank lines. When no
        steering files are configured, this is just the default prompt
        (Req 15.1, 15.2, 15.3).
        """

        return "\n\n".join(self._system_segments())

    def assemble_system_messages(self) -> list[dict]:
        """Return the system context as an ordered list of message dicts.

        The first message carries the built-in default prompt; each subsequent
        message carries one configured steering file's contents, in listed
        order. Missing steering files are warned about and skipped. This
        message-level ordering directly supports Property 22.
        """

        return [
            {"role": "system", "content": segment}
            for segment in self._system_segments()
        ]

    # ----- Token estimation (Req 14.1) ----- #

    def estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate the token count of a Context_Window deterministically.

        The estimate is ``sum(ceil(len(serialized_text) / 4))`` over every
        message plus a fixed :data:`PER_MESSAGE_OVERHEAD_TOKENS` per message.
        It is deterministic and offline (Req 14.1): the same input always
        yields the same count and no network call is made.
        """

        total = 0
        for message in messages:
            chars = len(_serialize_message_text(message))
            total += math.ceil(chars / CHARS_PER_TOKEN)
            total += PER_MESSAGE_OVERHEAD_TOKENS
        return total

    # ----- Context-window assembly ----- #

    def assemble(self, session: Session) -> tuple[list[dict], CompactionInfo | None]:
        """Assemble the Context_Window for a turn, compacting if over limit.

        Builds the system messages (built-in default prompt first, then steering
        files in order, then any project memory file) followed by the session's
        conversation messages. The estimated token count of that window is
        computed with the deterministic offline heuristic (Req 14.1). When it is
        at or below the effective token limit the window is returned as-is with
        ``None`` for the compaction info (Req 14.2, 14.8).

        Ephemeral provider segments (e.g. the plan reminder) are appended after
        compaction; their token cost is reserved so the final window still fits
        the configured limit. They are never written to ``session.messages``.

        When the estimate exceeds the limit, :meth:`compact` is run before the
        window is returned: the middle region is summarized and, if still over
        the limit, retained-recent messages are dropped oldest-first. The
        compacted window is returned together with a populated
        :class:`CompactionInfo` so the ``AgentLoop``/Repl can surface the
        "conversation context was compacted" notice (Req 14.7).
        """

        base: list[dict] = self.assemble_system_messages()
        base.extend(_message_to_window_dict(m) for m in session.messages)

        # Ephemeral, non-persisted provider segments (e.g. the plan reminder).
        ephemeral = self._provider_segments(session)
        reserve = self.estimate_tokens(ephemeral) if ephemeral else 0
        # Never let the reservation drive the effective limit below a small floor.
        effective_limit = max(self.config.token_limit - reserve, 0)

        if self.estimate_tokens(base) <= effective_limit:
            return base + ephemeral, None

        result = self.compact(session, limit=effective_limit)
        return result.messages + ephemeral, result.info

    def _provider_segments(self, session: Session) -> list[dict]:
        """Collect ephemeral segments from all providers, in registration order."""
        segments: list[dict] = []
        for provider in self.providers:
            try:
                produced = provider.segments(session)
            except Exception:  # noqa: BLE001 - a bad provider must never break a turn
                produced = []
            if produced:
                segments.extend(produced)
        return segments

    def _project_memory_segments(self) -> list[str]:
        """Return the contents of the first existing project memory file, or [].

        Looks in ``self.workspace_root`` (only) for the configured filenames in
        priority order (default: FORGE.md then AGENTS.md). Discovery is disabled
        when ``workspace_root`` is None or no filenames are configured, which
        reproduces the pre-feature behavior. A file that exists but cannot be
        read or is not valid UTF-8 is warned about and skipped.
        """
        if self.workspace_root is None or not self.project_memory_filenames:
            return []
        for filename in self.project_memory_filenames:
            candidate = self.workspace_root / filename
            if not candidate.is_file():
                continue
            try:
                return [candidate.read_text(encoding="utf-8")]
            except (OSError, UnicodeDecodeError):
                warnings.warn(
                    f"Project memory file could not be read: {candidate}; skipping it.",
                    stacklevel=2,
                )
                return []
        return []

    def compact(self, session: Session, *, limit: int | None = None) -> CompactionResult:
        """Compact the session's Context_Window to fit within the token limit.

        Partitions the conversation (NOT the system/steering prompt, which is
        always retained) into three regions and rebuilds the window
        (Req 14.3-14.6, 14.8, 14.9):

        * the **original task** — the first user message — retained verbatim as
          "the original task and instructions" (Req 14.3, 14.9);
        * a **middle region** of older messages, replaced by a single synthetic
          summary message produced via :meth:`_summarize_middle`, preserving
          decisions and outcomes (Req 14.3, 14.4); and
        * the **retained-recent** messages — the most recent
          ``config.retained_recent_messages`` — plus any **pending tool calls**
          whose Tool_Results are not yet present, all retained (Req 14.5, 14.6).

        If the post-summary estimate still exceeds the limit, retained-recent
        messages are dropped from oldest to newest — never the task, the system
        prompt, or pending tool calls — until the window fits or nothing more can
        be dropped (Req 14.8, 14.9). If even the smallest well-formed window
        (system prompt + task + pending tool calls + summary) exceeds the limit,
        a warning is emitted and that window is used anyway (Req 14.9).

        Returns a :class:`CompactionResult` carrying the compacted window and a
        :class:`CompactionInfo` recording the summary and dropped-message counts.
        """

        system_messages = self.assemble_system_messages()
        conv = [_message_to_window_dict(m) for m in session.messages]
        n = len(conv)
        effective_limit = self.config.token_limit if limit is None else limit

        # Degenerate case: no conversation messages. Nothing to partition; the
        # system prompt alone is the smallest window we can produce.
        if n == 0:
            window = list(system_messages)
            if self.estimate_tokens(window) > effective_limit:
                self._warn_cannot_reduce()
            return CompactionResult(
                messages=window, info=CompactionInfo(occurred=True)
            )

        # Region (a): the original task is the first user message (Req 14.3).
        task_index = next(
            (i for i, m in enumerate(conv) if m.get("role") == "user"), 0
        )

        # Region (c) anchors: pending tool calls (Req 14.6) and the most recent
        # configured count of messages (Req 14.5) are always retained.
        pending = self._pending_tool_call_indices(session.messages)
        retained_n = max(0, self.config.retained_recent_messages)
        recent = (
            set(range(max(0, n - retained_n), n)) if retained_n > 0 else set()
        )

        keep = {task_index} | pending | recent
        middle_indices = [i for i in range(n) if i not in keep]

        # Region (b): summarize the middle into ONE synthetic message.
        middle_msgs = [conv[i] for i in middle_indices]
        summary_count = 1 if middle_msgs else 0
        summary_message = None
        if middle_msgs:
            summary_text = self._summarize_middle(middle_msgs)
            summary_message = self._build_summary_message(
                len(middle_msgs), summary_text
            )

        # Rebuild the conversation in original order, collapsing the middle
        # region to the single summary message at the position it began. Each
        # surviving entry is tagged droppable when it is a retained-recent
        # message that is neither the task nor a pending tool call.
        middle_set = set(middle_indices)
        protected = {task_index} | pending
        entries: list[dict] = []
        inserted_summary = False
        for i in range(n):
            if i in middle_set:
                if summary_message is not None and not inserted_summary:
                    entries.append({"msg": summary_message, "droppable": False})
                    inserted_summary = True
                continue
            droppable = (i in recent) and (i not in protected)
            entries.append({"msg": conv[i], "droppable": droppable})

        def build_window(items: list[dict]) -> list[dict]:
            w = list(system_messages)
            w.extend(item["msg"] for item in items)
            return w

        window = build_window(entries)

        # Drop retained-recent messages oldest-first until at/below the limit
        # or nothing droppable remains (Req 14.8, 14.9).
        dropped = 0
        while self.estimate_tokens(window) > effective_limit:
            idx = next(
                (k for k, item in enumerate(entries) if item["droppable"]), None
            )
            if idx is None:
                break
            entries.pop(idx)
            dropped += 1
            window = build_window(entries)

        # Smallest well-formed window still exceeds the limit: warn and proceed
        # with what we have (Req 14.9).
        if self.estimate_tokens(window) > effective_limit:
            self._warn_cannot_reduce()

        info = CompactionInfo(
            occurred=True,
            summary_message_count=summary_count,
            dropped_message_count=dropped,
        )
        return CompactionResult(messages=window, info=info)

    # ----- Compaction helpers ----- #

    @staticmethod
    def _warn_cannot_reduce() -> None:
        """Emit the Req 14.9 warning that the limit could not be reached."""

        warnings.warn(
            "Compaction could not reduce the estimated context to at or below "
            "the token limit; proceeding with the smallest well-formed window.",
            stacklevel=2,
        )

    @staticmethod
    def _pending_tool_call_indices(messages: list[Message]) -> set[int]:
        """Return indices of messages carrying unresolved Tool_Calls (Req 14.6).

        A Tool_Call is *pending* when no later tool message carries a
        Tool_Result with the matching ``call_id``. Any message that emitted at
        least one still-pending Tool_Call is retained so the in-flight tool
        exchange is never split by compaction.
        """

        resolved: set[str] = set()
        for m in messages:
            if m.tool_result is not None:
                resolved.add(m.tool_result.call_id)

        pending: set[int] = set()
        for i, m in enumerate(messages):
            if m.tool_calls and any(c.id not in resolved for c in m.tool_calls):
                pending.add(i)
        return pending

    def _build_summary_message(self, count: int, summary_text: str) -> dict:
        """Wrap the middle-region summary as a single synthetic window message."""

        body = (summary_text or "").strip()
        if not body:
            body = f"{count} earlier message(s) were summarized to conserve context."
        content = f"{SUMMARY_MESSAGE_PREFIX} ({count} message(s) summarized)\n{body}"
        return {"role": "user", "content": content}

    def _summarize_middle(self, middle: list[dict]) -> str:
        """Summarize the middle region using the injected summarizer.

        Contract for ``self.summarizer``:

        * a ``VertexClient``-like object (detected by a ``generate_stream``
          attribute) is wrapped: a bounded summarization request is streamed and
          its text fragments are concatenated;
        * any other callable is treated as ``Callable[[list[dict]], str]`` and
          invoked directly with the middle messages; and
        * ``None`` (or an unusable value) falls back to a deterministic local
          placeholder summary so compaction still works fully offline.
        """

        if not middle:
            return ""

        summarizer = self.summarizer
        if summarizer is None:
            return self._local_summary(middle)

        # Summarization issues one or more live model requests (several, for a
        # chunked large middle region). Any failure here — a rate limit, auth,
        # timeout, or other VertexError, or a raising callable summarizer — must
        # NOT crash the turn: compaction is invoked from inside AgentLoop.run_turn
        # outside its VertexError handling, so an escaping exception would abort
        # the whole turn and lose the graceful-degradation contract. Fall back to
        # the deterministic local placeholder summary instead, emitting a warning
        # so the degradation is observable. (Kept decoupled from forge.vertex by
        # catching broadly rather than importing VertexError.)
        try:
            if hasattr(summarizer, "generate_stream"):
                return self._summarize_via_vertex(summarizer, middle)
            if callable(summarizer):
                return summarizer(middle)
        except Exception:  # noqa: BLE001 - summarization must never crash a turn
            warnings.warn(
                "Context summarization failed; falling back to a local "
                "placeholder summary so the turn can proceed.",
                stacklevel=2,
            )
            return self._local_summary(middle)
        return self._local_summary(middle)

    def _summarize_via_vertex(self, client: object, middle: list[dict]) -> str:
        """Summarize via a ``VertexClient``-like streaming dependency.

        The middle region can be far larger than the model's input limit, so it
        is split into chunks each no larger than
        :data:`SUMMARY_INPUT_TOKEN_BUDGET` estimated tokens. When it fits in a
        single chunk a lone summarization request is sent. Otherwise every chunk
        is summarized independently and the partial summaries are consolidated
        into one final summary — recursively, if the concatenated partials are
        themselves still too large to summarize in one request. This guarantees
        no individual request exceeds the model's input limit (the failure mode
        that previously crashed compaction on very large contexts).
        """

        chunks = self._chunk_for_summary(middle)
        if len(chunks) <= 1:
            return self._summarize_chunk_via_vertex(client, middle)

        partials: list[str] = []
        total = len(chunks)
        for i, chunk in enumerate(chunks, start=1):
            text = self._summarize_chunk_via_vertex(client, chunk)
            partials.append(f"[Part {i} of {total}]\n{text}")

        # Consolidate the partial summaries into one. Represent each partial as a
        # message so the same chunking/estimation machinery applies; recurse if
        # the partials are still collectively too large for a single request.
        partial_msgs = [{"role": "user", "content": p} for p in partials]
        if self.estimate_tokens(partial_msgs) > SUMMARY_INPUT_TOKEN_BUDGET:
            return self._summarize_via_vertex(client, partial_msgs)
        return self._summarize_chunk_via_vertex(client, partial_msgs)

    def _chunk_for_summary(self, middle: list[dict]) -> list[list[dict]]:
        """Split ``middle`` into chunks within :data:`SUMMARY_INPUT_TOKEN_BUDGET`.

        Greedily packs messages in order so each chunk's estimated token count
        stays at or below the budget. A single message that on its own exceeds
        the budget becomes its own chunk (its oversized transcript is truncated
        by :meth:`_summarize_chunk_via_vertex` before the request is sent).
        """

        chunks: list[list[dict]] = []
        current: list[dict] = []
        current_tokens = 0
        for msg in middle:
            msg_tokens = self.estimate_tokens([msg])
            if current and current_tokens + msg_tokens > SUMMARY_INPUT_TOKEN_BUDGET:
                chunks.append(current)
                current = []
                current_tokens = 0
            current.append(msg)
            current_tokens += msg_tokens
        if current:
            chunks.append(current)
        return chunks

    def _summarize_chunk_via_vertex(
        self, client: object, messages: list[dict]
    ) -> str:
        """Send one bounded summarization request for ``messages`` and return text.

        Sends a single request (no tools) whose user message embeds a transcript
        of ``messages`` and concatenates the streamed text fragments
        (``TextDelta``-shaped events exposing ``.text``). As a final safeguard
        the transcript is truncated to :data:`SUMMARY_INPUT_TOKEN_BUDGET`
        estimated tokens so that even a single pathologically large message
        cannot push the request past the model's input limit.
        """

        transcript = "\n".join(_serialize_message_text(m) for m in messages)

        max_chars = SUMMARY_INPUT_TOKEN_BUDGET * CHARS_PER_TOKEN
        if len(transcript) > max_chars:
            transcript = (
                transcript[:max_chars]
                + "\n[transcript truncated to fit the summarization request]"
            )

        prompt = f"{SUMMARY_INSTRUCTION}\n\n{transcript}"
        contents = [{"role": "user", "content": prompt}]

        parts: list[str] = []
        for event in client.generate_stream(contents, []):  # type: ignore[attr-defined]
            text = getattr(event, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        return "".join(parts).strip()

    @staticmethod
    def _local_summary(middle: list[dict]) -> str:
        """Deterministic offline fallback when no Model summarizer is available."""

        return (
            f"{len(middle)} earlier message(s) were summarized locally to "
            "conserve context (no summarization model was available)."
        )
