# Forge — Phase 5 Implementation Plan (Strategic)

**Audience:** a coding agent implementing this end-to-end.
**Prerequisite:** Phases 1–4 merged.
**Scope:** Phase 5 only — the two largest, highest-ceiling bets:

- **M — Multi-provider support.** Abstract the model client behind a `Provider`
  interface so Forge can talk to Anthropic / OpenAI / local (OpenAI-compatible)
  models, not just Vertex/Gemini.
- **N — Subagents.** Let the main agent delegate a scoped task to a fresh
  sub-agent with its own context window and a restricted toolset, returning a
  single result.

These are sequenced M-before-N because a subagent should be able to run on a
cheaper/faster provider, which M makes possible. Each is independently
shippable; **M is the bigger, riskier change — do it first and stabilize it
before starting N.**

---

## 0. Ground rules

Same as `phase1.md` §0, plus:
- New optional provider SDKs (`anthropic`, `openai`) go in a `pyproject.toml`
  extra (`[project.optional-dependencies] providers = [...]`) and are imported
  **guardedly** (like `google-genai` in `vertex.py`) so Forge imports without
  them; a provider only hard-requires its SDK at first use.
- The rest of Forge already speaks the provider-agnostic `StreamEvent` union
  (`TextDelta`/`ToolCall`/`UsageReport`/`Done`) — **do not change that union or
  `AgentLoop`.** The whole point is to keep the blast radius inside the client
  layer + wiring.
- Test with `pytest -q --basetemp=C:\forge_tmp` (outside repo), then remove it.

**Reality check (from the review):** the `StreamEvent` union is the easy 20%.
The real work is per-provider translation of (a) tools/function-call schemas,
(b) message/content shapes, and (c) error taxonomies. Budget accordingly.

---

## 1. Architecture overview

| File | Change |
|------|--------|
| `forge/providers/__init__.py` (new) | `Provider` protocol; `build_provider(config)` factory; shared `StreamEvent` re-exports. |
| `forge/providers/base.py` (new) | `Provider` protocol + shared typed errors (move `VertexError`-family up to a provider-agnostic base). |
| `forge/providers/vertex.py` | The existing `forge/vertex.py` refactored into a `VertexProvider` implementing `Provider` (behavior preserved). |
| `forge/providers/anthropic.py` (new) | `AnthropicProvider`. |
| `forge/providers/openai.py` (new) | `OpenAIProvider` (also covers local OpenAI-compatible endpoints via `base_url`). |
| `forge/config.py` | `[provider]` table: `type` (vertex/anthropic/openai), model, and per-provider auth/endpoint fields. |
| `forge/app.py` | Use `build_provider(config)` instead of constructing `VertexClient` directly. |
| `forge/subagent.py` (new) | `SubAgentRunner` + the `delegate` tool. |
| `forge/tools/subagent.py` (new) | `DelegateTool` (`delegate`). |
| `tests/…` | Provider translation tests (fakes, offline), factory tests, subagent tests. |

---

## 2. Feature M — Multi-provider

### 2.1 `Provider` protocol (`forge/providers/base.py`)
Extract the surface `AgentLoop` and `ContextManager` already depend on:
```python
class Provider(Protocol):
    def generate_stream(self, contents: list[dict], tools: list[ToolSpec]
                        ) -> Iterator[StreamEvent]:
        ...
```
- Move the typed error hierarchy (`ProviderError` base, `CredentialsError`,
  `AuthorizationError`, `RateLimitError`, `RequestTimeoutError`,
  `ConfigMissingError`) here so every provider raises the **same** types the
  `AgentLoop` already catches. Keep `VertexError` as an alias of `ProviderError`
  for backward-compat imports, or update the (few) references.
- `StreamEvent`/`TextDelta`/`ToolCall`/`UsageReport`/`Done` stay defined once and
  are re-exported; providers import them from here.

### 2.2 Refactor Vertex into a provider (`forge/providers/vertex.py`)
- Move `forge/vertex.py` content into `forge/providers/vertex.py` as
  `VertexProvider(Provider)`. **Preserve behavior exactly** — the retry/backoff,
  interrupt polling, timeout guard, `_to_sdk_contents`/`_to_sdk_tools`,
  `thought_signature` handling, and error translation are all Gemini-correct and
  must not regress.
- Keep a shim at `forge/vertex.py` re-exporting the moved names so existing
  imports/tests (`tests/test_vertex*.py`) keep working, or update those imports.
  Prefer the shim to minimize churn; document it.
- The Gemini-specific translation stays **inside** this module.

### 2.3 New providers
Each implements `generate_stream(contents, tools)` and does its own translation:

**`AnthropicProvider`** (`forge/providers/anthropic.py`)
- Translate Forge wire-shape `contents` → Anthropic Messages API: system prompt
  → top-level `system`; `user`/`model`→`assistant`; tool calls → `tool_use`
  blocks; tool results → `tool_use`/`tool_result` content blocks (Anthropic
  requires tool_result to reference the `tool_use` id).
- Tools: `ToolSpec` → Anthropic `tools` (name/description/`input_schema`).
- Stream: map Anthropic SSE events → `TextDelta` (text deltas), `ToolCall`
  (tool_use blocks; accumulate partial JSON args then emit), `UsageReport`
  (input/output tokens from the final message), `Done`.
- Errors: map 401/403→`AuthorizationError`, 429→`RateLimitError` (honor
  `retry-after`), timeouts→`RequestTimeoutError`, missing key→`CredentialsError`.

**`OpenAIProvider`** (`forge/providers/openai.py`)
- Chat Completions (or Responses) API with `tools`/`tool_calls`. `model`→
  `assistant`; tool results → `role:"tool"` messages keyed by `tool_call_id`;
  system → a `system` message.
- Supports a configurable `base_url` so **local OpenAI-compatible servers**
  (Ollama, vLLM, LM Studio) work with the same provider.
- Stream deltas → `TextDelta`; assembled `tool_calls` → `ToolCall`; `usage` →
  `UsageReport`; `Done`.
- Same error mapping as above.

Each provider polls the shared `InterruptController` between chunks (reuse the
Vertex pattern) and applies `config.request_timeout_s` and rate-limit retry
using the existing backoff helpers (lift those into `providers/base.py` so all
providers share them).

### 2.4 Factory (`forge/providers/__init__.py`)
```python
def build_provider(config: Config, interrupt: InterruptController) -> Provider:
    t = config.provider_type
    if t == "vertex":    return VertexProvider(config, interrupt)
    if t == "anthropic": return AnthropicProvider(config, interrupt)
    if t == "openai":    return OpenAIProvider(config, interrupt)
    raise ConfigError(..., detail=f"unknown provider type {t!r}")
```

### 2.5 Config (`forge/config.py`)
```toml
[provider]
type = "vertex"                # vertex | anthropic | openai
model = "gemini-3.1-pro-preview"
# vertex: project, region (existing top-level keys still honored)
# anthropic: api_key_env = "ANTHROPIC_API_KEY"
# openai:    api_key_env = "OPENAI_API_KEY", base_url = "https://api.openai.com/v1"
```
- Add `provider_type: str = "vertex"` and per-provider fields
  (`api_key_env`, `base_url`) to `Config`. **Default `type="vertex"` and keep the
  existing `model`/`project`/`region` handling so current configs work unchanged.**
- Validate `type` against the known set (`ConfigError` naming the bad value).
- API keys are read from the named **environment variable** at provider init
  (never stored in the config file); a missing key → `CredentialsError` with a
  clear message. Do not log key values.
- `validate_required_config`/`check_adc` in `app.py` become provider-aware: ADC
  is only required for `vertex`; anthropic/openai require their env key instead.
  Factor the startup validation per provider.

### 2.6 Wiring (`forge/app.py`)
- Replace `vertex_client = VertexClient(config, interrupt)` with
  `provider = build_provider(config, interrupt)` and pass `provider` everywhere
  `vertex_client` was used (`AgentLoop`, `ContextManager` summarizer). Rename the
  `App.vertex_client` field to `App.provider` (update references + tests).
- Provider-specific startup checks replace the unconditional ADC check.

### 2.7 Tests
- `tests/test_provider_factory.py`: factory returns the right type; unknown type
  → `ConfigError`.
- `tests/test_provider_anthropic.py` / `test_provider_openai.py`: **offline**,
  inject a fake SDK/transport; assert wire-shape→provider translation (system
  extraction, tool_use/tool_result correlation, role mapping) and
  stream→`StreamEvent` mapping; assert error mapping (401/429/timeout →
  the shared typed errors). Mirror the existing `tests/test_vertex_contents.py`
  style.
- Existing `tests/test_vertex*.py` still pass (via the shim or updated imports).
- `tests/test_startup_validation.py`: extended for per-provider validation (ADC
  only for vertex; env-key for others).

---

## 3. Feature N — Subagents

### 3.1 `SubAgentRunner` (`forge/subagent.py`)
A subagent is a **fresh** `AgentLoop` run over an isolated session with a
restricted toolset and its own context window, returning a single text result.
```python
class SubAgentRunner:
    def __init__(self, provider, config, interrupt, *, tool_registry,
                 allowed_tools: set[str], max_turns: int = 1): ...
    def run(self, task: str, *, workspace_root: Path) -> SubAgentResult:
        """Run a scoped task to completion in an isolated session and return its
        final text + usage. Uses a fresh in-memory Session (never persisted to
        the user's session store), its own ContextManager, and a ToolExecutor
        limited to `allowed_tools`."""
```
- **Isolation:** new in-memory `Session` (not saved to `SessionStore`); its own
  `ContextManager` (a subagent-specific system prompt: "You are a sub-agent…
  return a concise result"); a `ToolExecutor` whose `enabled` is the intersection
  of the parent's tools and `allowed_tools`.
- **Tool restriction:** by default a subagent gets **read-only** tools
  (`read`, `search`, `repo_index`, `search_memory`) — delegation is for
  exploration/analysis. Writing/shell is opt-in per delegation and still subject
  to the Phase 2 policy/approver (pass the same approver through).
- **Bounded:** cap total turns and total tokens; a subagent cannot spawn further
  subagents (no `delegate` in its toolset) — prevents unbounded recursion.
- **Interrupt:** share the parent `InterruptController` so Ctrl-C stops a
  subagent too; bracket with `begin_turn`/`end_turn` as the main loop does.
- **Usage:** aggregate the subagent's `UsageSummary` and surface it so the parent
  turn's cost includes delegated work (reuse `aggregate_usage` from
  `verification.py` or a shared helper).

### 3.2 `DelegateTool` (`forge/tools/subagent.py`)
- `name="delegate"`, `read_only=True` (it does not itself mutate the workspace;
  any writes happen only if the subagent is explicitly granted write tools, which
  the policy still gates). Params: `task` (required), optional `tools` (subset of
  allowed tool names), optional `max_turns`.
- `run` builds a `SubAgentRunner` from the shared provider/config/registry and
  returns the subagent's final text as the tool result content, with usage in
  `meta`.
- Inject the runner factory via `ToolContext` (new optional field, like `memory`
  in Phase 3), or construct inside the tool from context-carried collaborators.

### 3.3 Config (`forge/config.py`)
```toml
[subagents]
enabled = false
default_tools = ["read", "search", "repo_index", "search_memory"]
max_turns = 4
```
Add `delegate` to `RECOGNIZED_TOOLS`/`DEFAULT_ENABLED_TOOLS` **only when
enabled**; default OFF. Update exposure/config tests.

### 3.4 Tests
`tests/test_subagent.py` (offline, fake provider): a delegated task runs in an
isolated session, cannot access disabled tools, cannot recurse (`delegate` absent
from its toolset), respects `max_turns`, and returns aggregated usage; interrupt
propagates. `tests/test_delegate_tool.py`: validation + result shaping.

---

## 4. Full test plan & DoD

Run `python -m pytest -q --basetemp=C:\forge_tmp`, then remove the temp dir.

**Definition of Done**
- [ ] `Provider` protocol + shared typed errors in `providers/base.py`; backoff/
      timeout/interrupt helpers shared across providers.
- [ ] Vertex refactored to `VertexProvider` with behavior preserved (existing
      vertex tests pass via shim or updated imports).
- [ ] `AnthropicProvider` and `OpenAIProvider` (incl. local `base_url`) with
      full content/tool/error translation; offline translation tests.
- [ ] `build_provider` factory; `[provider]` config; per-provider startup
      validation (ADC only for vertex); keys from env, never logged.
- [ ] `app.py` uses the provider abstraction end-to-end; `App.provider` replaces
      `App.vertex_client`; default config still works with no changes.
- [ ] `SubAgentRunner` (isolated session, restricted read-only tools by default,
      bounded turns/tokens, no recursion, shared interrupt, aggregated usage).
- [ ] `delegate` tool gated by config (default OFF); recognized-tool set + tests
      updated.
- [ ] Full suite green; temp base removed.

## 5. Out of scope / future
- Provider-specific niceties (Anthropic prompt caching, OpenAI reasoning-effort
  controls) beyond basic streaming/tools/usage — add later once the abstraction
  is stable.
- Parallel subagents / a subagent orchestration graph — single delegated task
  per call here.
- Per-model tokenizer-accurate counting — keep the existing offline heuristic.

---

## 6. Sequencing & risk

1. **M first, in isolation.** Land the `Provider` refactor with Vertex behavior
   preserved and the two new providers behind the config switch. Stabilize
   (offline translation tests + a manual smoke against each real API) before
   touching subagents.
2. **N second.** Subagents build on M (a subagent may target a cheaper provider)
   and reuse the Phase 1 `ContextManager`, Phase 2 policy/approver, and the
   Phase 3 read-only tools — so keep those seams intact.

**Biggest risk:** the per-provider content/tool translation (esp. tool-call /
tool-result correlation and the function-call/response part-count rules). Cover
each provider's translation with focused offline tests **before** wiring it into
`bootstrap`, exactly as `tests/test_vertex_contents.py` does for Gemini today.
```
