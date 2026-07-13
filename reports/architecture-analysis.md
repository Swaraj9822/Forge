# Architecture Analysis

## Architectural Style

- **Pattern:** Modular monolith — a single installable Python package (`forge`)
  with a console-script entry point. Appropriate for a single-user CLI.
- **Design:** Layered + ports/adapters leanings. Tools implement a `Protocol`
  (`forge/tools/base.py::Tool`); MCP tools are adapted to the same shape
  (`forge/mcp_client.py::McpToolAdapter`). Pure decision logic is separated from
  I/O in the verification phase.
- **Fit:** The architecture matches the problem and (apparent) team size of one.
  No over-engineering; no microservice sprawl. This is the right shape.

## Layering (presentation → application → domain → infrastructure)

| Layer | Where | Boundary quality |
|-------|-------|------------------|
| CLI / presentation | `__main__.py`, `repl.py` | Clean. `__main__` is a thin dispatch; `repl.py` owns terminal I/O and rendering. |
| Composition root | `app.py` | Single place all wiring happens (`bootstrap`). Excellent. |
| Application/orchestration | `agent.py`, `verification.py` | `AgentLoop.run_turn` and `VerificationCoordinator.run` orchestrate; collaborators injected. |
| Domain/pure | `context.py` (token estimate, compaction), `usage.py`, pure helpers in `verification.py`, `config.py` resolution | Pure and testable. |
| Infrastructure | `vertex.py` (network), `tools/*` (fs/shell/git/search), `mcp_client.py`, `session.py` (disk) | Isolated behind typed surfaces. |

No layer violations of note. Rendering is decoupled from the agent loop via the
`Renderer`/`VerificationRenderer` `Protocol`s, so the loop runs headless in tests.

## Dependency Direction & Cycles

- A real import cycle (`repl` ↔ `verification`, both importing `agent`) is
  deliberately avoided: `repl.py` imports verification types under
  `if TYPE_CHECKING:` only (annotation-only coupling). This is documented inline.
- `tools/base.py` imports `session.ToolCall`; `agent.py`, `context.py`,
  `vertex.py` all share that one `ToolCall` type — no duplicate models.
- `ToolContext.config` is intentionally typed `Any` to avoid a hard cycle with
  `forge.config`. Reasonable trade-off, documented.

**No circular dependencies found.**

## Module Map (selected)

```
Module: app.py
  Responsibility: load config, validate startup, wire the whole graph
  Depends on: config, session, interrupt, vertex, tools.*, mcp_client, context, agent, repl, verification
  Depended on by: __main__
  Coupling: High (by design — it is the composition root)  Cohesion: High

Module: agent.py
  Responsibility: drive a single turn (assemble→stream→execute tools→repeat)
  Depends on: context, interrupt, session, tools.base, usage, vertex
  Coupling: Medium  Cohesion: High

Module: vertex.py
  Responsibility: Gemini/Vertex streaming client + SDK translation + retry
  Coupling: Low (only config, interrupt, session, tools.base)  Cohesion: Medium
  Flags: god-module drift (872 lines, ~7 distinct responsibilities)

Module: verification.py
  Responsibility: opt-in post-turn verify + bounded self-correction
  Coupling: Medium  Cohesion: High
  Flags: exemplary pure/impure split

Module: tools/base.py
  Responsibility: Tool protocol + ToolExecutor (exposure = registry ∩ enabled)
  Coupling: Low  Cohesion: High
```

## Architecture Drift Analysis

The codebase has no git history (see `infrastructure-analysis.md`), so drift
cannot be measured against prior intent. Comparing the design docs in
`.kiro/specs/` against the implementation: the implementation tracks the design
closely (requirement IDs are cited throughout the code). **No meaningful drift
observed.**

## Dangerous Structural Patterns

- **God module:** `forge/vertex.py` only. Recommend extracting the SDK
  wire-shape translation (`_to_sdk_contents`, `_to_sdk_tools`, `_parse_chunk`,
  `_function_call_to_tool_call`) and the retry-hint helpers into a
  `vertex_translate.py` so `VertexClient` keeps just lifecycle + streaming.
- **Circular deps:** none.
- **Hidden global state:** none significant. State is threaded explicitly via
  `ToolContext.state` (e.g. the planning tool's todo list) rather than module
  globals. The guarded SDK imports (`_genai`, etc.) are module-level but are
  read-only capability flags.
- **Anemic domain model:** N/A — the domain (sessions, messages, todos) is data
  plus thin behavior, which is appropriate here.

## Verdict

Architecture is a clear strength. Score 90/100; the only structural
recommendation is to split `vertex.py`.
