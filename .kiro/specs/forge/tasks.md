# Implementation Plan: Forge

## Overview

This plan builds Forge bottom-up: foundational helpers (path scoping, config, session storage) first, then the tool protocol and each built-in tool, then the model/IO layer (Vertex client, usage, context/compaction), then the orchestration layer (agent loop, REPL), then MCP, app bootstrap, and CLI dispatch. Each step is incremental and ends by wiring new code into the components built before it.

Implementation language is Python (the design specifies a `forge` package with concrete Python signatures). Property-based tests use Hypothesis configured for at least 100 iterations (`@settings(max_examples=100)`); each property test is tagged with a `# Feature: forge, Property {n}: ...` comment. Test sub-tasks are marked optional with `*`.

## Tasks

- [x] 1. Set up project scaffolding and dependencies
  - [x] 1.1 Create the `forge` package layout and dependency manifest
    - Create `pyproject.toml` declaring runtime deps (`google-genai`, `mcp`, `tomli-w`, `prompt_toolkit`) and dev deps (`hypothesis`, `pytest`), with a `forge` console-script entry point pointing at `forge.__main__`
    - Create the package skeleton: `forge/__init__.py`, `forge/tools/__init__.py`, `forge/data/` directory, and `tests/__init__.py`
    - Configure package data (package-data / include settings in `pyproject.toml`) so `forge/data/` (including `system_prompt.md`) is bundled in the package and loadable at runtime via `importlib.resources`
    - Configure pytest discovery for the `tests/` directory
    - _Requirements: 11.1_

- [x] 2. Implement the workspace path-scoping helper
  - [x] 2.1 Implement `resolve_in_workspace` in `forge/tools/paths.py`
    - Canonicalize a candidate path against the workspace root (cwd) and raise/return an out-of-scope signal for any path that escapes the root
    - Expose a helper the read/write/edit/search tools can share
    - _Requirements: 11.8, 5.4, 6.6_
  - [x] 2.2 Write property test for workspace path-scoping
    - **Property 6: Workspace path-scoping invariant**
    - **Validates: Requirements 5.4, 6.6**
    - Use a `tmp_path` workspace and generate both interior and escaping (`../`, absolute, symlink-style) paths
    - _Requirements: 5.4, 6.6_

- [x] 3. Implement configuration management
  - [x] 3.1 Implement `Config` dataclasses and `ConfigManager` in `forge/config.py`
    - Define `ModelPricing`, `McpServerConfig`, and `Config` frozen dataclasses with all documented defaults
    - Implement `config_path()` and `sessions_dir()` OS-conventional resolution (Windows `%APPDATA%`, Unix/macOS `$XDG_CONFIG_HOME`/`$XDG_DATA_HOME` with `~/.config` and `~/.local/share` fallbacks)
    - Implement `load()` to read TOML via `tomllib`, merge documented defaults for absent values, drop+warn on unrecognized enabled tools, raise `ConfigError` with file path and line/column on syntax error, and apply all defaults when the file is absent
    - Implement `write_default(path)` (used by `forge init`) emitting the documented TOML structure with required `project`/`region` placeholders, using `tomli-w`
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.9, 12.1_
  - [x] 3.2 Write property test for config defaults merge
    - **Property 18: Config defaults merge**
    - **Validates: Requirements 11.4**
    - Generate partial config dicts (arbitrary omitted subsets) and assert omitted settings equal defaults and present settings are preserved
    - _Requirements: 11.4_
  - [x] 3.3 Write property test for init config round-trip
    - **Property 19: Init config round-trip**
    - **Validates: Requirements 12.1**
    - Assert `write_default` output parses as TOML and loads back to the documented defaults with `project`/`region` placeholders present
    - _Requirements: 12.1_
  - [x] 3.4 Write unit tests for config edge cases
    - Cover syntax-error reporting (file + line/column), unknown-tool warning, OS path resolution, and `init`-already-exists leaving the file unchanged
    - _Requirements: 11.6, 11.7, 11.2, 12.2_

- [x] 4. Implement session persistence
  - [x] 4.1 Implement session data models and JSON serialization in `forge/session.py`
    - Define `ToolCall`, `ToolResultRecord`, `Message`, `TodoItem`, `Usage`, `Session`, and `SessionMeta` dataclasses
    - Implement lossless to-JSON / from-JSON serialization for `Session`
    - _Requirements: 13.1_
  - [x] 4.2 Implement `SessionStore` (save/load/list/new) in `forge/session.py`
    - `new()` mints a UUIDv4 id and ISO-8601 UTC timestamps; `save()` writes atomically (temp file + `os.replace`) guarded by an in-process per-session lock so concurrent writes serialize; `load()` raises `CorruptSessionError` on parse failure leaving on-disk bytes untouched; `list()` returns id + creation timestamp per stored session
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.7_
  - [x] 4.3 Write property test for session serialization round-trip
    - **Property 20: Session serialization round-trip**
    - **Validates: Requirements 13.1, 13.4, 13.5**
    - Generate arbitrary Sessions (messages, tool calls, results, todos, usage), assert serialize→deserialize equality, and assert a saved session appears in `list()` with its id and creation timestamp
    - _Requirements: 13.1, 13.4, 13.5_
  - [x] 4.4 Write unit tests for session store edge cases
    - Cover atomic and sequential-write behavior, unknown-id load error, and corrupt-file handling (bytes left untouched, `CorruptSessionError` raised)
    - _Requirements: 13.2, 13.3, 13.6, 13.7_

- [x] 5. Implement the Tool protocol and ToolExecutor
  - [x] 5.1 Implement `Tool` protocol, `ToolResult`, `ToolSpec`, and `ToolExecutor` in `forge/tools/base.py`
    - Define the `Tool` protocol (`name`, `description`, `parameters`, `validate`, `run`), `ToolResult`, `ToolContext`, and `ToolSpec`
    - Implement `ToolExecutor` with a name→Tool registry, `specs()` returning only enabled (built-in + accepted MCP) tools, and `execute()` resolving by name (unknown/disabled → "unavailable"), running `validate` (failure → validation-error result with no side effects), checking the interrupt before/after, then running the tool
    - _Requirements: 4.1, 4.6, 4.7, 11.8, 16.2_
  - [x] 5.2 Write property test for tool exposure and availability
    - **Property 4: Only enabled, recognized tools are exposed and runnable**
    - **Validates: Requirements 4.6, 11.7, 11.8**
    - Generate enabled-name sets; assert `specs()` equals recognized tools ∩ enabled, and invoking a non-exposed name yields an "unavailable" result with no side effects
    - _Requirements: 4.6, 11.7, 11.8_
  - [x] 5.3 Write property test for validation safety
    - **Property 5: Invalid arguments never cause side effects**
    - **Validates: Requirements 4.7**
    - Generate tool calls with invalid args; assert a validation-error result and unchanged workspace/session state
    - _Requirements: 4.7_
  - [x] 5.4 Write unit test for tool-executor interrupt handling
    - Assert that a tripped interrupt checked before or after a tool run yields an "interrupted" `Tool_Result` via the executor (no tool side effects when tripped before)
    - _Requirements: 4.3, 4.4_

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement the read tool
  - [x] 7.1 Implement the read tool in `forge/tools/fs.py`
    - UTF-8 decode with binary detection, optional inclusive line range bounded to the file, not-found / out-of-scope (via `resolve_in_workspace`) / invalid-range / binary handling, and truncation at 2,000 lines or 1 MB with a truncated flag in `meta`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_
  - [x] 7.2 Write property test for read line-range slicing
    - **Property 7: Read line-range slice**
    - **Validates: Requirements 5.2**
    - _Requirements: 5.2_
  - [x] 7.3 Write property test for invalid line ranges
    - **Property 8: Invalid line range rejected**
    - **Validates: Requirements 5.5**
    - _Requirements: 5.5_
  - [x] 7.4 Write property test for read truncation cap
    - **Property 9: Read truncation cap**
    - **Validates: Requirements 5.7**
    - _Requirements: 5.7_
  - [x] 7.5 Write unit tests for read error paths
    - Cover not-found, out-of-scope, and binary-file results
    - _Requirements: 5.3, 5.4, 5.6_

- [x] 8. Implement the write tool
  - [x] 8.1 Implement the write tool in `forge/tools/fs.py`
    - Replace existing content, create missing parent directories, report bytes written, enforce out-of-scope rejection, and return a filesystem-error result leaving the filesystem unchanged on failure
    - Perform the write through a temp file + atomic replace (`os.replace`) where applicable, per the design's error-handling section, so a partial/failed write leaves the filesystem unchanged
    - _Requirements: 6.1, 6.2, 6.6, 6.8_
  - [x] 8.2 Write property test for write round-trip
    - **Property 10: Write round-trip, byte count, and parent creation**
    - **Validates: Requirements 6.1, 6.2, 5.1**
    - _Requirements: 6.1, 6.2, 5.1_
  - [x] 8.3 Write unit tests for write error paths
    - Cover out-of-scope rejection and filesystem-error handling leaving the filesystem unchanged
    - _Requirements: 6.6, 6.8_

- [x] 9. Implement the edit tool
  - [x] 9.1 Implement the edit tool in `forge/tools/fs.py`
    - Require the target string to occur exactly once (zero → not found, more than one → ambiguous), with not-found-file, out-of-scope, and filesystem-error handling that leaves the file byte-for-byte unchanged on any non-unique or error case
    - _Requirements: 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_
  - [x] 9.2 Write property test for edit uniqueness
    - **Property 11: Edit uniqueness invariant**
    - **Validates: Requirements 6.3, 6.4, 6.5**
    - _Requirements: 6.3, 6.4, 6.5_
  - [x] 9.3 Write unit tests for edit error paths
    - Cover not-found-file, out-of-scope, and filesystem-error handling
    - _Requirements: 6.6, 6.7, 6.8_

- [x] 10. Implement the shell tool
  - [x] 10.1 Implement the shell tool in `forge/tools/shell.py`
    - Run via the platform default shell (`cmd.exe /C` on Windows, `/bin/sh -c` on POSIX) with `cwd` at the workspace root, spawn in a new process group, capture stdout/stderr/exit code, enforce the 120s timeout and 30,000-char cap, reject empty commands, and terminate the process tree on interrupt returning an "interrupted" result
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  - [x] 10.2 Write property test for shell output cap
    - **Property 12: Shell and git output char cap** (shell portion)
    - **Validates: Requirements 7.5, 9.6**
    - _Requirements: 7.5_
  - [x] 10.3 Write unit tests for shell behavior
    - Cover happy path, non-zero exit (exit code + error output), timeout, empty-command, and interrupt termination
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.6_

- [x] 11. Implement the search tool
  - [x] 11.1 Implement the in-house search engine in `forge/tools/search.py`
    - Content mode: compile the pattern with `re` (invalid → error result), walk the workspace yielding path/1-based line number/matching line, apply the 100-match limit and 500-char line cap with truncation flags; glob mode: return workspace paths matching the glob via `pathlib`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_
  - [x] 11.2 Write property test for content-search correctness
    - **Property 13: Search match correctness**
    - **Validates: Requirements 8.1**
    - _Requirements: 8.1_
  - [x] 11.3 Write property test for search result and line caps
    - **Property 14: Search result and line caps**
    - **Validates: Requirements 8.4, 8.6**
    - _Requirements: 8.4, 8.6_
  - [x] 11.4 Write property test for glob correctness
    - **Property 15: Glob correctness**
    - **Validates: Requirements 8.2**
    - _Requirements: 8.2_
  - [x] 11.5 Write unit tests for search edge cases
    - Cover no-matches result and invalid-regex error result
    - _Requirements: 8.3, 8.5_

- [x] 12. Implement the git tool
  - [x] 12.1 Implement the git tool in `forge/tools/git.py`
    - Dispatch exactly {status, diff, log, show, add, commit, branch, checkout, stash} through the `git` binary in the workspace; reject unsupported operations, report not-a-repo, surface non-zero exit with captured stderr, and apply the 30,000-char cap with a truncation flag
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_
  - [x] 12.2 Write property test for git operation dispatch
    - **Property 16: Git operation dispatch**
    - **Validates: Requirements 9.1, 9.4**
    - _Requirements: 9.1, 9.4_
  - [x] 12.3 Write property test for git output cap
    - **Property 12: Shell and git output char cap** (git portion)
    - **Validates: Requirements 7.5, 9.6**
    - _Requirements: 9.6_
  - [x] 12.4 Write unit tests for git behavior
    - Cover happy path, not-a-repo result, and non-zero exit with captured error output
    - _Requirements: 9.2, 9.3, 9.5_

- [x] 13. Implement the planning tool
  - [x] 13.1 Implement the planning/todo tool in `forge/tools/planning.py`
    - Store up to 100 session-scoped todo items, update item status within {pending, in_progress, completed}, return the current list, signal the REPL on change, and return a no-op error result when updating an absent item or an out-of-set status
    - Support clearing/replacing the todo list (Req 10.5)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_
  - [x] 13.2 Write property test for todo invariants
    - **Property 17: Todo store, update, and status invariants**
    - **Validates: Requirements 10.1, 10.2, 10.4, 10.6**
    - _Requirements: 10.1, 10.2, 10.4, 10.6_

- [x] 14. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Implement the interrupt controller
  - [x] 15.1 Implement `InterruptController` in `forge/interrupt.py`
    - Install a `SIGINT` handler that is a no-op cancel while idle and sets a `threading.Event` during a turn; expose check/reset methods with sub-second polling semantics so generation and tools stop within 1 second
    - _Requirements: 4.2, 4.3, 4.4_
  - [x] 15.2 Write unit test for `InterruptController`
    - Verify it is a no-op while idle, sets the event during a turn, and that check/reset semantics behave correctly (check reflects a tripped event, reset clears it)
    - _Requirements: 4.2, 4.3, 4.4_

- [x] 16. Implement the Vertex AI client
  - [x] 16.1 Implement `VertexClient` and `StreamEvent` in `forge/vertex.py`
    - Lazily construct `genai.Client(vertexai=True, project, location)`; implement `generate_stream(contents, tools)` yielding `TextDelta | ToolCall | UsageReport | Done`, checking the interrupt between chunks; translate failures into typed exceptions (`CredentialsError`, `ConfigMissingError`, `AuthorizationError`, `RateLimitError`, `RequestTimeoutError`) with rate-limit retry up to `rate_limit_retries` (exponential backoff) and a 60s request timeout
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3.1, 3.4_
  - [x] 16.2 Write unit tests for Vertex error and retry paths
    - With injected/mocked responses, cover authorization error, rate-limit retry exhaustion, request timeout, and mid-stream interruption/error (partial tokens retained)
    - Assert generation aborts on a tripped interrupt within ~1 second (stream stops checking between chunks)
    - _Requirements: 2.5, 2.6, 2.8, 3.4, 4.2_

- [x] 17. Implement the usage tracker
  - [x] 17.1 Implement `UsageTracker` in `forge/usage.py`
    - Record per-response input/output token counts, compute cumulative totals, and compute cost from `Config.pricing`; when pricing is absent, report tokens with cost marked unavailable
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5_
  - [x] 17.2 Write property test for usage accumulation and cost
    - **Property 24: Usage accumulation and cost**
    - **Validates: Requirements 17.1, 17.2, 17.4**
    - _Requirements: 17.1, 17.2, 17.4_
  - [x] 17.3 Write unit test for cost-unavailable display
    - Assert tokens are reported with a cost-unavailable indication when pricing is absent
    - _Requirements: 17.5_

- [x] 18. Implement the context manager and compaction
  - [x] 18.1 Implement system-prompt assembly and token estimation in `forge/context.py`
    - Ship the built-in default prompt as `forge/data/system_prompt.md` loaded via `importlib.resources`; place it first followed by configured steering file contents in order, warning and skipping missing steering files; implement `estimate_tokens` as the deterministic `ceil(chars / 4)` + per-message overhead heuristic
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 14.1_
  - [x] 18.2 Implement compaction in `forge/context.py`
    - When the estimate exceeds `Token_Limit`, partition into system/task prompt, middle region, and retained-recent messages + pending tool calls; summarize the middle region via a bounded Model request preserving decisions/outcomes; drop oldest retained-recent messages if still over limit (never the task, system prompt, or pending tool calls); emit a warning and proceed with the smallest well-formed window if it cannot reach the limit; return `CompactionInfo` from `assemble`
    - `ContextManager` receives an injected `VertexClient` (or a summarization callable) used to summarize the middle region, making the dependency on `VertexClient` explicit
    - _Requirements: 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9_
  - [x] 18.3 Write property test for steering prompt ordering
    - **Property 22: Steering prompt ordering**
    - **Validates: Requirements 15.1, 15.2, 15.3**
    - _Requirements: 15.1, 15.2, 15.3_
  - [x] 18.4 Write property test for compaction bound and retention
    - **Property 21: Compaction bound and retention invariant**
    - **Validates: Requirements 14.1, 14.3, 14.5, 14.6, 14.8, 14.9**
    - Mock the summarization Model call so the property stays offline
    - _Requirements: 14.1, 14.3, 14.5, 14.6, 14.8, 14.9_
  - [x] 18.5 Write unit tests for compaction trigger and steering warnings
    - Cover the over-limit trigger, the compaction notice surfaced, and the missing-steering-file warning
    - _Requirements: 14.2, 14.4, 14.7, 15.4_

- [x] 19. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 20. Implement the agent loop
  - [x] 20.1 Implement `AgentLoop.run_turn` in `forge/agent.py`
    - Append the user message; loop: assemble context (capturing `CompactionInfo`), stream a model response (rendering tokens and tool-name announcements), collect tool calls, execute them in received order appending exactly one `Tool_Result` each, and continue until a response carries no tool calls; on interrupt stop promptly retaining completed messages/results; trigger persistence and return a `TurnResult` carrying the usage summary and any `CompactionInfo`
    - Wire together `ContextManager`, `VertexClient`, `ToolExecutor`, `UsageTracker`, and `SessionStore`
    - Depends on all built-in tools (read, write, edit, shell, search, git, planning) being implemented so the executor registry is complete
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 3.2, 4.1, 4.5, 14.7_
  - [x] 20.2 Write property test for tool-call execution order
    - **Property 3: Tool calls execute in received order**
    - **Validates: Requirements 1.4, 1.5**
    - _Requirements: 1.4, 1.5_
  - [x] 20.3 Write unit tests for agent loop control flow
    - Use a scripted mock model to cover no-tool-call response, multi-tool turn continuation, and interrupt retention of completed messages/results
    - _Requirements: 1.3, 1.5, 4.5_

- [x] 21. Implement the REPL
  - [x] 21.1 Implement `Repl` in `forge/repl.py`
    - Read input via `prompt_toolkit`; treat exact `/exit` and `/quit` as termination; ignore empty/whitespace-only input; otherwise call `AgentLoop.run_turn`; render streamed text per token, announce tool names, print an end-of-response indicator, render todo lists on change, render the compaction notice from `TurnResult`, and print the usage summary
    - On a stream that terminates with an error or interruption, render an error indicator and retain the partial response tokens already displayed
    - _Requirements: 1.1, 1.6, 1.7, 3.1, 3.2, 3.3, 3.4, 10.3, 14.7, 17.3_
  - [x] 21.2 Write property test for exit-command classification
    - **Property 1: Exit-command classification**
    - **Validates: Requirements 1.6**
    - _Requirements: 1.6_
  - [x] 21.3 Write property test for blank-input handling
    - **Property 2: Blank input is ignored**
    - **Validates: Requirements 1.7**
    - _Requirements: 1.7_
  - [x] 21.4 Write unit tests for REPL rendering flow
    - Cover prompt display on launch, no-tool-call response display, streaming output, end-of-response indicator, todo-list rendering when the list changes, and stream-error-indicator rendering with partial-token retention
    - _Requirements: 1.1, 1.3, 3.1, 3.3, 3.4, 10.3_

- [x] 22. Implement the MCP client
  - [x] 22.1 Implement `McpClient` in `forge/mcp_client.py`
    - `connect_all` connects to each configured server (30s budget each, warn+continue on failure), discovers tools, adapts them to the `Tool` protocol, and resolves name collisions by keeping the built-in tool and excluding+warning the conflicting MCP tool; `call` forwards a tool call and returns the response as a `Tool_Result` (errors/unreachable → failure result); `close` tears down connections
    - Register accepted MCP tools into the `ToolExecutor` registry
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6_
  - [x] 22.2 Write property test for MCP name-collision resolution
    - **Property 23: MCP name-collision resolution**
    - **Validates: Requirements 16.6**
    - _Requirements: 16.6_
  - [x] 22.3 Write integration tests against a stub MCP server
    - Over stdio, cover connect/discover, tool-call forwarding, connect-failure warning, and call-time error/unreachable handling
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5_

- [x] 23. Implement app bootstrap and startup validation
  - [x] 23.1 Implement `app.py` bootstrap, wiring, and startup validation
    - Load config (defaults applied when file absent), then validate required `project`/`region` before constructing the Vertex client — print guidance to run `forge init` and exit when missing; detect missing ADC and print the credentials message naming `gcloud auth application-default login`; construct and wire `ConfigManager`, `SessionStore`, `InterruptController`, `VertexClient`, `ToolExecutor` (built-in + MCP tools), `ContextManager`, `UsageTracker`, `AgentLoop`, and `Repl`
    - _Requirements: 2.2, 2.3, 2.4, 11.5, 12.3_
  - [x] 23.2 Write unit tests for startup validation
    - Cover missing-ADC message, and missing project/region directing the user to `forge init`
    - _Requirements: 2.3, 2.4, 12.3_

- [x] 24. Implement CLI dispatch
  - [x] 24.1 Implement `__main__.py` CLI dispatch
    - Parse `forge` (fresh REPL), `forge init` (create config when absent / report-and-leave when present), `forge list` (id + creation timestamp via `SessionStore.list`), and `forge resume <session_id>` (load session, seed the agent loop's context window; unknown id → error message; corrupt file → error naming the session); route into `app.py` bootstrap
    - _Requirements: 12.1, 12.2, 13.4, 13.5, 13.6, 13.7_
  - [x] 24.2 Write integration tests for CLI dispatch
    - Cover `init` create and already-exists, `list` output, `resume` happy path, unknown-id error, and corrupt-session error
    - _Requirements: 12.1, 12.2, 13.4, 13.5, 13.6, 13.7_

- [x] 25. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each task references specific requirements (granular sub-requirement clauses) for traceability.
- Property-based tests use Hypothesis at 100+ iterations and are tagged `# Feature: forge, Property {n}: ...`; network, Vertex AI, and MCP transports are mocked so logic-layer properties stay fast and offline, and filesystem properties run against a per-example `tmp_path` workspace.
- Non-PBT criteria (REPL control flow, Vertex error/retry/timeout, interrupt behavior, shell/git happy/error paths, session atomic/sequential writes and corrupt/unknown resume, init-exists and missing-required-value startup, compaction trigger/notice, steering warnings, cost-unavailable display, and MCP transport) are covered by example/unit and integration tests as described in the design's Testing Strategy.
- ADC authentication (2.2) is exercised as a startup smoke check; the git supported-set (9.1) and TOML-format (11.1) criteria are covered by dispatch and configuration assertions.
- Checkpoints ensure incremental validation at natural boundaries.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "3.1", "4.1", "15.1"] },
    { "id": 2, "tasks": ["2.2", "3.2", "3.3", "3.4", "4.2", "5.1", "15.2", "16.1", "17.1", "18.1"] },
    { "id": 3, "tasks": ["4.3", "4.4", "5.2", "5.3", "5.4", "7.1", "10.1", "11.1", "12.1", "13.1", "16.2", "17.2", "17.3", "18.2"] },
    { "id": 4, "tasks": ["7.2", "7.3", "7.4", "7.5", "8.1", "10.2", "10.3", "11.2", "11.3", "11.4", "11.5", "12.2", "12.3", "12.4", "13.2", "18.3", "18.4", "18.5", "22.1"] },
    { "id": 5, "tasks": ["8.2", "8.3", "9.1", "22.2", "22.3"] },
    { "id": 6, "tasks": ["9.2", "9.3", "20.1"] },
    { "id": 7, "tasks": ["20.2", "20.3", "21.1"] },
    { "id": 8, "tasks": ["21.2", "21.3", "21.4", "23.1"] },
    { "id": 9, "tasks": ["23.2", "24.1"] },
    { "id": 10, "tasks": ["24.2"] }
  ]
}
```
