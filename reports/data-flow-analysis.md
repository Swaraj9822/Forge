# Data Flow Analysis

## Request Lifecycle — one agent turn

```
User line (repl.py::run_once)
  → is_exit_command / is_blank guards  (no model call on exit/blank)
  → AgentLoop.run_turn(session, text)            [forge/agent.py]
      → append user Message
      → interrupt.begin_turn()                   [arms Ctrl-C → event]
      → loop:
          → ContextManager.assemble(session)     [forge/context.py]
              → system prompt + steering files + conversation
              → estimate_tokens; if > token_limit → compact()
          → VertexClient.generate_stream(contents, specs)  [forge/vertex.py]
              → _to_sdk_contents (wire→SDK, pull out system_instruction)
              → stream chunks → TextDelta / ToolCall / UsageReport / Done
              → interrupt + wall-clock timeout polled between chunks
              → rate-limit → capped/jittered backoff retry (pre-emit only)
          → append model Message (text + tool_calls)
          → if tool_calls: ToolExecutor.execute each in order  [tools/base.py]
              → exposure check (registry ∩ enabled)
              → interrupt check → validate → run → interrupt check
              → append one tool Message per call
              → write/edit success flags mutated_files
          → repeat until a model response has no tool calls
      → interrupt.end_turn()
      → mirror cumulative usage onto session; SessionStore.save (atomic)
  → (optional) VerificationCoordinator.run(session, result)   [verification.py]
      → gate (should_verify) → run verify cmd → bounded correction loop
      → append VerificationRecord; aggregate usage; save
  → Repl renders: streamed text (live), [end of response], errors/interrupt,
    todos-if-changed, usage summary
```

**Validated at each stage:** tool args are validated before `run` (side-effect
free); paths are scoped for fs/search tools; the model `contents` are translated
so only `user`/`model` roles reach Gemini and tool-result part counts match
function-call part counts.

**What can silently fail (by design, surfaced not hidden):**
- A `VertexError` ends the turn gracefully with partial text retained and the
  error surfaced on `TurnResult.error` (rendered as `[error] ...`).
- A missing steering file warns and is skipped.
- An MCP connect failure warns and continues with built-ins.
- Compaction that cannot reach the limit warns and proceeds with the smallest
  well-formed window.

## Side-Effects Map (write operations)

| Operation | DB/disk write | Process spawn | Network | Logged |
|-----------|---------------|---------------|---------|--------|
| `write`/`edit` tool | Atomic file write in workspace | — | — | result appended to session |
| `shell` tool | (whatever the command does) | Yes — platform shell, own process group | possible (command-dependent) | output captured + persisted |
| `git` tool | repo mutations (`add`/`commit`/`checkout`/`stash`) | Yes — `git` (list argv) | possible (`push`-like not in set) | output persisted |
| `planning` tool | none (state in ToolContext) | — | — | todos synced to session |
| Each turn | `SessionStore.save` (atomic JSON) | — | — | — |
| Verification | `SessionStore.save` + verify-command side effects | Yes — verify command via shell core | command-dependent | feedback appended |
| MCP tool call | server-dependent | server subprocess (at startup) | server-dependent | result persisted |

**Hidden / missing side-effects of note:** there is **no audit log of executed
shell/git commands** separate from the session transcript. Given SEC-001, an
explicit command-execution log would materially improve incident
reconstruction. This is the most notable "missing side effect."

## State Flow (frontend)

N/A — terminal UI. The only client-side "state" is the REPL's
`_last_rendered_todos` snapshot used to render the todo list only when it
changes; the session is the single source of truth.
