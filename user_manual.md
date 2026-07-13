# Forge User Manual

Forge is a terminal-based AI coding agent. You describe a task in plain
language; Forge sends it to a language model, streams the reply to your
terminal, and lets the model use a set of coding tools — read/write/edit files,
run shell commands, search the codebase, and run git — to complete the work.

It runs two ways:

- **Interactive REPL** — a conversational prompt (`forge`).
- **Headless** — a single scripted prompt for CI or automation (`forge -p "…"`).

This manual covers everything you need to configure and use Forge.

---

## Table of contents

1. [Installation](#1-installation)
2. [Quick start](#2-quick-start)
3. [Configuration](#3-configuration)
4. [Providers and authentication](#4-providers-and-authentication)
5. [Running Forge](#5-running-forge)
6. [Sessions](#6-sessions)
7. [Autonomy modes and approvals](#7-autonomy-modes-and-approvals)
8. [Built-in tools](#8-built-in-tools)
9. [REPL commands and mentions](#9-repl-commands-and-mentions)
10. [Context, steering, and project memory](#10-context-steering-and-project-memory)
11. [Durable memory](#11-durable-memory)
12. [Repository map](#12-repository-map)
13. [Planning and todos](#13-planning-and-todos)
14. [Checkpoints and undo](#14-checkpoints-and-undo)
15. [Verification loop](#15-verification-loop)
16. [Subagents (delegation)](#16-subagents-delegation)
17. [Parallel tool execution](#17-parallel-tool-execution)
18. [MCP servers](#18-mcp-servers)
19. [Usage, cost, and budgets](#19-usage-cost-and-budgets)
20. [Interrupting Forge](#20-interrupting-forge)
21. [Terminal UI](#21-terminal-ui)
22. [Files and locations](#22-files-and-locations)
23. [Troubleshooting](#23-troubleshooting)

---

## 1. Installation

Forge requires **Python 3.11 or newer**.

Install in editable mode with the development dependencies:

```
pip install -e ".[dev]"
```

Optional extras:

- Non-Google providers (Anthropic, OpenAI):

  ```
  pip install -e ".[providers]"
  ```

- The `mcp` package is required only if you configure MCP servers (see
  [section 18](#18-mcp-servers)).

After installation the `forge` command is on your PATH. You can also run it as a
module: `python -m forge`.

---

## 2. Quick start

```
# 1. Create a configuration file with documented defaults
forge init

# 2. Edit the config to set your model provider credentials
#    (for the default Vertex/Gemini provider: project + region)

# 3. Start the interactive agent
forge
```

At the `forge>` prompt, type a request in plain language:

```
forge> add a docstring to every public function in src/utils.py
```

Forge streams the model's response, announces each tool before it runs, and — in
the default safe mode — asks you to approve file writes and shell commands.

To leave the REPL, type `/exit` or `/quit`, or press `Ctrl-D`.

---

## 3. Configuration

### 3.1 Creating the config

```
forge init
```

This writes a `config.toml` populated with documented defaults and required
placeholders. If a config already exists, `forge init` leaves it untouched and
tells you so.

### 3.2 Where the config lives

| OS | Path |
|----|------|
| Windows | `%APPDATA%\forge\config.toml` |
| Linux / macOS | `$XDG_CONFIG_HOME/forge/config.toml`, else `~/.config/forge/config.toml` |

A missing config file is **not** an error — Forge applies all defaults. A config
file with a TOML **syntax error** is fatal and Forge reports the offending line
and column.

### 3.3 What a fresh `forge init` turns on

The generated config is deliberately safe and feature-rich:

- **supervised** autonomy mode (asks before mutations),
- checkpoints enabled (so `/undo` works),
- durable memory enabled,
- repository map enabled and injected,
- color output and spinner enabled,
- `@`-mention file expansion enabled,
- subagents **off** and the verification loop **unset** (opt-in; see their
  sections).

### 3.4 Full configuration reference

All keys are optional; anything absent falls back to the default shown.

**Top level / `[provider]`**

| Key | Default | Meaning |
|-----|---------|---------|
| `model` | `gemini-3.1-pro-preview` | Model id. |
| `project` | — | GCP project id (required for the Vertex provider). |
| `region` | — | GCP region (required for the Vertex provider). |
| `enabled_tools` | all built-ins except `delegate` | Which tools the model may use. |
| `steering_files` | `[]` | Extra instruction files prepended to the system prompt. |
| `provider.type` | `vertex` | `vertex`, `anthropic`, or `openai`. |
| `provider.model` | (falls back to `model`) | Per-provider model override. |
| `provider.api_key_env` | provider default | Env var holding the API key. |
| `provider.base_url` | — | Custom base URL (OpenAI-compatible endpoints). |
| `provider.thinking_level` | — (model default) | Reasoning depth for Gemini 3 models: `minimal`, `low`, `medium`, or `high`. See [section 4](#4-providers-and-authentication). |

**`[limits]`**

| Key | Default | Meaning |
|-----|---------|---------|
| `token_limit` | `200000` | Context budget; compaction triggers above it. |
| `retained_recent_messages` | `20` | Recent messages kept verbatim during compaction. |
| `request_timeout_s` | `60` | Per-model-request wall-clock timeout. |
| `shell_timeout_s` | `120` | Shell command timeout. |
| `output_cap_chars` | `30000` | Max characters captured from tool output. |
| `search_result_limit` | `100` | Max content-search matches. |
| `search_line_cap` | `500` | Max characters per matched line. |
| `read_max_lines` | `2000` | Read-tool line cap. |
| `read_max_bytes` | `1000000` | Read-tool byte cap (also the checkpoint size cap). |
| `rate_limit_retries` | `5` | Rate-limit retry attempts with backoff. |
| `mcp_connect_timeout_s` | `30` | Per-MCP-server connect budget. |

**`[pricing]`** — enables cost estimates.

| Key | Default | Meaning |
|-----|---------|---------|
| `input_per_1k` | unset | USD per 1,000 input tokens. |
| `output_per_1k` | unset | USD per 1,000 output tokens. |

If either is unset, Forge shows token counts but reports "cost unavailable".

**`[context]`**

| Key | Default | Meaning |
|-----|---------|---------|
| `plan_reminder` | `true` | Re-inject the current todo plan each turn. |
| `project_memory` | `true` | Auto-load `FORGE.md`/`AGENTS.md` from the workspace. |

**`[policy]`** — autonomy and approvals (see [section 7](#7-autonomy-modes-and-approvals)).

| Key | Default | Meaning |
|-----|---------|---------|
| `mode` | `autopilot` (fresh init: `supervised`) | `autopilot`, `supervised`, or `readonly`. |
| `shell_allowlist` | `[]` | Shell commands auto-approved in supervised mode. |
| `show_diffs` | `false` (fresh init: `true`) | Show unified diffs for writes/edits. |

**`[checkpoint]`**

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `true` | Snapshot files before writes/edits so `/undo` works. |
| `keep_turns` | `10` | How many past turns are undoable. |

**`[memory]`** (see [section 11](#11-durable-memory))

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `false` (fresh init: `true`) | Turn on cross-session memory. |
| `max_records` | `500` | Cap on stored memories. |
| `inject_limit` | `5` | Max memories injected per turn. |
| `inject_char_budget` | `2000` | Character budget for injected memories. |

**`[repo_map]`** (see [section 12](#12-repository-map))

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `false` (fresh init: `true`) | Build the repository map. |
| `inject` | `false` (fresh init: `true`) | Inject the map into context each turn. |
| `char_budget` | `4000` | Character budget for the injected map. |

**`[ui]`, `[commands]`, `[parallel]`, `[mentions]`**

| Key | Default | Meaning |
|-----|---------|---------|
| `ui.color` | `false` (fresh init: `true`) | Colorized terminal output. |
| `ui.spinner` | `false` (fresh init: `true`) | Show a spinner while waiting on the model. |
| `commands.dir` | `.forge/commands` | Folder of custom slash-command files. |
| `parallel.enabled` | `false` | Run independent read-only tools concurrently. |
| `parallel.max_workers` | `4` | Worker cap for parallel execution. |
| `mentions.enabled` | `false` (fresh init: `true`) | Expand `@path` mentions to file contents. |

**`[subagents]`** (see [section 16](#16-subagents-delegation))

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `false` | Enable the `delegate` tool. |
| `default_tools` | `[read, search, repo_index, search_memory]` | Tools a subagent may use. |
| `max_turns` | `4` | Max round-trips per subagent. |

**`[verification]`** (see [section 15](#15-verification-loop))

| Key | Default | Meaning |
|-----|---------|---------|
| `command` | unset (disabled) | Command to run after a turn (e.g. `pytest -q`). |
| `trigger` | `on_file_change` | `on_file_change` or `always`. |
| `max_correction_iterations` | `3` | Self-correction attempts on failure. |
| `timeout_s` | `shell_timeout_s` | Timeout for the verify command. |

**`[mcp_servers]`** — a list of MCP server definitions (see [section 18](#18-mcp-servers)).

---

## 4. Providers and authentication

Forge speaks to three model providers. Select one with `provider.type`.

### Vertex AI / Gemini (default)

Requires a GCP `project` and `region` in the config, plus Application Default
Credentials on the machine:

```
gcloud auth application-default login
```

At startup Forge validates that `project`/`region` are set and probes ADC; a
missing credential prints the exact `gcloud` command to run.

#### Thinking level (Gemini 3)

Gemini 3 models perform internal reasoning before answering. Control how much
with `provider.thinking_level`:

```toml
[provider]
thinking_level = "high"   # minimal | low | medium | high
```

- Higher levels reason more deeply (better on complex tasks) at the cost of
  higher latency and more tokens; lower levels are faster and cheaper.
- When the key is **absent**, Forge sends no thinking parameter, so the model
  uses its own default (for Gemini 3 Pro that default is `high`).
- The level is a *relative allowance*, not a hard token cap; actual usage varies
  with task complexity.
- Reasoning ("thoughts") tokens are billed at the output-token rate; Forge folds
  them into the reported output token count and cost (see
  [section 19](#19-usage-cost-and-budgets)).
- This maps to the Gemini 3 `thinking_level` request parameter (which replaced
  the older `thinking_budget`). Older SDKs that don't accept it degrade
  gracefully — the request still runs, without the level applied.

### Anthropic

```toml
[provider]
type = "anthropic"
model = "claude-..."
```

Set your key in the environment (default variable `ANTHROPIC_API_KEY`, or name
your own via `provider.api_key_env`):

```
export ANTHROPIC_API_KEY=sk-ant-...
```

### OpenAI (and compatible endpoints)

```toml
[provider]
type = "openai"
model = "gpt-..."
base_url = "https://your-endpoint/v1"   # optional
```

Set `OPENAI_API_KEY` (or your `provider.api_key_env`). Use `base_url` for
OpenAI-compatible servers.

Forge handles rate limits with capped exponential backoff, enforces the request
timeout, and surfaces credential/authorization/timeout errors as readable
messages without losing your session.

---

## 5. Running Forge

### 5.1 Interactive REPL

```
forge
```

On launch Forge prints a short banner (model, provider, thinking level, autonomy
mode, tool count, workspace). You type a request; Forge shows a "thinking"
spinner while waiting on the model, then streams the reply. Each tool is
announced with its target before it runs — e.g. `[tool: read] src/app.py` or
`[tool: shell] $ pytest -q` — and each result is followed by a one-line outcome
(`-> 42 lines`, `-> wrote 1200 bytes`, or `-> error: …`). An end-of-response
marker with the turn's elapsed time closes the turn, followed by a usage
summary. If the model asked for a plan, the todo list is shown when it changes.

### 5.2 Headless (non-interactive)

Run a single prompt and exit — ideal for scripts and CI:

```
forge -p "list the python files in this repo"
forge -p "summarize README.md" --output json
echo "explain this error: ..." | forge -p -      # read prompt from stdin
```

Flags:

| Flag | Effect |
|------|--------|
| `-p`, `--prompt` | The prompt. `-` reads the whole prompt from stdin. |
| `--output text` | Stream the reply as plain text (default). |
| `--output json` | Emit a single parseable JSON object at the end. |
| `--yes` | Auto-approve every gated tool call (see below). |
| `--max-turns N` | Cap model round-trips; overrun exits with code 5. |
| `--max-cost X` | Cap estimated USD spend; overrun exits with code 5. |

**Approvals in headless mode.** Without `--yes`, a `supervised` or `readonly`
run **refuses** any gated mutation (returns a denied/forbidden tool result)
rather than hanging on a prompt no one can answer. Pass `--yes` to auto-approve
everything, matching autopilot behavior:

```
# Refuses write/edit/shell unless allowlisted (or you pass --yes)
forge -p "summarize the code" --output json

# Trusted environment: approve everything and bound the run
forge -p "run the tests and fix failures" --yes --max-turns 20 --max-cost 0.50
```

**JSON output** contains: `session_id`, `ok`, `response`, `error`,
`interrupted`, `budget_exceeded`, `mutated_files`, `usage`, `verification`, and
`todos`.

**Exit codes** (stable contract for CI):

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `1` | Startup/usage error (bad config, missing credentials, invalid flag combo). |
| `2` | The turn ended with a model/provider error. |
| `3` | The run was interrupted. |
| `4` | The verification command still failed after correction attempts. |
| `5` | A run budget (`--max-turns` / `--max-cost`) was exceeded. |

---

## 6. Sessions

Every run is a **session**: the full conversation, tool results, todo list, and
usage are persisted to disk automatically after each turn.

```
forge list                 # list saved sessions (id and creation time)
forge resume <session_id>  # restore a session and continue it in the REPL
```

Resuming reseeds the conversation, the todo plan, and cumulative usage totals so
work continues seamlessly. An unknown id reports the id; a corrupt session file
is reported by name and left untouched on disk.

Sessions are stored one JSON file per session (see [section 22](#22-files-and-locations)).

---

## 7. Autonomy modes and approvals

Forge's approval policy decides, per tool call, whether to run it, ask you, or
refuse it. It is set by `policy.mode`:

| Mode | Behavior |
|------|----------|
| `autopilot` | Runs everything, no prompts. |
| `supervised` | Prompts before **mutating** tools (write, edit, non-allowlisted shell, mutating git). Read-only tools never prompt. |
| `readonly` | **Forbids** mutating tools outright (no prompt); read-only tools run. |

### 7.1 The approval prompt

In supervised mode, a gated call shows a summary (and a diff preview for
writes/edits) and asks:

```
[approve] write wants to run: path=src/app.py, content=...
[y/n/a]
```

- `y` / `yes` — approve this once.
- `a` / `always` — approve and remember this action for the rest of the session
  (the same tool + target won't ask again).
- `n` / `no` / anything else / blank — deny (the model is told and can adapt).

### 7.2 Shell allowlist

In supervised mode, a shell command is auto-approved only if it matches an entry
in `policy.shell_allowlist` **and** contains no shell metacharacters
(`;`, `&`, `|`, backtick, `$(`, `>`, `<`, `&&`, `||`, newline). Any compound or
redirected command always asks.

Allowlist entries match by **token sequence**:

- a single token (`"git"`) matches any invocation of that program, while
- a multi-token entry (`"git status"`) matches only that exact leading
  sequence.

This lets you allow `git status`, `git diff`, `git log` without also
auto-approving `git reset --hard`. The default `forge init` allowlist ships the
narrow, safe git subcommands rather than a bare `git`.

### 7.3 Git operation gating

The `git` tool is gated per operation. Read-only operations
(`status`, `diff`, `log`, `show`, `branch`) run without a prompt; everything
else — `add`, `commit`, `checkout`, `stash`, and any unrecognized operation —
requires approval in supervised mode and is forbidden in readonly mode. (The
policy fails closed: an operation it doesn't recognize is treated as mutating.)

### 7.4 Diffs

With `policy.show_diffs = true`, Forge prints the unified diff of each
successful write/edit so you can see exactly what changed.

---

## 8. Built-in tools

Forge exposes these tools to the model. Which are available is controlled by
`enabled_tools`. Every tool is scoped to the **workspace** (the directory you
launched Forge from); paths that resolve outside it are refused.

| Tool | Read-only | What it does |
|------|-----------|--------------|
| `read` | yes | Return a UTF-8 file's contents, optionally a 1-based line range. Output is capped; binary files are refused. |
| `write` | no | Write a file atomically, creating parent directories. Preserves the existing file's permission bits. |
| `edit` | no | Modify a file in one of three modes (below), atomically. |
| `search` | yes | Search file **contents** by regex, or file **names** by glob. |
| `git` | per-op | Run a fixed set of git operations in the workspace repo. |
| `planning` | yes | Maintain a session todo list (see [section 13](#13-planning-and-todos)). |
| `remember` | no* | Save a durable memory (see [section 11](#11-durable-memory)). |
| `search_memory` | yes | Search durable memories. |
| `repo_index` | yes | Produce a structural map of the codebase. |
| `delegate` | — | Hand a scoped task to a subagent (opt-in; [section 16](#16-subagents-delegation)). |

\* `remember` writes only to Forge's own memory store, not your workspace.

### edit modes

- **replace** (default): replace a unique target string. Zero matches → "not
  found"; more than one → "ambiguous".
- **anchored**: replace a target that occurs between an `after` and/or `before`
  anchor, to disambiguate repeated text.
- **line_range**: replace lines `start_line`..`end_line` (1-based, inclusive).

### shell tool

Runs a command through the platform default shell (`cmd.exe /C` on Windows,
`/bin/sh -c` on Unix), rooted at the workspace. It enforces the timeout, caps
combined output, and terminates the whole process tree on timeout or interrupt.
It returns stdout, stderr, and the exit code.

### git tool

Dispatches exactly: `status`, `diff`, `log`, `show`, `add`, `commit`, `branch`,
`checkout`, `stash`. Any other operation is rejected. Output is capped; a
non-repository workspace returns a clear "not a git repository" result.

MCP servers can add more tools alongside these — see [section 18](#18-mcp-servers).

---

## 9. REPL commands and mentions

Anything you type is sent to the agent, **except** these reserved commands:

| Command | Effect |
|---------|--------|
| `/exit`, `/quit` | Leave the REPL. |
| `/undo` | Revert the most recent turn's file changes (see [section 14](#14-checkpoints-and-undo)). |
| `/help`, `/commands` | List built-in and custom commands. |
| `/cost` | Show cumulative session token usage and estimated cost. |
| `/tools` | List the tools currently available to the model. |
| `/model` | Show the active model, provider, and thinking level. |
| `/clear` | Clear the terminal screen. |
| `/<name> [args]` | Run a custom slash command (below). |

Blank/whitespace-only input just re-displays the prompt.

### Custom slash commands

Drop a markdown file in `.forge/commands/` (configurable via `commands.dir`).
The file name is the command name; its body becomes the prompt. Placeholders are
substituted:

- `$ARGUMENTS` → everything you typed after the command,
- `$1`, `$2`, … → individual whitespace-separated arguments.

Example — `.forge/commands/review.md`:

```
Review the file $1 for bugs and style issues, focusing on $ARGUMENTS.
```

Usage: `/review src/app.py error handling` expands to a full prompt before it
reaches the model. Run `/help` to see discovered custom commands.

### @-mentions

When `mentions.enabled` is on, an `@path` token in your message is expanded
inline to that file's fenced contents before the message is sent:

```
forge> explain what @src/config.py does and how it relates to @README.md
```

Mentions inside code blocks/inline code are left alone. Out-of-scope, missing,
binary, or oversized files are left as literal text and reported with a
`[warning]`. Included files are noted with `[included: …]`.

---

## 10. Context, steering, and project memory

Forge builds the model's context each turn from: the built-in system prompt,
your steering files, an optional project-memory file, and the conversation so
far.

### Steering files

List extra instruction files in `steering_files`; their contents are appended to
the system prompt in order. Use them for team conventions, architectural notes,
or "always do X" rules. A missing steering file is warned about and skipped.

### Project memory (`FORGE.md` / `AGENTS.md`)

When `context.project_memory` is on, Forge auto-loads the first of `FORGE.md`
then `AGENTS.md` found in the workspace root and prepends it as project context.
Put project-specific guidance there and it applies to every session in that
repo automatically.

### Automatic compaction

When the estimated context exceeds `token_limit`, Forge compacts it: it keeps
the original task and the most recent `retained_recent_messages` verbatim,
summarizes the older middle, and preserves any in-flight tool calls. If it still
doesn't fit, it drops the oldest retained messages. You'll see a
`[notice] conversation context was compacted` line. This is automatic and
lossless with respect to the ongoing task's key decisions; you don't need to do
anything.

---

## 11. Durable memory

With `memory.enabled`, Forge keeps a cross-session memory store so useful facts
survive between runs.

- The model saves notes with the **`remember`** tool and retrieves them with
  **`search_memory`** (keyword + recency ranking).
- Relevant memories are automatically injected into context each turn
  (conditioned on your latest message, bounded by `inject_limit` and
  `inject_char_budget`).
- **Secret redaction**: common secrets (API keys, tokens, passwords,
  `Authorization:` values, AWS keys, private-key blocks, long hex/base64 blobs)
  are redacted on write. This is best-effort defense-in-depth, not a guarantee —
  don't rely on it to store secrets safely.
- **Staleness**: a memory tied to files is dropped from results once those files
  change (or go missing) after the memory was recorded, so stale notes don't
  mislead the model.
- The store is capped at `max_records`; the oldest are pruned.

Memory is stored per-workspace at `.forge/memory.jsonl`.

---

## 12. Repository map

With `repo_map.enabled`, Forge builds a structural overview of your codebase:
file paths plus top-level (and one level of nested) symbols. Python files are
parsed with the standard library AST; other languages (JS/TS, Go, Rust, Ruby)
use best-effort regex. Noise directories (`.git`, `node_modules`, `__pycache__`,
virtualenvs, build output, etc.) and hidden files are skipped, and symlinks that
escape the workspace are ignored.

The model can request the map on demand with the `repo_index` tool. With
`repo_map.inject = true`, a budgeted version of the map (`char_budget`) is also
injected into context each turn to orient the model on large projects.

---

## 13. Planning and todos

For multi-step work the model maintains a session **todo list** with the
`planning` tool. It can replace the list, update an item's status, clear it, or
read it back. Each item has text and a status of `pending`, `in_progress`, or
`completed`.

Forge renders the list whenever it changes:

```
[todos]
  [x] Read the failing test
  [~] Fix the off-by-one in paginate()
  [ ] Re-run the suite
```

With `context.plan_reminder` on (default), the current plan is re-injected each
turn so the model keeps long tasks on track. The plan persists across turns and
is restored when you `forge resume`.

---

## 14. Checkpoints and undo

With `checkpoint.enabled` (default), Forge snapshots a file's contents before a
`write`/`edit` mutates it, grouped per turn. Undo the most recent turn's file
changes at any time:

```
forge> /undo
[undo] restored 2 file(s): src/app.py, src/util.py
```

- Undo reverts files a turn **created** (by deleting them) and files it
  **modified** (by restoring the prior bytes), preserving their permissions.
- Up to `checkpoint.keep_turns` past turns are undoable; run `/undo` repeatedly
  to walk back further.
- Files larger than the read byte cap (`read_max_bytes`) are not snapshotted;
  `/undo` will note it can't restore those.
- Checkpoints are workspace-local (`.forge/checkpoints`) and need no git.

Note: checkpoints cover the `write`/`edit` tools. Changes made by arbitrary
`shell` commands (e.g. `rm`, `mv`) are not captured by the checkpoint store.

---

## 15. Verification loop

Forge can automatically run a verification command after a turn and, on failure,
feed the output back to the model to self-correct. It is **opt-in**: set
`verification.command`.

```toml
[verification]
command = "pytest -q"
trigger = "on_file_change"        # or "always"
max_correction_iterations = 3
```

- **trigger** — `on_file_change` runs verification only after a turn that
  modified files; `always` runs it after every completed turn.
- After the command runs, Forge classifies the outcome (`passed`, `failed`,
  `timed_out`, `start_error`). On a correctable failure it starts a correction
  turn with the captured output as feedback and re-runs, up to
  `max_correction_iterations`.
- Progress is shown as `[verify]` lines. Tokens spent on correction turns are
  folded into the turn's reported usage.
- In headless mode, a still-failing verification yields exit code `4`.

---

## 16. Subagents (delegation)

With `subagents.enabled`, the model gains a `delegate` tool that hands a scoped
task to a fresh **subagent** — an isolated agent with its own short context
window and a restricted, mostly read-only toolset. This keeps focused
research/exploration from bloating the main conversation.

```toml
[subagents]
enabled = true
default_tools = ["read", "search", "repo_index", "search_memory"]
max_turns = 4
```

- A subagent runs to completion within `max_turns` round-trips and returns a
  concise result to the parent.
- Subagents cannot delegate again (no recursion).
- The subagent's token usage is folded into the parent turn's usage and cost.

---

## 17. Parallel tool execution

With `parallel.enabled`, Forge runs consecutive **read-only** tool calls
(`read`, `search`, `search_memory`, `repo_index`) concurrently, up to
`parallel.max_workers`. This speeds up turns where the model reads or searches
several things at once. Mutating tools always run sequentially in the order the
model requested them.

---

## 18. MCP servers

Forge can connect to external **Model Context Protocol** servers and expose
their tools alongside the built-ins. Requires the `mcp` package installed.

```toml
[[mcp_servers]]
name = "docs"
command = "uvx"
args = ["some-mcp-server@latest"]
env = { LOG_LEVEL = "ERROR" }
```

- At startup Forge launches each server, discovers its tools, and adds the
  accepted ones to the model's toolset. A server that fails to connect or times
  out (`mcp_connect_timeout_s`) is skipped with a warning; Forge continues with
  the rest.
- **Name collisions** are resolved deterministically: an MCP tool that clashes
  with a built-in is excluded (built-in wins); when two servers expose the same
  name, the first one connected wins. Each exclusion is warned about.
- MCP tools are treated as non-read-only, so they follow the same approval
  policy as other mutating tools.

If the `mcp` package isn't installed but servers are configured, Forge warns and
runs with the built-in tools only.

---

## 19. Usage, cost, and budgets

After each turn Forge prints a usage summary:

```
[usage] turn: 1.2k in / 340 out | session: 8.4k in / 2.1k out | cost: $0.012600 turn / $0.089000 session
```

- Token counts are always shown (per turn and cumulative for the session), and
  are humanized for readability (e.g. `8.4k`).
- **Cost** is shown only when `[pricing]` is configured; otherwise you'll see
  `cost unavailable`.
- For **thinking models**, reasoning ("thoughts") tokens are billed at the
  output-token rate. Forge folds them into the reported output token count and
  cost, so a thinking model's usage and spend are accurate rather than
  under-reported.
- `/cost` prints the cumulative session usage on demand at any time.

### Budgets (headless)

Bound a non-interactive run so it can't run away in CI:

- `--max-turns N` — stop after `N` model round-trips.
- `--max-cost X` — stop once estimated spend reaches `X` USD. This **requires**
  `[pricing]`; without it the run is refused up front (rather than silently
  never enforcing the cap).

Hitting either budget stops the run cleanly, skips any verification, sets
`budget_exceeded: true` in JSON output, and exits with code `5`.

```
forge -p "refactor the module and run tests" --yes --max-turns 20 --max-cost 0.50
```

---

## 20. Interrupting Forge

Press **Ctrl-C** while a turn is running to interrupt it. Forge stops model
generation and any running tool (including killing a shell command's whole
process tree) within about a second, retains everything completed so far, and
returns to the prompt. Pressing Ctrl-C at the idle prompt cancels the current
input line; **Ctrl-D** (or Ctrl-C at an empty prompt) exits the REPL.

---

## 21. Terminal UI

On launch the interactive REPL prints a **startup banner** summarizing the
active model, provider (and thinking level), autonomy mode, exposed tool count,
and workspace.

**While waiting on the model**, a "thinking" spinner is shown (with
`ui.spinner`); it clears the moment the first output arrives. Each turn's
end-of-response marker includes the elapsed wall-clock time.

**Tool visibility.** Every tool call is announced with a short description of
what it's acting on — the file path, shell command, search pattern, git
operation, and so on (e.g. `[tool: edit] src/app.py`). After a tool runs, a
concise result line follows: a success summary (`-> 42 lines`,
`-> wrote 1200 bytes`, `-> plan updated`), a failure message
(`-> error: file not found`), or a policy notice (`-> denied by approval
policy`).

**Color and diffs.** With `ui.color`, Forge colorizes tool announcements and
result lines (green ok / yellow warning / red error) and syntax-highlights diffs
(when `show_diffs` is on). Color and the spinner require a real terminal (TTY)
and the `rich` library; they degrade gracefully to plain ASCII text otherwise
(for example when output is redirected to a file or pipe).

**Usage line.** Token counts are humanized (`8.4k`) for readability; see
[section 19](#19-usage-cost-and-budgets).

---

## 22. Files and locations

| What | Location |
|------|----------|
| Config file | `%APPDATA%\forge\config.toml` (Windows) / `~/.config/forge/config.toml` (Unix) |
| Saved sessions | `%APPDATA%\forge\sessions\` (Windows) / `~/.local/share/forge/sessions\` (Unix) |
| Checkpoints | `<workspace>/.forge/checkpoints/` |
| Durable memory | `<workspace>/.forge/memory.jsonl` |
| Custom commands | `<workspace>/.forge/commands/*.md` |
| Project memory | `<workspace>/FORGE.md` or `AGENTS.md` |

The **workspace** is the directory you launch `forge` from; it is the security
boundary for all file, search, and git operations.

---

## 23. Troubleshooting

**"Required configuration value(s) missing: GCP project ID / region."**
Run `forge init`, then edit the config to set `project` and `region` (Vertex),
or switch `provider.type` and set the appropriate API-key environment variable.

**"Application Default Credentials (ADC) are unavailable."**
Run `gcloud auth application-default login`.

**"…API key is missing. Set the environment variable …"**
Export the key for your provider (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or your
`provider.api_key_env`).

**Config file … contains a syntax error.**
Fix the reported line/column in `config.toml` (it's TOML).

**A headless run refuses to write/edit/run shell.**
That's supervised/readonly mode declining a mutation non-interactively. Pass
`--yes` to auto-approve, or add safe commands to `policy.shell_allowlist`, or
set `policy.mode = "autopilot"` for trusted environments.

**`--max-cost` reports it requires pricing.**
Add `[pricing]` with `input_per_1k` and `output_per_1k`, or drop `--max-cost`
and use `--max-turns` instead.

**Color/spinner/diffs don't show.**
They only activate on a real terminal with `rich` installed; piping or
redirecting output disables them by design.

**A tool says a path is "out of scope."**
The path resolved outside the workspace. Launch Forge from the correct
directory, or use a path inside it.

**MCP servers didn't load.**
Ensure the `mcp` package is installed and each server's `command` is on PATH;
check startup warnings for per-server connect/timeout errors.
