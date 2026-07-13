# Forge — Phase 4 Implementation Plan (Ergonomics & UX)

**Audience:** a coding agent implementing this end-to-end.
**Prerequisite:** Phases 1–3 merged.
**Scope:** Phase 4 only — developer ergonomics:

- **J — `@file` mentions + custom slash commands.** Let the user inject file
  contents into a turn (`@path`) and define reusable prompt shortcuts
  (`/name …`) as markdown files.
- **K — Richer terminal UI.** Color, syntax-highlighted diffs, and a spinner —
  the *cheap* wins. Streaming markdown is explicitly deferred.
- **L — Parallel read-only tool execution.** Run batched independent `read` /
  `search` calls concurrently, preserving result order and interrupt latency.

Do **not** implement multi-provider or subagents (Phase 5). Do **not** build
streaming-markdown rendering (documented reason below).

---

## 0. Ground rules

Same as `phase1.md` §0. New optional dependency allowed **only** for K:
`rich` (add to `pyproject.toml` `[project].dependencies`), imported guardedly so
the package still imports if it is absent (mirror the `prompt_toolkit` lazy
import in `repl.py`). Everything else stays stdlib. Preserve the injectable
`out: TextIO` seam in `Repl`/`headless` so tests stay headless. Test with
`pytest -q --basetemp=C:\forge_tmp`, then remove the temp dir.

---

## 1. Architecture overview

| File | Change |
|------|--------|
| `forge/commands.py` (new) | `expand_mentions` (pure `@file` expansion) + `SlashCommandStore` (load/apply markdown command templates). |
| `forge/repl.py` | Classify `@`/`/name` input; expand before `run_turn`; list/handle custom commands. |
| `forge/tools/base.py` | `Tool.read_only` already exists (Phase 2); used here to select parallel-eligible calls. |
| `forge/agent.py` | `_execute_tool_calls`: run a leading run of read-only calls concurrently; preserve received order. |
| `forge/ui.py` (new) | Optional `rich`-backed renderer helpers (color, diff highlight, spinner) behind a plain-text fallback. |
| `forge/repl.py` | Route rendering through `ui` helpers when enabled. |
| `forge/config.py` | `[ui]` table (color, spinner); `[commands]` dir; `[parallel]` toggle + max workers. |
| `tests/…` | Mention expansion, slash commands, parallel ordering/interrupt, UI fallback. |

---

## 2. Feature J — `@file` mentions + slash commands

### 2.1 `@file` mention expansion (`forge/commands.py`, pure)
```python
def expand_mentions(text: str, workspace_root: Path, *, max_bytes: int
                    ) -> tuple[str, list[str], list[str]]:
    """Replace @path tokens with fenced file contents.

    Returns (expanded_text, included_paths, warnings). Each @token that resolves
    (via resolve_in_workspace) to a readable UTF-8 workspace file is appended as
    a fenced block:  `--- <relpath> ---\\n```\\n<contents>\\n``` `. Out-of-scope,
    missing, binary, or oversized (> max_bytes) files are left as literal text
    and reported in `warnings` (the model still sees the raw @token).
    """
```
- Token grammar: `@` followed by a path with no spaces (support quoted
  `@"a b.py"`). Do not treat emails / decorators as mentions — require the token
  to resolve to an existing workspace file; otherwise leave it untouched.
- Reuse `resolve_in_workspace` for scoping (never read outside the workspace).
- Cap each file by `config.read_max_bytes`; flag truncation.
- Property tests: a valid `@file` is replaced with a fenced block containing the
  file's contents; `@nonexistent` is left verbatim + warned; out-of-scope
  `@../secret` left verbatim + warned; multiple mentions all expand.

### 2.2 Custom slash commands (`forge/commands.py`)
```python
class SlashCommandStore:
    def __init__(self, dirs: list[Path]): ...       # e.g. workspace/.forge/commands
    def names(self) -> list[str]: ...
    def render(self, name: str, arg_text: str) -> str | None:
        """Load <name>.md, substitute $ARGUMENTS (and $1..$N) with arg_text,
        return the prompt text; None if the command is unknown."""
```
- A command is a markdown file `<name>.md`; its body becomes the prompt.
  `$ARGUMENTS` → the text after `/name`; `$1`,`$2`… → whitespace-split args.
- Discovery dir: `workspace/.forge/commands/` (and optionally a user-level dir).
- Keep it text-only (no execution) — a slash command just produces prompt text
  that is then run like any user turn.

### 2.3 REPL integration (`forge/repl.py`)
In `run_once`, after reading `line` and the existing `/exit`,`/quit`,`/undo`
checks, add:
- `line == "/help"` or `/commands` → list built-in + custom command names;
  return True (no turn).
- `line.startswith("/")` and first token is a known custom command → render via
  `SlashCommandStore`, then treat the rendered text as the turn input.
- Unknown `/xxx` → print `unknown command` hint; return True (do **not** send a
  stray slash command to the model).
- For a normal (non-command) line: run `expand_mentions` on it first; if any
  files were included, optionally echo `[included: a.py, b.py]`; pass the
  expanded text to `run_turn`.
- **Order of classification:** exit → undo → help/commands → custom slash →
  blank → (expand mentions) → run turn. Keep `is_exit_command` exact-match
  semantics intact (Property 1).

Add `SlashCommandStore` + mention config to the `Repl` constructor (optional,
default disabled so existing REPL tests are unaffected).

### 2.4 Tests
`tests/test_mentions.py` (property, per 2.1); `tests/test_slash_commands.py`
(render + `$ARGUMENTS`/`$1` substitution, unknown command, listing);
`tests/test_repl_commands.py` (classification order; `@file` expansion feeds the
turn; `/help` and unknown `/x` never invoke the agent loop — spy).

---

## 3. Feature K — Richer terminal UI

### 3.1 `forge/ui.py` (new)
A thin, optional layer over `rich`, with a plain-text fallback so nothing breaks
when `rich` is absent or output is redirected/non-TTY:
```python
try:
    from rich.console import Console
    from rich.syntax import Syntax
    _RICH = True
except Exception:  # noqa: BLE001
    _RICH = False

class Ui:
    def __init__(self, out, *, color: bool, spinner: bool): ...
    def tool_announcement(self, name: str) -> str: ...   # colored "[tool: x]"
    def render_diff(self, diff_text: str) -> None: ...    # syntax-highlight udiff
    def status(self, message: str): ...                   # spinner context manager (no-op fallback)
```
- **Only** color, diff highlighting, tool-announcement styling, and a spinner
  during model streaming. Everything degrades to the current plain `_writeln`
  when `_RICH` is false, `color` is off, or `out` is not a TTY
  (`getattr(out, "isatty", lambda: False)()`).
- The spinner must **not** interleave with streamed tokens: show it only while
  *waiting* for the first token, then stop it before streaming text (the 200
  ms/token budget from Req 3.1 must be preserved).

### 3.2 REPL integration
Route `on_tool` through `ui.tool_announcement`, print write/edit diffs (from
Phase 2 `meta["diff"]` when `show_diffs`) via `ui.render_diff`, and wrap the
pre-first-token wait in `ui.status(...)`. Keep all writes going through the
injectable `out` so tests can capture plain text (construct `Ui` with
`color=False, spinner=False` in tests).

### 3.3 Deferred (documented)
**Streaming markdown rendering is out of scope.** Markdown can't be rendered
until a block completes, which forces buffering that fights the token-latency
requirement. Note this in `forge/ui.py`'s module docstring so a future phase
picks it up deliberately.

### 3.4 Tests
`tests/test_ui.py`: with `color=False`/non-TTY, output equals the current
plain-text format (regression against `test_repl_rendering.py`); diff rendering
falls back to raw diff text when `rich` is unavailable (monkeypatch `_RICH`
False).

---

## 4. Feature L — Parallel read-only tool execution

### 4.1 `AgentLoop._execute_tool_calls` (`forge/agent.py`)
Today calls run strictly serially. Add concurrency **only** for a leading group
of read-only calls, preserving semantics:

Rules (all mandatory):
- **Eligibility:** a call is parallel-eligible iff the resolved tool's
  `read_only` is True **and** `tool_executor.is_exposed(name)` **and** the tool
  is one of the safe set (`read`, `search`, `search_memory`, `repo_index`) — do
  **not** parallelize `git` (even read ops shell out), `planning` (mutates
  `ctx.state`), or MCP tools (unknown side effects). Provide the safe set as a
  constant; MCP/git excluded even though `git status` is logically read-only.
- **Grouping:** only parallelize a *contiguous leading run* of eligible calls at
  the front of the batch; the moment a non-eligible call appears, run it and the
  rest serially. (Keeps ordering reasoning simple and safe.)
- **Ordering:** append exactly one `Tool_Result` per call in **received order**,
  regardless of completion order — the Vertex layer (`_to_sdk_contents`) requires
  function-response part count/order to match the call turn. Collect results into
  a list indexed by position, then append in order.
- **Interrupt:** poll `interrupt.check()` before dispatch; eligible tools are
  fast/bounded (`read`/`search`) so a thread pool is acceptable, but you cannot
  force-kill threads — dispatch the group, then as results arrive check the
  interrupt between them; if tripped, stop appending further results and return
  `interrupted=True` (results already computed for earlier positions are kept,
  matching today's "retain completed results" contract).
- **Executor reuse:** call `self.tool_executor.execute(call)` from worker threads.
  `ToolExecutor.execute` is stateless per call except the shared `ToolContext`;
  the eligible tools (`read`/`search`/…) do not mutate `ctx.state`, so concurrent
  execution is safe. Confirm no eligible tool writes to `ctx.state`.
- Use `concurrent.futures.ThreadPoolExecutor(max_workers=config.parallel_max_workers)`.

Sketch:
```python
def _execute_tool_calls(self, session, calls):
    mutated = False
    i = 0
    # leading eligible run
    lead = []
    while i < len(calls) and self._parallel_eligible(calls[i]):
        lead.append(calls[i]); i += 1
    if len(lead) > 1 and self._parallel_enabled:
        results = self._run_parallel(lead)  # ordered list, interrupt-aware
    else:
        results = [self._run_one(c) for c in lead]
    # append lead results in order (+ mutated/interrupt handling), then run the
    # remaining calls[i:] exactly as today (serial).
    ...
```

Keep the existing `_sync_todos`, `mutated_files` (`write`/`edit`), and
per-call interrupt-before-dispatch behavior intact.

### 4.2 Tests
`tests/test_parallel_tools.py`:
- Two `read` calls run concurrently (inject a fake executor whose `execute`
  sleeps; assert wall-clock < serial sum) — keep timing generous to avoid flake.
- **Ordering:** results are appended in received order even when the second call
  finishes first (fake executor with staggered delays).
- A batch of `[read, write, read]` runs the `write` (non-eligible) serially and
  does not parallelize across it.
- Interrupt tripped mid-group → `interrupted=True`, earlier results retained.
- `git`/`planning`/MCP calls are never parallelized (assert eligibility helper).
- Default (`parallel_enabled=False`) → identical behavior to today.

---

## 5. Config schema (`forge/config.py`)

```toml
[ui]
color = true
spinner = true

[commands]
dir = ".forge/commands"

[parallel]
enabled = false          # opt-in; default preserves serial behavior
max_workers = 4

[mentions]
enabled = true
```
- `Config` fields with back-compat defaults: `ui_color: bool = False`,
  `ui_spinner: bool = False`, `commands_dir: str = ".forge/commands"`,
  `parallel_enabled: bool = False`, `parallel_max_workers: int = 4`,
  `mentions_enabled: bool = False`.
  (Defaults OFF so existing behavior/tests are unchanged; `write_default` opts
  new configs into `color=true`, `spinner=true`, `mentions_enabled=true`.)
- Parse the tables; `write_default` emits them.

---

## 6. Wiring (`forge/app.py`) & test plan

- Build `Ui(out, color=config.ui_color, spinner=config.ui_spinner)`,
  `SlashCommandStore([root / config.commands_dir])`, and pass them + the mention
  settings into `Repl`.
- Pass `parallel_enabled`/`max_workers` into `AgentLoop`.
- Run `python -m pytest -q --basetemp=C:\forge_tmp` (outside repo), remove temp.
- New tests: `test_mentions.py`, `test_slash_commands.py`,
  `test_repl_commands.py`, `test_ui.py`, `test_parallel_tools.py`. Update
  `test_config_*` for the new tables.
- **Regression bar:** with all Phase 4 toggles OFF (dataclass defaults), the
  REPL, agent loop, and output are identical to Phase 3.

---

## 7. Definition of Done
- [ ] `expand_mentions` (property-tested, workspace-scoped, size-capped).
- [ ] `SlashCommandStore` (markdown templates, `$ARGUMENTS`/`$N`); REPL routes
      `/help`, custom commands, `@file`; classification order preserved; unknown
      commands never hit the model.
- [ ] `forge/ui.py` optional `rich` layer with plain-text/non-TTY fallback;
      color + diff highlight + pre-token spinner; streaming markdown deferred
      (documented).
- [ ] Parallel read-only execution for the safe set only; received-order
      results; interrupt-aware; git/planning/MCP excluded; default OFF.
- [ ] `[ui]`/`[commands]`/`[parallel]`/`[mentions]` config; defaults OFF, opted
      on in `write_default`.
- [ ] Full suite green; temp base removed.

## 8. Out of scope (Phase 5)
- Multi-provider models, subagents.
- Streaming markdown rendering.
- Executable slash commands (they only produce prompt text here).
```
