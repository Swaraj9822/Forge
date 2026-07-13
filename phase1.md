# Forge — Phase 1 Implementation Plan

**Audience:** a coding agent implementing this end-to-end.
**Scope:** Phase 1 only. Do **not** implement anything from later phases (approval
system, diff/checkpoint, memory store, repo map, multi-provider, etc.). Where
this document mentions later phases, it is only to keep the seams clean.

Phase 1 delivers three features plus one enabler:

- **Enabler — Context-provider seam:** a clean way to inject ephemeral,
  token-budgeted, query-conditioned segments into the assembled context window
  each turn, without persisting them to `session.messages`.
- **D — Plan-reminder injection:** re-emit the current todo list into context
  every turn so long tasks don't drift after compaction.
- **E — Project memory file auto-load:** automatically include a
  `FORGE.md`/`AGENTS.md` found in the workspace as system context.
- **A — Non-interactive (headless) mode:** `forge -p "prompt"` with
  `--output text|json` for CI/scripting.

---

## 0. Ground rules (read first)

1. **Match existing style.** Small, dependency-light modules; typed dataclasses;
   `from __future__ import annotations`; docstrings that explain *why*. Look at
   `forge/context.py`, `forge/config.py`, `forge/verification.py` as templates.
2. **No new heavy dependencies.** Everything in Phase 1 uses the stdlib
   (`argparse`, `json`, `difflib` not needed here, `pathlib`, `warnings`).
   Do not add `rich`, `tree-sitter`, or any provider SDKs.
3. **Backward compatibility is mandatory.** All new parameters are optional with
   defaults that reproduce today's behavior exactly. The existing test suite
   (237 tests) must still pass unchanged except where this document explicitly
   says to update a test.
4. **Test-first, property-style where natural.** The repo uses `pytest` +
   `hypothesis`. Add unit tests and, where a pure function has an invariant, a
   property test. Mirror the existing `tests/test_*.py` naming.
5. **Determinism.** Context assembly must stay offline and deterministic (no
   network in `ContextManager`). The plan reminder and project-file loading are
   pure/local.
6. **Run the suite** with a writable temp base on Windows:
   `python -m pytest -q --basetemp=_tmptest` then delete `_tmptest`.
7. **Definition of done** is in section 8. Tick every box.

---

## 1. Architecture overview of the changes

Files touched:

| File | Change |
|------|--------|
| `forge/context.py` | Add `ContextProvider` protocol; add `providers`, `workspace_root`, `project_memory_filenames` params to `ContextManager`; wire ephemeral segments into `assemble`; add project-memory-file discovery to `_system_segments`; add `limit` param to `compact`. |
| `forge/context_providers.py` (new) | `PlanReminderProvider` implementing `ContextProvider`. |
| `forge/config.py` | Add `[context]` config table → `Config.plan_reminder` and `Config.project_memory`; write them in `write_default`. |
| `forge/headless.py` (new) | Pure-ish `run_headless(...)` that runs one prompt and renders/serializes the result; renderers for streaming/capturing. |
| `forge/app.py` | Wire providers + workspace_root into `ContextManager`; add `run_prompt(...)`. |
| `forge/verification.py` | Add public `set_renderer` to `VerificationCoordinator`. |
| `forge/__main__.py` | Add `-p/--prompt` and `--output` options; dispatch to headless. |
| `tests/…` | New tests per feature; minor updates to config tests. |

Two seams introduced:

- **Context-provider seam** (`context.py`): used by D now; Phase 3 (memory) and
  the repo map will reuse it.
- **Headless run path** (`headless.py` + `app.run_prompt`): Phase 2's approval
  policy will later inject a policy here; for Phase 1 headless runs in plain
  autopilot (tools execute without prompting). This is a known, documented
  interim state — see section 7.

---

## 2. Enabler + Feature D — Context-provider seam & plan reminder

### 2.1 `ContextProvider` protocol (in `forge/context.py`)

Add near the top-level types (after the imports, before `ContextManager`):

```python
from typing import Protocol, runtime_checkable  # add to existing typing imports

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
```

### 2.2 `ContextManager.__init__` — new optional params

Current signature:
```python
def __init__(self, config, summarizer=None):
```
New signature (all additions optional, defaults preserve behavior):
```python
def __init__(
    self,
    config: Config,
    summarizer=None,
    providers: "list[ContextProvider] | None" = None,
    workspace_root: "Path | None" = None,
    project_memory_filenames: "tuple[str, ...]" = (),
) -> None:
    self.config = config
    self.summarizer = summarizer
    self.providers = list(providers) if providers else []
    self.workspace_root = workspace_root
    self.project_memory_filenames = tuple(project_memory_filenames)
```

Add a module constant near the other constants:
```python
# Per-project instruction files auto-loaded as system context (Feature E).
DEFAULT_PROJECT_MEMORY_FILES: tuple[str, ...] = ("FORGE.md", "AGENTS.md")
```

### 2.3 `compact` — accept an effective limit

So provider segments can reserve budget. Change the signature and the internal
limit source only:

```python
def compact(self, session: Session, *, limit: int | None = None) -> CompactionResult:
    ...
    effective_limit = self.config.token_limit if limit is None else limit
    # Replace every internal use of `self.config.token_limit` / `limit` in this
    # method body with `effective_limit`.
```
Do a careful find/replace **inside `compact` only** (there is a local `limit =
self.config.token_limit` today — rename that local to `effective_limit` and honor
the new parameter). Do not change `estimate_tokens` or `assemble_system_messages`.

### 2.4 `assemble` — reserve budget, compact, then append ephemeral segments

Replace the body of `assemble` with this logic (keep the docstring, extend it):

```python
def assemble(self, session: Session) -> tuple[list[dict], CompactionInfo | None]:
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
```

**Critical invariant:** when `self.providers` is empty, `ephemeral == []`,
`reserve == 0`, and the return value is identical to today. Add a test asserting
this (section 2.7).

### 2.5 `PlanReminderProvider` (new file `forge/context_providers.py`)

```python
"""Context providers that inject ephemeral, query-conditioned segments.

Providers implement :class:`forge.context.ContextProvider`. Their output is
appended to the assembled context window for a single turn and is never
persisted to the session. This module currently hosts the plan-reminder
provider (Feature D); later phases add a memory provider and a repo-map provider
here.
"""

from __future__ import annotations

from forge.session import Session

# Status glyphs kept consistent with the REPL's rendering.
_STATUS_LABEL = {
    "pending": "pending",
    "in_progress": "in progress",
    "completed": "done",
}


class PlanReminderProvider:
    """Re-emit the current todo list so the plan survives compaction (Req: D).

    Returns a single ephemeral ``user`` message summarizing ``session.todos`` so
    the model keeps the active plan in view on long tasks. Returns an empty list
    when there are no todos, so turns without a plan are unaffected.
    """

    HEADER = "[current plan — keep this in mind; do not restate it back to the user]"

    def segments(self, session: Session) -> list[dict]:
        todos = session.todos
        if not todos:
            return []
        lines = [self.HEADER]
        for t in todos:
            label = _STATUS_LABEL.get(t.status, t.status)
            lines.append(f"- ({label}) {t.text}")
        return [{"role": "user", "content": "\n".join(lines)}]
```

### 2.6 Project memory file discovery (Feature E, in `context.py`)

Extend `_system_segments` so the ordering is: **built-in default prompt →
steering files → project memory file**. Append at the end of `_system_segments`,
just before `return segments`:

```python
        # Feature E: auto-load a per-project instruction file from the workspace.
        segments.extend(self._project_memory_segments())
        return segments
```

Add the helper:

```python
def _project_memory_segments(self) -> list[str]:
    """Return the contents of the first existing project memory file, or [].

    Looks in ``self.workspace_root`` (only) for the configured filenames in
    priority order (default: FORGE.md then AGENTS.md). Discovery is disabled
    when ``workspace_root`` is None or no filenames are configured, which
    reproduces the pre-feature behavior. A file that exists but cannot be read
    or is not valid UTF-8 is warned about and skipped.
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
```

Notes:
- **Workspace-root only** (no ancestor walk) in Phase 1 — keeps within Forge's
  workspace boundary and stays deterministic. Do not read parent directories.
- First filename that exists wins; `FORGE.md` takes precedence over `AGENTS.md`.
- No size cap in Phase 1 (consistent with steering files). Do not add one.

### 2.7 Tests for the seam, plan reminder, and project file

Create `tests/test_context_providers.py`:

- `test_no_providers_window_is_unchanged`: build a `ContextManager` with and
  without an empty providers list; assert `assemble(session)` returns identical
  windows for a session with messages and no todos. (Guards the invariant.)
- `test_plan_reminder_appended_when_todos_present`: session with 2 todos;
  assert the last message in the assembled window is the plan-reminder `user`
  message and contains each todo's text and status label.
- `test_plan_reminder_absent_when_no_todos`: session with empty todos → no
  reminder message; window identical to no-provider case.
- `test_plan_reminder_not_persisted`: after `assemble`, assert `session.messages`
  does **not** contain the reminder (it is ephemeral).
- `test_plan_reminder_survives_compaction`: construct a config with a tiny
  `token_limit` so `assemble` compacts; assert the reminder is still the final
  message of the returned window and `CompactionInfo.occurred is True`.
- `test_budget_is_reserved_for_ephemeral_segments`: with a provider that returns
  a large segment, assert compaction is triggered at a lower effective limit
  (i.e. `assemble` compacts even though `base` alone is under `token_limit`).

Create `tests/test_project_memory.py`:

- `test_project_file_loaded_after_steering`: write a `FORGE.md` in a temp
  workspace; construct `ContextManager(config, workspace_root=tmp,
  project_memory_filenames=("FORGE.md","AGENTS.md"))`; assert
  `build_system_prompt()` ends with the file's contents and that a configured
  steering file (if any) precedes it.
- `test_forge_md_precedence_over_agents_md`: create both files; assert `FORGE.md`
  content is used and `AGENTS.md` is ignored.
- `test_no_project_file_is_noop`: empty workspace → `build_system_prompt()`
  equals the default-prompt-only result (identical to `workspace_root=None`).
- `test_unreadable_project_file_warns_and_skips`: create `FORGE.md` with invalid
  UTF-8 bytes; assert a warning is emitted and the segment is skipped.

Also confirm existing `tests/test_steering_order.py` and `tests/test_compaction_*`
still pass unchanged (they construct `ContextManager` without the new params).

---

## 3. Config changes (`forge/config.py`)

Add a `[context]` table with two booleans, both defaulting to `True`.

### 3.1 `Config` dataclass

Add two fields (frozen dataclass — append after the existing scalar fields,
before the collection fields is fine; keep defaults):

```python
    plan_reminder: bool = True
    project_memory: bool = True
```

### 3.2 Parse in `_from_raw`

```python
        context_raw = raw.get("context") or {}
        plan_reminder = bool(context_raw.get("plan_reminder", True))
        project_memory = bool(context_raw.get("project_memory", True))
```
and pass them into the `Config(...)` constructor:
```python
        return Config(
            ...,
            plan_reminder=plan_reminder,
            project_memory=project_memory,
            **merged_limits,
        )
```

### 3.3 `write_default`

Add a `[context]` table to the emitted document:
```python
        document["context"] = {"plan_reminder": True, "project_memory": True}
```
Place it near the other tables. Keep the file round-trippable through `load`.

### 3.4 Config test updates

- `tests/test_config_defaults.py`: assert `Config().plan_reminder is True` and
  `Config().project_memory is True`; assert loading a file with
  `[context]\nplan_reminder = false` yields `plan_reminder is False`.
- `tests/test_config_init.py`: the `write_default` output now includes a
  `[context]` table; update any exact-shape assertion and confirm the
  written file still round-trips to `Config(plan_reminder=True,
  project_memory=True, ...)`.
- Add `tests/test_config_context_table.py` if you prefer isolation: unknown
  keys under `[context]` are ignored; non-bool values are coerced via `bool()`.

---

## 4. Feature A — Non-interactive (headless) mode

### 4.1 CLI surface (`forge/__main__.py`)

Add two **top-level** options to `build_parser()` (they apply to the
no-subcommand invocation):

```python
    parser.add_argument(
        "-p", "--prompt",
        help="Run a single prompt non-interactively and exit. Use '-' to read "
             "the prompt from stdin.",
    )
    parser.add_argument(
        "--output", choices=["text", "json"], default="text",
        help="Output format for non-interactive runs (default: text).",
    )
```

In `main(...)`, before the "no subcommand → REPL" branch, add:

```python
    if args.command is None and args.prompt is not None:
        prompt = _read_prompt_arg(args.prompt)
        return _run_headless(prompt, output=args.output, out=out, err=err)
```

Helpers in `__main__.py`:

```python
def _read_prompt_arg(value: str) -> str:
    """Return the prompt text; '-' means read all of stdin."""
    if value == "-":
        return sys.stdin.read()
    return value


def _run_headless(prompt: str, *, output: str, out: TextIO, err: TextIO) -> int:
    """Route into app.run_prompt, handling the fatal startup errors like _run_repl."""
    from forge.app import run_prompt  # local import mirrors existing style
    try:
        return run_prompt(prompt, output=output, out=out, err=err)
    except ConfigError as exc:
        print(str(exc), file=err)
        return 1
    except StartupError as exc:  # defensive; run_prompt normally handles it
        print(exc.message, file=err)
        return exc.exit_code
```

Blank prompt handling: if `prompt.strip() == ""`, print an error to `err` and
return exit code 1 (do not start a turn).

### 4.2 `VerificationCoordinator.set_renderer` (`forge/verification.py`)

Add a tiny public setter so the headless path can control verification output:

```python
    def set_renderer(self, renderer: "VerificationRenderer | None") -> None:
        """Replace the progress renderer (used by the headless run path)."""
        self._renderer = renderer
```

### 4.3 `forge/headless.py` (new) — the testable core

Design goal: keep the "run one prompt and produce a result" logic **decoupled
from bootstrap**, exactly like the agent tests drive `AgentLoop` with fakes.
`run_headless` takes already-wired collaborators.

```python
"""Non-interactive (headless) execution for `forge -p`.

Runs a single prompt to completion (the agent loop already iterates until the
model emits no tool calls), optionally runs the post-turn verification phase,
and renders the result as plain text or a single JSON object. Kept independent
of bootstrap so it is unit-testable with fake collaborators (mirroring the
agent-loop tests).
"""

from __future__ import annotations

import json
from typing import TextIO

from forge.agent import AgentLoop, NullRenderer, TurnResult
from forge.session import Session
from forge.usage import UsageSummary

# Exit codes (documented contract for CI).
EXIT_OK = 0
EXIT_TURN_ERROR = 2
EXIT_INTERRUPTED = 3
EXIT_VERIFICATION_FAILED = 4


class _CapturingRenderer:
    """Accumulates streamed model text; optionally echoes it to a stream."""

    def __init__(self, echo: TextIO | None = None) -> None:
        self._parts: list[str] = []
        self._echo = echo

    def on_text(self, text: str) -> None:
        self._parts.append(text)
        if self._echo is not None:
            self._echo.write(text)
            self._echo.flush()

    def on_tool(self, name: str) -> None:
        if self._echo is not None:
            self._echo.write(f"\n[tool: {name}]\n")
            self._echo.flush()

    def on_compaction(self, info) -> None:  # noqa: ANN001 - matches Renderer
        return None

    @property
    def text(self) -> str:
        return "".join(self._parts)


def run_headless(
    agent_loop: AgentLoop,
    session: Session,
    verification_coordinator,
    prompt: str,
    *,
    output: str = "text",
    out: TextIO,
) -> int:
    """Run one prompt to completion and render/serialize the result.

    In ``text`` mode the model's response streams to ``out`` live; in ``json``
    mode nothing is written until a single JSON object is emitted at the end.
    Verification (when configured and gated in) runs after the turn; its
    correction-turn text is NOT included in the reported ``response``.
    """
    # Renderer: echo live only in text mode.
    renderer = _CapturingRenderer(echo=out if output == "text" else None)
    agent_loop.renderer = renderer
    # Silence verification progress in json mode; allow it in text mode.
    if verification_coordinator is not None:
        verification_coordinator.set_renderer(
            _VerifyText(out) if output == "text" else None
        )

    result: TurnResult = agent_loop.run_turn(session, prompt)
    response_text = renderer.text  # snapshot BEFORE verification correction turns

    # Swap to a null renderer so correction turns don't pollute the response.
    agent_loop.renderer = NullRenderer()

    phase = None
    turn_ok = not (result.interrupted or result.error)
    if verification_coordinator is not None and turn_ok:
        phase = verification_coordinator.run(session, result)

    usage = phase.usage if (phase is not None and phase.ran) else result.usage
    code = _exit_code(result, phase)

    if output == "json":
        _emit_json(out, session, result, phase, usage, response_text, code)
    else:
        _emit_text_footer(out, result, phase, usage)
    return code
```

Add the small helpers referenced above:

```python
class _VerifyText:
    """Minimal verification renderer that prints [verify] lines (text mode)."""
    def __init__(self, out: TextIO) -> None:
        self._out = out
    def on_verification_start(self, command: str) -> None:
        self._out.write(f"\n[verify] running: {command}\n"); self._out.flush()
    def on_verification_result(self, result) -> None:  # noqa: ANN001
        status = "passed" if result.outcome == "passed" else f"failed ({result.outcome})"
        self._out.write(f"[verify] {status}\n"); self._out.flush()
    def on_correction_iteration(self, iteration: int, max_iterations: int) -> None:
        self._out.write(f"[verify] correction {iteration}/{max_iterations}\n"); self._out.flush()
    def on_verification_cap_reached(self, result, iterations: int) -> None:  # noqa: ANN001
        self._out.write(f"[verify] cap reached ({iterations}); final: {result.outcome}\n")
        self._out.flush()


def _exit_code(result: TurnResult, phase) -> int:
    if result.error:
        return EXIT_TURN_ERROR
    if result.interrupted:
        return EXIT_INTERRUPTED
    if phase is not None and phase.ran and phase.final_result is not None:
        if phase.final_result.outcome != "passed":
            return EXIT_VERIFICATION_FAILED
    return EXIT_OK


def _usage_dict(u: UsageSummary) -> dict:
    return {
        "turn_input_tokens": u.turn_input_tokens,
        "turn_output_tokens": u.turn_output_tokens,
        "cumulative_input_tokens": u.cumulative_input_tokens,
        "cumulative_output_tokens": u.cumulative_output_tokens,
        "turn_cost": u.turn_cost,
        "cumulative_cost": u.cumulative_cost,
        "cost_available": u.cost_available,
    }


def _emit_json(out, session, result, phase, usage, response_text, code) -> None:
    verification = None
    if phase is not None and phase.ran:
        fr = phase.final_result
        verification = {
            "ran": True,
            "outcome": (fr.outcome if fr is not None else None),
            "iterations": phase.iterations_performed,
            "cap_reached": phase.cap_reached,
        }
    payload = {
        "session_id": session.id,
        "ok": code == EXIT_OK,
        "response": response_text,
        "error": result.error,
        "interrupted": result.interrupted,
        "mutated_files": result.mutated_files,
        "usage": _usage_dict(usage),
        "verification": verification,
        "todos": [{"id": t.id, "text": t.text, "status": t.status} for t in session.todos],
    }
    out.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    out.flush()


def _emit_text_footer(out, result, phase, usage) -> None:
    out.write("\n")
    if result.error:
        out.write(f"[error] {result.error}\n")
    elif result.interrupted:
        out.write("[interrupted]\n")
    # Reuse the same usage wording as the REPL for consistency.
    tok = (f"turn: {usage.turn_input_tokens} in / {usage.turn_output_tokens} out | "
           f"session: {usage.cumulative_input_tokens} in / "
           f"{usage.cumulative_output_tokens} out")
    cost = (f"cost: ${usage.turn_cost:.6f} turn / ${usage.cumulative_cost:.6f} session"
            if usage.cost_available else "cost unavailable")
    out.write(f"[usage] {tok} | {cost}\n")
    out.flush()
```

**Notes / gotchas:**
- After `bootstrap`, `agent_loop.renderer` is the `Repl` (the Repl wires itself
  in `__init__`). `run_headless` deliberately overrides it. This is why the
  headless core takes `agent_loop` and sets the renderer itself.
- `run_turn` already persists the session via `session_store`, so headless runs
  are saved and resumable.
- Warnings go to stderr by default, so they never corrupt JSON on stdout. Do not
  redirect warnings to `out`.

### 4.4 `app.run_prompt` (`forge/app.py`)

Add a wiring entry point that bootstraps then delegates to `run_headless`:

```python
def run_prompt(
    prompt: str,
    *,
    output: str = "text",
    config_path: Path | None = None,
    workspace_root: Path | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    """Bootstrap and run a single prompt non-interactively (Feature A).

    Returns the headless exit code. A StartupError (missing ADC / required
    config) is printed to stderr and its exit code returned, mirroring `main`.
    """
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    try:
        app = bootstrap(
            config_path=config_path,
            workspace_root=workspace_root,
        )
    except StartupError as exc:
        print(exc.message, file=err)
        return exc.exit_code

    from forge.headless import run_headless
    try:
        app.interrupt.install()
        return run_headless(
            app.agent_loop,
            app.session,
            app.verification_coordinator,
            prompt,
            output=output,
            out=out,
        )
    finally:
        app.close()
```

`app.close()` already tears down MCP + the SIGINT handler; reuse it.

### 4.5 Headless tests

Create `tests/test_headless.py` (drive with fakes, like `tests/test_agent_*`):

- Build a fake `AgentLoop` whose `run_turn` returns a scripted `TurnResult`
  (with a known `usage`) and appends a model message; or use the real
  `AgentLoop` with a fake `VertexClient` + fake tool executor as the agent tests
  do. Prefer the same fakes those tests already use.
- `test_text_mode_streams_and_prints_usage`: capturing `StringIO` as `out`;
  assert the streamed text and a `[usage]` footer are present; exit code 0.
- `test_json_mode_emits_single_object`: `out` is `StringIO`; assert the entire
  output parses as one JSON object with the documented keys; `response` equals
  the streamed text; no `[usage]` prose present.
- `test_json_mode_not_polluted_by_streaming`: assert stdout has no streamed
  fragments before the JSON (json mode uses the non-echoing renderer).
- `test_exit_code_on_turn_error`: fake TurnResult with `error="boom"` → exit
  code 2 and `error` populated in JSON.
- `test_exit_code_on_interrupt`: `interrupted=True` → exit code 3.
- `test_verification_failure_exit_code`: fake coordinator returning a phase with
  `ran=True`, `final_result.outcome="failed"` → exit code 4; `verification`
  object populated; correction-turn text excluded from `response`.
- `test_response_snapshot_excludes_correction_turns`: verify `response` is the
  first turn's text only.

CLI-level tests in `tests/test_cli_dispatch.py` (extend it):
- `main(["-p", "hello"], out=..., err=...)` dispatches to `run_prompt`
  (monkeypatch `forge.app.run_prompt` to a spy asserting args) and returns its
  code.
- `main(["-p", "-"], ...)` reads stdin (monkeypatch `sys.stdin`).
- Blank prompt (`-p "   "`) → exit code 1 with an error on `err`, no dispatch.
- `--output json` is threaded through to `run_prompt`.

---

## 5. Wiring changes (`forge/app.py`, `bootstrap`)

In `bootstrap`, at the `ContextManager` construction step (currently
`context_manager = ContextManager(config, summarizer=vertex_client)`), replace
with provider + workspace wiring:

```python
    from forge.context_providers import PlanReminderProvider
    from forge.context import DEFAULT_PROJECT_MEMORY_FILES

    providers = []
    if config.plan_reminder:
        providers.append(PlanReminderProvider())

    context_manager = ContextManager(
        config,
        summarizer=vertex_client,
        providers=providers,
        workspace_root=root,
        project_memory_filenames=(
            DEFAULT_PROJECT_MEMORY_FILES if config.project_memory else ()
        ),
    )
```

`root` is already computed earlier in `bootstrap`
(`root = Path(workspace_root) if workspace_root is not None else Path.cwd()`).
Ensure the `ContextManager` construction happens **after** `root` is defined
(move the two lines if needed — `root` is currently computed at the tool-executor
step; hoist it above the `ContextManager` construction, or reorder so `root` is
available). Keep everything else in the wiring order unchanged.

No other bootstrap changes. The Repl and verification coordinator continue to be
wired exactly as today for the interactive path.

---

## 6. Full test plan & commands

Run the whole suite with a writable temp base (Windows temp perms are flaky):

```
python -m pytest -q --basetemp=_tmptest
```
Then remove the temp dir:
```
Remove-Item -Recurse -Force _tmptest
```

New test files:
- `tests/test_context_providers.py` (section 2.7)
- `tests/test_project_memory.py` (section 2.7)
- `tests/test_headless.py` (section 4.5)
- optional `tests/test_config_context_table.py` (section 3.4)

Updated test files:
- `tests/test_config_defaults.py`, `tests/test_config_init.py` (section 3.4)
- `tests/test_cli_dispatch.py` (section 4.5)

Regression bar: **all previously-passing tests still pass.** If any existing
context/steering/compaction test changes output, that is a bug in the seam —
the empty-provider / `workspace_root=None` paths must be byte-identical to
today. Fix the seam, not the test (except the documented config-test updates).

---

## 7. Documented interim states (do NOT "fix" these in Phase 1)

- **Headless runs in autopilot.** With no approval system yet (Phase 2), a
  headless run executes shell/write/edit tools without prompting. This is
  intentional for Phase 1. Add a one-line note to `README.md` under a new
  "Non-interactive mode" heading warning that `-p` runs tools unattended, and
  that an approval policy arrives in a later phase. Do not build the policy now.
- **No `--session` resume in headless.** Each `forge -p` mints a fresh session
  (via the normal `bootstrap` path). Threading `--session <id>` into headless is
  out of scope; do not add it.
- **Ancestor search for project files is out of scope.** Workspace-root only.

---

## 8. Definition of Done (checklist)

- [ ] `ContextProvider` protocol added to `forge/context.py`.
- [ ] `ContextManager` accepts `providers`, `workspace_root`,
      `project_memory_filenames`; empty/None paths reproduce current behavior
      byte-for-byte.
- [ ] `compact` honors an optional `limit`; `assemble` reserves budget for
      ephemeral segments, compacts against the effective limit, and appends
      ephemeral segments last.
- [ ] `PlanReminderProvider` implemented in `forge/context_providers.py`;
      returns `[]` for empty todos; never persists to `session.messages`;
      survives compaction.
- [ ] Project memory file (`FORGE.md` → `AGENTS.md`) auto-loaded from the
      workspace root, ordered after steering files; unreadable file warns+skips.
- [ ] `Config.plan_reminder` and `Config.project_memory` added, parsed from
      `[context]`, written by `write_default`, round-trip verified.
- [ ] `forge -p/--prompt` and `--output text|json` implemented; `-` reads stdin;
      blank prompt → exit 1.
- [ ] `forge/headless.py` implemented with the documented JSON schema and exit
      codes (0 ok / 2 turn error / 3 interrupted / 4 verification failed).
- [ ] `app.run_prompt` bootstraps + delegates + tears down via `app.close()`.
- [ ] `VerificationCoordinator.set_renderer` added; json mode silences verify
      output; text mode streams response + prints usage footer.
- [ ] Bootstrap wires providers + workspace root into `ContextManager`.
- [ ] All new tests written; full suite green via
      `pytest --basetemp=_tmptest`; `_tmptest` removed.
- [ ] `README.md` updated with the "Non-interactive mode" note.

---

## 9. Manual verification (smoke)

After implementation, with a configured `config.toml` (real ADC), verify:

```
# JSON output, single object, parseable:
forge -p "list the python files in this repo" --output json

# stdin prompt:
echo "summarize forge/agent.py" | forge -p -

# Plan reminder: start an interactive session, have the model create todos,
# then run enough turns to trigger compaction; confirm the plan stays in view.

# Project file: add a FORGE.md with a distinctive instruction and confirm the
# model honors it in a fresh headless run.
```

For CI, assert on the exit code and (in json mode) parse stdout:
```
forge -p "run the tests and fix failures" --output json ; echo "exit=$LASTEXITCODE"
```

---

## 10. JSON output schema (reference)

```json
{
  "session_id": "uuid",
  "ok": true,
  "response": "final assistant text for the turn",
  "error": null,
  "interrupted": false,
  "mutated_files": false,
  "usage": {
    "turn_input_tokens": 0,
    "turn_output_tokens": 0,
    "cumulative_input_tokens": 0,
    "cumulative_output_tokens": 0,
    "turn_cost": null,
    "cumulative_cost": null,
    "cost_available": false
  },
  "verification": null,
  "todos": [{"id": "1", "text": "…", "status": "pending"}]
}
```
`verification` is `null` when the phase did not run, else
`{"ran": true, "outcome": "passed|failed|timed_out|start_error", "iterations": N, "cap_reached": false}`.
