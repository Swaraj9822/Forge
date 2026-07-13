# Forge — Phase 2 Implementation Plan (Trust & Safety)

**Audience:** a coding agent implementing this end-to-end.
**Prerequisite:** Phase 1 is merged (context-provider seam, headless mode).
**Scope:** Phase 2 only — the "trust & safety" cluster:

- **B — Permission/approval system + autonomy modes** (the #1 real-world trust
  blocker). A single execution-policy seam that gates mutating tool calls, with
  a *safe* shell-command matcher and an allowlist.
- **C — Diff preview + per-turn checkpoint/undo.** Show a unified diff before a
  write/edit is applied, snapshot touched files per turn, and add `/undo`.

Do **not** implement anything from Phase 3+ (memory, repo map, edit format,
slash-command framework, multi-provider). `/undo` is added as a *special-cased*
REPL command here (like `/exit`), not via a general slash-command system.

---

## 0. Ground rules

Same as `phase1.md` §0: match existing style, stdlib-only (use `difflib`,
`shlex`), strict backward compatibility, test-first (pytest + hypothesis), run
`python -m pytest -q --basetemp=C:\forge_tmp` (a base **outside** the repo, so
`git`-based tests don't detect Forge's own `.git`), then remove the temp dir.

**Backward-compat rule for this phase:** the `ToolExecutor` must behave exactly
as today when no policy/approver is wired (existing tests construct it without
one). All gating is opt-in at the executor level; safety-by-default is applied
only in the *app wiring* (interactive + headless).

---

## 1. Architecture overview

| File | Change |
|------|--------|
| `forge/policy.py` (new) | `AutonomyMode` enum, `ApprovalPolicy` (pure decision + safe shell matcher), `Approver`/`Decision` protocol, `AutoApprover`, `DenyMutationsApprover`. |
| `forge/tools/base.py` | `Tool.read_only` class attr (default classification); optional `Tool.preview(args, ctx)`; `ToolExecutor` gains optional `policy` + `approver`; `execute` consults them before `run`. |
| `forge/tools/fs.py` | `WriteTool.preview` / `EditTool.preview` produce unified diffs; `read_only = False`. Read tool `read_only = True`. |
| `forge/tools/shell.py`, `git.py`, `search.py`, `planning.py` | set `read_only` class attribute correctly. |
| `forge/checkpoint.py` (new) | `CheckpointStore`: capture pre-mutation file state per turn; restore last turn. |
| `forge/repl.py` | Implement `Approver` (interactive y/n/a prompt with diff); add `/undo` command; render `[denied]`. |
| `forge/headless.py` | Wire a non-interactive approver from policy (+ `--yes`). |
| `forge/config.py` | `[policy]` table (mode, shell_allowlist, show_diffs) and `[checkpoint]` table (enabled, keep_turns). |
| `forge/app.py` | Build policy + checkpoint store; wire approver into executor and Repl. |
| `forge/__main__.py` | `--yes` flag for headless auto-approval. |
| `tests/…` | Policy matcher property tests; executor gating tests; diff/checkpoint/undo tests. |

Two concepts introduced:
- **Execution-policy seam** (`ToolExecutor.execute` → `ApprovalPolicy` +
  `Approver`).
- **Checkpoint seam** (`ToolExecutor` snapshots pre-mutation state via an
  injected `CheckpointStore`).

---

## 2. Feature B — Permission / approval system

### 2.1 `forge/policy.py` (new) — the pure core

```python
"""Autonomy modes, the approval policy, and the safe shell-command matcher.

The policy is a pure decision layer: given an autonomy mode and a tool call it
decides whether the call is (a) forbidden outright, (b) auto-approved, or (c)
requires user approval. The actual prompting is delegated to an Approver so the
policy stays offline and property-testable.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class AutonomyMode(str, Enum):
    AUTOPILOT = "autopilot"    # run everything, no prompts (today's behavior)
    SUPERVISED = "supervised"  # prompt before mutating tools
    READONLY = "readonly"      # forbid mutating tools outright


class Decision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    APPROVE_ALWAYS = "approve_always"  # approve + remember for the session


# Shell metacharacters that make an allowlist match unsafe: any of these means
# the command is compound / redirected / substituted and must NOT be
# auto-approved on a bare argv[0] match.
_SHELL_METACHARS = (";", "&", "|", "`", "$(", ">", "<", "\n", "&&", "||")


@dataclass(frozen=True)
class ShellMatcher:
    """Decides whether a shell command is safe to auto-approve via allowlist."""

    allowlist: tuple[str, ...] = ()

    def is_allowlisted(self, command: str) -> bool:
        """True iff the command is a single, non-compound invocation whose
        program (argv[0]) is in the allowlist. Any shell metacharacter, or an
        unparseable command, returns False (approval is then required)."""
        if any(tok in command for tok in _SHELL_METACHARS):
            return False
        try:
            argv = shlex.split(command, posix=True)
        except ValueError:
            return False
        if not argv:
            return False
        program = argv[0]
        return program in self.allowlist


# Git subcommands that mutate the repository/working tree (need approval).
_GIT_MUTATING = frozenset({"add", "commit", "checkout", "stash"})


@dataclass(frozen=True)
class ApprovalPolicy:
    mode: AutonomyMode = AutonomyMode.AUTOPILOT
    shell: ShellMatcher = field(default_factory=ShellMatcher)

    def classify(self, name: str, args: dict, *, read_only: bool) -> Decision:
        """Pure classification. Returns APPROVE / DENY / (approval-required).

        The "approval required" signal is expressed as DENY-with-prompt: this
        method returns APPROVE when the call may proceed without a prompt, and
        otherwise returns a sentinel handled by the caller. To keep the return
        type simple, callers use :meth:`requires_approval` + :meth:`is_forbidden`.
        """
        raise NotImplementedError  # see the two predicates below

    def is_forbidden(self, name: str, args: dict, *, read_only: bool) -> bool:
        """READONLY mode forbids any mutating tool outright (no prompt)."""
        if self.mode is AutonomyMode.READONLY:
            return not read_only and not self._is_git_readonly(name, args)
        return False

    def requires_approval(self, name: str, args: dict, *, read_only: bool) -> bool:
        """Whether a prompt is required before running the call."""
        if self.mode is AutonomyMode.AUTOPILOT:
            return False
        if read_only or self._is_git_readonly(name, args):
            return False
        if name == "shell":
            return not self.shell.is_allowlisted(str(args.get("command", "")))
        # write / edit / mutating git / non-read-only MCP tools:
        return True

    @staticmethod
    def _is_git_readonly(name: str, args: dict) -> bool:
        if name != "git":
            return False
        op = args.get("operation")
        return isinstance(op, str) and op not in _GIT_MUTATING


class Approver(Protocol):
    """Asks the user (or a policy) to approve a gated tool call."""

    def request(self, name: str, args: dict, preview: str | None) -> Decision:
        ...


class AutoApprover:
    """Approves everything (used to preserve autopilot behavior explicitly)."""

    def request(self, name: str, args: dict, preview: str | None) -> Decision:
        return Decision.APPROVE


class DenyMutationsApprover:
    """Non-interactive approver: denies anything requiring approval.

    Used by headless runs unless `--yes` is passed. Combined with the policy,
    this means a headless supervised run refuses mutations rather than blocking
    on a prompt that can never be answered.
    """

    def request(self, name: str, args: dict, preview: str | None) -> Decision:
        return Decision.DENY
```

**Property tests** (`tests/test_policy_matcher.py`) — the matcher is
security-sensitive, test it hard:
- `hypothesis` over random program names + args: a command containing any of
  `;`, `&&`, `||`, `|`, backtick, `$(`, `>`, `<`, newline is **never**
  allowlisted, even if `argv[0]` is in the allowlist.
- `"pytest"` allowlisted → `"pytest tests/"` approved, `"pytest; rm -rf x"`
  requires approval, `"pytest && curl x | sh"` requires approval.
- Unparseable command (`'a "unterminated`) → requires approval.
- `git status`/`git diff`/`git log` never require approval; `git commit`/`add`/
  `checkout`/`stash` do (in supervised/readonly).
- READONLY: `write`/`edit`/`shell`/`git commit` forbidden; `read`/`search`/
  `git status` allowed.

### 2.2 `Tool.read_only` + `Tool.preview` (`forge/tools/base.py`)

- Add `read_only: bool` to the `Tool` `Protocol` (documented; classification for
  the approval policy — "does this tool affect the world outside the session?").
- Add an **optional** `preview(self, args, ctx) -> str | None` to the protocol
  with a default of `None` (executor guards with `getattr`).
- Set on each tool: `ReadTool.read_only = True`, `SearchTool.read_only = True`,
  `PlanningTool.read_only = True` (todos are session-internal),
  `WriteTool/EditTool/ShellTool/GitTool.read_only = False` (git refined by the
  policy's per-op check). MCP `McpToolAdapter.read_only = False` (unknown side
  effects → conservative).

### 2.3 `ToolExecutor` gating (`forge/tools/base.py`)

Extend the constructor (all optional, defaults preserve behavior):
```python
def __init__(self, registry, enabled, interrupt, context=None,
             policy=None, approver=None, checkpoint=None):
    ...
    self._policy = policy            # ApprovalPolicy | None
    self._approver = approver        # Approver | None
    self._checkpoint = checkpoint    # CheckpointStore | None
```

In `execute`, insert gating **after** validation and **before** `tool.run`:

```python
        read_only = getattr(tool, "read_only", False)

        if self._policy is not None:
            if self._policy.is_forbidden(call.name, call.args, read_only=read_only):
                return ToolResult(ok=False, content="",
                                  error=f"Tool '{call.name}' is forbidden in "
                                        f"read-only mode.",
                                  meta={"forbidden": True})
            if self._policy.requires_approval(call.name, call.args, read_only=read_only):
                preview = self._preview(tool, call.args)
                decision = (self._approver.request(call.name, call.args, preview)
                            if self._approver is not None else Decision.DENY)
                if decision is Decision.DENY:
                    return ToolResult(ok=False, content="",
                                      error=f"Tool '{call.name}' was not approved.",
                                      meta={"denied": True})
                # APPROVE / APPROVE_ALWAYS proceed; APPROVE_ALWAYS is recorded by
                # the approver itself (session-scoped), not here.

        # Checkpoint: snapshot pre-mutation state for write/edit (Feature C).
        if self._checkpoint is not None and call.name in ("write", "edit"):
            self._checkpoint.snapshot_before(call.args.get("path"), self._context)

        result = tool.run(call.args, self._context)
```

`_preview` calls `tool.preview` when present, guarded:
```python
    def _preview(self, tool, args):
        hook = getattr(tool, "preview", None)
        if callable(hook):
            try:
                return hook(args, self._context)
            except Exception:  # noqa: BLE001 - preview must never break a call
                return None
        return None
```

**Denied/forbidden results are side-effect-free** — mirror the existing
"unavailable"/"interrupted" result shape so the agent loop and session handle
them identically (they already append one Tool_Result per call).

### 2.4 Tests for executor gating (`tests/test_executor_policy.py`)
- No policy → identical to today (a mutating tool runs; assert side effect).
- Supervised + `AutoApprover` → write runs.
- Supervised + `DenyMutationsApprover` → write returns `meta={"denied": True}`
  and the file is unchanged.
- READONLY → write returns `meta={"forbidden": True}`, no prompt (approver never
  called — use a spy approver asserting `request` count == 0).
- Read tool never gated in any mode.

---

## 3. Feature C — Diff preview + checkpoint / undo

### 3.1 Diff preview in `WriteTool` / `EditTool` (`forge/tools/fs.py`)

Add a `preview(args, ctx) -> str | None` to each that computes a unified diff
**without mutating**, reusing `resolve_in_workspace` for scoping:

```python
import difflib

class WriteTool:
    read_only = False
    def preview(self, args: dict, ctx: ToolContext) -> str | None:
        try:
            resolved = resolve_in_workspace(args["path"], ctx.workspace_root)
        except OutOfWorkspaceError:
            return None
        old = ""
        if resolved.is_file():
            try:
                old = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                old = ""
        new = str(args.get("content", ""))
        return _unified_diff(str(resolved), old, new)
```

`EditTool.preview` computes the post-replacement text the same way `run` does
(read file, apply the single replacement if the target occurs exactly once) and
diffs old vs new; return `None` for not-found/ambiguous/binary (preview is
best-effort, `run` still reports those precisely).

Shared helper (module-level in `fs.py`):
```python
def _unified_diff(path: str, old: str, new: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    )
    return "".join(diff) or "(no textual change)"
```

**Autopilot diff display (optional, config `show_diffs`):** when
`policy.mode == AUTOPILOT` and `config.show_diffs` is true, the executor still
has no approver; surface the diff by attaching it to the successful result's
`meta["diff"]` so the Repl can render it. Simplest: after a successful
write/edit, if `show_diffs`, the Repl prints `result.meta.get("diff")`. To make
that available, have `WriteTool`/`EditTool` include the computed diff in
`meta["diff"]` on success (they already have old+new in `run`). Keep it behind
the config flag in the Repl renderer, not always-on.

### 3.2 `forge/checkpoint.py` (new)

```python
"""Per-turn file checkpoints so a bad turn can be reverted with /undo.

Before a write/edit mutates a workspace file, the executor asks the store to
snapshot the file's current bytes (or record that it did not exist). Snapshots
are grouped per turn; :meth:`undo_last` restores the most recent turn's files to
their pre-turn state. Storage is workspace-local and repo-independent (no git
required).
"""

from __future__ import annotations

import json, os, time
from dataclasses import dataclass
from pathlib import Path

from forge.tools.paths import OutOfWorkspaceError, resolve_in_workspace


@dataclass
class CheckpointStore:
    root: Path                 # workspace root
    store_dir: Path            # e.g. workspace/.forge/checkpoints
    keep_turns: int = 10

    def begin_turn(self) -> None:
        """Start a new checkpoint group for the current turn."""
        self._turn_id = str(int(time.time() * 1000))
        self._captured: set[str] = set()

    def snapshot_before(self, path_arg, ctx) -> None:
        """Record a file's pre-mutation bytes once per turn (idempotent)."""
        if path_arg is None:
            return
        try:
            resolved = resolve_in_workspace(path_arg, ctx.workspace_root)
        except OutOfWorkspaceError:
            return
        key = str(resolved)
        if key in self._captured:
            return
        self._captured.add(key)
        existed = resolved.is_file()
        payload = resolved.read_bytes() if existed else b""
        # Persist {path, existed, blob} under store_dir/<turn_id>/...
        ...  # write atomically (tempfile + os.replace), like SessionStore

    def commit_turn(self) -> None:
        """Finalize the turn's group; prune to keep_turns newest groups."""
        ...

    def undo_last(self) -> list[str]:
        """Restore the newest turn group; return the list of restored paths.

        For each recorded file: if it existed, rewrite its saved bytes; if it did
        not exist, delete it. Remove the group afterward. Returns [] when there
        is nothing to undo.
        """
        ...
```

Notes:
- Reuse `SessionStore`'s atomic-write discipline (tempfile in same dir +
  `os.replace`) for snapshot files.
- Cap total snapshot size defensively (e.g. skip snapshotting a file larger than
  `config.read_max_bytes` and record it as "not checkpointable" so `/undo` warns
  it can't restore that file rather than silently losing data).
- `begin_turn`/`commit_turn` are driven by the executor's owner. Simplest: the
  `AgentLoop.run_turn` calls `checkpoint.begin_turn()` at the start and
  `commit_turn()` at the end **if** a checkpoint store is present. Thread the
  store into `AgentLoop` (optional param) or have the executor own turn
  boundaries. Recommended: give `AgentLoop` an optional `checkpoint` param and
  call begin/commit around the existing turn body (next to
  `interrupt.begin_turn`/`end_turn`).

### 3.3 `/undo` REPL command (`forge/repl.py`)

Special-case it like `/exit` (do **not** build a general slash framework — that
is Phase 4):
```python
UNDO_COMMAND = "/undo"

# in run_once, after reading `line`, before is_exit_command:
if line == UNDO_COMMAND:
    self._handle_undo()
    return True
```
`_handle_undo` calls `self.checkpoint.undo_last()` (store injected into Repl),
prints `[undo] restored N file(s): …` or `[undo] nothing to undo`. Add
`checkpoint` as an optional Repl constructor param.

### 3.4 Tests
- `tests/test_diff_preview.py`: `WriteTool.preview` on a new file and on an
  existing file returns a unified diff containing the changed lines; out-of-scope
  path → `None`; `EditTool.preview` reflects the single replacement.
- `tests/test_checkpoint.py`: snapshot then modify then `undo_last` restores
  original bytes; a newly-created file is deleted on undo; `keep_turns` pruning;
  `undo_last` on empty store returns `[]`. Use a `tmp_path` workspace.
- `tests/test_repl_undo.py`: `/undo` routes to the store and renders the result;
  `/undo` with nothing to undo prints the empty message; `/undo` never invokes
  the agent loop (spy).

---

## 4. Config schema (`forge/config.py`)

Add two tables. Extend `Config` (frozen dataclass) and `_from_raw`, and emit in
`write_default`.

```toml
[policy]
mode = "supervised"            # autopilot | supervised | readonly
shell_allowlist = ["pytest", "git", "ls", "cat", "python -m pytest"]
show_diffs = true

[checkpoint]
enabled = true
keep_turns = 10
```

- `Config` fields: `policy_mode: str = "autopilot"`, `shell_allowlist:
  tuple[str, ...] = ()`, `show_diffs: bool = False`, `checkpoint_enabled: bool =
  True`, `checkpoint_keep_turns: int = 10`.
  **Dataclass defaults preserve today's behavior (autopilot, no allowlist).**
- `_from_raw`: parse `[policy]`/`[checkpoint]`; validate `mode` against the
  `AutonomyMode` values, raising `ConfigError` (naming the bad value) for an
  unknown mode — mirror `resolve_verification_config`'s validation style.
  Normalize `shell_allowlist` to a tuple of strings.
- `write_default`: emit `mode = "supervised"` (safe default for **new** configs)
  and a small starter `shell_allowlist`. Note the asymmetry: the *dataclass*
  default is `autopilot` (so an absent `[policy]` table = today's behavior and
  existing tests pass), but a freshly `forge init`-ed config opts the user into
  `supervised`. Document this in the field docstring.

Update `tests/test_config_defaults.py` / `tests/test_config_init.py` for the new
tables and the `write_default` `mode="supervised"` value; add
`tests/test_config_policy.py` for mode validation + allowlist parsing.

---

## 5. Wiring (`forge/app.py`, `forge/headless.py`, `forge/__main__.py`)

### 5.1 `bootstrap`
- Build the policy: `policy = ApprovalPolicy(mode=AutonomyMode(config.policy_mode),
  shell=ShellMatcher(tuple(config.shell_allowlist)))`.
- Build the checkpoint store when `config.checkpoint_enabled`:
  `CheckpointStore(root, root / ".forge" / "checkpoints", config.checkpoint_keep_turns)`.
- Pass `policy`, `checkpoint` into `_build_tool_executor` → `ToolExecutor`.
  The **approver** differs by run path, so do NOT bake it into the executor at
  bootstrap; set it after (executor gains a public `set_approver`), or pass it
  when the Repl / headless path is constructed.
- Interactive: after the Repl is built, `tool_executor.set_approver(repl)` (the
  Repl implements `Approver`). Also pass the checkpoint store into `Repl` and
  `AgentLoop`.

### 5.2 Repl as `Approver` (`forge/repl.py`)
Implement `request(name, args, preview) -> Decision`:
- Print a summary line: `[approve] {name} wants to run:` plus the command (for
  shell) or the diff `preview` (for write/edit).
- Read a single line via the existing input func: `y` → APPROVE, `a` →
  APPROVE_ALWAYS (record `name`+normalized-target in a session-scoped set so the
  same action isn't re-prompted), `n`/anything → DENY.
- APPROVE_ALWAYS bookkeeping lives in the Repl (a `set`), consulted before
  prompting again. Keep it simple and session-scoped.

### 5.3 Headless (`forge/headless.py`, `forge/__main__.py`)
- Add `--yes` to `build_parser` (headless auto-approve).
- `run_prompt`/`run_headless` gets a `yes: bool` param. Wire the approver:
  `AutoApprover()` when `--yes`, else `DenyMutationsApprover()`.
  `tool_executor.set_approver(that)` before running.
- Document: without `--yes`, a headless run in `supervised`/`readonly` mode
  refuses mutations (returns denied/forbidden tool results) rather than hanging.
  This **replaces the Phase 1 README caveat** — update that README note: headless
  is now safe-by-default (denies mutations) unless `--yes` is passed.

### 5.4 `AgentLoop` checkpoint boundaries (`forge/agent.py`)
Optional `checkpoint=None` constructor param; in `run_turn`, around the turn
body: `if self.checkpoint: self.checkpoint.begin_turn()` at the start and
`commit_turn()` in the `finally` (alongside `interrupt.end_turn()`).

---

## 6. Full test plan & commands

```
python -m pytest -q --basetemp=C:\forge_tmp
Remove-Item -Recurse -Force C:\forge_tmp
```
New: `test_policy_matcher.py`, `test_executor_policy.py`, `test_diff_preview.py`,
`test_checkpoint.py`, `test_repl_undo.py`, `test_config_policy.py`.
Updated: `test_config_defaults.py`, `test_config_init.py`, and any REPL/headless
test that now needs an approver (inject `AutoApprover` to keep old assertions).

**Regression bar:** with no policy/approver wired, the executor and every
existing test behave exactly as before Phase 2.

---

## 7. Out of scope (do NOT do in Phase 2)

- General slash-command framework and `@file` mentions (Phase 4). `/undo` is a
  one-off special case here.
- Git-based checkpoints (use the file-snapshot store; no shadow refs).
- Approving individual hunks of a diff (approve/deny is whole-call).
- Persisting APPROVE_ALWAYS across sessions (session-scoped only).

---

## 8. Definition of Done

- [ ] `forge/policy.py`: `AutonomyMode`, `ApprovalPolicy`, `ShellMatcher`,
      `Approver`/`Decision`, `AutoApprover`, `DenyMutationsApprover`.
- [ ] Safe shell matcher rejects all compound/metachar commands; property-tested.
- [ ] `Tool.read_only` set on every built-in + MCP adapter; `preview` on
      write/edit.
- [ ] `ToolExecutor` gates via policy+approver before `run`; forbidden/denied
      results are side-effect-free; no-policy path unchanged.
- [ ] `CheckpointStore` snapshots pre-mutation state per turn (atomic writes),
      prunes to `keep_turns`, restores/deletes on `undo_last`.
- [ ] `/undo` REPL command; Repl implements `Approver` with diff preview and
      y/n/a; `AgentLoop` drives begin/commit turn boundaries.
- [ ] `[policy]` + `[checkpoint]` config tables; mode validation; `write_default`
      emits `supervised`; dataclass default stays `autopilot`.
- [ ] Headless `--yes`; safe-by-default headless (denies mutations otherwise);
      README caveat updated.
- [ ] All new tests written; full suite green; temp base removed.
```
