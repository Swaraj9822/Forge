# Forge — Phase 3 Implementation Plan (Capability Depth)

**Audience:** a coding agent implementing this end-to-end.
**Prerequisite:** Phases 1–2 merged. This phase **reuses the Phase 1
context-provider seam** (`forge.context.ContextProvider`,
`ContextManager(providers=…)`).
**Scope:** Phase 3 only — deepen the agent's understanding of the project:

- **F — Durable cross-session memory (MemoryStore).** `remember` /
  `search_memory` tools + a memory `ContextProvider` that injects relevant,
  query-conditioned, token-budgeted memories each turn. Secret-redacted on write.
- **G — `repo_index` tool.** A lightweight, dependency-free repo map the model
  can call (file tree + top-level symbols/signatures).
- **H — Injected repo map (evolution of G).** Reuse G's indexer as a
  `ContextProvider` that injects a small, ranked, budgeted map each turn.
- **I — Robust edit format.** Add line-range / anchored edits so edits don't
  fail on ambiguous or whitespace-fragile targets.

Do **not** implement multi-provider, subagents, slash commands, or UI (Phase
4–5). No `tree-sitter` — use the stdlib `ast` for Python and regex for other
languages.

---

## 0. Ground rules

Same as `phase1.md` §0. Stdlib-only (`ast`, `re`, `json`, `pathlib`). Reuse the
existing offline/deterministic ethos of `ContextManager` — memory ranking and
the repo index must not make network calls. Test with
`pytest -q --basetemp=C:\forge_tmp` (base outside the repo), then remove it.

---

## 1. Architecture overview

| File | Change |
|------|--------|
| `forge/memory.py` (new) | `MemoryStore` (JSONL, append + query + prune), `redact_secrets`, keyword+recency ranking, path-staleness. |
| `forge/tools/memory.py` (new) | `RememberTool` (`remember`) + `SearchMemoryTool` (`search_memory`). |
| `forge/context_providers.py` | `MemoryProvider` and `RepoMapProvider` (reuse `RepoIndexer`). |
| `forge/repo_index.py` (new) | `RepoIndexer`: walk workspace, extract symbols, render a budgeted map. |
| `forge/tools/repo_index.py` (new) | `RepoIndexTool` (`repo_index`). |
| `forge/tools/fs.py` | Extend `EditTool` with line-range / anchored edit modes. |
| `forge/config.py` | `[memory]` and `[repo_map]` tables; add `remember`/`search_memory`/`repo_index` to `RECOGNIZED_TOOLS`. |
| `forge/app.py` | Wire the new tools + providers; build the `MemoryStore` and `RepoIndexer`. |
| `tests/…` | Store, redaction, ranking, tools, indexer, providers, and edit-mode tests. |

**Key gating note:** the three new tools must be added to
`forge.config.RECOGNIZED_TOOLS` and `DEFAULT_ENABLED_TOOLS`, or the executor's
`registry ∩ enabled` rule will drop them. Update those tuples and the
config-defaults tests.

---

## 2. Feature F — Durable memory

### 2.1 `forge/memory.py` (new)

Data model + store. JSONL, one memory per line, append-only with periodic prune.

```python
@dataclass(frozen=True)
class MemoryRecord:
    id: str
    text: str
    tags: tuple[str, ...]          # optional keywords supplied by the model
    paths: tuple[str, ...]         # workspace-relative files this memory is about
    created_at: str                # ISO-8601 UTC
    source: str                    # "model" | "system"
```

`MemoryStore`:
```python
class MemoryStore:
    def __init__(self, path: Path, *, max_records: int = 500): ...
    def add(self, text, *, tags=(), paths=(), source="model") -> MemoryRecord:
        """Redact secrets, append one JSONL line atomically, prune to max_records."""
    def all(self) -> list[MemoryRecord]:  # tolerant read; skip corrupt lines
    def search(self, query: str, *, limit: int, workspace_root: Path | None = None
               ) -> list[MemoryRecord]:
        """Keyword+recency ranked, staleness-filtered results (see 2.2/2.3)."""
    def prune(self) -> None:  # keep newest max_records, drop the rest
```

Persistence rules (mirror `SessionStore`):
- Append a single JSON line per `add`; for prune/rewrite, write a temp file in
  the same dir then `os.replace` (atomic).
- `all()` tolerates corrupt/partial lines: skip and continue (never raise), like
  `SessionStore.list`.
- Store location: **workspace-local** — `root / ".forge" / "memory.jsonl"` (so
  memories are per-project, matching how a repo map / AGENTS.md are per-project).

### 2.2 Secret redaction (`redact_secrets`, pure)
Best-effort defense-in-depth (do **not** oversell it — document as such):
- Redact common patterns before persisting: values that look like API keys /
  tokens (`(?i)(api[_-]?key|secret|token|password|authorization)\s*[:=]\s*\S+`),
  AWS keys (`AKIA[0-9A-Z]{16}`), PEM blocks (`-----BEGIN [^-]+PRIVATE KEY-----`
  … `-----END`), long high-entropy hex/base64 runs. Replace the sensitive span
  with `«redacted»`.
- Property test: for a corpus of synthetic secrets, the stored text never
  contains the original secret value; ordinary prose is unchanged.

### 2.3 Ranking + staleness (pure, offline)
- **Keyword score:** tokenize the query and each memory (`text` + `tags`);
  score = overlap count (case-insensitive), with tag matches weighted higher.
- **Recency:** tie-break / small boost by `created_at` (newer first).
- **Path staleness (invalidation):** when a memory lists `paths`, compare each
  file's current mtime to the memory's `created_at`; if the file changed after
  the memory was written, **down-rank or drop** it (configurable — default:
  down-rank, and annotate the injected text as "(may be stale)"). Missing files
  → drop. This is the "path-based staleness invalidation" from the design.
- Return top-`limit` by score.
- Property tests: relevance ordering (a memory sharing more query words ranks
  higher); a memory whose file mtime is newer than `created_at` is dropped/
  down-ranked; empty query → recency order.

### 2.4 Tools (`forge/tools/memory.py`)
- `RememberTool` (`name="remember"`, `read_only=False` — it writes to disk, but
  it is **not** a workspace-file mutation, so the Phase 2 policy should treat it
  as approval-exempt; simplest: classify `read_only=True` for approval since it
  touches only Forge's own store, not the user's project. Document the choice).
  Params: `text` (required), `tags` (optional array), `paths` (optional array).
  `run` → `store.add(...)`, returns the stored record id.
- `SearchMemoryTool` (`name="search_memory"`, `read_only=True`). Params: `query`
  (required), `limit` (optional, default from `[memory]`). `run` →
  `store.search(...)` rendered as text + `meta["results"]`.
- Both need the `MemoryStore`. Inject it via `ToolContext` — add an optional
  `memory` field to `ToolContext` (loosely typed, default `None`), set in the
  shared context at wiring time. (Mirrors how `state` is a session-scoped bag.)

### 2.5 `MemoryProvider` (`forge/context_providers.py`)
A `ContextProvider` that injects the most relevant memories each turn:
```python
class MemoryProvider:
    def __init__(self, store, *, limit, char_budget): ...
    def segments(self, session):
        query = self._latest_user_text(session)   # last user message
        if not query: return []
        hits = self.store.search(query, limit=self.limit,
                                 workspace_root=self.store.root_hint)
        if not hits: return []
        body = "\n".join(f"- {h.text}" for h in hits)  # truncate to char_budget
        return [{"role": "user",
                 "content": "[relevant project memory — background only]\n" + body}]
```
- **Query-conditioned** (uses the latest user message), **budgeted** (truncate to
  `char_budget`), **ephemeral** (the seam already never persists it). This is the
  design's key point: memory injection is retrieval, not static like steering.
- `_latest_user_text`: scan `session.messages` in reverse for the last
  `role=="user"` non-empty text (the just-appended prompt in `run_turn`).

### 2.6 Tests
`tests/test_memory_store.py` (add/all/prune/atomicity/corrupt-line tolerance),
`tests/test_memory_redaction.py` (property), `tests/test_memory_ranking.py`
(property: relevance + staleness), `tests/test_memory_tools.py`
(remember/search_memory happy + validation), `tests/test_memory_provider.py`
(query-conditioned injection, budget truncation, empty query → no segment,
ephemeral/not-persisted — reuse the Phase 1 invariant test style).

---

## 3. Feature G — `repo_index` tool

### 3.1 `forge/repo_index.py` (new) — `RepoIndexer`
Dependency-free structural map:
```python
class RepoIndexer:
    def __init__(self, root: Path, *, output_cap: int,
                 noise_dirs=frozenset({".git","node_modules","__pycache__",".venv",".forge"})): ...
    def build(self, *, budget_chars: int | None = None) -> str:
        """Return a text map: files (relative, sorted) each followed by its
        top-level symbols/signatures, capped to budget_chars/output_cap."""
```
- **Walk**: reuse the `SearchTool._walk_files` approach (os.walk, prune noise
  dirs, do not follow symlinks — stay in the workspace).
- **Symbol extraction:**
  - `.py`: parse with stdlib `ast`; emit `class`/`def`/`async def` names with
    arg signatures (top-level and one level of nesting for methods).
  - other languages: cheap regex for `function`/`class`/`def`/`export`/`func`
    definitions — best-effort, never raise on odd syntax.
  - undecodable/binary files: list the path only (no symbols).
- **Determinism & caching:** sort paths; cache the built map keyed on the set of
  `(path, mtime, size)` so repeated calls in one session don't re-walk. Rebuild
  when any file changes.
- **Cap:** truncate to `budget_chars` (or `output_cap`) with a
  `… (truncated)` marker, like the shell/git tools.

### 3.2 `RepoIndexTool` (`forge/tools/repo_index.py`)
- `name="repo_index"`, `read_only=True`. Optional params: `path` (limit to a
  subtree, scoped via `resolve_in_workspace`), `pattern` (only files matching a
  glob). `run` → `RepoIndexer(ctx.workspace_root, output_cap=…).build()`.
- Cap by `ctx.config.output_cap_chars` (fallback 30_000), matching `GitTool`.

### 3.3 Tests
`tests/test_repo_index.py`: a temp workspace with a couple of `.py` files →
map lists files and their `def`/`class` signatures; noise dirs pruned; symlink
not followed out of root; oversized map truncated + flagged; a syntactically
broken `.py` still lists the file (regex/`ast` failure degrades gracefully).

---

## 4. Feature H — Injected repo map (reuse G)

### 4.1 `RepoMapProvider` (`forge/context_providers.py`)
```python
class RepoMapProvider:
    def __init__(self, indexer: RepoIndexer, *, char_budget: int): ...
    def segments(self, session):
        # Only inject on the first turn / when the map changed, to save tokens.
        text = self.indexer.build(budget_chars=self.char_budget)
        if not text: return []
        return [{"role": "user",
                 "content": "[repository map — file/symbol overview]\n" + text}]
```
Design decisions to implement:
- **Budget:** hard char budget (e.g. from `[repo_map] char_budget`), separate
  from and smaller than the token limit; the Phase 1 seam already reserves the
  provider's tokens before compaction.
- **Freshness:** the indexer caches on `(path,mtime,size)`; the provider can
  inject every turn cheaply because unchanged repos return the cached map. To
  further save tokens, optionally only inject when the map hash changed since the
  last turn (track last-injected hash on the provider) — but injecting every turn
  is acceptable for v1 and simpler. Pick one and document it.
- **Ranking (optional refinement):** if the full map exceeds the budget,
  prioritize files referenced in recent conversation / touched this session, then
  the rest. Keep a simple heuristic; do not add PageRank/tree-sitter.

### 4.2 Tests
`tests/test_repo_map_provider.py`: injected as an ephemeral segment; respects
`char_budget`; empty repo → no segment; not persisted to `session.messages`;
gated by config (see §6).

---

## 5. Feature I — Robust edit format (`forge/tools/fs.py`)

Keep the existing unique-string `edit` behavior as the default mode, and add
optional modes so edits fail less on ambiguity/whitespace:

- **Anchored replace:** new optional args `after`/`before` — replace the unique
  occurrence of `target` that appears between the `after` and `before` anchors,
  disambiguating an otherwise-ambiguous target. If the target is still not unique
  within the anchor window, return the existing `ambiguous` result.
- **Line-range replace:** optional `start_line`/`end_line` (1-based inclusive) —
  replace exactly those lines with `replacement`, no target matching needed.
  Validate the range like `ReadTool` does (reuse its invalid-range semantics).
- **Mode selection:** infer from which args are present; explicit `mode`
  (`"replace" | "anchored" | "line_range"`) takes precedence. Exactly one mode
  applies per call; conflicting args → validation error.
- Preserve the Phase 2 `preview` (diff) and atomic-write behavior for every mode.

**Tests** (`tests/test_edit_modes.py`): line-range replace on known lines;
anchored replace disambiguates a target that occurs twice; invalid range
rejected; conflicting-mode args rejected in `validate`; default `replace` mode
unchanged (existing `test_edit_*` stay green — do not alter their behavior).

---

## 6. Config schema (`forge/config.py`)

```toml
[memory]
enabled = true
max_records = 500
inject_limit = 5          # memories injected per turn
inject_char_budget = 2000

[repo_map]
enabled = true
inject = true             # inject the map into context each turn (Feature H)
char_budget = 4000
```

- Add `RECOGNIZED_TOOLS += ("remember", "search_memory", "repo_index")` and to
  `DEFAULT_ENABLED_TOOLS`. **Update `tests/test_tool_exposure.py` and
  `test_config_defaults.py`** accordingly.
- `Config` fields (defaults preserve current behavior where sensible):
  `memory_enabled: bool = False`, `memory_max_records: int = 500`,
  `memory_inject_limit: int = 5`, `memory_inject_char_budget: int = 2000`,
  `repo_map_enabled: bool = False`, `repo_map_inject: bool = False`,
  `repo_map_char_budget: int = 4000`.
  **Rationale:** default the *providers* OFF at the dataclass level so absent
  tables reproduce today's context byte-for-byte (Phase 1 invariant), and have
  `write_default` opt new configs into `enabled = true`. Same asymmetry pattern
  as Phase 2's policy mode.
- Parse `[memory]`/`[repo_map]` in `_from_raw`; `write_default` emits both tables.

---

## 7. Wiring (`forge/app.py`)

- Build `MemoryStore(root / ".forge" / "memory.jsonl", max_records=…)` and
  `RepoIndexer(root, output_cap=config.output_cap_chars)` when enabled.
- Add `memory` to `ToolContext` and set it on the shared context so the memory
  tools can reach the store.
- Register the new tools in `build_builtin_registry` (or a parallel step) and
  ensure their names are enabled.
- Extend the `providers` list built for `ContextManager`:
  ```python
  if config.memory_enabled:
      providers.append(MemoryProvider(store, limit=config.memory_inject_limit,
                                      char_budget=config.memory_inject_char_budget))
  if config.repo_map_enabled and config.repo_map_inject:
      providers.append(RepoMapProvider(indexer, char_budget=config.repo_map_char_budget))
  ```
  Keep `PlanReminderProvider` first (from Phase 1). **Provider order matters for
  budget reservation** — put the plan reminder and memory (small, high-value)
  before the repo map (large) so the reminder is never crowded out.

---

## 8. Full test plan & DoD

Run: `python -m pytest -q --basetemp=C:\forge_tmp` then remove the temp dir.

New test files: `test_memory_store.py`, `test_memory_redaction.py`,
`test_memory_ranking.py`, `test_memory_tools.py`, `test_memory_provider.py`,
`test_repo_index.py`, `test_repo_map_provider.py`, `test_edit_modes.py`.
Updated: `test_tool_exposure.py`, `test_config_defaults.py`,
`test_config_init.py`.

**Definition of Done**
- [ ] `MemoryStore` (JSONL, atomic, corrupt-tolerant, pruned); `redact_secrets`
      (property-tested); keyword+recency ranking with path-staleness.
- [ ] `remember` / `search_memory` tools; `MemoryProvider` (query-conditioned,
      budgeted, ephemeral, not persisted).
- [ ] `RepoIndexer` (ast+regex, noise-pruned, symlink-safe, cached, capped);
      `repo_index` tool; `RepoMapProvider` (budgeted, ephemeral).
- [ ] `edit` gains anchored + line-range modes; default mode unchanged;
      preview + atomic writes preserved.
- [ ] New tools added to `RECOGNIZED_TOOLS`/`DEFAULT_ENABLED_TOOLS`; exposure
      tests updated.
- [ ] `[memory]`/`[repo_map]` config; providers OFF by dataclass default, ON in
      `write_default`; absent-table context is byte-identical to Phase 2.
- [ ] Wiring adds tools + providers in budget-safe order.
- [ ] Full suite green; temp base removed.

---

## 9. Out of scope (Phase 4–5)
- tree-sitter / PageRank ranking, embeddings-based memory retrieval.
- Multi-provider, subagents, slash commands, `@file`, rich UI.
- Cross-workspace (global) memory — memory is per-workspace here.
```
