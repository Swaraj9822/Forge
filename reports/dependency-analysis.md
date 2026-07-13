# Dependency Analysis

Source of truth: `pyproject.toml`.

## Direct Dependencies

| Package | Declared Version | Purpose | Usage | Risk |
|---------|------------------|---------|-------|------|
| `google-genai` | **unpinned** | Vertex AI / Gemini SDK (the model backend) | Core | Med — unpinned + fast-moving SDK; a breaking release can break `vertex.py` silently. Imported guardedly so absence degrades, but a *changed* API would not. |
| `mcp` | **unpinned** | Model Context Protocol client (external tool servers) | Core (optional feature) | Med — async API surface; unpinned. Imported guardedly. |
| `tomli-w` | **unpinned** | Writing the default `config.toml` (`forge init`) | Peripheral | Low |
| `prompt_toolkit` | **unpinned** | Interactive REPL line editing | Core (interactive only; injectable in tests) | Low |

Reading TOML uses the stdlib `tomllib` (Python ≥ 3.11), so no `tomli` runtime dep
is needed — `requires-python = ">=3.11"` is consistent with that. Good.

## Dev Dependencies

| Package | Version | Purpose | Risk |
|---------|---------|---------|------|
| `hypothesis` | unpinned | Property-based testing | Low |
| `pytest` | unpinned | Test runner | Low |

Note: the installed environment also has `pytest-asyncio` (it appears in the
traceback), but it is **not declared** in `pyproject.toml`'s dev extras. If any
test relies on it, that is an undeclared dev dependency (DEP-002, Low).

## Findings

```
CF-003 / DEP-001 — All dependencies are unpinned
Severity:  Medium   Confidence: High
File:      pyproject.toml (dependencies = ["google-genai", "mcp", ...])
Impact:    Non-reproducible installs; a transitive or direct breaking release
           can break Forge with no code change. For the two fast-moving SDKs
           (google-genai, mcp) this is the real risk.
Fix:       Pin with compatible-release specifiers, e.g.
           "google-genai>=X.Y,<X+1", "mcp>=A.B,<A+1", and commit a lockfile
           (pip-tools / uv) for reproducible dev/CI installs.
```

```
DEP-002 — pytest-asyncio used but not declared
Severity:  Low   Confidence: Medium
Evidence:  pytest_asyncio appears in the test traceback; absent from [dev] extras.
Fix:       Add it to the dev extras if tests depend on it, or remove the
           dependency if not.
```

## Supply-chain audit

A live `pip-audit` was **not** run (offline audit; not assuming network). No
known-CVE assessment is included. **Recommendation:** add `pip-audit` (or
`uv pip audit`) to CI so direct + transitive CVEs are tracked, especially for
`google-genai` and `mcp` and their transitive trees.

- **Unused deps:** none — all four direct deps are imported and used.
- **Abandoned deps:** none apparent (all are actively maintained projects).
- **Duplicated functionality:** none.

## Verdict

The dependency *set* is lean and appropriate. The only real action is **pinning
+ a lockfile + CI audit**, which converts the current "trust the author's local
env" posture into reproducible, monitored builds.
