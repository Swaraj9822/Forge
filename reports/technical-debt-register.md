# Technical Debt Register

## Root Cause Clustering

Most findings collapse into three root causes. Fixing the root resolves the
symptoms.

```
RC-1 — Undocumented, ungated agent trust model
Symptoms:
  - shell runs arbitrary commands, no sandbox/allowlist/confirmation  [forge/tools/shell.py]
  - fs path-scoping is bypassable, so it is not a real boundary        [forge/tools/paths.py vs shell.py]
  - tool output (incl. untrusted file/MCP content) re-enters context   [forge/agent.py]
  - no audit log of executed commands                                  [data-flow-analysis.md]
  - README does not state the trust model                              [README.md]
Impact:  A prompt-injection or careless command harms the operator's machine,
         and there is no record to reconstruct it.
Fix:     Document the trust model + add an opt-in command-approval gate and an
         executed-command audit log.
Effort:  Days

RC-2 — No engineering-process scaffolding
Symptoms:
  - no git repository / history                                        [workspace]
  - no CI test/lint gate                                               [no .github etc.]
  - no committed lint/format config (despite noqa markers)             [pyproject.toml]
  - unpinned dependencies, no lockfile                                 [pyproject.toml]
  - undeclared pytest-asyncio dev dep                                  [pyproject.toml]
  - full test suite cannot run locally (temp-dir ACL)                  [DX-002]
Impact:  Quality is carried by one author, not enforced; builds aren't
         reproducible; regressions can land silently.
Fix:     git init + remote; CI (pytest + ruff + pip-audit); pin deps + lockfile;
         fix local temp dir.
Effort:  Days

RC-3 — vertex.py concentration + duplicated fs/write boilerplate
Symptoms:
  - vertex.py is 872 lines spanning ~7 responsibilities                [forge/vertex.py]
  - atomic-write block triplicated (Write, Edit, SessionStore)         [fs.py, session.py]
  - binary-detection duplicated (Read, Edit)                           [fs.py]
Impact:  Highest-friction areas to change safely; minor duplication drift risk.
Fix:     Extract vertex translation helpers to a module; add shared atomicwrite
         and decode-or-binary helpers.
Effort:  Hours–Days
```

## Register

| ID | Severity | Area | Description | Evidence | Effort | Impact if Ignored |
|----|----------|------|-------------|----------|--------|-------------------|
| TD-01 | High | Security | Unsandboxed shell, no confirmation gate, no audit log | `forge/tools/shell.py`; `forge/tools/base.py::execute` | Days | Local RCE via prompt injection; unreconstructable incidents |
| TD-02 | High | Process | No version control in workspace | `git` not initialized | Hours | No history/rollback/review |
| TD-03 | High | Infra | No CI test/lint gate | no `.github/`, `tox.ini` | Hours | Regressions land unnoticed |
| TD-04 | Medium | Deps | Unpinned dependencies, no lockfile | `pyproject.toml` | Hours | Non-reproducible builds; SDK breakage |
| TD-05 | Medium | Quality | `vertex.py` god-module drift | `forge/vertex.py` (872 lines) | Days | Change-friction, harder review |
| TD-06 | Medium | DX | README lacks trust model + config docs | `README.md` | Hours | Operators unaware of RCE; slow onboarding |
| TD-07 | Medium | DX/Test | Local suite blocked by temp-dir ACL | pytest `tmp_path` PermissionError | Hours | Integration tests unverifiable locally |
| TD-08 | Low | Quality | Triplicated atomic-write / duplicated binary detection | `fs.py`, `session.py` | Hours | Minor drift risk |
| TD-09 | Low | Perf | Content search reads every file fully | `forge/tools/search.py` | Hours | Slow/memory-heavy on large repos |
| TD-10 | Low | Integrity | No cross-process session lock | `forge/session.py` | Hours | Lost turns if same session resumed twice concurrently |
| TD-11 | Low | Security | Session transcripts stored unencrypted | `forge/session.py` | Hours | Secrets-in-transcript at rest |
| TD-12 | Low | Lint | No committed ruff config despite `noqa` usage | `pyproject.toml` | Hours | Lint rules not reproducible |
| TD-13 | Low | Obs | No structured logging / `--verbose` | warnings-only across modules | Days | Harder field debugging |

Debt with no user-facing impact is deprioritized accordingly; nothing here is
"theoretical only" except arguably TD-10 (low likelihood in a single-user tool).
