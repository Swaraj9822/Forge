# Infrastructure & Operational Risk

Forge is a distributable CLI, not a hosted service, so "infrastructure" here
means packaging, CI/CD, and the engineering process around the repo. Many
production-service checklist items are N/A and are marked as such.

## Git History

**N/A — this workspace is not a git repository.** `git log` / `git shortlog`
both return `fatal: not a git repository`. Therefore the entire Section 3 of the
methodology (churn map, ownership/bus-factor, regression patterns, age analysis)
cannot be performed. This is itself the most significant infrastructure finding:

```
INFRA-001  No version control in the workspace
Severity:  High (process)   Confidence: High
Impact:    No history, no churn signal, no ownership record, no blame, no
           revert safety net, no PR review trail. For a codebase of this
           quality, the lack of VCS is the single biggest process risk.
Fix:       git init; commit; push to a remote; protect main.
```

## CI/CD Pipeline

```
INFRA-002  No CI configuration
Severity:  High (process)   Confidence: High
Evidence:  no .github/, .gitlab-ci.yml, azure-pipelines.yml, .circleci/, tox.ini
```

There is no automated test/lint gate. The strong test suite is only as good as
the discipline to run it locally — and locally it currently errors (temp-dir
ACL). Fix: a minimal GitHub Actions (or equivalent) workflow running
`pip install -e ".[dev]"` then `pytest` on push/PR for Python 3.11–3.13.

## Lint / Format gate

```
INFRA-003  No linter/formatter config committed
Severity:  Medium   Confidence: High
Evidence:  no ruff.toml/.ruff.toml/.flake8/setup.cfg; yet the code carries
           `# noqa: BLE001` markers implying ruff was used during development.
```

The `noqa: BLE001` comments indicate ruff was run at some point, but no config
is committed, so the rules aren't reproducible or enforced. Fix: add a
`[tool.ruff]` section to `pyproject.toml` and run it in CI.

## Production Readiness Checklist (scoped to a CLI tool)

| Item | Present | Severity if Missing |
|------|---------|---------------------|
| Input validation | Yes (tool `validate` + config validation) | — |
| Rate limiting (outbound) | Yes (Vertex backoff/retry) | — |
| Request timeouts | Yes (`request_timeout_s`, shell timeout) | — |
| Graceful shutdown / interrupt | Yes (`InterruptController`, process-tree kill) | — |
| Structured logging w/ trace IDs | **No** | Low — uses `warnings.warn`; no logging framework. For a CLI this is acceptable but limits debuggability. |
| Error tracking | N/A | No telemetry by design (privacy-positive for a local tool). |
| Secrets management | Yes (Google ADC; no secrets in repo) | — |
| Rollback capability | **No** | Tied to INFRA-001 (no VCS). |
| Version pinning | **No** | Medium — see `dependency-analysis.md` (CF-003). |
| Env separation | N/A | Single local environment. |

## Observability

- **Logging:** the project uses `warnings.warn` for non-fatal conditions
  (missing steering file, unrecognized tool, MCP connect failure, compaction
  could-not-reduce). There is no structured logger. For a single-user CLI this
  is a reasonable minimalist choice; a `--verbose`/`--debug` flag routing to
  `logging` would improve field debuggability (INFRA-004, Low).

## Incident Reconstruction (methodology §12.5)

Working through the heuristics:

1. **Missing timeouts:** none found — Vertex and shell both have wall-clock
   guards; MCP connect has a per-server budget + an outer guard.
2. **Shared mutable state:** session writes are locked per-session in-process
   (cross-process gap = DI-001).
3. **Dead-letter handling:** N/A (no queues).
4. **Unbounded operations:** read/shell/git/search outputs are all capped; the
   one unbounded-ish operation is the **content search reading every file**
   (PERF-001) — the most likely "why is Forge hanging?" incident on a big repo.
5. **Cascading failure:** an MCP server failing to connect warns and continues
   with built-ins (good, no cascade). SDK absence degrades gracefully.

Most likely real-world incident: **a destructive `shell` command executed via
prompt injection** (SEC-001) — High blast radius (operator's machine), Low
detectability (no audit log of executed commands), Hard recovery.

## Operational Risk Assessment

| Area | Risk Level | Worst Scenario |
|------|------------|----------------|
| Availability | Low | Single local process; crash only affects the user. |
| Recoverability | Medium | No VCS (INFRA-001) → no rollback of code; sessions are recoverable (atomic writes). |
| Deployability | Medium | `pip install -e` works; no release automation or pinned deps. |
| Observability | Medium | warnings-only; no command audit log. |
| Incident Response | Medium | No executed-command log makes a prompt-injection incident hard to reconstruct. |
