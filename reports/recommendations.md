# Recommendations & Roadmap

## Prioritization Matrix

```
                    LOW EFFORT                         HIGH EFFORT
HIGH IMPACT     Q1 ← Do First                      Q2 ← Plan Carefully
                - CF-002 git init                  - CF-001 trust model + approval
                - CF-003 CI gate                      gate + command audit log
                - CF-006 README trust/config docs
                - CF-007 fix local test env
LOW IMPACT      Q3 ← Schedule                      Q4 ← Accept or Drop
                - CF-004 pin deps + lockfile       - CF-005 split vertex.py
                - CF-008 search streaming             (improvement, not urgent)
                - CF-011 dedupe write helpers
                - CF-012 declare pytest-asyncio    - CF-009 cross-process lock
                                                      (accept risk: single user)
                                                   - CF-010 transcript-at-rest
                                                      (document; accept for local)
```

## Cost vs Value (P0/P1 only)

| Finding | Effort | Risk Reduction | Business Value | Recommendation |
|---------|--------|----------------|----------------|----------------|
| CF-001 Unsandboxed shell + no gate/audit | Days | High | High | Fix This Sprint — document now (hours), gate + audit next |
| CF-002 No version control | Hours | High | High | Fix Immediately |
| CF-003 No CI gate | Hours | High | High | Fix Immediately |

There are **no P0 (today)** findings: there is no live production system to
protect and no remotely exploitable bug. CF-001 is P1 because the risk is real
but is bounded to the operator's own machine and mitigated by running Forge only
in trusted workspaces in the meantime.

## Refactoring Roadmap

### Phase 1 — Stop the Bleeding (Week 1)
- **CF-002:** `git init`, commit the current tree, push to a remote, protect main.
  (Everything else depends on this.)
- **CF-003:** add a minimal CI workflow: `pip install -e ".[dev]"` → `ruff check`
  → `pytest` on py3.11–3.13.
- **CF-007:** fix the local temp-dir ACL (or `pytest --basetemp=.pytest-tmp`) and
  confirm a green suite so CI and local agree.
- **CF-006 (doc half of CF-001):** add a "Security & trust" section to the README
  stating that Forge executes commands the model proposes; advise trusted
  workspaces only.

### Phase 2 — Stabilization (Weeks 2–4)
- **CF-001 (controls):** add an opt-in `require_command_approval` config flag that
  prompts before `shell` and mutating `git` ops; add an executed-command audit
  log (separate from the session transcript).
- **CF-004:** pin runtime deps to compatible-release ranges and commit a lockfile;
  add `pip-audit` to CI. **CF-012:** declare `pytest-asyncio` (or drop it).
- **CF-012/CF-011:** add a shared `atomicwrite()` + `decode_or_binary()` helper
  and route Write/Edit/SessionStore through it.

### Phase 3 — Architecture & Performance (Month 2–3)
- **CF-005:** split `forge/vertex.py` — move `_to_sdk_contents`, `_to_sdk_tools`,
  `_parse_chunk`, `_function_call_to_tool_call`, and the retry-hint coercers into
  a `vertex_translate.py`; keep `VertexClient` as lifecycle + streaming + retry.
- **CF-008:** stream content search line-by-line, skip oversized files, respect
  `.gitignore`.

### Phase 4 — Strategic (Ongoing)
- Add a committed `[tool.ruff]` config (CF-012/TD-12).
- Optional `--verbose`/`logging` mode (TD-13).
- Decide explicitly on CF-009 (cross-process lock) and CF-010 (transcript at
  rest): for a single-user local tool, documenting and accepting the risk is a
  legitimate "Do Not Fix" — see below.

## Do Not Fix (with justification)
- **CF-009 cross-process session lock — Accept Risk.** Forge is a single-user
  interactive CLI; concurrently resuming the *same* session id in two processes
  is an unlikely operator action. Document the last-writer-wins behavior rather
  than adding locking machinery.
- **CF-010 transcript encryption — Document, don't encrypt.** Local-tool data at
  rest under the user's own data dir; encryption would add key-management
  complexity disproportionate to the threat. Document and offer `--no-persist`.

---

## Final Audit Verdict

> **Healthy.**

Forge is a small, genuinely well-built codebase: dependency-injected wiring,
clean layering, a `Protocol`-based tool model, an exemplary pure/impure split in
the verification phase, and a disciplined property-based test suite that passes
(221 green). There are **no critical or remotely exploitable defects** and very
little code rot.

It falls short of "Excellent" for two reasons, both about process and posture
rather than correctness: (1) the agent's **local code-execution trust model is
ungated and undocumented** — acceptable for an autonomous coding agent, but it
needs an opt-in guardrail, an audit log, and a clear README warning; and (2) the
project has **no version control, no CI, and unpinned dependencies**, so its
high quality is currently sustained by author discipline rather than enforced by
tooling.

Close the Phase 1 items (git, CI, fix the local test env, document the trust
model) and Forge moves solidly toward "Excellent." None of these block continued
development; they make the existing quality durable and the security posture
explicit.
