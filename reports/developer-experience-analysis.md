# Developer Experience Analysis

## Onboarding Friction

Estimated time to first safe change: **low (under an hour)** for the pure/unit
layers, currently **blocked** for the I/O-bound tests by a local environment
issue (DX-002).

- **README:** accurate but minimal (`README.md`). Covers install (`pip install
  -e ".[dev]"`) and `pytest`. Does not cover the security/trust model (see
  SEC-001) or `forge init` / config layout — a new user must read the code.
- **In-code documentation:** outstanding. Module docstrings explain intent and
  cite requirement IDs; functions explain *why*, not just *what*. This is the
  best onboarding asset in the project.
- **Specs:** `.kiro/specs/forge/` and `.kiro/specs/auto-verification-loop/`
  contain requirements/design/tasks that the code traces back to — excellent
  decision provenance.

## Debugging Experience

- Error messages are specific and actionable (e.g. the ADC error names the exact
  `gcloud` command; config errors name file + line/column).
- No structured logging / `--verbose` mode limits field debugging (INFRA-004).
- Local-vs-prod parity is essentially N/A (local tool).

## DX-001 — README omits the security/trust model and config docs

```
Severity: Medium   Confidence: High   File: README.md
```
A new operator cannot tell from the README that Forge will execute arbitrary
commands the model proposes, nor where config/sessions live. Add a "Security &
trust" section and a short config reference.

## DX-002 — Full test suite cannot run locally (temp-dir ACL)

```
Severity: Medium (local)   Confidence: High
Evidence: 34 tests error with
          PermissionError [WinError 5] Access is denied:
          'C:\Users\swara\AppData\Local\Temp\pytest-of-swara'
```
Every test using a real temp directory fails at fixture setup. This is an
environment/ACL problem (a stale, permission-restricted `pytest-of-swara`
folder), not a code defect, but it blocks verification of the integration tests.

**Fix.** Remove/recreate the stale temp dir, or point pytest at a writable base:
`pytest --basetemp=.pytest-tmp` (and add `.pytest-tmp/` to ignore lists), or set
`TMP`/`TEMP` to a writable location for the session. Then re-run `pytest` to
confirm a green suite.

## Automated Development Readiness

Scored against the methodology's four dimensions:

- **Code consistency** — High. File structure is predictable from module name.
- **Naming quality** — High. Names describe behavior; requirement IDs aid tracing.
- **Boundary clarity** — High. Changes scope cleanly to one module
  (composition is centralized in `app.py`).
- **Test reliability** — Mixed. Unit/property tests are reliable; the I/O tests
  fail for an environmental (not flaky-code) reason on this machine.

**Automated Development Readiness Score: 85/100.** Deductions: no VCS history to
reason about churn (INFRA-001), no enforced lint/coverage, and the local test
environment breakage. The code itself is very automation-friendly.

## Dependency Complexity

- Setup is a single `pip install -e ".[dev]"`. Shallow dependency graph (4 direct
  runtime deps). The only friction is unpinned versions (CF-003) and the
  undeclared `pytest-asyncio` (DEP-002).
