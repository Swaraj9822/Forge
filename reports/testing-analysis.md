# Testing Analysis

Testing is a clear strength. There are ~40 test files under `tests/` covering
unit behavior, integration wiring, and **property-based** invariants via
Hypothesis.

## Suite Result (this audit run)

```
python -m pytest -q
221 passed, 2 skipped, 34 errors in ~11s
```

**Important:** all 34 "errors" are the *same environment failure*, not code
defects:

```
PermissionError: [WinError 5] Access is denied:
'C:\Users\swara\AppData\Local\Temp\pytest-of-swara'
```

The errors occur during pytest's `tmp_path`/`tmpdir` fixture setup — every
erroring test is one that needs a real temp directory (shell/git/search/mcp/
verification integration tests that spawn processes or touch the filesystem).
The Windows temp folder `pytest-of-swara` has an ACL that denies access. This is
a **local environment problem** (a stale temp dir owned with restricted
permissions), and it blocks local verification of the I/O-bound tests. See
`developer-experience-analysis.md` (DX-002) for the fix.

## Coverage by Layer (qualitative)

| Layer | Coverage | Notes |
|-------|----------|-------|
| Pure/domain | Excellent | Property tests for compaction, search, git, planning, repl rendering, usage, config resolution, verification decision logic. |
| Tool behavior | Excellent | Per-tool unit tests incl. error/edge paths (read truncation, invalid range, edit uniqueness, write errors, shell output cap, validation safety). |
| Orchestration | Good | `test_agent_control_flow`, `test_agent_order`, `test_tool_executor_interrupt`, verification coordinator + interrupt timing. |
| Integration (process/fs) | Present but unverifiable here | Blocked by the temp-dir permission issue above. |
| Vertex client | Good (offline) | `test_vertex`, `test_vertex_contents` drive the translation/retry layer with injected fakes — no network needed. |

## Test Quality

- **Property tests** encode invariants (e.g. exposure = registry ∩ enabled;
  compaction always produces a well-formed window; git dispatch rejects
  unsupported ops). This is the highest-value form of testing and it is used
  well.
- **Offline-by-design:** `vertex.py` and `mcp_client.py` are written so the SDKs
  can be absent and fakes injected — tests need no network or live servers.
- **Interrupt timing tests** assert sub-second termination — good coverage of a
  tricky concurrency requirement.
- No obvious tautological/assertion-free tests were observed in the inventory
  (not every body was read line-by-line — Medium confidence).

## Gaps

- **No coverage measurement configured** (no `pytest-cov` / coverage gate). The
  qualitative coverage is high, but there is no enforced number.
- **No CI** means tests are not run automatically on change (see
  `infrastructure-analysis.md`).
- The temp-dir failure means a developer on this machine currently cannot get a
  green run of the full suite, masking regressions in the I/O-bound tests.

## Verdict

Testing 90/100. Fix the environment so the integration tests run, and add a
coverage report + CI gate to lock the quality in.
