# Data Integrity Review

The only persisted "business data" is the session store (one JSON file per
session) and the config file. There is no relational/transactional database, so
several subsections are N/A.

## Referential Integrity

N/A — no foreign keys / relational store. Within a session, tool-call → tool-result
correlation is by `call_id`; `forge/context.py::_pending_tool_call_indices`
correctly keeps an in-flight call+result pair together across compaction so the
Gemini "function-call/response part counts must match" invariant holds.

## Transactional Consistency — session writes

- **Atomic writes (good).** `forge/session.py::SessionStore.save` writes to a
  temp file in the *same* directory, `fsync`s, then `os.replace`s onto the
  target. A crash mid-write leaves either the old or the new file intact, never
  a half-written one. Temp files are cleaned up on failure.
- **Lossless round-trip (good).** `session_to_json` / `session_from_json`
  reconstruct nested dataclasses (not raw dicts), so equality holds across a
  save/load cycle. Verified by `tests/test_session_roundtrip.py`.
- **Usage continuity (good).** Cumulative token/cost totals are mirrored onto the
  session on every turn (`forge/agent.py::run_turn`) and re-seeded on resume
  (`forge/app.py` → `UsageTracker.seed`), so resumed sessions continue their
  tallies rather than restarting.

## Concurrency & Race Conditions

- **In-process per-session lock (good).** `SessionStore._lock_for` serializes
  concurrent saves to the *same* session id.
- **No cross-process lock (DI-001, Low).** Two `forge` processes that
  `resume` the same session id can interleave `os.replace`s; last-writer-wins,
  losing one process's turns. Unlikely in an interactive single-user tool but
  worth a note. Fix: an advisory lock file per session, or accept the risk
  explicitly.
- **Planning todo state** lives in the shared `ToolContext.state` and is synced
  to the session in `agent.py::_sync_todos`; single-threaded within a turn, so
  no race.

## Soft Delete Consistency

N/A — no soft-delete model.

## Migration Safety

N/A — no schema migrations. The session JSON shape is read defensively
(`session_from_dict` tolerates missing optional lists via `.get(..., [])`), and a
malformed file raises `CorruptSessionError` **without touching the bytes**
(`SessionStore.load`), so a bad file is reported, not corrupted further. `list()`
skips unreadable files rather than failing the whole listing. This is good
forward/backward-tolerant handling in lieu of formal migrations.

## Verdict

Data integrity is a strength (88/100). The only gap is the absence of a
cross-process session lock, which is Low impact for this tool.
