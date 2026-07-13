# Performance Analysis

Forge is a single-user interactive CLI, so latency is dominated by the model and
the network, not by Forge's own code. The findings below are the only
application-layer costs worth noting.

## PERF-001 — Content search reads every workspace file fully into memory

```
Severity:   Medium   Confidence: High
File:       forge/tools/search.py::_run_content / _walk_files / _read_text
Cost:       O(total bytes in workspace) per content search; each file fully
            read via path.read_bytes() then decoded then splitlines().
```

For each content search the engine walks the entire workspace (pruning only
`.git`, `node_modules`, `__pycache__`, `.venv`) and reads every non-pruned file
in full, even after the 100-match result cap is reached for *results* — the walk
still opens and decodes every file until the cap is hit, and large individual
files are fully materialized. On a large repo (e.g. hundreds of MB, build
outputs, vendored deps) a single search can be slow and memory-heavy.

**Fix.** Stream files line-by-line instead of `read_bytes()` + `splitlines()`;
add a per-file size skip (e.g. skip files > a few MB); short-circuit the file
walk once the result limit is reached (already partially done). Optionally
respect `.gitignore`.

## PERF-002 — Compaction drop-loop re-estimates the whole window each iteration

```
Severity:   Low   Confidence: High
File:       forge/context.py::compact (the `while estimate_tokens(window) > limit` loop)
Cost:       O(n^2) in the number of droppable messages; bounded by
            retained_recent_messages (default 20), so ~400 estimate ops worst case.
```

Each dropped message rebuilds and re-estimates the entire window. Bounded and
infrequent (only on over-limit assembly), so impact is small. Could subtract the
dropped message's contribution incrementally if it ever matters.

## Application layer

- **Blocking I/O:** the shell core (`forge/tools/shell.py`) correctly drains
  stdout/stderr on background threads and polls, so a chatty child never
  deadlocks on a full pipe. Good.
- **Token estimation:** deterministic, offline, O(n) per assemble. Fine.
- **No per-request crypto / heavy serialization in hot paths.**

## Database / Data Access

N/A — no database. Persistence is one JSON file per session.

## API Layer

N/A — no HTTP API. The only network client is the streaming Vertex call
(`forge/vertex.py`), which already applies a request timeout, a wall-clock guard
between chunks, and capped+jittered exponential backoff on rate limits.

## Frontend

N/A — terminal output only. Streaming writes are flushed per token to honor the
200ms/token responsiveness target.

## Scalability Limits

| Bottleneck | Estimated Failure Point | Assumption |
|------------|-------------------------|------------|
| Content search latency/memory | Noticeable on workspaces with very large/binary-heavy trees | Reads all non-pruned files per query |
| Context window | Compaction keeps requests under `token_limit` (default 200k) | Heuristic estimate ~4 chars/token |
| Concurrency | Single interactive user per process | CLI design |

Category: the tool is **not** scale-sensitive in the usual sense; the only real
knob is workspace size for search.
