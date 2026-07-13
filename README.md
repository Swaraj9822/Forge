# Forge

A minimal but complete terminal-based AI coding agent.

Forge runs as an interactive command-line REPL that drives an autonomous agent
loop: it sends user requests to a Gemini model hosted on Google Cloud Vertex AI,
streams responses to the terminal, and lets the model invoke a set of coding
tools (file read/write/edit, shell execution, codebase search, and git
operations) to complete tasks.

> **Documentation:** see the [User Manual](user_manual.md) for full
> configuration and usage, and the [Changelog](CHANGELOG.md) for notable
> changes.

## Development

Install in editable mode with dev dependencies:

```
pip install -e ".[dev]"
```

Run the test suite:

```
pytest
```

## Non-interactive mode

Forge supports a headless, scriptable mode via `forge -p "<prompt>"`:

```
forge -p "list the python files in this repo" --output json
```

Use `--output text` (default) for streamed plain text or `--output json` for a
single parseable JSON object. Pass `-p -` to read the prompt from stdin.

**Phase 2 — safe by default.** Headless runs respect the approval policy
configured under `[policy]` (`autopilot`, `supervised`, or `readonly`). Without
`--yes`, a `supervised` or `readonly` run refuses any gated mutation (returns a
denied/forbidden tool result) rather than hanging on a prompt that cannot be
answered. Pass `--yes` to auto-approve every gated call, matching the autopilot
behavior:

```
# Refuses write/edit/shell unless they are in the allowlist (or you pass --yes)
forge -p "summarize README.md" --output json

# Auto-approve everything (CI / trusted environments)
forge -p "run the tests and fix failures" --output json --yes
```

**Run budgets (CI safety).** Non-interactive runs accept `--max-turns` and
`--max-cost` to bound a run. `--max-turns N` caps the number of model
round-trips; `--max-cost X` caps the cumulative estimated spend in USD (and
requires model pricing under `[pricing]`, else the run is refused). Hitting
either budget stops the run and exits with code `5`:

```
forge -p "refactor the module and run tests" --yes --max-turns 20 --max-cost 0.50
```

Headless exit codes: `0` success, `2` turn error, `3` interrupted, `4`
verification failed, `5` run budget exceeded.
