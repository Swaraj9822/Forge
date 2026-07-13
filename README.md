# Forge

A minimal but complete terminal-based AI coding agent.

Forge runs as an interactive command-line REPL that drives an autonomous agent
loop: it sends user requests to a Gemini model hosted on Google Cloud Vertex AI,
streams responses to the terminal, and lets the model invoke a set of coding
tools (file read/write/edit, shell execution, codebase search, and git
operations) to complete tasks.

## Development

Install in editable mode with dev dependencies:

```
pip install -e ".[dev]"
```

Run the test suite:

```
pytest
```
