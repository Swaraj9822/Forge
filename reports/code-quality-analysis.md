# Code Quality Analysis

Overall quality is high. The dominant style is small modules, extensive
docstrings that cite requirement IDs, defensive `try/except` with documented
intent, and type hints throughout. The findings below are refinements, not rot.

## Complexity Hotspots

| File | Lines | Concern |
|------|-------|---------|
| `forge/vertex.py` | 872 | Multiple responsibilities (client lifecycle, exception translation, SDK wire translation, chunk parsing, retry/backoff, retry-hint coercion). File length >500 flagged. Functions are individually readable but the file is the hardest to change safely. |
| `forge/tools/fs.py` | 538 | Three tools (Read/Write/Edit) with repeated atomic-write + binary-detection + path-scoping boilerplate across Write and Edit. Candidate for a shared `_atomic_write` and `_decode_or_binary` helper. |
| `forge/context.py::compact` | ~120 | High cognitive load: region partitioning, summary insertion, drop loop. Correct and well-commented, but the densest single function. |
| `forge/verification.py::VerificationCoordinator.run` | ~90 | Long but linear and well-narrated; acceptable. |

No function exceeds reasonable nesting depth egregiously; cyclomatic complexity
is kept down by early returns.

## Code Smells

- **Duplicate logic (real):** the atomic temp-file-write + `os.replace` + cleanup
  block is copy-pasted in `WriteTool.run`, `EditTool.run` (`forge/tools/fs.py`)
  and structurally again in `SessionStore.save` (`forge/session.py`). Binary
  detection (`b"\x00" in raw` + UTF-8 decode) is duplicated in `ReadTool` and
  `EditTool`. Extract shared helpers. **Effort: Hours.**
- **Broad excepts:** many `except Exception:  # noqa: BLE001`. Each is documented
  ("degrade gracefully") and intentional given the defensive posture against SDK
  shape variance, but the breadth can mask real bugs. Low priority; consider
  narrowing where the expected exception type is known.
- **Magic numbers:** mostly avoided — limits live in `forge/config.py::DEFAULT_LIMITS`
  and module-level constants (`CHARS_PER_TOKEN`, `BACKOFF_*`, `_KILL_GRACE_S`).
  Good.
- **Boolean-trap parameters:** none notable. Keyword-only args are used for
  clarity (e.g. `bootstrap(*, ...)`).
- **Dead code:** none found. The `_*_IS_A_TOOL` module-level protocol assertions
  in `fs.py` are intentional type-checker aids, not dead code.
- **Naming:** consistent and behavior-describing. You can predict a file's
  structure from its module name — strong for both humans and automated tools.

## SOLID

- **SRP:** Well-followed except `vertex.py` (see hotspots).
- **OCP:** New tools are added by implementing the `Tool` protocol and enabling
  them — extension without modifying the executor. Good.
- **LSP:** Tools and renderers are structurally typed `Protocol`s; substitution
  holds (`NullRenderer`, `McpToolAdapter`).
- **ISP:** Renderer protocols are small and hooks are optional (invoked via
  `hasattr`/`getattr` guards in `agent.py`). Good.
- **DIP:** Exemplary — `ContextManager` depends on an injected `summarizer`
  abstraction; `AgentLoop` depends on injected collaborators; the composition
  root wires concretes.

## Design Patterns

- **Used well:** Strategy/Adapter (tools + MCP adapter), Dependency Injection
  (composition root), tagged-union stream events (`StreamEvent`), pure-core /
  imperative-shell (verification).
- **Misused:** none observed.
- **Missing that would help:** a tiny `atomicwrite(path, bytes)` utility to kill
  the triplicated write boilerplate.

## Verdict

Maintainability 88/100. The two concrete actions are (1) split `vertex.py` and
(2) de-duplicate the atomic-write and binary-detection helpers.
