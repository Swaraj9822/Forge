"""Optional rich-backed console rendering with plain-text fallback.

Streaming markdown rendering is out of scope. Markdown cannot be rendered
until a block completes, which forces buffering that fights the token-latency
requirement.
"""

from __future__ import annotations

import contextlib

from forge.memory import redact_secrets

try:
    from rich.console import Console
    from rich.syntax import Syntax
    _RICH = True
except Exception:  # noqa: BLE001
    _RICH = False


def _rich_escape(text: str) -> str:
    """Escape rich markup metacharacters so tool detail renders literally."""
    try:
        from rich.markup import escape

        return escape(text)
    except Exception:  # noqa: BLE001
        return text.replace("[", "\\[")


def _clip(value: object, limit: int = 72) -> str:
    """Collapse whitespace and truncate ``value`` to ``limit`` chars (ASCII-safe)."""
    text = " ".join(str(value).split())
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def describe_tool(name: str, args: dict | None) -> str | None:
    """Return a short, human-readable description of what a tool call is doing.

    Uses the primary argument(s) of each built-in tool (the file path, the
    search pattern, the shell command, and so on) so the terminal shows the
    *target* of a call, not just its name. Returns ``None`` when there is
    nothing useful to add beyond the tool name.
    """
    if not isinstance(args, dict):
        return None

    if name == "read":
        path = args.get("path")
        if not path:
            return None
        start, end = args.get("start_line"), args.get("end_line")
        if start or end:
            return f"{path}:{start or ''}-{end or ''}"
        return str(path)
    if name in ("write", "edit"):
        return str(args.get("path")) if args.get("path") else None
    if name == "search":
        needle = args.get("pattern") or args.get("glob")
        return f'"{_clip(needle)}"' if needle else None
    if name == "shell":
        cmd = args.get("command")
        return f"$ {_clip(cmd)}" if cmd else None
    if name == "git":
        op = args.get("operation")
        extra = args.get("args")
        if op and isinstance(extra, list) and extra:
            return _clip(f"{op} {' '.join(str(e) for e in extra)}")
        return str(op) if op else None
    if name == "planning":
        op = args.get("op") or args.get("action")
        return str(op) if op else "update"
    if name == "remember":
        return _clip(args.get("text")) if args.get("text") else None
    if name == "search_memory":
        query = args.get("query")
        return f'"{_clip(query)}"' if query else None
    if name == "repo_index":
        bits = [str(b) for b in (args.get("path"), args.get("pattern")) if b]
        return " ".join(bits) if bits else None
    if name == "delegate":
        return _clip(args.get("task")) if args.get("task") else None

    # Generic: surface the first non-empty string argument. This path handles
    # unknown/future tools (e.g. MCP) whose args are not modelled above and may
    # carry secrets (tokens, passwords), so the value is run through the same
    # best-effort secret redaction the memory store uses before display.
    for key, value in args.items():
        if isinstance(value, str) and value:
            return f"{key}={_clip(redact_secrets(value))}"
    return None


def _count_lines(text: str) -> int:
    stripped = text.strip("\n")
    return 0 if stripped == "" else stripped.count("\n") + 1


def summarize_result(name: str, result: object) -> str | None:
    """Return a one-line summary of a tool result (or an error message).

    Duck-typed on the :class:`~forge.tools.base.ToolResult` shape (``ok``,
    ``content``, ``error``, ``meta``) so this module stays decoupled from the
    tools layer. On failure it returns the (clipped) error text; on success it
    returns a compact, tool-appropriate outcome such as ``"42 lines"`` or
    ``"wrote 1200 bytes"``. Returns ``None`` when there is nothing to add.
    """
    ok = getattr(result, "ok", True)
    content = getattr(result, "content", "") or ""
    meta = getattr(result, "meta", None) or {}

    if not ok:
        err = getattr(result, "error", None) or content
        return _clip(err, 120) or "failed"

    truncated = " (truncated)" if meta.get("truncated") else ""

    if name in ("read", "search", "repo_index", "git", "shell", "search_memory"):
        n = _count_lines(content)
        return f"{n} line{'s' if n != 1 else ''}{truncated}"
    if name == "write":
        written = meta.get("bytes_written")
        return f"wrote {written} bytes" if written is not None else "written"
    if name == "edit":
        return "edited"
    if name == "planning":
        return "plan updated"
    if name == "remember":
        return "saved"
    if name == "delegate":
        return f"{len(content)} chars"
    return f"{len(content)} chars{truncated}" if content else None


class Ui:
    """Helper class for colored output, diff rendering, and spinners."""

    def __init__(self, out, *, color: bool, spinner: bool):
        self.out = out
        is_tty = getattr(out, "isatty", lambda: False)()
        self.use_rich = _RICH and color and is_tty
        self.use_spinner = _RICH and spinner and is_tty

        if self.use_rich:
            self.console = Console(file=out, force_terminal=True)
        else:
            self.console = None

    def tool_announcement(self, name: str) -> str:
        """Return a colored "[tool: name]" announcement string."""
        if self.use_rich and self.console:
            with self.console.capture() as capture:
                self.console.print(f"[bold blue]\\[tool: {name}][/bold blue]", end="")
            return "\n" + capture.get()
        return f"\n[tool: {name}]"

    def tool_call(self, name: str, detail: str | None = None) -> str:
        """Return a "[tool: name] detail" announcement, colored under rich.

        Keeps the ``[tool: name]`` prefix (so existing consumers/tests still
        match) and appends a dimmed detail describing the call's target.
        """
        if self.use_rich and self.console:
            with self.console.capture() as capture:
                self.console.print(
                    f"[bold blue]\\[tool: {name}][/bold blue]", end=""
                )
                if detail:
                    self.console.print(f" [dim]{_rich_escape(detail)}[/dim]", end="")
            return "\n" + capture.get()
        return f"\n[tool: {name}]" + (f" {detail}" if detail else "")

    def tool_result_line(self, text: str, kind: str = "ok") -> str:
        """Return an indented result line, colored by ``kind`` under rich.

        ``kind`` is one of ``"ok"`` (green), ``"warn"`` (yellow, for
        denied/forbidden) or ``"error"`` (red). Falls back to an indented
        ``"    -> text"`` line in plain mode.
        """
        if self.use_rich and self.console:
            color = {"ok": "green", "warn": "yellow", "error": "red"}.get(kind, "green")
            with self.console.capture() as capture:
                self.console.print(
                    f"    [dim]\u2514\u2500[/dim] [{color}]{_rich_escape(text)}[/{color}]",
                    end="",
                )
            return capture.get()
        return f"    -> {text}"

    def render_diff(self, diff_text: str) -> None:
        """Render a syntax-highlighted unified diff to out."""
        if self.use_rich and self.console:
            syntax = Syntax(
                diff_text, "diff", theme="monokai", background_color="default"
            )
            self.console.print(syntax)
        else:
            self.out.write(diff_text.rstrip() + "\n")
            self.out.flush()

    def status(self, message: str):
        """Return a spinner context manager, or a no-op fallback."""
        if self.use_spinner and self.console:
            return self.console.status(message)
        return contextlib.nullcontext()

    def banner(self, title: str, rows: list[tuple[str, str]]) -> str:
        """Return a formatted startup banner (colored under rich)."""
        if self.use_rich and self.console:
            with self.console.capture() as capture:
                self.console.print(f"[bold cyan]{_rich_escape(title)}[/bold cyan]")
                for label, value in rows:
                    self.console.print(
                        f"  [dim]{_rich_escape(label)}:[/dim] "
                        f"{_rich_escape(str(value))}"
                    )
            return capture.get().rstrip("\n")
        lines = [title]
        for label, value in rows:
            lines.append(f"  {label}: {value}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear the terminal screen when attached to a real terminal."""
        if self.use_rich and self.console:
            self.console.clear()
        else:
            is_tty = getattr(self.out, "isatty", lambda: False)()
            if is_tty:
                self.out.write("\033[2J\033[H")
                self.out.flush()
