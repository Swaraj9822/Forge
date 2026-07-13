"""Optional rich-backed console rendering with plain-text fallback.

Streaming markdown rendering is out of scope. Markdown cannot be rendered
until a block completes, which forces buffering that fights the token-latency
requirement.
"""

from __future__ import annotations

import contextlib

try:
    from rich.console import Console
    from rich.syntax import Syntax
    _RICH = True
except Exception:  # noqa: BLE001
    _RICH = False


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
