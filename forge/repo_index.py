"""Lightweight, dependency-free repository indexer (Feature G).

Provides a structural map of the codebase using:
- Python AST for .py files (top-level and nested class/function signatures)
- Regex for other languages (best-effort)
- Noise directory pruning
- Symlink safety
- Caching on (path, mtime, size)
- Budget-aware truncation
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any


# Noise directories to skip during walk
DEFAULT_NOISE_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", ".forge",
    "venv", "env", ".env", "dist", "build", ".tox", ".mypy_cache",
})

# Regex patterns for other languages (best-effort, never raise)
_LANG_PATTERNS: list[re.Pattern[str]] = [
    # JavaScript/TypeScript: function/class/const/let/var with function/arrow
    re.compile(
        r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(",
        re.MULTILINE,
    ),
    re.compile(
        r"^(?:export\s+)?class\s+(\w+)",
        re.MULTILINE,
    ),
    # Go: func/type
    re.compile(
        r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(",
        re.MULTILINE,
    ),
    re.compile(
        r"^type\s+(\w+)\s+(?:struct|interface)",
        re.MULTILINE,
    ),
    # Rust: fn/pub fn/struct/enum
    re.compile(
        r"^(?:pub\s+)?fn\s+(\w+)\s*[<(]",
        re.MULTILINE,
    ),
    re.compile(
        r"^(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)",
        re.MULTILINE,
    ),
    # Ruby: def/class/module
    re.compile(
        r"^def\s+(?:self\.)?(\w+)",
        re.MULTILINE,
    ),
    re.compile(
        r"^(?:class|module)\s+(\w+)",
        re.MULTILINE,
    ),
]


def _extract_python_symbols(content: str) -> list[str]:
    """Extract top-level and one-level-nested symbols from Python source.

    Uses stdlib ast for accurate parsing. Returns a list of signature strings
    like "class Foo:" or "def bar(x, y):".
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    symbols: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            # Extract class with bases
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.unparse(base))
            base_str = f"({', '.join(bases)})" if bases else ""
            symbols.append(f"class {node.name}{base_str}:")

            # One level of nesting for methods
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    prefix = (
                        "async "
                        if isinstance(child, ast.AsyncFunctionDef)
                        else ""
                    )
                    args = _format_args(child.args)
                    symbols.append(f"  {prefix}def {child.name}({args}):")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            args = _format_args(node.args)
            symbols.append(f"{prefix}def {node.name}({args}):")

    return symbols


def _format_args(args: ast.arguments) -> str:
    """Format function arguments as a simple string."""
    parts: list[str] = []

    # Positional args
    for arg in args.args:
        parts.append(arg.arg)

    # Keyword-only args
    if args.kwonlyargs:
        if args.posonlyargs or args.args:
            parts.append("*")
        for arg in args.kwonlyargs:
            parts.append(arg.arg)

    # **kwargs
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    return ", ".join(parts)


def _extract_regex_symbols(content: str) -> list[str]:
    """Extract symbols from non-Python files using regex (best-effort)."""
    symbols: list[str] = []
    for pattern in _LANG_PATTERNS:
        try:
            for match in pattern.finditer(content):
                name = match.group(1)
                if name and len(name) < 100:  # sanity check
                    symbols.append(name)
        except Exception:
            continue
    return symbols


class RepoIndexer:
    """Walks a workspace and builds a structural map of files and symbols.

    Features:
    - Dependency-free (stdlib ast + regex)
    - Noise directory pruning
    - Symlink safety (never follows out of root)
    - Caching on (path, mtime, size)
    - Budget-aware truncation
    """

    def __init__(
        self,
        root: Path,
        *,
        output_cap: int = 30_000,
        noise_dirs: frozenset[str] = DEFAULT_NOISE_DIRS,
    ) -> None:
        self._root = root.resolve()
        self._output_cap = output_cap
        self._noise_dirs = noise_dirs
        # Cache: (set of (path, mtime, size), built_text)
        self._cache_key: frozenset[tuple[str, float, int]] | None = None
        self._cache_text: str | None = None

    @property
    def root(self) -> Path:
        return self._root

    def build(self, *, budget_chars: int | None = None) -> str:
        """Return a text map: files with their top-level symbols/signatures.

        Capped to budget_chars (or output_cap) with a truncation marker.
        """
        cap = budget_chars if budget_chars is not None else self._output_cap

        # Check cache
        current_key = self._build_cache_key()
        if self._cache_key is not None and current_key == self._cache_key:
            if self._cache_text is not None:
                return self._truncate(self._cache_text, cap)

        # Build the map
        lines: list[str] = []
        for file_info in self._walk_files():
            rel_path = file_info["relative_path"]
            symbols = file_info["symbols"]

            lines.append(rel_path)
            for sym in symbols:
                lines.append(f"  {sym}")
            lines.append("")  # blank line between files

        text = "\n".join(lines)

        # Update cache
        self._cache_key = current_key
        self._cache_text = text

        return self._truncate(text, cap)

    def _build_cache_key(self) -> frozenset[tuple[str, float, int]]:
        """Build a cache key from (relative_path, mtime, size) of all files."""
        entries: list[tuple[str, float, int]] = []
        for entry in self._walk_files_raw():
            try:
                stat = entry.stat()
                entries.append(
                    (str(entry), stat.st_mtime, stat.st_size)
                )
            except OSError:
                continue
        return frozenset(entries)

    def _walk_files(self) -> list[dict[str, Any]]:
        """Walk the workspace and extract file info with symbols."""
        result: list[dict[str, Any]] = []
        for path in self._walk_files_raw():
            try:
                rel = path.relative_to(self._root)
            except ValueError:
                continue

            info: dict[str, Any] = {
                "absolute_path": path,
                "relative_path": str(rel),
                "symbols": [],
            }

            # Extract symbols based on extension
            ext = path.suffix.lower()
            if ext == ".py":
                try:
                    content = path.read_text(encoding="utf-8")
                    info["symbols"] = _extract_python_symbols(content)
                except (OSError, UnicodeDecodeError):
                    info["symbols"] = []
            elif ext in (".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".rb"):
                try:
                    content = path.read_text(encoding="utf-8")
                    info["symbols"] = _extract_regex_symbols(content)
                except (OSError, UnicodeDecodeError):
                    info["symbols"] = []
            # Other files: no symbols, just the path

            result.append(info)

        # Sort by relative path for determinism
        result.sort(key=lambda x: x["relative_path"])
        return result

    def _walk_files_raw(self) -> list[Path]:
        """Walk the workspace, yielding file paths (no symlinks out of root)."""
        result: list[Path] = []

        for dirpath, dirnames, filenames in os.walk(
            self._root, followlinks=False
        ):
            dir_path = Path(dirpath)

            # Prune noise directories (modify dirnames in-place)
            dirnames[:] = [
                d for d in dirnames
                if d not in self._noise_dirs
            ]

            # Skip hidden directories
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".")
            ]

            for filename in filenames:
                # Skip hidden files
                if filename.startswith("."):
                    continue

                file_path = dir_path / filename

                # Symlink safety: skip if resolved path is outside root
                try:
                    resolved = file_path.resolve()
                    if not str(resolved).startswith(str(self._root)):
                        continue
                except (OSError, ValueError):
                    continue

                result.append(file_path)

        return result

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """Truncate text to max_chars with a marker if needed."""
        if len(text) <= max_chars:
            return text
        # Find a clean break point (end of a line)
        truncated = text[: max_chars - 20]
        last_newline = truncated.rfind("\n")
        if last_newline > 0:
            truncated = truncated[:last_newline]
        return truncated + "\n… (truncated)"
