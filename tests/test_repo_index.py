"""Tests for the RepoIndexer engine and the repo_index tool."""

from __future__ import annotations

from pathlib import Path

from forge.interrupt import InterruptController
from forge.repo_index import RepoIndexer
from forge.tools.base import ToolContext
from forge.tools.repo_index import RepoIndexTool


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_root=tmp_path, interrupt=InterruptController())


def test_lists_python_symbols(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text(
        "class Foo:\n"
        "    def bar(self, x):\n"
        "        return x\n"
        "\n"
        "def top_level(a, b):\n"
        "    return a\n",
        encoding="utf-8",
    )
    text = RepoIndexer(tmp_path).build()
    assert "mod.py" in text
    assert "class Foo:" in text
    assert "def bar(self, x):" in text
    assert "def top_level(a, b):" in text


def test_async_def_is_extracted(tmp_path: Path) -> None:
    """Regression: async functions must not crash symbol extraction."""
    (tmp_path / "a.py").write_text(
        "async def fetch(url):\n    return url\n", encoding="utf-8"
    )
    text = RepoIndexer(tmp_path).build()
    assert "async def fetch(url):" in text


def test_noise_dirs_pruned(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("def a():\n    pass\n", encoding="utf-8")
    noisy = tmp_path / "node_modules"
    noisy.mkdir()
    (noisy / "lib.js").write_text("function x() {}\n", encoding="utf-8")
    text = RepoIndexer(tmp_path).build()
    assert "keep.py" in text
    assert "node_modules" not in text
    assert "lib.js" not in text


def test_broken_python_still_lists_file(tmp_path: Path) -> None:
    """A file that fails to parse is still listed (with no symbols)."""
    (tmp_path / "broken.py").write_text("def (:\n  bad syntax\n", encoding="utf-8")
    text = RepoIndexer(tmp_path).build()
    assert "broken.py" in text


def test_empty_repo_returns_empty(tmp_path: Path) -> None:
    assert RepoIndexer(tmp_path).build().strip() == ""


def test_truncation_marker(tmp_path: Path) -> None:
    for i in range(50):
        (tmp_path / f"m{i}.py").write_text(
            f"def f{i}(aaaaaaaaaa, bbbbbbbbbb):\n    pass\n", encoding="utf-8"
        )
    text = RepoIndexer(tmp_path).build(budget_chars=200)
    assert len(text) <= 200
    assert "… (truncated)" in text


def test_non_python_regex_symbols(tmp_path: Path) -> None:
    (tmp_path / "app.js").write_text(
        "export function handler() {}\nclass Widget {}\n", encoding="utf-8"
    )
    text = RepoIndexer(tmp_path).build()
    assert "app.js" in text
    assert "handler" in text
    assert "Widget" in text


# --------------------------------------------------------------------------- #
# RepoIndexTool
# --------------------------------------------------------------------------- #


def test_tool_is_read_only() -> None:
    assert RepoIndexTool().read_only is True


def test_tool_returns_index(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("def go():\n    pass\n", encoding="utf-8")
    result = RepoIndexTool().run({}, _ctx(tmp_path))
    assert result.ok is True
    assert "x.py" in result.content
    assert "def go():" in result.content


def test_tool_out_of_scope_path(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside"
    result = RepoIndexTool().run({"path": str(outside)}, _ctx(tmp_path))
    assert result.ok is False
    assert result.meta.get("out_of_scope") is True


def test_tool_pattern_filter(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("def a():\n    pass\n", encoding="utf-8")
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    result = RepoIndexTool().run({"pattern": "*.py"}, _ctx(tmp_path))
    assert result.ok is True
    assert "keep.py" in result.content
    assert "note.txt" not in result.content
