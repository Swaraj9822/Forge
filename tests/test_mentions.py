"""Tests for @file mention expansion."""

from __future__ import annotations

import tempfile
from pathlib import Path
from hypothesis import given
from hypothesis import strategies as st

from forge.commands import expand_mentions


def test_expand_mentions_basic() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()

        # Create some files
        file_a = root / "a.txt"
        file_a.write_text("Hello A", encoding="utf-8")

        file_b = root / "sub" / "b.txt"
        file_b.parent.mkdir(parents=True, exist_ok=True)
        file_b.write_text("Hello B", encoding="utf-8")

        # Test valid expansion
        text = "Check @a.txt and @sub/b.txt"
        expanded, included, warnings = expand_mentions(text, root, max_bytes=1000)

        assert "--- a.txt ---\n```\nHello A\n```" in expanded
        assert "--- sub/b.txt ---\n```\nHello B\n```" in expanded
        assert "a.txt" in included
        assert "sub/b.txt" in included
        assert not warnings


def test_expand_mentions_nonexistent_and_out_of_scope() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()

        text = "Check @nonexistent.txt and @../secret.txt"
        expanded, included, warnings = expand_mentions(text, root, max_bytes=1000)

        assert expanded == text
        assert not included
        assert len(warnings) == 2
        assert "nonexistent.txt" in warnings[0]
        assert "secret.txt" in warnings[1]


def test_expand_mentions_binary_and_oversized() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()

        # Binary file
        binary_file = root / "binary.bin"
        binary_file.write_bytes(b"\x80\x81\x82")

        # Oversized file
        large_file = root / "large.txt"
        large_file.write_text("A" * 100, encoding="utf-8")

        text = "Read @binary.bin and @large.txt"
        expanded, included, warnings = expand_mentions(text, root, max_bytes=10)

        assert expanded == text
        assert not included
        assert len(warnings) == 2
        assert "binary.bin" in warnings[0]
        assert "large.txt" in warnings[1]


def test_expand_mentions_quoted() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()

        file_space = root / "a b.txt"
        file_space.write_text("space content", encoding="utf-8")

        text = 'Check @"a b.txt"'
        expanded, included, warnings = expand_mentions(text, root, max_bytes=1000)

        assert "--- a b.txt ---\n```\nspace content\n```" in expanded
        assert "a b.txt" in included
        assert not warnings


def test_expand_mentions_code_blocks_ignored() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()

        file_a = root / "a.txt"
        file_a.write_text("Hello A", encoding="utf-8")

        text = (
            "Check @a.txt in normal text but ignore `code @a.txt` or:\n"
            "```python\n"
            "print('@a.txt')\n"
            "```"
        )
        expanded, included, warnings = expand_mentions(text, root, max_bytes=1000)

        # Only the first mention outside code blocks/inline should expand
        assert "--- a.txt ---\n```\nHello A\n```" in expanded
        # The other @a.txt mentions should remain verbatim
        assert "`code @a.txt`" in expanded
        assert "print('@a.txt')" in expanded
        assert len(included) == 1
        assert not warnings


@given(st.text())
def test_expand_mentions_never_crashes(text: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        expanded, included, warnings = expand_mentions(text, root, max_bytes=1000)
        assert isinstance(expanded, str)
        assert isinstance(included, list)
        assert isinstance(warnings, list)
        if "@" not in text:
            assert expanded == text
