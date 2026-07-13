"""Property-based test for steering prompt ordering.

# Feature: forge, Property 22: Steering prompt ordering
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.config import Config
from forge.context import ContextManager, load_default_system_prompt

# Filenames are restricted to safe lowercase letters so we never generate a
# Windows-reserved device name (NUL, CON, PRN, AUX, COM1..9, LPT1..9) as a
# file name on this host.
SAFE_NAME = st.text(
    alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    min_size=1,
    max_size=12,
)

# Steering file contents: arbitrary unicode text (no surrogates/control noise
# that would break round-tripping through a UTF-8 file).
CONTENT_TEXT = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=0x2FFF),
    min_size=0,
    max_size=80,
)


@st.composite
def steering_file_sets(draw: st.DrawFn) -> list[str]:
    """Generate an ordered list of distinguishable steering file contents.

    Each content is prefixed with its ordinal index so that even if Hypothesis
    draws identical bodies, every entry is unique and order is verifiable.
    """

    bodies = draw(st.lists(CONTENT_TEXT, min_size=0, max_size=5))
    return [f"[steering {i}]\n{body}" for i, body in enumerate(bodies)]


@settings(max_examples=10)
@given(steering_file_sets())
def test_steering_prompt_ordering(contents: list[str]) -> None:
    """For any ordered list of existing steering files, the assembled system
    context places the built-in default prompt first, followed by each steering
    file's contents in the configured order.

    Validates: Requirements 15.1, 15.2, 15.3
    """

    tmp_dir = Path(tempfile.mkdtemp(prefix="forge_steering_"))
    try:
        # Write each generated content to a real file using a unique,
        # safe-lowercase-letter filename, preserving the generated order.
        used_names: set[str] = set()
        paths: list[str] = []
        for idx, content in enumerate(contents):
            # Build a unique filename from safe letters plus the index.
            base = "abcdefghijklmnopqrstuvwxyz"[idx % 26]
            name = f"{base}{idx}.md"
            assert name not in used_names
            used_names.add(name)
            file_path = tmp_dir / name
            file_path.write_text(content, encoding="utf-8")
            paths.append(str(file_path))

        config = Config(steering_files=paths)
        manager = ContextManager(config)

        default_prompt = load_default_system_prompt()
        messages = manager.assemble_system_messages()

        # Req 15.2 / 15.3: the built-in default prompt is always first.
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == default_prompt

        # Req 15.1 / 15.2: the remaining messages are exactly the steering file
        # contents, in configured order.
        assert [m["content"] for m in messages[1:]] == contents
        assert all(m["role"] == "system" for m in messages)

        # Req 15.3: with no steering files, the only message is the default.
        if not contents:
            assert len(messages) == 1

        # build_system_prompt starts with the default prompt and contains each
        # steering content in configured order.
        combined = manager.build_system_prompt()
        assert combined.startswith(default_prompt)
        search_from = len(default_prompt)
        for content in contents:
            found = combined.find(content, search_from)
            assert found != -1, "steering content missing or out of order"
            search_from = found + len(content)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
