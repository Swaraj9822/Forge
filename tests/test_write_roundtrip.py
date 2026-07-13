"""Property-based test for the write tool's round-trip behavior.

# Feature: forge, Property 10: Write round-trip, byte count, and parent creation

Property 10 (Validates: Requirements 6.1, 6.2, 5.1): For any content and any
relative path within the Workspace (including paths whose parent directories do
not yet exist), after the write tool runs:

* reading the path back yields the written content (round-trip),
* the reported byte count equals the UTF-8 encoded length of the content, and
* all missing parent directories along the path now exist.

Each generated example builds an isolated workspace via ``tempfile.mkdtemp``
(per the environment note, ``tmp_path`` is unreliable on this host) and cleans
it up afterwards. Generated path segments use a safe lowercase-letter/digit
alphabet so they avoid OS-reserved characters and Windows-reserved device names.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.interrupt import InterruptController
from forge.tools.base import ToolContext
from forge.tools.fs import ReadTool, WriteTool

# --- Generators ---------------------------------------------------------------

# Windows reserved device names (case-insensitive). A safe path segment must
# never equal one of these, even with an extension, so we exclude them.
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

# Safe path segments: lowercase letters and digits only. This avoids path
# separators, the illegal Windows path characters (<>:"/\|?*), control
# characters, and the "." / ".." traversal segments entirely.
_safe_segment = (
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8)
    .filter(lambda s: s.lower() not in _WINDOWS_RESERVED)
)

# A relative path of 1..4 segments joined with "/", so the parent directories
# may not exist yet. An optional ".txt" extension is appended to the final
# segment to exercise both bare and extensioned filenames.
_relative_paths = st.builds(
    lambda parts, ext: "/".join(parts) + (".txt" if ext else ""),
    st.lists(_safe_segment, min_size=1, max_size=4),
    st.booleans(),
)

# Text content including unicode but excluding lone surrogates (category "Cs"),
# which cannot be UTF-8 encoded, and the NUL character (U+0000), which the read
# tool treats as a binary-file marker by contract (Req 5.6) and therefore is
# outside the UTF-8 *text* input space this round-trip property covers. Empty
# content is allowed.
_content = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=0,
    max_size=400,
)


# --- Property -----------------------------------------------------------------


@settings(max_examples=10)
@given(rel=_relative_paths, content=_content)
def test_write_round_trip_byte_count_and_parent_creation(
    rel: str, content: str
) -> None:
    """Write then read yields the same content; bytes and parents are correct."""
    workspace = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_write_")))
    try:
        ctx = ToolContext(
            workspace_root=workspace, interrupt=InterruptController()
        )

        result = WriteTool().run({"path": rel, "content": content}, ctx)

        # 6.1 - the write succeeds and reports the UTF-8 encoded byte count.
        assert result.ok, result.error
        assert result.meta["bytes_written"] == len(content.encode("utf-8"))

        # 6.2 - all missing parent directories along the path now exist.
        target = workspace / rel
        assert target.parent.is_dir()

        # 6.1 / 5.1 - reading the path back yields exactly the written content.
        read_result = ReadTool().run({"path": rel}, ctx)
        assert read_result.ok, read_result.error
        assert read_result.content == content
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
