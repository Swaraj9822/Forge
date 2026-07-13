"""Property-based test for the ``edit`` tool's uniqueness invariant.

# Feature: forge, Property 11: Edit uniqueness invariant

Property 11 (Validates: Requirements 6.3, 6.4, 6.5): For any file and target
string:

* if the target occurs **exactly once**, the edit tool replaces that single
  occurrence (``meta["replaced"] == 1``) and leaves all other bytes unchanged;
* if it occurs **zero** times, the result is "target not found"
  (``meta["not_found_target"]``);
* if it occurs **more than once**, the result is "ambiguous"
  (``meta["ambiguous"]``);
* and in both non-unique cases the file is left **byte-for-byte unchanged**.

Generation strategy
-------------------
Occurrence counts are made controllable by drawing the ``target`` from one
alphabet (uppercase letters + digits) and the surrounding "filler" segments from
a **disjoint** alphabet (lowercase letters). Because the filler can never share
a character with the target, no filler segment can introduce or split an
occurrence of the target. Content is built as ``target.join(fillers)`` with
``k + 1`` filler segments, inserting the target exactly ``k`` times for
``k in {0, 1, 2, 3}``.

To stay bulletproof regardless of any boundary subtleties, the assertions branch
on the *actual* occurrence count (``content.count(target)``) -- the same
``str.count`` the implementation uses -- rather than on the intended ``k``.

Environment notes
-----------------
A module-scoped workspace directory is created via ``tempfile.mkdtemp`` (the
pytest ``tmp_path`` fixture is intentionally avoided -- it does not combine with
Hypothesis ``@given``, and it is unreliable on this Windows host). The subject
file uses a fixed, non-reserved name and is rewritten for every example.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.interrupt import InterruptController
from forge.tools.base import ToolContext
from forge.tools.fs import EditTool

# --- Module-scoped workspace --------------------------------------------------
_WORKSPACE_ROOT = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_edit_")))
_SUBJECT_NAME = "subject.txt"  # fixed, non-reserved filename
_SUBJECT_PATH = _WORKSPACE_ROOT / _SUBJECT_NAME


@atexit.register
def _cleanup_dir() -> None:
    shutil.rmtree(_WORKSPACE_ROOT, ignore_errors=True)


# --- Generators ---------------------------------------------------------------
# Target chars: uppercase letters + digits. Filler chars: lowercase letters.
# The two alphabets are disjoint, so a filler segment can never contain (or help
# form) the target -- the occurrence count is governed entirely by the inserted
# separators.
_TARGET_ALPHABET = "XYZ0123456789"
_FILLER_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
# Replacement may be any text; it can freely overlap either alphabet.
_REPLACEMENT_ALPHABET = _FILLER_ALPHABET + _TARGET_ALPHABET + " \n"

_targets = st.text(alphabet=_TARGET_ALPHABET, min_size=1, max_size=5)
_replacements = st.text(alphabet=_REPLACEMENT_ALPHABET, min_size=0, max_size=8)
_fillers = st.text(alphabet=_FILLER_ALPHABET, min_size=0, max_size=6)


@st.composite
def _file_and_target(draw):
    """Build (content, target, replacement) with a controlled occurrence count."""
    target = draw(_targets)
    replacement = draw(_replacements)
    k = draw(st.integers(min_value=0, max_value=3))
    # k + 1 filler segments joined by the target yields exactly k insertions.
    fillers = draw(
        st.lists(_fillers, min_size=k + 1, max_size=k + 1)
    )
    content = target.join(fillers)
    return content, target, replacement


def _run_edit(target: str, replacement: str) -> tuple:
    """Run the edit tool against the subject file; return (result, file_bytes)."""
    ctx = ToolContext(
        workspace_root=_WORKSPACE_ROOT, interrupt=InterruptController()
    )
    result = EditTool().run(
        {"path": _SUBJECT_NAME, "target": target, "replacement": replacement},
        ctx,
    )
    return result, _SUBJECT_PATH.read_bytes()


# --- Property -----------------------------------------------------------------
@settings(max_examples=10)
@given(data=_file_and_target())
def test_edit_uniqueness_invariant(data: tuple) -> None:
    content, target, replacement = data

    original_bytes = content.encode("utf-8")
    _SUBJECT_PATH.write_bytes(original_bytes)

    # Branch on the ACTUAL occurrence count (the same str.count the tool uses).
    occurrences = content.count(target)

    result, file_bytes = _run_edit(target, replacement)

    if occurrences == 1:
        # 6.3 - exactly one occurrence: replace it; only that occurrence changes.
        assert result.ok is True
        assert result.meta.get("replaced") == 1
        expected = content.replace(target, replacement, 1).encode("utf-8")
        assert file_bytes == expected
    elif occurrences == 0:
        # 6.4 - zero occurrences: not found; file byte-for-byte unchanged.
        assert result.ok is False
        assert result.meta.get("not_found_target") is True
        assert file_bytes == original_bytes
    else:
        # 6.5 - more than one occurrence: ambiguous; file unchanged.
        assert result.ok is False
        assert result.meta.get("ambiguous") is True
        assert file_bytes == original_bytes
