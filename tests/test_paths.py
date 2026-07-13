"""Property-based tests for the workspace path-scoping helper.

# Feature: forge, Property 6: Workspace path-scoping invariant

Property 6 (Validates: Requirements 5.4, 6.6): For any path argument that
resolves OUTSIDE the workspace root, ``resolve_in_workspace`` signals
out-of-scope (raises ``OutOfWorkspaceError``) and performs no read/write outside
the workspace; for any path INSIDE the workspace, it returns a resolved path
that is within the root.

The workspace roots are built once via ``tempfile.mkdtemp`` at module scope and
cleaned up at exit. This deliberately avoids combining a function-scoped pytest
``tmp_path`` fixture with Hypothesis ``@given`` (which Hypothesis warns about,
since the fixture is created once and shared across all generated examples).
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from forge.tools.paths import OutOfWorkspaceError, resolve_in_workspace

# --- Module-scoped workspace and a sibling "outside" directory ----------------
# realpath both so that comparisons are stable on platforms where the system
# temp dir itself contains symlinks (e.g. macOS /var -> /private/var) or short
# (8.3) names on Windows.
_WORKSPACE_ROOT = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_ws_")))
_OUTSIDE_ROOT = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_outside_")))


@atexit.register
def _cleanup_dirs() -> None:
    for path in (_WORKSPACE_ROOT, _OUTSIDE_ROOT):
        shutil.rmtree(path, ignore_errors=True)


# A symlink-style escape: a link living INSIDE the workspace that points to the
# outside directory. Its name uses an uppercase letter and a dot so the interior
# path generator (lowercase letters / digits / _ / - only) can never collide
# with it. Symlink creation can require privileges on Windows, so the case is
# skipped when creation is not permitted.
_SYMLINK_NAME = "ESCAPE.LINK"
_SYMLINK_PATH = _WORKSPACE_ROOT / _SYMLINK_NAME
_SYMLINK_AVAILABLE = True
try:
    os.symlink(_OUTSIDE_ROOT, _SYMLINK_PATH, target_is_directory=True)
except (OSError, NotImplementedError, AttributeError):
    _SYMLINK_AVAILABLE = False


# --- Generators ---------------------------------------------------------------
# Safe path segments: never empty, never "." or ".." (the alphabet excludes
# dots), and free of OS-reserved characters.
_safe_segment = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-",
    min_size=1,
    max_size=8,
)

# Interior relative paths that always stay within the workspace root.
_interior_relpaths = st.lists(_safe_segment, min_size=1, max_size=5).map(
    lambda parts: "/".join(parts)
)

# Escaping traversal paths: at least one leading ".." guarantees the result
# lands above the workspace root regardless of any trailing segments.
_traversal_paths = st.builds(
    lambda n, tail: "/".join([".."] * n + tail),
    st.integers(min_value=1, max_value=4),
    st.lists(_safe_segment, min_size=0, max_size=3),
)

# Absolute paths rooted in the sibling "outside" directory.
_absolute_outside = st.lists(_safe_segment, min_size=0, max_size=3).map(
    lambda parts: str(_OUTSIDE_ROOT.joinpath(*parts))
)


# --- Properties ---------------------------------------------------------------
@settings(max_examples=10)
@given(rel=_interior_relpaths)
def test_interior_paths_resolve_within_workspace(rel: str) -> None:
    """Interior paths resolve to a path that is_relative_to the workspace root."""
    resolved = resolve_in_workspace(rel, workspace_root=_WORKSPACE_ROOT)
    assert resolved.is_relative_to(_WORKSPACE_ROOT)


@settings(max_examples=10)
@given(rel=_traversal_paths)
def test_traversal_escapes_raise_out_of_workspace(rel: str) -> None:
    """`../`-style traversal that escapes the root is signalled out-of-scope."""
    with pytest.raises(OutOfWorkspaceError):
        resolve_in_workspace(rel, workspace_root=_WORKSPACE_ROOT)


@settings(max_examples=10)
@given(abspath=_absolute_outside)
def test_absolute_outside_paths_raise_out_of_workspace(abspath: str) -> None:
    """Absolute paths outside the root are signalled out-of-scope."""
    with pytest.raises(OutOfWorkspaceError):
        resolve_in_workspace(abspath, workspace_root=_WORKSPACE_ROOT)


@pytest.mark.skipif(
    not _SYMLINK_AVAILABLE,
    reason="symlink creation requires privileges on this platform",
)
@settings(max_examples=10)
@given(tail=st.lists(_safe_segment, min_size=1, max_size=3))
def test_symlink_escapes_raise_out_of_workspace(tail: list[str]) -> None:
    """A path threaded through an in-workspace symlink to outside is rejected."""
    candidate = "/".join([_SYMLINK_NAME, *tail])
    with pytest.raises(OutOfWorkspaceError):
        resolve_in_workspace(candidate, workspace_root=_WORKSPACE_ROOT)
