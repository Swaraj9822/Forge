"""Shared workspace path-scoping helper.

The Workspace is the directory tree rooted at the current working directory from
which the CLI is launched; it is the security boundary used by the out-of-scope
path checks in the read, write, edit, and search tools.

``resolve_in_workspace`` canonicalizes a candidate path against the workspace
root and rejects anything that escapes it. Canonicalization uses ``realpath`` so
that ``..`` segments, absolute paths, and symlink-style escapes are all collapsed
before the containment check. The helper works for paths that do not yet exist
(e.g. a write target whose parents are missing): the longest existing prefix is
resolved through symlinks and the remaining components are appended verbatim.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["OutOfWorkspaceError", "resolve_in_workspace"]


class OutOfWorkspaceError(Exception):
    """Raised when a candidate path resolves outside the workspace root.

    Calling tools convert this into an "out of scope" ``ToolResult``.
    """

    def __init__(self, candidate: str | os.PathLike[str], workspace_root: Path):
        self.candidate = os.fspath(candidate)
        self.workspace_root = workspace_root
        super().__init__(
            f"path {self.candidate!r} resolves outside the workspace "
            f"{str(workspace_root)!r}"
        )


def resolve_in_workspace(
    candidate: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Canonicalize ``candidate`` against the workspace root.

    Args:
        candidate: The path supplied to a tool. May be relative (resolved against
            the workspace root) or absolute. Need not exist yet.
        workspace_root: The workspace boundary. Defaults to the current working
            directory from which the CLI was launched.

    Returns:
        The canonical absolute ``Path`` of ``candidate``, guaranteed to lie
        inside (or to be) the workspace root.

    Raises:
        OutOfWorkspaceError: If the canonical path escapes the workspace root.
    """
    root = Path(os.path.realpath(workspace_root if workspace_root is not None else Path.cwd()))

    candidate_path = Path(candidate)
    if not candidate_path.is_absolute():
        candidate_path = root / candidate_path

    resolved = Path(os.path.realpath(candidate_path))

    if not _is_within(resolved, root):
        raise OutOfWorkspaceError(candidate, root)

    return resolved


def _is_within(path: Path, root: Path) -> bool:
    """Return True if ``path`` is the workspace root or nested beneath it."""
    try:
        # ``is_relative_to`` treats ``root`` itself as within ``root``.
        return path.is_relative_to(root)
    except AttributeError:  # pragma: no cover - Python < 3.9 fallback
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False
