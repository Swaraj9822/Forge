"""Repo index tool: ``repo_index`` (Feature G).

Provides a lightweight, dependency-free structural map of the repository
that the model can call to understand the codebase layout.
"""

from __future__ import annotations

from typing import Any

from forge.tools.base import Tool, ToolContext, ToolResult
from forge.tools.paths import OutOfWorkspaceError, resolve_in_workspace


class RepoIndexTool:
    """The ``repo_index`` tool: return a structural map of the repository.

    Implements the :class:`~forge.tools.base.Tool` protocol.

    Classified as ``read_only=True`` since it only reads the filesystem.
    """

    name = "repo_index"
    description = (
        "Return a structural map of the repository showing files and their "
        "top-level symbols (classes, functions). Useful for understanding "
        "codebase layout. Optionally limit to a subtree or filter by pattern."
    )
    read_only = True
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Optional workspace-relative path to limit the index to "
                    "a subtree."
                ),
            },
            "pattern": {
                "type": "string",
                "description": (
                    "Optional glob pattern to filter files (e.g., '*.py')."
                ),
            },
        },
        "required": [],
    }

    def validate(self, args: dict) -> str | None:
        """Validate arguments."""
        if not isinstance(args, dict):
            return "Arguments must be an object."

        path = args.get("path")
        if path is not None and not isinstance(path, str):
            return "Argument 'path' must be a string."

        pattern = args.get("pattern")
        if pattern is not None and not isinstance(pattern, str):
            return "Argument 'pattern' must be a string."

        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Build and return the repository index."""
        from pathlib import Path

        from forge.repo_index import RepoIndexer

        # Get config for output cap
        config = getattr(ctx, "config", None)
        output_cap = getattr(config, "output_cap_chars", 30_000)
        if not isinstance(output_cap, int) or output_cap <= 0:
            output_cap = 30_000

        # Determine root directory
        root = ctx.workspace_root
        path_arg = args.get("path")
        if path_arg:
            try:
                root = resolve_in_workspace(path_arg, ctx.workspace_root)
            except OutOfWorkspaceError as exc:
                return ToolResult(
                    ok=False,
                    content="",
                    error=f"Path is out of scope: {exc.candidate}",
                    meta={"out_of_scope": True},
                )

        # Build the indexer
        indexer = RepoIndexer(root, output_cap=output_cap)

        # Build the index
        try:
            text = indexer.build(budget_chars=output_cap)
        except Exception as exc:
            return ToolResult(
                ok=False,
                content="",
                error=f"Failed to build repository index: {exc}",
                meta={"io_error": True},
            )

        if not text:
            return ToolResult(
                ok=True,
                content="No files found in the repository.",
                error=None,
                meta={},
            )

        # Apply pattern filter if specified (post-filter for simplicity)
        pattern = args.get("pattern")
        if pattern:
            import fnmatch

            lines = text.split("\n")
            filtered_lines: list[str] = []
            current_file = ""
            include_current = False

            for line in lines:
                if line and not line.startswith("  "):
                    # This is a file line
                    current_file = line
                    include_current = fnmatch.fnmatch(current_file, pattern)
                    if include_current:
                        filtered_lines.append(line)
                elif include_current:
                    # This is a symbol line under a matched file
                    filtered_lines.append(line)
                # else: skip symbol lines under non-matched files

            text = "\n".join(filtered_lines)

        if not text.strip():
            return ToolResult(
                ok=True,
                content="No files matched the pattern.",
                error=None,
                meta={},
            )

        truncated = len(text) >= output_cap
        meta: dict[str, Any] = {}
        if truncated:
            meta["truncated"] = True

        return ToolResult(
            ok=True,
            content=text,
            error=None,
            meta=meta,
        )


# Static assertion
_REPO_INDEX_TOOL_IS_A_TOOL: type[Tool] = RepoIndexTool  # type: ignore[assignment]
