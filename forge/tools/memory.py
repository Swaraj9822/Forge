"""Memory tools: ``remember`` and ``search_memory`` (Feature F).

Provides tools for the model to store and retrieve durable cross-session
memories. The ``remember`` tool writes to the MemoryStore with secret
redaction; ``search_memory`` retrieves relevant memories with keyword+recency
ranking and path-staleness filtering.

Both tools are classified as ``read_only=True`` for approval purposes since
they only touch Forge's own store, not the user's project files.
"""

from __future__ import annotations

from typing import Any

from forge.tools.base import Tool, ToolContext, ToolResult


class RememberTool:
    """The ``remember`` tool: store a durable memory.

    Implements the :class:`~forge.tools.base.Tool` protocol.

    Classified as ``read_only=True`` for approval purposes because it only
    writes to Forge's own memory store, not the user's project files.
    """

    name = "remember"
    description = (
        "Store a durable memory that persists across sessions. Memories are "
        "automatically redacted for secrets and can be tagged with keywords "
        "and associated file paths. Use this to remember important decisions, "
        "patterns, or project-specific knowledge."
    )
    read_only = True  # Touches only Forge's store, not user files
    parameters: dict = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The memory text to store.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional keywords for better search ranking.",
            },
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional workspace-relative file paths this memory is about. "
                    "Used for staleness detection."
                ),
            },
        },
        "required": ["text"],
    }

    def validate(self, args: dict) -> str | None:
        """Validate arguments."""
        if not isinstance(args, dict):
            return "Arguments must be an object."

        text = args.get("text")
        if text is None:
            return "Missing required argument 'text'."
        if not isinstance(text, str):
            return "Argument 'text' must be a string."
        if not text.strip():
            return "Argument 'text' must not be empty."

        tags = args.get("tags")
        if tags is not None:
            if not isinstance(tags, list):
                return "Argument 'tags' must be an array of strings."
            for t in tags:
                if not isinstance(t, str):
                    return "Argument 'tags' must contain only strings."

        paths = args.get("paths")
        if paths is not None:
            if not isinstance(paths, list):
                return "Argument 'paths' must be an array of strings."
            for p in paths:
                if not isinstance(p, str):
                    return "Argument 'paths' must contain only strings."

        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Store a memory in the MemoryStore."""
        text = args["text"]
        tags = tuple(args.get("tags", ())) if isinstance(args.get("tags"), list) else ()
        paths = tuple(args.get("paths", ())) if isinstance(args.get("paths"), list) else ()

        # Get the MemoryStore from ToolContext
        store = getattr(ctx, "memory", None)
        if store is None:
            return ToolResult(
                ok=False,
                content="",
                error="Memory store is not available.",
                meta={"unavailable": True},
            )

        try:
            record = store.add(text, tags=tags, paths=paths)
        except Exception as exc:
            return ToolResult(
                ok=False,
                content="",
                error=f"Failed to store memory: {exc}",
                meta={"io_error": True},
            )

        return ToolResult(
            ok=True,
            content=f"Memory stored with id: {record.id}",
            error=None,
            meta={"memory_id": record.id},
        )


class SearchMemoryTool:
    """The ``search_memory`` tool: search durable memories.

    Implements the :class:`~forge.tools.base.Tool` protocol.

    Classified as ``read_only=True`` since it only reads from the store.
    """

    name = "search_memory"
    description = (
        "Search durable cross-session memories by keyword. Returns relevant "
        "memories ranked by keyword overlap and recency. Stale memories "
        "(associated with files that changed after creation) are filtered out."
    )
    read_only = True
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query (keywords).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5).",
            },
        },
        "required": ["query"],
    }

    def validate(self, args: dict) -> str | None:
        """Validate arguments."""
        if not isinstance(args, dict):
            return "Arguments must be an object."

        query = args.get("query")
        if query is None:
            return "Missing required argument 'query'."
        if not isinstance(query, str):
            return "Argument 'query' must be a string."

        limit = args.get("limit")
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool):
                return "Argument 'limit' must be an integer."
            if limit < 1:
                return "Argument 'limit' must be >= 1."

        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Search the MemoryStore and return formatted results."""
        query = args["query"]
        limit = args.get("limit", 5)

        store = getattr(ctx, "memory", None)
        if store is None:
            return ToolResult(
                ok=False,
                content="",
                error="Memory store is not available.",
                meta={"unavailable": True},
            )

        try:
            from forge.memory import MemoryStore

            if not isinstance(store, MemoryStore):
                return ToolResult(
                    ok=False,
                    content="",
                    error="Memory store is not available.",
                    meta={"unavailable": True},
                )

            # Get workspace root from store's workspace_root property
            workspace_root = getattr(store, 'workspace_root', None)

            hits = store.search(
                query,
                limit=limit,
                workspace_root=workspace_root,
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                content="",
                error=f"Failed to search memories: {exc}",
                meta={"io_error": True},
            )

        if not hits:
            return ToolResult(
                ok=True,
                content="No relevant memories found.",
                error=None,
                meta={"results": []},
            )

        # Format results as readable text
        lines = [f"Found {len(hits)} relevant memory(ies):"]
        for i, hit in enumerate(hits, 1):
            lines.append(f"\n--- Memory {i} (id: {hit.id}) ---")
            lines.append(hit.text)
            if hit.tags:
                lines.append(f"Tags: {', '.join(hit.tags)}")
            if hit.paths:
                lines.append(f"Paths: {', '.join(hit.paths)}")
            lines.append(f"Created: {hit.created_at}")

        content = "\n".join(lines)
        results_meta = [
            {
                "id": h.id,
                "text": h.text,
                "tags": list(h.tags),
                "paths": list(h.paths),
                "created_at": h.created_at,
            }
            for h in hits
        ]

        return ToolResult(
            ok=True,
            content=content,
            error=None,
            meta={"results": results_meta},
        )


# Static assertions
_REMEMBER_TOOL_IS_A_TOOL: type[Tool] = RememberTool  # type: ignore[assignment]
_SEARCH_MEMORY_TOOL_IS_A_TOOL: type[Tool] = SearchMemoryTool  # type: ignore[assignment]
