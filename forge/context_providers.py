"""Context providers that inject ephemeral, query-conditioned segments.

Providers implement :class:`forge.context.ContextProvider`. Their output is
appended to the assembled context window for a single turn and is never
persisted to the session. This module hosts:

- :class:`PlanReminderProvider` (Feature D) — re-emits the current todo list
- :class:`MemoryProvider` (Feature F) — injects relevant memories each turn
- :class:`RepoMapProvider` (Feature H) — injects a structural repo map
"""

from __future__ import annotations

from pathlib import Path

from forge.session import Session

# Status glyphs kept consistent with the REPL's rendering.
_STATUS_LABEL = {
    "pending": "pending",
    "in_progress": "in progress",
    "completed": "done",
}


class PlanReminderProvider:
    """Re-emit the current todo list so the plan survives compaction (Req: D).

    Returns a single ephemeral ``user`` message summarizing ``session.todos`` so
    the model keeps the active plan in view on long tasks. Returns an empty list
    when there are no todos, so turns without a plan are unaffected.
    """

    HEADER = "[current plan — keep this in mind; do not restate it back to the user]"

    def segments(self, session: Session) -> list[dict]:
        todos = session.todos
        if not todos:
            return []
        lines = [self.HEADER]
        for t in todos:
            label = _STATUS_LABEL.get(t.status, t.status)
            lines.append(f"- ({label}) {t.text}")
        return [{"role": "user", "content": "\n".join(lines)}]


class MemoryProvider:
    """Inject the most relevant memories each turn (Feature F).

    Query-conditioned (uses the latest user message), budgeted (truncate to
    char_budget), and ephemeral (never persisted to session.messages).
    """

    HEADER = "[relevant project memory — background only]"

    def __init__(
        self,
        store: object,
        *,
        limit: int = 5,
        char_budget: int = 2000,
    ) -> None:
        self._store = store
        self._limit = limit
        self._char_budget = char_budget

    def segments(self, session: Session) -> list[dict]:
        """Return ephemeral memory segments conditioned on the latest user text."""
        query = self._latest_user_text(session)
        if not query:
            return []

        # Import here to avoid circular imports
        from forge.memory import MemoryStore

        if not isinstance(self._store, MemoryStore):
            return []

        # Get workspace root from store's workspace_root property
        workspace_root = getattr(self._store, 'workspace_root', None)

        hits = self._store.search(
            query,
            limit=self._limit,
            workspace_root=workspace_root,
        )
        if not hits:
            return []

        body = "\n".join(f"- {h.text}" for h in hits)

        # Truncate to char_budget
        if len(body) > self._char_budget:
            body = body[: self._char_budget - 20] + "\n… (truncated)"

        content = f"{self.HEADER}\n{body}"
        return [{"role": "user", "content": content}]

    @staticmethod
    def _latest_user_text(session: Session) -> str:
        """Scan session.messages in reverse for the last user message text."""
        for msg in reversed(session.messages):
            if msg.role == "user" and msg.text and msg.text.strip():
                return msg.text.strip()
        return ""


class RepoMapProvider:
    """Inject a structural repository map each turn (Feature H).

    Uses the RepoIndexer to build a file/symbol overview, budgeted to char_budget.
    Injected as an ephemeral segment that is never persisted.
    """

    HEADER = "[repository map — file/symbol overview]"

    def __init__(
        self,
        indexer: object,
        *,
        char_budget: int = 4000,
    ) -> None:
        self._indexer = indexer
        self._char_budget = char_budget
        self._last_hash: str | None = None

    def segments(self, session: Session) -> list[dict]:
        """Return the repo map as an ephemeral segment."""
        from forge.repo_index import RepoIndexer

        if not isinstance(self._indexer, RepoIndexer):
            return []

        text = self._indexer.build(budget_chars=self._char_budget)
        if not text:
            return []

        content = f"{self.HEADER}\n{text}"
        return [{"role": "user", "content": content}]
