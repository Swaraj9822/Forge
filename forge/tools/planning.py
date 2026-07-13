"""The planning / todo tracking tool.

This module implements the built-in ``planning`` tool (Requirement 10). The
tool lets the Model plan multi-step work by maintaining a *session-scoped*
todo list: it can replace the whole list, update the status of a single item,
clear the list, or read the current list back.

Session scoping (Req 10.5)
--------------------------
The todo list is stored in :attr:`ToolContext.state` under the ``"todos"`` key
rather than as instance state on the tool. ``ToolContext`` is created once per
session and shared across every :meth:`Tool.run` call in that session (see
:class:`forge.tools.base.ToolExecutor`), so storing the list there is what
makes it persist across turns within a session and reset between sessions.
Keeping it off the tool instance also means a single registered ``PlanningTool``
can serve concurrent/independent sessions without leaking state between them.

A small in-instance fallback (:attr:`PlanningTool._fallback_state`) is provided
only for the degenerate case where a context without a usable ``state`` mapping
is passed; normal wiring always supplies ``ctx.state``.

REPL render signal (Req 10.3)
-----------------------------
Rendering the list is the REPL/agent layer's job, not this tool's. To let that
layer detect a change, every *successful mutating* call returns
``meta={"todos_changed": True, "todos": [...]}``; the current serialized list is
also included in ``meta["todos"]`` on every successful call (including the
read-only ``get`` op, which sets ``todos_changed`` to ``False``). The agent loop
inspects ``meta["todos_changed"]`` to decide when to re-render.

Operations
----------
The operation is selected by the ``op`` argument (``action`` is accepted as an
alias):

``set`` (aliases: ``replace``, ``plan``)
    Replace the current list with a provided list of up to 100 items and return
    the stored list (Req 10.1). More than 100 items is rejected with an error
    result that leaves any existing list unchanged (Req 10.1).

``update`` (alias: ``status``)
    Update the status of the item identified by ``id`` and return the updated
    list (Req 10.2). An out-of-set status (Req 10.4/10.6) or an unknown id
    (Req 10.6) is rejected with an error result that leaves the list unchanged.

``clear`` (alias: ``reset``)
    Replace the current list with an empty list (Req 10.5).

``get`` (aliases: ``list``, ``show``)
    Return the current list without modifying it.

Item shape
----------
Each stored item has ``id``, ``text``, and ``status``. When setting items the
caller supplies ``text`` (required) and may supply ``status`` (defaults to
``"pending"``) and ``id`` (a sequential id is generated when absent). Items are
stored as :class:`forge.session.TodoItem` instances so they serialize uniformly
with the rest of the session; the tool returns them as plain dicts in
``ToolResult`` content/meta.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from forge.session import TodoItem
from forge.tools.base import ToolContext, ToolResult

__all__ = ["PlanningTool", "VALID_STATUSES", "MAX_ITEMS"]

# The only statuses a todo item may hold (Req 10.4).
VALID_STATUSES: frozenset[str] = frozenset({"pending", "in_progress", "completed"})

# The maximum number of items a single ``set`` may store (Req 10.1).
MAX_ITEMS: int = 100

# Operation aliases mapped to their canonical operation name.
_OP_ALIASES: dict[str, str] = {
    "set": "set",
    "replace": "set",
    "plan": "set",
    "update": "update",
    "status": "update",
    "clear": "clear",
    "reset": "clear",
    "get": "get",
    "list": "get",
    "show": "get",
}

_STATE_KEY = "todos"

_DEFAULT_STATUS = "pending"


@dataclass
class PlanningTool:
    """The ``planning`` tool implementing the :class:`Tool` protocol.

    Maintains a session-scoped todo list in :attr:`ToolContext.state`.
    """

    name: str = "planning"
    description: str = (
        "Plan and track multi-step work as a todo list for the current "
        "session. Use op='set' to (re)store a list of up to 100 task items, "
        "op='update' to change one item's status by id, op='clear' to empty "
        "the list, and op='get' to read the current list. Each item has an id, "
        "text, and a status of 'pending', 'in_progress', or 'completed'. The "
        "list persists across turns until it is replaced or cleared."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "description": (
                        "The operation to perform: 'set' to replace the whole "
                        "list, 'update' to change one item's status, 'clear' to "
                        "empty the list, or 'get' to read the current list."
                    ),
                    "enum": ["set", "update", "clear", "get"],
                },
                "items": {
                    "type": "array",
                    "description": (
                        "For op='set': the task items to store (max 100). Each "
                        "item provides 'text' and optionally 'status' and 'id'."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": sorted(VALID_STATUSES),
                            },
                        },
                        "required": ["text"],
                    },
                },
                "id": {
                    "type": "string",
                    "description": "For op='update': the id of the item to update.",
                },
                "status": {
                    "type": "string",
                    "description": "For op='update': the new status for the item.",
                    "enum": sorted(VALID_STATUSES),
                },
            },
            "required": ["op"],
        }
    )

    # Instance-level fallback used only when a context lacks a usable state map.
    _fallback_state: dict = field(default_factory=dict, repr=False)

    # -- validation (shape/type checks only) ---------------------------------

    def validate(self, args: dict) -> str | None:
        """Validate argument *shape* only; semantic checks happen in :meth:`run`.

        Returns ``None`` when the arguments are well-shaped, else an error
        string. Semantic rules whose violation must leave the list unchanged
        while still returning a descriptive Tool_Result (the >100-items cap, an
        unknown item id, and an out-of-set status) are intentionally handled in
        :meth:`run` so they surface as error :class:`ToolResult`s rather than
        validation failures that prevent the tool from running (Req 10.4/10.6).
        """
        if not isinstance(args, dict):
            return "Arguments must be an object."

        raw_op = args.get("op", args.get("action"))
        if raw_op is None:
            return "Missing required argument 'op'."
        if not isinstance(raw_op, str):
            return "Argument 'op' must be a string."
        if raw_op not in _OP_ALIASES:
            known = ", ".join(sorted(set(_OP_ALIASES)))
            return f"Unknown op '{raw_op}'. Expected one of: {known}."

        op = _OP_ALIASES[raw_op]

        if op == "set":
            items = args.get("items")
            if items is None:
                return "Op 'set' requires an 'items' list."
            if not isinstance(items, list):
                return "Argument 'items' must be a list."
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    return f"Item at index {index} must be an object."
                text = item.get("text")
                if text is None:
                    return f"Item at index {index} is missing 'text'."
                if not isinstance(text, str):
                    return f"Item at index {index} 'text' must be a string."
                status = item.get("status")
                if status is not None and not isinstance(status, str):
                    return f"Item at index {index} 'status' must be a string."
                item_id = item.get("id")
                if item_id is not None and not isinstance(item_id, str):
                    return f"Item at index {index} 'id' must be a string."

        elif op == "update":
            item_id = args.get("id")
            if item_id is None:
                return "Op 'update' requires an 'id'."
            if not isinstance(item_id, str):
                return "Argument 'id' must be a string."
            status = args.get("status")
            if status is None:
                return "Op 'update' requires a 'status'."
            if not isinstance(status, str):
                return "Argument 'status' must be a string."

        return None

    # -- execution -----------------------------------------------------------

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Execute the planning operation and return a :class:`ToolResult`."""
        op = _OP_ALIASES[args.get("op", args.get("action"))]

        if op == "set":
            return self._op_set(args, ctx)
        if op == "update":
            return self._op_update(args, ctx)
        if op == "clear":
            return self._op_clear(ctx)
        # op == "get"
        return self._op_get(ctx)

    # -- operations ----------------------------------------------------------

    def _op_set(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Replace the current list with up to 100 provided items (Req 10.1)."""
        items = args.get("items", [])

        # Semantic cap check: reject >100 items, leaving any existing list
        # unchanged (Req 10.1).
        if len(items) > MAX_ITEMS:
            return self._error_result(
                ctx,
                f"Cannot store {len(items)} items; the maximum is {MAX_ITEMS}.",
            )

        todos: list[TodoItem] = []
        used_ids: set[str] = set()
        for index, item in enumerate(items):
            status = item.get("status") or _DEFAULT_STATUS
            # An out-of-set status anywhere rejects the whole set, leaving the
            # existing list unchanged (Req 10.4).
            if status not in VALID_STATUSES:
                return self._error_result(
                    ctx,
                    f"Invalid status '{status}' for item at index {index}. "
                    f"Expected one of: {', '.join(sorted(VALID_STATUSES))}.",
                )
            item_id = item.get("id") or self._generate_id(index, used_ids)
            used_ids.add(item_id)
            todos.append(TodoItem(id=item_id, text=item["text"], status=status))

        self._set_todos(ctx, todos)
        return self._success_result(ctx, changed=True)

    def _op_update(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Update one item's status by id (Req 10.2, 10.4, 10.6)."""
        item_id = args["id"]
        status = args["status"]

        # Out-of-set status -> error, list unchanged (Req 10.4/10.6).
        if status not in VALID_STATUSES:
            return self._error_result(
                ctx,
                f"Invalid status '{status}'. Expected one of: "
                f"{', '.join(sorted(VALID_STATUSES))}.",
            )

        todos = self._get_todos(ctx)
        target = next((t for t in todos if t.id == item_id), None)

        # Unknown id -> error, list unchanged (Req 10.6).
        if target is None:
            return self._error_result(
                ctx, f"No todo item with id '{item_id}' was found."
            )

        target.status = status
        self._set_todos(ctx, todos)
        return self._success_result(ctx, changed=True)

    def _op_clear(self, ctx: ToolContext) -> ToolResult:
        """Clear the current list (Req 10.5)."""
        self._set_todos(ctx, [])
        return self._success_result(ctx, changed=True)

    def _op_get(self, ctx: ToolContext) -> ToolResult:
        """Return the current list without modifying it."""
        return self._success_result(ctx, changed=False)

    # -- state helpers -------------------------------------------------------

    def _state(self, ctx: ToolContext) -> dict:
        """Return the session-scoped state mapping, falling back if needed."""
        state = getattr(ctx, "state", None)
        if isinstance(state, dict):
            return state
        return self._fallback_state

    def _get_todos(self, ctx: ToolContext) -> list[TodoItem]:
        """Return the current todo list from session state (a live list)."""
        state = self._state(ctx)
        todos = state.get(_STATE_KEY)
        if todos is None:
            todos = []
            state[_STATE_KEY] = todos
        return todos

    def _set_todos(self, ctx: ToolContext, todos: list[TodoItem]) -> None:
        """Store ``todos`` as the current session-scoped todo list."""
        self._state(ctx)[_STATE_KEY] = todos

    # -- result helpers ------------------------------------------------------

    def _success_result(self, ctx: ToolContext, *, changed: bool) -> ToolResult:
        """Build a success result carrying the serialized list in content/meta."""
        serialized = self._serialize(self._get_todos(ctx))
        return ToolResult(
            ok=True,
            content=json.dumps(serialized, ensure_ascii=False),
            error=None,
            meta={"todos_changed": changed, "todos": serialized},
        )

    def _error_result(self, ctx: ToolContext, message: str) -> ToolResult:
        """Build an error result that leaves the list unchanged.

        The current (unchanged) list is still included in ``meta["todos"]`` so
        callers can render it, with ``todos_changed`` set to ``False``.
        """
        serialized = self._serialize(self._get_todos(ctx))
        return ToolResult(
            ok=False,
            content="",
            error=message,
            meta={"todos_changed": False, "todos": serialized},
        )

    @staticmethod
    def _serialize(todos: list[TodoItem]) -> list[dict]:
        """Serialize todo items to plain dicts (id, text, status)."""
        return [
            {"id": t.id, "text": t.text, "status": t.status} for t in todos
        ]

    @staticmethod
    def _generate_id(index: int, used: set[str]) -> str:
        """Generate a stable sequential id not already used within this set."""
        candidate = str(index + 1)
        bump = index + 1
        while candidate in used:
            bump += 1
            candidate = str(bump)
        return candidate
