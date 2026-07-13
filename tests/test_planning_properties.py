"""Property-based tests for the planning / todo tool invariants.

# Feature: forge, Property 17: Todo store, update, and status invariants

These properties exercise :class:`forge.tools.planning.PlanningTool` directly
against the real argument schema the tool expects (``op`` selecting
``set`` / ``update`` / ``clear`` / ``get``) and the real session-scoped state
carried on :attr:`forge.tools.base.ToolContext.state`. They assert the
universal invariants behind Requirement 10:

* 10.1 - a ``set`` of up to 100 items stores exactly those items and returns
  the stored list; more than 100 items is rejected and leaves the list
  unchanged.
* 10.2 - an ``update`` of a present item records the new status and returns the
  updated list (length and other items unchanged).
* 10.4 - every stored item's status stays within {pending, in_progress,
  completed}; an out-of-set status is rejected.
* 10.6 - updating an absent item, or using an out-of-set status, is a no-op
  error that leaves the current list unchanged.

Validates: Requirements 10.1, 10.2, 10.4, 10.6
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from forge.interrupt import InterruptController
from forge.tools.base import ToolContext, ToolResult
from forge.tools.planning import MAX_ITEMS, VALID_STATUSES, PlanningTool

# Sorted view of the only allowed statuses (Req 10.4).
VALID = sorted(VALID_STATUSES)  # ["completed", "in_progress", "pending"]

# Free-form text for item bodies; keeps things simple and JSON-safe.
SAFE_TEXT = st.text(max_size=20)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_ctx() -> ToolContext:
    """A fresh session-scoped context with an empty todo state.

    Each example gets its own ``state={}`` so the todo list is isolated, the
    same way a new Session starts empty (Req 10.5).
    """
    return ToolContext(
        workspace_root=Path.cwd(),
        interrupt=InterruptController(),
        state={},
    )


def run_tool(tool: PlanningTool, args: dict, ctx: ToolContext) -> ToolResult:
    """Run the tool the way the executor does: validate first, then run.

    Mirrors :meth:`forge.tools.base.ToolExecutor.execute`: a non-``None``
    validation error short-circuits to an error result and the tool never runs.
    """
    error = tool.validate(args)
    if error is not None:
        return ToolResult(
            ok=False, content="", error=error, meta={"validation_error": True}
        )
    return tool.run(args, ctx)


def current_todos(tool: PlanningTool, ctx: ToolContext) -> list[dict]:
    """Read the current todo list (serialized dicts) without mutating it."""
    result = run_tool(tool, {"op": "get"}, ctx)
    assert result.ok
    return result.meta["todos"]


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #


@st.composite
def valid_item(draw: st.DrawFn) -> dict:
    """An input item with required ``text`` and optional valid ``status``.

    ``id`` is intentionally omitted so the tool assigns deterministic
    sequential ids ("1".."n"); that keeps update-by-id tests unambiguous.
    """
    item: dict = {"text": draw(SAFE_TEXT)}
    if draw(st.booleans()):
        item["status"] = draw(st.sampled_from(VALID))
    return item


# A status string that is NOT one of the allowed statuses (Req 10.4 violation).
# min_size=1 keeps it clearly out-of-set (an empty string would be treated as a
# default on ``set`` but is still out-of-set on ``update``).
invalid_status = st.text(min_size=1, max_size=12).filter(lambda s: s not in VALID_STATUSES)


# --------------------------------------------------------------------------- #
# Property 17a - store invariant (Req 10.1, 10.4)
# --------------------------------------------------------------------------- #


@settings(max_examples=10, deadline=None)
@given(items=st.lists(valid_item(), max_size=MAX_ITEMS))
def test_set_stores_up_to_max_items_and_returns_list(items: list[dict]) -> None:
    """A ``set`` of up to 100 valid items stores exactly those items and
    returns the stored list; every stored status is within the allowed set.

    Validates: Requirements 10.1, 10.4
    """
    tool = PlanningTool()
    ctx = make_ctx()

    result = run_tool(tool, {"op": "set", "items": items}, ctx)

    assert result.ok
    assert result.meta["todos_changed"] is True

    stored = result.meta["todos"]
    # Same count and same texts in the same order (Req 10.1).
    assert len(stored) == len(items)
    assert [t["text"] for t in stored] == [i["text"] for i in items]
    # Every stored status is constrained to the allowed set (Req 10.4).
    assert all(t["status"] in VALID_STATUSES for t in stored)
    # Ids are unique so items remain individually addressable.
    assert len({t["id"] for t in stored}) == len(stored)

    # The list persists across a later read within the same session.
    assert current_todos(tool, ctx) == stored


@settings(max_examples=10, deadline=None)
@given(
    existing=st.lists(valid_item(), max_size=10),
    overflow=st.lists(valid_item(), min_size=MAX_ITEMS + 1, max_size=MAX_ITEMS + 10),
)
def test_set_over_cap_is_rejected_and_leaves_list_unchanged(
    existing: list[dict], overflow: list[dict]
) -> None:
    """A ``set`` of more than 100 items is rejected and leaves any existing
    list unchanged.

    Validates: Requirements 10.1
    """
    tool = PlanningTool()
    ctx = make_ctx()

    # Seed a known, valid current list first.
    seeded = run_tool(tool, {"op": "set", "items": existing}, ctx)
    assert seeded.ok
    before = current_todos(tool, ctx)

    result = run_tool(tool, {"op": "set", "items": overflow}, ctx)

    assert result.ok is False
    assert result.error is not None
    assert result.meta["todos_changed"] is False
    # The previously stored list is untouched (Req 10.1).
    assert current_todos(tool, ctx) == before


# --------------------------------------------------------------------------- #
# Property 17b - update invariant (Req 10.2, 10.4)
# --------------------------------------------------------------------------- #


@settings(max_examples=10, deadline=None)
@given(
    items=st.lists(valid_item(), min_size=1, max_size=20),
    new_status=st.sampled_from(VALID),
    index=st.integers(min_value=0),
)
def test_update_present_item_records_status_and_keeps_rest(
    items: list[dict], new_status: str, index: int
) -> None:
    """Updating a present item records the new status and returns the updated
    list, leaving the list length and the other items unchanged.

    Validates: Requirements 10.2, 10.4
    """
    tool = PlanningTool()
    ctx = make_ctx()

    stored = run_tool(tool, {"op": "set", "items": items}, ctx).meta["todos"]
    target = stored[index % len(stored)]
    target_id = target["id"]

    result = run_tool(
        tool, {"op": "update", "id": target_id, "status": new_status}, ctx
    )

    assert result.ok
    assert result.meta["todos_changed"] is True

    updated = result.meta["todos"]
    # Same number of items; the list was not truncated or grown (Req 10.2).
    assert len(updated) == len(stored)

    by_id = {t["id"]: t for t in updated}
    # The targeted item now holds the new (allowed) status (Req 10.2/10.4).
    assert by_id[target_id]["status"] == new_status
    assert by_id[target_id]["status"] in VALID_STATUSES

    # Every other item is byte-for-byte unchanged.
    for prev in stored:
        if prev["id"] == target_id:
            continue
        assert by_id[prev["id"]] == prev


# --------------------------------------------------------------------------- #
# Property 17c - no-op error invariants (Req 10.6, 10.4)
# --------------------------------------------------------------------------- #


@settings(max_examples=10, deadline=None)
@given(
    items=st.lists(valid_item(), max_size=20),
    absent_id=st.text(min_size=1, max_size=12),
    new_status=st.sampled_from(VALID),
)
def test_update_absent_item_is_noop_error(
    items: list[dict], absent_id: str, new_status: str
) -> None:
    """Updating an item that is not in the current list returns a not-found
    error and leaves the current list unchanged.

    Validates: Requirements 10.6
    """
    tool = PlanningTool()
    ctx = make_ctx()

    stored = run_tool(tool, {"op": "set", "items": items}, ctx).meta["todos"]
    existing_ids = {t["id"] for t in stored}
    # Ensure the chosen id really is absent.
    assume(absent_id not in existing_ids)

    result = run_tool(
        tool, {"op": "update", "id": absent_id, "status": new_status}, ctx
    )

    assert result.ok is False
    assert result.error is not None
    assert result.meta["todos_changed"] is False
    # The list is left exactly as it was (Req 10.6).
    assert current_todos(tool, ctx) == stored


@settings(max_examples=10, deadline=None)
@given(
    items=st.lists(valid_item(), min_size=1, max_size=20),
    index=st.integers(min_value=0),
    bad_status=invalid_status,
)
def test_update_out_of_set_status_is_noop_error(
    items: list[dict], index: int, bad_status: str
) -> None:
    """Updating a present item to a status outside the allowed set returns an
    error and leaves the current list unchanged.

    Validates: Requirements 10.4, 10.6
    """
    tool = PlanningTool()
    ctx = make_ctx()

    stored = run_tool(tool, {"op": "set", "items": items}, ctx).meta["todos"]
    target_id = stored[index % len(stored)]["id"]

    result = run_tool(
        tool, {"op": "update", "id": target_id, "status": bad_status}, ctx
    )

    assert result.ok is False
    assert result.error is not None
    assert result.meta["todos_changed"] is False
    # No item ever holds the out-of-set status; list is unchanged (Req 10.4/10.6).
    after = current_todos(tool, ctx)
    assert after == stored
    assert all(t["status"] in VALID_STATUSES for t in after)
