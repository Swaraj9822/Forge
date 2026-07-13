"""Unit tests for File_Mutation tracking on the turn result (task 3.2).

These tests drive :meth:`forge.agent.AgentLoop.run_turn` with a scripted mock
``VertexClient`` and a configurable ``ToolExecutor`` to verify how the loop sets
:attr:`forge.agent.TurnResult.mutated_files`.

A File_Mutation is defined precisely as a ``ToolResult`` with ``ok=True`` from a
tool named ``write`` or ``edit``. The Verification_Phase reads this flag as the
``on_file_change`` trigger signal. These tests assert:

* a successful ``write`` tool call sets ``mutated_files=True`` (Req 3.1);
* a successful ``edit`` tool call sets ``mutated_files=True`` (Req 3.1);
* a FAILED ``write``/``edit`` (``ok=False``) leaves it ``False`` (Req 3.2);
* a successful call to any other tool (``read``, ``shell``) leaves it ``False``
  (Req 3.2); and
* a turn with no tool calls leaves it ``False`` (Req 3.2).

Collaborators are in-process fakes (no network, no disk). The real
:class:`~forge.usage.UsageTracker` and
:class:`~forge.interrupt.InterruptController` are used.

Validates: Requirements 3.1, 3.2
"""

from __future__ import annotations

from forge.agent import AgentLoop
from forge.interrupt import InterruptController
from forge.session import Session, ToolCall, Usage
from forge.tools.base import ToolResult
from forge.usage import UsageTracker
from forge.vertex import Done, TextDelta, UsageReport


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeContextManager:
    """Minimal context manager: serialize messages, never compact."""

    def assemble(self, session: Session):
        contents = [
            {"role": m.role, "content": m.text} for m in session.messages
        ]
        return contents, None


class ScriptedVertexClient:
    """Yields a scripted list of stream events per ``generate_stream`` call."""

    def __init__(self, responses: list[list]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate_stream(self, contents, tools):
        self.calls += 1
        events = self._responses.pop(0)
        for event in events:
            yield event


class ConfigurableToolExecutor:
    """Returns a result for each call whose ``ok`` is driven by ``ok_by_name``.

    ``ok_by_name`` maps a tool name to the ``ok`` flag the executor should set
    on that tool's result. A name absent from the mapping defaults to a
    successful (``ok=True``) result. This lets a test exercise both successful
    and failed write/edit calls without bespoke executor subclasses.
    """

    def __init__(self, ok_by_name: dict[str, bool] | None = None) -> None:
        self._ok_by_name = ok_by_name or {}
        self.executed: list[str] = []

    def specs(self):
        return []

    def execute(self, call: ToolCall) -> ToolResult:
        self.executed.append(call.name)
        ok = self._ok_by_name.get(call.name, True)
        return ToolResult(
            ok=ok,
            content=f"ran {call.name}" if ok else None,
            error=None if ok else f"{call.name} failed",
        )


class RecordingSessionStore:
    """Captures saved sessions without touching disk."""

    def __init__(self) -> None:
        self.saved: list[Session] = []

    def save(self, session: Session) -> None:
        self.saved.append(session)


def _make_session() -> Session:
    return Session(
        id="session-under-test",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
        messages=[],
        todos=[],
        usage=Usage(input_tokens=0, output_tokens=0, estimated_cost=None),
    )


def _run_turn_with_tools(
    tool_names: list[str],
    *,
    ok_by_name: dict[str, bool] | None = None,
):
    """Drive one turn whose first response emits a call to each named tool.

    The first model response carries one tool call per name; the second
    response carries no tool calls, terminating the turn. Returns the
    ``TurnResult``.
    """

    calls = [
        ToolCall(id=f"call-{i}", name=name, args={})
        for i, name in enumerate(tool_names)
    ]
    first = [TextDelta(text="working"), *calls, Done()]
    second = [TextDelta(text="done"), UsageReport(2, 1), Done()]
    vertex = ScriptedVertexClient([first, second])
    executor = ConfigurableToolExecutor(ok_by_name)
    loop = AgentLoop(
        context_manager=FakeContextManager(),
        vertex_client=vertex,
        tool_executor=executor,
        usage_tracker=UsageTracker(),
        session_store=RecordingSessionStore(),
        interrupt=InterruptController(),
    )
    return loop.run_turn(session=_make_session(), user_text="go")


# --------------------------------------------------------------------------- #
# Successful write / edit set mutated_files (Req 3.1)
# --------------------------------------------------------------------------- #


def test_successful_write_sets_mutated_files() -> None:
    """A successful ``write`` tool call flags the turn as having mutated files.

    Validates: Requirements 3.1
    """

    result = _run_turn_with_tools(["write"])

    assert result.interrupted is False
    assert result.error is None
    assert result.mutated_files is True


def test_successful_edit_sets_mutated_files() -> None:
    """A successful ``edit`` tool call flags the turn as having mutated files.

    Validates: Requirements 3.1
    """

    result = _run_turn_with_tools(["edit"])

    assert result.mutated_files is True


# --------------------------------------------------------------------------- #
# Failed write / edit do NOT set mutated_files (Req 3.2)
# --------------------------------------------------------------------------- #


def test_failed_write_leaves_mutated_files_false() -> None:
    """A ``write`` call returning ``ok=False`` is not a File_Mutation.

    Validates: Requirements 3.2
    """

    result = _run_turn_with_tools(["write"], ok_by_name={"write": False})

    assert result.mutated_files is False


def test_failed_edit_leaves_mutated_files_false() -> None:
    """An ``edit`` call returning ``ok=False`` is not a File_Mutation.

    Validates: Requirements 3.2
    """

    result = _run_turn_with_tools(["edit"], ok_by_name={"edit": False})

    assert result.mutated_files is False


# --------------------------------------------------------------------------- #
# Other successful tools do NOT set mutated_files (Req 3.2)
# --------------------------------------------------------------------------- #


def test_successful_read_leaves_mutated_files_false() -> None:
    """A successful ``read`` call is not a File_Mutation.

    Validates: Requirements 3.2
    """

    result = _run_turn_with_tools(["read"])

    assert result.mutated_files is False


def test_successful_shell_leaves_mutated_files_false() -> None:
    """A successful ``shell`` call is not a File_Mutation.

    Validates: Requirements 3.2
    """

    result = _run_turn_with_tools(["shell"])

    assert result.mutated_files is False


def test_other_tools_with_failed_write_leaves_mutated_files_false() -> None:
    """A batch of non-mutating tools plus a failed write stays unflagged.

    Validates: Requirements 3.2
    """

    result = _run_turn_with_tools(
        ["read", "shell", "write"], ok_by_name={"write": False}
    )

    assert result.mutated_files is False


def test_mixed_batch_with_successful_edit_sets_mutated_files() -> None:
    """A batch mixing non-mutating tools with a successful edit is flagged.

    Validates: Requirements 3.1
    """

    result = _run_turn_with_tools(["read", "edit", "search"])

    assert result.mutated_files is True


# --------------------------------------------------------------------------- #
# A turn with no tool calls leaves mutated_files False (Req 3.2)
# --------------------------------------------------------------------------- #


def test_no_tool_calls_leaves_mutated_files_false() -> None:
    """A turn whose model response carries no tool calls is not flagged.

    Validates: Requirements 3.2
    """

    response = [TextDelta(text="just text"), UsageReport(3, 2), Done()]
    vertex = ScriptedVertexClient([response])
    executor = ConfigurableToolExecutor()
    loop = AgentLoop(
        context_manager=FakeContextManager(),
        vertex_client=vertex,
        tool_executor=executor,
        usage_tracker=UsageTracker(),
        session_store=RecordingSessionStore(),
        interrupt=InterruptController(),
    )

    result = loop.run_turn(session=_make_session(), user_text="hello")

    assert executor.executed == []
    assert result.mutated_files is False
