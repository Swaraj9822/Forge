"""Tests for translating Forge wire-shape messages into SDK ``contents``.

The :class:`~forge.vertex.VertexClient` cannot pass Forge's internal message
dicts (``{"role": "system"/"user"/"model"/"tool", "content": ...}``) to the
``google-genai`` SDK directly: Gemini's ``contents`` only accepts ``"user"`` and
``"model"`` roles with ``parts`` payloads, and the system prompt must travel as
``system_instruction``. :func:`forge.vertex._to_sdk_contents` performs that
translation; these tests pin its behavior.

When ``google-genai`` is not installed the translator returns its input
unchanged, so the structural assertions are skipped in that environment.
"""

from __future__ import annotations

import pytest

from forge.vertex import _to_sdk_contents

try:  # The structural assertions require the real SDK types.
    from google.genai import types as _types
except Exception:  # noqa: BLE001
    _types = None

requires_sdk = pytest.mark.skipif(
    _types is None, reason="google-genai is not installed"
)


def _part_kind(part) -> str:
    if getattr(part, "text", None):
        return "text"
    if getattr(part, "function_call", None) is not None:
        return "function_call"
    if getattr(part, "function_response", None) is not None:
        return "function_response"
    return "?"


@requires_sdk
def test_system_messages_become_system_instruction_not_content() -> None:
    wire = [
        {"role": "system", "content": "You are Forge."},
        {"role": "user", "content": "hi"},
    ]
    contents, system_instruction = _to_sdk_contents(wire)

    assert system_instruction == "You are Forge."
    # Only the user turn remains as content; the system turn is excluded.
    assert [c.role for c in contents] == ["user"]
    assert [_part_kind(p) for p in contents[0].parts] == ["text"]


@requires_sdk
def test_multiple_system_messages_are_joined() -> None:
    wire = [
        {"role": "system", "content": "First."},
        {"role": "system", "content": "Second."},
        {"role": "user", "content": "go"},
    ]
    _contents, system_instruction = _to_sdk_contents(wire)
    assert system_instruction == "First.\n\nSecond."


@requires_sdk
def test_model_tool_call_becomes_function_call_part() -> None:
    wire = [
        {
            "role": "model",
            "content": "calling",
            "tool_calls": [{"id": "c1", "name": "read", "args": {"path": "a.txt"}}],
        },
    ]
    contents, _si = _to_sdk_contents(wire)

    assert [c.role for c in contents] == ["model"]
    kinds = [_part_kind(p) for p in contents[0].parts]
    assert kinds == ["text", "function_call"]
    fc = contents[0].parts[1].function_call
    assert fc.name == "read"
    assert dict(fc.args) == {"path": "a.txt"}


@requires_sdk
def test_tool_result_becomes_user_function_response() -> None:
    wire = [
        {
            "role": "model",
            "content": None,
            "tool_calls": [{"id": "c1", "name": "read", "args": {}}],
        },
        {
            "role": "tool",
            "content": None,
            "tool_result": {
                "call_id": "c1",
                "ok": True,
                "content": "file body",
                "error": None,
                "meta": {},
            },
        },
    ]
    contents, _si = _to_sdk_contents(wire)

    # Gemini only accepts user/model roles, so a tool result is a user turn.
    assert contents[-1].role == "user"
    fr = contents[-1].parts[0].function_response
    # The function name is correlated back from the emitting call's id.
    assert fr.name == "read"


@requires_sdk
def test_parallel_tool_results_coalesce_into_one_turn() -> None:
    """Responses to a parallel-function-call turn travel in one user content.

    Gemini requires the turn following a function-call turn to carry exactly as
    many function-response parts as the call turn carried function-call parts.
    A model turn with N parallel calls is answered by N separate ``role="tool"``
    messages in Forge's history; they must be merged into a single user content
    of N function-response parts, not split across N turns (which the API
    rejects with "the number of function response parts is equal to the number
    of function call parts of the function call turn").
    """

    wire = [
        {
            "role": "model",
            "content": "running tools",
            "tool_calls": [
                {"id": "c1", "name": "shell", "args": {"cmd": "a"}},
                {"id": "c2", "name": "shell", "args": {"cmd": "b"}},
                {"id": "c3", "name": "shell", "args": {"cmd": "c"}},
            ],
        },
        {
            "role": "tool",
            "content": None,
            "tool_result": {"call_id": "c1", "ok": True, "content": "out-a",
                            "error": None, "meta": {}},
        },
        {
            "role": "tool",
            "content": None,
            "tool_result": {"call_id": "c2", "ok": True, "content": "out-b",
                            "error": None, "meta": {}},
        },
        {
            "role": "tool",
            "content": None,
            "tool_result": {"call_id": "c3", "ok": False, "content": "",
                            "error": "boom", "meta": {}},
        },
    ]
    contents, _si = _to_sdk_contents(wire)

    # The model call turn, then a SINGLE user turn answering all three calls.
    assert [c.role for c in contents] == ["model", "user"]

    call_parts = [
        p for p in contents[0].parts
        if getattr(p, "function_call", None) is not None
    ]
    response_parts = [
        p for p in contents[1].parts
        if getattr(p, "function_response", None) is not None
    ]
    # Part counts match — the invariant the Gemini API enforces.
    assert len(call_parts) == len(response_parts) == 3
    # Responses preserve call order and correlate names back from the call ids.
    assert [p.function_response.name for p in response_parts] == [
        "shell", "shell", "shell"
    ]


@requires_sdk
def test_tool_results_split_by_user_message_are_not_coalesced() -> None:
    """A real user turn between tool results breaks the coalescing run."""

    wire = [
        {"role": "model", "content": None,
         "tool_calls": [{"id": "c1", "name": "read", "args": {}}]},
        {"role": "tool", "content": None,
         "tool_result": {"call_id": "c1", "ok": True, "content": "x",
                         "error": None, "meta": {}}},
        {"role": "user", "content": "another question"},
        {"role": "model", "content": None,
         "tool_calls": [{"id": "c2", "name": "read", "args": {}}]},
        {"role": "tool", "content": None,
         "tool_result": {"call_id": "c2", "ok": True, "content": "y",
                         "error": None, "meta": {}}},
    ]
    contents, _si = _to_sdk_contents(wire)

    # Each tool result answers its own single-call turn; nothing is merged
    # across the intervening user turn.
    assert [c.role for c in contents] == [
        "model", "user", "user", "model", "user"
    ]
    assert len(contents[1].parts) == 1  # first tool result alone
    assert len(contents[4].parts) == 1  # second tool result alone


@requires_sdk
def test_empty_message_is_skipped() -> None:
    wire = [
        {"role": "user", "content": ""},  # no parts -> skipped
        {"role": "user", "content": "real"},
    ]
    contents, _si = _to_sdk_contents(wire)
    assert len(contents) == 1
    assert contents[0].parts[0].text == "real"


@requires_sdk
def test_no_system_message_yields_none_instruction() -> None:
    wire = [{"role": "user", "content": "hi"}]
    _contents, system_instruction = _to_sdk_contents(wire)
    assert system_instruction is None
