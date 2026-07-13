"""Non-interactive (headless) execution for `forge -p`.

Runs a single prompt to completion (the agent loop already iterates until the
model emits no tool calls), optionally runs the post-turn verification phase,
and renders the result as plain text or a single JSON object. Kept independent
of bootstrap so it is unit-testable with fake collaborators (mirroring the
agent-loop tests).
"""

from __future__ import annotations

import json
from typing import TextIO

from forge.agent import AgentLoop, NullRenderer, TurnResult
from forge.policy import AutoApprover, DenyMutationsApprover
from forge.session import Session
from forge.usage import UsageSummary

# Exit codes (documented contract for CI).
EXIT_OK = 0
EXIT_TURN_ERROR = 2
EXIT_INTERRUPTED = 3
EXIT_VERIFICATION_FAILED = 4


class _CapturingRenderer:
    """Accumulates streamed model text; optionally echoes it to a stream."""

    def __init__(self, echo: TextIO | None = None) -> None:
        self._parts: list[str] = []
        self._echo = echo

    def on_text(self, text: str) -> None:
        self._parts.append(text)
        if self._echo is not None:
            self._echo.write(text)
            self._echo.flush()

    def on_tool(self, name: str) -> None:
        if self._echo is not None:
            self._echo.write(f"\n[tool: {name}]\n")
            self._echo.flush()

    def on_compaction(self, info) -> None:  # noqa: ANN001 - matches Renderer
        return None

    @property
    def text(self) -> str:
        return "".join(self._parts)


class _VerifyText:
    """Minimal verification renderer that prints [verify] lines (text mode)."""

    def __init__(self, out: TextIO) -> None:
        self._out = out

    def on_verification_start(self, command: str) -> None:
        self._out.write(f"\n[verify] running: {command}\n")
        self._out.flush()

    def on_verification_result(self, result) -> None:  # noqa: ANN001
        status = (
            "passed" if result.outcome == "passed" else f"failed ({result.outcome})"
        )
        self._out.write(f"[verify] {status}\n")
        self._out.flush()

    def on_correction_iteration(self, iteration: int, max_iterations: int) -> None:
        self._out.write(f"[verify] correction {iteration}/{max_iterations}\n")
        self._out.flush()

    def on_verification_cap_reached(self, result, iterations: int) -> None:  # noqa: ANN001
        self._out.write(
            f"[verify] cap reached ({iterations}); final: {result.outcome}\n"
        )
        self._out.flush()


def run_headless(
    agent_loop: AgentLoop,
    session: Session,
    verification_coordinator,
    prompt: str,
    *,
    output: str = "text",
    out: TextIO,
    yes: bool = False,
) -> int:
    """Run one prompt to completion and render/serialize the result.

    In ``text`` mode the model's response streams to ``out`` live; in ``json``
    mode nothing is written until a single JSON object is emitted at the end.
    Verification (when configured and gated in) runs after the turn; its
    correction-turn text is NOT included in the reported ``response``.

    ``yes`` controls the headless approver (Phase 2, Feature B). When ``True``
    every gated call is auto-approved (matching the autopilot behavior);
    when ``False`` any gated mutation is denied so the run cannot hang on a
    prompt it cannot answer.
    """

    # Wire the approver on the executor before the turn runs. The executor
    # is constructed without an approver at bootstrap time; the run path
    # decides which one to use based on ``--yes``.
    tool_executor = getattr(agent_loop, "tool_executor", None)
    if tool_executor is not None and hasattr(tool_executor, "set_approver"):
        tool_executor.set_approver(
            AutoApprover() if yes else DenyMutationsApprover()
        )

    # Renderer: echo live only in text mode.
    renderer = _CapturingRenderer(echo=out if output == "text" else None)
    agent_loop.renderer = renderer
    # Silence verification progress in json mode; allow it in text mode.
    if verification_coordinator is not None:
        verification_coordinator.set_renderer(
            _VerifyText(out) if output == "text" else None
        )

    result: TurnResult = agent_loop.run_turn(session, prompt)
    response_text = renderer.text  # snapshot BEFORE verification correction turns

    # Swap to a null renderer so correction turns don't pollute the response.
    agent_loop.renderer = NullRenderer()

    phase = None
    turn_ok = not (result.interrupted or result.error)
    if verification_coordinator is not None and turn_ok:
        phase = verification_coordinator.run(session, result)

    usage = phase.usage if (phase is not None and phase.ran) else result.usage
    code = _exit_code(result, phase)

    if output == "json":
        _emit_json(out, session, result, phase, usage, response_text, code)
    else:
        _emit_text_footer(out, result, phase, usage)
    return code


def _exit_code(result: TurnResult, phase) -> int:
    if result.error:
        return EXIT_TURN_ERROR
    if result.interrupted:
        return EXIT_INTERRUPTED
    if phase is not None and phase.ran and phase.final_result is not None:
        if phase.final_result.outcome != "passed":
            return EXIT_VERIFICATION_FAILED
    return EXIT_OK


def _usage_dict(u: UsageSummary) -> dict:
    return {
        "turn_input_tokens": u.turn_input_tokens,
        "turn_output_tokens": u.turn_output_tokens,
        "cumulative_input_tokens": u.cumulative_input_tokens,
        "cumulative_output_tokens": u.cumulative_output_tokens,
        "turn_cost": u.turn_cost,
        "cumulative_cost": u.cumulative_cost,
        "cost_available": u.cost_available,
    }


def _emit_json(out, session, result, phase, usage, response_text, code) -> None:
    verification = None
    if phase is not None and phase.ran:
        fr = phase.final_result
        verification = {
            "ran": True,
            "outcome": (fr.outcome if fr is not None else None),
            "iterations": phase.iterations_performed,
            "cap_reached": phase.cap_reached,
        }
    payload = {
        "session_id": session.id,
        "ok": code == EXIT_OK,
        "response": response_text,
        "error": result.error,
        "interrupted": result.interrupted,
        "mutated_files": result.mutated_files,
        "usage": _usage_dict(usage),
        "verification": verification,
        "todos": [{"id": t.id, "text": t.text, "status": t.status} for t in session.todos],
    }
    out.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    out.flush()


def _emit_text_footer(out, result, phase, usage) -> None:
    out.write("\n")
    if result.error:
        out.write(f"[error] {result.error}\n")
    elif result.interrupted:
        out.write("[interrupted]\n")
    # Reuse the same usage wording as the REPL for consistency.
    tok = (
        f"turn: {usage.turn_input_tokens} in / {usage.turn_output_tokens} out | "
        f"session: {usage.cumulative_input_tokens} in / "
        f"{usage.cumulative_output_tokens} out"
    )
    cost = (
        f"cost: ${usage.turn_cost:.6f} turn / ${usage.cumulative_cost:.6f} session"
        if usage.cost_available
        else "cost unavailable"
    )
    out.write(f"[usage] {tok} | {cost}\n")
    out.flush()
