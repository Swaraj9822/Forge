"""The post-turn Review_Phase for Forge.

An opt-in phase that runs *after* an Agent_Loop turn that changed files. An
independent review agent — a fresh subagent with read-only tools and no access
to the implementer's reasoning — inspects the change against the original task
and plan, judges plan adherence, wiring, scope, and correctness, and returns a
structured verdict. On a "changes requested" verdict its findings are fed back
to the coding agent within a bounded self-correction loop, then the code is
re-reviewed.

Design mirrors :mod:`forge.verification`: pure, property-testable decision logic
(gate, verdict parsing, loop control, feedback/prompt formatting) is separated
from the I/O-bound :class:`ReviewCoordinator` that composes them with the
subagent runner and the shared :class:`~forge.agent.AgentLoop`.

Deterministic vs judgment split: *what changed* is supplied deterministically
(the checkpoint diff of the turn), and *whether tests pass* is left to the
separate verification phase. The reviewer's job is judgment — did the change
follow the plan, is it wired correctly, is its scope appropriate, does it miss
obvious cases — so it is given the diff and read-only tools, not the ability to
run or edit anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from forge.agent import AgentLoop, TurnResult
from forge.session import ReviewRecord, Session, SessionStore, Usage
from forge.subagent import SubAgentRunner
from forge.usage import UsageSummary
from forge.verification import aggregate_usage

__all__ = [
    "ReviewResult",
    "ReviewPhaseResult",
    "ReviewRenderer",
    "ReviewCoordinator",
    "REVIEW_SYSTEM_PROMPT",
    "should_review",
    "should_run_correction",
    "parse_verdict",
    "format_review_feedback",
    "build_review_prompt",
    "original_task",
    "format_plan",
]

# Verdict values.
APPROVED = "approved"
CHANGES_REQUESTED = "changes_requested"

# Bound the diff embedded in the reviewer prompt so a huge change does not blow
# the reviewer's context; the reviewer can still open files with its read tools.
_MAX_DIFF_CHARS = 20_000

# Cap on how many findings are parsed / fed back, to keep the loop focused.
_MAX_FINDINGS = 20


REVIEW_SYSTEM_PROMPT = (
    "You are an independent code reviewer for a coding agent. You did NOT write "
    "the code under review and have no stake in it — review it critically but "
    "fairly against the stated task and plan. Prefer objective, actionable "
    "findings (correctness, missing wiring, unjustified scope, missing cases) "
    "over style nitpicks. You have read-only tools only; never attempt to modify "
    "files or run state-changing commands. Be concise and decisive."
)


@dataclass(frozen=True)
class ReviewResult:
    """The structured outcome of one review pass.

    Attributes
    ----------
    verdict:
        ``"approved"`` or ``"changes_requested"``.
    findings:
        The actionable issues the reviewer raised (empty when approved).
    summary:
        The reviewer's raw response text, retained for display/records.
    """

    verdict: str
    findings: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass(frozen=True)
class ReviewPhaseResult:
    """The aggregate result of a Review_Phase, returned to the caller."""

    ran: bool
    final_result: ReviewResult | None
    iterations_performed: int
    cap_reached: bool
    interrupted: bool
    usage: UsageSummary


# --------------------------------------------------------------------------- #
# Pure decision logic
# --------------------------------------------------------------------------- #


def should_review(
    enabled: bool, trigger: str, mutated_files: bool, turn_ok: bool
) -> bool:
    """Decide whether the Review_Phase should run for a completed turn.

    Runs iff review is enabled AND the turn completed normally AND the trigger
    is satisfied (``"always"``, or ``"on_file_change"`` with a file mutation).
    Reviewing a turn that changed nothing would waste a model call, so
    ``on_file_change`` gates on ``mutated_files``.
    """
    return enabled and turn_ok and (trigger == "always" or mutated_files)


def should_run_correction(
    verdict: str,
    completed_iterations: int,
    max_iterations: int,
    interrupted: bool,
) -> bool:
    """Decide whether another correction iteration should run.

    Another correction runs iff no interrupt has halted the phase, the latest
    verdict requested changes, and fewer than ``max_iterations`` corrections
    have completed. A ``max_iterations`` of ``0`` reviews once with no
    correction (advisory-only).
    """
    return (
        not interrupted
        and verdict == CHANGES_REQUESTED
        and completed_iterations < max_iterations
    )


_VERDICT_RE = re.compile(r"VERDICT\s*[:\-]\s*([A-Za-z_ ]+)", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.*\S)\s*$")


def parse_verdict(text: str) -> ReviewResult:
    """Parse a reviewer's free-text response into a :class:`ReviewResult`.

    Looks for an explicit ``VERDICT: APPROVE`` / ``VERDICT: CHANGES_REQUESTED``
    marker (tolerant of casing and separators) and extracts bulleted/numbered
    findings. The parse fails *open* to ``approved`` when no verdict marker is
    present: review findings are advisory, and defaulting to "approved" avoids
    manufacturing a correction loop from an unparseable response (objective
    signals like tests remain the hard gate). A ``changes_requested`` verdict
    with no listed findings gets one generic finding so the feedback is never
    empty.
    """
    raw = text or ""

    verdict = APPROVED
    match = _VERDICT_RE.search(raw)
    if match:
        token = match.group(1).strip().upper()
        if any(w in token for w in ("CHANGE", "REQUEST", "REJECT", "FAIL", "BLOCK")):
            verdict = CHANGES_REQUESTED
        elif any(w in token for w in ("APPROVE", "PASS", "LGTM", "OK", "ACCEPT")):
            verdict = APPROVED

    findings: list[str] = []
    for line in raw.splitlines():
        m = _BULLET_RE.match(line)
        if m:
            item = m.group(1).strip()
            # Skip a bare "FINDINGS:" style header captured as a bullet.
            if item and not item.upper().startswith("VERDICT"):
                findings.append(item)
        if len(findings) >= _MAX_FINDINGS:
            break

    if verdict == CHANGES_REQUESTED and not findings:
        findings = ["Reviewer requested changes but listed no specific findings."]
    if verdict == APPROVED:
        findings = []

    return ReviewResult(verdict=verdict, findings=findings, summary=raw.strip())


def format_review_feedback(result: ReviewResult) -> str:
    """Render the correction feedback sent to the coding agent (a user turn)."""
    lines = "\n".join(f"- {f}" for f in result.findings) or "- (see review notes)"
    return (
        "An independent code review of your implementation requested changes. "
        "Address the following findings and nothing else:\n"
        "\n"
        f"{lines}\n"
        "\n"
        "Make only the changes needed to resolve these findings; do not introduce "
        "unrelated edits."
    )


def build_review_prompt(task: str, plan: str, diff: str) -> str:
    """Build the reviewer's task prompt from the task, plan, and change diff."""
    plan_block = plan or "(no explicit plan was recorded)"
    diff_block = diff or "(no diff available)"
    return (
        "Review the code change described below.\n"
        "\n"
        f"## Original task\n{task or '(unknown)'}\n"
        "\n"
        f"## Plan\n{plan_block}\n"
        "\n"
        f"## Diff of this change (a/ = before, b/ = after)\n{diff_block}\n"
        "\n"
        "Use your read-only tools to inspect the surrounding code and wiring as "
        "needed, then judge:\n"
        "1. Plan adherence — was the task/plan actually implemented?\n"
        "2. Wiring — is the new code integrated (imported, called, registered) "
        "and not dead?\n"
        "3. Scope — are edits to existing code justified by the task, with no "
        "unrelated or regressive changes?\n"
        "4. Correctness & gaps — obvious bugs, missing error handling, or cases "
        "the change should cover but doesn't.\n"
        "\n"
        "Do not modify files or run commands. End your response with exactly one "
        "of:\n"
        "VERDICT: APPROVE\n"
        "VERDICT: CHANGES_REQUESTED\n"
        "If requesting changes, follow it with a 'FINDINGS:' list of specific, "
        "actionable bullet items."
    )


def original_task(session: Session) -> str:
    """Return the first user message's text — the original task for the session."""
    for msg in session.messages:
        if msg.role == "user" and msg.text:
            return msg.text
    return ""


def format_plan(todos: list[Any]) -> str:
    """Render the session todo plan as a checklist, or ``""`` when empty."""
    if not todos:
        return ""
    return "\n".join(f"- [{t.status}] {t.text}" for t in todos)


# --------------------------------------------------------------------------- #
# Renderer protocol
# --------------------------------------------------------------------------- #


class ReviewRenderer(Protocol):
    """Optional UI hook driven during a Review_Phase (mirrors the verify one)."""

    def on_review_start(self) -> None:
        """Signal that a review pass is starting."""
        ...

    def on_review_result(self, result: ReviewResult) -> None:
        """Signal that a review pass finished with ``result``."""
        ...

    def on_review_correction(self, iteration: int, max_iterations: int) -> None:
        """Signal that correction ``iteration``/``max`` is starting."""
        ...

    def on_review_cap_reached(
        self, result: ReviewResult, iterations: int
    ) -> None:
        """Signal the cap was reached while changes were still requested."""
        ...


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# ReviewCoordinator
# --------------------------------------------------------------------------- #


class ReviewCoordinator:
    """Orchestrate the post-turn Review_Phase.

    Called after an Agent_Loop turn (in the REPL and headless paths). When the
    gate passes it runs an independent reviewer (a fresh subagent with the
    configured read-only tools and :data:`REVIEW_SYSTEM_PROMPT`), parses its
    verdict, and — on "changes requested" — feeds the findings back to the
    coding agent via a correction turn on the shared :class:`AgentLoop`, then
    re-reviews, bounded by ``review.max_iterations``.

    Strictly opt-in: with ``review.enabled = False`` (or the trigger not
    satisfied) the gate short-circuits and a ``ran=False`` result carrying the
    original turn's usage is returned, leaving end-of-turn rendering unchanged.
    """

    def __init__(
        self,
        config: Any,
        *,
        provider: Any,
        agent_loop: AgentLoop,
        session_store: SessionStore,
        interrupt: Any,
        tool_registry: dict,
        workspace_root: Path,
        checkpoint: Any | None = None,
        policy: Any | None = None,
        approver: Any | None = None,
        parent_context: Any | None = None,
        renderer: ReviewRenderer | None = None,
    ) -> None:
        self._config = config            # full Config (needed by the subagent)
        self._review = config.review     # ReviewConfig
        self._provider = provider
        self._agent_loop = agent_loop
        self._session_store = session_store
        self._interrupt = interrupt
        self._tool_registry = tool_registry
        self._workspace_root = workspace_root
        self._checkpoint = checkpoint
        self._policy = policy
        self._approver = approver
        self._parent_context = parent_context
        self._renderer = renderer

    def set_renderer(self, renderer: ReviewRenderer | None) -> None:
        """Replace the progress renderer (used by the headless run path)."""
        self._renderer = renderer

    # -- diff source --------------------------------------------------------

    def _build_diff(self) -> str:
        """Return the change diff for review, from the checkpoint store.

        Uses the checkpoint's per-turn snapshot diff (repo-independent, and
        precise to the ``write``/``edit`` mutations the turn made), capped to
        keep the reviewer prompt bounded. Returns a placeholder when no diff is
        available (no checkpoint, or nothing captured).
        """
        if self._checkpoint is not None:
            try:
                diff = self._checkpoint.diff_last_turn()
            except Exception:  # noqa: BLE001 - diff must never break the phase
                diff = ""
            if diff:
                if len(diff) > _MAX_DIFF_CHARS:
                    return diff[:_MAX_DIFF_CHARS] + "\n… (diff truncated)"
                return diff
        return "(no diff available)"

    # -- reviewer -----------------------------------------------------------

    def _run_review(self, prompt: str) -> ReviewResult:
        """Run the independent reviewer subagent and parse its verdict.

        The reviewer's token usage is folded into the shared usage tracker so
        the session's cumulative usage and cost include the review work.
        """
        max_turns = max(1, int(getattr(self._config, "subagents_max_turns", 4)))
        runner = SubAgentRunner(
            provider=self._provider,
            config=self._config,
            interrupt=self._interrupt,
            tool_registry=self._tool_registry,
            allowed_tools=set(self._review.tools),
            max_turns=max_turns,
            policy=self._policy,
            approver=self._approver,
            parent_context=self._parent_context,
            system_prompt=REVIEW_SYSTEM_PROMPT,
        )
        sub_result = runner.run(prompt, workspace_root=self._workspace_root)
        try:
            self._agent_loop.usage_tracker.record(
                sub_result.usage.input_tokens, sub_result.usage.output_tokens
            )
        except Exception:  # noqa: BLE001 - usage accounting must never break the phase
            pass
        return parse_verdict(sub_result.text)

    # -- orchestration ------------------------------------------------------

    def run(
        self, session: Session, turn_result: TurnResult
    ) -> ReviewPhaseResult:
        """Run the Review_Phase for a completed turn."""
        turn_ok = not (turn_result.interrupted or turn_result.error)
        if not should_review(
            self._review.enabled,
            self._review.trigger,
            turn_result.mutated_files,
            turn_ok,
        ):
            return ReviewPhaseResult(
                ran=False,
                final_result=None,
                iterations_performed=0,
                cap_reached=False,
                interrupted=False,
                usage=turn_result.usage,
            )

        task = original_task(session)
        plan = format_plan(session.todos)

        if self._renderer is not None:
            self._renderer.on_review_start()
        result = self._run_review(build_review_prompt(task, plan, self._build_diff()))
        if self._renderer is not None:
            self._renderer.on_review_result(result)

        completed = 0
        interrupted = False
        while should_run_correction(
            result.verdict, completed, self._review.max_iterations, interrupted
        ):
            if self._renderer is not None:
                self._renderer.on_review_correction(
                    completed + 1, self._review.max_iterations
                )

            feedback = format_review_feedback(result)
            correction = self._agent_loop.run_turn(session, feedback)

            if correction.interrupted or correction.error:
                interrupted = correction.interrupted
                break

            completed += 1

            if self._renderer is not None:
                self._renderer.on_review_start()
            result = self._run_review(
                build_review_prompt(task, plan, self._build_diff())
            )
            if self._renderer is not None:
                self._renderer.on_review_result(result)

        cap_reached = (
            result.verdict == CHANGES_REQUESTED
            and not interrupted
            and self._review.max_iterations > 0
            and completed >= self._review.max_iterations
        )
        if cap_reached and self._renderer is not None:
            self._renderer.on_review_cap_reached(result, completed)

        session.review_records.append(
            ReviewRecord(
                verdict=result.verdict,
                iterations=completed,
                cap_reached=cap_reached,
                findings=len(result.findings),
            )
        )

        # Persist with up-to-date cumulative usage (the reviewer's tokens were
        # recorded into the tracker but no run_turn followed the final review to
        # persist them, so mirror the tracker's cumulative onto the session).
        summary = self._agent_loop.usage_tracker.turn_summary()
        session.usage = Usage(
            input_tokens=summary.cumulative_input_tokens,
            output_tokens=summary.cumulative_output_tokens,
            estimated_cost=summary.cumulative_cost,
        )
        session.updated_at = _utc_now_iso()
        self._session_store.save(session)

        return ReviewPhaseResult(
            ran=True,
            final_result=result,
            iterations_performed=completed,
            cap_reached=cap_reached,
            interrupted=interrupted,
            usage=summary,
        )
