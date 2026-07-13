"""The post-turn Verification_Phase for Forge.

This module hosts the opt-in Verification_Phase that runs after an Agent_Loop
turn completes normally. It runs the user-configured Verify_Command, classifies
the outcome, and -- on failure -- feeds the captured output back to the Model
and re-runs the turn within a bounded self-correction loop.

The design separates pure, property-testable decision logic (outcome
classification, trigger gating, loop control, feedback formatting, usage
aggregation) from the I/O-bound runner and coordinator. This file is built up
incrementally: it begins with the result data models and the pure
``classify_outcome`` helper; later work adds the loop-control helpers, feedback
formatting, usage aggregation, and the ``VerificationRunner`` /
``VerificationCoordinator`` that compose them.

See the design document's "Data Models" and "VerificationRunner /
classify_outcome" sections. Requirements: 4.2, 4.3, 4.4, 4.5, 4.7.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from forge.agent import AgentLoop, TurnResult
from forge.config import VerificationConfig
from forge.interrupt import InterruptController
from forge.session import Session, SessionStore, VerificationRecord
from forge.tools.shell import CommandExecution, _cap, _render, execute_command
from forge.usage import UsageSummary

__all__ = [
    "VerificationResult",
    "VerificationPhaseResult",
    "VerificationRunner",
    "VerificationRenderer",
    "VerificationCoordinator",
    "classify_outcome",
    "should_verify",
    "should_run_correction",
    "format_feedback",
    "aggregate_usage",
]

# Verification_Result outcome statuses.
PASSED = "passed"
FAILED = "failed"
TIMED_OUT = "timed_out"
START_ERROR = "start_error"


@dataclass(frozen=True)
class VerificationResult:
    """The structured outcome of one Verify_Command execution.

    Attributes
    ----------
    outcome:
        One of ``"passed"``, ``"failed"``, ``"timed_out"``, or
        ``"start_error"`` (see :func:`classify_outcome`).
    exit_code:
        The command's exit code when the process ran to completion, else
        ``None`` (timed out, interrupted, or could not start).
    output:
        The captured combined output, possibly truncated to the configured
        output cap.
    truncated:
        ``True`` when the rendered combined output exceeded the configured
        output cap and was truncated.
    """

    outcome: str
    exit_code: int | None
    output: str
    truncated: bool = False


@dataclass(frozen=True)
class VerificationPhaseResult:
    """The aggregate result of a Verification_Phase, returned to the Repl.

    Attributes
    ----------
    ran:
        ``False`` when the gate skipped the phase (e.g. no Verify_Command, or
        the trigger condition was not satisfied); ``True`` otherwise.
    final_result:
        The final :class:`VerificationResult` produced by the phase, or
        ``None`` when the phase did not run.
    iterations_performed:
        The number of Correction_Iterations completed within the phase.
    cap_reached:
        ``True`` when the phase ended at Max_Correction_Iterations without a
        passing result.
    interrupted:
        ``True`` when a user interrupt halted the phase.
    usage:
        The usage summary aggregated across the original turn and all
        Correction_Iterations.
    """

    ran: bool
    final_result: VerificationResult | None
    iterations_performed: int
    cap_reached: bool
    interrupted: bool
    usage: UsageSummary


def classify_outcome(execution: CommandExecution) -> str:
    """Map a raw :class:`CommandExecution` to a Verification_Result outcome.

    The classification is a pure, total function over the execution flags,
    evaluated in priority order (Req 4.2, 4.3, 4.4, 4.5, 4.7):

    * a process that could not be started (``spawn_error``) -> ``"start_error"``;
    * otherwise a timed-out execution -> ``"timed_out"`` (regardless of whether
      process-tree termination itself succeeded);
    * otherwise an exit code of ``0`` -> ``"passed"``;
    * otherwise -> ``"failed"`` (the exit code and captured combined output are
      preserved by the caller on the :class:`VerificationResult`).
    """
    if execution.spawn_error is not None:
        return START_ERROR
    if execution.timed_out:
        return TIMED_OUT
    if execution.exit_code == 0:
        return PASSED
    return FAILED


def should_verify(
    command_present: bool,
    trigger: str,
    mutated_files: bool,
    turn_ok: bool,
) -> bool:
    """Decide whether the Verification_Phase should run for a completed turn.

    This is the gate evaluated first by the coordinator. The phase runs if and
    only if a Verify_Command is configured AND the turn completed normally AND
    the trigger condition is satisfied (Req 2.1, 2.2, 3.1, 3.2, 3.3, 3.4):

    * ``command_present`` -- a Verify_Command is configured (absent disables the
      feature, so the gate is always ``False``);
    * ``turn_ok`` -- the turn completed normally (not interrupted, no error);
    * the trigger is satisfied when it is ``"always"`` or, for
      ``"on_file_change"``, when the turn produced at least one File_Mutation
      (``mutated_files``).

    When this returns ``False`` the coordinator starts no Verify_Command process
    and appends no Verification_Feedback, preserving the unconfigured behavior.
    """
    return command_present and turn_ok and (trigger == "always" or mutated_files)


def should_run_correction(
    latest_outcome: str,
    completed_iterations: int,
    max_iterations: int,
    interrupted: bool,
) -> bool:
    """Decide whether another Correction_Iteration should run.

    This pure predicate drives the bounded self-correction loop so that it
    always terminates (Req 5.1, 5.2, 5.4, 5.6, 6.1, 6.3, 8.4). Another
    Correction_Iteration runs if and only if:

    * no interrupt has halted the phase (``not interrupted``);
    * the most recent outcome is correctable -- one of ``"failed"`` or
      ``"timed_out"``. A ``"passed"`` result terminates the loop, and a
      ``"start_error"`` is non-correctable (editing code will not make an
      unstartable command start);
    * fewer than ``max_iterations`` Correction_Iterations have completed.

    A ``max_iterations`` of ``0`` yields ``False`` immediately, so the initial
    Verify_Command runs once with no Correction_Iteration.
    """
    return (
        not interrupted
        and latest_outcome in {FAILED, TIMED_OUT}
        and completed_iterations < max_iterations
    )


def format_feedback(command: str, result: VerificationResult) -> str:
    """Render the Verification_Feedback text for a non-passing Verify_Command.

    This is a pure, deterministic renderer: the same ``command`` and
    :class:`VerificationResult` always produce the same string. The synthesized
    text becomes the ``user_text`` of a Correction_Iteration turn so the Model
    sees the failure details and can fix the underlying problem (Req 7.1, 7.2).

    The rendered message includes (Req 7.1):

    * the Verify_Command that was run;
    * the classified outcome status (``result.outcome``);
    * the exit code, or the literal ``"unavailable"`` when
      ``result.exit_code`` is ``None`` (timed out, interrupted, or could not
      start);
    * the captured combined output (``result.output``).

    The output header is ``"Output (truncated):"`` exactly when
    ``result.truncated`` is ``True`` (the combined output exceeded the
    configured output cap), and ``"Output:"`` otherwise (Req 7.2).
    """
    exit_code = "unavailable" if result.exit_code is None else str(result.exit_code)
    output_header = "Output (truncated):" if result.truncated else "Output:"
    return (
        "The verification command failed. Please fix the underlying problem.\n"
        "\n"
        f"Command: {command}\n"
        f"Status: {result.outcome}\n"
        f"Exit code: {exit_code}\n"
        f"{output_header}\n"
        f"{result.output}"
    )


def aggregate_usage(turn_usages: list[UsageSummary]) -> UsageSummary:
    """Aggregate per-turn usage across the turn and all Correction_Iterations.

    The Verification_Phase issues additional Model requests during its
    Correction_Iterations, each of which the :class:`~forge.usage.UsageTracker`
    records into both the per-turn and cumulative totals. ``turn_usages`` is the
    ordered list of per-turn :class:`~forge.usage.UsageSummary` snapshots -- the
    original turn first, followed by one snapshot per Correction_Iteration -- and
    this pure function folds them into the single summary the phase reports and
    persists (Req 10.1, 10.2, 10.3).

    The aggregation rule mirrors how the tracker already carries totals:

    * the aggregated *turn* token counts are the sum of every input summary's
      ``turn_input_tokens`` / ``turn_output_tokens`` (the per-turn tallies do not
      accumulate across turns, so they must be summed -- Req 10.1, 10.2);
    * the aggregated *cumulative* token counts are those of the FINAL summary,
      since the tracker already carries cumulative totals forward across turns,
      so the last snapshot holds the latest session totals (Req 10.2, 10.3);
    * ``cost_available`` is ``True`` only when every input summary reports cost
      as available; otherwise cost cannot be totalled and both cost fields are
      ``None`` (consistent with the tracker's "cost unavailable" handling);
    * when cost is available, ``turn_cost`` is the sum of each summary's
      ``turn_cost`` and ``cumulative_cost`` is the FINAL summary's
      ``cumulative_cost`` (the latest cumulative estimate carried forward).

    An empty ``turn_usages`` list yields a fully zeroed summary with
    ``cost_available=False`` and both cost fields ``None`` -- there are no
    requests to total and no final snapshot to carry, so the conservative,
    cost-unavailable zero summary is returned.
    """
    if not turn_usages:
        return UsageSummary(
            turn_input_tokens=0,
            turn_output_tokens=0,
            cumulative_input_tokens=0,
            cumulative_output_tokens=0,
            turn_cost=None,
            cumulative_cost=None,
            cost_available=False,
        )

    turn_input = sum(usage.turn_input_tokens for usage in turn_usages)
    turn_output = sum(usage.turn_output_tokens for usage in turn_usages)

    final = turn_usages[-1]
    cost_available = all(usage.cost_available for usage in turn_usages)

    if cost_available:
        turn_cost: float | None = sum(
            usage.turn_cost or 0.0 for usage in turn_usages
        )
        cumulative_cost = final.cumulative_cost
    else:
        turn_cost = None
        cumulative_cost = None

    return UsageSummary(
        turn_input_tokens=turn_input,
        turn_output_tokens=turn_output,
        cumulative_input_tokens=final.cumulative_input_tokens,
        cumulative_output_tokens=final.cumulative_output_tokens,
        turn_cost=turn_cost,
        cumulative_cost=cumulative_cost,
        cost_available=cost_available,
    )


class VerificationRunner:
    """Execute the Verify_Command and produce a :class:`VerificationResult`.

    The runner is the I/O-bound bridge between the Verification_Phase and the
    shared shell execution core. It reuses the exact machinery the ``shell``
    tool uses -- the platform default shell rooted at the Workspace, the
    wall-clock timeout, process-tree termination, sub-second interrupt polling,
    and the combined-output character cap -- so a Verify_Command runs under the
    same safety limits as any other shell command (Req 4.1).

    It is constructed once with the Workspace root and the shared
    :class:`~forge.interrupt.InterruptController`, then invoked per
    Verify_Command run by the :class:`VerificationCoordinator`.
    """

    def __init__(
        self, workspace_root: Path, interrupt: InterruptController
    ) -> None:
        """Bind the runner to a Workspace root and the shared interrupt controller.

        Parameters
        ----------
        workspace_root:
            The directory the Verify_Command runs in (the Workspace).
        interrupt:
            The shared :class:`~forge.interrupt.InterruptController` the shell
            core polls, so a Ctrl-C during the Verify_Command trips within one
            second. The coordinator brackets each run with
            ``begin_turn`` / ``end_turn`` so this controller is armed.
        """
        self._workspace_root = workspace_root
        self._interrupt = interrupt

    def run(
        self, command: str, *, timeout_s: int, output_cap: int
    ) -> VerificationResult:
        """Run ``command`` once and classify it into a :class:`VerificationResult`.

        The Verify_Command is executed through the shared
        :func:`~forge.tools.shell.execute_command` core and the raw
        :class:`~forge.tools.shell.CommandExecution` is mapped to an outcome via
        the pure :func:`classify_outcome` helper (Req 4.2, 4.3, 4.4, 4.5, 4.7).

        The captured combined output is rendered and capped reusing the shell
        core's :func:`~forge.tools.shell._render` and
        :func:`~forge.tools.shell._cap` helpers, so capping and truncation match
        :class:`~forge.tools.shell.ShellTool` exactly; the result is flagged
        ``truncated`` when the rendered combined output exceeded ``output_cap``
        (Req 4.6).

        When the process could not be started (``spawn_error``), the outcome is
        ``start_error``, the exit code is ``None``, and ``output`` carries a
        description of the failure (Req 4.7).
        """
        execution: CommandExecution = execute_command(
            command,
            workspace_root=self._workspace_root,
            interrupt=self._interrupt,
            timeout_s=timeout_s,
            output_cap=output_cap,
        )

        outcome = classify_outcome(execution)

        # A command that could not start has no streams to render; surface the
        # failure description as the output and leave the exit code unavailable.
        if outcome == START_ERROR:
            return VerificationResult(
                outcome=START_ERROR,
                exit_code=None,
                output=f"failed to start command: {execution.spawn_error}",
                truncated=False,
            )

        output, truncated = _cap(
            _render(execution.stdout, execution.stderr, execution.exit_code),
            output_cap,
        )

        return VerificationResult(
            outcome=outcome,
            exit_code=execution.exit_code,
            output=output,
            truncated=truncated,
        )


class VerificationRenderer(Protocol):
    """The optional UI hook the coordinator drives during a Verification_Phase.

    This mirrors the existing :class:`~forge.agent.Renderer` pattern: it is an
    optional, structurally-typed collaborator the coordinator invokes only when
    a renderer is wired. The concrete implementation lives on the ``Repl`` (see
    the design's "VerificationRenderer + Repl integration" section); the
    coordinator depends on this Protocol so it stays decoupled from the REPL and
    so the surfacing of progress indicators is independently testable (Req 9).

    The four hooks correspond to the user-visible phase milestones:

    * :meth:`on_verification_start` -- a Verify_Command is about to run (9.1);
    * :meth:`on_verification_result` -- a Verify_Command finished and was
      classified (9.2, 9.3);
    * :meth:`on_correction_iteration` -- a Correction_Iteration ``n`` of ``max``
      is starting (9.4);
    * :meth:`on_verification_cap_reached` -- the phase ended at the iteration
      cap without a passing result (9.5).
    """

    def on_verification_start(self, command: str) -> None:
        """Signal that ``command`` is about to be executed (Req 9.1)."""
        ...

    def on_verification_result(self, result: VerificationResult) -> None:
        """Signal that a Verify_Command finished with ``result`` (Req 9.2, 9.3)."""
        ...

    def on_correction_iteration(self, iteration: int, max_iterations: int) -> None:
        """Signal that Correction_Iteration ``iteration``/``max`` starts (Req 9.4)."""
        ...

    def on_verification_cap_reached(
        self, result: VerificationResult, iterations: int
    ) -> None:
        """Signal the cap was reached without a pass (Req 9.5)."""
        ...


class VerificationCoordinator:
    """Orchestrate the post-turn Verification_Phase.

    The coordinator is the single entry point the Repl calls after an
    Agent_Loop turn completes. It composes the pure decision helpers
    (:func:`should_verify`, :func:`should_run_correction`,
    :func:`format_feedback`, :func:`aggregate_usage`) with the I/O-bound
    :class:`VerificationRunner` and :class:`~forge.agent.AgentLoop` to drive the
    bounded self-correction loop, then records, persists, and reports the
    outcome (see the design's "VerificationCoordinator.run algorithm").

    The phase is strictly opt-in: when no Verify_Command is configured (or the
    trigger is not satisfied) the gate short-circuits and the coordinator
    returns a "not run" result carrying the original turn's usage, leaving the
    REPL's end-of-turn rendering and persistence exactly as they are today
    (Req 2, 3).

    A correction turn is just another :meth:`AgentLoop.run_turn` call whose
    "user" message is the synthesized Verification_Feedback, so it reuses the
    same context assembly, streaming, tool execution, message persistence, and
    interrupt bracketing (Req 7.3, 11.1). The Verify_Command itself runs
    *outside* ``run_turn``, so the coordinator brackets each run with
    :meth:`InterruptController.begin_turn` / :meth:`~InterruptController.end_turn`
    so a Ctrl-C during verification trips within one second (Req 8.1).
    """

    def __init__(
        self,
        config: VerificationConfig,
        runner: VerificationRunner,
        agent_loop: AgentLoop,
        session_store: SessionStore,
        interrupt: InterruptController,
        renderer: VerificationRenderer | None = None,
    ) -> None:
        """Bind the coordinator to its collaborators.

        Parameters
        ----------
        config:
            The resolved :class:`~forge.config.VerificationConfig`. An absent
            ``command`` disables the phase at the gate.
        runner:
            The :class:`VerificationRunner` that executes the Verify_Command.
        agent_loop:
            The shared :class:`~forge.agent.AgentLoop`; each Correction_Iteration
            is a :meth:`~forge.agent.AgentLoop.run_turn` call on it.
        session_store:
            The :class:`~forge.session.SessionStore` used to persist the session
            (with its appended :class:`~forge.session.VerificationRecord`) before
            returning.
        interrupt:
            The shared :class:`~forge.interrupt.InterruptController`; the
            coordinator brackets each Verify_Command run with its
            ``begin_turn`` / ``end_turn`` so verification observes Ctrl-C.
        renderer:
            An optional :class:`VerificationRenderer` driven for the user-visible
            progress indicators; when ``None`` no indicators are surfaced.
        """
        self._config = config
        self._runner = runner
        self._agent_loop = agent_loop
        self._session_store = session_store
        self._interrupt = interrupt
        self._renderer = renderer

    def set_renderer(self, renderer: "VerificationRenderer | None") -> None:
        """Replace the progress renderer (used by the headless run path)."""
        self._renderer = renderer

    def _run_verify(self) -> tuple[VerificationResult, bool]:
        """Run the Verify_Command once, bracketed for interrupt observation.

        The run is wrapped in :meth:`InterruptController.begin_turn` /
        :meth:`~InterruptController.end_turn` so the shell core's sub-second
        interrupt polling is armed and a Ctrl-C trips within one second
        (Req 8.1). The interrupt state is read via
        :meth:`InterruptController.check` *before* ``end_turn`` clears it, so the
        caller learns whether the user interrupted during this verification.

        Returns the :class:`VerificationResult` and a flag that is ``True`` when
        an interrupt was observed during the run.
        """
        self._interrupt.begin_turn()
        try:
            result = self._runner.run(
                self._config.command,
                timeout_s=self._config.timeout_s,
                output_cap=self._config.output_cap_chars,
            )
            interrupted = self._interrupt.check()
        finally:
            self._interrupt.end_turn()
        return result, interrupted

    def run(
        self, session: Session, turn_result: TurnResult
    ) -> VerificationPhaseResult:
        """Run the Verification_Phase for a completed turn.

        Follows the design's "VerificationCoordinator.run algorithm": gate,
        run the initial Verify_Command once, drive the bounded self-correction
        loop, then record/aggregate/persist and return.

        The gate (:func:`should_verify`) is evaluated first; when it is false
        the coordinator starts no process and appends no feedback, returning a
        ``ran=False`` result that carries the original turn's usage so the REPL
        renders and persists exactly as today (Req 2, 3).

        When gated in, the initial Verify_Command runs exactly once -- even when
        ``max_correction_iterations`` is ``0`` -- before any Correction_Iteration
        (Req 5.6, 5.7). The loop then feeds the failure back to the Model via a
        correction turn and re-runs the Verify_Command, bounded by
        :func:`should_run_correction` so it always terminates (Req 5, 6). An
        interrupted or errored correction turn halts the loop (Req 8.2, 8.4); an
        interrupt observed during a Verify_Command halts it too (Req 8.1).

        On completion the final :class:`VerificationRecord` is appended to the
        session, the per-turn usages are aggregated via :func:`aggregate_usage`,
        and the session is persisted with :meth:`SessionStore.save` (the atomic
        write completes before returning -- Req 8.3, 11.2) before the
        :class:`VerificationPhaseResult` is returned.
        """
        command_present = self._config.command is not None
        turn_ok = not (turn_result.interrupted or turn_result.error)
        if not should_verify(
            command_present,
            self._config.trigger,
            turn_result.mutated_files,
            turn_ok,
        ):
            return VerificationPhaseResult(
                ran=False,
                final_result=None,
                iterations_performed=0,
                cap_reached=False,
                interrupted=False,
                usage=turn_result.usage,
            )

        # Per-turn usage snapshots, original turn first, one per correction turn.
        usages: list[UsageSummary] = [turn_result.usage]

        if self._renderer is not None:
            self._renderer.on_verification_start(self._config.command)
        latest, interrupted = self._run_verify()
        if self._renderer is not None:
            self._renderer.on_verification_result(latest)

        completed_iterations = 0
        while should_run_correction(
            latest.outcome,
            completed_iterations,
            self._config.max_correction_iterations,
            interrupted,
        ):
            if self._renderer is not None:
                self._renderer.on_correction_iteration(
                    completed_iterations + 1,
                    self._config.max_correction_iterations,
                )

            feedback = format_feedback(self._config.command, latest)
            correction = self._agent_loop.run_turn(session, feedback)
            usages.append(correction.usage)

            # An interrupted or errored correction turn is a halt condition: we
            # begin no new iteration. Key off TurnResult flags rather than the
            # raw event, which run_turn's end_turn has already cleared (Req 8.2,
            # 8.4).
            if correction.interrupted or correction.error:
                interrupted = correction.interrupted
                break

            completed_iterations += 1

            if self._renderer is not None:
                self._renderer.on_verification_start(self._config.command)
            latest, verify_interrupted = self._run_verify()
            if self._renderer is not None:
                self._renderer.on_verification_result(latest)
            if verify_interrupted:
                interrupted = True
                break

        cap_reached = (
            latest.outcome != PASSED
            and not interrupted
            and completed_iterations >= self._config.max_correction_iterations
            and latest.outcome in {FAILED, TIMED_OUT}
        )
        if cap_reached and self._renderer is not None:
            self._renderer.on_verification_cap_reached(latest, completed_iterations)

        record = VerificationRecord(
            command=self._config.command,
            outcome=latest.outcome,
            exit_code=latest.exit_code,
            iterations=completed_iterations,
            cap_reached=cap_reached,
            truncated=latest.truncated,
        )
        session.verification_records.append(record)

        usage = aggregate_usage(usages)
        self._session_store.save(session)

        return VerificationPhaseResult(
            ran=True,
            final_result=latest,
            iterations_performed=completed_iterations,
            cap_reached=cap_reached,
            interrupted=interrupted,
            usage=usage,
        )
