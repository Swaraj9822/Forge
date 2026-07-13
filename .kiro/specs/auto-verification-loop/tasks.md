# Implementation Plan: Auto-Verification Loop

## Overview

This plan implements the post-turn Verification_Phase by reusing Forge's existing primitives. It proceeds bottom-up: first extract a shared shell execution core and add the config block, the `mutated_files` turn signal, and session persistence (all independent files), then build the pure decision helpers and data models in a new `forge/verification.py`, then the runner and coordinator that compose them, and finally the renderer and `app.py` wiring. Each step builds on the previous and ends by integrating into the live REPL with no orphaned code.

Implementation language: **Python** (matching the existing Forge codebase and its Hypothesis-based property suites).

## Tasks

- [x] 1. Extract a shared shell execution core
  - [x] 1.1 Extract `execute_command` + `CommandExecution` in `forge/tools/shell.py`
    - Add a frozen `CommandExecution` dataclass (`stdout`, `stderr`, `exit_code`, `timed_out`, `interrupted`, `spawn_error`)
    - Move the existing execution body (platform default shell, workspace rooting, wall-clock timeout, process-tree termination, sub-second interrupt polling, combined-output character cap) into a module-level `execute_command(command, *, workspace_root, interrupt, timeout_s, output_cap)` helper
    - Refactor `ShellTool.run` to delegate to `execute_command` and wrap the result into a `ToolResult` exactly as today (observable behavior unchanged)
    - _Requirements: 4.1, 8.1_

  - [x]* 1.2 Write tests for `execute_command` and the `ShellTool.run` refactor
    - Assert the raw `CommandExecution` fields the runner depends on (exit code, timed_out, spawn_error, capped output)
    - Run existing `test_shell_behavior.py` / `test_shell_output_cap.py` unchanged as the regression gate
    - _Requirements: 4.1_

- [x] 2. Add verification configuration
  - [x] 2.1 Add `VerificationConfig`, `resolve_verification_config`, and `ConfigManager` parsing
    - Add a frozen `VerificationConfig` dataclass to `forge/config.py` (`command`, `max_correction_iterations`, `trigger`, `timeout_s`, `output_cap_chars`) and attach it to `Config` as `verification`
    - Implement pure `resolve_verification_config(raw, *, shell_timeout_s, output_cap_chars)` applying defaults: `command` absent → `None`; `max_correction_iterations` absent → `3`; `trigger` absent → `on_file_change`; `timeout_s` absent → `shell_timeout_s`; `output_cap_chars` inherits `limits.output_cap_chars`
    - Parse the `[verification]` TOML table in `ConfigManager._from_raw` and raise `ConfigError` naming the offending value for a non-integer or `< 0` `max_correction_iterations`, and for a `trigger` not in `{on_file_change, always}` (also naming the allowed set)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.7, 1.8, 1.5, 1.6_

  - [x]* 2.2 Write property test for configuration resolution defaults
    - **Property 1: Configuration resolution applies documented defaults and reads present values**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.7, 1.8**

  - [x]* 2.3 Write property test for configuration validation
    - **Property 2: Configuration validation rejects invalid values**
    - **Validates: Requirements 1.5, 1.6**

- [x] 3. Add File_Mutation tracking to the turn result
  - [x] 3.1 Add `mutated_files` to `TurnResult` in `forge/agent.py`
    - Add `mutated_files: bool = False` to the frozen `TurnResult` dataclass
    - In `_execute_tool_calls`, set a turn-local flag when a `ToolResult` with `ok=True` is produced by a tool named `write` or `edit`, and surface it on the returned `TurnResult`
    - _Requirements: 3.1, 3.2_

  - [x]* 3.2 Write unit tests for File_Mutation tracking
    - Successful write/edit sets `mutated_files`; failed write/edit and other tools do not
    - _Requirements: 3.1, 3.2_

- [x] 4. Persist verification outcomes on the session
  - [x] 4.1 Add `VerificationRecord` and session serialization in `forge/session.py`
    - Add a `VerificationRecord` dataclass (`command`, `outcome`, `exit_code`, `iterations`, `cap_reached`, `truncated`) and `verification_records: list[VerificationRecord]` on `Session`
    - Extend `session_to_dict` / `session_from_dict` with lossless `*_to_dict` / `*_from_dict` helpers so the round-trip equality invariant holds
    - _Requirements: 11.2, 11.3_

  - [x]* 4.2 Write property test for verification record/feedback round-trip
    - **Property 9: Verification records and feedback round-trip losslessly**
    - **Validates: Requirements 11.1, 11.2, 11.3**

- [x] 5. Build verification data models and pure decision helpers (`forge/verification.py`)
  - [x] 5.1 Create `forge/verification.py` with result models and `classify_outcome`
    - Add frozen `VerificationResult` (`outcome`, `exit_code`, `output`, `truncated`) and `VerificationPhaseResult` (`ran`, `final_result`, `iterations_performed`, `cap_reached`, `interrupted`, `usage`)
    - Implement pure `classify_outcome(execution)`: `spawn_error` → `start_error`; `timed_out` → `timed_out`; `exit_code == 0` → `passed`; otherwise `failed` (preserving exit code and combined output)
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.7_

  - [x]* 5.2 Write property test for outcome classification
    - **Property 4: Outcome classification maps execution to status**
    - **Validates: Requirements 4.2, 4.3, 4.4, 4.5, 4.7**

  - [x] 5.3 Add `should_verify` and `should_run_correction` loop-control helpers
    - `should_verify(command_present, trigger, mutated_files, turn_ok)` → `command_present and turn_ok and (trigger == "always" or mutated_files)`
    - `should_run_correction(latest_outcome, completed_iterations, max_iterations, interrupted)` → `not interrupted and latest_outcome in {failed, timed_out} and completed_iterations < max_iterations` (makes `start_error` and `passed` non-correctable; `max == 0` yields `False`)
    - _Requirements: 2.1, 2.2, 3.1, 3.2, 3.3, 3.4, 5.1, 5.2, 5.4, 5.6, 6.1, 6.3, 8.4_

  - [x]* 5.4 Write property test for the trigger gate
    - **Property 3: Trigger decision gates the phase correctly**
    - **Validates: Requirements 2.1, 2.2, 3.1, 3.2, 3.3, 3.4**

  - [x]* 5.5 Write property test for bounded loop control
    - **Property 6: The correction loop is bounded and terminates correctly**
    - **Validates: Requirements 5.1, 5.2, 5.4, 5.5, 5.6, 5.7, 6.1, 6.3, 8.4**

  - [x] 5.6 Add `format_feedback` helper
    - Deterministically render the Verification_Feedback text including the Verify_Command, outcome status, exit code (or an explicit `unavailable` marker), the captured combined output, and a `(truncated)` marker exactly when `result.truncated`
    - _Requirements: 7.1, 7.2_

  - [x]* 5.7 Write property test for feedback formatting
    - **Property 7: Verification_Feedback includes the failure details**
    - **Validates: Requirements 7.1, 7.2**

  - [x] 5.8 Add `aggregate_usage` helper
    - Sum per-turn token counts across the original turn and all correction turns and carry the final cumulative totals, returning a single `UsageSummary`
    - _Requirements: 10.1, 10.2, 10.3_

  - [x]* 5.9 Write property test for phase usage aggregation
    - **Property 8: Phase usage aggregates every Model request**
    - **Validates: Requirements 10.1, 10.2, 10.3**

- [x] 6. Checkpoint - core helpers and models complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement the Verification_Runner
  - [x] 7.1 Implement `VerificationRunner` in `forge/verification.py`
    - `__init__(workspace_root, interrupt)` and `run(command, *, timeout_s, output_cap)`
    - Call the shared `execute_command`, map the result via `classify_outcome`, reuse the shell core's cap logic, and set `truncated=True` when the rendered combined output exceeded the cap
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [x]* 7.2 Write property test for output capping
    - **Property 5: Output capping truncates to the configured cap**
    - **Validates: Requirements 4.6, 7.2**

  - [x]* 7.3 Write integration tests for runner shell reuse and timeout
    - Trivial `exit 0` → `passed`; trivial non-zero exit → `failed` with exit code and output; a command sleeping past a short `timeout_s` → `timed_out` with the process tree terminated (1–3 representative examples)
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 8. Implement the Verification_Coordinator
  - [x] 8.1 Implement `VerificationCoordinator.run` in `forge/verification.py`
    - Gate with `should_verify`; on false return `VerificationPhaseResult(ran=False, ...)` carrying the original turn usage with no process started and no feedback appended
    - Bracket each Verify_Command run with `interrupt.begin_turn()` / `end_turn()`; run the initial command once even when `max_correction_iterations == 0`
    - Loop on `should_run_correction`: render the iteration indicator, build feedback via `format_feedback`, run `agent_loop.run_turn(session, feedback)`, break on the correction turn's `TurnResult.interrupted` or `TurnResult.error` (an errored correction turn is a halt condition), increment completed iterations, re-run the Verify_Command, break if the runner observed an interrupt
    - On cap reached without a pass, flag `cap_reached`; build and attach the `VerificationRecord`, aggregate usage via `aggregate_usage`, persist the session via `SessionStore.save` (atomic write completes before returning), and return the `VerificationPhaseResult`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 6.1, 6.2, 6.3, 7.3, 8.1, 8.2, 8.3, 8.4, 10.2, 10.3, 11.1, 11.2_

  - [x]* 8.2 Write coordinator loop ordering and bound tests
    - Drive a scripted runner (always-failing, fail-then-pass) with a mock `AgentLoop`: assert initial verify runs before any correction, feedback→run_turn→re-verify ordering, iteration counts, stop on first `passed`, zero corrections when `max == 0`, and cap-reached surfacing with the final result and count
    - _Requirements: 5.3, 5.5, 6.1, 6.2, 7.3_

  - [x]* 8.3 Write interrupt-retention unit test
    - A scripted interrupt mid-phase begins no new iteration and leaves previously appended feedback intact with the persisted session complete
    - _Requirements: 8.2, 8.3, 8.4_

  - [x]* 8.4 Write interrupt-timing integration tests
    - A long-running Verify_Command is terminated within ~1 second of an interrupt (process tree torn down, control returned to the prompt)
    - An interrupt during a correction turn halts the Verification_Phase within ~1 second and begins no new iteration
    - _Requirements: 8.1, 8.2_

- [x] 9. Checkpoint - runner and coordinator complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Surface verification status through the renderer
  - [x] 10.1 Add `VerificationRenderer` protocol and `Repl` integration in `forge/repl.py`
    - Define the optional, defensively-invoked `VerificationRenderer` protocol (`on_verification_start`, `on_verification_result`, `on_correction_iteration`, `on_verification_cap_reached`)
    - Implement the methods on `Repl`: running `[verify] running: <command>`, passed `[verify] passed`, failed `[verify] failed (<status>)`, iteration `[verify] correction iteration <n>/<max>`, and cap-reached (clear running indicator + `[verify] iteration cap reached (<max>); final status: <status>`)
    - Update `Repl.run_once` to call the coordinator after a non-interrupted/non-errored turn and render the phase's aggregated usage; when verification did not run, render exactly as today
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 2.3_

  - [x]* 10.2 Write renderer indicator unit tests
    - Assert running / passed / failed(status) / iteration `n/max` / cap-reached output, and that an absent command renders and persists exactly as today
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 2.3_

- [x] 11. Wire the phase into bootstrap
  - [x] 11.1 Construct and wire the coordinator in `forge/app.py`
    - In `bootstrap`, build the `VerificationRunner` (rooted at the workspace, sharing the `InterruptController`) and the `VerificationCoordinator` (sharing the `AgentLoop`, `SessionStore`, interrupt, and the `Repl` as `VerificationRenderer`), and pass the coordinator to the `Repl`
    - Keep the coordinator wired even when `config.verification.command` is `None` so it short-circuits at the gate with no special-casing
    - _Requirements: 2.1, 2.2, 2.3_

  - [x]* 11.2 Write opt-in equivalence integration test
    - With no Verify_Command configured, a completed turn returns to the prompt with no process started, no feedback appended, and identical end-of-turn rendering and persistence
    - _Requirements: 2.1, 2.2, 2.3_

- [x] 12. Final checkpoint - full feature integrated
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for a faster MVP.
- Each task references specific requirements clauses for traceability.
- Property tests (Properties 1–9) each run a minimum of 100 iterations with Hypothesis and are tagged `Feature: auto-verification-loop, Property {number}: {property_text}`.
- Generators should cover the prework edge cases: empty/whitespace commands, output exactly at and just over the cap, `max_correction_iterations` of 0 and large values, missing exit codes (timeouts/start errors), non-ASCII output, and sessions with zero, one, and many records/feedback messages.
- The `ShellTool.run` refactor preserves current behavior; the existing shell suites are the regression gate.
- Checkpoints ensure incremental validation at natural boundaries.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "3.1", "4.1"] },
    { "id": 1, "tasks": ["1.2", "2.2", "2.3", "3.2", "4.2", "5.1"] },
    { "id": 2, "tasks": ["5.2", "5.3"] },
    { "id": 3, "tasks": ["5.4", "5.5", "5.6"] },
    { "id": 4, "tasks": ["5.7", "5.8"] },
    { "id": 5, "tasks": ["5.9", "7.1"] },
    { "id": 6, "tasks": ["7.2", "7.3", "8.1"] },
    { "id": 7, "tasks": ["8.2", "8.3", "8.4", "10.1"] },
    { "id": 8, "tasks": ["10.2", "11.1"] },
    { "id": 9, "tasks": ["11.2"] }
  ]
}
```
