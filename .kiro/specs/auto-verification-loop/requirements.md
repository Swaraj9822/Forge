# Requirements Document

## Introduction

The Auto-Verification Loop is a new feature for the Forge terminal coding agent that brings Forge on par with agents like Claude Code and Codex. Today, after the Agent_Loop finishes a turn (the Model returns a response with no Tool_Calls), Forge returns control to the REPL prompt regardless of whether the code the agent just changed still builds, passes its tests, or lints clean. The user must notice and re-prompt manually.

This feature adds a post-turn Verification_Phase. When a user-configured Verify_Command is present and the turn changed files, Forge automatically runs that command (typically a test, build, or lint command), captures its result, and reports pass or fail to the user. When verification fails, Forge feeds the captured failure back into the agent so the Model can self-correct, repeating verification after each correction attempt. The self-correction loop is bounded by a configurable maximum number of iterations so it always terminates. The feature is opt-in: with no Verify_Command configured, Forge behaves exactly as it does today.

The Verification_Phase reuses the existing shell execution machinery (platform default shell, wall-clock timeout, combined-output character cap, process-tree termination, and sub-second interrupt polling), respects user Interrupts within one second, accrues the extra Model requests its correction iterations issue into the turn's usage and cost totals, and persists its outcomes in the Session. Verification status and iteration progress surface to the user through the REPL renderer.

## Glossary

- **Forge**: The complete terminal-based AI coding agent application.
- **REPL**: The read-eval-print loop that reads user input, drives the Agent_Loop, and renders output.
- **Agent_Loop**: The control component that orchestrates a turn by assembling the Context_Window, streaming the Model response, executing Tool_Calls, and feeding Tool_Results back until the Model returns a response with no Tool_Calls.
- **Model**: The Gemini model accessed through Vertex AI.
- **Turn**: One execution of the Agent_Loop, beginning when a user message (or a Verification_Feedback message) is appended and ending when the Model returns a response containing no Tool_Calls, or when the turn is halted by an Interrupt or an error.
- **Tool**: A discrete capability the Model can invoke (read, write, edit, shell, search, git, planning).
- **Tool_Result**: The structured output returned to the Model after a Tool runs, carrying success status, content, an optional error, and a meta mapping.
- **File_Mutation**: A successful (`ok = true`) Tool_Result produced by the write Tool or the edit Tool within a turn. A turn that produces at least one File_Mutation is said to have mutated files.
- **Verification_Phase**: The phase that runs after a Turn completes, in which Forge runs the Verify_Command, evaluates the Verification_Result, and conditionally drives the bounded self-correction loop.
- **Verify_Command**: The user-configured shell command line that Forge runs to verify the workspace (for example, a test, build, or lint command). Absence of a Verify_Command disables the feature.
- **Verification_Runner**: The component that executes the Verify_Command using the existing shell execution machinery and produces a Verification_Result.
- **Verification_Result**: The structured outcome of one Verify_Command execution, carrying an outcome status of `passed`, `failed`, `timed_out`, or `start_error`, the command exit code when available, and the captured (and possibly truncated) combined output.
- **Verification_Feedback**: A synthesized message describing a non-passing Verification_Result (including the Verify_Command, the outcome status, the exit code when available, and the captured output) that Forge appends to the conversation so the Model can self-correct.
- **Correction_Iteration**: One cycle of the bounded self-correction loop, consisting of appending a Verification_Feedback message, running a correction Turn through the Agent_Loop, and re-running the Verify_Command.
- **Max_Correction_Iterations**: The configured maximum number of Correction_Iterations Forge will perform for a single Verification_Phase before stopping.
- **Verification_Trigger**: The configured condition that determines when the Verification_Phase runs: `on_file_change` (only after a Turn that mutated files) or `always` (after every completed Turn).
- **Config_Manager**: The component that loads, validates, and provides access to configuration values, applying documented defaults for absent values.
- **Interrupt**: A user-initiated signal (for example, pressing Ctrl-C) that requests cancellation of in-progress work.
- **Session**: The persisted record of a conversation, including messages, Tool_Calls, Tool_Results, todos, and usage metadata.
- **Usage_Tracker**: The component that records per-response token counts and computes the turn and cumulative estimated cost.
- **Renderer**: The REPL component that renders streamed text, Tool announcements, the compaction notice, and (with this feature) Verification_Phase status and progress.
- **Workspace**: The directory tree, rooted at the current working directory, that scopes file, search, git, and shell operations.

## Default Configuration Values

These defaults are applied by the Config_Manager when a value is absent. They are grouped under a new `[verification]` TOML table that follows the existing deterministic, explicit-configuration style.

| Setting | Default Value |
| --- | --- |
| Verify_Command (`verification.command`) | No default; absent means the feature is disabled (opt-in) |
| Max_Correction_Iterations (`verification.max_correction_iterations`) | 3 |
| Verification_Trigger (`verification.trigger`) | `on_file_change` |
| Verification timeout (`verification.timeout_s`) | Inherits the configured shell command timeout (120 seconds) |
| Verification output cap | Inherits the configured shell and command output cap (30,000 characters) |

## Requirements

### Requirement 1: Verification Configuration

**User Story:** As a developer, I want to configure a verification command and its limits, so that Forge runs my project's tests, build, or lint after it edits my code.

#### Acceptance Criteria

1. THE Config_Manager SHALL read the Verify_Command from the `verification.command` configuration value.
2. WHERE the `verification.command` configuration value is absent, THE Config_Manager SHALL resolve the Verify_Command as absent.
3. WHERE the `verification.max_correction_iterations` configuration value is absent, THE Config_Manager SHALL resolve Max_Correction_Iterations to 3.
4. WHERE the `verification.trigger` configuration value is absent, THE Config_Manager SHALL resolve the Verification_Trigger to `on_file_change`.
5. IF the `verification.max_correction_iterations` configuration value is present and is not an integer greater than or equal to 0, THEN THE Config_Manager SHALL reject the configuration with an error identifying the offending value.
6. IF the `verification.trigger` configuration value is present and is not one of `on_file_change` or `always`, THEN THE Config_Manager SHALL reject the configuration with an error identifying the offending value and the allowed values.
7. WHERE the `verification.timeout_s` configuration value is absent, THE Config_Manager SHALL resolve the verification timeout to the configured shell command timeout.
8. WHERE the `verification.timeout_s` configuration value is present and is a positive integer, THE Config_Manager SHALL resolve the verification timeout to that configured value.

### Requirement 2: Opt-In Behavior When Unconfigured

**User Story:** As a developer who has not configured verification, I want Forge to behave exactly as before, so that the feature never changes my workflow until I opt in.

#### Acceptance Criteria

1. WHEN a Turn completes and the Verify_Command is absent, THE Forge SHALL return control to the REPL prompt without running a Verification_Phase and without starting any Verify_Command process.
2. WHEN the Verify_Command is absent, THE Forge SHALL NOT append any Verification_Feedback message to the Session.
3. WHEN the Verify_Command is absent, THE Forge SHALL produce the same end-of-turn rendering and Session persistence that it produces without this feature.

### Requirement 3: Verification Trigger Conditions

**User Story:** As a developer, I want verification to run at the right time, so that Forge checks my code after it changes files without wasting time when nothing changed.

#### Acceptance Criteria

1. WHEN a Turn completes, the Verify_Command is present, and the Verification_Trigger is `on_file_change`, THE Forge SHALL run the Verification_Phase only if the Turn produced at least one File_Mutation.
2. WHEN a Turn completes, the Verify_Command is present, the Verification_Trigger is `on_file_change`, and the Turn produced no File_Mutation, THE Forge SHALL return control to the REPL prompt without running the Verification_Phase.
3. WHERE the Verification_Trigger is `always` and the Verify_Command is present, THE Forge SHALL run the Verification_Phase after every completed Turn regardless of whether the Turn produced a File_Mutation.
4. WHEN a Turn is halted by an Interrupt or ends with an error, THE Forge SHALL return control to the REPL prompt without running the Verification_Phase.

### Requirement 4: Verification Command Execution

**User Story:** As a developer, I want the verification command to run with the same safety limits as shell commands, so that a slow or runaway verification cannot hang or flood my terminal.

#### Acceptance Criteria

1. WHEN the Verification_Phase runs the Verify_Command, THE Verification_Runner SHALL execute the Verify_Command through the platform default shell rooted at the Workspace using the existing shell execution machinery.
2. WHEN the Verify_Command completes with exit code 0, THE Verification_Runner SHALL produce a Verification_Result with outcome status `passed`.
3. WHEN the Verify_Command completes with a non-zero exit code, THE Verification_Runner SHALL produce a Verification_Result with outcome status `failed`, the exit code, and the captured combined output.
4. IF the Verify_Command runs longer than the configured verification timeout, THEN THE Verification_Runner SHALL terminate the Verify_Command process tree and produce a Verification_Result with outcome status `timed_out`.
5. IF the Verify_Command exceeds the configured verification timeout and termination of the process tree does not succeed, THEN THE Verification_Runner SHALL still produce a Verification_Result with outcome status `timed_out`.
6. WHEN the captured combined output of the Verify_Command exceeds the configured output cap, THE Verification_Runner SHALL truncate the captured output to the output cap and flag the Verification_Result as truncated.
7. IF the Verify_Command process cannot be started, THEN THE Verification_Runner SHALL produce a Verification_Result with outcome status `start_error` and a description of the failure.

### Requirement 5: Bounded Self-Correction Loop

**User Story:** As a developer, I want Forge to automatically fix verification failures it introduced, so that I get working code without re-prompting after every failed test run.

#### Acceptance Criteria

1. WHEN a Verification_Result has outcome status `passed`, THE Forge SHALL end the Verification_Phase and return control to the REPL prompt without performing a Correction_Iteration.
2. WHEN a Verification_Result has outcome status `failed` or `timed_out` and the number of completed Correction_Iterations is less than Max_Correction_Iterations, THE Forge SHALL perform a Correction_Iteration.
3. WHEN performing a Correction_Iteration, THE Forge SHALL append a Verification_Feedback message describing the Verification_Result, run a correction Turn through the Agent_Loop, and then re-run the Verify_Command.
4. WHILE the most recent Verification_Result has outcome status `failed` or `timed_out` and the number of completed Correction_Iterations is less than Max_Correction_Iterations, THE Forge SHALL continue performing Correction_Iterations.
5. WHEN a Correction_Iteration produces a Verification_Result with outcome status `passed`, THE Forge SHALL end the Verification_Phase and return control to the REPL prompt.
6. WHERE Max_Correction_Iterations is configured as 0, THE Forge SHALL run the Verify_Command and report the Verification_Result without performing any Correction_Iteration.
7. WHEN the Verification_Phase runs, THE Forge SHALL run the initial Verify_Command before any Correction_Iteration regardless of the configured Max_Correction_Iterations value.

### Requirement 6: Reaching the Iteration Cap

**User Story:** As a developer, I want the self-correction loop to always stop, so that a failure Forge cannot fix does not loop indefinitely or consume unbounded cost.

#### Acceptance Criteria

1. WHEN the number of completed Correction_Iterations reaches Max_Correction_Iterations and the most recent Verification_Result does not have outcome status `passed`, THE Forge SHALL end the Verification_Phase and return control to the REPL prompt.
2. WHEN the Verification_Phase ends because Max_Correction_Iterations was reached without a passing Verification_Result, THE Forge SHALL surface the final non-passing Verification_Result and the number of Correction_Iterations performed to the user.
3. THE Forge SHALL perform at most Max_Correction_Iterations Correction_Iterations within a single Verification_Phase.

### Requirement 7: Feeding Failures Back to the Model

**User Story:** As a developer, I want the agent to receive the actual failure output, so that its corrections target the real errors.

#### Acceptance Criteria

1. WHEN Forge performs a Correction_Iteration, THE Verification_Feedback message SHALL include the Verify_Command, the outcome status, the exit code when available, and the captured combined output of the failing Verification_Result.
2. WHEN the captured output included in a Verification_Feedback message would exceed the configured output cap, THE Forge SHALL include the truncated captured output and indicate that the output was truncated.
3. WHEN a Verification_Feedback message is appended, THE Agent_Loop SHALL process the correction Turn using the same context assembly, streaming, and Tool execution it uses for a user-submitted Turn.

### Requirement 8: Interrupt During Verification

**User Story:** As a developer, I want to stop a long verification or correction loop with Ctrl-C, so that I retain control during automated work.

#### Acceptance Criteria

1. WHEN a user issues an Interrupt while the Verify_Command is executing, THE Verification_Runner SHALL terminate the Verify_Command process tree within 1 second and return control to the REPL prompt.
2. WHEN a user issues an Interrupt during a correction Turn, THE Forge SHALL halt the Verification_Phase within 1 second and return control to the REPL prompt.
3. WHEN an Interrupt halts the Verification_Phase, THE Forge SHALL retain all messages, Tool_Results, and Verification_Feedback already appended to the Session, completing any in-progress Session write before returning control so that retention is all-or-nothing.
4. WHEN an Interrupt halts the Verification_Phase, THE Forge SHALL NOT begin a new Correction_Iteration.

### Requirement 9: Surfacing Verification Status to the User

**User Story:** As a developer, I want to see verification status and progress in my terminal, so that I know what Forge is checking and whether it succeeded.

#### Acceptance Criteria

1. WHEN the Verification_Phase begins running the Verify_Command, THE Renderer SHALL display an indicator naming that verification is running.
2. WHEN a Verification_Result with outcome status `passed` is produced, THE Renderer SHALL display a verification-passed indicator.
3. WHEN a Verification_Result with outcome status `failed`, `timed_out`, or `start_error` is produced, THE Renderer SHALL display a verification-failed indicator that includes the outcome status.
4. WHEN Forge begins a Correction_Iteration, THE Renderer SHALL display the current Correction_Iteration number and the configured Max_Correction_Iterations.
5. WHEN the Verification_Phase ends without a passing Verification_Result after reaching Max_Correction_Iterations, THE Renderer SHALL clear the verification-running indicator and display an indicator that the iteration cap was reached.

### Requirement 10: Usage and Cost Accounting Across Correction Iterations

**User Story:** As a developer, I want the cost of automatic corrections reflected in my usage totals, so that my reported cost matches what I actually spent.

#### Acceptance Criteria

1. WHEN a correction Turn issues a Model request during the Verification_Phase, THE Usage_Tracker SHALL record that request's token counts in both the turn and cumulative totals.
2. WHEN the Verification_Phase completes, THE Forge SHALL report a usage summary that includes the token counts and estimated cost of every Model request issued during the original Turn and all Correction_Iterations.
3. WHEN the Verification_Phase completes, THE Forge SHALL persist the cumulative token counts and estimated cost, including those accrued during Correction_Iterations, to the Session.

### Requirement 11: Persistence of Verification Outcomes

**User Story:** As a developer, I want verification outcomes saved with my session, so that resuming a session preserves the record of what was verified and corrected.

#### Acceptance Criteria

1. WHEN the Verification_Phase appends a Verification_Feedback message, THE Forge SHALL persist that message in the Session.
2. WHEN the Verification_Phase ends, THE Forge SHALL persist the Session including the final Verification_Result outcome status and the number of Correction_Iterations performed.
3. WHEN a Session containing Verification_Phase records is persisted and then restored, THE Session_Store SHALL reconstruct those records without loss.
