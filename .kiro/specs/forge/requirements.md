# Requirements Document

## Introduction

Forge is a minimal but complete terminal-based AI coding agent. Forge runs as an interactive command-line REPL that drives an autonomous agent loop: it sends user requests to a Gemini model hosted on Google Cloud Vertex AI, streams responses to the terminal, and lets the model invoke a set of coding tools (file read/write/edit, shell execution, codebase search, and git operations) to complete tasks. Forge operates autonomously by default while allowing the user to interrupt execution at any time.

Forge persists sessions for later resumption, manages conversation context with automatic compaction, supports user-defined system prompts and steering files, integrates external tools through the Model Context Protocol (MCP), and tracks token usage and cost. The scope is intentionally focused: the features a modern coding agent needs, kept simple.

## Glossary

- **Forge**: The complete terminal-based AI coding agent application described by this document.
- **CLI**: The command-line entry point that launches Forge in an interactive terminal session.
- **REPL**: The read-eval-print loop that reads user input, runs the agent loop, and prints output until the user exits.
- **Exit_Command**: The reserved keywords a user types to terminate the REPL. The reserved keywords are `/exit` and `/quit`.
- **Agent_Loop**: The control component that orchestrates a turn by sending conversation context to the Model, processing the streamed response, executing requested tool calls, and feeding tool results back to the Model until the turn completes.
- **Model**: The Gemini model `gemini-3.1-pro-preview` accessed through Google Cloud Vertex AI.
- **Vertex_Client**: The component that authenticates to and communicates with Vertex AI using the Google Cloud SDK.
- **ADC**: Application Default Credentials provided by the local gcloud environment, used by the Vertex_Client to authenticate.
- **Tool**: A discrete capability the Model can invoke during the Agent_Loop, including file read, file write, file edit, shell execution, codebase search, and git operations.
- **Tool_Executor**: The component that validates a tool call, executes the requested Tool, and returns a structured result to the Agent_Loop.
- **Tool_Call**: A structured request emitted by the Model naming a Tool and supplying arguments.
- **Tool_Result**: The structured output returned to the Model after a Tool runs, containing output data or an error.
- **Interrupt**: A user-initiated signal (for example, pressing Ctrl-C) that requests cancellation of in-progress Model generation or Tool execution.
- **Config_File**: A user-editable configuration file that defines the model identifier, GCP project ID, region, and enabled Tools.
- **Config_Manager**: The component that loads, validates, and provides access to configuration values.
- **Session**: A persisted record of a conversation, including messages, Tool_Calls, Tool_Results, and usage metadata.
- **Session_Store**: The component that persists, lists, and restores Sessions.
- **Context_Window**: The ordered set of messages and metadata sent to the Model for a turn.
- **Context_Manager**: The component that assembles the Context_Window and performs auto-compaction.
- **Compaction**: The process of summarizing earlier conversation content to reduce token count while preserving task-relevant information.
- **Steering_File**: A user-provided file containing a custom system prompt or guidance injected into the Context_Window.
- **MCP**: The Model Context Protocol, an open protocol for exposing external Tools to the agent.
- **MCP_Server**: An external process exposing Tools to Forge through MCP.
- **MCP_Client**: The component that connects to MCP_Servers, discovers their Tools, and invokes them.
- **Usage_Tracker**: The component that records token counts and computes estimated cost.
- **Token_Limit**: The configured maximum number of tokens permitted in the Context_Window before Compaction is triggered.
- **Workspace**: The directory tree that scopes file, search, and git operations, rooted at the current working directory from which the CLI is launched. The Workspace boundary is the security boundary used by the out-of-scope path checks in the file and search Tools.

## Default Configuration Values

The following table consolidates the documented default values referenced throughout this document. These defaults are applied by the Config_Manager when a value is absent from the Config_File.

| Setting | Default Value |
| --- | --- |
| Model identifier | `gemini-3.1-pro-preview` |
| GCP region | No default (required) |
| GCP project ID | No default (required) |
| Token_Limit | 200,000 tokens |
| Retained-recent-messages count (Compaction) | 20 messages |
| Request timeout (Vertex AI) | 60 seconds |
| Shell command timeout | 120 seconds |
| Shell and git output cap | 30,000 characters |
| Search result limit | 100 matches |
| Search matching-line cap | 500 characters |
| Read cap | 2,000 lines or 1 MB |
| Rate-limit retry attempts | 3 attempts |
| MCP connect timeout | 30 seconds |
| Enabled Tools default set | read, write, edit, shell, search, git, planning |

## Requirements

### Requirement 1: Interactive Agent REPL

**User Story:** As a developer, I want an interactive terminal session that accepts my requests and runs an agent loop, so that I can collaborate with the coding agent without leaving my terminal.

#### Acceptance Criteria

1. WHEN a user launches the CLI, THE Forge SHALL start a REPL that displays an input prompt and waits for user input.
2. WHEN a user submits text containing at least one non-whitespace character at the prompt, THE Agent_Loop SHALL send the submitted text together with the current Context_Window to the Model.
3. WHEN the Model returns a response that contains no Tool_Calls, THE Agent_Loop SHALL display the response and return control to the REPL prompt.
4. WHEN the Model returns a response that contains one or more Tool_Calls, THE Agent_Loop SHALL execute each Tool_Call in the order received and send each Tool_Result back to the Model within the same turn.
5. WHILE a turn is in progress, THE Agent_Loop SHALL continue sending Tool_Results to the Model until the Model returns a response containing no Tool_Calls.
6. WHEN a user submits input that exactly matches an Exit_Command keyword (`/exit` or `/quit`), THE Forge SHALL terminate the REPL and return control to the shell without invoking the Agent_Loop.
7. IF a user submits input that is empty or contains only whitespace characters, THEN THE Forge SHALL re-display the prompt without invoking the Agent_Loop.

### Requirement 2: Vertex AI and Gemini Integration

**User Story:** As a developer, I want Forge to use the Gemini model through Vertex AI, so that I can rely on a managed, enterprise-grade model backend.

#### Acceptance Criteria

1. WHEN the Vertex_Client sends a request to the Model, THE Vertex_Client SHALL target the model identifier, GCP project ID, and region provided by the Config_Manager.
2. THE Vertex_Client SHALL authenticate to Vertex AI using ADC.
3. IF ADC are unavailable when the Vertex_Client initializes, THEN THE Forge SHALL display an error message identifying the missing credentials and the command required to establish ADC.
4. IF the GCP project ID or region is absent from configuration when the Vertex_Client initializes, THEN THE Forge SHALL display an error message identifying the missing configuration value.
5. IF the Vertex_Client receives an authorization error from Vertex AI, THEN THE Forge SHALL display the error, preserve the current Session state, and return control to the REPL prompt.
6. IF the Vertex_Client receives a rate-limit response from Vertex AI, THEN THE Vertex_Client SHALL retry the request up to 3 attempts, and IF all retry attempts are exhausted, THEN THE Forge SHALL display a rate-limit error and return control to the REPL prompt.
7. WHEN the Vertex_Client receives a successful response from the Model, THE Agent_Loop SHALL process the response and return its output to the REPL.
8. IF a request to Vertex AI does not complete within 60 seconds, THEN THE Forge SHALL terminate the request, display a timeout error, preserve the current Session state, and return control to the REPL prompt.

### Requirement 3: Streaming Output

**User Story:** As a developer, I want to see the model's response as it is generated, so that I receive immediate feedback during long responses.

#### Acceptance Criteria

1. WHILE the Model is generating a response, THE Forge SHALL display each response token in the terminal within 200 milliseconds of receiving that token.
2. WHEN a Tool_Call is emitted during streaming, THE Forge SHALL display the name of the Tool being invoked before the Tool_Executor runs the Tool.
3. WHEN streaming of a response completes, THE Forge SHALL display a visible end-of-response indicator before returning control to the next step.
4. IF the Model response stream terminates with an error or is interrupted before completion, THEN THE Forge SHALL display an error indicator describing the interruption, SHALL retain the partial response tokens already displayed in the terminal, and SHALL return control to the next step.

### Requirement 4: Autonomous Tool Execution and Interrupt

**User Story:** As a developer, I want the agent to run tools automatically but let me stop it at any time, so that I get fast progress while retaining control.

#### Acceptance Criteria

1. WHEN the Model emits a Tool_Call for an enabled Tool, THE Tool_Executor SHALL execute the Tool without requesting user confirmation.
2. WHEN a user issues an Interrupt while the Model is generating a response, THE Forge SHALL stop generation within 1 second and return control to the REPL prompt.
3. WHEN a user issues an Interrupt while a Tool is executing, THE Tool_Executor SHALL stop the Tool within 1 second and return control to the REPL prompt.
4. WHEN an Interrupt halts an executing Tool, THE Tool_Executor SHALL return a Tool_Result indicating that the Tool was interrupted.
5. WHEN an Interrupt halts a turn, THE Forge SHALL retain all completed messages, Tool_Calls, and Tool_Results from that turn in the Session.
6. IF the Model emits a Tool_Call naming a Tool that is not enabled in configuration, THEN THE Tool_Executor SHALL return a Tool_Result describing the Tool as unavailable.
7. IF the Model emits a Tool_Call with arguments that fail validation, THEN THE Tool_Executor SHALL return a Tool_Result describing the validation error and SHALL NOT execute the Tool.

### Requirement 5: Read Files Tool

**User Story:** As a developer, I want the agent to read files, so that it can understand my code before making changes.

#### Acceptance Criteria

1. WHEN the read Tool is invoked with a file path that exists and contains valid UTF-8 text, THE Tool_Executor SHALL return the file contents in the Tool_Result.
2. WHERE a line range is supplied to the read Tool, THE Tool_Executor SHALL return only the lines within the specified range, bounded from line 1 to the last line of the file.
3. IF the read Tool is invoked with a path that does not exist, THEN THE Tool_Executor SHALL return a Tool_Result describing the path as not found.
4. IF the read Tool is invoked with a path that resolves outside the Workspace, THEN THE Tool_Executor SHALL return a Tool_Result describing the access as out of scope.
5. IF a supplied line range begins before line 1, ends after the last line of the file, or has a start line greater than the end line, THEN THE Tool_Executor SHALL return a Tool_Result describing the line range as invalid.
6. IF the read Tool is invoked with a file that does not contain valid UTF-8 text, THEN THE Tool_Executor SHALL return a Tool_Result describing the file as binary and SHALL exclude the raw file contents.
7. WHEN the read Tool returns contents that exceed the configured maximum of 2,000 lines or 1 MB, THE Tool_Executor SHALL return contents up to that limit and indicate that the contents were truncated.

### Requirement 6: Write and Edit Files Tool

**User Story:** As a developer, I want the agent to create and modify files, so that it can implement changes directly.

#### Acceptance Criteria

1. WHEN the write Tool is invoked with a file path and content, THE Tool_Executor SHALL write the content to the file, replacing any existing content at the path, and return a Tool_Result confirming the path and the count of bytes written.
2. WHEN the write Tool targets a path whose parent directory does not exist, THE Tool_Executor SHALL create all missing parent directories along the path before writing the file.
3. WHEN the edit Tool is invoked with a target string that occurs exactly once in the file, THE Tool_Executor SHALL replace the target string with the replacement string and return a Tool_Result confirming the change.
4. IF the edit Tool is invoked with a target string that occurs zero times in the file, THEN THE Tool_Executor SHALL return a Tool_Result describing the target string as not found and SHALL leave the file unchanged.
5. IF the edit Tool is invoked with a target string that occurs more than once in the file, THEN THE Tool_Executor SHALL return a Tool_Result describing the target string as ambiguous and SHALL leave the file unchanged.
6. IF a write or edit Tool targets a path that resolves outside the Workspace, THEN THE Tool_Executor SHALL return a Tool_Result describing the access as out of scope and SHALL leave the file system unchanged.
7. IF the edit Tool is invoked against a file path that does not exist, THEN THE Tool_Executor SHALL return a Tool_Result describing the file as not found and SHALL leave the file system unchanged.
8. IF a write or edit Tool fails due to a file system error such as insufficient permissions or an input/output failure, THEN THE Tool_Executor SHALL return a Tool_Result describing the failure and the affected path and SHALL leave the file system unchanged.

### Requirement 7: Shell Command Tool

**User Story:** As a developer, I want the agent to run shell commands, so that it can build, test, and inspect my project.

#### Acceptance Criteria

1. WHEN the shell Tool is invoked with a command, THE Tool_Executor SHALL execute the command within the Workspace and return its standard output, standard error, and exit code in the Tool_Result.
2. WHEN a shell command exits with a non-zero exit code, THE Tool_Executor SHALL include the exit code and the captured error output in the Tool_Result.
3. WHEN a shell command runs longer than the configured timeout of 120 seconds, THE Tool_Executor SHALL terminate the command and return a Tool_Result describing the timeout.
4. WHEN a user issues an Interrupt while a shell command is executing, THE Tool_Executor SHALL terminate the command process and return a Tool_Result describing the command as interrupted.
5. WHEN a shell command produces combined standard output and standard error exceeding the configured maximum of 30,000 characters, THE Tool_Executor SHALL return output up to that limit and indicate that the output was truncated.
6. IF the shell Tool is invoked with an empty command, THEN THE Tool_Executor SHALL return a Tool_Result describing the command as invalid.

### Requirement 8: Codebase Search Tool

**User Story:** As a developer, I want the agent to search my codebase by content and by file name, so that it can locate relevant code quickly.

#### Acceptance Criteria

1. WHEN the search Tool is invoked with a content pattern, THE Tool_Executor SHALL return matching file paths with line numbers and matching lines in the Tool_Result.
2. WHEN the search Tool is invoked with a file-name glob pattern, THE Tool_Executor SHALL return the list of file paths matching the glob in the Tool_Result.
3. WHEN a content or glob search produces no matches, THE Tool_Executor SHALL return a Tool_Result indicating that no matches were found.
4. WHEN the search Tool returns more matches than the configured result limit of 100 matches, THE Tool_Executor SHALL return results up to the limit and indicate that the results were truncated.
5. IF the search Tool is invoked with a content pattern that is not a valid regular expression, THEN THE Tool_Executor SHALL return a Tool_Result describing the pattern as invalid.
6. WHEN a matching line exceeds the configured maximum of 500 characters, THE Tool_Executor SHALL return the matching line truncated to that limit and indicate that the line was truncated.

### Requirement 9: Git Operations Tool

**User Story:** As a developer, I want the agent to perform git operations, so that it can inspect history and manage changes.

#### Acceptance Criteria

1. THE git Tool SHALL support exactly the following git operations: status, diff, log, show, add, commit, branch, checkout, and stash.
2. WHEN the git Tool is invoked with a supported git operation, THE Tool_Executor SHALL execute the operation in the Workspace repository and return the command output in the Tool_Result.
3. IF the git Tool is invoked while the Workspace is not a git repository, THEN THE Tool_Executor SHALL return a Tool_Result describing the Workspace as not a git repository.
4. IF the git Tool is invoked with an operation that is not in the supported set enumerated in criterion 1, THEN THE Tool_Executor SHALL return a Tool_Result describing the operation as unsupported.
5. IF a supported git operation completes with a non-zero exit code, THEN THE Tool_Executor SHALL return a Tool_Result that includes the exit code and the captured error output.
6. WHEN the git Tool returns command output exceeding the configured maximum of 30,000 characters, THE Tool_Executor SHALL return output up to that limit and indicate that the output was truncated.

### Requirement 10: Multi-Step Planning and Todo Tracking

**User Story:** As a developer, I want the agent to plan multi-step tasks and track progress, so that I can follow complex work as it proceeds.

#### Acceptance Criteria

1. WHEN the planning Tool is invoked with a list of up to 100 task items, THE Tool_Executor SHALL store the task items as the current todo list and return the stored list in the Tool_Result.
2. WHEN the planning Tool updates the status of a task item present in the current todo list, THE Tool_Executor SHALL record the new status and return the updated todo list in the Tool_Result.
3. WHERE a todo list exists for the current Session, THE Forge SHALL display each task item with the task item status in the terminal when the todo list changes.
4. THE Tool_Executor SHALL constrain each task item status to one of: pending, in progress, or completed.
5. WHILE the current Session remains active, THE Tool_Executor SHALL retain the current todo list and the task item statuses across turns until the planning Tool replaces or clears the todo list.
6. IF the planning Tool updates the status of a task item not present in the current todo list, THEN THE Tool_Executor SHALL return a Tool_Result indicating the task item was not found and SHALL leave the current todo list unchanged.

### Requirement 11: Configuration

**User Story:** As a developer, I want a configuration file for model and project settings, so that I can control how Forge connects and which tools are available.

#### Acceptance Criteria

1. THE Config_File SHALL be encoded in TOML format.
2. THE Config_Manager SHALL locate the Config_File using operating-system conventions: on Windows at `%APPDATA%\forge\config.toml`; on Unix and macOS at `$XDG_CONFIG_HOME/forge/config.toml` when `XDG_CONFIG_HOME` is set, otherwise at `~/.config/forge/config.toml`.
3. WHEN Forge starts, THE Config_Manager SHALL load the Config_File containing the model identifier, GCP project ID, region, and the set of enabled Tools.
4. WHERE a configuration value is absent from the Config_File, THE Config_Manager SHALL apply the documented default value for that configuration setting.
5. IF the Config_File is not present at the operating-system-conventional location, THEN THE Config_Manager SHALL apply the documented default values for all configuration settings and SHALL continue startup.
6. IF the Config_File contains a syntax error, THEN THE Forge SHALL display an error message identifying the Config_File and the line and column of the error, and SHALL stop startup.
7. IF the Config_File enables a Tool name that Forge does not provide, THEN THE Config_Manager SHALL display a warning naming the unrecognized Tool and SHALL continue startup.
8. THE Tool_Executor SHALL make available to the Model only the Tools listed as enabled by the Config_Manager.
9. THE Session_Store SHALL store Sessions under the operating-system-conventional user data directory: on Windows at `%APPDATA%\forge\sessions`; on Unix and macOS at `$XDG_DATA_HOME/forge/sessions` when `XDG_DATA_HOME` is set, otherwise at `~/.local/share/forge/sessions`.

### Requirement 12: First-Run Initialization

**User Story:** As a developer, I want a command to initialize my configuration, so that I can get Forge running with documented defaults and clear placeholders for required values.

#### Acceptance Criteria

1. WHEN a user runs the init command (for example, `forge init`) AND no Config_File exists at the operating-system-conventional location, THE Forge SHALL create a Config_File at the user-level configuration directory populated with the documented default values and with placeholders for the required values GCP project ID and region.
2. IF a user runs the init command AND a Config_File already exists at the operating-system-conventional location, THEN THE Forge SHALL report that configuration already exists and SHALL leave the existing Config_File unchanged.
3. IF the required configuration values GCP project ID or region are missing on normal startup, THEN THE Forge SHALL display a message directing the user to run the init command.

### Requirement 13: Session Persistence and Resume

**User Story:** As a developer, I want my conversations saved and resumable, so that I can continue work across separate terminal sessions.

#### Acceptance Criteria

1. WHEN a turn completes, THE Session_Store SHALL persist the Session to a user-level data directory including all messages, Tool_Calls, Tool_Results, and usage metadata.
2. WHEN the Session_Store persists a Session, THE Session_Store SHALL write the Session as a single atomic replacement of the stored Session file.
3. IF two writes to the same Session are requested concurrently, THEN THE Session_Store SHALL apply the writes sequentially.
4. WHEN a user requests a list of saved Sessions, THE Session_Store SHALL return each saved Session with its identifier and creation timestamp.
5. WHEN a user launches the CLI with a request to resume a specified Session, THE Session_Store SHALL load that Session and THE Agent_Loop SHALL continue using its restored messages as the Context_Window.
6. IF a user requests resumption of a Session identifier that does not exist, THEN THE Forge SHALL display an error message identifying the unknown Session identifier.
7. IF the stored Session file for a requested Session is corrupted or cannot be parsed, THEN THE Session_Store SHALL leave the stored Session data unchanged and THE Forge SHALL display an error message identifying the affected Session.

### Requirement 14: Context Auto-Compaction

**User Story:** As a developer, I want long conversations to be compacted automatically, so that the agent keeps working without exceeding model limits.

#### Acceptance Criteria

1. WHILE assembling a turn, THE Context_Manager SHALL compute the estimated token count of the Context_Window.
2. WHEN the estimated token count of the Context_Window exceeds the Token_Limit, THE Context_Manager SHALL perform Compaction before the Agent_Loop sends the request to the Model.
3. WHEN Compaction runs, THE Context_Manager SHALL replace earlier messages with a summary while retaining the original task and instructions.
4. WHEN Compaction runs, THE Context_Manager SHALL preserve the recorded decisions and outcomes from the replaced messages in the summary.
5. WHEN Compaction runs, THE Context_Manager SHALL retain the most recent messages up to the count configured by the Config_Manager.
6. WHEN Compaction runs, THE Context_Manager SHALL retain any pending Tool_Calls whose Tool_Results have not yet been produced.
7. WHEN Compaction completes, THE Forge SHALL display a notice that the conversation context was compacted.
8. WHEN Compaction completes, THE Context_Manager SHALL produce a Context_Window whose estimated token count does not exceed the Token_Limit.
9. IF Compaction cannot reduce the estimated token count of the Context_Window to at or below the Token_Limit, THEN THE Forge SHALL display a warning and THE Context_Manager SHALL proceed with the smallest well-formed Context_Window it can produce while retaining the original task and instructions and any pending Tool_Calls whose Tool_Results have not yet been produced.

### Requirement 15: Steering and System Prompt Customization

**User Story:** As a developer, I want to provide custom system prompts and steering files, so that I can shape the agent's behavior for my project.

#### Acceptance Criteria

1. WHERE one or more Steering_Files are configured, THE Context_Manager SHALL include the contents of each Steering_File in the Context_Window sent to the Model in the order the Steering_Files are listed in the configuration.
2. WHERE one or more Steering_Files are configured, THE Context_Manager SHALL place the built-in default system prompt first in the Context_Window, followed by the contents of the configured Steering_Files.
3. WHEN no Steering_File is configured, THE Context_Manager SHALL include the built-in default system prompt in the Context_Window.
4. IF a configured Steering_File path does not exist, THEN THE Forge SHALL display a warning naming the missing Steering_File and SHALL continue using the remaining available prompts.

### Requirement 16: MCP Tool Support

**User Story:** As a developer, I want Forge to use tools exposed over MCP, so that I can extend the agent with external capabilities.

#### Acceptance Criteria

1. WHERE one or more MCP_Servers are configured, THE MCP_Client SHALL attempt to connect to each configured MCP_Server at startup within 30 seconds per MCP_Server and SHALL discover the Tools each MCP_Server exposes.
2. WHEN MCP Tools are discovered, THE Tool_Executor SHALL make the discovered MCP Tools available to the Model alongside the built-in Tools.
3. WHEN the Model invokes a discovered MCP Tool, THE MCP_Client SHALL forward the Tool_Call to the corresponding MCP_Server and return the MCP_Server response as the Tool_Result.
4. IF a configured MCP_Server fails to connect within 30 seconds at startup, THEN THE Forge SHALL display a warning naming the MCP_Server and SHALL continue startup with the remaining Tools.
5. IF an invoked MCP_Server returns an error or becomes unreachable during a Tool_Call, THEN THE MCP_Client SHALL return a Tool_Result describing the failure.
6. IF a discovered MCP Tool has the same name as a built-in Tool, THEN THE Tool_Executor SHALL retain the built-in Tool for that name, SHALL exclude the conflicting MCP Tool from the Tools made available to the Model, and THE Forge SHALL display a warning naming the conflicting Tool and the MCP_Server.

### Requirement 17: Token Usage and Cost Tracking

**User Story:** As a developer, I want to see token usage and estimated cost, so that I can monitor my spending.

#### Acceptance Criteria

1. WHEN the Model returns a response, THE Usage_Tracker SHALL record the input token count and output token count reported for that response.
2. WHEN a turn completes, THE Usage_Tracker SHALL compute the cumulative input tokens, output tokens, and estimated cost for the current Session.
3. WHEN a turn completes, THE Forge SHALL display the token counts and estimated cost for that turn and the cumulative total for the Session.
4. THE Usage_Tracker SHALL compute estimated cost using the per-token pricing values provided by the Config_Manager for the active Model.
5. IF per-token pricing values for the active Model are not available from the Config_Manager, THEN THE Usage_Tracker SHALL omit the estimated cost and THE Forge SHALL display the token counts together with an indication that the estimated cost is unavailable.

## Assumptions and Dependencies

- Git is installed and available on the system PATH.
- A supported Python runtime is installed and available on the system PATH.
- gcloud Application Default Credentials (ADC) have been established in the local environment.
- Network access to Google Cloud Vertex AI is available.
- The shell Tool uses the platform default shell: `cmd` or PowerShell on Windows, and `/bin/sh` on Unix and macOS.
- The search Tool uses a built-in regular-expression and glob engine and does not depend on an external search binary.

## Out of Scope (v1)

- No integrated development environment (IDE) integration or graphical user interface (GUI).
- No multi-user mode or server mode.
- No remote session synchronization across machines.
- No model providers other than Gemini through Vertex AI.
