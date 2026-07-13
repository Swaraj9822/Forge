"""Slash commands and mention expansion for developer ergonomics."""

from __future__ import annotations

import re
from pathlib import Path

from forge.tools.paths import OutOfWorkspaceError, resolve_in_workspace


def get_code_spans(text: str) -> list[tuple[int, int]]:
    """Find start and end indices of markdown code blocks and inline code."""
    spans = []
    i = 0
    n = len(text)
    while i < n:
        if text[i:i+3] == "```":
            start = i
            i += 3
            end = text.find("```", i)
            if end == -1:
                spans.append((start, n))
                break
            spans.append((start, end + 3))
            i = end + 3
        elif text[i] == "`":
            start = i
            i += 1
            end = text.find("`", i)
            if end == -1:
                spans.append((start, n))
                break
            spans.append((start, end + 1))
            i = end + 1
        else:
            i += 1
    return spans


def is_index_in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    """Return True if the position falls within any of the spans."""
    for start, end in spans:
        if start <= pos < end:
            return True
    return False


def resolve_mention_candidate(
    candidate: str, workspace_root: Path
) -> tuple[Path | None, str, str | None]:
    """Find the longest prefix of candidate that resolves to an existing file."""
    punctuation = ".,?!:;)"

    current = candidate
    while current:
        try:
            resolved = resolve_in_workspace(current, workspace_root)
            if resolved.is_file():
                return resolved, current, None
        except OutOfWorkspaceError:
            return None, current, "out_of_scope"
        except Exception:
            pass

        if current and current[-1] in punctuation:
            current = current[:-1]
        else:
            break

    final_path = candidate
    while final_path and final_path[-1] in punctuation:
        final_path = final_path[:-1]
    if not final_path:
        final_path = candidate

    try:
        resolved = resolve_in_workspace(final_path, workspace_root)
        if not resolved.exists():
            return None, final_path, "missing"
        elif resolved.is_dir():
            return None, final_path, "is_dir"
        else:
            return resolved, final_path, None
    except OutOfWorkspaceError:
        return None, final_path, "out_of_scope"
    except Exception as exc:
        return None, final_path, f"error: {exc}"


def expand_mentions(
    text: str, workspace_root: Path, *, max_bytes: int
) -> tuple[str, list[str], list[str]]:
    """Replace @path tokens with fenced file contents.

    Returns (expanded_text, included_paths, warnings). Each @token that resolves
    (via resolve_in_workspace) to a readable UTF-8 workspace file is appended as
    a fenced block:  `--- <relpath> ---\\n```\\n<contents>\\n``` `. Out-of-scope,
    missing, binary, or oversized (> max_bytes) files are left as literal text
    and reported in `warnings` (the model still sees the raw @token).
    """
    code_spans = get_code_spans(text)

    pattern = re.compile(r'(?:\A|\s)@(?:"([^"]+)"|([^\s]+))')
    matches = list(pattern.finditer(text))

    expanded_text = text
    included_paths = []
    warnings = []

    for match in reversed(matches):
        start, end = match.span()
        at_pos = text.find("@", start, end)
        if at_pos == -1:
            continue

        if is_index_in_spans(at_pos, code_spans):
            continue

        group1 = match.group(1)
        group2 = match.group(2)

        is_quoted = group1 is not None
        candidate = group1 if is_quoted else group2

        if not candidate:
            continue

        if is_quoted:
            matched_token = f'"{candidate}"'
            try:
                resolved = resolve_in_workspace(candidate, workspace_root)
                if not resolved.exists():
                    resolved_path, warning = None, "missing"
                elif resolved.is_dir():
                    resolved_path, warning = None, "is_dir"
                else:
                    resolved_path, warning = resolved, None
            except OutOfWorkspaceError:
                resolved_path, warning = None, "out_of_scope"
            except Exception as exc:
                resolved_path, warning = None, f"error: {exc}"
        else:
            resolved_path, matched_subpart, warning = resolve_mention_candidate(
                candidate, workspace_root
            )
            matched_token = matched_subpart

        full_token_str = f"@{matched_token}"
        token_start_in_text = at_pos
        token_end_in_text = token_start_in_text + len(full_token_str)

        if resolved_path is not None:
            try:
                size = resolved_path.stat().st_size
                if size > max_bytes:
                    warnings.append(
                        f"Mention '{full_token_str}' exceeds maximum allowed "
                        f"size of {max_bytes} bytes (size: {size} bytes)."
                    )
                    continue

                with open(resolved_path, "r", encoding="utf-8") as f:
                    contents = f.read()

                relpath = resolved_path.relative_to(workspace_root).as_posix()
                fenced = f"--- {relpath} ---\n```\n{contents}\n```"

                expanded_text = (
                    expanded_text[:token_start_in_text]
                    + fenced
                    + expanded_text[token_end_in_text:]
                )
                included_paths.append(relpath)
            except UnicodeDecodeError:
                warnings.append(
                    f"Mention '{full_token_str}' is binary or not UTF-8 encoded."
                )
            except Exception as exc:
                warnings.append(
                    f"Mention '{full_token_str}' could not be read: {exc}"
                )
        else:
            if warning == "out_of_scope":
                warnings.append(
                    f"Mention '{full_token_str}' resolves outside the workspace."
                )
            elif warning == "missing":
                warnings.append(f"Mention '{full_token_str}' does not exist.")
            elif warning == "is_dir":
                warnings.append(
                    f"Mention '{full_token_str}' is a directory, not a file."
                )
            else:
                warnings.append(
                    f"Mention '{full_token_str}' could not be resolved: {warning}"
                )

    included_paths.reverse()
    warnings.reverse()

    return expanded_text, included_paths, warnings


class SlashCommandStore:
    """Discovers and renders custom markdown slash commands."""

    def __init__(self, dirs: list[Path]):
        self.dirs = [Path(d) for d in dirs]

    def names(self) -> list[str]:
        cmd_names = set()
        for d in self.dirs:
            if d.exists() and d.is_dir():
                for f in d.glob("*.md"):
                    cmd_names.add(f.stem)
        return sorted(list(cmd_names))

    def _find_file(self, name: str) -> Path | None:
        for d in self.dirs:
            if d.exists() and d.is_dir():
                f = d / f"{name}.md"
                if f.exists() and f.is_file():
                    return f
        return None

    def render(self, name: str, arg_text: str) -> str | None:
        """Load <name>.md, substitute $ARGUMENTS (and $1..$N) with arg_text,

        return the prompt text; None if the command is unknown.
        """
        f = self._find_file(name)
        if f is None:
            return None

        try:
            with open(f, "r", encoding="utf-8") as fh:
                body = fh.read()
        except Exception:
            return None

        args = arg_text.split()
        rendered = body.replace("$ARGUMENTS", arg_text)

        def repl(match: re.Match) -> str:
            num = int(match.group(0)[1:])
            if 1 <= num <= len(args):
                return args[num - 1]
            return ""

        rendered = re.sub(r"\$\d+", repl, rendered)
        return rendered
