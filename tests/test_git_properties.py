"""Property-based tests for the git tool (tasks 12.2 and 12.3).

This module hosts two design properties for :class:`forge.tools.git.GitTool`:

* **Property 16: Git operation dispatch** (Validates: Requirements 9.1, 9.4) --
  *for any* operation name, the git tool dispatches it through the ``git``
  binary if and only if the name is one of the supported operations
  ``{status, diff, log, show, add, commit, branch, checkout, stash}``, and
  returns an "unsupported" result for every other name without ever invoking
  git.

* **Property 12: Shell and git output char cap** (git portion)
  (Validates: Requirements 7.5, 9.6) -- *for any* command output, the git tool
  returns at most the configured cap of characters and flags the result as
  truncated whenever the output exceeds that cap (the documented default cap is
  30,000 characters).

Both tests are deliberately fast and fully offline: the dispatch test patches
``subprocess.run`` so the dispatch decision is observed without spawning git,
and the output-cap test is a white-box property over the tool's own ``_cap``
helper -- the exact truncation logic ``run`` applies to real git output. This
mirrors the sibling shell-cap test which reuses the shell tool's ``_render``
helper.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.interrupt import InterruptController
from forge.tools.base import ToolContext
from forge.tools.git import SUPPORTED_OPERATIONS, GitTool

# The default cap the tool falls back to when no config is supplied (Req 9.6).
_DEFAULT_CAP = 30_000


def _ctx(workspace="/nonexistent-workspace", **config_kwargs) -> ToolContext:
    """Build a ToolContext with a fake config carrying ``output_cap_chars``."""
    from pathlib import Path

    config = SimpleNamespace(**config_kwargs) if config_kwargs else None
    return ToolContext(
        workspace_root=Path(workspace),
        interrupt=InterruptController(),
        config=config,
    )


class _FakeCompleted:
    """Minimal stand-in for :class:`subprocess.CompletedProcess`."""

    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


# ---------------------------------------------------------------------------
# Property 16: Git operation dispatch (Req 9.1, 9.4)
# ---------------------------------------------------------------------------

# A grab-bag of names that are NOT in the supported set: common git subcommands
# the tool intentionally does not expose, case variants, and decorated forms.
_KNOWN_UNSUPPORTED = [
    "push",
    "pull",
    "clone",
    "fetch",
    "merge",
    "rebase",
    "reset",
    "rm",
    "mv",
    "tag",
    "init",
    "config",
    "remote",
    "STATUS",  # case-sensitive: not the lowercase "status"
    "Diff",
    "log ",  # trailing space
    " status",  # leading space
    "",  # empty operation name
    "status;rm -rf",  # injection-looking junk is still just "unsupported"
]

# Operations drawn from a mix of the supported set and arbitrary other strings,
# so a single test exercises both arms of the dispatch "iff".
_operation = st.one_of(
    st.sampled_from(sorted(SUPPORTED_OPERATIONS)),
    st.sampled_from(_KNOWN_UNSUPPORTED),
    st.text(max_size=12),
)


# Feature: forge, Property 16: Git operation dispatch
@settings(max_examples=10)
@given(operation=_operation)
def test_git_dispatches_iff_operation_supported(operation: str) -> None:
    """The git tool dispatches an operation iff it is in the supported set.

    Supported names are forwarded to ``git`` (a real ``git <operation>`` call is
    issued); every other name yields an "unsupported" result and ``git`` is
    never invoked.

    **Validates: Requirements 9.1, 9.4**
    """
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        # Record the argv of every git invocation so we can assert whether (and
        # how) git was dispatched. The is-repo probe must report "true" so a
        # supported operation proceeds to its dispatched call.
        calls.append(list(argv))
        return _FakeCompleted(stdout=b"true\n", stderr=b"", returncode=0)

    tool = GitTool()
    ctx = _ctx(output_cap_chars=_DEFAULT_CAP)

    with patch("forge.tools.git.subprocess.run", side_effect=fake_run):
        result = tool.run({"operation": operation}, ctx)

    if operation in SUPPORTED_OPERATIONS:
        # Dispatched: git was invoked and one call is exactly `git <operation>`.
        assert result.meta.get("unsupported") is None
        assert result.ok is True
        assert ["git", operation] in calls
    else:
        # Not dispatched: git was never invoked and the result says so.
        assert calls == []
        assert result.ok is False
        assert result.meta.get("unsupported") is True
        assert "unsupported" in (result.error or "")


def test_supported_set_is_exactly_the_documented_nine() -> None:
    """The supported set is exactly the nine operations from Requirement 9.1."""
    assert SUPPORTED_OPERATIONS == frozenset(
        {
            "status",
            "diff",
            "log",
            "show",
            "add",
            "commit",
            "branch",
            "checkout",
            "stash",
        }
    )


# ---------------------------------------------------------------------------
# Property 12 (git portion): output char cap (Req 7.5, 9.6)
# ---------------------------------------------------------------------------

# Feature: forge, Property 12: Shell and git output char cap
@settings(max_examples=10)
@given(
    text=st.text(max_size=600),
    cap=st.integers(min_value=0, max_value=600),
)
def test_git_output_is_capped_and_flags_truncation(text: str, cap: int) -> None:
    """The git tool caps output to ``cap`` chars and flags truncation exactly
    when the output exceeds the cap.

    Exercises the tool's own ``_cap`` helper -- the exact truncation logic
    ``GitTool.run`` applies to git's stdout/stderr -- across many output lengths
    and caps that straddle the boundary in both directions.

    **Validates: Requirements 7.5, 9.6**
    """
    ctx = _ctx(output_cap_chars=cap)

    capped, truncated = GitTool._cap(text, ctx)

    # Core cap guarantee: never return more than the cap (Req 9.6).
    assert len(capped) <= cap
    # Truncation flag is set exactly when the output exceeded the cap.
    assert truncated is (len(text) > cap)

    if truncated:
        assert capped == text[:cap]
    else:
        # Within the cap: returned whole, byte-for-byte unchanged.
        assert capped == text


def test_git_cap_defaults_to_30000_when_config_absent() -> None:
    """With no config, the cap falls back to the documented 30,000 chars.

    **Validates: Requirements 9.6**
    """
    ctx = _ctx()  # no config at all
    over = "a" * (_DEFAULT_CAP + 50)

    capped, truncated = GitTool._cap(over, ctx)

    assert truncated is True
    assert len(capped) == _DEFAULT_CAP
    assert capped == over[:_DEFAULT_CAP]


def test_git_cap_at_default_boundary_is_not_truncated() -> None:
    """Output exactly at the 30,000 default cap is returned whole, unflagged.

    **Validates: Requirements 9.6**
    """
    ctx = _ctx()  # no config -> 30,000 default
    exact = "b" * _DEFAULT_CAP

    capped, truncated = GitTool._cap(exact, ctx)

    assert truncated is False
    assert capped == exact
