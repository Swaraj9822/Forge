"""Property-based test for validation safety.

# Feature: forge, Property 5: Invalid arguments never cause side effects

Property 5 (Validates: Requirements 4.7): For any tool call whose arguments
FAIL validation, the ``ToolExecutor`` returns a validation-error
``ToolResult`` and the workspace and session state are left unchanged -- the
tool's ``run`` (which performs side effects) is never reached.

The workspace lives in a ``tempfile.mkdtemp`` directory created once at module
scope and cleaned up at exit. This deliberately avoids combining a
function-scoped pytest ``tmp_path`` fixture with Hypothesis ``@given`` (which
Hypothesis warns about, since the fixture would be created once and shared
across all generated examples) and sidesteps the host's ``tmp_path`` issues.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.interrupt import InterruptController
from forge.session import ToolCall
from forge.tools.base import ToolContext, ToolExecutor, ToolResult

# --- Module-scoped workspace --------------------------------------------------
# realpath so comparisons stay stable on platforms whose temp dir contains
# symlinks (macOS /var -> /private/var) or short (8.3) names on Windows.
_WORKSPACE_ROOT = Path(os.path.realpath(tempfile.mkdtemp(prefix="forge_valsafe_")))

# A pre-existing file so "workspace unchanged" is a meaningful assertion: its
# presence and contents must survive every rejected call.
_SENTINEL_NAME = "preexisting.txt"
_SENTINEL_CONTENT = "do not touch"
(_WORKSPACE_ROOT / _SENTINEL_NAME).write_text(_SENTINEL_CONTENT, encoding="utf-8")

# The name the side-effecting tool would create if its run() ever executed.
_SIDE_EFFECT_NAME = "SIDE_EFFECT.txt"


@atexit.register
def _cleanup_dir() -> None:
    shutil.rmtree(_WORKSPACE_ROOT, ignore_errors=True)


def _is_valid(args: object) -> bool:
    """The fake tool's validation contract.

    Args are valid iff they are a dict carrying a non-empty string ``path`` and
    a string ``content``. Everything else is invalid. Shared between the tool's
    ``validate`` and the generator's filter so generated args are guaranteed to
    be invalid per the tool's own rule.
    """
    return (
        isinstance(args, dict)
        and isinstance(args.get("path"), str)
        and args.get("path") != ""
        and isinstance(args.get("content"), str)
    )


@dataclass
class SideEffectingTool:
    """A :class:`Tool` whose ``run`` performs real side effects.

    ``run`` writes a file into the workspace AND mutates shared session state
    (``ctx.state``) and appends to ``log``. ``validate`` rejects invalid args
    per :func:`_is_valid`, so the ONLY thing that can stop ``run`` from firing
    is the executor honoring the validation error -- exactly what Property 5
    asserts.
    """

    log: list[str]
    name: str = "writer"
    description: str = "a side-effecting test tool"
    parameters: dict = field(default_factory=dict)

    def validate(self, args: dict) -> str | None:
        if _is_valid(args):
            return None
        return "invalid arguments: require non-empty string 'path' and string 'content'"

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        # Side effect 1: write into the workspace.
        (ctx.workspace_root / _SIDE_EFFECT_NAME).write_text(
            str(args.get("content", "")), encoding="utf-8"
        )
        # Side effect 2: mutate shared session state.
        ctx.state.setdefault("runs", []).append(args)
        # Side effect 3: record that run() executed.
        self.log.append("ran")
        return ToolResult(ok=True, content="wrote file")


def _snapshot_workspace() -> dict[str, str]:
    """Map of relative file name -> contents for every file in the workspace."""
    return {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted(_WORKSPACE_ROOT.iterdir())
        if p.is_file()
    }


# A non-empty string usable as a valid "path"/"content" value, so that each
# invalid category fails on exactly one axis (the missing/wrong field).
_valid_str = st.text(min_size=1, max_size=20)

# Values of the WRONG type for path/content (anything that is not a str).
_non_str = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.lists(st.integers(), max_size=3),
    st.dictionaries(st.text(max_size=5), st.integers(), max_size=3),
)

# Optional extra keys that never satisfy the contract on their own.
_extras = st.dictionaries(
    keys=st.sampled_from(["extra", "mode", "flag"]),
    values=st.one_of(st.text(max_size=5), st.integers()),
    max_size=3,
)


def _with_extras(base: dict, extras: dict) -> dict:
    merged = dict(extras)
    merged.update(base)  # base keys win so the invalid shape is preserved
    return merged


# Each branch is invalid by construction, so no filtering is needed: the
# generator stays fast while still covering every failure mode of the contract.
_invalid_args = st.one_of(
    # Empty dict -- both required keys missing.
    st.just({}),
    # Missing "content".
    st.builds(_with_extras, st.builds(lambda p: {"path": p}, _valid_str), _extras),
    # Missing "path".
    st.builds(_with_extras, st.builds(lambda c: {"content": c}, _valid_str), _extras),
    # "path" present but empty string.
    st.builds(
        _with_extras,
        st.builds(lambda c: {"path": "", "content": c}, _valid_str),
        _extras,
    ),
    # "path" wrong type.
    st.builds(
        _with_extras,
        st.builds(lambda p, c: {"path": p, "content": c}, _non_str, _valid_str),
        _extras,
    ),
    # "content" wrong type.
    st.builds(
        _with_extras,
        st.builds(lambda p, c: {"path": p, "content": c}, _valid_str, _non_str),
        _extras,
    ),
).filter(lambda a: not _is_valid(a))  # cheap safety guard; rarely rejects


@settings(max_examples=10)
@given(args=_invalid_args)
def test_invalid_arguments_never_cause_side_effects(args: dict) -> None:
    """A validation failure yields a validation-error result and no side effects.

    Validates: Requirements 4.7
    """
    log: list[str] = []
    tool = SideEffectingTool(log=log)
    interrupt = InterruptController()  # real controller, never tripped
    state: dict = {}
    context = ToolContext(
        workspace_root=_WORKSPACE_ROOT,
        interrupt=interrupt,
        state=state,
    )
    executor = ToolExecutor(
        registry={tool.name: tool},
        enabled={tool.name},  # tool IS enabled: only validation can stop run()
        interrupt=interrupt,
        context=context,
    )

    # Snapshot workspace + session state BEFORE the call.
    workspace_before = _snapshot_workspace()
    state_before = dict(state)

    result = executor.execute(ToolCall(id="call-1", name=tool.name, args=args))

    # --- The result is a validation error --------------------------------
    assert result.ok is False
    assert result.meta.get("validation_error") is True
    assert result.error is not None

    # --- No side effects: run() never executed ---------------------------
    assert log == []  # run() did not append
    assert state == state_before == {}  # session state unchanged
    # Workspace unchanged: same files, same contents, and crucially the
    # side-effect file was never created.
    workspace_after = _snapshot_workspace()
    assert workspace_after == workspace_before
    assert _SIDE_EFFECT_NAME not in workspace_after
    assert workspace_after == {_SENTINEL_NAME: _SENTINEL_CONTENT}
