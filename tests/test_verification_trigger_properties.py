"""Property-based test for the Verification_Phase trigger gate.

# Feature: auto-verification-loop, Property 3: Trigger decision gates the phase correctly

This test exercises the pure ``should_verify`` gate over every combination of
command presence, trigger value, File_Mutation flag, and turn-completion
status, asserting it returns ``True`` iff a Verify_Command is configured AND
the turn completed normally AND (the trigger is ``"always"`` OR the turn
mutated files).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.verification import should_verify

# The two valid Trigger values per the configuration spec.
TRIGGERS = ("on_file_change", "always")


@settings(max_examples=200)
@given(
    command_present=st.booleans(),
    trigger=st.sampled_from(TRIGGERS),
    mutated_files=st.booleans(),
    turn_ok=st.booleans(),
)
def test_trigger_decision_gates_the_phase_correctly(
    command_present: bool,
    trigger: str,
    mutated_files: bool,
    turn_ok: bool,
) -> None:
    """should_verify is equivalent to the documented gating expression.

    Validates: Requirements 2.1, 2.2, 3.1, 3.2, 3.3, 3.4
    """
    expected = (
        command_present
        and turn_ok
        and (trigger == "always" or mutated_files)
    )

    assert should_verify(command_present, trigger, mutated_files, turn_ok) is expected
