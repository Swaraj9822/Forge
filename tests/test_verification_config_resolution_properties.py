"""Property-based tests for verification configuration resolution defaults.

Feature: auto-verification-loop, Property 1: Configuration resolution applies documented defaults and reads present values

These properties exercise :func:`forge.config.resolve_verification_config`
directly. They assert the universal resolution invariants behind Requirement 1:

* 1.1 - ``command`` is read from ``verification.command`` when present.
* 1.2 - ``command`` resolves to ``None`` when absent (feature disabled).
* 1.3 - ``max_correction_iterations`` resolves to ``3`` when absent.
* 1.4 - ``trigger`` resolves to ``on_file_change`` when absent.
* 1.7 - ``timeout_s`` resolves to the provided ``shell_timeout_s`` when absent.
* 1.8 - ``timeout_s`` resolves to the provided positive integer when present.

The composite generator independently includes/omits each ``[verification]``
key so the input space covers absent keys, present values, zero and large
integers, and both allowed trigger values. ``output_cap_chars`` is always
inherited from the provided value.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.7, 1.8
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.config import (
    VERIFICATION_TRIGGERS,
    resolve_verification_config,
)

# Sentinel marking "this key is omitted from the raw [verification] mapping".
_ABSENT = object()

# A command value: present strings (including empty/whitespace) or absent.
_command_values = st.one_of(
    st.just(_ABSENT),
    st.text(max_size=40),
)

# A valid present max_correction_iterations (integer >= 0, covering zero and
# large values), or absent.
_max_iters_values = st.one_of(
    st.just(_ABSENT),
    st.integers(min_value=0, max_value=1_000_000),
)

# A valid present trigger (one of the allowed values), or absent.
_trigger_values = st.one_of(
    st.just(_ABSENT),
    st.sampled_from(VERIFICATION_TRIGGERS),
)

# A valid present positive-integer timeout, or absent.
_timeout_values = st.one_of(
    st.just(_ABSENT),
    st.integers(min_value=1, max_value=1_000_000),
)


@st.composite
def raw_verification_mappings(draw: st.DrawFn) -> dict[str, object]:
    """Build a raw ``[verification]`` mapping, independently including/omitting
    each key so the generated space covers every absent/present combination."""

    raw: dict[str, object] = {}

    command = draw(_command_values)
    if command is not _ABSENT:
        raw["command"] = command

    max_iters = draw(_max_iters_values)
    if max_iters is not _ABSENT:
        raw["max_correction_iterations"] = max_iters

    trigger = draw(_trigger_values)
    if trigger is not _ABSENT:
        raw["trigger"] = trigger

    timeout_s = draw(_timeout_values)
    if timeout_s is not _ABSENT:
        raw["timeout_s"] = timeout_s

    return raw


@settings(max_examples=200)
@given(
    raw=raw_verification_mappings(),
    shell_timeout_s=st.integers(min_value=1, max_value=1_000_000),
    output_cap_chars=st.integers(min_value=0, max_value=1_000_000),
)
def test_resolution_applies_defaults_and_reads_present_values(
    raw: dict[str, object],
    shell_timeout_s: int,
    output_cap_chars: int,
) -> None:
    """For any raw mapping and inherited values, resolution reads present
    values and applies the documented defaults for absent ones."""
    resolved = resolve_verification_config(
        raw,
        shell_timeout_s=shell_timeout_s,
        output_cap_chars=output_cap_chars,
    )

    # command: provided value when present, else None (Req 1.1, 1.2).
    if "command" in raw:
        assert resolved.command == raw["command"]
    else:
        assert resolved.command is None

    # max_correction_iterations: provided int when present, else 3 (Req 1.3).
    if "max_correction_iterations" in raw:
        assert resolved.max_correction_iterations == raw["max_correction_iterations"]
    else:
        assert resolved.max_correction_iterations == 3

    # trigger: provided allowed value when present, else on_file_change (Req 1.4).
    if "trigger" in raw:
        assert resolved.trigger == raw["trigger"]
    else:
        assert resolved.trigger == "on_file_change"

    # timeout_s: provided positive int when present, else shell_timeout_s
    # (Req 1.7, 1.8).
    if "timeout_s" in raw:
        assert resolved.timeout_s == raw["timeout_s"]
    else:
        assert resolved.timeout_s == shell_timeout_s

    # output_cap_chars always inherits the provided value.
    assert resolved.output_cap_chars == output_cap_chars
