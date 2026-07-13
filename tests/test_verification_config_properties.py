"""Property-based tests for verification configuration validation.

Feature: auto-verification-loop, Property 2: Configuration validation rejects invalid values

These properties exercise :func:`forge.config.resolve_verification_config`
directly. They assert the universal validation invariants behind Requirement 1:

* 1.5 - a present ``verification.max_correction_iterations`` that is not an
  integer ``>= 0`` (negative ints, booleans, or non-integers) is rejected with
  a :class:`ConfigError` naming the offending value.
* 1.6 - a present ``verification.trigger`` that is not one of
  ``on_file_change`` or ``always`` is rejected with a :class:`ConfigError`
  naming the offending value and the allowed values.

Note: ``bool`` is a subclass of ``int`` in Python, so ``True``/``False`` must be
rejected as non-integers even though they would otherwise satisfy ``>= 0``.

Validates: Requirements 1.5, 1.6
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from forge.config import (
    VERIFICATION_TRIGGERS,
    ConfigError,
    resolve_verification_config,
)

# Fixed, valid inherited values so the only thing under test is the validation
# of max_correction_iterations / trigger.
SHELL_TIMEOUT_S = 120
OUTPUT_CAP_CHARS = 30_000


# A value for max_correction_iterations that is NOT an integer >= 0. Covers:
# negative integers, booleans (bool is an int subclass and must be rejected),
# and assorted non-integers (floats, text, None, containers).
invalid_max_iters = st.one_of(
    st.integers(max_value=-1),
    st.booleans(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=10),
    st.none(),
    st.lists(st.integers(), max_size=3),
)

# A value for trigger that is NOT one of the allowed values. Covers arbitrary
# strings outside the allowed set plus non-string types.
invalid_trigger = st.one_of(
    st.text(max_size=10).filter(lambda s: s not in VERIFICATION_TRIGGERS),
    st.integers(),
    st.booleans(),
    st.none(),
    st.floats(allow_nan=False, allow_infinity=False),
)


@settings(max_examples=200)
@given(value=invalid_max_iters)
def test_invalid_max_correction_iterations_is_rejected(value: object) -> None:
    """Any non-(integer >= 0) max_correction_iterations raises ConfigError
    naming the offending value (Req 1.5)."""
    # A valid (absent) trigger ensures the raised error is about max_iters.
    raw = {"max_correction_iterations": value}

    with pytest.raises(ConfigError) as exc_info:
        resolve_verification_config(
            raw,
            shell_timeout_s=SHELL_TIMEOUT_S,
            output_cap_chars=OUTPUT_CAP_CHARS,
        )

    message = str(exc_info.value)
    # The error names the configuration key and the offending value.
    assert "max_correction_iterations" in message
    assert repr(value) in message


@settings(max_examples=200)
@given(value=invalid_trigger)
def test_invalid_trigger_is_rejected(value: object) -> None:
    """Any trigger not in {on_file_change, always} raises ConfigError naming
    the offending value and the allowed values (Req 1.6)."""
    # An absent max_correction_iterations keeps validation focused on trigger.
    raw = {"trigger": value}

    with pytest.raises(ConfigError) as exc_info:
        resolve_verification_config(
            raw,
            shell_timeout_s=SHELL_TIMEOUT_S,
            output_cap_chars=OUTPUT_CAP_CHARS,
        )

    message = str(exc_info.value)
    # The error names the offending value...
    assert repr(value) in message
    # ...and every allowed value so the user knows the valid options.
    for allowed in VERIFICATION_TRIGGERS:
        assert allowed in message
