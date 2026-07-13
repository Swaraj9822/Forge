"""Property-based test for Verification_Phase usage aggregation.

# Feature: auto-verification-loop, Property 8: Phase usage aggregates every Model request

Exercises the pure :func:`forge.verification.aggregate_usage` fold over the
ordered list of per-turn :class:`~forge.usage.UsageSummary` snapshots (the
original turn followed by one snapshot per Correction_Iteration). The property
is fully OFFLINE: it operates on plain data models, no network call is made.

Validates: Requirements 10.1, 10.2, 10.3.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.usage import UsageSummary
from forge.verification import aggregate_usage

# Token counts mirror what ``usage_metadata`` reports: non-negative integers.
TOKENS = st.integers(min_value=0, max_value=1_000_000)
# Cost estimates are non-negative floats; a summary may report cost as
# unavailable (None / cost_available=False) to exercise mixed availability.
COST = st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False)


@st.composite
def usage_summaries(draw: st.DrawFn) -> UsageSummary:
    """Generate a single per-turn UsageSummary with mixed cost availability."""

    cost_available = draw(st.booleans())
    turn_cost = draw(COST) if cost_available else None
    cumulative_cost = draw(COST) if cost_available else None
    return UsageSummary(
        turn_input_tokens=draw(TOKENS),
        turn_output_tokens=draw(TOKENS),
        cumulative_input_tokens=draw(TOKENS),
        cumulative_output_tokens=draw(TOKENS),
        turn_cost=turn_cost,
        cumulative_cost=cumulative_cost,
        cost_available=cost_available,
    )


@settings(max_examples=200)
@given(st.lists(usage_summaries(), min_size=1, max_size=12))
def test_phase_usage_aggregates_every_model_request(
    turn_usages: list[UsageSummary],
) -> None:
    """Property 8: aggregated turn tokens sum every request; cumulative is final.

    For any non-empty sequence of per-turn UsageSummaries, the aggregated turn
    token counts equal the sum of every turn's token counts, and the cumulative
    token counts equal the final summary's cumulative totals (Req 10.1, 10.2,
    10.3).
    """

    aggregated = aggregate_usage(turn_usages)

    assert aggregated.turn_input_tokens == sum(
        u.turn_input_tokens for u in turn_usages
    )
    assert aggregated.turn_output_tokens == sum(
        u.turn_output_tokens for u in turn_usages
    )

    final = turn_usages[-1]
    assert aggregated.cumulative_input_tokens == final.cumulative_input_tokens
    assert aggregated.cumulative_output_tokens == final.cumulative_output_tokens


def test_empty_list_yields_zeroed_cost_unavailable_summary() -> None:
    """Documented edge case: an empty list yields a zeroed, cost-unavailable summary."""

    aggregated = aggregate_usage([])

    assert aggregated.turn_input_tokens == 0
    assert aggregated.turn_output_tokens == 0
    assert aggregated.cumulative_input_tokens == 0
    assert aggregated.cumulative_output_tokens == 0
    assert aggregated.turn_cost is None
    assert aggregated.cumulative_cost is None
    assert aggregated.cost_available is False
