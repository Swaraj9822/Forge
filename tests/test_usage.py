"""Property-based test for usage accumulation and cost.

# Feature: forge, Property 24: Usage accumulation and cost
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from forge.config import Config, ModelPricing
from forge.usage import UsageTracker

# Per-response (input, output) token counts: non-negative integers, bounded to
# keep generated sequences realistic and float arithmetic well-behaved.
TOKEN_PAIRS = st.lists(
    st.tuples(st.integers(min_value=0, max_value=10**6), st.integers(min_value=0, max_value=10**6)),
    max_size=50,
)

# Per-1k-token rates: finite, non-negative, bounded so the pricing formula
# stays numerically stable (no overflow / precision blow-ups).
RATES = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)

# How many 1,000-token units a per-1k rate is divided by in the cost formula.
_TOKENS_PER_PRICING_UNIT = 1000


@settings(max_examples=10)
@given(pairs=TOKEN_PAIRS, in_rate=RATES, out_rate=RATES)
def test_usage_accumulation_and_cost_with_pricing(
    pairs: list[tuple[int, int]], in_rate: float, out_rate: float
) -> None:
    """For any sequence of recorded (input, output) token counts and any
    available pricing, cumulative/turn token totals equal the sums of the
    recorded counts and the estimated cost equals the pricing formula applied
    to those totals.

    Validates: Requirements 17.1, 17.2, 17.4
    """

    tracker = UsageTracker(ModelPricing(input_per_1k=in_rate, output_per_1k=out_rate))
    tracker.begin_turn()
    for input_tokens, output_tokens in pairs:
        tracker.record(input_tokens, output_tokens)

    summary = tracker.turn_summary()

    sum_in = sum(p[0] for p in pairs)
    sum_out = sum(p[1] for p in pairs)

    # Token accumulation: cumulative totals equal the sums (Req 17.1, 17.2).
    assert summary.cumulative_input_tokens == sum_in
    assert summary.cumulative_output_tokens == sum_out

    # Single turn: per-turn totals equal the same sums (Req 17.2).
    assert summary.turn_input_tokens == sum_in
    assert summary.turn_output_tokens == sum_out

    # Cost is available and matches the pricing formula on the totals (Req 17.4).
    expected_turn_cost = (
        (sum_in / _TOKENS_PER_PRICING_UNIT) * in_rate
        + (sum_out / _TOKENS_PER_PRICING_UNIT) * out_rate
    )
    assert summary.cost_available is True
    assert summary.turn_cost == pytest.approx(expected_turn_cost)
    assert summary.cumulative_cost == pytest.approx(expected_turn_cost)


@settings(max_examples=10)
@given(
    turn1=TOKEN_PAIRS,
    turn2=TOKEN_PAIRS,
    in_rate=RATES,
    out_rate=RATES,
)
def test_usage_per_turn_resets_cumulative_continues(
    turn1: list[tuple[int, int]],
    turn2: list[tuple[int, int]],
    in_rate: float,
    out_rate: float,
) -> None:
    """Across two turns, ``begin_turn`` resets the per-turn counters while the
    cumulative (session) totals keep growing, and cost on each reflects the
    pricing formula applied to the respective totals.

    Validates: Requirements 17.1, 17.2, 17.4
    """

    tracker = UsageTracker(ModelPricing(input_per_1k=in_rate, output_per_1k=out_rate))

    tracker.begin_turn()
    for input_tokens, output_tokens in turn1:
        tracker.record(input_tokens, output_tokens)

    # Start a second turn: per-turn counters reset, cumulative preserved.
    tracker.begin_turn()
    for input_tokens, output_tokens in turn2:
        tracker.record(input_tokens, output_tokens)

    summary = tracker.turn_summary()

    sum_in_1 = sum(p[0] for p in turn1)
    sum_out_1 = sum(p[1] for p in turn1)
    sum_in_2 = sum(p[0] for p in turn2)
    sum_out_2 = sum(p[1] for p in turn2)

    # Per-turn totals reflect only the second turn (reset by begin_turn).
    assert summary.turn_input_tokens == sum_in_2
    assert summary.turn_output_tokens == sum_out_2

    # Cumulative totals span both turns.
    assert summary.cumulative_input_tokens == sum_in_1 + sum_in_2
    assert summary.cumulative_output_tokens == sum_out_1 + sum_out_2

    expected_turn_cost = (
        (sum_in_2 / _TOKENS_PER_PRICING_UNIT) * in_rate
        + (sum_out_2 / _TOKENS_PER_PRICING_UNIT) * out_rate
    )
    expected_cumulative_cost = (
        ((sum_in_1 + sum_in_2) / _TOKENS_PER_PRICING_UNIT) * in_rate
        + ((sum_out_1 + sum_out_2) / _TOKENS_PER_PRICING_UNIT) * out_rate
    )
    assert summary.cost_available is True
    assert summary.turn_cost == pytest.approx(expected_turn_cost)
    assert summary.cumulative_cost == pytest.approx(expected_cumulative_cost)


@settings(max_examples=10)
@given(pairs=TOKEN_PAIRS, pricing=st.none() | st.builds(
    ModelPricing,
    input_per_1k=st.none() | RATES,
    output_per_1k=st.none() | RATES,
))
def test_usage_accumulation_without_pricing(
    pairs: list[tuple[int, int]], pricing: ModelPricing | None
) -> None:
    """Token accumulation holds regardless of pricing; when pricing is absent
    (no tracker pricing, or either rate is ``None``) cost is marked
    unavailable and the cost fields are ``None``.

    Validates: Requirements 17.1, 17.2
    """

    tracker = UsageTracker(pricing)
    tracker.begin_turn()
    for input_tokens, output_tokens in pairs:
        tracker.record(input_tokens, output_tokens)

    summary = tracker.turn_summary()

    sum_in = sum(p[0] for p in pairs)
    sum_out = sum(p[1] for p in pairs)

    # Token accumulation is independent of pricing availability (Req 17.1, 17.2).
    assert summary.cumulative_input_tokens == sum_in
    assert summary.cumulative_output_tokens == sum_out
    assert summary.turn_input_tokens == sum_in
    assert summary.turn_output_tokens == sum_out

    # Determine whether cost should be available for the generated pricing.
    pricing_available = (
        pricing is not None
        and pricing.input_per_1k is not None
        and pricing.output_per_1k is not None
    )
    assert summary.cost_available is pricing_available
    if not pricing_available:
        assert summary.turn_cost is None
        assert summary.cumulative_cost is None


# ---------------------------------------------------------------------------
# Unit test: cost-unavailable display (Req 17.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pricing",
    [
        pytest.param(None, id="no-pricing"),
        pytest.param(ModelPricing(input_per_1k=None, output_per_1k=None), id="both-none"),
        pytest.param(ModelPricing(input_per_1k=0.00125, output_per_1k=None), id="output-none"),
        pytest.param(ModelPricing(input_per_1k=None, output_per_1k=0.005), id="input-none"),
        pytest.param(Config(), id="config-default-pricing"),
    ],
)
def test_tokens_reported_with_cost_unavailable_when_pricing_absent(
    pricing: ModelPricing | Config | None,
) -> None:
    """When pricing is absent, the tracker still reports token counts but marks
    the estimated cost as unavailable (``cost_available`` is ``False`` and both
    cost fields are ``None``) so the REPL can render "cost unavailable".

    Validates: Requirements 17.5
    """

    tracker = UsageTracker(pricing)
    tracker.begin_turn()
    tracker.record(1500, 500)
    tracker.record(2500, 1000)

    summary = tracker.turn_summary()

    # Token counts are still reported (Req 17.5 reports tokens regardless).
    assert summary.turn_input_tokens == 4000
    assert summary.turn_output_tokens == 1500
    assert summary.cumulative_input_tokens == 4000
    assert summary.cumulative_output_tokens == 1500

    # Cost is marked unavailable rather than reported as a number.
    assert summary.cost_available is False
    assert summary.turn_cost is None
    assert summary.cumulative_cost is None
