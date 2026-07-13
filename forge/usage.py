"""Token usage and cost tracking for Forge.

Defines :class:`UsageSummary` (an immutable snapshot of a turn's and the
session's token tallies plus estimated cost) and :class:`UsageTracker`, which
records per-response token counts reported in ``usage_metadata``, accumulates
per-turn and cumulative (session) totals, and computes estimated cost from the
active model's :class:`~forge.config.ModelPricing`.

When per-token pricing is unavailable (either ``input_per_1k`` or
``output_per_1k`` is ``None``), the tracker reports token counts but marks the
estimated cost as unavailable so the REPL can render "cost unavailable".

See the design document's "UsageTracker" section and Property 24.
Requirements: 17.1, 17.2, 17.3, 17.4, 17.5.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.config import Config, ModelPricing

# Per-token pricing is quoted per 1,000 tokens, so cost divides token counts by
# this divisor before applying the rate (see design's cost formula).
_TOKENS_PER_PRICING_UNIT = 1000


@dataclass(frozen=True)
class UsageSummary:
    """Immutable snapshot of token usage and estimated cost.

    Carries both the current turn's tallies and the cumulative (session)
    totals, along with the estimated cost for each. When pricing is
    unavailable, ``turn_cost`` and ``cumulative_cost`` are ``None`` and
    ``cost_available`` is ``False`` so the REPL can render a
    "cost unavailable" indication (Req 17.5).
    """

    turn_input_tokens: int
    turn_output_tokens: int
    cumulative_input_tokens: int
    cumulative_output_tokens: int
    turn_cost: float | None
    cumulative_cost: float | None
    cost_available: bool


class UsageTracker:
    """Records token usage and computes estimated cost.

    ``record`` is called once per model response with the input/output token
    counts reported in ``usage_metadata`` (Req 17.1); it accumulates both the
    per-turn and cumulative (session) totals (Req 17.2). A turn may contain
    multiple responses (the multi-tool loop), so turns are delimited explicitly:
    :meth:`begin_turn` resets the per-turn counters at the start of each turn,
    and :meth:`turn_summary` reports the current turn's tallies plus the
    cumulative totals and estimated cost (Req 17.3, 17.4).
    """

    def __init__(self, pricing: ModelPricing | Config | None = None) -> None:
        """Construct with the active model's pricing.

        Accepts a :class:`~forge.config.ModelPricing` directly, or a
        :class:`~forge.config.Config` (whose ``pricing`` field is used). When
        omitted, cost is treated as unavailable.
        """

        if isinstance(pricing, Config):
            self._pricing: ModelPricing | None = pricing.pricing
        else:
            self._pricing = pricing

        self._turn_input = 0
        self._turn_output = 0
        self._cumulative_input = 0
        self._cumulative_output = 0

    def seed(self, input_tokens: int, output_tokens: int) -> None:
        """Seed the cumulative (session) counters from a restored session.

        Used when resuming a saved session so the cumulative totals continue
        from the persisted usage rather than restarting at zero (Req 17.2).
        Per-turn counters are unaffected.
        """

        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")

        self._cumulative_input = input_tokens
        self._cumulative_output = output_tokens

    def begin_turn(self) -> None:
        """Reset the per-turn counters at the start of a new turn.

        Cumulative (session) totals are preserved across turns.
        """

        self._turn_input = 0
        self._turn_output = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        """Record one response's token counts (Req 17.1, 17.2).

        Adds to both the per-turn and cumulative totals. Negative counts are
        not expected from ``usage_metadata`` and are rejected.
        """

        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")

        self._turn_input += input_tokens
        self._turn_output += output_tokens
        self._cumulative_input += input_tokens
        self._cumulative_output += output_tokens

    def turn_summary(self) -> UsageSummary:
        """Return the current turn's tallies plus cumulative totals and cost.

        Estimated cost is computed from the configured per-token pricing
        (Req 17.4). When pricing is unavailable, cost fields are ``None`` and
        ``cost_available`` is ``False`` (Req 17.5).
        """

        cost_available = self._cost_available()
        turn_cost = (
            self._cost(self._turn_input, self._turn_output)
            if cost_available
            else None
        )
        cumulative_cost = (
            self._cost(self._cumulative_input, self._cumulative_output)
            if cost_available
            else None
        )

        return UsageSummary(
            turn_input_tokens=self._turn_input,
            turn_output_tokens=self._turn_output,
            cumulative_input_tokens=self._cumulative_input,
            cumulative_output_tokens=self._cumulative_output,
            turn_cost=turn_cost,
            cumulative_cost=cumulative_cost,
            cost_available=cost_available,
        )

    def _cost_available(self) -> bool:
        """Cost is available only when both per-token rates are present."""

        return (
            self._pricing is not None
            and self._pricing.input_per_1k is not None
            and self._pricing.output_per_1k is not None
        )

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        """Compute cost from token counts and the configured per-1k rates.

        Cost = (input/1000 * input_per_1k) + (output/1000 * output_per_1k).
        Callers must ensure :meth:`_cost_available` is ``True`` first.
        """

        assert self._pricing is not None
        assert self._pricing.input_per_1k is not None
        assert self._pricing.output_per_1k is not None

        input_cost = (input_tokens / _TOKENS_PER_PRICING_UNIT) * self._pricing.input_per_1k
        output_cost = (
            output_tokens / _TOKENS_PER_PRICING_UNIT
        ) * self._pricing.output_per_1k
        return input_cost + output_cost
