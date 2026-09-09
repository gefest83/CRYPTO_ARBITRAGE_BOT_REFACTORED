"""Isolated Kronos config for phase 2 (no Settings coupling)."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["KronosConfig"]


class KronosConfig(BaseModel):
    """Offline-safe Kronos thresholds.

    All bps values are Decimal in basis points (1 bps = 0.01%).
    Costs are one-way entry costs by default (taker + half-spread + slippage).
    Set taker_fee_bps to round-trip (e.g. 20) if you want round-trip costing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # data window
    lookback: int = Field(default=400, ge=10, le=2048, description="Min 1m bars required")
    pred_len_1m: int = Field(default=5, ge=1, le=120, description="1m bars ahead to forecast")
    pred_len_5m: int = Field(default=1, ge=1, le=24, description="5m bars ahead")
    min_candles: int | None = Field(default=None, description="Override; default=lookback")

    # edge
    min_edge_bps: Decimal = Field(default=Decimal("15"), ge=0, description="Minimum net bps to trigger BUY/SELL")
    taker_fee_bps: Decimal = Field(default=Decimal("10"), ge=0)
    spread_bps: Decimal = Field(default=Decimal("5"), ge=0)
    slippage_bps: Decimal = Field(default=Decimal("5"), ge=0)
    cost_model: Literal["one_way", "round_trip"] = Field(default="one_way")

    # confirmation
    require_5m_confirmation: bool = Field(default=True)
    # if True, opposite 5m sign forces HOLD

    # freshness / integrity
    max_candle_age_ms: int = Field(default=90_000, ge=1_000, description="Last candle must be fresher than this")
    max_gap_ms: int = Field(default=90_000, ge=60_000, description="Max allowed gap between consecutive 1m candles")
    allow_incomplete_last: bool = Field(default=False, description="If False, incomplete last candle -> HOLD")

    @field_validator("min_edge_bps", "taker_fee_bps", "spread_bps", "slippage_bps", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: object) -> object:
        if isinstance(v, (int, float, str)):
            return Decimal(str(v))
        return v

    @property
    def effective_min_candles(self) -> int:
        return self.min_candles if self.min_candles is not None else self.lookback

    @property
    def total_cost_bps(self) -> Decimal:
        """Precomputed cost for cost_model."""
        if self.cost_model == "round_trip":
            # round-trip: 2*taker + spread + 2*slippage (conservative)
            return self.taker_fee_bps * 2 + self.spread_bps + self.slippage_bps * 2
        # one_way: taker + half spread + slippage
        return self.taker_fee_bps + (self.spread_bps / 2) + self.slippage_bps
