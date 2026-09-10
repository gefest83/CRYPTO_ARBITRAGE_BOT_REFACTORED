"""Execution constraints & fees for Binance Prediction Markets (research-only)."""

from __future__ import annotations

from decimal import Decimal

from app.research.prediction_markets.models import PredictionConstraints

__all__ = ["derive_constraints", "estimate_round_trip_cost_bps"]


def derive_constraints(raw_topic: dict | None = None) -> PredictionConstraints:
    """Derive constraints from a raw marketTopic (fees, slippage, collateral)."""
    if raw_topic is None:
        return PredictionConstraints()
    fee = raw_topic.get("feeRateBps")
    slip = raw_topic.get("slippageBps")
    collateral = raw_topic.get("collateral") or "USDT"
    try:
        fee_int = int(fee) if fee is not None else 200
    except Exception:
        fee_int = 200
    try:
        slip_int = int(slip) if slip is not None else 1200
    except Exception:
        slip_int = 1200
    return PredictionConstraints(
        fee_rate_bps=fee_int,
        slippage_bps=slip_int,
        collateral=str(collateral),
        chain_id=str(raw_topic.get("chainId", "56")),
        vendor=str(raw_topic.get("vendor", "PREDICT_FUN")),
    )


def estimate_round_trip_cost_bps(
    fee_bps: int,
    spread_bps: Decimal | None,
    price_impact_bps: Decimal | None = None,
) -> Decimal:
    """Estimate round-trip cost in bps: 2*fee + spread + impact.

    All prediction trades are YES/NO shares in [0,1]; spread is quoted
    bid/ask for the *same* outcome token.
    """
    total = Decimal(fee_bps * 2)
    if spread_bps is not None:
        total += spread_bps
    if price_impact_bps is not None:
        total += price_impact_bps
    return total
