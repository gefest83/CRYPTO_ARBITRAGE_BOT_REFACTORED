"""Binance Prediction Markets — Phase 1 Research (isolated).

Scope: BTC and ETH only, 5-minute and 15-minute prediction markets only.
BNB is explicitly out of scope.

This package is completely isolated from:
  triangle, transfer, execution, RiskEngine, ExecutionGuard,
  recovery, Telegram, AI Advisor, Kronos, existing exchange behavior.

It never places orders, moves funds, or touches LIVE/DEMO routing.
All network access is read-only and mockable.
"""

from app.research.prediction_markets.models import (  # noqa: F401
    HistoricalAvailability,
    MarketDuration,
    MarketIdentifiers,
    NormalizedMarket,
    OutcomeSnapshot,
    PredictionConstraints,
    PredictionMarketType,
    QuoteInspection,
    ResearchUniverse,
    TimestampQuality,
    UpdateFrequency,
)

__all__ = [
    "HistoricalAvailability",
    "MarketDuration",
    "MarketIdentifiers",
    "NormalizedMarket",
    "OutcomeSnapshot",
    "PredictionConstraints",
    "PredictionMarketType",
    "QuoteInspection",
    "ResearchUniverse",
    "TimestampQuality",
    "UpdateFrequency",
]
