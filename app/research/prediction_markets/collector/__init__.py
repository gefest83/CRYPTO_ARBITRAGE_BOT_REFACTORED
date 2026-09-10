"""Isolated historical data collector — Phase 2.

BTC/ETH 5m/15m only. No BNB. No trading, no wallet, no AI, no signals.

Architecture:
  models   — immutable observations (spot / prediction orderbook / trade / sync)
  storage  — JSONL + SQLite research-friendly persistence
  collector— orchestrator: WS pref + REST recovery, dedup, ordering, gaps, stale, expiry

All timestamps are integer ms since epoch (UTC), deterministic.
"""

from app.research.prediction_markets.collector.collector import HistoricalCollector  # noqa: F401
from app.research.prediction_markets.collector.models import (  # noqa: F401
    CollectorStats,
    PredictionOrderbookObservation,
    PredictionTradeObservation,
    RawObservation,
    SpotObservation,
    SynchronizedObservation,
)
from app.research.prediction_markets.collector.storage import CollectorStore  # noqa: F401

__all__ = [
    "CollectorStats",
    "CollectorStore",
    "HistoricalCollector",
    "PredictionOrderbookObservation",
    "PredictionTradeObservation",
    "RawObservation",
    "SpotObservation",
    "SynchronizedObservation",
]
