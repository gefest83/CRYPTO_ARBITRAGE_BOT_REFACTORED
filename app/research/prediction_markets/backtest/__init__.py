"""Phase 3 — Offline repricing research/backtest.

BTC/ETH 5m/15m only. BNB excluded.
Uses Phase 2 CollectorStore observations.
No trading, no wallet, no AI, no signals execution — pure offline analysis.

Invariant: signals use only observations at or before t0; future only for evaluation.
"""

from app.research.prediction_markets.backtest.analyzer import BacktestAnalyzer  # noqa: F401
from app.research.prediction_markets.backtest.cost_model import CostModel  # noqa: F401
from app.research.prediction_markets.backtest.models import (  # noqa: F401
    BacktestConfig,
    BacktestResult,
    GroupKey,
    GroupStats,
    ImpulseEvent,
    OutcomeMetrics,
)
from app.research.prediction_markets.backtest.signal import detect_impulses  # noqa: F401

__all__ = [
    "BacktestAnalyzer",
    "BacktestConfig",
    "BacktestResult",
    "CostModel",
    "GroupKey",
    "GroupStats",
    "ImpulseEvent",
    "OutcomeMetrics",
    "detect_impulses",
]
