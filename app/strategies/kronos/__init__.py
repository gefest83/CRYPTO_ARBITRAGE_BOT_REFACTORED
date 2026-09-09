"""Kronos scalping strategy — isolated vertical slice (offline, no I/O).

Phase 2 provides:
* config
* forecast model interface + mock predictor
* cost/edge calculation
* deterministic signal generation (BUY/SELL/HOLD)
* scanner (1m -> 5m confirmation, no lookahead, stale rejection)

No executor, no live market data, no DB changes, no torch.
"""

from app.strategies.kronos.config import KronosConfig
from app.strategies.kronos.cost import CostInputs, compute_cost_bps, compute_gross_bps, compute_net_bps
from app.strategies.kronos.forecast import KronosForecastModel, MockKronosPredictor
from app.strategies.kronos.scanner import KronosScanner
from app.strategies.kronos.signal import classify_signal
from app.strategies.kronos.synthetic import generate_synthetic_candles
from app.strategies.kronos.types import Candle, ForecastBar, KronosSignal, Signal

__all__ = [
    "Candle",
    "CostInputs",
    "ForecastBar",
    "KronosConfig",
    "KronosForecastModel",
    "KronosScanner",
    "KronosSignal",
    "MockKronosPredictor",
    "Signal",
    "classify_signal",
    "compute_cost_bps",
    "compute_gross_bps",
    "compute_net_bps",
    "generate_synthetic_candles",
]
