"""Shared types for Kronos offline slice."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from app.models.enums import ArbitrageStrategy

__all__ = ["Candle", "ForecastBar", "KronosSignal", "Signal"]


@dataclass(frozen=True, slots=True)
class Candle:
    """One 1m OHLCV bar.

    All prices as Decimal, timestamp is UTC, interval is always 1m in phase 2.
    ``complete`` = True means the bar is closed/final; False = still forming
    (must be rejected for signal generation).
    """

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    complete: bool = True

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            object.__setattr__(self, "timestamp", self.timestamp.replace(tzinfo=timezone.utc))


@dataclass(frozen=True, slots=True)
class ForecastBar:
    """One predicted bar (OHLCV). Only close is used for threshold today."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            object.__setattr__(self, "timestamp", self.timestamp.replace(tzinfo=timezone.utc))


# Signal is intentionally StrEnum-ish but plain string literal for offline isolation.
# Use Signal.BUY etc. Mirrors the future execution side.
from enum import StrEnum


class Signal(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class KronosSignal:
    """Deterministic output of the scanner. No trade persisted."""

    symbol: str
    signal: Signal
    reason: str
    strategy: ArbitrageStrategy = ArbitrageStrategy.KRONOS
    last_close: Decimal = Decimal("0")
    pred_close_1m: Decimal | None = None
    pred_close_5m: Decimal | None = None
    gross_bps_1m: Decimal = Decimal("0")
    gross_bps_5m: Decimal | None = None
    cost_bps: Decimal = Decimal("0")
    net_bps_1m: Decimal = Decimal("0")
    net_bps_5m: Decimal | None = None
    # input age in ms (for audit)
    data_age_ms: float = 0.0
    # whether 5m confirmation was applied and passed
    confirmation_applied: bool = False
    confirmation_passed: bool | None = None
