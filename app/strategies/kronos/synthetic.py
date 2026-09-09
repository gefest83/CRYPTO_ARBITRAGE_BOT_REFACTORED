"""Synthetic 1m OHLCV generator for offline tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Sequence

from app.strategies.kronos.types import Candle

__all__ = ["generate_synthetic_candles"]


def generate_synthetic_candles(
    *,
    start: datetime,
    count: int,
    start_price: Decimal = Decimal("100"),
    drift_bps_per_bar: Decimal = Decimal("0"),
    volatility_bps: Decimal = Decimal("0"),
    volume: Decimal = Decimal("10"),
) -> tuple[Candle, ...]:
    """Generate ``count`` 1m candles deterministically.

    - start_price: close of first bar
    - drift_bps_per_bar: deterministic drift in bps per bar (e.g. 1 = 0.01%)
    - volatility_bps: not used for deterministic tests (kept for future)
    Timestamps are 1m apart, starting at ``start`` (must be UTC).
    All bars are marked complete=True.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if count <= 0:
        return ()
    out: list[Candle] = []
    price = start_price
    for i in range(count):
        ts = start + timedelta(minutes=i)
        # apply drift cumulatively: price * (1 + drift/10000) each bar
        if i > 0 and drift_bps_per_bar != Decimal("0"):
            price = (price * (Decimal("1") + drift_bps_per_bar / Decimal("10000"))).quantize(
                Decimal("0.00000001")
            )
        # for simplicity OHLC all equal to close (spread 0 synthetic)
        out.append(
            Candle(
                timestamp=ts,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                complete=True,
            )
        )
    return tuple(out)
