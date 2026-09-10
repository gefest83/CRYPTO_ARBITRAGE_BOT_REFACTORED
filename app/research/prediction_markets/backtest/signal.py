"""Deterministic spot impulse detector — strictly uses data at or before t0."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.research.prediction_markets.collector.models import SpotObservation

__all__ = ["detect_impulses", "mid_price"]


def mid_price(obs: SpotObservation) -> Decimal | None:
    if obs.bid is None or obs.ask is None:
        # fallback to last_price
        return obs.last_price
    return (obs.bid + obs.ask) / Decimal("2")


def _impulse_bps(before: Decimal, at: Decimal) -> Decimal:
    if before == Decimal("0"):
        return Decimal("0")
    return (at - before) / before * Decimal("10000")


def detect_impulses(
    spot_observations: list[SpotObservation],
    threshold_bps: int,
    lookback_ms: int = 1000,
) -> list[dict[str, Any]]:
    """Return list of raw impulses: {t0_ms, symbol, before_mid, at_mid, impulse_bps, direction}.

    Deterministic: for each spot observation at time t as t0, look back to the latest
    observation with timestamp <= t - lookback_ms (at or before). If gap > 2*lookback, skip.

    No future data beyond t0 is inspected for signal definition.
    """
    # sort by exchange_ts_ms deterministically
    spots = sorted(spot_observations, key=lambda o: (o.exchange_ts_ms or 0, o.captured_at_ms))
    results: list[dict[str, Any]] = []
    # group by symbol to avoid cross-symbol comparison
    by_symbol: dict[str, list[SpotObservation]] = {}
    for o in spots:
        by_symbol.setdefault(o.symbol, []).append(o)

    for symbol, lst in by_symbol.items():
        for idx, cur in enumerate(lst):
            cur_mid = mid_price(cur)
            if cur_mid is None or cur.exchange_ts_ms is None:
                continue
            t0 = int(cur.exchange_ts_ms)
            target = t0 - lookback_ms
            # find before observation: latest with ts <= target
            before: SpotObservation | None = None
            for j in range(idx - 1, -1, -1):
                cand = lst[j]
                if cand.exchange_ts_ms is None:
                    continue
                if int(cand.exchange_ts_ms) <= target:
                    # ensure not too stale: within 2*lookback window start
                    if t0 - int(cand.exchange_ts_ms) <= 2 * lookback_ms + 1:
                        before = cand
                    break
            if before is None:
                continue
            before_mid = mid_price(before)
            if before_mid is None:
                continue
            bps = _impulse_bps(before_mid, cur_mid)
            if abs(bps) >= Decimal(threshold_bps):
                direction = 1 if bps > 0 else -1
                results.append(
                    {
                        "t0_ms": t0,
                        "symbol": symbol,
                        "before_mid": before_mid,
                        "at_mid": cur_mid,
                        "impulse_bps": bps,
                        "direction": direction,
                        "before_obs": before,
                        "at_obs": cur,
                    }
                )
    # deterministic sort
    results.sort(key=lambda r: (r["t0_ms"], r["symbol"]))
    return results
