"""Historical 1m OHLCV loader for Phase 3B backtest (Binance, research-only).

- Uses Binance public /api/v3/klines (no API keys, no trading).
- Strict no-lookahead: caller receives candles up to end time inclusive; no future leaked.
- Disk cache under data/kronos/ to avoid repeated network and to make backtests deterministic offline.
- Converts to app.strategies.kronos.types.Candle (Decimal, UTC, complete=True).

Isolated: does NOT touch execution, RiskEngine, Telegram, storage, AutoTrader.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Sequence

import httpx

from app.strategies.kronos.types import Candle

__all__ = [
    "BinanceKline",
    "fetch_binance_klines",
    "load_or_fetch_candles",
    "candles_from_csv",
    "save_candles_csv",
    "ensure_cache_dir",
]

BINANCE_BASE = "https://api.binance.com"
# Binance limits: 1000 per request
LIMIT = 1000
UTC = timezone.utc


def ensure_cache_dir(base: Path | None = None) -> Path:
    p = (base or Path("data/kronos"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _binance_symbol(symbol: str) -> str:
    """BTC/USDT -> BTCUSDT"""
    return symbol.strip().upper().replace("/", "")


def _ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _dt_from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


# Raw Binance kline: [openTime, open, high, low, close, volume, closeTime, quoteVol, trades, takerBuyBase, takerBuyQuote, ignore]
BinanceKline = tuple  # generic


def fetch_binance_klines(
    symbol: str,
    interval: str = "1m",
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = LIMIT,
    max_retries: int = 3,
    timeout: float = 10.0,
) -> list[list]:
    """Fetch klines from Binance public API with pagination handling for single request.

    This function fetches at most `limit` klines (single HTTP request). Caller handles pagination.
    """
    params: dict[str, str | int] = {
        "symbol": _binance_symbol(symbol),
        "interval": interval,
        "limit": limit,
    }
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    url = f"{BINANCE_BASE}/api/v3/klines"
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list):
                    raise RuntimeError(f"unexpected kline response {data}")
                return data
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"fetch failed after {max_retries}: {last_exc}")


def fetch_range(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> list[list]:
    """Fetch all klines in [start, end) via paginated requests."""
    out: list[list] = []
    cur = start
    # Binance interval to timedelta
    delta_map = {"1m": timedelta(minutes=1), "5m": timedelta(minutes=5), "1h": timedelta(hours=1)}
    step = delta_map.get(interval, timedelta(minutes=1))
    # We paginate by startTime cursor
    while cur < end:
        # Use batch limit
        remaining_ms = _ms(end) - _ms(cur)
        # Estimate how many bars left: not needed, just use limit
        batch = fetch_binance_klines(symbol, interval, _ms(cur), _ms(end), limit=LIMIT)
        if not batch:
            break
        out.extend(batch)
        # Next start is closeTime+1 of last bar
        last_close = batch[-1][6]  # closeTime
        cur = _dt_from_ms(int(last_close) + 1)
        if len(batch) < LIMIT:
            break
        # Be nice to Binance
        time.sleep(0.2)
        # Safety: avoid infinite loop if cur not advancing
        if len(out) > 200_000:
            break
    return out


def klines_to_candles(klines: list[list]) -> tuple[Candle, ...]:
    out: list[Candle] = []
    for k in klines:
        # k: [0 openTime,1 open,2 high,3 low,4 close,5 volume,6 closeTime,...]
        ts = _dt_from_ms(int(k[0]))
        # Use Decimal for prices
        o = Decimal(str(k[1]))
        h = Decimal(str(k[2]))
        lo = Decimal(str(k[3]))
        c = Decimal(str(k[4]))
        v = Decimal(str(k[5]))
        # Validate monotonic and non-positive guards later; keep as is
        out.append(Candle(timestamp=ts, open=o, high=h, low=lo, close=c, volume=v, complete=True))
    # Ensure sorted by timestamp
    out.sort(key=lambda x: x.timestamp)
    return tuple(out)


def save_candles_csv(path: Path, candles: Sequence[Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([c.timestamp.isoformat(), str(c.open), str(c.high), str(c.low), str(c.close), str(c.volume)])


def candles_from_csv(path: Path) -> tuple[Candle, ...]:
    out: list[Candle] = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ts = datetime.fromisoformat(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            out.append(
                Candle(
                    timestamp=ts,
                    open=Decimal(row["open"]),
                    high=Decimal(row["high"]),
                    low=Decimal(row["low"]),
                    close=Decimal(row["close"]),
                    volume=Decimal(row["volume"]),
                    complete=True,
                )
            )
    out.sort(key=lambda x: x.timestamp)
    return tuple(out)


def _cache_path(symbol: str, interval: str, start: datetime, end: datetime, cache_dir: Path) -> Path:
    safe_sym = _binance_symbol(symbol)
    s = start.strftime("%Y%m%d_%H%M")
    e = end.strftime("%Y%m%d_%H%M")
    return cache_dir / f"binance_{safe_sym}_{interval}_{s}_{e}.csv"


def load_or_fetch_candles(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = "1m",
    cache_dir: Path | None = None,
    use_cache: bool = True,
    save_cache: bool = True,
) -> tuple[Candle, ...]:
    """Load from cache if present, else fetch from Binance and optionally cache.

    Strict no-lookahead: returns only candles with timestamp < end and >= start.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    if start >= end:
        return ()

    cache_dir = ensure_cache_dir(cache_dir)
    path = _cache_path(symbol, interval, start, end, cache_dir)

    if use_cache and path.exists():
        try:
            candles = candles_from_csv(path)
            # Filter to exact range (cache file may have been for same range)
            filtered = tuple(c for c in candles if start <= c.timestamp < end)
            if filtered:
                return filtered
        except Exception:
            pass  # fall through to fetch

    # Fetch live
    klines = fetch_range(symbol, interval, start, end)
    candles = klines_to_candles(klines)
    # Filter.
    candles = tuple(c for c in candles if start <= c.timestamp < end)

    if save_cache and candles:
        try:
            save_candles_csv(path, candles)
        except Exception:
            pass
    return candles


def generate_synthetic_fallback(
    symbol: str,
    start: datetime,
    end: datetime,
) -> tuple[Candle, ...]:
    """Fallback deterministic synthetic candles when network unavailable (for tests)."""
    from app.strategies.kronos.synthetic import generate_synthetic_candles

    minutes = int((end - start).total_seconds() // 60)
    # Use symbol hash to vary price deterministically
    h = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
    price = Decimal(str(100 + (h % 50000)))
    return generate_synthetic_candles(start=start, count=minutes, start_price=price)
