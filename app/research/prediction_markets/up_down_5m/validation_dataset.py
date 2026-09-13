"""Valid BTC Up/Down 5m research dataset — real windows, real outcomes.

Problem fixed: the 64% figure used prediction windows with zero spot
overlap plus a spot-drift proxy label. Every :class:`ValidatedMarket` here
contains, for one exact 5m window:

* exact start/end (Predict.fun ``categorySlug`` ``btc-updown-5m-<start>``,
  cross-checked against stored ``resolution_ms``);
* BTC price series covering the whole window (Binance public 1s klines,
  bounded backfill cached locally — the missing source in stored data);
* UP/DOWN contract prices during the window (Predict.fun ``timeseries``
  ``chance`` — venue-recorded, existing API);
* the actual resolved outcome (Predict.fun ``outcomes`` WON/LOST plus
  venue-recorded Chainlink start/end in ``variantData`` — no Chainlink API).

Known gap, stated not hidden: stored prediction books never fall inside
their own windows (verified: 0 in all 7 sources), and closed-market books
are gone, so ``book_imb`` is None (vote abstains) unless in-window depth
reappears. Entry price for replay scoring is the UP chance / 100 at the
decision timestamp (real contract price, not an assumption).

Research-only, read-only, bounded (no streaming, no long collection).
Signal logic is never modified here — only evaluated.
"""

from __future__ import annotations

import bisect
from decimal import Decimal
from typing import Any

import httpx
from pydantic import Field, field_validator

from app.models.base import DomainModel, as_decimal
from app.research.prediction_markets.up_down_5m.settlement import SettlementOutcome

__all__ = [
    "BINANCE_KLINES_URL",
    "Kline",
    "ValidatedMarket",
    "ValidationResult",
    "build_signal_features",
    "contract_price_at",
    "parse_venue_outcome",
    "parse_window_from_slug",
    "run_validation",
]

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
FIVE_MIN_MS = 300_000


class Kline(DomainModel):
    """One Binance 1s kline (public, no auth; second resolution is required
    because the 8bps signal deadbands are invisible at 1m granularity)."""

    open_ts_ms: int = Field(ge=0)
    close_ts_ms: int = Field(ge=0)
    close: Decimal
    volume_base: Decimal
    taker_buy_base: Decimal

    @field_validator("close", "volume_base", "taker_buy_base", mode="before")
    @classmethod
    def _dec(cls, v):  # type: ignore[no-untyped-def]
        return as_decimal(v, field="kline")


class ValidatedMarket(DomainModel):
    """One fully-covered 5m market window with real outcome."""

    market_id: int = Field(gt=0)
    start_ts_ms: int = Field(ge=0)
    end_ts_ms: int = Field(ge=0)
    klines: tuple[Kline, ...]
    contract_series: tuple[tuple[int, Decimal], ...] = ()  # (ts_ms, UP chance 0..1)
    outcome: SettlementOutcome
    venue_start_price: Decimal | None = None
    venue_end_price: Decimal | None = None
    book_imb: Decimal | None = None  # None: no in-window depth anywhere stored
    notes: str = ""


class ValidationResult(DomainModel):
    market_id: int
    signal: str
    votes: int
    outcome: SettlementOutcome
    correct: bool | None  # None for HOLD or PUSH (not scored)
    entry_price: Decimal | None = None


def parse_window_from_slug(slug: str) -> tuple[int, int]:
    """Exact window from ``btc-updown-5m-<start_sec>`` slug."""
    try:
        start_sec = int(str(slug).strip().rsplit("-", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"unparseable market slug: {slug!r}") from exc
    if start_sec <= 0:
        raise ValueError(f"unparseable market slug: {slug!r}")
    return start_sec * 1000, start_sec * 1000 + FIVE_MIN_MS


def parse_venue_outcome(payload: dict[str, Any]) -> tuple[SettlementOutcome, Decimal | None, Decimal | None]:
    """Actual resolved outcome from venue ``outcomes`` + ``variantData``.

    PUSH when neither side WON (covers the venue 50-50 rule) or when the
    venue-recorded Chainlink anchors are exactly equal.
    """
    data = payload.get("data", payload)
    vd = data.get("variantData") or {}
    vstart = as_decimal(vd["startPrice"], field="startPrice") if vd.get("startPrice") is not None else None
    vend = as_decimal(vd["endPrice"], field="endPrice") if vd.get("endPrice") is not None else None
    won = [str(o.get("name", "")).upper() for o in data.get("outcomes") or [] if str(o.get("status", "")).upper() == "WON"]
    if len(won) == 1 and won[0] in ("UP", "DOWN"):
        return SettlementOutcome(won[0]), vstart, vend
    if vstart is not None and vend is not None and vstart == vend:
        return SettlementOutcome.PUSH, vstart, vend
    if not won:
        return SettlementOutcome.PUSH, vstart, vend
    raise ValueError(f"ambiguous venue outcome: {[ (o.get('name'), o.get('status')) for o in data.get('outcomes') or [] ]}")


def parse_contract_series(payload: dict[str, Any]) -> tuple[tuple[int, Decimal], ...]:
    """UP chance series (0..1) from ``timeseries`` response."""
    data = payload.get("data", payload)
    pts = data.get("series") or []
    out: list[tuple[int, Decimal]] = []
    for p in pts:
        try:
            out.append((int(p["x"]) * 1000, as_decimal(p["y"], field="chance") / Decimal("100")))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(sorted(out))


def contract_price_at(series: tuple[tuple[int, Decimal], ...], ts_ms: int) -> Decimal | None:
    """Latest UP price at/before t (strictly pre-decision)."""
    best: Decimal | None = None
    for ts, px in series:
        if ts <= int(ts_ms):
            best = px
        else:
            break
    return best


def parse_klines(rows: list[list[Any]]) -> list[Kline]:
    out: list[Kline] = []
    for r in rows:
        out.append(Kline(
            open_ts_ms=int(r[0]), close_ts_ms=int(r[6]),
            close=as_decimal(r[4], field="close"),
            volume_base=as_decimal(r[5], field="volume"),
            taker_buy_base=as_decimal(r[9], field="takerBuy"),
        ))
    return sorted(out, key=lambda k: k.close_ts_ms)


async def fetch_klines(
    symbol: str, start_ms: int, end_ms: int,
    transport: httpx.AsyncBaseTransport | None = None,
    interval: str = "1s",
) -> list[Kline]:
    """Bounded public klines backfill (no auth, one call per window).

    1s is primary (8bps deadbands need second resolution); 1m is the
    documented fallback where 1s retention is exhausted. Same closeTime
    no-look-ahead rule either way.
    """
    params = {"symbol": symbol, "interval": interval,
              "startTime": int(start_ms), "endTime": int(end_ms), "limit": 1000}
    async with httpx.AsyncClient(timeout=20.0, transport=transport) as client:
        r = await client.get(BINANCE_KLINES_URL, params=params)
        r.raise_for_status()
        return parse_klines(r.json())


def _close_at_or_before(klines: list[Kline], ts_ms: int) -> Kline | None:
    """Latest kline fully closed at/before t (closeTime rule: no look-ahead)."""
    idx = bisect.bisect_right([k.close_ts_ms for k in klines], int(ts_ms)) - 1
    return klines[idx] if idx >= 0 else None


def build_signal_features(
    market: ValidatedMarket,
    decision_ts_ms: int,
    momentum_window_ms: int = 30_000,
    accel_window_ms: int = 60_000,
    flow_window_ms: int = 60_000,
):  # type: ignore[no-untyped-def]
    """Map kline series onto :class:`SignalFeatures` (pre-decision only)."""
    from app.research.prediction_markets.up_down_5m.signal import compute_features

    ref = _close_at_or_before(market.klines, market.start_ts_ms)
    now = _close_at_or_before(market.klines, decision_ts_ms)
    m1 = _close_at_or_before(market.klines, decision_ts_ms - momentum_window_ms)
    m2 = _close_at_or_before(market.klines, decision_ts_ms - accel_window_ms)
    flow = None
    lo, hi = int(decision_ts_ms) - flow_window_ms, int(decision_ts_ms)
    buy = vol = Decimal("0")
    for k in market.klines:
        if lo < k.close_ts_ms <= hi:
            buy += k.taker_buy_base
            vol += k.volume_base
    if vol > 0:
        flow = (buy - (vol - buy)) / vol
    return compute_features(
        decision_ts_ms=int(decision_ts_ms),
        market_start_ms=market.start_ts_ms,
        market_end_ms=market.end_ts_ms,
        ref_price=ref.close if ref else None,
        mid_now=now.close if now else None,
        mid_momentum_ago=m1.close if m1 else None,
        mid_accel_ago=m2.close if m2 else None,
        flow_imb=flow,
        book_imb=market.book_imb,
    )


def run_validation(
    markets: list[ValidatedMarket],
    config=None,  # SignalConfig; None => researched defaults
    decision_offset_ms: int = 90_000,
) -> tuple[list[ValidationResult], dict[str, Any]]:
    """Run the UNCHANGED signal against real venue outcomes."""
    from app.research.prediction_markets.up_down_5m.signal import (
        Signal,
        SignalConfig,
        evaluate,
    )

    cfg = config or SignalConfig()
    results: list[ValidationResult] = []
    for m in markets:
        t = m.start_ts_ms + decision_offset_ms
        sig = evaluate(build_signal_features(m, t), cfg)
        px = contract_price_at(m.contract_series, t)
        if sig.signal == Signal.HOLD or m.outcome == SettlementOutcome.PUSH:
            correct = None
        else:
            correct = sig.signal.value == m.outcome.value
        results.append(ValidationResult(
            market_id=m.market_id, signal=sig.signal.value, votes=sig.votes,
            outcome=m.outcome, correct=correct, entry_price=px,
        ))
    scored = [r for r in results if r.correct is not None]
    hits = sum(1 for r in scored if r.correct)
    summary = {
        "markets": len(markets),
        "scored": len(scored),
        "hits": hits,
        "accuracy": hits / len(scored) if scored else None,
        "coverage": len(scored) / len(markets) if markets else 0.0,
        "label": "actual venue-resolved UP/DOWN (Predict.fun WON/LOST + recorded Chainlink anchors)",
    }
    return results, summary


def ensure_research_only() -> None:
    raise RuntimeError("validation dataset is research-only; live trading is disabled (fail-closed)")
