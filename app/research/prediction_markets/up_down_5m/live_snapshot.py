"""Wire BTC Up/Down 5m foundation to real Binance Prediction Market data.

Research-only, read-only. Reuses existing SAPI/WS infrastructure
(:mod:`app.research.prediction_markets.client`,
:mod:`app.research.prediction_markets.discovery`,
:mod:`app.research.prediction_markets.normalizer`). No new collector,
no long-running dataset, no trading.

What the venue actually exposes (verified against client + normalizer):
  obtainable via SAPI : market_id, start_ts/end_ts (``startDate``/``endDate``),
    UP bid/ask (YES-outcome ``order-book``), DOWN bid/ask (NO-outcome
    ``order-book``), trading status, last trade price.
  NOT exposed by SAPI: Chainlink BTC/USDT Top-of-Book mid (start, current,
    final). SAPI carries contract prices in [0,1], never the underlying
    Chainlink feed. Callers may inject a real Chainlink mid (provenance
    ``CHAINLINK_VENUE``) or a Binance spot ``bookTicker`` ToB mid as an
    explicitly labeled reference (provenance ``SPOT_REFERENCE`` — usable
    for display/timing, NEVER as a settlement input).

Outcome mapping assumption (documented): the inner market titled ``UP``
carries YES/NO outcomes where YES == UP wins and NO == DOWN wins
(matches existing fixtures and ``build_research_universe`` usage).
"""

from __future__ import annotations

import time
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import Field

from app.models.base import DomainModel, as_decimal
from app.research.prediction_markets.up_down_5m.market import (
    MarketState,
    UpDown5mMarket,
)
from app.research.prediction_markets.up_down_5m.settlement import SettlementOutcome

__all__ = [
    "FieldCoverage",
    "LiveUpDown5mSnapshot",
    "NoLiveMarketError",
    "PriceProvenance",
    "build_live_snapshot",
    "fetch_one_btc_5m_snapshot",
    "render_snapshot",
    "required_field_coverage",
    "settlement_preview",
]


class NoLiveMarketError(RuntimeError):
    """Raised when no live BTC Up/Down 5m market is obtainable (fail-closed)."""


class PriceProvenance(StrEnum):
    CHAINLINK_VENUE = "chainlink_venue"  # real Chainlink ToB mid supplied by caller
    SPOT_REFERENCE = "spot_reference"  # Binance spot bookTicker ToB mid; display only, NOT settlement
    UNAVAILABLE = "unavailable"  # venue does not expose; value is None


class FieldCoverage(DomainModel):
    """Presence + source of one required field."""

    name: str
    present: bool
    detail: str  # value summary or provenance/reason


class LiveUpDown5mSnapshot(DomainModel):
    """One live BTC Up/Down 5m market wired to the foundation model."""

    market: UpDown5mMarket
    checked_at_ms: int = Field(ge=0)
    trading_status: str | None = None
    start_provenance: PriceProvenance = PriceProvenance.UNAVAILABLE
    current_provenance: PriceProvenance = PriceProvenance.UNAVAILABLE
    end_provenance: PriceProvenance = PriceProvenance.UNAVAILABLE
    up_token_id: str | None = None
    down_token_id: str | None = None
    notes: str = ""


def _resolve_price(
    chainlink: Decimal | str | int | float | None,
    spot_ref: Decimal | str | int | float | None,
) -> tuple[Decimal | None, PriceProvenance]:
    if chainlink is not None:
        return as_decimal(chainlink, field="chainlink"), PriceProvenance.CHAINLINK_VENUE
    if spot_ref is not None:
        return as_decimal(spot_ref, field="spot_reference"), PriceProvenance.SPOT_REFERENCE
    return None, PriceProvenance.UNAVAILABLE


def build_live_snapshot(
    normalized: Any,
    *,
    chainlink_start: Decimal | str | int | float | None = None,
    chainlink_current: Decimal | str | int | float | None = None,
    chainlink_end: Decimal | str | int | float | None = None,
    spot_mid_at_start: Decimal | str | int | float | None = None,
    spot_mid_now: Decimal | str | int | float | None = None,
    checked_at_ms: int | None = None,
) -> LiveUpDown5mSnapshot:
    """Map one normalized BTC 5m market + books onto :class:`UpDown5mMarket`.

    Raises:
        ValueError: if the market is not a BTC 5m market with start/end,
            has no market_id, or lacks YES/NO (UP/DOWN) outcomes.
    """
    symbol = str(getattr(normalized, "symbol", "") or "").upper()
    if symbol.replace("USDT", "") != "BTC":
        raise ValueError(f"live wiring is BTC-only, got symbol={symbol!r}")
    duration = str(getattr(normalized, "duration", "") or "").lower()
    if duration != "5m":
        raise ValueError(f"live wiring is 5m-only, got duration={duration!r}")
    start_ms = getattr(normalized, "start_ms", None)
    end_ms = getattr(normalized, "end_ms", None)
    if start_ms is None or end_ms is None:
        raise ValueError("live wiring requires start_ms/end_ms (SAPI startDate/endDate)")
    identifiers = getattr(normalized, "identifiers", None)
    market_id = int(getattr(identifiers, "market_id", 0) or 0)
    if market_id <= 0:
        raise ValueError("live wiring requires a positive market_id")

    outcomes = list(getattr(normalized, "outcomes", None) or ())
    yes = next((o for o in outcomes if str(getattr(o, "name", "")).upper() == "YES"), None)
    no = next((o for o in outcomes if str(getattr(o, "name", "")).upper() == "NO"), None)
    if yes is None or no is None:
        raise ValueError("live wiring requires both YES (UP) and NO (DOWN) outcomes")

    start_price, start_prov = _resolve_price(chainlink_start, spot_mid_at_start)
    current_price, current_prov = _resolve_price(chainlink_current, spot_mid_now)
    end_price, end_prov = _resolve_price(chainlink_end, None)

    market = UpDown5mMarket(
        market_id=market_id,
        start_ts_ms=int(start_ms),
        end_ts_ms=int(end_ms),
        start_price=start_price,
        current_price=current_price,
        end_price=end_price,
        up_bid=getattr(yes, "best_bid", None),
        up_ask=getattr(yes, "best_ask", None),
        down_bid=getattr(no, "best_bid", None),
        down_ask=getattr(no, "best_ask", None),
    )
    notes = (
        "SAPI exposes contract books (YES=UP, NO=DOWN), not the Chainlink feed. "
        "SPOT_REFERENCE mids are display/timing only and must never settle. "
        "Settlement requires CHAINLINK_VENUE start+end."
    )
    return LiveUpDown5mSnapshot(
        market=market,
        checked_at_ms=int(checked_at_ms) if checked_at_ms is not None else int(time.time() * 1000),
        trading_status=getattr(normalized, "trading_status", None),
        start_provenance=start_prov,
        current_provenance=current_prov,
        end_provenance=end_prov,
        up_token_id=getattr(yes, "token_id", None),
        down_token_id=getattr(no, "token_id", None),
        notes=notes,
    )


def required_field_coverage(snapshot: LiveUpDown5mSnapshot) -> tuple[FieldCoverage, ...]:
    """Coverage of the 7 required recordable fields."""
    m = snapshot.market
    return (
        FieldCoverage(name="market_id", present=True, detail=f"market_id={m.market_id} (SAPI marketId)"),
        FieldCoverage(name="start_ts", present=True, detail=f"start_ts_ms={m.start_ts_ms} (SAPI startDate)"),
        FieldCoverage(name="end_ts", present=True, detail=f"end_ts_ms={m.end_ts_ms} (SAPI endDate)"),
        FieldCoverage(
            name="start_chainlink_tob_mid",
            present=m.start_price is not None,
            detail=f"{m.start_price} [{snapshot.start_provenance.value}]" if m.start_price is not None else f"missing [{snapshot.start_provenance.value}: SAPI does not expose Chainlink]",
        ),
        FieldCoverage(
            name="current_chainlink_tob_mid",
            present=m.current_price is not None,
            detail=f"{m.current_price} [{snapshot.current_provenance.value}]" if m.current_price is not None else f"missing [{snapshot.current_provenance.value}: SAPI does not expose Chainlink]",
        ),
        FieldCoverage(
            name="up_bid_ask",
            present=m.up_bid is not None and m.up_ask is not None,
            detail=f"bid={m.up_bid} ask={m.up_ask} (YES book {snapshot.up_token_id})" if m.up_bid is not None and m.up_ask is not None else "missing (YES order-book unavailable)",
        ),
        FieldCoverage(
            name="down_bid_ask",
            present=m.down_bid is not None and m.down_ask is not None,
            detail=f"bid={m.down_bid} ask={m.down_ask} (NO book {snapshot.down_token_id})" if m.down_bid is not None and m.down_ask is not None else "missing (NO order-book unavailable)",
        ),
        FieldCoverage(
            name="final_settlement_price_outcome",
            present=m.start_price is not None and m.end_price is not None and snapshot.end_provenance == PriceProvenance.CHAINLINK_VENUE,
            detail=(
                f"end={m.end_price} outcome={snapshot.market.settle().settlement.value} [chainlink_venue]"
                if m.start_price is not None and m.end_price is not None and snapshot.end_provenance == PriceProvenance.CHAINLINK_VENUE
                else f"withheld [{snapshot.end_provenance.value}: final Chainlink close not exposed by SAPI]"
            ),
        ),
    )


def settlement_preview(snapshot: LiveUpDown5mSnapshot) -> tuple[SettlementOutcome | None, str]:
    """Outcome if Chainlink anchors are venue-sourced, else (None, reason)."""
    m = snapshot.market
    if m.start_price is None or m.end_price is None:
        return None, "settlement withheld: Chainlink start/end anchors unavailable (SAPI does not expose Chainlink feed)"
    if snapshot.end_provenance != PriceProvenance.CHAINLINK_VENUE:
        return None, f"settlement withheld: end anchor is {snapshot.end_provenance.value}, not venue Chainlink (spot reference must never settle)"
    try:
        settled = m.settle()
    except ValueError as exc:
        return None, f"settlement withheld: {exc}"
    return settled.settlement, f"settled from Chainlink start={m.start_price} end={m.end_price}"


def render_snapshot(snapshot: LiveUpDown5mSnapshot, now_ms: int | None = None) -> str:
    """Render one complete market lifecycle/state snapshot (text, no secrets)."""
    now = int(now_ms) if now_ms is not None else snapshot.checked_at_ms
    m = snapshot.market
    state: MarketState = m.state_at(now)
    outcome, reason = settlement_preview(snapshot)
    lines = [
        "BTC Up/Down 5m — live lifecycle snapshot (research-only, no trading)",
        f"market_id={m.market_id} start_ts_ms={m.start_ts_ms} end_ts_ms={m.end_ts_ms} duration_ms={m.duration_ms}",
        f"trading_status={snapshot.trading_status} state@{now}={state.value} checked_at_ms={snapshot.checked_at_ms}",
        f"start_chainlink_tob_mid={m.start_price} [{snapshot.start_provenance.value}]",
        f"current_chainlink_tob_mid={m.current_price} [{snapshot.current_provenance.value}]",
        f"up_bid={m.up_bid} up_ask={m.up_ask} up_mid={m.up_mid} (YES {snapshot.up_token_id})",
        f"down_bid={m.down_bid} down_ask={m.down_ask} down_mid={m.down_mid} (NO {snapshot.down_token_id})",
        f"final_settlement_price={m.end_price} [{snapshot.end_provenance.value}] outcome={outcome.value if outcome else None} ({reason})",
        "coverage:",
    ]
    for cov in required_field_coverage(snapshot):
        mark = "OK" if cov.present else "MISSING"
        lines.append(f"  [{mark}] {cov.name}: {cov.detail}")
    lines.append(f"notes: {snapshot.notes}")
    return "\n".join(lines)


async def fetch_one_btc_5m_snapshot(
    client: Any,
    *,
    chainlink_start: Decimal | str | int | float | None = None,
    chainlink_current: Decimal | str | int | float | None = None,
    chainlink_end: Decimal | str | int | float | None = None,
    spot_mid_at_start: Decimal | str | int | float | None = None,
    spot_mid_now: Decimal | str | int | float | None = None,
    checked_at_ms: int | None = None,
    prefer_open: bool = True,
) -> LiveUpDown5mSnapshot:
    """Fetch ONE live BTC 5m market via existing SAPI client + normalizer.

    Reuses :func:`discover_markets` and per-token ``get_order_book`` (same
    pattern as ``build_research_universe``). Read-only; never quotes/trades.

    Raises:
        NoLiveMarketError: if no BTC 5m market (or no OPEN one when
            ``prefer_open``) is obtainable.
    """
    from app.research.prediction_markets.discovery import discover_markets
    from app.research.prediction_markets.normalizer import normalize_market_topic

    topics = await discover_markets(client)
    # Structural pre-filter to BTC 5m windows before spending detail calls.
    candidates: list[dict[str, Any]] = []
    for t in topics:
        sym = str(t.get("symbol") or "").upper()
        if sym.replace("USDT", "") != "BTC":
            continue
        start, end = t.get("startDate"), t.get("endDate")
        if start is None or end is None or abs(int(end) - int(start) - 300_000) > 5_000:
            continue
        candidates.append(t)
    if not candidates:
        raise NoLiveMarketError("no BTC 5m market topics discovered via SAPI market/list+search")

    enriched: list[dict[str, Any]] = []
    for t in candidates:
        try:
            detail = await client.get_market_detail(t["marketTopicId"])
            enriched.append(detail if isinstance(detail, dict) and "marketTopicId" in detail else t)
        except Exception:
            enriched.append(t)

    orderbooks: dict[str, dict[str, Any]] = {}
    for topic in enriched:
        for inner in topic.get("markets") or []:
            for o in inner.get("outcomes") or []:
                tok = str(o.get("tokenId") or "")
                if not tok or tok in orderbooks:
                    continue
                try:
                    orderbooks[tok] = await client.get_order_book(
                        topic.get("vendor", "PREDICT_FUN"), inner.get("marketId"), tok
                    )
                except Exception:
                    continue

    normalizedbest: Any | None = None
    for topic in enriched:
        nm = normalize_market_topic(topic, orderbooks=orderbooks)
        if nm is None or nm.symbol.replace("USDT", "") != "BTC" or str(nm.duration).lower() != "5m":
            continue
        if prefer_open and not nm.is_active:
            if normalizedbest is None:
                normalizedbest = nm  # keep as fallback if nothing OPEN
            continue
        normalizedbest = nm
        break
    if normalizedbest is None:
        raise NoLiveMarketError("no BTC 5m market normalized from live SAPI topics (all out of scope or bookless)")

    return build_live_snapshot(
        normalizedbest,
        chainlink_start=chainlink_start,
        chainlink_current=chainlink_current,
        chainlink_end=chainlink_end,
        spot_mid_at_start=spot_mid_at_start,
        spot_mid_now=spot_mid_now,
        checked_at_ms=checked_at_ms,
    )


def ensure_research_only() -> None:
    """Fail-closed guard — always raises; live wiring must never trade."""
    raise RuntimeError("up_down_5m live wiring is research-only; live trading is disabled (fail-closed)")
