"""Normalization: raw Binance SAPI payloads -> immutable internal models.

Pure functions, no I/O, no BNB, deterministic.

Scope guard: only BTC/ETH, only 5m/15m.
"""

from __future__ import annotations

from decimal import Decimal

from app.models.base import as_decimal

from app.research.prediction_markets.models import (
    MarketDuration,
    MarketIdentifiers,
    NormalizedMarket,
    OutcomeSnapshot,
    PredictionMarketType,
    _infer_duration,
)

__all__ = ["normalize_market_topic", "normalize_outcome"]


def _symbol_to_market_type(symbol: str, duration: MarketDuration) -> PredictionMarketType | None:
    s = symbol.upper().replace("USDT", "")
    if s == "BTC" and duration == MarketDuration.M5:
        return PredictionMarketType.BTC_5M
    if s == "BTC" and duration == MarketDuration.M15:
        return PredictionMarketType.BTC_15M
    if s == "ETH" and duration == MarketDuration.M5:
        return PredictionMarketType.ETH_5M
    if s == "ETH" and duration == MarketDuration.M15:
        return PredictionMarketType.ETH_15M
    return None


def normalize_outcome(raw: dict, orderbook: dict | None = None, last_price: str | Decimal | None = None) -> OutcomeSnapshot:
    price = as_decimal(raw.get("price", "0"), field="price")
    chance = None
    if raw.get("chance") is not None:
        try:
            chance = as_decimal(raw["chance"], field="chance")
        except Exception:
            chance = None
    bids = ()
    asks = ()
    best_bid = None
    best_ask = None
    bid_size = None
    ask_size = None
    ts = None
    if orderbook:
        ts = orderbook.get("timestamp")
        for b in orderbook.get("bids", []):
            # orderbook bids: {"price": "...", "size": "..."} or [price,size]
            if isinstance(b, dict):
                bids = bids + ((as_decimal(b["price"], field="bid_price"), as_decimal(b["size"], field="bid_size")),)
            else:
                bids = bids + ((as_decimal(b[0], field="bid_price"), as_decimal(b[1], field="bid_size")),)
        for a in orderbook.get("asks", []):
            if isinstance(a, dict):
                asks = asks + ((as_decimal(a["price"], field="ask_price"), as_decimal(a["size"], field="ask_size")),)
            else:
                asks = asks + ((as_decimal(a[0], field="ask_price"), as_decimal(a[1], field="ask_size")),)
        if bids:
            # bids sorted descending; best is max price
            best_bid = max(b[0] for b in bids)
            # size at best
            for p, s in bids:
                if p == best_bid:
                    bid_size = s
                    break
        if asks:
            best_ask = min(a[0] for a in asks)
            for p, s in asks:
                if p == best_ask:
                    ask_size = s
                    break
    lp = None
    if last_price is not None:
        try:
            lp = as_decimal(last_price, field="lastTradePrice")
        except Exception:
            lp = None
    return OutcomeSnapshot(
        name=str(raw.get("name", "YES")),
        token_id=str(raw.get("tokenId", "")),
        price=price,
        chance=chance,
        index=raw.get("index"),
        best_bid=best_bid,
        best_ask=best_ask,
        bid_size=bid_size,
        ask_size=ask_size,
        bids=bids,
        asks=asks,
        last_trade_price=lp,
        timestamp_ms=ts,
    )


def normalize_market_topic(
    raw_topic: dict,
    orderbooks: dict[str, dict] | None = None,
    last_prices: dict[int, str | Decimal] | None = None,
) -> NormalizedMarket | None:
    """Normalize one marketTopic. Returns None if out of scope (non-BTC/ETH or non-5m/15m)."""
    symbol = str(raw_topic.get("symbol") or raw_topic.get("variantData", {}).get("priceFeedSymbol") or "").upper()
    # fall back to slug/title heuristics
    if not symbol:
        title = str(raw_topic.get("title") or raw_topic.get("slug") or "")
        if "BTC" in title.upper():
            symbol = "BTCUSDT"
        elif "ETH" in title.upper():
            symbol = "ETHUSDT"
    # scope guard: BTC/ETH only
    base = symbol.replace("USDT", "")
    if base not in ("BTC", "ETH"):
        return None

    start = raw_topic.get("startDate")
    end = raw_topic.get("endDate")
    duration = _infer_duration(start, end)
    if duration not in (MarketDuration.M5, MarketDuration.M15):
        return None

    mtype = _symbol_to_market_type(symbol, duration)
    if mtype is None:
        return None

    markets = raw_topic.get("markets") or []
    # Prediction markets have exactly one inner market (UP) with YES/NO outcomes; use first
    inner = markets[0] if markets else {}
    outcomes_raw = inner.get("outcomes") or []
    market_id = inner.get("marketId")
    if market_id is None:
        # fallback: try raw_topic marketId
        market_id = raw_topic.get("marketId")
    if market_id is None:
        return None

    orderbooks = orderbooks or {}
    last_prices = last_prices or {}

    outcomes: list[OutcomeSnapshot] = []
    identifiers: list[MarketIdentifiers] = []
    for o in outcomes_raw:
        tok = str(o.get("tokenId", ""))
        ob = orderbooks.get(tok)
        lp = None
        # last_prices keyed by marketId (one per market), not token
        # if we have price for this marketId, attach to both outcomes
        if isinstance(market_id, int) and market_id in last_prices:
            lp = last_prices[market_id]
        outcomes.append(normalize_outcome(o, orderbook=ob, last_price=lp))

    # Build identifiers for each token (YES/NO)
    # We create one NormalizedMarket per topic (contains both outcomes); identifiers are per token
    # Caller should expand identifiers separately if needed.

    liquidity = None
    volume = None
    try:
        if raw_topic.get("liquidity") is not None:
            liquidity = as_decimal(raw_topic["liquidity"], field="liquidity")
        if inner.get("liquidity") is not None and liquidity is None:
            liquidity = as_decimal(inner["liquidity"], field="liquidity")
    except Exception:
        liquidity = None
    try:
        if raw_topic.get("tradeVolume") is not None:
            volume = as_decimal(raw_topic["tradeVolume"], field="tradeVolume")
    except Exception:
        volume = None

    # Build primary identifiers from first outcome (YES) for the NormalizedMarket wrapper
    first_token = outcomes[0].token_id if outcomes else ""
    primary_identifiers = MarketIdentifiers(
        market_topic_id=int(raw_topic["marketTopicId"]),
        market_id=int(market_id),
        token_id=first_token,
        slug=str(raw_topic.get("slug", "")),
        vendor=str(raw_topic.get("vendor", "PREDICT_FUN")),
        chain_id=str(raw_topic.get("chainId", "56")),
        symbol=symbol,
        duration=duration,
        market_type=mtype,
    )

    return NormalizedMarket(
        identifiers=primary_identifiers,
        slug=str(raw_topic.get("slug", "")),
        title=str(raw_topic.get("title", "")),
        question=raw_topic.get("question"),
        symbol=symbol,
        duration=duration,
        market_type=mtype,
        start_ms=start,
        end_ms=end,
        resolution_ms=end,
        status=raw_topic.get("status"),
        trading_status=inner.get("tradingStatus") or raw_topic.get("status"),
        liquidity=liquidity,
        volume=volume,
        participant_count=raw_topic.get("participantCount"),
        fee_rate_bps=raw_topic.get("feeRateBps"),
        outcomes=tuple(outcomes),
        raw_topic=raw_topic,
    )


def extract_identifiers(market: NormalizedMarket) -> list[MarketIdentifiers]:
    """Expand a normalized market into per-token identifiers (YES + NO)."""
    out: list[MarketIdentifiers] = []
    for o in market.outcomes:
        if o.name.upper() not in ("YES", "NO"):
            continue
        # skip empty token
        if not o.token_id:
            continue
        out.append(
            MarketIdentifiers(
                market_topic_id=market.identifiers.market_topic_id,
                market_id=market.identifiers.market_id,
                token_id=o.token_id,
                slug=market.slug,
                vendor=market.identifiers.vendor,
                chain_id=market.identifiers.chain_id,
                symbol=market.symbol,
                duration=market.duration,
                market_type=market.market_type,
            )
        )
    return out
