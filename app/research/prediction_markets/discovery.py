"""Discovery orchestrator: find BTC/ETH 5m/15m markets, enrich, normalize."""

from __future__ import annotations

from decimal import Decimal

from app.research.prediction_markets.analyzer import (
    analyze_historical_availability,
    analyze_timestamp_quality,
    analyze_update_frequency,
)
from app.research.prediction_markets.client import BinancePredictionClient
from app.research.prediction_markets.constraints import derive_constraints
from app.research.prediction_markets.hypothesis import assess_hypothesis
from app.research.prediction_markets.models import ResearchUniverse
from app.research.prediction_markets.normalizer import extract_identifiers, normalize_market_topic

__all__ = ["discover_markets", "build_research_universe"]


async def discover_markets(client: BinancePredictionClient) -> list[dict]:
    """Discover BTC/ETH prediction topics via market/list + search fallback.

    Returns raw topic dicts filtered to crypto/UpDown categories.
    The client is signed — in tests this hits the mock transport.
    """
    topics: list[dict] = []
    seen: set[int] = set()

    # Primary: list by crypto category
    try:
        data = await client.list_markets(l1_category="crypto", limit=100)
        for t in data.get("marketTopics", []):
            tid = t.get("marketTopicId")
            if tid and tid not in seen:
                seen.add(tid)
                topics.append(t)
    except Exception:
        pass

    # Fallback: search BTC/ETH
    for q in ("BTC", "ETH", "bitcoin", "ethereum"):
        try:
            results = await client.search_markets(q, top_k=50)
            for t in results if isinstance(results, list) else []:
                tid = t.get("marketTopicId")
                if tid and tid not in seen:
                    seen.add(tid)
                    topics.append(t)
        except Exception:
            continue

    return topics


async def build_research_universe(client: BinancePredictionClient) -> ResearchUniverse:
    """Full pipeline: discover -> detail -> orderbook -> normalize -> analyze."""
    raw_topics = await discover_markets(client)

    # Enrich with detail for accurate variantData/start/end
    enriched: list[dict] = []
    for t in raw_topics:
        try:
            detail = await client.get_market_detail(t["marketTopicId"])
            # Binance detail returns single topic dict; merge markets if present
            enriched.append(detail if isinstance(detail, dict) and "marketTopicId" in detail else t)
        except Exception:
            enriched.append(t)

    # For each topic, optionally fetch orderbook + lastTradePrice for YES/NO
    orderbooks: dict[str, dict] = {}
    last_prices: dict[int, str] = {}
    for topic in enriched:
        for m in topic.get("markets") or []:
            mid = m.get("marketId")
            vendor = topic.get("vendor", "PREDICT_FUN")
            for o in m.get("outcomes") or []:
                tok = str(o.get("tokenId", ""))
                if not tok:
                    continue
                try:
                    ob = await client.get_order_book(vendor, mid, tok)
                    orderbooks[tok] = ob
                except Exception:
                    pass
            if mid is not None:
                try:
                    lp = await client.get_last_trade_price(mid)
                    # response: {"marketId": ..., "lastTradePrice": "..."}
                    price = lp.get("lastTradePrice") if isinstance(lp, dict) else None
                    if price is not None:
                        last_prices[int(mid)] = str(price)
                except Exception:
                    pass

    # Normalize (scope-filtered)
    normalized = []
    for topic in enriched:
        # pass per-token orderbooks; last_prices keyed by marketId
        nm = normalize_market_topic(topic, orderbooks=orderbooks, last_prices=last_prices)
        if nm is not None:
            normalized.append(nm)

    # Constraints from first active market
    constraints = derive_constraints(normalized[0].raw_topic if normalized else None)

    # Analyzers
    ts_quality = analyze_timestamp_quality(normalized[0].raw_topic if normalized else None)
    update_freq = analyze_update_frequency()
    historical = analyze_historical_availability()

    # identifiers
    all_ids = []
    for nm_item in normalized:
        all_ids.extend(extract_identifiers(nm_item))

    market_types = tuple(sorted({m.market_type for m in normalized}, key=lambda x: x.value))

    # spreads for hypothesis
    spreads: list[Decimal] = []
    for m in normalized:
        for o in m.outcomes:
            if o.spread_bps is not None:
                spreads.append(o.spread_bps)

    testable, notes = assess_hypothesis(normalized, constraints, historical, spreads)

    return ResearchUniverse(
        markets=tuple(normalized),
        market_types_found=market_types,
        identifiers=tuple(all_ids),
        constraints=constraints,
        timestamp_quality=ts_quality,
        update_frequency=update_freq,
        historical=historical,
        hypothesis_testable=testable,
        hypothesis_notes=notes,
    )
