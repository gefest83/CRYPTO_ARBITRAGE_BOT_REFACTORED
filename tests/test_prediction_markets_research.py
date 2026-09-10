"""Phase 1 — Binance Prediction Markets research (isolated, mocked)."""

from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from app.research.prediction_markets.analyzer import (
    analyze_historical_availability,
    analyze_timestamp_quality,
    analyze_update_frequency,
)
from app.research.prediction_markets.client import BinancePredictionClient
from app.research.prediction_markets.constraints import derive_constraints, estimate_round_trip_cost_bps
from app.research.prediction_markets.discovery import build_research_universe
from app.research.prediction_markets.endpoints import ALL_ENDPOINTS
from app.research.prediction_markets.hypothesis import assess_hypothesis
from app.research.prediction_markets.models import MarketDuration, PredictionMarketType
from app.research.prediction_markets.normalizer import extract_identifiers, normalize_market_topic


# --- Fixtures: raw mock payloads matching Binance legacy docs shape ---

def _topic(market_topic_id: int, symbol: str, start: int, end: int, slug: str, mid: int, token_yes: str, token_no: str) -> dict:
    return {
        "marketTopicId": market_topic_id,
        "vendor": "PREDICT_FUN",
        "chainId": "56",
        "slug": slug,
        "title": f"{symbol} Price 5m Up or Down?" if end - start == 300_000 else f"{symbol} Price 15m Up or Down?",
        "question": f"Will {symbol} go UP?",
        "description": "Resolves YES if price higher at resolution.",
        "symbol": symbol,
        "participantCount": 1000,
        "collateral": "USDT",
        "feeRateBps": 200,
        "slippageBps": 1200,
        "liquidity": "45000.00",
        "tradeVolume": "150000.00",
        "publishedAt": start - 60_000,
        "startDate": start,
        "endDate": end,
        "status": "REGISTERED",
        "markets": [
            {
                "marketId": mid,
                "externalId": f"ext_{mid}",
                "title": "UP",
                "status": "REGISTERED",
                "tradingStatus": "OPEN",
                "liquidity": "25000.00",
                "tradeVolume": "90000.00",
                "outcomes": [
                    {"name": "YES", "price": "0.52", "chance": "0.52", "index": 0, "tokenId": token_yes},
                    {"name": "NO", "price": "0.48", "chance": "0.48", "index": 1, "tokenId": token_no},
                ],
            }
        ],
    }


BTC_5M_TOPIC = _topic(1001, "BTCUSDT", 1_748_131_200_000, 1_748_131_500_000, "btc-price-5m-up-or-down", 9001, "tok_btc5_yes", "tok_btc5_no")
BTC_15M_TOPIC = _topic(1002, "BTCUSDT", 1_748_131_200_000, 1_748_132_100_000, "btc-price-15m-up-or-down", 9002, "tok_btc15_yes", "tok_btc15_no")
ETH_5M_TOPIC = _topic(1003, "ETHUSDT", 1_748_131_200_000, 1_748_131_500_000, "eth-price-5m-up-or-down", 9003, "tok_eth5_yes", "tok_eth5_no")
ETH_15M_TOPIC = _topic(1004, "ETHUSDT", 1_748_131_200_000, 1_748_132_100_000, "eth-price-15m-up-or-down", 9004, "tok_eth15_yes", "tok_eth15_no")
# Out of scope
BNB_5M_TOPIC = _topic(1005, "BNBUSDT", 1_748_131_200_000, 1_748_131_500_000, "bnb-price-5m-up-or-down", 9005, "tok_bnb5_yes", "tok_bnb5_no")
BTC_1H_TOPIC = _topic(1006, "BTCUSDT", 1_748_131_200_000, 1_748_134_800_000, "btc-price-1h-up-or-down", 9006, "tok_btc1h_yes", "tok_btc1h_no")

ORDERBOOK_YES = {
    "outcome": "YES",
    "tokenId": "tok_btc5_yes",
    "timestamp": 1_748_131_400_000,
    "bids": [{"price": "0.51", "size": "5000"}, {"price": "0.50", "size": "12000"}],
    "asks": [{"price": "0.52", "size": "3000"}, {"price": "0.53", "size": "7500"}],
}
LAST_PRICE = {"marketId": 9001, "lastTradePrice": "0.52"}
QUOTE_RESP = {
    "quoteId": "q_test_123",
    "tokenId": "tok_btc5_yes",
    "chance": "0.52",
    "vendor": "PREDICT_FUN",
    "side": "BUY",
    "amountIn": "1000000000000000000",
    "amountOut": "1923070000000000000",
    "feeAmount": "20000000000000000",
    "averagePrice": 0.52,
    "lastPrice": 0.52,
    "priceImpact": 0.001,
    "timestamp": 1_748_131_400_000,
    "chainId": "56",
    "feeRateBps": 200,
    "slippageBps": 1200,
    "expireAt": 1_748_131_430_000,
}


# --- Normalizer tests ---

def test_normalizer_accepts_btc_eth_5m_15m() -> None:
    for topic in (BTC_5M_TOPIC, BTC_15M_TOPIC, ETH_5M_TOPIC, ETH_15M_TOPIC):
        nm = normalize_market_topic(topic)
        assert nm is not None, f"should accept {topic['slug']}"
        assert nm.symbol in ("BTCUSDT", "ETHUSDT")
        assert nm.duration in (MarketDuration.M5, MarketDuration.M15)
        assert nm.identifiers.market_topic_id == topic["marketTopicId"]
        assert nm.identifiers.token_id in (topic["markets"][0]["outcomes"][0]["tokenId"],)
        assert nm.identifiers.slug == topic["slug"]
        assert len(nm.outcomes) == 2
        assert nm.volume is not None
        assert nm.liquidity is not None


def test_normalizer_rejects_bnb_and_wrong_duration() -> None:
    assert normalize_market_topic(BNB_5M_TOPIC) is None
    assert normalize_market_topic(BTC_1H_TOPIC) is None


def test_normalizer_resolves_identifiers_and_outcomes() -> None:
    nm = normalize_market_topic(BTC_5M_TOPIC, orderbooks={"tok_btc5_yes": ORDERBOOK_YES}, last_prices={9001: "0.52"})
    assert nm is not None
    ids = extract_identifiers(nm)
    assert len(ids) == 2
    assert {i.token_id for i in ids} == {"tok_btc5_yes", "tok_btc5_no"}
    # all share same marketId/marketTopicId
    assert all(i.market_id == 9001 for i in ids)
    assert all(i.market_topic_id == 1001 for i in ids)
    # slug + symbol propagated
    assert all(i.slug == "btc-price-5m-up-or-down" for i in ids)

    yes = next(o for o in nm.outcomes if o.name == "YES")
    assert yes.best_bid == Decimal("0.51")
    assert yes.best_ask == Decimal("0.52")
    assert yes.spread == Decimal("0.01")
    assert yes.last_trade_price == Decimal("0.52")
    assert yes.timestamp_ms == 1_748_131_400_000


def test_market_types_are_exactly_btc_eth_5m_15m() -> None:
    mapping = {
        (BTC_5M_TOPIC["marketTopicId"], "BTCUSDT", 300_000): PredictionMarketType.BTC_5M,
        (BTC_15M_TOPIC["marketTopicId"], "BTCUSDT", 900_000): PredictionMarketType.BTC_15M,
        (ETH_5M_TOPIC["marketTopicId"], "ETHUSDT", 300_000): PredictionMarketType.ETH_5M,
        (ETH_15M_TOPIC["marketTopicId"], "ETHUSDT", 900_000): PredictionMarketType.ETH_15M,
    }
    for topic in (BTC_5M_TOPIC, BTC_15M_TOPIC, ETH_5M_TOPIC, ETH_15M_TOPIC):
        nm = normalize_market_topic(topic)
        assert nm is not None
        assert nm.market_type == mapping[(topic["marketTopicId"], topic["symbol"], topic["endDate"] - topic["startDate"])]


def test_immutable_models_reject_extra_bnb() -> None:
    nm = normalize_market_topic(BTC_5M_TOPIC)
    assert nm is not None
    # frozen
    with pytest.raises(Exception):
        nm.symbol = "BNBUSDT"  # type: ignore[misc]


# --- Constraints / fees ---

def test_constraints_and_round_trip() -> None:
    c = derive_constraints(BTC_5M_TOPIC)
    assert c.fee_rate_bps == 200
    assert c.slippage_bps == 1200
    assert c.collateral == "USDT"
    assert c.supports_market_orders is True
    total = estimate_round_trip_cost_bps(200, Decimal("193"))  # 0.01 spread on 0.515 mid ~194 bps
    assert total == Decimal(200 * 2) + Decimal("193")


# --- Analyzers ---

def test_timestamp_quality() -> None:
    tq = analyze_timestamp_quality(BTC_5M_TOPIC)
    assert tq.has_exchange_timestamp is True
    assert tq.has_start_end_resolution is True
    assert tq.ordering_guaranteed is False


def test_update_frequency() -> None:
    uf = analyze_update_frequency()
    assert uf.ws_latency_expected_ms == 200
    assert uf.rest_weight_per_call == 200
    assert uf.is_active_only is True
    uf2 = analyze_update_frequency([100, 200, 150])
    assert uf2.measured_interval_ms == pytest.approx(150.0)
    assert uf2.sample_count == 3


def test_historical_availability() -> None:
    h = analyze_historical_availability()
    assert h.binance_provides_history is False
    assert h.predict_fun_has_history is True
    assert "snapshot-only" in h.notes


# --- Hypothesis ---

def test_hypothesis_testable_with_active_markets() -> None:
    nm = normalize_market_topic(BTC_5M_TOPIC, orderbooks={"tok_btc5_yes": ORDERBOOK_YES})
    assert nm is not None
    c = derive_constraints(BTC_5M_TOPIC)
    h = analyze_historical_availability()
    spreads = [o.spread_bps for o in nm.outcomes if o.spread_bps is not None]
    testable, notes = assess_hypothesis([nm], c, h, spreads)
    assert testable is True
    assert "Active markets" in notes


def test_hypothesis_not_testable_no_markets() -> None:
    c = derive_constraints(None)
    h = analyze_historical_availability()
    testable, notes = assess_hypothesis([], c, h)
    assert testable is False


# --- Client with mocked transport ---

def _mock_transport(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    params = dict(request.url.params)
    if path.endswith("/market/list"):
        return httpx.Response(200, json={"marketTopics": [BTC_5M_TOPIC, BTC_15M_TOPIC, ETH_5M_TOPIC, ETH_15M_TOPIC, BNB_5M_TOPIC, BTC_1H_TOPIC], "total": 6, "offset": 0, "limit": 100, "hasMore": False})
    if path.endswith("/market/search"):
        q = params.get("query", "").upper()
        topics = [t for t in [BTC_5M_TOPIC, BTC_15M_TOPIC, ETH_5M_TOPIC, ETH_15M_TOPIC] if q in t["symbol"]]
        return httpx.Response(200, json=topics)
    if path.endswith("/market/detail"):
        tid = int(params.get("marketTopicId", 0))
        mapping = {t["marketTopicId"]: t for t in [BTC_5M_TOPIC, BTC_15M_TOPIC, ETH_5M_TOPIC, ETH_15M_TOPIC, BNB_5M_TOPIC, BTC_1H_TOPIC]}
        return httpx.Response(200, json=mapping.get(tid, BTC_5M_TOPIC))
    if path.endswith("/order-book"):
        return httpx.Response(200, json=ORDERBOOK_YES)
    if path.endswith("/last-trade-price"):
        return httpx.Response(200, json=LAST_PRICE)
    if path.endswith("/trade/get-quote"):
        return httpx.Response(200, json=QUOTE_RESP)
    if path.endswith("/category/list"):
        return httpx.Response(200, json={"categories": [{"id": "crypto", "name": "Crypto"}]})
    if path.endswith("/wallet/list"):
        return httpx.Response(200, json={"wallets": [{"walletAddress": "0xabc", "walletId": "wid"}]})
    return httpx.Response(404, json={"msg": "not mocked"})


@pytest.mark.asyncio
async def test_client_mocked_endpoints() -> None:
    transport = httpx.MockTransport(_mock_transport)
    async with BinancePredictionClient("key", "secret", transport=transport) as client:
        cats = await client.list_categories()
        assert "categories" in cats

        data = await client.list_markets(l1_category="crypto", limit=100)
        assert len(data["marketTopics"]) == 6

        detail = await client.get_market_detail(1001)
        assert detail["marketTopicId"] == 1001

        ob = await client.get_order_book("PREDICT_FUN", 9001, "tok_btc5_yes")
        assert ob["tokenId"] == "tok_btc5_yes"
        assert ob["bids"][0]["price"] == "0.51"

        lp = await client.get_last_trade_price(9001)
        assert lp["lastTradePrice"] == "0.52"

        # quote inspection (read-only)
        quote = await client.get_quote_inspection("0xabc", "tok_btc5_yes", "BUY", "1000000000000000000")
        assert quote["quoteId"] == "q_test_123"
        assert quote["averagePrice"] == 0.52
        assert "priceImpact" in quote

        spec = client.quote_spec()
        assert spec["places_order"] is False
        assert spec["read_only"] is True
        assert spec["endpoint"] == "/sapi/v1/w3w/wallet/prediction/trade/get-quote"


@pytest.mark.asyncio
async def test_discovery_pipeline_end_to_end() -> None:
    transport = httpx.MockTransport(_mock_transport)
    async with BinancePredictionClient("key", "secret", transport=transport) as client:
        universe = await build_research_universe(client)
        # Only BTC/ETH 5m/15m survive (BNB + 1h filtered)
        assert len(universe.markets) == 4
        types = set(universe.market_types_found)
        assert types == {PredictionMarketType.BTC_5M, PredictionMarketType.BTC_15M, PredictionMarketType.ETH_5M, PredictionMarketType.ETH_15M}
        # identifiers: 2 outcomes per market = 8
        assert len(universe.identifiers) == 8
        assert all(i.vendor == "PREDICT_FUN" for i in universe.identifiers)
        assert all(i.chain_id == "56" for i in universe.identifiers)
        # no BNB
        assert not any("BNB" in i.symbol for i in universe.identifiers)
        # durations
        assert {m.duration for m in universe.markets} == {MarketDuration.M5, MarketDuration.M15}
        # metadata
        assert universe.constraints.fee_rate_bps == 200
        assert universe.timestamp_quality.has_start_end_resolution is True
        assert universe.update_frequency.ws_latency_expected_ms == 200
        assert universe.historical.binance_provides_history is False
        assert universe.hypothesis_testable is True


def test_endpoints_registry_complete() -> None:
    assert ALL_ENDPOINTS["market_list"] == "/sapi/v1/w3w/wallet/prediction/market/list"
    assert ALL_ENDPOINTS["order_book"] == "/sapi/v1/w3w/wallet/prediction/order-book"
    assert ALL_ENDPOINTS["get_quote"] == "/sapi/v1/w3w/wallet/prediction/trade/get-quote"
    assert ALL_ENDPOINTS["market_detail"] == "/sapi/v1/w3w/wallet/prediction/market/detail"
    assert ALL_ENDPOINTS["last_trade_price"] == "/sapi/v1/w3w/wallet/prediction/order-book/last-trade-price"


def test_isolation_no_forbidden_imports() -> None:
    """Research package must not import forbidden modules."""
    import pathlib, re
    root = pathlib.Path("app/research/prediction_markets")
    forbidden = ["triangle", "transfer", "execution", "risk", "RiskEngine", "ExecutionGuard", "recovery", "telegram", "kronos", "ai advisor"]
    for p in root.glob("*.py"):
        text = p.read_text(encoding="utf-8").lower()
        for kw in forbidden:
            # allow the word in docstrings that says "must not modify triangle"
            # but not as an import
            if f"import {kw}" in text or f"from app.{kw}" in text or f"from app.strategies.{kw}" in text or f"from app.execution" in text:
                raise AssertionError(f"{p.name} imports forbidden module {kw}")
