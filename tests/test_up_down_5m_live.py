"""Live wiring tests: SAPI/normalizer -> UpDown5mMarket snapshot (research-only).

All offline via httpx MockTransport. No network, no trading, no credentials.
"""

from __future__ import annotations

from decimal import Decimal

import httpx
import pytest

from app.research.prediction_markets.client import BinancePredictionClient
from app.research.prediction_markets.normalizer import normalize_market_topic
from app.research.prediction_markets.up_down_5m.live_snapshot import (
    NoLiveMarketError,
    PriceProvenance,
    build_live_snapshot,
    ensure_research_only,
    fetch_one_btc_5m_snapshot,
    render_snapshot,
    required_field_coverage,
    settlement_preview,
)
from app.research.prediction_markets.up_down_5m.market import MarketState
from app.research.prediction_markets.up_down_5m.settlement import SettlementOutcome

T0 = 1_748_131_200_000
T1 = T0 + 300_000


def _topic(symbol: str = "BTCUSDT", start: int = T0, end: int = T1, trading: str = "OPEN") -> dict:
    return {
        "marketTopicId": 1001,
        "vendor": "PREDICT_FUN",
        "chainId": "56",
        "slug": "btc-price-5m-up-or-down",
        "title": f"{symbol} Price 5m Up or Down?",
        "question": f"Will {symbol} go UP?",
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
                "marketId": 9001,
                "externalId": "ext_9001",
                "title": "UP",
                "status": "REGISTERED",
                "tradingStatus": trading,
                "liquidity": "25000.00",
                "tradeVolume": "90000.00",
                "outcomes": [
                    {"name": "YES", "price": "0.52", "chance": "0.52", "index": 0, "tokenId": "tok_yes"},
                    {"name": "NO", "price": "0.48", "chance": "0.48", "index": 1, "tokenId": "tok_no"},
                ],
            }
        ],
    }


BOOK_YES = {
    "tokenId": "tok_yes",
    "timestamp": T0 + 1000,
    "bids": [{"price": "0.51", "size": "5000"}],
    "asks": [{"price": "0.53", "size": "3000"}],
}
BOOK_NO = {
    "tokenId": "tok_no",
    "timestamp": T0 + 1000,
    "bids": [{"price": "0.46", "size": "4000"}],
    "asks": [{"price": "0.48", "size": "3500"}],
}


def _normalized(**kw) -> object:
    topic = _topic(**kw) if kw else _topic()
    return normalize_market_topic(topic, orderbooks={"tok_yes": BOOK_YES, "tok_no": BOOK_NO})


# ---------- field mapping ----------

def test_snapshot_maps_ids_window_and_books() -> None:
    nm = _normalized()
    assert nm is not None
    snap = build_live_snapshot(nm, spot_mid_at_start="68000", spot_mid_now="68050", checked_at_ms=T0 + 1000)
    m = snap.market
    assert m.market_id == 9001
    assert m.start_ts_ms == T0 and m.end_ts_ms == T1
    # YES -> UP, NO -> DOWN
    assert m.up_bid == Decimal("0.51") and m.up_ask == Decimal("0.53")
    assert m.down_bid == Decimal("0.46") and m.down_ask == Decimal("0.48")
    assert snap.up_token_id == "tok_yes" and snap.down_token_id == "tok_no"
    assert snap.trading_status == "OPEN"


def test_spot_reference_provenance_never_settles() -> None:
    nm = _normalized()
    assert nm is not None
    snap = build_live_snapshot(
        nm, spot_mid_at_start="68000", spot_mid_now="68100", checked_at_ms=T0 + 1000,
    )
    assert snap.start_provenance == PriceProvenance.SPOT_REFERENCE
    assert snap.current_provenance == PriceProvenance.SPOT_REFERENCE
    assert snap.market.start_price == Decimal("68000")
    outcome, reason = settlement_preview(snap)
    assert outcome is None  # no Chainlink end anchor -> withheld
    assert "withheld" in reason
    cov = {c.name: c for c in required_field_coverage(snap)}
    assert cov["market_id"].present and cov["start_ts"].present and cov["end_ts"].present
    assert cov["up_bid_ask"].present and cov["down_bid_ask"].present
    assert cov["final_settlement_price_outcome"].present is False


def test_chainlink_anchors_enable_settlement_preview() -> None:
    nm = _normalized()
    assert nm is not None
    snap = build_live_snapshot(
        nm,
        chainlink_start="68000",
        chainlink_current="68050",
        chainlink_end="68100",
        checked_at_ms=T1 + 1000,
    )
    assert snap.start_provenance == PriceProvenance.CHAINLINK_VENUE
    assert snap.end_provenance == PriceProvenance.CHAINLINK_VENUE
    outcome, _ = settlement_preview(snap)
    assert outcome == SettlementOutcome.UP
    cov = {c.name: c for c in required_field_coverage(snap)}
    assert cov["final_settlement_price_outcome"].present is True


def test_chainlink_wins_over_spot_reference() -> None:
    nm = _normalized()
    assert nm is not None
    snap = build_live_snapshot(
        nm, chainlink_start="68000", spot_mid_at_start="67900", checked_at_ms=T0 + 1000,
    )
    assert snap.market.start_price == Decimal("68000")
    assert snap.start_provenance == PriceProvenance.CHAINLINK_VENUE


def test_missing_prices_are_unavailable_not_fabricated() -> None:
    nm = _normalized()
    assert nm is not None
    snap = build_live_snapshot(nm, checked_at_ms=T0 + 1000)
    assert snap.market.start_price is None and snap.market.current_price is None
    assert snap.start_provenance == PriceProvenance.UNAVAILABLE
    cov = {c.name: c for c in required_field_coverage(snap)}
    assert cov["start_chainlink_tob_mid"].present is False


# ---------- guards ----------

def test_rejects_non_btc() -> None:
    nm = normalize_market_topic(_topic(symbol="ETHUSDT"), orderbooks={"tok_yes": BOOK_YES, "tok_no": BOOK_NO})
    assert nm is not None
    with pytest.raises(ValueError, match="BTC-only"):
        build_live_snapshot(nm, checked_at_ms=T0)


def test_rejects_non_5m() -> None:
    nm = normalize_market_topic(
        _topic(end=T0 + 900_000), orderbooks={"tok_yes": BOOK_YES, "tok_no": BOOK_NO}
    )
    assert nm is not None  # normalizer accepts 15m; wiring must refuse
    with pytest.raises(ValueError, match="5m-only"):
        build_live_snapshot(nm, checked_at_ms=T0)


def test_rejects_missing_down_outcome() -> None:
    topic = _topic()
    topic["markets"][0]["outcomes"] = [
        {"name": "YES", "price": "0.52", "chance": "0.52", "index": 0, "tokenId": "tok_yes"},
    ]
    nm = normalize_market_topic(topic, orderbooks={"tok_yes": BOOK_YES})
    assert nm is not None
    with pytest.raises(ValueError, match="YES.*NO|NO.*YES|both YES"):
        build_live_snapshot(nm, checked_at_ms=T0)


# ---------- end-to-end via mocked SAPI ----------

def _mock_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    params = dict(request.url.params)
    if path.endswith("/market/list"):
        return httpx.Response(200, json={"marketTopics": [_topic()], "total": 1})
    if path.endswith("/market/search"):
        return httpx.Response(200, json=[_topic()])
    if path.endswith("/market/detail"):
        return httpx.Response(200, json=_topic())
    if path.endswith("/order-book") and not path.endswith("last-trade-price"):
        tok = params.get("tokenId", "")
        return httpx.Response(200, json=BOOK_NO if tok == "tok_no" else BOOK_YES)
    if path.endswith("/last-trade-price"):
        return httpx.Response(200, json={"marketId": 9001, "lastTradePrice": "0.52"})
    if path.endswith("/category/list"):
        return httpx.Response(200, json={"categories": [{"id": "crypto", "name": "Crypto"}]})
    return httpx.Response(404, json={"msg": "not mocked"})


@pytest.mark.asyncio
async def test_fetch_one_btc_5m_end_to_end_mocked() -> None:
    transport = httpx.MockTransport(_mock_handler)
    async with BinancePredictionClient("k", "s", transport=transport) as client:
        snap = await fetch_one_btc_5m_snapshot(
            client, spot_mid_at_start="68000", spot_mid_now="68010", checked_at_ms=T0 + 5000,
        )
    assert snap.market.market_id == 9001
    assert snap.market.up_bid == Decimal("0.51")
    assert snap.market.down_ask == Decimal("0.48")
    assert snap.market.state_at(T0 + 5000) == MarketState.OPEN


@pytest.mark.asyncio
async def test_fetch_raises_fail_closed_when_no_btc_5m() -> None:
    def _empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"marketTopics": [], "total": 0})

    transport = httpx.MockTransport(_empty)
    async with BinancePredictionClient("k", "s", transport=transport) as client:
        with pytest.raises(NoLiveMarketError):
            await fetch_one_btc_5m_snapshot(client, checked_at_ms=T0)


# ---------- render ----------

def test_render_contains_full_lifecycle_snapshot() -> None:
    nm = _normalized()
    assert nm is not None
    snap = build_live_snapshot(
        nm, spot_mid_at_start="68000", spot_mid_now="68010", checked_at_ms=T0 + 5000,
    )
    text = render_snapshot(snap, T0 + 5000)
    for label in (
        "market_id=", "start_ts_ms=", "end_ts_ms=",
        "start_chainlink_tob_mid=", "current_chainlink_tob_mid=",
        "up_bid=", "up_ask=", "down_bid=", "down_ask=",
        "final_settlement_price=", "state@", "coverage:",
    ):
        assert label in text, f"render missing {label!r}"
    assert "no trading" in text.lower() or "research-only" in text.lower()


def test_live_wiring_is_research_only() -> None:
    with pytest.raises(RuntimeError):
        ensure_research_only()


def test_no_trading_imports_in_new_wiring() -> None:
    import pathlib

    for p in (
        pathlib.Path("app/research/prediction_markets/up_down_5m/live_snapshot.py"),
        pathlib.Path("scripts/run_up_down_5m_snapshot.py"),
    ):
        text = p.read_text(encoding="utf-8").lower()
        for kw in ("from app.execution", "from app.risk", "from app.recovery", "from app.telegram", "from app.agent", "place_order(", "submit_order(", "batch_redeem("):
            assert kw not in text, f"{p.name} must stay research-only, found {kw!r}"
