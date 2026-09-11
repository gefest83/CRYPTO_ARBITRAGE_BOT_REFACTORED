"""Focused tests for Predict.fun + Binance Spot WebSocket upgrade — research only."""

from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.research.prediction_markets.binance_spot_ws import (
    BINANCE_WS_TOPICS,
    build_spot_subscription,
    parse_spot_ws_message,
)
from app.research.prediction_markets.collector.collector import HistoricalCollector
from app.research.prediction_markets.collector.models import SourceKind
from app.research.prediction_markets.collector.storage import CollectorStore
from app.research.prediction_markets.predict_ws import (
    PREDICT_WS_CHANNEL,
    PREDICT_WS_URL,
    build_predict_subscription,
    parse_predict_ws_message,
)

T0 = 1_748_131_200_000
RES_5M = T0 + 300_000


# --- Predict.fun WS parsing ---

def test_predict_ws_parse_snapshot_preserves_timestamps():
    msg = {
        "channel": "orderbook",
        "data": {
            "marketId": 2191278,
            "bids": [[0.51, 100], [0.50, 200]],
            "asks": [[0.53, 150], [0.54, 300]],
            "updateTimestampMs": T0 + 1200,
            "sequence": 42,
            "symbol": "BTCUSDT",
            "duration": "5m",
            "resolution_ms": RES_5M,
        },
    }
    parsed = parse_predict_ws_message(msg)
    assert parsed is not None
    assert parsed["marketId"] == 2191278
    assert parsed["updateTimestampMs"] == T0 + 1200
    assert parsed["sequence"] == 42
    assert parsed["symbol"] == "BTCUSDT"
    # Collector should preserve these without fabricating
    store = CollectorStore(memory_only=True)
    coll = HistoricalCollector(store, stale_threshold_ms=5000, prediction_stale_threshold_ms=300000, resolution_by_market={2191278: RES_5M})
    captured = T0 + 1250
    obs = coll.ingest_prediction_orderbook(parsed, captured_at_ms=captured)
    assert obs is not None
    assert obs.captured_at_ms == captured  # exact local receive time
    assert obs.exchange_ts_ms == T0 + 1200  # preserved from payload
    assert obs.update_ts_ms == T0 + 1200
    assert obs.market_id == 2191278


def test_predict_ws_parse_official_push_format():
    """Official docs: {"type":"M","topic":"predictOrderbook/123","data":{...}} must be parsed."""
    msg = {
        "type": "M",
        "topic": "predictOrderbook/2191278",
        "data": {
            "marketId": 2191278,
            "bids": [[0.51, 100]],
            "asks": [[0.53, 150]],
            "updateTimestampMs": T0 + 1200,
            "sequence": 42,
            "version": 1,
        },
    }
    parsed = parse_predict_ws_message(msg)
    assert parsed is not None
    assert parsed["marketId"] == 2191278
    assert parsed["updateTimestampMs"] == T0 + 1200
    # Also test that subscription ack is ignored
    ack = {"type": "R", "requestId": 0, "success": True}
    assert parse_predict_ws_message(ack) is None


def test_predict_ws_parse_update_with_same_timestamp_not_discarded():
    """Requirement 6: unchanged updateTimestampMs must not cause discard if snapshot is valid."""
    store = CollectorStore(memory_only=True)
    coll = HistoricalCollector(store, stale_threshold_ms=5000, prediction_stale_threshold_ms=300000, resolution_by_market={9001: RES_5M})
    # First snapshot
    msg1 = {
        "marketId": 9001,
        "tokenId": "tok_yes",
        "symbol": "BTCUSDT",
        "updateTimestampMs": T0 + 1000,
        "sequence": None,  # no sequence to trigger content-hash dedup
        "bids": [[0.51, 100]],
        "asks": [[0.53, 150]],
        "duration": "5m",
        "resolution_ms": RES_5M,
    }
    obs1 = coll.ingest_prediction_orderbook(msg1, captured_at_ms=T0 + 1050)
    assert obs1 is not None
    assert store.count() == 1
    # Second snapshot with SAME updateTimestampMs but different bids (valid WS snapshot, not duplicate)
    msg2 = {
        "marketId": 9001,
        "tokenId": "tok_yes",
        "symbol": "BTCUSDT",
        "updateTimestampMs": T0 + 1000,  # unchanged
        "sequence": None,
        "bids": [[0.52, 120]],  # different price
        "asks": [[0.54, 160]],
        "duration": "5m",
        "resolution_ms": RES_5M,
    }
    obs2 = coll.ingest_prediction_orderbook(msg2, captured_at_ms=T0 + 1100)
    # Should NOT be discarded as duplicate
    assert obs2 is not None, "valid WS snapshot with same updateTimestampMs but different book should not be discarded"
    assert store.count() == 2
    assert coll.stats.duplicates_dropped == 0
    # True duplicate with same content SHOULD be deduped
    obs3 = coll.ingest_prediction_orderbook(msg2, captured_at_ms=T0 + 1150)
    assert obs3 is None
    assert coll.stats.duplicates_dropped == 1


def test_predict_ws_parse_preserves_no_fabrication():
    """Do not fabricate exchange timestamp if missing."""
    store = CollectorStore(memory_only=True)
    coll = HistoricalCollector(store, stale_threshold_ms=5000, prediction_stale_threshold_ms=300000, resolution_by_market={9001: RES_5M})
    msg = {
        "marketId": 9001,
        "tokenId": "tok_yes",
        "symbol": "BTCUSDT",
        # No updateTimestampMs
        "bids": [[0.51, 100]],
        "asks": [[0.53, 150]],
        "duration": "5m",
        "resolution_ms": RES_5M,
    }
    captured = T0 + 1000
    obs = coll.ingest_prediction_orderbook(msg, captured_at_ms=captured)
    assert obs is not None
    assert obs.captured_at_ms == captured
    # Exchange timestamps should be None, not fabricated as captured
    assert obs.exchange_ts_ms is None
    assert obs.update_ts_ms is None


# --- Subscription ---

def test_predict_ws_build_subscription_exact_format():
    # Official docs: {"requestId":0,"method":"subscribe","params":["predictOrderbook/123"]} one topic per request
    subs = build_predict_subscription([2191278, 2191279, 2191278])  # duplicate should be deduped and sorted
    assert len(subs) == 2
    assert subs[0] == {"requestId": 0, "method": "subscribe", "params": ["predictOrderbook/2191278"]}
    assert subs[1] == {"requestId": 1, "method": "subscribe", "params": ["predictOrderbook/2191279"]}
    # Ensure WS URL is correct
    assert PREDICT_WS_URL == "wss://ws.predict.fun/ws"
    # Also test single helper matches docs
    from app.research.prediction_markets.predict_ws import build_predict_subscription_single

    single = build_predict_subscription_single(2191278, request_id=0)
    assert single == {"requestId": 0, "method": "subscribe", "params": ["predictOrderbook/2191278"]}


def test_predict_ws_subscription_filters_btc_eth_only():
    """Discovery must filter BTC/ETH 5m/15m, BNB excluded."""
    from app.research.prediction_markets.predict_ws import _filter_btc_eth_5m_15m

    cats = [
        {"slug": "btc-updown-5m-123", "variantData": {"priceFeedSymbol": "BTCUSDT"}, "tags": [{"name": "5 Min"}], "endsAt": "2026-09-11T12:00:00.000Z", "markets": [{"id": 1}]},
        {"slug": "eth-updown-15m-123", "variantData": {"priceFeedSymbol": "ETHUSDT"}, "tags": [{"name": "15 Min"}], "endsAt": "2026-09-11T12:00:00.000Z", "markets": [{"id": 2}]},
        {"slug": "bnb-updown-5m-123", "variantData": {"priceFeedSymbol": "BNBUSDT"}, "tags": [{"name": "5 Min"}], "endsAt": "2026-09-11T12:00:00.000Z", "markets": [{"id": 3}]},
        {"slug": "random-market", "variantData": {"priceFeedSymbol": "SOLUSDT"}, "tags": [], "endsAt": "2026-09-11T12:00:00.000Z", "markets": [{"id": 4}]},
    ]
    filtered = _filter_btc_eth_5m_15m(cats)  # type: ignore[arg-type]
    mids = [f["market_id"] for f in filtered]
    assert 1 in mids and 2 in mids
    assert 3 not in mids  # BNB excluded
    assert 4 not in mids  # SOL excluded


def test_binance_spot_ws_subscription_exact():
    sub = build_spot_subscription()
    assert sub["method"] == "SUBSCRIBE"
    assert "btcusdt@bookTicker" in sub["params"]
    assert "ethusdt@bookTicker" in sub["params"]
    assert sub["params"] == BINANCE_WS_TOPICS
    assert sub["id"] == 1
    # Ensure topics are BTC/ETH only, BNB excluded
    assert not any("bnb" in t for t in sub["params"])


# --- Binance Spot WS parsing ---

def test_binance_spot_ws_parse_bookTicker_preserves_exchange_ts():
    msg = {
        "e": "bookTicker",
        "E": T0 + 1000,
        "s": "BTCUSDT",
        "b": "68000.00",
        "B": "0.5",
        "a": "68000.50",
        "A": "0.6",
        "u": 12345,
    }
    parsed = parse_spot_ws_message(msg)
    assert parsed is not None
    assert parsed["symbol"] == "BTCUSDT"
    assert parsed["exchange_ts_ms"] == T0 + 1000  # preserved E
    assert parsed["bid"] == "68000.00"
    assert parsed["ask"] == "68000.50"
    assert parsed["sequence"] == 12345

    store = CollectorStore(memory_only=True)
    coll = HistoricalCollector(store, stale_threshold_ms=5000, prediction_stale_threshold_ms=300000)
    captured = T0 + 1050
    obs = coll.ingest_spot(parsed, captured_at_ms=captured)
    assert obs is not None
    assert obs.captured_at_ms == captured
    assert obs.exchange_ts_ms == T0 + 1000
    assert obs.bid == Decimal("68000.00")


def test_binance_spot_ws_parse_combined_stream_wrapper():
    msg = {
        "stream": "btcusdt@bookTicker",
        "data": {
            "e": "bookTicker",
            "E": T0 + 2000,
            "s": "ETHUSDT",
            "b": "2500.00",
            "B": "1.2",
            "a": "2500.10",
            "A": "0.8",
        },
    }
    parsed = parse_spot_ws_message(msg)
    assert parsed is not None
    assert parsed["symbol"] == "ETHUSDT"
    assert parsed["exchange_ts_ms"] == T0 + 2000


def test_binance_spot_ws_bnb_excluded():
    msg = {"e": "bookTicker", "E": T0 + 1000, "s": "BNBUSDT", "b": "600.00", "a": "600.10"}
    parsed = parse_spot_ws_message(msg)
    # Parser filters BNB at WS layer (BTC/ETH only) -> None, collector also would drop
    assert parsed is None
    # Even if parser passed through, collector must drop BNB
    store = CollectorStore(memory_only=True)
    coll = HistoricalCollector(store)
    # Direct ingest with BNB should be dropped
    obs = coll.ingest_spot({"symbol": "BNBUSDT", "bid": "600", "ask": "600.1", "exchange_ts_ms": T0 + 1000}, captured_at_ms=T0 + 1050)
    assert obs is None  # BNB excluded
    assert store.count() == 0


def test_collector_preserves_timestamps_and_not_fabricate_spot():
    """Do not fabricate exchange timestamp if missing - keep None."""
    store = CollectorStore(memory_only=True)
    coll = HistoricalCollector(store)
    data = {"symbol": "BTCUSDT", "bid": "68000", "ask": "68001"}  # no exchange_ts
    captured = T0 + 1000
    obs = coll.ingest_spot(data, captured_at_ms=captured)
    assert obs is not None
    assert obs.captured_at_ms == captured
    assert obs.exchange_ts_ms is None  # not fabricated


# --- Reconnect and snapshot/update handling ---

def test_collector_reconnect_increments_counter():
    store = CollectorStore(memory_only=True)
    coll = HistoricalCollector(store)
    assert coll.stats.reconnects == 0
    coll.on_reconnect("ws closed")
    assert coll.stats.reconnects == 1
    coll.on_reconnect("another")
    assert coll.stats.reconnects == 2


@pytest.mark.asyncio
async def test_predict_ws_reconnect_triggers_rest_snapshot(tmp_path):
    """On WS disconnect, collector should increment reconnect and trigger REST snapshot."""
    # Mock REST via PredictFunClient with MockTransport
    import httpx
    from app.research.prediction_markets.predict_client import PredictFunClient
    from app.research.prediction_markets.predict_ws import PredictFunWSClient

    # Mock categories to return one BTC market
    cat = {
        "slug": "btc-updown-5m-123",
        "variantData": {"priceFeedSymbol": "BTCUSDT"},
        "tags": [{"name": "5 Min"}],
        "endsAt": "2099-01-01T00:00:00.000Z",
        "markets": [{"id": 2199999, "title": "BTC 5m"}],
    }

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/v1/categories"):
            return httpx.Response(200, json={"success": True, "data": [cat], "cursor": None})
        if req.url.path.endswith("/orderbook"):
            return httpx.Response(200, json={"success": True, "data": {"marketId": 2199999, "bids": [[0.51, 100]], "asks": [[0.53, 100]], "updateTimestampMs": T0 + 1000}})
        return httpx.Response(404, json={})

    transport = httpx.MockTransport(handler)
    rest_client = PredictFunClient("test_key", transport=transport, base_url="https://api.predict.fun")
    store = CollectorStore(memory_only=True)
    coll = HistoricalCollector(store, stale_threshold_ms=5000, prediction_stale_threshold_ms=300000, resolution_by_market={2199999: 4_000_000_000_000})

    # Mock WS that immediately fails then recovers
    class FakeWS:
        def __init__(self, fail_first=True):
            self.fail_first = fail_first
            self.sent = []
            self.closed = False

        async def send(self, msg):
            self.sent.append(msg)

        async def close(self):
            self.closed = True

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.fail_first:
                self.fail_first = False
                raise RuntimeError("simulated disconnect")
            # After reconnect, send one snapshot
            await asyncio.sleep(0.01)
            raise StopAsyncIteration

    # Track connects
    connects = []

    async def fake_connect(*args, **kwargs):
        ws = FakeWS()
        connects.append(ws)
        return ws

    ws_client = PredictFunWSClient("test_key", coll, rest_client=rest_client, ws_url="wss://ws.predict.fun/ws", connect_factory=fake_connect)
    stop = asyncio.Event()

    # Run for a short time then stop
    task = asyncio.create_task(ws_client.run(stop))
    await asyncio.sleep(0.15)
    stop.set()
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except asyncio.TimeoutError:
        task.cancel()

    # Should have attempted reconnect and incremented counter
    assert coll.stats.reconnects >= 1
    await rest_client.close()


@pytest.mark.asyncio
async def test_spot_ws_reconnect_and_sequence_gap():
    store = CollectorStore(memory_only=True)
    coll = HistoricalCollector(store, stale_threshold_ms=5000, prediction_stale_threshold_ms=300000)
    # Simulate spot WS messages with gap
    msgs = [
        {"e": "bookTicker", "E": T0 + 1000, "s": "BTCUSDT", "b": "68000", "a": "68001", "u": 1},
        {"e": "bookTicker", "E": T0 + 1100, "s": "BTCUSDT", "b": "68001", "a": "68002", "u": 3},  # gap 1->3
    ]
    for m in msgs:
        parsed = parse_spot_ws_message(m)
        captured = int(m["E"]) + 5
        coll.ingest_spot(parsed, captured_at_ms=captured)
    assert coll.stats.gaps_detected == 1
    assert store.count() == 2


def test_predict_ws_snapshot_vs_update_handling():
    """Snapshot (full book) and update (delta) both handled, timestamps preserved."""
    store = CollectorStore(memory_only=True)
    coll = HistoricalCollector(store, resolution_by_market={9001: RES_5M}, stale_threshold_ms=5000, prediction_stale_threshold_ms=300000)
    # Snapshot with full bids/asks
    snap = {
        "marketId": 9001,
        "tokenId": "tok_yes",
        "symbol": "BTCUSDT",
        "updateTimestampMs": T0 + 1000,
        "sequence": 100,
        "bids": [[0.51, 100], [0.50, 200]],
        "asks": [[0.53, 150]],
        "duration": "5m",
        "resolution_ms": RES_5M,
    }
    obs_snap = coll.ingest_prediction_orderbook(snap, captured_at_ms=T0 + 1050)
    assert obs_snap is not None
    assert obs_snap.depth == 2
    assert obs_snap.captured_at_ms == T0 + 1050
    assert obs_snap.update_ts_ms == T0 + 1000

    # Update with same timestamp but different book (valid, not duplicate due to content hash)
    upd = {
        "marketId": 9001,
        "tokenId": "tok_yes",
        "symbol": "BTCUSDT",
        "updateTimestampMs": T0 + 1000,  # same
        "sequence": None,  # no seq
        "bids": [[0.52, 110]],
        "asks": [[0.54, 160]],
        "duration": "5m",
        "resolution_ms": RES_5M,
    }
    obs_upd = coll.ingest_prediction_orderbook(upd, captured_at_ms=T0 + 1100)
    assert obs_upd is not None  # not discarded
    assert obs_upd.best_bid == Decimal("0.52")


def test_5m_and_15m_markets_available():
    """Verify 5m and 15m filtering preserves both durations."""
    from collections import Counter

    from app.research.prediction_markets.predict_ws import _filter_btc_eth_5m_15m

    cats = [
        {"slug": "btc-updown-5m-1", "variantData": {"priceFeedSymbol": "BTCUSDT"}, "tags": [{"name": "5 Min"}], "endsAt": "2026-09-11T12:00:00.000Z", "markets": [{"id": 10}]},
        {"slug": "btc-updown-15m-1", "variantData": {"priceFeedSymbol": "BTCUSDT"}, "tags": [{"name": "15 Min"}], "endsAt": "2026-09-11T12:00:00.000Z", "markets": [{"id": 11}]},
        {"slug": "eth-updown-5m-1", "variantData": {"priceFeedSymbol": "ETHUSDT"}, "tags": [{"name": "5 Min"}], "endsAt": "2026-09-11T12:00:00.000Z", "markets": [{"id": 12}]},
        {"slug": "eth-updown-15m-1", "variantData": {"priceFeedSymbol": "ETHUSDT"}, "tags": [{"name": "15 Min"}], "endsAt": "2026-09-11T12:00:00.000Z", "markets": [{"id": 13}]},
    ]
    filtered = _filter_btc_eth_5m_15m(cats)  # type: ignore[arg-type]
    durs = Counter(f["duration"] for f in filtered)
    syms = Counter(f["symbol"] for f in filtered)
    assert durs["5m"] == 2
    assert durs["15m"] == 2
    assert syms["BTCUSDT"] == 2
    assert syms["ETHUSDT"] == 2
