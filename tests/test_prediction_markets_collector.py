"""Phase 2 — Historical collector deterministic tests (mocked WS/REST)."""

from __future__ import annotations

import json
import pathlib
from decimal import Decimal

import httpx
import pytest

from app.research.prediction_markets.client import BinancePredictionClient
from app.research.prediction_markets.collector.collector import HistoricalCollector
from app.research.prediction_markets.collector.models import SourceKind
from app.research.prediction_markets.collector.storage import CollectorStore

# fixed times for determinism
T0 = 1_748_131_200_000  # start
RES_5M = T0 + 300_000
RES_15M = T0 + 900_000


def _spot(bid="68000.1", ask="68000.5", ts=T0 + 1_000, seq=1, symbol="BTCUSDT"):
    return {"symbol": symbol, "bid": bid, "ask": ask, "bid_qty": "0.5", "ask_qty": "0.6", "exchange_ts_ms": ts, "sequence": seq}


def _pred_ob(market_id=9001, token="tok_btc5_yes", ts=T0 + 1_200, seq=10, symbol="BTCUSDT", resolution=RES_5M, bids=None, asks=None):
    return {
        "marketId": market_id,
        "tokenId": token,
        "symbol": symbol,
        "updateTimestampMs": ts,
        "sequence": seq,
        "resolution_ms": resolution,
        "bids": bids or [["0.51", "5000"], ["0.50", "1200"]],
        "asks": asks or [["0.52", "3000"], ["0.53", "7500"]],
        "outcome": "YES",
        "duration": "5m",
    }


def _pred_trade(market_id=9001, token="tok_btc5_yes", ts=T0 + 1_500, price="0.52", symbol="BTCUSDT"):
    return {"marketId": market_id, "tokenId": token, "symbol": symbol, "timestamp": ts, "price": price, "resolution_ms": RES_5M}


def test_collector_records_all_required_fields(tmp_path: pathlib.Path) -> None:
    store = CollectorStore(base_dir=tmp_path / "data", memory_only=True)
    coll = HistoricalCollector(store, resolution_by_market={9001: RES_5M, 9002: RES_15M})

    spot = coll.ingest_spot(_spot(), captured_at_ms=T0 + 1_050)
    assert spot is not None
    assert spot.market_id is None  # spot has no marketId
    assert spot.symbol == "BTCUSDT"
    assert spot.exchange_ts_ms == T0 + 1_000
    assert spot.spread == Decimal("0.4")
    assert spot.spread_bps is not None
    assert spot.source == SourceKind.SPOT_ORDERBOOK

    ob = coll.ingest_prediction_orderbook(_pred_ob(), captured_at_ms=T0 + 1_250)
    assert ob is not None
    assert ob.market_id == 9001
    assert ob.token_id == "tok_btc5_yes"
    assert ob.exchange_ts_ms == T0 + 1_200
    assert ob.update_ts_ms == T0 + 1_200
    assert ob.time_to_resolution_ms == RES_5M - (T0 + 1_200)
    assert ob.spread == Decimal("0.01")
    assert ob.depth == 2
    assert ob.best_bid == Decimal("0.51")
    assert ob.best_ask == Decimal("0.52")

    trade = coll.ingest_prediction_trade(_pred_trade(), captured_at_ms=T0 + 1_550)
    assert trade is not None
    assert trade.market_id == 9001
    assert trade.price == Decimal("0.52")
    assert trade.time_to_resolution_ms == RES_5M - (T0 + 1_500)

    # storage has 3
    assert store.count() == 3
    rows = store.query(limit=10, order="exchange_ts_ms")
    assert len(rows) == 3
    # deterministic ordering by exchange_ts_ms
    assert rows[0]["exchange_ts_ms"] == T0 + 1_000
    assert rows[1]["exchange_ts_ms"] == T0 + 1_200


def test_bnb_excluded_everywhere(tmp_path: pathlib.Path) -> None:
    store = CollectorStore(base_dir=tmp_path / "data", memory_only=True)
    coll = HistoricalCollector(store)
    assert coll.ingest_spot(_spot(symbol="BNBUSDT"), captured_at_ms=T0 + 1_000) is None
    assert coll.ingest_prediction_orderbook(_pred_ob(symbol="BNBUSDT"), captured_at_ms=T0 + 1_200) is None
    assert coll.ingest_prediction_trade(_pred_trade(symbol="BNBUSDT"), captured_at_ms=T0 + 1_500) is None
    assert store.count() == 0


def test_duplicate_dropped(tmp_path: pathlib.Path) -> None:
    store = CollectorStore(base_dir=tmp_path / "data", memory_only=True)
    coll = HistoricalCollector(store, resolution_by_market={9001: RES_5M})
    data = _pred_ob(ts=T0 + 1000, seq=1)
    assert coll.ingest_prediction_orderbook(data, captured_at_ms=T0 + 1100) is not None
    assert coll.ingest_prediction_orderbook(data, captured_at_ms=T0 + 1100) is None  # duplicate
    assert coll.stats.duplicates_dropped == 1
    assert store.count() == 1

    spot_data = _spot(ts=T0 + 2000, seq=5)
    assert coll.ingest_spot(spot_data, captured_at_ms=T0 + 2100) is not None
    assert coll.ingest_spot(spot_data, captured_at_ms=T0 + 2100) is None
    assert coll.stats.duplicates_dropped == 2


def test_out_of_order_dropped(tmp_path: pathlib.Path) -> None:
    store = CollectorStore(base_dir=tmp_path / "data", memory_only=True)
    coll = HistoricalCollector(store, resolution_by_market={9001: RES_5M}, out_of_order_tolerance_ms=1000)
    # watermark = 5000
    coll.ingest_prediction_orderbook(_pred_ob(ts=T0 + 5000, seq=10), captured_at_ms=T0 + 5100)
    # 3000 is 2000 behind watermark -> dropped (tolerance 1000)
    assert coll.ingest_prediction_orderbook(_pred_ob(ts=T0 + 3000, seq=11), captured_at_ms=T0 + 5100) is None
    assert coll.stats.out_of_order_dropped == 1
    # within tolerance (4500) accepted
    assert coll.ingest_prediction_orderbook(_pred_ob(ts=T0 + 4500, seq=12), captured_at_ms=T0 + 5200) is not None


def test_stale_snapshot_dropped(tmp_path: pathlib.Path) -> None:
    store = CollectorStore(base_dir=tmp_path / "data", memory_only=True)
    coll = HistoricalCollector(store, stale_threshold_ms=5000, resolution_by_market={9001: RES_5M})
    # captured 10s after event -> stale
    assert coll.ingest_prediction_orderbook(_pred_ob(ts=T0 + 1000), captured_at_ms=T0 + 20_000) is None
    assert coll.stats.stale_dropped == 1
    # spot stale too
    assert coll.ingest_spot(_spot(ts=T0 + 1000), captured_at_ms=T0 + 20_000) is None
    assert coll.stats.stale_dropped == 2


def test_market_expiration_dropped(tmp_path: pathlib.Path) -> None:
    store = CollectorStore(base_dir=tmp_path / "data", memory_only=True)
    coll = HistoricalCollector(store, resolution_by_market={9001: RES_5M})
    # at resolution -> expired
    assert coll.ingest_prediction_orderbook(_pred_ob(ts=RES_5M, seq=1), captured_at_ms=RES_5M + 100) is None
    assert coll.ingest_prediction_orderbook(_pred_ob(ts=RES_5M + 1000, seq=2), captured_at_ms=RES_5M + 1100) is None
    assert coll.stats.expired_dropped == 2
    # before resolution ok
    assert coll.ingest_prediction_orderbook(_pred_ob(ts=RES_5M - 1, seq=3), captured_at_ms=RES_5M) is not None


def test_missing_sequence_gap_detected(tmp_path: pathlib.Path) -> None:
    store = CollectorStore(base_dir=tmp_path / "data", memory_only=True)
    coll = HistoricalCollector(store, resolution_by_market={9001: RES_5M})
    coll.ingest_prediction_orderbook(_pred_ob(ts=T0 + 1000, seq=1), captured_at_ms=T0 + 1100)
    coll.ingest_prediction_orderbook(_pred_ob(ts=T0 + 2000, seq=5), captured_at_ms=T0 + 2100)  # gap 1->5
    assert coll.stats.gaps_detected == 1
    coll.ingest_spot(_spot(ts=T0 + 3000, seq=10), captured_at_ms=T0 + 3100)
    coll.ingest_spot(_spot(ts=T0 + 4000, seq=12), captured_at_ms=T0 + 4100)  # gap
    assert coll.stats.gaps_detected == 2


def test_reconnect_and_rest_recovery(tmp_path: pathlib.Path) -> None:
    # mock REST via httpx MockTransport that returns lastTradePrice
    def handler(request: httpx.Request) -> httpx.Response:
        if "last-trade-price" in request.url.path:
            return httpx.Response(200, json={"marketId": 9001, "lastTradePrice": "0.55"})
        return httpx.Response(404, json={})

    transport = httpx.MockTransport(handler)
    store = CollectorStore(base_dir=tmp_path / "data", memory_only=True)
    client = BinancePredictionClient("k", "s", transport=transport)
    # use far-future resolution so not expired at test time (real now is 2026)
    future_res = 9_000_000_000_000
    coll = HistoricalCollector(store, client=client, resolution_by_market={9001: future_res})
    assert coll.stats.reconnects == 0
    coll.on_reconnect("ws closed")
    assert coll.stats.reconnects == 1

    # REST snapshot recovery
    import asyncio
    count = asyncio.run(coll.rest_snapshot_recovery())
    assert count == 1
    assert coll.stats.rest_snapshots == 1
    assert store.count() == 1
    # the snapshot is a trade observation
    rows = store.query(limit=5)
    assert rows[0]["source"] == "prediction_trade"


@pytest.mark.asyncio
async def test_mocked_websocket_stream_with_reconnect_and_ordering(tmp_path: pathlib.Path) -> None:
    """Simulate WS stream: messages, duplicate, out-of-order, gap, stale, reconnect."""
    store = CollectorStore(base_dir=tmp_path / "data", memory_only=True)
    coll = HistoricalCollector(store, resolution_by_market={9001: RES_5M}, stale_threshold_ms=5000, out_of_order_tolerance_ms=500)

    # simulated WS messages in order
    msgs = [
        _pred_ob(ts=T0 + 1000, seq=1),
        _pred_ob(ts=T0 + 1100, seq=2),
        _pred_ob(ts=T0 + 1100, seq=2),  # duplicate
        _pred_ob(ts=T0 + 1200, seq=4),  # gap (missing 3)
        _pred_ob(ts=T0 + 1150, seq=5),  # out-of-order but within 500? 1200-1150=50 <500 so accepted? watermark 1200, 1150 diff 50 <500 => accepted
        _pred_ob(ts=T0 + 600, seq=6),   # too old -> dropped
        _spot(ts=T0 + 1050, seq=10, bid="68001", ask="68001.5"),
    ]
    for m in msgs:
        if "marketId" in m:
            coll.ingest_prediction_orderbook(m, captured_at_ms=m["updateTimestampMs"] + 50)
        else:
            coll.ingest_spot(m, captured_at_ms=m["exchange_ts_ms"] + 50)

    # simulate reconnect
    coll.on_reconnect()
    assert coll.stats.reconnects == 1
    assert coll.stats.duplicates_dropped == 1
    assert coll.stats.gaps_detected >= 1
    assert coll.stats.out_of_order_dropped == 1
    # total accepted: seq1, seq2, seq4, seq5 (1150 within tol), spot =5
    assert store.count() == 5


def test_storage_jsonl_and_sqlite_format(tmp_path: pathlib.Path) -> None:
    base = tmp_path / "research"
    store = CollectorStore(base_dir=base, memory_only=False)
    coll = HistoricalCollector(store, resolution_by_market={9001: RES_5M})
    coll.ingest_spot(_spot(ts=T0 + 1000, seq=1), captured_at_ms=T0 + 1005)
    coll.ingest_prediction_orderbook(_pred_ob(ts=T0 + 1100, seq=2), captured_at_ms=T0 + 1105)
    # JSONL file exists
    day = "2025-05-17"  # from T0? T0 is 2025-05-26 approx; compute dynamically
    import datetime

    expected_day = datetime.datetime.fromtimestamp((T0 + 1005) / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
    jpath = base / "observations" / f"{expected_day}.jsonl"
    assert jpath.exists()
    lines = jpath.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["symbol"] == "BTCUSDT"
    assert "spread" in rec or "spread_bps" in rec
    # SQLite queryable
    rows = store.query(symbol="BTCUSDT", limit=10)
    assert len(rows) == 2
    # deterministic ordering
    rows_ts_sorted = store.query(order="exchange_ts_ms", limit=10)
    assert rows_ts_sorted[0]["exchange_ts_ms"] < rows_ts_sorted[1]["exchange_ts_ms"]
    # schema has expected columns
    assert "market_id" in rows[0]
    assert "token_id" in rows[0]
    assert "time_to_resolution_ms" in rows[0]
    assert "raw_json" in rows[0]
    store.close()


def test_synchronization_resolution(tmp_path: pathlib.Path) -> None:
    store = CollectorStore(base_dir=tmp_path / "data", memory_only=True)
    coll = HistoricalCollector(store, resolution_by_market={9001: RES_5M}, sync_window_ms=500)
    # spot at 1000, pred at 1200 -> delta 200 within window
    coll.ingest_spot(_spot(ts=T0 + 1000, seq=1), captured_at_ms=T0 + 1010)
    pred = coll.ingest_prediction_orderbook(_pred_ob(ts=T0 + 1200, seq=2), captured_at_ms=T0 + 1210)
    assert pred is not None
    sync = coll.synchronized_view(pred)
    assert sync is not None
    assert sync.delta_ms == 200
    assert sync.spot.exchange_ts_ms == T0 + 1000
    assert sync.prediction.update_ts_ms == T0 + 1200
    assert sync.time_to_resolution_ms == RES_5M - (T0 + 1200)
    # outside window -> no sync
    pred2 = coll.ingest_prediction_orderbook(_pred_ob(ts=T0 + 5000, seq=3), captured_at_ms=T0 + 5010)
    # spot still at 1000, delta 4000 >500
    assert coll.synchronized_view(pred2) is None
    # after new spot, sync works again
    coll.ingest_spot(_spot(ts=T0 + 4900, seq=2), captured_at_ms=T0 + 4910)
    assert coll.synchronized_view(pred2) is not None


def test_five_and_15m_durations_and_eth(tmp_path: pathlib.Path) -> None:
    store = CollectorStore(base_dir=tmp_path / "data", memory_only=True)
    coll = HistoricalCollector(store, resolution_by_market={9001: RES_5M, 9002: RES_15M})
    # BTC 5m
    assert coll.ingest_prediction_orderbook(_pred_ob(market_id=9001, ts=T0 + 1000, seq=1, symbol="BTCUSDT", resolution=RES_5M), captured_at_ms=T0 + 1010) is not None
    # BTC 15m (uses ETH? ensure both)
    assert coll.ingest_prediction_orderbook(_pred_ob(market_id=9002, ts=T0 + 1000, seq=1, symbol="BTCUSDT", resolution=RES_15M), captured_at_ms=T0 + 1010) is not None
    # ETH 5m
    assert coll.ingest_prediction_orderbook(_pred_ob(market_id=9001, ts=T0 + 1000, seq=1, symbol="ETHUSDT", resolution=RES_5M, token="tok_eth5_yes"), captured_at_ms=T0 + 1010) is not None
    eth_spot = coll.ingest_spot(_spot(symbol="ETHUSDT", ts=T0 + 1000, seq=1), captured_at_ms=T0 + 1010)
    assert eth_spot is not None
    assert eth_spot.symbol == "ETHUSDT"


def test_isolation_no_forbidden_imports() -> None:
    import pathlib
    root = pathlib.Path("app/research/prediction_markets/collector")
    for p in root.glob("*.py"):
        text = p.read_text(encoding="utf-8").lower()
        for kw in ["from app.strategies.triangular", "from app.strategies.transfer", "from app.execution", "from app.risk", "from app.recovery", "from app.telegram", "from app.agent", "from app.strategies.kronos"]:
            assert kw not in text, f"{p.name} imports forbidden {kw}"
