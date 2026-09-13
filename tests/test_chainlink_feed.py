"""Chainlink resolution feed tests — auth, v3 decode, resolution math (mocked).

All offline via httpx MockTransport + synthetic v3 blobs. No network,
no credentials, no trading.
"""

from __future__ import annotations

import hashlib
import hmac
from decimal import Decimal

import httpx
import pytest

from app.research.prediction_markets.up_down_5m.chainlink_feed import (
    BTC_USD_CEX_V3_FEED_ID,
    ChainlinkFeedClient,
    ChainlinkFeedError,
    ChainlinkReport,
    attach_chainlink_resolution,
    build_5m_candles,
    build_auth_headers,
    decode_v3_report,
    final_close,
    market_start_price,
    resolve_market,
    tob_mid,
)
from app.research.prediction_markets.up_down_5m.live_snapshot import (
    PriceProvenance,
    build_live_snapshot,
    settle_from_chainlink,
)
from app.research.prediction_markets.up_down_5m.market import UpDown5mMarket
from app.research.prediction_markets.up_down_5m.settlement import SettlementOutcome

T0 = 1_748_131_200_000
T1 = T0 + 300_000
SCALE = 10**18


def _word(value: int) -> bytes:
    return int(value).to_bytes(32, byteorder="big", signed=True)


def _blob(price: str, bid: str, ask: str) -> str:
    words = [
        _word(0),  # feedId placeholder
        _word(T0 // 1000),  # validFrom
        _word(T0 // 1000),  # observations
        _word(0),  # nativeFee
        _word(0),  # linkFee
        _word(T0 // 1000 + 3600),  # expiresAt
        _word(int(Decimal(price) * SCALE)),
        _word(int(Decimal(bid) * SCALE)),
        _word(int(Decimal(ask) * SCALE)),
    ]
    return "0x" + b"".join(words).hex()


def _envelope(price: str, bid: str, ask: str, obs_sec: int) -> dict:
    return {
        "feedID": BTC_USD_CEX_V3_FEED_ID,
        "validFromTimestamp": str(obs_sec - 1),
        "observationsTimestamp": str(obs_sec),
        "fullReport": _blob(price, bid, ask),
    }


def _report(price: str, ts_ms: int, spread: str = "2") -> ChainlinkReport:
    bid = Decimal(price) - Decimal(spread)
    ask = Decimal(price) + Decimal(spread)
    rep = decode_v3_report(_envelope(price, str(bid), str(ask), ts_ms // 1000))
    return rep.model_copy(update={"observations_ts_ms": ts_ms, "valid_from_ts_ms": ts_ms - 1000})


# ---------- auth ----------

def test_auth_headers_match_official_scheme() -> None:
    hdrs = build_auth_headers("key-uuid", "secret", "GET", "/api/v1/reports/latest?feedID=0xabc", timestamp_ms=1716211845123)
    assert hdrs["Authorization"] == "key-uuid"
    assert hdrs["X-Authorization-Timestamp"] == "1716211845123"
    body_hash = hashlib.sha256(b"").hexdigest()
    expected = hmac.new(
        b"secret",
        f"GET /api/v1/reports/latest?feedID=0xabc {body_hash} key-uuid 1716211845123".encode(),
        hashlib.sha256,
    ).hexdigest()
    assert hdrs["X-Authorization-Signature-SHA256"] == expected


# ---------- v3 decode ----------

def test_decode_v3_blob_bid_ask_benchmark() -> None:
    rep = decode_v3_report(_envelope("68000.5", "68000", "68001", T0 // 1000))
    assert rep.feed_id == BTC_USD_CEX_V3_FEED_ID
    assert rep.bid == Decimal("68000")
    assert rep.ask == Decimal("68001")
    assert rep.benchmark == Decimal("68000.5")
    assert rep.mid == Decimal("68000.5")
    assert rep.observations_ts_ms == (T0 // 1000) * 1000


def test_decode_rejects_inconsistent_book() -> None:
    with pytest.raises(ChainlinkFeedError):
        decode_v3_report(_envelope("68000", "68002", "68001", T0 // 1000))  # bid > ask


def test_decode_rejects_short_blob() -> None:
    env = _envelope("68000", "67999", "68001", T0 // 1000)
    env["fullReport"] = "0x1234"
    with pytest.raises(ChainlinkFeedError):
        decode_v3_report(env)


# ---------- resolution math ----------

def test_tob_mid_exact_decimal() -> None:
    assert tob_mid(Decimal("68000"), Decimal("68002")) == Decimal("68001")
    with pytest.raises(ValueError):
        tob_mid(Decimal("0"), Decimal("1"))


def test_start_price_is_latest_at_or_before_start() -> None:
    reps = [_report("67900", T0 - 60_000), _report("68000", T0 - 5_000), _report("68100", T0 + 5_000)]
    assert market_start_price(reps, T0).mid == Decimal("68000")


def test_start_price_fail_closed_without_cover() -> None:
    with pytest.raises(ValueError):
        market_start_price([_report("68100", T0 + 5_000)], T0)


def test_final_close_is_last_before_end() -> None:
    reps = [_report("68000", T1 - 10_000), _report("68100", T1 - 1_000), _report("68200", T1 + 1_000)]
    assert final_close(reps, T1).mid == Decimal("68100")


def test_final_close_fail_closed_without_cover() -> None:
    with pytest.raises(ValueError):
        final_close([_report("68200", T1 + 1_000)], T1)


def test_candles_close_feeds_final() -> None:
    reps = [_report("68000", T1 - 120_000), _report("68050", T1 - 60_000), _report("68100", T1 - 1_000)]
    candles = build_5m_candles(reps)
    assert candles
    assert candles[-1].close == Decimal("68100")
    assert candles[-1].close_ms <= T1


# ---------- attach + provenance ----------

def _market_5m(mid: int = 10918165) -> UpDown5mMarket:
    return UpDown5mMarket(
        market_id=mid, start_ts_ms=T0, end_ts_ms=T1,
        up_bid=Decimal("0.51"), up_ask=Decimal("0.53"),
        down_bid=Decimal("0.47"), down_ask=Decimal("0.49"),
    )


def test_attach_binds_chainlink_to_market() -> None:
    reps = [_report("68000", T0 - 5_000), _report("68100", T1 - 1_000)]
    res = resolve_market(reps, 10918165, T0, T1)
    assert res.start_price == Decimal("68000") and res.end_price == Decimal("68100")
    m = attach_chainlink_resolution(_market_5m(), res)
    assert m.start_price == Decimal("68000") and m.end_price == Decimal("68100")
    assert m.settle().settlement == SettlementOutcome.UP


def test_attach_rejects_wrong_market() -> None:
    reps = [_report("68000", T0 - 5_000), _report("68100", T1 - 1_000)]
    res = resolve_market(reps, 10918165, T0, T1)
    with pytest.raises(ValueError, match="mismatch"):
        attach_chainlink_resolution(_market_5m(mid=999), res)


def test_settle_from_chainlink_ok_and_spot_refused() -> None:
    from app.research.prediction_markets.normalizer import normalize_market_topic

    topic = {
        "marketTopicId": 1001, "vendor": "PREDICT_FUN", "chainId": "56",
        "slug": "btc-price-5m-up-or-down", "title": "BTCUSDT Price 5m Up or Down?",
        "symbol": "BTCUSDT", "startDate": T0, "endDate": T1, "status": "REGISTERED",
        "markets": [{
            "marketId": 10918165, "title": "UP", "tradingStatus": "CLOSED",
            "outcomes": [
                {"name": "YES", "price": "0.52", "tokenId": "tok_yes"},
                {"name": "NO", "price": "0.48", "tokenId": "tok_no"},
            ],
        }],
    }
    books = {
        "tok_yes": {"tokenId": "tok_yes", "bids": [{"price": "0.51", "size": "1"}], "asks": [{"price": "0.53", "size": "1"}]},
        "tok_no": {"tokenId": "tok_no", "bids": [{"price": "0.47", "size": "1"}], "asks": [{"price": "0.49", "size": "1"}]},
    }
    nm = normalize_market_topic(topic, orderbooks=books)
    assert nm is not None

    # spot reference must never settle
    spot_snap = build_live_snapshot(nm, spot_mid_at_start="68000", spot_mid_now="68100", checked_at_ms=T1)
    with pytest.raises(ValueError, match="must never settle|refusing to settle"):
        settle_from_chainlink(spot_snap)

    # CHAINLINK provenance settles
    cl_snap = build_live_snapshot(
        nm, chainlink_start="68000", chainlink_current="68050", chainlink_end="68100", checked_at_ms=T1,
    ).model_copy(update={"start_provenance": PriceProvenance.CHAINLINK, "current_provenance": PriceProvenance.CHAINLINK, "end_provenance": PriceProvenance.CHAINLINK})
    assert settle_from_chainlink(cl_snap) == SettlementOutcome.UP


# ---------- mocked client ----------

def _mock_transport(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/discovery"):
        return httpx.Response(200, json={"feeds": [{
            "assetName": "Bitcoin", "baseAsset": "BTC", "quoteAsset": "USD",
            "attributeType": "CexPrice", "feedType": "Crypto", "status": "live",
            "schemaVersion": "V3", "feedId": BTC_USD_CEX_V3_FEED_ID,
        }]})
    if path.endswith("/reports/latest"):
        return httpx.Response(200, json={"report": _envelope("68000", "67999", "68001", T0 // 1000)})
    if path.endswith("/reports/page"):
        return httpx.Response(200, json={"reports": [
            _envelope("68000", "67999", "68001", (T0 - 5_000) // 1000),
            _envelope("68100", "68099", "68101", (T1 - 1_000) // 1000),
        ]})
    return httpx.Response(404, json={"msg": "not mocked"})


@pytest.mark.asyncio
async def test_client_discovery_and_decode_mocked() -> None:
    transport = httpx.MockTransport(_mock_transport)
    async with ChainlinkFeedClient("k", "s", transport=transport) as client:
        feeds = await client.discovery(feed_type="crypto")
        assert any(f["feedId"] == BTC_USD_CEX_V3_FEED_ID for f in feeds)
        latest = await client.get_latest_report(BTC_USD_CEX_V3_FEED_ID)
        assert latest.mid == Decimal("68000")
        page = await client.page_reports(BTC_USD_CEX_V3_FEED_ID, T0 // 1000, limit=2)
        assert [r.mid for r in page] == [Decimal("68000"), Decimal("68100")]


@pytest.mark.asyncio
async def test_client_reports_require_credentials() -> None:
    transport = httpx.MockTransport(_mock_transport)
    async with ChainlinkFeedClient(None, None, transport=transport) as client:
        with pytest.raises(ChainlinkFeedError, match="credentials absent"):
            await client.get_latest_report(BTC_USD_CEX_V3_FEED_ID)


@pytest.mark.asyncio
async def test_client_maps_server_auth_error() -> None:
    def _deny(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "missing auth headers"})

    transport = httpx.MockTransport(_deny)
    async with ChainlinkFeedClient("k", "s", transport=transport) as client:
        with pytest.raises(ChainlinkFeedError) as exc:
            await client.get_latest_report(BTC_USD_CEX_V3_FEED_ID)
        assert exc.value.status == 400


def test_research_only_isolation() -> None:
    import pathlib

    for p in (
        pathlib.Path("app/research/prediction_markets/up_down_5m/chainlink_feed.py"),
        pathlib.Path("scripts/run_up_down_5m_chainlink_replay.py"),
    ):
        text = p.read_text(encoding="utf-8").lower()
        for kw in ("from app.execution", "from app.risk", "from app.recovery", "from app.telegram", "from app.agent", "place_order(", "submit_order(", "batch_redeem("):
            assert kw not in text, f"{p.name} must stay research-only, found {kw!r}"
    from app.research.prediction_markets.up_down_5m.chainlink_feed import ChainlinkFeedClient as C

    with pytest.raises(RuntimeError):
        C("k", "s").ensure_research_only()
