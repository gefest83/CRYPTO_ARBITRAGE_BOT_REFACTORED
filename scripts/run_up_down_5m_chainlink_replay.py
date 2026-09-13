"""Deterministic end-to-end Chainlink replay for one BTC Up/Down 5m market.

Default (fixture): fully offline and deterministic. Replays market
``--market-id`` (default 10918165) over a FIXTURE trace of Chainlink v3
report mids across its 5m window: start price -> final 5m close -> attach
(CHAINLINK provenance) -> binary settlement -> paper replay PnL.
The fixture is clearly labeled and proves the pipeline, not the market.

Live (--live): attempts the real path — one bounded Chainlink
``page_reports`` over [start-60s, end] plus SAPI books — and fails closed
with an exact blocker message when credentials/data are unavailable.
Read-only; never quotes, never trades, never prints secrets.

Spot is never a settlement substitute: settlement goes exclusively through
``settle_from_chainlink`` (CHAINLINK provenance or refusal).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from decimal import Decimal

T0 = 1_748_131_200_000  # fixture window start (labeled FIXTURE, not live)
T1 = T0 + 300_000


def _fixture_reports(market_id: int):  # type: ignore[no-untyped-def]
    """Deterministic Chainlink-shaped trace: 68000 -> 68100 (UP), 5s grid."""
    from app.research.prediction_markets.up_down_5m.chainlink_feed import ChainlinkReport

    reports: list[ChainlinkReport] = []
    steps = 72  # [T0-60s, T1-1s] at 5s
    for i in range(steps):
        ts = T0 - 60_000 + i * 5_000
        frac = i / (steps - 1)
        mid = Decimal("68000") + Decimal("100") * Decimal(str(frac))
        reports.append(ChainlinkReport(
            feed_id="fixture-chainlink-btc-usd-v3",
            observations_ts_ms=ts,
            valid_from_ts_ms=ts - 1000,
            bid=mid - Decimal("1"),
            ask=mid + Decimal("1"),
            benchmark=mid,
        ))
    return reports, market_id


def _fixture_books() -> dict:
    return {
        "tok_yes": {"tokenId": "tok_yes", "bids": [{"price": "0.51", "size": "5000"}], "asks": [{"price": "0.53", "size": "3000"}]},
        "tok_no": {"tokenId": "tok_no", "bids": [{"price": "0.47", "size": "4000"}], "asks": [{"price": "0.49", "size": "3500"}]},
    }


def _fixture_topic(market_id: int) -> dict:
    return {
        "marketTopicId": 1001, "vendor": "PREDICT_FUN", "chainId": "56",
        "slug": "btc-price-5m-up-or-down", "title": "BTCUSDT Price 5m Up or Down?",
        "symbol": "BTCUSDT", "startDate": T0, "endDate": T1, "status": "REGISTERED",
        "markets": [{
            "marketId": market_id, "title": "UP", "tradingStatus": "CLOSED",
            "outcomes": [
                {"name": "YES", "price": "0.52", "tokenId": "tok_yes"},
                {"name": "NO", "price": "0.48", "tokenId": "tok_no"},
            ],
        }],
    }


def run_fixture(market_id: int, entry_side: str, entry_size: str) -> int:
    from app.research.prediction_markets.normalizer import normalize_market_topic
    from app.research.prediction_markets.up_down_5m.chainlink_feed import (
        attach_chainlink_resolution,
        resolve_market,
    )
    from app.research.prediction_markets.up_down_5m.live_snapshot import (
        PriceProvenance,
        build_live_snapshot,
        render_snapshot,
        settle_from_chainlink,
    )
    from app.research.prediction_markets.up_down_5m.replay import (
        ReplayDecision,
        UpDown5mReplay,
    )

    print(f"market_id={market_id} mode=FIXTURE (offline, deterministic; NOT live data)")
    reports, _ = _fixture_reports(market_id)
    nm = normalize_market_topic(_fixture_topic(market_id), orderbooks=_fixture_books())
    assert nm is not None
    resolution = resolve_market(reports, market_id, T0, T1, feed_id="fixture-chainlink-btc-usd-v3")
    base = build_live_snapshot(nm, checked_at_ms=T1).market
    market = attach_chainlink_resolution(base, resolution)
    snapshot = build_live_snapshot(
        nm,
        chainlink_start=resolution.start_price,
        chainlink_current=resolution.end_price,
        chainlink_end=resolution.end_price,
        checked_at_ms=T1,
    ).model_copy(update={
        "market": market,
        "start_provenance": PriceProvenance.CHAINLINK,
        "current_provenance": PriceProvenance.CHAINLINK,
        "end_provenance": PriceProvenance.CHAINLINK,
    })
    outcome = settle_from_chainlink(snapshot)
    print(f"chainlink start={resolution.start_price} (report@{resolution.start_report_ts_ms}) "
          f"final_close={resolution.end_price} (report@{resolution.end_report_ts_ms}) outcome={outcome.value}")
    entry_price = market.up_ask if entry_side == "UP" else market.down_ask
    decision = ReplayDecision(
        market_id=market_id, side=entry_side, entry_price=entry_price,
        size=Decimal(entry_size), entry_ts_ms=T0 + 1000,
    )
    summary = UpDown5mReplay().run([market], [decision])
    fill = summary.fills[0]
    print(f"replay: side={fill.side} entry={fill.entry_price} size={fill.size} "
          f"payout={fill.payout} fee={fill.fee} net={fill.net} win={fill.win}")
    print(render_snapshot(snapshot, T1))
    print("trading: none (research-only fail-closed)")
    return 0


async def run_live(args: argparse.Namespace) -> int:
    from app.research.prediction_markets.up_down_5m.chainlink_feed import (
        BTC_USD_CEX_V3_FEED_ID,
        ChainlinkFeedClient,
        resolve_chainlink_credentials,
        resolve_market,
    )

    creds = resolve_chainlink_credentials()
    if creds is None:
        print("BLOCKER: no Chainlink Data Streams credentials "
              "(CHAINLINK_DATA_STREAMS_KEY/SECRET). Report endpoints require HMAC auth; "
              "unauthenticated calls return 400 missing UserId/Timestamp/HmacSignature. "
              "Live historical Chainlink data is inaccessible from this environment.")
        return 2
    if args.start_ms is None or args.end_ms is None:
        print("BLOCKER: live mode needs --start-ms/--end-ms (market window is unknown: "
              "Binance SAPI key is rejected with -2008 Invalid Api-Key ID and "
              "Predict.fun returns 404 for this market id).")
        return 2
    key, secret = creds
    try:
        async with ChainlinkFeedClient(key, secret) as client:
            start_sec = (int(args.start_ms) - 60_000) // 1000
            limit = max(1, min(500, (int(args.end_ms) - int(args.start_ms)) // 1000 + 30))
            reports = await client.page_reports(BTC_USD_CEX_V3_FEED_ID, start_sec, limit=limit)
    except Exception as exc:
        print(f"BLOCKER: live Chainlink fetch failed (fail-closed): {str(exc)[:300]}")
        return 3
    try:
        resolution = resolve_market(reports, int(args.market_id), int(args.start_ms), int(args.end_ms))
    except ValueError as exc:
        print(f"BLOCKER: resolution inputs incomplete: {exc}")
        return 3
    print(f"market_id={args.market_id} mode=LIVE start={resolution.start_price} "
          f"final_close={resolution.end_price}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic Chainlink replay for one BTC Up/Down 5m market (no trading)")
    ap.add_argument("--market-id", default="10918165")
    ap.add_argument("--entry-side", default="UP", choices=["UP", "DOWN"])
    ap.add_argument("--entry-size", default="1")
    ap.add_argument("--live", action="store_true", help="attempt real Chainlink+SAPI path (needs creds + window)")
    ap.add_argument("--start-ms", default=None, help="live mode: market start ms")
    ap.add_argument("--end-ms", default=None, help="live mode: market end ms")
    args = ap.parse_args(argv)
    if args.live:
        return asyncio.run(run_live(args))
    return run_fixture(int(args.market_id), args.entry_side, args.entry_size)


if __name__ == "__main__":
    sys.exit(main())
