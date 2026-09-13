"""Research-only diagnostic: print ONE BTC Up/Down 5m lifecycle snapshot.

Default (mock): fully offline, deterministic, no credentials, no network.
Prints one complete market lifecycle/state snapshot wired through the real
SAPI client + discovery + normalizer code paths (httpx MockTransport).

Live (--live): attempts ONE real SAPI read (market/list + detail +
order-book for a single BTC 5m market) with a short timeout. Read-only;
never quotes, never trades. Requires Binance SAPI key/secret in env/.env
(same keys as the research runtime). Any failure prints a fail-closed
blocker message and exits non-zero. Never prints secrets.

Spot reference mids (--spot-start/--spot-now) are labeled SPOT_REFERENCE
(display/timing only, never settlement). Real Chainlink mids
(--chainlink-start/--chainlink-current/--chainlink-end) are labeled
CHAINLINK_VENUE.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import httpx

T0 = 1_748_131_200_000
T1 = T0 + 300_000

MOCK_TOPIC = {
    "marketTopicId": 1001,
    "vendor": "PREDICT_FUN",
    "chainId": "56",
    "slug": "btc-price-5m-up-or-down",
    "title": "BTCUSDT Price 5m Up or Down?",
    "question": "Will BTCUSDT go UP?",
    "symbol": "BTCUSDT",
    "participantCount": 1000,
    "collateral": "USDT",
    "feeRateBps": 200,
    "slippageBps": 1200,
    "liquidity": "45000.00",
    "tradeVolume": "150000.00",
    "publishedAt": T0 - 60_000,
    "startDate": T0,
    "endDate": T1,
    "status": "REGISTERED",
    "markets": [
        {
            "marketId": 9001,
            "externalId": "ext_9001",
            "title": "UP",
            "status": "REGISTERED",
            "tradingStatus": "OPEN",
            "liquidity": "25000.00",
            "tradeVolume": "90000.00",
            "outcomes": [
                {"name": "YES", "price": "0.52", "chance": "0.52", "index": 0, "tokenId": "tok_btc5_yes"},
                {"name": "NO", "price": "0.48", "chance": "0.48", "index": 1, "tokenId": "tok_btc5_no"},
            ],
        }
    ],
}

MOCK_BOOKS = {
    "tok_btc5_yes": {
        "tokenId": "tok_btc5_yes",
        "timestamp": T0 + 200_000,
        "bids": [{"price": "0.51", "size": "5000"}],
        "asks": [{"price": "0.53", "size": "3000"}],
    },
    "tok_btc5_no": {
        "tokenId": "tok_btc5_no",
        "timestamp": T0 + 200_000,
        "bids": [{"price": "0.46", "size": "4000"}],
        "asks": [{"price": "0.48", "size": "3500"}],
    },
}


def _mock_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    params = dict(request.url.params)
    if path.endswith("/market/list"):
        return httpx.Response(200, json={"marketTopics": [MOCK_TOPIC], "total": 1})
    if path.endswith("/market/search"):
        return httpx.Response(200, json=[MOCK_TOPIC])
    if path.endswith("/market/detail"):
        return httpx.Response(200, json=MOCK_TOPIC)
    if path.endswith("/order-book") and not path.endswith("last-trade-price"):
        tok = params.get("tokenId", "")
        return httpx.Response(200, json=MOCK_BOOKS.get(tok, MOCK_BOOKS["tok_btc5_yes"]))
    if path.endswith("/last-trade-price"):
        return httpx.Response(200, json={"marketId": 9001, "lastTradePrice": "0.52"})
    if path.endswith("/category/list"):
        return httpx.Response(200, json={"categories": [{"id": "crypto", "name": "Crypto"}]})
    return httpx.Response(404, json={"msg": "not mocked"})


def _resolve_sapi_credentials() -> tuple[str, str] | None:
    try:
        from app.exchanges.credentials import EnvCredentialsProvider

        creds = EnvCredentialsProvider().get("binance")
        if creds is not None and creds.api_key and creds.secret:
            return creds.api_key.strip(), creds.secret.strip()
    except Exception:
        pass
    for k_env, s_env in (
        ("CAT_RESEARCH__BINANCE_APIKEY", "CAT_RESEARCH__BINANCE_SECRET"),
        ("CAT_KEY_BINANCE_APIKEY", "CAT_KEY_BINANCE_SECRET"),
        ("BINANCE_API_KEY", "BINANCE_API_SECRET"),
    ):
        k, s = os.getenv(k_env), os.getenv(s_env)
        if k and s and k.strip() and s.strip():
            return k.strip(), s.strip()
    return None


async def _run(args: argparse.Namespace) -> int:
    from app.research.prediction_markets.client import BinancePredictionClient
    from app.research.prediction_markets.up_down_5m.live_snapshot import (
        fetch_one_btc_5m_snapshot,
        render_snapshot,
    )

    print("mode=" + ("live (read-only SAPI)" if args.live else "mock (offline, deterministic)"))
    print("research-only: no quotes, no orders, no fund movement")
    now_ms = int(time.time() * 1000)

    if args.live:
        creds = _resolve_sapi_credentials()
        if creds is None:
            print("BLOCKER: no Binance SAPI credentials found (CAT_KEY_BINANCE_APIKEY/SECRET).")
            print("SAPI market data is signed even for reads; cannot fetch live without them.")
            return 2
        api_key, api_secret = creds
        try:
            async with BinancePredictionClient(api_key, api_secret, timeout=10.0) as client:
                snapshot = await fetch_one_btc_5m_snapshot(
                    client,
                    chainlink_start=args.chainlink_start,
                    chainlink_current=args.chainlink_current,
                    chainlink_end=args.chainlink_end,
                    spot_mid_at_start=args.spot_start,
                    spot_mid_now=args.spot_now,
                    checked_at_ms=now_ms,
                )
        except Exception as exc:
            print(f"BLOCKER: live SAPI fetch failed (fail-closed): {str(exc)[:300]}")
            return 3
    else:
        transport = httpx.MockTransport(_mock_handler)
        async with BinancePredictionClient("mock-key", "mock-secret", transport=transport) as client:
            snapshot = await fetch_one_btc_5m_snapshot(
                client,
                chainlink_start=args.chainlink_start,
                chainlink_current=args.chainlink_current,
                chainlink_end=args.chainlink_end,
                spot_mid_at_start=args.spot_start,
                spot_mid_now=args.spot_now,
                checked_at_ms=T0 + 200_000,  # deterministic mid-window view
            )
            now_ms = T0 + 200_000

    print(render_snapshot(snapshot, now_ms))
    print("trading: none (research-only fail-closed)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Research-only BTC Up/Down 5m lifecycle snapshot (no trading)")
    ap.add_argument("--live", action="store_true", help="attempt one real read-only SAPI fetch (needs SAPI creds)")
    ap.add_argument("--spot-start", default=None, help="Binance spot ToB mid at market start (reference only)")
    ap.add_argument("--spot-now", default=None, help="Binance spot ToB mid now (reference only)")
    ap.add_argument("--chainlink-start", default=None, help="real Chainlink ToB mid at market start")
    ap.add_argument("--chainlink-current", default=None, help="real Chainlink ToB mid now")
    ap.add_argument("--chainlink-end", default=None, help="real Chainlink final close (5m candle before end)")
    args = ap.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
