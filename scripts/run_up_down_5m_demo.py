"""DEMO run for the frozen BTC Up/Down 5m drift signal (simulation only).

Mock (default): fully offline, deterministic. Simulates four DEMO lifecycles
(UP entry -> WIN, DOWN entry -> WIN, weak -> HOLD/SKIP, reversal -> EXIT)
through the real DEMO code path and prints structured DEMO logs.

Live (--live): ONE read-only SAPI fetch (market/list + detail + order-book)
plus operator-supplied spot reference mids. Still simulation-only: prints what
DEMO WOULD do. Never quotes, never trades. LIVE trading stays disabled.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from decimal import Decimal

T0 = 1_748_131_200_000
T1 = T0 + 300_000


def _mock_cases():  # type: ignore[no-untyped-def]
    from app.research.prediction_markets.up_down_5m.demo import DemoConfig, run_demo_market

    cfg = DemoConfig()
    cases = [
        # (market_id, spot_start, spot_entry, spot_exit, settlement)
        (9001, "68000", "68025", "68030", "UP"),  # +3.7bps UP, holds -> WIN
        (9002, "68000", "67975", "67970", "DOWN"),  # -3.7bps DOWN, holds -> WIN
        (9003, "68000", "68005", "68005", "UP"),  # +0.7bps weak -> SKIP
        (9004, "68000", "68025", "67970", "DOWN"),  # UP then reversal -> EXIT
    ]
    out = []
    for mid, s0, se, sx, st in cases:
        out.append(run_demo_market(
            market_id=mid, market_start_ms=T0, market_end_ms=T1,
            spot_start=Decimal(s0), spot_entry=Decimal(se), spot_exit=Decimal(sx),
            settlement=st, config=cfg,
        ))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BTC Up/Down 5m DEMO drift signal (simulation only, no trading)")
    ap.add_argument("--live", action="store_true", help="one read-only SAPI fetch, still simulation-only")
    ap.add_argument("--spot-start", default=None, help="spot reference mid at market start")
    ap.add_argument("--spot-now", default=None, help="spot reference mid now (entry check)")
    ap.add_argument("--spot-exit", default=None, help="spot reference mid at exit check")
    args = ap.parse_args(argv)

    from app.research.prediction_markets.up_down_5m.demo import DemoConfig, render_demo_log, run_demo_market

    print("mode=DEMO (simulation only, no orders, no fund movement)")
    print("research-only: live trading disabled (fail-closed)")

    if not args.live:
        for d in _mock_cases():
            print(render_demo_log(d))
        print("trading: none (demo simulation only)")
        return 0

    async def _live() -> int:
        import httpx

        from app.exchanges.credentials import EnvCredentialsProvider
        from app.research.prediction_markets.client import BinancePredictionClient
        from app.research.prediction_markets.up_down_5m.live_snapshot import fetch_one_btc_5m_snapshot

        creds = EnvCredentialsProvider().get("binance")
        if creds is None or not creds.api_key or not creds.secret:
            print("BLOCKER: no Binance SAPI credentials (DEMO read needs signed market data).")
            return 2
        if args.spot_start is None or args.spot_now is None:
            print("BLOCKER: --spot-start and --spot-now spot references required (Chainlink never a signal).")
            return 2
        now_ms = int(time.time() * 1000)
        try:
            async with BinancePredictionClient(creds.api_key.strip(), creds.secret.strip(), timeout=10.0) as client:
                snap = await fetch_one_btc_5m_snapshot(
                    client, spot_mid_at_start=args.spot_start, spot_mid_now=args.spot_now,
                    checked_at_ms=now_ms,
                )
        except Exception as exc:
            print(f"BLOCKER: live SAPI fetch failed (fail-closed): {str(exc)[:300]}")
            return 3
        m = snap.market
        d = run_demo_market(
            market_id=m.market_id, market_start_ms=m.start_ts_ms, market_end_ms=m.end_ts_ms,
            spot_start=Decimal(str(args.spot_start)), spot_entry=Decimal(str(args.spot_now)),
            spot_exit=Decimal(str(args.spot_exit)) if args.spot_exit else Decimal(str(args.spot_now)),
            settlement=None, config=DemoConfig(),
        )
        print(render_demo_log(d))
        print("trading: none (demo simulation only)")
        return 0

    return asyncio.run(_live())


if __name__ == "__main__":
    sys.exit(main())
