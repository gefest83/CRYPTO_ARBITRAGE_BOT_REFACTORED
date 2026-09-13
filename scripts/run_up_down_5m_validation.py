"""Validate the UNCHANGED BTC Up/Down 5m signal against real venue outcomes.

Bounded and research-only: for --max-markets BTC 5m markets from the
existing collector DB it performs 3 read-only calls per market
(Predict.fun get_market + timeseries chance, Binance public klines),
caches everything under data/research/up_down_5m_validation/ (gitignored),
then evaluates the frozen signal at --offset-ms against the actual
venue-resolved UP/DOWN. No Chainlink API, no new collector, no trading,
no signal-logic changes. Re-runs are fully offline via --no-fetch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from pathlib import Path

CACHE = Path("data/research/up_down_5m_validation")


def _db_markets(db: str, max_markets: int) -> list[tuple[int, int]]:
    con = sqlite3.connect(db)
    rows = con.execute(
        "select market_id, max(resolution_ms) from observations "
        "where source='prediction_orderbook' and symbol='BTCUSDT' and duration='5m' "
        "and market_id is not null and resolution_ms is not null "
        "group by market_id order by market_id limit %d" % int(max_markets)
    ).fetchall()
    con.close()
    return [(int(m), int(r)) for m, r in rows]


async def _fetch_market(predict, market_id: int) -> dict:
    return await predict.get_market(market_id)


async def _fetch_series(predict, market_id: int, start_ms: int, end_ms: int) -> dict:
    return await predict.get_timeseries(
        market_id, metric="chance",
        from_sec=start_ms // 1000 - 120, to_sec=end_ms // 1000 + 60, limit=100,
    )


async def main_async(args: argparse.Namespace) -> int:
    from dotenv import dotenv_values

    from app.research.prediction_markets.predict_client import PredictFunClient
    from app.research.prediction_markets.up_down_5m.signal import SignalConfig
    from app.research.prediction_markets.up_down_5m.validation_dataset import (
        ValidatedMarket,
        fetch_klines,
        parse_contract_series,
        parse_venue_outcome,
        parse_window_from_slug,
        run_validation,
    )

    CACHE.mkdir(parents=True, exist_ok=True)
    markets = _db_markets(args.db, args.max_markets)
    print(f"candidate_markets={len(markets)}")

    vals = dotenv_values(".env") or {}
    token = (vals.get("CAT_RESEARCH__PREDICT_API_KEY") or "").strip()
    if not token and not args.no_fetch:
        print("BLOCKER: no Predict.fun token (CAT_RESEARCH__PREDICT_API_KEY); use --no-fetch with cache.")
        return 2

    dataset: list[ValidatedMarket] = []
    skipped: list[str] = []
    async with PredictFunClient(token or "cache-only") as predict:
        for mid, res_ms in markets:
            cpath = CACHE / f"market_{mid}.json"
            if args.no_fetch or cpath.exists():
                if not cpath.exists():
                    skipped.append(f"{mid}:no-cache")
                    continue
                blob = json.loads(cpath.read_text(encoding="utf-8"))
            else:
                try:
                    payload = await _fetch_market(predict, mid)
                    data = payload.get("data", payload)
                    slug = str(data.get("categorySlug", ""))
                    start_ms, end_ms = parse_window_from_slug(slug)
                    if end_ms != res_ms:
                        skipped.append(f"{mid}:slug/DB window mismatch")
                        continue
                    series = await _fetch_series(predict, mid, start_ms, end_ms)
                    klines = await fetch_klines("BTCUSDT", start_ms - 180_000, end_ms)
                    blob = {
                        "market_id": mid, "start_ms": start_ms, "end_ms": end_ms,
                        "payload": payload, "series": series,
                        "klines": [[k.open_ts_ms, k.close_ts_ms, str(k.close), str(k.volume_base), str(k.taker_buy_base)] for k in klines],
                    }
                    cpath.write_text(json.dumps(blob), encoding="utf-8")
                    await asyncio.sleep(0.3)  # polite pacing, bounded run
                except Exception as exc:
                    skipped.append(f"{mid}:{str(exc)[:120]}")
                    continue
            try:
                from app.research.prediction_markets.up_down_5m.validation_dataset import (
                    Kline,
                    parse_contract_series as pcs,
                )
                from app.research.prediction_markets.up_down_5m.validation_dataset import (
                    fetch_klines as _fetch_klines,
                )
                outcome, vstart, vend = parse_venue_outcome(blob["payload"])
                klines = [Kline(open_ts_ms=k[0], close_ts_ms=k[1], close=k[2], volume_base=k[3], taker_buy_base=k[4]) for k in blob["klines"]]
                interval = blob.get("kline_interval", "1s")
                if len(klines) < 5 and not args.no_fetch:
                    # 1s retention exhausted for older windows: 1m fallback, same rules.
                    klines = await _fetch_klines("BTCUSDT", blob["start_ms"] - 180_000, blob["end_ms"], interval="1m")
                    blob["klines"] = [[k.open_ts_ms, k.close_ts_ms, str(k.close), str(k.volume_base), str(k.taker_buy_base)] for k in klines]
                    blob["kline_interval"] = "1m"
                    cpath.write_text(json.dumps(blob), encoding="utf-8")
                    interval = "1m"
                if len(klines) < 5:
                    skipped.append(f"{mid}:thin-klines")
                    continue
                dataset.append(ValidatedMarket(
                    market_id=blob["market_id"], start_ts_ms=blob["start_ms"], end_ts_ms=blob["end_ms"],
                    klines=tuple(klines), contract_series=pcs(blob["series"]),
                    outcome=outcome, venue_start_price=vstart, venue_end_price=vend,
                    notes=f"klines=Binance public {interval} backfill; outcome=venue WON/LOST + recorded Chainlink anchors",
                ))
            except Exception as exc:
                skipped.append(f"{mid}:parse:{str(exc)[:120]}")
    print(f"validated_markets={len(dataset)} skipped={len(skipped)}")
    for s in skipped[:10]:
        print("  skip:", s)
    if not dataset:
        print("BLOCKER: empty validation dataset.")
        return 2

    results, summary = run_validation(dataset, SignalConfig(), args.offset_ms)
    rep = {"summary": summary, "offset_ms": args.offset_ms,
           "results": [r.model_dump(mode="json") for r in results]}
    out = CACHE / "validation_report.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    acc = summary["accuracy"]
    print(f"sample_count={summary['markets']} scored={summary['scored']} "
          f"accuracy={acc:.3f} coverage={summary['coverage']:.3f}" if acc is not None else "no scored markets")
    for r in results:
        if r.correct is not None:
            print(f"  m={r.market_id} sig={r.signal} votes={r.votes} outcome={r.outcome.value} "
                  f"correct={r.correct} entry={r.entry_price}")
    print(f"wrote {out} trading: none (research-only)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate frozen Up/Down 5m signal vs venue outcomes (no trading)")
    ap.add_argument("--db", default="data/research/prediction_markets_multiday/collector.db")
    ap.add_argument("--max-markets", type=int, default=40)
    ap.add_argument("--offset-ms", type=int, default=90_000)
    ap.add_argument("--no-fetch", action="store_true", help="offline: use cache only")
    args = ap.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
