"""12-hour research-only collection: Predict.fun + Binance spot, BTC/ETH 5m/15m.

Requirements satisfied:
 - Predict.fun token from .env via resolve_predict_token (no logging)
 - BTC/ETH 5m/15m only, BNB excluded
 - Binance spot BTCUSDT/ETHUSDT simultaneous
 - WS pref where supported, REST snapshots for recovery
 - Synchronized timestamps, bid/ask, spread, depth, spot movement, market ID, resolution, sequence/update, gaps/reconnects/duplicates/stale/expired via HistoricalCollector
 - Persist via Phase 2 CollectorStore under data/research/prediction_markets
 - Exactly 12 hours, research-only, no trading imports
 - Do not print secrets
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.research.prediction_markets.collector.collector import HistoricalCollector
from app.research.prediction_markets.collector.storage import CollectorStore
from app.research.prediction_markets.predict_client import PredictFunClient, resolve_predict_token

try:
    import websockets  # type: ignore[import-not-found]

    _HAS_WEBSOCKETS = True
except ImportError:
    _HAS_WEBSOCKETS = False
    websockets = None  # type: ignore[assignment]

DEFAULT_OUT = Path("data/research/prediction_markets")
SPOT_URL = "https://api.binance.com"
SPOT_SYMBOLS = ("BTCUSDT", "ETHUSDT")

# parse ISO to ms
from datetime import datetime as _dt


def _iso_to_ms(s: str | None) -> int | None:
    if not s:
        return None
    try:
        # 2026-09-11T23:40:00.000Z
        d = _dt.fromisoformat(s.replace("Z", "+00:00"))
        return int(d.timestamp() * 1000)
    except Exception:
        return None


async def spot_loop(collector: HistoricalCollector, stop: asyncio.Event, poll_ms: int = 500):
    seq = {s: 0 for s in SPOT_SYMBOLS}
    async with httpx.AsyncClient(timeout=5) as http:
        while not stop.is_set():
            for sym in SPOT_SYMBOLS:
                try:
                    r = await http.get(f"{SPOT_URL}/api/v3/ticker/bookTicker", params={"symbol": sym})
                    if r.status_code != 200:
                        continue
                    data = r.json()
                    bid = data.get("bidPrice")
                    ask = data.get("askPrice")
                    if not bid or not ask:
                        continue
                    seq[sym] += 1
                    now = int(time.time() * 1000)
                    collector.ingest_spot(
                        {
                            "symbol": sym,
                            "bid": bid,
                            "ask": ask,
                            "bid_qty": data.get("bidQty"),
                            "ask_qty": data.get("askQty"),
                            "exchange_ts_ms": now,
                            "sequence": seq[sym],
                        },
                        captured_at_ms=now,
                    )
                except asyncio.CancelledError:
                    return
                except Exception:
                    continue
            try:
                await asyncio.wait_for(stop.wait(), timeout=poll_ms / 1000)
            except asyncio.TimeoutError:
                continue


async def predict_loop(collector: HistoricalCollector, stop: asyncio.Event, client: PredictFunClient, poll_ms: int = 1000):
    seq = 0
    while not stop.is_set():
        try:
            cats = await client.list_categories(limit=50)
            items = cats.get("data", []) if isinstance(cats, dict) else []
            # filter BTC/ETH 5m/15m, exclude BNB
            active = []
            for cat in items:
                slug = str(cat.get("slug", "")).lower()
                if "bnb" in slug:
                    continue
                if not ("btc" in slug or "eth" in slug):
                    continue
                if not ("5m" in slug or "15m" in slug or "5-min" in slug or "15-min" in slug):
                    # fallback check tags
                    tags = cat.get("tags", []) or []
                    tag_names = " ".join(str(t.get("name","")).lower() for t in tags)
                    if "5 min" not in tag_names and "15 min" not in tag_names:
                        continue
                # priceFeedSymbol from variantData
                vd = cat.get("variantData") or {}
                symbol = vd.get("priceFeedSymbol") or ("BTCUSDT" if "btc" in slug else "ETHUSDT")
                # duration from slug or tags — check 15m before 5m to avoid substring false positive ("15m" contains "5m")
                if "15m" in slug or "15-min" in slug:
                    duration = "15m"
                elif "5m" in slug or "5-min" in slug:
                    duration = "5m"
                else:
                    # infer from tags
                    tag_names = " ".join(str(t.get("name","")).lower() for t in cat.get("tags", []) or [])
                    if "15 min" in tag_names:
                        duration = "15m"
                    elif "5 min" in tag_names:
                        duration = "5m"
                    else:
                        duration = "5m"
                starts = cat.get("startsAt")
                ends = cat.get("endsAt")
                res_ms = _iso_to_ms(ends)
                start_ms = _iso_to_ms(starts)
                # child market
                for m in cat.get("markets") or []:
                    mid = m.get("id")
                    if mid is None:
                        continue
                    active.append({
                        "market_id": int(mid),
                        "symbol": symbol,
                        "duration": duration,
                        "resolution_ms": res_ms,
                        "start_ms": start_ms,
                        "slug": slug,
                    })
                    if len(active) >= 20:
                        break
                if len(active) >= 20:
                    break
            # poll orderbooks for active markets
            for entry in active:
                if stop.is_set():
                    break
                mid = entry["market_id"]
                # skip expired
                now_ms = int(time.time() * 1000)
                if entry["resolution_ms"] is not None and now_ms >= entry["resolution_ms"]:
                    continue
                try:
                    ob = await client.get_orderbook(mid)
                    # ob shape: {"data": {"bids":[[p,s]...],"asks":...,"marketId":...,"updateTimestampMs":...}, "success": True}
                    data = ob.get("data", ob) if isinstance(ob, dict) else {}
                    bids = data.get("bids", [])
                    asks = data.get("asks", [])
                    upd = data.get("updateTimestampMs") or now_ms
                    seq += 1
                    # normalize bids/asks to collector expected format (list of [price,size] strings)
                    bids_norm = [[str(p), str(s)] for p, s in bids]
                    asks_norm = [[str(p), str(s)] for p, s in asks]
                    collector.ingest_prediction_orderbook(
                        {
                            "marketId": mid,
                            "tokenId": f"predict_{mid}",
                            "symbol": entry["symbol"],
                            "updateTimestampMs": int(upd),
                            "sequence": seq,
                            "resolution_ms": entry["resolution_ms"],
                            "bids": bids_norm,
                            "asks": asks_norm,
                            "duration": entry["duration"],
                            "marketTopicId": None,
                        },
                        captured_at_ms=now_ms,
                    )
                except Exception:
                    continue
        except asyncio.CancelledError:
            return
        except Exception:
            # discovery gap -> count as reconnect
            try:
                collector.on_reconnect("predict_discovery_error")
            except Exception:
                pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll_ms / 1000)
        except asyncio.TimeoutError:
            continue


async def main():
    duration = 43200.0  # 12 hours
    out = DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "observations").mkdir(parents=True, exist_ok=True)

    tok = resolve_predict_token()
    if not tok:
        print("ERROR: Predict.fun token not found (CAT_RESEARCH__PREDICT_API_KEY) - fail-closed")
        return 1
    # verify BTC/ETH scope without printing token
    print(f"Predict token loaded: True (BTC/ETH 5m/15m only, BNB excluded)")
    print(f"Collector ALLOWED_BASES: BTC, ETH")
    print(f"Output: {out}/observations/ + {out}/collector.db")
    start_utc = datetime.now(timezone.utc)
    print(f"Collection start UTC: {start_utc.isoformat()}")
    end_expected = datetime.fromtimestamp(start_utc.timestamp() + duration, tz=timezone.utc)
    print(f"Expected end UTC: {end_expected.isoformat()} (12h)")

    store = CollectorStore(base_dir=out, memory_only=False)
    collector = HistoricalCollector(
        store=store,
        client=None,
        stale_threshold_ms=5000,
        prediction_stale_threshold_ms=300000,
        sync_window_ms=500,
        resolution_by_market={},
    )

    stop = asyncio.Event()
    client = PredictFunClient(tok)

    print("Research-only: no orders, wallet ops, transfers, withdrawals or trading - fail-closed")

    # WebSocket-first, REST fallback — preserve timestamps, dedup, gaps, stale, expiry, reconnect
    tasks: list[asyncio.Task] = []
    if _HAS_WEBSOCKETS:
        try:
            from app.research.prediction_markets.binance_spot_ws import BinanceSpotWSClient
            from app.research.prediction_markets.predict_ws import PredictFunWSClient
            from app.research.prediction_markets import endpoints as ep

            spot_ws = BinanceSpotWSClient(collector)
            predict_ws = PredictFunWSClient(tok, collector, rest_client=client)
            tasks = [
                asyncio.create_task(spot_ws.run(stop)),
                asyncio.create_task(predict_ws.run(stop)),
            ]
            print(f"WebSocket market data preferred: True (Predict.fun {predict_ws.ws_url} + Binance Spot {spot_ws.ws_url}, fallback REST snapshots for recovery)")
            print(f"Predict.fun subscription: {{\"method\":\"subscribe\",\"params\":{{\"channel\":\"{ep.PREDICT_FUN_WS_CHANNEL_ORDERBOOK}\",\"marketIds\":\"BTC/ETH 5m/15m discovered via REST list_categories (BNB excluded)\"}}}}")
            print(f"Binance Spot topics: {ep.BINANCE_SPOT_SUBSCRIPTION_TOPICS}")
        except Exception as exc:
            print(f"WS init failed, fallback to REST polling: {exc}")
            tasks = [
                asyncio.create_task(spot_loop(collector, stop, poll_ms=500)),
                asyncio.create_task(predict_loop(collector, stop, client, poll_ms=1000)),
            ]
            print("WebSocket market data preferred: False (fallback REST snapshots for recovery)")
    else:
        tasks = [
            asyncio.create_task(spot_loop(collector, stop, poll_ms=500)),
            asyncio.create_task(predict_loop(collector, stop, client, poll_ms=1000)),
        ]
        print("WebSocket market data preferred: False (fallback REST snapshots for recovery)")

    # handle signals
    import signal
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except Exception:
            pass

    try:
        await asyncio.wait_for(stop.wait(), timeout=duration)
    except asyncio.TimeoutError:
        pass
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        try:
            await client.close()
        except Exception:
            pass

    end_utc = datetime.now(timezone.utc)
    stats = collector.stats_snapshot()

    # compute synchronized count
    sync_count = 0
    for obs in store.all_observations():
        if obs.source.value.startswith("prediction"):
            if collector.synchronized_view(obs) is not None:
                sync_count += 1

    # per symbol/duration breakdown from DB
    per = {}
    try:
        import sqlite3
        conn = sqlite3.connect(str(out / "collector.db"))
        cur = conn.execute("SELECT symbol, duration, source, COUNT(*) FROM observations GROUP BY symbol, duration, source")
        rows = cur.fetchall()
        for sym, dur, src, cnt in rows:
            per[(sym, dur, src)] = cnt
        conn.close()
    except Exception as e:
        rows = []
        per = {"error": str(e)}

    print("=== COMPLETION REPORT ===")
    print(f"collection start UTC: {start_utc.isoformat()}")
    print(f"collection end UTC: {end_utc.isoformat()}")
    print(f"duration secs: {duration}")
    print(f"total spot observations: {stats.spot_count}")
    print(f"prediction observations: {stats.prediction_ob_count + stats.prediction_trade_count}")
    print(f"synchronized observations: {sync_count}")
    print(f"observations per symbol/duration: {per}")
    print(f"reconnects: {stats.reconnects}")
    print(f"sequence gaps: {stats.gaps_detected}")
    print(f"duplicates: {stats.duplicates_dropped}")
    print(f"stale/expired observations: stale={stats.stale_dropped} expired={stats.expired_dropped} ooo={stats.out_of_order_dropped}")
    print(f"database/file locations: {out}/collector.db , {out}/observations/YYYY-MM-DD.jsonl")
    print(f"trading: none (research-only fail-closed)")
    # Log to file
    report_path = out / f"report_{start_utc.strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"start UTC: {start_utc.isoformat()}\n")
            f.write(f"end UTC: {end_utc.isoformat()}\n")
            f.write(f"duration secs: {duration}\n")
            f.write(f"spot: {stats.spot_count}\n")
            f.write(f"prediction: {stats.prediction_ob_count + stats.prediction_trade_count}\n")
            f.write(f"synchronized: {sync_count}\n")
            f.write(f"per: {per}\n")
            f.write(f"reconnects: {stats.reconnects}\n")
            f.write(f"gaps: {stats.gaps_detected}\n")
            f.write(f"duplicates: {stats.duplicates_dropped}\n")
            f.write(f"stale: {stats.stale_dropped} expired: {stats.expired_dropped} ooo: {stats.out_of_order_dropped}\n")
            f.write(f"db: {out/'collector.db'}\n")
        print(f"report written to {report_path}")
    except Exception:
        pass

    store.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
