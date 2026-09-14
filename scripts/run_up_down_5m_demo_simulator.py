"""Read-only DEMO simulator run: frozen signal + real mechanics, ≥20 markets.

Live reads only (never places orders, never enables LIVE):

* Binance public klines / bookTicker (spot reference for the frozen signal).
* Predict.fun ``get_market`` (window/outcome/feeRateBps) and ``get_orderbook``
  (real UP bid/ask at entry/exit instants for simulated fills).

Flow per market: resolve BTC id → wait +90s → entry book poll + drift signal
→ wait +180s → exit book poll + reversal check → wait settlement → poll
venue outcome → price via :mod:`demo_simulator` → print per-trade and
cumulative logs (entry prices, exits, settlements, win/loss, gross/net,
cumulative PnL, drawdown, ROI).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from decimal import Decimal
from pathlib import Path

logging.disable(logging.CRITICAL)

PROG_DEFAULT = r"C:\Users\derkd\AppData\Local\Temp\opencode\demo_sim_progress.jsonl"


def _up_book_from_orderbook(payload: object) -> tuple[Decimal | None, Decimal | None]:
    """Best UP bid/ask from a get_orderbook payload (tolerant to shape)."""
    def _num(x: object) -> Decimal | None:
        try:
            return Decimal(str(x))
        except Exception:
            return None

    node: object = payload
    if isinstance(node, dict) and isinstance(node.get("data"), dict):
        node = node["data"]
    # shape A: {"bids": [{"price":..}], "asks": [...]} (maybe under outcome key)
    cands = [node]
    if isinstance(node, dict):
        for key in ("up", "UP", "yes", "YES"):
            if isinstance(node.get(key), dict):
                cands.append(node[key])
    for c in cands:
        if not isinstance(c, dict):
            continue
        bids, asks = c.get("bids"), c.get("asks")
        try:
            bb = _num(bids[0]["price"] if isinstance(bids[0], dict) else bids[0][0]) if bids else None
            ba = _num(asks[0]["price"] if isinstance(asks[0], dict) else asks[0][0]) if asks else None
        except Exception:
            bb, ba = None, None
        if bb is not None and ba is not None:
            return bb, ba
    return None, None


def _up_book_from_market(payload: dict) -> tuple[Decimal | None, Decimal | None]:
    """Fallback: UP outcome bestBid/bestAsk from get_market payload."""
    try:
        for o in payload.get("data", {}).get("outcomes", []) or []:
            if str(o.get("name", "")).upper() == "UP":
                bb = o.get("bestBid")
                ba = o.get("bestAsk")
                bb = Decimal(str(bb["price"] if isinstance(bb, dict) else bb)) if bb is not None else None
                ba = Decimal(str(ba["price"] if isinstance(ba, dict) else ba)) if ba is not None else None
                return bb, ba
    except Exception:
        pass
    return None, None


async def _spot_mid() -> Decimal:
    import httpx

    async with httpx.AsyncClient(timeout=10) as h:
        r = await h.get("https://api.binance.com/api/v3/ticker/bookTicker", params={"symbol": "BTCUSDT"})
        j = r.json()
        return (Decimal(j["bidPrice"]) + Decimal(j["askPrice"])) / Decimal("2")


async def main_async(args: argparse.Namespace) -> int:
    from app.research.prediction_markets.predict_client import PredictFunClient, resolve_predict_token
    from app.research.prediction_markets.up_down_5m.demo import (
        DemoConfig, decide_demo_entry, decide_demo_exit, render_demo_log, run_demo_market,
    )
    from app.research.prediction_markets.up_down_5m.demo_simulator import (
        DemoFill, DemoSimConfig, simulate_trade, summarize,
    )
    from app.research.prediction_markets.up_down_5m.live_snapshot import PriceProvenance
    from app.research.prediction_markets.up_down_5m.signal import Position
    from app.research.prediction_markets.up_down_5m.validation_dataset import fetch_klines

    print("mode=DEMO read-only simulator (simulated fills, no orders, LIVE disabled)", flush=True)
    prog = Path(args.progress)
    prog.write_text("", encoding="utf-8")
    sim_cfg = DemoSimConfig(stake_shares=Decimal(str(args.stake)))
    demo_cfg = DemoConfig()
    n = int(args.markets)

    def close_at(kl: list, ts: int) -> Decimal | None:
        best = None
        for k in kl:
            if k.close_ts_ms <= ts:
                best = k
        return best.close if best else None

    async def book_up(client: PredictFunClient, mid: int) -> tuple[Decimal | None, Decimal | None, int]:
        try:
            ob = await client.get_orderbook(mid)
            bb, ba = _up_book_from_orderbook(ob)
            if bb is not None and ba is not None:
                return bb, ba, 200
        except Exception:
            pass
        try:
            m = await client.get_market(mid)
            bb, ba = _up_book_from_market(m)
            fee = int((m.get("data", {}) or {}).get("feeRateBps") or 200)
            return bb, ba, fee
        except Exception:
            return None, None, 200

    async def venue(client: PredictFunClient, mid: int) -> tuple[str | None, str | None, str | None, list]:
        try:
            m = await client.get_market(mid)
            d = m.get("data", {})
            outs = [(o.get("name"), o.get("status")) for o in d.get("outcomes", []) or []]
            vd = d.get("variantData") or {}
            won = [str(x).upper() for x, s in outs if str(s).upper() == "WON"]
            st = won[0] if len(won) == 1 and won[0] in ("UP", "DOWN") else None
            return st, vd.get("startPrice"), vd.get("endPrice"), outs
        except Exception:
            return None, None, None, []

    async def find_btc(client: PredictFunClient, start_sec: int, cursor: int) -> tuple[int | None, int]:
        want = f"btc-updown-5m-{start_sec}"
        mid, guard = cursor, 0
        while guard < 4000:
            guard += 1
            try:
                m = await client.get_market(mid)
                slug = (m.get("data", {}) or {}).get("categorySlug") or ""
                if slug == want:
                    return mid, mid + 1
                if "btc-updown-5m" in slug:
                    try:
                        s = int(slug.rsplit("-", 1)[-1])
                    except ValueError:
                        mid += 1
                        continue
                    mid += max(1, min(300, (start_sec - s) // 300 * 100)) if s < start_sec else 1
                    continue
            except Exception:
                pass
            mid += 1
            if guard % 25 == 0:
                await asyncio.sleep(2)
        return None, cursor

    async def klines(a: int, b: int) -> list:
        try:
            return await fetch_klines("BTCUSDT", a, b)
        except Exception as exc:
            print(f"klines failed: {str(exc)[:120]}", flush=True)
            return []

    tok = resolve_predict_token()
    if not tok:
        print("BLOCKER: no Predict.fun token.")
        return 2
    sim_trades = []
    async with PredictFunClient(tok) as client:
        now = int(time.time())
        first = ((now + 1500) // 300) * 300  # first full window with discovery budget
        starts = [first + i * 300 for i in range(n)]
        print(f"windows n={n} from {starts[0]} to {starts[-1]}", flush=True)
        cursor = int(args.cursor_hint)
        for idx, s in enumerate(starts):
            start, end = s * 1000, s * 1000 + 300_000
            print(f"[{idx+1}/{n}] resolving id for window {s} ...", flush=True)
            mid, cursor = await find_btc(client, s, cursor)
            if mid is None:
                print(f"window {s}: id unresolved -> SKIP-UNRESOLVED", flush=True)
                continue
            t_e, t_x = start + 90_000, start + 180_000
            while int(time.time() * 1000) < t_e:
                await asyncio.sleep(5)
            kl = await klines(start - 180_000, start + 200_000)
            ref, me = close_at(kl, start), close_at(kl, t_e)
            px = await _spot_mid()
            sig_e, drift_e, act_e = decide_demo_entry(
                market_id=mid, market_start_ms=start, market_end_ms=end,
                spot_start=ref, spot_now=me,
                provenance=PriceProvenance.SPOT_REFERENCE, config=demo_cfg)
            bb_e, ba_e, fee_e = await book_up(client, mid)
            print(f"market_id={mid} spot_start={ref} current_BTC={px} elapsed={int(time.time()*1000)-start} "
                  f"drift90={drift_e} signal={sig_e.value} sim_entry={act_e.value} "
                  f"up_bid={bb_e} up_ask={ba_e}", flush=True)
            pos = Position.NONE
            if act_e.value == "ENTER_UP":
                pos = Position.LONG_UP
            elif act_e.value == "ENTER_DOWN":
                pos = Position.LONG_DOWN
            while int(time.time() * 1000) < t_x:
                await asyncio.sleep(5)
            kl2 = await klines(start - 180_000, end)
            mx = close_at(kl2, t_x)
            if pos == Position.NONE:
                sig_x_v, act_x_v = "HOLD", "HOLD_POSITION"
                bb_x, ba_x = None, None
            else:
                sig_x, _dx, act_x = decide_demo_exit(
                    market_id=mid, market_start_ms=start, market_end_ms=end, position=pos,
                    spot_start=ref, spot_now=mx,
                    provenance=PriceProvenance.SPOT_REFERENCE, config=demo_cfg)
                bb_x, ba_x, _ = await book_up(client, mid)
                sig_x_v, act_x_v = sig_x.value, act_x.value
            print(f"market_id={mid} exit180 signal={sig_x_v} action={act_x_v} up_bid={bb_x} up_ask={ba_x}", flush=True)
            while int(time.time() * 1000) < end + 5_000:
                await asyncio.sleep(5)
            st, vs, ve, outs = None, None, None, []
            for i in range(12):
                st, vs, ve, outs = await venue(client, mid)
                if st:
                    break
                await asyncio.sleep(30)
            dec = run_demo_market(market_id=mid, market_start_ms=start, market_end_ms=end,
                                  spot_start=ref, spot_entry=me, spot_exit=mx,
                                  settlement=st, config=demo_cfg)
            print(render_demo_log(dec), flush=True)
            side = "SKIP"
            if dec.entry_action.value == "ENTER_UP":
                side = "UP"
            elif dec.entry_action.value == "ENTER_DOWN":
                side = "DOWN"
            fill = None
            if side != "SKIP" and bb_e is not None and ba_e is not None:
                fill = DemoFill(market_id=mid, up_bid_entry=bb_e, up_ask_entry=ba_e,
                                up_bid_exit=bb_x, up_ask_exit=ba_x, fee_bps=fee_e)
            tr = simulate_trade(market_id=mid, side=side, fill=fill,
                                settlement=dec.settlement.value if dec.settlement else st,
                                exited=(dec.result == "EXIT"), config=sim_cfg)
            sim_trades.append(tr)
            summ = summarize(sim_trades)
            print(f"TRADE market_id={mid} side={tr.side} entry_px={tr.entry_price} "
                  f"exit_px={tr.exit_price} shares={tr.shares} settlement={tr.settlement} "
                  f"result={tr.result} gross={tr.gross} fees={tr.fees} slip={tr.slippage} "
                  f"net={tr.net} staked={tr.staked}", flush=True)
            print(f"CUM trades={summ.trades} W={summ.wins} L={summ.losses} X={summ.exits} "
                  f"gross={summ.gross} fees={summ.fees} net={summ.total_net} "
                  f"avg={summ.avg_net} dd={summ.max_drawdown} roi={summ.roi} "
                  f"staked={summ.total_staked}", flush=True)
            with prog.open("a", encoding="utf-8") as f:
                f.write(tr.model_dump_json() + "\n")
        summ = summarize(sim_trades)
        print(f"=== FINAL trades={summ.trades} wins={summ.wins} losses={summ.losses} "
              f"exits={summ.exits} pushes={summ.pushes} skips={summ.skips} "
              f"gross={summ.gross} fees={summ.fees} slippage={summ.slippage} "
              f"net={summ.total_net} avg={summ.avg_net} dd={summ.max_drawdown} "
              f"roi={summ.roi} staked={summ.total_staked} ===", flush=True)
        print("trading: none (read-only DEMO simulator, no orders placed, LIVE disabled)", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only DEMO trade simulator (no trading)")
    ap.add_argument("--markets", type=int, default=20)
    ap.add_argument("--stake", default="10", help="shares per entered market")
    ap.add_argument("--cursor-hint", default="2252107", help="recent BTC id to start id search")
    ap.add_argument("--progress", default=PROG_DEFAULT)
    args = ap.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
