"""Read-only DEMO + replay capture: frozen ETH flow +20s over N new ETH 5m markets.

READ-ONLY DEMO (simulation only, no orders, LIVE disabled, no withdrawals):

* Entry at +20s: ETH taker flow imbalance over prior 30s from Binance public
  ETHUSDT 1s klines (closeTime rule, strictly pre-decision).
  flow >= +0.25 -> BUY UP, flow <= -0.25 -> BUY DOWN, else HOLD.
  Frozen: EthFlowConfig defaults (strategy NOT modified here).
* At +90s and +180s, opposite +-2bps ETH spot drift -> EXIT at then-current
  contract price; otherwise hold to settlement.
* Simulated fills from read-only Predict.fun orderbook/market books
  (UP ask for UP entry, 1 - UP bid for DOWN entry; mirrored salvage on EXIT).
  Never quotes, never trades. LIVE refused fail-closed.
* Settlement from read-only venue outcome poll (WON/LOST -> UP/DOWN, else PUSH).

Replay capture (per market, under --replay-dir, gitignored research data just
like the BTC validation cache):

* ``market_<id>.json``: market_id/start_ms/end_ms/symbol/payload (final
  get_market incl. outcomes + variantData)/series (chance timeseries)/klines
  (ETHUSDT 1s ``[open, close, close, vol, taker]`` covering start-180s..end).
  Loadable offline with ``validation_dataset`` helpers
  (Kline/parse_contract_series/parse_venue_outcome) exactly like the BTC
  ``up_down_5m_validation`` cache.
* Exact book legs + used prices + flow/signal/decisions/economics are stored
  in the per-trade record (progress JSONL), so both the signal path
  (backtest_eth_flow on mids) and the exact fill economics (simulate_trade
  on spread books) are reproducible offline without contacting the venue.

Strategy is NOT modified here: all decisions call the frozen
app.research.prediction_markets.up_down_5m.eth_flow_signal functions with
default EthFlowConfig. No BTC markets are touched.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import logging

logging.disable(logging.CRITICAL)

REPLAY_DEFAULT = "data/research/up_down_5m_eth_validation"


def _num(x: object) -> Decimal | None:
    try:
        return Decimal(str(x))
    except Exception:
        return None


def _up_book_from_orderbook(payload: object) -> tuple[Decimal | None, Decimal | None]:
    node: object = payload
    if isinstance(node, dict) and isinstance(node.get("data"), dict):
        node = node["data"]
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
    try:
        if isinstance(node, dict):
            for o in node.get("outcomes") or []:
                if isinstance(o, dict) and str(o.get("name", "")).upper() == "UP":
                    bb = o.get("bestBid")
                    ba = o.get("bestAsk")
                    bb = _num(bb["price"] if isinstance(bb, dict) else bb) if bb is not None else None
                    ba = _num(ba["price"] if isinstance(ba, dict) else ba) if ba is not None else None
                    if bb is not None and ba is not None:
                        return bb, ba
    except Exception:
        pass
    return None, None


def _up_book_from_market(payload: dict) -> tuple[Decimal | None, Decimal | None, int]:
    try:
        data = payload.get("data", {}) or {}
        for o in data.get("outcomes", []) or []:
            if str(o.get("name", "")).upper() == "UP":
                bb = o.get("bestBid")
                ba = o.get("bestAsk")
                bb = _num(bb["price"] if isinstance(bb, dict) else bb) if bb is not None else None
                ba = _num(ba["price"] if isinstance(ba, dict) else ba) if ba is not None else None
                fee = int(data.get("feeRateBps") or 200)
                if bb is not None and ba is not None:
                    return bb, ba, fee
    except Exception:
        pass
    return None, None, 200


async def _sleep_until(ts_ms: int) -> None:
    while True:
        now = int(time.time() * 1000)
        gap = ts_ms - now
        if gap <= 0:
            return
        await asyncio.sleep(min(5.0, gap / 1000.0))


async def main_async(args: argparse.Namespace) -> int:
    from app.research.prediction_markets.predict_client import PredictFunClient, resolve_predict_token
    from app.research.prediction_markets.up_down_5m.demo import DemoMode, ensure_demo_only
    from app.research.prediction_markets.up_down_5m.demo_simulator import (
        DemoFill,
        DemoSimConfig,
        simulate_trade,
        summarize,
    )
    from app.research.prediction_markets.up_down_5m.eth_flow_signal import (
        EthFlowConfig,
        compute_drift_bps,
        compute_flow_imb,
        decide_eth_exit_drift,
        decide_eth_flow,
    )
    from app.research.prediction_markets.up_down_5m.validation_dataset import fetch_klines

    ensure_demo_only(DemoMode.DEMO)  # fail-closed: LIVE refused
    cfg = EthFlowConfig()  # frozen, defaults only — never modified
    assert cfg.entry_offset_ms == 20_000
    assert cfg.spot_symbol == "ETHUSDT"
    sim_cfg = DemoSimConfig(mode=DemoMode.DEMO, stake_shares=Decimal(str(args.stake)))
    print("mode=DEMO read-only ETH flow (+20s, simulation only, no orders, LIVE disabled)", flush=True)
    print(f"frozen: symbol={cfg.spot_symbol} entry={cfg.entry_offset_ms} flow_thr={cfg.flow_thr} "
          f"exits={list(cfg.exit_offsets_ms)} exit_thr={cfg.exit_drift_thr_bps} "
          f"fee={cfg.fee_bps} slip={cfg.slippage_per_share}", flush=True)

    pairs: list[tuple[int, int]] = []
    for spec in str(args.windows).split(","):
        spec = spec.strip()
        if not spec:
            continue
        start_s, mid = spec.split(":")
        pairs.append((int(start_s), int(mid)))
    if len(pairs) != int(args.markets):
        print(f"BLOCKER: --windows must list exactly --markets={args.markets} start_sec:market_id pairs "
              f"(got {len(pairs)}).", flush=True)
        return 2
    print(f"windows n={len(pairs)} {pairs[0][0]} -> {pairs[-1][0]}", flush=True)

    replay_dir = Path(args.replay_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)
    prog = replay_dir / "demo_progress.jsonl"
    prog.write_text("", encoding="utf-8")

    tok = resolve_predict_token()
    if not tok:
        print("BLOCKER: no Predict.fun token.")
        return 2

    async def klines(a: int, b: int) -> list:
        try:
            return await fetch_klines(cfg.spot_symbol, a, b)
        except Exception as exc:
            print(f"klines failed [{a},{b}]: {str(exc)[:150]}", flush=True)
            return []

    def close_at(kl: list, ts: int):  # strictly pre-decision: closeTime <= ts
        best = None
        for k in kl:
            if k.close_ts_ms <= ts:
                best = k
        return best

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
            bb, ba, fee = _up_book_from_market(m)
            return bb, ba, fee
        except Exception:
            return None, None, 200

    async def venue_outcome(client: PredictFunClient, mid: int) -> tuple[str | None, list]:
        try:
            m = await client.get_market(mid)
            d = m.get("data", {}) or {}
            outs = [(o.get("name"), o.get("status")) for o in d.get("outcomes", []) or []]
            won = [str(x).upper() for x, s in outs if str(s).upper() == "WON"]
            if len(won) == 1 and won[0] in ("UP", "DOWN"):
                return won[0], outs
            vd = d.get("variantData") or {}
            vs, ve = vd.get("startPrice"), vd.get("endPrice")
            if vs is not None and ve is not None and str(vs) == str(ve):
                return "PUSH", outs
            if not won:
                return None, outs
            return None, outs
        except Exception as exc:
            print(f"venue poll {mid} failed: {str(exc)[:150]}", flush=True)
            return None, []

    sim_trades = []
    async with PredictFunClient(tok) as client:
        for idx, (s, mid) in enumerate(pairs):
            start, end = s * 1000, s * 1000 + 300_000
            t_e = start + cfg.entry_offset_ms
            print(f"[{idx+1}/{len(pairs)}] market_id={mid} window={s} waiting +20s ...", flush=True)
            await _sleep_until(t_e + 1500)  # small buffer so the entry kline has closed
            kl = await klines(start - 120_000, t_e)
            ref = close_at(kl, start)
            lo, hi = int(t_e) - cfg.flow_window_ms, int(t_e)
            buy = vol = Decimal("0")
            for k in kl:
                if lo < k.close_ts_ms <= hi:
                    buy += k.taker_buy_base
                    vol += k.volume_base
            flow = compute_flow_imb(buy, vol) if vol > 0 else None
            res = decide_eth_flow(flow_imb=flow, config=cfg)  # frozen call
            sig = res.signal.value
            bb_e, ba_e, fee_e = await book_up(client, mid)
            if sig == "UP":
                entry_px = ba_e
            elif sig == "DOWN":
                entry_px = (Decimal("1") - bb_e) if bb_e is not None else None
            else:
                entry_px = None
            print(f"market_id={mid} flow20={flow} signal={sig} entry_px={entry_px} "
                  f"up_bid={bb_e} up_ask={ba_e} klines={len(kl)}", flush=True)

            pos = sig if sig in ("UP", "DOWN") and entry_px is not None else "SKIP"
            exit_px = None
            exited = False
            exit_eo = None
            bb_x, ba_x = None, None
            if pos in ("UP", "DOWN"):
                for eo in cfg.exit_offsets_ms:
                    te = start + int(eo)
                    await _sleep_until(te + 1500)
                    kl2 = await klines(start - 120_000, te)
                    ref2 = close_at(kl2, start)
                    now = close_at(kl2, te)
                    d = compute_drift_bps(
                        now.close if now else None, ref2.close if ref2 else None
                    )
                    if decide_eth_exit_drift(side=pos, drift_bps=d, config=cfg):  # frozen call
                        bb_x, ba_x, _ = await book_up(client, mid)
                        if pos == "UP":
                            exit_px = bb_x
                        else:
                            exit_px = (Decimal("1") - ba_x) if ba_x is not None else None
                        if exit_px is not None:
                            exited = True
                            exit_eo = int(eo)
                            print(f"market_id={mid} EXIT@{eo} drift={d} exit_px={exit_px}", flush=True)
                            break
                        print(f"market_id={mid} exit signal @{eo} drift={d} but no book -> HOLD", flush=True)
            else:
                for eo in cfg.exit_offsets_ms:
                    await _sleep_until(min(start + int(eo) + 1500, end))
                    if int(time.time() * 1000) >= end:
                        break

            await _sleep_until(end + 8_000)
            st = None
            outs_last: list = []
            for i in range(int(args.settle_polls)):
                st, outs_last = await venue_outcome(client, mid)
                if st is not None:
                    break
                await asyncio.sleep(int(args.settle_interval))
            print(f"market_id={mid} settlement={st} outcomes={outs_last} exited={exited}", flush=True)

            side = pos if pos in ("UP", "DOWN") else "SKIP"
            fill = None
            if side != "SKIP" and bb_e is not None and ba_e is not None:
                if exited and exit_px is None:
                    exited = False
                if exited:
                    if side == "UP":
                        fill = DemoFill(market_id=mid, up_bid_entry=bb_e, up_ask_entry=ba_e,
                                        up_bid_exit=exit_px, up_ask_exit=ba_e, fee_bps=fee_e)
                    else:
                        implied_ask_exit = (Decimal("1") - exit_px) if exit_px is not None else None
                        fill = DemoFill(market_id=mid, up_bid_entry=bb_e, up_ask_entry=ba_e,
                                        up_bid_exit=bb_e, up_ask_exit=implied_ask_exit, fee_bps=fee_e)
                else:
                    fill = DemoFill(market_id=mid, up_bid_entry=bb_e, up_ask_entry=ba_e, fee_bps=fee_e)
            tr = simulate_trade(market_id=mid, side=side, fill=fill,
                                settlement=st, exited=exited, config=sim_cfg)
            sim_trades.append(tr)
            summ = summarize(sim_trades)
            print(f"TRADE market_id={mid} flow={flow} side={tr.side} entry_px={tr.entry_price} "
                  f"exit_px={tr.exit_price} shares={tr.shares} settlement={tr.settlement} "
                  f"result={tr.result} gross={tr.gross} fees={tr.fees} slip={tr.slippage} "
                  f"net={tr.net} staked={tr.staked}", flush=True)
            print(f"CUM trades={summ.trades} W={summ.wins} L={summ.losses} X={summ.exits} "
                  f"gross={summ.gross} fees={summ.fees} net={summ.total_net} "
                  f"avg={summ.avg_net} dd={summ.max_drawdown} roi={summ.roi} "
                  f"staked={summ.total_staked}", flush=True)
            rec = {"market_id": mid, "window_start_sec": s, "flow20": str(flow) if flow is not None else None,
                   "signal": sig, "entry_px": str(tr.entry_price),
                   "entry_book": {"up_bid": str(bb_e) if bb_e is not None else None,
                                  "up_ask": str(ba_e) if ba_e is not None else None, "fee_bps": fee_e},
                   "exit_offset_ms": exit_eo,
                   "exit_book": {"up_bid": str(bb_x) if bb_x is not None else None,
                                 "up_ask": str(ba_x) if ba_x is not None else None},
                   "exit_px": str(tr.exit_price) if tr.exit_price is not None else None,
                   "settlement": tr.settlement.value if tr.settlement is not None else st,
                   "result": tr.result, "gross": str(tr.gross), "fees": str(tr.fees),
                   "slippage": str(tr.slippage), "net": str(tr.net), "staked": str(tr.staked)}
            with prog.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

            # ---- replay capture (same shape as BTC up_down_5m_validation cache) ----
            try:
                full_kl = await klines(start - 180_000, end)
                krows = [[k.open_ts_ms, k.close_ts_ms, str(k.close), str(k.volume_base),
                          str(k.taker_buy_base)] for k in full_kl]
                try:
                    series = await client._get(
                        f"/v1/markets/{mid}/timeseries",
                        {"metric": "chance", "from": str(s - 120), "to": str(s + 360), "limit": "100"},
                    )
                except Exception as exc:
                    print(f"market_id={mid} timeseries capture failed: {str(exc)[:150]}", flush=True)
                    series = {"data": {"series": []}}
                try:
                    payload = await client.get_market(mid)
                except Exception as exc:
                    print(f"market_id={mid} payload capture failed: {str(exc)[:150]}", flush=True)
                    payload = {"data": None}
                blob = {"market_id": mid, "start_ms": start, "end_ms": end, "symbol": cfg.spot_symbol,
                        "payload": payload, "series": series, "klines": krows, "kline_interval": "1s"}
                (replay_dir / f"market_{mid}.json").write_text(json.dumps(blob), encoding="utf-8")
                print(f"market_id={mid} replay saved klines={len(krows)}", flush=True)
            except Exception as exc:
                print(f"market_id={mid} replay capture BLOCKER: {str(exc)[:200]}", flush=True)
        summ = summarize(sim_trades)
        cov = (summ.trades / len(pairs)) if pairs else 0.0
        wr = (summ.wins / summ.trades) if summ.trades else None
        print(f"=== FINAL markets={len(pairs)} trades={summ.trades} coverage={cov:.3f} "
              f"wins={summ.wins} losses={summ.losses} exits={summ.exits} pushes={summ.pushes} "
              f"skips={summ.skips} win_rate={wr} gross={summ.gross} fees={summ.fees} "
              f"slippage={summ.slippage} net={summ.total_net} avg={summ.avg_net} "
              f"dd={summ.max_drawdown} roi={summ.roi} staked={summ.total_staked} ===", flush=True)
        print(f"replay dataset: {replay_dir}/market_<id>.json + demo_progress.jsonl "
              f"(offline-replayable, no venue needed)", flush=True)
        print("trading: none (read-only DEMO, no orders placed, LIVE disabled)", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only DEMO+replay: frozen ETH flow +20s (no trading)")
    ap.add_argument("--markets", type=int, default=10)
    ap.add_argument("--stake", default="1", help="shares per entered market (1 matches per-share units)")
    ap.add_argument("--windows", required=True,
                    help="comma-separated start_sec:market_id pairs, exactly --markets of them, ascending")
    ap.add_argument("--replay-dir", default=REPLAY_DEFAULT)
    ap.add_argument("--settle-polls", default="20")
    ap.add_argument("--settle-interval", default="30", help="seconds between settlement polls")
    args = ap.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
