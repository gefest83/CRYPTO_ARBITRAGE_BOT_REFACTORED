"""Threshold research for the BTC Up/Down 5m entry signal (research-only).

Reads the EXISTING collector DB (default: multiday). No new collection,
no network, no trading, no Chainlink anywhere in this script.

Method (all strictly pre-decision, no look-ahead):
  per BTC 5m market (start = resolution - 300s): reference = spot mid at
  start; features at t = start + --offset-ms from spot books (captured_at),
  spot trades (exchange ts, signed by ``m`` flag) and the latest prediction
  book (update_ts) dated <= t. Grid-searches vote deadbands; label = spot
  mid drift into expiry vs reference.

The label is a DIRECTIONAL-CONSISTENCY PROXY, not settlement: true
settlement-threshold validation needs Chainlink history (blocked). Results
select the committed :class:`SignalConfig` defaults; the JSON report is
written to data/research/up_down_5m_thresholds.json.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import sys
import time
from decimal import Decimal
from pathlib import Path

BPS = Decimal("10000")


def _load_spot_books(con: sqlite3.Connection) -> tuple[list[int], list[Decimal]]:
    rows = con.execute(
        "select captured_at_ms, bid, ask from observations "
        "where source='spot_orderbook' and symbol='BTCUSDT' and bid is not null "
        "order by captured_at_ms"
    ).fetchall()
    ts = [int(r[0]) for r in rows]
    mids = [(Decimal(str(r[1])) + Decimal(str(r[2]))) / Decimal("2") for r in rows]
    return ts, mids


def _load_spot_trades(con: sqlite3.Connection) -> list[tuple[int, Decimal, Decimal, bool]]:
    out: list[tuple[int, Decimal, Decimal, bool]] = []
    for ts, price, raw in con.execute(
        "select exchange_ts_ms, price, raw_json from observations "
        "where source='spot_trade' and symbol='BTCUSDT' and exchange_ts_ms is not null "
        "order by exchange_ts_ms"
    ):
        try:
            inner = (json.loads(raw or "{}").get("raw") or {})
            m = bool(inner.get("m", True))
            q = inner.get("q") or "0"
        except Exception:
            continue
        out.append((int(ts), Decimal(str(price)), Decimal(str(q)), m))
    return out


def _markets(con: sqlite3.Connection, max_markets: int) -> list[tuple[int, int, int]]:
    rows = con.execute(
        "select market_id, max(resolution_ms) from observations "
        "where source='prediction_orderbook' and symbol='BTCUSDT' and duration='5m' "
        "and market_id is not null and resolution_ms is not null "
        "group by market_id order by market_id limit %d" % int(max_markets)
    ).fetchall()
    return [(int(m), int(r) - 300_000, int(r)) for m, r in rows]


def _synthetic_windows(bts: list[int], step_ms: int = 300_000) -> list[tuple[int, int, int]]:
    """5m-grid windows inside spot coverage (fallback; labeled synthetic).

    Used because recorded prediction-market windows do not overlap spot
    coverage in any existing DB (verified across all five datasets).
    mid = 0 marks synthetic (no venue market id).
    """
    if not bts:
        return []
    lo = bts[0] + 300_000
    hi = bts[-1] - 60_000
    lo = (lo // 300_000) * 300_000
    out = []
    t = lo
    while t + 300_000 <= hi:
        out.append((0, t, t + 300_000))
        t += step_ms
    return out


def _book_imb_any(con: sqlite3.Connection, t: int) -> Decimal | None:
    row = con.execute(
        "select raw_json from observations where source='prediction_orderbook' "
        "and symbol='BTCUSDT' and update_ts_ms is not null and update_ts_ms<=? "
        "order by update_ts_ms desc limit 1",
        (int(t),),
    ).fetchone()
    if not row or not row[0]:
        return None
    try:
        payload = json.loads(row[0])
        bids = payload.get("bids") or []
        asks = payload.get("asks") or []
        bv = sum(Decimal(str(x[1])) for x in bids[:5])
        av = sum(Decimal(str(x[1])) for x in asks[:5])
    except Exception:
        return None
    if bv + av <= 0:
        return None
    return (bv - av) / (bv + av)


def _latest_le(ts: list[int], vals: list, t: int):  # type: ignore[no-untyped-def]
    i = bisect.bisect_right(ts, int(t)) - 1
    return vals[i] if i >= 0 else None


def _book_imb(con: sqlite3.Connection, market_id: int, t: int) -> Decimal | None:
    row = con.execute(
        "select raw_json from observations where source='prediction_orderbook' "
        "and market_id=? and update_ts_ms is not null and update_ts_ms<=? "
        "order by update_ts_ms desc limit 1",
        (market_id, int(t)),
    ).fetchone()
    if not row or not row[0]:
        return None
    try:
        payload = json.loads(row[0])
        bids = payload.get("bids") or []
        asks = payload.get("asks") or []
        bv = sum(Decimal(str(x[1])) for x in bids[:5])
        av = sum(Decimal(str(x[1])) for x in asks[:5])
    except Exception:
        return None
    if bv + av <= 0:
        return None
    return (bv - av) / (bv + av)


def _features_for_market(con, bts, bmids, trades, tts, mid, start, t, off_cfg):  # type: ignore[no-untyped-def]
    mom_w, acc_w, flow_w = off_cfg
    ref = _latest_le(bts, bmids, start + 1000)
    now = _latest_le(bts, bmids, t)
    m1 = _latest_le(bts, bmids, t - mom_w)
    m2 = _latest_le(bts, bmids, t - acc_w)
    flow = None
    if tts:
        lo = bisect.bisect_right(tts, t - flow_w)
        hi = bisect.bisect_right(tts, t)
        buy = sell = Decimal("0")
        for _, _, q, m in trades[lo:hi]:
            if m:
                sell += q
            else:
                buy += q
        if buy + sell > 0:
            flow = (buy - sell) / (buy + sell)
    book = _book_imb(con, mid, t)
    end_mid = _latest_le(bts, bmids, start + 300_000 - 1000)
    label = None
    if ref and end_mid and ref > 0 and end_mid != ref:
        label = "UP" if end_mid > ref else "DOWN"
    feats = {"ref": ref, "now": now, "m1": m1, "m2": m2, "flow": flow, "book": book}
    return feats, label


def _bps(now: Decimal, base: Decimal) -> Decimal:
    return (now - base) / base * BPS


def _decide(feats: dict, drift_thr: Decimal, mom_thr: Decimal, flow_thr: Decimal, book_thr: Decimal) -> str:
    if feats["ref"] is None or feats["now"] is None:
        return "HOLD"
    drift = _bps(feats["now"], feats["ref"])
    votes = 0
    votes += 1 if drift >= drift_thr else (-1 if drift <= -drift_thr else 0)
    if feats["m1"] is not None:
        mom = _bps(feats["now"], feats["m1"])
        accel = _bps(feats["m1"], feats["m2"]) if feats["m2"] is not None else None
        mom = _bps(feats["now"], feats["m1"])
        if mom >= mom_thr and (accel is None or accel >= -Decimal("12")):
            votes += 1
        elif mom <= -mom_thr and (accel is None or accel <= Decimal("12")):
            votes -= 1
    if feats["flow"] is not None:
        votes += 1 if feats["flow"] >= flow_thr else (-1 if feats["flow"] <= -flow_thr else 0)
    if feats["book"] is not None:
        votes += 1 if feats["book"] >= book_thr else (-1 if feats["book"] <= -book_thr else 0)
    if votes >= 2:
        return "UP"
    if votes <= -2:
        return "DOWN"
    return "HOLD"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grid-search Up/Down 5m signal deadbands on existing DB (no trading)")
    ap.add_argument("--db", default="data/research/prediction_markets_multiday/collector.db")
    ap.add_argument("--max-markets", type=int, default=400)
    ap.add_argument("--offset-ms", type=int, default=90_000)
    ap.add_argument("--out", default="data/research/up_down_5m_thresholds.json")
    args = ap.parse_args(argv)
    t0 = time.time()
    con = sqlite3.connect(args.db)
    bts, bmids = _load_spot_books(con)
    trades = _load_spot_trades(con)
    tts = [t for t, _, _, _ in trades]
    markets = _markets(con, args.max_markets)
    smin, smax = (bts[0], bts[-1]) if bts else (0, 0)
    overlap = [(m, s, e) for m, s, e in markets if s >= smin and e <= smax + 60_000]
    mode = "real-markets"
    if not overlap:
        markets = _synthetic_windows(bts)[: args.max_markets]
        mode = "synthetic-5m-windows"
    else:
        markets = overlap
    print(f"mode={mode} windows={len(markets)}")

    mom_w, acc_w, flow_w = 30_000, 60_000, 60_000
    dataset = []
    for mid, start, end in markets:
        t = start + args.offset_ms
        feats, label = _features_for_market(con, bts, bmids, trades, tts, mid, start, t, (mom_w, acc_w, flow_w))
        if mode.startswith("synthetic"):
            feats = dict(feats)
            feats["book"] = _book_imb_any(con, t)
        if label is None or feats["now"] is None or feats["ref"] is None:
            continue
        dataset.append((mid, feats, label))
    print(f"labeled_markets={len(dataset)}")
    if not dataset:
        print("BLOCKER: no labeled markets (spot coverage gap around windows?)")
        return 2

    grid = []
    for d in ("5", "8", "10", "15", "20"):
        for m in ("5", "8", "10", "15"):
            for f in ("0.05", "0.10", "0.20"):
                for b in ("0.05", "0.08", "0.15"):
                    grid.append((Decimal(d), Decimal(m), Decimal(f), Decimal(b)))
    rows = []
    for dt, mt, ft, bt in grid:
        n = hit = 0
        for _, feats, label in dataset:
            sig = _decide(feats, dt, mt, ft, bt)
            if sig == "HOLD":
                continue
            n += 1
            hit += 1 if sig == label else 0
        acc = hit / n if n else 0.0
        rows.append({"drift": str(dt), "mom": str(mt), "flow": str(ft), "book": str(bt),
                     "traded": n, "coverage": n / len(dataset), "accuracy": acc})
    rows.sort(key=lambda r: (r["accuracy"], r["coverage"]), reverse=True)
    eligible = [r for r in rows if r["coverage"] >= 0.20] or rows
    best = eligible[0]
    print("top-10 (accuracy, coverage, traded, drift/mom/flow/book):")
    for r in rows[:10]:
        print(f"  acc={r['accuracy']:.3f} cov={r['coverage']:.3f} n={r['traded']} "
              f"d={r['drift']} m={r['mom']} f={r['flow']} b={r['book']}")
    print(f"BEST coverage>=20%: {best}")

    # reversal-exit check on winner: second decision at +180s
    rev = rev_ok = entered = 0
    for mid, feats, label in dataset:
        sig1 = _decide(feats, Decimal(best["drift"]), Decimal(best["mom"]), Decimal(best["flow"]), Decimal(best["book"]))
        if sig1 == "HOLD":
            continue
        entered += 1
        start = next(s for m, s, _ in markets if m == mid)
        t2 = start + 180_000
        feats2, _ = _features_for_market(con, bts, bmids, trades, tts, mid, start, t2, (mom_w, acc_w, flow_w))
        if feats2["now"] is None or feats2["ref"] is None:
            continue
        sig2 = _decide(feats2, Decimal(best["drift"]), Decimal(best["mom"]), Decimal(best["flow"]), Decimal(best["book"]))
        if (sig1 == "UP" and sig2 == "DOWN") or (sig1 == "DOWN" and sig2 == "UP"):
            rev += 1
            rev_ok += 1 if sig2 == label else 0
    print(f"entered={entered} reversals={rev} reversal_correct={rev_ok} "
          f"({'%.3f' % (rev_ok / rev) if rev else 'n/a'})")
    report = {
        "db": args.db, "mode": mode, "offset_ms": args.offset_ms, "labeled_markets": len(dataset),
        "elapsed_secs": round(time.time() - t0, 1),
        "label": "spot-drift proxy vs reference (NOT settlement; Chainlink history blocked)",
        "best": best, "top10": rows[:10],
        "reversal": {"entered": entered, "reversals": rev, "reversal_correct": rev_ok},
        "note": "UP/DOWN contract edge vs settlement unmeasurable here: prediction books lack "
                "outcome side labels and Chainlink history is inaccessible.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out} trading: none (research-only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
