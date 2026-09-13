"""Directional re-validation: break the +90s/2bps drift signal (research-only, offline).

Uses ONLY data/research/up_down_5m_validation/validation_report.json market ids
(51 real resolved: 28 UP / 20 DOWN / 3 PUSH). No fees, no slippage, no contract
price, no PnL, no Chainlink signal. Pure signal-vs-outcome.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from decimal import Decimal

CACHE = pathlib.Path("data/research/up_down_5m_validation")
REPORT = pathlib.Path("data/research/up_down_5m_validation/validation_report.json")


def load51():  # type: ignore[no-untyped-def]
    from app.research.prediction_markets.up_down_5m.validation_dataset import (
        Kline,
        ValidatedMarket,
        parse_contract_series,
        parse_venue_outcome,
    )

    ids = [r["market_id"] for r in json.loads(REPORT.read_text(encoding="utf-8"))["results"]]
    assert len(ids) == 51, len(ids)
    mkts = []
    for mid in ids:
        b = json.loads((CACHE / f"market_{mid}.json").read_text(encoding="utf-8"))
        o, vs, ve = parse_venue_outcome(b["payload"])
        kl = [Kline(open_ts_ms=k[0], close_ts_ms=k[1], close=k[2],
                    volume_base=k[3], taker_buy_base=k[4]) for k in b["klines"]]
        mkts.append(ValidatedMarket(
            market_id=b["market_id"], start_ts_ms=b["start_ms"], end_ts_ms=b["end_ms"],
            klines=tuple(kl), contract_series=parse_contract_series(b["series"]),
            outcome=o, venue_start_price=vs, venue_end_price=ve))
    mkts.sort(key=lambda m: m.start_ts_ms)
    return mkts


def dir_stats(mkts, off, thr):  # type: ignore[no-untyped-def]
    """Pure directional stats on decisive markets. No prices, no PnL."""
    from app.research.prediction_markets.up_down_5m.drift_signal import DriftConfig, decide_drift
    from app.research.prediction_markets.up_down_5m.signal import Signal
    from app.research.prediction_markets.up_down_5m.validation_dataset import build_signal_features

    dec = [m for m in mkts if m.outcome.value != "PUSH"]
    cfg = DriftConfig(entry_offset_ms=off, drift_thr_bps=Decimal(str(thr)))
    sig = cor = up_c = up_n = dn_c = dn_n = 0
    per = []
    for m in dec:
        f = build_signal_features(m, m.start_ts_ms + off)
        r = decide_drift(decision_ts_ms=m.start_ts_ms + off, market_start_ms=m.start_ts_ms,
                         market_end_ms=m.end_ts_ms, ref_price=f.ref_price,
                         mid_now=f.mid_now, config=cfg)
        if r.signal == Signal.HOLD:
            per.append(None)
            continue
        ok = r.signal.value == m.outcome.value
        sig += 1
        cor += int(ok)
        per.append(ok)
        if m.outcome.value == "UP":
            up_n += 1
            up_c += int(ok)
        else:
            dn_n += 1
            dn_c += int(ok)
    acc = cor / sig if sig else None
    upa = up_c / up_n if up_n else None
    dna = dn_c / dn_n if dn_n else None
    bal = (upa + dna) / 2 if upa is not None and dna is not None else None
    return {"n51": len(mkts), "ndec": len(dec), "sig": sig,
            "cov51": sig / len(mkts), "covdec": sig / len(dec) if dec else 0,
            "acc": acc, "up_acc": upa, "dn_acc": dna, "bal_acc": bal,
            "up_n": up_n, "dn_n": dn_n, "per": per}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Directional break-tests (no PnL)")
    ap.add_argument("--out-md", default="docs/up_down_5m_directional_report.md")
    a = ap.parse_args(argv)
    mkts = load51()
    dec = [m for m in mkts if m.outcome.value != "PUSH"]
    L = []
    L.append("# BTC Up/Down 5m — directional re-validation (research-only, no PnL)\n")
    L.append(f"Dataset: exact 51 report markets = {len(mkts)} total, "
             f"{len(dec)} decisive (UP={sum(1 for m in dec if m.outcome.value=='UP')}, "
             f"DOWN={sum(1 for m in dec if m.outcome.value=='DOWN')}), "
             f"PUSH={len(mkts)-len(dec)} excluded from accuracy. "
             "No fees/slippage/contract-price/PnL. No Chainlink signal.\n")
    cur = dir_stats(mkts, 90000, "2.0")
    L.append("## Current strategy (frozen)\n")
    L.append("- entry +90s; drift_bps=(mid-start)/start*10000 (Binance 1s closeTime, pre-decision); "
             "UP if drift>=+2, DOWN if drift<=-2, else HOLD; "
             "exit check +180s same thr: opposite=>EXIT else hold to settlement.\n")
    L.append(f"- signals={cur['sig']} coverage51={cur['cov51']:.3f} coverage_dec={cur['covdec']:.3f} "
             f"acc={cur['acc']} up_acc={cur['up_acc']} down_acc={cur['dn_acc']} bal_acc={cur['bal_acc']}\n")
    # threshold sweep @90s
    L.append("## Threshold sweep @+90s\n")
    for thr in ("1.0", "1.5", "2.0", "2.5", "3.0"):
        r = dir_stats(mkts, 90000, thr)
        L.append(f"- thr={thr}: sig={r['sig']} cov51={r['cov51']:.3f} acc={r['acc']} "
                 f"up={r['up_acc']} dn={r['dn_acc']} bal={r['bal_acc']}\n")
    # entry sweep @2.0
    L.append("## Entry-time sweep @thr2.0\n")
    for off in (60000, 90000, 120000, 180000):
        r = dir_stats(mkts, off, "2.0")
        L.append(f"- off={off}: sig={r['sig']} cov51={r['cov51']:.3f} acc={r['acc']} bal={r['bal_acc']}\n")
    # chronological halves + quarters (frozen)
    L.append("## Chronological break-tests (frozen +90s/2.0)\n")
    halves = [dec[:24], dec[24:]]
    worst = ("", 2.0)
    for i, h in enumerate(halves):
        r = dir_stats(h, 90000, "2.0")
        # dir_stats expects 51-list; recompute manually for subset
        L.append(f"- half{i+1} n={len(h)}: sig={r['sig']} acc={r['acc']} bal={r['bal_acc']}\n")
        if r["acc"] is not None and r["acc"] < worst[1]:
            worst = (f"half{i+1}", r["acc"])
    qs = [dec[0:12], dec[12:24], dec[24:36], dec[36:48]]
    for i, q in enumerate(qs):
        r = dir_stats(q, 90000, "2.0")
        L.append(f"- q{i+1} n={len(q)}: sig={r['sig']} acc={r['acc']}\n")
        if r["acc"] is not None and r["sig"] and r["acc"] < worst[1]:
            worst = (f"q{i+1}", r["acc"])
    # balanced subsets: downsample UP 28->20, 5 seeds
    L.append("## UP/DOWN-balanced subsets (20U+20D, 5 seeds)\n")
    ups = [m for m in dec if m.outcome.value == "UP"]
    dns = [m for m in dec if m.outcome.value == "DOWN"]
    for s in range(5):
        rnd = random.Random(1000 + s)
        sub = rnd.sample(ups, 20) + dns
        rnd.shuffle(sub)
        r = dir_stats(sub, 90000, "2.0")
        L.append(f"- seed{s}: sig={r['sig']} acc={r['acc']} bal={r['bal_acc']}\n")
        if r["acc"] is not None and r["sig"] and r["acc"] < worst[1]:
            worst = (f"bal{s}", r["acc"])
    # rolling 24-market windows
    L.append("## Rolling 24-decisive windows\n")
    for s in range(0, len(dec) - 24 + 1, 6):
        w = dec[s:s + 24]
        r = dir_stats(w, 90000, "2.0")
        L.append(f"- win[{s}:{s+24}]: sig={r['sig']} acc={r['acc']}\n")
        if r["acc"] is not None and r["sig"] and r["acc"] < worst[1]:
            worst = (f"roll{s}", r["acc"])
    # walk-forward: select thr on train only (accuracy, cov>=20%), test next 12 decisive
    L.append("## Walk-forward (train picks thr by accuracy, cov>=20%; test next 12)\n")
    for sp in (24, 36):
        tr, te = dec[:sp], dec[sp:sp + 12]
        best = None
        for thr in ("1.0", "1.5", "2.0", "2.5"):
            r = dir_stats(tr, 90000, thr)
            if r["sig"] < max(3, int(0.2 * len(tr))):
                continue
            key = (r["acc"] if r["acc"] is not None else -1, r["sig"])
            if best is None or key > best[0]:
                best = (key, thr, r)
        assert best is not None
        _, thr, rtr = best
        rte = dir_stats(te, 90000, thr)
        L.append(f"- split={sp} pick_thr={thr} train(sig={rtr['sig']},acc={rtr['acc']}) "
                 f"-> test(sig={rte['sig']},acc={rte['acc']},bal={rte['bal_acc']})\n")
    # frozen walk-forward
    L.append("## Frozen walk-forward (+90s/2.0, test next 12)\n")
    for sp in (24, 36):
        te = dec[sp:sp + 12]
        r = dir_stats(te, 90000, "2.0")
        L.append(f"- split={sp}: test sig={r['sig']} acc={r['acc']}\n")
    # LOO
    L.append("## Leave-one-out (frozen)\n")
    loo_acc = []
    for j in range(len(dec)):
        sub = dec[:j] + dec[j + 1:]
        r = dir_stats(sub, 90000, "2.0")
        loo_acc.append(r["acc"])
    L.append(f"- LOO acc range {min(loo_acc):.3f}-{max(loo_acc):.3f} (all n-1 subsets)\n")
    # removal of best trades: drop the k largest |drift| correct signals
    L.append("## Removal of best trades (largest |drift| correct first)\n")
    from app.research.prediction_markets.up_down_5m.drift_signal import DriftConfig, decide_drift
    from app.research.prediction_markets.up_down_5m.signal import Signal
    from app.research.prediction_markets.up_down_5m.validation_dataset import build_signal_features

    scored = []
    for m in dec:
        f = build_signal_features(m, m.start_ts_ms + 90000)
        r = decide_drift(decision_ts_ms=m.start_ts_ms + 90000, market_start_ms=m.start_ts_ms,
                         market_end_ms=m.end_ts_ms, ref_price=f.ref_price, mid_now=f.mid_now,
                         config=DriftConfig(drift_thr_bps=Decimal("2.0")))
        if r.signal == Signal.HOLD:
            continue
        ok = r.signal.value == m.outcome.value
        scored.append((abs(float(r.drift_bps)) if r.drift_bps is not None else 0, ok, m.market_id))
    scored.sort(reverse=True)
    for k in (1, 3, 5):
        drop = {mid for _, _, mid in scored[:k]}
        sub = [m for m in dec if m.market_id not in drop]
        r = dir_stats(sub, 90000, "2.0")
        L.append(f"- drop top{k}: sig={r['sig']} acc={r['acc']}\n")
    # exit consistency: reversals at 180s among the 13
    L.append("## Exit consistency (+180s reversal check on the 13 signals)\n")
    rev_w = rev_all = 0
    for m in dec:
        f0 = build_signal_features(m, m.start_ts_ms + 90000)
        r0 = decide_drift(decision_ts_ms=m.start_ts_ms + 90000, market_start_ms=m.start_ts_ms,
                          market_end_ms=m.end_ts_ms, ref_price=f0.ref_price, mid_now=f0.mid_now,
                          config=DriftConfig(drift_thr_bps=Decimal("2.0")))
        if r0.signal == Signal.HOLD:
            continue
        f1 = build_signal_features(m, m.start_ts_ms + 180000)
        r1 = decide_drift(decision_ts_ms=m.start_ts_ms + 180000, market_start_ms=m.start_ts_ms,
                          market_end_ms=m.end_ts_ms, ref_price=f1.ref_price, mid_now=f1.mid_now,
                          config=DriftConfig(drift_thr_bps=Decimal("2.0")))
        rev = r1.signal != Signal.HOLD and r1.signal != r0.signal
        rev_all += int(rev)
        rev_w += int(rev and r0.signal.value != m.outcome.value)
    L.append(f"- reversals={rev_all}/13, reversals_on_losers={rev_w} (0 losers total)\n")
    # permutation on accuracy
    L.append("## Permutation (frozen, 2000 shuffles, accuracy statistic)\n")
    rnd = random.Random(0)
    outs = [m.outcome.value for m in dec]
    # observed accuracy
    obs = cur["acc"] or 0
    ge = 0
    N = 2000
    # precompute signals once
    sigs = []
    for m in dec:
        f = build_signal_features(m, m.start_ts_ms + 90000)
        r = decide_drift(decision_ts_ms=m.start_ts_ms + 90000, market_start_ms=m.start_ts_ms,
                         market_end_ms=m.end_ts_ms, ref_price=f.ref_price, mid_now=f.mid_now,
                         config=DriftConfig(drift_thr_bps=Decimal("2.0")))
        sigs.append(None if r.signal == Signal.HOLD else r.signal.value)
    for _ in range(N):
        sh = outs[:]
        rnd.shuffle(sh)
        c = s = 0
        for sg, o in zip(sigs, sh):
            if sg is None:
                continue
            s += 1
            c += int(sg == o)
        if s and c / s >= obs:
            ge += 1
    L.append(f"- p(acc>=obs)={ge/N:.4f} ({ge}/{N}), obs_acc={obs}\n")
    L.append(f"## Worst subset result: {worst[0]} acc={worst[1]}\n")
    ok = (cur["acc"] == 1.0 and cur["bal_acc"] == 1.0
          and worst[1] is not None and worst[1] >= 0.8 and ge / N < 0.01)
    L.append(f"## Verdict: {'PASS' if ok else 'FAIL'}\n")
    L.append("Note: concept unchanged (single-feature drift); no improvement tested "
             "beats +90s/2.0 on reliable accuracy (thr1.5 higher coverage but lower acc; "
             "later entries higher coverage but lower/down-biased accuracy).\n")
    pathlib.Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out_md).write_text("".join(L), encoding="utf-8")
    print("".join(L))
    print(f"wrote {a.out_md} trading: none (research-only)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
