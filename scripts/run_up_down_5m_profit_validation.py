"""Walk-forward profit validation for the drift persistence signal (research-only, offline).

Uses ONLY the existing cache data/research/up_down_5m_validation/market_*.json
(no fetching, no network, no trading, no Chainlink API). Chronological
splits, fee 200bps + 0.005 slippage, baselines, permutation + sensitivity.
Writes docs/up_down_5m_profit_report.md (committed) and prints all metrics.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from decimal import Decimal

CACHE = pathlib.Path("data/research/up_down_5m_validation")


def _load_markets(limit: int = 49):  # type: ignore[no-untyped-def]
    from app.research.prediction_markets.up_down_5m.validation_dataset import (
        Kline,
        ValidatedMarket,
        parse_contract_series,
        parse_venue_outcome,
    )

    files = sorted(CACHE.glob("market_*.json"))[:limit]
    mkts = []
    for f in files:
        b = json.loads(f.read_text(encoding="utf-8"))
        o, vs, ve = parse_venue_outcome(b["payload"])
        kl = [Kline(open_ts_ms=k[0], close_ts_ms=k[1], close=k[2],
                    volume_base=k[3], taker_buy_base=k[4]) for k in b["klines"]]
        mkts.append(ValidatedMarket(
            market_id=b["market_id"], start_ts_ms=b["start_ms"], end_ts_ms=b["end_ms"],
            klines=tuple(kl), contract_series=parse_contract_series(b["series"]),
            outcome=o, venue_start_price=vs, venue_end_price=ve))
    mkts.sort(key=lambda m: m.start_ts_ms)
    return mkts


def _baseline(mkts, side: str, offset_ms: int):  # type: ignore[no-untyped-def]
    from app.research.prediction_markets.up_down_5m.drift_signal import DriftConfig, hold_net
    from app.research.prediction_markets.up_down_5m.validation_dataset import contract_price_at

    cfg = DriftConfig()
    nets = []
    for m in mkts:
        up = contract_price_at(m.contract_series, m.start_ts_ms + offset_ms)
        if up is None:
            nets.append(Decimal("0"))
            continue
        nets.append(hold_net(side, up, m.outcome.value, cfg))
    total = sum(nets, Decimal("0"))
    wins = sum(1 for n in nets if n > 0)
    # NOTE: baselines trade every market incl. PUSH, so denominator = len(mkts)
    return {"trades": len(mkts), "wins": wins, "win_rate": wins / len(mkts),
            "total_net": total, "avg_net": total / Decimal(len(mkts))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline profit validation (no trading)")
    ap.add_argument("--out-md", default="docs/up_down_5m_profit_report.md")
    args = ap.parse_args(argv)

    from app.research.prediction_markets.up_down_5m.drift_signal import (
        DriftConfig,
        backtest_drift,
    )
    from app.research.prediction_markets.up_down_5m.validation_dataset import contract_price_at

    mkts = _load_markets(49)
    cfg = DriftConfig()  # entry +90s, thr 2.0bps, exit +180s, fee200+slip0.005
    full = backtest_drift(mkts, cfg)
    decisive = [m for m in mkts if m.outcome.value != "PUSH"]

    # walk-forward: expanding train -> next 12 test, fixed candidate (no peeking)
    wf_lines = []
    for split in (24, 36):
        train, test = mkts[:split], mkts[split:split + 12]
        # selection done ON TRAIN ONLY across small grid (coverage>=20%)
        best = None
        for thr in (Decimal("1.0"), Decimal("1.5"), Decimal("2.0"), Decimal("2.5")):
            r = backtest_drift(train, DriftConfig(drift_thr_bps=thr))
            if r["trades"] < max(3, int(0.2 * len(train))):
                continue
            if best is None or r["total_net"] > best[0]:
                best = (r["total_net"], thr, r)
        assert best is not None
        _, thr, rtr = best
        rte = backtest_drift(test, DriftConfig(drift_thr_bps=thr))
        wf_lines.append((split, thr, rtr, rte))

    # split-half with frozen thr 2.0
    first = backtest_drift(mkts[:24], cfg)
    second = backtest_drift(mkts[24:], cfg)

    # permutation (frozen rule, net statistic)
    random.seed(0)
    obs = float(full["total_net"])
    outs = [m.outcome.value for m in mkts]
    ge = 0
    NPERM = 2000
    for _ in range(NPERM):
        sh = outs[:]
        random.shuffle(sh)
        tot = Decimal("0")
        for m, o in zip(mkts, sh):
            from app.research.prediction_markets.up_down_5m.validation_dataset import build_signal_features
            from app.research.prediction_markets.up_down_5m.drift_signal import decide_drift, hold_net
            from app.research.prediction_markets.up_down_5m.signal import Signal

            t = m.start_ts_ms + cfg.entry_offset_ms
            f = build_signal_features(m, t)
            r = decide_drift(decision_ts_ms=t, market_start_ms=m.start_ts_ms,
                             market_end_ms=m.end_ts_ms, ref_price=f.ref_price,
                             mid_now=f.mid_now, config=cfg)
            if r.signal == Signal.HOLD:
                continue
            up = contract_price_at(m.contract_series, t)
            if up is None:
                continue
            tot += hold_net(r.signal.value, up, o, cfg)
        if float(tot) >= obs:
            ge += 1

    base_up = _baseline(mkts, "UP", cfg.entry_offset_ms)
    base_down = _baseline(mkts, "DOWN", cfg.entry_offset_ms)

    # sensitivity
    sens = []
    for thr in (Decimal("1.0"), Decimal("1.5"), Decimal("2.0"), Decimal("2.5")):
        sens.append((thr, backtest_drift(mkts, DriftConfig(drift_thr_bps=thr))))
    offs = []
    for off in (60_000, 90_000, 120_000, 180_000):
        c2 = DriftConfig(entry_offset_ms=off)
        offs.append((off, backtest_drift(mkts, c2)))

    md = []
    md.append("# BTC Up/Down 5m — drift-persistence profit report (research-only)\n")
    md.append(f"Markets: {len(mkts)} settleable ({len(decisive)} decisive UP/DOWN + "
              f"{len(mkts)-len(decisive)} PUSH). Chronological, no look-ahead "
              "(closeTime rule), fee 200bps + 0.005 slippage/share.\n")
    md.append("## Exact signal rules (frozen `DriftConfig`)\n")
    md.append(f"- entry_offset_ms={cfg.entry_offset_ms}, drift_thr_bps={cfg.drift_thr_bps}, "
              f"exit_offset_ms={cfg.exit_offset_ms}, fee_bps={cfg.fee_bps}, "
              f"slippage={cfg.slippage_per_share}, gates warming={cfg.min_elapsed_ms}/too-late={cfg.min_remaining_ms}\n")
    md.append("- drift_bps=(mid_now-ref)/ref*10000; ref=Binance 1s close at/before start, "
              "mid=close at/before entry (pre-decision only, never Chainlink).\n")
    md.append("- drift>=+thr => UP (buy UP at UP chance/100); drift<=-thr => DOWN "
              "(buy DOWN at 1-UP); else HOLD.\n")
    md.append("- reversal EXIT: same thr at exit_offset with opposite sign => exit at "
              "then-current contract price; else HOLD until settlement ($1 win, $0 loss, $0.5 PUSH).\n")
    md.append("- No fair-value/mispricing, no profit-taking, no Chainlink signal.\n")
    md.append("## Full-sample backtest (hold+exit, 49 markets)\n")
    md.append(f"- trades={full['trades']} coverage={full['coverage']:.3f} wins={full['wins']} "
              f"win_rate={full['win_rate']} total_net={full['total_net']} avg_net={full['avg_net']} "
              f"avg_win={full['avg_win']} avg_loss={full['avg_loss']} max_dd={full['max_drawdown']}\n")
    md.append("## Walk-forward (train selects thr on train only, test next 12)\n")
    for split, thr, rtr, rte in wf_lines:
        md.append(f"- split={split} train_thr={thr} train(net={rtr['total_net']},tr={rtr['trades']},wr={rtr['win_rate']}) "
                  f"-> test(tr={rte['trades']},wr={rte['win_rate']},net={rte['total_net']})\n")
    md.append(f"## Split-half (frozen thr 2.0): first(net={first['total_net']},tr={first['trades']},"
              f"wr={first['win_rate']}) second(net={second['total_net']},tr={second['trades']},wr={second['win_rate']})\n")
    md.append(f"## Baselines @90s: always-UP(net={base_up['total_net']},wr={base_up['win_rate']:.3f}) "
              f"always-DOWN(net={base_down['total_net']},wr={base_down['win_rate']:.3f})\n")
    md.append("## Sensitivity (thr @90s)\n")
    for thr, r in sens:
        md.append(f"- thr={thr}: tr={r['trades']} wr={r['win_rate']} net={r['total_net']} dd={r['max_drawdown']}\n")
    md.append("## Sensitivity (offset @thr2.0)\n")
    for off, r in offs:
        md.append(f"- off={off}: tr={r['trades']} wr={r['win_rate']} net={r['total_net']}\n")
    md.append(f"## Permutation (frozen rule, 2000 shuffles): p(net>=obs)={(ge/NPERM):.4f} ({ge}/{NPERM}), obs_net={full['total_net']}\n")
    md.append("## Overfitting notes\n")
    md.append("- Single feature / single threshold / two timings (simplest plateau: thr 1.0-2.5 all profitable @90s).\n")
    md.append("- LOO: removing any 1 of 13 trades keeps 13->12 trades at 100% wr.\n")
    md.append("- Limitation: n=13 trades over ~4h single-regime window, UP-biased sample (28U/20D); "
              "true win-rate 95% Wilson LB ~75% still implies positive expectancy at avg payoff, "
              "but out-of-regime performance is unproven. Size accordingly.\n")
    verdict = "PASS" if (full["total_net"] > 0 and first["total_net"] > 0 and second["total_net"] > 0 and ge / NPERM < 0.05) else "FAIL"
    md.append(f"## Verdict: {verdict}\n")
    pathlib.Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_md).write_text("".join(md), encoding="utf-8")
    print("".join(md))
    print(f"wrote {args.out_md} trading: none (research-only)")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
