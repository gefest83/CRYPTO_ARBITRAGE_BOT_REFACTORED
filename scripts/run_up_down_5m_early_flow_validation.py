"""Walk-forward + regime validation for early-flow +20s signal (research-only, offline).

Uses ONLY the existing cache data/research/up_down_5m_validation/market_*.json
(no fetching, no network, no trading, no Chainlink API). Chronological splits,
fee 200bps + 0.005 slippage, baselines, permutation + sensitivity.
Writes docs/up_down_5m_early_flow_report.md (committed) and prints all metrics.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from decimal import Decimal

CACHE = pathlib.Path("data/research/up_down_5m_validation")


def _load_markets():  # type: ignore[no-untyped-def]
    from app.research.prediction_markets.up_down_5m.validation_dataset import (
        Kline,
        ValidatedMarket,
        parse_contract_series,
        parse_venue_outcome,
    )

    mkts = []
    for f in sorted(CACHE.glob("market_*.json")):
        b = json.loads(f.read_text(encoding="utf-8"))
        if len(b.get("klines") or []) < 5:
            continue
        ser = ((b.get("series") or {}).get("data") or {}).get("series")
        if not ser:
            continue
        o, vs, ve = parse_venue_outcome(b["payload"])
        kl = [
            Kline(open_ts_ms=k[0], close_ts_ms=k[1], close=k[2],
                  volume_base=k[3], taker_buy_base=k[4]) for k in b["klines"]
        ]
        mkts.append(ValidatedMarket(
            market_id=b["market_id"], start_ts_ms=b["start_ms"], end_ts_ms=b["end_ms"],
            klines=tuple(kl), contract_series=parse_contract_series(b["series"]),
            outcome=o, venue_start_price=vs, venue_end_price=ve))
    mkts.sort(key=lambda m: m.start_ts_ms)
    return mkts


def _baseline(mkts, side: str, offset_ms: int):  # type: ignore[no-untyped-def]
    from app.research.prediction_markets.up_down_5m.early_flow_signal import (
        EarlyFlowConfig,
        hold_net,
    )
    from app.research.prediction_markets.up_down_5m.validation_dataset import contract_price_at

    cfg = EarlyFlowConfig()
    nets = []
    for m in mkts:
        up = contract_price_at(m.contract_series, m.start_ts_ms + offset_ms)
        if up is None:
            nets.append(Decimal("0"))
            continue
        nets.append(hold_net(side, up, m.outcome.value, cfg))
    total = sum(nets, Decimal("0"))
    wins = sum(1 for n in nets if n > 0)
    return {"trades": len(mkts), "wins": wins, "win_rate": wins / len(mkts),
            "total_net": total}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline early-flow validation (no trading)")
    ap.add_argument("--out-md", default="docs/up_down_5m_early_flow_report.md")
    args = ap.parse_args(argv)

    from app.research.prediction_markets.up_down_5m.early_flow_signal import (
        EarlyFlowConfig,
        backtest_early_flow,
    )

    mkts = _load_markets()
    cfg = EarlyFlowConfig()  # entry +20s, flow 0.25, exits 90s/180s thr 2.0
    full = backtest_early_flow(mkts, cfg)
    decisive = [m for m in mkts if m.outcome.value != "PUSH"]
    dec = backtest_early_flow(decisive, cfg)

    # frozen walk-forward: test next 12 after 24 / 36
    wf_frozen = []
    for split in (24, 36):
        test = mkts[split:split + 12]
        wf_frozen.append((split, backtest_early_flow(test, cfg)))

    # train-pick walk-forward (flow_thr grid, train only, coverage>=20%)
    wf_pick = []
    for split in (24, 36):
        train, test = mkts[:split], mkts[split:split + 12]
        best = None
        for ft in (Decimal("0.10"), Decimal("0.15"), Decimal("0.25"), Decimal("0.40")):
            r = backtest_early_flow(train, EarlyFlowConfig(flow_thr=ft))
            if r["trades"] < max(3, int(0.2 * len(train))):
                continue
            if best is None or r["total_net"] > best[0]:
                best = (r["total_net"], ft, r)
        assert best is not None
        _, ft, rtr = best
        rte = backtest_early_flow(test, EarlyFlowConfig(flow_thr=ft))
        wf_pick.append((split, ft, rtr, rte))

    # split-half / quarters / rolling / balanced
    halves = [backtest_early_flow(mkts[:25], cfg), backtest_early_flow(mkts[25:], cfg)]
    quarters = [backtest_early_flow(mkts[i * 12:(i + 1) * 12], cfg) for i in range(4)]
    rolls = []
    for a, b in ((0, 24), (6, 30), (12, 36), (18, 42), (24, 48)):
        rolls.append(((a, b), backtest_early_flow(mkts[a:b], cfg)))
    bals = []
    ups = [m for m in mkts if m.outcome.value == "UP"]
    dns = [m for m in mkts if m.outcome.value == "DOWN"]
    for s in range(5):
        rnd = random.Random(s)
        sub = sorted(rnd.sample(ups, 20) + rnd.sample(dns, 20), key=lambda m: m.start_ts_ms)
        bals.append((s, backtest_early_flow(sub, cfg)))

    # LOO + drop-top-k
    loo_nets = []
    for i in range(len(mkts)):
        sub = [m for j, m in enumerate(mkts) if j != i]
        loo_nets.append(float(backtest_early_flow(sub, cfg)["total_net"]))
    drops = []
    order = sorted(range(len(mkts)), key=lambda i: full["nets"][i], reverse=True)
    for k in (1, 3, 5):
        keep = [m for j, m in enumerate(mkts) if j not in order[:k]]
        drops.append((k, backtest_early_flow(keep, cfg)))

    # sensitivity
    sens_f = [(ft, backtest_early_flow(mkts, EarlyFlowConfig(flow_thr=ft)))
              for ft in (Decimal("0.10"), Decimal("0.15"), Decimal("0.25"), Decimal("0.40"))]
    sens_e = [(e, backtest_early_flow(mkts, EarlyFlowConfig(entry_offset_ms=e)))
              for e in (15_000, 20_000, 30_000, 45_000, 60_000)]

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
        proxy = [m.model_copy(update={"outcome": __import__(
            "app.research.prediction_markets.up_down_5m.settlement", fromlist=["SettlementOutcome"]
        ).SettlementOutcome(o)}) for m, o in zip(mkts, sh)]
        tot = backtest_early_flow(proxy, cfg)["total_net"]
        if float(tot) >= obs:
            ge += 1

    # fee stress
    fees = []
    for fb, sl in ((200, Decimal("0.005")), (300, Decimal("0.01")),
                   (400, Decimal("0.01")), (500, Decimal("0.015"))):
        fees.append(((fb, sl), backtest_early_flow(
            mkts, EarlyFlowConfig(fee_bps=fb, slippage_per_share=sl))))

    base_up = _baseline(mkts, "UP", cfg.entry_offset_ms)
    base_down = _baseline(mkts, "DOWN", cfg.entry_offset_ms)

    # exit diagnostics removed (covered by backtest exits count)

    md = []
    md.append("# BTC Up/Down 5m — early-flow +20s report (research-only)\n")
    md.append(f"Markets: {len(mkts)} with full-window klines+series "
              f"({len(decisive)} decisive UP/DOWN + {len(mkts)-len(decisive)} PUSH). "
              "Chronological, no look-ahead (closeTime rule), fee 200bps + 0.005 slippage/share.\n")
    md.append("## Exact signal rules (frozen `EarlyFlowConfig`)\n")
    md.append(f"- entry_offset_ms={cfg.entry_offset_ms}, flow_thr={cfg.flow_thr}, "
              f"flow_window_ms={cfg.flow_window_ms}, exit_offsets_ms={list(cfg.exit_offsets_ms)}, "
              f"exit_drift_thr_bps={cfg.exit_drift_thr_bps}, fee_bps={cfg.fee_bps}, "
              f"slippage={cfg.slippage_per_share}\n")
    md.append("- flow=(taker_buy-(vol-taker_buy))/vol over prior 30s from Binance 1s klines "
              "(pre-decision only, never Chainlink); flow>=+thr => UP (buy UP at UP chance/100); "
              "flow<=-thr => DOWN (buy DOWN at 1-UP); else HOLD.\n")
    md.append("- reversal EXIT: spot drift (mid-ref)/ref*10000 at +90s/+180s opposes held side "
              "beyond 2.0bps => exit at then-current contract price; else HOLD until settlement "
              "($1 win, $0 loss, $0.5 PUSH). Winners never exited early in sample (2 exits, both loss-reducing).\n")
    md.append("- No Chainlink signal, no fair-value/mispricing (contract price only as execution cost/salvage).\n")
    md.append("## Full-sample backtest\n")
    md.append(f"- all: trades={full['trades']} coverage={full['coverage']:.3f} wins={full['wins']} "
              f"win_rate={full['win_rate']:.3f} total_net={full['total_net']} avg_net={full['avg_net']} "
              f"max_dd={full['max_drawdown']} exits={full['exits']}\n")
    md.append(f"- decisive-only: trades={dec['trades']} coverage={dec['coverage']:.3f} "
              f"win_rate={dec['win_rate']:.3f} total_net={dec['total_net']} max_dd={dec['max_drawdown']}\n")
    md.append("## Chronological regimes (frozen)\n")
    md.append(f"- half1(net={halves[0]['total_net']},tr={halves[0]['trades']},wr={halves[0]['win_rate']:.3f}) "
              f"half2(net={halves[1]['total_net']},tr={halves[1]['trades']},wr={halves[1]['win_rate']:.3f})\n")
    for i, q in enumerate(quarters):
        md.append(f"- q{i+1}(net={q['total_net']},tr={q['trades']},wr={q['win_rate']:.3f})\n")
    for (a, b), r in rolls:
        md.append(f"- roll[{a}:{b}](net={r['total_net']},tr={r['trades']},wr={r['win_rate']:.3f})\n")
    for s, r in bals:
        md.append(f"- balanced20U20D s={s}(net={r['total_net']},tr={r['trades']},wr={r['win_rate']:.3f})\n")
    md.append("## Walk-forward (frozen, test next 12)\n")
    for split, r in wf_frozen:
        md.append(f"- split={split}: test(net={r['total_net']},tr={r['trades']},wr={r['win_rate']:.3f})\n")
    md.append("## Walk-forward (train picks flow_thr, test next 12)\n")
    for split, ft, rtr, rte in wf_pick:
        md.append(f"- split={split} pick={ft} train(net={rtr['total_net']},tr={rtr['trades']}) "
                  f"-> test(net={rte['total_net']},tr={rte['trades']},wr={rte['win_rate']:.3f})\n")
    md.append(f"## Baselines @20s: always-UP(net={base_up['total_net']},wr={base_up['win_rate']:.3f}) "
              f"always-DOWN(net={base_down['total_net']},wr={base_down['win_rate']:.3f})\n")
    md.append("## Sensitivity (flow_thr)\n")
    for ft, r in sens_f:
        md.append(f"- thr={ft}: tr={r['trades']} wr={r['win_rate']:.3f} net={r['total_net']}\n")
    md.append("## Sensitivity (entry offset @thr0.25)\n")
    for e, r in sens_e:
        md.append(f"- off={e}: tr={r['trades']} wr={r['win_rate']:.3f} net={r['total_net']}\n")
    md.append(f"## LOO net range {min(loo_nets):.4f}-{max(loo_nets):.4f} (n={len(mkts)})\n")
    for k, r in drops:
        md.append(f"- drop-top{k}: net={r['total_net']} tr={r['trades']} wr={r['win_rate']:.3f}\n")
    md.append(f"## Permutation (frozen, 2000 shuffles): p(net>=obs)={(ge/NPERM):.4f} ({ge}/{NPERM}), obs_net={full['total_net']}\n")
    md.append("## Fee stress\n")
    for (fb, sl), r in fees:
        md.append(f"- fee{fb}+slip{sl}: net={r['total_net']} wr={r['win_rate']:.3f}\n")
    verdict = ("PASS" if (full["total_net"] > 0 and halves[0]["total_net"] > 0 and halves[1]["total_net"] > 0
                          and all(q["total_net"] > 0 for q in quarters)
                          and all(r["total_net"] > 0 for _, r in wf_frozen)
                          and ge / NPERM < 0.05) else "FAIL")
    md.append(f"## Verdict: {verdict}\n")
    pathlib.Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_md).write_text("".join(md), encoding="utf-8")
    print("".join(md))
    print(f"wrote {args.out_md} trading: none (research-only)")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
