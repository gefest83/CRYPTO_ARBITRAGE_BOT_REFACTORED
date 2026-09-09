"""Phase 3B CLI – research-only historical backtest for Kronos-mini.

Uses real NeoQuasar/Kronos-mini + Tokenizer-2k (lazy, isolated).
Fetches real Binance 1m OHLCV for BTC/USDT, ETH/USDT, SOL/USDT.
Walk-forward, no lookahead, enter t+1 at next open, hold 5m, realistic costs.

Usage:
  py -3.13 scripts/run_kronos_backtest.py --days 7
  py -3.13 scripts/run_kronos_backtest.py --days 7 --symbols BTC/USDT ETH/USDT --stride 5
  py -3.13 scripts/run_kronos_backtest.py --days 7 --use-mock  (no torch/HF, for CI)
  py -3.13 scripts/run_kronos_backtest.py --days 14 --output data/kronos/backtest_14d.json

Output: console table + machine-readable JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

# Ensure workspace root
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from app.strategies.kronos.backtest import BacktestConfig, BacktestEngine, SignalGenerator
from app.strategies.kronos.data import ensure_cache_dir, load_or_fetch_candles

UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kronos-mini Phase 3B backtest (research-only)")
    p.add_argument("--days", type=int, default=7, help="historical days to fetch (7-14 recommended for initial, up to 30)")
    p.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"], help="symbols")
    p.add_argument("--stride", type=int, default=5, help="bars between decisions (5=no overlap, 1=every minute)")
    p.add_argument("--lookback", type=int, default=400, help="1m lookback for Kronos")
    p.add_argument("--min-edge", type=str, default="15", help="min edge bps (e.g. 15)")
    p.add_argument("--taker", type=str, default="10", help="taker fee bps")
    p.add_argument("--spread", type=str, default="5", help="spread bps")
    p.add_argument("--slippage", type=str, default="5", help="slippage bps")
    p.add_argument("--cost-model", type=str, default="round_trip", choices=["one_way", "round_trip"])
    p.add_argument("--end", type=str, default=None, help="end datetime ISO (default now, UTC)")
    p.add_argument("--use-mock", action="store_true", help="use MockKronosPredictor instead of real model (no torch/HF, deterministic)")
    p.add_argument("--output", type=str, default="data/kronos/backtest_result.json", help="JSON output path")
    p.add_argument("--no-cache", action="store_true", help="ignore disk cache, refetch")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    days = args.days
    symbols = [s.strip().upper() for s in args.symbols]
    stride = args.stride
    lookback = args.lookback

    # Determine time window: last `days` days up to `end` (or now, truncated to minute)
    if args.end:
        end = datetime.fromisoformat(args.end)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
    else:
        end = datetime.now(UTC).replace(second=0, microsecond=0)
        # Align to minute boundary and ensure complete candle
        end = end - timedelta(minutes=1)
    start = end - timedelta(days=days)
    # Align start to minute
    start = start.replace(second=0, microsecond=0)

    print(f"Phase 3B backtest – {days}d {symbols}  {start.isoformat()} -> {end.isoformat()}  stride={stride}")
    print(f"Costs: taker {args.taker}bps spread {args.spread}bps slippage {args.slippage}bps model {args.cost_model}  edge {args.min_edge}bps")
    if args.use_mock:
        print("Using MockKronosPredictor (no torch/HF)")

    cfg = BacktestConfig(
        lookback_1m=lookback,
        lookback_5m=max(10, lookback // 5),
        pred_len_1m=5,
        pred_len_5m=1,
        horizon_bars=5,
        stride_bars=stride,
        min_edge_bps=Decimal(args.min_edge),
        taker_fee_bps=Decimal(args.taker),
        spread_bps=Decimal(args.spread),
        slippage_bps=Decimal(args.slippage),
        cost_model=args.cost_model,
    )

    # Load predictor (real or mock) – lazy
    predictor = None
    if not args.use_mock:
        from app.strategies.kronos.real import RealKronosPredictor

        predictor = RealKronosPredictor(variant="mini", T=1.0, top_p=0.9, top_k=0, sample_count=1)
        print("Loading Kronos-mini (NeoQuasar/Kronos-mini + Tokenizer-2k)...")
        t0 = time.perf_counter()
        try:
            init_s = predictor.load()
        except Exception as exc:
            print(f"FAILED to load Kronos-mini: {exc}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            sys.exit(2)
        print(f"  loaded in {init_s:.1f}s  device={predictor.device}")
    else:
        from app.strategies.kronos.forecast import MockKronosPredictor

        # Deterministic mock with small drift to generate some signals
        predictor = MockKronosPredictor(drift_bps_per_bar=Decimal("8"))

    signal_gen = SignalGenerator(cfg, predictor_1m=predictor, predictor_5m=predictor)
    engine = BacktestEngine(cfg)

    ensure_cache_dir(Path("data/kronos"))

    all_results: dict = {
        "config": {
            "days": days,
            "symbols": symbols,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "stride": stride,
            "lookback_1m": cfg.lookback_1m,
            "lookback_5m": cfg.lookback_5m,
            "horizon_bars": cfg.horizon_bars,
            "cost_model": cfg.cost_model,
            "taker_fee_bps": str(cfg.taker_fee_bps),
            "spread_bps": str(cfg.spread_bps),
            "slippage_bps": str(cfg.slippage_bps),
            "min_edge_bps": str(cfg.min_edge_bps),
            "cost_bps": str(cfg.cost_bps()),
            "use_mock": args.use_mock,
        },
        "per_symbol": {},
        "aggregate": {},
        "runtime": {},
    }

    total_infer = 0
    total_time = 0.0
    agg_trades: dict[str, list] = {s: [] for s in ("kronos_1m", "kronos_5m", "momentum", "buy_hold")}
    agg_equity: dict[str, list[float]] = {s: [1.0] for s in agg_trades}
    per_strategy_time: dict[str, float] = {}

    for sym in symbols:
        print(f"\n=== {sym} ===")
        # Fetch 1m candles
        print(f"  fetching {sym} 1m {start.isoformat()}..{end.isoformat()} ...")
        try:
            candles = load_or_fetch_candles(sym, start, end, interval="1m", use_cache=not args.no_cache)
        except Exception as exc:
            print(f"  fetch failed for {sym}: {exc}", file=sys.stderr)
            continue
        print(f"  got {len(candles)} 1m candles ({candles[0].timestamp.isoformat() if candles else 'none'} -> {candles[-1].timestamp.isoformat() if candles else 'none'})")
        if len(candles) < cfg.lookback_1m + cfg.horizon_bars + 5:
            print(f"  insufficient candles for {sym}, need {cfg.lookback_1m + cfg.horizon_bars + 5}, skipping")
            continue

        # Run all 4 strategies
        t0 = time.perf_counter()
        results = engine.run_all(sym, candles, signal_gen)
        dt = time.perf_counter() - t0
        print(f"  backtest wall {dt:.1f}s  (inferences: kronos_1m {results['kronos_1m'].inference_count}, kronos_5m {results['kronos_5m'].inference_count})")

        sym_entry: dict = {}
        for strat, res in results.items():
            m = res.metrics
            print(
                f"  {strat:12} trades {m.trade_count:4}  net {m.net_pnl_bps:7.1f}bps ({m.net_pnl_pct:.2f}%)  win {m.win_rate*100:4.1f}%  PF {m.profit_factor:5.2f}  DD {m.max_drawdown_pct:4.1f}%  Sharpe {m.sharpe_ratio:5.2f}  avg {m.avg_net_bps:6.1f}bps"
            )
            print(f"             avg_win {m.avg_win_bps:6.1f}  avg_loss {m.avg_loss_bps:6.1f}  turnover {m.turnover:.0f}  infer {res.inference_count} time {res.inference_time_s:.1f}s")
            sym_entry[strat] = {
                "net_pnl_bps": str(m.net_pnl_bps),
                "net_pnl_pct": str(m.net_pnl_pct),
                "trade_count": m.trade_count,
                "win_rate": m.win_rate,
                "avg_win_bps": str(m.avg_win_bps),
                "avg_loss_bps": str(m.avg_loss_bps),
                "profit_factor": m.profit_factor,
                "max_drawdown_pct": m.max_drawdown_pct,
                "sharpe_ratio": m.sharpe_ratio,
                "turnover": m.turnover,
                "avg_net_bps": str(m.avg_net_bps),
                "std_net_bps": m.std_net_bps,
                "inference_count": res.inference_count,
                "inference_time_s": res.inference_time_s,
                "horizon_bars": res.horizon_bars,
                "stride_bars": res.stride_bars,
            }
            # aggregate
            agg_trades[strat].extend(res.trades)
            # For aggregate equity, we need to combine per-symbol equity curves additive? Simplest: average equity across symbols
            # We'll later compute aggregate metrics from all trades combined
            total_infer += res.inference_count
            total_time += res.inference_time_s
            per_strategy_time[strat] = per_strategy_time.get(strat, 0) + res.inference_time_s

        all_results["per_symbol"][sym] = sym_entry

    # Aggregate across symbols per strategy
    print(f"\n=== Aggregate across {len(symbols)} symbols ===")
    from app.strategies.kronos.metrics import compute_metrics as _cm

    # For aggregate we combine all trades of same strategy across symbols
    for strat in ("kronos_1m", "kronos_5m", "momentum", "buy_hold"):
        trades = agg_trades[strat]
        # Build combined equity curve: start 1.0, sequentially apply each trade's net
        equity = [1.0]
        cur = 1.0
        for t in trades:
            cur *= 1.0 + float(t.net_bps) / 10000.0
            equity.append(cur)
        # Daily returns: need to aggregate per day across symbols? Use global daily from trades
        # Reuse helper
        from app.strategies.kronos.backtest import _daily_returns_from_trades

        # Create dummy candles for daily grouping? Use trades entry_time
        daily_map: dict[str, float] = {}
        for tr in trades:
            day = tr.entry_time[:10]
            daily_map[day] = daily_map.get(day, 0.0) + float(tr.net_bps) / 10000.0
        daily_rets = [daily_map[d] for d in sorted(daily_map)]
        metrics = _cm(trades, equity, daily_rets)
        print(
            f"  {strat:12} trades {metrics.trade_count:4}  net {metrics.net_pnl_bps:7.1f}bps ({metrics.net_pnl_pct:.2f}%)  win {metrics.win_rate*100:4.1f}%  PF {metrics.profit_factor:5.2f}  DD {metrics.max_drawdown_pct:4.1f}%  Sharpe {metrics.sharpe_ratio:5.2f}"
        )
        all_results["aggregate"][strat] = {
            "net_pnl_bps": str(metrics.net_pnl_bps),
            "net_pnl_pct": str(metrics.net_pnl_pct),
            "trade_count": metrics.trade_count,
            "win_rate": metrics.win_rate,
            "avg_win_bps": str(metrics.avg_win_bps),
            "avg_loss_bps": str(metrics.avg_loss_bps),
            "profit_factor": metrics.profit_factor,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "turnover": metrics.turnover,
            "avg_net_bps": str(metrics.avg_net_bps),
            "std_net_bps": metrics.std_net_bps,
            "inference_time_s": per_strategy_time.get(strat, 0.0),
        }

    # Runtime
    # Estimate 30d practicality: linear extrapolation from days
    est_30d_infer = total_infer * (30 / days) if days else 0
    est_30d_time = total_time * (30 / days) if days else 0
    all_results["runtime"] = {
        "total_inference_count": total_infer,
        "total_inference_time_s": total_time,
        "avg_latency_ms": (total_time / total_infer * 1000) if total_infer else 0,
        "per_strategy_time": per_strategy_time,
        "est_30d_inference_count": est_30d_infer,
        "est_30d_time_s": est_30d_time,
        "est_30d_hours": est_30d_time / 3600,
        "practical_30d_cpu": est_30d_time < 3600 * 8,  # <8h considered practical on CPU
    }
    print(f"\nRuntime: total inferences {total_infer}  time {total_time:.1f}s  avg {total_time/total_infer*1000:.0f}ms" if total_infer else "Runtime: no inferences (mock or hold)")
    print(f"  est 30d: inferences {est_30d_infer:.0f}  time {est_30d_time/3600:.1f}h  practical_cpu={all_results['runtime']['practical_30d_cpu']}")

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved machine-readable result to {out_path}")

    # Brief verdict
    kronos_1m_net = Decimal(all_results["aggregate"]["kronos_1m"]["net_pnl_bps"])
    kronos_5m_net = Decimal(all_results["aggregate"]["kronos_5m"]["net_pnl_bps"])
    mom_net = Decimal(all_results["aggregate"]["momentum"]["net_pnl_bps"])
    bh_net = Decimal(all_results["aggregate"]["buy_hold"]["net_pnl_bps"])
    print("\nVerdict (after costs):")
    print(f"  kronos_1m {kronos_1m_net}bps vs momentum {mom_net}bps vs buy_hold {bh_net}bps")
    if kronos_1m_net > mom_net and kronos_1m_net > bh_net and kronos_1m_net > 0:
        print("  -> Kronos-mini 1m shows positive edge over baselines (but check Sharpe/win rate for significance).")
    elif kronos_1m_net > mom_net:
        print("  -> Kronos-mini 1m beats momentum but not buy_hold or not positive – no clear edge.")
    else:
        print("  -> No useful edge over simple baselines after costs.")

    print("\nDone. Research-only, no DEMO/LIVE triggered.")


if __name__ == "__main__":
    main()
