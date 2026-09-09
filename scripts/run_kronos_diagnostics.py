"""Phase 3C CLI – Kronos-mini diagnostic analysis only (research, isolated).

Reuses existing Binance 1m CSV cache in data/kronos/ whenever possible.
Reuses disk forecast cache in data/kronos/forecast_cache/ to avoid rerunning
expensive Kronos-mini inference.

Walk-forward, no lookahead: at t use only candles through t; forecast;
realized = (t+5 close - t+1 open)/open.

Measures separately for kronos_1m and kronos_5m:
  distribution p50/p75/p90/p95/max, coverage above thresholds,
  directional accuracy, correlation, gross/net PnL, trades/win/PF,
  threshold sensitivity 5/10/15/20/30/50.

Usage:
  py -3.13 scripts/run_kronos_diagnostics.py --symbols BTC/USDT --days 7 --stride 60
  py -3.13 scripts/run_kronos_diagnostics.py --symbols BTC/USDT ETH/USDT --days 3 --stride 30
  py -3.13 scripts/run_kronos_diagnostics.py --use-mock   (no torch, quick validation)
  py -3.13 scripts/run_kronos_diagnostics.py --cache-only (never hit network)

Output: console + machine-readable JSON under data/kronos/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from app.strategies.kronos.backtest import BacktestConfig
from app.strategies.kronos.cost import compute_gross_bps
from app.strategies.kronos.data import candles_from_csv, ensure_cache_dir, load_or_fetch_candles
from app.strategies.kronos.diagnostics import (
    COVERAGE_THRESHOLDS,
    DEFAULT_THRESHOLDS,
    Sample,
    directional_accuracy,
    distribution_stats,
    pearson_corr,
    threshold_coverage,
    threshold_sensitivity,
    trade_stats_for_threshold,
)
from app.strategies.kronos.scanner import resample_to_5m

UTC = timezone.utc
CACHE_DIR = Path("data/kronos")
FORECAST_CACHE = CACHE_DIR / "forecast_cache"


def _sanitize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "")


def find_best_cached_csv(symbol: str, cache_dir: Path = CACHE_DIR) -> Path | None:
    """Pick the cached 1m CSV for symbol with the most rows (no network)."""
    safe = _sanitize_symbol(symbol)
    cands = sorted(cache_dir.glob(f"binance_{safe}_1m_*.csv"))
    if not cands:
        return None
    best: Path | None = None
    best_n = -1
    for p in cands:
        try:
            # cheap row count without full parse
            with p.open("r") as f:
                n = sum(1 for _ in f) - 1
            if n > best_n:
                best_n = n
                best = p
        except Exception:
            continue
    return best


def load_candles_reuse_cache(
    symbol: str,
    start: datetime,
    end: datetime,
    cache_only: bool,
) -> tuple:
    """Load candles preferring existing CSV cache; fetch only if needed and allowed."""
    # 1. Try best cached file sliced to [start, end)
    best = find_best_cached_csv(symbol)
    if best is not None:
        try:
            all_c = candles_from_csv(best)
            sliced = tuple(c for c in all_c if start <= c.timestamp < end)
            # If sliced covers most of requested range, use it without fetch
            requested_min = int((end - start).total_seconds() // 60)
            if len(sliced) >= requested_min - 5 and len(sliced) > 0:
                return sliced, {"source": f"cache:{best.name}", "cached_rows": len(all_c)}
            # If cache file covers a superset but different window, still usable if enough bars
            if len(sliced) >= 500:
                return sliced, {"source": f"cache:{best.name}", "cached_rows": len(all_c)}
        except Exception:
            pass
    if cache_only:
        # Fall back to whatever best file has (even if window differs), sliced loosely
        if best is not None:
            try:
                all_c = candles_from_csv(best)
                return all_c, {"source": f"cache:{best.name}", "cached_rows": len(all_c), "note": "full-file fallback"}
            except Exception as exc:
                raise RuntimeError(f"no usable cache for {symbol}: {exc}")
        raise RuntimeError(f"no cached CSV for {symbol} and --cache-only set")
    # Fetch (uses load_or_fetch which itself caches)
    candles = load_or_fetch_candles(symbol, start, end, interval="1m", use_cache=True)
    return candles, {"source": "fetch-or-exact-cache"}


def _window_hash(candles) -> str:
    h = hashlib.md5()
    h.update(str(candles[0].timestamp).encode())
    h.update(str(candles[-1].timestamp).encode())
    h.update(str(candles[-1].close).encode())
    h.update(str(len(candles)).encode())
    return h.hexdigest()[:12]


def _forecast_cache_path(symbol: str, strategy: str, t_iso: str, lookback: int, model_tag: str = "mini") -> Path:
    safe = _sanitize_symbol(symbol)
    t_safe = t_iso.replace(":", "").replace("+", "p")
    return FORECAST_CACHE / f"{safe}_{strategy}_lb{lookback}_{model_tag}_{t_safe}.json"


def get_pred_bps_cached(
    symbol: str,
    strategy: str,
    window,
    t_iso: str,
    lookback: int,
    predictor,
    pred_len: int,
    model_tag: str = "mini",
) -> tuple[float, bool, float]:
    """Return (pred_bps, cache_hit, infer_time_s). Walk-forward safe: window ends at t."""
    FORECAST_CACHE.mkdir(parents=True, exist_ok=True)
    path = _forecast_cache_path(symbol, strategy, t_iso, lookback, model_tag)
    wh = _window_hash(window)
    if path.exists():
        try:
            obj = json.loads(path.read_text())
            if obj.get("window_hash") == wh and "pred_bps" in obj:
                return float(obj["pred_bps"]), True, 0.0
        except Exception:
            pass
    # Miss: run inference (single predict call, no future leaked)
    t0 = time.perf_counter()
    forecast = predictor.predict(window, pred_len)
    dt = time.perf_counter() - t0
    last_close = window[-1].close
    pred_close = forecast[-1].close
    gross = compute_gross_bps(last_close, pred_close)
    pred_bps = float(gross)
    try:
        path.write_text(json.dumps({"pred_bps": pred_bps, "window_hash": wh, "t": t_iso, "strategy": strategy}))
    except Exception:
        pass
    return pred_bps, False, dt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kronos-mini Phase 3C diagnostics (research-only)")
    p.add_argument("--symbols", nargs="+", default=["BTC/USDT"], help="symbols (default BTC only for CPU)")
    p.add_argument("--days", type=int, default=7, help="days of history (uses cache when possible)")
    p.add_argument("--stride", type=int, default=60, help="bars between decisions")
    p.add_argument("--lookback", type=int, default=400, help="1m lookback")
    p.add_argument("--end", type=str, default=None, help="end ISO (default: latest cached timestamp)")
    p.add_argument("--use-mock", action="store_true", help="MockKronosPredictor instead of real mini")
    p.add_argument("--cache-only", action="store_true", help="never hit network; use CSV cache only")
    p.add_argument("--max-samples", type=int, default=0, help="cap samples per strategy (0=no cap)")
    p.add_argument("--output", type=str, default="data/kronos/diagnostics_3c.json", help="JSON output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols]
    stride = args.stride
    lookback = args.lookback
    lookback_5m = max(10, lookback // 5)

    cfg = BacktestConfig(
        lookback_1m=lookback,
        lookback_5m=lookback_5m,
        pred_len_1m=5,
        pred_len_5m=1,
        horizon_bars=5,
        stride_bars=stride,
    )
    cost = cfg.cost_bps()
    print(f"Phase 3C diagnostics – symbols={symbols} stride={stride} lookback={lookback} cost={cost}bps")
    print(f"Thresholds sensitivity: {list(DEFAULT_THRESHOLDS)} | coverage: {list(COVERAGE_THRESHOLDS)}")

    # Predictor (lazy, real mini only). model_tag isolates disk forecast cache.
    if args.use_mock:
        from app.strategies.kronos.forecast import MockKronosPredictor

        predictor = MockKronosPredictor(drift_bps_per_bar=Decimal("8"))
        model_tag = "mock"
        print("Using MockKronosPredictor (no torch)")
    else:
        from app.strategies.kronos.real import RealKronosPredictor

        predictor = RealKronosPredictor(variant="mini", T=1.0, top_p=0.9, top_k=0, sample_count=1)
        model_tag = "mini"
        print("Loading Kronos-mini (NeoQuasar/Kronos-mini + Tokenizer-2k)...")
        try:
            init_s = predictor.load()
        except Exception as exc:
            print(f"FAILED to load Kronos-mini: {exc}", file=sys.stderr)
            sys.exit(2)
        print(f"  loaded in {init_s:.1f}s device={predictor.device}")

    ensure_cache_dir(CACHE_DIR)

    # Determine time window: prefer latest cached timestamps to avoid fetch
    if args.end:
        end = datetime.fromisoformat(args.end)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        start = end - timedelta(days=args.days)
    else:
        # Use latest cached candle across symbols as end
        latest: datetime | None = None
        for sym in symbols:
            best = find_best_cached_csv(sym)
            if best is not None:
                try:
                    all_c = candles_from_csv(best)
                    if all_c:
                        ts = all_c[-1].timestamp + timedelta(minutes=1)
                        if latest is None or ts > latest:
                            latest = ts
                except Exception:
                    pass
        if latest is not None:
            end = latest
            start = end - timedelta(days=args.days)
            print(f"Using cached window {start.isoformat()} -> {end.isoformat()} (no fetch needed)")
        else:
            end = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=1)
            start = end - timedelta(days=args.days)

    per_symbol: dict = {}
    all_samples: dict[str, list[Sample]] = {"kronos_1m": [], "kronos_5m": []}
    runtime = {"inference_count": 0, "inference_time_s": 0.0, "cache_hits": 0, "cache_misses": 0}

    for sym in symbols:
        print(f"\n=== {sym} ===")
        try:
            candles, src_info = load_candles_reuse_cache(sym, start, end, cache_only=args.cache_only)
        except Exception as exc:
            print(f"  no data for {sym}: {exc}", file=sys.stderr)
            continue
        print(f"  source={src_info.get('source')} candles={len(candles)}")
        if len(candles) < lookback + cfg.horizon_bars + 5:
            print(f"  insufficient candles, skipping")
            continue

        samples_1m: list[Sample] = []
        samples_5m: list[Sample] = []
        t_list = list(range(lookback - 1, len(candles) - cfg.horizon_bars - 1, stride))
        if args.max_samples and len(t_list) > args.max_samples:
            # evenly subsample to cap runtime
            step = len(t_list) / args.max_samples
            t_list = [t_list[int(i * step)] for i in range(args.max_samples)]
        print(f"  decision points: {len(t_list)} (stride {stride})")

        for t in t_list:
            # realized long bps: entry t+1 open -> exit t+5 close
            entry = candles[t + 1].open
            exit_c = candles[t + cfg.horizon_bars].close
            if entry <= 0 or exit_c <= 0:
                continue
            realized = float(compute_gross_bps(entry, exit_c))
            t_iso = candles[t].timestamp.isoformat()

            # kronos_1m: window ends at t (no lookahead)
            window_1m = candles[t - lookback + 1 : t + 1]
            try:
                pred_1m, hit, dt = get_pred_bps_cached(sym, "kronos_1m", window_1m, t_iso, lookback, predictor, cfg.pred_len_1m, model_tag)
            except Exception as exc:
                print(f"  1m predict failed at {t_iso}: {exc}", file=sys.stderr)
                continue
            runtime["inference_count"] += 0 if hit else 1
            runtime["inference_time_s"] += dt
            runtime["cache_hits" if hit else "cache_misses"] += 1
            samples_1m.append(Sample(timestamp=t_iso, pred_bps=pred_1m, realized_bps=realized))

            # kronos_5m: resample same 1m window (ends at t), predict 1x5m bar
            window_5m_full = resample_to_5m(window_1m)
            if len(window_5m_full) < 10:
                continue
            use_len = min(len(window_5m_full), lookback_5m)
            window_5m = window_5m_full[-use_len:]
            try:
                pred_5m, hit5, dt5 = get_pred_bps_cached(sym, "kronos_5m", window_5m, t_iso, lookback_5m, predictor, cfg.pred_len_5m, model_tag)
            except Exception as exc:
                print(f"  5m predict failed at {t_iso}: {exc}", file=sys.stderr)
                continue
            # count 5m inference separately (cache key differs so hit tracking shared is fine)
            runtime["inference_count"] += 0 if hit5 else 1
            runtime["inference_time_s"] += dt5
            runtime["cache_hits" if hit5 else "cache_misses"] += 1
            samples_5m.append(Sample(timestamp=t_iso, pred_bps=pred_5m, realized_bps=realized))

        print(f"  collected: 1m n={len(samples_1m)}, 5m n={len(samples_5m)}")
        all_samples["kronos_1m"].extend(samples_1m)
        all_samples["kronos_5m"].extend(samples_5m)

        # Per-symbol diagnostics (store raw samples count + stats; full stats in aggregate)
        per_symbol[sym] = {
            "n_1m": len(samples_1m),
            "n_5m": len(samples_5m),
            "window": {"start": candles[0].timestamp.isoformat(), "end": candles[-1].timestamp.isoformat(), "bars": len(candles)},
            "source": src_info.get("source"),
        }

    # Aggregate diagnostics per strategy
    strategies: dict = {}
    for strat in ("kronos_1m", "kronos_5m"):
        samples = all_samples[strat]
        preds = [s.pred_bps for s in samples]
        reals = [s.realized_bps for s in samples]
        dist = distribution_stats(preds)
        cov = threshold_coverage(preds)
        acc = directional_accuracy(preds, reals)
        corr = pearson_corr(preds, reals)
        # Realized distribution for context
        real_dist = distribution_stats(reals)
        sens = threshold_sensitivity(samples, DEFAULT_THRESHOLDS, cost)
        # Default-threshold (15bps) headline like backtest
        headline = trade_stats_for_threshold(samples, 15.0, cost)
        strategies[strat] = {
            "n": len(samples),
            "pred_distribution": dist,
            "realized_distribution": real_dist,
            "coverage_abs_pred": cov,
            "directional_accuracy": acc,
            "correlation": corr,
            "headline_thr15": headline,
            "gross_pnl_bps_thr15": headline["gross_pnl_bps"],
            "net_pnl_bps_thr15": headline["net_pnl_bps"],
            "sensitivity": sens,
        }
        print(f"\n--- {strat} (n={len(samples)}) ---")
        print(f"  pred p50 {dist['p50']:.2f} p75 {dist['p75']:.2f} p90 {dist['p90']:.2f} p95 {dist['p95']:.2f} max {dist['max']:.2f} abs_mean {dist['abs_mean']:.2f}")
        print(f"  coverage>15: {cov['15.0']['count']} ({cov['15.0']['pct']:.1f}%)  >35(cost): n/a  >50: {cov['50.0']['count']} ({cov['50.0']['pct']:.1f}%)")
        print(f"  dir_acc {acc['accuracy']*100:.1f}% (long {acc['long_acc']*100:.1f}% n={acc['long_n']}, short {acc['short_acc']*100:.1f}% n={acc['short_n']})  corr {corr:+.3f}")
        print(f"  thr15: trades {headline['trade_count']} gross {headline['gross_pnl_bps']:.1f} net {headline['net_pnl_bps']:.1f} win {headline['win_rate']*100:.1f}% PF {headline['profit_factor']:.2f}")

    # Cost analysis: what fraction filtered by 35bps cost?
    # For each strategy, count |pred_gross| > cost vs > cost+thr
    cost_f = float(cost)
    cost_analysis: dict = {}
    for strat in ("kronos_1m", "kronos_5m"):
        preds = [s.pred_bps for s in all_samples[strat]]
        n = len(preds)
        above_cost = sum(1 for v in preds if abs(v) > cost_f)
        above_cost_edge = sum(1 for v in preds if abs(v) > cost_f + 15.0)
        cost_analysis[strat] = {
            "cost_bps": cost_f,
            "n": n,
            "above_cost_count": above_cost,
            "above_cost_pct": (above_cost / n * 100) if n else 0.0,
            "above_cost_plus_15_count": above_cost_edge,
            "above_cost_plus_15_pct": (above_cost_edge / n * 100) if n else 0.0,
        }

    out = {
        "config": {
            "symbols": symbols,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "stride": stride,
            "lookback_1m": lookback,
            "lookback_5m": lookback_5m,
            "horizon_bars": cfg.horizon_bars,
            "cost_model": cfg.cost_model,
            "cost_bps": str(cost),
            "use_mock": args.use_mock,
            "cache_only": args.cache_only,
            "thresholds": list(DEFAULT_THRESHOLDS),
            "coverage_thresholds": list(COVERAGE_THRESHOLDS),
        },
        "per_symbol": per_symbol,
        "strategies": strategies,
        "cost_analysis": cost_analysis,
        "runtime": {
            **runtime,
            "avg_latency_ms": (runtime["inference_time_s"] / max(1, runtime["cache_misses"]) * 1000),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved diagnostics JSON to {out_path}")
    print(f"Runtime: inferences {runtime['cache_misses']} (+{runtime['cache_hits']} cache hits) time {runtime['inference_time_s']:.1f}s")
    print("Done. Research-only, no DEMO/LIVE triggered.")


if __name__ == "__main__":
    main()
