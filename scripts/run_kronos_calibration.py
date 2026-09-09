"""Final cheap Kronos-mini calibration experiment (research-only, isolated).

BTC/USDT only, existing cached Binance 1m data, existing Kronos-mini
tokenizer/model. No production/DEMO/LIVE/RiskEngine/execution/Telegram
changes. No 30-day backtest.

Grid: T in {0.5, 1.0, 1.5} x top_p in {0.8, 0.9, 1.0}, 5-minute horizon
(kronos_1m: 400x1m -> pred_len 5). sample_count=1, top_k=0, clip=5.

Walk-forward, no lookahead: at t use only candles through t; realized =
(t+5 close - t+1 open)/open (long basis, bps).

Reuses data/kronos/binance_*.csv (cache-only, never fetches) and
data/kronos/forecast_cache/ (model_tag isolates configs; baseline
T=1.0/top_p=0.9 reuses the existing "mini" cache entries).

Measures per config: predicted-return distribution, directional accuracy,
Pearson correlation, gross/net PnL and trade counts at 5/10/15/20/30bps.

Usage:
  py -3.13 scripts/run_kronos_calibration.py --days 2 --stride 120
  py -3.13 scripts/run_kronos_calibration.py --days 2 --stride 120 --max-samples 24
  py -3.13 scripts/run_kronos_calibration.py --output data/kronos/calibration_3c.json
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
from app.strategies.kronos.data import candles_from_csv, ensure_cache_dir
from app.strategies.kronos.diagnostics import (
    Sample,
    directional_accuracy,
    distribution_stats,
    pearson_corr,
    threshold_coverage,
    trade_stats_for_threshold,
)

UTC = timezone.utc
CACHE_DIR = Path("data/kronos")
FORECAST_CACHE = CACHE_DIR / "forecast_cache"

GRID_T = (0.5, 1.0, 1.5)
GRID_TOP_P = (0.8, 0.9, 1.0)
SENS_THRESHOLDS = (5.0, 10.0, 15.0, 20.0, 30.0)


def _sanitize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "")


def _window_hash(window) -> str:
    h = hashlib.md5()
    h.update(str(window[0].timestamp).encode())
    h.update(str(window[-1].timestamp).encode())
    h.update(str(window[-1].close).encode())
    h.update(str(len(window)).encode())
    return h.hexdigest()[:12]


def _cache_path(symbol: str, t_iso: str, lookback: int, model_tag: str) -> Path:
    safe = _sanitize_symbol(symbol)
    t_safe = t_iso.replace(":", "").replace("+", "p")
    return FORECAST_CACHE / f"{safe}_kronos_1m_lb{lookback}_{model_tag}_{t_safe}.json"


def _model_tag(t: float, top_p: float) -> str:
    # Baseline reuses existing diagnostics cache ("mini") -> zero new inference.
    if t == 1.0 and top_p == 0.9:
        return "mini"
    return f"mini_T{t}_p{top_p}"


def _seed_for(t_iso: str, t: float, top_p: float) -> int:
    h = hashlib.md5(f"{t_iso}|{t}|{top_p}".encode()).hexdigest()[:8]
    return int(h, 16) % (2**31)


def predict_bps(symbol: str, window, t_iso: str, lookback: int, predictor, pred_len: int, model_tag: str, temp: float, top_p: float) -> tuple[float, bool, float]:
    FORECAST_CACHE.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, t_iso, lookback, model_tag)
    wh = _window_hash(window)
    if path.exists():
        try:
            obj = json.loads(path.read_text())
            if obj.get("window_hash") == wh and "pred_bps" in obj:
                return float(obj["pred_bps"]), True, 0.0
        except Exception:
            pass
    # Deterministic seed per (t, config) so reruns are comparable.
    try:
        import torch

        torch.manual_seed(_seed_for(t_iso, temp, top_p))
    except Exception:
        pass
    t0 = time.perf_counter()
    forecast = predictor.predict(window, pred_len)
    dt = time.perf_counter() - t0
    gross = compute_gross_bps(window[-1].close, forecast[-1].close)
    pred_bps = float(gross)
    try:
        path.write_text(json.dumps({"pred_bps": pred_bps, "window_hash": wh, "t": t_iso, "model_tag": model_tag}))
    except Exception:
        pass
    return pred_bps, False, dt


def find_best_cached_csv(symbol: str) -> Path | None:
    safe = _sanitize_symbol(symbol)
    cands = sorted(CACHE_DIR.glob(f"binance_{safe}_1m_*.csv"))
    best: Path | None = None
    best_n = -1
    for p in cands:
        try:
            with p.open("r") as f:
                n = sum(1 for _ in f) - 1
            if n > best_n:
                best_n = n
                best = p
        except Exception:
            continue
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cheap Kronos-mini calibration experiment (BTC, cached data only)")
    p.add_argument("--days", type=int, default=2, help="days of cached history to use")
    p.add_argument("--stride", type=int, default=120, help="bars between decisions")
    p.add_argument("--lookback", type=int, default=400, help="1m lookback")
    p.add_argument("--max-samples", type=int, default=0, help="cap decision points (0=no cap)")
    p.add_argument("--output", type=str, default="data/kronos/calibration_3c.json", help="JSON output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbol = "BTC/USDT"
    stride = args.stride
    lookback = args.lookback

    cfg = BacktestConfig(lookback_1m=lookback, horizon_bars=5, stride_bars=stride)
    cost = cfg.cost_bps()
    print(f"Calibration experiment – {symbol} lookback={lookback} stride={stride} cost={cost}bps")
    print(f"Grid T={list(GRID_T)} x top_p={list(GRID_TOP_P)} (9 configs, 5-min horizon, sample_count=1)")

    # Load candles from cache only (never fetch).
    best = find_best_cached_csv(symbol)
    if best is None:
        print(f"No cached CSV for {symbol} in {CACHE_DIR}", file=sys.stderr)
        sys.exit(2)
    all_c = candles_from_csv(best)
    print(f"Cache file: {best.name} ({len(all_c)} bars, {all_c[0].timestamp.isoformat()} -> {all_c[-1].timestamp.isoformat()})")
    end = all_c[-1].timestamp + timedelta(minutes=1)
    start = end - timedelta(days=args.days)
    candles = tuple(c for c in all_c if start <= c.timestamp < end)
    print(f"Window: {start.isoformat()} -> {end.isoformat()} ({len(candles)} bars)")
    if len(candles) < lookback + cfg.horizon_bars + 5:
        print("Insufficient cached bars for window", file=sys.stderr)
        sys.exit(2)

    from app.strategies.kronos.real import RealKronosPredictor

    # One shared model load; per-config lightweight predictor handles (same weights).
    base = RealKronosPredictor(variant="mini", T=1.0, top_p=0.9, top_k=0, sample_count=1)
    print("Loading Kronos-mini (NeoQuasar/Kronos-mini + Tokenizer-2k)...")
    try:
        init_s = base.load()
    except Exception as exc:
        print(f"FAILED to load Kronos-mini: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"  loaded in {init_s:.1f}s device={base.device}")
    # Share loaded weights across configs by reusing underlying handles.
    shared = (base._tokenizer, base._model, base._predictor)

    ensure_cache_dir(CACHE_DIR)

    t_list = list(range(lookback - 1, len(candles) - cfg.horizon_bars - 1, stride))
    if args.max_samples and len(t_list) > args.max_samples:
        step = len(t_list) / args.max_samples
        t_list = [t_list[int(i * step)] for i in range(args.max_samples)]
    print(f"Decision points: {len(t_list)}")

    # Realized (config-independent, computed once, no lookahead beyond evaluation).
    realized_by_t: dict[int, tuple[str, float]] = {}
    for t in t_list:
        entry = candles[t + 1].open
        exit_c = candles[t + cfg.horizon_bars].close
        if entry <= 0 or exit_c <= 0:
            continue
        realized_by_t[t] = (candles[t].timestamp.isoformat(), float(compute_gross_bps(entry, exit_c)))

    results: dict = {}
    total_miss = 0
    total_hit = 0
    total_time = 0.0

    for temp in GRID_T:
        for top_p in GRID_TOP_P:
            tag = _model_tag(temp, top_p)
            pred = RealKronosPredictor(variant="mini", T=temp, top_p=top_p, top_k=0, sample_count=1)
            # Reuse loaded weights (avoid 9 model loads).
            pred._tokenizer, pred._model, pred._predictor = shared
            pred._loaded = True
            key = f"T{temp}_p{top_p}"
            samples: list[Sample] = []
            miss = 0
            hit = 0
            inf_time = 0.0
            for t, (t_iso, realized) in realized_by_t.items():
                window = candles[t - lookback + 1 : t + 1]
                try:
                    pred_bps, was_hit, dt = predict_bps(symbol, window, t_iso, lookback, pred, cfg.pred_len_1m, tag, temp, top_p)
                except Exception as exc:
                    print(f"  predict failed {key} {t_iso}: {exc}", file=sys.stderr)
                    continue
                if was_hit:
                    hit += 1
                else:
                    miss += 1
                    inf_time += dt
                samples.append(Sample(timestamp=t_iso, pred_bps=pred_bps, realized_bps=realized))
            preds = [s.pred_bps for s in samples]
            reals = [s.realized_bps for s in samples]
            dist = distribution_stats(preds)
            cov = threshold_coverage(preds)
            acc = directional_accuracy(preds, reals)
            corr = pearson_corr(preds, reals)
            sens = {str(thr): trade_stats_for_threshold(samples, thr, cost) for thr in SENS_THRESHOLDS}
            # gross/net at default 15 for headline
            headline = sens["15.0"]
            results[key] = {
                "T": temp,
                "top_p": top_p,
                "n": len(samples),
                "pred_distribution": dist,
                "coverage_abs_pred": cov,
                "directional_accuracy": acc,
                "correlation": corr,
                "gross_pnl_bps_thr15": headline["gross_pnl_bps"],
                "net_pnl_bps_thr15": headline["net_pnl_bps"],
                "sensitivity": sens,
                "inference_new": miss,
                "inference_cached": hit,
                "inference_time_s": inf_time,
            }
            total_miss += miss
            total_hit += hit
            total_time += inf_time
            print(
                f"  {key}: n={len(samples)} abs_mean={dist['abs_mean']:.2f} p95={dist['p95']:.2f} max={dist['max']:.2f} "
                f"acc={acc['accuracy']*100:.1f}% corr={corr:+.3f} net15={headline['net_pnl_bps']:.1f} "
                f"trades15={headline['trade_count']} (new={miss} cached={hit} {inf_time:.0f}s)"
            )

    out = {
        "config": {
            "symbol": symbol,
            "days": args.days,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "stride": stride,
            "lookback_1m": lookback,
            "horizon_bars": cfg.horizon_bars,
            "cost_model": cfg.cost_model,
            "cost_bps": str(cost),
            "grid_T": list(GRID_T),
            "grid_top_p": list(GRID_TOP_P),
            "sample_count": 1,
            "top_k": 0,
            "seed": "torch.manual_seed(md5(t_iso|T|top_p)) per inference",
            "cache": "disk forecast_cache, baseline T1.0/p0.9 reuses existing 'mini' entries",
        },
        "results": results,
        "runtime": {
            "inference_new": total_miss,
            "inference_cached": total_hit,
            "inference_time_s": total_time,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved calibration JSON to {out_path}")
    print(f"Runtime: new inferences {total_miss} (+{total_hit} cache hits) {total_time:.0f}s")
    print("Done. Research-only, no DEMO/LIVE triggered.")


if __name__ == "__main__":
    main()
