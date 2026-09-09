"""Phase 3A benchmark – real Kronos predictor (isolated).

Benchmarks BOTH models:
  1. NeoQuasar/Kronos-mini  (Tokenizer-2k, ctx 2048, 4.1M) – primary scalping candidate
  2. NeoQuasar/Kronos-base  (Tokenizer-base, ctx 512, 102.3M) – benchmark only

Isolated: does NOT connect to trading/DEMO/execution/AutoTrader/Telegram/storage/AI agent.
Downloads HF models only for this benchmark.

Measures:
  - init/download time
  - first inference latency (cold)
  - warm inference latency (avg of N)
  - 5-symbol batch latency
  - RAM, VRAM (if CUDA)

Usage:
  py -3.13 scripts/benchmark_kronos.py
  py -3.13 scripts/benchmark_kronos.py --lookback 400 --pred-len 5 --warm-runs 5
  py -3.13 scripts/benchmark_kronos.py --variants mini  (only mini)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Sequence

# Ensure workspace root
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from app.strategies.kronos.real import MODEL_SPECS, RealKronosPredictor, detect_environment
from app.strategies.kronos.synthetic import generate_synthetic_candles
from app.strategies.kronos.types import Candle

UTC = timezone.utc


def _mem_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _vram_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)  # type: ignore
    except Exception:
        pass
    return None


def _time_fn(fn, *args, **kwargs) -> tuple[float, object]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return dt, out


def benchmark_variant(
    variant: str,
    candles_1m: Sequence[Candle],
    candles_5m_equiv: Sequence[Candle],
    pred_len: int,
    warm_runs: int,
    batch_candles: Sequence[Sequence[Candle]],
) -> dict:
    spec = MODEL_SPECS[variant]  # type: ignore
    print(f"\n{'='*72}")
    print(f"Benchmarking {spec.display_name}")
    print(f"  model:     {spec.model_id}")
    print(f"  tokenizer: {spec.tokenizer_id}")
    print(f"  max_ctx:   {spec.max_context}  params: {spec.params}")
    print(f"{'='*72}")

    predictor = RealKronosPredictor(variant=variant, T=1.0, top_p=0.9, top_k=0, sample_count=1)  # type: ignore

    mem_before = _mem_mb()
    vram_before = _vram_mb()

    # -- init / download
    print(f"[{variant}] loading (download if needed)...")
    t0 = time.perf_counter()
    try:
        init_time = predictor.load()
    except Exception as exc:
        print(f"[{variant}] FAILED to load: {exc}", file=sys.stderr)
        raise
    wall_init = time.perf_counter() - t0
    # predictor.load already returns elapsed; use wall to be safe
    init_t = init_time if init_time is not None else wall_init
    print(f"[{variant}] init/download time: {init_t:.2f}s (wall {wall_init:.2f}s)")

    mem_after_load = _mem_mb()
    vram_after_load = _vram_mb()
    ram_used = (mem_after_load - mem_before) if mem_before and mem_after_load else None
    vram_used = (vram_after_load - vram_before) if vram_before is not None and vram_after_load is not None else vram_after_load

    # -- 1 symbol, 1m forecast
    print(f"[{variant}] 1-symbol 1m  cold inference (lookback={len(candles_1m)} pred_len={pred_len})...")
    first_1m_lat, out_1m = _time_fn(predictor.predict, candles_1m, pred_len)
    print(f"  first 1m latency: {first_1m_lat*1000:.1f} ms  -> {len(out_1m)} bars, close={out_1m[-1].close}")

    # warm runs
    warm_lats: list[float] = []
    for i in range(warm_runs):
        dt, _ = _time_fn(predictor.predict, candles_1m, pred_len)
        warm_lats.append(dt)
    warm_avg = sum(warm_lats) / len(warm_lats) if warm_lats else 0
    warm_p50 = sorted(warm_lats)[len(warm_lats)//2] if warm_lats else 0
    print(f"  warm 1m avg: {warm_avg*1000:.1f} ms  p50 {warm_p50*1000:.1f} ms  (n={warm_runs})")

    # -- 1 symbol, 5m forecast
    # For 5m we feed 5m-resampled candles. Use same pred_len for comparability.
    # If candles_5m_equiv shorter (e.g. 80), still predict pred_len.
    print(f"[{variant}] 1-symbol 5m  cold (input {len(candles_5m_equiv)} 5m-bars pred_len={pred_len})...")
    # 5m predictor may need shorter context; but spec says same sampling params
    first_5m_lat, out_5m = _time_fn(predictor.predict, candles_5m_equiv, pred_len)
    print(f"  first 5m latency: {first_5m_lat*1000:.1f} ms -> close={out_5m[-1].close}")
    warm_5m_lats: list[float] = []
    for _ in range(warm_runs):
        dt, _ = _time_fn(predictor.predict, candles_5m_equiv, pred_len)
        warm_5m_lats.append(dt)
    warm_5m_avg = sum(warm_5m_lats) / len(warm_5m_lats) if warm_5m_lats else 0
    print(f"  warm 5m avg: {warm_5m_avg*1000:.1f} ms")

    # -- 5-symbol batch
    print(f"[{variant}] 5-symbol batch (batch={len(batch_candles)} each {len(batch_candles[0])} bars pred_len={pred_len})...")
    # first batch (cold batch)
    t0 = time.perf_counter()
    try:
        batch_out = predictor.predict_batch(batch_candles, pred_len)
        batch_lat = time.perf_counter() - t0
        batch_ok = True
        print(f"  batch latency: {batch_lat*1000:.1f} ms -> {len(batch_out)} series, each {len(batch_out[0])} bars")
        # warm batch
        batch_warm: list[float] = []
        for _ in range(warm_runs):
            dt, _ = _time_fn(predictor.predict_batch, batch_candles, pred_len)
            batch_warm.append(dt)
        batch_warm_avg = sum(batch_warm) / len(batch_warm) if batch_warm else batch_lat
        print(f"  batch warm avg: {batch_warm_avg*1000:.1f} ms  per-symbol {batch_warm_avg/len(batch_candles)*1000:.1f} ms")
    except Exception as exc:
        batch_lat = float("nan")
        batch_warm_avg = float("nan")
        batch_ok = False
        print(f"  batch FAILED: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()

    mem_peak = _mem_mb()
    vram_peak = _vram_mb()

    # -- verify compatibility with scanner (offline, no I/O)
    try:
        from app.strategies.kronos.config import KronosConfig
        from app.strategies.kronos.scanner import KronosScanner

        cfg = KronosConfig(
            lookback=min(400, len(candles_1m)),
            pred_len_1m=pred_len,
            pred_len_5m=pred_len,
            min_edge_bps=Decimal("15"),
            require_5m_confirmation=False,
            max_candle_age_ms=600_000,
            max_gap_ms=600_000,
        )
        scanner = KronosScanner(cfg, predictor)
        now = candles_1m[-1].timestamp + timedelta(seconds=10)
        sig = scanner.scan("BTC/USDT", candles_1m, now=now)
        print(f"  scanner check: signal={sig.signal} reason={sig.reason} net_1m={sig.net_bps_1m}")
        scanner_ok = True
    except Exception as exc:
        print(f"  scanner compatibility FAILED: {exc}", file=sys.stderr)
        scanner_ok = False

    predictor.unload()

    return {
        "variant": variant,
        "spec": spec,
        "init_time_s": init_t,
        "first_1m_ms": first_1m_lat * 1000,
        "warm_1m_ms": warm_avg * 1000,
        "first_5m_ms": first_5m_lat * 1000,
        "warm_5m_ms": warm_5m_avg * 1000,
        "batch_ms": batch_lat * 1000 if batch_ok else float("nan"),
        "batch_warm_ms": batch_warm_avg * 1000 if batch_ok else float("nan"),
        "ram_mb_before": mem_before,
        "ram_mb_after": mem_after_load,
        "ram_delta_mb": ram_used,
        "ram_peak_mb": mem_peak,
        "vram_mb": vram_peak,
        "vram_after_load_mb": vram_after_load,
        "scanner_ok": scanner_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kronos Phase 3A benchmark (isolated)")
    parser.add_argument("--lookback", type=int, default=400, help="lookback bars (1m)")
    parser.add_argument("--pred-len", type=int, default=5, help="forecast horizon (same for 1m and 5m)")
    parser.add_argument("--warm-runs", type=int, default=3, help="warm runs to average")
    parser.add_argument("--variants", nargs="+", default=["mini", "base"], choices=["mini", "base"])
    args = parser.parse_args()

    env = detect_environment()
    print("Environment")
    print(f"  python: {env['python']}  {env['platform']}")
    print(f"  torch:  {env.get('torch_version')}  cuda={env.get('cuda_available')}  device={env['device']}")
    if env.get("cuda_device_name"):
        print(f"  cuda dev: {env['cuda_device_name']}")
    print(f"  RAM total {env.get('ram_total_gb')} GB  avail {env.get('ram_available_gb')} GB ({env.get('ram_percent')}%)")
    # CPU
    try:
        import psutil

        print(f"  CPU logical {psutil.cpu_count(logical=True)} physical {psutil.cpu_count(logical=False)}")
    except Exception:
        pass

    # Generate SAME input data for both models
    # Use deterministic synthetic candles with small drift to mimic real market
    start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    lookback = args.lookback
    pred_len = args.pred_len

    print(f"\nGenerating synthetic data: lookback={lookback} pred_len={pred_len} (same for both models)")
    # 1m candles: 400 bars starting at 100
    candles_1m = generate_synthetic_candles(start=start, count=lookback, start_price=Decimal("50000"), drift_bps_per_bar=Decimal("0.2"))
    # 5m equivalent: resample conceptually – generate 5m candles at 5m interval for fair bench
    # Option A: resample 1m -> 5m via scanner helper; Option B: generate directly at 5m
    # We do both: resample to get ~80 bars, pad to 80 if needed
    from app.strategies.kronos.scanner import resample_to_5m

    # For 5m bench, we want ~80 bars (400/5). Use resampled.
    candles_5m = resample_to_5m(candles_1m)
    # If resampled yields fewer than expected (e.g., 80), pad or use as is
    print(f"  1m candles: {len(candles_1m)} (e.g. {candles_1m[0].close} -> {candles_1m[-1].close})")
    print(f"  5m resampled: {len(candles_5m)} bars (from 1m)")
    # For consistent benchmark length, if 5m too short (<80) keep as is; batch also uses 1m length
    if len(candles_5m) < 10:
        # fallback: generate separate 5m synthetic
        print("  5m resampled too short, generating 5m synthetic directly")
        candles_5m = generate_synthetic_candles(start=start, count=max(80, pred_len * 10), start_price=Decimal("50000"))[::5]  # type: ignore
    # ensure at least pred_len bars for prediction
    print(f"  5m candles used: {len(candles_5m)}")

    # 5-symbol batch: 5 variants with different drifts/prices
    batch_candles: list[Sequence[Candle]] = []
    for i, (price, drift) in enumerate([(Decimal("50000"), Decimal("0.2")), (Decimal("3000"), Decimal("-0.1")), (Decimal("600"), Decimal("0.3")), (Decimal("1.0"), Decimal("0.0")), (Decimal("100000"), Decimal("0.15"))]):
        c = generate_synthetic_candles(start=start, count=lookback, start_price=price, drift_bps_per_bar=drift)
        batch_candles.append(c)
    print(f"  batch: {len(batch_candles)} symbols, each {len(batch_candles[0])} bars")

    results: list[dict] = []
    failed: list[str] = []
    for variant in args.variants:
        try:
            r = benchmark_variant(variant, candles_1m, candles_5m, pred_len, args.warm_runs, batch_candles)
            results.append(r)
        except Exception as exc:
            print(f"[{variant}] benchmark FAILED – stopping as required: {exc}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            failed.append(variant)
            # per task: if installation or model loading fails, report exact failure and stop
            if variant == args.variants[0]:
                print("First variant failed – aborting further benchmarks.", file=sys.stderr)
                break

    # -- comparison table
    print("\n" + "=" * 100)
    print("Comparison Table (Phase 3A)")
    print("=" * 100)
    header = f"{'Model':<12} {'Params':<8} {'Device':<8} {'Init s':<8} {'Warm 1-sym ms':<14} {'Warm 5-sym batch ms':<18} {'RAM MB':<10} {'VRAM MB':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        spec = r["spec"]
        ram = r["ram_delta_mb"]
        ram_str = f"{ram:.0f}" if ram is not None else "n/a"
        # also show peak
        vram = r["vram_mb"]
        vram_str = f"{vram:.0f}" if vram is not None else "n/a (CPU)"
        print(
            f"{spec.display_name:<12} {spec.params:<8} {detect_environment()['device']:<8} {r['init_time_s']:<8.1f} {r['warm_1m_ms']:<14.0f} {r['batch_warm_ms']:<18.0f} {ram_str:<10} {vram_str:<10}"
        )
    if failed:
        print(f"\nFailed variants: {', '.join(failed)}")

    # -- recommendation
    print("\nRecommendation (based on this environment)")
    print("-" * 60)
    env_dev = detect_environment()["device"]
    for r in results:
        warm = r["warm_1m_ms"]
        batch = r["batch_warm_ms"]
        name = r["spec"].display_name
        # thresholds: 1m scalping needs < ~200ms warm per symbol and <1s batch?
        # On CPU, 4GB RAM, i3-2120, likely too slow
        if warm < 300 and batch < 2000:
            verdict = "suitable for 1m scalping (warm latency OK)"
        elif warm < 2000:
            verdict = "suitable only for offline research (too slow for 1m)"
        else:
            verdict = "not practical in current environment (too slow / heavy)"
        print(f"  {name}: warm {warm:.0f} ms, batch {batch:.0f} ms -> {verdict}")
    # also note mini vs base
    if results:
        mini = next((x for x in results if x["variant"] == "mini"), None)
        base = next((x for x in results if x["variant"] == "base"), None)
        if mini and base:
            print(f"\nMini vs Base: mini {mini['warm_1m_ms']:.0f} ms vs base {base['warm_1m_ms']:.0f} ms (ratio {base['warm_1m_ms']/mini['warm_1m_ms']:.1f}x)")
            print("Primary candidate for future scalping is Kronos-mini per task; base is benchmark-only.")
            if mini["warm_1m_ms"] < base["warm_1m_ms"]:
                print("Mini is faster as expected (4.1M vs 102.3M).")
        if env_dev == "cpu":
            print("Note: Running on CPU only – latencies are 5-10x higher than CUDA; not indicative of GPU scalping viability.")
        print("\nBenchmark complete. HF cache remains for reuse; no trading/execution was triggered.")


if __name__ == "__main__":
    main()
