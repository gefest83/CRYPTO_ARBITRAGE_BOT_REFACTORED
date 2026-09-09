# Kronos Research — Parked

> **Research-only. Not connected to live/DEMO trading, RiskEngine, execution, Telegram, triangle or transfer strategies.**

This directory contains the isolated Kronos scalping research stack (Phases 2–3C + calibration). All production trading remains unchanged.

## What was implemented

- **Forecast interface + mock** (`forecast.py`, `types.py`) — `KronosForecastModel` protocol, deterministic `MockKronosPredictor` (no torch/network, used for CI).
- **Cost / signal logic** (`cost.py`, `signal.py`, `config.py`) — `CostInputs` (`taker 10bps, spread 5bps, slippage 5bps`), `compute_gross_bps / compute_cost_bps / compute_net_bps`, `classify_signal` strict `> / <` threshold, optional 5m confirmation. Reused verbatim in backtests.
- **Scanner** (`scanner.py`) — offline validation, `resample_to_5m`, no lookahead (slices to `lookback` tail), stale/gap guards.
- **Real adapter** (`real.py` + `model/kronos.py`, `model/module.py`) — lazy `RealKronosPredictor` wrapping the vendored upstream `shiyu-coder/Kronos` (`Kronos`, `KronosTokenizer`, `KronosPredictor`). Model/tokenizer are downloaded from Hugging Face only when `load()` is called; normal tests never import `torch`.
- **Historical data loader** (`data.py`) — Binance public `api/v3/klines` for `1m` OHLCV, paginated, disk-cached under `data/kronos/binance_*.csv`, returns `Candle` (`Decimal`, `UTC`, `complete=True`).
- **Backtest engine** (`backtest.py`) — strict walk-forward (`t` uses only candles through `t`, forecast, enter at `t+1` open, hold 5m, exit at `t+5` close), 4 strategies, realistic `round_trip` costs, forecast-disk cache `data/kronos/forecast_cache/`.
- **Metrics** (`metrics.py`) — net PnL, trade count, win rate, avg win/loss, profit factor, max drawdown, Sharpe (daily `mean/std*sqrt252`), turnover, runtime/inference cost.
- **Diagnostics** (`diagnostics.py`) — pure walk-forward diagnostics over `(pred_bps, realized_bps)` pairs.
- **Synthetic helper** (`synthetic.py`) — deterministic candle generator for tests.

Isolation: no imports from `app.execution`, `app.risk`, `app.exchanges`, `app.auto`, `app.telegram`, `app.storage` beyond the `Candle`/`ForecastBar` value types.

## Models tested

| Model | HF ID | Tokenizer | Context | Params |
|---|---|---|---|---|
| **Kronos-mini** (primary) | `NeoQuasar/Kronos-mini` | `NeoQuasar/Kronos-Tokenizer-2k` | 2048 | 4.1M |
| Kronos-base (benchmark only) | `NeoQuasar/Kronos-base` | `NeoQuasar/Kronos-Tokenizer-base` | 512 | 102.3M |

Both use `T=1.0, top_p=0.9, top_k=0, sample_count=1, clip=5` unless a calibration grid overrides `T/top_p`. `mini` is the only candidate ever considered for live 1m scalping; `base` is benchmark-only.

## Backtest / diagnostic methodology

- **Universe:** Binance `BTC/USDT`, `ETH/USDT`, `SOL/USDT` real `1m` OHLCV, cached (e.g. `data/kronos/binance_BTCUSDT_1m_20260902_1912_20260909_1912.csv` 10080 bars for 7d).
- **Walk-forward / no lookahead:** at decision `t` the model sees `window = candles[t-lookback+1 .. t]` only. `pred_bps = (pred_close - last_close)/last_close *10000` (long basis). Realized 5m gross `= (exit_close - entry_open)/entry_open *10000` (`t+1` open → `t+5` close). A short inverts the sign; net `= gross - cost_bps` (35bps `round_trip` = `2*taker+spread+2*slippage`).
- **Strategies compared:** `kronos_1m` (400×1m → pred_len 5), `kronos_5m` (same 400×1m resampled to ~80×5m → pred_len 1), momentum (20-bar), buy-and-hold (single long). All use the same `min_edge_bps` + cost threshold via `compute_net_bps` + `classify_signal`.
- **Threshold sensitivity:** 5/10/15/20/30/50 bps as research-only sweeps.
- **Diagnostics:** per `kronos_1m`/`kronos_5m` — predicted-return distribution (`p50/p75/p90/p95/max`, `abs_*`), coverage above 5/10/15/20/30/40/50/70 bps, directional accuracy (sign agreement), Pearson correlation, gross/net PnL, trade count/win/PF at each threshold. Uses `app/strategies/kronos/diagnostics.py` pure functions; strict walk-forward pairs builder in `scripts/run_kronos_diagnostics.py`.
- **Caching:** forecast-disk cache `data/kronos/forecast_cache/*_{strategy}_lb{lookback}_{model_tag}_{t}.json` keyed by window hash avoids rerunning 1.4s 1m / 0.12s 5m inferences on CPU.
- **Horizon:** fixed 5 minutes throughout.

## Key results

- **Benchmark (i3-2120 CPU, 8GB, `torch 2.14+cpu`, no CUDA):** `mini` warm 1.4–1.6s (400×1m) vs `base` ~21s; `5m` ~0.12s. Batch 5 symbols ~7s (`mini`) vs ~104s (`base`). `mini` ~14× faster as expected.
- **7d walk-forward (stride 60, 162 pts/symbol):** 
  - `kronos_1m` / `kronos_5m` flat: BTC 7d `0 trades 0.0bps` at `edge15 round_trip`; BTC+ETH aggregate `0 / 1 losing trade -46.9bps`. Momentum `4 trades -58.6bps BTC` / `286 trades -9928bps 3-symbol aggregate`; buy-and-hold `+21bps BTC` / `+734bps aggregate`. 
  - **Diagnostic (BTC+ETH 7d, n=324):** `kronos_1m` `abs_mean 5.7bps`, `p95 13.3`, `max 30.2`, `>15 6.2%`, `>50 0%`; accuracy `51.2%`, corr `+0.045`. `kronos_5m` `abs_mean 6.0`, `p95 15.3`, `max 140`, `>15 8.6%`; accuracy `49.1%`, corr `-0.147`. Headline `edge15` net 0.0 / -46.9 gross -11.9, win 0%, PF 0. Sensitivity monontonic but all 6 thresholds net ≤0. **0 forecasts clear `35bps` cost** for 1m (0/324), 1/324 for 5m — cost zeroes the signal set.

## Calibration result

- **Grid:** BTC only, 2d (21 pts/config), cached data only, 9 configs `T {0.5,1.0,1.5} × top_p {0.8,0.9,1.0}`, 5-min horizon, `torch.manual_seed(md5(t|T|top_p))` per inference, baseline `T1.0/p0.9` reuses 21 existing `mini` cache hits. Report `data/kronos/calibration_3c.json`.
- **Outcome:** spread responds to temperature (abs_mean `3→14bps`, max `8→50bps` from `T0.5→T1.5`), confirming damping is partly sampling, but **no config material improves signal**: 0–1 trades at any threshold, net `0.0` or single `+2.7bps` (1 trade) statistically meaningless, accuracy `48–71%` on `n=21` inside ±22pp noise band, correlation swings `+0.08→+0.60` unstably. Widening the distribution does not create edge because `35bps` remains binding.

## Final conclusion

**Kronos-mini is parked for 1m/5m scalping.** Across 7-day real Binance samples, strict walk-forward, realistic `35bps round_trip` costs, and a full calibration sweep, the model shows no directional edge (accuracy ≈50%, corr ≈0), damped forecasts (`p95` ~13–15bps vs cost 35), zero repeatable gross/net PnL, and 0% win rate at every research threshold. Costs are binding, but even loosening temperature does not yield tradeable signal — the limitation is signal quality, not just cost. Further Kronos research is justified only as cheap, GPU-based, longer-horizon or mean-calibration hypotheses; no 30-day CPU backtest or live wiring is warranted.

## Scripts / reports

- `scripts/benchmark_kronos.py` — warm/cold/batch latency, RAM/VRAM, scanner check.
- `scripts/run_kronos_backtest.py` — 4-strategy walk-forward, `--days/--stride/--min-edge`, JSON to `data/kronos/backtest_*.json`.
- `scripts/run_kronos_diagnostics.py` — distribution/coverage/accuracy/correlation/sensitivity, `--cache-only` reuses CSV+forecast cache, JSON to `data/kronos/diagnostics_3c_*.json`.
- `scripts/run_kronos_calibration.py` — T/top_p grid, JSON to `data/kronos/calibration_3c.json`.

All Kronos JSON reports and `data/kronos/binance_*.csv` caches are retained on disk (ignored by `.gitignore` via `data/`, kept for reproducibility).
