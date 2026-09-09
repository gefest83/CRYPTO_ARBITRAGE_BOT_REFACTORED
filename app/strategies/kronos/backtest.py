"""Phase 3B walk-forward backtest engine (research-only, isolated).

Strict walk-forward, no lookahead:
  at t, use only candles through t; forecast; enter at t+1 using next open; hold 5m.

Four strategies:
  1. kronos_1m  – real Kronos-mini 1m → 5-min (pred_len 5)
  2. kronos_5m  – real Kronos-mini 5m → 5-min (pred_len 1, 5m resampled)
  3. momentum   – non-Kronos baseline (20-bar momentum)
  4. buy_hold   – single long from first to last

Costs: reuse app.strategies.kronos.cost / signal logic.
Cache/reuse forecasts per backtest run.
No connection to DEMO/LIVE/RiskEngine/Telegram.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from app.strategies.kronos.config import KronosConfig
from app.strategies.kronos.cost import CostInputs, compute_cost_bps, compute_gross_bps, compute_net_bps
from app.strategies.kronos.metrics import EquityPoint, Metrics, Trade, compute_metrics, compute_max_drawdown, compute_sharpe
from app.strategies.kronos.scanner import resample_to_5m
from app.strategies.kronos.signal import classify_signal
from app.strategies.kronos.types import Candle, ForecastBar, Signal

UTC = timezone.utc

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BacktestEngine",
    "SignalGenerator",
]

# ---------------------------------------------------------------- config
@dataclass(frozen=True, slots=True)
class BacktestConfig:
    # Kronos params (reuses KronosConfig for signal)
    lookback_1m: int = 400
    lookback_5m: int = 80  # for 5m mode: 80*5m = 400m history
    pred_len_1m: int = 5  # 5 *1m =5min
    pred_len_5m: int = 1  # 1 *5m =5min
    horizon_bars: int = 5  # holding horizon in 1m bars
    stride_bars: int = 5  # step between decisions (5 = no overlap, 1 = every minute)
    # costs / edge
    min_edge_bps: Decimal = Decimal("15")
    taker_fee_bps: Decimal = Decimal("10")
    spread_bps: Decimal = Decimal("5")
    slippage_bps: Decimal = Decimal("5")
    cost_model: str = "round_trip"  # realistic for backtest; switch to one_way to match scanner
    # momentum baseline period
    momentum_period: int = 20

    def cost_bps(self) -> Decimal:
        ci = CostInputs(
            taker_fee_bps=self.taker_fee_bps,
            spread_bps=self.spread_bps,
            slippage_bps=self.slippage_bps,
            cost_model=self.cost_model,
        )
        return compute_cost_bps(ci)

    def to_kronos_config_1m(self) -> KronosConfig:
        return KronosConfig(
            lookback=self.lookback_1m,
            pred_len_1m=self.pred_len_1m,
            pred_len_5m=self.pred_len_5m,
            min_edge_bps=self.min_edge_bps,
            taker_fee_bps=self.taker_fee_bps,
            spread_bps=self.spread_bps,
            slippage_bps=self.slippage_bps,
            cost_model=self.cost_model,  # type: ignore
            require_5m_confirmation=False,
            max_candle_age_ms=600_000,
            max_gap_ms=600_000,
            allow_incomplete_last=False,
        )

    def to_kronos_config_5m(self) -> KronosConfig:
        # For 5m mode we still use same thresholds but lookback 80
        return KronosConfig(
            lookback=self.lookback_5m,
            pred_len_1m=self.pred_len_5m,  # trick: reuse 1m slot for 5m horizon
            pred_len_5m=1,
            min_edge_bps=self.min_edge_bps,
            taker_fee_bps=self.taker_fee_bps,
            spread_bps=self.spread_bps,
            slippage_bps=self.slippage_bps,
            cost_model=self.cost_model,  # type: ignore
            require_5m_confirmation=False,
            max_candle_age_ms=600_000,
            max_gap_ms=600_000,
            allow_incomplete_last=False,
        )


# ---------------------------------------------------------------- result
@dataclass(frozen=True, slots=True)
class BacktestResult:
    strategy: str
    symbol: str
    trades: tuple[Trade, ...]
    equity: tuple[EquityPoint, ...]
    metrics: Metrics
    inference_count: int
    inference_time_s: float
    # per trade details for debugging
    horizon_bars: int
    stride_bars: int


# ---------------------------------------------------------------- execution helpers
def _executable_entry_exit(
    candles: Sequence[Candle],
    t_idx: int,
    horizon: int,
    side: Signal,
    cost_bps: Decimal,
) -> tuple[Decimal, Decimal, Decimal, Decimal] | None:
    """Return (entry_price, exit_price, gross_bps, net_bps) for trade starting at t.

    t is decision index (candle at t used for signal). Entry at t+1 open, exit at t+1+horizon-1 close.
    No lookahead beyond t for signal, but entry/exit uses future candles t+1..t+horizon (evaluated after entry).

    Returns None if not enough future candles.
    side must be BUY or SELL. HOLD never calls this.
    """
    entry_idx = t_idx + 1
    exit_idx = t_idx + horizon  # because entry at t+1 open, hold horizon bars: exit at close of t+horizon
    # Need horizon bars after t: so exit_idx < len(candles)
    if exit_idx >= len(candles) or entry_idx >= len(candles):
        return None
    entry_candle = candles[entry_idx]
    exit_candle = candles[exit_idx]
    entry_price = entry_candle.open
    exit_price = exit_candle.close
    if entry_price <= 0 or exit_price <= 0:
        return None
    if side == Signal.BUY:
        gross = compute_gross_bps(entry_price, exit_price)
    elif side == Signal.SELL:
        # For short, gross is (entry - exit)/entry
        gross = compute_gross_bps(exit_price, entry_price)  # this gives negative if exit>entry
        # compute_gross_bps does (pred - last)/last ; for short we want (entry - exit)/entry = - (exit-entry)/entry
        # So we can compute as - compute_gross_bps(entry, exit)
        # Let's use direct: (entry - exit)/entry *10000 = - (exit-entry)/entry
        gross = (-compute_gross_bps(entry_price, exit_price))
        # But compute_gross_bps already handles quant, so invert
    else:
        return None
    net = compute_net_bps(gross, cost_bps)
    # Note: compute_net_bps clamps to 0 if cost > gross, we want signed net; but original clamps.
    # For backtest we want actual net = gross - cost (signed) not clamped? However spec says reuse existing logic.
    # We'll keep clamped behavior for signal, but for PnL we want real net = gross - cost_bps (if BUY) or gross - cost_bps etc without clamping?
    # To be realistic, net should be gross - cost_bps without clamping (can be negative).
    # We'll compute raw net = gross - cost_bps for BUY, gross - cost_bps for SELL? Actually SELL gross already negative if price up, so gross - cost? Need signed.
    # For simplicity, use compute_net_bps which clamps at 0; that would under-count losses. So we compute raw.
    # Override: raw net = gross - cost_bps if gross>0 else gross + cost_bps? No for PnL we always subtract cost.
    # Realistic: net = gross - cost_bps for long, net = gross - cost_bps for short as well (cost always reduces PnL)
    # We'll compute both and store real_net = gross - cost_bps (signed)
    # But to match production signal logic, we keep net_signal = compute_net_bps
    # For PnL, use real_net
    real_net = (gross - cost_bps).quantize(Decimal("0.0001")) if side == Signal.BUY else (gross - cost_bps).quantize(Decimal("0.0001"))
    # For short, gross negative when price up, so gross - cost will be more negative, correct.
    # However if gross is positive for short (price down), net = positive gross - cost
    return entry_price, exit_price, gross, real_net


def _daily_returns_from_trades(trades: Sequence[Trade], candles: Sequence[Candle]) -> list[float]:
    """Aggregate trade net returns by calendar day for Sharpe."""
    if not trades:
        return []
    # Group by entry_time date
    from collections import defaultdict

    daily: dict[str, float] = defaultdict(float)
    for t in trades:
        day = t.entry_time[:10]  # YYYY-MM-DD
        daily[day] += float(t.net_bps) / 10000.0
    # Sorted by day
    sorted_days = sorted(daily.keys())
    return [daily[d] for d in sorted_days]


# ---------------------------------------------------------------- signal generators
class SignalGenerator:
    """Generates signals for each strategy. Supports caching for Kronos."""

    def __init__(self, config: BacktestConfig, predictor_1m=None, predictor_5m=None):
        self.config = config
        self.predictor_1m = predictor_1m
        self.predictor_5m = predictor_5m
        # cache: (strategy, hash(window)) -> (Signal, net_bps, gross_bps)
        self._cache: dict[tuple[str, str], tuple[Signal, Decimal, Decimal]] = {}
        self.inference_count = 0
        self.inference_time_s = 0.0

    def _hash_window(self, candles: Sequence[Candle]) -> str:
        # Hash of window closes + timestamps to key cache
        h = hashlib.md5()
        for c in candles:
            h.update(str(c.timestamp).encode())
            h.update(str(c.close).encode())
        return h.hexdigest()

    def kronos_1m_signal(self, window_1m: Sequence[Candle]) -> tuple[Signal, Decimal, Decimal]:
        """Use predictor_1m on 1m window, pred_len 5."""
        if self.predictor_1m is None:
            return Signal.HOLD, Decimal("0"), Decimal("0")
        key = ("kronos_1m", self._hash_window(window_1m))
        if key in self._cache:
            return self._cache[key]
        t0 = time.perf_counter()
        try:
            forecast: Sequence[ForecastBar] = self.predictor_1m.predict(window_1m, self.config.pred_len_1m)  # type: ignore
        except Exception:
            self._cache[key] = (Signal.HOLD, Decimal("0"), Decimal("0"))
            return self._cache[key]
        dt = time.perf_counter() - t0
        self.inference_count += 1
        self.inference_time_s += dt
        if len(forecast) != self.config.pred_len_1m:
            self._cache[key] = (Signal.HOLD, Decimal("0"), Decimal("0"))
            return self._cache[key]
        last_close = window_1m[-1].close
        pred_close = forecast[-1].close
        gross = compute_gross_bps(last_close, pred_close)
        cost = self.config.cost_bps()
        net = compute_net_bps(gross, cost)
        sig, _ = classify_signal(net_bps_1m=net, min_edge_bps=self.config.min_edge_bps)
        self._cache[key] = (sig, net, gross)
        return sig, net, gross

    def kronos_5m_signal(self, window_1m: Sequence[Candle]) -> tuple[Signal, Decimal, Decimal]:
        """Resample 1m window to 5m, use predictor_5m (or same predictor) with pred_len 1."""
        if self.predictor_5m is None and self.predictor_1m is None:
            return Signal.HOLD, Decimal("0"), Decimal("0")
        predictor = self.predictor_5m or self.predictor_1m
        # Resample to 5m
        candles_5m = resample_to_5m(window_1m)
        # Need at least 10 5m bars; use whatever available up to lookback_5m (handles 79 vs 80 alignment)
        if len(candles_5m) < 10:
            return Signal.HOLD, Decimal("0"), Decimal("0")
        use_len = min(len(candles_5m), self.config.lookback_5m)
        window_5m = candles_5m[-use_len:]
        key = ("kronos_5m", self._hash_window(window_5m))
        if key in self._cache:
            return self._cache[key]
        t0 = time.perf_counter()
        try:
            forecast = predictor.predict(window_5m, self.config.pred_len_5m)  # type: ignore
        except Exception:
            self._cache[key] = (Signal.HOLD, Decimal("0"), Decimal("0"))
            return self._cache[key]
        dt = time.perf_counter() - t0
        self.inference_count += 1
        self.inference_time_s += dt
        if len(forecast) != self.config.pred_len_5m:
            self._cache[key] = (Signal.HOLD, Decimal("0"), Decimal("0"))
            return self._cache[key]
        last_close = window_5m[-1].close
        pred_close = forecast[-1].close
        gross = compute_gross_bps(last_close, pred_close)
        cost = self.config.cost_bps()
        net = compute_net_bps(gross, cost)
        sig, _ = classify_signal(net_bps_1m=net, min_edge_bps=self.config.min_edge_bps)
        self._cache[key] = (sig, net, gross)
        return sig, net, gross

    def momentum_signal(self, window_1m: Sequence[Candle]) -> tuple[Signal, Decimal, Decimal]:
        """Simple momentum: (close[t] / close[t-N] -1)*10000. Same threshold as Kronos."""
        n = self.config.momentum_period
        if len(window_1m) < n + 1:
            return Signal.HOLD, Decimal("0"), Decimal("0")
        last = window_1m[-1].close
        past = window_1m[-1 - n].close
        if past <= 0 or last <= 0:
            return Signal.HOLD, Decimal("0"), Decimal("0")
        gross = compute_gross_bps(past, last)
        cost = self.config.cost_bps()
        net = compute_net_bps(gross, cost)
        sig, _ = classify_signal(net_bps_1m=net, min_edge_bps=self.config.min_edge_bps)
        return sig, net, gross

    def reset_stats(self):
        self.inference_count = 0
        self.inference_time_s = 0.0
        # keep cache across runs for same data


# ---------------------------------------------------------------- engine
class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.cost_bps = self.config.cost_bps()

    def run_strategy(
        self,
        symbol: str,
        candles: Sequence[Candle],
        strategy: str,
        signal_gen: SignalGenerator | None = None,
    ) -> BacktestResult:
        """Run single strategy walk-forward.

        strategy in {"kronos_1m","kronos_5m","momentum","buy_hold"}
        signal_gen required for kronos/momentum, not for buy_hold.
        """
        assert strategy in ("kronos_1m", "kronos_5m", "momentum", "buy_hold")
        if signal_gen is not None:
            # Reset cache stats per strategy run? Keep cumulative.
            pass

        trades: list[Trade] = []
        # For buy_hold, single trade
        if strategy == "buy_hold":
            if len(candles) < self.config.lookback_1m + self.config.horizon_bars + 1:
                # Not enough data
                equity = (EquityPoint(timestamp=candles[0].timestamp.isoformat(), equity=1.0),)
                metrics = compute_metrics([], [1.0], [])
                return BacktestResult(strategy, symbol, (), equity, metrics, 0, 0.0, self.config.horizon_bars, self.config.stride_bars)
            t = self.config.lookback_1m - 1  # first decision point
            entry_idx = t + 1
            exit_idx = len(candles) - 1
            entry_price = candles[entry_idx].open
            exit_price = candles[exit_idx].close
            gross = compute_gross_bps(entry_price, exit_price)
            net = (gross - self.cost_bps).quantize(Decimal("0.0001"))
            trades.append(
                Trade(
                    entry_time=candles[entry_idx].timestamp.isoformat(),
                    exit_time=candles[exit_idx].timestamp.isoformat(),
                    side="BUY",
                    entry_price=entry_price,
                    exit_price=exit_price,
                    gross_bps=gross,
                    cost_bps=self.cost_bps,
                    net_bps=net,
                )
            )
            # equity
            equity_vals = [1.0, 1.0 + float(net) / 10000.0]
            equity = (
                EquityPoint(timestamp=candles[entry_idx].timestamp.isoformat(), equity=1.0),
                EquityPoint(timestamp=candles[exit_idx].timestamp.isoformat(), equity=equity_vals[1]),
            )
            daily_rets = _daily_returns_from_trades(trades, candles)
            metrics = compute_metrics(trades, equity_vals, daily_rets)
            inf_c = signal_gen.inference_count if signal_gen else 0
            inf_t = signal_gen.inference_time_s if signal_gen else 0.0
            return BacktestResult(strategy, symbol, tuple(trades), equity, metrics, inf_c, inf_t, self.config.horizon_bars, self.config.stride_bars)

        # For other strategies: walk-forward
        start_t = self.config.lookback_1m - 1
        # Need horizon bars after t+1 for exit, so last t is len - horizon -1
        end_t = len(candles) - self.config.horizon_bars - 1
        if end_t < start_t:
            equity = (EquityPoint(timestamp=candles[0].timestamp.isoformat(), equity=1.0),)
            metrics = compute_metrics([], [1.0], [])
            return BacktestResult(strategy, symbol, (), equity, metrics, 0, 0.0, self.config.horizon_bars, self.config.stride_bars)

        equity_curve: list[float] = [1.0]
        equity_points: list[EquityPoint] = [EquityPoint(timestamp=candles[start_t].timestamp.isoformat(), equity=1.0)]
        cur_equity = 1.0

        t = start_t
        # Use while to allow stride
        while t <= end_t:
            window = candles[t - self.config.lookback_1m + 1 : t + 1] if strategy != "kronos_5m" else candles[max(0, t - 400 + 1) : t + 1]
            # For kronos_5m we pass 1m window to signal_gen which resamples internally; so use 400 window for both but 5m uses resampled
            # For momentum use same 1m window
            if strategy == "kronos_1m":
                sig, _, _ = signal_gen.kronos_1m_signal(window) if signal_gen else (Signal.HOLD, Decimal("0"), Decimal("0"))
            elif strategy == "kronos_5m":
                sig, _, _ = signal_gen.kronos_5m_signal(window) if signal_gen else (Signal.HOLD, Decimal("0"), Decimal("0"))
            elif strategy == "momentum":
                sig, _, _ = signal_gen.momentum_signal(window) if signal_gen else (Signal.HOLD, Decimal("0"), Decimal("0"))
            else:
                sig = Signal.HOLD

            if sig in (Signal.BUY, Signal.SELL):
                res = _executable_entry_exit(candles, t, self.config.horizon_bars, sig, self.cost_bps)
                if res is not None:
                    entry_price, exit_price, gross, net = res
                    # For PnL we already have net (real)
                    trades.append(
                        Trade(
                            entry_time=candles[t + 1].timestamp.isoformat(),
                            exit_time=candles[t + self.config.horizon_bars].timestamp.isoformat(),
                            side=sig,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            gross_bps=gross,
                            cost_bps=self.cost_bps,
                            net_bps=net,
                        )
                    )
                    cur_equity *= 1.0 + float(net) / 10000.0
                    equity_curve.append(cur_equity)
                    equity_points.append(EquityPoint(timestamp=candles[t + self.config.horizon_bars].timestamp.isoformat(), equity=cur_equity))
                else:
                    # Not enough future, skip
                    equity_curve.append(cur_equity)
                    equity_points.append(EquityPoint(timestamp=candles[t].timestamp.isoformat(), equity=cur_equity))
            else:
                # HOLD: no trade, equity unchanged but we still record point for drawdown? Keep flat.
                equity_curve.append(cur_equity)
                equity_points.append(EquityPoint(timestamp=candles[t].timestamp.isoformat(), equity=cur_equity))

            t += self.config.stride_bars

        # If HOLDs produced flat equity points but we want equity only on trade exits, we already have curve.
        # For metrics we need equity curve including all steps (including flat).
        # Use equity_curve list
        daily_rets = _daily_returns_from_trades(trades, candles)
        metrics = compute_metrics(trades, equity_curve, daily_rets)
        inf_c = signal_gen.inference_count if signal_gen else 0
        inf_t = signal_gen.inference_time_s if signal_gen else 0.0
        return BacktestResult(strategy, symbol, tuple(trades), tuple(equity_points), metrics, inf_c, inf_t, self.config.horizon_bars, self.config.stride_bars)

    def run_all(
        self,
        symbol: str,
        candles: Sequence[Candle],
        signal_gen: SignalGenerator | None = None,
    ) -> dict[str, BacktestResult]:
        results: dict[str, BacktestResult] = {}
        # For kronos strategies we need signal_gen with predictor; for momentum we need same gen; for buy_hold none
        # To avoid cross-contamination of inference counts, we clone counts per strategy? But we want cache reuse across strategies where possible.
        # We'll run sequentially and capture counts delta.
        for strat in ("kronos_1m", "kronos_5m", "momentum", "buy_hold"):
            # Reset inference counts before each? But task wants runtime/inference cost per strategy. So per result we report count for that strategy only.
            # We'll create fresh SignalGenerator per strategy if needed, but share predictor handles.
            # Approach: use single gen but snapshot count before/after.
            if signal_gen is not None:
                before_c = signal_gen.inference_count
                before_t = signal_gen.inference_time_s
            else:
                before_c, before_t = 0, 0.0
            res = self.run_strategy(symbol, candles, strat, signal_gen if strat != "buy_hold" else None)
            if signal_gen is not None:
                # Adjust result to show delta only for that strategy
                # But BacktestResult already stores cumulative; we want delta
                delta_c = signal_gen.inference_count - before_c
                delta_t = signal_gen.inference_time_s - before_t
                # Patch result with delta (create new)
                res = BacktestResult(
                    strategy=res.strategy,
                    symbol=res.symbol,
                    trades=res.trades,
                    equity=res.equity,
                    metrics=res.metrics,
                    inference_count=delta_c,
                    inference_time_s=delta_t,
                    horizon_bars=res.horizon_bars,
                    stride_bars=res.stride_bars,
                )
            results[strat] = res
        return results
