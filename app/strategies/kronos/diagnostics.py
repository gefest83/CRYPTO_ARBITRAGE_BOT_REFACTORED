"""Phase 3C diagnostic calculations (research-only, isolated, deterministic).

Pure functions over walk-forward (pred_bps, realized_bps) pairs.
No torch, no network, no lookahead — caller builds pairs with strict
t-only windows (see scripts/run_kronos_diagnostics.py).

Conventions:
- pred_bps: predicted 5-minute gross move in bps (long basis):
    (pred_close - last_close)/last_close * 10000
- realized_bps: realized 5-minute gross move in bps (long basis):
    (exit_close - entry_open)/entry_open * 10000
    where entry is t+1 open, exit is t+5 close (same as backtest engine).
- Signal for threshold thr reuses production logic:
    net_pred = compute_net_bps(pred_gross, cost)  (clamped, production)
    BUY iff net_pred > thr, SELL iff net_pred < -thr, else HOLD.
- Trade PnL (diagnostic, realistic, unclamped):
    gross_trade = realized if BUY else -realized
    net_trade = gross_trade - cost_bps
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from app.strategies.kronos.cost import compute_net_bps

__all__ = [
    "Sample",
    "quantile",
    "distribution_stats",
    "threshold_coverage",
    "directional_accuracy",
    "pearson_corr",
    "trade_stats_for_threshold",
    "threshold_sensitivity",
    "DEFAULT_THRESHOLDS",
    "COVERAGE_THRESHOLDS",
]

DEFAULT_THRESHOLDS: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 30.0, 50.0)
COVERAGE_THRESHOLDS: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 70.0)


@dataclass(frozen=True, slots=True)
class Sample:
    timestamp: str
    pred_bps: float
    realized_bps: float


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile in [0,1]. Returns 0.0 for empty."""
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))
    s = sorted(values)
    n = len(s)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def distribution_stats(values: Sequence[float]) -> dict:
    """Distribution of predicted returns (signed bps)."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "abs_mean": 0.0,
            "abs_p50": 0.0,
            "abs_p90": 0.0,
            "abs_max": 0.0,
        }
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1) if n >= 2 else 0.0
    std = math.sqrt(var)
    abs_vals = [abs(x) for x in vals]
    abs_mean = sum(abs_vals) / n
    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": float(min(vals)),
        "p50": quantile(vals, 0.50),
        "p75": quantile(vals, 0.75),
        "p90": quantile(vals, 0.90),
        "p95": quantile(vals, 0.95),
        "max": float(max(vals)),
        "abs_mean": abs_mean,
        "abs_p50": quantile(abs_vals, 0.50),
        "abs_p90": quantile(abs_vals, 0.90),
        "abs_max": float(max(abs_vals)),
    }


def threshold_coverage(values: Sequence[float], thresholds: Sequence[float] = COVERAGE_THRESHOLDS) -> dict:
    """Count/% of |pred| above each threshold. Uses absolute predicted size."""
    vals = [float(v) for v in values]
    n = len(vals)
    out: dict[str, dict] = {}
    for thr in thresholds:
        c = sum(1 for v in vals if abs(v) > float(thr))
        out[str(thr)] = {"count": c, "pct": (c / n * 100.0) if n else 0.0}
    return out


def directional_accuracy(preds: Sequence[float], realized: Sequence[float]) -> dict:
    """Sign agreement: correct iff (pred>0 and real>0) or (pred<0 and real<0).

    Zero pred or zero realized counts as incorrect (no edge). Returns
    accuracy, n, correct, plus long-only / short-only splits.
    """
    p = [float(x) for x in preds]
    r = [float(x) for x in realized]
    assert len(p) == len(r)
    n = len(p)
    if n == 0:
        return {"n": 0, "correct": 0, "accuracy": 0.0, "long_n": 0, "long_acc": 0.0, "short_n": 0, "short_acc": 0.0}
    correct = sum(1 for a, b in zip(p, r) if (a > 0 and b > 0) or (a < 0 and b < 0))
    long_idx = [i for i, a in enumerate(p) if a > 0]
    short_idx = [i for i, a in enumerate(p) if a < 0]
    long_correct = sum(1 for i in long_idx if r[i] > 0)
    short_correct = sum(1 for i in short_idx if r[i] < 0)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "long_n": len(long_idx),
        "long_acc": (long_correct / len(long_idx)) if long_idx else 0.0,
        "short_n": len(short_idx),
        "short_acc": (short_correct / len(short_idx)) if short_idx else 0.0,
    }


def pearson_corr(preds: Sequence[float], realized: Sequence[float]) -> float:
    """Pearson correlation. 0.0 if undefined (n<2 or zero variance)."""
    p = [float(x) for x in preds]
    r = [float(x) for x in realized]
    n = len(p)
    if n < 2 or len(r) != n:
        return 0.0
    mp = sum(p) / n
    mr = sum(r) / n
    cov = sum((a - mp) * (b - mr) for a, b in zip(p, r))
    vp = sum((a - mp) ** 2 for a in p)
    vr = sum((b - mr) ** 2 for b in r)
    if vp == 0 or vr == 0:
        return 0.0
    return cov / math.sqrt(vp * vr)


def _signal_for_pred(pred_gross: float, thr: float, cost: Decimal) -> int:
    """+1 BUY, -1 SELL, 0 HOLD using production compute_net_bps."""
    net = compute_net_bps(Decimal(str(pred_gross)), cost)
    fthr = float(thr)
    fn = float(net)
    if fn > fthr:
        return 1
    if fn < -fthr:
        return -1
    return 0


def trade_stats_for_threshold(
    samples: Sequence[Sample],
    threshold_bps: float,
    cost_bps: Decimal,
) -> dict:
    """Apply threshold to samples -> trades -> gross/net PnL, win rate, PF."""
    cost_f = float(cost_bps)
    gross_trades: list[float] = []
    net_trades: list[float] = []
    for s in samples:
        side = _signal_for_pred(s.pred_bps, float(threshold_bps), cost_bps)
        if side == 0:
            continue
        gross = float(s.realized_bps) if side > 0 else -float(s.realized_bps)
        net = gross - cost_f
        gross_trades.append(gross)
        net_trades.append(net)
    n = len(net_trades)
    gross_sum = float(sum(gross_trades)) if gross_trades else 0.0
    net_sum = float(sum(net_trades)) if net_trades else 0.0
    if n == 0:
        return {
            "threshold_bps": float(threshold_bps),
            "trade_count": 0,
            "gross_pnl_bps": 0.0,
            "net_pnl_bps": 0.0,
            "win_rate": 0.0,
            "avg_win_bps": 0.0,
            "avg_loss_bps": 0.0,
            "profit_factor": 0.0,
        }
    wins = [x for x in net_trades if x > 0]
    losses = [x for x in net_trades if x < 0]
    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    sum_w = sum(wins)
    sum_l = abs(sum(losses)) if losses else 0.0
    pf = (sum_w / sum_l) if sum_l != 0 else (float("inf") if sum_w > 0 else 0.0)
    return {
        "threshold_bps": float(threshold_bps),
        "trade_count": n,
        "gross_pnl_bps": gross_sum,
        "net_pnl_bps": net_sum,
        "win_rate": win_rate,
        "avg_win_bps": avg_win,
        "avg_loss_bps": avg_loss,
        "profit_factor": pf,
    }


def threshold_sensitivity(
    samples: Sequence[Sample],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    cost_bps: Decimal = Decimal("35"),
) -> dict:
    out: dict[str, dict] = {}
    for thr in thresholds:
        out[str(thr)] = trade_stats_for_threshold(samples, float(thr), cost_bps)
    return out
