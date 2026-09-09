"""Metrics for Phase 3B walk-forward backtest (isolated, deterministic).

Computes per-symbol and aggregate:
  net PnL, trade count, win rate, avg win/loss, profit factor, max drawdown, Sharpe, turnover.

Pure functions, no I/O, no lookahead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

__all__ = [
    "Trade",
    "EquityPoint",
    "compute_metrics",
    "compute_max_drawdown",
    "compute_sharpe",
    "Metrics",
]


@dataclass(frozen=True, slots=True)
class Trade:
    entry_time: str  # iso
    exit_time: str
    side: str  # BUY/SELL/HOLD
    entry_price: Decimal
    exit_price: Decimal
    gross_bps: Decimal
    cost_bps: Decimal
    net_bps: Decimal
    # net return in fraction: net_bps/10000


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: str
    equity: float  # cumulative equity (start 1.0)


@dataclass(frozen=True, slots=True)
class Metrics:
    net_pnl_bps: Decimal
    net_pnl_pct: Decimal
    trade_count: int
    win_rate: float
    avg_win_bps: Decimal
    avg_loss_bps: Decimal
    profit_factor: float  # sum wins / abs(sum losses)
    max_drawdown_pct: float
    sharpe_ratio: float  # daily Sharpe (252)
    turnover: float  # trades per day *2 or total turnover
    avg_net_bps: Decimal
    std_net_bps: float


def compute_max_drawdown(equity: Sequence[float]) -> float:
    """Max drawdown as fraction (0-1). equity starts at 1.0."""
    peak = equity[0] if equity else 1.0
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak != 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_sharpe(daily_returns: Sequence[float]) -> float:
    """Annualized Sharpe from daily net returns (mean/std * sqrt(252)). Returns 0 if std=0."""
    if len(daily_returns) < 2:
        return 0.0
    mean = sum(daily_returns) / len(daily_returns)
    var = sum((x - mean) ** 2 for x in daily_returns) / (len(daily_returns) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    # Annualize: 252 trading days
    return (mean / std) * math.sqrt(252)


def compute_metrics(
    trades: Sequence[Trade],
    equity: Sequence[float],
    daily_returns: Sequence[float] | None = None,
) -> Metrics:
    if not trades:
        return Metrics(
            net_pnl_bps=Decimal("0"),
            net_pnl_pct=Decimal("0"),
            trade_count=0,
            win_rate=0.0,
            avg_win_bps=Decimal("0"),
            avg_loss_bps=Decimal("0"),
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            turnover=0.0,
            avg_net_bps=Decimal("0"),
            std_net_bps=0.0,
        )

    net_bps_list = [t.net_bps for t in trades]
    net_sum = sum(net_bps_list, Decimal("0"))
    net_pct = (net_sum / Decimal("100"))  # bps to pct? 100bps=1% => divide 100
    # Actually bps/100 = pct? 100 bps =1% => bps/100 = pct
    # Keep as net_pnl_bps and net_pnl_pct
    wins = [t for t in trades if t.net_bps > 0]
    losses = [t for t in trades if t.net_bps < 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_win = (sum((t.net_bps for t in wins), Decimal("0")) / len(wins)) if wins else Decimal("0")
    avg_loss = (sum((t.net_bps for t in losses), Decimal("0")) / len(losses)) if losses else Decimal("0")
    sum_wins = sum((t.net_bps for t in wins), Decimal("0"))
    sum_losses_abs = abs(sum((t.net_bps for t in losses), Decimal("0")))
    if sum_losses_abs == 0:
        pf = float("inf") if sum_wins > 0 else 0.0
    else:
        pf = float(sum_wins / sum_losses_abs)

    # Drawdown
    dd = compute_max_drawdown(equity)
    # Sharpe
    if daily_returns is None:
        # Fallback: compute daily from equity if not provided: use trade-level returns?
        # Use per-trade net returns as proxy
        per_trade_rets = [float(t.net_bps) / 10000.0 for t in trades]
        # Approximate daily: aggregate per day? Use trade rets directly for Sharpe without annualization scaling?
        # We'll compute trade-level Sharpe and annualize with sqrt(252*288) not needed – use trade sharpe
        if len(per_trade_rets) >= 2:
            mean = sum(per_trade_rets) / len(per_trade_rets)
            var = sum((x - mean) ** 2 for x in per_trade_rets) / (len(per_trade_rets) - 1)
            std = math.sqrt(var) if var > 0 else 0
            sharpe = (mean / std * math.sqrt(252 * 288 / 5)) if std != 0 else 0.0  # approx for 5m stride
            # But to keep stable, use simple mean/std without annualization
            # We'll provide raw
        else:
            sharpe = 0.0
    else:
        sharpe = compute_sharpe(daily_returns)

    avg_net = net_sum / len(trades) if trades else Decimal("0")
    # std of net_bps
    if len(net_bps_list) >= 2:
        mean_f = float(avg_net)
        var = sum((float(x) - mean_f) ** 2 for x in net_bps_list) / (len(net_bps_list) - 1)
        std_bps = math.sqrt(var)
    else:
        std_bps = 0.0

    # Turnover: trades per day proxy, or total notional turnover 2*trade_count
    # Provide trades per day if we can estimate days from equity length?
    turnover = float(len(trades) * 2)  # total legs

    return Metrics(
        net_pnl_bps=net_sum,
        net_pnl_pct=net_pct,
        trade_count=len(trades),
        win_rate=win_rate,
        avg_win_bps=avg_win,
        avg_loss_bps=avg_loss,
        profit_factor=pf,
        max_drawdown_pct=dd * 100,
        sharpe_ratio=sharpe,
        turnover=turnover,
        avg_net_bps=avg_net,
        std_net_bps=std_bps,
    )
