"""Phase 3C deterministic tests – pure diagnostics, no torch/network."""

from __future__ import annotations

from decimal import Decimal

from app.strategies.kronos.diagnostics import (
    Sample,
    directional_accuracy,
    distribution_stats,
    pearson_corr,
    quantile,
    threshold_coverage,
    threshold_sensitivity,
    trade_stats_for_threshold,
)


def test_quantile_basic():
    vals = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert quantile(vals, 0.5) == 20.0
    assert quantile(vals, 0.0) == 0.0
    assert quantile(vals, 1.0) == 40.0
    assert quantile([], 0.5) == 0.0


def test_distribution_stats():
    vals = [1.0, 2.0, 3.0, 4.0, 100.0]
    d = distribution_stats(vals)
    assert d["count"] == 5
    assert d["p50"] == 3.0
    assert d["max"] == 100.0
    assert d["p95"] > d["p90"] >= d["p75"] >= d["p50"]
    assert d["abs_max"] == 100.0


def test_distribution_empty():
    d = distribution_stats([])
    assert d["count"] == 0
    assert d["p95"] == 0.0


def test_threshold_coverage():
    vals = [3.0, -8.0, 20.0, -50.0, 100.0]
    cov = threshold_coverage(vals, [5.0, 10.0, 50.0])
    assert cov["5.0"]["count"] == 4  # |3| excluded
    assert cov["10.0"]["count"] == 3
    assert cov["50.0"]["count"] == 1
    assert cov["5.0"]["pct"] == 80.0


def test_directional_accuracy():
    preds = [10.0, -10.0, 5.0, -5.0, 0.0]
    reals = [8.0, -3.0, -2.0, 4.0, 0.0]
    # correct: idx0 (++/), idx1 (--/) ; idx2,3 wrong; idx4 zero pred -> incorrect
    acc = directional_accuracy(preds, reals)
    assert acc["n"] == 5
    assert acc["correct"] == 2
    assert acc["accuracy"] == 0.4
    assert acc["long_n"] == 2
    assert acc["short_n"] == 2


def test_directional_empty():
    acc = directional_accuracy([], [])
    assert acc["accuracy"] == 0.0


def test_pearson_perfect():
    assert pearson_corr([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert pearson_corr([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert pearson_corr([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) == 0.0
    assert pearson_corr([1.0], [2.0]) == 0.0


def test_trade_stats_threshold_filters_cost():
    # cost 35 round-trip: pred 50 gross -> net 15 (clamped 50-35) > thr 10 => BUY
    # pred 20 gross -> net 0 (clamped, 20-35<0 => 0) => HOLD for thr>=0
    samples = [
        Sample(timestamp="t1", pred_bps=50.0, realized_bps=40.0),
        Sample(timestamp="t2", pred_bps=20.0, realized_bps=100.0),
        Sample(timestamp="t3", pred_bps=-50.0, realized_bps=-40.0),
    ]
    cost = Decimal("35")
    stats = trade_stats_for_threshold(samples, 10.0, cost)
    # t1 BUY: gross 40, net 5 ; t3 SELL: gross 40, net 5 ; t2 HOLD
    assert stats["trade_count"] == 2
    assert stats["gross_pnl_bps"] == 80.0
    assert stats["net_pnl_bps"] == 10.0
    assert stats["win_rate"] == 1.0
    assert stats["profit_factor"] == float("inf")


def test_trade_stats_losses():
    samples = [
        Sample(timestamp="t1", pred_bps=60.0, realized_bps=-100.0),  # BUY wrong
        Sample(timestamp="t2", pred_bps=-60.0, realized_bps=100.0),  # SELL wrong
    ]
    cost = Decimal("35")
    stats = trade_stats_for_threshold(samples, 10.0, cost)
    assert stats["trade_count"] == 2
    assert stats["win_rate"] == 0.0
    assert stats["net_pnl_bps"] == (-135.0 * 2)
    assert stats["profit_factor"] == 0.0


def test_threshold_sensitivity_monotonic_trades():
    samples = [
        Sample(timestamp=f"t{i}", pred_bps=float(v), realized_bps=10.0) for i, v in enumerate([5, 15, 25, 60])
    ]
    cost = Decimal("35")
    sens = threshold_sensitivity(samples, [5.0, 15.0, 50.0], cost)
    # higher threshold => fewer or equal trades
    assert sens["5.0"]["trade_count"] >= sens["15.0"]["trade_count"] >= sens["50.0"]["trade_count"]


def test_no_lookahead_contract_documented():
    # Samples must be built walk-forward; this test pins the Sample schema
    s = Sample(timestamp="2026-09-02T20:00:00+00:00", pred_bps=1.5, realized_bps=-2.5)
    assert isinstance(s.pred_bps, float)
    assert isinstance(s.realized_bps, float)
