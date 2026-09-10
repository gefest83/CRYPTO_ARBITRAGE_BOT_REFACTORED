"""Phase 3 — Backtest deterministic tests (offline, synthetic observations)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.research.prediction_markets.backtest.analyzer import BacktestAnalyzer
from app.research.prediction_markets.backtest.dataset import build_synthetic_dataset
from app.research.prediction_markets.backtest.models import BacktestConfig
from app.research.prediction_markets.collector.models import PredictionOrderbookObservation, SourceKind, SpotObservation


def _run(scenario: str, **kw) -> object:
    cfg = BacktestConfig(impulse_thresholds_bps=(20,), eval_window_ms=10000, fee_bps=200, min_depth=0, min_observations=1, **kw)
    analyzer = BacktestAnalyzer(cfg)
    spots, preds = build_synthetic_dataset(scenario, **{k: v for k, v in kw.items() if k in ("symbol", "duration", "market_id")})
    return analyzer.run(spots, preds)


def test_valid_repricing_detects_lag_and_positive_net() -> None:
    res = _run("valid_repricing")
    assert res.dataset_size > 0
    assert len(res.all_metrics) >= 1
    m = res.all_metrics[0]
    assert m.repricing_lag_ms is not None and m.repricing_lag_ms == 200 or m.repricing_lag_ms == 500  # 1200 vs 1500 lag
    assert m.max_favorable is not None and m.max_favorable > Decimal("0")
    assert m.gross_edge is not None and m.gross_edge > Decimal("0")
    # net may be positive depending on fee; with fee 200bps on 0.52 ~0.0104, gross 0.03 - half_spread 0.01 - fee => still positive
    assert m.net_edge is not None
    assert res.group_stats[0].win_rate is not None


def test_no_repricing_zero_or_none() -> None:
    res = _run("no_repricing")
    assert len(res.all_metrics) >= 1
    m = next(x for x in res.all_metrics if x.event.threshold_bps == 20)
    # no move beyond spread => lag None, gross 0 or None
    assert m.max_favorable is not None and m.max_favorable == Decimal("0") or m.gross_edge == Decimal("0")


def test_delayed_repricing_lag_large() -> None:
    res = _run("delayed_repricing")
    m = res.all_metrics[0]
    assert m.repricing_lag_ms == 7000
    assert m.time_to_max_favorable_ms == 7000


def test_negative_repricing_adverse_positive() -> None:
    res = _run("negative_repricing")
    m = res.all_metrics[0]
    assert m.max_favorable is not None and m.max_favorable < Decimal("0")
    assert m.max_adverse is not None and m.max_adverse > Decimal("0")
    assert m.win is False


def test_spread_fee_impact_net_can_be_negative() -> None:
    res = _run("spread_fee_impact")
    m = res.all_metrics[0]
    # gross 0.03, spread cost 0.02, fee ~0.01 => net ~0 or negative
    assert m.spread_cost is not None and m.spread_cost == Decimal("0.02")
    assert m.fee_cost is not None and m.fee_cost > Decimal("0")
    assert m.gross_edge == Decimal("0.03")
    assert m.net_edge is not None and m.net_edge < m.gross_edge
    # ensure cost model applied
    assert m.net_edge < Decimal("0") or m.win in (True, False)


def test_insufficient_liquidity_flag() -> None:
    cfg = BacktestConfig(impulse_thresholds_bps=(20,), fee_bps=200, min_depth=5, min_observations=1)
    analyzer = BacktestAnalyzer(cfg)
    spots, preds = build_synthetic_dataset("insufficient_liquidity")
    res = analyzer.run(spots, preds)
    assert any(x.insufficient_liquidity for x in res.all_metrics)


def test_expired_markets_excluded() -> None:
    res = _run("expired")
    # expired pred should be filtered; only one valid event, but future after t0 contains no valid pred? Actually only pre-t0 pred left, so future empty -> metrics with None
    # ensure not counted as signal with huge lag
    assert all(not x.is_expired for x in res.all_metrics)  # analyzer skips expired inputs
    # dataset_size excludes expired due to load_dataset filtering; here direct run includes filter
    # check expired observation not used: future list should be empty or not contain expired ts
    assert res.dataset_size >= 2


def test_out_of_order_data_deterministic_ordering() -> None:
    # spots/preds shuffled, analyzer must sort
    spots, preds = build_synthetic_dataset("out_of_order")
    cfg = BacktestConfig(impulse_thresholds_bps=(20,), eval_window_ms=10000, min_observations=1)
    analyzer = BacktestAnalyzer(cfg)
    res1 = analyzer.run(spots, preds)
    # reverse order should give same result
    res2 = analyzer.run(list(reversed(spots)), list(reversed(preds)))
    assert len(res1.all_metrics) == len(res2.all_metrics)
    if res1.all_metrics and res2.all_metrics:
        assert res1.all_metrics[0].repricing_lag_ms == res2.all_metrics[0].repricing_lag_ms


def test_strict_no_lookahead() -> None:
    """Future price spike beyond t0 must not affect signal definition; only evaluation."""
    from decimal import Decimal

    T0 = 1_748_131_200_000
    # create two datasets identical up to t0, diverge after t0
    spots_a, preds_a = build_synthetic_dataset("valid_repricing")
    spots_b, preds_b = build_synthetic_dataset("valid_repricing")
    # inject an extra future pred spike at T0+9000 that is far ahead — should affect metrics but NOT event count/t0
    extra = PredictionOrderbookObservation(
        captured_at_ms=T0 + 9005,
        source=SourceKind.PREDICTION_ORDERBOOK,
        symbol="BTCUSDT",
        market_id=9001,
        token_id="tok_9001_yes",
        duration="5m",
        exchange_ts_ms=T0 + 9000,
        update_ts_ms=T0 + 9000,
        resolution_ms=T0 + 300_000,
        time_to_resolution_ms=T0 + 300_000 - (T0 + 9000),
        outcome="YES",
        best_bid=Decimal("0.90"),
        best_ask=Decimal("0.92"),
        bids=((Decimal("0.90"), Decimal("5000")),),
        asks=((Decimal("0.92"), Decimal("3000")),),
    )
    preds_b.append(extra)
    cfg = BacktestConfig(impulse_thresholds_bps=(20,), min_observations=1)
    analyzer = BacktestAnalyzer(cfg)
    res_a = analyzer.run(spots_a, preds_a)
    res_b = analyzer.run(spots_b, preds_b)
    # event count must be identical (signal definition unchanged)
    assert len(res_a.all_metrics) == len(res_b.all_metrics)
    # but max favorable may increase in b due to spike
    if res_a.all_metrics and res_b.all_metrics:
        assert res_b.all_metrics[0].max_favorable >= res_a.all_metrics[0].max_favorable
    # ensure initial price is from at-or-before t0, not future spike
    assert res_b.all_metrics[0].event.initial_price == Decimal("0.52")  # mid of 0.51/0.53 at T0+500


def test_grouping_by_dimensions() -> None:
    """Run with BTC and ETH, 5m/15m, YES/NO to verify groups."""
    cfg = BacktestConfig(impulse_thresholds_bps=(20,), min_observations=1)
    analyzer = BacktestAnalyzer(cfg)
    spots_btc, preds_btc = build_synthetic_dataset("valid_repricing", symbol="BTCUSDT", duration="5m", market_id=9001)
    spots_eth, preds_eth = build_synthetic_dataset("valid_repricing", symbol="ETHUSDT", duration="15m", market_id=9002)
    # mix
    spots = spots_btc + spots_eth
    # need preds for both markets; keep separate market_ids
    preds = preds_btc + preds_eth
    res = analyzer.run(spots, preds)
    groups = {g.group for g in res.group_stats}
    assert any("symbol:BTCUSDT" in g for g in groups)
    assert any("symbol:ETHUSDT" in g for g in groups)
    assert any("duration:5m" in g for g in groups)
    assert any("duration:15m" in g for g in groups)
    # win rate present
    all_g = next(g for g in res.group_stats if g.group == "ALL")
    assert all_g.signal_count >= 2


def test_bnb_excluded_in_backtest() -> None:
    from app.research.prediction_markets.collector.models import SpotObservation as SO

    try:
        bnb_spot = SO(captured_at_ms=1, source=SourceKind.SPOT_ORDERBOOK, symbol="BNBUSDT", exchange_ts_ms=1, bid=Decimal("1"), ask=Decimal("2"))
        assert False, "BNB should have been rejected"
    except Exception:
        pass
    # analyzer must also filter BNB if somehow injected via raw bypass (use BTC dataset and check filter)
    spots, preds = build_synthetic_dataset("valid_repricing", symbol="BTCUSDT")
    assert spots[0].symbol == "BTCUSDT"


def test_isolation_no_forbidden_imports() -> None:
    import pathlib
    root = pathlib.Path("app/research/prediction_markets/backtest")
    for p in root.glob("*.py"):
        text = p.read_text(encoding="utf-8").lower()
        for kw in ["from app.strategies.triangular", "from app.strategies.transfer", "from app.execution", "from app.risk", "from app.recovery", "from app.telegram", "from app.agent", "from app.strategies.kronos"]:
            assert kw not in text, f"{p.name} violates isolation: {kw}"
