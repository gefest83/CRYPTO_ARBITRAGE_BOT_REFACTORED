"""Profit-signal tests: determinism, no look-ahead, exit policy (mocked/offline)."""

from __future__ import annotations

from decimal import Decimal

from app.research.prediction_markets.up_down_5m.drift_signal import (
    DriftConfig,
    backtest_drift,
    compute_drift_bps,
    decide_drift,
    ensure_research_only,
    exit_net,
    hold_net,
    manage_with_exit,
)
from app.research.prediction_markets.up_down_5m.signal import Action, Position, Signal
from app.research.prediction_markets.up_down_5m.validation_dataset import (
    Kline,
    ValidatedMarket,
    build_signal_features,
)
from app.research.prediction_markets.up_down_5m.settlement import SettlementOutcome

T0 = 1_748_131_200_000
T1 = T0 + 300_000
CFG = DriftConfig()


def _mkline(ts: int, close: str) -> Kline:
    return Kline(open_ts_ms=ts, close_ts_ms=ts + 999, close=Decimal(close),
                 volume_base=Decimal("1"), taker_buy_base=Decimal("0.6"))


def _market(outcome: SettlementOutcome = SettlementOutcome.UP) -> ValidatedMarket:
    kl = tuple(_mkline(T0 - 120_000 + i * 1000, "100") for i in range(420))
    return ValidatedMarket(market_id=7, start_ts_ms=T0, end_ts_ms=T1,
                           klines=kl, contract_series=((T0, Decimal("0.5")),),
                           outcome=outcome)


def test_drift_math_bps() -> None:
    assert compute_drift_bps(Decimal("100"), Decimal("100")) == Decimal("0")
    assert compute_drift_bps(Decimal("101"), Decimal("100")) == Decimal("100")
    assert compute_drift_bps(None, Decimal("100")) is None
    assert compute_drift_bps(Decimal("100"), Decimal("0")) is None


def test_up_down_hold_threshold() -> None:
    up = decide_drift(decision_ts_ms=T0 + 90_000, market_start_ms=T0, market_end_ms=T1,
                      ref_price=Decimal("100"), mid_now=Decimal("100.03"), config=CFG)  # 3bps
    assert up.signal == Signal.UP
    dn = decide_drift(decision_ts_ms=T0 + 90_000, market_start_ms=T0, market_end_ms=T1,
                      ref_price=Decimal("100"), mid_now=Decimal("99.97"), config=CFG)
    assert dn.signal == Signal.DOWN
    weak = decide_drift(decision_ts_ms=T0 + 90_000, market_start_ms=T0, market_end_ms=T1,
                        ref_price=Decimal("100"), mid_now=Decimal("100.01"), config=CFG)  # 1bps
    assert weak.signal == Signal.HOLD


def test_time_gates_and_missing() -> None:
    assert decide_drift(decision_ts_ms=T0 + 10_000, market_start_ms=T0, market_end_ms=T1,
                        ref_price=Decimal("100"), mid_now=Decimal("110"), config=CFG).signal == Signal.HOLD
    assert decide_drift(decision_ts_ms=T1 - 60_000, market_start_ms=T0, market_end_ms=T1,
                        ref_price=Decimal("100"), mid_now=Decimal("110"), config=CFG).signal == Signal.HOLD
    assert decide_drift(decision_ts_ms=T0 + 90_000, market_start_ms=T0, market_end_ms=T1,
                        ref_price=None, mid_now=Decimal("110"), config=CFG).signal == Signal.HOLD


def test_deterministic() -> None:
    kw = {"decision_ts_ms": T0 + 90_000, "market_start_ms": T0, "market_end_ms": T1,
          "ref_price": Decimal("100"), "mid_now": Decimal("101"), "config": CFG}
    assert decide_drift(**kw) == decide_drift(**kw)


def test_manage_entry_hold_reversal_exit() -> None:
    assert manage_with_exit(Position.NONE, Signal.UP) == Action.ENTER_UP
    assert manage_with_exit(Position.NONE, Signal.DOWN) == Action.ENTER_DOWN
    assert manage_with_exit(Position.NONE, Signal.HOLD) == Action.HOLD_POSITION
    assert manage_with_exit(Position.LONG_UP, Signal.UP, Signal.DOWN) == Action.EXIT
    assert manage_with_exit(Position.LONG_DOWN, Signal.DOWN, Signal.UP) == Action.EXIT
    # winners never exit early
    assert manage_with_exit(Position.LONG_UP, Signal.UP, Signal.UP) == Action.HOLD_POSITION
    assert manage_with_exit(Position.LONG_UP, Signal.UP, Signal.HOLD) == Action.HOLD_POSITION
    assert manage_with_exit(Position.LONG_DOWN, Signal.DOWN, Signal.HOLD) == Action.HOLD_POSITION


def test_hold_net_beats_cost_with_fee() -> None:
    cfg = DriftConfig(fee_bps=200, slippage_per_share=Decimal("0.005"))
    win = hold_net("UP", Decimal("0.5"), "UP", cfg)  # 1-0.505-fee
    loss = hold_net("UP", Decimal("0.5"), "DOWN", cfg)
    assert win > Decimal("0") > loss
    assert win + (-loss) == Decimal("1")  # win/loss symmetric around payout gap


def test_exit_salvages_vs_hold_loss() -> None:
    cfg = DriftConfig()
    hold = hold_net("UP", Decimal("0.6"), "DOWN", cfg)  # ~-0.62
    esc = exit_net("UP", Decimal("0.6"), Decimal("0.4"), cfg)  # salvage 0.4
    assert esc > hold  # smaller loss
    assert esc < Decimal("0")  # still a loss, no fabrication


def test_no_lookahead_future_kline_ignored() -> None:
    m = _market()
    f0 = build_signal_features(m, T0 + 90_000)
    kl2 = tuple(m.klines) + (_mkline(T0 + 200_000, "200"),)
    m2 = m.model_copy(update={"klines": kl2})
    f1 = build_signal_features(m2, T0 + 90_000)
    assert f1.mid_now == f0.mid_now


def test_backtest_synthetic_trend_profits() -> None:
    # steady climb: drift at 90s must be UP and hold must win vs UP outcome
    kl = tuple(_mkline(T0 - 120_000 + i * 1000, str(100 + i * 0.01)) for i in range(420))
    m = ValidatedMarket(market_id=9, start_ts_ms=T0, end_ts_ms=T1, klines=kl,
                        contract_series=((T0, Decimal("0.5")),), outcome=SettlementOutcome.UP)
    # loose thr to force entry on synthetic micro-drift
    res = backtest_drift([m], DriftConfig(drift_thr_bps=Decimal("0.1")))
    assert res["trades"] == 1 and res["total_net"] > Decimal("0")


def test_no_chainlink_no_trading_imports() -> None:
    import ast
    import pathlib

    for p in (pathlib.Path("app/research/prediction_markets/up_down_5m/drift_signal.py"),
              pathlib.Path("scripts/run_up_down_5m_profit_validation.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        mods = []
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                mods.append(n.module or "")
            elif isinstance(n, ast.Import):
                mods.extend(a.name for a in n.names)
        assert not any("chainlink" in str(x).lower() for x in mods), p
        text = p.read_text(encoding="utf-8").lower()
        for kw in ("from app.execution", "from app.risk", "from app.recovery",
                   "from app.telegram", "from app.agent", "place_order(", "submit_order("):
            assert kw not in text, f"{p.name}: {kw}"
    try:
        ensure_research_only()
        raise AssertionError("must raise")
    except RuntimeError:
        pass
