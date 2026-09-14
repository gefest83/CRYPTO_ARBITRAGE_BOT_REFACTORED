"""Early-flow +20s tests: determinism, no look-ahead, enter/exit policy (mocked)."""

from __future__ import annotations

from decimal import Decimal

from app.research.prediction_markets.up_down_5m.early_flow_signal import (
    EarlyFlowConfig,
    backtest_early_flow,
    compute_drift_bps,
    compute_flow_imb,
    decide_early_flow,
    decide_exit_drift,
    ensure_research_only,
    exit_net,
    hold_net,
    manage_with_exit,
)
from app.research.prediction_markets.up_down_5m.signal import Action, Position, Signal
from app.research.prediction_markets.up_down_5m.settlement import SettlementOutcome
from app.research.prediction_markets.up_down_5m.validation_dataset import Kline, ValidatedMarket

T0 = 1_748_131_200_000
T1 = T0 + 300_000
CFG = EarlyFlowConfig()


def _mkline(ts: int, close: str, buy: str = "0.6") -> Kline:
    return Kline(open_ts_ms=ts, close_ts_ms=ts + 999, close=Decimal(close),
                 volume_base=Decimal("1"), taker_buy_base=Decimal(buy))


def _market(outcome: SettlementOutcome = SettlementOutcome.UP) -> ValidatedMarket:
    kl = tuple(_mkline(T0 - 120_000 + i * 1000, "100") for i in range(420))
    return ValidatedMarket(market_id=7, start_ts_ms=T0, end_ts_ms=T1,
                           klines=kl, contract_series=((T0, Decimal("0.5")),),
                           outcome=outcome)


def test_flow_math() -> None:
    assert compute_flow_imb(Decimal("0.6"), Decimal("1")) == Decimal("0.2")
    assert compute_flow_imb(Decimal("0"), Decimal("1")) == Decimal("-1")
    assert compute_flow_imb(Decimal("0.5"), Decimal("1")) == Decimal("0")
    assert compute_flow_imb(Decimal("0.6"), Decimal("0")) is None
    assert compute_drift_bps(Decimal("101"), Decimal("100")) == Decimal("100")
    assert compute_drift_bps(None, Decimal("100")) is None


def test_up_down_hold_threshold() -> None:
    assert decide_early_flow(flow_imb=Decimal("0.5"), config=CFG).signal == Signal.UP
    assert decide_early_flow(flow_imb=Decimal("-0.5"), config=CFG).signal == Signal.DOWN
    assert decide_early_flow(flow_imb=Decimal("0.01"), config=CFG).signal == Signal.HOLD
    assert decide_early_flow(flow_imb=None, config=CFG).signal == Signal.HOLD


def test_exit_drift_opposition_only() -> None:
    assert decide_exit_drift(side="UP", drift_bps=Decimal("-5"), config=CFG) is True
    assert decide_exit_drift(side="UP", drift_bps=Decimal("5"), config=CFG) is False
    assert decide_exit_drift(side="DOWN", drift_bps=Decimal("5"), config=CFG) is True
    assert decide_exit_drift(side="DOWN", drift_bps=Decimal("-5"), config=CFG) is False
    assert decide_exit_drift(side="UP", drift_bps=None, config=CFG) is False


def test_manage_entry_hold_reversal_exit() -> None:
    assert manage_with_exit(Position.NONE, Signal.UP) == Action.ENTER_UP
    assert manage_with_exit(Position.NONE, Signal.DOWN) == Action.ENTER_DOWN
    assert manage_with_exit(Position.NONE, Signal.HOLD) == Action.HOLD_POSITION
    assert manage_with_exit(Position.LONG_UP, Signal.UP, True) == Action.EXIT
    assert manage_with_exit(Position.LONG_DOWN, Signal.DOWN, True) == Action.EXIT
    assert manage_with_exit(Position.LONG_UP, Signal.UP, False) == Action.HOLD_POSITION
    assert manage_with_exit(Position.LONG_DOWN, Signal.DOWN, False) == Action.HOLD_POSITION


def test_hold_exit_economics() -> None:
    cfg = EarlyFlowConfig(fee_bps=200, slippage_per_share=Decimal("0.005"))
    win = hold_net("UP", Decimal("0.5"), "UP", cfg)
    loss = hold_net("UP", Decimal("0.5"), "DOWN", cfg)
    assert win > Decimal("0") > loss
    esc = exit_net("UP", Decimal("0.5"), Decimal("0.3"), cfg)
    assert esc > loss
    assert esc < Decimal("0")


def test_backtest_entry_timing_20s() -> None:
    # steady taker-buy pressure into +20s must enter UP
    kl = tuple(_mkline(T0 - 120_000 + i * 1000, "100", buy="0.8") for i in range(420))
    m = ValidatedMarket(market_id=9, start_ts_ms=T0, end_ts_ms=T1, klines=kl,
                        contract_series=((T0, Decimal("0.5")),), outcome=SettlementOutcome.UP)
    res = backtest_early_flow([m], CFG)
    assert res["trades"] == 1 and res["total_net"] > Decimal("0")
    assert CFG.entry_offset_ms == 20_000 and CFG.entry_offset_ms <= 60_000


def test_no_lookahead_future_kline_ignored() -> None:
    m = _market()
    r0 = backtest_early_flow([m], CFG)
    kl2 = tuple(m.klines) + (_mkline(T0 + 200_000, "200", buy="1"),)
    m2 = m.model_copy(update={"klines": kl2})
    r1 = backtest_early_flow([m2], CFG)
    assert r1["total_net"] == r0["total_net"]


def test_no_chainlink_no_trading_imports() -> None:
    import ast
    import pathlib

    for p in (pathlib.Path("app/research/prediction_markets/up_down_5m/early_flow_signal.py"),
              pathlib.Path("scripts/run_up_down_5m_early_flow_validation.py")):
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
    # contract price must only appear as execution cost/salvage, never as signal:
    # entry decision takes flow only; exit decision takes drift only
    import inspect as _inspect

    from app.research.prediction_markets.up_down_5m import early_flow_signal as _m

    assert "contract" not in _inspect.getsource(_m.decide_early_flow).lower()
    assert "contract" not in _inspect.getsource(_m.decide_exit_drift).lower()
    try:
        ensure_research_only()
        raise AssertionError("must raise")
    except RuntimeError:
        pass
