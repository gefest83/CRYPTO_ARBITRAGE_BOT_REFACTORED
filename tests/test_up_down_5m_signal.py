"""Entry-signal tests: determinism, no look-ahead, enter/exit policy (mocked).

Offline, no I/O, no Chainlink, no trading.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.research.prediction_markets.up_down_5m.signal import (
    Action,
    Position,
    Signal,
    SignalConfig,
    compute_features,
    ensure_research_only,
    evaluate,
    exit_to_skip,
    manage,
)

T0 = 1_748_131_200_000
T1 = T0 + 300_000
CFG = SignalConfig()  # researched defaults


def _feats(up: bool = True, **kw):  # type: ignore[no-untyped-def]
    s = 1 if up else -1
    base = {
        "decision_ts_ms": T0 + 90_000,
        "market_start_ms": T0,
        "market_end_ms": T1,
        "ref_price": Decimal("68000"),
        "mid_now": Decimal("68000") + Decimal("10") * s,  # ~14.7bps drift
        "mid_momentum_ago": Decimal("68000") + Decimal("4") * s,
        "mid_accel_ago": Decimal("68000"),
        "flow_imb": Decimal("0.3") * s,
        "book_imb": Decimal("0.2") * s,
    }
    base.update(kw)
    return compute_features(**base)  # type: ignore[arg-type]


def test_up_and_down_majority_vote() -> None:
    assert evaluate(_feats(up=True), CFG).signal == Signal.UP
    assert evaluate(_feats(up=False), CFG).signal == Signal.DOWN


def test_hold_on_weak_or_missing() -> None:
    weak = _feats(mid_now=Decimal("68001"), mid_momentum_ago=Decimal("68001"),  # ~1.5bps
                  mid_accel_ago=Decimal("68001"), flow_imb=Decimal("0.01"), book_imb=Decimal("0.01"))
    assert evaluate(weak, CFG).signal == Signal.HOLD
    f = _feats()
    f = f.model_copy(update={"flow_imb": None, "book_imb": None, "mid_momentum_ago": None})
    assert evaluate(f, CFG).signal == Signal.HOLD  # drift alone (+14.7bps = 1 vote) insufficient
    assert evaluate(_feats(ref_price=None), CFG).signal == Signal.HOLD


def test_time_gates() -> None:
    assert evaluate(_feats(decision_ts_ms=T0 + 10_000), CFG).signal == Signal.HOLD  # warming
    assert evaluate(_feats(decision_ts_ms=T1 - 60_000), CFG).signal == Signal.HOLD  # too-late


def test_deterministic_repeated_calls() -> None:
    f = _feats()
    assert evaluate(f, CFG) == evaluate(f, CFG)


def test_custom_thresholds_change_vote() -> None:
    strict = SignalConfig(drift_entry_bps=Decimal("50"), momentum_bps=Decimal("50"),
                          flow_threshold=Decimal("0.9"), book_threshold=Decimal("0.9"))
    assert evaluate(_feats(), strict).signal == Signal.HOLD


def test_manage_immediate_entry() -> None:
    assert manage(Position.NONE, Signal.UP) == Action.ENTER_UP
    assert manage(Position.NONE, Signal.DOWN) == Action.ENTER_DOWN
    assert manage(Position.NONE, Signal.HOLD) == Action.HOLD_POSITION


def test_manage_reversal_exits_hold_otherwise() -> None:
    assert manage(Position.LONG_UP, Signal.DOWN) == Action.EXIT
    assert manage(Position.LONG_DOWN, Signal.UP) == Action.EXIT
    assert manage(Position.LONG_UP, Signal.UP) == Action.HOLD_POSITION
    assert manage(Position.LONG_UP, Signal.HOLD) == Action.HOLD_POSITION
    assert manage(Position.LONG_DOWN, Signal.DOWN) == Action.HOLD_POSITION
    assert manage(Position.LONG_DOWN, Signal.HOLD) == Action.HOLD_POSITION


def test_exit_maps_to_replay_skip_inside_window() -> None:
    d = exit_to_skip(10918165, T0 + 100_000)
    assert d.side == "SKIP" and d.market_id == 10918165


def test_entry_replay_roundtrip_hold_to_settlement() -> None:
    from app.research.prediction_markets.up_down_5m.market import UpDown5mMarket
    from app.research.prediction_markets.up_down_5m.replay import ReplayDecision, UpDown5mReplay

    res = evaluate(_feats(), CFG)
    act = manage(Position.NONE, res.signal)
    assert act == Action.ENTER_UP  # BUY UP now
    market = UpDown5mMarket(
        market_id=10918165, start_ts_ms=T0, end_ts_ms=T1,
        start_price=Decimal("68000"), end_price=Decimal("68100"),
        current_price=Decimal("68100"),
        up_bid=Decimal("0.51"), up_ask=Decimal("0.53"),
        down_bid=Decimal("0.47"), down_ask=Decimal("0.49"),
    )
    dec = ReplayDecision(market_id=10918165, side="UP", entry_price=Decimal("0.53"),
                         size=Decimal("1"), entry_ts_ms=T0 + 90_000)
    summary = UpDown5mReplay().run([market], [dec])
    assert summary.traded == 1 and summary.fills[0].win is True


def test_no_chainlink_in_signal_source() -> None:
    import pathlib

    src = pathlib.Path("app/research/prediction_markets/up_down_5m/signal.py").read_text(encoding="utf-8").lower()
    assert "chainlink" in src  # ban documented...
    assert "from app.research.prediction_markets.up_down_5m.chainlink_feed" not in src
    assert "import chainlink" not in src
    # ...but only in comments/docstrings, never as data flow
    import ast

    tree = ast.parse(pathlib.Path("app/research/prediction_markets/up_down_5m/signal.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").lower())
    assert not any("chainlink" in m for m in imported)


def test_research_only_isolation() -> None:
    import pathlib

    for p in (
        pathlib.Path("app/research/prediction_markets/up_down_5m/signal.py"),
        pathlib.Path("scripts/run_up_down_5m_threshold_research.py"),
    ):
        text = p.read_text(encoding="utf-8").lower()
        for kw in ("from app.execution", "from app.risk", "from app.recovery", "from app.telegram",
                   "from app.agent", "place_order(", "submit_order(", "batch_redeem("):
            assert kw not in text, f"{p.name} must stay research-only, found {kw!r}"
    with pytest.raises(RuntimeError):
        ensure_research_only()
