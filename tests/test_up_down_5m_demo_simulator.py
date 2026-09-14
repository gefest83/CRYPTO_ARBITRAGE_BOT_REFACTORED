"""Simulator tests: fills, fees, slippage, settlement, drawdown, ROI (offline)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.research.prediction_markets.up_down_5m.demo import DemoMode
from app.research.prediction_markets.up_down_5m.demo_simulator import (
    DemoFill,
    DemoSimConfig,
    ensure_research_only,
    simulate_trade,
    summarize,
)

CFG = DemoSimConfig(stake_shares=Decimal("10"), fee_bps=200, slippage_per_share=Decimal("0.005"))


def _fill(**kw) -> DemoFill:  # type: ignore[no-untyped-def]
    base = {"market_id": 1, "up_bid_entry": Decimal("0.49"), "up_ask_entry": Decimal("0.51"),
            "up_bid_exit": Decimal("0.40"), "up_ask_exit": Decimal("0.42")}
    base.update(kw)
    return DemoFill(**base)


def test_up_win_pays_one_minus_costs() -> None:
    t = simulate_trade(market_id=1, side="UP", fill=_fill(), settlement="UP", config=CFG)
    assert t.result == "WIN"
    # cost=0.51*10=5.10 slip=0.05 fee=5.15*0.02=0.103 staked=5.253 payout=10
    assert t.staked == Decimal("5.10") + Decimal("0.05") + Decimal("5.15") * Decimal("200") / Decimal("10000")
    assert t.net == Decimal("10") - t.staked
    assert t.gross == Decimal("10") - Decimal("5.10")


def test_down_entry_uses_implied_ask() -> None:
    t = simulate_trade(market_id=1, side="DOWN", fill=_fill(), settlement="DOWN", config=CFG)
    assert t.result == "WIN"
    assert t.entry_price == Decimal("1") - Decimal("0.49")  # 1 - UP bid


def test_loss_and_push() -> None:
    loss = simulate_trade(market_id=1, side="UP", fill=_fill(), settlement="DOWN", config=CFG)
    assert loss.result == "LOSS" and loss.net < Decimal("0")
    push = simulate_trade(market_id=1, side="UP", fill=_fill(), settlement="PUSH", config=CFG)
    assert push.result == "PUSH" and push.gross == Decimal("5") - Decimal("5.10")


def test_exit_salvage_beats_hold_loss() -> None:
    hold = simulate_trade(market_id=1, side="UP", fill=_fill(), settlement="DOWN", config=CFG)
    ext = simulate_trade(market_id=1, side="UP", fill=_fill(), settlement="DOWN", exited=True, config=CFG)
    assert ext.result == "EXIT"
    assert ext.exit_price == Decimal("0.40")
    assert ext.net > hold.net  # salvaged >0 beats holding to zero
    assert ext.net < Decimal("0")


def test_skip_and_pending() -> None:
    s = simulate_trade(market_id=1, side="SKIP", fill=None, settlement=None, config=CFG)
    assert s.result == "SKIP" and s.net == Decimal("0")
    p = simulate_trade(market_id=2, side="UP", fill=_fill(), settlement=None, config=CFG)
    assert p.result == "PENDING"


def test_summary_drawdown_roi() -> None:
    w = simulate_trade(market_id=1, side="UP", fill=_fill(), settlement="UP", config=CFG)
    l = simulate_trade(market_id=2, side="UP", fill=_fill(), settlement="DOWN", config=CFG)
    s = summarize([w, l])
    assert (s.trades, s.wins, s.losses) == (2, 1, 1)
    assert s.total_net == w.net + l.net
    assert s.avg_net == s.total_net / Decimal("2")
    assert s.roi == s.total_net / s.total_staked
    # equity rises then falls below peak: dd = peak - trough
    assert s.max_drawdown == -l.net
    s2 = summarize([w, w])
    assert s2.max_drawdown == Decimal("0")


def test_live_disabled_and_determinism() -> None:
    cfg = DemoSimConfig(mode=DemoMode.LIVE)
    with pytest.raises(RuntimeError):
        simulate_trade(market_id=1, side="UP", fill=_fill(), settlement="UP", config=cfg)
    with pytest.raises(RuntimeError):
        ensure_research_only()
    a = simulate_trade(market_id=1, side="UP", fill=_fill(), settlement="UP", config=CFG)
    b = simulate_trade(market_id=1, side="UP", fill=_fill(), settlement="UP", config=CFG)
    assert a == b


def test_no_forbidden_imports() -> None:
    import ast
    import pathlib

    for p in (
        pathlib.Path("app/research/prediction_markets/up_down_5m/demo_simulator.py"),
        pathlib.Path("scripts/run_up_down_5m_demo_simulator.py"),
    ):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        mods = []
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                mods.append(n.module or "")
            elif isinstance(n, ast.Import):
                mods.extend(a.name for a in n.names)
        assert not any("chainlink_feed" in str(m).lower() for m in mods), p
        text = p.read_text(encoding="utf-8").lower()
        for kw in ("from app.execution", "from app.risk", "from app.recovery",
                   "from app.telegram", "from app.agent", "place_order(",
                   "submit_order(", "batch_redeem("):
            assert kw not in text, f"{p.name}: {kw}"
