"""DEMO tests: frozen rules, spot-only signal, LIVE disabled, logging (offline)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.research.prediction_markets.up_down_5m.demo import (
    DemoConfig,
    DemoMode,
    decide_demo_entry,
    decide_demo_exit,
    ensure_demo_only,
    ensure_research_only,
    render_demo_log,
    run_demo_market,
)
from app.research.prediction_markets.up_down_5m.live_snapshot import PriceProvenance
from app.research.prediction_markets.up_down_5m.signal import Action, Position, Signal

T0 = 1_748_131_200_000
T1 = T0 + 300_000


def test_frozen_entry_rules() -> None:
    sig, drift, act = decide_demo_entry(
        market_id=1, market_start_ms=T0, market_end_ms=T1,
        spot_start=Decimal("68000"), spot_now=Decimal("68025"),  # +3.7bps
    )
    assert sig == Signal.UP and act == Action.ENTER_UP
    assert drift is not None and drift >= Decimal("2.0")
    sig, _, act = decide_demo_entry(
        market_id=1, market_start_ms=T0, market_end_ms=T1,
        spot_start=Decimal("68000"), spot_now=Decimal("67975"),  # -3.7bps
    )
    assert sig == Signal.DOWN and act == Action.ENTER_DOWN
    sig, _, act = decide_demo_entry(
        market_id=1, market_start_ms=T0, market_end_ms=T1,
        spot_start=Decimal("68000"), spot_now=Decimal("68005"),  # +0.7bps weak
    )
    assert sig == Signal.HOLD and act == Action.HOLD_POSITION


def test_exit_reversal_vs_hold_winner() -> None:
    # reversal: LONG_UP + DOWN exit => EXIT
    sig, _, act = decide_demo_exit(
        market_id=1, market_start_ms=T0, market_end_ms=T1, position=Position.LONG_UP,
        spot_start=Decimal("68000"), spot_now=Decimal("67970"),
    )
    assert sig == Signal.DOWN and act == Action.EXIT
    # winner holds: LONG_UP + UP exit => HOLD
    sig, _, act = decide_demo_exit(
        market_id=1, market_start_ms=T0, market_end_ms=T1, position=Position.LONG_UP,
        spot_start=Decimal("68000"), spot_now=Decimal("68030"),
    )
    assert act == Action.HOLD_POSITION


def test_full_lifecycle_win_skip_exit() -> None:
    win = run_demo_market(
        market_id=11, market_start_ms=T0, market_end_ms=T1,
        spot_start=Decimal("68000"), spot_entry=Decimal("68025"),
        spot_exit=Decimal("68030"), settlement="UP",
    )
    assert win.result == "WIN" and win.entry_action == Action.ENTER_UP
    skip = run_demo_market(
        market_id=12, market_start_ms=T0, market_end_ms=T1,
        spot_start=Decimal("68000"), spot_entry=Decimal("68005"),
        spot_exit=Decimal("68005"), settlement="UP",
    )
    assert skip.result == "SKIP"
    ext = run_demo_market(
        market_id=13, market_start_ms=T0, market_end_ms=T1,
        spot_start=Decimal("68000"), spot_entry=Decimal("68025"),
        spot_exit=Decimal("67970"), settlement="DOWN",
    )
    assert ext.result == "EXIT" and ext.exit_action == Action.EXIT


def test_live_disabled_fail_closed() -> None:
    with pytest.raises(RuntimeError):
        ensure_demo_only(DemoMode.LIVE)
    with pytest.raises(RuntimeError):
        ensure_demo_only("LIVE")
    with pytest.raises(RuntimeError):
        ensure_research_only()
    with pytest.raises(RuntimeError):
        run_demo_market(
            market_id=1, market_start_ms=T0, market_end_ms=T1,
            spot_start=Decimal("68000"), spot_entry=Decimal("68025"),
            spot_exit=Decimal("68030"), settlement="UP",
            config=DemoConfig(mode=DemoMode.LIVE),
        )


def test_chainlink_never_a_signal() -> None:
    with pytest.raises(ValueError, match="never Chainlink"):
        decide_demo_entry(
            market_id=1, market_start_ms=T0, market_end_ms=T1,
            spot_start=Decimal("68000"), spot_now=Decimal("68025"),
            provenance=PriceProvenance.CHAINLINK_VENUE,
        )
    with pytest.raises(ValueError, match="never Chainlink"):
        decide_demo_exit(
            market_id=1, market_start_ms=T0, market_end_ms=T1, position=Position.LONG_UP,
            spot_start=Decimal("68000"), spot_now=Decimal("67970"),
            provenance=PriceProvenance.CHAINLINK,
        )


def test_demo_log_shows_required_fields() -> None:
    d = run_demo_market(
        market_id=9001, market_start_ms=T0, market_end_ms=T1,
        spot_start=Decimal("68000"), spot_entry=Decimal("68025"),
        spot_exit=Decimal("68030"), settlement="UP",
    )
    text = render_demo_log(d).lower()
    for kw in ("signal=", "drift_entry_bps=", "entry=", "exit=", "settlement=", "result="):
        assert kw in text
    assert "live_trading=disabled" in text
    assert "win" in text


def test_deterministic() -> None:
    kw = {"market_id": 5, "market_start_ms": T0, "market_end_ms": T1,
          "spot_start": Decimal("68000"), "spot_entry": Decimal("68025"),
          "spot_exit": Decimal("68030"), "settlement": "UP"}
    assert run_demo_market(**kw) == run_demo_market(**kw)


def test_no_forbidden_imports_and_no_live_trading() -> None:
    import ast
    import pathlib

    for p in (
        pathlib.Path("app/research/prediction_markets/up_down_5m/demo.py"),
        pathlib.Path("scripts/run_up_down_5m_demo.py"),
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
            assert kw not in text, f"{p.name} must stay demo-only, found {kw!r}"
