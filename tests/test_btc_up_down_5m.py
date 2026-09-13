"""BTC Up/Down 5m foundation tests — settlement rule + state transitions + replay.

Research-only; offline; no network; no live trading.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.research.prediction_markets.up_down_5m.market import (
    MarketState,
    UpDown5mMarket,
)
from app.research.prediction_markets.up_down_5m.replay import (
    ReplayDecision,
    UpDown5mReplay,
)
from app.research.prediction_markets.up_down_5m.settlement import (
    SettlementOutcome,
    payout_per_share,
    settle_up_down_5m,
)

T0 = 1_748_131_200_000  # window start
T1 = T0 + 300_000  # window end (5m)


def _market(**kw) -> UpDown5mMarket:
    base = {"market_id": 9001, "start_ts_ms": T0, "end_ts_ms": T1}
    base.update(kw)
    return UpDown5mMarket(**base)


# ---------- settlement rule ----------

def test_settle_up_wins_on_higher_end() -> None:
    assert settle_up_down_5m(Decimal("68000"), Decimal("68001")) == SettlementOutcome.UP
    assert settle_up_down_5m("68000.00", "68000.01") == SettlementOutcome.UP


def test_settle_down_wins_on_lower_end() -> None:
    assert settle_up_down_5m(Decimal("68000"), Decimal("67999")) == SettlementOutcome.DOWN


def test_settle_equal_is_push_50_50() -> None:
    assert settle_up_down_5m(Decimal("68000"), Decimal("68000")) == SettlementOutcome.PUSH
    assert payout_per_share("UP", SettlementOutcome.PUSH) == Decimal("0.5")
    assert payout_per_share("DOWN", SettlementOutcome.PUSH) == Decimal("0.5")


def test_settle_payout_winner_loser() -> None:
    assert payout_per_share("UP", SettlementOutcome.UP) == Decimal("1")
    assert payout_per_share("DOWN", SettlementOutcome.UP) == Decimal("0")
    assert payout_per_share("DOWN", SettlementOutcome.DOWN) == Decimal("1")
    assert payout_per_share("UP", SettlementOutcome.DOWN) == Decimal("0")


def test_settle_rejects_non_positive_chainlink() -> None:
    with pytest.raises(ValueError):
        settle_up_down_5m(Decimal("0"), Decimal("68000"))
    with pytest.raises(ValueError):
        settle_up_down_5m(Decimal("68000"), Decimal("-1"))


def test_settle_exact_decimal_compare_no_float() -> None:
    # 1 sat above start must still be UP (exact Decimal, no epsilon).
    assert settle_up_down_5m(Decimal("68000.00000000"), Decimal("68000.00000001")) == SettlementOutcome.UP


# ---------- market state transitions ----------

def test_state_upcoming_open_closed_settled() -> None:
    m = _market(start_price=Decimal("68000"), end_price=None)
    assert m.state_at(T0 - 1) == MarketState.UPCOMING
    assert m.state_at(T0) == MarketState.OPEN
    assert m.state_at(T1 - 1) == MarketState.OPEN
    assert m.can_trade_at(T0 + 1000) is True
    # after end without end_price -> CLOSED, no trading
    assert m.state_at(T1) == MarketState.CLOSED
    assert m.can_trade_at(T1) is False
    assert m.can_trade_at(T0 - 1) is False


def test_state_settled_when_anchors_known() -> None:
    m = _market(start_price=Decimal("68000"), end_price=Decimal("68100"))
    assert m.state_at(T1) == MarketState.SETTLED
    # time-driven: still OPEN before end even if anchors are pre-filled
    assert m.state_at(T0 + 1000) == MarketState.OPEN
    settled = m.settle()
    assert settled.settlement == SettlementOutcome.UP
    assert settled.state_at(T1) == MarketState.SETTLED


def test_settle_requires_both_chainlink_anchors() -> None:
    with pytest.raises(ValueError):
        _market(start_price=Decimal("68000"), end_price=None).settle()
    with pytest.raises(ValueError):
        _market(start_price=None, end_price=Decimal("68000")).settle()


def test_market_rejects_non_5m_window() -> None:
    with pytest.raises(Exception):
        _market(end_ts_ms=T0 + 900_000)  # 15m is out of scope for this 5m model


def test_market_rejects_contract_price_outside_01() -> None:
    with pytest.raises(Exception):
        _market(up_bid=Decimal("1.5"), up_ask=Decimal("1.6"))
    with pytest.raises(Exception):
        _market(down_bid=Decimal("-0.1"))


def test_position_and_quote_helpers() -> None:
    m = _market(
        start_price=Decimal("68000"),
        up_bid=Decimal("0.51"), up_ask=Decimal("0.53"),
        down_bid=Decimal("0.47"), down_ask=Decimal("0.49"),
    )
    assert m.up_mid == Decimal("0.52")
    assert m.down_mid == Decimal("0.48")
    m2 = m.with_position("UP", Decimal("2"))
    assert m2.position_up == Decimal("2") and m2.position_down == Decimal("0")
    with pytest.raises(ValueError):
        m.with_position("UP", Decimal("-1"))


# ---------- replay vs binary settlement ----------

def _replay_market(mid: int, start: str, end: str) -> UpDown5mMarket:
    return UpDown5mMarket(
        market_id=mid, start_ts_ms=T0, end_ts_ms=T1,
        start_price=Decimal(start), end_price=Decimal(end),
        current_price=Decimal(end),
    )


def test_replay_up_win_pays_one_minus_cost_fee() -> None:
    engine = UpDown5mReplay(fee_bps=200)
    markets = [_replay_market(1, "68000", "68100")]  # UP
    decisions = [ReplayDecision(market_id=1, side="UP", entry_price=Decimal("0.52"), size=Decimal("1"), entry_ts_ms=T0 + 1000)]
    summary = engine.run(markets, decisions)
    assert summary.traded == 1 and summary.wins == 1
    f = summary.fills[0]
    assert f.outcome == SettlementOutcome.UP
    assert f.payout_per_share == Decimal("1")
    assert f.payout == Decimal("1")
    # cost 0.52 + fee 0.52*0.02=0.0104 -> net 0.4696
    assert f.net == Decimal("1") - Decimal("0.52") - Decimal("0.0104")


def test_replay_loser_goes_to_zero() -> None:
    engine = UpDown5mReplay(fee_bps=200)
    markets = [_replay_market(1, "68000", "67900")]  # DOWN
    decisions = [ReplayDecision(market_id=1, side="UP", entry_price=Decimal("0.52"), size=Decimal("1"), entry_ts_ms=T0 + 1000)]
    summary = engine.run(markets, decisions)
    assert summary.wins == 0
    assert summary.fills[0].payout == Decimal("0")
    assert summary.fills[0].win is False


def test_replay_push_pays_half() -> None:
    engine = UpDown5mReplay(fee_bps=0)
    markets = [_replay_market(1, "68000", "68000")]  # PUSH
    decisions = [ReplayDecision(market_id=1, side="UP", entry_price=Decimal("0.40"), size=Decimal("2"), entry_ts_ms=T0 + 1000)]
    summary = engine.run(markets, decisions)
    f = summary.fills[0]
    assert f.outcome == SettlementOutcome.PUSH
    assert f.payout == Decimal("0.5") * Decimal("2")
    assert f.net == Decimal("1.0") - Decimal("0.80")


def test_replay_rejects_entry_outside_open_window() -> None:
    engine = UpDown5mReplay()
    markets = [_replay_market(1, "68000", "68100")]
    # entry at/after end -> look-ahead / late, must raise
    with pytest.raises(ValueError):
        engine.run(markets, [ReplayDecision(market_id=1, side="UP", entry_price=Decimal("0.5"), size=Decimal("1"), entry_ts_ms=T1)])
    with pytest.raises(ValueError):
        engine.run(markets, [ReplayDecision(market_id=1, side="UP", entry_price=Decimal("0.5"), size=Decimal("1"), entry_ts_ms=T0 - 1)])


def test_replay_skip_scores_zero_and_excluded_from_win_rate() -> None:
    engine = UpDown5mReplay()
    markets = [_replay_market(1, "68000", "68100"), _replay_market(2, "68000", "67900")]
    decisions = [
        ReplayDecision(market_id=1, side="SKIP", entry_price=Decimal("0.5"), size=Decimal("1"), entry_ts_ms=T0 + 1),
        ReplayDecision(market_id=2, side="DOWN", entry_price=Decimal("0.5"), size=Decimal("1"), entry_ts_ms=T0 + 1),
    ]
    summary = engine.run(markets, decisions)
    assert summary.skipped == 1 and summary.traded == 1
    assert summary.win_rate == 1.0


def test_replay_is_research_only_fail_closed() -> None:
    engine = UpDown5mReplay()
    with pytest.raises(RuntimeError):
        engine.ensure_research_only()


def test_replay_no_live_trading_imports() -> None:
    import pathlib

    root = pathlib.Path("app/research/prediction_markets/up_down_5m")
    for p in root.glob("*.py"):
        text = p.read_text(encoding="utf-8").lower()
        for kw in ("from app.execution", "from app.risk", "from app.recovery", "from app.telegram", "from app.agent", "place_order(", "submit_order(", "batch_redeem("):
            assert kw not in text, f"{p.name} must stay research-only, found {kw!r}"


def test_spot_proxy_backtest_is_not_used_by_replay() -> None:
    """Guard: settlement replay must not depend on the deprecated spot-proxy path."""
    import pathlib

    replay_src = pathlib.Path("app/research/prediction_markets/up_down_5m/replay.py").read_text(encoding="utf-8")
    assert "detect_impulses" not in replay_src
    assert "BacktestAnalyzer" not in replay_src
    assert "settle_up_down_5m" in replay_src
