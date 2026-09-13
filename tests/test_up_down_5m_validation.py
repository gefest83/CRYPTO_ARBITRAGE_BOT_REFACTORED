"""Validation-dataset tests: real-window schema, venue outcomes, no look-ahead.

All offline (synthetic klines/payloads, mocked klines transport). Signal
logic itself is untouched and only evaluated, never modified here.
"""

from __future__ import annotations

from decimal import Decimal

import httpx
import pytest

from app.research.prediction_markets.up_down_5m.settlement import SettlementOutcome
from app.research.prediction_markets.up_down_5m.validation_dataset import (
    Kline,
    ValidatedMarket,
    build_signal_features,
    contract_price_at,
    ensure_research_only,
    fetch_klines,
    parse_contract_series,
    parse_venue_outcome,
    parse_window_from_slug,
    run_validation,
)

T0 = 1_789_319_100_000
T1 = T0 + 300_000


def _kline(open_ms: int, close: str, vol: str = "1", tb: str = "0.6") -> Kline:
    return Kline(open_ts_ms=open_ms, close_ts_ms=open_ms + 999,
                 close=Decimal(close), volume_base=Decimal(vol), taker_buy_base=Decimal(tb))


def _market(**kw):  # type: ignore[no-untyped-def]
    kl = [_kline(T0 - 120_000 + i * 1000, "100") for i in range(420)]
    base = {"market_id": 2232903, "start_ts_ms": T0, "end_ts_ms": T1,
            "klines": tuple(kl), "outcome": SettlementOutcome.UP}
    base.update(kw)
    return ValidatedMarket(**base)


def test_slug_window_exact() -> None:
    assert parse_window_from_slug("btc-updown-5m-1789319100") == (1_789_319_100_000, 1_789_319_400_000)
    with pytest.raises(ValueError):
        parse_window_from_slug("btc-updown-5m-soon")


def test_venue_outcome_mapping() -> None:
    up = {"data": {"outcomes": [{"name": "Up", "status": "WON"}, {"name": "Down", "status": "LOST"}],
                   "variantData": {"startPrice": 1, "endPrice": 2}}}
    assert parse_venue_outcome(up)[0] == SettlementOutcome.UP
    down = {"data": {"outcomes": [{"name": "Up", "status": "LOST"}, {"name": "Down", "status": "WON"}],
                    "variantData": {"startPrice": 2, "endPrice": 1}}}
    assert parse_venue_outcome(down)[0] == SettlementOutcome.DOWN
    tied = {"data": {"outcomes": [{"name": "Up", "status": "TIED"}, {"name": "Down", "status": "TIED"}],
                    "variantData": {"startPrice": 5, "endPrice": 5}}}
    assert parse_venue_outcome(tied)[0] == SettlementOutcome.PUSH
    amb = {"data": {"outcomes": [{"name": "Up", "status": "WON"}, {"name": "Down", "status": "WON"}], "variantData": {}}}
    with pytest.raises(ValueError):
        parse_venue_outcome(amb)


def test_contract_series_pre_decision_only() -> None:
    series = parse_contract_series({"data": {"series": [
        {"x": T0 // 1000 - 60, "y": 50}, {"x": T0 // 1000 + 120, "y": 67}]}})
    assert series[0] == (T0 - 60_000, Decimal("0.5"))
    assert contract_price_at(series, T0 + 90_000) == Decimal("0.5")  # future point ignored
    assert contract_price_at(series, T0 - 61_000) is None


def test_kline_close_time_rule_no_lookahead() -> None:
    m = _market()
    f_early = build_signal_features(m, T0 + 90_000)
    # all closes are 100 (flat) except inject a future spike beyond decision
    kl = list(m.klines) + [_kline(T0 + 200_000, "200")]
    m2 = m.model_copy(update={"klines": tuple(kl)})
    f_same = build_signal_features(m2, T0 + 90_000)
    assert f_same.mid_now == f_early.mid_now == Decimal("100")


def test_flow_from_taker_buy() -> None:
    m = _market()
    f = build_signal_features(m, T0 + 90_000)
    assert f.flow_imb == pytest.approx(Decimal("0.2"))  # 0.6 buy of 1.0 vol


def test_run_validation_scores_against_venue_outcome() -> None:
    from app.research.prediction_markets.up_down_5m.signal import SignalConfig

    cfg = SignalConfig(drift_entry_bps=Decimal("1"), momentum_bps=Decimal("1"),
                       flow_threshold=Decimal("0.05"), book_threshold=Decimal("5"))
    kl = [_kline(T0 - 120_000 + i * 1000, str(100 + i * 0.01)) for i in range(420)]  # steady climb
    m = _market(klines=tuple(kl), outcome=SettlementOutcome.UP)
    results, summary = run_validation([m], cfg, decision_offset_ms=90_000)
    assert summary["markets"] == 1 and summary["scored"] == 1
    assert summary["accuracy"] == 1.0 and summary["coverage"] == 1.0
    assert results[0].correct is True


def test_push_and_hold_excluded_from_accuracy() -> None:
    m = _market(outcome=SettlementOutcome.PUSH)
    results, summary = run_validation([m], None, decision_offset_ms=90_000)
    assert summary["scored"] == 0 and summary["accuracy"] is None
    assert results[0].correct is None


@pytest.mark.asyncio
async def test_fetch_klines_mocked() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        assert "api.binance.com" in str(request.url)
        assert request.url.params["interval"] == "1s"
        return httpx.Response(200, json=[
            [T0, "100", "101", "99", "100.5", "2.0", T0 + 999, "0", 1, "1.2", "0", "0"],
        ])

    kl = await fetch_klines("BTCUSDT", T0 - 1000, T0 + 1000, transport=httpx.MockTransport(_handler))
    assert len(kl) == 1 and kl[0].close == Decimal("100.5") and kl[0].taker_buy_base == Decimal("1.2")


def test_no_chainlink_no_trading_in_adapter() -> None:
    import ast
    import pathlib

    for p in (pathlib.Path("app/research/prediction_markets/up_down_5m/validation_dataset.py"),
              pathlib.Path("scripts/run_up_down_5m_validation.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        mods = []
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                mods.append(n.module or "")
            elif isinstance(n, ast.Import):
                mods.extend(a.name for a in n.names)
        assert not any("chainlink" in str(m).lower() for m in mods), p
        text = p.read_text(encoding="utf-8").lower()
        for kw in ("from app.execution", "from app.risk", "place_order(", "submit_order("):
            assert kw not in text, f"{p.name}: {kw}"
    with pytest.raises(RuntimeError):
        ensure_research_only()
