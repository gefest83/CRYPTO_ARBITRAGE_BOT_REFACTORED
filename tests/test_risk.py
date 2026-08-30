"""Risk engine: every protection, including the kill switch and fail-closed rules."""

from decimal import Decimal

from app.models.enums import ArbitrageStrategy
from app.models.risk import RiskLimits
from app.risk import RiskEngine
from app.risk.rules import RiskContext

D = Decimal


def _context(**overrides) -> RiskContext:
    params = {
        "strategy": ArbitrageStrategy.TRIANGLE,
        "notional_quote": D("500"),
        "net_profit_bps": D("50"),
        "slippage_bps": D("5"),
        "data_age_ms": 100.0,
        "daily_pnl": D("0"),
        "open_transfers": 0,
    }
    params.update(overrides)
    return RiskContext(**params)


def _engine(**limits) -> RiskEngine:
    return RiskEngine(RiskLimits(**limits) if limits else RiskLimits())


def test_clean_context_approves():
    assessment = _engine().evaluate(_context())
    assert assessment.approved
    assert assessment.violations == ()


def test_kill_switch_blocks_everything():
    engine = _engine()
    assessment = engine.evaluate(_context(kill_switch_engaged=True))
    assert not assessment.approved
    assert any(v.rule == "kill_switch" for v in assessment.violations)


def test_min_net_profit_rejects_marginal_trade():
    engine = _engine(min_net_profit_bps=D("30"))
    assessment = engine.evaluate(_context(net_profit_bps=D("25")))
    assert not assessment.approved
    assert any(v.rule == "min_net_profit" for v in assessment.violations)


def test_max_trade_size_rejects_oversized_trade():
    engine = _engine(max_trade_size=D("1000"))
    assessment = engine.evaluate(_context(notional_quote=D("1500")))
    assert not assessment.approved
    violation = next(v for v in assessment.violations if v.rule == "max_trade_size")
    assert violation.limit == D("1000")
    assert violation.actual == D("1500")


def test_max_daily_loss_blocks_after_breach():
    engine = _engine(max_daily_loss=D("250"))
    ok = engine.evaluate(_context(daily_pnl=D("-100")))
    assert ok.approved
    breached = engine.evaluate(_context(daily_pnl=D("-250")))
    assert not breached.approved
    assert any(v.rule == "max_daily_loss" for v in breached.violations)


def test_max_open_transfers_applies_to_transfer_strategy_only():
    engine = _engine(max_open_transfers=2)
    # triangle strategy is not gated by the transfer cap
    assert engine.evaluate(_context(open_transfers=5)).approved
    assessment = engine.evaluate(_context(strategy=ArbitrageStrategy.TRANSFER, open_transfers=2))
    assert not assessment.approved
    assert any(v.rule == "max_open_transfers" for v in assessment.violations)


def test_max_slippage_rejects():
    engine = _engine(max_slippage_bps=D("15"))
    assessment = engine.evaluate(_context(slippage_bps=D("20")))
    assert not assessment.approved
    assert any(v.rule == "max_slippage" for v in assessment.violations)


def test_stale_data_rejects():
    engine = _engine(max_data_age_ms=2500)
    assessment = engine.evaluate(_context(data_age_ms=4000.0))
    assert not assessment.approved
    assert any(v.rule == "max_data_age" for v in assessment.violations)


def test_exchange_and_asset_exposure_limits():
    engine = _engine(max_exchange_exposure=D("1000"), max_asset_exposure=D("2000"))
    context = _context(
        exchange_exposure={"binance": D("1500")},
        asset_exposure={"SOL": D("2500")},
    )
    assessment = engine.evaluate(context)
    assert not assessment.approved
    rules = {v.rule for v in assessment.violations}
    assert "max_exchange_exposure" in rules
    assert "max_asset_exposure" in rules


def test_broken_rule_fails_closed():
    class BrokenRule:
        name = "broken"

        def check(self, context, limits):
            raise RuntimeError("boom")

    engine = RiskEngine(RiskLimits(), rules=[BrokenRule()])
    assessment = engine.evaluate(_context())
    assert not assessment.approved
    assert assessment.violations[0].severity.value == "critical"


def test_negative_net_profit_always_rejected_by_default():
    engine = _engine()  # default min 10 bps
    assert not engine.evaluate(_context(net_profit_bps=D("-5"))).approved
