"""Transfer planner math: fees, withdrawal costs, net profit, sizing."""

from decimal import Decimal

from app.models.transfer import TransferPlan
from app.strategies.transfer.planner import TransferPlanner

D = Decimal


def _plan(**overrides) -> TransferPlan:
    params = {
        "source_exchange": "binance",
        "dest_exchange": "okx",
        "asset": "SOL",
        "network": "SOL",
        "amount": D("10"),
        "buy_price": D("100"),
        "sell_price": D("105"),
        "buy_fee_bps": D("10"),
        "sell_fee_bps": D("10"),
        "withdrawal_fee": D("0.01"),
    }
    params.update(overrides)
    return TransferPlan(**params)


def test_gross_and_net_profit_math():
    plan = _plan()
    assert plan.buy_cost_quote == D("1000")
    assert plan.sell_proceeds_quote == D("1050")
    # trading fees: 1000 * 10bps + 1050 * 10bps = 1 + 1.05
    assert plan.trading_fees_quote == D("2.05")
    # withdrawal fee valued at sell price: 0.01 * 105
    assert plan.withdrawal_cost_quote == D("1.05")
    assert plan.gross_profit_quote == D("50")
    assert plan.net_profit_quote == D("50") - D("2.05") - D("1.05")
    assert plan.net_profit_bps == (plan.net_profit_quote / D("1000") * D("10000"))


def test_min_profit_rejection():
    planner = TransferPlanner.__new__(TransferPlanner)
    planner._settings = None  # type: ignore[assignment]
    # Directly exercise evaluate() logic via a real settings-backed planner
    from app.config.settings import Settings

    planner = TransferPlanner(Settings())
    # ~46.9 bps net: above the default 30 bps floor
    assert planner.evaluate(_plan()) is not None
    # Barely profitable plan below the floor -> rejected
    low = _plan(sell_price=D("100.5"))
    assert low.net_profit_bps < planner.min_net_profit_bps
    assert planner.evaluate(low) is None


def test_nonsense_inputs_never_produce_opportunities():
    from app.config.settings import Settings

    planner = TransferPlanner(Settings())
    assert planner.evaluate(_plan(amount=D("0"))) is None
    assert planner.evaluate(_plan(buy_price=D("0"))) is None
    assert planner.evaluate(_plan(sell_price=D("0"))) is None
    assert planner.evaluate(_plan(source_exchange="binance", dest_exchange="binance")) is None
    assert planner.evaluate(_plan(sell_price=D("99"))) is None  # negative spread


def test_network_cost_included():
    plan = _plan(network_cost_quote=D("2"))
    assert plan.withdrawal_cost_quote == D("1.05") + D("2")
    assert plan.net_profit_quote == D("50") - D("2.05") - D("3.05")


def test_spread_bps():
    plan = _plan()
    assert plan.spread_bps == D("500")  # 5% spread


def test_executable_amount_honours_constraints():
    from app.config.settings import Settings

    planner = TransferPlanner(Settings())
    # withdrawal minimum dominates
    assert planner.executable_amount(
        requested_amount=D("0.5"),
        withdrawal_min=D("1"),
        available_quote=D("10000"),
        buy_price=D("100"),
    ) == D("0")
    # affordability dominates
    assert planner.executable_amount(
        requested_amount=D("10"),
        withdrawal_min=D("0.1"),
        available_quote=D("250"),
        buy_price=D("100"),
    ) == D("2.5")
    # notional cap dominates
    assert planner.executable_amount(
        requested_amount=D("10"),
        withdrawal_min=D("0.1"),
        available_quote=D("100000"),
        buy_price=D("100"),
        max_notional=D("250"),
    ) == D("2.5")
    # amount cap dominates
    assert planner.executable_amount(
        requested_amount=D("10"),
        withdrawal_min=D("0.1"),
        available_quote=D("100000"),
        buy_price=D("100"),
        max_amount=D("3"),
    ) == D("3")
