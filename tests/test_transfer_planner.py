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

    planner = TransferPlanner(Settings(_env_file=None))
    # ~469 bps net (buy 100, sell 105) – above 70 bps floor -> accepted
    assert planner.evaluate(_plan()) is not None
    assert _plan().net_profit_bps >= planner.min_net_profit_bps
    # Barely profitable plan below the floor -> rejected
    low = _plan(sell_price=D("100.5"))
    assert low.net_profit_bps < planner.min_net_profit_bps
    assert planner.evaluate(low) is None


def test_nonsense_inputs_never_produce_opportunities():
    from app.config.settings import Settings

    planner = TransferPlanner(Settings(_env_file=None))
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

    planner = TransferPlanner(Settings(_env_file=None))
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


def test_default_transfer_assets_is_top50():
    from app.config.settings import Settings, DEFAULT_TRANSFER_ASSETS

    settings = Settings(_env_file=None)
    assert len(settings.transfer.assets) == 50
    assert len(DEFAULT_TRANSFER_ASSETS) == 50
    # Must include core liquid assets and be supported on all venues
    for core in ("BTC", "ETH", "SOL", "BNB", "XRP", "TRX"):
        assert core in settings.transfer.assets
    # Triangular unchanged
    assert settings.arbitrage.triangle_assets == ("BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "LINK", "AVAX", "TRX")


def test_transfer_min_net_profit_is_70bps():
    from app.config.settings import Settings

    settings = Settings(_env_file=None)
    assert settings.transfer.min_net_profit_bps == D("70")


def test_notional_sizing_100_500():
    from app.config.settings import Settings

    settings = Settings(_env_file=None)
    assert settings.transfer.min_notional_quote == D("100")
    assert settings.transfer.max_notional_quote == D("500")
    planner = TransferPlanner(settings)
    # $100 notional at 100 price => 1.0 coin; $500 => 5.0 coin
    assert planner.executable_amount_from_notional(
        min_notional=D("100"),
        max_notional=D("500"),
        available_quote=D("10000"),
        buy_price=D("100"),
        withdrawal_min=D("0.1"),
    ) == D("5")
    # available < min => 0
    assert planner.executable_amount_from_notional(
        min_notional=D("100"),
        max_notional=D("500"),
        available_quote=D("50"),
        buy_price=D("100"),
        withdrawal_min=D("0.1"),
    ) == D("0")
    # withdrawal_min violation
    assert planner.executable_amount_from_notional(
        min_notional=D("100"),
        max_notional=D("500"),
        available_quote=D("10000"),
        buy_price=D("100"),
        withdrawal_min=D("10"),
    ) == D("0")
    # max_amount cap truncates
    assert planner.executable_amount_from_notional(
        min_notional=D("100"),
        max_notional=D("500"),
        available_quote=D("10000"),
        buy_price=D("100"),
        withdrawal_min=D("0.1"),
        max_amount=D("2"),
    ) == D("2")
    # amount*price < min after max_amount truncation => 0
    assert planner.executable_amount_from_notional(
        min_notional=D("100"),
        max_notional=D("500"),
        available_quote=D("10000"),
        buy_price=D("100"),
        withdrawal_min=D("0.1"),
        max_amount=D("0.5"),
    ) == D("0")
    # cheap coin (SHIB) with tiny price: 500 / 0.00002 = 25M
    assert planner.executable_amount_from_notional(
        min_notional=D("100"),
        max_notional=D("500"),
        available_quote=D("10000"),
        buy_price=D("0.00002"),
        withdrawal_min=D("1000000"),
    ) == D("25000000.00000000")


def test_notional_sizing_uses_buy_price():
    from app.config.settings import Settings

    planner = TransferPlanner(Settings(_env_file=None))
    # $500 at 2500 price => 0.2 coin
    amt = planner.executable_amount_from_notional(
        min_notional=D("100"),
        max_notional=D("500"),
        available_quote=D("10000"),
        buy_price=D("2500"),
        withdrawal_min=D("0.001"),
    )
    assert amt == D("0.20000000")
    # $500 at 0.3 price => 1666.66 coin
    amt2 = planner.executable_amount_from_notional(
        min_notional=D("100"),
        max_notional=D("500"),
        available_quote=D("10000"),
        buy_price=D("0.3"),
        withdrawal_min=D("0.1"),
    )
    assert amt2 == (D("500") / D("0.3")).quantize(D("0.00000001"))
