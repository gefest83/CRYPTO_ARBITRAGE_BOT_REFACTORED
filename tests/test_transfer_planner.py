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
    # ~469 bps net (buy 100, sell 105) – above 50 bps floor -> accepted
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


def test_transfer_min_net_profit_is_50bps():
    from app.config.settings import Settings

    settings = Settings(_env_file=None)
    assert settings.transfer.min_net_profit_bps == D("50")


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


# ---------------------------------------------------------------------------
# Market-data sanity guard (Layer A + B) – pure, no symbol hard-coding
# ---------------------------------------------------------------------------

def _book(*, venue: str, bids, asks):
    from app.models.market_data import OrderBook, OrderBookLevel
    from app.models.symbol import Symbol

    return OrderBook(
        exchange_id=venue,
        symbol=Symbol(base="TEST", quote="USDT"),
        bids=tuple(OrderBookLevel(price=D(str(p)), amount=D(str(a))) for p, a in bids),
        asks=tuple(OrderBookLevel(price=D(str(p)), amount=D(str(a))) for p, a in asks),
    )


def test_transfer_guard_missing_buy_ask():
    from app.strategies.transfer.planner import validate_transfer_books

    buy = _book(venue="binance", bids=[(100, 1)], asks=[])
    sell = _book(venue="okx", bids=[(105, 1)], asks=[(106, 1)])
    assert validate_transfer_books(buy, sell, max_gross_divergence_bps=D("5000")) == "invalid_book"


def test_transfer_guard_missing_sell_bid():
    from app.strategies.transfer.planner import validate_transfer_books

    buy = _book(venue="binance", bids=[(99, 1)], asks=[(100, 1)])
    sell = _book(venue="okx", bids=[], asks=[(106, 1)])
    assert validate_transfer_books(buy, sell, max_gross_divergence_bps=D("5000")) == "invalid_book"


def test_transfer_guard_empty_asks():
    from app.strategies.transfer.planner import validate_transfer_books

    buy = _book(venue="binance", bids=[(99, 1)], asks=[])
    sell = _book(venue="okx", bids=[(105, 1)], asks=[(106, 1)])
    assert validate_transfer_books(buy, sell, max_gross_divergence_bps=D("5000")) == "invalid_book"


def test_transfer_guard_crossed_book():
    from app.strategies.transfer.planner import validate_transfer_books

    # buy book crossed: bid 101 >= ask 100
    buy = _book(venue="binance", bids=[(101, 1)], asks=[(100, 1)])
    sell = _book(venue="okx", bids=[(105, 1)], asks=[(106, 1)])
    assert validate_transfer_books(buy, sell, max_gross_divergence_bps=D("5000")) == "invalid_book"
    # sell book crossed
    buy2 = _book(venue="binance", bids=[(99, 1)], asks=[(100, 1)])
    sell2 = _book(venue="okx", bids=[(107, 1)], asks=[(106, 1)])
    assert validate_transfer_books(buy2, sell2, max_gross_divergence_bps=D("5000")) == "invalid_book"


def test_transfer_guard_intra_spread_over_1000_bps():
    from app.strategies.transfer.planner import validate_transfer_books

    # buy spread (100-120)/110*10000 = 1818 bps >1000
    buy = _book(venue="binance", bids=[(100, 1)], asks=[(120, 1)])
    sell = _book(venue="okx", bids=[(105, 1)], asks=[(106, 1)])
    assert validate_transfer_books(buy, sell, max_gross_divergence_bps=D("5000")) == "invalid_book"


def test_transfer_guard_strk_like_119M_rejected():
    from app.strategies.transfer.planner import validate_transfer_books

    buy = _book(venue="binance", bids=[(0.025, 1)], asks=[(0.026, 1)])
    sell = _book(venue="okx", bids=[(312, 1)], asks=[(315, 1)])
    # gross ≈ (312-0.026)/0.026*10000 ≈ 119M bps
    assert validate_transfer_books(buy, sell, max_gross_divergence_bps=D("5000")) == "gross_divergence"


def test_transfer_guard_tia_like_145k_rejected():
    from app.strategies.transfer.planner import validate_transfer_books

    buy = _book(venue="binance", bids=[(0.35, 1)], asks=[(0.353, 1)])
    sell = _book(venue="okx", bids=[(5.5, 1)], asks=[(5.6, 1)])
    # gross ≈ (5.5-0.353)/0.353*10000 ≈ 145k bps
    assert validate_transfer_books(buy, sell, max_gross_divergence_bps=D("5000")) == "gross_divergence"


def test_transfer_guard_4000_bps_accepted():
    from app.strategies.transfer.planner import validate_transfer_books

    # gross 4000 <5000 → pass
    buy = _book(venue="binance", bids=[(99, 1)], asks=[(100, 1)])
    sell = _book(venue="okx", bids=[(140, 1)], asks=[(141, 1)])
    assert validate_transfer_books(buy, sell, max_gross_divergence_bps=D("5000")) is None


def test_transfer_guard_config_override_2000_bps():
    from app.strategies.transfer.planner import validate_transfer_books

    buy = _book(venue="binance", bids=[(99, 1)], asks=[(100, 1)])
    # gross 2500 bps
    sell = _book(venue="okx", bids=[(125, 1)], asks=[(126, 1)])
    assert validate_transfer_books(buy, sell, max_gross_divergence_bps=D("2000")) == "gross_divergence"
    assert validate_transfer_books(buy, sell, max_gross_divergence_bps=D("3000")) is None


def test_transfer_guard_normal_route_reaches_profitability_check():
    from app.strategies.transfer.planner import validate_transfer_books

    # normal small spread: buy 100, sell 100.5 gross 50 bps <5000, intra spread 1% <10% → pass
    buy = _book(venue="binance", bids=[(99, 1)], asks=[(100, 1)])
    sell = _book(venue="okx", bids=[(100.5, 1)], asks=[(101, 1)])
    assert validate_transfer_books(buy, sell, max_gross_divergence_bps=D("5000")) is None


def test_transfer_guard_eth_like_128bps_not_rejected():
    from app.strategies.transfer.planner import validate_transfer_books

    # ETH ~128 bps divergence: buy 2461, sell 2430? Use buy 2461 sell 2492 ≈128 bps
    buy = _book(venue="binance", bids=[(2460, 1)], asks=[(2461, 1)])
    sell = _book(venue="okx", bids=[(2492, 1)], asks=[(2493, 1)])
    # gross (2492-2461)/2461*10000 ≈126 bps <5000
    assert validate_transfer_books(buy, sell, max_gross_divergence_bps=D("5000")) is None
    # also verify intra-spread is tiny (<1000)
    assert validate_transfer_books(buy, sell, max_gross_divergence_bps=D("5000")) is None
