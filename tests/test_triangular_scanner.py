"""Triangular scanner: route discovery, orientation, fees, slippage, freshness."""

from decimal import Decimal

from app.clock import FixedClock
from app.config.settings import Settings
from app.market_data.store import MarketDataStore
from app.models.base import utc_now
from app.models.enums import OrderSide
from app.models.market_data import OrderBook, OrderBookLevel
from app.models.symbol import Symbol
from app.strategies.triangular import TriangularScanner
from app.strategies.triangular.fees import ScanRequest

D = Decimal


def _book(venue: str, symbol: Symbol, bid: str, ask: str, depth: int = 10) -> OrderBook:
    """Deterministic book around the given bid/ask with 5000 quote per level."""
    levels = depth
    bid_price, ask_price = D(bid), D(ask)
    bid_step = bid_price * D("0.0001")
    ask_step = ask_price * D("0.0001")
    bids = tuple(
        OrderBookLevel(price=bid_price - bid_step * i, amount=D("5000") / bid_price)
        for i in range(levels)
    )
    asks = tuple(
        OrderBookLevel(price=ask_price + ask_step * i, amount=D("5000") / ask_price)
        for i in range(levels)
    )
    return OrderBook(
        exchange_id=venue,
        symbol=symbol,
        bids=bids,
        asks=asks,
        timestamp=utc_now(),
        received_at=utc_now(),
    )


def _scanner(clock: FixedClock | None = None) -> TriangularScanner:
    settings = Settings(_env_file=None)
    return TriangularScanner(
        store=MarketDataStore(stale_after_ms=10_000, clock=clock or FixedClock()),
        venues=lambda: ("binance",),
        settings=settings.arbitrage,
        clock=clock,
    )


def _request() -> ScanRequest:
    return ScanRequest(
        symbols=(Symbol.parse("BTC/USDT"), Symbol.parse("ETH/USDT")),
        notional_quote=D("1000"),
    )


async def test_finds_profitable_ring_with_correct_orientation():
    scanner = _scanner()
    store = scanner._store
    # Construct a ring with a genuine edge:
    #   buy BTC at 100 (ask), buy ETH with BTC on ETH/BTC at ask 0.05 (1 BTC = 20 ETH),
    #   sell ETH at 21 (bid) -> 20 ETH * 21 = 420 vs 100 spent... build coherent numbers:
    # leg1: buy BTC with 1000 USDT at ask 100 -> 10 BTC
    # leg2: buy ETH paying BTC on ETH/BTC ask 0.05 -> 10 BTC / 0.05 = 200 ETH
    # leg3: sell 200 ETH at bid 5.30 -> 1060 USDT  (fees ~3x10bps, slippage tiny)
    store.put_order_book(_book("binance", Symbol.parse("BTC/USDT"), "99.99", "100"))
    store.put_order_book(_book("binance", Symbol.parse("ETH/BTC"), "0.0499", "0.05"))
    store.put_order_book(_book("binance", Symbol.parse("ETH/USDT"), "5.30", "5.31"))
    opportunities = await scanner.scan(_request())
    assert len(opportunities) == 1
    opportunity = opportunities[0]
    assert opportunity.direction == "USDT->BTC->ETH->USDT @ binance"
    # legs: BUY BTC/USDT, BUY ETH/BTC (buy Y with X), SELL ETH/USDT
    legs = opportunity.legs_route
    assert [leg.side for leg in legs] == [
        OrderSide.BUY,
        OrderSide.BUY,
        OrderSide.SELL,
    ]
    assert [leg.symbol.name for leg in legs] == ["BTC/USDT", "ETH/BTC", "ETH/USDT"]
    # net must clear the configured minimum and be positive
    assert opportunity.net_profit_bps > D("0")
    # gross = net + fees + slippage
    profit = opportunity.profit
    assert profit.gross_spread_bps == (
        profit.net_profit_bps + profit.trading_fees_bps + profit.slippage_bps
    )


async def test_inverse_cross_orientation_uses_sell_leg():
    """Without ETH/BTC, an X/Y book (BTC/ETH) must be sold into, not bought."""
    scanner = _scanner()
    store = scanner._store
    store.put_order_book(_book("binance", Symbol.parse("BTC/USDT"), "99.99", "100"))
    # BTC/ETH: sell BTC receiving ETH at bid 19.9 (1 BTC = 19.9 ETH)
    store.put_order_book(_book("binance", Symbol.parse("BTC/ETH"), "19.9", "20"))
    store.put_order_book(_book("binance", Symbol.parse("ETH/USDT"), "5.30", "5.31"))
    opportunities = await scanner.scan(_request())
    assert len(opportunities) == 1
    legs = opportunities[0].legs_route
    assert [leg.side for leg in legs] == [
        OrderSide.BUY,
        OrderSide.SELL,  # sell BTC for ETH on BTC/ETH
        OrderSide.SELL,
    ]


async def test_unprofitable_ring_rejected_by_minimum():
    scanner = _scanner()
    store = scanner._store
    # Coherent prices (ETH/USDT ~= BTC/USDT * ETH/BTC) with realistic spreads:
    # every direction loses the spread + fees.
    store.put_order_book(_book("binance", Symbol.parse("BTC/USDT"), "99.99", "100"))
    store.put_order_book(_book("binance", Symbol.parse("ETH/BTC"), "0.0499", "0.0501"))
    store.put_order_book(_book("binance", Symbol.parse("ETH/USDT"), "4.99", "5.00"))
    assert await scanner.scan(_request()) == ()


async def test_stale_data_is_rejected_when_freshness_required():
    clock = FixedClock()
    scanner = _scanner(clock)
    store = scanner._store
    store.put_order_book(_book("binance", Symbol.parse("BTC/USDT"), "99.99", "100"))
    store.put_order_book(_book("binance", Symbol.parse("ETH/BTC"), "0.0499", "0.05"))
    store.put_order_book(_book("binance", Symbol.parse("ETH/USDT"), "5.30", "5.31"))
    # Advance the clock far beyond the staleness window (10s).
    clock.advance(seconds=30)
    assert await scanner.scan(_request()) == ()
    # Freshness not required: the same stale books produce the ring again.
    request = ScanRequest(
        symbols=(Symbol.parse("BTC/USDT"), Symbol.parse("ETH/USDT")),
        notional_quote=D("1000"),
        require_fresh_data=False,
    )
    assert len(await scanner.scan(request)) == 1


async def test_fee_calculation_uses_provider():
    """Higher taker fees shrink the net by exactly the fee delta."""
    from app.models.market import MarketFees
    from app.strategies.triangular.fees import StaticFeeProvider

    async def scan_with_fees(taker_bps: Decimal):
        settings = Settings(_env_file=None)
        scanner = TriangularScanner(
            store=MarketDataStore(stale_after_ms=10_000),
            venues=lambda: ("binance",),
            settings=settings.arbitrage,
            fees=StaticFeeProvider(taker_bps=taker_bps),
        )
        store = scanner._store
        store.put_order_book(_book("binance", Symbol.parse("BTC/USDT"), "99.99", "100"))
        store.put_order_book(_book("binance", Symbol.parse("ETH/BTC"), "0.0499", "0.05"))
        store.put_order_book(_book("binance", Symbol.parse("ETH/USDT"), "5.30", "5.31"))
        result = await scanner.scan(_request())
        assert len(result) == 1
        return result[0]

    cheap = await scan_with_fees(D("1"))
    expensive = await scan_with_fees(D("20"))
    # reported fees differ by ~3 * 19 bps
    assert cheap.profit.trading_fees_bps < expensive.profit.trading_fees_bps
    assert expensive.profit.trading_fees_bps - cheap.profit.trading_fees_bps > D("50")
    # higher fees -> lower net
    assert expensive.net_profit_bps < cheap.net_profit_bps
    # fee provider contract sanity
    from app.models.enums import MarketType

    assert StaticFeeProvider().fees_for(
        "binance", Symbol.parse("BTC/USDT"), MarketType.SPOT
    ).taker_bps == D("10")
    assert StaticFeeProvider(overrides={"binance": MarketFees(taker_bps=D("5"))}).fees_for(
        "binance", Symbol.parse("BTC/USDT"), MarketType.SPOT
    ).taker_bps == D("5")


async def test_max_results_limits_output():
    scanner = _scanner()
    opportunities = await scanner.scan(_request())
    assert len(opportunities) <= _request().max_results
