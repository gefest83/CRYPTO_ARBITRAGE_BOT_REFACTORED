"""Fill simulator: bid/ask direction, slippage cap, partial fills, fee model."""

from decimal import Decimal

from app.execution.fill_simulator import FillSimulator
from app.models.base import utc_now
from app.models.enums import OrderSide
from app.models.market_data import OrderBook, OrderBookLevel
from app.models.symbol import Symbol

D = Decimal


def _book(levels: list[tuple[str, str]], side: str) -> OrderBook:
    """Book with the given (price, amount) levels on one side only."""
    parsed = [OrderBookLevel(price=D(p), amount=D(a)) for p, a in levels]
    empty = ()
    return OrderBook(
        exchange_id="binance",
        symbol=Symbol.parse("ETH/USDT"),
        bids=parsed if side == "bids" else empty,
        asks=parsed if side == "asks" else empty,
        timestamp=utc_now(),
        received_at=utc_now(),
    )


def test_buy_walks_asks_ascending():
    sim = FillSimulator(taker_fee_bps=D("10"), max_slippage_bps=D("1000"))
    book = _book([("100", "1"), ("110", "1"), ("120", "1")], "asks")
    fill = sim.simulate(book, OrderSide.BUY, base_amount=D("2"))
    # 1 @ 100 + 1 @ 110
    assert fill.filled_amount + fill.fee == D("2")  # gross base filled before fee
    assert fill.average_price == D("105")
    assert fill.quote_amount == D("210")


def test_sell_walks_bids_descending():
    sim = FillSimulator(taker_fee_bps=D("10"), max_slippage_bps=D("1000"))
    book = _book([("120", "1"), ("110", "1"), ("100", "1")], "bids")
    fill = sim.simulate(book, OrderSide.SELL, base_amount=D("2"))
    assert fill.filled_amount == D("2")
    assert fill.average_price == D("115")  # 120 then 110
    gross = D("230")
    expected_net = gross - gross * D("10") / D("10000")
    assert fill.quote_amount == expected_net


def test_slippage_cap_stops_the_walk():
    sim = FillSimulator(max_slippage_bps=D("50"))  # 0.5% cap
    book = _book([("100", "1"), ("101", "1")], "asks")
    # level 2 at 101 is 100 bps above the reference 100 -> capped out
    fill = sim.simulate(book, OrderSide.BUY, base_amount=D("2"))
    assert fill.fill_ratio < D("1")
    assert fill.filled_amount + fill.fee == D("1")  # only level 1 filled


def test_cap_beyond_book_is_partial_fill():
    sim = FillSimulator(max_slippage_bps=D("10000"))
    book = _book([("100", "1")], "asks")
    fill = sim.simulate(book, OrderSide.BUY, base_amount=D("5"))
    assert fill.fill_ratio == D("0.2")
    assert not fill.is_complete


def test_empty_book_rejects():
    sim = FillSimulator()
    book = _book([], "asks")
    fill = sim.simulate(book, OrderSide.BUY, base_amount=D("1"))
    assert fill.is_rejected
    assert fill.rejected_reason == "empty_book"


def test_zero_amount_rejects():
    sim = FillSimulator()
    book = _book([("100", "1")], "asks")
    assert sim.simulate(book, OrderSide.BUY, base_amount=D("0")).is_rejected


def test_quote_budget_buy_never_overspends():
    sim = FillSimulator(taker_fee_bps=D("10"), max_slippage_bps=D("10000"))
    book = _book([("100", "5"), ("110", "5")], "asks")
    fill = sim.simulate(book, OrderSide.BUY, quote_amount=D("300"))
    assert fill.quote_amount == D("300")  # gross spend exactly the budget
    assert fill.filled_amount + fill.fee == D("3")  # 3 ETH gross


def test_fee_charged_in_received_currency():
    """BUY: fee reduces base received; SELL: fee reduces quote received."""
    sim = FillSimulator(taker_fee_bps=D("100"))  # 1%
    asks = _book([("100", "1")], "asks")
    buy = sim.simulate(asks, OrderSide.BUY, base_amount=D("1"))
    assert buy.fee == D("0.01")  # 1% of 1 ETH
    assert buy.filled_amount == D("0.99")
    assert buy.quote_amount == D("100")  # gross spend unchanged

    bids = _book([("100", "1")], "bids")
    sell = sim.simulate(bids, OrderSide.SELL, base_amount=D("1"))
    assert sell.fee == D("1")  # 1% of 100 USDT
    assert sell.quote_amount == D("99")
    assert sell.filled_amount == D("1")  # full base sold


def test_slippage_measured_against_top_of_book():
    sim = FillSimulator(max_slippage_bps=D("10000"))
    book = _book([("100", "1"), ("110", "1")], "asks")
    fill = sim.simulate(book, OrderSide.BUY, base_amount=D("2"))
    # average 105 vs reference 100 -> 500 bps
    assert fill.slippage_bps == D("500.0000")
    assert fill.reference_price == D("100")
