"""Market data: store freshness, order-book estimation direction, service refresh."""

from decimal import Decimal

from app.clock import FixedClock
from app.config.settings import Settings
from app.exchanges.manager import ExchangeManager
from app.market_data.order_book import (
    estimate_for_base_amount,
    estimate_for_quote_amount,
)
from app.market_data.service import MarketDataService
from app.market_data.store import MarketDataStore
from app.models.base import utc_now
from app.models.enums import OrderSide
from app.models.market_data import OrderBook, OrderBookLevel
from app.models.symbol import Symbol

D = Decimal


def _book(bid: str = "99", ask: str = "101", levels: int = 5) -> OrderBook:
    step = D("1")
    bids = tuple(OrderBookLevel(price=D(bid) - step * i, amount=D("10")) for i in range(levels))
    asks = tuple(OrderBookLevel(price=D(ask) + step * i, amount=D("10")) for i in range(levels))
    return OrderBook(
        exchange_id="binance",
        symbol=Symbol.parse("ETH/USDT"),
        bids=bids,
        asks=asks,
        timestamp=utc_now(),
        received_at=utc_now(),
    )


def test_store_detects_staleness():
    clock = FixedClock()
    store = MarketDataStore(stale_after_ms=2000, clock=clock)
    book = _book()
    store.put_order_book(book)
    assert store.is_fresh(store.age_ms(book))
    clock.advance(seconds=3)
    assert store.is_stale(store.age_ms(book))


def test_store_filters_fresh_books_only():
    clock = FixedClock()
    store = MarketDataStore(stale_after_ms=1000, clock=clock)
    store.put_order_book(_book())
    books = store.order_books_for(Symbol.parse("ETH/USDT"), fresh_only=True)
    assert len(books) == 1
    clock.advance(seconds=2)
    assert store.order_books_for(Symbol.parse("ETH/USDT"), fresh_only=True) == ()
    assert len(store.order_books_for(Symbol.parse("ETH/USDT"))) == 1  # still cached


def test_estimate_buy_walks_asks_sell_walks_bids():
    book = _book()
    buy = estimate_for_base_amount(book, OrderSide.BUY, D("15"))
    # 10 @ 101 + 5 @ 102 = 1010 + 510 = 1520
    assert buy.quote_amount == D("1520")
    assert buy.average_price == D("101.33333333")
    sell = estimate_for_base_amount(book, OrderSide.SELL, D("15"))
    # 10 @ 99 + 5 @ 98 = 990 + 490 = 1480
    assert sell.quote_amount == D("1480")


def test_estimate_quote_budget_floor_rathers_than_overspend():
    book = _book()
    estimate = estimate_for_quote_amount(book, OrderSide.BUY, D("1010"))
    # 10 @ 101 = 1010 exactly; 10 more would cost 1020 -> partial 0.0980 base
    assert estimate.quote_amount <= D("1010")
    assert estimate.filled_amount <= D("10")


def test_books_sorted_correctly():
    book = _book()
    assert [level.price for level in book.bids] == sorted(
        (level.price for level in book.bids), reverse=True
    )
    assert [level.price for level in book.asks] == sorted(level.price for level in book.asks)
    assert book.side(OrderSide.BUY) is book.asks
    assert book.side(OrderSide.SELL) is book.bids


def test_out_of_order_book_updates_are_ignored():
    """H-10: an older snapshot (websocket replay / reconnect duplicate) must
    never replace a newer one; equal and newer updates are preserved."""
    from datetime import timedelta

    clock = FixedClock()
    store = MarketDataStore(stale_after_ms=60_000, clock=clock)

    newer = _book()
    store.put_order_book(newer)

    older = newer.model_copy(
        update={
            "timestamp": newer.timestamp - timedelta(seconds=30),
            "bids": (OrderBookLevel(price=D("1"), amount=D("1")),),
        }
    )
    store.put_order_book(older)  # out-of-order replay — must be dropped
    kept = store.order_book("binance", Symbol.parse("ETH/USDT"))
    assert kept.timestamp == newer.timestamp
    assert kept.bids == newer.bids

    equal = newer.model_copy(
        update={"bids": (OrderBookLevel(price=D("50"), amount=D("5")),)}
    )
    store.put_order_book(equal)  # same snapshot instant, refreshed content
    kept = store.order_book("binance", Symbol.parse("ETH/USDT"))
    assert kept.bids == equal.bids

    freshest = newer.model_copy(update={"timestamp": newer.timestamp + timedelta(seconds=5)})
    store.put_order_book(freshest)
    assert store.order_book("binance", Symbol.parse("ETH/USDT")).timestamp == (
        freshest.timestamp
    )


def test_future_timestamp_is_stale():
    """H-9: a materially future timestamp (replay/tampering/clock break)
    must not pass freshness validation."""
    from datetime import timedelta

    clock = FixedClock()
    store = MarketDataStore(stale_after_ms=2000, clock=clock)
    future = _book()
    future = future.model_copy(update={"timestamp": clock.now() + timedelta(seconds=60)})
    store.put_order_book(future)
    assert store.is_stale(store.age_ms(future))
    assert store.order_books_for(Symbol.parse("ETH/USDT"), fresh_only=True) == ()


def test_small_future_clock_skew_is_tolerated():
    """H-9: a few seconds of venue/local clock skew stay fresh (the venue
    clock is ahead of the local one)."""
    from datetime import timedelta

    clock = FixedClock()
    store = MarketDataStore(stale_after_ms=2000, clock=clock)
    skewed = _book()
    skewed = skewed.model_copy(update={"timestamp": clock.now() + timedelta(seconds=2)})
    store.put_order_book(skewed)
    assert store.is_fresh(store.age_ms(skewed))


async def test_service_refresh_isolates_venue_failures(tmp_path):
    settings = Settings(_env_file=None)
    mgr = ExchangeManager(settings)
    await mgr.open_all()
    try:
        store = MarketDataStore(stale_after_ms=60_000)
        service = MarketDataService(manager=mgr, store=store, config=settings.market_data)
        outcome = await service.refresh_order_books(
            [Symbol.parse("ETH/USDT"), Symbol.parse("BTC/USDT")]
        )
        assert outcome.succeeded == outcome.requested == 6  # 3 venues x 2 symbols
        assert store.order_book("binance", Symbol.parse("ETH/USDT")) is not None
    finally:
        await mgr.close()


async def test_service_skips_breaker_open_venues(tmp_path):
    settings = Settings(_env_file=None)
    mgr = ExchangeManager(settings)
    await mgr.open_all()
    try:
        store = MarketDataStore(stale_after_ms=60_000)
        service = MarketDataService(manager=mgr, store=store, config=settings.market_data)
        # open the breaker for every venue
        for venue in mgr.enabled_ids():
            for _ in range(settings.exchanges.breaker_failure_threshold):
                mgr.record_failure(venue)
        outcome = await service.refresh_order_books([Symbol.parse("ETH/USDT")])
        assert outcome.requested == 0
    finally:
        await mgr.close()
