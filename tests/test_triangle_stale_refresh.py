"""Focused tests for DEMO stale-order-book single-refresh fix.

Covers:
  * stale cached book -> successful refresh -> execution can continue
  * stale cached book -> refresh still stale/missing -> execution rejected
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from app.config.settings import Settings
from app.execution.fill_simulator import FillSimulator
from app.execution.guard import ExecutionGuard
from app.market_data.store import MarketDataStore
from app.models.base import utc_now
from app.models.enums import TradingMode
from app.models.market_data import OrderBook, OrderBookLevel
from app.models.symbol import Symbol
from app.recovery import ExecutionRecovery
from app.storage.engine import Database
from app.storage.repositories import AuditLogRepository, TradeRepository
from app.strategies.triangular.executor import TriangleExecutor


def _sym(name: str) -> Symbol:
    return Symbol.parse(name)


def _book(symbol: str, *, stale: bool = False) -> OrderBook:
    now = utc_now()
    ts = now - timedelta(seconds=5) if stale else now
    return OrderBook(
        exchange_id="binance",
        symbol=_sym(symbol),
        bids=(OrderBookLevel(price=Decimal("100"), amount=Decimal("10")),),
        asks=(OrderBookLevel(price=Decimal("101"), amount=Decimal("10")),),
        timestamp=ts,
        received_at=ts,
    )


class FakeMarket:
    """Fake MarketDataService that records refresh calls and optionally fixes the store."""

    def __init__(self, store: MarketDataStore, *, fix_on_refresh: bool = True, keep_stale: bool = False):
        self.store = store
        self.calls: list[tuple[str, str]] = []
        self.fix_on_refresh = fix_on_refresh
        self.keep_stale = keep_stale

    async def refresh_order_books(self, symbols, *, exchange_ids=None):
        for sym in symbols:
            for venue in (exchange_ids or ["binance"]):
                self.calls.append((venue, sym.name))
                if self.fix_on_refresh:
                    # Put a fresh book
                    fresh = _book(sym.name, stale=self.keep_stale)
                    # If keep_stale, we put a still-stale book to simulate refresh failure
                    self.store.put_order_book(fresh)
                # else: do nothing -> still missing/stale
        # mimic RefreshOutcome
        from app.market_data.service import RefreshOutcome

        return RefreshOutcome(requested=len(self.calls), succeeded=len(self.calls) if self.fix_on_refresh else 0)


def _settings() -> Settings:
    base = Settings(_env_file=None)
    return base.model_copy(update={"trading": base.trading.model_copy(update={"mode": TradingMode.DEMO})})


async def _executor(tmp_path, store: MarketDataStore, market) -> tuple[TriangleExecutor, Database]:
    settings = _settings()
    db = Database(settings.database.model_copy(update={"url": f"sqlite+aiosqlite:///{tmp_path / 'db.db'}"}))
    await db.create_schema()
    trades = TradeRepository(db)
    audit = AuditLogRepository(db)
    from app.config.modes import policy_for

    guard = ExecutionGuard(policy_for(TradingMode.DEMO))
    executor = TriangleExecutor(
        settings=settings,
        store=store,
        manager=type("M", (), {"adapter": lambda self, v: None, "enabled_ids": lambda self: ("binance",), "is_breaker_open": lambda self, v: False})(),
        guard=guard,
        recovery=ExecutionRecovery(),
        fill_simulator=FillSimulator(taker_fee_bps=Decimal("10"), max_slippage_bps=Decimal("500")),
        precision=None,
        trade_repo=trades,
        audit=audit,
        paper_wallets=None,
        market=market,
    )
    return executor, db


@pytest.mark.asyncio
async def test_stale_book_successful_refresh_allows_execution(tmp_path):
    """Stale cached book -> one refresh puts fresh book -> _fresh_book succeeds."""
    store = MarketDataStore(stale_after_ms=2000)
    # Put stale book for BNB/USDT
    store.put_order_book(_book("BNB/USDT", stale=True))
    # Also need fresh books for other legs to make preview succeed later, but this test focuses on _fresh_book directly
    market = FakeMarket(store, fix_on_refresh=True, keep_stale=False)
    executor, db = await _executor(tmp_path, store, market)
    try:
        # Initially stale
        assert store.is_stale(store.age_ms(store.order_book("binance", _sym("BNB/USDT"))))
        # First call should trigger one refresh and then succeed
        book = await executor._fresh_book("binance", _sym("BNB/USDT"))
        assert book is not None
        assert not store.is_stale(store.age_ms(book))
        assert len(market.calls) == 1
        assert market.calls[0] == ("binance", "BNB/USDT")
        # Second call with fresh book should not trigger another refresh
        market.calls.clear()
        book2 = await executor._fresh_book("binance", _sym("BNB/USDT"))
        assert book2 is not None
        assert len(market.calls) == 0
    finally:
        await db.dispose()


@pytest.mark.asyncio
async def test_stale_book_refresh_still_stale_rejected(tmp_path):
    """Stale cached book -> refresh still stale/missing -> fail closed."""
    store = MarketDataStore(stale_after_ms=2000)
    store.put_order_book(_book("LINK/BNB", stale=True))
    # Market that refreshes but keeps stale (simulates venue still not delivering)
    market = FakeMarket(store, fix_on_refresh=True, keep_stale=True)
    executor, db = await _executor(tmp_path, store, market)
    try:
        with pytest.raises(ValueError, match="stale order book"):
            await executor._fresh_book("binance", _sym("LINK/BNB"))
        # Exactly one refresh attempt
        assert len(market.calls) == 1
        # Even after refresh, book is still stale, so still raises
        market.calls.clear()
        with pytest.raises(ValueError, match="stale order book"):
            await executor._fresh_book("binance", _sym("LINK/BNB"))
        assert len(market.calls) == 1
    finally:
        await db.dispose()


@pytest.mark.asyncio
async def test_missing_book_refresh_still_missing_rejected(tmp_path):
    """Missing cached book -> refresh does not create it -> fail closed."""
    store = MarketDataStore(stale_after_ms=2000)
    # No book initially
    market = FakeMarket(store, fix_on_refresh=False)
    executor, db = await _executor(tmp_path, store, market)
    try:
        with pytest.raises(ValueError, match="no order book cached"):
            await executor._fresh_book("binance", _sym("BNB/USDT"))
        assert len(market.calls) == 1
    finally:
        await db.dispose()


@pytest.mark.asyncio
async def test_preview_cycle_uses_refresh_for_stale_cross(tmp_path):
    """Full preview: stale cross book refreshed once allows cycle to proceed."""
    store = MarketDataStore(stale_after_ms=2000)
    # Prepare three books: leg1 and leg3 fresh, leg2 (cross) stale
    store.put_order_book(_book("BNB/USDT", stale=False))
    store.put_order_book(_book("LINK/BNB", stale=True))
    store.put_order_book(_book("LINK/USDT", stale=False))
    market = FakeMarket(store, fix_on_refresh=True, keep_stale=False)
    executor, db = await _executor(tmp_path, store, market)
    try:
        from app.models.arbitrage import ArbitrageLeg
        from app.models.enums import OrderSide

        legs = (
            type("L", (), {"symbol": _sym("BNB/USDT"), "side": OrderSide.BUY})(),
            type("L", (), {"symbol": _sym("LINK/BNB"), "side": OrderSide.BUY})(),
            type("L", (), {"symbol": _sym("LINK/USDT"), "side": OrderSide.SELL})(),
        )
        # Should succeed after one refresh of LINK/BNB
        result = await executor._preview_cycle("binance", legs, Decimal("100"))
        assert result is not None
        # Only the stale symbol should have been refreshed once
        assert ("binance", "LINK/BNB") in market.calls
        assert len([c for c in market.calls if c[1] == "LINK/BNB"]) == 1
    finally:
        await db.dispose()


@pytest.mark.asyncio
async def test_preview_cycle_still_stale_after_refresh_returns_none(tmp_path):
    """Preview still fails if refreshed cross remains stale."""
    store = MarketDataStore(stale_after_ms=2000)
    store.put_order_book(_book("BNB/USDT", stale=False))
    store.put_order_book(_book("LINK/BNB", stale=True))
    store.put_order_book(_book("LINK/USDT", stale=False))
    market = FakeMarket(store, fix_on_refresh=True, keep_stale=True)
    executor, db = await _executor(tmp_path, store, market)
    try:
        from app.models.enums import OrderSide

        legs = (
            type("L", (), {"symbol": _sym("BNB/USDT"), "side": OrderSide.BUY})(),
            type("L", (), {"symbol": _sym("LINK/BNB"), "side": OrderSide.BUY})(),
            type("L", (), {"symbol": _sym("LINK/USDT"), "side": OrderSide.SELL})(),
        )
        result = await executor._preview_cycle("binance", legs, Decimal("100"))
        assert result is None
    finally:
        await db.dispose()
