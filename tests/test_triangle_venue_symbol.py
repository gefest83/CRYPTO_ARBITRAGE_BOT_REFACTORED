"""Regression for venue-symbol bug: cross only for venues that list it."""

import pathlib
import tempfile

import pytest
from unittest.mock import AsyncMock, patch
from app.config.settings import Settings, TradingSettings
from app.models.enums import TradingMode
from app.services import build_app
from tests.conftest import make_settings

@pytest.mark.asyncio
async def test_cross_not_in_watch_if_inactive_on_all_venues(tmp_path: pathlib.Path):
    # Use real DEMO settings with default triangle assets (includes BNB, AVAX)
    settings = Settings(_env_file=None)
    settings = settings.model_copy(update={
        "trading": TradingSettings(mode=TradingMode.DEMO, allow_live=False, base_currency="USDT"),
        "database": settings.database.model_copy(update={"url": f"sqlite+aiosqlite:///{tmp_path / 'db.db'}"}),
    })
    app = await build_app(settings)
    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        await app.manager.open_all()
        from app.services import _init_watchlist
        await _init_watchlist(app)
        # AVAX/BNB is inactive on binance and not listed on okx/bybit, so it should NOT be in watch
        watch_names = {s.name for s in app.watch_symbols}
        assert "AVAX/BNB" not in watch_names, f"AVAX/BNB should not be in watch when inactive, got {watch_names}"
        assert "BNB/AVAX" not in watch_names
        # But AVAX/BTC is active on at least one venue, so it should be in watch
        assert "AVAX/BTC" in watch_names
        await app.manager.close()
        await app.db.dispose()

@pytest.mark.asyncio
async def test_triangle_scanner_does_not_use_unsupported_cross(tmp_path: pathlib.Path):
    # Verify that a venue that doesn't list AVAX/BNB does not produce a route for it
    from app.market_data.store import MarketDataStore
    from app.models.market_data import OrderBook, OrderBookLevel
    from app.models.symbol import Symbol
    from app.models.base import utc_now
    from decimal import Decimal
    from app.strategies.triangular.scanner import TriangularScanner
    from app.config.settings import ArbitrageSettings

    store = MarketDataStore(stale_after_ms=60_000)
    # Create books for binance (has AVAX/BNB) and okx (does not)
    # For this test, we simulate that okx has no AVAX/BNB book
    def book(venue, symbol_name):
        sym = Symbol.parse(symbol_name)
        return OrderBook(
            exchange_id=venue,
            symbol=sym,
            bids=(OrderBookLevel(price=Decimal("100"), amount=Decimal("10")),),
            asks=(OrderBookLevel(price=Decimal("101"), amount=Decimal("10")),),
            timestamp=utc_now(),
            received_at=utc_now(),
        )
    # Put AVAX/USDT, BNB/USDT, and AVAX/BNB only for binance (profitable)
    # Use favorable prices for binance AVAX/BNB to ensure the triangle is profitable
    def book_profitable(venue, symbol_name):
        sym = Symbol.parse(symbol_name)
        # For profitable: AVAX/BNB bid 2.0 (so AVAX->BNB gives 2 BNB per AVAX)
        # and BNB/USDT bid 105 (so BNB->USDT gives 105 per BNB)
        # AVAX/USDT ask 100 (so USDT->AVAX costs 100 per AVAX)
        # Cycle: 100 USDT -> 1 AVAX (100) -> 2 BNB (2.0) -> 210 USDT (2*105) => net +110
        if symbol_name == "AVAX/BNB" and venue == "binance":
            return OrderBook(
                exchange_id=venue,
                symbol=sym,
                bids=(OrderBookLevel(price=Decimal("2.0"), amount=Decimal("100")),),
                asks=(OrderBookLevel(price=Decimal("2.01"), amount=Decimal("100")),),
                timestamp=utc_now(),
                received_at=utc_now(),
            )
        if symbol_name == "BNB/USDT" and venue == "binance":
            return OrderBook(
                exchange_id=venue,
                symbol=sym,
                bids=(OrderBookLevel(price=Decimal("105"), amount=Decimal("100")),),
                asks=(OrderBookLevel(price=Decimal("105.5"), amount=Decimal("100")),),
                timestamp=utc_now(),
                received_at=utc_now(),
            )
        return book(venue, symbol_name)
    store.put_order_book(book_profitable("binance", "AVAX/USDT"))
    store.put_order_book(book_profitable("binance", "BNB/USDT"))
    store.put_order_book(book_profitable("binance", "AVAX/BNB"))
    # okx has no AVAX/BNB and its BNB/USDT is not favorable (bid 100 vs binance 105)
    store.put_order_book(book("okx", "AVAX/USDT"))
    store.put_order_book(book("okx", "BNB/USDT"))
    # okx has no AVAX/BNB
    scanner = TriangularScanner(
        store=store,
        venues=lambda: ("binance", "okx"),
        settings=ArbitrageSettings(triangle_assets=("AVAX", "BNB"), triangle_min_net_bps=Decimal("0")),
    )
    # Scan should find one opportunity for binance (with AVAX/BNB) but not for okx
    from app.strategies.triangular.fees import ScanRequest
    req = ScanRequest(symbols=(Symbol.parse("AVAX/USDT"), Symbol.parse("BNB/USDT")), notional_quote=Decimal("100"), min_net_profit_bps=Decimal("0"))
    opps = await scanner.scan(req)
    # At least one opp for binance
    binance_opps = [o for o in opps if o.buy_leg.exchange_id == "binance"]
    okx_opps = [o for o in opps if o.buy_leg.exchange_id == "okx"]
    assert len(binance_opps) >= 1, f"binance should have AVAX->BNB opp, got {opps}"
    assert len(okx_opps) == 0, f"okx should not have AVAX->BNB opp without the cross, got {okx_opps}"
    # Verify the binance opp uses AVAX/BNB
    assert any(o.legs_route[1].symbol.name == "AVAX/BNB" for o in binance_opps)
