"""Triangle execution in PAPER mode (integration through AppServices)."""

import asyncio
from decimal import Decimal

import pytest
from app.config.modes import policy_for
from app.config.settings import Settings
from app.errors import ExecutionDisabledError
from app.execution.fill_simulator import FillSimulator
from app.execution.guard import ExecutionGuard
from app.market_data.store import MarketDataStore
from app.models.arbitrage import ArbitrageLeg, ArbitrageOpportunity
from app.models.balance import Balance, BalanceSnapshot
from app.models.base import utc_now
from app.models.enums import OrderStatus as VenueOrderStatus
from app.models.enums import TradeStatus as VenueTradeStatus
from app.models.enums import TradingMode
from app.models.market_data import OrderBook, OrderBookLevel
from app.models.order import Order as VenueOrder
from app.models.symbol import Symbol
from app.recovery import ExecutionRecovery
from app.services import AppServices
from app.storage.engine import Database
from app.storage.repositories import AuditLogRepository, TradeRepository
from app.strategies.triangular.executor import TriangleExecutor

D = Decimal


async def test_paper_triangle_executes_profitably(services: AppServices):
    opportunities = await services.scan_triangles()
    assert opportunities, "the simulated venues must produce triangle rings"
    best = opportunities[0]
    trade, assessment = await services.execute_triangle(best)
    assert assessment is not None and assessment.approved
    assert trade.status.value == "completed"
    assert trade.input_amount > 0
    assert trade.output_amount > 0
    assert len(trade.orders) == 3  # all three legs recorded
    assert trade.net_profit > 0
    assert trade.net_profit_bps > 0


async def test_risk_rejects_oversized_trade(services: AppServices, tmp_path):
    opportunities = await services.scan_triangles(notional_quote=D("10"))
    assert opportunities
    # Force a notional far above the risk limit (default 1000).
    oversized = opportunities[0].model_copy(
        update={"size_notional_quote": D("999999")},
    )
    trade, assessment = await services.execute_triangle(oversized)
    assert assessment is not None and not assessment.approved
    assert trade.status.value == "failed"
    assert any("max_trade_size" in v.rule for v in assessment.violations)


async def test_risk_rejects_below_minimum_profit(services: AppServices):
    opportunities = await services.scan_triangles()
    assert opportunities
    weak = opportunities[-1].model_copy(
        update={"profit": opportunities[-1].profit.model_copy(update={"gross_spread_bps": D("0")})}
    )
    trade, assessment = await services.execute_triangle(weak)
    assert assessment is not None and not assessment.approved
    assert trade.status.value == "failed"


async def test_kill_switch_blocks_execution_before_any_order(services: AppServices):
    await services.engage_kill_switch("test kill")
    try:
        opportunities = await services.scan_triangles()
        assert opportunities
        with pytest.raises(ExecutionDisabledError):
            await services.execute_triangle(opportunities[0])
    finally:
        await services.release_kill_switch()


async def test_kill_switch_persists_across_restart(services: AppServices, tmp_path):
    await services.engage_kill_switch("persist me")
    from app.services import build_app, shutdown_app, start_app

    # rebuild a fresh app from the same database (simulated restart)
    settings = services.settings
    await shutdown_app(services)
    app2 = await build_app(settings.model_copy(update={"database": settings.database}))
    await start_app(app2)
    try:
        assert app2.guard.is_halted
        assert app2.guard.halt_reason == "persist me"
    finally:
        await app2.release_kill_switch()
        await shutdown_app(app2)


async def test_stale_market_data_refuses_execution(services: AppServices):
    """Books older than the staleness window must not be traded against."""
    opportunities = await services.scan_triangles()
    assert opportunities
    # Age every cached book beyond the window.
    from datetime import timedelta

    for key in list(services.store._books.keys()):
        book = services.store._books[key]
        services.store._books[key] = book.model_copy(
            update={"timestamp": book.timestamp - timedelta(seconds=30)}
        )
    trade, assessment = await services.execute_triangle(opportunities[0])
    if assessment is not None and assessment.approved:
        # risk passed (data age checked from the opportunity, which is fresh);
        # the executor must still refuse per-leg on stale books
        assert trade.status.value == "failed"
        assert "stale" in (trade.error or "").lower()
    else:
        assert any("max_data_age" in v.rule for v in assessment.violations)


async def test_paper_wallet_debits_and_credits(services: AppServices):
    wallet = services.paper_wallets["binance"]
    before = wallet.free("USDT")
    wallet.debit("USDT", D("100"))
    assert wallet.free("USDT") == before - D("100")
    wallet.credit("USDT", D("50"))
    assert wallet.free("USDT") == before - D("50")
    with pytest.raises(ValueError):
        wallet.debit("USDT", wallet.free("USDT") + D("1"))


# ---------------------------------------------------------------------------
# Venue-order semantics (B-1) and crash-safe execution state (H-3/H-4).
#
# Unit tests with a scripted venue adapter in DEMO mode: the fake counts
# create_order calls so every test can assert that no duplicate orders are
# ever placed.
# ---------------------------------------------------------------------------


# Profitable fake route: 100 USDT -> 1 BTC @100 -> 50 ETH @0.02 -> 110 USDT @2.2
_PRICES = {
    "BTC/USDT": (D("100"), D("100.5")),   # bid, ask
    "ETH/BTC": (D("0.0198"), D("0.02")),
    "ETH/USDT": (D("2.2"), D("2.205")),
}


def _deep_book(symbol: str) -> OrderBook:
    bid, ask = _PRICES[symbol]
    return OrderBook(
        exchange_id="binance",
        symbol=_sym(symbol),
        bids=tuple(
            OrderBookLevel(price=bid - D("0.001") * i, amount=D("1000")) for i in range(5)
        ),
        asks=tuple(
            OrderBookLevel(price=ask + D("0.001") * i, amount=D("1000")) for i in range(5)
        ),
        timestamp=utc_now(),
        received_at=utc_now(),
    )


def _sym(name: str):
    return Symbol.parse(name)


class ScriptedVenue:
    """Fake exchange adapter with programmable order outcomes.

    ``create``:
      * ``fill``   — the venue accepts and fills immediately (response returned)
      * ``timeout`` — the venue accepts and fills, but the response is lost
                     (create_order sleeps past the leg timeout)
      * ``raise``  — the venue raises before any order exists

    ``query``:
      * ``ok``   — fetch_open_orders reports the venue's view
      * ``fail`` — the venue is unreachable for recovery queries
    """

    def __init__(self, *, create: str = "fill", query: str = "ok") -> None:
        self.create_behaviour = create
        self.query_behaviour = query
        self.create_calls: list = []
        #: client_order_id -> venue order (the venue's record of the order)
        self.orders_by_client_id: dict[str, VenueOrder] = {}
        #: observed during create_order: was the EXECUTING trade persisted?
        self.executing_persisted_at_submit: list[bool] = []
        self._trade_repo: TradeRepository | None = None

    def bind_trade_repo(self, repo: TradeRepository) -> None:
        self._trade_repo = repo

    async def fetch_balances(self) -> BalanceSnapshot:
        return BalanceSnapshot(
            exchange_id="binance",
            balances=(Balance(exchange_id="binance", asset="USDT", free=D("100000")),),
            timestamp=utc_now(),
        )

    async def create_order(self, request) -> VenueOrder:
        self.create_calls.append(request)
        if self._trade_repo is not None:
            self.executing_persisted_at_submit.append(bool(await self._trade_repo.list_executing()))
        bid, ask = _PRICES[request.symbol.name]
        average = ask if request.side.value == "buy" else bid
        # The venue's own record of the accepted order (used by recovery).
        self.orders_by_client_id[request.client_order_id] = VenueOrder(
            exchange_id="binance",
            exchange_order_id=f"E{len(self.create_calls)}",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            status=VenueOrderStatus.FILLED,
            amount=request.amount,
            filled_amount=request.amount,
            average_price=average,
        )
        if self.create_behaviour == "timeout":
            await asyncio.sleep(1.0)  # response lost; wait_for cancels us
            raise asyncio.CancelledError
        if self.create_behaviour == "raise":
            raise RuntimeError("venue error before acceptance")
        return self.orders_by_client_id[request.client_order_id]

    async def fetch_order(self, order_id, *, symbol):
        if self.query_behaviour == "fail":
            raise RuntimeError("venue unreachable")
        return next(
            (o for o in self.orders_by_client_id.values() if o.exchange_order_id == order_id),
            None,
        )

    async def fetch_open_orders(self, *, symbol=None):
        if self.query_behaviour == "fail":
            raise RuntimeError("venue unreachable")
        return tuple(
            order
            for order in self.orders_by_client_id.values()
            if symbol is None or order.symbol == symbol
        )


class ScriptedManager:
    def __init__(self, adapter: ScriptedVenue) -> None:
        self._adapter = adapter

    def adapter(self, venue: str) -> ScriptedVenue:
        return self._adapter

    def enabled_ids(self):
        return ("binance",)


def _venue_settings(**overrides) -> Settings:
    base = Settings(_env_file=None)
    execution = base.execution.model_copy(update={"leg_timeout_seconds": 0.3})
    trading = base.trading.model_copy(update={"mode": TradingMode.DEMO})
    return base.model_copy(
        update={"trading": trading, "execution": execution, **overrides}
    )


def _opportunity(notional: str = "100") -> ArbitrageOpportunity:
    legs = (
        ArbitrageLeg(
            exchange_id="binance",
            symbol=_sym("BTC/USDT"),
            side=_side_buy(),
            price=D("100"),
            amount=D("1"),
        ),
        ArbitrageLeg(
            exchange_id="binance",
            symbol=_sym("ETH/BTC"),
            side=_side_buy(),
            price=D("0.02"),
            amount=D("50"),
        ),
        ArbitrageLeg(
            exchange_id="binance",
            symbol=_sym("ETH/USDT"),
            side=_side_sell(),
            price=D("2.2"),
            amount=D("50"),
        ),
    )
    return ArbitrageOpportunity(
        symbol=_sym("BTC/USDT"),
        buy_leg=legs[0],
        sell_leg=legs[2],
        legs_route=legs,
        size_notional_quote=D(notional),
        direction="USDT->BTC->ETH->USDT",
        data_age_ms=0.0,
    )


def _side_buy():
    from app.models.enums import OrderSide

    return OrderSide.BUY


def _side_sell():
    from app.models.enums import OrderSide

    return OrderSide.SELL


async def _venue_executor(
    tmp_path,
    *,
    create: str = "fill",
    query: str = "ok",
) -> tuple[TriangleExecutor, ScriptedVenue, TradeRepository, Database]:
    settings = _venue_settings()
    db = Database(
        settings.database.model_copy(
            update={"url": f"sqlite+aiosqlite:///{tmp_path / 'tri.db'}"}
        )
    )
    await db.create_schema()
    trades = TradeRepository(db)
    audit = AuditLogRepository(db)
    store = MarketDataStore(stale_after_ms=60_000)
    for name in _PRICES:
        store.put_order_book(_deep_book(name))
    venue = ScriptedVenue(create=create, query=query)
    venue.bind_trade_repo(trades)
    executor = TriangleExecutor(
        settings=settings,
        store=store,
        manager=ScriptedManager(venue),
        guard=ExecutionGuard(policy_for(TradingMode.DEMO)),
        recovery=ExecutionRecovery(),
        fill_simulator=FillSimulator(
            taker_fee_bps=D("10"), max_slippage_bps=D("500")
        ),
        precision=None,
        trade_repo=trades,
        audit=audit,
        paper_wallets=None,
    )
    return executor, venue, trades, db


def _close(a, b, tolerance: str = "0.000001") -> bool:
    return abs(a - b) <= D(tolerance)


async def test_venue_legs_count_as_real_fills(tmp_path):
    """B-1: a DEMO/LIVE cycle whose legs fill on the venue must complete —
    the returned Order is the authoritative result, not a rejection."""
    executor, venue, trades, db = await _venue_executor(tmp_path, create="fill")
    try:
        trade = await executor.execute(_opportunity())
        assert trade.status.value == "completed", trade.error
        assert len(venue.create_calls) == 3
        assert len(trade.orders) == 3
        # 100 USDT -> ~1 BTC @100.5 -> ~49.75 ETH @0.02 -> ~109.45 USDT @2.2
        assert _close(trade.input_amount, D("100"))
        expected_eth = (D("100") / D("100.5")).quantize(D("0.00000001")) / D("0.02")
        assert _close(trade.output_amount, expected_eth * D("2.2"))
        assert trade.net_profit > 0
        persisted = await trades.get(trade.id)
        assert persisted is not None
        assert persisted.status.value == "completed"
    finally:
        await db.dispose()


async def test_timeout_recovered_to_filled_completes_cycle(tmp_path):
    """B-1: every create_order times out, recovery finds the venue orders
    FILLED — the cycle must complete, not fail."""
    executor, venue, _trades, db = await _venue_executor(
        tmp_path, create="timeout", query="ok"
    )
    try:
        trade = await executor.execute(_opportunity())
        assert trade.status.value == "completed", trade.error
        assert len(venue.create_calls) == 3  # exactly one order per leg
        assert trade.output_amount > 0
        assert trade.net_profit > 0
    finally:
        await db.dispose()


async def test_timeout_unresolved_escalates_to_manual_review(tmp_path):
    """B-2/H-5: a leg whose outcome cannot be established lands the trade in
    MANUAL_REVIEW — never a plain rejection, never a duplicate order."""
    executor, venue, _trades, db = await _venue_executor(
        tmp_path, create="timeout", query="fail"
    )
    try:
        trade = await executor.execute(_opportunity())
        assert trade.status.value == "manual_review", trade.error
        assert "unconfirmed" in (trade.error or "")
        # exactly one order was placed; no unwind, no retry
        assert len(venue.create_calls) == 1
    finally:
        await db.dispose()


async def test_trade_state_persisted_before_leg_1(tmp_path):
    """H-3: the EXECUTING trade record exists in storage before the first
    venue order is submitted (observed from inside create_order)."""
    executor, venue, _trades, db = await _venue_executor(tmp_path, create="fill")
    try:
        await executor.execute(_opportunity())
        assert venue.executing_persisted_at_submit, "create_order was never called"
        assert all(venue.executing_persisted_at_submit), (
            "the EXECUTING trade was not persisted before a venue submission"
        )
    finally:
        await db.dispose()


def _order_payload(
    symbol: str,
    side,
    amount: str,
    average: str,
    *,
    status=VenueOrderStatus.FILLED,
) -> dict:
    import json

    order = VenueOrder(
        exchange_id="binance",
        exchange_order_id="E-x",
        client_order_id=f"cat{'0' * 12}",
        symbol=_sym(symbol),
        side=side,
        order_type=_market(),
        status=status,
        amount=D(amount),
        filled_amount=D(amount),
        average_price=D(average),
    )
    computed = set(order.model_computed_fields)
    return json.loads(order.model_dump_json(exclude=computed))


def _market():
    from app.models.enums import OrderType

    return OrderType.MARKET


def _interrupted_trade(orders: tuple[dict, ...]):
    from app.models.enums import ArbitrageStrategy

    return __import__("app.models.trade", fromlist=["TradeRecord"]).TradeRecord(
        strategy=ArbitrageStrategy.TRIANGLE,
        mode=TradingMode.DEMO,
        exchange_id="binance",
        route="USDT->BTC->ETH->USDT",
        symbols=("BTC/USDT", "ETH/BTC", "ETH/USDT"),
        input_amount=D("100"),
        status=VenueTradeStatus.EXECUTING,
        orders=orders,
    )


async def test_resume_before_leg_1_fails_safely(tmp_path):
    """H-3: a crash before any submission resumes to FAILED (nothing can be
    on the venue) and places no orders."""
    executor, venue, trades, db = await _venue_executor(tmp_path)
    try:
        trade = _interrupted_trade(())
        await trades.save(trade)
        resumed = await executor.resume_open_trades()
        assert len(resumed) == 1
        assert resumed[0].status.value == "failed"
        assert "interrupted before leg 1" in (resumed[0].error or "")
        assert venue.create_calls == []
    finally:
        await db.dispose()


async def test_resume_after_leg_1_requires_manual_review(tmp_path):
    """H-3: a crash after leg 1 filled resumes to MANUAL_REVIEW with the
    held inventory spelled out — never auto-continues the cycle."""
    executor, venue, trades, db = await _venue_executor(tmp_path)
    try:
        leg1 = _order_payload("BTC/USDT", _side_buy(), "1", "100")
        trade = _interrupted_trade((leg1,))
        await trades.save(trade)
        resumed = await executor.resume_open_trades()
        assert resumed[0].status.value == "manual_review"
        error = resumed[0].error or ""
        assert "1 BTC" in error
        assert "manual" in error.lower()
        assert venue.create_calls == []  # no leg 2, no unwind, no retry
    finally:
        await db.dispose()


async def test_resume_after_leg_2_requires_manual_review(tmp_path):
    """H-3: crash after leg 2 filled — same fail-closed behaviour."""
    executor, venue, trades, db = await _venue_executor(tmp_path)
    try:
        leg1 = _order_payload("BTC/USDT", _side_buy(), "1", "100")
        leg2 = _order_payload("ETH/BTC", _side_buy(), "50", "0.02")
        trade = _interrupted_trade((leg1, leg2))
        await trades.save(trade)
        resumed = await executor.resume_open_trades()
        assert resumed[0].status.value == "manual_review"
        assert "50 ETH" in (resumed[0].error or "")
        assert venue.create_calls == []
    finally:
        await db.dispose()


async def test_resume_after_all_legs_filled_completes(tmp_path):
    """H-3: a crash after the last leg filled computes the final P&L and
    completes the trade from venue evidence alone."""
    executor, venue, trades, db = await _venue_executor(tmp_path)
    try:
        leg1 = _order_payload("BTC/USDT", _side_buy(), "1", "100")
        leg2 = _order_payload("ETH/BTC", _side_buy(), "50", "0.02")
        leg3 = _order_payload("ETH/USDT", _side_sell(), "50", "2.2")
        trade = _interrupted_trade((leg1, leg2, leg3))
        await trades.save(trade)
        resumed = await executor.resume_open_trades()
        assert resumed[0].status.value == "completed", resumed[0].error
        assert resumed[0].net_profit == D("10")
        assert venue.create_calls == []
    finally:
        await db.dispose()


async def test_resume_with_unconfirmed_order_escalates(tmp_path):
    """H-3 + B-2: an interrupted leg whose venue state cannot be established
    escalates to MANUAL_REVIEW (recovery queried, nothing placed)."""
    executor, venue, trades, db = await _venue_executor(tmp_path, query="fail")
    try:
        pending = _order_payload(
            "BTC/USDT", _side_buy(), "1", "100", status=VenueOrderStatus.PENDING
        )
        trade = _interrupted_trade((pending,))
        await trades.save(trade)
        resumed = await executor.resume_open_trades()
        assert resumed[0].status.value == "manual_review"
        assert venue.create_calls == []
    finally:
        await db.dispose()


async def test_mid_cycle_kill_switch_holding_goes_to_manual_review(tmp_path):
    """A kill switch engaged after leg 1 filled must not silently fail the
    trade while inventory is held — MANUAL_REVIEW, no unwind order."""
    executor, venue, _trades, db = await _venue_executor(tmp_path, create="fill")
    try:
        # fill leg 1, then engage the kill switch before leg 2
        original_create = venue.create_order
        state = {"calls": 0}

        async def _create_then_halt(request):
            order = await original_create(request)
            state["calls"] += 1
            if state["calls"] == 1:
                executor._guard.engage_kill_switch("test halt")
            return order

        venue.create_order = _create_then_halt
        trade = await executor.execute(_opportunity())
        assert trade.status.value == "manual_review", trade.error
        assert "holding" in (trade.error or "")
        # leg 1 only — the kill switch stopped leg 2 before any submission
        assert len(venue.create_calls) == 1
    finally:
        await db.dispose()
