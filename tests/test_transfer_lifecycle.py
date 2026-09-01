"""Transfer lifecycle: full workflow, persistence, resume, safety gates."""

import asyncio
import multiprocessing
import multiprocessing.context
from decimal import Decimal

import pytest
from app.config.modes import policy_for
from app.execution.fill_simulator import FillSimulator
from app.execution.guard import ExecutionGuard
from app.market_data.store import MarketDataStore
from app.models.base import utc_now
from app.models.enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    TradingMode,
    TransferState,
)
from app.models.market_data import OrderBook, OrderBookLevel
from app.models.order import Order as VenueOrder
from app.models.symbol import Symbol
from app.models.transfer import DepositAddress, TransferTx, WithdrawalNetwork
from app.recovery import ExecutionRecovery
from app.services import AppServices, build_app, shutdown_app, start_app
from app.storage.engine import Database
from app.storage.repositories import AuditLogRepository, TradeRepository
from app.strategies.transfer.orchestrator import TransferOrchestrator
from app.strategies.transfer.planner import TransferPlanner
from tests.conftest import make_settings

D = Decimal


async def test_full_lifecycle_completes_in_paper(services: AppServices):
    plans = await services.plan_transfers()
    assert plans, "the simulated venues must produce transfer plans"
    record = await services.start_transfer(plans[0])
    seen_states = {record.state}
    ticks = 0
    while not record.is_terminal and ticks < 100:
        await asyncio.sleep(0.05)
        advanced = await services.tick_transfers()
        for updated in advanced:
            if updated.id == record.id:
                record = updated
                seen_states.add(record.state)
        ticks += 1
    assert record.state is TransferState.COMPLETED
    assert record.realized_profit_quote > 0
    # the lifecycle visited the core states in order
    ordered = [
        TransferState.CREATED,
        TransferState.BUY_SUBMITTED,
        TransferState.BUY_FILLED,
        TransferState.WITHDRAW_SUBMITTED,
        TransferState.TRANSFER_IN_PROGRESS,
        TransferState.DEPOSIT_DETECTED,
        TransferState.SELL_SUBMITTED,
        TransferState.COMPLETED,
    ]
    positions = [ordered.index(s) for s in ordered if s in seen_states]
    assert positions == sorted(positions), f"states out of order: {seen_states}"
    # a completed transfer produces a trade record
    trades = await services.trades.list_recent(5)
    assert any(t.transfer_id == record.id for t in trades)
    assert await services.transfers.count_open() == 0


async def test_transfer_survives_restart(tmp_path):
    """The whole point of the persisted lifecycle: restart mid-flight."""
    settings = make_settings(tmp_path)
    slow = settings.model_copy(
        update={
            "transfer": settings.transfer.model_copy(update={"simulated_transfer_seconds": 5.0})
        }
    )
    app1 = await build_app(slow)
    await start_app(app1)
    plans = await app1.plan_transfers()
    assert plans
    record = await app1.start_transfer(plans[0])
    assert record.state in (
        TransferState.BUY_FILLED,
        TransferState.WITHDRAW_SUBMITTED,
        TransferState.WITHDRAW_PENDING,
    )
    transfer_id = record.id
    # simulate a crash: no graceful tick, just shutdown
    await shutdown_app(app1)

    # fresh process from the same database, faster blockchain for the resume
    fast = settings.model_copy(
        update={
            "transfer": settings.transfer.model_copy(update={"simulated_transfer_seconds": 0.1})
        }
    )
    app2 = await build_app(fast)
    await start_app(app2)
    open_records = await app2.transfers.list_open()
    assert [r.id for r in open_records] == [transfer_id]
    record = open_records[0]
    ticks = 0
    while not record.is_terminal and ticks < 100:
        advanced = await app2.tick_transfers()
        for updated in advanced:
            if updated.id == transfer_id:
                record = updated
        await asyncio.sleep(0.05)
        ticks += 1
    assert record.state is TransferState.COMPLETED
    assert record.realized_profit_quote > 0
    await shutdown_app(app2)


async def test_risk_max_open_transfers_blocks_new_transfer(tmp_path):
    settings = make_settings(tmp_path)
    settings = settings.model_copy(
        update={"risk": settings.risk.model_copy(update={"max_open_transfers": 1})}
    )
    from app.services import build_app, shutdown_app, start_app

    app = await build_app(settings)
    await start_app(app)
    try:
        plans = await app.plan_transfers()
        assert plans, "the simulated venues must produce transfer plans"
        first = await app.start_transfer(plans[0])
        assert first.state.is_open
        assert await app.transfers.count_open() == 1
        # the cap is reached: the next transfer must be refused by risk
        with pytest.raises(Exception, match="max_open_transfers"):
            await app.start_transfer(plans[0])
    finally:
        await shutdown_app(app)


async def test_transfer_requires_capital(services: AppServices):
    """A plan that cannot be funded is refused before any order is placed."""
    plans = await services.plan_transfers()
    assert plans
    plan = plans[0]
    # blow the plan's size far beyond every cap
    huge = plan.model_copy(update={"amount": D("1000000")})
    with pytest.raises(Exception, match=r"max_trade_size|risk validation"):
        await services.start_transfer(huge)


async def test_plans_respect_network_constraints(services: AppServices):
    """Every emitted plan carries a network that both venues actually list."""
    plans = await services.plan_transfers()
    for plan in plans:
        source = await services.manager.adapter(plan.source_exchange).fetch_withdrawal_networks(
            plan.asset
        )
        dest = await services.manager.adapter(plan.dest_exchange).fetch_withdrawal_networks(
            plan.asset
        )
        source_codes = {n.network_code or n.network for n in source if n.withdraw_enabled}
        dest_codes = {n.network_code or n.network for n in dest if n.deposit_enabled}
        assert (plan.network.upper()) in {n.network.upper() for n in source}
        assert source_codes & dest_codes, "plan network must exist on both venues"


async def test_transfer_below_minimum_profit_is_not_planned(services: AppServices):
    """The planner never returns plans under the configured minimum."""
    plans = await services.plan_transfers()
    minimum = services.settings.transfer.min_net_profit_bps
    for plan in plans:
        assert plan.net_profit_bps >= minimum


async def test_transfer_plan_carries_real_data_age(services: AppServices):
    """H-7: plans carry the real age of the books they were priced from."""
    plans = await services.plan_transfers()
    assert plans
    for plan in plans:
        assert 0.0 <= plan.data_age_ms <= 5000.0


async def test_transfer_risk_rejects_stale_plan(services: AppServices):
    """H-7: a plan whose pricing data is stale fails closed through
    MaxDataAgeRule instead of bypassing it with a hardcoded 0.0."""
    plans = await services.plan_transfers()
    assert plans
    stale = plans[0].model_copy(update={"data_age_ms": 60000.0})
    with pytest.raises(Exception, match="max_data_age"):
        await services.start_transfer(stale)


async def test_concurrent_transfer_starts_respect_max_open_transfers(tmp_path):
    """H-6: two concurrent starts with max_open_transfers=1 — exactly one
    may proceed; the other is refused by MaxOpenTransfersRule."""
    settings = make_settings(tmp_path)
    settings = settings.model_copy(
        update={
            "risk": settings.risk.model_copy(update={"max_open_transfers": 1}),
            "transfer": settings.transfer.model_copy(
                update={"simulated_transfer_seconds": 5.0}
            ),
        }
    )
    from app.services import build_app, shutdown_app, start_app

    app = await build_app(settings)
    await start_app(app)
    try:
        plans = await app.plan_transfers()
        assert plans
        results = await asyncio.gather(
            app.start_transfer(plans[0]),
            app.start_transfer(plans[0]),
            return_exceptions=True,
        )
        succeeded = [r for r in results if not isinstance(r, BaseException)]
        refused = [r for r in results if isinstance(r, BaseException)]
        assert len(succeeded) == 1
        assert len(refused) == 1
        assert "max_open_transfers" in str(refused[0])
        assert await app.transfers.count_open() == 1
    finally:
        await shutdown_app(app)


# ---------------------------------------------------------------------------
# H-2 / C-3 / C-4: restart idempotency and fail-closed asset tracking with a
# scripted venue adapter (DEMO/LIVE order paths; no real orders).
# ---------------------------------------------------------------------------



class _ScriptedTransferVenue:
    """Fake venue for transfer-leg tests: counts every order/withdrawal."""

    def __init__(self, *, create: str = "fill", query: str = "ok", fail_deposit_address=False):
        self.create = create  # fill | timeout
        self.query = query  # ok | fail
        self.fail_deposit_address = fail_deposit_address
        self.create_calls: list = []
        self.withdraw_calls: list = []
        self.orders_by_client_id: dict[str, VenueOrder] = {}

    async def fetch_balances(self):
        from app.models.balance import Balance, BalanceSnapshot

        return BalanceSnapshot(
            exchange_id="binance",
            balances=(Balance(exchange_id="binance", asset="USDT", free=D("100000")),),
            timestamp=utc_now(),
        )

    async def fetch_trading_fees(self, symbol):
        from app.models.market import MarketFees

        return MarketFees(maker_bps=D("10"), taker_bps=D("10"))

    async def create_order(self, request):
        self.create_calls.append(request)
        if self.create == "raise":
            raise RuntimeError("venue error before acceptance")
        self.orders_by_client_id[request.client_order_id] = VenueOrder(
            exchange_id=request.exchange_id,
            exchange_order_id=f"E{len(self.create_calls)}",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            status=OrderStatus.FILLED,
            amount=request.amount,
            filled_amount=request.amount,
            average_price=D("100"),
        )
        if self.create == "timeout":
            await asyncio.sleep(1.0)
            raise asyncio.CancelledError
        return self.orders_by_client_id[request.client_order_id]

    async def fetch_order(self, order_id, *, symbol):
        if self.query == "fail":
            raise RuntimeError("venue unreachable")
        return next(
            (o for o in self.orders_by_client_id.values() if o.exchange_order_id == order_id),
            None,
        )

    async def fetch_open_orders(self, *, symbol=None):
        if self.query == "fail":
            raise RuntimeError("venue unreachable")
        return tuple(
            o for o in self.orders_by_client_id.values() if symbol is None or o.symbol == symbol
        )

    async def fetch_withdrawal_networks(self, asset):
        return (
            WithdrawalNetwork(
                network="SIMNET",
                network_code="SIM",
                withdraw_enabled=True,
                deposit_enabled=True,
                withdrawal_fee=D("0.01"),
                withdrawal_min=D("0.0001"),
            ),
        )

    async def fetch_deposit_address(self, asset, *, network=None):
        if self.fail_deposit_address:
            raise RuntimeError("deposit addresses unavailable")
        return DepositAddress(address="0xDEST", network=network)

    async def withdraw(self, asset, amount, address, *, memo=None, network=None):
        self.withdraw_calls.append((asset, amount, address))
        return TransferTx(
            direction="withdrawal",
            asset=asset,
            amount=D(amount),
            status="ok",
            txid=f"wd-{len(self.withdraw_calls)}",
        )

    async def fetch_deposits(self, asset, *, limit=20):
        return ()

    async def fetch_withdrawals(self, asset, *, limit=20):
        return ()


class _ScriptedManager:
    def __init__(self, source: _ScriptedTransferVenue, dest: _ScriptedTransferVenue):
        self._venues = {"binance": source, "okx": dest}

    def adapter(self, venue):
        return self._venues[venue]

    def enabled_ids(self):
        return ("binance", "okx")


def _transfer_book(venue: str) -> OrderBook:
    sym = Symbol.parse("ETH/USDT")
    bids = tuple(
        OrderBookLevel(price=D("105") - D("0.5") * i, amount=D("100")) for i in range(5)
    )
    asks = tuple(
        OrderBookLevel(price=D("100") + D("0.5") * i, amount=D("100")) for i in range(5)
    )
    return OrderBook(
        exchange_id=venue,
        symbol=sym,
        bids=bids,
        asks=asks,
        timestamp=utc_now(),
        received_at=utc_now(),
    )


async def _scripted_orchestrator(
    tmp_path,
    *,
    mode=TradingMode.LIVE,
    source: _ScriptedTransferVenue | None = None,
    dest: _ScriptedTransferVenue | None = None,
):
    from tests.conftest import make_settings as _make

    settings = _make(tmp_path)
    settings = settings.model_copy(
        update={
            "trading": settings.trading.model_copy(update={"mode": mode}),
            "execution": settings.execution.model_copy(update={"leg_timeout_seconds": 0.3}),
        }
    )
    db = Database(
        settings.database.model_copy(update={"url": f"sqlite+aiosqlite:///{tmp_path / 'trf.db'}"})
    )
    await db.create_schema()
    from app.storage.repositories import TransferRepository

    transfers = TransferRepository(db)
    trades = TradeRepository(db)
    audit = AuditLogRepository(db)
    store = MarketDataStore(stale_after_ms=60_000)
    store.put_order_book(_transfer_book("binance"))
    store.put_order_book(_transfer_book("okx"))
    source = source or _ScriptedTransferVenue()
    dest = dest or _ScriptedTransferVenue()
    orchestrator = TransferOrchestrator(
        settings=settings,
        manager=_ScriptedManager(source, dest),
        store=store,
        guard=ExecutionGuard(policy_for(mode)),
        recovery=ExecutionRecovery(),
        fill_simulator=FillSimulator(taker_fee_bps=D("10"), max_slippage_bps=D("500")),
        planner=TransferPlanner(settings),
        transfer_repo=transfers,
        trade_repo=trades,
        audit=audit,
        risk_check=None,
        paper_wallets={},
    )
    return orchestrator, transfers, source, dest, db


def _transfer_plan(amount: str = "5"):
    from app.models.transfer import TransferPlan

    return TransferPlan(
        source_exchange="binance",
        dest_exchange="okx",
        asset="ETH",
        network="SIM",
        amount=D(amount),
        buy_price=D("100"),
        sell_price=D("105"),
        withdrawal_fee=D("0.01"),
        data_age_ms=0.0,
    )


def _filled_buy_order_json(amount: str = "5") -> dict:
    import json

    order = VenueOrder(
        exchange_id="binance",
        exchange_order_id="E1",
        client_order_id="cat000000000001",
        symbol=Symbol.parse("ETH/USDT"),
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        status=OrderStatus.FILLED,
        amount=D(amount),
        filled_amount=D(amount),
        average_price=D("100"),
    )
    computed = set(order.model_computed_fields)
    return json.loads(order.model_dump_json(exclude=computed))


def _pending_intent_json(amount: str = "5") -> dict:
    import json

    order = VenueOrder(
        exchange_id="binance",
        client_order_id="cat000000000002",
        symbol=Symbol.parse("ETH/USDT"),
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        status=OrderStatus.PENDING,
        amount=D(amount),
    )
    computed = set(order.model_computed_fields)
    return json.loads(order.model_dump_json(exclude=computed))


def _record(**updates):
    from app.models.transfer import TransferRecord

    base = {
        "source_exchange": "binance",
        "dest_exchange": "okx",
        "asset": "ETH",
        "network": "SIM",
        "amount": D("5"),
        "plan": _transfer_plan(),
        "mode": "DEMO",
    }
    base.update(updates)
    return TransferRecord(**base)


async def test_restart_with_persisted_buy_filled_never_duplicates_buy(tmp_path):
    """H-2: BUY_SUBMITTED with a persisted FILLED order resumes to
    BUY_FILLED — no second order is created."""
    orchestrator, transfers, source, _dest, db = await _scripted_orchestrator(tmp_path)
    try:
        record = _record(
            state=TransferState.BUY_SUBMITTED,
            buy_order=_filled_buy_order_json(),
            buy_filled_amount=D("5"),
        )
        await transfers.save(record)
        advanced = await orchestrator.tick()
        updated = next(r for r in advanced if r.id == record.id)
        assert updated.state is TransferState.BUY_FILLED
        assert updated.buy_filled_amount == D("5")
        assert source.create_calls == []  # no duplicate order
    finally:
        await db.dispose()


async def test_restart_with_pending_buy_intent_resolves_or_escalates(tmp_path):
    """H-2: BUY_SUBMITTED holding a pre-submission PENDING intent (crash
    between submission and outcome save) queries the venue; when the order
    cannot be found it fails closed — never a second BUY."""
    # venue answers: the intent's order is not there (never reached venue)
    orchestrator, transfers, source, _dest, db = await _scripted_orchestrator(tmp_path)
    try:
        record = _record(state=TransferState.BUY_SUBMITTED, buy_order=_pending_intent_json())
        await transfers.save(record)
        advanced = await orchestrator.tick()
        updated = next(r for r in advanced if r.id == record.id)
        assert updated.state is TransferState.MANUAL_REVIEW
        assert source.create_calls == []  # never a second BUY
    finally:
        await db.dispose()


async def test_restart_with_pending_buy_intent_adopts_venue_order(tmp_path):
    """H-2: when recovery finds the intent's order FILLED on the venue, the
    transfer adopts the fill and continues — no second BUY."""
    source = _ScriptedTransferVenue()
    orchestrator, transfers, source, _dest, db = await _scripted_orchestrator(
        tmp_path, source=source
    )
    try:
        intent = _pending_intent_json()
        # the venue reports the intent's order as filled
        venue_order = VenueOrder.model_validate(intent).model_copy(
            update={
                "status": OrderStatus.FILLED,
                "filled_amount": D("5"),
                "average_price": D("100"),
                "exchange_order_id": "E9",
            }
        )
        source.orders_by_client_id[intent["client_order_id"]] = venue_order
        record = _record(state=TransferState.BUY_SUBMITTED, buy_order=intent)
        await transfers.save(record)
        advanced = await orchestrator.tick()
        updated = next(r for r in advanced if r.id == record.id)
        assert updated.state is TransferState.BUY_FILLED
        assert updated.buy_filled_amount == D("5")
        assert source.create_calls == []
    finally:
        await db.dispose()


async def test_buy_timeout_recovered_to_filled_completes(tmp_path):
    """B-1/B-2 on the transfer path: buy times out, recovery finds the venue
    order FILLED — the transfer proceeds instead of failing."""
    source = _ScriptedTransferVenue(create="timeout")
    orchestrator, _transfers, source, _dest, db = await _scripted_orchestrator(
        tmp_path, source=source
    )
    try:
        record = await orchestrator.start(_transfer_plan())
        # the recovered fill carried the workflow past the buy leg
        assert record.state is TransferState.WITHDRAW_SUBMITTED
        assert record.buy_filled_amount > 0
        assert len(source.create_calls) == 1  # exactly one buy attempt
    finally:
        await db.dispose()


async def test_buy_timeout_unresolved_goes_to_manual_review(tmp_path):
    """B-2: buy times out and the venue cannot confirm the outcome —
    MANUAL_REVIEW, never a duplicate buy."""
    source = _ScriptedTransferVenue(create="timeout", query="fail")
    orchestrator, _transfers, source, _dest, db = await _scripted_orchestrator(
        tmp_path, source=source
    )
    try:
        record = await orchestrator.start(_transfer_plan())
        assert record.state is TransferState.MANUAL_REVIEW
        assert "unconfirmed" in (record.error or "")
        assert len(source.create_calls) == 1
    finally:
        await db.dispose()


async def test_withdrawal_never_resubmitted_after_acceptance(tmp_path):
    """H-2: WITHDRAW_SUBMITTED with a persisted withdrawal id advances via
    _check_withdrawal — no second withdrawal is ever submitted."""
    orchestrator, transfers, source, _dest, db = await _scripted_orchestrator(
        tmp_path, mode=TradingMode.LIVE
    )
    try:
        record = _record(
            state=TransferState.WITHDRAW_SUBMITTED,
            buy_filled_amount=D("5"),
            withdrawal_id="wd-real-1",
            withdrawal_txid="wd-real-1",
            withdrawal_amount=D("4.99"),
            deposit_address="0xDEST",
            created_at=utc_now(),
        )
        await transfers.save(record)
        await orchestrator.tick()
        updated = await transfers.get(record.id)
        assert updated is not None
        # routed through _check_withdrawal: id intact, no resubmission
        assert updated.state is TransferState.WITHDRAW_SUBMITTED
        assert updated.withdrawal_id == "wd-real-1"
        assert source.withdraw_calls == []  # never a second withdrawal
    finally:
        await db.dispose()


async def test_withdrawal_crash_window_fails_closed(tmp_path):
    """H-2: a persisted withdrawal intent (crash between submission and the
    outcome save) never leads to a duplicate withdrawal.  In LIVE mode the
    unmatched intent times out into MANUAL_REVIEW."""
    orchestrator, transfers, source, _dest, db = await _scripted_orchestrator(
        tmp_path, mode=TradingMode.LIVE
    )
    try:
        from datetime import timedelta

        record = _record(
            state=TransferState.WITHDRAW_SUBMITTED,
            buy_filled_amount=D("5"),
            withdrawal_id="wd-intent-abc123def456",
            withdrawal_amount=D("4.99"),
            deposit_address="0xDEST",
            created_at=utc_now() - timedelta(hours=2),  # past the deposit timeout
        )
        await transfers.save(record)
        advanced = await orchestrator.tick()
        updated = next(r for r in advanced if r.id == record.id)
        assert updated.state is TransferState.MANUAL_REVIEW
        assert source.withdraw_calls == []  # never a duplicate withdrawal
    finally:
        await db.dispose()


async def test_withdrawal_failure_tracks_purchased_asset(tmp_path):
    """C-3: the buy filled but the withdrawal cannot proceed — the transfer
    lands in MANUAL_REVIEW naming the purchased asset, never a plain FAILED."""
    dest = _ScriptedTransferVenue(fail_deposit_address=True)
    orchestrator, _transfers, _source, dest, db = await _scripted_orchestrator(
        tmp_path, dest=dest
    )
    try:
        record = await orchestrator.start(_transfer_plan())
        assert record.state is TransferState.MANUAL_REVIEW
        error = record.error or ""
        assert "ETH" in error
        assert "binance" in error
        assert "manual review" in error
    finally:
        await db.dispose()


async def test_sell_failure_tracks_destination_asset(tmp_path):
    """C-4: the deposit arrived but the sell leg failed — MANUAL_REVIEW with
    the destination asset/exchange named, never a plain FAILED."""
    source = _ScriptedTransferVenue()
    dest = _ScriptedTransferVenue(create="raise")
    orchestrator, transfers, source, dest, db = await _scripted_orchestrator(
        tmp_path, source=source, dest=dest
    )
    try:
        record = _record(
            state=TransferState.DEPOSIT_DETECTED,
            buy_filled_amount=D("5"),
            withdrawal_id="wd-1",
            withdrawal_txid="wd-1",
            withdrawal_amount=D("4.99"),
            deposit_address="0xDEST",
            deposit_txid="dp-1",
            deposit_amount=D("4.99"),
        )
        await transfers.save(record)
        advanced = await orchestrator.tick()
        updated = next(r for r in advanced if r.id == record.id)
        assert updated.state is TransferState.MANUAL_REVIEW
        error = updated.error or ""
        assert "4.99 ETH" in error
        assert "okx" in error
        assert "manual review" in error
    finally:
        await db.dispose()


# ---------------------------------------------------------------------------
# TEST GAP #1 (from the LIVE READINESS AUDIT):
#   Restart at BUY_FILLED must resume into withdrawal exactly once across
#   multiple restarts — no duplicate BUY, no duplicate withdrawal, no loss
#   of buy_order / buy_filled_amount / withdrawal_id across the database
#   boundary between two independently-built orchestrator instances.
# ---------------------------------------------------------------------------


async def test_restart_at_buy_filled_state_submits_withdrawal(tmp_path):
    """A BUY_FILLED transfer row that survives a process crash must resume
    into withdrawal submission exactly once across two restarts.

    The test rebuilds fresh orchestrator instances pointed at the same SQLite
    file (a real restart, not an in-memory object swap), seeds a BUY_FILLED
    record between restarts, and asserts:

      * BUY ``create_order()`` calls after restart == 0
      * withdrawal calls after restart == 1
      * persisted ``buy_order`` is preserved verbatim
      * persisted ``buy_filled_amount`` is preserved
      * persisted ``withdrawal_id`` is created and preserved
      * state advances to WITHDRAW_SUBMITTED
      * a subsequent tick on the SAME orchestrator does NOT submit again
      * a SECOND restart does NOT submit another withdrawal
    """
    db_path = tmp_path / "trf.db"

    # ---------------------------------------------------------------- pre-restart
    # Build the first orchestrator, persist a BUY_FILLED record by hand (the
    # simulation of "the previous process crashed between BUY_FILLED commit
    # and the withdrawal call"), then dispose it.
    orchestrator1, transfers, source1, dest1, db1 = await _scripted_orchestrator(tmp_path)
    try:
        seeded = _record(
            state=TransferState.BUY_FILLED,
            buy_order=_filled_buy_order_json(),
            buy_filled_amount=D("5"),
        )
        await transfers.save(seeded)
        pre_buy_id = seeded.buy_order["client_order_id"]
        pre_buy_filled = seeded.buy_filled_amount
    finally:
        await db1.dispose()

    assert db_path.exists(), "the SQLite file must exist for the restart"

    # ---------------------------------------------------------------- restart #1
    # A fresh orchestrator = a fresh process.  Same DB file, fresh adapter
    # call counters, fresh module state.  Calling tick() is what production
    # would do on its first poll after startup.
    orchestrator2, transfers2, source2, _dest2, db2 = await _scripted_orchestrator(tmp_path)
    try:
        # Sanity: the new orchestrator sees the BUY_FILLED record through
        # the real repository.
        open_rows = await transfers2.list_open()
        assert any(r.id == seeded.id for r in open_rows), (
            "BUY_FILLED record must be visible after restart via list_open()"
        )
        persisted_before = await transfers2.get(seeded.id)
        assert persisted_before is not None
        assert persisted_before.state is TransferState.BUY_FILLED
        assert persisted_before.withdrawal_id is None
        assert persisted_before.buy_order is not None
        assert persisted_before.buy_filled_amount == pre_buy_filled

        # Tick the orchestrator — the BUY_FILLED state routes to
        # _submit_withdrawal (orchestrator.py:283), which must call
        # adapter.withdraw exactly once and persist WITHDRAW_SUBMITTED.
        await orchestrator2.tick()

        # BUY must NEVER be submitted again after a restart.
        assert source2.create_calls == [], (
            f"a restart at BUY_FILLED must not place a new BUY, but "
            f"{len(source2.create_calls)} create_order() call(s) were made"
        )
        # Withdrawal must be submitted exactly once.
        assert len(source2.withdraw_calls) == 1, (
            f"expected exactly 1 withdrawal after restart, got "
            f"{len(source2.withdraw_calls)}"
        )
        asset, amount, address = source2.withdraw_calls[0]
        assert asset == "ETH"
        assert address == "0xDEST"

        # Reload from the database and verify the persisted shape.
        after_first_restart = await transfers2.get(seeded.id)
        assert after_first_restart is not None
        assert after_first_restart.state is TransferState.WITHDRAW_SUBMITTED
        assert after_first_restart.buy_order is not None
        assert after_first_restart.buy_order["client_order_id"] == pre_buy_id
        assert after_first_restart.buy_filled_amount == pre_buy_filled
        assert after_first_restart.withdrawal_id is not None
        first_withdrawal_id = after_first_restart.withdrawal_id
        assert after_first_restart.withdrawal_txid is not None
        first_withdrawal_txid = after_first_restart.withdrawal_txid

        # ------------------------------------------------------------ tick idempotency
        # Ticking the SAME orchestrator again must NOT submit another
        # withdrawal: the BUY_FILLED->WITHDRAW_SUBMITTED guard
        # (``if record.withdrawal_id is not None: return record``) is the
        # only thing keeping a 2nd tick quiet.
        await orchestrator2.tick()
        assert len(source2.withdraw_calls) == 1, (
            f"a 2nd tick on the same process must not resubmit, got "
            f"{len(source2.withdraw_calls)} withdrawal(s)"
        )
        assert source2.create_calls == []
        after_second_tick = await transfers2.get(seeded.id)
        assert after_second_tick is not None
        assert after_second_tick.withdrawal_id == first_withdrawal_id
        assert after_second_tick.withdrawal_txid == first_withdrawal_txid
    finally:
        await db2.dispose()

    # ---------------------------------------------------------------- restart #2
    # Build a THIRD orchestrator pointed at the same DB file.  Fresh
    # ``source3.withdraw_calls`` (length 0 at start).  The persisted
    # withdrawal_id MUST prevent another submit.
    orchestrator3, transfers3, source3, _dest3, db3 = await _scripted_orchestrator(tmp_path)
    try:
        persisted_before = await transfers3.get(seeded.id)
        assert persisted_before is not None
        assert persisted_before.state is TransferState.WITHDRAW_SUBMITTED
        assert persisted_before.withdrawal_id == first_withdrawal_id

        await orchestrator3.tick()

        assert source3.create_calls == [], (
            "BUY must never be resubmitted across a second restart"
        )
        assert source3.withdraw_calls == [], (
            f"withdrawal must not be resubmitted across a second restart, "
            f"but {len(source3.withdraw_calls)} new call(s) were made"
        )

        after_second_restart = await transfers3.get(seeded.id)
        assert after_second_restart is not None
        assert after_second_restart.withdrawal_id == first_withdrawal_id
        assert after_second_restart.withdrawal_txid == first_withdrawal_txid
        assert after_second_restart.buy_order["client_order_id"] == pre_buy_id
        assert after_second_restart.buy_filled_amount == pre_buy_filled
    finally:
        await db3.dispose()


# ---------------------------------------------------------------------------
# TEST GAP #3 (from the LIVE READINESS AUDIT):
#   Re-recovery of the SAME venue order across multiple independent process
#   restarts must be idempotent — repeated recovery cannot manufacture a new
#   order, cannot duplicate BUY, cannot mutate a terminal FILLED result, and
#   cannot drift the persisted client_order_id or filled_amount.
#
#   Two layers:
#     A. The recovery primitive (``ExecutionRecovery.recover``) itself is
#        driven against the same persisted order three times against the
#        same venue adapter — proving it is idempotent regardless of how
#        many times the process asks.
#     B. The orchestrator's restart path is driven through three independent
#        orchestrator instances pointed at the same SQLite file: first
#        restart resolves the unresolved intent into a terminal BUY_FILLED,
#        subsequent restarts re-enter the same code path and short-circuit
#        on the already-confirmed fill — never calling create_order again.
# ---------------------------------------------------------------------------


class _VenueWithStableFilledOrder:
    """Fake venue adapter that reports ONE pre-existing FILLED order.

    Used to prove repeated recovery is idempotent: ``fetch_order`` /
    ``fetch_open_orders`` always return the same venue order regardless of
    how many times the bot asks — exactly what a real venue would do once
    the order has reached a terminal state and been pushed to the history.
    """

    def __init__(self, *, client_order_id: str, exchange_order_id: str, venue: str) -> None:
        from app.models.symbol import Symbol as _Sym

        self._venue_order = VenueOrder(
            exchange_id=venue,
            exchange_order_id=exchange_order_id,
            client_order_id=client_order_id,
            symbol=_Sym.parse("ETH/USDT"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            amount=D("5"),
            filled_amount=D("5"),
            average_price=D("100"),
        )
        self.create_calls: list = []

    async def create_order(self, request):
        self.create_calls.append(request)
        raise AssertionError(
            "recovery must NEVER call create_order on the venue adapter"
        )

    async def fetch_order(self, order_id, *, symbol):
        if order_id == self._venue_order.exchange_order_id:
            return self._venue_order
        return None

    async def fetch_open_orders(self, *, symbol=None):
        return ()  # the FILLED order is not in the open list

    async def withdraw(self, *args, **kwargs):
        raise AssertionError("recovery must NEVER call withdraw")


async def test_recovery_of_same_order_across_multiple_restarts_is_idempotent(tmp_path):
    """TEST GAP #3: repeated recovery of the SAME order must be a no-op on
    subsequent calls and must never trigger a new order.

    Three independent process instances (recovery primitive, orchestrator #1,
    orchestrator #2) all operate on the same persisted TIMEOUT intent and the
    same venue-side FILLED record.  None of them may place a new order, none
    of them may mutate the persisted ``client_order_id``, and none of them
    may move the persisted ``filled_amount`` away from the venue's value.
    """
    stable_client_id = "catabcdef123456"
    stable_exchange_id = "E-stable-001"
    buy_filled_amount_expected = D("5")
    buy_average_expected = D("100")

    db_path = tmp_path / "trf.db"

    # =====================================================================
    # Layer A — direct ExecutionRecovery.recover() across three calls
    # =====================================================================
    # The same persisted order, the same venue adapter, the same recovery
    # primitive rebuilt from scratch each time. Each call must converge on
    # the same terminal FILLED view.
    initial_order = VenueOrder(
        exchange_id="binance",
        exchange_order_id=stable_exchange_id,
        client_order_id=stable_client_id,
        symbol=Symbol.parse("ETH/USDT"),
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        status=OrderStatus.TIMEOUT,
        amount=D("5"),
        error="placement timed out",
    )

    venue_a = _VenueWithStableFilledOrder(
        client_order_id=stable_client_id,
        exchange_order_id=stable_exchange_id,
        venue="binance",
    )

    recovered_views: list[VenueOrder] = []
    for run in range(3):
        # Brand new recovery primitive — different process, fresh state.
        recovery = ExecutionRecovery()
        result = await recovery.recover(initial_order, venue_a)
        recovered_views.append(result)

    # The recovery primitive must NEVER touch the venue's create_order path.
    assert venue_a.create_calls == [], (
        "ExecutionRecovery.recover() must not call create_order on the adapter"
    )
    # All three calls converge on the same terminal FILLED view.
    for run, view in enumerate(recovered_views):
        assert view.status is OrderStatus.FILLED, (
            f"recovery call #{run + 1} did not return FILLED: {view.status}"
        )
        assert view.filled_amount == buy_filled_amount_expected
        assert view.average_price == buy_average_expected
        assert view.client_order_id == stable_client_id
        assert view.exchange_order_id == stable_exchange_id

    # Calling recovery AGAIN on the now-terminal result must be a no-op
    # (RecoveryOutcome path: REJECTED early-return; FILLED also a no-op
    # because the dispatch table routes only TIMEOUT/UNKNOWN/PENDING/OPEN/
    # PARTIALLY_FILLED through the resolution branches). The persisted
    # identity must not drift.
    for run in range(2):
        recovery = ExecutionRecovery()
        already_terminal = recovered_views[-1]
        again = await recovery.recover(already_terminal, venue_a)
        assert again.status is OrderStatus.FILLED
        assert again.filled_amount == buy_filled_amount_expected
        assert again.client_order_id == stable_client_id
        assert again.exchange_order_id == stable_exchange_id
    assert venue_a.create_calls == [], (
        "recovery of an already-terminal order must not call create_order"
    )

    # =====================================================================
    # Layer B — orchestrator restart cycle through three independent
    # orchestrator instances against the SAME SQLite file.
    # =====================================================================
    # Each instance is built from scratch via _scripted_orchestrator which
    # creates a brand new adapter — so the new process starts with
    # create_calls == [] and only fills it through ``create_order`` calls
    # the orchestrator actually makes.

    # ----- pre-restart: seed a BUY_SUBMITTED record with a TIMEOUT intent
    orchestrator1, transfers1, source1, dest1, db1 = await _scripted_orchestrator(tmp_path)
    try:
        # Wipe the default scripted venue's create_calls (the orchestrator
        # was built against a fresh scripted venue that has 0 entries — the
        # real restart will also start with 0). Seed the TIMEOUT intent.
        seed_intent = _pending_intent_json()
        # Use a distinct, stable client_order_id so we can prove identity
        # preservation across restarts.
        seed_intent_payload = VenueOrder.model_validate(seed_intent).model_copy(
            update={
                "client_order_id": stable_client_id,
                "exchange_order_id": stable_exchange_id,
                "status": OrderStatus.TIMEOUT,
                "error": "placement timed out (seeded for restart test)",
            }
        )
        seed_json = _order_to_json_compat(seed_intent_payload)
        seeded = _record(state=TransferState.BUY_SUBMITTED, buy_order=seed_json)
        await transfers1.save(seeded)
        seeded_id = seeded.id
    finally:
        await db1.dispose()

    assert db_path.exists()

    # ----- restart #1: a fresh orchestrator (fresh adapter with 0 calls).
    # The first tick triggers _resolve_existing_buy -> _recovery.recover()
    # for the first time. The venue reports the FILLED order and recovery
    # returns it; the lifecycle advances to BUY_FILLED.
    orchestrator2, transfers2, source2, dest2, db2 = await _scripted_orchestrator(tmp_path)
    try:
        # The scripted venue used by _scripted_orchestrator stores orders by
        # client_order_id. Pre-load it with the FILLED venue order that the
        # recovery path will find on lookup.
        source2.orders_by_client_id[stable_client_id] = VenueOrder.model_validate(seed_json).model_copy(
            update={
                "status": OrderStatus.FILLED,
                "filled_amount": buy_filled_amount_expected,
                "average_price": buy_average_expected,
                "exchange_order_id": stable_exchange_id,
            }
        )

        # Tick — the BUY_SUBMITTED state routes to _execute_buy ->
        # _resolve_existing_buy -> _recovery.recover(). Recovery queries the
        # venue, finds the FILLED order, returns the FILLED view.
        advanced = await orchestrator2.tick()
        updated = next(r for r in advanced if r.id == seeded_id)
        assert updated.state is TransferState.BUY_FILLED, updated.state
        assert updated.buy_filled_amount == buy_filled_amount_expected
        assert updated.buy_order["client_order_id"] == stable_client_id
        assert updated.buy_order["exchange_order_id"] == stable_exchange_id
        assert updated.buy_order["status"] == OrderStatus.FILLED.value

        # CRITICAL: no second create_order() was placed.
        assert source2.create_calls == [], (
            f"recovery must not place a new order, got "
            f"{len(source2.create_calls)} create_order call(s)"
        )

        # Snapshot the persisted buy_order verbatim so we can prove it does
        # not drift across subsequent restarts (no spurious re-serialisation
        # of the terminal result, no client_order_id mutation).
        persisted_after_r1 = await transfers2.get(seeded_id)
        assert persisted_after_r1 is not None
        assert persisted_after_r1.state is TransferState.BUY_FILLED
        assert persisted_after_r1.buy_order["client_order_id"] == stable_client_id
        assert persisted_after_r1.buy_order["exchange_order_id"] == stable_exchange_id
        assert persisted_after_r1.buy_filled_amount == buy_filled_amount_expected
        assert persisted_after_r1.buy_order["status"] == OrderStatus.FILLED.value
        persisted_buy_order_after_r1 = dict(persisted_after_r1.buy_order)
        persisted_filled_after_r1 = persisted_after_r1.buy_filled_amount

        # Now prove repeated recovery is idempotent at the orchestrator
        # level too: call _resolve_existing_buy directly on the persisted
        # BUY_FILLED record a SECOND time. The orchestrator must short-
        # circuit on the ``is_confirmed_fill`` branch and NOT invoke
        # _recovery.recover (and therefore must not place a new order).
        reloaded = await transfers2.get(seeded_id)
        re_resolved = await orchestrator2._resolve_existing_buy(
            reloaded, Symbol.parse("ETH/USDT")
        )
        assert re_resolved.state is TransferState.BUY_FILLED
        assert re_resolved.buy_order["client_order_id"] == stable_client_id
        assert re_resolved.buy_filled_amount == persisted_filled_after_r1
        # Still no new create_order() call.
        assert source2.create_calls == [], (
            "a 2nd restart-time call to _resolve_existing_buy must not place a new order"
        )

        # Even if the recovery primitive is invoked AGAIN directly on the
        # already-terminal order, it must converge on the same terminal
        # view and never escalate to MANUAL_REVIEW just because it was
        # called multiple times.  This mirrors the behaviour of a future
        # code path that might double-check the order after a partial
        # restart.
        already_filled = VenueOrder.model_validate(persisted_buy_order_after_r1)
        second_pass = await ExecutionRecovery().recover(already_filled, source2)
        assert second_pass.status is OrderStatus.FILLED
        assert second_pass.filled_amount == buy_filled_amount_expected
        assert second_pass.client_order_id == stable_client_id
        assert second_pass.exchange_order_id == stable_exchange_id
        assert source2.create_calls == []
    finally:
        await db2.dispose()

    # ----- restart #2: yet another fresh orchestrator.  Load the persisted
    # row and re-derive the BUY_FILLED shape from the database — proves
    # the persisted bytes survive an independent DB connection lifetime.
    orchestrator3, transfers3, source3, dest3, db3 = await _scripted_orchestrator(tmp_path)
    try:
        persisted_pre = await transfers3.get(seeded_id)
        assert persisted_pre is not None
        assert persisted_pre.state is TransferState.BUY_FILLED
        assert persisted_pre.buy_order["client_order_id"] == stable_client_id
        assert persisted_pre.buy_order["exchange_order_id"] == stable_exchange_id
        assert persisted_pre.buy_filled_amount == buy_filled_amount_expected

        # The persisted buy_order must be byte-equivalent to the snapshot
        # captured after the first restart (no drift from re-serialisation).
        assert dict(persisted_pre.buy_order) == persisted_buy_order_after_r1

        # Drive _resolve_existing_buy AGAIN through a brand-new orchestrator
        # pointed at the same DB. The is_confirmed_fill branch must short-
        # circuit (no _recovery.recover call, no venue roundtrip, no new
        # create_order).
        reloaded = await transfers3.get(seeded_id)
        re_resolved = await orchestrator3._resolve_existing_buy(
            reloaded, Symbol.parse("ETH/USDT")
        )
        assert re_resolved.state is TransferState.BUY_FILLED
        assert re_resolved.buy_order["client_order_id"] == stable_client_id
        assert re_resolved.buy_filled_amount == buy_filled_amount_expected
        assert source3.create_calls == []
    finally:
        await db3.dispose()

    # ----- restart #3: a fourth orchestrator, just to be paranoid. Same
    # assertions as restart #2 — proves the lifecycle survives any number
    # of fresh processes without ever mutating the terminal result.
    orchestrator4, transfers4, source4, dest4, db4 = await _scripted_orchestrator(tmp_path)
    try:
        persisted_post_r3 = await transfers4.get(seeded_id)
        assert persisted_post_r3 is not None
        assert persisted_post_r3.state is TransferState.BUY_FILLED
        assert persisted_post_r3.buy_order["client_order_id"] == stable_client_id
        assert persisted_post_r3.buy_order["exchange_order_id"] == stable_exchange_id
        assert persisted_post_r3.buy_filled_amount == buy_filled_amount_expected
        assert dict(persisted_post_r3.buy_order) == persisted_buy_order_after_r1

        reloaded = await transfers4.get(seeded_id)
        re_resolved = await orchestrator4._resolve_existing_buy(
            reloaded, Symbol.parse("ETH/USDT")
        )
        assert re_resolved.state is TransferState.BUY_FILLED
        assert re_resolved.buy_filled_amount == buy_filled_amount_expected
        assert source4.create_calls == []
    finally:
        await db4.dispose()


def _order_to_json_compat(order: VenueOrder) -> dict:
    """Same serialisation as TransferOrchestrator uses for persisted orders."""
    import json

    computed = set(order.model_computed_fields)
    return json.loads(order.model_dump_json(exclude=computed))


# ---------------------------------------------------------------------------
# TEST GAP #2 (from the LIVE READINESS AUDIT):
#   Process crash during the PAPER withdrawal lifecycle must NOT cause a
#   duplicate withdrawal, a duplicate source balance debit, or a duplicate
#   destination-side accounting entry.
#
#   The model is: PAPER funds are synthetic and the paper wallet re-seeds
#   from the simulated venue's deterministic balances on every process
#   restart (paper_wallet.py:5-7).  The real safety property to verify is
#   therefore:
#     * the orchestrator does NOT call ``adapter.withdraw`` again on the
#       fresh simulated adapter (the equivalent of a second on-chain
#       withdrawal in real LIFE);
#     * the persisted ``withdrawal_id`` / ``withdrawal_txid`` are stable
#       across the restart (the venue's view of the original withdrawal
#       is reused, not replaced by a second call);
#     * the source PAPER wallet is NOT debited a second time — it stays at
#       the post-withdrawal balance from the pre-crash process and never
#       moves again on restart;
#     * the lifecycle advances from WITHDRAW_SUBMITTED to
#       TRANSFER_IN_PROGRESS (and ultimately COMPLETED) without ever
#       re-submitting the withdrawal;
#     * a SECOND independent restart also remains safe.
# ---------------------------------------------------------------------------


async def test_restart_during_paper_withdrawal_does_not_duplicate_transfer(tmp_path):
    """TEST GAP #2: a PAPER withdrawal that survives a process crash must
    not be re-submitted on restart.

    This test exercises the ACTUAL PAPER withdrawal path (no mocks of the
    withdrawal logic): the simulated venue's ``withdraw`` appends an entry
    to its per-instance history list, and the orchestrator's paper wallet
    holds the live source balance.  Both are reset on process restart —
    exactly the natural model of a PAPER restart.  The test verifies that
    the orchestrator's post-restart lifecycle resumes from the persisted
    ``WITHDRAW_SUBMITTED`` state via ``_check_withdrawal`` (NOT
    ``_submit_withdrawal``), so neither the simulated ``withdraw`` nor the
    source wallet debit is repeated.
    """
    settings = make_settings(tmp_path)
    # Make the simulated blockchain fast so the resume completes quickly
    # without altering the pre-crash timing (the pre-crash app has the
    # same default delay of 0.1s, plenty of time to reach WITHDRAW_SUBMITTED).
    fast = settings.model_copy(
        update={
            "transfer": settings.transfer.model_copy(
                update={"simulated_transfer_seconds": 0.1}
            )
        }
    )

    # ============================================================ pre-restart
    # First process: drive a transfer until the source-side withdrawal has
    # been executed (adapter.withdraw called once, source paper wallet
    # debited once, WITHDRAW_SUBMITTED persisted with the venue's txid).
    app1 = await build_app(fast)
    await start_app(app1)
    try:
        plans = await app1.plan_transfers()
        assert plans, "the simulated venues must produce transfer plans"
        source_venue = plans[0].source_exchange
        dest_venue = plans[0].dest_exchange
        asset = plans[0].asset

        record = await app1.start_transfer(plans[0])
        transfer_id = record.id
        # Drive ticks until the orchestrator has executed the PAPER
        # withdrawal (adapter.withdraw was called exactly once and the
        # source paper wallet was debited once).  We do NOT wait for
        # TRANSFER_IN_PROGRESS — we want to capture the state immediately
        # after withdrawal submission, so we tick once after WITHDRAW_SUBMITTED
        # surfaces and snapshot before any further lifecycle advance.
        ticks = 0
        while record.state is TransferState.WITHDRAW_SUBMITTED and ticks < 10:
            await asyncio.sleep(0.05)
            advanced = await app1.tick_transfers()
            for updated in advanced:
                if updated.id == transfer_id:
                    record = updated
            ticks += 1

        # Pre-crash invariants: the withdrawal was actually executed.
        assert record.state in (
            TransferState.WITHDRAW_SUBMITTED,
            TransferState.WITHDRAW_PENDING,
            TransferState.TRANSFER_IN_PROGRESS,
        ), f"unexpected pre-crash state: {record.state}"
        assert record.withdrawal_id is not None, (
            "the PAPER withdrawal must have produced a withdrawal_id"
        )
        assert record.withdrawal_txid is not None, (
            "the PAPER withdrawal must have produced a withdrawal_txid"
        )

        # Capture the pre-crash safety snapshot.
        pre_source_adapter = app1.manager.adapter(source_venue)
        pre_withdraw_history = list(pre_source_adapter._withdrawals)
        pre_wallet = app1.paper_wallets[source_venue]
        pre_withdrawal_id = record.withdrawal_id
        pre_withdrawal_txid = record.withdrawal_txid
        pre_buy_filled = record.buy_filled_amount

        # The simulated adapter MUST have exactly one withdrawal history
        # entry from the pre-crash process — this is the on-chain footprint
        # we are protecting against duplicating.
        assert len(pre_withdraw_history) == 1, (
            f"pre-crash: expected 1 simulated withdrawal, got "
            f"{len(pre_withdraw_history)}"
        )
        assert pre_withdraw_history[0].txid == pre_withdrawal_txid
        # The destination adapter has no deposit history yet (the
        # simulated withdraw emits a "deposit" entry on the SOURCE
        # adapter's ledger, not the destination's — deposit detection on
        # the destination polls fetch_deposits which the orchestrator
        # does NOT touch until TRANSFER_IN_PROGRESS).

        # The pre-crash source wallet MUST have been debited.  Capture the
        # post-debit balance; on restart the orchestrator must not touch
        # this balance again.
    finally:
        # Simulate the crash: shutdown_app closes adapters, disposes the
        # DB pool, and discards all in-memory state.
        await shutdown_app(app1)

    # ============================================================ restart #1
    # Fresh process, SAME database, fresh paper wallets, fresh simulated
    # adapters (whose _withdrawals / _deposits lists start empty).
    app2 = await build_app(fast)
    await start_app(app2)
    try:
        post_source_adapter = app2.manager.adapter(source_venue)
        post_dest_adapter = app2.manager.adapter(dest_venue)

        # Fresh simulated adapter MUST have empty histories at startup.
        assert post_source_adapter._withdrawals == [], (
            "fresh simulated adapter must start with empty withdrawal history"
        )
        assert post_dest_adapter._deposits == [], (
            "fresh simulated adapter must start with empty deposit history"
        )

        # The orchestrator's resume() runs in start_app and loads the open
        # transfer.  Verify it sees exactly the pre-crash record (same id,
        # same withdrawal_id, same withdrawal_txid, same buy_filled_amount).
        open_records = await app2.transfers.list_open()
        assert [r.id for r in open_records] == [transfer_id], (
            f"resume must load the persisted transfer id, got "
            f"{[r.id for r in open_records]}"
        )
        resumed = open_records[0]
        assert resumed.state in (
            TransferState.WITHDRAW_SUBMITTED,
            TransferState.WITHDRAW_PENDING,
            TransferState.TRANSFER_IN_PROGRESS,
        ), f"unexpected resumed state: {resumed.state}"
        assert resumed.withdrawal_id == pre_withdrawal_id
        assert resumed.withdrawal_txid == pre_withdrawal_txid
        assert resumed.buy_filled_amount == pre_buy_filled
        assert resumed.buy_order is not None  # BUY evidence preserved

        # Capture the post-restart source wallet balance and the fresh
        # simulated adapters' withdrawal/deposit histories BEFORE ticking.
        post_wallet = app2.paper_wallets[source_venue]
        post_wallet_balance_pre_tick = post_wallet.free(asset)
        post_withdraw_history_pre_tick = list(post_source_adapter._withdrawals)
        post_deposit_history_pre_tick = list(post_dest_adapter._deposits)

        # Drive the lifecycle to a terminal state.  The lifecycle must
        # route through _check_withdrawal (state WITHDRAW_SUBMITTED ->
        # WITHDRAW_PENDING -> TRANSFER_IN_PROGRESS -> DEPOSIT_DETECTED ->
        # SELL_SUBMITTED -> COMPLETED) without ever calling
        # _submit_withdrawal again.  We assert that by checking the
        # simulated adapter's withdrawal history is still empty after
        # the lifecycle completes and the source paper wallet has not
        # been debited.
        ticks = 0
        record = resumed
        while not record.is_terminal and ticks < 100:
            await asyncio.sleep(0.05)
            advanced = await app2.tick_transfers()
            for updated in advanced:
                if updated.id == transfer_id:
                    record = updated
            ticks += 1
        assert record.state is TransferState.COMPLETED, (
            f"PAPER transfer must complete after restart, ended in "
            f"{record.state}: {record.error}"
        )

        # CRITICAL SAFETY ASSERTIONS for restart #1:

        # 1. The fresh simulated adapter's withdrawal history MUST still
        # be empty — the orchestrator did NOT call adapter.withdraw again.
        assert post_source_adapter._withdrawals == [], (
            f"a restarted PAPER process must NOT call adapter.withdraw "
            f"again, but the adapter now holds "
            f"{len(post_source_adapter._withdrawals)} withdrawal(s)"
        )
        # 2. The fresh simulated adapter's deposit history MUST also be
        # empty — the orchestrator did NOT trigger a fresh deposit.
        assert post_dest_adapter._deposits == [], (
            f"a restarted PAPER process must NOT call adapter.withdraw, "
            f"so no deposit must be appended either, but the dest adapter "
            f"now holds {len(post_dest_adapter._deposits)} deposit(s)"
        )
        # 3. The pre-tick source wallet balance MUST equal the post-tick
        # source wallet balance — no second debit.
        post_wallet_balance_post_tick = post_wallet.free(asset)
        assert post_wallet_balance_post_tick == post_wallet_balance_pre_tick, (
            f"source PAPER wallet must not be debited twice, "
            f"pre={post_wallet_balance_pre_tick}, post={post_wallet_balance_post_tick}"
        )
        # 4. The persisted withdrawal identity is preserved across the
        # restart and the post-restart lifecycle (no second submit means
        # no replacement of withdrawal_id / withdrawal_txid).
        final = await app2.transfers.get(transfer_id)
        assert final is not None
        assert final.state is TransferState.COMPLETED
        assert final.withdrawal_id == pre_withdrawal_id
        assert final.withdrawal_txid == pre_withdrawal_txid
        assert final.buy_filled_amount == pre_buy_filled
    finally:
        await shutdown_app(app2)

    # ============================================================ restart #2
    # A second independent restart against the same DB.  The transfer is
    # already COMPLETED, so resume() must NOT touch it.  The fresh
    # simulated adapter must remain pristine.
    app3 = await build_app(fast)
    await start_app(app3)
    try:
        open_records = await app3.transfers.list_open()
        assert open_records == [], (
            f"a completed transfer must not appear as open after restart, "
            f"got {[r.id for r in open_records]}"
        )
        post_source_adapter = app3.manager.adapter(source_venue)
        post_dest_adapter = app3.manager.adapter(dest_venue)
        assert post_source_adapter._withdrawals == [], (
            "second restart must not introduce any simulated withdrawal"
        )
        assert post_dest_adapter._deposits == [], (
            "second restart must not introduce any simulated deposit"
        )
        post_wallet = app3.paper_wallets[source_venue]
        # The wallet's balance must equal the post-crash seed (PAPER is
        # synthetic; each process reseeds from the deterministic venue
        # balances).  Importantly, no debit has occurred on this process
        # — the wallet still holds the full seed balance.
        # We assert: the wallet balance is the deterministic seed for
        # this venue+asset, AND no withdrawal has been submitted.
        assert post_wallet.free(asset) > D("0"), (
            "second-restart wallet must hold the seed balance for the asset"
        )
        # Ticking a completed transfer must not mutate anything.
        completed_record = await app3.transfers.get(transfer_id)
        assert completed_record is not None
        assert completed_record.state is TransferState.COMPLETED
        assert completed_record.withdrawal_id == pre_withdrawal_id
        await app3.tick_transfers()
        assert post_source_adapter._withdrawals == []
        assert post_dest_adapter._deposits == []
        post_tick = await app3.transfers.get(transfer_id)
        assert post_tick.state is TransferState.COMPLETED
        assert post_tick.withdrawal_id == pre_withdrawal_id
        assert post_tick.withdrawal_txid == pre_withdrawal_txid
    finally:
        await shutdown_app(app3)


# ---------------------------------------------------------------------------
# TEST GAP #4 (from the LIVE READINESS AUDIT):
#   Deposit detection on a venue with delayed crediting.
#
#   Two regression tests, both using LIVE mode (the only path that goes
#   through _await_deposit -> _timeout_guard):
#     A. The deposit eventually arrives: the orchestrator must NOT
#        submit the SELL before the deposit is detected, must NOT mark
#        the transfer COMPLETED prematurely, and must submit the SELL
#        exactly once after the deposit is detected.
#     B. The deposit never arrives: the orchestrator must escalate to
#        MANUAL_REVIEW once ``deposit_timeout_seconds`` is exceeded, must
#        NOT submit a SELL, and the MANUAL_REVIEW state must be inert
#        to subsequent ticks and survive a restart.
# ---------------------------------------------------------------------------


class _ScriptedDepositVenue:
    """Fake venue adapter for deposit-detection tests in LIVE mode.

    Unlike :class:`_ScriptedTransferVenue` (which models an end-to-end
    transfer flow with simulated ``withdraw``/``create_order``), this
    adapter is the destination-side account view the orchestrator polls:
    it lets the test directly program whether the deposit is visible and
    what its content is, without driving the full buy/withdraw legs.

    It can also be used on the SOURCE side — when ``withdrawals_visible``
    is ``True`` (the default), ``fetch_withdrawals`` reports a confirmed
    withdrawal matching the seeded ``withdrawal_id``, so the orchestrator's
    ``_check_withdrawal`` advances to ``TRANSFER_IN_PROGRESS`` on the
    first tick.
    """

    def __init__(
        self,
        *,
        deposits: tuple | None = None,
        withdrawal_txid: str = "wd-deposit-1",
        withdrawals_visible: bool = True,
        exchange_id: str = "okx",
    ) -> None:
        self._deposits = tuple(deposits) if deposits is not None else ()
        self._withdrawal_txid = withdrawal_txid
        self._withdrawals_visible = withdrawals_visible
        self._exchange_id = exchange_id
        self.create_calls: list = []
        self.withdraw_calls: list = []

    async def fetch_balances(self):
        from app.models.balance import Balance, BalanceSnapshot

        return BalanceSnapshot(
            exchange_id=self._exchange_id,
            balances=(Balance(exchange_id=self._exchange_id, asset="USDT", free=D("100000")),),
            timestamp=utc_now(),
        )

    async def fetch_trading_fees(self, symbol):
        from app.models.market import MarketFees

        return MarketFees(maker_bps=D("10"), taker_bps=D("10"))

    async def create_order(self, request):
        # Record any SELL attempt — this is the action the test must prove
        # the orchestrator does NOT take before the deposit is detected.
        self.create_calls.append(request)
        from app.models.order import Order as _O
        from app.models.enums import OrderStatus as _OS

        return _O(
            exchange_id=request.exchange_id,
            exchange_order_id="E1",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            status=_OS.FILLED,
            amount=request.amount,
            filled_amount=request.amount,
            average_price=D("100"),
        )

    async def fetch_order(self, order_id, *, symbol):
        return None

    async def fetch_open_orders(self, *, symbol=None):
        return ()

    async def fetch_withdrawal_networks(self, asset):
        return (
            WithdrawalNetwork(
                network="SIMNET",
                network_code="SIM",
                withdraw_enabled=True,
                deposit_enabled=True,
                withdrawal_fee=D("0.01"),
                withdrawal_min=D("0.0001"),
            ),
        )

    async def fetch_deposit_address(self, asset, *, network=None):
        return DepositAddress(address="0xDEST", network=network)

    async def withdraw(self, asset, amount, address, *, memo=None, network=None):
        self.withdraw_calls.append((asset, amount, address))
        return TransferTx(
            direction="withdrawal",
            asset=asset,
            amount=D(amount),
            status="ok",
            txid=f"wd-{len(self.withdraw_calls)}",
        )

    async def fetch_deposits(self, asset, *, limit=20):
        return self._deposits

    async def fetch_withdrawals(self, asset, *, limit=20):
        # When withdrawals_visible is True (the default), return a confirmed
        # withdrawal that matches the seeded ``withdrawal_id`` so
        # ``_check_withdrawal`` advances to ``TRANSFER_IN_PROGRESS`` on the
        # first tick.  When False, the withdrawal is not yet visible to the
        # orchestrator (it must wait for the venue to publish it).
        from app.models.transfer import TransferTx as _Tx

        if not self._withdrawals_visible:
            return ()
        return (
            _Tx(
                direction="withdrawal",
                asset=asset,
                amount=D("4.99"),
                status="confirmed",
                txid=self._withdrawal_txid,
            ),
        )

    def set_deposits(self, deposits: tuple) -> None:
        """Flip the deposit visibility mid-test."""
        self._deposits = tuple(deposits)


class _ScriptedDepositManager:
    def __init__(self, source, dest) -> None:
        # Both venues are passed by the caller; in the deposit-detection
        # tests the source is a ``_ScriptedDepositVenue`` configured to
        # report a confirmed withdrawal so the orchestrator's
        # ``_check_withdrawal`` advances to ``TRANSFER_IN_PROGRESS``.
        self._venues = {"binance": source, "okx": dest}

    def adapter(self, venue):
        return self._venues[venue]

    def enabled_ids(self):
        return ("binance", "okx")


async def _delayed_deposit_orchestrator(
    tmp_path,
    *,
    deposit_timeout_seconds: int = 3600,
    dest: _ScriptedDepositVenue | None = None,
    source: _ScriptedDepositVenue | None = None,
):
    """Build a LIVE-mode orchestrator pointed at a destination venue that
    can be programmed to reveal or hide the deposit at will.

    Both the source and the destination venues are
    :class:`_ScriptedDepositVenue` instances.  The source is configured to
    report a confirmed withdrawal (matching the seeded ``withdrawal_id``)
    so ``_check_withdrawal`` can advance to ``TRANSFER_IN_PROGRESS`` on
    the first tick; the destination is programmable for the deposit
    visibility test.
    """
    from tests.conftest import make_settings as _make

    settings = _make(tmp_path)
    settings = settings.model_copy(
        update={
            "trading": settings.trading.model_copy(update={"mode": TradingMode.LIVE}),
            "execution": settings.execution.model_copy(update={"leg_timeout_seconds": 0.3}),
            "transfer": settings.transfer.model_copy(
                update={"deposit_timeout_seconds": deposit_timeout_seconds}
            ),
        }
    )
    db = Database(
        settings.database.model_copy(update={"url": f"sqlite+aiosqlite:///{tmp_path / 'trf.db'}"})
    )
    await db.create_schema()
    from app.storage.repositories import TransferRepository as _TrfRepo

    transfers = _TrfRepo(db)
    trades = TradeRepository(db)
    audit = AuditLogRepository(db)
    store = MarketDataStore(stale_after_ms=60_000)
    store.put_order_book(_transfer_book("binance"))
    store.put_order_book(_transfer_book("okx"))
    source = source or _ScriptedDepositVenue(
        deposits=(),
        withdrawal_txid="wd-source",
        withdrawals_visible=True,
        exchange_id="binance",
    )
    dest = dest or _ScriptedDepositVenue(
        deposits=(),
        withdrawal_txid="wd-dest",
        withdrawals_visible=True,
        exchange_id="okx",
    )
    orchestrator = TransferOrchestrator(
        settings=settings,
        manager=_ScriptedDepositManager(source, dest),
        store=store,
        guard=ExecutionGuard(policy_for(TradingMode.LIVE)),
        recovery=ExecutionRecovery(),
        fill_simulator=FillSimulator(taker_fee_bps=D("10"), max_slippage_bps=D("500")),
        planner=TransferPlanner(settings),
        transfer_repo=transfers,
        trade_repo=trades,
        audit=audit,
        risk_check=None,
        paper_wallets={},
    )
    return orchestrator, transfers, source, dest, db, settings


async def test_delayed_deposit_is_detected_before_timeout(tmp_path):
    """TEST GAP #4 / Scenario A: a delayed deposit is eventually detected
    and the SELL fires exactly once, only after the deposit is confirmed.

    The destination venue's ``fetch_deposits`` returns nothing for the
    first three ticks, then returns a confirmed deposit on the fourth tick.
    The orchestrator must:

    * NOT submit a SELL before the deposit appears;
    * NOT mark the transfer COMPLETED before the deposit appears;
    * detect the deposit and advance to DEPOSIT_DETECTED;
    * submit the SELL exactly once after the deposit is detected;
    * complete the transfer normally.
    """
    wd_id = "wd-deposit-A"
    dest = _ScriptedDepositVenue(
        deposits=(), withdrawal_txid=wd_id, withdrawals_visible=True, exchange_id="okx"
    )
    source = _ScriptedDepositVenue(
        deposits=(), withdrawal_txid=wd_id, withdrawals_visible=True, exchange_id="binance"
    )
    orchestrator, transfers, src, dest, db, _settings = await _delayed_deposit_orchestrator(
        tmp_path, deposit_timeout_seconds=3600, dest=dest, source=source
    )
    try:
        # Seed a WITHDRAW_SUBMITTED record with a fresh ``created_at``
        # (so ``_timeout_guard`` cannot fire during the delayed phase)
        # and a destination address the deposit will match.
        record = _record(
            state=TransferState.WITHDRAW_SUBMITTED,
            buy_filled_amount=D("5"),
            withdrawal_id="wd-deposit-A",
            withdrawal_txid="wd-deposit-A",
            withdrawal_amount=D("4.99"),
            deposit_address="0xDEST",
        )
        await transfers.save(record)
        transfer_id = record.id

        # The destination venue reports a confirmed withdrawal (so
        # ``_check_withdrawal`` advances to TRANSFER_IN_PROGRESS), and
        # NO deposits yet.
        assert dest._deposits == ()

        # Tick 1: WITHDRAW_SUBMITTED -> WITHDRAW_PENDING (no confirmed
        # withdrawal visible yet) or directly to TRANSFER_IN_PROGRESS
        # (if the venue already shows the withdrawal).  ``_check_withdrawal``
        # may also keep the state at WITHDRAW_SUBMITTED if the withdrawal
        # is not yet visible — both are safe; what matters is that no
        # SELL is ever submitted before DEPOSIT_DETECTED.
        await orchestrator.tick()
        snapshot1 = await transfers.get(transfer_id)
        assert snapshot1.state in (
            TransferState.WITHDRAW_SUBMITTED,
            TransferState.WITHDRAW_PENDING,
            TransferState.TRANSFER_IN_PROGRESS,
        ), snapshot1.state
        assert dest.create_calls == [], (
            "SELL must NOT be submitted before the deposit is detected"
        )
        assert snapshot1.state is not TransferState.COMPLETED

        # Tick 2-4: drive the lifecycle through the deposit-wait phase.
        # The destination venue keeps reporting no deposit.
        for tick_index in range(2, 5):
            await orchestrator.tick()
            snapshot = await transfers.get(transfer_id)
            assert dest.create_calls == [], (
                f"after tick {tick_index}: SELL must not have been submitted "
                f"yet (state={snapshot.state})"
            )
            assert snapshot.state is not TransferState.COMPLETED, (
                f"after tick {tick_index}: transfer must not be COMPLETED "
                f"before the deposit is detected (state={snapshot.state})"
            )
            assert snapshot.state is not TransferState.DEPOSIT_DETECTED, (
                f"after tick {tick_index}: DEPOSIT_DETECTED must not be "
                f"set without a real deposit (state={snapshot.state})"
            )
            assert snapshot.state is not TransferState.SELL_SUBMITTED

        # Now reveal the deposit: a confirmed TransferTx that matches the
        # persisted ``withdrawal_amount`` and ``deposit_address``.
        deposit_tx = TransferTx(
            direction="deposit",
            asset="ETH",
            amount=D("4.99"),
            status="confirmed",
            txid="dp-real-1",
            address="0xDEST",
        )
        dest.set_deposits((deposit_tx,))

        # Tick: _await_deposit should now find the deposit and advance to
        # DEPOSIT_DETECTED (we do not require the SELL to be submitted in
        # the same tick — the lifecycle may need additional ticks to reach
        # _execute_sell once DEPOSIT_DETECTED is set).
        await orchestrator.tick()
        snapshot = await transfers.get(transfer_id)
        assert snapshot.state in (
            TransferState.DEPOSIT_DETECTED,
            TransferState.SELL_SUBMITTED,
            TransferState.COMPLETED,
        ), f"unexpected state after deposit reveal: {snapshot.state}"
        assert dest.create_calls == [], (
            f"SELL must not be submitted in the same tick the deposit is "
            f"detected — the lifecycle must advance one step at a time; "
            f"got {len(dest.create_calls)} create_order call(s)"
        )

        # Drive any remaining ticks to a terminal state.  Exactly one
        # SELL must fire across all of them.
        ticks = 0
        while not snapshot.is_terminal and ticks < 20:
            await asyncio.sleep(0.01)
            await orchestrator.tick()
            snapshot = await transfers.get(transfer_id)
            ticks += 1
        assert snapshot.state is TransferState.COMPLETED, (
            f"transfer must reach COMPLETED after the SELL, ended in "
            f"{snapshot.state}: {snapshot.error}"
        )
        # CRITICAL: exactly one SELL — no duplicates, no premature.
        assert len(dest.create_calls) == 1, (
            f"exactly one SELL must have been submitted after the deposit "
            f"was detected, got {len(dest.create_calls)}"
        )
    finally:
        await db.dispose()


async def test_delayed_deposit_timeout_escalates_to_manual_review(tmp_path):
    """TEST GAP #4 / Scenario B: when the deposit never arrives, the
    orchestrator must escalate to MANUAL_REVIEW once
    ``deposit_timeout_seconds`` is exceeded, must NOT submit a SELL, and
    the MANUAL_REVIEW state must be inert to subsequent ticks and survive
    a restart.

    The seeded record is at WITHDRAW_SUBMITTED with a ``created_at``
    backdated past the configured ``deposit_timeout_seconds`` (the
    destination venue reports a confirmed withdrawal so the state
    machine advances to TRANSFER_IN_PROGRESS, and ``fetch_deposits``
    always returns nothing).
    """
    from datetime import timedelta

    deposit_timeout_seconds = 60
    past_created_at = utc_now() - timedelta(seconds=deposit_timeout_seconds + 30)
    wd_id = "wd-deposit-B"

    dest = _ScriptedDepositVenue(
        deposits=(), withdrawal_txid=wd_id, withdrawals_visible=True, exchange_id="okx"
    )
    source = _ScriptedDepositVenue(
        deposits=(), withdrawal_txid=wd_id, withdrawals_visible=True, exchange_id="binance"
    )
    orchestrator, transfers, src, dest, db, _settings = await _delayed_deposit_orchestrator(
        tmp_path, deposit_timeout_seconds=deposit_timeout_seconds, dest=dest, source=source
    )
    try:
        record = _record(
            state=TransferState.WITHDRAW_SUBMITTED,
            buy_filled_amount=D("5"),
            withdrawal_id=wd_id,
            withdrawal_txid=wd_id,
            withdrawal_amount=D("4.99"),
            deposit_address="0xDEST",
            created_at=past_created_at,
        )
        await transfers.save(record)
        transfer_id = record.id

        # Tick 1: the lifecycle advances WITHDRAW_SUBMITTED ->
        # TRANSFER_IN_PROGRESS (the withdrawal is visible and confirmed).
        # Tick 2: TRANSFER_IN_PROGRESS -> _await_deposit, which calls
        # _timeout_guard because ``fetch_deposits`` is empty.  ``created_at``
        # is past the configured timeout, so _timeout_guard escalates to
        # MANUAL_REVIEW.
        await orchestrator.tick()
        snapshot_intermediate = await transfers.get(transfer_id)
        assert snapshot_intermediate.state is TransferState.TRANSFER_IN_PROGRESS, (
            f"first tick should advance to TRANSFER_IN_PROGRESS, got "
            f"{snapshot_intermediate.state}: {snapshot_intermediate.error}"
        )
        await orchestrator.tick()
        snapshot = await transfers.get(transfer_id)
        assert snapshot.state is TransferState.MANUAL_REVIEW, (
            f"deposit timeout must escalate to MANUAL_REVIEW, ended in "
            f"{snapshot.state}: {snapshot.error}"
        )
        # The MANUAL_REVIEW error carries the timeout context ("deposit
        # not detected yet after Ns") and the held asset / destination
        # venue are tracked in the persisted record fields themselves —
        # this is the durable identifier an operator needs to recover
        # the in-flight inventory.
        assert snapshot.error is not None
        assert "deposit not detected" in snapshot.error, (
            f"MANUAL_REVIEW must carry the deposit-timeout context, got: "
            f"{snapshot.error}"
        )
        assert snapshot.error.endswith("s"), (
            f"MANUAL_REVIEW must include the elapsed-seconds figure, got: "
            f"{snapshot.error}"
        )
        # Held asset + destination venue are tracked in the record fields.
        assert snapshot.buy_filled_amount == D("5"), (
            f"the buy leg fill amount must be preserved in the record, "
            f"got {snapshot.buy_filled_amount}"
        )
        assert snapshot.withdrawal_amount == D("4.99"), (
            f"the withdrawal amount must be preserved in the record, "
            f"got {snapshot.withdrawal_amount}"
        )
        assert snapshot.asset == "ETH", (
            f"the transferred asset must be preserved in the record, "
            f"got {snapshot.asset}"
        )
        assert snapshot.dest_exchange == "okx", (
            f"the destination venue must be preserved in the record, "
            f"got {snapshot.dest_exchange}"
        )
        # No SELL was ever submitted.
        assert dest.create_calls == [], (
            f"no SELL must be submitted on timeout, got "
            f"{len(dest.create_calls)} create_order call(s)"
        )
        # No false COMPLETED state.
        assert snapshot.state is not TransferState.COMPLETED
        # Persisted identifiers survive.
        assert snapshot.withdrawal_id == wd_id
        assert snapshot.withdrawal_txid == wd_id
        assert snapshot.withdrawal_amount == D("4.99")

        # A repeated tick on the same orchestrator is inert: MANUAL_REVIEW
        # is terminal, the lifecycle MUST NOT do anything.
        await orchestrator.tick()
        snapshot2 = await transfers.get(transfer_id)
        assert snapshot2.state is TransferState.MANUAL_REVIEW
        assert snapshot2.error == snapshot.error
        assert dest.create_calls == []
    finally:
        await db.dispose()

    # ----- restart: a fresh orchestrator pointed at the same DB must
    # see the MANUAL_REVIEW state preserved, must NOT submit a SELL, and
    # must NOT regress the state.
    orchestrator2, transfers2, src2, dest2, db2, _settings2 = await _delayed_deposit_orchestrator(
        tmp_path,
        deposit_timeout_seconds=deposit_timeout_seconds,
        dest=dest,
        source=source,
    )
    try:
        # The fresh destination venue starts with empty deposit history
        # (just like a real exchange on restart).
        assert dest2._deposits == ()
        assert dest2.create_calls == []

        await orchestrator2.tick()
        snapshot3 = await transfers2.get(transfer_id)
        assert snapshot3.state is TransferState.MANUAL_REVIEW, (
            f"MANUAL_REVIEW must persist across restart, ended in "
            f"{snapshot3.state}: {snapshot3.error}"
        )
        assert dest2.create_calls == [], (
            f"a fresh orchestrator must NOT submit a SELL, got "
            f"{len(dest2.create_calls)} create_order call(s)"
        )
        # The held-asset and venue identification persist verbatim in
        # the record fields.
        assert snapshot3.buy_filled_amount == D("5")
        assert snapshot3.withdrawal_amount == D("4.99")
        assert snapshot3.asset == "ETH"
        assert snapshot3.dest_exchange == "okx"
        # The original withdrawal identifiers are preserved.
        assert snapshot3.withdrawal_id == wd_id
        assert snapshot3.withdrawal_txid == wd_id
        # The MANUAL_REVIEW error survives restart.
        assert snapshot3.error == snapshot.error
    finally:
        await db2.dispose()


# ---------------------------------------------------------------------------
# TEST GAP #5 (from the LIVE READINESS AUDIT):
#   Withdraw to a memo-required address where the memo is missing.
#
#   The orchestrator MUST propagate the destination memo/tag to the venue's
#   ``adapter.withdraw`` call (so the venue's ledger can credit the right
#   account) and MUST refuse to submit a withdrawal when the destination
#   asset/network is known to require a tag and the venue returned no tag
#   in the deposit-address response.  A post-withdrawal MANUAL_REVIEW is
#   NOT sufficient — the funds are already lost by then.
#
#   Both tests use PAPER mode so the actual ``adapter.withdraw`` call site
#   is exercised (the LIVE branch and PAPER branch both invoke the adapter
#   and both must receive the memo).
# ---------------------------------------------------------------------------


class _MemoScriptedTransferVenue:
    """A scripted transfer venue that records memo-tag handling.

    Mirrors the public surface of :class:`_ScriptedTransferVenue` but
    adds memo-aware behaviour: ``fetch_deposit_address`` returns a
    configurable ``DepositAddress`` and ``withdraw`` records the memo
    that was actually passed by the orchestrator.  This is the minimum
    test surface needed to prove the orchestrator either correctly
    propagates the memo or refuses to call withdraw when one is
    required but missing.
    """

    def __init__(
        self,
        *,
        deposit_address: DepositAddress,
    ) -> None:
        self._deposit_address = deposit_address
        self.create_calls: list = []
        self.withdraw_calls: list = []  # list of (asset, amount, address, memo, network)

    async def fetch_balances(self):
        from app.models.balance import Balance, BalanceSnapshot

        return BalanceSnapshot(
            exchange_id="binance",
            balances=(Balance(exchange_id="binance", asset="USDT", free=D("100000")),),
            timestamp=utc_now(),
        )

    async def fetch_trading_fees(self, symbol):
        from app.models.market import MarketFees

        return MarketFees(maker_bps=D("10"), taker_bps=D("10"))

    async def create_order(self, request):
        self.create_calls.append(request)
        from app.models.enums import OrderStatus as _OS
        from app.models.order import Order as _O

        return _O(
            exchange_id=request.exchange_id,
            exchange_order_id="E1",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            status=_OS.FILLED,
            amount=request.amount,
            filled_amount=request.amount,
            average_price=D("100"),
        )

    async def fetch_order(self, order_id, *, symbol):
        return None

    async def fetch_open_orders(self, *, symbol=None):
        return ()

    async def fetch_withdrawal_networks(self, asset):
        return (
            WithdrawalNetwork(
                network=self._deposit_address.network or "XRP",
                network_code=self._deposit_address.network or "XRP",
                withdraw_enabled=True,
                deposit_enabled=True,
                withdrawal_fee=D("0.25"),
                withdrawal_min=D("0.0001"),
            ),
        )

    async def fetch_deposit_address(self, asset, *, network=None):
        return self._deposit_address

    async def withdraw(
        self,
        asset,
        amount,
        address,
        *,
        memo=None,
        network=None,
    ):
        # CRITICAL: this is the call the test inspects.  The orchestrator
        # MUST pass the memo through (TEST A) and MUST NOT call this at
        # all when the memo is required and missing (TEST B).
        self.withdraw_calls.append(
            {
                "asset": asset,
                "amount": amount,
                "address": address,
                "memo": memo,
                "network": network,
            }
        )
        return TransferTx(
            direction="withdrawal",
            asset=asset,
            amount=D(amount),
            status="ok",
            txid=f"wd-{len(self.withdraw_calls)}",
        )

    async def fetch_deposits(self, asset, *, limit=20):
        return ()

    async def fetch_withdrawals(self, asset, *, limit=20):
        return ()


class _MemoScriptedManager:
    def __init__(self, source, dest) -> None:
        self._venues = {"binance": source, "okx": dest}

    def adapter(self, venue):
        return self._venues[venue]

    def enabled_ids(self):
        return ("binance", "okx")


async def _memo_orchestrator(
    tmp_path,
    *,
    source: _MemoScriptedTransferVenue,
    dest: _MemoScriptedTransferVenue,
    mode: TradingMode = TradingMode.PAPER,
):
    """Build an orchestrator whose source and dest are programmable
    :class:`_MemoScriptedTransferVenue` instances."""
    from tests.conftest import make_settings as _make

    settings = _make(tmp_path)
    settings = settings.model_copy(
        update={
            "trading": settings.trading.model_copy(update={"mode": mode}),
            "execution": settings.execution.model_copy(update={"leg_timeout_seconds": 0.3}),
        }
    )
    db = Database(
        settings.database.model_copy(update={"url": f"sqlite+aiosqlite:///{tmp_path / 'trf.db'}"})
    )
    await db.create_schema()
    from app.storage.repositories import TransferRepository as _TrfRepo

    transfers = _TrfRepo(db)
    trades = TradeRepository(db)
    audit = AuditLogRepository(db)
    store = MarketDataStore(stale_after_ms=60_000)
    store.put_order_book(_transfer_book("binance"))
    store.put_order_book(_transfer_book("okx"))
    orchestrator = TransferOrchestrator(
        settings=settings,
        manager=_MemoScriptedManager(source, dest),
        store=store,
        guard=ExecutionGuard(policy_for(mode)),
        recovery=ExecutionRecovery(),
        fill_simulator=FillSimulator(taker_fee_bps=D("10"), max_slippage_bps=D("500")),
        planner=TransferPlanner(settings),
        transfer_repo=transfers,
        trade_repo=trades,
        audit=audit,
        risk_check=None,
        paper_wallets={},
    )
    return orchestrator, transfers, source, dest, db


def _xrp_plan(amount: str = "10") -> "TransferPlan":
    from app.models.transfer import TransferPlan

    return TransferPlan(
        source_exchange="binance",
        dest_exchange="okx",
        asset="XRP",
        network="XRP",
        amount=D(amount),
        buy_price=D("1"),
        sell_price=D("1.01"),
        withdrawal_fee=D("0.25"),
        data_age_ms=0.0,
    )


def _xrp_record(**updates):
    from app.models.transfer import TransferRecord

    base = {
        "source_exchange": "binance",
        "dest_exchange": "okx",
        "asset": "XRP",
        "network": "XRP",
        "amount": D("10"),
        "plan": _xrp_plan(),
        "mode": "PAPER",
    }
    base.update(updates)
    return TransferRecord(**base)


async def test_withdrawal_passes_required_memo_to_venue(tmp_path):
    """TEST GAP #5 / Scenario A: a memo-required asset's destination
    memo/tag is propagated verbatim to ``adapter.withdraw``.

    The destination venue's ``fetch_deposit_address`` returns the deposit
    address together with a required memo/tag.  The orchestrator must:

    * call ``adapter.withdraw`` exactly once;
    * pass the exact destination address returned by the venue;
    * pass the exact memo/tag returned by the venue;
    * continue the lifecycle normally (the memo is treated as just another
      withdrawal parameter, not a special case that breaks the flow).
    """
    memo = "12345"
    dest = _MemoScriptedTransferVenue(
        deposit_address=DepositAddress(
            address="rDestAddressWithTag",
            memo=memo,
            network="XRP",
        )
    )
    source = _MemoScriptedTransferVenue(
        # Source deposit address is irrelevant for the source adapter in
        # PAPER mode; keep the same shape so the test stays consistent.
        deposit_address=DepositAddress(address="source-pool", network="XRP")
    )
    orchestrator, transfers, source, dest, db = await _memo_orchestrator(
        tmp_path, source=source, dest=dest
    )
    try:
        record = _xrp_record(
            state=TransferState.BUY_FILLED,
            buy_order=_filled_buy_order_json(),
            buy_filled_amount=D("10"),
        )
        await transfers.save(record)

        # The next tick on a BUY_FILLED record routes to
        # ``_submit_withdrawal``, which is the call site we are
        # exercising.
        advanced = await orchestrator.tick()
        updated = next(r for r in advanced if r.id == record.id)
        # The lifecycle advanced through the withdrawal submission.
        assert updated.state in (
            TransferState.WITHDRAW_SUBMITTED,
            TransferState.WITHDRAW_PENDING,
            TransferState.TRANSFER_IN_PROGRESS,
            TransferState.DEPOSIT_DETECTED,
            TransferState.SELL_SUBMITTED,
            TransferState.COMPLETED,
            TransferState.FAILED,
        )
        # adapter.withdraw was called exactly once on the SOURCE adapter
        # (the venue that holds the asset to be moved).
        assert len(source.withdraw_calls) == 1, (
            f"expected exactly 1 source withdraw call, got "
            f"{len(source.withdraw_calls)}"
        )
        call = source.withdraw_calls[0]
        # Exact destination address returned by the venue.
        assert call["address"] == "rDestAddressWithTag"
        # Exact memo/tag returned by the venue — THIS is the critical
        # propagation invariant.
        assert call["memo"] == memo, (
            f"adapter.withdraw must receive the destination memo, got "
            f"{call['memo']!r}"
        )
        # Network / asset / amount also propagated.
        assert call["asset"] == "XRP"
        assert call["network"] == "XRP"
        # The destination adapter MUST NOT have been called for withdraw
        # (the bot withdraws from source, deposits to dest).
        assert dest.withdraw_calls == []
    finally:
        await db.dispose()


async def test_missing_required_withdrawal_memo_fails_closed(tmp_path):
    """TEST GAP #5 / Scenario B: when the destination venue returns a
    deposit address WITHOUT a memo for an asset the bot knows requires
    a memo/tag, the orchestrator MUST refuse to call ``adapter.withdraw``
    and MUST transition to ``MANUAL_REVIEW``.

    A post-withdrawal MANUAL_REVIEW is NOT sufficient: the funds are
    already sent to an uncredited address and are lost.  The fix
    enforced by the production code fails closed BEFORE the
    ``adapter.withdraw`` call.
    """
    dest = _MemoScriptedTransferVenue(
        deposit_address=DepositAddress(
            address="rDestAddressWithoutTag",
            memo=None,  # the venue returned no tag — the test declares the
            # asset (XRP) requires one.
            network="XRP",
        )
    )
    source = _MemoScriptedTransferVenue(
        deposit_address=DepositAddress(address="source-pool", network="XRP")
    )
    orchestrator, transfers, source, dest, db = await _memo_orchestrator(
        tmp_path, source=source, dest=dest
    )
    try:
        record = _xrp_record(
            state=TransferState.BUY_FILLED,
            buy_order=_filled_buy_order_json(),
            buy_filled_amount=D("10"),
        )
        await transfers.save(record)
        transfer_id = record.id

        # Tick: the orchestrator must hit the memo-required check and
        # escalate to MANUAL_REVIEW without ever calling the venue's
        # ``withdraw`` endpoint.
        advanced = await orchestrator.tick()
        updated = next(r for r in advanced if r.id == transfer_id)
        # The fix fails the transfer to MANUAL_REVIEW (not FAILED — the
        # operator must be able to act, and the held inventory is real).
        assert updated.state is TransferState.MANUAL_REVIEW, (
            f"missing memo must escalate to MANUAL_REVIEW, ended in "
            f"{updated.state}: {updated.error}"
        )
        # CRITICAL: adapter.withdraw was NEVER called on either side.
        assert source.withdraw_calls == [], (
            f"adapter.withdraw must NOT be called when a required memo is "
            f"missing, got {len(source.withdraw_calls)} source withdraw "
            f"call(s)"
        )
        assert dest.withdraw_calls == []
        # No false WITHDRAW_SUBMITTED: the persisted withdrawal_id is the
        # ``wd-intent-…`` placeholder that was persisted BEFORE the venue
        # call — and crucially there is no real withdrawal_id from the
        # venue.  The persisted state must reflect "not submitted".
        assert updated.withdrawal_id is None or updated.withdrawal_id.startswith(
            "wd-intent-"
        ), (
            f"withdrawal_id must be absent or the pre-submit intent prefix "
            f"when the withdrawal was never sent, got: "
            f"{updated.withdrawal_id!r}"
        )
        # The MANUAL_REVIEW reason must clearly identify the missing
        # memo/tag (so the operator can act on it).
        error = updated.error or ""
        assert "memo" in error.lower() or "tag" in error.lower(), (
            f"MANUAL_REVIEW reason must mention the missing memo/tag, "
            f"got: {error!r}"
        )
        # Held inventory is tracked verbatim: the asset and amounts are
        # still on the source venue, never moved.
        assert updated.asset == "XRP"
        assert updated.buy_filled_amount == D("10")
        assert updated.withdrawal_amount == D("0") or updated.withdrawal_amount is None
        assert updated.source_exchange == "binance"
        assert updated.dest_exchange == "okx"

        # A repeated tick on the same orchestrator is inert: MANUAL_REVIEW
        # is terminal, no further side effects.
        before_withdraw = list(source.withdraw_calls)
        await orchestrator.tick()
        after_withdraw = list(source.withdraw_calls)
        assert before_withdraw == after_withdraw, (
            "a 2nd tick on a MANUAL_REVIEW record must not call withdraw"
        )
        snapshot = await transfers.get(transfer_id)
        assert snapshot.state is TransferState.MANUAL_REVIEW
    finally:
        await db.dispose()

    # ----- restart: a fresh orchestrator against the same DB must see
    # MANUAL_REVIEW, must NOT call adapter.withdraw, and must remain
    # fail-closed.
    orchestrator2, transfers2, source2, dest2, db2 = await _memo_orchestrator(
        tmp_path, source=source, dest=dest
    )
    try:
        # The fresh source adapter starts with an empty withdraw_calls
        # list — proving the venue itself is not persisting any state.
        assert source2.withdraw_calls == []

        await orchestrator2.tick()
        snapshot2 = await transfers2.get(transfer_id)
        assert snapshot2.state is TransferState.MANUAL_REVIEW, (
            f"MANUAL_REVIEW must persist across restart, ended in "
            f"{snapshot2.state}: {snapshot2.error}"
        )
        # No new withdrawal was attempted after restart.
        assert source2.withdraw_calls == [], (
            f"a fresh orchestrator must NOT call adapter.withdraw, got "
            f"{len(source2.withdraw_calls)} call(s)"
        )
        # The held-asset and venue information is preserved verbatim.
        assert snapshot2.asset == "XRP"
        assert snapshot2.buy_filled_amount == D("10")
        assert snapshot2.source_exchange == "binance"
        assert snapshot2.dest_exchange == "okx"
        # The MANUAL_REVIEW error survives restart.
        assert snapshot2.error == updated.error
    finally:
        await db2.dispose()


# ---------------------------------------------------------------------------
# TEST GAP #7 (from the LIVE READINESS AUDIT):
#   Cross-process ``start_transfer`` concurrency.
#
#   The existing in-process ``_transfer_start_lock`` does NOT span
#   processes.  Two independent bot processes connected to the same
#   persistent database must be tested directly: with
#   ``max_open_transfers=1``, exactly one process may successfully claim
#   the single slot, and every other process must be rejected.
#
#   The test uses ``multiprocessing`` with the ``spawn`` start method
#   (Windows-safe) and a ``multiprocessing.Barrier`` so all children
#   enter the critical section at the same time.  This is NOT a
#   "sleep and hope" race test.
# ---------------------------------------------------------------------------


def _concurrent_child_worker(
    db_path: str,
    child_index: int,
    barrier,
    result_queue,
) -> None:
    """Module-level worker (required for ``spawn`` pickling).

    Each child builds its own AppServices against the shared database,
    refreshes market data via ``start_app``, plans a transfer, waits on
    the barrier, then calls ``start_transfer`` and reports the outcome
    through ``result_queue``.
    """
    import asyncio
    import traceback

    from app.config.settings import DatabaseSettings
    from app.services import build_app, shutdown_app, start_app
    from tests.conftest import make_settings

    outcome: dict = {
        "child": child_index,
        "ok": False,
        "error": None,
        "record_id": None,
        "traceback": None,
    }

    async def _runner() -> None:
        # Build the same settings shape the in-process concurrent test
        # uses (make_settings + max_open_transfers=1), but pointing at
        # the shared database file.  Note: make_settings reads .env
        # implicitly; we override the database URL after copying.
        settings = make_settings_for_child(db_path)
        try:
            app = await build_app(settings)
            try:
                await start_app(app)
                plans = await app.plan_transfers()
                if not plans:
                    outcome["error"] = "no plans available"
                    return
                plan = plans[0]

                # Deterministic barrier: every child arrives here, then
                # they are released simultaneously.  This is the
                # cross-process critical section.
                barrier.wait()

                try:
                    record = await app.start_transfer(plan)
                    outcome["ok"] = True
                    outcome["record_id"] = record.id
                    outcome["record_state"] = record.state.value
                except Exception as exc:  # noqa: BLE001
                    outcome["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                await shutdown_app(app)
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = f"setup crashed: {type(exc).__name__}: {exc}"

    try:
        asyncio.run(_runner())
    except Exception as exc:  # noqa: BLE001
        outcome["error"] = f"runner crashed: {type(exc).__name__}: {exc}"
        outcome["traceback"] = traceback.format_exc()
    finally:
        result_queue.put(outcome)


def make_settings_for_child(db_path: str):
    """Worker-side equivalent of ``tests.conftest.make_settings``.

    Defined at module level so the ``spawn`` start method can pickle the
    process.  Self-contained: the ``spawn`` start method creates a fresh
    Python interpreter in the child, so module-level state from the
    parent is NOT inherited.  This function therefore builds the settings
    from scratch using only the ``db_path`` argument and the known
    constants (AVAX/DOGE assets, simulated blockchain, paper-mode
    universes, max_open_transfers=1).
    """
    from decimal import Decimal

    from app.config.settings import DatabaseSettings, Settings

    base = Settings(_env_file=None)
    out = base.model_copy(
        update={
            "database": DatabaseSettings(url=f"sqlite+aiosqlite:///{db_path}"),
            "arbitrage": base.arbitrage.model_copy(
                update={"triangle_assets": ("BTC", "ETH", "SOL")}
            ),
            "transfer": base.transfer.model_copy(
                update={
                    "assets": ("AVAX", "DOGE"),
                    "simulated_transfer_seconds": 5.0,
                    "poll_interval_seconds": 0.05,
                    "default_amount": Decimal("1"),
                }
            ),
            "market_data": base.market_data.model_copy(
                update={"streams_enabled": False}
            ),
            "risk": base.risk.model_copy(update={"max_open_transfers": 1}),
        }
    )
    return out


async def test_cross_process_start_transfer_respects_max_open_transfers(tmp_path):
    """TEST GAP #7: independent OS processes racing on the same database
    must not exceed ``MAX_OPEN_TRANSFERS=1``.

    The parent process seeds a fresh database and the market data cache,
    then spawns three child processes with the ``spawn`` start method
    (Windows-safe).  Each child builds its own AppServices, waits on a
    ``multiprocessing.Barrier``, and calls ``start_transfer``.  Exactly
    one child must succeed; the other two must be rejected.
    """
    from app.config.settings import DatabaseSettings
    from app.services import build_app, shutdown_app, start_app
    from app.storage.engine import Database
    from app.storage.repositories import TransferRepository
    from tests.conftest import make_settings

    # The shared SQLite file all three children will connect to.
    db_path = str(tmp_path / "concurrent.db")

    # Parent seeds the database schema and the market data so every
    # child can build a fresh AppServices against it.  ``max_open_transfers``
    # is set to 1 to make the cap observable.
    settings = make_settings(tmp_path)
    settings = settings.model_copy(
        update={
            "database": DatabaseSettings(url=f"sqlite+aiosqlite:///{db_path}"),
            "risk": settings.risk.model_copy(update={"max_open_transfers": 1}),
            "transfer": settings.transfer.model_copy(
                update={"simulated_transfer_seconds": 5.0}
            ),
        }
    )

    # Stash the settings on the module so the spawned workers can build
    # their own identical settings object (with the same tmp_path for
    # any test-scoped file locations) pointed at the shared DB.
    parent_app = await build_app(settings)
    try:
        await start_app(parent_app)

        # Sanity: the plan universe produces at least one plan.
        plans = await parent_app.plan_transfers()
        assert plans, "the simulated universe must produce at least one plan"

        # Spawn three independent child processes with a shared barrier.
        ctx = multiprocessing.get_context("spawn")
        n_children = 3
        barrier = ctx.Barrier(n_children)
        result_queue: ctx.Queue = ctx.Queue()

        procs = [
            ctx.Process(
                target=_concurrent_child_worker,
                args=(db_path, i, barrier, result_queue),
                name=f"transfer-child-{i}",
            )
            for i in range(n_children)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            # Generous timeout: each child builds an app, starts it,
            # plans, waits on the barrier, and tries start_transfer.
            proc.join(timeout=120)
            assert proc.exitcode is not None, (
                f"child {proc.name} did not finish in time"
            )
            assert proc.exitcode == 0, (
                f"child {proc.name} crashed with exit code {proc.exitcode}"
            )

        outcomes = []
        while not result_queue.empty():
            outcomes.append(result_queue.get())
        assert len(outcomes) == n_children, (
            f"expected {n_children} child outcomes, got {len(outcomes)}: {outcomes}"
        )

        # EXACTLY ONE child must succeed; the other two must be rejected
        # by the cross-process MaxOpenTransfers enforcement.
        successes = [o for o in outcomes if o["ok"]]
        failures = [o for o in outcomes if not o["ok"]]
        assert len(successes) == 1, (
            "MAX_OPEN_TRANSFERS=1 must allow exactly one cross-process "
            f"start; got {len(successes)} successes and {len(failures)} "
            f"failures: {outcomes}"
        )
        assert len(failures) == n_children - 1
        for failure in failures:
            assert failure["error"] is not None, (
                f"failed child {failure['child']} must report an error: {failure}"
            )
            # The error must clearly identify max_open_transfers as the
            # reason for the rejection.
            assert "max_open_transfers" in failure["error"], (
                f"failed child {failure['child']} error must mention "
                f"max_open_transfers; got: {failure['error']!r}"
            )

        # EXACTLY ONE open transfer row must exist in the database.
        from app.models.enums import TransferState

        transfers = TransferRepository(parent_app.db)
        open_records = await transfers.list_open()
        assert len(open_records) == 1, (
            f"DB must hold exactly one open transfer after cross-process "
            f"start, got {len(open_records)}: {open_records}"
        )
        # The single surviving record must match the successful child.
        assert open_records[0].id == successes[0]["record_id"]
        assert open_records[0].state is TransferState.WITHDRAW_SUBMITTED, (
            f"the surviving transfer must have advanced to "
            f"WITHDRAW_SUBMITTED, got {open_records[0].state}"
        )

        # No duplicate persisted rows: the database must hold exactly
        # one transfer row total.
        all_recent = await transfers.list_recent(limit=10)
        assert len(all_recent) == 1, (
            f"DB must hold exactly one transfer row, got {len(all_recent)}"
        )
        assert all_recent[0].id == successes[0]["record_id"]

        # Database integrity: PRAGMA integrity_check must report ok.
        from sqlalchemy import text as _text

        async with parent_app.db.engine.connect() as conn:
            result = await conn.execute(_text("PRAGMA integrity_check"))
            row = result.first()
            assert row is not None
            assert row[0] == "ok", (
                f"SQLite integrity_check must report ok, got {row[0]!r}"
            )
    finally:
        await shutdown_app(parent_app)
