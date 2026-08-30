"""Storage round-trips: trades, transfers, balances, audit log, bot state."""

from decimal import Decimal

from app.models.enums import (
    ArbitrageStrategy,
    TradeStatus,
    TradingMode,
    TransferState,
)
from app.models.trade import TradeRecord
from app.models.transfer import TransferPlan, TransferRecord
from app.storage.engine import Database
from app.storage.repositories import (
    AuditLogRepository,
    BalanceRepository,
    BotStateRepository,
    TradeRepository,
    TransferRepository,
)

D = Decimal


async def _db(tmp_path):
    from app.config.settings import DatabaseSettings

    db = Database(DatabaseSettings(url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"))
    await db.create_schema()
    return db


async def test_trade_round_trip(tmp_path):
    db = await _db(tmp_path)
    try:
        trades = TradeRepository(db)
        trade = TradeRecord(
            strategy=ArbitrageStrategy.TRIANGLE,
            mode=TradingMode.PAPER,
            exchange_id="binance",
            route="USDT->BTC->ETH->USDT",
            symbols=("BTC/USDT", "ETH/BTC", "ETH/USDT"),
            input_amount=D("1000"),
            output_amount=D("1004"),
            fees_quote=D("1.2"),
            net_profit=D("2.8"),
            net_profit_bps=D("28"),
            status=TradeStatus.COMPLETED,
        )
        await trades.save(trade)
        loaded = await trades.get(trade.id)
        assert loaded is not None
        assert loaded.route == trade.route
        assert loaded.symbols == trade.symbols
        assert loaded.net_profit == D("2.8")
        assert loaded.status is TradeStatus.COMPLETED
        recent = await trades.list_recent(10)
        assert [t.id for t in recent] == [trade.id]
        # daily pnl aggregation
        assert await trades.realized_pnl_today() == D("2.8")
    finally:
        await db.dispose()


async def test_transfer_lifecycle_persistence(tmp_path):
    db = await _db(tmp_path)
    try:
        transfers = TransferRepository(db)
        plan = TransferPlan(
            source_exchange="binance",
            dest_exchange="okx",
            asset="SOL",
            network="SOL",
            amount=D("10"),
            buy_price=D("100"),
            sell_price=D("105"),
        )
        record = TransferRecord(
            source_exchange="binance",
            dest_exchange="okx",
            asset="SOL",
            network="SOL",
            amount=D("10"),
            plan=plan,
        )
        # CREATED -> ... -> COMPLETED with persistence at each step
        for state in (
            TransferState.BUY_FILLED,
            TransferState.WITHDRAW_SUBMITTED,
            TransferState.TRANSFER_IN_PROGRESS,
            TransferState.COMPLETED,
        ):
            record = record.with_state(state)
            await transfers.save(record)
            loaded = await transfers.get(record.id)
            assert loaded is not None
            assert loaded.state is state
            assert loaded.plan.buy_price == D("100")  # plan survives round trips
        assert await transfers.count_open() == 0
        # an open record is listed for resume
        open_record = record.with_state(TransferState.DEPOSIT_DETECTED)
        await transfers.save(open_record)
        listed = await transfers.list_open()
        assert [r.id for r in listed] == [open_record.id]
        assert await transfers.count_open() == 1
    finally:
        await db.dispose()


async def test_balance_snapshots_replace_per_exchange(tmp_path):
    db = await _db(tmp_path)
    try:
        from app.models.balance import Balance, BalanceSnapshot

        repo = BalanceRepository(db)
        snapshot = BalanceSnapshot(
            exchange_id="binance",
            balances=(
                Balance(exchange_id="binance", asset="USDT", free=D("100")),
                Balance(exchange_id="binance", asset="BTC", free=D("1")),
            ),
        )
        await repo.save_snapshot(snapshot)
        # second snapshot replaces the first
        updated = BalanceSnapshot(
            exchange_id="binance",
            balances=(Balance(exchange_id="binance", asset="USDT", free=D("250")),),
        )
        await repo.save_snapshot(updated)
        loaded = await repo.load_all()
        assert set(loaded) == {"binance"}
        assert loaded["binance"].free_of("USDT") == D("250")
        assert loaded["binance"].free_of("BTC") == D("0")  # gone
        assert await repo.free_of("binance", "USDT") == D("250")
    finally:
        await db.dispose()


async def test_balance_snapshot_mutation_guard(tmp_path):
    """Domain models are frozen: updates go through model_copy, never assignment."""
    from app.models.balance import Balance, BalanceSnapshot

    snapshot = BalanceSnapshot(
        exchange_id="binance",
        balances=(Balance(exchange_id="binance", asset="USDT", free=D("100")),),
    )
    bigger = snapshot.model_copy(
        update={
            "balances": (
                *snapshot.balances,
                Balance(exchange_id="binance", asset="BTC", free=D("2")),
            )
        }
    )
    assert bigger.free_of("BTC") == D("2")
    assert snapshot.free_of("BTC") == D("0")  # original untouched


async def test_audit_log_and_bot_state(tmp_path):
    db = await _db(tmp_path)
    try:
        audit = AuditLogRepository(db)
        await audit.log("TEST_EVENT", "something happened", {"key": "value"})
        rows = await audit.list_recent(5)
        assert len(rows) == 1
        assert rows[0].action == "TEST_EVENT"
        assert rows[0].context == {"key": "value"}

        state = BotStateRepository(db)
        assert await state.get("missing") is None
        await state.set("kill_switch", {"engaged": True, "reason": "test"})
        assert (await state.get("kill_switch"))["engaged"] is True
        await state.delete("kill_switch")
        assert await state.get("kill_switch") is None
    finally:
        await db.dispose()


async def test_schema_drift_recreates_database(tmp_path):
    """An old file with a drifted schema is renamed to .bak, not crashed on."""
    from app.config.settings import DatabaseSettings
    from sqlalchemy import text

    path = tmp_path / "drift.db"
    db = Database(DatabaseSettings(url=f"sqlite+aiosqlite:///{path}"))
    await db.create_schema()
    # simulate drift: drop a column by creating an incompatible legacy table
    async with db.session() as session:
        await session.execute(text("DROP TABLE trades"))
        await session.execute(text("CREATE TABLE trades (id TEXT PRIMARY KEY, legacy TEXT)"))
    await db.dispose()

    db2 = Database(DatabaseSettings(url=f"sqlite+aiosqlite:///{path}"))
    await db2.create_schema()  # must detect drift and rebuild
    trades = TradeRepository(db2)
    await trades.save(
        TradeRecord(
            strategy=ArbitrageStrategy.TRIANGLE,
            exchange_id="binance",
            route="r",
            status=TradeStatus.COMPLETED,
        )
    )
    assert await trades.realized_pnl_today() == D("0")
    assert (tmp_path / "drift.db.bak").exists()
    await db2.dispose()
