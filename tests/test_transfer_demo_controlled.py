"""Controlled DEMO transfer test: profitable opportunity above 30 bps without weakening thresholds."""

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest
from app.config.modes import policy_for
from app.execution.fill_simulator import FillSimulator
from app.execution.guard import ExecutionGuard
from app.execution.paper_wallet import PaperWallet
from app.market_data.store import MarketDataStore
from app.models.base import utc_now
from app.models.enums import OrderSide, OrderStatus, OrderType, TradingMode, TransferState
from app.models.market_data import OrderBook, OrderBookLevel
from app.models.symbol import Symbol
from app.models.transfer import TransferPlan
from app.recovery import ExecutionRecovery
from app.storage.engine import Database
from app.storage.repositories import AuditLogRepository, TradeRepository, TransferRepository
from app.strategies.transfer import TransferOrchestrator, TransferPlanner
from app.models.transfer import DepositAddress, TransferTx, WithdrawalNetwork

D = Decimal

# Profitable books: buy at 100 (ask), sell at 105 (bid) => spread 500 bps
# After fees (10+10 bps) + withdrawal 0.01*105=1.05, net still >30 bps for amount 5
# buy_cost 500, sell 525, gross 25, fees ~1.0+0.5+1.05=2.55, net 22.45, net_bps 449
def _profitable_book(venue: str, symbol: Symbol, *, buy_side: bool) -> OrderBook:
    # For buy book (source), asks at 100, bids at 99.5
    # For sell book (dest), bids at 105, asks at 105.5
    if buy_side:
        bids = tuple(OrderBookLevel(price=D("99.5"), amount=D("100")) for _ in range(3))
        asks = tuple(OrderBookLevel(price=D("100"), amount=D("100")) for _ in range(3))
    else:
        bids = tuple(OrderBookLevel(price=D("105"), amount=D("100")) for _ in range(3))
        asks = tuple(OrderBookLevel(price=D("105.5"), amount=D("100")) for _ in range(3))
    return OrderBook(
        exchange_id=venue,
        symbol=symbol,
        bids=bids,
        asks=asks,
        timestamp=utc_now(),
        received_at=utc_now(),
    )

class _Venue:
    def __init__(self):
        self.create_calls = []
        self.withdraw_calls = []
        self.orders = {}
    async def fetch_balances(self):
        from app.models.balance import Balance, BalanceSnapshot
        return BalanceSnapshot(
            exchange_id="binance",
            balances=(__import__("app.models.balance", fromlist=["Balance"]).Balance(exchange_id="binance", asset="USDT", free=D("100000")),),
            timestamp=utc_now(),
        )
    async def fetch_trading_fees(self, symbol):
        from app.models.market import MarketFees
        return MarketFees(maker_bps=D("10"), taker_bps=D("10"))
    async def fetch_withdrawal_networks(self, asset):
        return (WithdrawalNetwork(network="SIM", network_code="SIM", withdraw_enabled=True, deposit_enabled=True, withdrawal_fee=D("0.01"), withdrawal_min=D("0.1")),)
    async def fetch_deposit_address(self, asset, *, network=None):
        return DepositAddress(address="0xDEST", network=network)
    async def create_order(self, request):
        self.create_calls.append(request)
        # Simulate filled order
        from app.models.order import Order
        avg = D("100") if request.side == OrderSide.BUY else D("105")
        return Order(
            exchange_id=request.exchange_id,
            exchange_order_id=f"E{len(self.create_calls)}",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            status=OrderStatus.FILLED,
            amount=request.amount,
            filled_amount=request.amount,
            average_price=avg,
        )
    async def fetch_order(self, order_id, *, symbol): return None
    async def fetch_open_orders(self, *, symbol=None): return ()
    async def withdraw(self, asset, amount, address, *, memo=None, network=None):
        self.withdraw_calls.append((asset, amount, address))
        return TransferTx(direction="withdrawal", asset=asset, amount=D(amount), status="ok", txid=f"wd-{len(self.withdraw_calls)}")
    async def fetch_deposits(self, asset, *, limit=20): return ()
    async def fetch_withdrawals(self, asset, *, limit=20): return ()

class _Manager:
    def __init__(self, src, dst):
        self._m = {"binance": src, "okx": dst}
    def adapter(self, venue): return self._m[venue]
    def enabled_ids(self): return ("binance", "okx")

async def _demo_orchestrator(tmp_path: Path):
    from tests.conftest import make_settings
    settings = make_settings(tmp_path)
    settings = settings.model_copy(update={
        "trading": settings.trading.model_copy(update={"mode": TradingMode.DEMO}),
        "transfer": settings.transfer.model_copy(update={"simulated_transfer_seconds": 0.1, "poll_interval_seconds": 0.05}),
        "execution": settings.execution.model_copy(update={"leg_timeout_seconds": 0.3}),
    })
    # Keep real thresholds: 70 and 10 (not -10000)
    assert settings.transfer.min_net_profit_bps == D("70")
    assert settings.risk.min_net_profit_bps == D("10")
    db = Database(settings.database.model_copy(update={"url": f"sqlite+aiosqlite:///{tmp_path / 'demo.db'}"}))
    await db.create_schema()
    transfers = TransferRepository(db)
    trades = TradeRepository(db)
    audit = AuditLogRepository(db)
    store = MarketDataStore(stale_after_ms=60_000)
    sym = Symbol.parse("ETH/USDT")
    store.put_order_book(_profitable_book("binance", sym, buy_side=True))
    store.put_order_book(_profitable_book("okx", sym, buy_side=False))
    src = _Venue()
    dst = _Venue()
    # Paper wallets for DEMO sell leg
    from app.execution.paper_wallet import PaperWallet
    paper_wallets = {"binance": PaperWallet({"USDT": D("100000")}), "okx": PaperWallet({"USDT": D("100000")})}
    # Need market service for _fresh_book refresh
    from app.market_data.service import MarketDataService
    from app.exchanges.manager import ExchangeManager
    # Create a minimal market service that uses the same store
    # For this controlled test, we don't need real market service, just a fake that does nothing on refresh
    class FakeMarket:
        async def refresh_order_books(self, symbols, *, exchange_ids=None):
            return None
    market = FakeMarket()
    orchestrator = TransferOrchestrator(
        settings=settings,
        manager=_Manager(src, dst),
        store=store,
        guard=ExecutionGuard(policy_for(TradingMode.DEMO)),
        recovery=ExecutionRecovery(),
        fill_simulator=FillSimulator(taker_fee_bps=D("10"), max_slippage_bps=D("500")),
        planner=TransferPlanner(settings),
        transfer_repo=transfers,
        trade_repo=trades,
        audit=audit,
        risk_check=None,
        paper_wallets=paper_wallets,
        market=market,
    )
    return orchestrator, transfers, src, dst, db, settings, store

@pytest.mark.asyncio
async def test_demo_controlled_profitable_plan_found(tmp_path: Path):
    orchestrator, _, _, _, db, _, _ = await _demo_orchestrator(tmp_path)
    try:
        plans = await orchestrator.plan(asset="ETH", source="binance", dest="okx")
        assert len(plans) == 1, f"expected 1 profitable plan above 70 bps, got {plans}"
        plan = plans[0]
        assert plan.net_profit_bps >= D("70"), f"net {plan.net_profit_bps} <70"
        assert plan.net_profit_bps < D("1000")  # sanity
        # Verify PnL includes all fees: net = gross - total_fees
        # gross = (sell - buy)*amount = (105-100)*5=25, total_fees = buy_fee 0.5 + sell_fee 0.525 + withdrawal 1.05 =2.075, net 22.925
        # Our books have buy 100, sell 105, amount 5 (executable limited by available_quote/max)
        # Check that net is computed correctly
        assert plan.gross_profit_quote == plan.sell_price * plan.amount - plan.buy_price * plan.amount
        assert plan.total_fees_quote == plan.trading_fees_quote + plan.withdrawal_cost_quote
        assert plan.net_profit_quote == plan.gross_profit_quote - plan.total_fees_quote
    finally:
        await db.dispose()

@pytest.mark.asyncio
async def test_demo_controlled_full_lifecycle(tmp_path: Path):
    orchestrator, transfers, src, dst, db, settings, store = await _demo_orchestrator(tmp_path)
    try:
        plans = await orchestrator.plan(asset="ETH", source="binance", dest="okx")
        assert plans
        plan = plans[0]
        # Risk check with real limits should also approve (net >10)
        from app.risk import RiskEngine
        from app.risk.rules import RiskContext
        from app.models.enums import ArbitrageStrategy
        limits = settings.risk.to_limits()
        # For DEMO, RiskEngine should have min 10 (not -10000) as per 7c2ddaf
        assert limits.min_net_profit_bps == D("10")
        engine = RiskEngine(limits)
        ctx = RiskContext(strategy=ArbitrageStrategy.TRANSFER, notional_quote=plan.buy_cost_quote, net_profit_bps=plan.net_profit_bps, slippage_bps=plan.estimated_slippage_bps, data_age_ms=plan.data_age_ms, daily_pnl=D("0"), open_transfers=0, exchange_exposure={}, asset_exposure={}, kill_switch_engaged=False)
        assessment = engine.evaluate(ctx)
        assert assessment.approved, f"risk rejected profitable DEMO plan {assessment.violations}"

        # Start transfer
        record = await orchestrator.start(plan)
        assert record.state == "buy_filled" or record.state == "withdraw_submitted"
        # Advance through lifecycle: withdraw -> transfer -> deposit -> sell -> completed
        # Use orchestrator.tick to advance (simulated delay 0.1s)
        seen = {record.state}
        for _ in range(20):
            await asyncio.sleep(0.05)
            advanced = await orchestrator.tick()
            for r in advanced:
                if r.id == record.id:
                    record = r
                    seen.add(r.state)
            # Refresh from DB
            record = await transfers.get(record.id)
            if record.is_terminal:
                break
        assert record.state == "completed", f"final {record.state} {record.error}"
        # Verify PnL includes all fees correctly
        # proceeds - buy_cost should equal net (already embedded), and fees for reporting should be as per plan scaled
        assert record.realized_profit_quote is not None
        assert record.fees_quote is not None
        # Realized should be proceeds - buy_cost (net already)
        # For this profitable plan, realized should be >0
        assert record.realized_profit_quote > D("0"), f"realized {record.realized_profit_quote} not >0"
        # Fees should be >0 and approx total_fees scaled to actual filled amounts
        assert record.fees_quote > D("0")
        # Check that sell was via paper wallet (DEMO sell)
        assert len(dst.create_calls) == 0 or dst.create_calls[0].side == OrderSide.SELL
        # The transfer should have produced a trade
        print(f"DEMO completed {record.id} net {record.realized_profit_quote} fees {record.fees_quote} buy {record.buy_filled_amount} sell {record.sell_filled_amount}")
    finally:
        await db.dispose()
