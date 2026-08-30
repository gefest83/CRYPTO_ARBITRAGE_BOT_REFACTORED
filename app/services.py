"""Application services: the single composition root.

Everything the CLI and the Telegram bot do goes through :class:`AppServices` —
one object, built once per process, that owns:

* the exchange manager (fixed binance/okx/bybit universe),
* market data (WebSocket-first, REST fallback),
* risk engine + persistent kill switch,
* the two strategies (triangular scanner/executor, transfer orchestrator),
* storage (trades, transfers, balances, audit log, bot state).

No trading logic lives in the CLI or Telegram layers; they are interfaces.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.config.logging_config import get_logger
from app.config.modes import policy_for
from app.config.settings import Settings, get_settings
from app.execution.fill_simulator import FillSimulator
from app.execution.guard import ExecutionGuard
from app.execution.order_gate import live_session_gate, never_place_orders, sandbox_only
from app.execution.paper_wallet import PaperWallet
from app.execution.precision import InstrumentFilters, StaticPrecisionProvider
from app.market_data.service import MarketDataService
from app.market_data.store import MarketDataStore
from app.models.arbitrage import ArbitrageOpportunity
from app.models.balance import BalanceSnapshot
from app.models.base import DEC0
from app.models.enums import ArbitrageStrategy, TradeStatus, TradingMode
from app.models.risk import RiskAssessment
from app.models.symbol import Symbol
from app.models.trade import TradeRecord
from app.models.transfer import TransferPlan, TransferRecord
from app.recovery import ExecutionRecovery
from app.risk import RiskEngine, RiskStateTracker
from app.risk.rules import RiskContext, exposure_maps_for_transfers
from app.storage.engine import Database
from app.storage.repositories import (
    AuditLogRepository,
    BalanceRepository,
    BotStateRepository,
    TradeRepository,
    TransferRepository,
)
from app.strategies.transfer import TransferOrchestrator, TransferPlanner
from app.strategies.triangular import TriangleExecutor, TriangularScanner
from app.strategies.triangular.fees import ScanRequest

__all__ = ["AppServices", "build_app", "shutdown_app"]

logger = get_logger("runtime")

_AUTO_FLAG_KEY = "auto_trading"

#: Fallback taker fee (bps) used when a venue does not publish account fees.
_DEFAULT_TAKER_BPS = Decimal("10")


@dataclass(slots=True)
class AppServices:
    settings: Settings
    db: Database
    guard: ExecutionGuard
    manager: Any  # ExchangeManager (typed loosely to avoid import cycle noise)
    store: MarketDataStore
    market: MarketDataService
    risk: RiskEngine
    risk_state: RiskStateTracker
    scanner: TriangularScanner
    executor: TriangleExecutor
    orchestrator: TransferOrchestrator | None
    trades: TradeRepository
    transfers: TransferRepository
    balance_repo: BalanceRepository
    audit: AuditLogRepository
    bot_state: BotStateRepository
    paper_wallets: dict[str, PaperWallet] = field(default_factory=dict)
    watch_symbols: tuple[Symbol, ...] = ()
    started_at: float = field(default_factory=time.monotonic)
    _cached_exchange_exposure: dict = field(default_factory=dict)
    _cached_asset_exposure: dict = field(default_factory=dict)

    # ------------------------------------------------------------ risk
    def risk_environment(self):
        """Synchronous snapshot of the risk environment (cached exposures)."""
        return self.risk_state.snapshot(
            kill_switch_engaged=self.guard.is_halted,
            exchange_exposure=self._cached_exchange_exposure,
            asset_exposure=self._cached_asset_exposure,
        )

    async def refresh_risk_exposure(self) -> None:
        """Recompute exposure maps from the open transfers (async, cached)."""
        open_records = await self.transfers.list_open()
        exchange, asset = exposure_maps_for_transfers(open_records)
        self._cached_exchange_exposure = exchange
        self._cached_asset_exposure = asset
        self.risk_state.open_transfers = len(open_records)

    def validate_triangle(self, opportunity: ArbitrageOpportunity) -> RiskAssessment:
        environment = self.risk_environment()
        context = RiskContext(
            strategy=ArbitrageStrategy.TRIANGLE,
            notional_quote=opportunity.size_notional_quote,
            net_profit_bps=opportunity.net_profit_bps,
            slippage_bps=opportunity.profit.slippage_bps,
            data_age_ms=opportunity.data_age_ms,
            daily_pnl=environment.daily_pnl,
            open_transfers=environment.open_transfers,
            exchange_exposure=environment.exchange_exposure,
            asset_exposure=environment.asset_exposure,
            kill_switch_engaged=environment.kill_switch_engaged,
        )
        return self.risk.evaluate(context)

    def validate_transfer(self, plan: TransferPlan) -> RiskAssessment:
        environment = self.risk_environment()
        context = RiskContext(
            strategy=ArbitrageStrategy.TRANSFER,
            notional_quote=plan.buy_cost_quote,
            net_profit_bps=plan.net_profit_bps,
            slippage_bps=plan.estimated_slippage_bps,
            data_age_ms=0.0,  # plans are built from fresh books or rejected
            daily_pnl=environment.daily_pnl,
            open_transfers=environment.open_transfers,
            exchange_exposure=environment.exchange_exposure,
            asset_exposure=environment.asset_exposure,
            kill_switch_engaged=environment.kill_switch_engaged,
        )
        return self.risk.evaluate(context)

    # ------------------------------------------------------------ triangle
    async def scan_triangles(
        self, notional_quote: Decimal | None = None
    ) -> tuple[ArbitrageOpportunity, ...]:
        """Refresh books if needed and scan all venues for triangle cycles."""
        await self.ensure_market_data()
        request = ScanRequest(
            symbols=tuple(
                Symbol(base=asset, quote=self.settings.trading.base_currency)
                for asset in self.settings.arbitrage.triangle_assets
            ),
            notional_quote=notional_quote or self.settings.arbitrage.default_notional_quote,
            min_net_profit_bps=self.settings.arbitrage.triangle_min_net_bps,
            max_results=self.settings.arbitrage.max_results,
            require_fresh_data=self.settings.arbitrage.require_fresh_data,
            ttl_ms=self.settings.arbitrage.scan_ttl_ms,
        )
        opportunities = await self.scanner.scan(request)
        return opportunities

    async def execute_triangle(
        self, opportunity: ArbitrageOpportunity
    ) -> tuple[TradeRecord, RiskAssessment | None]:
        """Risk-validate and execute one triangle opportunity."""
        self.guard.ensure_can_trade()
        assessment = self.validate_triangle(opportunity)
        if not assessment.approved:
            await self.audit.log(
                "TRIANGLE_RISK_REJECTED",
                "; ".join(assessment.reasons),
                {"opportunity_id": opportunity.id},
            )
            return (
                TradeRecord(
                    strategy=ArbitrageStrategy.TRIANGLE,
                    mode=self.settings.mode,
                    exchange_id=opportunity.buy_leg.exchange_id,
                    route=opportunity.direction or "",
                    input_amount=opportunity.size_notional_quote,
                    net_profit_bps=opportunity.net_profit_bps,
                    status=self._risk_status(),
                    error="; ".join(assessment.reasons),
                ),
                assessment,
            )
        trade = await self.executor.execute(opportunity)
        if trade.status.value == "completed":
            self.risk_state.register_realized_pnl(trade.net_profit)
        await self.refresh_risk_exposure()
        return trade, assessment

    # ------------------------------------------------------------ transfer
    async def plan_transfers(
        self,
        *,
        asset: str | None = None,
        amount: Decimal | None = None,
        source: str | None = None,
        dest: str | None = None,
    ) -> list[TransferPlan]:
        await self.ensure_market_data()
        return await self.orchestrator.plan(asset=asset, amount=amount, source=source, dest=dest)

    async def start_transfer(self, plan: TransferPlan) -> TransferRecord:
        record = await self.orchestrator.start(plan)
        await self.refresh_risk_exposure()
        return record

    async def tick_transfers(self) -> list[TransferRecord]:
        """Advance all open transfers; refresh exposure and daily P&L."""
        advanced = await self.orchestrator.tick()
        for record in advanced:
            if record.state.value == "completed":
                self.risk_state.register_transfer_closed(realized_pnl=record.realized_profit_quote)
                await self.trades.save(
                    TradeRecord(
                        strategy=ArbitrageStrategy.TRANSFER,
                        mode=self.settings.mode,
                        exchange_id=f"{record.source_exchange}->{record.dest_exchange}",
                        route=f"USDT->{record.asset}->{record.network}->{record.asset}/USDT",
                        symbols=(f"{record.asset}/{self.settings.trading.base_currency}",),
                        input_amount=record.plan.buy_cost_quote,
                        output_amount=record.sell_proceeds_quote,
                        fees_quote=record.fees_quote,
                        net_profit=record.realized_profit_quote,
                        net_profit_bps=(
                            record.realized_profit_quote
                            / record.plan.buy_cost_quote
                            * Decimal("10000")
                            if record.plan.buy_cost_quote > DEC0
                            else DEC0
                        ),
                        status=TradeStatus.COMPLETED,
                        transfer_id=record.id,
                    )
                )
        await self.refresh_risk_exposure()
        return advanced

    # ------------------------------------------------------------ balances
    async def balances(self) -> dict[str, BalanceSnapshot]:
        """Current balances per venue (paper wallets or live fetch)."""
        snapshots: dict[str, BalanceSnapshot] = {}
        for venue in self.manager.enabled_ids():
            if (
                self.settings.mode is TradingMode.PAPER
                and self.settings.exchanges.simulate_in_paper
            ):
                wallet = self.paper_wallets.get(venue)
                if wallet is not None:
                    from app.models.balance import Balance

                    snapshots[venue] = BalanceSnapshot(
                        exchange_id=venue,
                        balances=tuple(
                            Balance(exchange_id=venue, asset=asset, free=free)
                            for asset, free in sorted(wallet.snapshot().items())
                            if free > DEC0
                        ),
                    )
                    continue
            try:
                adapter = self.manager.adapter(venue)
                snapshot = await adapter.fetch_balances()
                snapshots[venue] = snapshot
            except Exception as exc:  # noqa: BLE001 - isolate venues
                await self.audit.log("BALANCE_FETCH_FAILED", str(exc), {"exchange_id": venue})
        for snapshot in snapshots.values():
            await self.balance_repo.save_snapshot(snapshot)
        return snapshots

    # ------------------------------------------------------------ status
    async def status(self) -> dict[str, Any]:
        await self.refresh_risk_exposure()
        recent_trades = await self.trades.list_recent(5)
        open_transfers = await self.transfers.list_open()
        return {
            "mode": self.settings.mode.value,
            "uptime_seconds": round(time.monotonic() - self.started_at, 1),
            "guard": self.guard.status(),
            "auto_trading": await self.auto_trading_enabled(),
            "exchanges": self.manager.status_snapshot(),
            "market_data": self.store.stats(),
            "risk": {
                "daily_pnl": str(self.risk_state.daily_pnl),
                "open_transfers": len(open_transfers),
                "limits": {
                    "max_trade_size": str(self.risk.limits.max_trade_size),
                    "min_net_profit_bps": str(self.risk.limits.min_net_profit_bps),
                    "max_daily_loss": str(self.risk.limits.max_daily_loss),
                    "max_open_transfers": self.risk.limits.max_open_transfers,
                },
            },
            "transfers_open": [
                {
                    "id": r.id,
                    "route": f"{r.source_exchange}->{r.dest_exchange}",
                    "asset": r.asset,
                    "network": r.network,
                    "amount": str(r.amount),
                    "state": r.state.value,
                }
                for r in open_transfers
            ],
            "recent_trades": [
                {
                    "id": t.id,
                    "strategy": t.strategy.value,
                    "route": t.route,
                    "status": t.status.value,
                    "net_profit": str(t.net_profit),
                }
                for t in recent_trades
            ],
        }

    # ------------------------------------------------------------ market data
    async def ensure_market_data(self) -> None:
        """Refresh cached books for the watchlist when they are missing/stale."""
        watch = self.watch_symbols
        if not watch:
            return
        stale_needed = [
            symbol
            for symbol in watch
            if not any(self.store.order_book(venue, symbol) for venue in self.manager.enabled_ids())
        ]
        symbols_to_refresh = (
            watch
            if stale_needed
            else tuple(
                dict.fromkeys(
                    symbol
                    for symbol in watch
                    for venue in self.manager.enabled_ids()
                    if (book := self.store.order_book(venue, symbol)) is None
                    or self.store.is_stale(self.store.age_ms(book))
                )
            )
        )
        if symbols_to_refresh:
            await self.market.refresh_order_books(tuple(symbols_to_refresh))

    # ------------------------------------------------------------ bot state
    async def auto_trading_enabled(self) -> bool:
        value = await self.bot_state.get(_AUTO_FLAG_KEY)
        return bool(value) if isinstance(value, bool) else False

    async def set_auto_trading(self, enabled: bool) -> None:
        await self.bot_state.set(_AUTO_FLAG_KEY, enabled)
        self.guard.enable_trading(enabled=enabled)
        await self.audit.log("AUTO_TRADING_ENABLED" if enabled else "AUTO_TRADING_DISABLED")

    async def engage_kill_switch(self, reason: str) -> None:
        self.guard.engage_kill_switch(reason)
        await self.guard.persist(self.bot_state)
        await self.audit.log("KILL_SWITCH_ENGAGED", reason)
        await self.set_auto_trading(False)

    async def release_kill_switch(self) -> None:
        self.guard.release_kill_switch()
        await self.guard.persist(self.bot_state)
        await self.audit.log("KILL_SWITCH_RELEASED")

    @staticmethod
    def _risk_status() -> TradeStatus:
        return TradeStatus.FAILED


async def build_app(settings: Settings | None = None) -> AppServices:
    """Compose the whole application (no network I/O beyond schema creation)."""
    settings = settings or get_settings()
    db = Database(settings.database)
    if settings.database.auto_create_schema:
        await db.create_schema()
    else:
        db.start()

    trades = TradeRepository(db)
    transfers = TransferRepository(db)
    balances = BalanceRepository(db)
    audit = AuditLogRepository(db)
    bot_state = BotStateRepository(db)

    policy = policy_for(settings.mode)
    guard = ExecutionGuard(policy, trading_enabled=False)
    await guard.restore(bot_state)  # kill switch survives restarts (fail-closed)

    # Order gate per mode: PAPER never places, DEMO only sandboxed venues,
    # LIVE requires the enabled guard (plus the settings-level triple opt-in
    # that already refused to load otherwise).
    if settings.mode is TradingMode.PAPER:
        gate = never_place_orders
    elif settings.mode is TradingMode.DEMO:
        gate = sandbox_only
    else:
        gate = live_session_gate(guard)

    from app.exchanges.manager import ExchangeManager  # local import: composition root

    manager = ExchangeManager(settings, order_gate=gate)
    store = MarketDataStore(stale_after_ms=settings.market_data.stale_after_ms)
    market = MarketDataService(manager=manager, store=store, config=settings.market_data)

    risk = RiskEngine(settings.risk.to_limits())
    risk_state = RiskStateTracker()
    risk_state.sync_from_storage(
        daily_pnl=await trades.realized_pnl_today(),
        open_transfers=await transfers.list_open(),
    )

    fill_simulator = FillSimulator(
        taker_fee_bps=_DEFAULT_TAKER_BPS,
        max_slippage_bps=Decimal(str(settings.execution.paper_max_slippage_bps)),
    )
    recovery = ExecutionRecovery()

    scanner = TriangularScanner(
        store=store,
        venues=lambda: (v for v in manager.enabled_ids() if not manager.is_breaker_open(v)),
        settings=settings.arbitrage,
    )

    # Paper wallets: deterministic seed balances per simulated venue.
    paper_wallets: dict[str, PaperWallet] = {}
    if settings.mode is TradingMode.PAPER and settings.exchanges.simulate_in_paper:
        for venue in manager.enabled_ids():
            paper_wallets[venue] = PaperWallet()

    precision = await _build_precision(manager)
    executor = TriangleExecutor(
        settings=settings,
        store=store,
        manager=manager,
        guard=guard,
        recovery=recovery,
        fill_simulator=fill_simulator,
        precision=precision,
        trade_repo=trades,
        audit=audit,
        paper_wallets=paper_wallets or None,
    )

    services = AppServices(
        settings=settings,
        db=db,
        guard=guard,
        manager=manager,
        store=store,
        market=market,
        risk=risk,
        risk_state=risk_state,
        scanner=scanner,
        executor=executor,
        orchestrator=None,  # wired below (needs services for risk checks)
        trades=trades,
        transfers=transfers,
        balance_repo=balances,
        audit=audit,
        bot_state=bot_state,
        paper_wallets=paper_wallets,
    )

    orchestrator = TransferOrchestrator(
        settings=settings,
        manager=manager,
        store=store,
        guard=guard,
        recovery=recovery,
        fill_simulator=fill_simulator,
        planner=TransferPlanner(settings),
        transfer_repo=transfers,
        trade_repo=trades,
        audit=audit,
        risk_check=services.validate_transfer,
        paper_wallets=paper_wallets,
    )
    services.orchestrator = orchestrator
    return services


async def start_app(services: AppServices) -> None:
    """Open adapters, prime market data, restore runtime flags and transfers."""
    await services.manager.open_all()
    await _init_paper_wallets(services)
    await _init_watchlist(services)

    # Streams first (WebSocket-first policy), then a REST round for every
    # pair no stream covers yet.
    if services.settings.market_data.streams_enabled:
        await services.market.start_streams(services.watch_symbols)
    await services.market.refresh_order_books(services.watch_symbols)

    if await services.auto_trading_enabled():
        services.guard.enable_trading(enabled=True)
        logger.info("auto_trading_restored")

    await services.orchestrator.resume()
    await services.refresh_risk_exposure()
    await services.audit.log("APP_STARTED", str(services.settings.mode))


async def shutdown_app(services: AppServices) -> None:
    """Stop streams, close adapters, dispose the database."""
    await services.market.stop_streams()
    await services.manager.close()
    await services.db.dispose()
    await services.audit.log("APP_STOPPED")


async def _init_paper_wallets(services: AppServices) -> None:
    """Seed paper wallets from the simulated adapters' deterministic balances."""
    if not services.paper_wallets:
        return
    for venue, wallet in services.paper_wallets.items():
        adapter = services.manager.adapter(venue)
        snapshot = await adapter.fetch_balances()
        wallet.__init__({b.asset: b.free for b in snapshot.balances})


async def _init_watchlist(services: AppServices) -> None:
    """Build the watchlist: USDT pairs of the triangle assets + their crosses."""
    assets = set(services.settings.arbitrage.triangle_assets)
    assets.update(services.settings.transfer.assets)
    quote = services.settings.trading.base_currency
    watch: list[Symbol] = [Symbol(base=asset, quote=quote) for asset in sorted(assets)]
    listed: set[str] = set()
    for venue in services.manager.enabled_ids():
        try:
            markets = await services.manager.adapter(venue).load_markets()
        except Exception as exc:  # noqa: BLE001 - one venue failing is isolated
            logger.warning("load_markets_failed", extra={"venue": venue, "error": str(exc)})
            continue
        for market in markets:
            listed.add(market.symbol.name)
    # Cross pairs among the watched assets that at least one venue lists.
    watched_assets = sorted(assets)
    for i, quote_asset in enumerate(watched_assets):
        for base_asset in watched_assets[i + 1 :]:
            for name in (f"{base_asset}/{quote_asset}", f"{quote_asset}/{base_asset}"):
                if name in listed:
                    watch.append(Symbol.parse(name))
                    break
    services.watch_symbols = tuple(dict.fromkeys(watch))


async def _build_precision(manager) -> StaticPrecisionProvider:
    """Instrument filters from every venue's load_markets (best effort)."""
    by_venue: dict[tuple[str, str], InstrumentFilters] = {}
    by_symbol: dict[str, InstrumentFilters] = {}
    for venue in manager.enabled_ids():
        try:
            markets = await manager.adapter(venue).load_markets()
        except Exception:  # noqa: BLE001 - precision is best-effort
            continue
        for market in markets:
            filters = InstrumentFilters.from_market(market.precision, market.limits)
            by_venue[(venue, market.symbol.name)] = filters
            by_symbol.setdefault(market.symbol.name, filters)
    return StaticPrecisionProvider(by_venue=by_venue, by_symbol=by_symbol)
