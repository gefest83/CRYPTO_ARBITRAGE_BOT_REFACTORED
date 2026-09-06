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

import asyncio
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
from app.models.enums import TradingMode
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
_STRATEGY_KEY = "active_strategy"

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
    #: Background Telegram runner (None when disabled / not configured).
    telegram_runner: Any = None
    #: Background auto-trading loop controller (None only during build; always
    #: non-None once :func:`build_app` returns).
    auto_controller: Any = None
    _cached_exchange_exposure: dict = field(default_factory=dict)
    _cached_asset_exposure: dict = field(default_factory=dict)
    #: Serialises transfer starts (H-6): together with an in-lock exposure
    #: refresh this guarantees the MaxOpenTransfersRule always sees the true
    #: open-transfer count, so concurrent starts cannot race past the cap.
    _transfer_start_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # --- AI Advisor (optional, isolated subsystem — analytical only, never executive) ---
    # The advisor shares ``db``; no second database. ``None`` when the advisor
    # has not been wired via :func:`app.agent.build_agent`.
    agent_knowledge: Any = field(default=None)
    agent_experiences: Any = field(default=None)
    agent_lessons: Any = field(default=None)
    agent_recommendations: Any = field(default=None)
    agent_recommendation_service: Any = field(default=None)
    agent_knowledge_service: Any = field(default=None)
    agent_approval_service: Any = field(default=None)
    agent_audit: Any = field(default=None)
    #: Phase 4: read-only trading-journal access for the AI Agent (wired by
    #: :func:`app.agent.build_agent`; ``None`` until then).
    agent_journal: Any = field(default=None)
    #: Phase 5: memory overlay stores + reflection scheduler (wired by
    #: :func:`app.agent.build_agent`; ``None`` until then).
    agent_feedback: Any = field(default=None)
    agent_lesson_history: Any = field(default=None)
    agent_memory_labels: Any = field(default=None)
    agent_reflection: Any = field(default=None)
    #: Phase 6: event-driven operator notifications (wired by
    #: :func:`app.agent.build_agent`; ``None`` until then; explicit scan only).
    agent_notifications: Any = field(default=None)
    #: Phase 7: evidence-based recommendation engine (wired by
    #: :func:`app.agent.build_agent`; ``None`` until then; persistence only).
    agent_recommender: Any = field(default=None)

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
            # H-7: the real age of the pricing data the plan was built from
            # (measured at planning time) — a stale plan now fails closed
            # through MaxDataAgeRule instead of bypassing it with 0.0.
            data_age_ms=plan.data_age_ms,
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
        """Start one transfer workflow.

        H-6: starts are serialised and the open-transfer counter is refreshed
        from storage inside the lock, so the risk check inside
        ``orchestrator.start`` always sees the true count — concurrent starts
        cannot both pass MaxOpenTransfersRule on a stale in-memory snapshot.
        """
        async with self._transfer_start_lock:
            await self.refresh_risk_exposure()
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
        auto_flag = await self.auto_trading_enabled()
        auto_loop_running = (
            await self.auto_controller.is_running()
            if self.auto_controller is not None
            else False
        )
        active_strategy = await self.get_active_strategy()
        return {
            "mode": self.settings.mode.value,
            "uptime_seconds": round(time.monotonic() - self.started_at, 1),
            "guard": self.guard.status(),
            "auto_trading": auto_flag,
            "auto_loop_running": auto_loop_running,
            "active_strategy": active_strategy or "not_set",
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

    async def get_active_strategy(self) -> str | None:
        """Active DEMO strategy: 'triangle' or 'transfer', or None if not chosen."""
        value = await self.bot_state.get(_STRATEGY_KEY)
        if isinstance(value, str) and value in ("triangle", "transfer"):
            return value
        return None

    async def set_active_strategy(self, strategy: str) -> None:
        if strategy not in ("triangle", "transfer"):
            raise ValueError(f"unknown strategy: {strategy}")
        await self.bot_state.set(_STRATEGY_KEY, strategy)
        await self.audit.log("ACTIVE_STRATEGY_SET", strategy)

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
    # Initialise logging exactly once at the entry point of the application.
    # The function is idempotent (a no-op on repeat calls unless ``force=True``).
    from app.config.logging_config import configure_logging

    configure_logging(settings.logging)
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

    limits = settings.risk.to_limits()
    risk = RiskEngine(limits)
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

    fee_provider = await _build_fee_provider(manager, settings)

    scanner = TriangularScanner(
        store=store,
        venues=lambda: (v for v in manager.enabled_ids() if not manager.is_breaker_open(v)),
        settings=settings.arbitrage,
        fees=fee_provider,
    )

    # Paper wallets: deterministic seed balances per simulated venue.
    # For DEMO, the blockchain leg is simulated, so the destination sell
    # leg needs a paper wallet to hold the simulated deposit. Real DEMO
    # buy/sell legs still go through the venue, but the transfer's
    # simulated deposit is credited to the paper wallet for the sell.
    paper_wallets: dict[str, PaperWallet] = {}
    if settings.mode is TradingMode.PAPER and settings.exchanges.simulate_in_paper:
        for venue in manager.enabled_ids():
            paper_wallets[venue] = PaperWallet()
    elif settings.mode is TradingMode.DEMO:
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
        market=market,
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
        market=market,
    )
    services.orchestrator = orchestrator
    # AutoTradingController: single owner of the AutoTrader background loop.
    # Both the CLI start_auto / stop_auto and the Telegram /start_trading /
    # /stop_trading commands must go through this controller.
    from app.auto_controller import AutoTradingController

    services.auto_controller = AutoTradingController(services)
    return services


async def start_app(
    services: AppServices, *, start_telegram: bool = True, start_streams: bool = True
) -> None:
    """Open adapters, prime market data, restore runtime flags and transfers."""
    await services.manager.open_all()
    # DEMO preflight: verify credentials/balances/fees/market data per venue
    # before any trading — fail-closed per venue, never crash the app.
    if services.settings.mode is TradingMode.DEMO:
        try:
            from app.exchanges.preflight import run_demo_preflight

            await run_demo_preflight(services.manager, services.settings, services.store)
        except Exception as exc:  # noqa: BLE001 - preflight must never block startup
            logger.warning("demo_preflight_failed", extra={"error": str(exc)[:300]})
    await _init_paper_wallets(services)
    await _init_watchlist(services)

    # REST-prime USDT pairs BEFORE starting WS streams: ccxt.pro clients
    # share rate limits / connections between WS subscriptions and REST, so
    # priming while 150 WS streams (re)subscribe — especially during an OKX
    # poison-recovery wave — stalls REST for minutes. REST-first (10s, no WS
    # contention), then WS streams take over live updates for the same pairs.
    # Crosses (triangle-only) are fetched on demand by the first scan.
    _base_ccy = services.settings.trading.base_currency
    _prime_symbols = tuple(s for s in services.watch_symbols if s.quote == _base_ccy) or services.watch_symbols
    await services.market.refresh_order_books(_prime_symbols)
    # Long-running processes (telegram / start_auto) keep live WS streams;
    # one-shot CLI commands (status/scan/...) use REST-only priming above and
    # skip WS entirely — otherwise every short command pays WS subscribe,
    # poison-recovery and close costs (minutes) just to print and exit.
    if start_streams and services.settings.market_data.streams_enabled:
        await services.market.start_streams(services.watch_symbols)
        try:
            await asyncio.wait_for(_wait_for_ws_warmup(services), timeout=8.0)
        except TimeoutError:
            pass

    if await services.auto_trading_enabled():
        services.guard.enable_trading(enabled=True)
        logger.info("auto_trading_restored")

    await services.orchestrator.resume()
    # H-3/H-4: resolve trades interrupted mid-cycle (queries only; incomplete
    # cycles fail closed into MANUAL_REVIEW, never auto-continued).
    resumed_trades = await services.executor.resume_open_trades()
    if resumed_trades:
        logger.info(
            "triangle_trades_resumed",
            extra={"count": len(resumed_trades)},
        )
    await services.refresh_risk_exposure()
    await services.audit.log("APP_STARTED", str(services.settings.mode))

    # Telegram operator interface: optional, fail-safe.  A startup failure
    # never blocks the bot (auto trading is unaffected) and never enables it.
    # One-shot CLI commands (status/scan/balances/...) skip it entirely —
    # the Telegram getMe round-trip (often 60-90s to api.telegram.org from
    # this network) would otherwise dominate every short command.
    if start_telegram:
        await _start_telegram(services)


async def shutdown_app(services: AppServices) -> None:
    """Stop streams, close adapters, dispose the database."""
    import traceback

    logger.info("shutdown_app_invoked", extra={"caller": traceback.extract_stack()[-2].name})
    # Auto-trading loop first: stop any in-flight cycle before tearing down
    # state.  The controller's shutdown is idempotent and never raises.
    if services.auto_controller is not None:
        await services.auto_controller.shutdown()
    # Telegram first: stop accepting new commands before tearing down state.
    await _stop_telegram(services)
    await services.market.stop_streams()
    await services.manager.close()
    await services.db.dispose()
    await services.audit.log("APP_STOPPED")
    logger.info("shutdown_app_complete")


async def _init_paper_wallets(services: AppServices) -> None:
    """Seed paper wallets from the simulated adapters' deterministic balances."""
    if not services.paper_wallets:
        return
    for venue, wallet in services.paper_wallets.items():
        adapter = services.manager.adapter(venue)
        try:
            snapshot = await adapter.fetch_balances()
            wallet.__init__({b.asset: b.free for b in snapshot.balances})
        except Exception as exc:  # noqa: BLE001 - DEMO fetch may timeout, seed with defaults
            logger.warning("paper_wallet_seed_failed", extra={"venue": venue, "error": str(exc)[:200]})
            # Fallback to large deterministic seed so transfer E2E can proceed in DEMO
            # Use the same seed as SimulatedExchangeAdapter but with venue offset
            try:
                from app.exchanges.simulated import _SEED_INVENTORY, _BALANCE_QUANTA
                from hashlib import blake2b

                def _offset(asset: str) -> Decimal:
                    digest = blake2b(f"{venue}:{asset}".encode(), digest_size=8).digest()
                    v = int.from_bytes(digest, "big") / float(1 << 64)
                    return Decimal(str(round(v * 2 - 1, 6)))

                seeded: dict[str, Decimal] = {}
                for asset, (base, spread) in _SEED_INVENTORY.items():
                    quanta = _BALANCE_QUANTA.get(asset, Decimal("0.01"))
                    amt = max(Decimal("0"), (base + _offset(asset) * spread).quantize(quanta))
                    if amt > 0:
                        seeded[asset] = amt
                # Ensure at least USDT and common transfer assets are well funded
                for a in ("USDT", "BTC", "ETH", "SOL", "AVAX", "DOGE", "LINK", "XRP", "ADA", "TRX"):
                    seeded.setdefault(a, Decimal("10000") if a == "USDT" else Decimal("100"))
                wallet.__init__(seeded)
            except Exception:
                wallet.__init__({"USDT": Decimal("10000")})


async def _wait_for_ws_warmup(services: AppServices) -> None:
    """Wait until WS streams cover a useful share of the watchlist.

    Returns early once at least half of the supported (venue, symbol) book
    pairs are delivering via WS, or after the caller's timeout. Prevents the
    REST priming round from duplicating WS work on every boot.
    """
    from app.models.enums import StreamKind

    watch = services.watch_symbols
    if not watch:
        return
    supported_total = 0
    for venue in services.manager.enabled_ids():
        for symbol in watch:
            if (venue.strip().lower(), symbol.name) not in services.market._unsupported_pairs:
                supported_total += 1
    if supported_total <= 0:
        return
    target = max(1, supported_total // 2)
    for _ in range(80):  # up to ~8s at 0.1s polls; caller caps with wait_for
        covered = services.market.get_stream_covered_pairs(StreamKind.ORDER_BOOK)
        if len(covered) >= target:
            logger.info(
                "ws_warmup_covered",
                extra={"covered": len(covered), "supported": supported_total},
            )
            return
        await asyncio.sleep(0.1)


async def _init_watchlist(services: AppServices) -> None:
    """Build the watchlist: USDT pairs of all assets + triangle crosses.

    USDT legs cover the full transfer universe (50 assets) plus triangular
    legs. Cross pairs (X/Y) are only needed by the triangular scanner, which
    walks cycles among `triangle_assets` (10 assets) — crosses among the
    wider 50-asset transfer universe are never scanned, so including them
    only burns WS slots / REST calls and widens the WS-poisoning surface
    (e.g. OKX rejects ATOM/ETH over WS while REST lists it).
    """
    assets = set(services.settings.arbitrage.triangle_assets)
    assets.update(services.settings.transfer.assets)
    quote = services.settings.trading.base_currency
    watch: list[Symbol] = [Symbol(base=asset, quote=quote) for asset in sorted(assets)]
    # Venue-specific listed sets: only include a cross for a venue if that
    # venue actually lists it as active. The global watch is the union of
    # per-venue actives, but the check below uses per-venue data.
    listed_per_venue: dict[str, set[str]] = {}
    listed: set[str] = set()
    for venue in services.manager.enabled_ids():
        try:
            markets = await services.manager.adapter(venue).load_markets()
        except Exception as exc:  # noqa: BLE001 - one venue failing is isolated
            logger.warning("load_markets_failed", extra={"venue": venue, "error": str(exc)})
            continue
        venue_listed: set[str] = set()
        for market in markets:
            if not market.active:
                continue
            venue_listed.add(market.symbol.name)
            listed.add(market.symbol.name)
        listed_per_venue[venue] = venue_listed
    # Cross pairs among the TRIANGLE assets only (the scanner's cycle universe)
    # that at least one venue lists as active.
    # The watch remains global (union), but scanner will filter per-venue via
    # _venue_spot_books, so a cross like AVAX/BNB that is only active on
    # binance will not be attempted on okx/bybit.
    watched_assets = sorted(set(services.settings.arbitrage.triangle_assets))
    for i, quote_asset in enumerate(watched_assets):
        for base_asset in watched_assets[i + 1 :]:
            for name in (f"{base_asset}/{quote_asset}", f"{quote_asset}/{base_asset}"):
                if name in listed:
                    watch.append(Symbol.parse(name))
                    break
    services.watch_symbols = tuple(dict.fromkeys(watch))
    # Pre-seed permanently-unsupported (venue, symbol) pairs from the
    # load_markets active sets: a market no venue lists as active never
    # enters the watch at all (cross filter above), but a market listed on
    # one venue and missing on another must not burn WS retries + REST calls
    # on the venue that lacks it. Seeding here makes both start_streams and
    # refresh_order_books skip those pairs without any network attempt.
    if listed_per_venue:
        seeded = 0
        for venue, venue_listed in listed_per_venue.items():
            for symbol in services.watch_symbols:
                if symbol.name not in venue_listed:
                    try:
                        services.market.mark_unsupported(
                            venue, symbol.name, "not listed (load_markets)"
                        )
                        seeded += 1
                    except Exception:
                        pass
        if seeded:
            logger.info(
                "watchlist_unsupported_seeded",
                extra={"pairs": seeded, "symbols": len(services.watch_symbols)},
            )


async def _build_precision(manager) -> StaticPrecisionProvider:
    """Instrument filters from every venue's load_markets (best effort)."""
    from app.exchanges.profiles import resolve_profile

    by_venue: dict[tuple[str, str], InstrumentFilters] = {}
    by_symbol: dict[str, InstrumentFilters] = {}
    for venue in manager.enabled_ids():
        markets = None
        try:
            markets = await manager.adapter(venue).load_markets()
        except Exception as exc:  # noqa: BLE001 - precision is best-effort
            # Binance DEMO: load_markets on demo host may timeout — fallback only
            # for precision metadata, trading stays DEMO-only.
            profile = None
            try:
                profile = resolve_profile(venue)
            except Exception:
                pass
            if profile is not None and profile.demo_no_load_markets:
                logger.info(
                    "precision_demo_load_failed_fallback_to_prod",
                    extra={"exchange_id": venue, "error": str(exc)[:160]},
                )
                markets = await _load_production_markets_for_precision(venue)
            else:
                logger.debug(
                    "precision_load_failed", extra={"exchange_id": venue, "error": str(exc)[:160]}
                )
                continue
        if not markets:
            # Still nothing — static simulated fallback for core symbols
            markets = _simulated_markets_for_precision(venue, manager)
            if markets:
                logger.warning(
                    "precision_fallback_to_simulated",
                    extra={"exchange_id": venue, "symbols": len(markets)},
                )
        for market in markets or ():
            filters = InstrumentFilters.from_market(market.precision, market.limits)
            by_venue[(venue, market.symbol.name)] = filters
            by_symbol.setdefault(market.symbol.name, filters)
    return StaticPrecisionProvider(by_venue=by_venue, by_symbol=by_symbol)


async def _load_production_markets_for_precision(venue: str):
    """Load production markets for precision only (no trading, no credentials)."""
    try:
        from app.exchanges.base import AdapterOptions
        from app.exchanges.ccxt_adapter import CCXTAdapter
        from app.exchanges.profiles import resolve_profile
        from app.models.exchange import Exchange

        profile = resolve_profile(venue)
        exchange = Exchange(id=profile.id, name=profile.display_name, adapter="ccxt")
        options = AdapterOptions(sandbox=False, enable_rate_limit=True)
        adapter = CCXTAdapter(exchange, credentials=None, options=options, order_gate=None)
        await adapter.open()
        try:
            markets = await adapter.load_markets()
            return markets
        finally:
            await adapter.close()
    except Exception as exc:  # noqa: BLE001 - fallback to simulated
        logger.debug(
            "production_precision_load_failed",
            extra={"exchange_id": venue, "error": str(exc)[:160]},
        )
        return None


def _simulated_markets_for_precision(venue: str, manager):
    """Static fallback using simulated magnitude-appropriate precision.

    Uses the deterministic simulated adapter's precision_for to generate
    filters for the core watchlist symbols when real market metadata is
    unavailable (e.g. Binance DEMO).  This is clearly a fallback, not real
    venue data, but prevents orders being rejected for missing filters.
    """
    try:
        from app.exchanges.simulated import SimulatedExchangeAdapter
        from app.models.market import Market
        from app.models.enums import MarketType
        from app.models.symbol import Symbol

        # Core symbols that must have filters for triangular
        base_currency = "USDT"
        try:
            base_currency = manager._settings.trading.base_currency  # type: ignore[attr-defined]
        except Exception:
            pass
        assets = set()
        try:
            assets.update(manager._settings.arbitrage.triangle_assets)  # type: ignore[attr-defined]
            assets.update(manager._settings.transfer.assets)  # type: ignore[attr-defined]
        except Exception:
            assets.update(["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "LINK", "AVAX", "TRX"])
        symbols = [Symbol(base=a, quote=base_currency) for a in sorted(assets)]
        # Add cross pairs that exist in simulation
        markets = []
        for sym in symbols:
            prec = SimulatedExchangeAdapter.precision_for(sym)
            # Use conservative limits similar to simulation
            from app.models.market import MarketLimits

            markets.append(
                Market(
                    exchange_id=venue,
                    symbol=sym,
                    market_type=MarketType.SPOT,
                    native_symbol=sym.name,
                    active=True,
                    precision=prec,
                    limits=MarketLimits(min_amount=None, min_cost=None),
                )
            )
        return tuple(markets)
    except Exception:
        return None


async def _build_fee_provider(manager, settings):
    """Build fee provider for triangular scanner.

    PAPER: static 10 bps fallback (simulated venues).
    DEMO: try real venue fees via ``fetch_trading_fees``; if unavailable
    use market-fee fallback from ``load_markets``; if still unavailable log
    fallback and assume 10 bps. Never hardcode a new fee — use real data
    when possible, otherwise clearly log fallback.
    """
    from app.strategies.triangular.fees import StaticFeeProvider
    from app.models.symbol import Symbol

    if settings.mode is not TradingMode.DEMO:
        return StaticFeeProvider()
    overrides: dict[str, object] = {}
    probe = Symbol(base="BTC", quote=settings.trading.base_currency)
    for venue in manager.enabled_ids():
        fee = None
        try:
            adapter = manager.adapter(venue)
            fee = await adapter.fetch_trading_fees(probe)
            logger.info(
                "demo_fee_real",
                extra={"exchange_id": venue, "taker_bps": str(fee.taker_bps), "maker_bps": str(fee.maker_bps), "account": fee.is_account_specific},
            )
        except Exception as exc:  # noqa: BLE001 - try market fallback
            try:
                adapter = manager.adapter(venue)
                markets = await adapter.load_markets()
                for m in markets:
                    if m.symbol == probe:
                        fee = m.fees
                        break
                if fee is None and markets:
                    fee = markets[0].fees
                if fee is not None:
                    logger.info(
                        "demo_fee_from_markets",
                        extra={"exchange_id": venue, "taker_bps": str(fee.taker_bps)},
                    )
                else:
                    logger.warning(
                        "demo_fee_fallback",
                        extra={"exchange_id": venue, "error": str(exc)[:160], "fallback_bps": "10"},
                    )
            except Exception as exc2:  # noqa: BLE001
                logger.warning(
                    "demo_fee_fallback",
                    extra={"exchange_id": venue, "error": str(exc2)[:160], "fallback_bps": "10"},
                )
        if fee is not None:
            overrides[venue.lower()] = fee  # type: ignore[assignment]
    if overrides:
        return StaticFeeProvider(overrides=overrides)  # type: ignore[arg-type]
    logger.warning("demo_fee_all_fallback", extra={"fallback_bps": "10"})
    return StaticFeeProvider()


# ---------------------------------------------------------------------------
# Telegram lifecycle (read-only / safe-control operator interface).
# ---------------------------------------------------------------------------

async def _start_telegram(services: AppServices) -> None:
    """Optionally start the Telegram bot as a background task.

    Fail-safe by construction:
    * missing / unconfigured bot  -> no-op (Telegram is optional);
    * empty operator allow-list   -> no-op (fail-closed);
    * startup failure (no network, bad token, httpx missing, ...) -> logged
      and swallowed: trading is NEVER enabled by Telegram failures.
    """
    from app.telegram.runner import TelegramRunner

    if not services.settings.telegram.is_configured:
        return
    if not services.settings.telegram.has_any_operator:
        logger.info("telegram_allow_list_empty_skipping_startup")
        return
    runner = TelegramRunner(services)
    services.telegram_runner = runner
    try:
        await runner.start()
    except Exception as exc:  # noqa: BLE001 - Telegram is optional, never fatal
        logger.warning(
            "telegram_start_failed",
            extra={"error": str(exc)[:200]},
        )
        services.telegram_runner = None


async def _stop_telegram(services: AppServices) -> None:
    """Stop the Telegram bot if it was started; idempotent and quiet."""
    runner = services.telegram_runner
    services.telegram_runner = None
    if runner is None:
        return
    try:
        await runner.stop()
    except Exception as exc:  # noqa: BLE001 - shutdown must stay quiet
        logger.warning(
            "telegram_stop_failed",
            extra={"error": str(exc)[:200]},
        )
