"""Triangular arbitrage executor.

Executes a validated 3-leg opportunity (``USDT → X → Y → USDT`` on ONE venue)
sequentially, in the current trading mode:

* **PAPER** — every leg is filled by :class:`~app.execution.fill_simulator`
  against the live order-book snapshot; funds move inside the paper wallet.
* **DEMO / LIVE** — real market orders through the venue adapter, wrapped in
  per-leg timeouts; REJECTED / PARTIALLY_FILLED / TIMEOUT / UNKNOWN outcomes
  go through :class:`~app.recovery.ExecutionRecovery`.

Safety properties:

* the kill switch is re-checked before *every* leg (not just the first);
* every book is freshness-checked immediately before it is used;
* a non-destructive preview of the whole cycle must clear
  ``min_net_at_execute_bps`` before any order is placed — if profitability is
  uncertain the cycle is abandoned;
* a partial fill on one leg rescales the following legs to what is actually
  held (never trade amounts you do not have);
* mid-cycle failures unwind what is held back to USDT; if the unwind itself
  fails the trade lands in MANUAL_REVIEW — never silently ignored.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from app.config.logging_config import get_logger
from app.config.settings import Settings
from app.errors import ExecutionDisabledError
from app.execution.fill_simulator import FillSimulator, SimulatedFill
from app.execution.guard import ExecutionGuard
from app.execution.paper_wallet import PaperWallet
from app.execution.precision import PrecisionProvider
from app.models.arbitrage import ArbitrageOpportunity
from app.models.base import DEC0, utc_now
from app.models.enums import (
    ArbitrageStrategy,
    OrderSide,
    OrderStatus,
    OrderType,
    TradeStatus,
    TradingMode,
)
from app.models.order import Order, OrderRequest
from app.models.symbol import Symbol
from app.models.trade import TradeRecord
from app.recovery import ExecutionRecovery
from app.storage.repositories import AuditLogRepository, TradeRepository

__all__ = ["TriangleExecutor"]

logger = get_logger("strategies.triangular.executor")

_QUANTUM = Decimal("0.00000001")


class TriangleExecutor:
    """Sequential 3-leg executor for one venue."""

    def __init__(
        self,
        *,
        settings: Settings,
        store,
        manager,
        guard: ExecutionGuard,
        recovery: ExecutionRecovery,
        fill_simulator: FillSimulator,
        precision: PrecisionProvider | None = None,
        trade_repo: TradeRepository,
        audit: AuditLogRepository,
        paper_wallets: dict[str, PaperWallet] | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._manager = manager
        self._guard = guard
        self._recovery = recovery
        self._sim = fill_simulator
        self._precision = precision
        self._trades = trade_repo
        self._audit = audit
        self._wallets = paper_wallets or {}
        self._mode = settings.mode

    # ---------------------------------------------------------------- public
    async def execute(self, opportunity: ArbitrageOpportunity) -> TradeRecord:
        """Execute one opportunity; always returns a persisted TradeRecord."""
        legs = opportunity.legs_route
        spend = opportunity.size_notional_quote
        route_text = opportunity.direction or "USDT->?->?->USDT"
        trade = TradeRecord(
            strategy=ArbitrageStrategy.TRIANGLE,
            mode=self._mode,
            exchange_id=opportunity.buy_leg.exchange_id,
            route=route_text,
            symbols=tuple(leg.symbol.name for leg in legs),
            input_amount=spend,
            status=TradeStatus.EXECUTING,
        )
        await self._audit.log(
            "TRIANGLE_EXECUTE_STARTED",
            route_text,
            {"trade_id": trade.id, "venue": trade.exchange_id, "notional": str(spend)},
        )

        try:
            trade = await self._run_cycle(trade, opportunity, legs, spend)
        except ExecutionDisabledError as exc:
            trade = trade.with_status(TradeStatus.FAILED, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - never lose the trade record
            logger.error(
                "triangle_execution_error",
                extra={"trade_id": trade.id, "error": str(exc)},
            )
            trade = trade.with_status(TradeStatus.FAILED, error=f"{type(exc).__name__}: {exc}")

        await self._trades.save(trade)
        await self._audit.log(
            "TRIANGLE_EXECUTE_FINISHED",
            route_text,
            {
                "trade_id": trade.id,
                "status": trade.status.value,
                "net_profit": str(trade.net_profit),
            },
        )
        return trade

    # ---------------------------------------------------------------- cycle
    async def _run_cycle(
        self,
        trade: TradeRecord,
        opportunity: ArbitrageOpportunity,
        legs,
        spend: Decimal,
    ) -> TradeRecord:
        venue = trade.exchange_id
        symbols = [leg.symbol for leg in legs]

        # --- preview: expected value must clear the execution-time floor ---
        preview = self._preview_cycle(venue, legs, spend)
        if preview is None:
            return trade.with_status(
                TradeStatus.FAILED,
                error="pre-trade preview failed (stale data, empty book or "
                "slippage cap) — profitability uncertain",
            )
        expected_net = preview - spend
        floor_bps = Decimal(str(self._settings.execution.min_net_at_execute_bps))
        if spend > DEC0 and (expected_net / spend) * Decimal("10000") < floor_bps:
            return trade.with_status(
                TradeStatus.FAILED,
                error=f"pre-trade preview net {expected_net} below execution "
                f"floor ({floor_bps} bps) — abandoned before any leg",
            )

        # --- capital check ---
        await self._check_capital(venue, spend)

        # --- leg 1: BUY X with USDT ---
        fill1, order1 = await self._fill_leg(venue, symbols[0], OrderSide.BUY, quote_amount=spend)
        trade = self._absorb_order(trade, order1)
        if fill1 is None or fill1.filled_amount <= DEC0:
            return self._fail(trade, "leg 1 (buy) rejected or empty fill")
        held_base = fill1.filled_amount
        # All fees are converted into USDT for reporting: leg 1's fee is in
        # USDT, leg 2's fee is in X (priced at leg 1's average), leg 3's in USDT.
        total_fees = fill1.fee
        total_slip = fill1.slippage_bps
        actual_spend = fill1.quote_amount
        trade = trade.model_copy(update={"input_amount": actual_spend})

        # --- leg 2: X -> Y on the cross book ---
        side2 = legs[1].side
        if side2 is OrderSide.BUY:
            # Buy Y paying X: the cross book's quote currency is X.
            fill2, order2 = await self._fill_leg(
                venue, symbols[1], OrderSide.BUY, quote_amount=held_base
            )
        else:
            fill2, order2 = await self._fill_leg(
                venue, symbols[1], OrderSide.SELL, base_amount=held_base
            )
        trade = self._absorb_order(trade, order2)
        if fill2 is None or fill2.filled_amount <= DEC0:
            return await self._fail_with_unwind(
                trade,
                venue,
                symbols[0],
                held_base,
                "leg 2 (cross) rejected or empty fill",
                total_fees,
                actual_spend,
            )
        held_y = fill2.filled_amount if side2 is OrderSide.BUY else fill2.quote_amount
        # Leg 2's fee is denominated in the cross book's quote currency (X);
        # value it in USDT at leg 1's average price.
        total_fees += fill2.fee * fill1.average_price
        total_slip += fill2.slippage_bps

        # --- leg 3: SELL Y for USDT ---
        fill3, order3 = await self._fill_leg(venue, symbols[2], OrderSide.SELL, base_amount=held_y)
        trade = self._absorb_order(trade, order3)
        if fill3 is None or fill3.filled_amount <= DEC0 or fill3.quote_amount <= DEC0:
            return await self._fail_with_unwind(
                trade,
                venue,
                symbols[2],
                held_y,
                "leg 3 (sell) rejected or empty fill",
                total_fees,
                actual_spend,
            )
        total_fees += fill3.fee
        total_slip += fill3.slippage_bps
        proceeds = fill3.quote_amount

        # Fees are already reflected in the net amounts (fee-on-received
        # model), so the net profit is simply proceeds minus spend.
        net_profit = proceeds - actual_spend
        net_bps = (
            (net_profit / actual_spend * Decimal("10000")).quantize(Decimal("0.01"))
            if actual_spend > DEC0
            else DEC0
        )
        return trade.with_status(
            TradeStatus.COMPLETED,
            output_amount=proceeds.quantize(_QUANTUM),
            fees_quote=total_fees.quantize(_QUANTUM),
            slippage_bps=total_slip.quantize(Decimal("0.0001")),
            net_profit=net_profit.quantize(_QUANTUM),
            net_profit_bps=net_bps,
        )

    # ---------------------------------------------------------------- legs
    async def _fill_leg(
        self,
        venue: str,
        symbol: Symbol,
        side: OrderSide,
        *,
        base_amount: Decimal | None = None,
        quote_amount: Decimal | None = None,
    ) -> tuple[SimulatedFill | None, Order]:
        """Fill one leg in the current mode; returns (fill, order).

        ``fill`` is ``None`` for exchange legs — the authoritative view is the
        returned :class:`~app.models.order.Order`.
        """
        self._guard.ensure_can_trade()  # kill switch re-checked every leg
        book = self._fresh_book(venue, symbol)

        if self._mode is TradingMode.PAPER:
            fill = self._sim.simulate(
                book, side, base_amount=base_amount, quote_amount=quote_amount
            )
            if fill.is_rejected:
                order = Order(
                    exchange_id=venue,
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.REJECTED,
                    amount=base_amount or DEC0,
                    error=fill.rejected_reason,
                )
                return None, order
            self._apply_paper_wallet(venue, side, symbol, fill)
            order = Order(
                exchange_id=venue,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                status=OrderStatus.FILLED,
                amount=fill.filled_amount,
                filled_amount=fill.filled_amount,
                average_price=fill.average_price,
                fee_paid=fill.fee,
                fee_currency=symbol.base if side is OrderSide.BUY else symbol.quote,
            )
            return fill, order

        # DEMO / LIVE: a real market order through the adapter.
        amount = base_amount
        if amount is None:
            # Quote-budget buy: estimate the base size from the current book.
            estimate_book = book
            amount = (
                quote_amount / estimate_book.asks[0].price
                if side is OrderSide.BUY and estimate_book.asks
                else DEC0
            ).quantize(_QUANTUM)
            if amount <= DEC0:
                return None, Order(
                    exchange_id=venue,
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.REJECTED,
                    error="cannot size leg from empty book",
                )
        filters = self._precision.filters_for(venue, symbol.name) if self._precision else None
        request = OrderRequest(
            exchange_id=venue,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            amount=amount,
        )
        if filters is not None:
            reference = (
                book.asks[0].price if side is OrderSide.BUY and book.asks else book.bids[0].price
            )
            request, reason = _apply(request, filters, reference)
            if reason is not None:
                return None, Order(
                    exchange_id=venue,
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    status=OrderStatus.REJECTED,
                    amount=amount,
                    error=f"precision filter: {reason}",
                )
        adapter = self._manager.adapter(venue)
        try:
            order = await asyncio.wait_for(
                adapter.create_order(request),
                timeout=self._settings.execution.leg_timeout_seconds,
            )
        except TimeoutError:
            order = Order(
                exchange_id=venue,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                amount=request.amount,
                client_order_id=request.client_order_id,
                status=OrderStatus.TIMEOUT,
                error="placement timed out",
            )
        except Exception as exc:  # noqa: BLE001 - exchange failure -> unknown
            order = Order(
                exchange_id=venue,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                amount=request.amount,
                client_order_id=request.client_order_id,
                status=OrderStatus.UNKNOWN,
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
        if order.status in (OrderStatus.TIMEOUT, OrderStatus.UNKNOWN, OrderStatus.PARTIALLY_FILLED):
            order = await self._recovery.recover(order, adapter)
        return None, order

    # ---------------------------------------------------------------- helpers
    def _fresh_book(self, venue: str, symbol: Symbol):
        book = self._store.order_book(venue, symbol)
        if book is None:
            raise ValueError(f"no order book cached for {venue} {symbol.name}")
        if self._store.is_stale(self._store.age_ms(book)):
            raise ValueError(
                f"stale order book for {venue} {symbol.name} "
                f"({self._store.age_ms(book):.0f} ms) — refusing to execute"
            )
        return book

    def _preview_cycle(self, venue: str, legs, spend: Decimal) -> Decimal | None:
        """Non-destructive walk of the three current books; expected proceeds."""
        try:
            symbols = [leg.symbol for leg in legs]
            fill1 = self._sim.simulate(
                self._fresh_book(venue, symbols[0]), OrderSide.BUY, quote_amount=spend
            )
            if fill1.is_rejected:
                return None
            side2 = legs[1].side
            if side2 is OrderSide.BUY:
                fill2 = self._sim.simulate(
                    self._fresh_book(venue, symbols[1]),
                    OrderSide.BUY,
                    quote_amount=fill1.filled_amount,
                )
            else:
                fill2 = self._sim.simulate(
                    self._fresh_book(venue, symbols[1]),
                    OrderSide.SELL,
                    base_amount=fill1.filled_amount,
                )
            if fill2.is_rejected:
                return None
            held_y = fill2.filled_amount if side2 is OrderSide.BUY else fill2.quote_amount
            fill3 = self._sim.simulate(
                self._fresh_book(venue, symbols[2]), OrderSide.SELL, base_amount=held_y
            )
            if fill3.is_rejected:
                return None
            return fill3.quote_amount
        except ValueError:
            return None

    async def _check_capital(self, venue: str, spend: Decimal) -> None:
        """Ensure the quote amount is available before the first leg.

        PAPER debits the in-memory wallet (fail closed on insufficient funds);
        DEMO/LIVE asks the venue for its real free balance.
        """
        if self._mode is TradingMode.PAPER:
            wallet = self._wallets.get(venue)
            if wallet is not None:
                wallet.debit("USDT", spend)  # raises when insufficient
            return
        adapter = self._manager.adapter(venue)
        snapshot = await adapter.fetch_balances()
        quote = self._settings.trading.base_currency
        free = snapshot.free_of(quote)
        if free < spend:
            raise ValueError(f"insufficient {quote} on {venue}: need {spend}, free {free}")

    def _apply_paper_wallet(
        self, venue: str, side: OrderSide, symbol: Symbol, fill: SimulatedFill
    ) -> None:
        wallet = self._wallets.get(venue)
        if wallet is None:
            return
        if side is OrderSide.BUY:
            wallet.debit(symbol.quote, fill.quote_amount)
            wallet.credit(symbol.base, fill.filled_amount)
        else:
            wallet.debit(symbol.base, fill.filled_amount)
            wallet.credit(symbol.quote, fill.quote_amount)

    def _absorb_order(self, trade: TradeRecord, order: Order) -> TradeRecord:
        orders = (*trade.orders, _order_to_json(order))
        return trade.model_copy(update={"orders": orders, "updated_at": utc_now()})

    def _fail(self, trade: TradeRecord, reason: str) -> TradeRecord:
        logger.warning("triangle_leg_failed", extra={"trade_id": trade.id, "reason": reason})
        return trade.with_status(TradeStatus.FAILED, error=reason)

    async def _fail_with_unwind(
        self,
        trade: TradeRecord,
        venue: str,
        unwind_symbol: Symbol,
        unwind_amount: Decimal,
        reason: str,
        fees_so_far: Decimal,
        spend: Decimal,
    ) -> TradeRecord:
        """A later leg failed: sell what we hold back to USDT (mitigation)."""
        try:
            fill, order = await self._fill_leg(
                venue, unwind_symbol, OrderSide.SELL, base_amount=unwind_amount
            )
            trade = self._absorb_order(trade, order)
            # Net-of-fee proceeds (fee-on-received model).
            recovered = (
                fill.quote_amount
                if fill is not None
                else order.filled_amount * (order.average_price or DEC0)
            )
            fees = fees_so_far
            if recovered <= DEC0:
                return trade.with_status(
                    TradeStatus.MANUAL_REVIEW,
                    error=f"{reason}; unwind ALSO failed — manual review required",
                )
            # Fees are embedded in the net amounts; realised net is proceeds
            # minus spend.
            net = recovered - spend
            return trade.with_status(
                TradeStatus.COMPLETED,
                output_amount=recovered.quantize(_QUANTUM),
                fees_quote=fees.quantize(_QUANTUM),
                net_profit=net.quantize(_QUANTUM),
                net_profit_bps=(
                    (net / spend * Decimal("10000")).quantize(Decimal("0.01"))
                    if spend > DEC0
                    else DEC0
                ),
                error=reason,
            )
        except Exception as exc:  # noqa: BLE001 - unwind failure is manual review
            return trade.with_status(
                TradeStatus.MANUAL_REVIEW,
                error=f"{reason}; unwind error: {exc}",
            )


def _order_to_json(order: Order) -> dict:
    import json

    computed = set(order.model_computed_fields)
    return json.loads(order.model_dump_json(exclude=computed))


def _apply(request: OrderRequest, filters, reference_price: Decimal):
    from app.execution.precision import apply_filters

    return apply_filters(request, filters, reference_price=reference_price)
