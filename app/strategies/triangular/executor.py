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

Crash safety (H-3/H-4):

* the trade record is persisted as EXECUTING *before* leg 1 is submitted;
* before every venue order a PENDING intent carrying the ``client_order_id``
  is persisted, and the outcome replaces it after the venue answers — a
  restart can always tell that an order *may* exist on the venue;
* on restart :meth:`resume_open_trades` resolves unconfirmed orders through
  recovery and then fails closed: a cycle that already submitted a leg is
  never continued automatically — it lands in MANUAL_REVIEW with explicit
  held-inventory information (only a fully filled 3-leg cycle is completed
  by computing its final P&L).

Venue-order semantics (B-1): for DEMO/LIVE legs the returned
:class:`~app.models.order.Order` is the authoritative result.  A
venue-confirmed terminal order with ``filled_amount > 0`` is a REAL fill of
that amount — never a rejection.  Orders whose final state cannot be
established (UNKNOWN / MANUAL_REVIEW / still open on the venue) never cause
a new order or an unwind: the trade escalates to MANUAL_REVIEW.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
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

#: Callback fired with the fully-built request right before a venue order is
#: submitted (used to persist a PENDING intent with the client_order_id).
SubmitNotifier = Callable[["OrderRequest"], Awaitable[None]]


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
        market=None,
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
        self._market = market

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
        # H-3: the trade must exist in storage BEFORE any order is placed —
        # a crash mid-cycle leaves a resumable record instead of lost state.
        await self._trades.save(trade)
        await self._audit.log(
            "TRIANGLE_EXECUTE_STARTED",
            route_text,
            {"trade_id": trade.id, "venue": trade.exchange_id, "notional": str(spend)},
        )

        try:
            trade = await self._run_cycle(trade, opportunity, legs, spend)
        except ExecutionDisabledError as exc:
            persisted = await self._persisted_or_local(trade)
            trade = self._interrupted(persisted, f"execution disabled: {exc}")
        except Exception as exc:  # noqa: BLE001 - never lose the trade record
            logger.error(
                "triangle_execution_error",
                extra={"trade_id": trade.id, "error": str(exc)},
            )
            persisted = await self._persisted_or_local(trade)
            trade = self._interrupted(persisted, f"{type(exc).__name__}: {exc}")
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

    def _interrupted(self, trade: TradeRecord, reason: str) -> TradeRecord:
        """A cycle aborted between legs.  If any leg already filled we may
        hold inventory on the venue — that is MANUAL_REVIEW, never a plain
        FAILED (which would abandon the held asset)."""
        if self._held_inventory(trade.orders) != "":
            return trade.with_status(
                TradeStatus.MANUAL_REVIEW,
                error=f"{reason}; holding {self._held_inventory(trade.orders)} — "
                "manual review required",
            )
        return trade.with_status(TradeStatus.FAILED, error=reason)

    async def _persisted_or_local(self, trade: TradeRecord) -> TradeRecord:
        """The persisted record is the source of truth: per-leg saves may be
        ahead of the caller's in-memory copy when the cycle aborted."""
        try:
            persisted = await self._trades.get(trade.id)
        except Exception:  # noqa: BLE001 - fall back to the local copy
            return trade
        return persisted if persisted is not None else trade

    # ---------------------------------------------------------------- resume
    async def resume_open_trades(self) -> list[TradeRecord]:
        """H-3/H-4: restart recovery for trades interrupted mid-cycle.

        Loads every EXECUTING trade, resolves unconfirmed orders through
        :class:`ExecutionRecovery` (queries only — never places orders), and
        then fails closed:

        * no order was ever submitted → FAILED (nothing can be on the venue);
        * all three legs venue-confirmed filled → COMPLETED with final P&L;
        * every submitted order venue-confirmed empty → FAILED (nothing held);
        * anything else (a leg may have filled / cannot be confirmed) →
          MANUAL_REVIEW with explicit held-inventory information.  A partial
          cycle is NEVER continued automatically.
        """
        interrupted = await self._trades.list_executing()
        if not interrupted:
            return []
        logger.info("triangle_resume_started", extra={"count": len(interrupted)})
        resumed: list[TradeRecord] = []
        for trade in interrupted:
            try:
                updated = await self._resume_one(trade)
            except Exception as exc:  # noqa: BLE001 - resume must fail closed
                logger.error(
                    "triangle_resume_error",
                    extra={"trade_id": trade.id, "error": str(exc)[:300]},
                )
                updated = trade.with_status(
                    TradeStatus.MANUAL_REVIEW,
                    error=f"resume failed ({type(exc).__name__}: {exc}) — manual review",
                )
            await self._trades.save(updated)
            resumed.append(updated)
        return resumed

    async def _resume_one(self, trade: TradeRecord) -> TradeRecord:
        orders: list[Order] = []
        for payload in trade.orders:
            order = self._order_from_json(payload)
            if order is None:
                return trade.with_status(
                    TradeStatus.MANUAL_REVIEW,
                    error="resume: an order record could not be parsed — manual review",
                )
            orders.append(order)

        if not orders:
            return trade.with_status(
                TradeStatus.FAILED,
                error="interrupted before leg 1 was submitted — nothing on the venue",
            )

        # Resolve every unconfirmed order against its venue (queries only).
        for i, order in enumerate(orders):
            if not self._is_unconfirmed(order):
                continue
            adapter = self._manager.adapter(order.exchange_id)
            orders[i] = await self._recovery.recover(order, adapter)
        trade = trade.model_copy(
            update={
                "orders": tuple(_order_to_json(o) for o in orders),
                "updated_at": utc_now(),
            }
        )

        unconfirmed = [o for o in orders if self._is_unconfirmed(o)]
        if unconfirmed:
            order = unconfirmed[-1]
            return trade.with_status(
                TradeStatus.MANUAL_REVIEW,
                error=f"resume: leg outcome unconfirmed ({order.status.value}) for "
                f"order {order.client_order_id or order.exchange_order_id or '?'} "
                f"on {order.exchange_id} — manual review required",
            )

        confirmed_fills = [o for o in orders if o.is_confirmed_fill]
        if not confirmed_fills:
            return trade.with_status(
                TradeStatus.FAILED,
                error="resume: every submitted leg is confirmed not filled — "
                "nothing was held",
            )

        if len(orders) == 3 and len(confirmed_fills) == 3:
            return self._complete_from_orders(trade, orders)

        held = self._held_inventory(trade.orders)
        return trade.with_status(
            TradeStatus.MANUAL_REVIEW,
            error=f"resume: cycle incomplete after crash — holding {held} on "
            f"{trade.exchange_id}; manual unwind/decision required",
        )

    def _complete_from_orders(self, trade: TradeRecord, orders: list[Order]) -> TradeRecord:
        """Final accounting for a cycle whose three legs all venue-filled."""
        fill1 = _venue_fill_view(orders[0])
        fill2 = _venue_fill_view(orders[1])
        fill3 = _venue_fill_view(orders[2])
        if fill1 is None or fill2 is None or fill3 is None:
            return trade.with_status(
                TradeStatus.MANUAL_REVIEW,
                error="resume: filled legs cannot be valued (missing average "
                "price) — manual review",
            )
        spend = fill1.quote_amount
        proceeds = fill3.quote_amount
        fees = fill1.fee + fill2.fee * fill1.average_price + fill3.fee
        net_profit = proceeds - spend
        return trade.with_status(
            TradeStatus.COMPLETED,
            input_amount=spend,
            output_amount=proceeds.quantize(_QUANTUM),
            fees_quote=fees.quantize(_QUANTUM),
            net_profit=net_profit.quantize(_QUANTUM),
            net_profit_bps=(
                (net_profit / spend * Decimal("10000")).quantize(Decimal("0.01"))
                if spend > DEC0
                else DEC0
            ),
        )

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
        preview = await self._preview_cycle(venue, legs, spend)
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
        fill1, order1 = await self._fill_leg(
            venue, symbols[0], OrderSide.BUY, quote_amount=spend, on_submit=self._notifier(trade)
        )
        trade = self._absorb_order(trade, order1)
        await self._trades.save(trade)
        if self._is_unconfirmed(order1):
            return self._unconfirmed(trade, 1, order1)
        fill1 = fill1 if fill1 is not None else _venue_fill_view(order1)
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
                venue,
                symbols[1],
                OrderSide.BUY,
                quote_amount=held_base,
                on_submit=self._notifier(trade),
            )
        else:
            fill2, order2 = await self._fill_leg(
                venue,
                symbols[1],
                OrderSide.SELL,
                base_amount=held_base,
                on_submit=self._notifier(trade),
            )
        trade = self._absorb_order(trade, order2)
        await self._trades.save(trade)
        if self._is_unconfirmed(order2):
            return self._unconfirmed(trade, 2, order2)
        fill2 = fill2 if fill2 is not None else _venue_fill_view(order2)
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
        fill3, order3 = await self._fill_leg(
            venue, symbols[2], OrderSide.SELL, base_amount=held_y, on_submit=self._notifier(trade)
        )
        trade = self._absorb_order(trade, order3)
        await self._trades.save(trade)
        if self._is_unconfirmed(order3):
            return self._unconfirmed(trade, 3, order3)
        fill3 = fill3 if fill3 is not None else _venue_fill_view(order3)
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
        on_submit: SubmitNotifier | None = None,
    ) -> tuple[SimulatedFill | None, Order]:
        """Fill one leg in the current mode; returns (fill, order).

        ``fill`` is ``None`` for exchange legs — the authoritative view is the
        returned :class:`~app.models.order.Order`.  ``on_submit`` (venue legs
        only) is fired with the final :class:`~app.models.order.OrderRequest`
        immediately before submission so the caller can persist a PENDING
        intent identifying the order by its ``client_order_id``.
        """
        self._guard.ensure_can_trade()  # kill switch re-checked every leg
        book = await self._fresh_book(venue, symbol)

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
        if on_submit is not None:
            # H-3/H-4: persist the intent BEFORE the order exists on the
            # venue, so a crash between submission and persistence is
            # recoverable through the client_order_id.
            await on_submit(request)
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
        if not order.client_order_id:
            # Some venues do not echo the client id; keep the intent identity
            # so the persisted outcome replaces the PENDING placeholder.
            order = order.model_copy(update={"client_order_id": request.client_order_id})
        if order.status in (
            OrderStatus.TIMEOUT,
            OrderStatus.UNKNOWN,
            OrderStatus.PARTIALLY_FILLED,
        ):
            order = await self._recovery.recover(order, adapter)
        return None, order

    # ---------------------------------------------------------------- helpers
    async def _fresh_book(self, venue: str, symbol: Symbol):
        book = self._store.order_book(venue, symbol)
        if book is not None and not self._store.is_stale(self._store.age_ms(book)):
            return book
        # Stale or missing -> one on-demand refresh in DEMO (single attempt, fail-closed if still stale)
        # PAPER/LIVE keep original fail-closed without refresh to avoid changing their behavior.
        if self._mode is TradingMode.DEMO and self._market is not None:
            try:
                await self._market.refresh_order_books([symbol], exchange_ids=[venue])
            except Exception:
                pass
            book = self._store.order_book(venue, symbol)
            if book is not None and not self._store.is_stale(self._store.age_ms(book)):
                return book
        if book is None:
            raise ValueError(f"no order book cached for {venue} {symbol.name}")
        raise ValueError(
            f"stale order book for {venue} {symbol.name} "
            f"({self._store.age_ms(book):.0f} ms) — refusing to execute"
        )

    async def _preview_cycle(self, venue: str, legs, spend: Decimal) -> Decimal | None:
        """Non-destructive walk of the three current books; expected proceeds."""
        try:
            symbols = [leg.symbol for leg in legs]
            fill1 = self._sim.simulate(
                await self._fresh_book(venue, symbols[0]), OrderSide.BUY, quote_amount=spend
            )
            if fill1.is_rejected:
                return None
            side2 = legs[1].side
            if side2 is OrderSide.BUY:
                fill2 = self._sim.simulate(
                    await self._fresh_book(venue, symbols[1]),
                    OrderSide.BUY,
                    quote_amount=fill1.filled_amount,
                )
            else:
                fill2 = self._sim.simulate(
                    await self._fresh_book(venue, symbols[1]),
                    OrderSide.SELL,
                    base_amount=fill1.filled_amount,
                )
            if fill2.is_rejected:
                return None
            held_y = fill2.filled_amount if side2 is OrderSide.BUY else fill2.quote_amount
            fill3 = self._sim.simulate(
                await self._fresh_book(venue, symbols[2]), OrderSide.SELL, base_amount=held_y
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
        """Record a leg outcome; replaces the PENDING placeholder (matched by
        ``client_order_id``) when one exists, otherwise appends."""
        payload = _order_to_json(order)
        orders = list(trade.orders)
        if order.client_order_id:
            for i, existing in enumerate(orders):
                if existing.get("client_order_id") == order.client_order_id:
                    orders[i] = payload
                    break
            else:
                orders.append(payload)
        else:
            orders.append(payload)
        return trade.model_copy(update={"orders": tuple(orders), "updated_at": utc_now()})

    def _notifier(self, trade: TradeRecord) -> SubmitNotifier:
        """Build the on_submit hook that persists a PENDING leg intent."""

        async def _note(request: OrderRequest) -> None:
            placeholder = Order(
                exchange_id=request.exchange_id,
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                amount=request.amount,
                client_order_id=request.client_order_id,
                status=OrderStatus.PENDING,
            )
            updated = self._absorb_order(trade, placeholder)
            await self._trades.save(updated)

        return _note

    # ---------------------------------------------------------------- status
    @staticmethod
    def _is_unconfirmed(order: Order) -> bool:
        """The venue could not establish a final state for this order.

        Unconfirmed orders (still open, partially filled and live, or marked
        MANUAL_REVIEW) must never trigger a new order or an unwind.
        """
        return order.outcome_unconfirmed

    def _unconfirmed(self, trade: TradeRecord, leg: int, order: Order) -> TradeRecord:
        note = (
            f"leg {leg} outcome unconfirmed ({order.status.value}"
            f"{f': {order.error}' if order.error else ''}) — order "
            f"{order.client_order_id or order.exchange_order_id or '?'} on "
            f"{order.exchange_id} may have filled; manual review required"
        )
        logger.warning("triangle_leg_unconfirmed", extra={"trade_id": trade.id, "leg": leg})
        return trade.with_status(TradeStatus.MANUAL_REVIEW, error=note)

    @staticmethod
    def _order_from_json(data: dict) -> Order | None:
        try:
            return Order.model_validate(data)
        except Exception:  # noqa: BLE001 - a corrupt entry must not break resume
            return None

    def _held_inventory(self, order_payloads: tuple[dict, ...]) -> str:
        """Human-readable description of inventory held after confirmed fills."""
        held: list[str] = []
        for payload in order_payloads:
            order = self._order_from_json(payload)
            if order is None:
                continue
            if order.is_confirmed_fill:
                held.append(f"{order.filled_amount} {order.symbol.base} (from {order.symbol.name})")
        return "; ".join(held)

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
                venue,
                unwind_symbol,
                OrderSide.SELL,
                base_amount=unwind_amount,
                on_submit=self._notifier(trade),
            )
            trade = self._absorb_order(trade, order)
            await self._trades.save(trade)
            if self._is_unconfirmed(order):
                return trade.with_status(
                    TradeStatus.MANUAL_REVIEW,
                    error=f"{reason}; unwind order outcome unconfirmed "
                    f"({order.status.value}) — manual review required",
                )
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


def _venue_fill_view(order: Order) -> SimulatedFill | None:
    """B-1: build a fill view from venue-confirmed order evidence.

    Returns ``None`` when the order is not a venue-confirmed fill (no fill
    amount, unconfirmed status, or no average price to value it).  The view
    follows the paper :class:`SimulatedFill` semantics:

    * BUY — ``filled_amount`` is base received NET of a base-denominated fee;
      ``quote_amount`` is the gross quote spent (plus a quote fee);
    * SELL — ``quote_amount`` is quote received NET of a quote fee.

    ``fee`` is expressed in the leg's quote currency for reporting.
    """
    if order.filled_amount <= DEC0:
        return None
    if not order.is_confirmed_fill:
        return None
    average = order.average_price
    if average is None or average <= DEC0:
        return None
    filled = order.filled_amount
    fee = order.fee_paid or DEC0
    fee_in_base = fee if order.fee_currency == order.symbol.base else DEC0
    fee_in_quote = fee if order.fee_currency == order.symbol.quote else DEC0
    fee_quote_report = fee_in_quote + fee_in_base * average
    if order.side is OrderSide.BUY:
        return SimulatedFill(
            side=order.side,
            symbol=order.symbol,
            filled_amount=(filled - fee_in_base).quantize(_QUANTUM),
            quote_amount=(filled * average + fee_in_quote).quantize(_QUANTUM),
            average_price=average,
            reference_price=average,
            fee=fee_quote_report.quantize(_QUANTUM),
            slippage_bps=DEC0,
            fill_ratio=(filled / order.amount if order.amount > DEC0 else Decimal("1")).quantize(
                Decimal("0.000001")
            ),
            rejected_reason=None,
        )
    return SimulatedFill(
        side=order.side,
        symbol=order.symbol,
        filled_amount=filled,
        quote_amount=(filled * average - fee_in_quote).quantize(_QUANTUM),
        average_price=average,
        reference_price=average,
        fee=fee_quote_report.quantize(_QUANTUM),
        slippage_bps=DEC0,
        fill_ratio=(filled / order.amount if order.amount > DEC0 else Decimal("1")).quantize(
            Decimal("0.000001")
        ),
        rejected_reason=None,
    )


def _apply(request: OrderRequest, filters, reference_price: Decimal):
    from app.execution.precision import apply_filters

    return apply_filters(request, filters, reference_price=reference_price)
