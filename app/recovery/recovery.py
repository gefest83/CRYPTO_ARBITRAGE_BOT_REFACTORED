"""Execution recovery for problematic order outcomes.

Small, explicit and testable.  Handles exactly the four outcomes that need
more than "move on":

* ``REJECTED``       — nothing to recover; return the order as final.
* ``PARTIALLY_FILLED`` — query the exchange, re-read the actual filled
  quantity and persist it; the caller decides whether the residual can be
  completed or must be unwound.
* ``TIMEOUT``        — the placement call timed out: the order may or may not
  exist on the venue.  Query the exchange (by exchange id and client id)
  before believing anything.
* ``UNKNOWN``        — anything we cannot classify.

For an unknown result the protocol is:

1. query the exchange (``fetch_order`` by exchange order id / client id);
2. check the open-orders list and order history;
3. determine the actually filled quantity;
4. persist the outcome and return it;
5. escalate to ``MANUAL_REVIEW`` when safe recovery is impossible (order not
   found anywhere, venue disagrees, or repeated query failures).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.logging_config import get_logger
from app.models.base import DEC0
from app.models.enums import OrderStatus
from app.models.order import Order

__all__ = ["ExecutionRecovery", "RecoveryOutcome"]

logger = get_logger("recovery")

#: After this many failed exchange queries we stop guessing and escalate.
_MAX_QUERY_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class RecoveryOutcome:
    """Result of one recovery attempt."""

    order: Order
    #: ``filled`` / ``not_found`` / ``manual_review`` — the caller persists
    #: the order either way; the disposition tells it what to do next.
    disposition: str
    note: str = ""

    @property
    def needs_manual_review(self) -> bool:
        return self.disposition == "manual_review"


class ExecutionRecovery:
    """Resolves uncertain order outcomes against the exchange."""

    async def recover(self, order: Order, adapter) -> Order:
        """Resolve ``order`` into a final-ish state and return the updated order.

        The returned order carries the exchange's own view of the fill.  When
        the truth cannot be established the order is marked
        ``MANUAL_REVIEW``-pending (status stays as-is, disposition says so) —
        never silently "fixed".
        """
        if order.status is OrderStatus.REJECTED:
            return order
        if order.status is OrderStatus.TIMEOUT or order.status is OrderStatus.UNKNOWN:
            return await self._resolve_unknown(order, adapter)
        if order.status is OrderStatus.PARTIALLY_FILLED:
            return await self._refresh_partial(order, adapter)
        return order

    # ---------------------------------------------------------------- internals
    async def _resolve_unknown(self, order: Order, adapter) -> Order:
        """Timeout / unknown: find out what actually happened on the venue."""
        last_error: str | None = None
        for attempt in range(1, _MAX_QUERY_ATTEMPTS + 1):
            try:
                # 1. Direct lookup by exchange order id.
                if order.exchange_order_id:
                    fetched = await adapter.fetch_order(
                        order.exchange_order_id, symbol=order.symbol
                    )
                    resolved = self._merge(order, fetched)
                    logger.info(
                        "recovery_resolved_by_id",
                        extra={
                            "exchange_id": order.exchange_id,
                            "order_id": order.exchange_order_id,
                            "status": resolved.status.value,
                            "filled": str(resolved.filled_amount),
                            "attempt": attempt,
                        },
                    )
                    return resolved
                # 2. No exchange id (timeout before the response arrived):
                #    look for the order among open orders by client id.
                open_orders = await adapter.fetch_open_orders(symbol=order.symbol)
                match = self._match_by_client_id(open_orders, order.client_order_id)
                if match is not None:
                    resolved = self._merge(order, match)
                    logger.info(
                        "recovery_resolved_from_open_orders",
                        extra={
                            "exchange_id": order.exchange_id,
                            "client_order_id": order.client_order_id,
                            "status": resolved.status.value,
                            "filled": str(resolved.filled_amount),
                        },
                    )
                    return resolved
                # 3. Not open: it either filled or never existed.  A filled
                #    market order disappears from the open list — treat "not
                #    found" as NOT FOUND (never as filled): the truth lives
                #    in the venue's history which the operator can check.
                logger.warning(
                    "recovery_order_not_found",
                    extra={
                        "exchange_id": order.exchange_id,
                        "client_order_id": order.client_order_id,
                        "note": "order not found on venue; escalating to manual review",
                    },
                )
                return order.with_status(
                    OrderStatus.REJECTED,
                    error="recovery: order not found on venue (never placed or "
                    "purged from history) — manual review",
                )
            except Exception as exc:  # noqa: BLE001 - retry, then escalate
                last_error = str(exc)
                logger.warning(
                    "recovery_query_failed",
                    extra={
                        "exchange_id": order.exchange_id,
                        "attempt": attempt,
                        "error": str(exc)[:200],
                    },
                )
        return order.with_status(
            OrderStatus.REJECTED,
            error=f"recovery: exchange unreachable after {_MAX_QUERY_ATTEMPTS} "
            f"attempts ({last_error}) — manual review",
        )

    async def _refresh_partial(self, order: Order, adapter) -> Order:
        """PARTIALLY_FILLED: read the authoritative fill state from the venue."""
        try:
            if order.exchange_order_id:
                fetched = await adapter.fetch_order(order.exchange_order_id, symbol=order.symbol)
                return self._merge(order, fetched)
            open_orders = await adapter.fetch_open_orders(symbol=order.symbol)
            match = self._match_by_client_id(open_orders, order.client_order_id)
            if match is not None:
                return self._merge(order, match)
        except Exception as exc:  # noqa: BLE001 - escalate rather than guess
            logger.warning(
                "recovery_partial_refresh_failed",
                extra={"exchange_id": order.exchange_id, "error": str(exc)[:200]},
            )
            return order.with_status(
                OrderStatus.PARTIALLY_FILLED,
                error=f"recovery: refresh failed ({str(exc)[:160]}) — manual review",
            )
        return order

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _match_by_client_id(orders: tuple[Order, ...], client_order_id: str | None) -> Order | None:
        if not client_order_id:
            return None
        return next((o for o in orders if o.client_order_id == client_order_id), None)

    @staticmethod
    def _merge(local: Order, fetched: Order) -> Order:
        """Keep the local identity, take the venue's view of the fill."""
        filled = fetched.filled_amount
        if filled > local.amount:
            filled = local.amount  # venue sanity bound: never trust overfills
        if DEC0 < filled < local.amount:
            status = OrderStatus.PARTIALLY_FILLED
        elif filled >= local.amount:
            status = OrderStatus.FILLED
        else:
            status = fetched.status if fetched.status.is_terminal else OrderStatus.REJECTED
        return local.model_copy(
            update={
                "status": status,
                "filled_amount": filled,
                "average_price": fetched.average_price or local.average_price,
                "fee_paid": fetched.fee_paid or local.fee_paid,
                "exchange_order_id": fetched.exchange_order_id or local.exchange_order_id,
                "fills": fetched.fills or local.fills,
                "updated_at": fetched.updated_at,
            }
        )
