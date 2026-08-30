"""Instrument precision filters for order placement.

Every order the executors send must be expressible on its venue:

* ``amount`` is a multiple of the instrument's ``amount_step``;
* a limit ``price`` is a multiple of ``price_tick``;
* the resulting notional clears ``min_cost`` and the amount clears
  ``min_amount``.

Amounts are rounded **down** to the step — never up: rounding an order up
could exceed the capital the plan reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, runtime_checkable

from app.models.market import MarketLimits, MarketPrecision
from app.models.order import OrderRequest

__all__ = [
    "InstrumentFilters",
    "PrecisionProvider",
    "StaticPrecisionProvider",
    "apply_filters",
    "round_step_down",
]

_QUANTUM = Decimal("0.00000001")


@dataclass(frozen=True, slots=True)
class InstrumentFilters:
    """Tradability constraints of one instrument on one venue."""

    amount_step: Decimal | None = None
    price_tick: Decimal | None = None
    min_amount: Decimal | None = None
    min_cost: Decimal | None = None
    #: Linear-contract face value in base units (perps); ``None``/1 means the
    #: amount is already quoted in base units.
    contract_size: Decimal | None = None

    @classmethod
    def from_market(
        cls,
        precision: MarketPrecision | None,
        limits: MarketLimits | None,
        *,
        contract_size: Decimal | None = None,
    ) -> InstrumentFilters:
        precision = precision or MarketPrecision()
        limits = limits or MarketLimits()
        return cls(
            amount_step=precision.resolved_amount_step(),
            price_tick=precision.resolved_price_tick(),
            min_amount=limits.min_amount,
            min_cost=limits.min_cost,
            contract_size=contract_size,
        )


def round_step_down(value: Decimal, step: Decimal | None) -> Decimal:
    """Largest multiple of ``step`` not exceeding ``value`` (exact Decimal)."""
    if step is None or step <= 0:
        return value
    steps = (value / step).to_integral_value(rounding="ROUND_FLOOR")
    return (steps * step).quantize(_QUANTUM)


@runtime_checkable
class PrecisionProvider(Protocol):
    """Venue/instrument -> filters; ``None`` means "no data, do not restrict"."""

    def filters_for(self, exchange_id: str, symbol_name: str) -> InstrumentFilters | None: ...


@dataclass(slots=True)
class StaticPrecisionProvider:
    """In-memory map built once from the venues' ``load_markets`` output.

    Keyed by ``(exchange_id, symbol_name)`` with a fallback map keyed by
    symbol only, so one venue's metadata can serve every simulated venue.
    """

    by_venue: dict[tuple[str, str], InstrumentFilters]
    by_symbol: dict[str, InstrumentFilters]

    def filters_for(self, exchange_id: str, symbol_name: str) -> InstrumentFilters | None:
        key = (exchange_id.lower(), symbol_name)
        if key in self.by_venue:
            return self.by_venue[key]
        return self.by_symbol.get(symbol_name)


def apply_filters(
    request: OrderRequest,
    filters: InstrumentFilters | None,
    *,
    reference_price: Decimal,
) -> tuple[OrderRequest, str | None]:
    """Round ``request`` onto the instrument grid; reject below venue minimums.

    Returns ``(rounded_request, None)`` on success or ``(request, reason)``
    when the (rounded) order cannot exist on this market:
    ``below_min_amount`` / ``below_min_notional``.  Market orders carry no
    price, so the tick rule applies to limit prices only; the *notional*
    check uses ``reference_price`` (the leg's scanned VWAP).
    """
    if filters is None:
        return request, None

    amount = request.amount
    contract_size = filters.contract_size
    if contract_size is not None and contract_size > 1:
        # Whole contracts only: floor(base / contract_size) × contract_size.
        contracts = (amount / contract_size).to_integral_value(rounding="ROUND_FLOOR")
        amount = (contracts * contract_size).quantize(_QUANTUM)

    amount = round_step_down(amount, filters.amount_step)
    if filters.min_amount is not None and amount < filters.min_amount:
        return request, "below_min_amount"
    notional = amount * reference_price
    if filters.min_cost is not None and notional < filters.min_cost:
        return request, "below_min_notional"
    if amount <= 0:
        # Sub-step / sub-contract residue: nothing tradable would be sent.
        return request, "below_min_amount"

    updates: dict[str, object] = {"amount": amount}
    if request.price is not None and filters.price_tick is not None:
        updates["price"] = round_step_down(request.price, filters.price_tick)
    return request.model_copy(update=updates), None
