"""Deterministic paper fill simulator.

Walks a live order-book snapshot from the :class:`~app.market_data.store`
the way a taker market order would:

* BUY consumes asks (ascending), SELL consumes bids (descending) — never the
  wrong side;
* a per-leg slippage cap (from settings, not hardcoded) stops the walk when a
  level's price would breach ``reference × (1 ± cap)``;
* partial fills are *possible* (thin books, tight cap) and reported honestly
  through ``fill_ratio`` — the executor routes them into recovery instead of
  pretending the leg filled;
* the taker fee is charged in quote currency, like most spot venues.

Everything is Decimal; the simulator is deterministic given the same book.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.models.base import DEC0
from app.models.enums import OrderSide
from app.models.market_data import OrderBook
from app.models.symbol import Symbol

__all__ = ["FillSimulator", "SimulatedFill"]

_BPS = Decimal("10000")
_QUANTUM = Decimal("0.00000001")


@dataclass(frozen=True, slots=True)
class SimulatedFill:
    """Outcome of one simulated taker fill.

    Fee model (consistent across legs, never cross-currency): the taker fee is
    charged in the currency *received*:

    * BUY — ``filled_amount`` is the base **net of the fee**; ``quote_amount``
      is the gross quote spent;
    * SELL — ``quote_amount`` is the quote **net of the fee**;
      ``filled_amount`` is the full base sold.

    ``fee`` is the fee expressed in the received currency (reporting only —
    it is already reflected in the net amounts).
    """

    side: OrderSide
    symbol: Symbol
    #: BUY: base received net of fee.  SELL: full base sold.
    filled_amount: Decimal
    #: BUY: gross quote spent.  SELL: quote received net of fee.
    quote_amount: Decimal
    #: Average fill price (quote per base), before fees.
    average_price: Decimal
    #: Top-of-book reference price the slippage is measured against.
    reference_price: Decimal
    #: Taker fee in the received currency (already deducted from the nets).
    fee: Decimal
    #: Slippage vs top of book in bps (positive = worse than reference).
    slippage_bps: Decimal
    #: Fraction of the requested size that filled (0..1).
    fill_ratio: Decimal
    #: ``None`` when at least something filled; otherwise the reject reason.
    rejected_reason: str | None

    @property
    def is_rejected(self) -> bool:
        return self.rejected_reason is not None

    @property
    def is_complete(self) -> bool:
        return not self.is_rejected and self.fill_ratio >= Decimal("0.999")


class FillSimulator:
    """Fills paper orders against order-book snapshots."""

    def __init__(
        self,
        *,
        taker_fee_bps: Decimal = Decimal("10"),
        max_slippage_bps: Decimal = Decimal("50"),
    ) -> None:
        self._taker_fee_bps = taker_fee_bps
        self._max_slippage_bps = max_slippage_bps

    @property
    def taker_fee_bps(self) -> Decimal:
        return self._taker_fee_bps

    def simulate(
        self,
        book: OrderBook,
        side: OrderSide,
        *,
        base_amount: Decimal | None = None,
        quote_amount: Decimal | None = None,
    ) -> SimulatedFill:
        """Simulate one taker fill; exactly one of the amounts must be given.

        ``base_amount`` trades a base size (SELL legs), ``quote_amount`` spends
        a quote budget (BUY legs).
        """
        if (base_amount is None) == (quote_amount is None):
            raise ValueError("exactly one of base_amount/quote_amount is required")
        if base_amount is not None and base_amount <= DEC0:
            return self._rejected(book, side, "zero_amount")
        if quote_amount is not None and quote_amount <= DEC0:
            return self._rejected(book, side, "zero_amount")

        levels = book.side(side)
        if not levels:
            return self._rejected(book, side, "empty_book")

        reference = levels[0].price
        direction = 1 if side is OrderSide.BUY else -1
        cap_multiplier = Decimal(1) + direction * self._max_slippage_bps / _BPS
        price_cap = (reference * cap_multiplier).quantize(_QUANTUM)

        filled = DEC0
        quote = DEC0
        if base_amount is not None:
            remaining = base_amount
            for level in levels:
                if side is OrderSide.BUY and level.price > price_cap:
                    break
                if side is OrderSide.SELL and level.price < price_cap:
                    break
                take = min(remaining, level.amount)
                filled += take
                quote += take * level.price
                remaining -= take
                if remaining <= DEC0:
                    break
            if filled <= DEC0:
                return self._rejected(book, side, "insufficient_liquidity_within_slippage_cap")
            requested = base_amount
        else:
            remaining_quote = quote_amount
            for level in levels:
                if side is OrderSide.BUY and level.price > price_cap:
                    break
                if side is OrderSide.SELL and level.price < price_cap:
                    break
                level_quote = level.price * level.amount
                take_quote = min(remaining_quote, level_quote)
                filled += (take_quote / level.price).quantize(_QUANTUM)
                quote += take_quote
                remaining_quote -= take_quote
                if remaining_quote <= DEC0:
                    break
            if filled <= DEC0 or quote <= DEC0:
                return self._rejected(book, side, "insufficient_liquidity_within_slippage_cap")
            requested = quote_amount

        average = (quote / filled).quantize(_QUANTUM)
        slippage = self._slippage_bps(reference, average, side)
        fee_rate = self._taker_fee_bps / _BPS
        if side is OrderSide.BUY:
            # Fee charged in the base received.
            fee = (filled * fee_rate).quantize(_QUANTUM)
            filled_net = (filled - fee).quantize(_QUANTUM)
        else:
            # Fee charged in the quote received.
            fee = (quote * fee_rate).quantize(_QUANTUM)
            quote = (quote - fee).quantize(_QUANTUM)
            filled_net = filled
        fill_ratio = (
            filled / requested
            if base_amount is not None and requested > DEC0
            else min(Decimal(1), quote / requested)
        )
        return SimulatedFill(
            side=side,
            symbol=book.symbol,
            filled_amount=filled_net.quantize(_QUANTUM),
            quote_amount=quote.quantize(_QUANTUM),
            average_price=average,
            reference_price=reference,
            fee=fee,
            slippage_bps=slippage,
            fill_ratio=fill_ratio.quantize(Decimal("0.000001")),
            rejected_reason=None,
        )

    def _rejected(self, book: OrderBook, side: OrderSide, reason: str) -> SimulatedFill:
        levels = book.side(side)
        reference = levels[0].price if levels else DEC0
        return SimulatedFill(
            side=side,
            symbol=book.symbol,
            filled_amount=DEC0,
            quote_amount=DEC0,
            average_price=reference,
            reference_price=reference,
            fee=DEC0,
            slippage_bps=DEC0,
            fill_ratio=DEC0,
            rejected_reason=reason,
        )

    @staticmethod
    def _slippage_bps(reference: Decimal, average: Decimal, side: OrderSide) -> Decimal:
        if reference <= DEC0:
            return DEC0
        delta = average - reference if side is OrderSide.BUY else reference - average
        return (delta / reference * _BPS).quantize(Decimal("0.0001"))
