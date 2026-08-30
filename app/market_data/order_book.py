"""Order-book mathematics.

Pure functions only — no I/O, no state.  This is the module the arbitrage engine
relies on to answer "can this spread actually be executed for $X?", instead of
naively comparing top-of-book prices.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_DOWN, Decimal

from app.errors import ValidationError
from app.models.base import DEC0
from app.models.enums import OrderSide
from app.models.market_data import ExecutionEstimate, OrderBook, OrderBookLevel

__all__ = [
    "available_liquidity",
    "crossable_liquidity",
    "estimate_for_base_amount",
    "estimate_for_quote_amount",
    "max_executable_notional",
    "vwap",
]

_BPS = Decimal("10000")
_QUANT = Decimal("0.00000001")


def _floor_amount(value: Decimal) -> Decimal:
    """Quantise a tradable size downwards.

    Rounding a size up can make the resulting order cost more than the requested
    budget (and trip the per-trade capital limit), so sizes always round down.
    """
    return value.quantize(_QUANT, rounding=ROUND_DOWN)


def available_liquidity(
    levels: Sequence[OrderBookLevel], *, max_levels: int | None = None
) -> Decimal:
    """Total base amount offered by ``levels``."""
    selected = levels if max_levels is None else levels[:max_levels]
    return sum((level.amount for level in selected), DEC0)


def vwap(levels: Sequence[OrderBookLevel], amount: Decimal) -> Decimal | None:
    """Volume-weighted average price for ``amount``; ``None`` if liquidity is short."""
    if amount <= DEC0:
        return None
    remaining = amount
    quote = DEC0
    for level in levels:
        take = min(remaining, level.amount)
        quote += take * level.price
        remaining -= take
        if remaining <= DEC0:
            return (quote / amount).quantize(_QUANT)
    return None


def estimate_for_base_amount(
    book: OrderBook, side: OrderSide, amount: Decimal
) -> ExecutionEstimate:
    """Walk the book for a base-currency ``amount`` (e.g. 0.5 BTC)."""
    if amount <= DEC0:
        raise ValidationError("amount must be positive", amount=str(amount))

    levels = book.side(side)
    if not levels:
        raise ValidationError(
            f"empty {'ask' if side is OrderSide.BUY else 'bid'} side",
            exchange_id=book.exchange_id,
            symbol=book.symbol.name,
        )

    reference = levels[0].price
    remaining = amount
    quote = DEC0
    consumed = 0

    for level in levels:
        take = min(remaining, level.amount)
        if take <= DEC0:
            continue
        quote += take * level.price
        remaining -= take
        consumed += 1
        if remaining <= DEC0:
            break

    filled = amount - remaining
    average = (quote / filled).quantize(_QUANT) if filled > DEC0 else reference
    return ExecutionEstimate(
        side=side,
        requested_amount=amount,
        filled_amount=filled,
        quote_amount=quote.quantize(_QUANT),
        average_price=average,
        reference_price=reference,
        slippage_bps=_slippage_bps(reference, average, side),
        levels_consumed=consumed,
        is_complete=remaining <= DEC0,
    )


def estimate_for_quote_amount(
    book: OrderBook, side: OrderSide, quote_amount: Decimal
) -> ExecutionEstimate:
    """Walk the book for a quote-currency budget (e.g. "buy BTC for $50 000")."""
    if quote_amount <= DEC0:
        raise ValidationError("quote_amount must be positive", quote_amount=str(quote_amount))

    levels = book.side(side)
    if not levels:
        raise ValidationError(
            f"empty {'ask' if side is OrderSide.BUY else 'bid'} side",
            exchange_id=book.exchange_id,
            symbol=book.symbol.name,
        )

    reference = levels[0].price
    remaining_quote = quote_amount
    filled = DEC0
    spent = DEC0
    consumed = 0

    for level in levels:
        level_quote = level.price * level.amount
        take_quote = min(remaining_quote, level_quote)
        if take_quote <= DEC0:
            continue
        take_base = take_quote / level.price
        filled += take_base
        spent += take_quote
        remaining_quote -= take_quote
        consumed += 1
        if remaining_quote <= DEC0:
            break

    average = (spent / filled).quantize(_QUANT) if filled > DEC0 else reference
    requested_base = _floor_amount(quote_amount / reference)
    return ExecutionEstimate(
        side=side,
        requested_amount=requested_base,
        requested_quote=quote_amount,
        filled_amount=_floor_amount(filled),
        quote_amount=spent.quantize(_QUANT),
        average_price=average,
        reference_price=reference,
        slippage_bps=_slippage_bps(reference, average, side),
        levels_consumed=consumed,
        is_complete=remaining_quote <= DEC0,
    )


def max_executable_notional(
    book: OrderBook, side: OrderSide, *, max_levels: int | None = None
) -> Decimal:
    """Quote-currency volume available on one side of the book."""
    levels = book.side(side)
    selected = levels if max_levels is None else levels[:max_levels]
    return sum((level.notional for level in selected), DEC0).quantize(_QUANT)


def crossable_liquidity(buy_book: OrderBook, sell_book: OrderBook) -> tuple[Decimal, Decimal]:
    """Liquidity that is actually profitable to cross between two books.

    Walks ``buy_book.asks`` against ``sell_book.bids`` while the bid still exceeds
    the ask and returns ``(base_amount, quote_amount_at_buy_prices)``.  Summing a
    whole book side instead would overstate the executable size, because deep
    levels are past the crossing point where the spread is already negative.
    """
    asks, bids = buy_book.asks, sell_book.bids
    ask_index = bid_index = 0
    ask_left = asks[0].amount if asks else DEC0
    bid_left = bids[0].amount if bids else DEC0
    base = quote = DEC0

    while ask_index < len(asks) and bid_index < len(bids):
        ask, bid = asks[ask_index], bids[bid_index]
        if bid.price <= ask.price:
            break
        take = min(ask_left, bid_left)
        if take > DEC0:
            base += take
            quote += take * ask.price
        ask_left -= take
        bid_left -= take
        if ask_left <= DEC0:
            ask_index += 1
            if ask_index < len(asks):
                ask_left = asks[ask_index].amount
        if bid_left <= DEC0:
            bid_index += 1
            if bid_index < len(bids):
                bid_left = bids[bid_index].amount

    return _floor_amount(base), quote.quantize(_QUANT)


def _slippage_bps(reference: Decimal, average: Decimal, side: OrderSide) -> Decimal:
    if reference <= DEC0:
        return DEC0
    delta = average - reference if side is OrderSide.BUY else reference - average
    return (delta / reference * _BPS).quantize(Decimal("0.0001"))
