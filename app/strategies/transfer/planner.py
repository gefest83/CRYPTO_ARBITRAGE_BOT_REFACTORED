"""Transfer arbitrage planner: the economics of one cross-venue transfer.

Pure math, no I/O.  Given executable prices (ask where we buy, bid where we
sell), fees and the withdrawal fee, it computes gross/net profit and decides
whether the plan clears the configured minimum.

Buy price  = ask on the source venue      (we take liquidity)
Sell price = bid on the destination venue (we hit the bid)

Both prices must be depth-derived when the caller has books available (see
:func:`plan_from_books`); ticker mids are a last-resort fallback and must
never silently replace book pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.config.settings import Settings
from app.models.base import DEC0
from app.models.enums import OrderSide
from app.models.market_data import OrderBook
from app.models.transfer import TransferPlan

__all__ = ["TransferPlanner", "price_pair", "validate_transfer_books"]

_BPS = Decimal("10000")
_QUANTUM = Decimal("0.00000001")

#: Intra-book spread threshold for Layer A (1000 bps = 10 %).
_MAX_INTRA_SPREAD_BPS = Decimal("1000")


def validate_transfer_books(
    buy_book: OrderBook,
    sell_book: OrderBook,
    *,
    max_gross_divergence_bps: Decimal,
) -> str | None:
    """Pure market-data sanity guard for Transfer (Layers A + B).

    Returns ``None`` when both books are sane, otherwise an explicit
    rejection reason:

    * ``invalid_book`` – missing best ask/bid, empty side, crossed book
      (best bid ≥ best ask), or intra-book spread > 1000 bps.
    * ``gross_divergence`` – cross-venue gross ``|gross| > max`` where
      ``gross = (sell_bid - buy_ask) / buy_ask * 10000``.

    No I/O, no symbol hard-coding, no side effects – fully testable.
    """
    # Layer A: per-book validity
    buy_ask = buy_book.best_ask
    sell_bid = sell_book.best_bid
    if buy_ask is None or sell_bid is None:
        return "invalid_book"
    if not buy_book.asks or not sell_book.bids:
        return "invalid_book"
    if buy_book.is_crossed or sell_book.is_crossed:
        return "invalid_book"
    # best_bid < best_ask must hold (is_crossed already checks bid ≥ ask,
    # but be explicit for clarity)
    if buy_book.best_bid is not None and buy_book.best_bid >= buy_ask:
        return "invalid_book"
    if sell_book.best_bid is not None and sell_book.best_ask is not None:
        if sell_book.best_bid >= sell_book.best_ask:
            return "invalid_book"
    # intra-book spread sanity
    for book in (buy_book, sell_book):
        mid = book.mid
        if mid is None or mid <= DEC0:
            continue
        spread_bps = (book.best_ask - book.best_bid) / mid * _BPS  # type: ignore[operator]
        if spread_bps > _MAX_INTRA_SPREAD_BPS:
            return "invalid_book"

    # Layer B: cross-venue gross divergence
    if buy_ask <= DEC0:
        return "invalid_book"
    gross_bps = (sell_bid - buy_ask) / buy_ask * _BPS
    if abs(gross_bps) > max_gross_divergence_bps:
        return "gross_divergence"
    return None


@dataclass(frozen=True, slots=True)
class PricePair:
    """Executable prices derived from real order books."""

    buy_price: Decimal
    sell_price: Decimal
    #: Depth-derived (True) or fallback pricing (False) — reported honestly.
    from_books: bool
    buy_slippage_bps: Decimal = DEC0
    sell_slippage_bps: Decimal = DEC0


def price_pair(buy_book: OrderBook, sell_book: OrderBook, amount: Decimal) -> PricePair:
    """VWAP price for a base ``amount`` on both venues.

    BUY side walks the source asks; SELL side walks the destination bids.
    When the books are too thin for the whole amount, the achievable amount is
    lower — the planner still prices what the books can carry and reports the
    slippage, and the orchestrator sizes down accordingly.
    """
    from app.market_data.order_book import estimate_for_base_amount

    buy = estimate_for_base_amount(buy_book, OrderSide.BUY, amount)
    sell = estimate_for_base_amount(sell_book, OrderSide.SELL, amount)
    return PricePair(
        buy_price=buy.average_price.quantize(_QUANTUM),
        sell_price=sell.average_price.quantize(_QUANTUM),
        from_books=True,
        buy_slippage_bps=buy.slippage_bps,
        sell_slippage_bps=sell.slippage_bps,
    )


class TransferPlanner:
    """Evaluates transfer plans against the configured minimum profit."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def min_net_profit_bps(self) -> Decimal:
        return self._settings.transfer.min_net_profit_bps

    def evaluate(self, plan: TransferPlan) -> TransferPlan | None:
        """Return ``plan`` when it clears the minimum net profit, else ``None``.

        Fail-closed on nonsense inputs: non-positive amounts or prices never
        produce an "opportunity".
        """
        if plan.amount <= DEC0 or plan.buy_price <= DEC0 or plan.sell_price <= DEC0:
            return None
        if plan.source_exchange == plan.dest_exchange:
            return None
        if plan.net_profit_bps < self.min_net_profit_bps:
            return None
        return plan

    def build(
        self,
        *,
        source_exchange: str,
        dest_exchange: str,
        asset: str,
        network: str,
        amount: Decimal,
        prices: PricePair,
        buy_fee_bps: Decimal,
        sell_fee_bps: Decimal,
        withdrawal_fee: Decimal,
        network_cost_quote: Decimal = DEC0,
    ) -> TransferPlan:
        return TransferPlan(
            source_exchange=source_exchange,
            dest_exchange=dest_exchange,
            asset=asset,
            network=network,
            amount=amount,
            buy_price=prices.buy_price,
            sell_price=prices.sell_price,
            buy_fee_bps=buy_fee_bps,
            sell_fee_bps=sell_fee_bps,
            withdrawal_fee=withdrawal_fee,
            network_cost_quote=network_cost_quote,
            estimated_slippage_bps=prices.buy_slippage_bps + prices.sell_slippage_bps,
        )

    def executable_amount(
        self,
        *,
        requested_amount: Decimal,
        withdrawal_min: Decimal,
        available_quote: Decimal,
        buy_price: Decimal,
        max_amount: Decimal | None = None,
        max_notional: Decimal | None = None,
    ) -> Decimal:
        """Largest tradable amount honouring every constraint (floored, > 0).

        Constraints: the requested amount, the venue's withdrawal minimum, the
        available quote budget, the configured amount cap and the notional cap.
        Returns ``0`` when nothing tradable remains.
        """
        amount = requested_amount
        if max_amount is not None:
            amount = min(amount, max_amount)
        if max_notional is not None and buy_price > DEC0:
            amount = min(amount, (max_notional / buy_price).quantize(_QUANTUM))
        if buy_price > DEC0:
            affordable = (available_quote / buy_price).quantize(_QUANTUM)
            amount = min(amount, affordable)
        amount = min(amount, (requested_amount).quantize(_QUANTUM))
        amount = amount.quantize(_QUANTUM)
        if amount <= DEC0 or amount < withdrawal_min:
            return DEC0
        return amount

    def executable_amount_from_notional(
        self,
        *,
        min_notional: Decimal,
        max_notional: Decimal,
        available_quote: Decimal,
        buy_price: Decimal,
        withdrawal_min: Decimal,
        max_amount: Decimal | None = None,
    ) -> Decimal:
        """USDT-notional sizing: $100-500 → coin amount.

        Caps `max_notional` by `available_quote`, requires `max_notional >=
        min_notional`, derives `amount = max_notional / buy_price` (best
        notional for absolute profit at same bps), honours `max_amount` and
        `withdrawal_min`.  Returns ``0`` when nothing tradable remains.
        """
        effective_max = min(max_notional, available_quote)
        if effective_max < min_notional or buy_price <= DEC0:
            return DEC0
        amount = (effective_max / buy_price).quantize(_QUANTUM)
        if max_amount is not None:
            amount = min(amount, max_amount)
        amount = amount.quantize(_QUANTUM)
        if amount <= DEC0 or amount < withdrawal_min:
            return DEC0
        # Ensure derived notional still >= min (covers max_amount truncation)
        if amount * buy_price < min_notional:
            return DEC0
        return amount
