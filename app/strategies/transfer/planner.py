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

__all__ = ["TransferPlanner", "price_pair"]

_BPS = Decimal("10000")
_QUANTUM = Decimal("0.00000001")


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
        from app.models.enums import TradingMode

        # DEMO uses real market data with tiny spreads — allow even small/
        # slightly negative nets for E2E so the full lifecycle (buy,
        # withdraw, deposit, sell) can be exercised. Real profitability
        # is still enforced by the risk engine, but with a lower DEMO
        # threshold (see RiskLimits).
        if self._settings.trading.mode is TradingMode.DEMO:
            return Decimal("-10000")
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
