"""Market data snapshots: tickers, order books, trades, execution estimates."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field, computed_field, field_validator, model_validator

from app.models.base import DEC0, DomainModel, utc_now
from app.models.enums import MarketType, OrderSide
from app.models.symbol import Symbol

__all__ = [
    "ExecutionEstimate",
    "OrderBook",
    "OrderBookLevel",
    "Ticker",
    "Trade",
]

_BPS = Decimal("10000")


class _Quote(DomainModel):
    """Common identity of every market-data snapshot."""

    exchange_id: str
    symbol: Symbol
    market_type: MarketType = MarketType.SPOT
    timestamp: datetime = Field(default_factory=utc_now)
    received_at: datetime = Field(default_factory=utc_now)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def age_ms(self) -> float:
        """Age relative to the exchange timestamp (freshness of the data)."""
        return max(0.0, (utc_now() - self.timestamp).total_seconds() * 1000.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def transport_latency_ms(self) -> float:
        return max(0.0, (self.received_at - self.timestamp).total_seconds() * 1000.0)


class Ticker(_Quote):
    bid: Decimal | None = None
    ask: Decimal | None = None
    last: Decimal | None = None
    bid_volume: Decimal | None = None
    ask_volume: Decimal | None = None
    volume_24h: Decimal | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mid(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread_bps(self) -> Decimal | None:
        mid = self.mid
        if mid is None or mid <= DEC0 or self.bid is None or self.ask is None:
            return None
        return (self.ask - self.bid) / mid * _BPS


class OrderBookLevel(DomainModel):
    price: Decimal
    amount: Decimal

    @property
    def notional(self) -> Decimal:
        return self.price * self.amount


class OrderBook(_Quote):
    """Depth snapshot. ``bids`` descend by price, ``asks`` ascend by price."""

    bids: tuple[OrderBookLevel, ...] = ()
    asks: tuple[OrderBookLevel, ...] = ()
    sequence: int | None = None

    @field_validator("bids")
    @classmethod
    def _sort_bids(cls, levels: tuple[OrderBookLevel, ...]) -> tuple[OrderBookLevel, ...]:
        return tuple(sorted(levels, key=lambda level: level.price, reverse=True))

    @field_validator("asks")
    @classmethod
    def _sort_asks(cls, levels: tuple[OrderBookLevel, ...]) -> tuple[OrderBookLevel, ...]:
        return tuple(sorted(levels, key=lambda level: level.price))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0].price if self.bids else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0].price if self.asks else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mid(self) -> Decimal | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2

    @property
    def is_crossed(self) -> bool:
        return bool(self.bids and self.asks and self.bids[0].price >= self.asks[0].price)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def depth(self) -> int:
        return max(len(self.bids), len(self.asks))

    def side(self, side: OrderSide) -> tuple[OrderBookLevel, ...]:
        """Levels that a taker order of ``side`` would consume."""
        return self.asks if side is OrderSide.BUY else self.bids


class Trade(_Quote):
    price: Decimal
    amount: Decimal
    side: OrderSide | None = None
    trade_id: str | None = None


class ExecutionEstimate(DomainModel):
    """Result of walking an order book for a requested size."""

    side: OrderSide
    requested_amount: Decimal
    filled_amount: Decimal
    quote_amount: Decimal
    average_price: Decimal
    reference_price: Decimal
    slippage_bps: Decimal
    levels_consumed: int
    is_complete: bool
    #: Set when the walk was driven by a quote-currency budget rather than a size.
    requested_quote: Decimal | None = None

    @model_validator(mode="after")
    def _check(self) -> ExecutionEstimate:
        if self.filled_amount < DEC0 or self.quote_amount < DEC0:
            raise ValueError("execution estimate cannot be negative")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fill_ratio(self) -> Decimal:
        """Fraction of the request that was filled.

        For a quote-budget walk the ratio is measured in quote currency, because
        the base-amount request is only an estimate derived from the top of book.
        """
        if self.requested_quote is not None:
            if self.requested_quote <= DEC0:
                return DEC0
            return self.quote_amount / self.requested_quote
        if self.requested_amount <= DEC0:
            return DEC0
        return self.filled_amount / self.requested_amount
