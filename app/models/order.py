"""Order domain models."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import Field, computed_field

from app.models.base import DEC0, DomainModel, utc_now
from app.models.enums import MarketType, OrderSide, OrderStatus, OrderType, TimeInForce
from app.models.symbol import Symbol

__all__ = ["Fill", "Order", "OrderRequest"]


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class OrderRequest(DomainModel):
    """Intent to place an order. Produced by execution planning, never by the API."""

    exchange_id: str
    symbol: Symbol
    market_type: MarketType = MarketType.SPOT
    side: OrderSide
    order_type: OrderType = OrderType.LIMIT
    amount: Decimal
    price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.IOC
    reduce_only: bool = False
    client_order_id: str = Field(default_factory=lambda: _new_id("cat"))
    metadata: tuple[tuple[str, str], ...] = ()

    @property
    def notional(self) -> Decimal | None:
        return None if self.price is None else self.price * self.amount


class Fill(DomainModel):
    fill_id: str | None = None
    price: Decimal
    amount: Decimal
    fee: Decimal = DEC0
    fee_currency: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)

    @property
    def notional(self) -> Decimal:
        return self.price * self.amount


class Order(DomainModel):
    """Order as tracked by the terminal (superset of the exchange payload)."""

    id: str = Field(default_factory=lambda: _new_id("ord"))
    exchange_id: str
    exchange_order_id: str | None = None
    client_order_id: str | None = None
    symbol: Symbol
    market_type: MarketType = MarketType.SPOT
    side: OrderSide
    order_type: OrderType = OrderType.LIMIT
    status: OrderStatus = OrderStatus.PENDING
    amount: Decimal
    filled_amount: Decimal = DEC0
    price: Decimal | None = None
    average_price: Decimal | None = None
    fee_paid: Decimal = DEC0
    fee_currency: str | None = None
    fills: tuple[Fill, ...] = ()
    position_id: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def remaining_amount(self) -> Decimal:
        return max(DEC0, self.amount - self.filled_amount)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fill_ratio(self) -> Decimal:
        return DEC0 if self.amount <= DEC0 else self.filled_amount / self.amount

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_open(self) -> bool:
        return not self.status.is_terminal

    @property
    def is_partially_filled(self) -> bool:
        return DEC0 < self.filled_amount < self.amount

    @property
    def filled_notional(self) -> Decimal:
        if self.average_price is not None:
            return self.average_price * self.filled_amount
        return sum((fill.notional for fill in self.fills), DEC0)

    def with_status(self, status: OrderStatus, *, error: str | None = None) -> Order:
        return self.model_copy(
            update={"status": status, "error": error or self.error, "updated_at": utc_now()}
        )
