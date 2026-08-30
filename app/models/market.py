"""Market (tradable instrument on a concrete exchange)."""

from __future__ import annotations

from decimal import Decimal

from app.models.base import DEC0, DomainModel
from app.models.enums import MarketType
from app.models.symbol import Symbol

__all__ = ["Market", "MarketFees", "MarketLimits", "MarketPrecision"]


class MarketFees(DomainModel):
    """Account-specific fee schedule. Never hardcode a global fee constant."""

    maker_bps: Decimal = Decimal("10")
    taker_bps: Decimal = Decimal("10")
    is_account_specific: bool = False


class MarketLimits(DomainModel):
    min_amount: Decimal | None = None
    max_amount: Decimal | None = None
    min_cost: Decimal | None = None
    min_price: Decimal | None = None
    max_price: Decimal | None = None


class MarketPrecision(DomainModel):
    """Price/amount granularity.

    ``price``/``amount`` are decimal *places* (legacy integer form);
    ``price_tick`` / ``amount_step`` are the explicit step sizes.
    When the steps are absent they are derived from the decimal places:
    ``10 ** -decimals``.
    """

    price: int | None = None
    amount: int | None = None
    price_tick: Decimal | None = None
    amount_step: Decimal | None = None

    def resolved_price_tick(self) -> Decimal | None:
        if self.price_tick is not None:
            return self.price_tick
        if self.price is not None:
            return Decimal(1).scaleb(-self.price)
        return None

    def resolved_amount_step(self) -> Decimal | None:
        if self.amount_step is not None:
            return self.amount_step
        if self.amount is not None:
            return Decimal(1).scaleb(-self.amount)
        return None


class Market(DomainModel):
    """A symbol as offered by one exchange, including trading constraints."""

    exchange_id: str
    symbol: Symbol
    market_type: MarketType
    native_symbol: str
    active: bool = True
    fees: MarketFees = MarketFees()
    limits: MarketLimits = MarketLimits()
    precision: MarketPrecision = MarketPrecision()
    contract_size: Decimal | None = None
    inverse: bool = False

    @property
    def key(self) -> str:
        return f"{self.exchange_id}:{self.symbol.name}:{self.market_type.value}"

    @property
    def is_derivative(self) -> bool:
        return self.market_type in (MarketType.FUTURES, MarketType.SWAP)

    def satisfies_min_notional(self, notional: Decimal) -> bool:
        return notional >= (self.limits.min_cost or DEC0)
