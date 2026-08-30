"""Balance domain models."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field, computed_field, field_validator

from app.models.base import DEC0, DomainModel, utc_now

__all__ = ["Balance", "BalanceSnapshot"]


class Balance(DomainModel):
    """Balance of one asset on one exchange."""

    exchange_id: str
    asset: str
    free: Decimal = DEC0
    used: Decimal = DEC0
    valuation_currency: str | None = None
    valuation: Decimal | None = None

    @field_validator("asset", mode="before")
    @classmethod
    def _upper(cls, value: str) -> str:
        return str(value).strip().upper()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> Decimal:
        return self.free + self.used

    @property
    def is_empty(self) -> bool:
        return self.total <= DEC0


class BalanceSnapshot(DomainModel):
    """All balances of one exchange at a point in time."""

    exchange_id: str
    balances: tuple[Balance, ...] = ()
    timestamp: datetime = Field(default_factory=utc_now)

    def get(self, asset: str) -> Balance | None:
        needle = asset.strip().upper()
        return next((b for b in self.balances if b.asset == needle), None)

    def free_of(self, asset: str) -> Decimal:
        balance = self.get(asset)
        return balance.free if balance else DEC0

    @property
    def non_empty(self) -> tuple[Balance, ...]:
        return tuple(b for b in self.balances if not b.is_empty)

    @property
    def total_valuation(self) -> Decimal:
        return sum((b.valuation for b in self.balances if b.valuation is not None), DEC0)
