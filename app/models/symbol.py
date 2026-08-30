"""Trading symbol value object."""

from __future__ import annotations

from pydantic import field_validator

from app.errors import ValidationError
from app.models.base import DomainModel

__all__ = ["Symbol"]


class Symbol(DomainModel):
    """Exchange-agnostic instrument identifier, e.g. ``BTC/USDT`` or ``BTC/USDT:USDT``.

    The terminal never passes exchange-native symbols across module boundaries;
    adapters translate between :class:`Symbol` and their native format.
    """

    base: str
    quote: str
    settle: str | None = None

    @field_validator("base", "quote", "settle", mode="before")
    @classmethod
    def _normalise(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip().upper()
        return text or None

    @field_validator("base", "quote")
    @classmethod
    def _require(cls, value: str | None) -> str:
        if not value:
            raise ValueError("base and quote currencies are required")
        return value

    @classmethod
    def parse(cls, raw: str) -> Symbol:
        """Parse ``BASE/QUOTE`` or ``BASE/QUOTE:SETTLE`` notation."""
        text = str(raw).strip().upper()
        if "/" not in text:
            raise ValidationError(f"invalid symbol: {raw!r}", symbol=raw)
        pair, _, settle = text.partition(":")
        base, _, quote = pair.partition("/")
        if not base or not quote:
            raise ValidationError(f"invalid symbol: {raw!r}", symbol=raw)
        return cls(base=base, quote=quote, settle=settle or None)

    @property
    def name(self) -> str:
        return f"{self.base}/{self.quote}" + (f":{self.settle}" if self.settle else "")

    @property
    def is_derivative(self) -> bool:
        return self.settle is not None

    def spot(self) -> Symbol:
        """Spot projection of a derivative symbol."""
        return self if self.settle is None else Symbol(base=self.base, quote=self.quote)

    def __str__(self) -> str:
        return self.name
