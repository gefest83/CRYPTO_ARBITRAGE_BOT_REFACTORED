"""Scanning primitives shared by the triangular strategy.

The scanner is decoupled from the exchange layer through two tiny injected
providers:

* :data:`VenueProvider` — where the venue list comes from (the runtime);
* :class:`FeeProvider` — per-venue, per-symbol taker/maker fees.  The static
  fallback knows the common default (10 bps taker); account-specific fees are
  fetched from the venue when available and cached by the runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol, runtime_checkable

from app.models.enums import MarketType
from app.models.market import MarketFees
from app.models.symbol import Symbol

__all__ = [
    "FeeProvider",
    "ScanRequest",
    "StaticFeeProvider",
    "VenueProvider",
]

#: Source of the venue universe (ids only — no exchange names in the strategy).
VenueProvider = Callable[[], Sequence[str]]


@runtime_checkable
class FeeProvider(Protocol):
    def fees_for(self, exchange_id: str, symbol: Symbol, market_type: MarketType) -> MarketFees: ...


@dataclass(frozen=True, slots=True)
class StaticFeeProvider:
    """Fallback fee source: venue-specific overrides on a common default."""

    taker_bps: Decimal = Decimal("10")
    maker_bps: Decimal = Decimal("8")
    overrides: dict[str, MarketFees] = field(default_factory=dict)

    def fees_for(self, exchange_id: str, symbol: Symbol, market_type: MarketType) -> MarketFees:
        return self.overrides.get(exchange_id.strip().lower()) or MarketFees(
            maker_bps=self.maker_bps, taker_bps=self.taker_bps
        )


@dataclass(frozen=True, slots=True)
class ScanRequest:
    """What one scan should look at."""

    #: USDT-quoted symbols whose assets may start/end a cycle.
    symbols: tuple[Symbol, ...]
    #: Notional (quote) spent on the first leg of every cycle.
    notional_quote: Decimal = Decimal("1000")
    #: Minimum net profit (bps); ``None`` falls back to settings.
    min_net_profit_bps: Decimal | None = None
    require_fresh_data: bool = True
    max_results: int = 20
    ttl_ms: int = 1500
