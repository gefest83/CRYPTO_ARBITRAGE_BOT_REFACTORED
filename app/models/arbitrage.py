"""Arbitrage domain models: legs, profit breakdown, opportunity."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import Field, computed_field

from app.models.base import DEC0, DomainModel, utc_now
from app.models.enums import ArbitrageStrategy, MarketType, OpportunityStatus, OrderSide
from app.models.market_data import ExecutionEstimate
from app.models.risk import RiskViolation
from app.models.symbol import Symbol

__all__ = [
    "ArbitrageLeg",
    "ArbitrageOpportunity",
    "ProfitBreakdown",
]


class ArbitrageLeg(DomainModel):
    """One leg of an arbitrage operation, priced against real order-book depth."""

    exchange_id: str
    symbol: Symbol
    market_type: MarketType = MarketType.SPOT
    side: OrderSide
    price: Decimal
    amount: Decimal
    fee_bps: Decimal = DEC0
    estimate: ExecutionEstimate | None = None

    @property
    def notional(self) -> Decimal:
        return self.price * self.amount

    @property
    def slippage_bps(self) -> Decimal:
        return self.estimate.slippage_bps if self.estimate else DEC0

    @property
    def fill_ratio(self) -> Decimal:
        """Fraction of requested amount that was filled."""
        if self.estimate is None:
            return DEC0
        return self.estimate.fill_ratio


class ProfitBreakdown(DomainModel):
    """Full cost decomposition in basis points (1 bps = 0.01%).

    Kept in bps so that spreads stay comparable across notionals; absolute
    quote-currency values are derived from ``notional_quote``.
    """

    notional_quote: Decimal = DEC0
    gross_spread_bps: Decimal = DEC0
    trading_fees_bps: Decimal = DEC0
    slippage_bps: Decimal = DEC0
    transfer_bps: Decimal = DEC0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_costs_bps(self) -> Decimal:
        return self.trading_fees_bps + self.slippage_bps + self.transfer_bps

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_profit_bps(self) -> Decimal:
        return self.gross_spread_bps - self.total_costs_bps

    @property
    def gross_profit_quote(self) -> Decimal:
        return self.notional_quote * self.gross_spread_bps / Decimal("10000")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_profit_quote(self) -> Decimal:
        return self.notional_quote * self.net_profit_bps / Decimal("10000")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_profitable(self) -> bool:
        return self.net_profit_bps > DEC0


class ArbitrageOpportunity(DomainModel):
    """A validated, costed arbitrage possibility.

    The model is venue-agnostic: legs reference exchanges by id only, which is
    what keeps the strategy core free of any hardcoded exchange list.
    """

    id: str = Field(default_factory=lambda: f"opp-{uuid.uuid4().hex[:12]}")
    strategy: ArbitrageStrategy = ArbitrageStrategy.TRIANGLE
    symbol: Symbol
    buy_leg: ArbitrageLeg
    sell_leg: ArbitrageLeg
    profit: ProfitBreakdown = ProfitBreakdown()
    status: OpportunityStatus = OpportunityStatus.DETECTED
    max_notional_quote: Decimal = DEC0
    data_age_ms: float = 0.0
    latency_ms: float = 0.0
    detected_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    notes: tuple[str, ...] = ()
    #: Structured reasons a rejected opportunity did not pass risk.
    risk_violations: tuple[RiskViolation, ...] = ()
    #: Notional actually spent on the first leg.
    size_notional_quote: Decimal = DEC0
    #: Human-readable route description, e.g. "USDT→BTC→ETH→USDT @ binance".
    direction: str | None = None
    #: The full ordered multi-leg route (buy_leg/sell_leg stay first/last).
    legs_route: tuple[ArbitrageLeg, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_profit_bps(self) -> Decimal:
        return self.profit.net_profit_bps

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_profit_quote(self) -> Decimal:
        return self.profit.net_profit_quote

    @computed_field  # type: ignore[prop-decorator]
    @property
    def exchange_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(leg.exchange_id for leg in (self.buy_leg, self.sell_leg)))

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and utc_now() >= self.expires_at

    @property
    def is_actionable(self) -> bool:
        return (
            self.profit.is_profitable
            and not self.is_expired
            and self.status in (OpportunityStatus.DETECTED, OpportunityStatus.VALIDATED)
        )

    def with_status(
        self,
        status: OpportunityStatus,
        *,
        note: str | None = None,
        violations: tuple[RiskViolation, ...] = (),
    ) -> ArbitrageOpportunity:
        notes = (*self.notes, note) if note else self.notes
        update: dict[str, object] = {"status": status, "notes": notes}
        if violations:
            update["risk_violations"] = violations
        return self.model_copy(update=update)
