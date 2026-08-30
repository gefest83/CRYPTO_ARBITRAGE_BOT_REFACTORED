"""Risk domain models: limits, violations, assessment."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field

from app.models.base import DomainModel, utc_now
from app.models.enums import RiskLevel

__all__ = ["RiskAssessment", "RiskLimits", "RiskViolation"]


class RiskLimits(DomainModel):
    """Hard limits enforced before any order leaves the bot.

    Every limit is fail-closed: a trade that cannot be proven within limits is
    rejected.  Defaults are conservative paper-mode numbers; an operator going
    to DEMO/LIVE must set them to match real balances (see ``.env.example``).
    """

    #: Maximum notional (quote currency) of a single trade.
    max_trade_size: Decimal = Decimal("1000")
    #: Minimum net profit (bps) required before execution is allowed.
    min_net_profit_bps: Decimal = Decimal("10")
    #: Maximum aggregated loss (quote currency) per UTC day; breach engages
    #: the kill switch automatically.
    max_daily_loss: Decimal = Decimal("250")
    #: Maximum simultaneously open transfer workflows.
    max_open_transfers: int = 3
    #: Maximum exposure (quote currency) per exchange.
    max_exchange_exposure: Decimal = Decimal("150000")
    #: Maximum exposure (quote currency) per asset across exchanges.
    max_asset_exposure: Decimal = Decimal("750000")
    #: Maximum acceptable estimated slippage (bps).
    max_slippage_bps: Decimal = Decimal("15")
    #: Maximum acceptable market-data age at execution time (ms).
    max_data_age_ms: int = 2500


class RiskViolation(DomainModel):
    """A single broken rule; carries limit vs. actual for the audit log."""

    rule: str
    message: str
    limit: Decimal | None = None
    actual: Decimal | None = None
    #: Venue id / asset / leg the violation is about, when the rule is scoped.
    subject: str | None = None
    severity: RiskLevel = RiskLevel.HIGH


class RiskAssessment(DomainModel):
    """Outcome of a risk evaluation. ``approved`` is the only gate execution reads."""

    approved: bool
    violations: tuple[RiskViolation, ...] = ()
    level: RiskLevel = RiskLevel.LOW
    evaluated_at: datetime = Field(default_factory=utc_now)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(v.message for v in self.violations)
