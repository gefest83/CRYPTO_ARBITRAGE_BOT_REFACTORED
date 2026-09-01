"""Pre-trade risk rules.

Every rule is a pure function of :class:`RiskContext` against
:class:`~app.models.risk.RiskLimits` and returns a
:class:`~app.models.risk.RiskViolation` when breached.  A rule that raises is
converted by the engine into a CRITICAL violation — risk fails closed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from app.models.base import DEC0
from app.models.enums import ArbitrageStrategy, RiskLevel
from app.models.risk import RiskLimits, RiskViolation

__all__ = [
    "DEFAULT_RISK_RULES",
    "RiskContext",
    "RiskRule",
]


@dataclass(frozen=True, slots=True)
class RiskContext:
    """Everything the rules need about one candidate trade.

    The caller (runtime) must fill every field from live state — defaults are
    deliberately conservative so a half-built context fails closed instead of
    silently disabling a rule.
    """

    strategy: ArbitrageStrategy
    #: Notional of the trade in quote currency (USDT).
    notional_quote: Decimal
    #: Expected net profit in bps after fees/slippage/transfer costs.
    net_profit_bps: Decimal
    #: Estimated execution slippage in bps.
    slippage_bps: Decimal
    #: Age of the market data the decision is based on (ms).
    data_age_ms: float
    #: Realised P&L of the current UTC day (quote currency).
    daily_pnl: Decimal
    #: Open transfer workflows right now.
    open_transfers: int
    #: Post-trade exposure per exchange in quote currency.
    exchange_exposure: Mapping[str, Decimal] | None = None
    #: Post-trade exposure per asset in quote currency.
    asset_exposure: Mapping[str, Decimal] | None = None
    #: Kill switch state (mirrored from the guard for evaluation).
    kill_switch_engaged: bool = False


class RiskRule:
    """Base class: ``name`` is the stable machine-readable rule code."""

    name: str = "rule"

    def check(self, context: RiskContext, limits: RiskLimits) -> RiskViolation | None:
        raise NotImplementedError


def _violation(
    rule: str,
    message: str,
    *,
    limit: Decimal | None = None,
    actual: Decimal | None = None,
    severity: RiskLevel = RiskLevel.HIGH,
    subject: str | None = None,
) -> RiskViolation:
    return RiskViolation(
        rule=rule,
        message=message,
        limit=limit,
        actual=actual,
        severity=severity,
        subject=subject,
    )


class KillSwitchRule(RiskRule):
    """The kill switch stops ALL new execution immediately."""

    name = "kill_switch"

    def check(self, context: RiskContext, limits: RiskLimits) -> RiskViolation | None:
        if context.kill_switch_engaged:
            return _violation(
                self.name,
                "kill switch is engaged: new execution is stopped",
                severity=RiskLevel.CRITICAL,
            )
        return None


class MaxDailyLossRule(RiskRule):
    """Today's realised loss must stay inside the configured bound."""

    name = "max_daily_loss"

    def check(self, context: RiskContext, limits: RiskLimits) -> RiskViolation | None:
        loss = -context.daily_pnl
        if loss >= limits.max_daily_loss:
            return _violation(
                self.name,
                f"daily loss {loss} reached the limit {limits.max_daily_loss}",
                limit=limits.max_daily_loss,
                actual=loss,
                severity=RiskLevel.CRITICAL,
            )
        return None


class MinNetProfitRule(RiskRule):
    """Never execute an uncertain or negative-margin trade."""

    name = "min_net_profit"

    def check(self, context: RiskContext, limits: RiskLimits) -> RiskViolation | None:
        if context.net_profit_bps < limits.min_net_profit_bps:
            return _violation(
                self.name,
                f"net profit {context.net_profit_bps} bps below minimum "
                f"{limits.min_net_profit_bps} bps",
                limit=limits.min_net_profit_bps,
                actual=context.net_profit_bps,
                severity=RiskLevel.MEDIUM,
            )
        return None


class MaxTradeSizeRule(RiskRule):
    """Hard cap on the notional of a single trade."""

    name = "max_trade_size"

    def check(self, context: RiskContext, limits: RiskLimits) -> RiskViolation | None:
        if context.notional_quote > limits.max_trade_size:
            return _violation(
                self.name,
                f"trade size {context.notional_quote} exceeds maximum {limits.max_trade_size}",
                limit=limits.max_trade_size,
                actual=context.notional_quote,
            )
        return None


class MaxSlippageRule(RiskRule):
    """Estimated slippage must stay within tolerance."""

    name = "max_slippage"

    def check(self, context: RiskContext, limits: RiskLimits) -> RiskViolation | None:
        if context.slippage_bps > limits.max_slippage_bps:
            return _violation(
                self.name,
                f"slippage {context.slippage_bps} bps exceeds maximum "
                f"{limits.max_slippage_bps} bps",
                limit=limits.max_slippage_bps,
                actual=context.slippage_bps,
            )
        return None


class MaxOpenTransfersRule(RiskRule):
    """Cap the number of simultaneously in-flight transfer workflows."""

    name = "max_open_transfers"

    def check(self, context: RiskContext, limits: RiskLimits) -> RiskViolation | None:
        if context.strategy is not ArbitrageStrategy.TRANSFER:
            return None
        if context.open_transfers >= limits.max_open_transfers:
            return _violation(
                self.name,
                f"{context.open_transfers} open transfers reached the maximum "
                f"{limits.max_open_transfers}",
                limit=Decimal(str(limits.max_open_transfers)),
                actual=Decimal(str(context.open_transfers)),
            )
        return None


class MaxExchangeExposureRule(RiskRule):
    """Per-exchange exposure cap (post-trade)."""

    name = "max_exchange_exposure"

    def check(self, context: RiskContext, limits: RiskLimits) -> RiskViolation | None:
        exposure = context.exchange_exposure
        if not exposure:
            return None
        for exchange_id, value in exposure.items():
            if value > limits.max_exchange_exposure:
                return _violation(
                    self.name,
                    f"exchange {exchange_id} exposure {value} exceeds maximum "
                    f"{limits.max_exchange_exposure}",
                    limit=limits.max_exchange_exposure,
                    actual=value,
                    subject=exchange_id,
                )
        return None


class MaxAssetExposureRule(RiskRule):
    """Per-asset exposure cap across exchanges (post-trade)."""

    name = "max_asset_exposure"

    def check(self, context: RiskContext, limits: RiskLimits) -> RiskViolation | None:
        exposure = context.asset_exposure
        if not exposure:
            return None
        for asset, value in exposure.items():
            if value > limits.max_asset_exposure:
                return _violation(
                    self.name,
                    f"asset {asset} exposure {value} exceeds maximum {limits.max_asset_exposure}",
                    limit=limits.max_asset_exposure,
                    actual=value,
                    subject=asset,
                )
        return None


class MaxDataAgeRule(RiskRule):
    """Never execute on stale market data — a stale price is a fake spread."""

    name = "max_data_age"

    def check(self, context: RiskContext, limits: RiskLimits) -> RiskViolation | None:
        if context.data_age_ms > limits.max_data_age_ms:
            # ``inf`` (e.g. a materially future timestamp, H-9, or missing
            # data) is stale but cannot be converted to a finite int.
            actual = (
                Decimal(str(int(context.data_age_ms)))
                if math.isfinite(context.data_age_ms)
                else None
            )
            return _violation(
                self.name,
                f"market data age {context.data_age_ms:.0f} ms exceeds maximum "
                f"{limits.max_data_age_ms} ms",
                limit=Decimal(str(limits.max_data_age_ms)),
                actual=actual,
                severity=RiskLevel.CRITICAL,
            )
        return None


#: Order matters only for reporting; every rule is evaluated.
DEFAULT_RISK_RULES: tuple[RiskRule, ...] = (
    KillSwitchRule(),
    MaxDailyLossRule(),
    MaxDataAgeRule(),
    MinNetProfitRule(),
    MaxTradeSizeRule(),
    MaxSlippageRule(),
    MaxOpenTransfersRule(),
    MaxExchangeExposureRule(),
    MaxAssetExposureRule(),
)


def exposure_maps_for_transfers(
    open_transfers,  # Sequence[TransferRecord]
) -> tuple[dict[str, Decimal], dict[str, Decimal]]:
    """Post-trade exposure maps implied by the currently open transfers.

    * exchange exposure: the source venue carries the bought asset's value
      until the deposit is credited on the destination venue (then it flips);
    * asset exposure: the transferred asset's value at the plan's buy price.
    """
    exchange: dict[str, Decimal] = {}
    asset: dict[str, Decimal] = {}
    for record in open_transfers:
        value = record.plan.buy_cost_quote
        # While in flight the capital is spread across both venues; once the
        # deposit is detected it belongs to the destination.
        source_weight = Decimal("0.5") if record.deposit_amount > DEC0 else Decimal("1")
        exchange[record.source_exchange] = (
            exchange.get(record.source_exchange, DEC0) + value * source_weight
        )
        exchange[record.dest_exchange] = exchange.get(record.dest_exchange, DEC0) + value * (
            Decimal("1") - source_weight
        )
        asset[record.asset] = asset.get(record.asset, DEC0) + value
    return exchange, asset
