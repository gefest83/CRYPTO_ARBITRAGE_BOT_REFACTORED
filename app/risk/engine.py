"""Central risk engine.

Single checkpoint that every future execution path must pass.  It is
deliberately synchronous and pure: given a context, the verdict is deterministic
and logged.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.config.logging_config import get_logger
from app.models.enums import RiskLevel
from app.models.risk import RiskAssessment, RiskLimits, RiskViolation

from .rules import DEFAULT_RISK_RULES, RiskContext, RiskRule

__all__ = ["RiskEngine"]

logger = get_logger("risk.engine")

_SEVERITY_ORDER: dict[RiskLevel, int] = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


class RiskEngine:
    """Evaluates all rules and returns a single verdict."""

    def __init__(self, limits: RiskLimits, *, rules: Sequence[RiskRule] | None = None) -> None:
        self._limits = limits
        self._rules = tuple(rules if rules is not None else DEFAULT_RISK_RULES)

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    @property
    def rules(self) -> tuple[RiskRule, ...]:
        return self._rules

    def with_limits(self, limits: RiskLimits) -> RiskEngine:
        return RiskEngine(limits, rules=self._rules)

    def evaluate(self, context: RiskContext) -> RiskAssessment:
        violations: list[RiskViolation] = []
        for rule in self._rules:
            try:
                violation = rule.check(context, self._limits)
            except Exception as exc:  # noqa: BLE001 - a broken rule must fail closed
                logger.error("risk_rule_failed", extra={"rule": rule.name, "error": str(exc)})
                violations.append(
                    RiskViolation(
                        rule=rule.name,
                        message=f"rule evaluation failed: {exc}",
                        severity=RiskLevel.CRITICAL,
                    )
                )
                continue
            if violation is not None:
                violations.append(violation)

        approved = not violations
        level = (
            RiskLevel.LOW
            if approved
            else max(
                (v.severity for v in violations), key=lambda severity: _SEVERITY_ORDER[severity]
            )
        )
        return RiskAssessment(approved=approved, violations=tuple(violations), level=level)
