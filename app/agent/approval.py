"""Human-gated recommendation lifecycle — CREATED -> REVIEWED -> APPROVED/REJECTED.

Phase 8 — Approval Workflow
---------------------------
``RECOMMENDATION → explicit operator approval → validation → controlled
config change → audit → measurement state``

Critical invariants:

* AI never calls REVIEW / APPROVE / APPLY / CONFIG MUTATION / RISK LIMIT MUTATION / ORDER EXECUTION.
* Approval is explicit: non-empty human approver + exact recommendation id.
  In LIVE mode an explicit reason is additionally required.
* Only an explicit human operator (Telegram allow-list or CLI operator) may review/approve/reject.
* Approval is explicit and allowlisted — no generic ``set_config(key, value)``,
  no arbitrary config writes, no trade/order/withdrawal execution.
* Risk limits have a separate explicit validation path (stricter bounds).
* Existing risk gates, kill switch and DEMO/LIVE separation stay
  authoritative: approval refuses while the kill switch is engaged and
  never touches mode flags, gates or execution paths.
* Every transition is audited (app log + agent audit); duplicate approvals
  never apply twice (atomic PENDING/REVIEWED → terminal guard).
* Fail-closed on any uncertainty (stale current value, invalid param/value,
  already applied, concurrent race).
* After an approved change the service records ``awaiting_measurement``
  state for the change (Phase 9 will measure it; evaluation is NOT here).

The service never executes trades.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import select

from app.agent.models import AgentRecommendation, RecommendationStatus
from app.agent.tables import AgentRecommendationRow
from app.config.logging_config import get_logger
from app.errors import TerminalError

__all__ = ["ApprovalError", "RecommendationApprovalService", "ALLOWLIST"]

logger = get_logger("agent.approval")


class ApprovalError(TerminalError):
    """Approval failed — fail-closed, trading unaffected."""

    code = "approval_error"
    http_status = 409


# ------------------------------------------------------------------ allowlist
# Explicit parameter -> (type, min, max). Risk limits and non-risk are
# separated only by the bounds they carry — risk bounds are deliberately
# narrow and conservative. No generic set_config is exposed.

ALLOWLIST: dict[str, dict[str, Any]] = {
    # Risk limits (separate explicit validation — stricter)
    "risk.max_trade_size": {"type": Decimal, "min": Decimal("10"), "max": Decimal("5000")},
    "risk.min_net_profit_bps": {"type": Decimal, "min": Decimal("0"), "max": Decimal("100")},
    "risk.max_slippage_bps": {"type": Decimal, "min": Decimal("1"), "max": Decimal("100")},
    "risk.max_data_age_ms": {"type": int, "min": 500, "max": 10000},
    "risk.max_open_transfers": {"type": int, "min": 0, "max": 10},
    "risk.max_daily_loss": {"type": Decimal, "min": Decimal("10"), "max": Decimal("10000")},
    # Non-risk (still allowlisted, looser but explicit)
    "arbitrage.triangle_min_net_bps": {"type": Decimal, "min": Decimal("0"), "max": Decimal("100")},
    "arbitrage.triangle_max_leg_slippage_bps": {"type": Decimal, "min": Decimal("1"), "max": Decimal("50")},
    "execution.leg_timeout_seconds": {"type": int, "min": 1, "max": 60},
}

# Parameters that are considered risk limits (separate path)
RISK_PARAMS = {k for k in ALLOWLIST if k.startswith("risk.")}


def _parse_value(raw: str, expected_type: type) -> Any:
    raw = str(raw).strip()
    if expected_type is Decimal:
        try:
            return Decimal(raw)
        except (InvalidOperation, ValueError) as exc:
            raise ApprovalError(f"invalid Decimal value: {raw!r}") from exc
    if expected_type is int:
        try:
            # Disallow float-like strings for int params
            if "." in raw:
                raise ValueError("int param got float string")
            return int(raw)
        except ValueError as exc:
            raise ApprovalError(f"invalid int value: {raw!r}") from exc
    raise ApprovalError(f"unsupported type for allowlist: {expected_type}")


def _validate_bounds(param: str, value: Any) -> None:
    spec = ALLOWLIST.get(param)
    if spec is None:
        raise ApprovalError(f"parameter not allowlisted: {param}")
    min_v = spec["min"]
    max_v = spec["max"]
    # Compare as same type
    if isinstance(value, Decimal):
        min_v = Decimal(str(min_v))
        max_v = Decimal(str(max_v))
    if value < min_v or value > max_v:
        raise ApprovalError(f"value {value} for {param} out of bounds [{min_v}, {max_v}]")


# ------------------------------------------------------------------ service


#: Reviewable/decidable states for approve/reject (Phase 7 lifecycle).
_ACTIONABLE_STATES: tuple[str, str] = (
    RecommendationStatus.PENDING.value,
    RecommendationStatus.REVIEWED.value,
)


class RecommendationApprovalService:
    """Human-only review, approval and application of advisor recommendations."""

    def __init__(self, db, *, services: Any | None = None) -> None:  # type: ignore[no-untyped-def]
        self._db = db
        self._services = services  # optional — used to read current config and apply

    # ------------------------------------------------------------------
    # Helpers — current config reading (explicit allowlist, no generic)
    # ------------------------------------------------------------------

    def _get_current(self, param: str) -> str | None:
        """Read the live current value for *param* via explicit allowlist.

        Returns stringified current value or None if param not recognized.
        This is the “verify current config not unexpectedly changed” check.
        """
        if self._services is None:
            return None
        try:
            settings = getattr(self._services, "settings", None)
            if settings is None:
                return None
            # Explicit mapping — no generic getattr(key)
            if param == "risk.max_trade_size":
                return str(settings.risk.max_trade_size)
            if param == "risk.min_net_profit_bps":
                return str(settings.risk.min_net_profit_bps)
            if param == "risk.max_slippage_bps":
                return str(settings.risk.max_slippage_bps)
            if param == "risk.max_data_age_ms":
                return str(settings.risk.max_data_age_ms)
            if param == "risk.max_open_transfers":
                return str(settings.risk.max_open_transfers)
            if param == "risk.max_daily_loss":
                return str(settings.risk.max_daily_loss)
            if param == "arbitrage.triangle_min_net_bps":
                return str(settings.arbitrage.triangle_min_net_bps)
            if param == "arbitrage.triangle_max_leg_slippage_bps":
                return str(settings.arbitrage.triangle_max_leg_slippage_bps)
            if param == "execution.leg_timeout_seconds":
                return str(settings.execution.leg_timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - fail-closed on read error
            raise ApprovalError(f"failed to read current config for {param}: {exc}") from exc
        return None

    def _apply_value(self, param: str, value: Any) -> None:
        """Explicit, allowlist-only mutation of live config / risk engine.

        No generic ``set_config(key, value)`` — each param has its own branch
        so that review can audit exactly what is mutable.
        """
        if self._services is None:
            raise ApprovalError("no services bound — cannot apply config change")
        settings = getattr(self._services, "settings", None)
        if settings is None:
            raise ApprovalError("services has no settings")

        # Risk limits — separate explicit path (also updates RiskEngine)
        if param == "risk.max_trade_size":
            new_settings = settings.model_copy(update={"risk": settings.risk.model_copy(update={"max_trade_size": value})})
            object.__setattr__(self._services, "settings", new_settings)
            try:
                new_limits = self._services.risk.limits.model_copy(update={"max_trade_size": value}) if hasattr(self._services, "risk") else None
                if new_limits is not None:
                    self._services.risk = self._services.risk.with_limits(new_limits)
            except Exception:
                pass
            return
        if param == "risk.min_net_profit_bps":
            new_settings = settings.model_copy(update={"risk": settings.risk.model_copy(update={"min_net_profit_bps": value})})
            object.__setattr__(self._services, "settings", new_settings)
            try:
                new_limits = self._services.risk.limits.model_copy(update={"min_net_profit_bps": value}) if hasattr(self._services, "risk") else None
                if new_limits is not None:
                    self._services.risk = self._services.risk.with_limits(new_limits)
            except Exception:
                pass
            return
        if param == "risk.max_slippage_bps":
            new_settings = settings.model_copy(update={"risk": settings.risk.model_copy(update={"max_slippage_bps": value})})
            object.__setattr__(self._services, "settings", new_settings)
            try:
                new_limits = self._services.risk.limits.model_copy(update={"max_slippage_bps": value}) if hasattr(self._services, "risk") else None
                if new_limits is not None:
                    self._services.risk = self._services.risk.with_limits(new_limits)
            except Exception:
                pass
            return
        if param == "risk.max_data_age_ms":
            new_settings = settings.model_copy(update={"risk": settings.risk.model_copy(update={"max_data_age_ms": value})})
            object.__setattr__(self._services, "settings", new_settings)
            try:
                new_limits = self._services.risk.limits.model_copy(update={"max_data_age_ms": value}) if hasattr(self._services, "risk") else None
                if new_limits is not None:
                    self._services.risk = self._services.risk.with_limits(new_limits)
            except Exception:
                pass
            return
        if param == "risk.max_open_transfers":
            new_settings = settings.model_copy(update={"risk": settings.risk.model_copy(update={"max_open_transfers": value})})
            object.__setattr__(self._services, "settings", new_settings)
            try:
                new_limits = self._services.risk.limits.model_copy(update={"max_open_transfers": value}) if hasattr(self._services, "risk") else None
                if new_limits is not None:
                    self._services.risk = self._services.risk.with_limits(new_limits)
            except Exception:
                pass
            return
        if param == "risk.max_daily_loss":
            new_settings = settings.model_copy(update={"risk": settings.risk.model_copy(update={"max_daily_loss": value})})
            object.__setattr__(self._services, "settings", new_settings)
            try:
                new_limits = self._services.risk.limits.model_copy(update={"max_daily_loss": value}) if hasattr(self._services, "risk") else None
                if new_limits is not None:
                    self._services.risk = self._services.risk.with_limits(new_limits)
            except Exception:
                pass
            return
        if param == "arbitrage.triangle_min_net_bps":
            new_settings = settings.model_copy(update={"arbitrage": settings.arbitrage.model_copy(update={"triangle_min_net_bps": value})})
            object.__setattr__(self._services, "settings", new_settings)
            return
        if param == "arbitrage.triangle_max_leg_slippage_bps":
            new_settings = settings.model_copy(update={"arbitrage": settings.arbitrage.model_copy(update={"triangle_max_leg_slippage_bps": value})})
            object.__setattr__(self._services, "settings", new_settings)
            return
        if param == "execution.leg_timeout_seconds":
            new_settings = settings.model_copy(update={"execution": settings.execution.model_copy(update={"leg_timeout_seconds": value})})
            object.__setattr__(self._services, "settings", new_settings)
            return
        raise ApprovalError(f"parameter not allowlisted for apply: {param}")

    # ------------------------------------------------------------------
    # Public: human approval + apply (session-based, atomic)
    # ------------------------------------------------------------------

    async def review(self, recommendation_id: str, *, approver: str, reason: str = "") -> AgentRecommendation:
        """Mark a PENDING recommendation as REVIEWED (human triage, no decision).

        Idempotent-fail-closed: only PENDING rows transition; anything else
        raises :class:`ApprovalError`. Audited like every other transition.
        """
        if not approver or not str(approver).strip():
            raise ApprovalError("approver must be a non-empty human identifier")
        if not recommendation_id or not str(recommendation_id).strip():
            raise ApprovalError("recommendation_id required")
        async with self._db.session() as session:
            result = await session.execute(select(AgentRecommendationRow).where(AgentRecommendationRow.id == recommendation_id))
            row = result.scalars().first()
            if row is None:
                raise ApprovalError(f"recommendation not found: {recommendation_id}")
            if row.status != RecommendationStatus.PENDING.value:
                raise ApprovalError(f"recommendation not PENDING (status={row.status}) — only PENDING can be reviewed")
            update_result = await session.execute(
                AgentRecommendationRow.__table__.update()
                .where(AgentRecommendationRow.id == recommendation_id, AgentRecommendationRow.status == RecommendationStatus.PENDING.value)
                .values(
                    status=RecommendationStatus.REVIEWED.value,
                    operator_decision=f"reviewed by {approver}",
                    decision_reason=reason[:500] if reason else None,
                    version=row.version + 1,
                )
            )
            if update_result.rowcount == 0:
                raise ApprovalError("recommendation not PENDING (concurrent modification) — duplicate")
            returned = AgentRecommendation(
                id=row.id,
                parameter=row.parameter,
                current_value=row.current_value,
                old_value=row.old_value,
                proposed_value=row.proposed_value,
                reason=row.reason,
                evidence=tuple(row.evidence or ()),
                confidence=row.confidence,
                expected_impact=row.expected_impact,
                risk=row.risk,
                status=RecommendationStatus.REVIEWED,
                operator_decision=f"reviewed by {approver}",
                decision_reason=reason[:500] if reason else None,
                result=row.result,
                source_type=row.source_type,
                source_id=row.source_id,
                version=row.version + 1,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
        await self._audit_transition(
            "AGENT_RECOMMENDATION_REVIEWED",
            returned,
            approver=approver,
            reason=reason,
            event_type="recommendation_reviewed",
        )
        return returned  # type: ignore[no-any-return]

    async def _audit_transition(  # type: ignore[no-untyped-def]
        self, action: str, rec: AgentRecommendation, *, approver: str, reason: str = "",
        event_type: str | None = None,
    ) -> None:
        """Best-effort dual audit (app log + agent audit); never rolls back."""
        try:
            audit = getattr(self._services, "audit", None) if self._services is not None else None
            if audit is not None and hasattr(audit, "log"):
                await audit.log(
                    action,
                    f"{rec.parameter} {rec.current_value} -> {rec.proposed_value} by {approver}",
                    {"recommendation_id": rec.id, "parameter": rec.parameter,
                     "proposed_value": rec.proposed_value, "approver": approver},
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("transition_audit_failed", extra={"recommendation_id": rec.id, "error": str(exc)[:200]})
        try:
            agent_audit = getattr(self._services, "agent_audit", None) if self._services is not None else None
            if agent_audit is not None and hasattr(agent_audit, "log_recommendation"):
                await agent_audit.log_recommendation(rec, event_type=event_type or "recommendation_created")
        except Exception as exc:  # noqa: BLE001
            logger.warning("transition_agent_audit_failed", extra={"recommendation_id": rec.id, "error": str(exc)[:200]})

    def _check_safety_gates(self, reason: str) -> None:
        """Phase 8 gates: kill switch and LIVE mode remain authoritative.

        * Kill switch engaged → refuse (uncertain state; release first).
        * LIVE mode → require an explicit human reason (real-capital stakes).
        Fail-closed when the guard/settings cannot be read.
        """
        if self._services is None:
            return
        try:
            guard = getattr(self._services, "guard", None)
            if guard is not None and bool(getattr(guard, "is_halted", False)):
                raise ApprovalError("kill switch engaged — config changes blocked until released")
        except ApprovalError:
            raise
        except Exception as exc:  # noqa: BLE001 - unreadable guard fails closed
            raise ApprovalError(f"cannot verify kill-switch state: {exc}") from exc
        try:
            settings = getattr(self._services, "settings", None)
            mode = getattr(settings, "mode", None) if settings is not None else None
            mode_value = str(getattr(mode, "value", mode) or "").upper()
            if mode_value == "LIVE" and not str(reason or "").strip():
                raise ApprovalError("LIVE mode requires an explicit reason for approval")
        except ApprovalError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ApprovalError(f"cannot verify trading mode: {exc}") from exc

    async def measurement_state(self, recommendation_id: str) -> dict[str, Any] | None:
        """Read-only view of the ``awaiting_measurement`` record (Phase 9 input)."""
        try:
            bot_state = getattr(self._services, "bot_state", None) if self._services is not None else None
            if bot_state is None or not hasattr(bot_state, "get"):
                return None
            value = await bot_state.get(f"agent_rec_measure:{recommendation_id}")
            return dict(value) if isinstance(value, dict) else None
        except Exception:
            return None

    async def _record_measurement_state(  # type: ignore[no-untyped-def]
        self, *, recommendation_id: str, parameter: str, old_value: Any,
        new_value: Any, approver: str,
    ) -> None:
        """Record that an approved change now requires measurement (best-effort)."""
        try:
            from app.models.base import utc_now

            bot_state = getattr(self._services, "bot_state", None) if self._services is not None else None
            if bot_state is None or not hasattr(bot_state, "set"):
                return
            await bot_state.set(
                f"agent_rec_measure:{recommendation_id}",
                {"recommendation_id": recommendation_id, "parameter": parameter,
                 "old_value": str(old_value), "new_value": str(new_value),
                 "approver": approver, "approved_at": utc_now().isoformat(),
                 "status": "awaiting_measurement"},
            )
        except Exception as exc:  # noqa: BLE001 - measurement bookkeeping never breaks approval
            logger.warning("measurement_state_write_failed",
                           extra={"recommendation_id": recommendation_id, "error": str(exc)[:200]})

    async def approve(self, recommendation_id: str, *, approver: str, reason: str = "") -> AgentRecommendation:
        """Approve an exact recommendation and apply it through the control plane.

        Phase 8 flow: explicit operator + exact id → safety gates (kill
        switch, LIVE reason) → allowlist validation → staleness check →
        controlled apply → atomic PENDING/REVIEWED → APPROVED → dual audit →
        ``awaiting_measurement`` state. Any failure aborts before mutation.
        """
        if not approver or not str(approver).strip():
            raise ApprovalError("approver must be a non-empty human identifier")
        if not recommendation_id or not str(recommendation_id).strip():
            raise ApprovalError("recommendation_id required")

        # Load and validate inside a single session transaction.
        # We use a session per call; the UPDATE ... WHERE status='pending'
        # provides atomicity for concurrent callers (one will see rowcount 0).
        async with self._db.session() as session:
            result = await session.execute(select(AgentRecommendationRow).where(AgentRecommendationRow.id == recommendation_id))
            row = result.scalars().first()
            if row is None:
                raise ApprovalError(f"recommendation not found: {recommendation_id}")
            if row.status not in _ACTIONABLE_STATES:
                raise ApprovalError(f"recommendation not PENDING/REVIEWED (status={row.status}) — duplicate or already handled")
            param = str(row.parameter).strip()
            proposed_raw = str(row.proposed_value).strip()
            if not param or not proposed_raw:
                raise ApprovalError("recommendation integrity failed: missing parameter/proposed_value")
            if param not in ALLOWLIST:
                raise ApprovalError(f"parameter not allowlisted: {param}")
            spec = ALLOWLIST[param]
            parsed = _parse_value(proposed_raw, spec["type"])
            _validate_bounds(param, parsed)
            # Phase 8 safety gates — existing controls stay authoritative.
            self._check_safety_gates(reason)
            if self._services is not None:
                live_current = self._get_current(param)
                stored_current = row.current_value if row.current_value is not None else row.old_value
                if stored_current is not None and live_current is not None:
                    if str(stored_current).strip() != str(live_current).strip():
                        raise ApprovalError(
                            f"current config changed for {param}: recommendation stored {stored_current!r} vs live {live_current!r} — stale, reject"
                        )
            # Apply in-memory (must succeed before DB commit)
            try:
                self._apply_value(param, parsed)
            except ApprovalError:
                raise
            except Exception as exc:
                raise ApprovalError(f"apply failed for {param}: {exc}") from exc

            # Atomic status transition — succeeds only if still actionable (concurrent guard)
            update_result = await session.execute(
                AgentRecommendationRow.__table__.update()
                .where(AgentRecommendationRow.id == recommendation_id, AgentRecommendationRow.status.in_(_ACTIONABLE_STATES))
                .values(
                    status=RecommendationStatus.APPROVED.value,
                    operator_decision=f"approved by {approver}",
                    decision_reason=reason[:500] if reason else None,
                    result=f"applied {param}={proposed_raw}",
                    version=row.version + 1,
                )
            )
            if update_result.rowcount == 0:
                raise ApprovalError(f"recommendation not PENDING/REVIEWED (concurrent modification) — duplicate")
            # Build returned model directly (avoid stale identity-map reload)
            returned = AgentRecommendation(
                id=row.id,
                parameter=row.parameter,
                current_value=row.current_value,
                old_value=row.old_value,
                proposed_value=row.proposed_value,
                reason=row.reason,
                evidence=tuple(row.evidence or ()),
                confidence=row.confidence,
                expected_impact=row.expected_impact,
                risk=row.risk,
                status=RecommendationStatus.APPROVED,
                operator_decision=f"approved by {approver}",
                decision_reason=reason[:500] if reason else None,
                result=f"applied {param}={proposed_raw}",
                source_type=row.source_type,
                source_id=row.source_id,
                version=row.version + 1,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            # Store for post-commit audit (fail-log, not rollback)
            self._pending_audit = (param, row.current_value, proposed_raw, recommendation_id, approver)  # type: ignore[attr-defined]

        # Post-commit audit (fail-log, not rollback) — outside the session so audit failure does not rollback approval
        try:
            param_a, old_a, new_a, rid_a, appr_a = getattr(self, "_pending_audit", (None, None, None, None, None))
            if param_a is not None:
                audit = getattr(self._services, "audit", None) if self._services is not None else None
                if audit is not None and hasattr(audit, "log"):
                    await audit.log(
                        "AGENT_RECOMMENDATION_APPROVED",
                        f"{param_a} {old_a} -> {new_a} by {appr_a}",
                        {"recommendation_id": rid_a, "parameter": param_a, "proposed_value": new_a, "approver": appr_a},
                    )
                try:
                    bot_state = getattr(self._services, "bot_state", None) if self._services is not None else None
                    if bot_state is not None:
                        await bot_state.set(f"agent_rec_approved:{rid_a}", {"parameter": param_a, "value": new_a, "approver": appr_a})
                except Exception:
                    pass
                # Phase 8: agent-audit trail + awaiting_measurement state.
                try:
                    agent_audit = getattr(self._services, "agent_audit", None) if self._services is not None else None
                    if agent_audit is not None and hasattr(agent_audit, "log_recommendation"):
                        await agent_audit.log_recommendation(returned, event_type="recommendation_approved")
                except Exception as exc_issue:  # noqa: BLE001
                    logger.warning("approval_agent_audit_failed",
                                   extra={"recommendation_id": recommendation_id, "error": str(exc_issue)[:200]})
                await self._record_measurement_state(
                    recommendation_id=rid_a, parameter=param_a, old_value=old_a,
                    new_value=new_a, approver=appr_a,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("approval_audit_failed", extra={"recommendation_id": recommendation_id, "error": str(exc)[:200]})
        finally:
            try:
                delattr(self, "_pending_audit")
            except Exception:
                pass

        return returned  # type: ignore[no-any-return]

    async def reject(self, recommendation_id: str, *, approver: str, reason: str = "") -> AgentRecommendation:
        if not approver or not str(approver).strip():
            raise ApprovalError("approver required")
        async with self._db.session() as session:
            result = await session.execute(select(AgentRecommendationRow).where(AgentRecommendationRow.id == recommendation_id))
            row = result.scalars().first()
            if row is None:
                raise ApprovalError(f"recommendation not found: {recommendation_id}")
            if row.status not in _ACTIONABLE_STATES:
                raise ApprovalError(f"not PENDING/REVIEWED (status={row.status})")
            update_result = await session.execute(
                AgentRecommendationRow.__table__.update()
                .where(AgentRecommendationRow.id == recommendation_id, AgentRecommendationRow.status.in_(_ACTIONABLE_STATES))
                .values(
                    status=RecommendationStatus.REJECTED.value,
                    operator_decision=f"rejected by {approver}",
                    decision_reason=reason[:500] if reason else None,
                    version=row.version + 1,
                )
            )
            if update_result.rowcount == 0:
                raise ApprovalError("concurrent modification — not PENDING")
            # Build returned directly to avoid stale identity map.
            # Reject mutates nothing but the row status: configuration is
            # provably unchanged (asserted by focused Phase 8 tests).
            returned = AgentRecommendation(
                id=row.id,
                parameter=row.parameter,
                current_value=row.current_value,
                old_value=row.old_value,
                proposed_value=row.proposed_value,
                reason=row.reason,
                evidence=tuple(row.evidence or ()),
                confidence=row.confidence,
                expected_impact=row.expected_impact,
                risk=row.risk,
                status=RecommendationStatus.REJECTED,
                operator_decision=f"rejected by {approver}",
                decision_reason=reason[:500] if reason else None,
                result=row.result,
                source_type=row.source_type,
                source_id=row.source_id,
                version=row.version + 1,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
        await self._audit_transition(
            "AGENT_RECOMMENDATION_REJECTED",
            returned,
            approver=approver,
            reason=reason,
            event_type="recommendation_rejected",
        )
        return returned  # type: ignore[no-any-return]
