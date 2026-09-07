"""Human-gated recommendation lifecycle — CREATED -> REVIEWED -> APPROVED/REJECTED.

Phase 8 — Approval Workflow
---------------------------
``RECOMMENDATION → explicit operator approval → validation → controlled
config change → audit → measurement state``

Hardened approval flow (crash-safe, persistent, idempotent, fail-closed)::

    validate (allowlist, bounds, gates, staleness)
        ↓
    atomic durable claim + persistent config write (ONE db transaction)
        ↓
    runtime settings + RiskEngine synchronization (fail-closed, rollback)
        ↓
    audit + awaiting_measurement state

Critical invariants:

* AI never calls REVIEW / APPROVE / APPLY / CONFIG MUTATION / RISK LIMIT MUTATION / ORDER EXECUTION.
* Approval is explicit: non-empty human approver + exact recommendation id.
  In LIVE mode an explicit reason is additionally required.
* Only an explicit human operator (Telegram allow-list or CLI operator) may review/approve/reject.
* Approval is explicit and allowlisted — no generic config writes,
  no arbitrary config writes, no trade/order execution.
* Risk limits have a separate explicit validation path (stricter bounds).
* Existing risk gates, kill switch and DEMO/LIVE separation stay
  authoritative: approval refuses while the kill switch is engaged and
  never touches mode flags, gates or execution paths.
* Every transition is audited (app log + agent audit); duplicate approvals
  never apply twice (atomic PENDING/REVIEWED → terminal guard claimed
  BEFORE any runtime mutation; the loser of a race mutates nothing).
* Fail-closed on any uncertainty (stale current value, invalid param/value,
  already applied, concurrent race, persistence failure, risk-sync failure).
* Approved values are persisted durably in the existing ``bot_state``
  key/value table (``agent_config:<param>``) inside the same database
  transaction as the APPROVED claim, and re-applied on startup via
  :func:`restore_approved_config`, so approvals survive restarts.
* RiskEngine synchronization never fails silently: a sync failure rolls
  back runtime, persisted and row state and raises instead of reporting
  a false successful approval.
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

__all__ = [
    "ALLOWLIST",
    "CONFIG_KEY_PREFIX",
    "ApprovalError",
    "RecommendationApprovalService",
    "config_key",
    "load_persisted_overrides",
    "restore_approved_config",
]

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

# ------------------------------------------------------------------ persistent config
# Approved values are stored durably in the EXISTING ``bot_state`` key/value
# table (same database, same transactional machinery — no second store, no
# migration). One key per allowlisted parameter; values are the validated
# string forms. Never secrets: the allowlist contains only safe tuning
# numbers, never API keys, tokens, passwords or credentials.

#: Prefix for durable per-parameter keys in the ``bot_state`` table.
CONFIG_KEY_PREFIX = "agent_config:"


def config_key(param: str) -> str:
    """Durable ``bot_state`` key holding the approved value for *param*."""
    return f"{CONFIG_KEY_PREFIX}{param}"


async def load_persisted_overrides(db: Any) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """Read all persisted approved values (validated allowlist only).

    Unknown keys are ignored; malformed values are skipped with a warning
    (fail-closed: never applied). Returns ``{param: raw_string}``.
    """
    from app.storage.tables import BotStateRow

    overrides: dict[str, str] = {}
    async with db.session() as session:
        for param in ALLOWLIST:
            row = await session.get(BotStateRow, config_key(param))
            if row is None or row.value is None:
                continue
            raw = str(row.value).strip()
            if not raw:
                continue
            try:
                parsed = _parse_value(raw, ALLOWLIST[param]["type"])
                _validate_bounds(param, parsed)
            except ApprovalError as exc:
                logger.warning("persisted_config_skipped",
                               extra={"parameter": param, "error": str(exc)[:200]})
                continue
            overrides[param] = raw
    return overrides


async def restore_approved_config(services: Any) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """Re-apply persisted approved values to a freshly built service set.

    Called on application startup (see :func:`app.services.build_app`) so
    that ``services.settings`` and ``services.risk`` agree with the durable
    state after a restart. Invalid persisted values are skipped with a
    warning and never applied. Returns the applied ``{param: raw}`` map.
    """
    db = getattr(services, "db", None)
    if db is None:
        return {}
    try:
        overrides = await load_persisted_overrides(db)
    except Exception as exc:  # noqa: BLE001 - startup stays alive; caller logs
        logger.warning("persisted_config_load_failed", extra={"error": str(exc)[:200]})
        raise
    if not overrides:
        return {}
    # Apply through a detached approval helper so the explicit allowlist
    # mapping is shared with the live approval path (no second code path).
    helper = RecommendationApprovalService(db, services=services)
    applied: dict[str, str] = {}
    for param, raw in sorted(overrides.items()):
        try:
            parsed = _parse_value(raw, ALLOWLIST[param]["type"])
            _validate_bounds(param, parsed)
            helper._apply_value(param, parsed)
        except Exception as exc:  # noqa: BLE001 - one bad key must not break startup
            logger.warning("persisted_config_apply_failed",
                           extra={"parameter": param, "error": str(exc)[:200]})
            continue
        applied[param] = raw
    if applied:
        logger.info("persisted_config_restored", extra={"parameters": sorted(applied)})
    return applied


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

    def _require_risk_engine(self) -> Any:  # type: ignore[no-untyped-def]
        """Return the bound RiskEngine or fail closed (risk params only)."""
        engine = getattr(self._services, "risk", None) if self._services is not None else None
        if engine is None or not hasattr(engine, "limits") or not hasattr(engine, "with_limits"):
            raise ApprovalError("risk engine unavailable — risk limit change blocked (fail-closed)")
        return engine

    @staticmethod
    def _risk_field(param: str) -> str:
        return param.partition(".")[2]

    def _apply_value(self, param: str, value: Any) -> None:
        """Explicit, allowlist-only mutation of live config / risk engine.

        Fail-closed: both replacement objects are built BEFORE either is
        installed, and any RiskEngine failure raises (never swallowed), so
        settings and risk limits cannot silently diverge. Each param has
        its own branch so that review can audit exactly what is mutable.
        """
        if self._services is None:
            raise ApprovalError("no services bound — cannot apply config change")
        settings = getattr(self._services, "settings", None)
        if settings is None:
            raise ApprovalError("services has no settings")

        # Risk limits — separate explicit path (settings AND RiskEngine move
        # together; a sync failure raises and leaves both untouched).
        if param in RISK_PARAMS:
            field = self._risk_field(param)
            try:
                new_settings = settings.model_copy(
                    update={"risk": settings.risk.model_copy(update={field: value})}
                )
            except Exception as exc:
                raise ApprovalError(f"apply failed for {param}: {exc}") from exc
            try:
                engine = self._require_risk_engine()
                new_limits = engine.limits.model_copy(update={field: value})
                new_engine = engine.with_limits(new_limits)
            except ApprovalError:
                raise
            except Exception as exc:
                raise ApprovalError(f"risk synchronization failed for {param}: {exc}") from exc
            object.__setattr__(self._services, "settings", new_settings)
            try:
                object.__setattr__(self._services, "risk", new_engine)
            except Exception as exc:
                # Settings moved but risk did not — restore settings at once
                # so the two can never silently diverge.
                try:
                    object.__setattr__(self._services, "settings", settings)
                except Exception:
                    pass
                raise ApprovalError(f"risk synchronization failed for {param}: {exc}") from exc
            return
        if param == "arbitrage.triangle_min_net_bps":
            try:
                new_settings = settings.model_copy(update={"arbitrage": settings.arbitrage.model_copy(update={"triangle_min_net_bps": value})})
            except Exception as exc:
                raise ApprovalError(f"apply failed for {param}: {exc}") from exc
            object.__setattr__(self._services, "settings", new_settings)
            return
        if param == "arbitrage.triangle_max_leg_slippage_bps":
            try:
                new_settings = settings.model_copy(update={"arbitrage": settings.arbitrage.model_copy(update={"triangle_max_leg_slippage_bps": value})})
            except Exception as exc:
                raise ApprovalError(f"apply failed for {param}: {exc}") from exc
            object.__setattr__(self._services, "settings", new_settings)
            return
        if param == "execution.leg_timeout_seconds":
            try:
                new_settings = settings.model_copy(update={"execution": settings.execution.model_copy(update={"leg_timeout_seconds": value})})
            except Exception as exc:
                raise ApprovalError(f"apply failed for {param}: {exc}") from exc
            object.__setattr__(self._services, "settings", new_settings)
            return
        raise ApprovalError(f"parameter not allowlisted for apply: {param}")

    async def _persist_in_session(self, session: Any, param: str, value_str: str) -> None:  # type: ignore[no-untyped-def]
        """Durable write of one approved value inside the caller's transaction.

        Separate seam (not the ``bot_state`` repository) so the write can
        share the approval claim's database transaction — and so tests can
        inject a persistence failure deterministically.
        """
        from app.models.base import utc_now
        from app.storage.tables import BotStateRow

        row = BotStateRow(key=config_key(param), value=value_str, updated_at=utc_now())
        await session.merge(row)

    async def _rollback_claim(  # type: ignore[no-untyped-def]
        self, *, recommendation_id: str, from_status: str, param: str,
        old_persisted: Any, old_settings: Any, old_risk: Any,
    ) -> None:
        """Best-effort rollback after a post-claim runtime failure.

        Restores runtime settings/risk, the durable config key and the
        recommendation row so no false APPROVED state survives. Never
        raises — the original failure stays authoritative.
        """
        if self._services is not None:
            try:
                if old_settings is not None:
                    object.__setattr__(self._services, "settings", old_settings)
            except Exception as exc:  # noqa: BLE001
                logger.warning("approval_rollback_settings_failed",
                               extra={"recommendation_id": recommendation_id, "error": str(exc)[:200]})
            try:
                if old_risk is not None:
                    object.__setattr__(self._services, "risk", old_risk)
            except Exception as exc:  # noqa: BLE001
                logger.warning("approval_rollback_risk_failed",
                               extra={"recommendation_id": recommendation_id, "error": str(exc)[:200]})
        try:
            bot_state = getattr(self._services, "bot_state", None) if self._services is not None else None
            if bot_state is not None:
                if old_persisted is None:
                    try:
                        await bot_state.delete(config_key(param))
                    except Exception:
                        pass
                else:
                    await bot_state.set(config_key(param), old_persisted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("approval_rollback_persist_failed",
                           extra={"recommendation_id": recommendation_id, "error": str(exc)[:200]})
        try:
            async with self._db.session() as session:
                await session.execute(
                    AgentRecommendationRow.__table__.update()
                    .where(AgentRecommendationRow.id == recommendation_id,
                           AgentRecommendationRow.status == RecommendationStatus.APPROVED.value)
                    .values(status=from_status, operator_decision=None,
                            decision_reason=None, result="rolled back: runtime sync failed")
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("approval_rollback_row_failed",
                           extra={"recommendation_id": recommendation_id, "error": str(exc)[:200]})

    async def _audit_approval_failure(  # type: ignore[no-untyped-def]
        self, *, recommendation_id: str, param: str, stage: str, error: str,
    ) -> None:
        """Record a failed approval without ever fabricating APPROVED."""
        try:
            audit = getattr(self._services, "audit", None) if self._services is not None else None
            if audit is not None and hasattr(audit, "log"):
                await audit.log(
                    "AGENT_RECOMMENDATION_APPROVAL_FAILED",
                    f"{param} stage={stage}: {error}"[:500],
                    {"recommendation_id": recommendation_id, "parameter": param,
                     "stage": stage, "error": error[:300]},
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("approval_failure_audit_failed",
                           extra={"recommendation_id": recommendation_id, "error": str(exc)[:200]})
        try:
            from app.agent.audit import AgentAuditEvent

            agent_audit = getattr(self._services, "agent_audit", None) if self._services is not None else None
            if agent_audit is not None and hasattr(agent_audit, "log"):
                await agent_audit.log(AgentAuditEvent(
                    event_type="recommendation_approval_failed",
                    evidence_count=0,
                    confidence=0.0,
                    action="FAILED",
                    details={"recommendation_id": recommendation_id, "parameter": param,
                             "stage": stage, "error": error[:300]},
                    source_type="system",
                    source_id="phase8:approval",
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("approval_failure_agent_audit_failed",
                           extra={"recommendation_id": recommendation_id, "error": str(exc)[:200]})

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

        Hardened Phase 8 flow: explicit operator + exact id → allowlist /
        bounds validation → safety gates (kill switch, LIVE reason) →
        staleness check → ATOMIC durable claim (PENDING/REVIEWED → APPROVED
        plus the persistent ``bot_state`` config write in ONE database
        transaction) → runtime settings + RiskEngine synchronization
        (fail-closed with rollback) → dual audit + ``awaiting_measurement``
        state.

        Concurrency: the conditional UPDATE claims the row only when it is
        still actionable, so of concurrent approve/approve or approve/reject
        callers exactly one wins; every loser raises WITHOUT mutating any
        configuration. A failed claim or failed persistence applies nothing.
        A failed runtime sync rolls durable, runtime and row state back and
        raises — never a false APPROVED.
        """
        if not approver or not str(approver).strip():
            raise ApprovalError("approver must be a non-empty human identifier")
        if not recommendation_id or not str(recommendation_id).strip():
            raise ApprovalError("recommendation_id required")
        if self._services is None:
            raise ApprovalError("no services bound — cannot approve config change")

        # ---- Phase 1: load + validate (no mutation of any state) ----
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
            live_current = self._get_current(param)
            stored_current = row.current_value if row.current_value is not None else row.old_value
            if stored_current is not None and live_current is not None:
                if str(stored_current).strip() != str(live_current).strip():
                    raise ApprovalError(
                        f"current config changed for {param}: recommendation stored {stored_current!r} vs live {live_current!r} — stale, reject"
                    )
            from_status = str(row.status)
            from_version = int(row.version)
            # Snapshot row fields for the returned model (session scope ends).
            snap = {
                "id": row.id, "parameter": row.parameter, "current_value": row.current_value,
                "old_value": row.old_value, "proposed_value": row.proposed_value,
                "reason_text": row.reason, "evidence": tuple(row.evidence or ()),
                "confidence": row.confidence, "expected_impact": row.expected_impact,
                "risk": row.risk, "source_type": row.source_type, "source_id": row.source_id,
                "created_at": row.created_at, "updated_at": row.updated_at,
            }

        # ---- Phase 2: snapshot runtime + durable state (still no mutation) ----
        old_settings = getattr(self._services, "settings", None)
        old_risk = getattr(self._services, "risk", None)
        try:
            bot_state = getattr(self._services, "bot_state", None)
            old_persisted = await bot_state.get(config_key(param)) if bot_state is not None else None
        except Exception as exc:
            raise ApprovalError(f"cannot read persistent config for {param}: {exc}") from exc

        # ---- Phase 3: atomic durable claim + persistent config (ONE txn) ----
        # The conditional UPDATE is the idempotency guard: only a row that is
        # still PENDING/REVIEWED transitions, so concurrent or repeated
        # approvals cannot apply twice. The persistent config write shares
        # the transaction — a persistence failure rolls the claim back and
        # nothing is applied anywhere.
        try:
            async with self._db.session() as session:
                update_result = await session.execute(
                    AgentRecommendationRow.__table__.update()
                    .where(AgentRecommendationRow.id == recommendation_id,
                           AgentRecommendationRow.status.in_(_ACTIONABLE_STATES))
                    .values(
                        status=RecommendationStatus.APPROVED.value,
                        operator_decision=f"approved by {approver}",
                        decision_reason=reason[:500] if reason else None,
                        result=f"applied {param}={proposed_raw}",
                        version=from_version + 1,
                    )
                )
                if update_result.rowcount == 0:
                    raise ApprovalError("recommendation not PENDING/REVIEWED (concurrent modification) — duplicate")
                await self._persist_in_session(session, param, proposed_raw)
        except ApprovalError:
            raise
        except Exception as exc:
            await self._audit_approval_failure(
                recommendation_id=recommendation_id, param=param,
                stage="durable_claim", error=str(exc)[:300],
            )
            raise ApprovalError(f"durable approval claim failed for {param}: {exc}") from exc

        # ---- Phase 4: runtime synchronization (fail-closed with rollback) ----
        # Durable state is committed; runtime cannot join that transaction, so
        # a sync failure explicitly rolls durable + runtime + row state back
        # instead of leaving a false APPROVED behind.
        try:
            self._apply_value(param, parsed)
        except ApprovalError as exc:
            await self._rollback_claim(
                recommendation_id=recommendation_id, from_status=from_status, param=param,
                old_persisted=old_persisted, old_settings=old_settings, old_risk=old_risk,
            )
            await self._audit_approval_failure(
                recommendation_id=recommendation_id, param=param,
                stage="risk_sync" if "risk" in str(exc).lower() or param in RISK_PARAMS else "runtime_apply",
                error=str(exc)[:300],
            )
            raise
        except Exception as exc:
            await self._rollback_claim(
                recommendation_id=recommendation_id, from_status=from_status, param=param,
                old_persisted=old_persisted, old_settings=old_settings, old_risk=old_risk,
            )
            await self._audit_approval_failure(
                recommendation_id=recommendation_id, param=param,
                stage="runtime_apply", error=str(exc)[:300],
            )
            raise ApprovalError(f"apply failed for {param}: {exc}") from exc

        returned = AgentRecommendation(
            id=snap["id"],
            parameter=snap["parameter"],
            current_value=snap["current_value"],
            old_value=snap["old_value"],
            proposed_value=snap["proposed_value"],
            reason=snap["reason_text"],
            evidence=snap["evidence"],
            confidence=snap["confidence"],
            expected_impact=snap["expected_impact"],
            risk=snap["risk"],
            status=RecommendationStatus.APPROVED,
            operator_decision=f"approved by {approver}",
            decision_reason=reason[:500] if reason else None,
            result=f"applied {param}={proposed_raw}",
            source_type=snap["source_type"],
            source_id=snap["source_id"],
            version=from_version + 1,
            created_at=snap["created_at"],
            updated_at=snap["updated_at"],
        )

        # ---- Phase 5: success audit (fail-log, never masks the approval) ----
        try:
            audit = getattr(self._services, "audit", None) if self._services is not None else None
            if audit is not None and hasattr(audit, "log"):
                await audit.log(
                    "AGENT_RECOMMENDATION_APPROVED",
                    f"{param} {snap['current_value']} -> {proposed_raw} by {approver}",
                    {"recommendation_id": recommendation_id, "parameter": param,
                     "proposed_value": proposed_raw, "approver": approver},
                )
            try:
                bot_state = getattr(self._services, "bot_state", None) if self._services is not None else None
                if bot_state is not None:
                    await bot_state.set(f"agent_rec_approved:{recommendation_id}",
                                        {"parameter": param, "value": proposed_raw, "approver": approver})
            except Exception:
                pass
            try:
                agent_audit = getattr(self._services, "agent_audit", None) if self._services is not None else None
                if agent_audit is not None and hasattr(agent_audit, "log_recommendation"):
                    await agent_audit.log_recommendation(returned, event_type="recommendation_approved")
            except Exception as exc_issue:  # noqa: BLE001
                logger.warning("approval_agent_audit_failed",
                               extra={"recommendation_id": recommendation_id, "error": str(exc_issue)[:200]})
            await self._record_measurement_state(
                recommendation_id=recommendation_id, parameter=param, old_value=snap["current_value"],
                new_value=proposed_raw, approver=approver,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("approval_audit_failed", extra={"recommendation_id": recommendation_id, "error": str(exc)[:200]})

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
