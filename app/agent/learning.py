"""Phase 9 — Long-term Learning (closed loop, final AI Agent phase).

``Recommendation → Approved Change → Measurement → Outcome → Feedback →
Memory/Lessons``

The :class:`LearningEngine` closes the loop opened in Phase 8:

* measures approved recommendations against deterministic journal/config
  data (before vs after windows split at approval time);
* verdicts improved / worsened / unchanged / insufficient_data;
* persists measurements (new ``agent_measurements`` table — existing
  tables untouched) and flips the Phase 8 record to ``measured``;
* incorporates operator feedback (useful/wrong/ignore/approve/reject);
* revises or versions lessons from measured outcomes (contradictions
  retained, never silently overwritten; freshness/decay blended).

Hard invariants (tested):

* only APPROVED recommendations are measured; anything else fails closed;
* LLM text is never measured evidence — the optional ``llm`` only appends
  an explanatory note, and its failure changes nothing;
* learning never mutates trading parameters, never approves/applies
  anything, never bypasses safety/risk/LIVE controls (feedback kinds
  approve/reject delegate to the human-gated approval service);
* Knowledge, Journal, Conversation Memory and Trading Memory stay separate
  stores — measurement lessons are tagged ``measurement`` + ``system`` and
  never written into knowledge or journal tables.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import select

from app.agent.journal import MIN_SAMPLE_FOR_CONCLUSIONS, aggregate_stats
from app.agent.models import (
    Lesson,
    Measurement,
    MeasurementOutcome,
    RecommendationStatus,
    SourceType,
)
from app.agent.tables import AgentMeasurementRow
from app.config.logging_config import get_logger
from app.errors import TerminalError
from app.storage.engine import Database

__all__ = [
    "FEEDBACK_KINDS",
    "LearningEngine",
    "MeasurementError",
    "MeasurementRepository",
    "PARAMETER_METRICS",
]

logger = get_logger("agent.learning")


class MeasurementError(TerminalError):
    """Measurement/learning failed — fail-closed, trading unaffected."""

    code = "measurement_error"
    http_status = 409


#: Operator feedback verbs accepted by :meth:`LearningEngine.record_feedback`.
FEEDBACK_KINDS: tuple[str, ...] = ("useful", "wrong", "ignore", "approve", "reject")

#: Deterministic journal metric per allowlisted parameter.
#: ``higher_is_better`` decides the improved/worsened direction.
PARAMETER_METRICS: dict[str, tuple[str, bool]] = {
    "risk.max_slippage_bps": ("avg_slippage_bps", False),
    "arbitrage.triangle_max_leg_slippage_bps": ("avg_slippage_bps", False),
    "risk.max_trade_size": ("fail_rate", False),
    "execution.leg_timeout_seconds": ("fail_rate", False),
    "risk.max_data_age_ms": ("fail_rate", False),
    "risk.min_net_profit_bps": ("avg_net_bps", True),
    "arbitrage.triangle_min_net_bps": ("avg_net_bps", True),
    "risk.max_daily_loss": ("total_pnl", True),
    "risk.max_open_transfers": ("avg_net_bps", True),
}


def _dec(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError, AttributeError):
        return None


def _window_metric(trades: list[Any], metric: str) -> tuple[Decimal | None, dict[str, Any]]:
    """Deterministic metric over completed-trade windows (plus context stats)."""
    completed = [t for t in trades if _status_of(t) == "completed"]
    stats = aggregate_stats(trades)
    if metric == "avg_slippage_bps":
        values = [(t, _dec(getattr(t, "slippage_bps", None))) for t in completed]
        values = [(t, v) for t, v in values if v is not None]
        if not values:
            return None, {"n": 0, "stats": stats}
        avg = sum((v for _, v in values), Decimal("0")) / len(values)
        return avg, {"n": len(values), "stats": stats,
                     "trade_ids": [t.id for t, _ in values[:40]]}
    if metric == "fail_rate":
        terminal = [t for t in trades if _status_of(t) in ("completed", "failed", "manual_review")]
        if not terminal:
            return None, {"n": 0, "stats": stats}
        failed = [t for t in terminal if _status_of(t) in ("failed", "manual_review")]
        rate = Decimal(len(failed)) / Decimal(len(terminal))
        return rate, {"n": len(terminal), "stats": stats,
                      "trade_ids": [t.id for t in terminal[:40]]}
    if metric == "avg_net_bps":
        values = [(t, _dec(getattr(t, "net_profit_bps", None))) for t in completed]
        values = [(t, v) for t, v in values if v is not None]
        if not values:
            return None, {"n": 0, "stats": stats}
        avg = sum((v for _, v in values), Decimal("0")) / len(values)
        return avg, {"n": len(values), "stats": stats,
                     "trade_ids": [t.id for t, _ in values[:40]]}
    if metric == "total_pnl":
        if not completed:
            return None, {"n": 0, "stats": stats}
        total = _dec(stats["total_pnl"])
        return total, {"n": len(completed), "stats": stats,
                       "trade_ids": [t.id for t in completed[:40]]}
    return None, {"n": 0, "stats": stats}


def _status_of(trade: Any) -> str:
    status = getattr(trade, "status", "?")
    return str(getattr(status, "value", status))


def _confidence_for_windows(n_before: int, n_after: int) -> float:
    smallest = min(n_before, n_after)
    if smallest < MIN_SAMPLE_FOR_CONCLUSIONS:
        return 0.0
    return round(min(0.85, 0.5 + 0.05 * smallest), 4)


def _row_to_measurement(row: AgentMeasurementRow) -> Measurement:
    return Measurement(
        id=row.id,
        recommendation_id=row.recommendation_id,
        parameter=row.parameter,
        old_value=row.old_value,
        new_value=row.new_value,
        metric=row.metric,
        higher_is_better=bool(row.higher_is_better),
        before_value=row.before_value,
        after_value=row.after_value,
        delta=row.delta,
        n_before=row.n_before,
        n_after=row.n_after,
        outcome=row.outcome,
        confidence=row.confidence,
        reason=row.reason,
        feedback_kind=row.feedback_kind,
        feedback_by=row.feedback_by,
        feedback_at=row.feedback_at,
        lesson_id=row.lesson_id,
        evidence_before=tuple(row.evidence_before or ()),
        evidence_after=tuple(row.evidence_after or ()),
        details=dict(row.details or {}),
        source_type=row.source_type,
        source_id=row.source_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _measurement_to_row(m: Measurement) -> AgentMeasurementRow:
    return AgentMeasurementRow(
        id=m.id,
        recommendation_id=m.recommendation_id,
        parameter=m.parameter,
        old_value=m.old_value,
        new_value=m.new_value,
        metric=m.metric,
        higher_is_better=bool(m.higher_is_better),
        before_value=m.before_value,
        after_value=m.after_value,
        delta=m.delta,
        n_before=m.n_before,
        n_after=m.n_after,
        outcome=m.outcome.value if isinstance(m.outcome, MeasurementOutcome) else str(m.outcome),
        confidence=m.confidence,
        reason=m.reason,
        feedback_kind=m.feedback_kind,
        feedback_by=m.feedback_by,
        feedback_at=m.feedback_at,
        lesson_id=m.lesson_id,
        evidence_before=list(m.evidence_before),
        evidence_after=list(m.evidence_after),
        details=dict(m.details),
        source_type=m.source_type,
        source_id=m.source_id,
        created_at=m.created_at,
        updated_at=m.updated_at,
    )


class MeasurementRepository:
    """Persistence for :class:`Measurement` (new Phase 9 table only)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, measurement: Measurement) -> Measurement:
        async with self._db.session() as session:
            await session.merge(_measurement_to_row(measurement))
        return measurement

    async def get(self, measurement_id: str) -> Measurement | None:
        async with self._db.session() as session:
            row = await session.get(AgentMeasurementRow, measurement_id)
            return _row_to_measurement(row) if row else None

    async def get_by_recommendation(self, recommendation_id: str) -> Measurement | None:
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentMeasurementRow)
                .where(AgentMeasurementRow.recommendation_id == recommendation_id)
                .order_by(AgentMeasurementRow.created_at.desc())
                .limit(1)
            )
            row = result.scalars().first()
            return _row_to_measurement(row) if row else None

    async def list_recent(self, limit: int = 20) -> list[Measurement]:
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentMeasurementRow).order_by(AgentMeasurementRow.created_at.desc()).limit(limit)
            )
            return [_row_to_measurement(r) for r in result.scalars()]


class LearningEngine:
    """Closed learning loop over approved recommendations.

    Bound to :class:`AppServices` (duck-typed). Reads journal/config,
    memory, feedback and audit; writes ONLY measurements, memory
    (experiences/lessons via existing repos), feedback rows and audit
    events. Never touches trading, execution, risk, recovery or config —
    except through the human-gated approval service for explicit
    approve/reject feedback.
    """

    def __init__(self, services: Any) -> None:  # type: ignore[no-untyped-def]
        self._services = services
        from app.storage.engine import Database as _Database  # local import: composition root parity

        db: _Database = services.db
        self._measurements = MeasurementRepository(db)

    @property
    def measurements(self) -> MeasurementRepository:
        return self._measurements

    # ------------------------------------------------------------ measurement

    async def measure(
        self,
        recommendation_id: str,
        *,
        min_sample: int = MIN_SAMPLE_FOR_CONCLUSIONS,
        llm: Any | None = None,
    ) -> Measurement:
        """Measure one APPROVED recommendation before vs after approval.

        Windows split journal trades at approval time; the metric follows
        :data:`PARAMETER_METRICS`. Persists the verdict, flips the Phase 8
        record to ``measured`` and audits ``measurement_completed``. Only
        APPROVED rows are measurable — anything else raises
        :class:`MeasurementError`.
        """
        rec = await self._require_approved(recommendation_id)
        state = await self._approval_service().measurement_state(recommendation_id)
        approved_at = self._parse_time((state or {}).get("approved_at"))
        if approved_at is None:
            try:
                approved_at = rec.updated_at
            except Exception:
                approved_at = None
        metric, higher_is_better = PARAMETER_METRICS.get(rec.parameter, ("avg_net_bps", True))
        try:
            trades = await self._services.trades.list_recent(limit=200)
        except Exception as exc:
            raise MeasurementError(f"journal read failed: {exc}") from exc
        before = [t for t in trades if getattr(t, "created_at", None) is not None and t.created_at < approved_at] if approved_at else []
        after = [t for t in trades if getattr(t, "created_at", None) is not None and approved_at is not None and t.created_at >= approved_at] if approved_at else list(trades)
        # Confound check: live config must still hold the approved value.
        live_current = self._live_param(rec.parameter)
        confounded = live_current is not None and str(live_current).strip() != str(rec.proposed_value).strip()
        before_value, before_ctx = _window_metric(before, metric)
        after_value, after_ctx = _window_metric(after, metric)
        n_before = int(before_ctx.get("n", 0))
        n_after = int(after_ctx.get("n", 0))
        if confounded:
            outcome: MeasurementOutcome = MeasurementOutcome.INSUFFICIENT
            reason = (f"insufficient_data: live {rec.parameter}={live_current} no longer holds "
                      f"approved {rec.proposed_value} — window confounded")
            delta = Decimal("0")
        elif n_before < min_sample or n_after < min_sample:
            outcome = MeasurementOutcome.INSUFFICIENT
            reason = (f"insufficient_data: n_before={n_before}, n_after={n_after}, "
                      f"need {min_sample} each")
            delta = Decimal("0")
        elif before_value is None or after_value is None:
            outcome = MeasurementOutcome.INSUFFICIENT
            reason = f"insufficient_data: metric {metric} unavailable in one window"
            delta = Decimal("0")
        else:
            delta = after_value - before_value
            if delta == 0:
                outcome = MeasurementOutcome.UNCHANGED
                reason = f"no change: {metric} {before_value} → {after_value} (n={n_before}/{n_after})"
            elif (delta > 0) == bool(higher_is_better):
                outcome = MeasurementOutcome.IMPROVED
                reason = f"improved: {metric} {before_value} → {after_value} (Δ{delta}, n={n_before}/{n_after})"
            else:
                outcome = MeasurementOutcome.WORSENED
                reason = f"worsened: {metric} {before_value} → {after_value} (Δ{delta}, n={n_before}/{n_after})"
        confidence = _confidence_for_windows(n_before, n_after) if outcome not in (MeasurementOutcome.INSUFFICIENT,) else 0.0
        details: dict[str, Any] = {
            "stats_before": before_ctx.get("stats"),
            "stats_after": after_ctx.get("stats"),
            "metric": metric,
            "higher_is_better": higher_is_better,
        }
        if llm is not None:
            try:
                note = await self._explain_with_llm(rec, metric, before_value, after_value, delta, llm)
                if note:
                    details["ai_note"] = note
            except Exception as exc:  # noqa: BLE001 - LLM failure never changes the verdict
                logger.warning("learning_llm_failed",
                               extra={"recommendation_id": recommendation_id, "error": str(exc)[:200]})
        measurement = Measurement(
            recommendation_id=rec.id,
            parameter=rec.parameter,
            old_value=str(rec.current_value or rec.old_value or "?"),
            new_value=str(rec.proposed_value),
            metric=metric,
            higher_is_better=bool(higher_is_better),
            before_value=str(before_value) if before_value is not None else "n/a",
            after_value=str(after_value) if after_value is not None else "n/a",
            delta=str(delta),
            n_before=n_before,
            n_after=n_after,
            outcome=outcome,
            confidence=confidence,
            reason=reason[:2000],
            evidence_before=tuple(before_ctx.get("trade_ids", [])),
            evidence_after=tuple(after_ctx.get("trade_ids", [])),
            details=details,
            source_type=rec.source_type,
            source_id=rec.source_id,
        )
        await self._measurements.save(measurement)
        await self._mark_measured(recommendation_id, measurement.id)
        await self._audit_measurement(measurement)
        return measurement

    # ------------------------------------------------------------ feedback

    async def record_feedback(
        self,
        *,
        recommendation_id: str | None = None,
        measurement_id: str | None = None,
        kind: str,
        approver: str,
        comment: str = "",
    ) -> dict[str, Any]:
        """Incorporate operator feedback (useful/wrong/ignore/approve/reject).

        approve/reject delegate to the human-gated approval service (exact
        recommendation, audited transitions). useful/wrong also persist
        ±1 :class:`AgentFeedback` rows. ignore records the kind without a
        rating. Every path audits ``measurement_feedback``.
        """
        from app.agent.approval import ApprovalError

        kind = str(kind or "").strip().lower()
        if kind not in FEEDBACK_KINDS:
            raise MeasurementError(f"unknown feedback kind: {kind!r} (useful/wrong/ignore/approve/reject)")
        if not approver or not str(approver).strip():
            raise ApprovalError("approver must be a non-empty human identifier")
        measurement = None
        if measurement_id is not None:
            measurement = await self._measurements.get(measurement_id)
            if measurement is None:
                raise MeasurementError(f"measurement not found: {measurement_id}")
            recommendation_id = recommendation_id or measurement.recommendation_id
        if not recommendation_id:
            raise MeasurementError("recommendation_id required (directly or via measurement_id)")
        if kind in ("approve", "reject"):
            service = self._approval_service()
            if kind == "approve":
                result = await service.approve(recommendation_id, approver=str(approver), reason=comment)
            else:
                result = await service.reject(recommendation_id, approver=str(approver), reason=comment)
            transitioned = result
        else:
            transitioned = None
            if kind in ("useful", "wrong"):
                await self._services.agent_feedback.submit(
                    target_type="recommendation", target_id=recommendation_id,
                    rating=1 if kind == "useful" else -1,
                    comment=comment, approver=str(approver),
                )
        if measurement is not None:
            await self._update_measurement_feedback(measurement, kind, approver, comment)
        await self._audit_feedback(recommendation_id, measurement.id if measurement else None,
                                   kind, approver, comment)
        return {"recommendation_id": recommendation_id,
                "measurement_id": measurement.id if measurement else None,
                "kind": kind,
                "transitioned_to": str(getattr(getattr(transitioned, "status", ""), "value", getattr(transitioned, "status", None))) if transitioned else None}

    # ------------------------------------------------------------ learn

    async def learn_from_measurement(self, measurement_id: str) -> dict[str, Any]:
        """Fold a measured outcome into memory (new or versioned lesson).

        Insufficient outcomes are skipped explicitly. Contradictory outcomes
        create a separate lesson and audit the conflict — old lessons are
        never silently overwritten. Otherwise the overlapping lesson is
        revised (version bump + history) with decay-blended confidence.
        Returns ``{"lesson_id": ..., "versioned": bool, "conflict": ...}``.
        """
        from app.agent.extraction import detect_contradiction, outcome_direction
        from app.agent.memory import decayed_confidence, memory_age_days, revise_lesson

        measurement = await self._measurements.get(measurement_id)
        if measurement is None:
            raise MeasurementError(f"measurement not found: {measurement_id}")
        if measurement.outcome == MeasurementOutcome.INSUFFICIENT:
            return {"lesson_id": None, "versioned": False, "conflict": None,
                    "skipped": f"insufficient outcome: {measurement.reason[:160]}"}
        window_ids = list(measurement.evidence_after) or list(measurement.evidence_before)
        after_stats = (measurement.details or {}).get("stats_after") or {}
        new_direction = outcome_direction(
            after_stats.get("avg_net_bps", measurement.after_value),
            self._fail_rate(after_stats),
        )
        lesson_repo = self._services.agent_lessons
        label_repo = getattr(self._services, "agent_memory_labels", None)
        try:
            lessons = await lesson_repo.list_all()
        except Exception:
            lessons = []
        overlapping = [les for les in lessons
                       if set(getattr(les, "evidence", ()) or ()) & set(window_ids)]
        for les in overlapping:
            existing_direction = "unknown"
            try:
                if label_repo is not None:
                    label = await label_repo.get_label("lesson", les.id)
                    if label is not None and label.get("direction"):
                        existing_direction = str(label["direction"])
            except Exception:
                pass
            conflict = detect_contradiction(
                existing_direction=existing_direction,
                new_avg_net_bps=after_stats.get("avg_net_bps", 0),
                new_fail_rate=self._fail_rate(after_stats),
                existing_lesson_id=les.id,
                new_evidence_ids=window_ids,
            )
            if conflict is not None:
                new_lesson = await self._create_outcome_lesson(measurement, new_direction)
                await self._audit_conflict(measurement, les.id, conflict, new_lesson.id)
                await self._link_measurement_lesson(measurement, new_lesson.id)
                return {"lesson_id": new_lesson.id, "versioned": False, "conflict": conflict}
        if overlapping:
            target = sorted(overlapping, key=lambda les: len(set(getattr(les, "evidence", ()) or ()) & set(window_ids)),
                            reverse=True)[0]
            age = memory_age_days(getattr(target, "created_at", None))
            blended = round((decayed_confidence(float(target.confidence), age) + float(measurement.confidence)) / 2, 4)
            updated = await revise_lesson(
                lesson_repo, self._services.agent_lesson_history, target.id,
                new_evidence_ids=tuple(window_ids),
                confidence=min(0.9, max(0.0, blended)),
                reason=f"measurement {measurement.id}: {measurement.outcome.value} "
                       f"({measurement.metric} Δ{measurement.delta}); "
                       f"feedback={measurement.feedback_kind or 'none'}",
            )
            if label_repo is not None:
                try:
                    await label_repo.set_label(target_type="lesson", target_id=updated.id,
                                               learning_type="observation",
                                               sample_size=len(updated.related_experience_ids) or len(updated.evidence),
                                               direction=new_direction)
                except Exception:
                    pass
            await self._link_measurement_lesson(measurement, updated.id)
            await self._audit_learned(measurement, updated.id, versioned=True, conflict=None)
            return {"lesson_id": updated.id, "versioned": True, "conflict": None}
        new_lesson = await self._create_outcome_lesson(measurement, new_direction)
        await self._link_measurement_lesson(measurement, new_lesson.id)
        await self._audit_learned(measurement, new_lesson.id, versioned=False, conflict=None)
        return {"lesson_id": new_lesson.id, "versioned": False, "conflict": None}

    # ------------------------------------------------------------ queries

    async def list_recent_measurements(self, limit: int = 20) -> list[Measurement]:
        return await self._measurements.list_recent(limit=min(int(limit), 100))

    async def get_by_recommendation(self, recommendation_id: str) -> Measurement | None:
        return await self._measurements.get_by_recommendation(recommendation_id)

    # ------------------------------------------------------------ internals

    def _approval_service(self) -> Any:  # type: ignore[no-untyped-def]
        return self._services.agent_approval_service

    async def _require_approved(self, recommendation_id: str) -> Any:  # type: ignore[no-untyped-def]
        try:
            rec = await self._services.agent_recommendation_service.repository.get(recommendation_id)
        except Exception as exc:
            raise MeasurementError(f"recommendation read failed: {exc}") from exc
        if rec is None:
            raise MeasurementError(f"recommendation not found: {recommendation_id}")
        status = str(getattr(getattr(rec, "status", ""), "value", rec.status))
        if status != RecommendationStatus.APPROVED.value:
            raise MeasurementError(f"only APPROVED recommendations are measurable (status={status})")
        return rec

    @staticmethod
    def _parse_time(raw: Any) -> datetime | None:  # type: ignore[no-untyped-def]
        if not raw:
            return None
        try:
            from datetime import datetime as _dt

            return _dt.fromisoformat(str(raw))
        except Exception:
            return None

    def _live_param(self, parameter: str) -> str | None:
        try:
            settings = self._services.settings
        except Exception:
            return None
        try:
            section, _, name = parameter.partition(".")
            group = getattr(settings, section, None)
            return str(getattr(group, name)) if group is not None else None
        except Exception:
            return None

    @staticmethod
    def _fail_rate(stats: dict[str, Any]) -> float:
        try:
            n = int(stats.get("n", 0) or 0)
            failed = int(stats.get("failed", 0) or 0) + int(stats.get("manual_review", 0) or 0)
            return (failed / n) if n else 0.0
        except Exception:
            return 0.0

    async def _mark_measured(self, recommendation_id: str, measurement_id: str) -> None:
        try:
            bot_state = getattr(self._services, "bot_state", None)
            if bot_state is None or not hasattr(bot_state, "get"):
                return
            key = f"agent_rec_measure:{recommendation_id}"
            state = await bot_state.get(key)
            if isinstance(state, dict):
                state = dict(state)
                state["status"] = "measured"
                state["measurement_id"] = measurement_id
                await bot_state.set(key, state)
        except Exception as exc:  # noqa: BLE001 - bookkeeping never breaks measurement
            logger.warning("learning_mark_measured_failed",
                           extra={"recommendation_id": recommendation_id, "error": str(exc)[:200]})

    async def _update_measurement_feedback(  # type: ignore[no-untyped-def]
        self, measurement: Measurement, kind: str, approver: str, comment: str,
    ) -> Measurement:
        from app.models.base import utc_now

        updated = measurement.model_copy(update={
            "feedback_kind": kind,
            "feedback_by": str(approver),
            "feedback_at": utc_now(),
            "details": dict(measurement.details or {}, feedback_comment=str(comment or "")[:500]),
            "updated_at": utc_now(),
        })
        await self._measurements.save(updated)
        return updated

    async def _audit_measurement(self, measurement: Measurement) -> None:
        try:
            from app.agent.audit import AgentAuditEvent

            await self._services.agent_audit.log(AgentAuditEvent(
                event_type="measurement_completed",
                evidence_count=measurement.sample_size,
                confidence=float(measurement.confidence),
                action=measurement.outcome.value.upper(),
                details={"measurement_id": measurement.id,
                         "recommendation_id": measurement.recommendation_id,
                         "parameter": measurement.parameter,
                         "metric": measurement.metric,
                         "before": measurement.before_value,
                         "after": measurement.after_value,
                         "delta": measurement.delta,
                         "outcome": measurement.outcome.value},
                source_type=SourceType.SYSTEM.value,
                source_id=f"phase9:{measurement.parameter}",
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("learning_audit_failed", extra={"error": str(exc)[:200]})

    async def _audit_feedback(  # type: ignore[no-untyped-def]
        self, recommendation_id: str, measurement_id: str | None,
        kind: str, approver: str, comment: str,
    ) -> None:
        try:
            from app.agent.audit import AgentAuditEvent

            await self._services.agent_audit.log(AgentAuditEvent(
                event_type="measurement_feedback",
                evidence_count=1,
                confidence=0.0,
                action=kind.upper(),
                details={"recommendation_id": recommendation_id,
                         "measurement_id": measurement_id,
                         "kind": kind, "approver": str(approver),
                         "comment": str(comment or "")[:300]},
                source_type=SourceType.SYSTEM.value,
                source_id="phase9:feedback",
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("learning_feedback_audit_failed", extra={"error": str(exc)[:200]})

    async def _audit_learned(  # type: ignore[no-untyped-def]
        self, measurement: Measurement, lesson_id: str, *, versioned: bool,
        conflict: dict | None,
    ) -> None:
        try:
            from app.agent.audit import AgentAuditEvent

            await self._services.agent_audit.log(AgentAuditEvent(
                event_type="measurement_learned",
                evidence_count=measurement.sample_size,
                confidence=float(measurement.confidence),
                action="LEARNED",
                details={"measurement_id": measurement.id, "lesson_id": lesson_id,
                         "versioned": versioned, "conflict": conflict,
                         "outcome": measurement.outcome.value},
                source_type=SourceType.SYSTEM.value,
                source_id="phase9:learn",
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("learning_learned_audit_failed", extra={"error": str(exc)[:200]})

    async def _audit_conflict(self, measurement: Measurement, lesson_id: str, conflict: dict[str, Any], new_lesson_id: str) -> None:  # type: ignore[no-untyped-def]
        try:
            from app.agent.audit import AgentAuditEvent

            await self._services.agent_audit.log(AgentAuditEvent(
                event_type="contradiction",
                evidence_count=len(measurement.evidence_after),
                confidence=float(measurement.confidence),
                action="CONFLICT",
                details=dict(conflict, measurement_id=measurement.id, new_lesson_id=new_lesson_id),
                source_type=SourceType.SYSTEM.value,
                source_id="phase9:learn",
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("learning_conflict_audit_failed", extra={"error": str(exc)[:200]})

    async def _create_outcome_lesson(self, measurement: Measurement, direction: str) -> Any:  # type: ignore[no-untyped-def]
        from app.agent.models import Lesson

        lesson = Lesson(
            title=f"Measured outcome: {measurement.parameter} {measurement.outcome.value}"[:256],
            content=(f"Approved change {measurement.parameter} {measurement.old_value} → "
                     f"{measurement.new_value} measured {measurement.outcome.value}: "
                     f"{measurement.metric} {measurement.before_value} → {measurement.after_value} "
                     f"(Δ{measurement.delta}, n={measurement.n_before}/{measurement.n_after}). "
                     f"{measurement.reason[:500]}")[:4000],
            pattern=f"measurement:{measurement.outcome.value}:{measurement.metric}"[:2000],
            confidence=float(measurement.confidence),
            evidence=tuple(list(measurement.evidence_after) or list(measurement.evidence_before)),
            related_experience_ids=(),
            source_type=SourceType.SYSTEM.value,
            source_id=f"phase9:{measurement.id}",
        )
        await self._services.agent_lessons.save(lesson)
        try:
            label_repo = getattr(self._services, "agent_memory_labels", None)
            if label_repo is not None:
                await label_repo.set_label(target_type="lesson", target_id=lesson.id,
                                           learning_type="observation",
                                           sample_size=len(lesson.evidence) or measurement.sample_size,
                                           direction=direction)
        except Exception:
            pass
        return lesson

    async def _link_measurement_lesson(self, measurement: Measurement, lesson_id: str) -> None:
        try:
            updated = measurement.model_copy(update={"lesson_id": lesson_id})
            await self._measurements.save(updated)
        except Exception as exc:  # noqa: BLE001
            logger.warning("learning_link_failed", extra={"error": str(exc)[:200]})

    async def _explain_with_llm(  # type: ignore[no-untyped-def]
        self, rec: Any, metric: str, before_value: Any, after_value: Any, delta: Any, llm: Any,
    ) -> str | None:
        from app.agent.providers.base import LLMMessage, LLMRequest, filter_secrets_from_text

        prompt = (
            "Describe in one sentence this measured outcome. "
            f"Use only these validated numbers: metric={metric}, "
            f"before={before_value}, after={after_value}, delta={delta}."
        )
        response = await llm.complete(LLMRequest(messages=(LLMMessage(role="user", content=prompt),)))
        text = filter_secrets_from_text(response.content or "")[:200].strip()
        if len(text) < 5:
            raise ValueError("empty LLM explanation")
        return text

