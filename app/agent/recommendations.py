"""Recommendation model and service.

The recommendation layer is deliberately *isolated* from configuration mutation:

* ``RecommendationService.create`` persists an :class:`AgentRecommendation`
  and returns it. That is the *only* allowed side effect.
* There is *no* method that writes to ``app.config.settings`` or
  ``RiskLimits`` or any risk limit. Attempting to call a non-existent
  ``apply`` path raises :class:`RecommendationApplyBlocked`.
* Any future ``APPROVE -> validation gate -> apply -> audit`` flow must be
  implemented as a *separate* deterministic service that validates the
  recommendation, asks for explicit human approval, and audits the result.
  Phase 1 does not implement that path and the boundary is enforced here.

Secret filtering is applied before any stored recommendation is exposed to
the LLM: ``recommendation.model_dump`` never contains keys.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from app.agent.models import AgentRecommendation, RecommendationStatus
from app.agent.tables import AgentRecommendationRow
from app.config.logging_config import get_logger
from app.errors import TerminalError
from app.storage.engine import Database

__all__ = ["RecommendationApplyBlocked", "RecommendationRepository", "RecommendationService"]

logger = get_logger("agent.recommendations")


class RecommendationApplyBlocked(TerminalError):
    """Raised when code attempts to apply a recommendation without human approval.

    The advisor is an analytical layer; it may not mutate trading parameters,
    risk limits, or configuration directly. A separate, audited, human-gated
    flow would be required to change anything.
    """

    code = "recommendation_apply_blocked"
    http_status = 409


# ------------------------------------------------------------------ mapping


def _rec_to_row(rec: AgentRecommendation) -> AgentRecommendationRow:
    return AgentRecommendationRow(
        id=rec.id,
        parameter=rec.parameter,
        current_value=rec.current_value,
        old_value=rec.old_value,
        proposed_value=rec.proposed_value,
        reason=rec.reason,
        evidence=list(rec.evidence),
        confidence=rec.confidence,
        expected_impact=rec.expected_impact,
        risk=rec.risk,
        status=rec.status.value if isinstance(rec.status, RecommendationStatus) else str(rec.status),
        operator_decision=rec.operator_decision,
        decision_reason=rec.decision_reason,
        result=rec.result,
        source_type=rec.source_type,
        source_id=rec.source_id,
        version=rec.version,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
    )


def _row_to_rec(row: AgentRecommendationRow) -> AgentRecommendation:
    return AgentRecommendation(
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
        status=row.status,  # coerced by pydantic
        operator_decision=row.operator_decision,
        decision_reason=row.decision_reason,
        result=row.result,
        source_type=row.source_type,
        source_id=row.source_id,
        version=row.version,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


# ------------------------------------------------------------------ repository


class RecommendationRepository:
    """Persistence for :class:`AgentRecommendation`."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, rec: AgentRecommendation) -> AgentRecommendation:
        row = _rec_to_row(rec)
        async with self._db.session() as session:
            await session.merge(row)
        logger.info(
            "recommendation_saved",
            extra={"id": rec.id, "parameter": rec.parameter, "status": rec.status, "confidence": rec.confidence},
        )
        return rec

    async def get(self, rec_id: str) -> AgentRecommendation | None:
        async with self._db.session() as session:
            row = await session.get(AgentRecommendationRow, rec_id)
            return _row_to_rec(row) if row else None

    async def list_recent(self, limit: int = 20) -> list[AgentRecommendation]:
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentRecommendationRow).order_by(AgentRecommendationRow.created_at.desc()).limit(limit)
            )
            return [_row_to_rec(r) for r in result.scalars()]

    async def list_by_status(self, status: str | RecommendationStatus) -> list[AgentRecommendation]:
        val = status.value if isinstance(status, RecommendationStatus) else str(status)
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentRecommendationRow)
                .where(AgentRecommendationRow.status == val)
                .order_by(AgentRecommendationRow.created_at.desc())
            )
            return [_row_to_rec(r) for r in result.scalars()]

    async def list_by_parameter(self, parameter: str) -> list[AgentRecommendation]:
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentRecommendationRow)
                .where(AgentRecommendationRow.parameter == parameter)
                .order_by(AgentRecommendationRow.created_at.desc())
            )
            return [_row_to_rec(r) for r in result.scalars()]

    async def list_all(self) -> list[AgentRecommendation]:
        async with self._db.session() as session:
            result = await session.execute(select(AgentRecommendationRow).order_by(AgentRecommendationRow.created_at))
            return [_row_to_rec(r) for r in result.scalars()]


# ------------------------------------------------------------------ service


class RecommendationService:
    """Analytical recommendation service.

    *Creates* recommendations; it does not *apply* them. The boundary is
    enforced by not exposing any ``apply`` / ``commit`` / ``mutate_config``
    method. If such a method is added in the future it must require:

    ``AI recommendation -> human APPROVE -> deterministic validation/safety
    gate -> apply -> audit``

    Phase 1 has no apply path at all.
    """

    def __init__(self, repository: RecommendationRepository) -> None:
        self._repo = repository

    @property
    def repository(self) -> RecommendationRepository:
        return self._repo

    async def create(
        self,
        parameter: str,
        *,
        current_value: str | None,
        proposed_value: str,
        reason: str,
        evidence: tuple[str, ...] | list[str] = (),
        confidence: float = 0.5,
        expected_impact: str = "",
        risk: str = "",
        source_type: str = "manual",
        source_id: str = "advisor",
        status: RecommendationStatus | str = RecommendationStatus.PENDING,
    ) -> AgentRecommendation:
        """Persist a new recommendation and return it.

        The only side effect is the database write; no configuration, risk
        limit, or exchange state is mutated.
        """
        rec = AgentRecommendation(
            parameter=parameter,
            current_value=current_value,
            old_value=current_value,
            proposed_value=proposed_value,
            reason=reason,
            evidence=tuple(evidence),
            confidence=confidence,
            expected_impact=expected_impact,
            risk=risk,
            status=status if isinstance(status, RecommendationStatus) else RecommendationStatus(str(status)),
            source_type=str(source_type),
            source_id=str(source_id),
        )
        await self._repo.save(rec)
        return rec

    async def create_from_model(self, rec: AgentRecommendation) -> AgentRecommendation:
        """Persist an already-constructed recommendation."""
        await self._repo.save(rec)
        return rec

    async def list_recent(self, limit: int = 20) -> list[AgentRecommendation]:
        return await self._repo.list_recent(limit=limit)

    async def list_pending(self) -> list[AgentRecommendation]:
        return await self._repo.list_by_status(RecommendationStatus.PENDING)

    # ------------------------------------------------------------------
    # Explicitly blocked: no config mutation path
    # ------------------------------------------------------------------

    def apply(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
        """Blocked — recommendations cannot be applied directly.

        A future phase may implement a *separate* human-gated service that
        validates, audits, and applies a recommendation. No code path inside
        the advisor may ever mutate configuration without that gate.
        """
        raise RecommendationApplyBlocked(
            "direct application of advisor recommendations is blocked — "
            "human approval + validation gate required (not implemented in Phase 1)"
        )

    def mutate_config(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
        raise RecommendationApplyBlocked("advisor cannot mutate configuration directly")

    def update_risk_limits(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
        raise RecommendationApplyBlocked("advisor cannot modify risk limits directly")
