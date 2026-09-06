"""Agent audit persistence — analysis and recommendation lifecycle events.

Every important agent event is persisted with provenance, provider/model,
and timestamps — but never credentials. The table is crash/restart safe
because it lives in the same DB and uses transactional writes.

Events:
* analysis — one per AgentCore.handle() (facts/observations/hypotheses,
  evidence_count/confidence/action, provider/model)
* recommendation_created — when a recommendation is synthesised
* recommendation_approved / rejected — via approval service
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import Field
from sqlalchemy import Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.agent.models import DomainModel, utc_now
from app.storage.base import UTC_DATETIME, Base

# Import JSON type
from sqlalchemy import JSON as SA_JSON

__all__ = ["AgentAuditEvent", "AgentAuditRepository", "AgentAuditRow"]


class AgentAuditRow(Base):
    __tablename__ = "agent_audit"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(32))  # analysis, recommendation_created, approved, rejected, provider_call
    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    query: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_count: Mapped[int | None] = mapped_column(nullable=True)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    details: Mapped[dict[str, Any] | None] = mapped_column(SA_JSON, nullable=True)
    source_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)

    __table_args__ = (
        Index("ix_agent_audit_type", "event_type"),
        Index("ix_agent_audit_created", "created_at"),
    )


class AgentAuditEvent(DomainModel):
    """Immutable audit event — no secrets."""

    id: str = Field(default_factory=lambda: f"aad-{uuid.uuid4().hex[:12]}")
    event_type: str
    provider: str | None = None
    model: str | None = None
    query: str | None = None
    evidence_count: int | None = None
    confidence: float | None = None
    action: str | None = None
    details: dict[str, Any] | None = None
    source_type: str | None = None
    source_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


def _event_to_row(ev: AgentAuditEvent) -> AgentAuditRow:
    return AgentAuditRow(
        id=ev.id,
        event_type=ev.event_type,
        provider=ev.provider,
        model=ev.model,
        query=ev.query[:2000] if ev.query else None,
        evidence_count=ev.evidence_count,
        confidence=ev.confidence,
        action=ev.action,
        details=ev.details,
        source_type=ev.source_type,
        source_id=ev.source_id,
        created_at=ev.created_at,
    )


def _row_to_event(row: AgentAuditRow) -> AgentAuditEvent:
    return AgentAuditEvent(
        id=row.id,
        event_type=row.event_type,
        provider=row.provider,
        model=row.model,
        query=row.query,
        evidence_count=row.evidence_count,
        confidence=row.confidence,
        action=row.action,
        details=row.details,
        source_type=row.source_type,
        source_id=row.source_id,
        created_at=row.created_at,
    )


class AgentAuditRepository:
    """Persistence for agent audit events."""

    def __init__(self, db):  # type: ignore[no-untyped-def]
        self._db = db

    async def log(self, event: AgentAuditEvent) -> AgentAuditEvent:
        # Never persist credentials — defensive filter on details
        if event.details:
            # Ensure no secret keys in details
            txt = str(event.details).lower()
            for forbidden in ("api_key", "secret", "password", "cat_key", "cat_telegram", "bearer"):
                if forbidden in txt:
                    # Replace details with redacted note
                    event = event.model_copy(update={"details": {"note": "<redacted>"}})
                    break
        row = _event_to_row(event)
        async with self._db.session() as session:
            session.add(row)
        return event

    async def log_analysis(
        self,
        *,
        provider: str | None,
        model: str | None,
        query: str | None,
        evidence_count: int | None,
        confidence: float | None,
        action: str | None,
        facts_count: int | None = None,
        hypotheses_count: int | None = None,
        recommendation_id: str | None = None,
    ) -> AgentAuditEvent:
        ev = AgentAuditEvent(
            event_type="analysis",
            provider=provider,
            model=model,
            query=query[:500] if query else None,
            evidence_count=evidence_count,
            confidence=confidence,
            action=action,
            details={
                "facts_count": facts_count,
                "hypotheses_count": hypotheses_count,
                "recommendation_id": recommendation_id,
            },
            source_type="agent",
            source_id="analysis",
        )
        return await self.log(ev)

    async def log_recommendation(self, rec, *, event_type: str = "recommendation_created") -> AgentAuditEvent:
        ev = AgentAuditEvent(
            event_type=event_type,
            provider=None,
            model=None,
            query=None,
            evidence_count=len(rec.evidence) if hasattr(rec, "evidence") else None,
            confidence=getattr(rec, "confidence", None),
            action=getattr(rec, "status", None).value if hasattr(getattr(rec, "status", None), "value") else str(getattr(rec, "status", "")),
            details={
                "recommendation_id": rec.id,
                "parameter": rec.parameter,
                "proposed_value": rec.proposed_value,
                "reason": (rec.reason[:200] if hasattr(rec, "reason") else None),
            },
            source_type=rec.source_type if hasattr(rec, "source_type") else "recommendation",
            source_id=rec.source_id if hasattr(rec, "source_id") else rec.id,
        )
        return await self.log(ev)

    async def list_recent(self, limit: int = 20) -> list[AgentAuditEvent]:
        from sqlalchemy import select

        async with self._db.session() as session:
            result = await session.execute(select(AgentAuditRow).order_by(AgentAuditRow.created_at.desc()).limit(limit))
            return [_row_to_event(r) for r in result.scalars()]

    async def count(self) -> int:
        from sqlalchemy import func, select

        async with self._db.session() as session:
            result = await session.execute(select(func.count()).select_from(AgentAuditRow))
            return int(result.scalar_one())
