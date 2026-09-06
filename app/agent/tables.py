"""SQLAlchemy tables for the AI Advisor subsystem.

Four tables cover the advisor persistence surface:

* ``agent_knowledge``      — ingested documentation / research / knowledge docs
* ``agent_experiences``    — structured memory of situations & outcomes
* ``agent_lessons``        — distilled patterns from experiences
* ``agent_recommendations`` — advisor suggestions awaiting human approval

All tables live in the SAME database as the trading bot (``trades``,
``transfers``, ``balances``, ``audit_log``, ``bot_state``). There is no
second database — restarts use the same file and the same
``Base.metadata.create_all`` path.

Every row carries provenance columns (``source_type``, ``source_id``,
``version``, ``status``, ``created_at``, ``updated_at``) so the advisor
never forgets where a piece of knowledge came from.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from sqlalchemy import JSON as SA_JSON

from app.models.base import utc_now
from app.storage.base import UTC_DATETIME, Base

__all__ = [
    "AgentExperienceRow",
    "AgentKnowledgeRow",
    "AgentLessonRow",
    "AgentRecommendationRow",
]


class AgentKnowledgeRow(Base):
    __tablename__ = "agent_knowledge"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(256))
    category: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str]] = mapped_column(SA_JSON, default=list)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(SA_JSON, nullable=True)
    # provenance
    source_type: Mapped[str] = mapped_column(String(32))
    source_id: Mapped[str] = mapped_column(String(256))
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)

    __table_args__ = (
        Index("ix_agent_knowledge_category", "category"),
        Index("ix_agent_knowledge_source_id", "source_id"),
        Index("ix_agent_knowledge_created_at", "created_at"),
    )


class AgentExperienceRow(Base):
    __tablename__ = "agent_experiences"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    situation: Mapped[str] = mapped_column(Text)
    observation: Mapped[str] = mapped_column(Text)
    decision: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    lesson: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(default=0.5)
    evidence: Mapped[list[str]] = mapped_column(SA_JSON, default=list)
    tags: Mapped[list[str]] = mapped_column(SA_JSON, default=list)
    # provenance
    source_type: Mapped[str] = mapped_column(String(32))
    source_id: Mapped[str] = mapped_column(String(256))
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)

    __table_args__ = (
        Index("ix_agent_experiences_source_id", "source_id"),
        Index("ix_agent_experiences_created_at", "created_at"),
    )


class AgentLessonRow(Base):
    __tablename__ = "agent_lessons"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(256))
    content: Mapped[str] = mapped_column(Text)
    pattern: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(default=0.5)
    evidence: Mapped[list[str]] = mapped_column(SA_JSON, default=list)
    related_experience_ids: Mapped[list[str]] = mapped_column(SA_JSON, default=list)
    # provenance
    source_type: Mapped[str] = mapped_column(String(32))
    source_id: Mapped[str] = mapped_column(String(256))
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)

    __table_args__ = (
        Index("ix_agent_lessons_created_at", "created_at"),
    )


class AgentRecommendationRow(Base):
    __tablename__ = "agent_recommendations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    parameter: Mapped[str] = mapped_column(String(128))
    current_value: Mapped[str | None] = mapped_column(String(256), nullable=True)
    old_value: Mapped[str | None] = mapped_column(String(256), nullable=True)
    proposed_value: Mapped[str] = mapped_column(String(256))
    reason: Mapped[str] = mapped_column(Text)
    evidence: Mapped[list[str]] = mapped_column(SA_JSON, default=list)
    confidence: Mapped[float] = mapped_column(default=0.5)
    expected_impact: Mapped[str] = mapped_column(Text, default="")
    risk: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    operator_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    # provenance
    source_type: Mapped[str] = mapped_column(String(32))
    source_id: Mapped[str] = mapped_column(String(256))
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)

    __table_args__ = (
        Index("ix_agent_recommendations_parameter", "parameter"),
        Index("ix_agent_recommendations_status", "status"),
        Index("ix_agent_recommendations_created_at", "created_at"),
    )
