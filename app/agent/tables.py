"""SQLAlchemy tables for the AI Advisor subsystem.

Nine tables cover the advisor persistence surface (Phase 5):

* ``agent_knowledge``      — ingested documentation / research / knowledge docs
* ``agent_knowledge_chunks`` — deterministic retrieval chunks (Phase 3)
* ``agent_experiences``    — structured memory of situations & outcomes
* ``agent_lessons``        — distilled patterns from experiences
* ``agent_recommendations`` — advisor suggestions awaiting human approval
* ``agent_audit``          — analysis / recommendation / reflection actions
* ``agent_feedback``       — human feedback on lessons/recommendations (Phase 5)
* ``agent_lesson_history`` — superseded lesson versions, auditable (Phase 5)
* ``agent_memory_labels``  — learning-type / sample-size / direction overlay (Phase 5)
* ``agent_measurements``   — before/after measurements of approved changes (Phase 9)

Phase 5 store mapping: ``ai_memory`` → experiences+lessons; ``ai_experiences``
→ agent_experiences; ``ai_lessons`` → agent_lessons; ``ai_recommendations`` →
agent_recommendations; ``ai_knowledge_sources``/``ai_knowledge_documents`` →
agent_knowledge(+chunks); ``ai_actions`` → agent_audit; ``ai_feedback`` →
agent_feedback.

All tables live in the SAME database as the trading bot (``trades``,
``transfers``, ``balances``, ``audit_log``, ``bot_state``). There is no
second database — restarts use the same file and the same
``Base.metadata.create_all`` path. Phase 5 adds only NEW tables; existing
tables are never altered, so upgrades never trigger schema-drift rebuilds.

Every row carries provenance columns (``source_type``, ``source_id``,
``version``, ``status``, ``created_at``, ``updated_at``) so the advisor
never forgets where a piece of knowledge came from.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from sqlalchemy import JSON as SA_JSON

from app.models.base import utc_now
from app.storage.base import UTC_DATETIME, Base

__all__ = [
    "AgentExperienceRow",
    "AgentFeedbackRow",
    "AgentKnowledgeChunkRow",
    "AgentKnowledgeRow",
    "AgentLessonHistoryRow",
    "AgentLessonRow",
    "AgentMeasurementRow",
    "AgentMemoryLabelRow",
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


class AgentKnowledgeChunkRow(Base):
    """Phase 3 retrieval unit — one deterministic chunk of a knowledge doc.

    New table (``create_all`` creates it on fresh DBs; existing DBs gain it
    without touching ``agent_knowledge``). Every chunk denormalizes its
    parent provenance so retrieval never loses the source.
    """

    __tablename__ = "agent_knowledge_chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(32))
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    section: Mapped[str] = mapped_column(String(256), default="")
    content: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String(256), default="")
    category: Mapped[str] = mapped_column(String(32), default="research")
    source_type: Mapped[str] = mapped_column(String(32), default="document")
    source_id: Mapped[str] = mapped_column(String(256), default="")
    document_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    verification_status: Mapped[str] = mapped_column(String(32), default="unverified")
    confidence: Mapped[float] = mapped_column(default=0.5)
    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)

    __table_args__ = (
        Index("ix_agent_chunks_doc_id", "doc_id"),
        Index("ix_agent_chunks_category", "category"),
        Index("ix_agent_chunks_source_id", "source_id"),
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


class AgentFeedbackRow(Base):
    """Phase 5 ``ai_feedback`` — human-only feedback on agent artifacts."""

    __tablename__ = "agent_feedback"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    target_type: Mapped[str] = mapped_column(String(32))  # lesson | recommendation
    target_id: Mapped[str] = mapped_column(String(32))
    rating: Mapped[int] = mapped_column(Integer)  # +1 | -1
    comment: Mapped[str] = mapped_column(Text, default="")
    approver: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)

    __table_args__ = (
        Index("ix_agent_feedback_target", "target_type", "target_id"),
        Index("ix_agent_feedback_created_at", "created_at"),
    )


class AgentLessonHistoryRow(Base):
    """Phase 5 lesson versioning — superseded snapshots stay auditable."""

    __tablename__ = "agent_lesson_history"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    lesson_id: Mapped[str] = mapped_column(String(32))
    version: Mapped[int] = mapped_column(Integer)
    snapshot: Mapped[dict[str, Any]] = mapped_column(SA_JSON)
    reason: Mapped[str] = mapped_column(Text, default="")
    superseded_by: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)

    __table_args__ = (
        Index("ix_agent_lesson_history_lesson", "lesson_id", "version"),
    )


class AgentMemoryLabelRow(Base):
    """Phase 5 learning-type overlay — no alterations to existing tables.

    One row per experience/lesson carrying its :class:`LearningType`,
    explicit sample size, outcome direction and optional supersession link.
    Decay affects retrieval weight only; rows (and history) are never
    deleted because of decay.
    """

    __tablename__ = "agent_memory_labels"

    target_type: Mapped[str] = mapped_column(String(32), primary_key=True)  # experience | lesson
    target_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    learning_type: Mapped[str] = mapped_column(String(32), default="observation")
    sample_size: Mapped[int] = mapped_column(Integer, default=1)
    direction: Mapped[str] = mapped_column(String(32), default="unknown")
    supersedes: Mapped[str | None] = mapped_column(String(32), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)

    __table_args__ = (
        Index("ix_agent_memory_labels_type", "learning_type"),
    )


class AgentMeasurementRow(Base):
    """Phase 9 ``measurement`` — before/after verdicts for approved changes.

    New table only; existing tables are never altered. ``details`` carries
    window stats, trade id lists (bounded) and the optional AI note.
    """

    __tablename__ = "agent_measurements"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    recommendation_id: Mapped[str] = mapped_column(String(32))
    parameter: Mapped[str] = mapped_column(String(128))
    old_value: Mapped[str] = mapped_column(String(256))
    new_value: Mapped[str] = mapped_column(String(256))
    metric: Mapped[str] = mapped_column(String(64))
    higher_is_better: Mapped[bool] = mapped_column(default=False)
    before_value: Mapped[str] = mapped_column(String(64))
    after_value: Mapped[str] = mapped_column(String(64))
    delta: Mapped[str] = mapped_column(String(64))
    n_before: Mapped[int] = mapped_column(Integer, default=0)
    n_after: Mapped[int] = mapped_column(Integer, default=0)
    outcome: Mapped[str] = mapped_column(String(32), default="insufficient")
    confidence: Mapped[float] = mapped_column(default=0.0)
    reason: Mapped[str] = mapped_column(Text, default="")
    feedback_kind: Mapped[str | None] = mapped_column(String(32), nullable=True)
    feedback_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    feedback_at: Mapped[datetime | None] = mapped_column(UTC_DATETIME, nullable=True)
    lesson_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    evidence_before: Mapped[list[str]] = mapped_column(SA_JSON, default=list)
    evidence_after: Mapped[list[str]] = mapped_column(SA_JSON, default=list)
    details: Mapped[dict[str, Any] | None] = mapped_column(SA_JSON, nullable=True)
    source_type: Mapped[str] = mapped_column(String(32), default="system")
    source_id: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)

    __table_args__ = (
        Index("ix_agent_measurements_rec", "recommendation_id"),
        Index("ix_agent_measurements_outcome", "outcome"),
        Index("ix_agent_measurements_created_at", "created_at"),
    )
