"""Persistent memory: experiences, lessons, feedback, labels, history (Phase 5).

The memory layer survives restarts because it is backed by the *same*
SQLAlchemy/SQLite database as the trading bot. No second database is created.

Every record carries provenance (source_type / source_id / version / status /
created_at) so that audits can reconstruct where a memory came from.

Phase 5 stores (``ai_*`` mapping): ``ai_memory`` → experiences + lessons;
``ai_experiences`` → :class:`ExperienceRepository`; ``ai_lessons`` →
:class:`LessonRepository`; ``ai_feedback`` → :class:`FeedbackRepository`;
lesson versions → :class:`LessonHistoryRepository`; learning-type overlay →
:class:`MemoryLabelRepository`. Freshness/decay is computed, never stored
destructively: historical records are never deleted because of decay.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from sqlalchemy import select

from app.agent.models import AgentFeedback, Experience, LearningType, Lesson
from app.agent.tables import (
    AgentExperienceRow,
    AgentFeedbackRow,
    AgentLessonHistoryRow,
    AgentLessonRow,
    AgentMemoryLabelRow,
)
from app.storage.engine import Database

__all__ = [
    "ExperienceRepository",
    "FeedbackRepository",
    "LessonHistoryRepository",
    "LessonRepository",
    "MemoryLabelRepository",
    "decayed_confidence",
    "memory_age_days",
    "memory_freshness",
    "revise_lesson",
]

#: Freshness half-life: a 30-day-old memory counts half as much in retrieval.
FRESHNESS_HALF_LIFE_DAYS = 30.0


# ------------------------------------------------------------------ helpers


def _exp_to_row(exp: Experience) -> AgentExperienceRow:
    return AgentExperienceRow(
        id=exp.id,
        situation=exp.situation,
        observation=exp.observation,
        decision=exp.decision,
        result=exp.result,
        lesson=exp.lesson,
        confidence=exp.confidence,
        evidence=list(exp.evidence),
        tags=list(exp.tags),
        source_type=exp.source_type,
        source_id=exp.source_id,
        version=exp.version,
        status=exp.status,
        created_at=exp.created_at,
        updated_at=exp.updated_at,
    )


def _row_to_exp(row: AgentExperienceRow) -> Experience:
    return Experience(
        id=row.id,
        situation=row.situation,
        observation=row.observation,
        decision=row.decision,
        result=row.result,
        lesson=row.lesson,
        confidence=row.confidence,
        evidence=tuple(row.evidence or ()),
        tags=tuple(row.tags or ()),
        source_type=row.source_type,
        source_id=row.source_id,
        version=row.version,
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _lesson_to_row(lesson: Lesson) -> AgentLessonRow:
    return AgentLessonRow(
        id=lesson.id,
        title=lesson.title,
        content=lesson.content,
        pattern=lesson.pattern,
        confidence=lesson.confidence,
        evidence=list(lesson.evidence),
        related_experience_ids=list(lesson.related_experience_ids),
        source_type=lesson.source_type,
        source_id=lesson.source_id,
        version=lesson.version,
        status=lesson.status,
        created_at=lesson.created_at,
        updated_at=lesson.updated_at,
    )


def _row_to_lesson(row: AgentLessonRow) -> Lesson:
    return Lesson(
        id=row.id,
        title=row.title,
        content=row.content,
        pattern=row.pattern,
        confidence=row.confidence,
        evidence=tuple(row.evidence or ()),
        related_experience_ids=tuple(row.related_experience_ids or ()),
        source_type=row.source_type,
        source_id=row.source_id,
        version=row.version,
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


# ------------------------------------------------------------------ repositories


class ExperienceRepository:
    """Persistence for :class:`Experience`."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, exp: Experience) -> Experience:
        row = _exp_to_row(exp)
        async with self._db.session() as session:
            await session.merge(row)
        return exp

    async def get(self, exp_id: str) -> Experience | None:
        async with self._db.session() as session:
            row = await session.get(AgentExperienceRow, exp_id)
            return _row_to_exp(row) if row else None

    async def list_recent(self, limit: int = 20) -> list[Experience]:
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentExperienceRow).order_by(AgentExperienceRow.created_at.desc()).limit(limit)
            )
            return [_row_to_exp(r) for r in result.scalars()]

    async def search(self, query: str, limit: int = 20) -> list[Experience]:
        """Naïve LIKE search over situation/observation/decision fields.

        Phase 1 does not need vector search; a substring match is sufficient
        to prove the memory retrieval path. Future phases may add embeddings.
        """
        pattern = f"%{query}%"
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentExperienceRow)
                .where(
                    (AgentExperienceRow.situation.like(pattern))
                    | (AgentExperienceRow.observation.like(pattern))
                    | (AgentExperienceRow.decision.like(pattern))  # type: ignore[arg-type]
                )
                .order_by(AgentExperienceRow.created_at.desc())
                .limit(limit)
            )
            return [_row_to_exp(r) for r in result.scalars()]

    async def list_all(self) -> list[Experience]:
        async with self._db.session() as session:
            result = await session.execute(select(AgentExperienceRow).order_by(AgentExperienceRow.created_at))
            return [_row_to_exp(r) for r in result.scalars()]

    async def count(self) -> int:
        from sqlalchemy import func

        async with self._db.session() as session:
            result = await session.execute(select(func.count()).select_from(AgentExperienceRow))
            return int(result.scalar_one())


class LessonRepository:
    """Persistence for :class:`Lesson`."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, lesson: Lesson) -> Lesson:
        row = _lesson_to_row(lesson)
        async with self._db.session() as session:
            await session.merge(row)
        return lesson

    async def get(self, lesson_id: str) -> Lesson | None:
        async with self._db.session() as session:
            row = await session.get(AgentLessonRow, lesson_id)
            return _row_to_lesson(row) if row else None

    async def list_recent(self, limit: int = 20) -> list[Lesson]:
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentLessonRow).order_by(AgentLessonRow.created_at.desc()).limit(limit)
            )
            return [_row_to_lesson(r) for r in result.scalars()]

    async def list_all(self) -> list[Lesson]:
        async with self._db.session() as session:
            result = await session.execute(select(AgentLessonRow).order_by(AgentLessonRow.created_at))
            return [_row_to_lesson(r) for r in result.scalars()]

    async def search(self, query: str, limit: int = 20) -> list[Lesson]:
        pattern = f"%{query}%"
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentLessonRow)
                .where(
                    (AgentLessonRow.title.like(pattern))
                    | (AgentLessonRow.content.like(pattern))
                    | (AgentLessonRow.pattern.like(pattern))  # type: ignore[arg-type]
                )
                .order_by(AgentLessonRow.created_at.desc())
                .limit(limit)
            )
            return [_row_to_lesson(r) for r in result.scalars()]


# ------------------------------------------------------------------ freshness / decay (Phase 5 §5)


def memory_age_days(created_at: datetime | None, *, now: datetime | None = None) -> float:
    """Age of a memory record in days (0.0 for missing timestamps)."""
    if created_at is None:
        return 0.0
    try:
        from app.models.base import utc_now

        ref = now or utc_now()
        naive_created = created_at.replace(tzinfo=None) if created_at.tzinfo else created_at
        naive_ref = ref.replace(tzinfo=None) if ref.tzinfo else ref
        return max(0.0, (naive_ref - naive_created).total_seconds() / 86400.0)
    except Exception:
        return 0.0


def freshness_weight(age_days: float, *, half_life_days: float = FRESHNESS_HALF_LIFE_DAYS) -> float:
    """Deterministic decay weight: 1.0 when new, halving every half-life.

    Older observations never vanish (weight stays > 0); decay affects
    retrieval relevance/confidence only.
    """
    if age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def decayed_confidence(confidence: float, age_days: float) -> float:
    """Effective confidence after decay (rounded for stable test assertions)."""
    return round(float(confidence) * freshness_weight(age_days), 4)


def memory_freshness(
    created_at: datetime | None,
    confidence: float,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Freshness metadata bundle for retrieval (age, weight, decayed confidence)."""
    age = memory_age_days(created_at, now=now)
    weight = round(freshness_weight(age), 4)
    return {
        "age_days": round(age, 2),
        "weight": weight,
        "confidence": float(confidence),
        "decayed_confidence": decayed_confidence(confidence, age),
    }


# ------------------------------------------------------------------ feedback (Phase 5 ai_feedback)


class FeedbackRepository:
    """Persistence for :class:`AgentFeedback` (human-only)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def submit(
        self,
        *,
        target_type: str,
        target_id: str,
        rating: int,
        comment: str = "",
        approver: str,
    ) -> AgentFeedback:
        """Record human feedback. ``approver`` must be non-empty (fail-closed)."""
        feedback = AgentFeedback(
            target_type=target_type,
            target_id=target_id,
            rating=rating,
            comment=comment,
            approver=approver,
        )
        row = AgentFeedbackRow(
            id=feedback.id,
            target_type=feedback.target_type,
            target_id=feedback.target_id,
            rating=feedback.rating,
            comment=feedback.comment,
            approver=feedback.approver,
            created_at=feedback.created_at,
        )
        async with self._db.session() as session:
            await session.merge(row)
        return feedback

    async def list_for_target(self, target_type: str, target_id: str) -> list[AgentFeedback]:
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentFeedbackRow)
                .where(
                    AgentFeedbackRow.target_type == str(target_type).strip().lower(),
                    AgentFeedbackRow.target_id == str(target_id),
                )
                .order_by(AgentFeedbackRow.created_at)
            )
            return [
                AgentFeedback(
                    id=r.id,
                    target_type=r.target_type,
                    target_id=r.target_id,
                    rating=r.rating,
                    comment=r.comment,
                    approver=r.approver,
                    created_at=r.created_at,
                )
                for r in result.scalars()
            ]

    async def score_for_target(self, target_type: str, target_id: str) -> dict[str, Any]:
        """Net feedback score (+1/-1 sums) with sample size."""
        items = await self.list_for_target(target_type, target_id)
        return {
            "n": len(items),
            "score": sum(i.rating for i in items),
            "up": sum(1 for i in items if i.rating > 0),
            "down": sum(1 for i in items if i.rating < 0),
        }


# ------------------------------------------------------------------ lesson history (Phase 5 §7 versioning)


class LessonHistoryRepository:
    """Auditable snapshots of superseded lesson versions."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def snapshot(
        self, lesson: Lesson, *, reason: str = "", superseded_by: str | None = None
    ) -> dict[str, Any]:
        """Persist the current lesson state before it is revised."""
        entry = {
            "id": f"lh-{uuid.uuid4().hex[:12]}",
            "lesson_id": lesson.id,
            "version": lesson.version,
            "snapshot": lesson.model_dump(mode="json"),
            "reason": reason,
            "superseded_by": superseded_by,
        }
        row = AgentLessonHistoryRow(
            id=entry["id"],
            lesson_id=lesson.id,
            version=lesson.version,
            snapshot=entry["snapshot"],
            reason=reason,
            superseded_by=superseded_by,
        )
        async with self._db.session() as session:
            await session.merge(row)
        return entry

    async def get_versions(self, lesson_id: str) -> list[dict[str, Any]]:
        """All snapshots for ``lesson_id``, oldest first (auditable history)."""
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentLessonHistoryRow)
                .where(AgentLessonHistoryRow.lesson_id == lesson_id)
                .order_by(AgentLessonHistoryRow.version)
            )
            return [
                {
                    "id": r.id,
                    "lesson_id": r.lesson_id,
                    "version": r.version,
                    "snapshot": r.snapshot,
                    "reason": r.reason,
                    "superseded_by": r.superseded_by,
                    "created_at": r.created_at,
                }
                for r in result.scalars()
            ]


# ------------------------------------------------------------------ memory labels (Phase 5 overlay)


class MemoryLabelRepository:
    """Learning-type / sample-size / direction overlay for experiences+lessons.

    Existing tables are never altered; labels live in ``agent_memory_labels``
    keyed by ``(target_type, target_id)``. ``target_type`` is ``experience``
    or ``lesson``; ``learning_type`` is a :class:`LearningType` value.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def set_label(
        self,
        *,
        target_type: str,
        target_id: str,
        learning_type: str | LearningType,
        sample_size: int = 1,
        direction: str = "unknown",
        supersedes: str | None = None,
    ) -> dict[str, Any]:
        lt = learning_type.value if isinstance(learning_type, LearningType) else str(learning_type).strip().lower()
        if lt not in tuple(t.value for t in LearningType):
            raise ValueError(f"unknown learning type: {learning_type!r}")
        label = {
            "target_type": str(target_type).strip().lower(),
            "target_id": str(target_id),
            "learning_type": lt,
            "sample_size": max(0, int(sample_size)),
            "direction": str(direction).strip().lower() or "unknown",
            "supersedes": supersedes,
        }
        row = AgentMemoryLabelRow(
            target_type=label["target_type"],
            target_id=label["target_id"],
            learning_type=label["learning_type"],
            sample_size=label["sample_size"],
            direction=label["direction"],
            supersedes=supersedes,
        )
        async with self._db.session() as session:
            await session.merge(row)
        return label

    async def get_label(self, target_type: str, target_id: str) -> dict[str, Any] | None:
        async with self._db.session() as session:
            row = await session.get(
                AgentMemoryLabelRow, (str(target_type).strip().lower(), str(target_id))
            )
            if row is None:
                return None
            return {
                "target_type": row.target_type,
                "target_id": row.target_id,
                "learning_type": row.learning_type,
                "sample_size": row.sample_size,
                "direction": row.direction,
                "supersedes": row.supersedes,
                "updated_at": row.updated_at,
            }

    async def labels_for(self, target_type: str, target_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Batch lookup (bounded); missing ids simply absent from the result."""
        out: dict[str, dict[str, Any]] = {}
        for tid in target_ids[:50]:
            label = await self.get_label(target_type, tid)
            if label is not None:
                out[tid] = label
        return out


# ------------------------------------------------------------------ versioning (Phase 5 §7)


async def revise_lesson(
    lesson_repo: LessonRepository,
    history_repo: LessonHistoryRepository,
    lesson_id: str,
    *,
    new_evidence_ids: list[str] | tuple[str, ...] = (),
    new_experience_ids: list[str] | tuple[str, ...] = (),
    content: str | None = None,
    confidence: float | None = None,
    reason: str = "",
) -> Lesson:
    """Revise a lesson with new evidence, keeping the previous version auditable.

    The current snapshot is stored in history BEFORE the update; the lesson
    keeps its id with ``version + 1`` and the union of evidence ids, so the
    current version is distinguishable from historical ones and every update
    identifies its source evidence. Raises ``KeyError`` for unknown lessons.
    """
    from app.models.base import utc_now

    lesson = await lesson_repo.get(lesson_id)
    if lesson is None:
        raise KeyError(f"lesson not found: {lesson_id}")
    merged_evidence = tuple(dict.fromkeys(list(lesson.evidence) + [str(e) for e in new_evidence_ids]))
    merged_related = tuple(dict.fromkeys(list(lesson.related_experience_ids) + [str(e) for e in new_experience_ids]))
    updated = lesson.model_copy(update={
        "evidence": merged_evidence,
        "related_experience_ids": merged_related,
        "content": content if content is not None else lesson.content,
        "confidence": confidence if confidence is not None else lesson.confidence,
        "version": lesson.version + 1,
        "updated_at": utc_now(),
    })
    await history_repo.snapshot(
        lesson,
        reason=reason or "lesson revised",
        superseded_by=f"{lesson.id}@v{updated.version}",
    )
    await lesson_repo.save(updated)
    return updated
