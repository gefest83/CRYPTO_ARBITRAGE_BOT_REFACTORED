"""Persistent memory: experiences and lessons.

The memory layer survives restarts because it is backed by the *same*
SQLAlchemy/SQLite database as the trading bot. No second database is created.

Every record carries provenance (source_type / source_id / version / status /
created_at) so that audits can reconstruct where a memory came from.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from app.agent.models import Experience, Lesson
from app.agent.tables import AgentExperienceRow, AgentLessonRow
from app.storage.engine import Database

__all__ = ["ExperienceRepository", "LessonRepository"]


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
