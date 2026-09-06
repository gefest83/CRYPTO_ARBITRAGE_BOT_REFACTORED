"""Live/Journal context collection for the advisor.

Context is the *current* snapshot the advisor reasons over: recent trades,
journal entries, active risk state, market-data freshness and the knowledge
slices that are relevant to a user question.

The collector touches only the read-only tools — no trading state is mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent.knowledge import KnowledgeService
from app.agent.memory import ExperienceRepository, LessonRepository
from app.agent.recommendations import RecommendationRepository
from app.agent.tools import AgentTools

__all__ = ["AgentContext", "ContextCollector"]


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Snapshot gathered for one advisor request."""

    # raw tool outputs (already sanitized)
    recent_trades: tuple[dict[str, Any], ...] = ()
    trade_statistics: dict[str, Any] = field(default_factory=dict)
    scan_statistics: dict[str, Any] = field(default_factory=dict)
    current_parameters: dict[str, Any] = field(default_factory=dict)
    risk_state: dict[str, Any] = field(default_factory=dict)
    exchange_status: dict[str, Any] = field(default_factory=dict)
    balances: dict[str, Any] = field(default_factory=dict)
    recent_journal: tuple[dict[str, Any], ...] = ()
    previous_recommendations: tuple[dict[str, Any], ...] = ()
    # memory / knowledge slices
    experiences: tuple[dict[str, Any], ...] = ()
    lessons: tuple[dict[str, Any], ...] = ()
    knowledge_hits: tuple[dict[str, Any], ...] = ()
    # request metadata
    query: str | None = None
    language: str | None = None

    def summary(self) -> str:
        """Concise text summary for the LLM prompt / human report."""
        lines: list[str] = [
            f"Recent trades: {len(self.recent_trades)}",
            f"Trade stats: {self.trade_statistics}",
            f"Scan stats: {self.scan_statistics}",
            f"Risk: {self.risk_state}",
            f"Exchanges: {list(self.exchange_status.keys())}",
            f"Balances venues: {list(self.balances.keys())}",
            f"Journal entries: {len(self.recent_journal)}",
            f"Previous recommendations: {len(self.previous_recommendations)}",
            f"Experiences in context: {len(self.experiences)}",
            f"Lessons in context: {len(self.lessons)}",
            f"Knowledge hits: {len(self.knowledge_hits)}",
        ]
        if self.query:
            lines.append(f"Query: {self.query}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "recent_trades": list(self.recent_trades),
            "trade_statistics": self.trade_statistics,
            "scan_statistics": self.scan_statistics,
            "current_parameters": self.current_parameters,
            "risk_state": self.risk_state,
            "exchange_status": self.exchange_status,
            "balances": self.balances,
            "recent_journal": list(self.recent_journal),
            "previous_recommendations": list(self.previous_recommendations),
            "experiences": list(self.experiences),
            "lessons": list(self.lessons),
            "knowledge_hits": list(self.knowledge_hits),
            "query": self.query,
        }


class ContextCollector:
    """Collects :class:`AgentContext` via the read-only tools plus memory/KB retrieval.

    The collector is deliberately *synchronous-looking* but fully async: each
    data source is fetched via the allowlisted tools, no direct DB/SQL is
    exposed, and no credentials are ever included.
    """

    def __init__(
        self,
        tools: AgentTools,
        *,
        knowledge_service: KnowledgeService | None = None,
        experience_repo: ExperienceRepository | None = None,
        lesson_repo: LessonRepository | None = None,
        recommendation_repo: RecommendationRepository | None = None,
    ) -> None:
        self._tools = tools
        self._knowledge = knowledge_service
        self._experiences = experience_repo
        self._lessons = lesson_repo
        self._recommendations = recommendation_repo

    async def collect(
        self,
        query: str | None = None,
        language: str | None = None,
        *,
        include_balances: bool = True,
        trade_limit: int = 20,
        journal_limit: int = 20,
        memory_limit: int = 10,
    ) -> AgentContext:
        """Gather a full context snapshot.

        ``query`` is used to retrieve the most relevant knowledge / memory
        slices (substring search). When ``None`` the most recent items are
        returned instead.
        """
        # Tool-sourced sections (read-only)
        recent_trades = await self._tools.get_recent_trades(limit=trade_limit)
        trade_stats = await self._tools.get_trade_statistics()
        scan_stats = await self._tools.get_scan_statistics()
        current_params = await self._tools.get_current_parameters()
        risk_state = await self._tools.get_risk_state()
        exchange_status = await self._tools.get_exchange_status()
        balances: dict[str, Any] = {}
        if include_balances:
            try:
                balances = await self._tools.get_balances()
            except Exception:
                balances = {}
        recent_journal = await self._tools.get_recent_journal(limit=journal_limit)
        previous_recs = await self._tools.get_previous_recommendations(limit=10)

        # Memory / KB slices (filtered by query when provided)
        experiences: list[dict[str, Any]] = []
        lessons: list[dict[str, Any]] = []
        knowledge_hits: list[dict[str, Any]] = []

        if query and self._knowledge is not None:
            try:
                hits = await self._knowledge.search(query, limit=memory_limit)
                knowledge_hits = [
                    {"id": h.id, "title": h.title, "category": str(h.category), "summary": h.summary, "source_id": h.source_id}
                    for h in hits
                ]
            except Exception:
                pass
        elif self._knowledge is not None:
            try:
                recent = await self._knowledge.get_recent(limit=memory_limit)
                knowledge_hits = [
                    {"id": h.id, "title": h.title, "category": str(h.category), "summary": h.summary}
                    for h in recent
                ]
            except Exception:
                pass

        if self._experiences is not None:
            try:
                exps = (
                    await self._experiences.search(query, limit=memory_limit)
                    if query
                    else await self._experiences.list_recent(limit=memory_limit)
                )
                experiences = [
                    {"id": e.id, "situation": e.situation, "observation": e.observation, "lesson": e.lesson, "confidence": e.confidence}
                    for e in exps
                ]
            except Exception:
                pass

        if self._lessons is not None:
            try:
                les = (
                    await self._lessons.search(query, limit=memory_limit)
                    if query
                    else await self._lessons.list_recent(limit=memory_limit)
                )
                lessons = [
                    {"id": le.id, "title": le.title, "content": le.content, "confidence": le.confidence}
                    for le in les
                ]
            except Exception:
                pass

        return AgentContext(
            recent_trades=tuple(recent_trades),
            trade_statistics=trade_stats,
            scan_statistics=scan_stats,
            current_parameters=current_params,
            risk_state=risk_state,
            exchange_status=exchange_status,
            balances=balances,
            recent_journal=tuple(recent_journal),
            previous_recommendations=tuple(previous_recs),
            experiences=tuple(experiences),
            lessons=tuple(lessons),
            knowledge_hits=tuple(knowledge_hits),
            query=query,
            language=language,
        )
