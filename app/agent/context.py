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
    # Phase 4: deterministic journal-analysis results (FACTS, computed by app code)
    journal_analysis: tuple[dict[str, Any], ...] = ()
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
            f"Journal analyses: {len(self.journal_analysis)}",
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
            "journal_analysis": list(self.journal_analysis),
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
        journal_reader: Any | None = None,
        label_repository: Any | None = None,
    ) -> None:
        self._tools = tools
        self._knowledge = knowledge_service
        self._experiences = experience_repo
        self._lessons = lesson_repo
        self._recommendations = recommendation_repo
        self._journal = journal_reader
        self._labels = label_repository

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
        """Gather a full context snapshot — bounded and treated as untrusted.

        ``query`` is used to retrieve the most relevant knowledge / memory
        slices (substring search). When ``None`` the most recent items are
        returned instead.

        Phase 3C hardening: all retrieval is hard-capped (trades 5, journal
        5, memory/knowledge 3) and every free-text field is truncated to
        500 chars. The raw stored text keeps provenance, but the in-context
        copy is bounded so that no unrestricted dump reaches the LLM.
        """
        # Hard caps — even if caller asks for more, we bound for LLM safety
        trade_limit = min(int(trade_limit), 5)
        journal_limit = min(int(journal_limit), 5)
        memory_limit = min(int(memory_limit), 3)
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
        previous_recs = await self._tools.get_previous_recommendations(limit=5)

        # Memory / KB slices (filtered by query when provided)
        experiences: list[dict[str, Any]] = []
        lessons: list[dict[str, Any]] = []
        knowledge_hits: list[dict[str, Any]] = []

        if query and self._knowledge is not None:
            try:
                # Phase 3: chunk-level relevance retrieval with provenance.
                # Falls back to document search when chunks are unavailable.
                chunk_hits = []
                try:
                    chunk_hits = await self._knowledge.search_chunks(query, limit=memory_limit)
                except Exception:
                    chunk_hits = []
                if chunk_hits:
                    knowledge_hits = chunk_hits
                else:
                    hits = await self._knowledge.search(query, limit=memory_limit)
                    knowledge_hits = [
                        {
                            "id": h.id,
                            "title": h.title,
                            "category": str(h.category),
                            "summary": h.summary,
                            "source_id": h.source_id,
                            "source_type": h.source_type,
                            "document_path": h.effective_document_path,
                            "section": h.section,
                            "verification_status": h.verification_status,
                            "confidence": h.confidence,
                        }
                        for h in hits
                    ]
            except Exception:
                pass
        elif self._knowledge is not None:
            try:
                recent = await self._knowledge.get_recent(limit=memory_limit)
                knowledge_hits = [
                    {
                        "id": h.id,
                        "title": h.title,
                        "category": str(h.category),
                        "summary": h.summary,
                        "source_id": h.source_id,
                        "source_type": h.source_type,
                        "document_path": h.effective_document_path,
                        "section": h.section,
                        "verification_status": h.verification_status,
                        "confidence": h.confidence,
                    }
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
                label_map = await _labels_for(self._labels, "experience", [e.id for e in exps])
                experiences = _rerank_by_decay([_enrich_experience(e, label_map) for e in exps])[:memory_limit]
            except Exception:
                pass

        if self._lessons is not None:
            try:
                les = (
                    await self._lessons.search(query, limit=memory_limit)
                    if query
                    else await self._lessons.list_recent(limit=memory_limit)
                )
                label_map = await _labels_for(self._labels, "lesson", [le.id for le in les])
                lessons = _rerank_by_decay([_enrich_lesson(le, label_map) for le in les])[:memory_limit]
            except Exception:
                pass

        # Phase 4: deterministic journal analysis (bounded, provenance-preserving).
        # The LLM never queries the journal itself; the collector runs the
        # read-only tools and stores computed FACTS for analysis/prompting.
        journal_analysis: list[dict[str, Any]] = []
        if self._journal is not None:
            try:
                journal_analysis = await _collect_journal_analyses(
                    self._journal, query, limit=memory_limit
                )
            except Exception:
                journal_analysis = []

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
            journal_analysis=tuple(journal_analysis),
            query=query,
            language=language,
        )


_JOURNAL_STRATEGY_WORDS: frozenset[str] = frozenset({"strategy", "strategies", "triangle", "transfer", "performs best"})
_JOURNAL_EXCHANGE_WORDS: frozenset[str] = frozenset({"exchange", "exchanges", "binance", "okx", "bybit", "venue"})
_JOURNAL_TRADE_WORDS: frozenset[str] = frozenset(
    {"trade", "order", "fee", "fees", "slippage", "fail", "failed", "failure", "execution", "profit", "loss", "pnl"}
)
_JOURNAL_PERIOD_WORDS: frozenset[str] = frozenset(
    {"today", "yesterday", "week", "period", "compare", "comparison", "versus", "vs", "quality"}
)
_JOURNAL_MISSED_WORDS: frozenset[str] = frozenset({"missed", "miss", "opportunit"})


async def _labels_for(label_repo: Any, target_type: str, ids: list[str]) -> dict[str, dict[str, Any]]:
    """Batch label lookup (best-effort, empty map when unwired)."""
    if label_repo is None or not ids:
        return {}
    try:
        return await label_repo.labels_for(target_type, ids)
    except Exception:
        return {}


def _enrich_experience(exp: Any, label_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Experience dict + Phase 5 provenance: type, sample size, freshness."""
    from app.agent.memory import memory_freshness

    label = label_map.get(getattr(exp, "id", ""), {})
    freshness = memory_freshness(getattr(exp, "created_at", None), float(getattr(exp, "confidence", 0.5)))
    return {
        "id": exp.id,
        "situation": exp.situation[:500],
        "observation": exp.observation[:500],
        "lesson": (exp.lesson[:500] if exp.lesson else None),
        "confidence": exp.confidence,
        "learning_type": label.get("learning_type", "observation"),
        "sample_size": int(label.get("sample_size", 1)),
        "freshness": freshness,
        "decayed_confidence": freshness["decayed_confidence"],
        "source_id": exp.source_id,
        "source_type": exp.source_type,
        "evidence": list(getattr(exp, "evidence", ()) or ()),
        "created_at": exp.created_at.isoformat() if getattr(exp, "created_at", None) else None,
    }


def _enrich_lesson(lesson: Any, label_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Lesson dict + Phase 5 provenance: type, sample size, freshness."""
    from app.agent.memory import memory_freshness

    label = label_map.get(getattr(lesson, "id", ""), {})
    related = list(getattr(lesson, "related_experience_ids", ()) or ())
    freshness = memory_freshness(getattr(lesson, "created_at", None), float(getattr(lesson, "confidence", 0.5)))
    return {
        "id": lesson.id,
        "title": lesson.title[:200],
        "content": lesson.content[:500],
        "confidence": lesson.confidence,
        "learning_type": label.get("learning_type", "observation"),
        "sample_size": int(label.get("sample_size", len(related))),
        "freshness": freshness,
        "decayed_confidence": freshness["decayed_confidence"],
        "source_id": lesson.source_id,
        "source_type": lesson.source_type,
        "evidence": list(getattr(lesson, "evidence", ()) or ()),
        "related_experience_ids": related,
        "version": getattr(lesson, "version", 1),
        "created_at": lesson.created_at.isoformat() if getattr(lesson, "created_at", None) else None,
    }


def _rerank_by_decay(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-rank memories by decayed confidence (stable: score desc, id asc).

    Decay affects relevance only — no record is dropped for being old.
    """
    return sorted(items, key=lambda d: (-float(d.get("decayed_confidence", 0.0)), str(d.get("id", ""))))


async def _collect_journal_analyses(reader: Any, query: str | None, limit: int) -> list[dict[str, Any]]:
    """Run bounded deterministic journal analyses triggered by ``query``."""
    limit = max(1, min(int(limit), 3))
    results: list[dict[str, Any]] = []
    lowered = (query or "").lower()

    def _has(words: frozenset[str]) -> bool:
        return any(w in lowered for w in words)

    # 1. Explicit trade IDs always resolve first (provenance preserved).
    if query:
        from app.agent.journal import TRADE_ID_RE

        for trade_id in dict.fromkeys(TRADE_ID_RE.findall(query)):
            try:
                results.append(await reader.analyze_trade(trade_id))
            except Exception:
                continue
            if len(results) >= limit:
                return results

    if not query:
        return results

    # 2. Period comparison ("today", "compare ...", "execution quality ...").
    if _has(_JOURNAL_PERIOD_WORDS):
        try:
            results.append(await reader.compare_today_vs_yesterday())
        except Exception:
            pass
        if len(results) >= limit:
            return results[:limit]

    # 3. Strategy / exchange performance ("which performs best ...").
    if _has(_JOURNAL_STRATEGY_WORDS):
        try:
            results.append(await reader.strategy_performance())
        except Exception:
            pass
        if len(results) >= limit:
            return results[:limit]
    if _has(_JOURNAL_EXCHANGE_WORDS):
        try:
            results.append(await reader.exchange_performance())
        except Exception:
            pass
        if len(results) >= limit:
            return results[:limit]

    # 4. Missed opportunities (explicit insufficient_data when absent).
    if _has(_JOURNAL_MISSED_WORDS):
        try:
            results.append(await reader.missed_opportunities())
        except Exception:
            pass
        if len(results) >= limit:
            return results[:limit]

    # 5. Trade/fee/slippage/failure questions without an ID: analyze the
    # most recent trade so the answer cites a concrete journal record.
    if _has(_JOURNAL_TRADE_WORDS) and not results:
        try:
            recent = await reader.list_trades(limit=1)
            if recent:
                results.append(await reader.analyze_trade(str(recent[0]["id"])))
            else:
                results.append(
                    {
                        "kind": "trade_analysis",
                        "status": "insufficient_data",
                        "reason": "no journaled trades",
                    }
                )
        except Exception:
            pass
    return results[:limit]
