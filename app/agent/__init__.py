"""AI Advisor subsystem — analytical, read-only, never executive.

Architecture (Phase 1)
----------------------

    Existing Bot
        | scans / trades / journal / state
        v
    AI Advisor (this package)
        |
        +-- Knowledge Base  (repository documentation)
        +-- Memory DB       (experiences / lessons / recommendations)
        +-- Live/Journal Context (read-only tools)
        v
    Reflection / Pattern Analysis
        |
        v
    LLM Provider abstraction
        |
        v
    Recommendation
        |
        v
    Human approval (future)

Safety
------

* The advisor may READ, ANALYZE, REMEMBER, REFLECT, RECOMMEND.
* It may NOT place orders, cancel orders, withdraw, access credentials,
  bypass risk, mutate parameters or risk limits, or execute arbitrary code.
* The only allowed side effect of a recommendation is *persistence*; any
  future config change requires ``human APPROVE -> validation gate -> apply
  -> audit`` and that path is explicitly blocked in Phase 1.

Integration
-----------

Use :func:`build_agent` to wire the advisor from an existing
:class:`app.services.AppServices` instance. No second database is created;
everything lives in the same SQLite/PostgreSQL file and the same
``Base.metadata.create_all`` machinery.
"""

from __future__ import annotations

from app.agent.context import AgentContext, ContextCollector
from app.agent.core import AgentCore, AgentRequest, AgentResponse
from app.agent.knowledge import KnowledgeRepository, KnowledgeService
from app.agent.memory import ExperienceRepository, LessonRepository
from app.agent.models import (
    AgentRecommendation,
    Experience,
    KnowledgeCategory,
    KnowledgeDocument,
    Lesson,
    RecommendationStatus,
    ReflectionObservation,
    ReflectionResult,
    SourceType,
)
from app.agent.providers.base import LLMProvider, NullProvider, EchoProvider
from app.agent.providers import create_provider as _create_provider  # lazy, no network

from app.agent.recommendations import RecommendationRepository, RecommendationService
from app.agent.reflection import ReflectionEngine
from app.agent.tools import AgentTools

__all__ = [
    "AgentContext",
    "AgentCore",
    "AgentRecommendation",
    "AgentRequest",
    "AgentResponse",
    "AgentTools",
    "ContextCollector",
    "EchoProvider",
    "Experience",
    "ExperienceRepository",
    "KnowledgeCategory",
    "KnowledgeDocument",
    "KnowledgeRepository",
    "KnowledgeService",
    "Lesson",
    "LessonRepository",
    "LLMProvider",
    "NullProvider",
    "RecommendationRepository",
    "RecommendationService",
    "RecommendationStatus",
    "ReflectionEngine",
    "ReflectionObservation",
    "ReflectionResult",
    "SourceType",
    "build_agent",
]


def build_agent(services, *, llm: LLMProvider | None = None):  # type: ignore[no-untyped-def]
    """Compose the advisor from an existing :class:`AppServices`.

    Returns ``(core, knowledge_service, experience_repo, lesson_repo,
    recommendation_repo, tools)`` so callers can drive ingestion or ad-hoc
    queries without going through :class:`AgentCore`.

    The advisor shares :attr:`services.db`; no new database is created.
    """
    from app.storage.engine import Database  # local import: composition root

    db: Database = services.db

    # Repositories (all backed by the shared database)
    knowledge_repo = KnowledgeRepository(db)
    knowledge_service = KnowledgeService(knowledge_repo)
    exp_repo = ExperienceRepository(db)
    lesson_repo = LessonRepository(db)
    rec_repo = RecommendationRepository(db)
    rec_service = RecommendationService(rec_repo)

    # Expose advisor repos on services for tool access (non-intrusive, optional)
    # Tools will probe for these attributes; setting them makes get_memory /
    # get_previous_recommendations work without modifying AppServices class.
    services.agent_knowledge = knowledge_repo  # type: ignore[attr-defined]
    services.agent_experiences = exp_repo  # type: ignore[attr-defined]
    services.agent_lessons = lesson_repo  # type: ignore[attr-defined]
    services.agent_recommendations = rec_repo  # type: ignore[attr-defined]
    services.agent_recommendation_service = rec_service  # type: ignore[attr-defined]
    services.agent_knowledge_service = knowledge_service  # type: ignore[attr-defined]

    tools = AgentTools(services)
    collector = ContextCollector(
        tools,
        knowledge_service=knowledge_service,
        experience_repo=exp_repo,
        lesson_repo=lesson_repo,
        recommendation_repo=rec_repo,
    )
    reflection = ReflectionEngine()
    # Provider selection: explicit ``llm`` wins; otherwise derive from settings
    # without performing any network I/O.
    if llm is None:
        try:
            llm = _create_provider(getattr(services, "settings", None))
        except Exception:
            llm = NullProvider()
    core = AgentCore(collector=collector, reflection=reflection, llm=llm)

    return core, knowledge_service, exp_repo, lesson_repo, rec_service, tools
