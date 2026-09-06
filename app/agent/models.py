"""AI Advisor domain models.

All models are immutable :class:`DomainModel` subclasses (frozen, extra="forbid")
consistent with the rest of the project. Every record preserves its source
metadata (source_type / source_id / created_at / version / status) so that
audits can trace provenance.

No model stores secrets (API keys, .env values, DB credentials). Content is
plain analysis / documentation / recommendation text.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator

from app.models.base import DomainModel, utc_now

__all__ = [
    "AgentRecommendation",
    "EvidenceRef",
    "Experience",
    "KnowledgeCategory",
    "KnowledgeDocument",
    "Lesson",
    "RecommendationStatus",
    "ReflectionObservation",
    "ReflectionResult",
    "SourceType",
]


# ------------------------------------------------------------------ enums


class KnowledgeCategory(StrEnum):
    BOT = "bot"
    EXCHANGE = "exchange"
    TRADING = "trading"
    RESEARCH = "research"


class SourceType(StrEnum):
    DOCUMENT = "document"
    REPOSITORY = "repository"
    TRADE = "trade"
    TRANSFER = "transfer"
    JOURNAL = "journal"
    MEMORY = "memory"
    MANUAL = "manual"
    SYSTEM = "system"
    LLM = "llm"


class RecommendationStatus(StrEnum):
    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DISMISSED = "dismissed"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


# ------------------------------------------------------------------ helpers


class EvidenceRef(DomainModel):
    """Pointer to the data that supports an observation / recommendation."""

    source_type: str = Field(description="What kind of source (trade, journal, document, ...)")
    source_id: str = Field(description="Identifier of the source record / file")
    detail: str | None = None
    at: datetime = Field(default_factory=utc_now)


# ------------------------------------------------------------------ knowledge


class KnowledgeDocument(DomainModel):
    """One ingested piece of project knowledge.

    The :attr:`content` holds the document text (or an excerpt). The four
    top-level categories mirror the architecture diagram:

    * BOT — bot behaviour, CLI, Telegram, configuration
    * EXCHANGE — Binance / OKX / Bybit venue specifics
    * TRADING — strategy, risk, execution, recovery
    * RESEARCH — market-data, analysis notes
    """

    id: str = Field(default_factory=lambda: f"kd-{uuid.uuid4().hex[:12]}")
    title: str
    category: KnowledgeCategory
    content: str
    summary: str | None = None
    tags: tuple[str, ...] = ()
    # provenance (mandatory — every record preserves its source)
    source_type: str = Field(default=SourceType.DOCUMENT.value)
    source_id: str = Field(description="File path / URL / repo identifier")
    version: int = Field(default=1, ge=1)
    status: str = Field(default="active")
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("title", "content", "source_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("field must be non-empty")
        return str(value).strip()

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value


# ------------------------------------------------------------------ experience / lesson


class Experience(DomainModel):
    """Structured memory of one observed situation.

    The six load-bearing fields are deliberately explicit so that reflection
    can be audited later: situation / observation / decision / result /
    lesson / confidence / source.

    ``decision`` and ``result`` are kept separate (``decision`` = what the
    bot chose, ``result`` = what actually happened) to distinguish intent
    from outcome.
    """

    id: str = Field(default_factory=lambda: f"exp-{uuid.uuid4().hex[:12]}")
    situation: str
    observation: str
    decision: str | None = None
    result: str | None = None
    lesson: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    # provenance
    source_type: str = Field(default=SourceType.MEMORY.value)
    source_id: str = Field(description="Trade id / journal id / transfer id / manual")
    version: int = Field(default=1, ge=1)
    status: str = Field(default="active")
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("situation", "observation", "source_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("field must be non-empty")
        return str(value).strip()


class Lesson(DomainModel):
    """Distilled pattern extracted from one or more experiences."""

    id: str = Field(default_factory=lambda: f"les-{uuid.uuid4().hex[:12]}")
    title: str
    content: str
    pattern: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()
    related_experience_ids: tuple[str, ...] = ()
    # provenance
    source_type: str = Field(default=SourceType.MEMORY.value)
    source_id: str = Field(description="Experience id(s) or manual source")
    version: int = Field(default=1, ge=1)
    status: str = Field(default="active")
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("title", "content", "source_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("field must be non-empty")
        return str(value).strip()


# ------------------------------------------------------------------ recommendation


class AgentRecommendation(DomainModel):
    """What the advisor suggests a human should consider changing.

    *Does not* apply the change. The only allowed side effect is persistence;
    application requires ``human APPROVE -> validation gate -> apply -> audit``
    which is *not* implemented in Phase 1.

    Every field listed in the spec is present:

    ``parameter / current_value / proposed_value / reason / evidence /
    confidence / expected_impact / risk / status`` plus provenance and the
    operator-decision audit trail.
    """

    id: str = Field(default_factory=lambda: f"rec-{uuid.uuid4().hex[:12]}")
    parameter: str = Field(description="Dot-path of the config/risk parameter, e.g. risk.max_trade_size")
    current_value: str | None = Field(default=None, description="Stringified current value")
    old_value: str | None = Field(default=None, description="Alias of current_value for memory queries")
    proposed_value: str = Field(description="Stringified proposed value")
    reason: str
    evidence: tuple[str, ...] = Field(default=(), description="Supporting trade ids / knowledge ids / observations")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    expected_impact: str = Field(default="", description="What the change is expected to achieve")
    risk: str = Field(default="", description="Downside if the recommendation is wrong")
    status: RecommendationStatus = RecommendationStatus.PENDING
    # operator audit trail (filled after human review — not by the AI)
    operator_decision: str | None = None
    decision_reason: str | None = None
    result: str | None = None
    # provenance
    source_type: str = Field(default=SourceType.MEMORY.value)
    source_id: str = Field(description="Reflection id / experience id / manual")
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("parameter", "proposed_value", "reason", "source_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("field must be non-empty")
        return str(value).strip()

    @field_validator("parameter", mode="before")
    @classmethod
    def _strip_param(cls, value: str) -> str:
        return str(value).strip()

    def with_status(self, status: RecommendationStatus, **updates: Any) -> AgentRecommendation:
        data: dict[str, Any] = dict(updates)
        data["status"] = status
        data["updated_at"] = utc_now()
        return self.model_copy(update=data)


# ------------------------------------------------------------------ reflection


class ReflectionObservation(DomainModel):
    """One structured insight produced by the Reflection Engine (V2).

    Phase 2 adds ``probable_cause``, ``recurring_pattern``, ``evidence_count``
    so that analysis can distinguish facts vs hypotheses.
    """

    what_happened: str
    what_expected: str
    what_differed: str
    possible_pattern: str | None = None
    # Phase 2 additions (with defaults for backward compat)
    probable_cause: str | None = None
    recurring_pattern: str | None = None
    evidence_count: int = Field(default=0, ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    has_enough_evidence: bool = True
    evidence: tuple[str, ...] = ()
    source_type: str = Field(default=SourceType.MEMORY.value)
    source_id: str = Field(default="reflection")
    created_at: datetime = Field(default_factory=utc_now)


class ReflectionResult(DomainModel):
    """Outcome of a reflection run.

    When :attr:`action` is ``"NO_ACTION"`` the engine concluded there is
    insufficient evidence for a useful insight. Callers must not create
    recommendations from NO_ACTION results.

    Phase 2 exposes ``evidence_count`` / ``confidence`` at the top level for
    deterministic gating that the LLM cannot bypass.
    """

    action: str = Field(description="NO_ACTION or INSIGHT")
    observation: ReflectionObservation | None = None
    lesson: Lesson | None = None
    recommendation: AgentRecommendation | None = None
    reason: str = ""
    # Phase 2 explicit fields (mirrors observation when present)
    evidence_count: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    has_enough_evidence: bool = False

    @property
    def is_no_action(self) -> bool:
        return self.action == "NO_ACTION"
