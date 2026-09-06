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
    "AgentFeedback",
    "AgentRecommendation",
    "BOT_KNOWLEDGE",
    "EXCHANGE_KNOWLEDGE",
    "EvidenceRef",
    "Experience",
    "KnowledgeCategory",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "LearningType",
    "Lesson",
    "RESEARCH_KNOWLEDGE",
    "RecommendationStatus",
    "ReflectionObservation",
    "ReflectionResult",
    "SourceType",
    "TRADING_KNOWLEDGE",
    "normalize_knowledge_category",
]


# ------------------------------------------------------------------ enums


class KnowledgeCategory(StrEnum):
    BOT = "bot"
    EXCHANGE = "exchange"
    TRADING = "trading"
    RESEARCH = "research"


# Phase 3 aliases: project plan names for the same four categories.
# Repository knowledge uses these; external knowledge uses the same
# categories but a non-repository ``source_type`` so the two stay distinct.
BOT_KNOWLEDGE: KnowledgeCategory = KnowledgeCategory.BOT
EXCHANGE_KNOWLEDGE: KnowledgeCategory = KnowledgeCategory.EXCHANGE
TRADING_KNOWLEDGE: KnowledgeCategory = KnowledgeCategory.TRADING
RESEARCH_KNOWLEDGE: KnowledgeCategory = KnowledgeCategory.RESEARCH

_CATEGORY_ALIASES: dict[str, KnowledgeCategory] = {
    "bot": KnowledgeCategory.BOT,
    "bot_knowledge": KnowledgeCategory.BOT,
    "exchange": KnowledgeCategory.EXCHANGE,
    "exchange_knowledge": KnowledgeCategory.EXCHANGE,
    "trading": KnowledgeCategory.TRADING,
    "trading_knowledge": KnowledgeCategory.TRADING,
    "research": KnowledgeCategory.RESEARCH,
    "research_knowledge": KnowledgeCategory.RESEARCH,
}


def normalize_knowledge_category(value: Any) -> KnowledgeCategory:
    """Normalize ``BOT`` / ``BOT_KNOWLEDGE`` (any case) to :class:`KnowledgeCategory`.

    Raises ``ValueError`` for unknown categories.
    """
    if isinstance(value, KnowledgeCategory):
        return value
    key = str(value).strip().lower()
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    raise ValueError(f"unknown knowledge category: {value!r}")


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


class LearningType(StrEnum):
    """Phase 5 learning-item types — every persisted learning item has one.

    * FACT — deterministic data computed by application code from the journal
      (trade outcomes, aggregates, period deltas). Never authored by the LLM.
    * OBSERVATION — a recorded market/execution observation (experiences,
      reflection observations). Deterministic or human-recorded.
    * HYPOTHESIS — LLM-proposed interpretation. Stored only as audit events,
      never auto-promoted to FACT (no promotion path exists by design).
    * RECOMMENDATION — a gated suggestion awaiting human approval.
    """

    FACT = "fact"
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    RECOMMENDATION = "recommendation"


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

    Phase 3 record metadata (all persisted; new fields live in the row's
    ``extra_metadata`` JSON so no schema migration is required):

    * ``id`` — knowledge ID
    * ``source_type`` / ``source_id`` — source / provenance (repository vs external)
    * ``document_path`` — explicit document/path (defaults to ``source_id``)
    * ``section`` — section within the document (markdown header or "")
    * ``verification_status`` — ``unverified`` / ``verified``
    * ``confidence`` — 0.0..1.0 trust in the record
    """

    id: str = Field(default_factory=lambda: f"kd-{uuid.uuid4().hex[:12]}")
    title: str
    category: KnowledgeCategory
    content: str
    summary: str | None = None
    tags: tuple[str, ...] = ()
    # Phase 3 metadata (defaults keep Phase 1 constructors working)
    section: str = Field(default="")
    document_path: str | None = Field(default=None, description="Explicit document/path; defaults to source_id")
    verification_status: str = Field(default="unverified")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
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
        if isinstance(value, KnowledgeCategory):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            if key in _CATEGORY_ALIASES:
                return _CATEGORY_ALIASES[key]
            return key
        return value

    @field_validator("section", mode="before")
    @classmethod
    def _coerce_section(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("document_path", mode="before")
    @classmethod
    def _coerce_doc_path(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("verification_status", mode="before")
    @classmethod
    def _coerce_verification(cls, value: Any) -> str:
        text = str(value).strip().lower() if value is not None else "unverified"
        if text not in ("unverified", "verified"):
            raise ValueError("verification_status must be 'unverified' or 'verified'")
        return text

    @property
    def source(self) -> str:
        """Unified ``source_type:source_id`` provenance string."""
        return f"{self.source_type}:{self.source_id}"

    @property
    def effective_document_path(self) -> str:
        """Explicit document path, falling back to ``source_id``."""
        return self.document_path or self.source_id


class KnowledgeChunk(DomainModel):
    """One deterministic chunk of a knowledge document.

    Chunks are the retrieval unit: the AI receives relevant chunks instead
    of entire documents. Every chunk preserves document provenance plus its
    own ``chunk_index`` / ``section`` so answers can cite the exact source.
    """

    id: str = Field(default_factory=lambda: f"kc-{uuid.uuid4().hex[:12]}")
    doc_id: str
    chunk_index: int = Field(ge=0)
    section: str = Field(default="")
    content: str
    # denormalized provenance (mirrors the parent document at index time)
    title: str = Field(default="")
    category: KnowledgeCategory = KnowledgeCategory.RESEARCH
    source_type: str = Field(default=SourceType.DOCUMENT.value)
    source_id: str = Field(description="Parent document source_id")
    document_path: str | None = None
    verification_status: str = Field(default="unverified")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("doc_id", "content", "source_id")
    @classmethod
    def _non_empty_chunk(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("field must be non-empty")
        return str(value).strip()

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_chunk_category(cls, value: Any) -> Any:
        if isinstance(value, KnowledgeCategory):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            if key in _CATEGORY_ALIASES:
                return _CATEGORY_ALIASES[key]
            return key
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


# ------------------------------------------------------------------ feedback (Phase 5)


class AgentFeedback(DomainModel):
    """Human operator feedback on an agent artifact (lesson/recommendation).

    ``ai_feedback`` store. Only a non-empty human ``approver`` may submit;
    the LLM can never author feedback (no code path passes LLM text here).
    ``rating`` is +1 (useful) / -1 (wrong); ``comment`` is free text.
    """

    id: str = Field(default_factory=lambda: f"fdb-{uuid.uuid4().hex[:12]}")
    target_type: str = Field(description="lesson | recommendation")
    target_id: str = Field(description="Id of the lesson/recommendation reviewed")
    rating: int = Field(description="+1 useful, -1 wrong")
    comment: str = Field(default="")
    approver: str = Field(description="Non-empty human identifier")
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("target_type", "target_id", "approver")
    @classmethod
    def _non_empty_feedback(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("field must be non-empty")
        return str(value).strip()

    @field_validator("target_type", mode="before")
    @classmethod
    def _coerce_target(cls, value: Any) -> str:
        text = str(value).strip().lower()
        if text not in ("lesson", "recommendation"):
            raise ValueError("target_type must be 'lesson' or 'recommendation'")
        return text

    @field_validator("rating", mode="before")
    @classmethod
    def _coerce_rating(cls, value: Any) -> int:
        rating = int(value)
        if rating not in (1, -1):
            raise ValueError("rating must be +1 or -1")
        return rating


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
