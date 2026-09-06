"""Phase 5 — deterministic experience extraction pipeline.

``Trade/Event → Journal → Experience Extraction → Validation/Aggregation →
Lesson → Memory``

Every function here is deterministic application code: financial figures,
sample sizes and confidences are computed from persisted journal data, never
by the LLM. Missing values are represented explicitly (``"n/a"`` / ``None``),
never invented.

Learning-type discipline (:class:`LearningType`):

* journal-derived experiences → ``FACT`` when all key metrics are present,
  else ``OBSERVATION``;
* synthesized lessons → ``OBSERVATION`` (patterns need human/operator trust);
* LLM text → ``HYPOTHESIS`` via :func:`record_llm_hypothesis` only, stored as
  an ``agent_audit`` event — there is deliberately NO path that persists LLM
  output as an experience, lesson or fact.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.agent.journal import MIN_SAMPLE_FOR_CONCLUSIONS
from app.agent.models import Experience, LearningType, Lesson, SourceType
from app.config.logging_config import get_logger

__all__ = [
    "aggregate_to_lesson",
    "compute_lesson_confidence",
    "detect_contradiction",
    "extract_experience",
    "lesson_direction",
    "outcome_direction",
    "record_llm_hypothesis",
    "validate_experience",
]

logger = get_logger("agent.extraction")

#: Experience fields that must be real journal data to qualify as FACT.
_FACT_REQUIRED_FIELDS: tuple[str, ...] = ("net_profit", "net_profit_bps", "fees_quote", "status")


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def outcome_direction(avg_net_bps: Any, fail_rate: float) -> str:
    """Deterministic outcome regime: ``positive`` | ``negative`` | ``mixed``."""
    try:
        avg = Decimal(str(avg_net_bps))
    except Exception:
        avg = Decimal("0")
    if avg < 0 or fail_rate >= 0.5:
        return "negative"
    if avg > 0 and fail_rate < 0.3:
        return "positive"
    return "mixed"


def lesson_direction(lesson: Lesson, *, avg_net_bps: Any = 0, fail_rate: float = 0.0) -> str:
    """Direction tag for a lesson from its evidence statistics.

    Computed deterministically from caller-supplied evidence stats (recomputed
    from the journal at synthesis/contradiction time, never stored blindly).
    """
    _ = lesson  # signature keeps the evidence contract explicit
    return outcome_direction(avg_net_bps, fail_rate)


def extract_experience(
    journal_view: dict[str, Any],
    *,
    transfer_plan: dict[str, Any] | None = None,
    market_conditions: str | None = None,
) -> Experience:
    """Build an :class:`Experience` from a Phase 4 sanitized journal view.

    Preserves experience ID linkage (``source_id`` + ``evidence`` = trade
    id), timestamps (model defaults to now; journal time embedded in text),
    strategy/exchange/route, expected+realized edge, fees, slippage,
    execution time, order results, failure reason, market conditions,
    outcome and confidence. Unavailable data is explicit (``"n/a"``).
    """
    trade_id = _text(journal_view.get("id")) or "unknown"
    strategy = _text(journal_view.get("strategy")) or "n/a"
    exchange = _text(journal_view.get("exchange_id")) or "n/a"
    route = _text(journal_view.get("route")) or "n/a"
    status = _text(journal_view.get("status")) or "n/a"
    realized = _text(journal_view.get("net_profit")) or "n/a"
    realized_bps = _text(journal_view.get("net_profit_bps")) or "n/a"
    fees = _text(journal_view.get("fees_quote")) or "n/a"
    slippage = _text(journal_view.get("slippage_bps")) or "n/a"
    duration = journal_view.get("execution_duration_s")
    duration_txt = f"{duration}s" if duration is not None else "n/a"
    created = _text(journal_view.get("created_at")) or "n/a"

    if transfer_plan is not None:
        expected_txt = _text(transfer_plan.get("net_profit_quote")) or "n/a"
    else:
        expected_txt = "n/a (expected edge is not journaled for this trade)"

    legs = journal_view.get("orders") or []
    leg_bits: list[str] = []
    for leg in legs if isinstance(legs, list) else []:
        if not isinstance(leg, dict):
            continue
        leg_bits.append(
            f"{leg.get('symbol', '?')} {leg.get('side', '?')} "
            f"{leg.get('status', '?')} fill={leg.get('fill_ratio', '?')}"
        )
    legs_txt = "; ".join(leg_bits) if leg_bits else "n/a (no legs journaled)"
    failure = _text(journal_view.get("error")) or "none"
    market_txt = market_conditions or "n/a (market snapshot not journaled per trade)"

    situation = (
        f"[{created}] {strategy} trade on {exchange} route {route}: "
        f"status={status}, realized={realized} ({realized_bps} bps), "
        f"expected={expected_txt}, fees={fees}, slippage={slippage} bps, "
        f"duration={duration_txt}"
    )
    observation = (
        f"outcome={status}; legs: {legs_txt}; "
        f"failure={failure}; market: {market_txt}"
    )
    decision = f"executed {strategy} {route} on {exchange}"
    result = f"{status}: net {realized} ({realized_bps} bps), fees {fees}"

    # Confidence heuristic (deterministic): clean completions score higher,
    # failures/manual_review lower — capped, never a strong claim alone.
    if status == "completed" and "n/a" not in (realized, fees):
        confidence = 0.7
    elif status in ("failed", "manual_review"):
        confidence = 0.6
    else:
        confidence = 0.5

    return Experience(
        situation=situation[:2000],
        observation=observation[:2000],
        decision=decision[:500],
        result=result[:500],
        confidence=confidence,
        evidence=(trade_id,),
        tags=("phase5", "trade", strategy, exchange, status),
        source_type=SourceType.TRADE.value,
        source_id=trade_id,
    )


def validate_experience(exp: Experience) -> list[str]:
    """Validate an extracted experience; returns error strings (empty = valid)."""
    errors: list[str] = []
    if not exp.situation.strip():
        errors.append("situation is empty")
    if not exp.observation.strip():
        errors.append("observation is empty")
    if not exp.source_id.strip():
        errors.append("source trade id is missing")
    if not exp.evidence:
        errors.append("evidence trade ids are missing")
    if not (0.0 <= float(exp.confidence) <= 1.0):
        errors.append("confidence out of bounds")
    return errors


def compute_lesson_confidence(experiences: list[Experience]) -> float:
    """Average confidence, capped when the sample is insufficient."""
    if not experiences:
        return 0.0
    avg = sum(float(e.confidence) for e in experiences) / len(experiences)
    if len(experiences) < MIN_SAMPLE_FOR_CONCLUSIONS:
        avg = min(avg, 0.44)  # below action gates: no strong conclusions
    return round(min(0.9, max(0.0, avg)), 4)


def aggregate_to_lesson(
    theme: str,
    experiences: list[Experience],
    *,
    pattern: str,
    title: str | None = None,
) -> Lesson | None:
    """Synthesize a lesson from validated experiences (None when insufficient).

    Reuses the Phase 4 minimum-sample rule: fewer than
    :data:`MIN_SAMPLE_FOR_CONCLUSIONS` experiences → explicit ``None``
    (insufficient data), never a weak lesson masquerading as insight.
    """
    if len(experiences) < MIN_SAMPLE_FOR_CONCLUSIONS:
        logger.info(
            "lesson_insufficient_sample",
            extra={"theme": theme, "n": len(experiences), "required": MIN_SAMPLE_FOR_CONCLUSIONS},
        )
        return None
    for exp in experiences:
        if validate_experience(exp):
            logger.warning("lesson_invalid_experience", extra={"experience_id": exp.id})
            return None
    evidence = tuple(dict.fromkeys(eid for e in experiences for eid in e.evidence))
    related = tuple(e.id for e in experiences)
    return Lesson(
        title=(title or f"Pattern: {theme}")[:256],
        content=pattern[:4000],
        pattern=pattern[:2000],
        confidence=compute_lesson_confidence(experiences),
        evidence=evidence,
        related_experience_ids=related,
        source_type=SourceType.SYSTEM.value,
        source_id=f"phase5:{theme}",
    )


def detect_contradiction(
    *,
    existing_direction: str,
    new_avg_net_bps: Any,
    new_fail_rate: float,
    existing_lesson_id: str,
    new_evidence_ids: list[str],
) -> dict[str, Any] | None:
    """Detect conflicting evidence regimes (pure, deterministic).

    Returns a conflict record when the new batch direction opposes the
    existing lesson direction (positive↔negative); ``mixed`` never conflicts.
    The caller must persist the record (audit) and retain BOTH lessons —
    silent overwrites are forbidden.
    """
    new_direction = outcome_direction(new_avg_net_bps, new_fail_rate)
    opposing = {("positive", "negative"), ("negative", "positive")}
    if (existing_direction, new_direction) not in opposing:
        return None
    return {
        "kind": "contradiction",
        "existing_lesson_id": existing_lesson_id,
        "existing_direction": existing_direction,
        "new_direction": new_direction,
        "new_evidence_ids": list(new_evidence_ids),
        "resolution": "retained_both",
    }


async def record_llm_hypothesis(
    audit_repo: Any,
    text: str,
    *,
    source: str = "llm",
    query: str | None = None,
) -> dict[str, Any]:
    """Persist LLM text ONLY as a hypothesis audit event (never a fact).

    This is the single sanctioned sink for LLM-authored interpretations.
    There is intentionally no function that writes LLM output into
    experiences, lessons, knowledge or recommendations.
    """
    from app.agent.audit import AgentAuditEvent

    event = AgentAuditEvent(
        event_type="llm_hypothesis",
        provider="llm",
        model=None,
        query=(query or "")[:500] if query else None,
        evidence_count=0,
        confidence=0.0,
        action="HYPOTHESIS",
        details={"text": text[:2000], "learning_type": LearningType.HYPOTHESIS.value},
        source_type=SourceType.LLM.value,
        source_id=source,
    )
    await audit_repo.log(event)
    return {"event_id": event.id, "learning_type": LearningType.HYPOTHESIS.value}
