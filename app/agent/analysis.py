"""Structured analysis pipeline — FACTS / OBSERVATIONS / HYPOTHESES / RECOMMENDATIONS.

The advisor must never let speculation become fact. ``StructuredAnalysis``
enforces that separation:

* **FACTS** — verbatim, truncated snapshots from read-only tools, trade
  records, market-data freshness, risk limits. No LLM invents them.
* **OBSERVATIONS** — deterministic reflection (what_happened / what_expected /
  what_differed plus evidence_count / confidence). The LLM may *propose* an
  observation, but the deterministic ``ReflectionEngine`` remains the source
  of truth; the LLM text is stored as a *hypothesis* string, never promoted.
* **HYPOTHESES** — LLM-proposed explanations, explicitly labelled as
  speculation (filtered for secrets, capped length).
* **RECOMMENDATIONS** — only emitted after the deterministic safety gate:
  ``evidence_count < 5 -> NO_ACTION`` and ``confidence < 0.45 -> NO_ACTION``.
  The gate cannot be bypassed by colluding LLM content.

Context sizes are bounded (see ``MAX_*`` constants) so that no unrestricted
database dump reaches the prompt — future token-budget changes can tune the
caps without touching callers.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.agent.context import AgentContext
from app.agent.models import AgentRecommendation, DomainModel, ReflectionResult, utc_now
from app.agent.providers.base import filter_secrets_from_text, sanitize_untrusted_text
from app.config.logging_config import get_logger

__all__ = ["StructuredAnalysis", "AnalysisEngine", "BoundedContext"]

logger = get_logger("agent.analysis")

# ------------------------------------------------------------------ bounds
# Keep prompts predictable and avoid dumping thousands of journal rows.
MAX_FACTS = 8
MAX_OBSERVATIONS = 5
MAX_HYPOTHESES = 3
MAX_RECOMMENDATIONS = 2
MAX_CONTEXT_CHARS = 6000
MAX_SINGLE_FACT_CHARS = 500
MAX_SINGLE_HYPOTHESIS_CHARS = 800
MAX_KNOWLEDGE_HITS = 3
MAX_TRADES_IN_FACTS = 5
MAX_JOURNAL_IN_FACTS = 5


class BoundedContext(DomainModel):
    """Truncated, safe-to-log context summary."""

    summary: str
    facts_count: int
    truncated: bool = False


class StructuredAnalysis(DomainModel):
    """Analysis that keeps facts distinct from speculation.

    * ``facts`` — deterministic, truncated snapshots (trade stats, risk, etc.)
    * ``observations`` — deterministic reflection strings
    * ``hypotheses`` — LLM-proposed (or fallback) speculations, never facts
    * ``recommendations`` — deterministic, gated (may be empty)
    * ``has_enough_evidence`` / ``evidence_count`` / ``confidence`` echoed
      for audit
    * ``action`` is ``NO_ACTION`` when gates fail (even if hypotheses exist)
    """

    facts: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    recommendations: tuple[AgentRecommendation, ...] = ()
    evidence_count: int = Field(ge=0, default=0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    has_enough_evidence: bool = False
    action: str = Field(description="NO_ACTION or INSIGHT")
    llm_hypothesis_raw: str | None = None
    context_summary: str = ""
    bounded: bool = True
    created_at: Any = Field(default_factory=utc_now)

    @property
    def is_no_action(self) -> bool:
        return self.action == "NO_ACTION"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "... (truncated)"


def _bound_list(items: list[str], max_items: int, max_chars_per: int) -> tuple[str, ...]:
    bounded: list[str] = []
    for item in items[:max_items]:
        # Treat every item as untrusted data: secret filter + injection sanitize + bound
        sanitized = sanitize_untrusted_text(filter_secrets_from_text(item), max_chars=max_chars_per)
        bounded.append(_truncate(sanitized, max_chars_per))
    return tuple(bounded)


class AnalysisEngine:
    """Builds :class:`StructuredAnalysis` from context + reflection + LLM.

    The engine is deterministic — the LLM output is stored as a hypothesis and
    never auto-promotes to a fact or recommendation. Gates are applied *after*
    the LLM proposes, and they cannot be skipped.
    """

    def __init__(self, *, max_context_chars: int = MAX_CONTEXT_CHARS) -> None:
        self._max_context = max_context_chars

    def build(
        self,
        *,
        context: AgentContext,
        reflection: ReflectionResult | None,
        llm_output: str | None,
        previous_recommendations: list[dict[str, Any]] | None = None,
        llm_malformed: bool = False,
    ) -> StructuredAnalysis:
        """Assemble structured analysis with deterministic gating.

        ``llm_output`` is treated as *hypothesis* only. If ``llm_malformed``,
        it is stored as a truncated note and fails closed (NO_ACTION when
        gates already fail).
        """
        # 1. FACTS — bounded, deterministic snapshots (no LLM)
        raw_facts: list[str] = []

        # Trades fact (bounded count)
        trades_fact = f"recent trades: {len(context.recent_trades)}"
        if context.trade_statistics:
            trades_fact += f" | stats total={context.trade_statistics.get('total', 0)} completed={context.trade_statistics.get('completed', 0)} failed={context.trade_statistics.get('failed', 0)} total_pnl={context.trade_statistics.get('total_pnl', '0')}"
        raw_facts.append(_truncate(trades_fact, MAX_SINGLE_FACT_CHARS))

        # Sample trade facts (truncated)
        for t in list(context.recent_trades)[:MAX_TRADES_IN_FACTS]:
            raw_facts.append(
                _truncate(
                    f"trade {t.get('id','?')[:12]} {t.get('strategy','?')} {t.get('status','?')} route={t.get('route','')[:30]} net={t.get('net_profit','0')}",
                    MAX_SINGLE_FACT_CHARS,
                )
            )

        # Parameters fact (already non-secret)
        cp = context.current_parameters or {}
        risk = cp.get("risk", {}) if isinstance(cp, dict) else {}
        raw_facts.append(_truncate(f"params risk: {risk}", MAX_SINGLE_FACT_CHARS))
        raw_facts.append(_truncate(f"risk_state: {context.risk_state}", MAX_SINGLE_FACT_CHARS))
        raw_facts.append(_truncate(f"exchanges: {list(context.exchange_status.keys())}", MAX_SINGLE_FACT_CHARS))
        # Journal (bounded)
        for j in list(context.recent_journal)[:MAX_JOURNAL_IN_FACTS]:
            raw_facts.append(_truncate(f"journal {j.get('action','?')}: {j.get('message','')[:60]}", MAX_SINGLE_FACT_CHARS))
        # Previous recommendations fact
        if previous_recommendations is not None:
            raw_facts.append(_truncate(f"previous_recommendations: {len(previous_recommendations)}", MAX_SINGLE_FACT_CHARS))
        elif context.previous_recommendations:
            raw_facts.append(_truncate(f"previous_recommendations: {len(context.previous_recommendations)}", MAX_SINGLE_FACT_CHARS))
        # Knowledge facts — repository vs external stay labeled; assumptions/
        # hypotheses/recommendations never enter facts (separation enforced).
        for hit in list(context.knowledge_hits)[:MAX_KNOWLEDGE_HITS]:
            if not isinstance(hit, dict):
                continue
            source_type = str(hit.get("source_type", "repository") or "repository")
            source_id = str(hit.get("source_id", "?") or "?")[:120]
            section = str(hit.get("section", "") or "")[:80]
            verification = str(hit.get("verification_status", "unverified") or "unverified")[:16]
            title = str(hit.get("title", "") or "")[:80]
            summary = str(hit.get("summary") or hit.get("excerpt") or "")[:200]
            label = "repo-fact" if source_type == "repository" else "external-fact"
            where = f"{source_id}#{section}" if section else source_id
            raw_facts.append(
                _truncate(f"{label} [{where}|{verification}] {title}: {summary}", MAX_SINGLE_FACT_CHARS)
            )

        facts = _bound_list(raw_facts, MAX_FACTS, MAX_SINGLE_FACT_CHARS)

        # 2. OBSERVATIONS — deterministic reflection only
        observations: list[str] = []
        evidence_count = 0
        confidence = 0.0
        has_enough = False
        action = "NO_ACTION"

        if reflection is not None:
            evidence_count = getattr(reflection, "evidence_count", 0) or (reflection.observation.evidence_count if reflection.observation else 0)
            confidence = getattr(reflection, "confidence", 0.0) or (reflection.observation.confidence if reflection.observation else 0.0)
            has_enough = bool(getattr(reflection, "has_enough_evidence", False))
            action = reflection.action
            if reflection.observation is not None:
                obs = reflection.observation
                observations.append(
                    _truncate(
                        f"what_happened={obs.what_happened} | expected={obs.what_expected} | differed={obs.what_differed} | cause={obs.probable_cause or '-'} | recurring={obs.recurring_pattern or obs.possible_pattern or '-'} | confidence={obs.confidence:.2f} | evidence={obs.evidence_count}",
                        MAX_SINGLE_FACT_CHARS,
                    )
                )
            elif reflection.reason:
                observations.append(_truncate(f"reflection reason: {reflection.reason}", MAX_SINGLE_FACT_CHARS))
        else:
            observations.append("no reflection (no trades/experiences)")

        observations_b = _bound_list(observations, MAX_OBSERVATIONS, MAX_SINGLE_FACT_CHARS)

        # 3. HYPOTHESES — LLM output is speculation, never a fact
        hypotheses: list[str] = []
        llm_hypothesis_raw = None
        if llm_output is not None:
            llm_hypothesis_raw = _truncate(filter_secrets_from_text(llm_output), MAX_SINGLE_HYPOTHESIS_CHARS)
            if llm_malformed:
                hypotheses.append(_truncate(f"LLM malformed output (treated as hypothesis, not fact): {llm_hypothesis_raw[:200]}", MAX_SINGLE_HYPOTHESIS_CHARS))
            else:
                hypotheses.append(_truncate(f"LLM hypothesis: {llm_hypothesis_raw[:400]}", MAX_SINGLE_HYPOTHESIS_CHARS))
        elif reflection is not None and reflection.observation is not None:
            # Fallback hypothesis derived from deterministic pattern (still labelled hypothesis)
            pat = reflection.observation.possible_pattern or "no deterministic pattern"
            hypotheses.append(_truncate(f"deterministic hypothesis: {pat[:300]}", MAX_SINGLE_HYPOTHESIS_CHARS))

        hypotheses_b = _bound_list(hypotheses, MAX_HYPOTHESES, MAX_SINGLE_HYPOTHESIS_CHARS)

        # 4. RECOMMENDATIONS — gated deterministically (cannot be bypassed)
        # Gates per spec: evidence_count <5 -> NO_ACTION, confidence <0.45 -> NO_ACTION
        # These gates are already applied in reflection, but we re-enforce here so that
        # even a hallucinated LLM recommendation is discarded.
        gated_action = action
        if evidence_count < 5:
            gated_action = "NO_ACTION"
            has_enough = False
        if confidence < 0.45:
            gated_action = "NO_ACTION"
            has_enough = False

        # If LLM tried to hallucinate a recommendation when gated, it is ignored.
        # Actual recommendations are only the deterministic one (if any) passed via
        # ``reflection.recommendation`` and gated above — AgentCore decides eligibility.

        # For this engine, we surface zero recommendations when NO_ACTION; when INSIGHT,
        # the caller (AgentCore) may attach the gated recommendation separately.
        # We expose empty here to make the separation explicit — facts/observations/hypotheses/recommendations.
        recommendations: tuple[AgentRecommendation, ...] = ()
        # Note: if the caller has a gated recommendation, they can set it; this engine does not auto-create.

        # 5. Bounded context summary (for logging / prompt)
        summary_parts = [
            f"facts: {len(facts)}",
            f"observations: {len(observations_b)}",
            f"hypotheses: {len(hypotheses_b)}",
            f"evidence_count: {evidence_count}",
            f"confidence: {confidence:.2f}",
            f"action: {gated_action}",
        ]
        ctx_summary = _truncate(" | ".join(summary_parts) + " | " + context.summary()[:1000], self._max_context)

        return StructuredAnalysis(
            facts=facts,
            observations=observations_b,
            hypotheses=hypotheses_b,
            recommendations=recommendations,
            evidence_count=evidence_count,
            confidence=confidence,
            has_enough_evidence=has_enough,
            action=gated_action,
            llm_hypothesis_raw=llm_hypothesis_raw,
            context_summary=ctx_summary,
            bounded=True,
        )

    def attach_recommendation(
        self,
        analysis: StructuredAnalysis,
        recommendation: AgentRecommendation | None,
    ) -> StructuredAnalysis:
        """Return a copy of ``analysis`` with ``recommendation`` attached if gated passes.

        If analysis is NO_ACTION, attachment is refused (returns unchanged).
        """
        if analysis.is_no_action or recommendation is None:
            return analysis
        # Re-enforce gates before attachment
        if analysis.evidence_count < 5 or analysis.confidence < 0.45:
            return analysis
        return analysis.model_copy(update={"recommendations": (recommendation,)})
