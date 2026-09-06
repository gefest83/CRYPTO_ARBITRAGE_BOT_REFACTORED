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


def _format_memory_fact(prefix: str, item: dict[str, Any]) -> str | None:
    """Render one retrieved memory as a labeled line (never a journal-fact)."""
    freshness = item.get("freshness") if isinstance(item.get("freshness"), dict) else {}
    age = freshness.get("age_days", "?")
    decayed = item.get("decayed_confidence", item.get("confidence", "?"))
    title = str(item.get("title") or item.get("situation") or "")[:100]
    evidence = item.get("evidence") or []
    ev_txt = ",".join(str(e)[:12] for e in list(evidence)[:3]) if evidence else str(item.get("source_id", "?"))[:16]
    return (
        f"{prefix} [{item.get('id', '?')}|{item.get('learning_type', 'observation')}|"
        f"conf={item.get('confidence', '?')}|decayed={decayed}|n={item.get('sample_size', '?')}|"
        f"age={age}d|src={ev_txt}] {title}"
    )


def _format_journal_fact(entry: Any) -> str | None:
    """Render one journal-analysis result as a labeled FACT line (or None)."""
    if not isinstance(entry, dict):
        return None
    kind = str(entry.get("kind", "?"))
    if kind == "trade_analysis":
        if entry.get("status") == "insufficient_data":
            return f"journal-fact [trade:{entry.get('trade_id', '?')}] insufficient_data: {entry.get('reason', '')[:120]}"
        realized = entry.get("realized", {}) if isinstance(entry.get("realized"), dict) else {}
        expected = entry.get("expected", {}) if isinstance(entry.get("expected"), dict) else {}
        diff = entry.get("difference", {}) if isinstance(entry.get("difference"), dict) else {}
        fees = entry.get("fees", {}) if isinstance(entry.get("fees"), dict) else {}
        slip = entry.get("slippage", {}) if isinstance(entry.get("slippage"), dict) else {}
        exp_txt = (
            f"expected net={expected.get('net_profit')} ({expected.get('net_profit_bps')} bps)"
            if expected.get("net_profit") is not None
            else "expected=insufficient_data"
        )
        diff_txt = f"diff net={diff.get('net_profit')}" if diff and diff.get("net_profit") is not None else "diff=n/a"
        return (
            f"journal-fact [trade:{entry.get('trade_id', '?')}|{entry.get('strategy', '?')}|"
            f"{entry.get('exchange_id', '?')}|{entry.get('status', '?')}] "
            f"realized net={realized.get('net_profit')} ({realized.get('net_profit_bps')} bps); "
            f"{exp_txt}; {diff_txt}; fees={fees.get('fees_quote')}; "
            f"slippage={slip.get('slippage_bps')} bps; outcome={entry.get('outcome', '?')}"
        )
    if kind == "aggregation":
        groups = entry.get("groups", {}) if isinstance(entry.get("groups"), dict) else {}
        bits = []
        for name in sorted(groups)[:4]:
            g = groups[name]
            bits.append(f"{name}: n={g.get('n', 0)} pnl={g.get('total_pnl', '0')} win={g.get('win_rate', '0')}%")
        detail = "; ".join(bits) if bits else "no groups"
        return f"journal-fact [aggregate by {entry.get('by', '?')} n={entry.get('total_n', 0)}] {detail}"
    if kind == "period_comparison":
        return (
            f"journal-fact [periods {entry.get('label_a', 'a')} n={entry.get('n_a', 0)} vs "
            f"{entry.get('label_b', 'b')} n={entry.get('n_b', 0)}] "
            f"delta_pnl={entry.get('delta_total_pnl')} conclusion={entry.get('conclusion', '')[:100]}"
        )
    if kind == "missed_opportunities":
        if entry.get("status") == "insufficient_data":
            return f"journal-fact [missed] insufficient_data: {entry.get('reason', '')[:120]}"
        return f"journal-fact [missed] n={entry.get('n', 0)}: {entry.get('reason', '')[:120]}"
    return None


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

        # Journal-analysis facts — deterministic app computations (FACTS).
        # Placed early so bounded truncation keeps computed answers over raw
        # dumps. Hypotheses and recommendations never enter here; the LLM
        # receives these lines as data it must explain, not override.
        for entry in list(getattr(context, "journal_analysis", ()) or ())[:MAX_KNOWLEDGE_HITS]:
            line = _format_journal_fact(entry)
            if line:
                raw_facts.append(_truncate(line, MAX_SINGLE_FACT_CHARS))

        # Memory facts — labeled distinctly from authoritative journal facts:
        # memories are past observations/lessons with decayed confidence, never
        # current journal truth. Provenance, confidence, sample size and
        # freshness ride on every line.
        for exp in list(getattr(context, "experiences", ()) or ())[:2]:
            if not isinstance(exp, dict):
                continue
            line = _format_memory_fact("memory-observation", exp)
            if line:
                raw_facts.append(_truncate(line, MAX_SINGLE_FACT_CHARS))
        for les in list(getattr(context, "lessons", ()) or ())[:2]:
            if not isinstance(les, dict):
                continue
            line = _format_memory_fact("memory-lesson", les)
            if line:
                raw_facts.append(_truncate(line, MAX_SINGLE_FACT_CHARS))

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
