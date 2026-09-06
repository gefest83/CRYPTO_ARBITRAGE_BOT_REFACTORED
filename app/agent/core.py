"""Agent Core: request -> context -> memory -> knowledge -> analysis -> response.

The core is independent of Telegram and CLI. It owns the pipeline:

    request
        -> context collection (tools)
        -> memory retrieval (experiences / lessons)
        -> knowledge retrieval (docs)
        -> analysis (reflection + LLM provider abstraction)
        -> response (insight / NO_ACTION / recommendation)

The only allowed output of the analysis stage is a :class:`AgentRecommendation`
or a ``NO_ACTION`` reflection. There is no code path from LLM output directly
to configuration mutation — the boundary is enforced by
:class:`RecommendationService.apply` raising :class:`RecommendationApplyBlocked`.

Secrets are filtered before any prompt is sent to the LLM and before any
response is persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.context import AgentContext, ContextCollector
from app.agent.models import AgentRecommendation, ReflectionResult
from app.agent.providers.base import LLMProvider, NullProvider, filter_secrets_from_text
from app.agent.reflection import ReflectionEngine
from app.config.logging_config import get_logger

__all__ = ["AgentCore", "AgentRequest", "AgentResponse"]

logger = get_logger("agent.core")


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Inbound request to the advisor."""

    query: str
    language: str | None = None
    include_balances: bool = True
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Outbound response from the advisor — analytical, never executive."""

    query: str
    context: AgentContext
    reflection: ReflectionResult | None
    recommendation: AgentRecommendation | None
    llm_output: str | None
    language: str | None = None
    # Convenience flag
    is_no_action: bool = False

    def summary(self, lang: str | None = None) -> str:
        """Human-readable summary (respects requested language if provided)."""
        effective = lang or self.language or "en"
        if self.is_no_action or self.reflection is None or self.reflection.is_no_action:
            reason = self.reflection.reason if self.reflection else "no reflection executed"
            note = "Insufficient evidence — no recommendation at this time."
            if effective.startswith("ru"):
                note = "Недостаточно данных — рекомендации нет."
            return f"{note} ({reason})"
        parts: list[str] = []
        if self.reflection and self.reflection.observation:
            obs = self.reflection.observation
            parts.append(f"Observation: {obs.what_happened}")
            if obs.possible_pattern:
                parts.append(f"Pattern: {obs.possible_pattern} (confidence {obs.confidence:.2f})")
        if self.recommendation:
            rec = self.recommendation
            parts.append(f"Recommendation: {rec.parameter} {rec.current_value} -> {rec.proposed_value} — {rec.reason}")
        if self.llm_output:
            parts.append(f"LLM: {self.llm_output[:500]}")
        return "\n".join(parts) if parts else "No insight — see reflection."


class AgentCore:
    """Independent analytical core — no Telegram or CLI coupling."""

    def __init__(
        self,
        collector: ContextCollector,
        reflection: ReflectionEngine | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self._collector = collector
        self._reflection = reflection or ReflectionEngine()
        self._llm = llm or NullProvider()

    @property
    def llm_provider(self) -> LLMProvider:
        return self._llm

    @property
    def reflection_engine(self) -> ReflectionEngine:
        return self._reflection

    # ------------------------------------------------------------------ pipeline

    async def handle(self, request: AgentRequest) -> AgentResponse:
        """Execute the full pipeline for one advisor request."""
        query = request.query.strip() if request.query else ""
        if not query:
            query = "status overview"

        # 1. Context collection (read-only tools + memory + knowledge)
        context = await self._collector.collect(
            query=query,
            language=request.language,
            include_balances=request.include_balances,
        )

        # 2. Memory / knowledge are already in context; reflection consumes recent trades
        # Prefer structured memory when available, else fall back to trades.
        reflection: ReflectionResult | None = None
        recommendation: AgentRecommendation | None = None
        llm_output: str | None = None

        # Use get_memory trade dicts for reflection
        recent_trades = list(context.recent_trades)  # already list of dicts
        if recent_trades:
            try:
                reflection = self._reflection.reflect_on_trades(recent_trades)
            except Exception as exc:  # noqa: BLE001 - reflection must never crash the pipeline
                logger.warning("reflection_failed", extra={"error": str(exc)[:300]})
                reflection = ReflectionResult(action="NO_ACTION", reason=f"reflection error: {exc}")

        # 3. If reflection produced NO_ACTION, try experiences fallback before giving up
        if (reflection is None or reflection.is_no_action) and context.experiences:
            try:
                # Build minimal Experience-like dicts from context experiences if trade reflection yielded nothing
                from app.agent.models import Experience, SourceType

                exps = [
                    Experience(
                        situation=e.get("situation", "unknown"),
                        observation=e.get("observation", "unknown"),
                        source_id=e.get("id", "ctx"),
                        source_type=SourceType.MEMORY.value,
                        confidence=float(e.get("confidence", 0.5)),
                    )
                    for e in context.experiences
                ]
                exp_result = self._reflection.reflect_on_experiences(exps)
                if not exp_result.is_no_action:
                    reflection = exp_result
            except Exception:
                pass

        # 4. Maybe derive a recommendation (gated — never mutates config)
        if reflection is not None and not reflection.is_no_action:
            try:
                recommendation = self._reflection.maybe_recommend(
                    reflection,
                    current_parameters=context.current_parameters,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("recommendation_synthesis_failed", extra={"error": str(exc)[:300]})

        # 5. LLM analysis — safe boundary (prompt filtered for secrets, response filtered too)
        # Only call LLM when there is something to analyse; skip on pure NO_ACTION to save calls
        if reflection is not None and not reflection.is_no_action:
            try:
                prompt = _build_llm_prompt(query, context, reflection, recommendation)
                safe_prompt = filter_secrets_from_text(prompt)
                from app.agent.providers.base import LLMRequest, LLMMessage

                response = await self._llm.complete(
                    LLMRequest(messages=(LLMMessage(role="user", content=safe_prompt),))
                )
                llm_output = filter_secrets_from_text(response.content)
            except Exception as exc:  # noqa: BLE001 - LLM failures are non-fatal
                logger.warning("llm_analysis_failed", extra={"error": str(exc)[:300]})
                llm_output = None

        is_no_action = reflection is None or reflection.is_no_action

        return AgentResponse(
            query=query,
            context=context,
            reflection=reflection,
            recommendation=recommendation,
            llm_output=llm_output,
            language=request.language,
            is_no_action=is_no_action,
        )

    async def status(self, language: str | None = None) -> dict[str, Any]:
        """Quick health/status snapshot — no LLM call, pure read-only context."""
        ctx = await self._collector.collect(query=None, language=language, include_balances=False)
        return {
            "recent_trades": len(ctx.recent_trades),
            "trade_stats": ctx.trade_statistics,
            "experiences": len(ctx.experiences),
            "lessons": len(ctx.lessons),
            "knowledge_docs": len(ctx.knowledge_hits),
            "risk": ctx.risk_state,
            "exchanges": list(ctx.exchange_status.keys()),
            "language": language or "en",
        }


# ------------------------------------------------------------------ llm prompt


def _build_llm_prompt(
    query: str,
    context: AgentContext,
    reflection: ReflectionResult | None,
    recommendation: AgentRecommendation | None,
) -> str:
    """Build a sanitized prompt for the LLM — never includes secrets."""
    parts: list[str] = [
        "You are the AI Advisor for a crypto arbitrage bot. You are ANALYTICAL only.",
        "You may NOT place orders, transfer funds, or mutate configuration.",
        "Provide a concise analysis and, if evidence supports it, justify a recommendation.",
        "",
        f"User query: {query}",
        "",
        "Context summary:",
        context.summary()[:3000],
        "",
    ]
    if reflection and reflection.observation:
        obs = reflection.observation
        parts.extend(
            [
                "Reflection:",
                f"- What happened: {obs.what_happened}",
                f"- What expected: {obs.what_expected}",
                f"- What differed: {obs.what_differed}",
                f"- Pattern: {obs.possible_pattern} (confidence {obs.confidence:.2f})",
                f"- Evidence: {list(obs.evidence)}",
                "",
            ]
        )
    elif reflection:
        parts.append(f"Reflection: {reflection.action} — {reflection.reason}\n")

    if recommendation:
        parts.extend(
            [
                "Derived recommendation (requires human approval, not yet applied):",
                f"- Parameter: {recommendation.parameter}",
                f"- Current: {recommendation.current_value} -> Proposed: {recommendation.proposed_value}",
                f"- Reason: {recommendation.reason}",
                f"- Confidence: {recommendation.confidence:.2f}",
                "",
            ]
        )

    # Truncate knowledge but include titles
    if context.knowledge_hits:
        parts.append("Relevant knowledge (titles):")
        for hit in context.knowledge_hits[:5]:
            title = hit.get("title", "") if isinstance(hit, dict) else getattr(hit, "title", "")
            parts.append(f"- {title}")
        parts.append("")

    parts.append("Respond concisely (max 400 words). If evidence is weak, say NO_ACTION and why.")
    return "\n".join(parts)
