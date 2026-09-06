"""Reflection Engine — structured analysis of completed experiences / trades.

The engine answers:

* what happened
* what was expected
* what differed
* possible pattern
* confidence
* whether there is enough evidence

When evidence is insufficient it returns ``NO_ACTION`` — the advisor must
not create a recommendation from thin air. No automatic trading decision is
ever emitted.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.agent.models import (
    AgentRecommendation,
    Experience,
    Lesson,
    RecommendationStatus,
    ReflectionObservation,
    ReflectionResult,
    SourceType,
)
from app.config.logging_config import get_logger

__all__ = ["ReflectionEngine"]

logger = get_logger("agent.reflection")

# Minimum number of completed trades / experiences for a non-trivial insight.
# Below this threshold the engine returns NO_ACTION (insufficient evidence).
_MIN_EVIDENCE_COUNT = 5

# Minimum confidence for an insight to be considered actionable.
_MIN_INSIGHT_CONFIDENCE = 0.6


class ReflectionEngine:
    """First-generation reflection: heuristics + template analysis.

    Future phases may delegate to an LLM; Phase 1 uses deterministic rules
    so that the pipeline is testable offline. The interface is the contract
    that must remain stable.
    """

    def __init__(self, min_evidence: int = _MIN_EVIDENCE_COUNT) -> None:
        self._min_evidence = min_evidence

    # ------------------------------------------------------------------
    # low-level: analyse a batch of experiences / trades
    # ------------------------------------------------------------------

    def reflect_on_experiences(
        self,
        experiences: list[Experience],
        *,
        extra_context: dict[str, Any] | None = None,
    ) -> ReflectionResult:
        """Produce a structured observation from ``experiences``.

        Returns ``NO_ACTION`` when ``len(experiences) < min_evidence`` or when
        confidence is too low. Never raises; failing closed — insufficient
        evidence — is safer than hallucinating a pattern.
        """
        if len(experiences) < self._min_evidence:
            logger.info(
                "reflection_no_action_insufficient_experiences",
                extra={"count": len(experiences), "required": self._min_evidence},
            )
            return ReflectionResult(
                action="NO_ACTION",
                reason=f"insufficient evidence: {len(experiences)} experiences, need {self._min_evidence}",
            )

        # Simple pattern detection: count how often observation mentions
        # slippage / profit / failure and synthesise a narrative.
        total = len(experiences)
        with_lesson = sum(1 for e in experiences if e.lesson)
        # Average confidence across the batch
        avg_conf = sum(e.confidence for e in experiences) / total if total else 0.0

        # Heuristic: low average confidence => insufficient evidence
        if avg_conf < _MIN_INSIGHT_CONFIDENCE and with_lesson == 0:
            return ReflectionResult(
                action="NO_ACTION",
                reason=f"low confidence: avg {avg_conf:.2f} and no explicit lessons",
            )

        # Determine dominant theme (naïve keyword scan)
        theme = _dominant_theme([e.observation for e in experiences] + [e.situation for e in experiences])

        # Build a deterministic observation
        situations = "; ".join(e.situation[:80] for e in experiences[:3])
        observations = "; ".join(e.observation[:80] for e in experiences[:3])
        expected = extra_context.get("expected", "profitable execution within risk limits") if extra_context else "profitable execution within risk limits"
        differed = _infer_differed(theme, experiences)
        pattern = _infer_pattern(theme, experiences) if avg_conf >= _MIN_INSIGHT_CONFIDENCE else None
        confidence = min(0.95, max(0.35, avg_conf))

        has_enough = total >= self._min_evidence and confidence >= 0.45

        if not has_enough:
            return ReflectionResult(
                action="NO_ACTION",
                reason=f"evidence threshold not met (count={total}, confidence={confidence:.2f})",
            )

        evidence_ids = [e.id for e in experiences[:5]]
        obs = ReflectionObservation(
            what_happened=observations or situations,
            what_expected=expected,
            what_differed=differed,
            possible_pattern=pattern,
            confidence=confidence,
            has_enough_evidence=True,
            evidence=tuple(evidence_ids),
            source_type=SourceType.MEMORY.value,
            source_id="reflection:experiences",
        )

        # Optionally produce a lesson when a strong pattern is detected
        lesson: Lesson | None = None
        if pattern and confidence >= _MIN_INSIGHT_CONFIDENCE:
            lesson = Lesson(
                title=f"Pattern: {theme}",
                content=pattern,
                pattern=pattern,
                confidence=confidence,
                evidence=tuple(evidence_ids),
                related_experience_ids=tuple(evidence_ids),
                source_type=SourceType.SYSTEM.value,
                source_id="reflection",
            )

        return ReflectionResult(
            action="INSIGHT",
            observation=obs,
            lesson=lesson,
            reason=f"insight from {total} experiences (theme={theme})",
        )

    def reflect_on_trades(
        self,
        trades: list[dict[str, Any]],
        *,
        extra_context: dict[str, Any] | None = None,
    ) -> ReflectionResult:
        """Analyse a list of trade dicts (as returned by :meth:`AgentTools.get_recent_trades`).

        The dict shape matches :meth:`AgentTools.get_recent_trades`; any
        missing keys are handled defensively.
        """
        if len(trades) < self._min_evidence:
            return ReflectionResult(
                action="NO_ACTION",
                reason=f"insufficient trade evidence: {len(trades)} trades, need {self._min_evidence}",
            )

        # Compute aggregate signals
        statuses = [str(t.get("status", "")) for t in trades]
        failed = sum(1 for s in statuses if s in ("failed", "manual_review"))
        completed = sum(1 for s in statuses if s == "completed")
        total = len(trades)
        fail_rate = failed / total if total else 1.0

        # Profit signal
        profits: list[Decimal] = []
        for t in trades:
            try:
                profits.append(Decimal(str(t.get("net_profit", "0") or "0")))
            except Exception:
                profits.append(Decimal("0"))
        avg_profit = sum(profits, Decimal("0")) / len(profits) if profits else Decimal("0")

        # Confidence heuristic: more completed trades => higher confidence; high fail rate => pattern
        has_pattern = fail_rate >= 0.4 or completed == 0
        confidence = 0.5
        if total >= self._min_evidence:
            confidence += 0.1
        if has_pattern:
            confidence += 0.15
        confidence = min(0.9, max(0.3, confidence))

        if confidence < 0.45 and total < self._min_evidence * 2:
            return ReflectionResult(
                action="NO_ACTION",
                reason=f"trade evidence not confident enough (fail_rate={fail_rate:.2f}, confidence={confidence:.2f})",
            )

        if has_pattern and fail_rate >= 0.5:
            differed = f"high failure rate ({failed}/{total} failed or manual_review)"
            pattern = f"Repeated failures ({fail_rate:.0%}) — possible venue / slippage / stale-data issue; review risk limits and market-data freshness."
        elif avg_profit < 0:
            differed = f"average trade is unprofitable (avg net {avg_profit})"
            pattern = "Negative expectancy — scanner thresholds may be too loose or fees underestimated."
        else:
            differed = f"mixed outcomes ({completed} completed, {failed} failed)"
            pattern = "Execution outcomes are mixed; more data needed to isolate a dominant cause."

        expected = extra_context.get("expected", "consistent profitable execution") if extra_context else "consistent profitable execution"
        happened = f"{total} trades: {completed} completed, {failed} failed/manual_review, avg profit {avg_profit}"

        obs = ReflectionObservation(
            what_happened=happened,
            what_expected=expected,
            what_differed=differed,
            possible_pattern=pattern,
            confidence=confidence,
            has_enough_evidence=True,
            evidence=tuple(str(t.get("id", "")) for t in trades[:5]),
            source_type=SourceType.TRADE.value,
            source_id="reflection:trades",
        )

        return ReflectionResult(
            action="INSIGHT" if has_pattern else "NO_ACTION",
            observation=obs if has_pattern else None,
            reason=differed if has_pattern else f"no strong pattern (fail_rate={fail_rate:.2f})",
        )

    # ------------------------------------------------------------------
    # recommendation synthesis (gated — never auto-mutates config)
    # ------------------------------------------------------------------

    def maybe_recommend(
        self,
        result: ReflectionResult,
        *,
        current_parameters: dict[str, Any] | None = None,
    ) -> AgentRecommendation | None:
        """Optionally derive a recommendation from a reflection result.

        Returns ``None`` for ``NO_ACTION`` results (no recommendation when
        evidence is insufficient — the architectural NO_ACTION guarantee).
        When an insight exists, returns a *pending* recommendation that still
        requires human approval; it does NOT mutate any config.
        """
        if result.is_no_action or result.observation is None:
            return None
        # Guard: confidence must be actionable
        if result.observation.confidence < _MIN_INSIGHT_CONFIDENCE:
            return None
        pattern = result.observation.possible_pattern
        if not pattern:
            return None

        # Derive a conservative parameter suggestion — never a high-risk leap.
        # The suggestion is generic in Phase 1; future phases can specialize
        # per failure class (slippage -> triangle_max_leg_slippage, data age -> max_data_age_ms, etc.)
        param, old, proposed = _suggest_parameter(pattern, current_parameters)

        return AgentRecommendation(
            parameter=param,
            current_value=old,
            old_value=old,
            proposed_value=proposed,
            reason=pattern,
            evidence=result.observation.evidence,
            confidence=result.observation.confidence,
            expected_impact="Investigate and validate the observed pattern; proposed tuning is conservative.",
            risk="If the pattern is misidentified, tightening thresholds may reduce opportunity flow.",
            status=RecommendationStatus.PENDING,
            source_type=SourceType.SYSTEM.value,
            source_id="reflection",
        )


# ------------------------------------------------------------------ heuristics


def _dominant_theme(texts: list[str]) -> str:
    joined = " ".join(texts).lower()
    scores: dict[str, int] = {
        "slippage": joined.count("slippage") + joined.count("slip"),
        "profit": joined.count("profit") + joined.count("loss") + joined.count("pnl"),
        "stale": joined.count("stale") + joined.count("age") + joined.count("fresh"),
        "failure": joined.count("failed") + joined.count("rejected") + joined.count("manual"),
        "liquidity": joined.count("liquidity") + joined.count("depth") + joined.count("vwap"),
    }
    return max(scores, key=lambda k: scores[k]) if any(scores.values()) else "general"


def _infer_differed(theme: str, experiences: list[Experience]) -> str:
    mapping = {
        "slippage": "observed slippage exceeded expected tolerance",
        "profit": "realised profit diverged from planned profit",
        "stale": "execution used stale market data vs. expected fresh books",
        "failure": "execution failed or required manual review vs. expected clean completion",
        "liquidity": "depth was thinner than assumed — VWAP walked beyond estimate",
        "general": "outcome diverged from expected profitable, risk-limited execution",
    }
    return mapping.get(theme, mapping["general"])


def _infer_pattern(theme: str, experiences: list[Experience]) -> str:
    mapping = {
        "slippage": "Repeated high slippage — per-leg slippage tolerance may be too loose or books too stale; consider tightening triangle_max_leg_slippage_bps or reducing notional.",
        "profit": "Profit variance is high — scanner fee model may be miscalibrated; compare fees vs. realised fees.",
        "stale": "Market data is frequently stale at execution; consider lowering stale_after_ms or gating with require_fresh_data.",
        "failure": "Repeated REJECTED / MANUAL_REVIEW outcomes — recovery paths are engaged; review risk limits and venue health.",
        "liquidity": "Depth is overestimated — scanner VWAP assumes more liquidity than the book delivers; reduce notional or tighten slippage.",
        "general": "Execution outcomes show variance from plan — collect more evidence before tuning.",
    }
    return mapping.get(theme, mapping["general"])


def _suggest_parameter(
    pattern: str,
    current_params: dict[str, Any] | None,
) -> tuple[str, str | None, str]:
    """Map a pattern string to a conservative parameter suggestion.

    Returns (parameter_path, old_value_str_or_None, proposed_value_str).
    The proposal is intentionally modest (± small delta) — never a large swing.
    """
    low = pattern.lower()
    if "slippage" in low:
        old = _get_param(current_params, "max_slippage_bps") or "15"
        try:
            proposed = str(max(5, int(Decimal(old) * Decimal("0.85"))))
        except Exception:
            proposed = "12"
        return ("risk.max_slippage_bps", old, proposed)
    if "stale" in low or "fresh" in low:
        old = _get_param(current_params, "max_data_age_ms") or "2500"
        try:
            proposed = str(max(800, int(Decimal(old) * Decimal("0.80"))))
        except Exception:
            proposed = "2000"
        return ("risk.max_data_age_ms", old, proposed)
    if "profit" in low or "fee" in low:
        old = _get_param(current_params, "min_net_profit_bps") or "10"
        try:
            proposed = str(int(Decimal(old) + Decimal("5")))
        except Exception:
            proposed = "15"
        return ("risk.min_net_profit_bps", old, proposed)
    # Default: conservative trade-size reduction (never increase risk)
    old = _get_param(current_params, "max_trade_size") or "1000"
    try:
        proposed = str(Decimal(old) * Decimal("0.90"))
    except Exception:
        proposed = "900"
    return ("risk.max_trade_size", old, proposed)


def _get_param(params: dict[str, Any] | None, key: str) -> str | None:
    if not params or not isinstance(params, dict):
        return None
    # Try flat risk map first (tools.get_risk_state / status()["risk"]["limits"] style)
    if key in params:
        return str(params[key])
    risk = params.get("risk") if isinstance(params.get("risk"), dict) else None
    if risk is None:
        # Try nested: current_parameters["risk"][key]
        risk = params.get("risk", {})
    if isinstance(risk, dict) and key in risk:
        return str(risk[key])
    # Also check limits sub-map
    limits = risk.get("limits") if isinstance(risk, dict) else None
    if isinstance(limits, dict) and key in limits:
        return str(limits[key])
    return None
