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

Phase 5 adds :class:`ReflectionScheduler` — deterministic reflection
triggers (post-trade lightweight, N-trade aggregate, daily, weekly) driven
by :class:`AgentSettings` intervals. The scheduler computes statistics and
evidence first and never calls the LLM itself: trivial events stay cheap,
and the LLM (if configured) only interprets validated evidence later
through the normal :class:`AgentCore` pipeline.
"""

from __future__ import annotations

from datetime import datetime
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

__all__ = ["ReflectionEngine", "ReflectionScheduler"]

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
                evidence_count=len(experiences),
                confidence=0.0,
                has_enough_evidence=False,
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
                evidence_count=total,
                confidence=avg_conf,
                has_enough_evidence=False,
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
                evidence_count=total,
                confidence=confidence,
                has_enough_evidence=False,
            )

        evidence_ids = [e.id for e in experiences[:5]]
        # Phase 2: probable_cause and recurring_pattern are deterministic derivations
        probable_cause = _infer_probable_cause(theme, experiences)
        recurring = pattern  # same deterministic pattern counted as recurring
        obs = ReflectionObservation(
            what_happened=observations or situations,
            what_expected=expected,
            what_differed=differed,
            possible_pattern=pattern,
            probable_cause=probable_cause,
            recurring_pattern=recurring,
            evidence_count=total,
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
            evidence_count=total,
            confidence=confidence,
            has_enough_evidence=True,
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
                evidence_count=len(trades),
                confidence=0.0,
                has_enough_evidence=False,
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
                evidence_count=total,
                confidence=confidence,
                has_enough_evidence=False,
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
        # Phase 2 additional detail
        probable_cause = _infer_trade_probable_cause(fail_rate, avg_profit)
        recurring = pattern if has_pattern else None
        obs = ReflectionObservation(
            what_happened=happened,
            what_expected=expected,
            what_differed=differed,
            possible_pattern=pattern,
            probable_cause=probable_cause,
            recurring_pattern=recurring,
            evidence_count=total,
            confidence=confidence,
            has_enough_evidence=True,
            evidence=tuple(str(t.get("id", "")) for t in trades[:5]),
            source_type=SourceType.TRADE.value,
            source_id="reflection:trades",
        )

        if has_pattern:
            return ReflectionResult(
                action="INSIGHT",
                observation=obs,
                reason=differed,
                evidence_count=total,
                confidence=confidence,
                has_enough_evidence=True,
            )
        # NO_ACTION must still carry evidence_count/confidence for gating visibility
        return ReflectionResult(
            action="NO_ACTION",
            observation=None,
            reason=f"no strong pattern (fail_rate={fail_rate:.2f})",
            evidence_count=total,
            confidence=confidence,
            has_enough_evidence=False,
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


def _infer_probable_cause(theme: str, experiences: list[Experience]) -> str:
    mapping = {
        "slippage": "slippage tolerance miscalibrated or books stale at execution",
        "profit": "fee model or scanner notional misaligned with realised costs",
        "stale": "market-data freshness gate not strict enough",
        "failure": "venue rejection / recovery path triggered repeatedly",
        "liquidity": "VWAP overestimates available depth",
        "general": "unidentified execution variance — more evidence needed",
    }
    return mapping.get(theme, mapping["general"])


def _infer_trade_probable_cause(fail_rate: float, avg_profit: Decimal) -> str:
    if fail_rate >= 0.5:
        return "high venue failure rate — connectivity, balance, or risk rejection"
    if avg_profit < 0:
        return "negative expectancy — scanner threshold vs fees imbalance"
    if fail_rate >= 0.3:
        return "intermittent failures — depth or staleness likely"
    return "mixed outcomes — insufficient signal for single cause"


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


# ------------------------------------------------------------------ Phase 5 scheduler


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


#: bot_state keys for scheduler cursors (restart-safe, no trading impact).
CURSOR_POST_TRADE = "agent_reflection_cursor"
CURSOR_N_MARK = "agent_reflection_n_mark"
CURSOR_DAILY = "agent_reflection_last_daily"
CURSOR_WEEKLY = "agent_reflection_last_weekly"

_TERMINAL_TRADE_STATUSES: frozenset[str] = frozenset({"completed", "failed", "manual_review"})


class ReflectionScheduler:
    """Deterministic reflection triggers over the persisted journal.

    Triggers (all configurable via :class:`AgentSettings`):

    * post-trade — lightweight extraction for each new terminal trade
      (bounded per run, no LLM);
    * N-trade — aggregate reflection every ``reflection_n_trades`` new
      terminal trades (lesson candidate via :class:`ReflectionEngine`);
    * daily / weekly — aggregate reflection over the period window.

    Statistics and evidence are computed first; the scheduler itself never
    calls any LLM provider (it holds no provider reference). State lives in
    ``bot_state`` so restarts resume cursors instead of duplicating work.
    """

    def __init__(self, engine: ReflectionEngine | None = None) -> None:
        self._engine = engine or ReflectionEngine()

    # ------------------------------------------------------------ entry point

    async def run_due(self, services: Any, *, now: datetime | None = None) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        """Run every due trigger once; returns a provenance-rich report."""
        from app.agent.extraction import (  # local import: avoid cycle at module load
            aggregate_to_lesson,
            detect_contradiction,
            extract_experience,
            outcome_direction,
            validate_experience,
        )

        now = now or _utcnow()
        settings = getattr(getattr(services, "settings", None), "agent", None)
        report: dict[str, Any] = {
            "triggered": [],
            "experiences_created": [],
            "lessons_created": [],
            "skipped": {},
            "at": now.isoformat(),
        }
        if settings is not None and not bool(getattr(settings, "reflection_enabled", True)):
            report["skipped"]["all"] = "reflection_enabled=false"
            return report

        journal = getattr(services, "agent_journal", None)
        exp_repo = getattr(services, "agent_experiences", None)
        lesson_repo = getattr(services, "agent_lessons", None)
        label_repo = getattr(services, "agent_memory_labels", None)
        audit_repo = getattr(services, "agent_audit", None)
        bot_state = getattr(services, "bot_state", None)
        if journal is None or exp_repo is None or lesson_repo is None or bot_state is None:
            report["skipped"]["all"] = "agent not wired (journal/experiences/lessons/bot_state missing)"
            return report

        # 1. Post-trade lightweight reflection (always checked first).
        try:
            created = await self._run_post_trade(
                services, journal, exp_repo, label_repo, audit_repo, bot_state,
                extract_experience, validate_experience, now=now,
            )
            report["experiences_created"].extend(created)
            if created:
                report["triggered"].append("post_trade")
            else:
                report["skipped"]["post_trade"] = "no new terminal trades"
        except Exception as exc:  # noqa: BLE001 - scheduler must never crash callers
            logger.warning("reflection_post_trade_failed", extra={"error": str(exc)[:200]})
            report["skipped"]["post_trade"] = f"error: {exc}"

        # 2. N-trade aggregate reflection.
        try:
            lesson_id = await self._run_n_trade(
                services, journal, exp_repo, lesson_repo, label_repo, audit_repo, bot_state,
                aggregate_to_lesson, detect_contradiction, outcome_direction, now=now,
            )
            if lesson_id is not None:
                report["triggered"].append("n_trade")
                report["lessons_created"].append(lesson_id)
            else:
                report["skipped"].setdefault("n_trade", "threshold not reached or insufficient evidence")
        except Exception as exc:  # noqa: BLE001
            logger.warning("reflection_n_trade_failed", extra={"error": str(exc)[:200]})
            report["skipped"]["n_trade"] = f"error: {exc}"

        # 3. Daily / weekly aggregate reflection.
        for trigger, cursor_key, enabled_attr, hours_attr in (
            ("daily", CURSOR_DAILY, "reflection_daily_enabled", "reflection_daily_hours"),
            ("weekly", CURSOR_WEEKLY, "reflection_weekly_enabled", "reflection_weekly_hours"),
        ):
            try:
                lesson_id = await self._run_periodic(
                    services, journal, exp_repo, lesson_repo, label_repo, audit_repo, bot_state,
                    aggregate_to_lesson, detect_contradiction, outcome_direction,
                    trigger=trigger, cursor_key=cursor_key,
                    enabled_attr=enabled_attr, hours_attr=hours_attr, now=now,
                )
                if lesson_id is not None:
                    report["triggered"].append(trigger)
                    report["lessons_created"].append(lesson_id)
                else:
                    report["skipped"].setdefault(trigger, "not due or insufficient evidence")
            except Exception as exc:  # noqa: BLE001
                logger.warning("reflection_periodic_failed", extra={"trigger": trigger, "error": str(exc)[:200]})
                report["skipped"][trigger] = f"error: {exc}"
        return report

    # ------------------------------------------------------------ triggers

    async def _run_post_trade(  # type: ignore[no-untyped-def]
        self, services: Any, journal: Any, exp_repo: Any, label_repo: Any,
        audit_repo: Any, bot_state: Any, extract_experience: Any,
        validate_experience: Any, *, now: datetime, limit: int = 10,
    ) -> list[str]:
        from app.agent.models import LearningType

        cursor = await bot_state.get(CURSOR_POST_TRADE)
        views = await journal.list_trades(limit=200)
        fresh = [v for v in views if v.get("status") in _TERMINAL_TRADE_STATUSES]
        # Cursor stores max processed created_at ISO; process only newer.
        new = [v for v in fresh if not cursor or str(v.get("created_at", "")) > str(cursor)]
        new.sort(key=lambda v: str(v.get("created_at", "")))
        created: list[str] = []
        processed_cursor = cursor
        for view in new[:limit]:
            plan = await journal.transfer_plan(view.get("transfer_id"))
            exp = extract_experience(view, transfer_plan=plan)
            if validate_experience(exp):
                continue
            await exp_repo.save(exp)
            created.append(exp.id)
            if label_repo is not None:
                direction = _trade_direction(view)
                is_fact = all(view.get(f) not in (None, "", "n/a") for f in ("net_profit", "fees_quote", "status"))
                await label_repo.set_label(
                    target_type="experience", target_id=exp.id,
                    learning_type=LearningType.FACT if is_fact else LearningType.OBSERVATION,
                    sample_size=1, direction=direction,
                )
            processed_cursor = view.get("created_at") or processed_cursor
        if processed_cursor != cursor:
            await bot_state.set(CURSOR_POST_TRADE, processed_cursor)
        if audit_repo is not None and created:
            from app.agent.audit import AgentAuditEvent

            await audit_repo.log(AgentAuditEvent(
                event_type="reflection_post_trade",
                evidence_count=len(created),
                action="REFLECTED",
                details={"experience_ids": created[:10]},
                source_type="reflection",
                source_id="post_trade",
            ))
        # Keep the total counter for the N-trade trigger in sync.
        total = await bot_state.get(CURSOR_N_MARK)
        if total is None:
            await bot_state.set(CURSOR_N_MARK, 0)
        return created

    async def _run_n_trade(  # type: ignore[no-untyped-def]
        self, services: Any, journal: Any, exp_repo: Any, lesson_repo: Any,
        label_repo: Any, audit_repo: Any, bot_state: Any, aggregate_to_lesson: Any,
        detect_contradiction: Any, outcome_direction: Any, *, now: datetime,
    ) -> str | None:
        settings = getattr(getattr(services, "settings", None), "agent", None)
        threshold = int(getattr(settings, "reflection_n_trades", 20) or 20)
        mark = await bot_state.get(CURSOR_N_MARK)
        mark = int(mark) if isinstance(mark, (int, float)) or (isinstance(mark, str) and mark.isdigit()) else 0
        # Deterministic gate: reflect when the experience store grew
        # by >= threshold since the last N-mark.
        current_total = await exp_repo.count()
        if current_total - mark < threshold:
            return None
        window = await exp_repo.list_recent(limit=threshold)
        if len(window) < threshold:
            return None
        result = self._engine.reflect_on_experiences(window)
        await bot_state.set(CURSOR_N_MARK, current_total)
        if result.is_no_action or result.lesson is None:
            if audit_repo is not None:
                from app.agent.audit import AgentAuditEvent

                await audit_repo.log(AgentAuditEvent(
                    event_type="reflection_n_trade",
                    evidence_count=len(window),
                    action="NO_ACTION",
                    details={"reason": result.reason},
                    source_type="reflection",
                    source_id="n_trade",
                ))
            return None
        return await self._persist_lesson_candidate(
            lesson_repo, label_repo, audit_repo, result.lesson, window,
            aggregate_to_lesson, detect_contradiction, outcome_direction,
            trigger="n_trade",
        )

    async def _run_periodic(  # type: ignore[no-untyped-def]
        self, services: Any, journal: Any, exp_repo: Any, lesson_repo: Any,
        label_repo: Any, audit_repo: Any, bot_state: Any, aggregate_to_lesson: Any,
        detect_contradiction: Any, outcome_direction: Any, *, trigger: str,
        cursor_key: str, enabled_attr: str, hours_attr: str, now: datetime,
    ) -> str | None:
        settings = getattr(getattr(services, "settings", None), "agent", None)
        if settings is not None and not bool(getattr(settings, enabled_attr, True)):
            return None
        interval_hours = float(getattr(settings, hours_attr, 24.0) or 24.0)
        last_raw = await bot_state.get(cursor_key)
        if last_raw is not None:
            try:
                from datetime import datetime as _dt

                last = _dt.fromisoformat(str(last_raw))
                elapsed_h = (now - last).total_seconds() / 3600.0 if last.tzinfo else None
                if elapsed_h is None:
                    elapsed_h = 0.0
                if elapsed_h < interval_hours:
                    return None
            except Exception:
                pass
        window_start = now - _timedelta_hours(interval_hours * (7 if trigger == "weekly" else 1))
        experiences = await self._experiences_since(exp_repo, window_start)
        await bot_state.set(cursor_key, now.isoformat())
        if len(experiences) < _min_sample():
            if audit_repo is not None:
                from app.agent.audit import AgentAuditEvent

                await audit_repo.log(AgentAuditEvent(
                    event_type=f"reflection_{trigger}",
                    evidence_count=len(experiences),
                    action="NO_ACTION",
                    details={"reason": f"insufficient evidence: {len(experiences)}"},
                    source_type="reflection",
                    source_id=trigger,
                ))
            return None
        result = self._engine.reflect_on_experiences(experiences)
        if result.is_no_action or result.lesson is None:
            return None
        return await self._persist_lesson_candidate(
            lesson_repo, label_repo, audit_repo, result.lesson, experiences,
            aggregate_to_lesson, detect_contradiction, outcome_direction,
            trigger=trigger,
        )

    # ------------------------------------------------------------ helpers

    async def _lesson_direction(self, lesson: Any, label_repo: Any) -> str:  # type: ignore[no-untyped-def]
        """Stored direction for an existing lesson (label overlay, else unknown)."""
        try:
            if label_repo is not None:
                label = await label_repo.get_label("lesson", getattr(lesson, "id", ""))
                if label is not None and label.get("direction"):
                    return str(label["direction"])
        except Exception:
            pass
        return "unknown"

    async def _recent_experience_ids(self, exp_repo: Any, *, limit: int) -> list[str]:  # type: ignore[no-untyped-def]
        try:
            return [e.id for e in await exp_repo.list_recent(limit=limit)]
        except Exception:
            return []

    async def _experiences_since(self, exp_repo: Any, since: datetime) -> list[Any]:  # type: ignore[no-untyped-def]
        try:
            all_exps = await exp_repo.list_all()
        except Exception:
            try:
                all_exps = await exp_repo.list_recent(limit=500)
            except Exception:
                return []
        out = []
        for exp in all_exps:
            created = getattr(exp, "created_at", None)
            try:
                if created is not None and created >= since:
                    out.append(exp)
            except Exception:
                continue
        return out

    async def _persist_lesson_candidate(  # type: ignore[no-untyped-def]
        self, lesson_repo: Any, label_repo: Any, audit_repo: Any, lesson: Any,
        experiences: list[Any], aggregate_to_lesson: Any, detect_contradiction: Any,
        outcome_direction: Any, *, trigger: str,
    ) -> str | None:
        from app.agent.models import LearningType

        # Re-derive direction deterministically from evidence trade outcomes.
        trade_ids = [eid for e in experiences for eid in (e.evidence or ())]
        direction = _direction_from_experiences(experiences)
        # Contradiction check against same-theme active lessons (never overwrite).
        try:
            existing = await lesson_repo.list_all()
        except Exception:
            existing = []
        theme = _theme_of_lesson_title(getattr(lesson, "title", ""))
        for other in existing:
            if _theme_of_lesson_title(getattr(other, "title", "")) != theme or other.id == lesson.id:
                continue
            other_dir = await self._lesson_direction(other, label_repo)
            conflict = detect_contradiction(
                existing_direction=other_dir,
                new_avg_net_bps=_avg_bps_of_experiences(experiences),
                new_fail_rate=_fail_rate_of_experiences(experiences),
                existing_lesson_id=other.id,
                new_evidence_ids=trade_ids,
            )
            if conflict is not None and audit_repo is not None:
                from app.agent.audit import AgentAuditEvent

                await audit_repo.log(AgentAuditEvent(
                    event_type="contradiction",
                    evidence_count=len(trade_ids),
                    action="CONFLICT",
                    details=conflict,
                    source_type="reflection",
                    source_id=trigger,
                ))
        await lesson_repo.save(lesson)
        if label_repo is not None:
            await label_repo.set_label(
                target_type="lesson", target_id=lesson.id,
                learning_type=LearningType.OBSERVATION,
                sample_size=len(experiences), direction=direction,
            )
        if audit_repo is not None:
            from app.agent.audit import AgentAuditEvent

            await audit_repo.log(AgentAuditEvent(
                event_type=f"reflection_{trigger}",
                evidence_count=len(experiences),
                confidence=float(getattr(lesson, "confidence", 0.0)),
                action="INSIGHT",
                details={"lesson_id": lesson.id, "theme": theme, "direction": direction},
                source_type="reflection",
                source_id=trigger,
            ))
        return lesson.id


def _utcnow() -> datetime:
    from app.models.base import utc_now

    return utc_now()


def _timedelta_hours(hours: float):  # type: ignore[no-untyped-def]
    from datetime import timedelta

    return timedelta(hours=float(hours))


def _min_sample() -> int:
    try:
        from app.agent.journal import MIN_SAMPLE_FOR_CONCLUSIONS
    except Exception:
        return 5
    return int(MIN_SAMPLE_FOR_CONCLUSIONS)


def _trade_direction(view: dict[str, Any]) -> str:
    status = str(view.get("status", ""))
    try:
        from decimal import Decimal as _D

        bps = _D(str(view.get("net_profit_bps", "0") or "0"))
    except Exception:
        bps = _D("0")
    if status in ("failed", "manual_review") or bps < 0:
        return "negative"
    if status == "completed" and bps > 0:
        return "positive"
    return "mixed"


def _theme_of_lesson_title(title: str) -> str:
    text = str(title or "")
    if text.lower().startswith("pattern:"):
        return text.split(":", 1)[1].strip().lower()
    return text.strip().lower()[:80]


def _fail_rate_of_experiences(experiences: list[Any]) -> float:  # type: ignore[no-untyped-def]
    if not experiences:
        return 0.0
    failed = 0
    for exp in experiences:
        haystack = f"{getattr(exp, 'situation', '')} {getattr(exp, 'observation', '')} {getattr(exp, 'result', '')}".lower()
        tags = [str(t).lower() for t in (getattr(exp, "tags", ()) or ())]
        if "failed" in tags or "manual_review" in tags or "failed" in haystack or "manual review" in haystack:
            failed += 1
    return failed / len(experiences)


def _avg_bps_of_experiences(experiences: list[Any]) -> float:  # type: ignore[no-untyped-def]
    import re as _re

    values: list[float] = []
    for exp in experiences:
        text = f"{getattr(exp, 'situation', '')} {getattr(exp, 'result', '')}"
        match = _re.search(r"(-?\d+(?:\.\d+)?)\s*bps", text)
        if match:
            try:
                values.append(float(match.group(1)))
            except Exception:
                continue
    if not values:
        return 0.0
    return sum(values) / len(values)


def _direction_from_experiences(experiences: list[Any]) -> str:  # type: ignore[no-untyped-def]
    from app.agent.extraction import outcome_direction

    return outcome_direction(_avg_bps_of_experiences(experiences), _fail_rate_of_experiences(experiences))
