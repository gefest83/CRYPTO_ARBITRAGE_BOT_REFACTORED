"""Phase 7 — evidence-based Recommendation Engine.

``Journal + Memory + Knowledge + deterministic statistics → recommendations``

The engine derives conservative tuning candidates from persisted evidence:

* journal aggregates (completed-trade slippage / failure-rate / PnL edge);
* memory lessons/experiences as supporting evidence ids;
* knowledge docs as supporting evidence ids (titles only, never authority).

Every number (profit, edge, fees, slippage, sample size, confidence) is
computed here by deterministic application code. An optional LLM may draft
explanatory prose for the ``reason`` field, but it can never create
authoritative numbers — on any LLM failure the template reason is used.

Critical safety (tested):

* generation persists rows only (status PENDING); it NEVER mutates
  configuration, risk limits, thresholds, execution settings, LIVE mode,
  and NEVER executes trades. Phase 8 approval workflow is NOT implemented.
* insufficient samples → explicit skip reasons, no weak recommendations;
* exact duplicates (same parameter + proposed value among PENDING/REVIEWED)
  return the existing row — no duplicate rows;
* conflicting values for the same parameter are both retained for the
  human, with a ``recommendation_conflict`` audit event.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from app.agent.approval import ALLOWLIST
from app.agent.journal import MIN_SAMPLE_FOR_CONCLUSIONS
from app.agent.models import AgentRecommendation, RecommendationStatus, SourceType
from app.config.logging_config import get_logger

__all__ = ["RecommendationEngine"]

logger = get_logger("agent.recommender")


def _dec(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError, AttributeError):
        return None


def _clamp_to_allowlist(parameter: str, value: Decimal) -> Decimal:
    spec = ALLOWLIST.get(parameter)
    if spec is None:
        raise ValueError(f"parameter not allowlisted: {parameter}")
    minimum = Decimal(str(spec["min"]))
    maximum = Decimal(str(spec["max"]))
    return max(minimum, min(maximum, value))


def _confidence_for_sample(n: int) -> float:
    """Deterministic confidence from sample size (capped, never strong alone)."""
    if n < MIN_SAMPLE_FOR_CONCLUSIONS:
        return 0.0
    return round(min(0.85, 0.55 + 0.02 * n), 4)


class RecommendationEngine:
    """Deterministic, evidence-based recommendation generation.

    Bound to :class:`AppServices` (duck-typed) for journal reads, current
    parameters, memory/knowledge evidence, persistence, audit and optional
    Phase 6 notification. Holds no trading/execution references.
    """

    #: Deterministic rules evaluated in a stable order.
    RULES: tuple[str, ...] = ("slippage", "failure_rate", "profit_edge")

    def __init__(self, services: Any) -> None:  # type: ignore[no-untyped-def]
        self._services = services

    # ------------------------------------------------------------ entry point

    async def generate(self, *, limit: int = 5, llm: Any | None = None) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        """Generate evidence-based recommendations (persistence-only side effect).

        Returns ``{"created": [...], "deduplicated": [...], "skipped": {...},
        "notified": [...]}`` with recommendation ids and explicit reasons.
        Never raises for data problems (insufficient data is reported, not
        thrown); never mutates anything but the recommendations table.
        """
        report: dict[str, Any] = {"created": [], "deduplicated": [], "skipped": {}, "notified": []}
        try:
            trades = await self._recent_trades(limit=100)
        except Exception as exc:  # noqa: BLE001 - journal failure → explicit skip
            logger.warning("recommender_journal_failed", extra={"error": str(exc)[:200]})
            report["skipped"]["journal"] = f"journal read failed: {exc}"
            return report
        if not trades:
            report["skipped"]["journal"] = "insufficient_data: no journaled trades"
            return report
        try:
            memory_ids = await self._memory_evidence_ids()
            knowledge_ids = await self._knowledge_evidence_ids()
        except Exception:
            memory_ids, knowledge_ids = [], []
        for rule in self.RULES:
            if len(report["created"]) >= max(1, int(limit)):
                break
            try:
                candidate = self._build_candidate(rule, trades)
            except Exception as exc:  # noqa: BLE001 - one bad rule stops nothing
                logger.warning("recommender_rule_failed", extra={"rule": rule, "error": str(exc)[:200]})
                report["skipped"][rule] = f"error: {exc}"
                continue
            if candidate is None:
                continue
            if candidate.get("insufficient"):
                report["skipped"][candidate["rule"]] = candidate["insufficient"]
                continue
            try:
                outcome = await self._persist_candidate(candidate, memory_ids, knowledge_ids, llm=llm)
            except Exception as exc:  # noqa: BLE001
                logger.warning("recommender_persist_failed", extra={"rule": rule, "error": str(exc)[:200]})
                report["skipped"][rule] = f"persist failed: {exc}"
                continue
            if outcome["status"] == "created":
                report["created"].append(outcome["id"])
                if await self._notify_new(outcome["recommendation"]):
                    report["notified"].append(outcome["id"])
            else:
                report["deduplicated"].append(outcome["id"])
        return report

    # ------------------------------------------------------------ inputs

    async def _recent_trades(self, *, limit: int = 100) -> list[Any]:  # type: ignore[no-untyped-def]
        return await self._services.trades.list_recent(limit=min(int(limit), 200))

    async def _memory_evidence_ids(self, *, limit: int = 5) -> list[str]:
        ids: list[str] = []
        for repo_attr in ("agent_experiences", "agent_lessons"):
            try:
                repo = getattr(self._services, repo_attr, None)
                if repo is None or not hasattr(repo, "list_recent"):
                    continue
                for item in await repo.list_recent(limit=limit):
                    ids.append(getattr(item, "id", "?"))
            except Exception:
                continue
        return ids[:10]

    async def _knowledge_evidence_ids(self, *, limit: int = 3) -> list[str]:
        try:
            service = getattr(self._services, "agent_knowledge_service", None)
            if service is None or not hasattr(service, "get_recent"):
                return []
            return [doc.id for doc in await service.get_recent(limit=limit)]
        except Exception:
            return []

    def _current_param(self, parameter: str) -> str | None:
        """Read the live current value for an allowlisted parameter (no write)."""
        try:
            settings = self._services.settings
        except Exception:
            return None
        try:
            section, _, name = parameter.partition(".")
            group = getattr(settings, section, None)
            if group is None:
                return None
            return str(getattr(group, name))
        except Exception:
            return None

    # ------------------------------------------------------------ rules (deterministic)

    @staticmethod
    def _status_of(trade: Any) -> str:
        status = getattr(trade, "status", "?")
        return str(getattr(status, "value", status))

    def _build_candidate(self, rule: str, trades: list[Any]) -> dict[str, Any] | None:  # type: ignore[no-untyped-def]
        if rule == "slippage":
            return self._slippage_candidate(trades)
        if rule == "failure_rate":
            return self._failure_candidate(trades)
        if rule == "profit_edge":
            return self._profit_candidate(trades)
        return None

    def _slippage_candidate(self, trades: list[Any]) -> dict[str, Any] | None:  # type: ignore[no-untyped-def]
        completed = [t for t in trades if self._status_of(t) == "completed"]
        slips = [(t, _dec(getattr(t, "slippage_bps", None))) for t in completed]
        slips = [(t, s) for t, s in slips if s is not None]
        if len(slips) < MIN_SAMPLE_FOR_CONCLUSIONS:
            return {"rule": "slippage",
                    "insufficient": f"insufficient_data: {len(slips)} completed trades with slippage, need {MIN_SAMPLE_FOR_CONCLUSIONS}"}
        avg = sum((s for _, s in slips), Decimal("0")) / len(slips)
        limit = _dec(self._current_param("risk.max_slippage_bps"))
        if limit is None or avg <= limit:
            return {"rule": "slippage",
                    "insufficient": f"no signal: avg slippage {avg} bps within limit {limit} bps"}
        proposed = _clamp_to_allowlist("risk.max_slippage_bps", (limit * Decimal("0.85")).quantize(Decimal("1")))
        return {
            "rule": "slippage",
            "parameter": "risk.max_slippage_bps",
            "current_value": str(limit),
            "proposed_value": str(proposed),
            "reason": (f"Avg realized slippage {avg:.1f} bps over {len(slips)} completed trades "
                       f"exceeds limit {limit} bps; tighten tolerance conservatively."),
            "evidence": [t.id for t, _ in slips[:20]],
            "confidence": _confidence_for_sample(len(slips)),
            "expected_impact": "Fewer high-slippage fills; may reduce fill rate.",
            "risk": "Too tight a tolerance could reject viable cycles.",
            "stats": {"avg_slippage_bps": str(avg), "n": len(slips), "limit_bps": str(limit)},
        }

    def _failure_candidate(self, trades: list[Any]) -> dict[str, Any] | None:  # type: ignore[no-untyped-def]
        window = trades[:20]
        terminal = [t for t in window if self._status_of(t) in ("completed", "failed", "manual_review")]
        if len(terminal) < MIN_SAMPLE_FOR_CONCLUSIONS:
            return {"rule": "failure_rate",
                    "insufficient": f"insufficient_data: {len(terminal)} terminal trades, need {MIN_SAMPLE_FOR_CONCLUSIONS}"}
        failed = [t for t in terminal if self._status_of(t) in ("failed", "manual_review")]
        rate = len(failed) / len(terminal)
        if rate < 0.5:
            return {"rule": "failure_rate",
                    "insufficient": f"no signal: failure rate {rate:.0%} below 50% over {len(terminal)} trades"}
        current = _dec(self._current_param("risk.max_trade_size"))
        if current is None:
            return {"rule": "failure_rate", "insufficient": "no signal: current max_trade_size unreadable"}
        proposed = _clamp_to_allowlist("risk.max_trade_size", (current * Decimal("0.9")).quantize(Decimal("1")))
        if proposed == current:
            return {"rule": "failure_rate", "insufficient": "no signal: already at allowlist floor"}
        return {
            "rule": "failure_rate",
            "parameter": "risk.max_trade_size",
            "current_value": str(current),
            "proposed_value": str(proposed),
            "reason": (f"{len(failed)}/{len(terminal)} recent trades failed or need review "
                       f"({rate:.0%}); reduce exposure conservatively."),
            "evidence": [t.id for t in terminal[:20]],
            "confidence": _confidence_for_sample(len(terminal)),
            "expected_impact": "Smaller worst-case loss per trade while failures persist.",
            "risk": "Lower size also lowers profit on recovery.",
            "stats": {"fail_rate": round(rate, 4), "n": len(terminal)},
        }

    def _profit_candidate(self, trades: list[Any]) -> dict[str, Any] | None:  # type: ignore[no-untyped-def]
        completed = [t for t in trades if self._status_of(t) == "completed"]
        pnls = [(t, _dec(getattr(t, "net_profit_bps", None))) for t in completed]
        pnls = [(t, p) for t, p in pnls if p is not None]
        if len(pnls) < MIN_SAMPLE_FOR_CONCLUSIONS:
            return {"rule": "profit_edge",
                    "insufficient": f"insufficient_data: {len(pnls)} completed trades, need {MIN_SAMPLE_FOR_CONCLUSIONS}"}
        avg = sum((p for _, p in pnls), Decimal("0")) / len(pnls)
        threshold = _dec(self._current_param("arbitrage.triangle_min_net_bps"))
        if threshold is None or avg >= threshold:
            return {"rule": "profit_edge",
                    "insufficient": f"no signal: avg edge {avg} bps meets threshold {threshold} bps"}
        proposed = _clamp_to_allowlist("arbitrage.triangle_min_net_bps", threshold + Decimal("5"))
        if proposed == threshold:
            return {"rule": "profit_edge", "insufficient": "no signal: already at allowlist ceiling"}
        return {
            "rule": "profit_edge",
            "parameter": "arbitrage.triangle_min_net_bps",
            "current_value": str(threshold),
            "proposed_value": str(proposed),
            "reason": (f"Avg realized edge {avg:.1f} bps over {len(pnls)} completed trades "
                       f"undercuts threshold {threshold} bps; raise the bar conservatively."),
            "evidence": [t.id for t, _ in pnls[:20]],
            "confidence": _confidence_for_sample(len(pnls)),
            "expected_impact": "Filters marginal cycles that realize below costs.",
            "risk": "Fewer opportunities overall.",
            "stats": {"avg_net_bps": str(avg), "n": len(pnls), "threshold_bps": str(threshold)},
        }

    # ------------------------------------------------------------ persist + audit + notify

    async def _persist_candidate(  # type: ignore[no-untyped-def]
        self, candidate: dict[str, Any], memory_ids: list[str], knowledge_ids: list[str],
        *, llm: Any | None = None,
    ) -> dict[str, Any]:
        service = self._services.agent_recommendation_service
        evidence = list(candidate["evidence"]) + list(memory_ids) + list(knowledge_ids)
        # Deduplication: same parameter + proposed value among actionable rows.
        try:
            existing = await service.repository.list_by_parameter(candidate["parameter"])
        except Exception:
            existing = []
        actionable = [r for r in existing
                      if str(getattr(getattr(r, "status", ""), "value", r.status))
                      in (RecommendationStatus.PENDING.value, RecommendationStatus.REVIEWED.value)]
        for row in actionable:
            if str(row.proposed_value).strip() == str(candidate["proposed_value"]).strip():
                return {"status": "deduplicated", "id": row.id, "recommendation": row}
        # Contradiction: same parameter, different value — retain both, audit it.
        conflicts = [r for r in actionable if r.id]
        reason = candidate["reason"]
        if conflicts:
            reason += f" Conflicts with {conflicts[0].id}; both retained for human review."
        if llm is not None:
            try:
                note = await self._explain_with_llm(candidate, llm)
                if note:
                    reason += f" AI note: {note}"
            except Exception as exc:  # noqa: BLE001 - LLM failure → template reason stands
                logger.warning("recommender_llm_failed", extra={"rule": candidate["rule"], "error": str(exc)[:200]})
        rec = await service.create(
            candidate["parameter"],
            current_value=candidate.get("current_value"),
            proposed_value=candidate["proposed_value"],
            reason=reason[:2000],
            evidence=tuple(evidence[:40]),
            confidence=candidate["confidence"],
            expected_impact=candidate.get("expected_impact", ""),
            risk=candidate.get("risk", ""),
            source_type=SourceType.SYSTEM.value,
            source_id=f"phase7:{candidate['rule']}",
            status=RecommendationStatus.PENDING,
        )
        try:
            await self._services.agent_audit.log_recommendation(rec, event_type="recommendation_created")
        except Exception as exc:  # noqa: BLE001 - audit failure must not break generation
            logger.warning("recommender_audit_failed", extra={"error": str(exc)[:200]})
        if conflicts:
            await self._audit_conflict(rec, conflicts[0])
        return {"status": "created", "id": rec.id, "recommendation": rec}

    async def _audit_conflict(self, new_rec: AgentRecommendation, existing_rec: AgentRecommendation) -> None:  # type: ignore[no-untyped-def]
        try:
            from app.agent.audit import AgentAuditEvent

            await self._services.agent_audit.log(AgentAuditEvent(
                event_type="recommendation_conflict",
                evidence_count=len(new_rec.evidence),
                confidence=float(new_rec.confidence),
                action="CONFLICT",
                details={"new_id": new_rec.id, "existing_id": existing_rec.id,
                         "parameter": new_rec.parameter,
                         "new_value": new_rec.proposed_value,
                         "existing_value": existing_rec.proposed_value,
                         "resolution": "retained_both"},
                source_type=SourceType.SYSTEM.value,
                source_id="phase7:recommender",
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("recommender_conflict_audit_failed", extra={"error": str(exc)[:200]})

    async def _notify_new(self, rec: AgentRecommendation) -> bool:
        """Push a ``new_recommendation`` event through the Phase 6 layer."""
        try:
            notifier = getattr(self._services, "agent_notifications", None)
            if notifier is None or not hasattr(notifier, "notify"):
                return False
            result = await notifier.notify(
                "new_recommendation",
                {"parameter": str(rec.parameter)[:80],
                 "old": str(rec.current_value or rec.old_value or "-")[:40],
                 "proposed": str(rec.proposed_value)[:40],
                 "reason": str(rec.reason or "")[:120],
                 "confidence": f"{float(rec.confidence):.2f}",
                 "rec_id": rec.id},
                {"recommendation_id": rec.id},
            )
            return int(result.get("sent", 0)) > 0
        except Exception as exc:  # noqa: BLE001 - notification never breaks generation
            logger.warning("recommender_notify_failed", extra={"error": str(exc)[:200]})
            return False

    async def _explain_with_llm(self, candidate: dict[str, Any], llm: Any) -> str | None:  # type: ignore[no-untyped-def]
        """LLM drafts prose from validated numbers (or raises → template stands)."""
        from app.agent.providers.base import LLMMessage, LLMRequest, filter_secrets_from_text

        stats = ", ".join(f"{k}={v}" for k, v in (candidate.get("stats") or {}).items())
        prompt = (
            "Explain in one sentence why this parameter change is suggested. "
            f"Use only these validated numbers: {stats[:400]}. "
            f"Parameter {candidate['parameter']} "
            f"{candidate.get('current_value')} -> {candidate['proposed_value']}."
        )
        response = await llm.complete(LLMRequest(messages=(LLMMessage(role="user", content=prompt),)))
        text = filter_secrets_from_text(response.content or "")[:200].strip()
        if len(text) < 5:
            raise ValueError("empty LLM explanation")
        return text

