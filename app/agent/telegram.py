"""Telegram service boundary for the AI Advisor (Phase 1 — preparation).

The existing Telegram bot in :mod:`app.telegram.bot` is intentionally *not*
redesigned in Phase 1. This module provides a **clean, isolated service/API
boundary** so that Phase 2 can add operator commands without touching the
advisor core:

    /ai                – overview / help
    /ai status         – advisor health, memory counts
    /ai report         – last reflection insight (or NO_ACTION)
    /ai recommendations – pending recommendations
    /ai memory         – recent experiences / lessons
    /ai balance        – balances via the read-only tool path

The layer is deliberately thin:

* it owns *no* trading logic;
* it only reads through :class:`AgentCore` / :class:`AgentTools`;
* it respects the existing :mod:`app.telegram.i18n` language setting;
  future AI notifications must use the operator's selected Telegram language.

Phase 1 wires the adapter but does not yet register the handlers in
:class:`app.telegram.bot.TelegramBot`. A follow-up phase can add them in
one place without opening the advisor core to Telegram-specific concerns.
"""

from __future__ import annotations

from typing import Any

from app.agent.core import AgentCore, AgentRequest
from app.agent.tools import AgentTools
from app.telegram.i18n import t

__all__ = ["AgentTelegramAdapter", "AI_COMMANDS"]


AI_COMMANDS: tuple[str, ...] = (
    "/ai",
    "/ai status",
    "/ai report",
    "/ai recommendations",
    "/ai memory",
    "/ai balance",
    "/ai approve",
    "/ai reject",
)

# Additional translations for AI Advisor (English / Russian).
# These keys are namespaced under ``ai_*`` so they never clash with the
# existing bot texts. They use the same :func:`t` mechanism so Phase 2
# notifications can call ``t("ai_...", lang)`` just like the rest of the bot.
AI_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "ai_help": (
            "AI Advisor — analytical layer (read-only):\n"
            "/ai status          - advisor health & memory counts\n"
            "/ai report          - last reflection / insight\n"
            "/ai recommendations - pending recommendations (human approval required)\n"
            "/ai memory          - recent experiences & lessons\n"
            "/ai balance         - balances per venue (via bot services)\n"
            "/ai approve <id>    - approve and apply a recommendation (human-only, allowlisted)\n"
            "/ai reject <id>     - reject a pending recommendation\n"
            "\n"
            "The advisor can READ, ANALYZE, REMEMBER, REFLECT and RECOMMEND. "
            "It cannot execute trades, withdraw funds, or modify configuration directly."
        ),
        "ai_status_title": "AI Advisor status:",
        "ai_status_trades": "  recent trades: {count}",
        "ai_status_experiences": "  experiences: {count}",
        "ai_status_lessons": "  lessons: {count}",
        "ai_status_knowledge": "  knowledge docs: {count}",
        "ai_status_recommendations": "  pending recommendations: {count}",
        "ai_report_no_action": "No actionable insight — {reason}",
        "ai_report_insight": (
            "Insight (confidence {confidence:.2f}):\n"
            "  happened: {happened}\n"
            "  expected: {expected}\n"
            "  differed: {differed}\n"
            "  pattern: {pattern}"
        ),
        "ai_recommendations_empty": "No pending recommendations.",
        "ai_recommendations_header": "Pending recommendations ({count}):",
        "ai_recommendations_line": "  {parameter}: {old} -> {proposed} — {reason} (conf {confidence:.2f})",
        "ai_memory_header": "Memory (recent {count}):",
        "ai_memory_experience": "  exp {id}: {situation} | {observation} (conf {confidence:.2f})",
        "ai_memory_lesson": "  les {id}: {title} — {content} (conf {confidence:.2f})",
        "ai_memory_empty": "No memory entries yet.",
        "ai_balance_title": "Balances per venue:",
        "ai_balance_line": "  {venue}: {assets}",
        "ai_balance_empty": "No balances available.",
        "ai_unknown_subcommand": "Unknown /ai subcommand. Try /ai for help.",
        "ai_not_configured": "AI Advisor is not configured.",
        "ai_approve_ok": "Approved {parameter}: {old} -> {proposed} (by {approver})",
        "ai_approve_fail": "Approve failed: {error}",
        "ai_reject_ok": "Rejected {id} (by {approver})",
        "ai_reject_fail": "Reject failed: {error}",
        "ai_approve_usage": "Usage: /ai approve <recommendation_id>",
        "ai_reject_usage": "Usage: /ai reject <recommendation_id>",
    },
    "ru": {
        "ai_help": (
            "AI-советник — аналитический слой (только чтение):\n"
            "/ai status          - состояние и счётчики памяти\n"
            "/ai report          - последний инсайт рефлексии\n"
            "/ai recommendations - ожидающие рекомендации (требуют подтверждения)\n"
            "/ai memory          - недавние опыты и уроки\n"
            "/ai balance         - балансы по площадкам\n"
            "/ai approve <id>    - подтвердить и применить рекомендацию (только оператор, allowlist)\n"
            "/ai reject <id>     - отклонить ожидующую рекомендацию\n"
            "\n"
            "Советник может ЧИТАТЬ, АНАЛИЗИРОВАТЬ, ЗАПОМИНАТЬ, РЕФЛЕКСИРОВАТЬ и РЕКОМЕНДОВАТЬ. "
            "Он не исполняет сделки, не выводит средства и не меняет конфигурацию напрямую."
        ),
        "ai_status_title": "Статус AI-советника:",
        "ai_status_trades": "  недавние сделки: {count}",
        "ai_status_experiences": "  опыты: {count}",
        "ai_status_lessons": "  уроки: {count}",
        "ai_status_knowledge": "  документы знаний: {count}",
        "ai_status_recommendations": "  ожидающие рекомендации: {count}",
        "ai_report_no_action": "Нет применимых инсайтов — {reason}",
        "ai_report_insight": (
            "Инсайт (уверенность {confidence:.2f}):\n"
            "  случилось: {happened}\n"
            "  ожидалось: {expected}\n"
            "  отличие: {differed}\n"
            "  паттерн: {pattern}"
        ),
        "ai_recommendations_empty": "Нет ожидающих рекомендаций.",
        "ai_recommendations_header": "Ожидающие рекомендации ({count}):",
        "ai_recommendations_line": "  {parameter}: {old} -> {proposed} — {reason} (увер {confidence:.2f})",
        "ai_memory_header": "Память (последние {count}):",
        "ai_memory_experience": "  опыт {id}: {situation} | {observation} (увер {confidence:.2f})",
        "ai_memory_lesson": "  урок {id}: {title} — {content} (увер {confidence:.2f})",
        "ai_memory_empty": "Пока нет записей в памяти.",
        "ai_balance_title": "Балансы по площадкам:",
        "ai_balance_line": "  {venue}: {assets}",
        "ai_balance_empty": "Балансы недоступны.",
        "ai_unknown_subcommand": "Неизвестная подкоманда /ai. Попробуйте /ai для справки.",
        "ai_not_configured": "AI-советник не настроен.",
        "ai_approve_ok": "Одобрено {parameter}: {old} -> {proposed} (кем {approver})",
        "ai_approve_fail": "Ошибка подтверждения: {error}",
        "ai_reject_ok": "Отклонено {id} (кем {approver})",
        "ai_reject_fail": "Ошибка отклонения: {error}",
        "ai_approve_usage": "Использование: /ai approve <id>",
        "ai_reject_usage": "Использование: /ai reject <id>",
    },
}


def _ai_t(key: str, lang: str | None, **kwargs: Any) -> str:
    """Translate ``key`` using advisor translations, falling back to ``t``.

    The advisor keeps its own small table so that Phase 2 does not need to
    pollute the core bot's TRANSLATIONS dict before handlers are registered.
    The fallback ensures a key that is missing in ``AI_TRANSLATIONS`` still
    resolves via the main i18n system when the latter is populated with the
    same keys.
    """
    effective = lang if lang in ("en", "ru") else "en"
    table = AI_TRANSLATIONS.get(effective, AI_TRANSLATIONS["en"])
    template = table.get(key)
    if template is None:
        # Fallback to the global bot translations (so a Phase 2 that merges
        # the tables still works).
        return t(key, effective, **kwargs)
    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template


class AgentTelegramAdapter:
    """Telegram-facing facade over :class:`AgentCore` + approval.

    * Read-only paths (``/ai status`` etc.) go via ``AgentCore`` /
      ``AgentTools`` and never mutate config.
    * Human-gated paths (``/ai approve`` / ``/ai reject``) go via the
      explicit ``RecommendationApprovalService`` with allowlist + audit.
      AI output can never invoke them.

    Every public method is ``async`` and returns a *localized* string ready
    to be sent via :meth:`TelegramClient.send_message`. No Telegram-specific
    logic lives in :class:`AgentCore` — this adapter is the only place where
    language selection happens.

    Usage (Phase 2 wiring sketch)
    --------------------------------

    .. code-block:: python

        adapter = AgentTelegramAdapter(core, tools, approval_service)
        # inside TelegramBot.handle_update:
        if text.startswith("/ai"):
            lang = await lang_getter(user_id)
            reply = await adapter.dispatch(text, lang=lang, approver=str(user_id))
            await self._safe_send(chat_id, reply)
    """

    def __init__(
        self,
        core: AgentCore | None,
        tools: AgentTools | None = None,
        approval_service: Any | None = None,
    ) -> None:
        self._core = core
        self._tools = tools
        self._approval = approval_service

    async def dispatch(self, text: str, *, lang: str | None = None, approver: str | None = None) -> str:
        """Route ``/ai*`` text to the appropriate sub-handler.

        ``text`` is expected to be the full message (e.g. ``"/ai status"``).
        Unknown subcommands fall back to :meth:`help`.
        """
        effective = lang if lang in ("en", "ru") else "en"
        if self._core is None and self._tools is None:
            return _ai_t("ai_not_configured", effective)

        parts = text.strip().split()
        if not parts or parts[0].lower() != "/ai":
            return _ai_t("ai_help", effective)

        sub = parts[1].lower() if len(parts) > 1 else ""
        # Legacy: bare /ai is help
        if sub == "":
            return await self.help(lang=effective)
        if sub == "status":
            return await self.status(lang=effective)
        if sub == "report":
            return await self.report(lang=effective)
        if sub == "recommendations":
            return await self.recommendations(lang=effective)
        if sub == "memory":
            return await self.memory(lang=effective)
        if sub == "balance":
            return await self.balance(lang=effective)
        if sub == "approve":
            rec_id = parts[2].strip() if len(parts) > 2 else ""
            if not rec_id:
                return _ai_t("ai_approve_usage", effective)
            return await self.approve(rec_id, lang=effective, approver=approver)
        if sub == "reject":
            rec_id = parts[2].strip() if len(parts) > 2 else ""
            if not rec_id:
                return _ai_t("ai_reject_usage", effective)
            return await self.reject(rec_id, lang=effective, approver=approver)

        return _ai_t("ai_unknown_subcommand", effective)

    async def help(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        return _ai_t("ai_help", effective)

    async def status(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._core is None:
            return _ai_t("ai_not_configured", effective)
        try:
            st = await self._core.status(language=effective)
        except Exception:
            return _ai_t("ai_not_configured", effective)

        lines: list[str] = [
            _ai_t("ai_status_title", effective),
            _ai_t("ai_status_trades", effective, count=st.get("recent_trades", 0)),
            _ai_t("ai_status_experiences", effective, count=st.get("experiences", 0)),
            _ai_t("ai_status_lessons", effective, count=st.get("lessons", 0)),
            _ai_t("ai_status_knowledge", effective, count=st.get("knowledge_docs", 0)),
            _ai_t("ai_status_recommendations", effective, count=len(st.get("risk", {}).get("recommendations_pending", [])) if isinstance(st.get("risk"), dict) else 0),
        ]
        # Also show pending recommendations count via direct tool query if available
        if self._tools is not None:
            try:
                recs = await self._tools.get_previous_recommendations(limit=20)
                pending = sum(1 for r in recs if str(r.get("status", "")).lower() == "pending")
                lines[-1] = _ai_t("ai_status_recommendations", effective, count=pending)
            except Exception:
                pass
        return "\n".join(lines)

    async def report(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._core is None:
            return _ai_t("ai_not_configured", effective)
        try:
            resp = await self._core.handle(AgentRequest(query="report", language=effective))
        except Exception as exc:  # noqa: BLE001 - telegram must never propagate exceptions
            return f"{_ai_t('ai_report_no_action', effective, reason=str(exc)[:120])}"
        if resp.is_no_action or resp.reflection is None or resp.reflection.is_no_action:
            reason = resp.reflection.reason if resp.reflection else "no data"
            return _ai_t("ai_report_no_action", effective, reason=reason)
        obs = resp.reflection.observation
        if obs is None:
            return _ai_t("ai_report_no_action", effective, reason=resp.reflection.reason if resp.reflection else "empty")
        return _ai_t(
            "ai_report_insight",
            effective,
            confidence=obs.confidence,
            happened=obs.what_happened[:120],
            expected=obs.what_expected[:120],
            differed=obs.what_differed[:180],
            pattern=(obs.possible_pattern or "-")[:180],
        )

    async def recommendations(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._tools is None:
            return _ai_t("ai_not_configured", effective)
        try:
            recs = await self._tools.get_previous_recommendations(limit=10)
        except Exception:
            return _ai_t("ai_recommendations_empty", effective)
        pending = [r for r in recs if str(r.get("status", "")).lower() in ("pending", "draft")]
        if not pending:
            return _ai_t("ai_recommendations_empty", effective)
        lines: list[str] = [_ai_t("ai_recommendations_header", effective, count=len(pending))]
        for rec in pending[:5]:
            lines.append(
                _ai_t(
                    "ai_recommendations_line",
                    effective,
                    parameter=rec.get("parameter", "?"),
                    old=rec.get("old_value") or rec.get("current_value") or "-",
                    proposed=rec.get("proposed_value", "?"),
                    reason=(rec.get("reason", "") or "")[:80],
                    confidence=float(rec.get("confidence", 0.5)),
                )
            )
        return "\n".join(lines)

    async def memory(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._tools is None:
            return _ai_t("ai_not_configured", effective)
        try:
            mem = await self._tools.get_memory(limit=5)
        except Exception:
            return _ai_t("ai_memory_empty", effective)
        exps = mem.get("experiences", []) if isinstance(mem, dict) else []
        les = mem.get("lessons", []) if isinstance(mem, dict) else []
        if not exps and not les:
            return _ai_t("ai_memory_empty", effective)
        lines: list[str] = [_ai_t("ai_memory_header", effective, count=len(exps) + len(les))]
        for exp in exps[:3]:
            lines.append(
                _ai_t(
                    "ai_memory_experience",
                    effective,
                    id=str(exp.get("id", "?"))[:10],
                    situation=str(exp.get("situation", ""))[:40],
                    observation=str(exp.get("observation", ""))[:40],
                    confidence=float(exp.get("confidence", 0.5)),
                )
            )
        for lesson in les[:2]:
            lines.append(
                _ai_t(
                    "ai_memory_lesson",
                    effective,
                    id=str(lesson.get("id", "?"))[:10],
                    title=str(lesson.get("title", ""))[:30],
                    content=str(lesson.get("content", ""))[:40],
                    confidence=float(lesson.get("confidence", 0.5)),
                )
            )
        return "\n".join(lines)

    async def balance(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._tools is None:
            return _ai_t("ai_not_configured", effective)
        try:
            snaps = await self._tools.get_balances()
        except Exception:
            return _ai_t("ai_balance_empty", effective)
        if not snaps:
            return _ai_t("ai_balance_empty", effective)
        lines: list[str] = [_ai_t("ai_balance_title", effective)]
        for venue, data in snaps.items():
            bals = data.get("balances", []) if isinstance(data, dict) else []
            assets = ", ".join(f"{b.get('asset')} {b.get('free')}" for b in bals[:5]) if bals else "-"
            lines.append(_ai_t("ai_balance_line", effective, venue=venue, assets=assets))
        return "\n".join(lines)

    async def approve(self, rec_id: str, *, lang: str | None = None, approver: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._approval is None:
            return _ai_t("ai_not_configured", effective)
        if not approver:
            # Must be human — fail-closed if approver missing
            return _ai_t("ai_approve_fail", effective, error="approver required")
        try:
            # Redact rec_id (no secret, but still defensive)
            from app.exchanges.sanitize import redact_secrets as _redact

            rec_id = _redact(rec_id).strip()
            result = await self._approval.approve(rec_id, approver=str(approver), reason=f"telegram /ai approve by {approver}")
            return _ai_t(
                "ai_approve_ok",
                effective,
                parameter=result.parameter,
                old=result.old_value or result.current_value or "-",
                proposed=result.proposed_value,
                approver=approver,
            )
        except Exception as exc:  # noqa: BLE001 - telegram must not leak internal details
            from app.exchanges.sanitize import redact_secrets as _redact

            safe = _redact(str(exc))[:200]
            return _ai_t("ai_approve_fail", effective, error=safe)

    async def reject(self, rec_id: str, *, lang: str | None = None, approver: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._approval is None:
            return _ai_t("ai_not_configured", effective)
        if not approver:
            return _ai_t("ai_reject_fail", effective, error="approver required")
        try:
            from app.exchanges.sanitize import redact_secrets as _redact

            rec_id = _redact(rec_id).strip()
            await self._approval.reject(rec_id, approver=str(approver), reason=f"telegram /ai reject by {approver}")
            return _ai_t("ai_reject_ok", effective, id=rec_id[:12], approver=approver)
        except Exception as exc:  # noqa: BLE001
            from app.exchanges.sanitize import redact_secrets as _redact

            safe = _redact(str(exc))[:200]
            return _ai_t("ai_reject_fail", effective, error=safe)
