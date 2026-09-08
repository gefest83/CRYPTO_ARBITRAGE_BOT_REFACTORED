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

import asyncio
import hashlib
import time as _time
from typing import Any

from app.agent.core import AgentCore, AgentRequest
from app.agent.tools import AgentTools
from app.config.logging_config import get_logger
from app.telegram.i18n import t

__all__ = ["AgentTelegramAdapter", "AI_COMMANDS"]

logger = get_logger("agent.telegram")

# Separate NL rate/budget limits (independent from /ai report budget)
try:
    from app.agent.providers.openrouter import RateLimiter as _RateLimiter  # type: ignore[import-not-found]

    _NL_RATE_LIMITER: Any = _RateLimiter(per_minute=20, per_hour=100, per_day=300)
except Exception:
    _NL_RATE_LIMITER = None


AI_COMMANDS: tuple[str, ...] = (
    "/ai",
    "/ai status",
    "/ai report",
    "/ai recommendations",
    "/ai memory",
    "/ai balance",
    "/ai approve",
    "/ai reject",
    "/ai measurements",
    "/ai feedback",
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
            "/ai measurements [id] - measured outcomes of approved changes (or one rec)\n"
            "/ai feedback <id> <useful|wrong|ignore|approve|reject> [comment] - operator feedback\n"
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
        "ai_measurements_empty": "No measured outcomes yet.",
        "ai_measurements_header": "Measured outcomes ({count}):",
        "ai_measurements_line": "  {parameter}: {outcome} ({metric} {before} → {after}, n={n_before}/{n_after}) [rec:{rec_id}]",
        "ai_measurements_detail": (
            "Measurement for {rec_id}:\n"
            "  parameter: {parameter} ({old} → {new})\n"
            "  outcome: {outcome} — {reason}\n"
            "  metric: {metric} {before} → {after} (Δ{delta}, n={n_before}/{n_after}, conf {confidence:.2f})\n"
            "  feedback: {feedback} | lesson: {lesson}"
        ),
        "ai_measurements_missing": "No measurement for {rec_id} yet.",
        "ai_feedback_ok": "Feedback recorded: {kind} on {rec_id} (by {approver})",
        "ai_feedback_fail": "Feedback failed: {error}",
        "ai_feedback_usage": "Usage: /ai feedback <recommendation_id> <useful|wrong|ignore|approve|reject> [comment]",
        "ai_nl_trades_title": "Recent trades ({count}):",
        "ai_nl_trades_empty": "No trades found. The bot has not recorded any trades yet.",
        "ai_nl_trades_line": "  {strategy} {route} [{status}] net {net_profit} ({created})",
        "ai_nl_trades_more": "  ... and {remaining} more (see CLI for the full list)",
        "ai_nl_trade_stats": (
            "Trade statistics (last {total}):\n"
            "  completed: {completed}\n"
            "  failed: {failed}\n"
            "  manual review: {manual_review}\n"
            "  total PnL: {pnl}\n"
            "  avg net: {bps} bps\n"
            "  win rate: {win}%"
        ),
        "ai_nl_scan_stats": (
            "Scan statistics:\n"
            "  tickers: {tickers}\n"
            "  order books: {books}\n"
            "  venues with data: {venues}"
        ),
        "ai_nl_opportunities_found": "Current opportunities (read-only, no execution):",
        "ai_nl_opportunities_none": (
            "No profitable routes right now. Scanner ran on current market data "
            "but nothing is above the configured profitability threshold."
        ),
        "ai_nl_opportunities_stale": (
            "Scanner has incomplete/stale market data — opportunity check is unreliable right now."
        ),
        "ai_nl_opportunities_line": "  {kind} {desc} net {bps} bps",
        "ai_nl_exchange_status": "Exchange status:",
        "ai_nl_exchange_line": "  {venue}: {status} (keys {creds})",
        "ai_nl_exchange_unavailable": "  {venue}: unavailable — {reason}",
        "ai_nl_bot_status": "Bot status:",
        "ai_nl_risk_title": "Risk state:",
        "ai_nl_params_title": "Current parameters:",
        "ai_nl_journal_title": "Recent journal ({count}):",
        "ai_nl_journal_empty": "Journal is empty.",
        "ai_nl_journal_line": "  {ts} {action}: {message}",
        "ai_nl_why_title": "Why is the bot not trading — diagnostics (read-only):",
        "ai_nl_why_no_blocker": "No clear blocker found in the available data.",
        "ai_nl_help_full": (
            "AI Advisor — I answer read-only questions (English / Русский):\n"
            "- balances: 'show me balances', 'покажи балансы'\n"
            "- trades: 'show recent trades', 'были ли сегодня сделки'\n"
            "- scan statistics: 'how many opportunities did the scanner find', 'что показал последний скан'\n"
            "- opportunities/routes: 'are there any arbitrage opportunities', 'есть ли арбитражные возможности'\n"
            "- exchange status: 'which exchanges are online', 'какие биржи онлайн'\n"
            "- bot status: 'what is the bot status', 'что сейчас происходит'\n"
            "- risk state: 'what is the current risk state', 'какое состояние риска'\n"
            "- parameters: 'show current parameters', 'какие сейчас параметры'\n"
            "- journal: 'show journal', 'покажи журнал'\n"
            "- memory: 'show agent memory', 'покажи память агента'\n"
            "- recommendations: 'show recommendations', 'покажи рекомендации'\n"
            "- why not trading: 'why is the bot not trading', 'почему бот не торгует'\n"
            "- how the bot works: 'how does the bot trade', 'как торгует бот'\n"
            "\n"
            "Trading, withdrawals, configuration changes and approvals stay protected — "
            "use the explicit commands (/ai approve <id>, /pause, /resume, CLI) for those."
        ),
        "ai_nl_unknown": (
            "I can answer read-only questions about balances, trades, scan statistics, "
            "opportunities/routes, exchange status, bot status, risk state, parameters, "
            "journal, memory and recommendations. Try 'what can you do?' for examples."
        ),
        "ai_nl_privileged_refused": (
            "Refused: that action is privileged and cannot be done via natural language. "
            "Use the explicit workflow (e.g. /ai approve <id>, /pause, CLI) as an authorized operator."
        ),
        "ai_nl_balance_unavailable": "  {venue}: unavailable — balance request failed",
        "ai_nl_balance_more": "  ... and {remaining} more balances not shown",
        "ai_nl_trades_today_title": "Trades today ({count}):",
        "ai_nl_trades_empty_today": "No trades today. The bot has not recorded any trades on the current calendar date.",
        "ai_nl_trades_last_hour_title": "Trades in the last hour ({count}):",
        "ai_nl_trades_empty_last_hour": "No trades in the last hour.",
        "ai_nl_threshold_line": "(governing threshold: {value} bps — {source}, active strategy: {strategy})",
        "ai_nl_risk_gate_line": "(execution is additionally gated by risk min_net_profit_bps: {value} bps)",
        "ai_nl_scan_took": "(scan took {ms} ms)",
        "ai_nl_bot_operation_title": "How this bot works (read-only explanation):",
        "ai_nl_insufficient_data": "Available data is insufficient to establish a reason — no reason is invented.",
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
            "/ai measurements [id] - измеренные исходы применённых изменений (или одна)\n"
            "/ai feedback <id> <useful|wrong|ignore|approve|reject> [комментарий] - отзыв оператора\n"
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
        "ai_measurements_empty": "Измеренных исходов пока нет.",
        "ai_measurements_header": "Измеренные исходы ({count}):",
        "ai_measurements_line": "  {parameter}: {outcome} ({metric} {before} → {after}, n={n_before}/{n_after}) [рек:{rec_id}]",
        "ai_measurements_detail": (
            "Измерение для {rec_id}:\n"
            "  параметр: {parameter} ({old} → {new})\n"
            "  исход: {outcome} — {reason}\n"
            "  метрика: {metric} {before} → {after} (Δ{delta}, n={n_before}/{n_after}, увер {confidence:.2f})\n"
            "  отзыв: {feedback} | урок: {lesson}"
        ),
        "ai_measurements_missing": "Измерения для {rec_id} пока нет.",
        "ai_feedback_ok": "Отзыв записан: {kind} по {rec_id} (кем {approver})",
        "ai_feedback_fail": "Ошибка отзыва: {error}",
        "ai_feedback_usage": "Использование: /ai feedback <id> <useful|wrong|ignore|approve|reject> [комментарий]",
        "ai_nl_trades_title": "Последние сделки ({count}):",
        "ai_nl_trades_empty": "Сделок не найдено. Бот пока не записал ни одной сделки.",
        "ai_nl_trades_line": "  {strategy} {route} [{status}] net {net_profit} ({created})",
        "ai_nl_trades_more": "  ... и ещё {remaining} (полный список в CLI)",
        "ai_nl_trade_stats": (
            "Статистика сделок (последние {total}):\n"
            "  завершено: {completed}\n"
            "  неудачно: {failed}\n"
            "  ручная проверка: {manual_review}\n"
            "  суммарный PnL: {pnl}\n"
            "  средний net: {bps} bps\n"
            "  доля успешных: {win}%"
        ),
        "ai_nl_scan_stats": (
            "Статистика сканирования:\n"
            "  тикеры: {tickers}\n"
            "  стаканы: {books}\n"
            "  площадки с данными: {venues}"
        ),
        "ai_nl_opportunities_found": "Текущие возможности (только чтение, без исполнения):",
        "ai_nl_opportunities_none": (
            "Прибыльных маршрутов сейчас нет. Сканер отработал по текущим данным, "
            "но ничего не выше настроенного порога прибыльности."
        ),
        "ai_nl_opportunities_stale": (
            "У сканера неполные/устаревшие рыночные данные — оценка возможностей сейчас ненадёжна."
        ),
        "ai_nl_opportunities_line": "  {kind} {desc} net {bps} bps",
        "ai_nl_exchange_status": "Статус бирж:",
        "ai_nl_exchange_line": "  {venue}: {status} (ключи {creds})",
        "ai_nl_exchange_unavailable": "  {venue}: недоступна — {reason}",
        "ai_nl_bot_status": "Статус бота:",
        "ai_nl_risk_title": "Состояние риска:",
        "ai_nl_params_title": "Текущие параметры:",
        "ai_nl_journal_title": "Журнал (последние {count}):",
        "ai_nl_journal_empty": "Журнал пуст.",
        "ai_nl_journal_line": "  {ts} {action}: {message}",
        "ai_nl_why_title": "Почему бот не торгует — диагностика (только чтение):",
        "ai_nl_why_no_blocker": "Явной причины блокировки по доступным данным не найдено.",
        "ai_nl_help_full": (
            "AI-советник — отвечаю на вопросы только для чтения (English / Русский):\n"
            "- балансы: 'покажи балансы', 'show me balances'\n"
            "- сделки: 'покажи последние сделки', 'were there any trades today'\n"
            "- статистика сканирования: 'что показал последний скан', 'how many opportunities did the scanner find'\n"
            "- возможности/маршруты: 'есть ли арбитражные возможности', 'are there any arbitrage opportunities'\n"
            "- статус бирж: 'какие биржи онлайн', 'which exchanges are online'\n"
            "- статус бота: 'что сейчас происходит', 'what is the bot status'\n"
            "- риск: 'какое состояние риска', 'what is the current risk state'\n"
            "- параметры: 'какие сейчас параметры', 'show current parameters'\n"
            "- журнал: 'покажи журнал', 'show journal'\n"
            "- память: 'покажи память агента', 'show agent memory'\n"
            "- рекомендации: 'покажи рекомендации', 'show recommendations'\n"
            "- почему нет сделок: 'почему бот не торгует', 'why is the bot not trading'\n"
            "- как работает бот: 'как торгует бот', 'how does the bot trade'\n"
            "\n"
            "Торговля, выводы, изменение конфигурации и подтверждения остаются защищёнными — "
            "используйте явные команды (/ai approve <id>, /pause, /resume, CLI)."
        ),
        "ai_nl_unknown": (
            "Могу отвечать на вопросы только для чтения: балансы, сделки, статистика сканирования, "
            "возможности/маршруты, статус бирж и бота, риск, параметры, журнал, память и рекомендации. "
            "Спросите 'что ты умеешь?' для примеров."
        ),
        "ai_nl_privileged_refused": (
            "Отклонено: это привилегированное действие нельзя выполнить естественным языком. "
            "Используйте явный процесс (/ai approve <id>, /pause, CLI) как авторизованный оператор."
        ),
        "ai_nl_balance_unavailable": "  {venue}: недоступна — запрос баланса не удался",
        "ai_nl_balance_more": "  ... и ещё {remaining} балансов не показано",
        "ai_nl_trades_today_title": "Сделки сегодня ({count}):",
        "ai_nl_trades_empty_today": "Сегодня сделок не было. На текущую календарную дату бот сделок не записывал.",
        "ai_nl_trades_last_hour_title": "Сделки за последний час ({count}):",
        "ai_nl_trades_empty_last_hour": "За последний час сделок не было.",
        "ai_nl_threshold_line": "(действующий порог: {value} bps — {source}, активная стратегия: {strategy})",
        "ai_nl_risk_gate_line": "(исполнение дополнительно ограничено риск-лимитом min_net_profit_bps: {value} bps)",
        "ai_nl_scan_took": "(скан занял {ms} мс)",
        "ai_nl_bot_operation_title": "Как работает этот бот (объяснение, только чтение):",
        "ai_nl_insufficient_data": "Доступных данных недостаточно для установления причины — причина не выдумывается.",
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


def _finalize_telegram(text: str) -> str:
    """Bound and sanitize Telegram output — no secret leakage, never exceeds 3000 chars.

    All adapter outputs go through this so that even a malicious DB value
    cannot produce an unbounded or secret-leaking Telegram message.
    """
    from app.agent.providers.base import filter_secrets_from_text, sanitize_untrusted_text
    from app.exchanges.sanitize import redact_secrets as _redact

    # Layered defense: redact, secret filter, injection sanitize, then bound
    cleaned = _redact(text)
    cleaned = filter_secrets_from_text(cleaned)
    # Note: we do not fully sanitize the whole translated template (it is trusted),
    # but we bound the final length strictly.
    if len(cleaned) > 3000:
        cleaned = cleaned[:2980] + "... (truncated)"
    return cleaned


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
        learning: Any | None = None,
    ) -> None:
        self._core = core
        self._tools = tools
        self._approval = approval_service
        self._learning = learning
        # Per-adapter NL rate limiter instance (separate from /ai report budget)
        try:
            from app.agent.providers.openrouter import RateLimiter as _RL  # type: ignore

            self._nl_limiter: Any = _RL(per_minute=20, per_hour=100, per_day=300)
        except Exception:
            self._nl_limiter = None

    async def dispatch(self, text: str, *, lang: str | None = None, approver: str | None = None) -> str:
        """Route ``/ai*`` text to the appropriate sub-handler.

        ``text`` is expected to be the full message (e.g. ``"/ai status"``).
        Unknown subcommands fall back to :meth:`help`.
        """
        effective = lang if lang in ("en", "ru") else "en"
        if self._core is None and self._tools is None:
            return _finalize_telegram(_ai_t("ai_not_configured", effective))

        parts = text.strip().split()
        if not parts or parts[0].lower() != "/ai":
            return _finalize_telegram(_ai_t("ai_help", effective))

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
                return _finalize_telegram(_ai_t("ai_approve_usage", effective))
            return await self.approve(rec_id, lang=effective, approver=approver)
        if sub == "reject":
            rec_id = parts[2].strip() if len(parts) > 2 else ""
            if not rec_id:
                return _finalize_telegram(_ai_t("ai_reject_usage", effective))
            return await self.reject(rec_id, lang=effective, approver=approver)
        if sub == "measurements":
            rec_id = parts[2].strip() if len(parts) > 2 else ""
            return await self.measurements(rec_id or None, lang=effective)
        if sub == "feedback":
            rec_id = parts[2].strip() if len(parts) > 2 else ""
            kind = parts[3].strip().lower() if len(parts) > 3 else ""
            comment = " ".join(parts[4:]).strip() if len(parts) > 4 else ""
            if not rec_id or not kind:
                return _finalize_telegram(_ai_t("ai_feedback_usage", effective))
            return await self.feedback(rec_id, kind, comment, lang=effective, approver=approver)

        return _finalize_telegram(_ai_t("ai_unknown_subcommand", effective))

    async def help(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        return _finalize_telegram(_ai_t("ai_help", effective))

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
        return _finalize_telegram("\n".join(lines))

    async def report(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._core is None:
            return _ai_t("ai_not_configured", effective)
        try:
            resp = await self._core.handle(AgentRequest(query="report", language=effective))
        except Exception as exc:  # noqa: BLE001 - telegram must never propagate exceptions
            from app.exchanges.sanitize import redact_secrets as _redact
            from app.agent.providers.base import filter_secrets_from_text as _filt, sanitize_untrusted_text as _san

            safe = _san(_filt(_redact(str(exc)[:200])))
            return _finalize_telegram(f"{_ai_t('ai_report_no_action', effective, reason=safe[:120])}")
        # Use structured analysis if available for richer report (Phase 3D)
        # Graceful LLM failure: if analysis is None, fall back to reflection
        if resp.is_no_action or resp.reflection is None or resp.reflection.is_no_action:
            reason = resp.reflection.reason if resp.reflection else "no data"
            # Sanitize reason (untrusted)
            from app.agent.providers.base import sanitize_untrusted_text as _san

            safe_reason = _san(reason)[:200]
            return _finalize_telegram(_ai_t("ai_report_no_action", effective, reason=safe_reason))
        # Prefer analysis for evidence_count/action when present
        analysis = getattr(resp, "analysis", None)
        obs = resp.reflection.observation
        if obs is None:
            return _finalize_telegram(_ai_t("ai_report_no_action", effective, reason=resp.reflection.reason if resp.reflection else "empty"))
        # Build structured report with Phase 3D fields
        from app.agent.providers.base import sanitize_untrusted_text as _san, filter_secrets_from_text as _filt
        from app.exchanges.sanitize import redact_secrets as _redact

        def _safe(s: str, n: int) -> str:
            return _san(_filt(_redact(str(s))))[:n]

        # Include evidence_count, probable_cause, recurring_pattern for usefulness
        evidence_cnt = getattr(obs, "evidence_count", 0) or getattr(resp, "analysis", None) and getattr(resp.analysis, "evidence_count", 0) or 0
        action = getattr(analysis, "action", resp.reflection.action) if analysis else resp.reflection.action
        probable = getattr(obs, "probable_cause", None) or "-"
        recurring = getattr(obs, "recurring_pattern", None) or obs.possible_pattern or "-"
        # Bounded, sanitized
        result = _ai_t(
            "ai_report_insight",
            effective,
            confidence=obs.confidence,
            happened=_safe(obs.what_happened, 120),
            expected=_safe(obs.what_expected, 120),
            differed=_safe(obs.what_differed, 180),
            pattern=_safe(recurring, 180),
        )
        # Append structured footer (bounded, never exceeds telegram limit)
        footer = f"\n  evidence: {evidence_cnt} | action: {action} | cause: {_safe(probable, 80)}"
        combined = result + footer
        # Hard bound: ensure telegram output never exceeds 3000 chars (Phase 3D)
        if len(combined) > 3000:
            combined = combined[:2980] + "... (truncated)"
        return _finalize_telegram(combined)

    async def recommendations(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._tools is None:
            return _finalize_telegram(_ai_t("ai_not_configured", effective))
        try:
            recs = await self._tools.get_previous_recommendations(limit=10)
        except Exception:
            return _finalize_telegram(_ai_t("ai_recommendations_empty", effective))
        pending = [r for r in recs if str(r.get("status", "")).lower() in ("pending", "draft", "reviewed")]
        if not pending:
            return _finalize_telegram(_ai_t("ai_recommendations_empty", effective))
        lines: list[str] = [_ai_t("ai_recommendations_header", effective, count=len(pending))]
        for rec in pending[:5]:
            # Phase 7: surface lifecycle status + evidence sample size (n).
            evidence = rec.get("evidence") or ()
            status = str(rec.get("status", "pending")).lower()
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
                + f" [n={len(evidence)} {status}]"
            )
        return _finalize_telegram("\n".join(lines))

    async def memory(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._tools is None:
            return _finalize_telegram(_ai_t("ai_not_configured", effective))
        try:
            mem = await self._tools.get_memory(limit=5)
        except Exception:
            return _finalize_telegram(_ai_t("ai_memory_empty", effective))
        exps = mem.get("experiences", []) if isinstance(mem, dict) else []
        les = mem.get("lessons", []) if isinstance(mem, dict) else []
        if not exps and not les:
            return _finalize_telegram(_ai_t("ai_memory_empty", effective))
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
        return _finalize_telegram("\n".join(lines))

    def _clean_nl(self, text: str) -> str:
        """Redact secrets without the 3000-char cap (bot chunks long replies)."""
        from app.agent.providers.base import filter_secrets_from_text
        from app.exchanges.sanitize import redact_secrets as _redact

        return filter_secrets_from_text(_redact(text))

    def _services(self) -> Any | None:
        tools = self._tools
        if tools is None:
            return None
        return getattr(tools, "_services", None)

    def _enabled_venues(self) -> list[str]:
        try:
            svc = self._services()
            if svc is not None and getattr(svc, "manager", None) is not None:
                return list(svc.manager.enabled_ids())
        except Exception:
            pass
        return []

    async def balance(
        self, *, lang: str | None = None, venue: str | None = None, asset: str | None = None
    ) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._tools is None:
            return _finalize_telegram(_ai_t("ai_not_configured", effective))
        try:
            snaps = await self._tools.get_balances()
        except Exception:
            return _finalize_telegram(_ai_t("ai_balance_empty", effective))
        return self._clean_nl(self._format_balances(snaps, effective, venue=venue, asset=asset))

    def _format_balances(
        self, snaps: dict[str, Any], effective: str, *, venue: str | None = None, asset: str | None = None
    ) -> str:
        """Format *all* non-zero balances with venue/asset/free/used.

        No artificial ``[:5]`` truncation. Partial venue failures (enabled
        venue missing from ``snaps``) are reported explicitly instead of
        being hidden. Very long outputs carry an explicit "+N more" note so
        the Telegram chunking layer can split them safely.
        """
        from decimal import Decimal

        venue_f = (venue or "").strip().lower() or None
        asset_f = (asset or "").strip().upper() or None
        lines: list[str] = [_ai_t("ai_balance_title", effective)]
        shown = 0
        hidden = 0
        # Per-venue cap keeps single replies usable; overflow is counted,
        # never silently dropped.
        _PER_VENUE_SOFT_CAP = 60
        venues = sorted(snaps.keys()) if isinstance(snaps, dict) else []
        if venue_f:
            venues = [v for v in venues if v.strip().lower() == venue_f]
            if not venues and isinstance(snaps, dict) and snaps:
                # Requested venue returned nothing — still report it.
                lines.append(_ai_t("ai_nl_balance_unavailable", effective, venue=venue_f))
                return "\n".join(lines)
        if not venues and not snaps:
            return _ai_t("ai_balance_empty", effective)
        for v in venues:
            data = snaps.get(v, {}) if isinstance(snaps, dict) else {}
            bals = data.get("balances", []) if isinstance(data, dict) else []
            rows: list[str] = []
            for b in bals:
                try:
                    a = str(b.get("asset", "?")).upper()
                    if asset_f and a != asset_f:
                        continue
                    free = str(b.get("free", "0"))
                    used = str(b.get("used", "0"))
                    # Skip dust / zero balances (both free and used are zero).
                    try:
                        if Decimal(str(free)) == 0 and Decimal(str(used)) == 0:
                            continue
                    except Exception:
                        pass
                    rows.append(f"{a}: free {free} / used {used}")
                except Exception:
                    continue
            if not rows:
                lines.append(f"{v}: -")
                continue
            lines.append(f"{v}:")
            if len(rows) > _PER_VENUE_SOFT_CAP:
                shown_rows = rows[:_PER_VENUE_SOFT_CAP]
                hidden += len(rows) - _PER_VENUE_SOFT_CAP
            else:
                shown_rows = rows
            for r in shown_rows:
                lines.append(f"  {r}")
                shown += 1
        # Partial failures: enabled venues with no snapshot are unavailable.
        try:
            enabled = {e.strip().lower() for e in self._enabled_venues()}
            present = {str(v).strip().lower() for v in (snaps.keys() if isinstance(snaps, dict) else [])}
            if venue_f:
                enabled = {e for e in enabled if e == venue_f}
            for missing in sorted(enabled - present):
                lines.append(_ai_t("ai_nl_balance_unavailable", effective, venue=missing))
        except Exception:
            pass
        if hidden:
            lines.append(_ai_t("ai_nl_balance_more", effective, remaining=hidden))
        if shown == 0 and hidden == 0 and len(lines) <= 1:
            return _ai_t("ai_balance_empty", effective)
        return "\n".join(lines)

    # ------------------------------------------------------------- NL intents
    async def _active_strategy(self) -> str:
        """Active runtime strategy (``triangle`` default, mirrors AutoTrader)."""
        svc = self._services()
        if svc is None:
            return "triangle"
        get_active = getattr(svc, "get_active_strategy", None)
        if get_active is not None:
            try:
                active = await get_active()
                if active in ("triangle", "transfer"):
                    return active
            except Exception:
                pass
        # Fallback: status() view carries the same field.
        try:
            st = await svc.status()
            if isinstance(st, dict) and st.get("active_strategy") in ("triangle", "transfer"):
                return str(st.get("active_strategy"))
        except Exception:
            pass
        return "triangle"

    async def _threshold_context(self) -> dict[str, str]:
        """Thresholds actually governing the active runtime path (read-only).

        Root cause of the historical "5 bps vs 10 bps" confusion: the
        *scanner* filters triangles at ``arbitrage.triangle_min_net_bps``
        (default 5) while the *risk engine* additionally gates every
        execution at ``risk.min_net_profit_bps`` (default 10) via
        ``MinNetProfitRule``. Transfer plans filter at
        ``transfer.min_net_profit_bps`` (default 50). This helper reports
        the strategy threshold that governs *opportunity detection* plus
        the risk gate that governs *execution* — never inventing values.
        """
        strategy = await self._active_strategy()
        strat_value: str | None = None
        strat_source = ""
        risk_value: str | None = None
        svc = self._services()
        settings = getattr(svc, "settings", None) if svc is not None else None
        try:
            if strategy == "transfer":
                transfer = getattr(settings, "transfer", None)
                if transfer is not None and getattr(transfer, "min_net_profit_bps", None) is not None:
                    strat_value = str(transfer.min_net_profit_bps)
                    strat_source = "transfer.min_net_profit_bps"
            else:
                arbitrage = getattr(settings, "arbitrage", None)
                if arbitrage is not None and getattr(arbitrage, "triangle_min_net_bps", None) is not None:
                    strat_value = str(arbitrage.triangle_min_net_bps)
                    strat_source = "arbitrage.triangle_min_net_bps"
        except Exception:
            pass
        try:
            risk = getattr(settings, "risk", None)
            if risk is not None and getattr(risk, "min_net_profit_bps", None) is not None:
                risk_value = str(risk.min_net_profit_bps)
        except Exception:
            pass
        return {
            "strategy": strategy,
            "threshold": strat_value or "?",
            "source": strat_source or "unknown",
            "risk_threshold": risk_value or "?",
        }

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    async def _deterministic_dispatch(self, intent: str, entities: dict, effective: str) -> str | None:
        """Try deterministic fast-path. Returns string if handled, else None."""
        if intent == "BALANCE_QUERY":
            return await self.balance(lang=effective, venue=entities.get("venue"), asset=entities.get("asset"))
        if intent == "TRADES_QUERY":
            return await self.nl_trades(lang=effective, period=entities.get("period"))
        if intent == "TRADE_STATS_QUERY":
            return await self.nl_trade_stats(lang=effective)
        if intent == "SCAN_STATS_QUERY":
            return await self.nl_scan_stats(lang=effective)
        if intent == "OPPORTUNITIES_QUERY":
            return await self.nl_opportunities(lang=effective)
        if intent == "EXCHANGE_STATUS_QUERY":
            return await self.nl_exchange_status(lang=effective)
        if intent == "BOT_STATUS_QUERY":
            return await self.nl_bot_status(lang=effective)
        if intent == "BOT_OPERATION_QUERY":
            return await self.nl_bot_operation(lang=effective)
        if intent == "RISK_QUERY":
            return await self.nl_risk(lang=effective)
        if intent == "PARAMETERS_QUERY":
            return await self.nl_parameters(lang=effective)
        if intent == "MEMORY_QUERY":
            return await self.memory(lang=effective)
        if intent == "JOURNAL_QUERY":
            return await self.nl_journal(lang=effective)
        if intent == "RECOMMENDATIONS_QUERY":
            return await self.recommendations(lang=effective)
        if intent == "WHY_NOT_TRADING_QUERY":
            return await self.nl_why_not_trading(lang=effective)
        if intent == "AI_HELP":
            return self._clean_nl(_ai_t("ai_nl_help_full", effective))
        return None

    def _nl_system_prompt(self, effective: str, threshold_ctx: dict[str, str]) -> str:
        if effective == "ru":
            return (
                "Ты — AI-советник крипто-арбитражного бота. Отвечай только на основе реальных данных из инструментов. "
                "Не выдумывай пороги/параметры — используй только live-контекст. "
                f"Текущий порог: {threshold_ctx.get('threshold','?')} bps ({threshold_ctx.get('source','?')}), "
                f"риск-лимит: {threshold_ctx.get('risk_threshold','?')} bps, стратегия: {threshold_ctx.get('strategy','?')}. "
                "Используй доступные инструменты (только чтение) для ответа. Отвечай на языке оператора (русский)."
            )
        return (
            "You are the AI Advisor for a crypto arbitrage bot. Answer grounded in real tool data only. "
            "Never invent thresholds/parameters — use only the live context provided. "
            f"Current threshold: {threshold_ctx.get('threshold','?')} bps ({threshold_ctx.get('source','?')}), "
            f"risk gate: {threshold_ctx.get('risk_threshold','?')} bps, strategy: {threshold_ctx.get('strategy','?')}. "
            "Use available read-only tools to answer. Respond in the operator's language (English)."
        )

    async def _execute_llm_tools_flow(self, text: str, effective: str) -> str | None:
        """LLM tool-calling flow. Returns final answer or None if should fallback."""
        if self._tools is None:
            return None
        provider = None
        if self._core is not None:
            provider = getattr(self._core, "_llm", None) or getattr(self._core, "llm_provider", None)
        if provider is None or not getattr(provider, "supports_tool_calling", False):
            return None
        limiter = getattr(self, "_nl_limiter", None) or _NL_RATE_LIMITER
        if limiter is not None:
            try:
                limiter.check()
            except Exception:
                return None
            try:
                limiter.record()
            except Exception:
                pass
        try:
            threshold_ctx = await self._threshold_context()
        except Exception:
            threshold_ctx = {"threshold": "?", "source": "unknown", "strategy": "triangle", "risk_threshold": "?"}
        from app.agent.providers.base import filter_secrets_from_text, sanitize_untrusted_text
        from app.agent.tools import ToolAccessBlocked
        safe_user = sanitize_untrusted_text(filter_secrets_from_text(text or ""), max_chars=1000)[:1000]
        system_prompt = self._nl_system_prompt(effective, threshold_ctx)
        safe_system = filter_secrets_from_text(system_prompt)[:1500]
        prompt_hash = self._hash_text(safe_system + "|" + safe_user)
        from app.agent.nl_tool_executor import MAX_NL_TOOL_CALLS, NL_TOOL_DEFINITIONS, NLToolExecutor, validate_tool_call
        from app.agent.providers.base import LLMMessage, LLMRequest
        executor = NLToolExecutor(self._tools)
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=safe_system),
            LLMMessage(role="user", content=safe_user),
        ]
        tool_names_used: list[str] = []
        corrective_retries = 0
        total_calls = 0
        final_content: str | None = None
        while total_calls < MAX_NL_TOOL_CALLS:
            req = LLMRequest(messages=tuple(messages), tools=NL_TOOL_DEFINITIONS, tool_choice="auto")
            resp = await provider.complete(req)
            filtered_content = filter_secrets_from_text(resp.content or "")
            if not resp.tool_calls:
                final_content = filtered_content
                break
            tool_calls = list(resp.tool_calls)[: MAX_NL_TOOL_CALLS - total_calls]
            messages.append(LLMMessage(role="assistant", content=filtered_content, tool_calls=tuple(tool_calls)))
            round_success = False
            for tc in tool_calls:
                if total_calls >= MAX_NL_TOOL_CALLS:
                    break
                try:
                    validated = validate_tool_call(tc.name, tc.arguments)
                except Exception as exc:
                    if corrective_retries < 1:
                        corrective_retries += 1
                        err_msg = sanitize_untrusted_text(filter_secrets_from_text(str(exc)[:300]), max_chars=500)
                        messages.append(LLMMessage(role="tool", content="Tool error: " + err_msg + " -- use valid allowlisted tools only.", tool_call_id=tc.id, name=tc.name))
                        try:
                            logger.warning("ai_nl_tool_blocked", extra={"tool": tc.name, "error": err_msg[:200], "routing_path": "llm_tools"})
                        except Exception:
                            pass
                        round_success = False
                        break
                    else:
                        try:
                            logger.warning("ai_nl_tool_blocked", extra={"tool": tc.name, "routing_path": "llm_tools", "outcome": "fallback"})
                        except Exception:
                            pass
                        raise ToolAccessBlocked("malformed tool call " + tc.name) from exc
                try:
                    result_str = await executor.execute(tc.name, validated)
                except Exception as exc:
                    if corrective_retries < 1 and isinstance(exc, ToolAccessBlocked):
                        corrective_retries += 1
                        messages.append(LLMMessage(role="tool", content="ToolAccessBlocked: " + filter_secrets_from_text(str(exc))[:300], tool_call_id=tc.id, name=tc.name))
                        round_success = False
                        break
                    raise
                messages.append(LLMMessage(role="tool", content=result_str, tool_call_id=tc.id, name=tc.name))
                tool_names_used.append(tc.name)
                total_calls += 1
                round_success = True
            if not round_success and corrective_retries and total_calls == 0:
                continue
            if not round_success:
                break
            if total_calls >= MAX_NL_TOOL_CALLS:
                break
        if final_content is None:
            try:
                final_req = LLMRequest(messages=tuple(messages))
                final_resp = await provider.complete(final_req)
                final_content = filter_secrets_from_text(final_resp.content or "")
            except Exception:
                return None
        if not final_content or len(final_content.strip()) < 5:
            return None
        from app.agent.providers.base import filter_secrets_from_text as _filt, sanitize_untrusted_text as _san
        from app.exchanges.sanitize import redact_secrets as _redact
        cleaned = _redact(final_content)
        cleaned = _filt(cleaned)
        cleaned = _san(cleaned, max_chars=3500)[:3500]
        answer_hash = self._hash_text(cleaned)
        try:
            if self._core is not None and getattr(self._core, "_audit", None) is not None:
                audit = self._core._audit  # type: ignore
                from app.agent.audit import AgentAuditEvent
                ev = AgentAuditEvent(
                    event_type="nl_llm_tools",
                    provider=getattr(provider, "name", "unknown"),
                    model=getattr(provider, "model", None),
                    query=prompt_hash[:500],
                    details={"prompt_hash": prompt_hash, "answer_hash": answer_hash, "tools": tool_names_used[:5], "routing_path": "llm_tools"},
                )
                await audit.log(ev)
        except Exception:
            pass
        self._last_nl_tools = tool_names_used  # type: ignore
        return self._clean_nl(cleaned)

    async def handle_natural_language(
        self, text: str, *, lang: str | None = None, approver: str | None = None
    ) -> str:
        """Hybrid NL: pre-gate -> deterministic fast-path -> LLM tool-calling -> router fallback.

        Entire NL handling (provider retries + tool rounds + final generation) is wrapped
        in a single 12s deadline. On timeout/error/budget exhaustion, falls back to
        deterministic router. Every request logs routing_path without secrets.
        """
        from app.agent.nl_router import UNKNOWN, detect_intent, is_privileged_request
        from app.agent.tools import ToolAccessBlocked

        effective = lang if lang in ("en", "ru") else "en"
        if self._core is None and self._tools is None:
            return self._clean_nl(_ai_t("ai_not_configured", effective))
        # 1. Privileged pre-gate BEFORE any LLM call
        if is_privileged_request(text or ""):
            logger.info("ai_nl_request", extra={"intent": "PRIVILEGED_REFUSED", "duration_ms": 0, "routing_path": "privileged_refused", "outcome": "refused"})
            return self._clean_nl(_ai_t("ai_nl_privileged_refused", effective))
        intent, entities = detect_intent(text or "")
        started = _time.perf_counter()
        routing_path = "unknown"
        self._last_nl_tools = []  # type: ignore
        tool_names: list[str] = []
        outcome = "ok"
        try:
            # 2. Deterministic fast-path — skip for complex arbitrary questions that should use LLM tools
            low = (text or "").lower()
            # Bypass fast-path for ambiguous / explanatory / recommendation / small-talk.
            # Keep obvious read-only queries (balances, today trades, status) on fast-path.
            # Do not add new deterministic intents; use LLM semantic path.
            is_small_talk = any(ph in low for ph in ["привет", "как зовут", "кто ты", "как тебя зовут", "здравствуй"])
            is_recommendation = any(ph in low for ph in ["как улучшить", "чтобы были сделки", "чтобы было больше сделок", "как сделать чтобы были", "посоветуй", "порекомендуй", "что делать чтобы"])
            is_complex_llm_candidate = is_small_talk or is_recommendation or any(ph in low for ph in ["что у нас сейчас", "вообще происходит", "объясни мне", "простыми словами", "почему сегодня"])
            fast = None
            if not is_complex_llm_candidate:
                fast = await self._deterministic_dispatch(intent, entities, effective)
            if fast is not None and intent != UNKNOWN:
                # AI_HELP and known intents are fast-path; also covers UNKNOWN->help but we want LLM for UNKNOWN
                routing_path = "fast_path"
                return fast
            # If intent is UNKNOWN, we try LLM before fallback
            # 3. LLM tool-calling under 12s deadline (covers retries+tools+final)
            # Providers without native tool support will return None -> fallback
            try:
                async with asyncio.timeout(12):
                    llm_answer = await self._execute_llm_tools_flow(text or "", effective)
                    if llm_answer is not None:
                        routing_path = "llm_tools"
                        # Extract tool names from audit? For logging, we already have
                        return llm_answer
            except TimeoutError:
                outcome = "timeout"
                routing_path = "router_fallback"
                logger.warning("ai_nl_timeout", extra={"routing_path": routing_path, "duration_ms": int((_time.perf_counter() - started) * 1000)})
            except ToolAccessBlocked as exc:
                outcome = "tool_blocked"
                routing_path = "router_fallback"
                try:
                    from app.agent.audit import AgentAuditEvent

                    if self._core is not None and getattr(self._core, "_audit", None) is not None:
                        audit = self._core._audit  # type: ignore
                        ev = AgentAuditEvent(event_type="ToolAccessBlocked", query=self._hash_text(text or "")[:500], details={"error": filter_secrets_from_text(str(exc))[:200], "routing_path": routing_path})
                        await audit.log(ev)
                except Exception:
                    pass
            except Exception as exc:
                from app.exchanges.sanitize import redact_secrets as _redact

                _ = _redact(str(exc))[:200]
                outcome = "llm_error"
                routing_path = "router_fallback"
            # 4. Router fallback (deterministic UNKNOWN help or intent-based)
            routing_path = "router_fallback" if routing_path == "unknown" else routing_path
            if intent == UNKNOWN:
                # Small-talk should be answered naturally, not with technical capability list
                if is_small_talk:
                    if effective == "ru":
                        return self._clean_nl("Привет! Я — AI-советник бота. Помогаю с анализом арбитража, балансами и сделками. Спроси, например, «покажи балансы» или «как улучшить, чтобы были сделки?»")
                    return self._clean_nl("Hi! I'm the bot's AI advisor — I help with balances, trades and analysis. Try 'show me balances' or 'how to get more trades?'")
                outcome = outcome if outcome != "ok" else "fallback"
                return self._clean_nl(_ai_t("ai_nl_unknown", effective))
            # For known intents that fast-path already handled, we wouldn't be here.
            # Fallback: try deterministic again or unknown
            fallback = await self._deterministic_dispatch(intent, entities, effective)
            if fallback is not None:
                return fallback
            return self._clean_nl(_ai_t("ai_nl_unknown", effective))
        except Exception as exc:  # noqa: BLE001 - NL must never crash telegram
            from app.exchanges.sanitize import redact_secrets as _redact

            safe = _redact(str(exc))[:200]
            _ = safe
            routing_path = "router_fallback"
            outcome = "exception"
            return self._clean_nl(_ai_t("ai_nl_unknown", effective))
        finally:
            try:
                duration_ms = int((_time.perf_counter() - started) * 1000)
                # Never log secrets or raw credentials
                from app.agent.providers.base import filter_secrets_from_text as _ff

                safe_intent = _ff(intent)[:40]
                # Use tool names from last LLM run if available
                logged_tools = tool_names[:5]
                try:
                    if hasattr(self, "_last_nl_tools") and getattr(self, "_last_nl_tools"):
                        logged_tools = list(getattr(self, "_last_nl_tools"))[:5]
                except Exception:
                    pass
                logger.info("ai_nl_request", extra={"intent": safe_intent, "duration_ms": duration_ms, "routing_path": routing_path, "tools": logged_tools, "outcome": outcome})
                # Clear for next request
                try:
                    self._last_nl_tools = []  # type: ignore
                except Exception:
                    pass
            except Exception:
                pass

    @staticmethod
    def _trade_local_date(raw: Any) -> Any | None:
        """Parse a trade ``created_at`` ISO value to an app-local date."""
        from datetime import datetime as _dt

        if not raw:
            return None
        try:
            parsed = _dt.fromisoformat(str(raw))
        except Exception:
            return None
        try:
            if parsed.tzinfo is None:
                # Domain timestamps are UTC; naive values are treated as UTC.
                from datetime import UTC as _UTC

                parsed = parsed.replace(tzinfo=_UTC)
            return parsed.astimezone().date()
        except Exception:
            return None

    def _filter_trades_by_period(self, trades: list[dict[str, Any]], period: str | None) -> list[dict[str, Any]]:
        """Apply a strict time window; ``None``/``recent`` keeps history order."""
        from datetime import datetime as _dt
        from datetime import timedelta as _td

        if period == "today":
            today = _dt.now().astimezone().date()
            return [t for t in trades if self._trade_local_date(t.get("created_at")) == today]
        if period == "last_hour":
            now = _dt.now().astimezone()
            cutoff = now - _td(hours=1)
            out: list[dict[str, Any]] = []
            for t in trades:
                raw = t.get("created_at")
                try:
                    parsed = _dt.fromisoformat(str(raw)) if raw else None
                except Exception:
                    parsed = None
                if parsed is None:
                    continue
                try:
                    if parsed.tzinfo is None:
                        from datetime import UTC as _UTC

                        parsed = parsed.replace(tzinfo=_UTC)
                    if parsed >= cutoff:
                        out.append(t)
                except Exception:
                    continue
            return out
        return list(trades)

    async def nl_trades(self, *, lang: str | None = None, limit: int = 10, period: str | None = None) -> str:
        """Recent trades, optionally restricted to a strict time window.

        ``period="today"`` returns *only* trades from the current
        application-local calendar date (never stale history); when empty
        it explicitly says so instead of showing old trades.
        ``None``/``"recent"`` keeps the legacy recent-history behavior.
        """
        effective = lang if lang in ("en", "ru") else "en"
        if self._tools is None:
            return self._clean_nl(_ai_t("ai_not_configured", effective))
        fetch_limit = 100 if period in ("today", "last_hour") else limit
        try:
            trades = await self._tools.get_recent_trades(limit=fetch_limit)
        except Exception:
            return self._clean_nl(_ai_t("ai_nl_trades_empty", effective))
        if period in ("today", "last_hour"):
            windowed = self._filter_trades_by_period(trades or [], period)
            if not windowed:
                if period == "today":
                    return self._clean_nl(_ai_t("ai_nl_trades_empty_today", effective))
                return self._clean_nl(_ai_t("ai_nl_trades_empty_last_hour", effective))
            title_key = "ai_nl_trades_today_title" if period == "today" else "ai_nl_trades_last_hour_title"
            lines: list[str] = [_ai_t(title_key, effective, count=len(windowed))]
            for tr in windowed[:limit]:
                lines.append(
                    _ai_t(
                        "ai_nl_trades_line",
                        effective,
                        strategy=str(tr.get("strategy", "?")),
                        route=str(tr.get("route", "?"))[:60],
                        status=str(tr.get("status", "?")),
                        net_profit=str(tr.get("net_profit", "?")),
                        created=str(tr.get("created_at", "?"))[:19],
                    )
                )
            if len(windowed) > limit:
                lines.append(_ai_t("ai_nl_trades_more", effective, remaining=len(windowed) - limit))
            return self._clean_nl("\n".join(lines))
        if not trades:
            return self._clean_nl(_ai_t("ai_nl_trades_empty", effective))
        lines = [_ai_t("ai_nl_trades_title", effective, count=len(trades))]
        for tr in trades[:limit]:
            lines.append(
                _ai_t(
                    "ai_nl_trades_line",
                    effective,
                    strategy=str(tr.get("strategy", "?")),
                    route=str(tr.get("route", "?"))[:60],
                    status=str(tr.get("status", "?")),
                    net_profit=str(tr.get("net_profit", "?")),
                    created=str(tr.get("created_at", "?"))[:19],
                )
            )
        if len(trades) > limit:
            lines.append(_ai_t("ai_nl_trades_more", effective, remaining=len(trades) - limit))
        return self._clean_nl("\n".join(lines))

    async def nl_trade_stats(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._tools is None:
            return self._clean_nl(_ai_t("ai_not_configured", effective))
        try:
            st = await self._tools.get_trade_statistics()
        except Exception:
            return self._clean_nl(_ai_t("ai_nl_trades_empty", effective))
        return self._clean_nl(
            _ai_t(
                "ai_nl_trade_stats",
                effective,
                total=st.get("total", 0),
                completed=st.get("completed", 0),
                failed=st.get("failed", 0),
                manual_review=st.get("manual_review", 0),
                pnl=st.get("total_pnl", "0"),
                bps=st.get("avg_net_bps", "0"),
                win=st.get("win_rate", "0"),
            )
        )

    async def nl_scan_stats(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._tools is None:
            return self._clean_nl(_ai_t("ai_not_configured", effective))
        try:
            stats = await self._tools.get_scan_statistics()
        except Exception:
            stats = {}
        if not isinstance(stats, dict) or not stats:
            stats = {"tickers": 0, "order_books": 0, "exchanges": 0}
        if "note" in stats:
            return self._clean_nl(
                _ai_t(
                    "ai_nl_scan_stats",
                    effective,
                    tickers=stats.get("tickers", 0),
                    books=stats.get("order_books", 0),
                    venues=stats.get("exchanges", 0),
                )
                + f" ({stats.get('note')})"
            )
        return self._clean_nl(
            _ai_t(
                "ai_nl_scan_stats",
                effective,
                tickers=stats.get("tickers", 0),
                books=stats.get("order_books", 0),
                venues=stats.get("exchanges", 0),
            )
        )

    async def nl_opportunities(self, *, lang: str | None = None) -> str:
        """Read-only current opportunities, scoped to the active strategy.

        Latency: cheap ``store.stats()`` first — with zero order books the
        answer is "stale data" with *no* scan at all. Otherwise exactly one
        strategy-scoped calculation runs (triangle *or* transfer, never
        both), timed and logged. Strictly read-only: no orders, no config.
        """
        import time as _time

        effective = lang if lang in ("en", "ru") else "en"
        svc = self._services()
        if svc is None:
            return self._clean_nl(_ai_t("ai_not_configured", effective))
        try:
            stats = await self._tools.get_scan_statistics() if self._tools else {}
        except Exception:
            stats = {}
        books = int((stats or {}).get("order_books", 0) or 0) if isinstance(stats, dict) else 0
        if books == 0:
            lines: list[str] = [_ai_t("ai_nl_opportunities_stale", effective)]
            lines.append(await self._threshold_text(effective))
            return self._clean_nl("\n".join(lines))
        strategy = await self._active_strategy()
        triangles: Any = ()
        plans: Any = []
        scan_ms: int | None = None
        try:
            started = _time.perf_counter()
            if strategy == "transfer":
                try:
                    plans = await svc.plan_transfers()
                except Exception:
                    plans = []
            else:
                try:
                    triangles = await svc.scan_triangles()
                except Exception:
                    triangles = ()
            scan_ms = int((_time.perf_counter() - started) * 1000)
            try:
                logger.info("ai_nl_scan", extra={"strategy": strategy, "duration_ms": scan_ms})
            except Exception:
                pass
        except Exception:
            pass
        lines = []
        if (not triangles) and (not plans):
            lines.append(_ai_t("ai_nl_opportunities_none", effective))
            lines.append(await self._threshold_text(effective))
            if scan_ms is not None:
                lines.append(_ai_t("ai_nl_scan_took", effective, ms=scan_ms))
            return self._clean_nl("\n".join(lines))
        lines.append(_ai_t("ai_nl_opportunities_found", effective))
        for opp in list(triangles or [])[:5]:
            try:
                desc = getattr(opp, "direction", "?") or "?"
                bps = getattr(opp, "net_profit_bps", "?")
                lines.append(_ai_t("ai_nl_opportunities_line", effective, kind="triangle", desc=str(desc)[:60], bps=str(bps)))
            except Exception:
                continue
        for plan in list(plans or [])[:5]:
            try:
                desc = f"{plan.source_exchange}->{plan.dest_exchange} {plan.asset} {plan.amount}"
                lines.append(_ai_t("ai_nl_opportunities_line", effective, kind="transfer", desc=desc[:60], bps=str(plan.net_profit_bps)))
            except Exception:
                continue
        lines.append(await self._threshold_text(effective))
        if scan_ms is not None:
            lines.append(_ai_t("ai_nl_scan_took", effective, ms=scan_ms))
        lines.append("(no execution — view-only)" if effective == "en" else "(без исполнения — только просмотр)")
        return self._clean_nl("\n".join(lines))

    async def _threshold_text(self, effective: str) -> str:
        """One-line threshold provenance for the active runtime path."""
        ctx = await self._threshold_context()
        return _ai_t(
            "ai_nl_threshold_line",
            effective,
            value=ctx["threshold"],
            source=ctx["source"],
            strategy=ctx["strategy"],
        )

    async def nl_exchange_status(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._tools is None:
            return self._clean_nl(_ai_t("ai_not_configured", effective))
        try:
            snap = await self._tools.get_exchange_status()
        except Exception:
            snap = {}
        lines: list[str] = [_ai_t("ai_nl_exchange_status", effective)]
        if not snap:
            enabled = self._enabled_venues()
            if enabled:
                for v in enabled:
                    lines.append(_ai_t("ai_nl_exchange_unavailable", effective, venue=v, reason="no data"))
            else:
                lines.append("-")
            return self._clean_nl("\n".join(lines))
        for venue in sorted(snap.keys()):
            info = snap.get(venue, {}) if isinstance(snap, dict) else {}
            lines.append(
                _ai_t(
                    "ai_nl_exchange_line",
                    effective,
                    venue=venue,
                    status=info.get("status", "?") if isinstance(info, dict) else "?",
                    creds=info.get("credentials", "?") if isinstance(info, dict) else "?",
                )
            )
        # Partial failures: enabled venues missing from snapshot.
        try:
            enabled = {e.strip().lower() for e in self._enabled_venues()}
            present = {str(v).strip().lower() for v in snap.keys()}
            for missing in sorted(enabled - present):
                lines.append(_ai_t("ai_nl_exchange_unavailable", effective, venue=missing, reason="no snapshot"))
        except Exception:
            pass
        return self._clean_nl("\n".join(lines))

    async def nl_bot_status(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        svc = self._services()
        if svc is None:
            return self._clean_nl(_ai_t("ai_not_configured", effective))
        try:
            st = await svc.status()
        except Exception:
            return self._clean_nl(_ai_t("ai_not_configured", effective))
        guard = st.get("guard", {}) if isinstance(st, dict) else {}
        lines = [
            _ai_t("ai_nl_bot_status", effective),
            f"mode: {st.get('mode', '?')}",
            f"kill switch: {guard.get('halted', '?')} ({guard.get('halt_reason', '')})".rstrip(),
            f"auto trading: {st.get('auto_trading', '?')} / loop: {st.get('auto_loop_running', '?')}",
            f"strategy: {st.get('active_strategy', '?')}",
        ]
        return self._clean_nl("\n".join(lines))

    async def nl_bot_operation(self, *, lang: str | None = None) -> str:
        """Explain how the bot actually trades (read-only, from live config).

        Describes the real pipeline — strategies, market data, scanning,
        profitability, sizing, risk, execution, recovery, persistence and
        DEMO-vs-LIVE behavior — using current runtime values. Never invents
        components; every named stage exists in this repository.
        """
        effective = lang if lang in ("en", "ru") else "en"
        svc = self._services()
        if svc is None:
            return self._clean_nl(_ai_t("ai_not_configured", effective))
        settings = getattr(svc, "settings", None)
        strategy = await self._active_strategy()
        ctx = await self._threshold_context()
        venues: list[str] = []
        try:
            venues = list(self._enabled_venues())
        except Exception:
            pass
        venues_s = ", ".join(venues) if venues else "?"
        mode = "?"
        tri_assets = ""
        trf_assets = ""
        interval = "?"
        try:
            if settings is not None:
                mode = str(getattr(getattr(settings, "mode", "?"), "value", getattr(settings, "mode", "?")))
                tri_assets = ", ".join(list(getattr(getattr(settings, "arbitrage", None), "triangle_assets", ()) or ()))
                trf_assets = ", ".join(list(getattr(getattr(settings, "transfer", None), "assets", ()) or ())[:8])
                interval = str(getattr(getattr(settings, "execution", None), "auto_interval_seconds", "?"))
        except Exception:
            pass
        if effective == "ru":
            lines = [
                _ai_t("ai_nl_bot_operation_title", effective),
                f"Сейчас активна стратегия: {strategy}. Режим: {mode}. Площадки: {venues_s}.",
                "",
                "Как бот торгует (реальный конвейер):",
                "1. Рыночные данные: MarketDataService (сначала WebSocket, fallback — REST) собирает стаканы/тикеры в MarketDataStore; устаревшие данные отбрасываются.",
                f"2. Сканирование: triangle — TriangularScanner ищет циклы среди активов ({tri_assets or '?'}); transfer — TransferPlanner/Orchestrator ищут межбиржевые различия ({trf_assets or '?'}).",
                f"3. Прибыльность: чистая прибыль после комиссий и проскальзывания сравнивается с порогом стратегии ({ctx['threshold']} bps, {ctx['source']}); размер позиции ограничен авто-ноционалом.",
                f"4. Риск: RiskEngine (лимиты) + ExecutionGuard (стоп-кран); исполнение дополнительно требует net >= риск-лимита ({ctx['risk_threshold']} bps, risk.min_net_profit_bps).",
                f"5. Исполнение: цикл AutoTrader (каждые {interval} c) проверяет флаг автоторговли, валидирует риск и исполняет через TriangleExecutor / start_transfer. Ордера: PAPER — никогда, DEMO — только песочница, LIVE — только через guard.",
                "6. Учёт и восстановление: результат пишется в TradeRepository и audit-журнал; прерванные циклы ExecutionRecovery переводит в MANUAL_REVIEW, а не продолжает автоматически.",
                "",
                "Что может мешать исполнению: стоп-кран, выключенная автоторговля, отсутствие стратегии, нет стаканов/устаревшие данные, возможности ниже порога, недоступная биржа, риск-лимиты, провал preflight.",
            ]
        else:
            lines = [
                _ai_t("ai_nl_bot_operation_title", effective),
                f"Active strategy: {strategy}. Mode: {mode}. Venues: {venues_s}.",
                "",
                "How the bot trades (actual pipeline):",
                "1. Market data: MarketDataService (WebSocket-first, REST fallback) feeds order books/tickers into MarketDataStore; stale quotes are discarded.",
                f"2. Scanning: triangle — TriangularScanner walks cycles over ({tri_assets or '?'}); transfer — TransferPlanner/Orchestrator look for cross-venue spreads ({trf_assets or '?'}).",
                f"3. Profitability: net profit after taker fees and slippage is compared against the strategy threshold ({ctx['threshold']} bps, {ctx['source']}); position size is capped by the auto notional.",
                f"4. Risk: RiskEngine (limits) + ExecutionGuard (kill switch); execution additionally requires net >= risk limit ({ctx['risk_threshold']} bps, risk.min_net_profit_bps).",
                f"5. Execution: the AutoTrader cycle (every {interval}s) checks the auto-trading flag, risk-validates and executes via TriangleExecutor / start_transfer. Orders: PAPER — never, DEMO — sandbox only, LIVE — guard-gated.",
                "6. Recording & recovery: results go to TradeRepository and the audit journal; interrupted cycles are failed-closed into MANUAL_REVIEW by ExecutionRecovery, never auto-continued.",
                "",
                "What can prevent execution: kill switch, auto-trading off, no active strategy, missing/stale books, opportunities below threshold, venue offline, risk limits, preflight failure.",
            ]
        return self._clean_nl("\n".join(lines))

    async def nl_risk(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._tools is None:
            return self._clean_nl(_ai_t("ai_not_configured", effective))
        try:
            rs = await self._tools.get_risk_state()
        except Exception:
            return self._clean_nl(_ai_t("ai_not_configured", effective))
        lines = [
            _ai_t("ai_nl_risk_title", effective),
            f"daily pnl: {rs.get('daily_pnl', '?')}",
            f"open transfers: {rs.get('open_transfers', '?')}",
            f"limits: {rs.get('limits', {})}",
            f"kill switch: {rs.get('kill_switch', {})}",
        ]
        return self._clean_nl("\n".join(lines))

    async def nl_parameters(self, *, lang: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._tools is None:
            return self._clean_nl(_ai_t("ai_not_configured", effective))
        try:
            params = await self._tools.get_current_parameters()
        except Exception:
            return self._clean_nl(_ai_t("ai_not_configured", effective))
        lines = [_ai_t("ai_nl_params_title", effective)]
        try:
            import json as _json

            lines.append(_json.dumps(params, ensure_ascii=False, default=str)[:2500])
        except Exception:
            lines.append(str(params)[:2500])
        return self._clean_nl("\n".join(lines))

    async def nl_journal(self, *, lang: str | None = None, limit: int = 10) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._tools is None:
            return self._clean_nl(_ai_t("ai_not_configured", effective))
        try:
            rows = await self._tools.get_recent_journal(limit=limit)
        except Exception:
            rows = []
        if not rows:
            return self._clean_nl(_ai_t("ai_nl_journal_empty", effective))
        lines: list[str] = [_ai_t("ai_nl_journal_title", effective, count=len(rows))]
        for r in rows[:limit]:
            lines.append(
                _ai_t(
                    "ai_nl_journal_line",
                    effective,
                    ts=str(r.get("ts", "?"))[:19],
                    action=str(r.get("action", "?"))[:40],
                    message=str(r.get("message", ""))[:120],
                )
            )
        return self._clean_nl("\n".join(lines))

    async def nl_why_not_trading(self, *, lang: str | None = None) -> str:
        """Explain blockers from cheap read-only diagnostics (no fresh scan).

        Uses a single ``status()`` snapshot plus store/exchange/risk/trade
        statistics already available at runtime. No ``scan_triangles()`` /
        ``plan_transfers()`` call: freshness-gated opportunity detection
        belongs to the opportunities handler. Anything not supported by the
        data is explicitly marked as insufficient — never invented.
        """
        effective = lang if lang in ("en", "ru") else "en"
        svc = self._services()
        if svc is None or self._tools is None:
            return self._clean_nl(_ai_t("ai_not_configured", effective))
        blockers: list[str] = []
        cheap_ok = False
        # Single status snapshot: kill switch / guard / auto flag / strategy / loop.
        try:
            st = await svc.status()
            cheap_ok = True
            guard = st.get("guard", {}) if isinstance(st, dict) else {}
            if str(guard.get("halted", "")).lower() == "true":
                blockers.append(f"kill switch ENGAGED ({guard.get('halt_reason', '')})")
            if str(guard.get("trading_enabled", "")).lower() in ("false", "0"):
                blockers.append("trading disabled by execution guard")
            if isinstance(st, dict) and not st.get("auto_trading", False):
                blockers.append("auto-trading flag is OFF (strategy not running)" if effective == "en" else "флаг автоторговли ВЫКЛ (стратегия не запущена)")
            if isinstance(st, dict) and st.get("active_strategy", "not_set") == "not_set":
                blockers.append("no active strategy selected" if effective == "en" else "активная стратегия не выбрана")
            if isinstance(st, dict) and st.get("auto_trading", False) and not st.get("auto_loop_running", False):
                blockers.append("auto-trading flag is ON but the loop is not running" if effective == "en" else "флаг автоторговли ВКЛ, но цикл не запущен")
            # Preflight hint: DEMO venues that never delivered market data.
            md = st.get("market_data", {}) if isinstance(st, dict) else {}
            if isinstance(md, dict) and int(md.get("order_books", 0) or 0) == 0:
                blockers.append("missing order books (market data unavailable — possible preflight failure)" if effective == "en" else "нет стаканов (рыночные данные недоступны — возможен провал preflight)")
        except Exception:
            pass
        # Market-data freshness (cheap store stats, no refresh).
        try:
            stats = await self._tools.get_scan_statistics()
            cheap_ok = True
            if isinstance(stats, dict):
                if int(stats.get("order_books", 0) or 0) == 0:
                    marker = "missing order books" if effective == "en" else "нет стаканов"
                    if marker not in blockers:
                        blockers.append(marker)
        except Exception:
            pass
        # Exchange availability (cheap snapshot, no I/O).
        try:
            snap = await self._tools.get_exchange_status()
            cheap_ok = True
            if isinstance(snap, dict):
                for venue, info in snap.items():
                    status = str((info or {}).get("status", "")).lower() if isinstance(info, dict) else ""
                    if status in ("offline", "degraded"):
                        blockers.append(f"exchange {venue} {status}")
                    if isinstance(info, dict) and info.get("private_blocked"):
                        blockers.append(f"exchange {venue} private calls blocked")
        except Exception:
            pass
        # Failed-trade evidence (cheap DB aggregate).
        try:
            tstats = await self._tools.get_trade_statistics()
            cheap_ok = True
            if isinstance(tstats, dict):
                if int(tstats.get("failed", 0) or 0) > 0:
                    blockers.append(f"failed trades observed: {tstats.get('failed')}")
                if int(tstats.get("manual_review", 0) or 0) > 0:
                    blockers.append(f"manual-review cases: {tstats.get('manual_review')}")
        except Exception:
            pass
        lines = [_ai_t("ai_nl_why_title", effective)]
        if blockers:
            for b in blockers[:10]:
                lines.append(f"  - {b}"[:300])
        elif not cheap_ok:
            lines.append(_ai_t("ai_nl_insufficient_data", effective))
        else:
            lines.append(_ai_t("ai_nl_why_no_blocker", effective))
        # Threshold provenance: the value actually governing the active path,
        # plus the risk execution gate (the 5-vs-10 root cause, made explicit).
        try:
            ctx = await self._threshold_context()
            lines.append(
                _ai_t(
                    "ai_nl_threshold_line",
                    effective,
                    value=ctx["threshold"],
                    source=ctx["source"],
                    strategy=ctx["strategy"],
                )
            )
            if str(ctx.get("risk_threshold", "?")) != "?":
                lines.append(_ai_t("ai_nl_risk_gate_line", effective, value=ctx["risk_threshold"]))
        except Exception:
            pass
        return self._clean_nl("\n".join(lines))

    async def approve(self, rec_id: str, *, lang: str | None = None, approver: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._approval is None:
            return _finalize_telegram(_ai_t("ai_not_configured", effective))
        if not approver:
            # Must be human — fail-closed if approver missing
            return _finalize_telegram(_ai_t("ai_approve_fail", effective, error="approver required"))
        try:
            # Redact rec_id (no secret, but still defensive)
            from app.exchanges.sanitize import redact_secrets as _redact

            rec_id = _redact(rec_id).strip()
            result = await self._approval.approve(rec_id, approver=str(approver), reason=f"telegram /ai approve by {approver}")
            return _finalize_telegram(
                _ai_t(
                    "ai_approve_ok",
                    effective,
                    parameter=result.parameter,
                    old=result.old_value or result.current_value or "-",
                    proposed=result.proposed_value,
                    approver=approver,
                )
            )
        except Exception as exc:  # noqa: BLE001 - telegram must not leak internal details
            from app.exchanges.sanitize import redact_secrets as _redact

            safe = _redact(str(exc))[:200]
            return _finalize_telegram(_ai_t("ai_approve_fail", effective, error=safe))

    async def reject(self, rec_id: str, *, lang: str | None = None, approver: str | None = None) -> str:
        effective = lang if lang in ("en", "ru") else "en"
        if self._approval is None:
            return _finalize_telegram(_ai_t("ai_not_configured", effective))
        if not approver:
            return _finalize_telegram(_ai_t("ai_reject_fail", effective, error="approver required"))
        try:
            from app.exchanges.sanitize import redact_secrets as _redact

            rec_id = _redact(rec_id).strip()
            await self._approval.reject(rec_id, approver=str(approver), reason=f"telegram /ai reject by {approver}")
            return _finalize_telegram(_ai_t("ai_reject_ok", effective, id=rec_id[:12], approver=approver))
        except Exception as exc:  # noqa: BLE001
            from app.exchanges.sanitize import redact_secrets as _redact

            safe = _redact(str(exc))[:200]
            return _finalize_telegram(_ai_t("ai_reject_fail", effective, error=safe))

    async def measurements(self, rec_id: str | None = None, *, lang: str | None = None) -> str:
        """Phase 9 visibility: measured outcomes (list or one recommendation)."""
        effective = lang if lang in ("en", "ru") else "en"
        if self._learning is None:
            return _finalize_telegram(_ai_t("ai_not_configured", effective))
        try:
            if rec_id:
                from app.exchanges.sanitize import redact_secrets as _redact

                measurement = await self._learning.get_by_recommendation(_redact(rec_id).strip())
                if measurement is None:
                    state = None
                    try:
                        approval = getattr(self._learning, "_services", None)
                        approval = getattr(approval, "agent_approval_service", None) if approval else self._approval
                        if approval is not None and hasattr(approval, "measurement_state"):
                            state = await approval.measurement_state(_redact(rec_id).strip())
                    except Exception:
                        state = None
                    if state is not None and state.get("status") == "awaiting_measurement":
                        return _finalize_telegram(
                            _ai_t("ai_measurements_detail", effective, rec_id=rec_id[:12],
                                  parameter=state.get("parameter", "?"), old=state.get("old_value", "?"),
                                  new=state.get("new_value", "?"), outcome="awaiting_measurement",
                                  reason="approved, not yet measured", metric="-", before="-", after="-",
                                  delta="-", n_before=0, n_after=0, confidence=0.0,
                                  feedback="-", lesson="-"))
                    return _finalize_telegram(_ai_t("ai_measurements_missing", effective, rec_id=rec_id[:12]))
                return _finalize_telegram(
                    _ai_t("ai_measurements_detail", effective, rec_id=measurement.recommendation_id[:12],
                          parameter=measurement.parameter, old=measurement.old_value, new=measurement.new_value,
                          outcome=measurement.outcome.value
                          if hasattr(measurement.outcome, "value") else str(measurement.outcome),
                          reason=(measurement.reason or "")[:160], metric=measurement.metric,
                          before=measurement.before_value, after=measurement.after_value,
                          delta=measurement.delta, n_before=measurement.n_before, n_after=measurement.n_after,
                          confidence=float(measurement.confidence),
                          feedback=measurement.feedback_kind or "-",
                          lesson=(measurement.lesson_id[:12] if measurement.lesson_id else "-")))
            items = await self._learning.list_recent_measurements(limit=5)
            if not items:
                return _finalize_telegram(_ai_t("ai_measurements_empty", effective))
            lines: list[str] = [_ai_t("ai_measurements_header", effective, count=len(items))]
            for m in items:
                lines.append(
                    _ai_t("ai_measurements_line", effective, parameter=m.parameter,
                          outcome=m.outcome.value if hasattr(m.outcome, "value") else str(m.outcome),
                          metric=m.metric, before=m.before_value, after=m.after_value,
                          n_before=m.n_before, n_after=m.n_after,
                          rec_id=m.recommendation_id[:12]))
            return _finalize_telegram("\n".join(lines))
        except Exception as exc:  # noqa: BLE001 - telegram must not leak internal details
            from app.exchanges.sanitize import redact_secrets as _redact

            safe = _redact(str(exc))[:200]
            return _finalize_telegram(_ai_t("ai_measurements_missing", effective, rec_id=(rec_id or "?")[:12])
                                      if rec_id else _ai_t("ai_measurements_empty", effective) + f" ({safe[:60]})")

    async def feedback(self, rec_id: str, kind: str, comment: str = "", *, lang: str | None = None,
                       approver: str | None = None) -> str:
        """Phase 9 visibility: operator feedback (human-gated like approve)."""
        effective = lang if lang in ("en", "ru") else "en"
        if self._learning is None:
            return _finalize_telegram(_ai_t("ai_not_configured", effective))
        if not approver:
            return _finalize_telegram(_ai_t("ai_feedback_fail", effective, error="approver required"))
        try:
            from app.exchanges.sanitize import redact_secrets as _redact

            clean_id = _redact(rec_id).strip()
            kwargs: dict[str, Any] = {"recommendation_id": clean_id,
                                      "kind": str(kind).strip().lower(),
                                      "approver": str(approver), "comment": comment}
            try:
                existing = await self._learning.get_by_recommendation(clean_id)
                if existing is not None:
                    kwargs["measurement_id"] = existing.id
            except Exception:
                pass
            result = await self._learning.record_feedback(**kwargs)
            return _finalize_telegram(
                _ai_t("ai_feedback_ok", effective, kind=result["kind"],
                      rec_id=str(result["recommendation_id"])[:12], approver=approver))
        except Exception as exc:  # noqa: BLE001
            from app.exchanges.sanitize import redact_secrets as _redact

            safe = _redact(str(exc))[:200]
            return _finalize_telegram(_ai_t("ai_feedback_fail", effective, error=safe))
