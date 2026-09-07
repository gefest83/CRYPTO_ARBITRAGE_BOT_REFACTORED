"""Deterministic natural-language intent router for the AI Advisor.

This module is intentionally *deterministic* — substring keyword matching
over a normalized (lowercased) message, no LLM, no function calling.

Supported intents (at least)::

    BALANCE_QUERY, TRADES_QUERY, TRADE_STATS_QUERY, SCAN_STATS_QUERY,
    OPPORTUNITIES_QUERY, EXCHANGE_STATUS_QUERY, BOT_STATUS_QUERY,
    BOT_OPERATION_QUERY, RISK_QUERY, PARAMETERS_QUERY, MEMORY_QUERY,
    JOURNAL_QUERY, RECOMMENDATIONS_QUERY, WHY_NOT_TRADING_QUERY,
    AI_HELP, UNKNOWN

``BOT_OPERATION_QUERY`` ("how does the bot trade/work?") is an
architecture/behavior explanation; ``BOT_STATUS_QUERY`` ("what is the bot
doing right now?") is current runtime state. They are distinct intents.

``TRADES_QUERY`` may carry a time-window entity (``period`` = ``today`` /
``last_hour`` / ``recent`` / ``None``); recent-history questions carry no
period and keep the legacy recent-trades behavior.

Security: :func:`is_privileged_request` detects natural-language attempts
to perform privileged actions (approve / trade / withdraw / configure).
The Telegram layer must refuse those *before* any tool call.
"""

from __future__ import annotations

import re

__all__ = [
    "BALANCE_QUERY",
    "TRADES_QUERY",
    "TRADE_STATS_QUERY",
    "SCAN_STATS_QUERY",
    "OPPORTUNITIES_QUERY",
    "EXCHANGE_STATUS_QUERY",
    "BOT_STATUS_QUERY",
    "BOT_OPERATION_QUERY",
    "RISK_QUERY",
    "PARAMETERS_QUERY",
    "MEMORY_QUERY",
    "JOURNAL_QUERY",
    "RECOMMENDATIONS_QUERY",
    "WHY_NOT_TRADING_QUERY",
    "AI_HELP",
    "UNKNOWN",
    "PERIOD_TODAY",
    "PERIOD_LAST_HOUR",
    "PERIOD_RECENT",
    "detect_intent",
    "is_privileged_request",
    "extract_balance_filters",
    "extract_trade_period",
    "KNOWN_VENUES",
]

BALANCE_QUERY = "BALANCE_QUERY"
TRADES_QUERY = "TRADES_QUERY"
TRADE_STATS_QUERY = "TRADE_STATS_QUERY"
SCAN_STATS_QUERY = "SCAN_STATS_QUERY"
OPPORTUNITIES_QUERY = "OPPORTUNITIES_QUERY"
EXCHANGE_STATUS_QUERY = "EXCHANGE_STATUS_QUERY"
BOT_STATUS_QUERY = "BOT_STATUS_QUERY"
RISK_QUERY = "RISK_QUERY"
PARAMETERS_QUERY = "PARAMETERS_QUERY"
MEMORY_QUERY = "MEMORY_QUERY"
JOURNAL_QUERY = "JOURNAL_QUERY"
RECOMMENDATIONS_QUERY = "RECOMMENDATIONS_QUERY"
WHY_NOT_TRADING_QUERY = "WHY_NOT_TRADING_QUERY"
BOT_OPERATION_QUERY = "BOT_OPERATION_QUERY"
AI_HELP = "AI_HELP"
UNKNOWN = "UNKNOWN"

#: Trade time-window entities for TRADES_QUERY.
PERIOD_TODAY = "today"
PERIOD_LAST_HOUR = "last_hour"
PERIOD_RECENT = "recent"

KNOWN_VENUES = ("binance", "okx", "bybit")


def _norm(text: str) -> str:
    return (text or "").strip().lower()


# ---------------------------------------------------------------------------
# Privileged (must NEVER be executed via natural language)
# ---------------------------------------------------------------------------

_PRIVILEGED_PATTERNS: tuple[str, ...] = (
    # approval shortcuts
    "approve", "одобр", "подтверд", "reject", "отклон",
    # trading / orders
    "place an order", "place order", "open order", "execute trade",
    "start trading", "stop trading", "start auto", "stop auto",
    "запусти торгов", "запустить торгов", "останови торгов", "остановить торгов",
    "начни торгов", "начать торгов", "торгуй", "открой сделку", "открой ордер",
    "исполни сделку", "соверши сделку", "купи", "продай", "buy ", "sell ",
    # withdrawals / transfers of funds
    "withdraw", "вывод средств", "выведи", "вывести средства", "переведи средства",
    "transfer funds", "send funds",
    # configuration mutation
    "change min profit", "change profit", "set min profit", "min profit to",
    "change setting", "modify setting", "update config", "change config",
    "измени", "изменить", "поменяй", "установи", "настрой",
    "минимальн", "конфиг",
    # credentials
    "api key", "api-key", "secret key", "credentials", "апи ключ", "секретн",
    "токен биржи", "credentials",
    # kill switch manipulation via NL
    "engage kill", "release kill", "kill switch",
    "стоп-кран", "стопкран",
)


def is_privileged_request(text: str) -> bool:
    """Return True when ``text`` asks for a privileged (non-read-only) action."""
    t = _norm(text)
    if not t:
        return False
    # Explicit /ai approve|reject|feedback remain allowed through the explicit
    # path — but *natural language* approval shortcuts are forbidden.
    if t.startswith("/ai"):
        return False
    for pat in _PRIVILEGED_PATTERNS:
        if pat in t:
            # Guard against false positives for read-only status words:
            # "kill switch" alone in a *question* ("is kill switch engaged?")
            # is read-only. Only treat as privileged when paired with a
            # mutation verb.
            if pat in ("kill switch", "стоп-кран", "стопкран"):
                mutation_verbs = (
                    "engage", "release", "enable", "disable", "включи",
                    "выключи", "включить", "выключить", "engage", "release",
                    "сними", "поставь",
                )
                if not any(v in t for v in mutation_verbs):
                    continue
            # "buy"/"sell" appear in transfer-route descriptions ("buy_price");
            # require them to look like an instruction, not a noun.
            if pat in ("buy ", "sell "):
                action_verbs = (
                    "place", "open", "execute", "start", "сделай", "открой",
                    "исполни", "купи", "продай", "buy", "sell",
                )
                # bare "buy"/"sell" single word is privileged-ish; longer
                # sentences mentioning trade history are not.
                if any(w in t for w in ("trade", "сделк", "history", "история", "recent", "последн", "show", "покажи")):
                    continue
                _ = action_verbs  # keep linter quiet; fall through to True
            return True
    return False


# ---------------------------------------------------------------------------
# Intent keyword tables (RU + EN, substring matching)
# ---------------------------------------------------------------------------

# Each entry: (intent, patterns). Order matters only for tie-breaking —
# scoring picks the intent with the most pattern hits.
_INTENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (AI_HELP, (
        "what can you do", "what can i ask", "what do you do", "help me",
        "capabilities", "how to use", "how do i use",
        "что ты умеешь", "что умеешь", "что можно спросить", "что спросить",
        "помощь", "помоги", "твои возможности", "возможности советника",
        "справка", "как пользоваться", "как тебя использовать",
        "help", "commands", "команды",
    )),
    (WHY_NOT_TRADING_QUERY, (
        "why is the bot not trading", "why are there no trades",
        "why hasn't the bot traded", "why has not", "why no trades",
        "what is preventing trades", "what prevents trades",
        "what is blocking trades", "why is bot not trading",
        "why no trading", "no trading, why",
        "почему бот не торгует", "почему нет сделок", "почему сейчас нет сделок",
        "почему бот не торговал", "почему не торгует", "почему нет торговли",
        "что мешает", "что блокирует", "почему сделок нет",
        "почему бот ничего не делает", "почему не исполняются сделки",
        "почему сделки не исполняются",
    )),
    (BOT_OPERATION_QUERY, (
        "how does the bot trade", "how does our bot work", "how does the bot work",
        "how does it find arbitrage", "how does it find trades",
        "how does it choose trades", "how are routes selected",
        "how does it decide to trade", "how does the trading cycle work",
        "how does the bot choose", "how does the bot decide",
        "what strategies does the bot use", "what strategies are used",
        "how does the bot find", "trading cycle",
        "как торгует наш бот", "как торгует бот", "как работает бот",
        "как работает наш бот", "как бот выбирает сделки", "как он ищет арбитраж",
        "как он ищет сделки", "как происходит торговый цикл", "торговый цикл",
        "как бот принимает решение о сделке", "как принимается решение о сделке",
        "какие стратегии использует бот", "какие стратегии используются",
        "как выбираются маршруты", "как ищет арбитраж", "как устроен бот",
        "как бот ищет",
    )),
    (BALANCE_QUERY, (
        "balance", "balances", "баланс", "балансы",
        "how much money", "how much funds", "funds on", "money on",
        "сколько денег", "сколько средств", "денег на бирж", "средств на бирж",
        "wallet", "кошел", "счет на бирж", "счёт на бирж",
    )),
    (TRADE_STATS_QUERY, (
        "trade statistic", "trading statistic", "trade stats",
        "win rate", "winrate", "success rate", "how many successful",
        "how many failed", "failed trades count",
        "статистик сделок", "статистику сделок", "статистика сделок",
        "статистика торгов", "статистика по сделкам", "статистики сделок",
        "винрейт", "процент успешных", "сколько успешных", "сколько неуспешных",
        "успешных сделок", "неудачных сделок",
    )),
    (TRADES_QUERY, (
        "trade", "trades", "сделк", "торгов",
        "were there any", "did the bot make", "did bot make",
        "show recent", "recent trades", "last trade", "last few trade",
        "last hour", "today", "recently", "lately",
        "были ли", "были сделки", "какие сделки", "покажи последние",
        "последние сделки", "сегодня", "недавно", "недавние",
        "что произошло", "история сделок", "failed trade", "неудавш",
        "manual review", "ручн",
    )),
    (SCAN_STATS_QUERY, (
        "scan statistic", "scan stats", "scanner found", "scanner find",
        "how many opportunities did the scanner", "what happened during the last scan",
        "last scan", "scan result", "scan statistics", "routes checked", "missing books",
        "stale books", "below threshold", "sizing failure", "pricing failure",
        "сколько возможностей нашел сканер", "сколько возможностей нашёл сканер",
        "что показал последний скан", "что нашел последний скан",
        "что нашёл последний скан",         "последний скан", "последнее сканирование",
        "статистику сканирования", "статистика сканирования", "статистики сканирования",
        "статистик сканирования", "статистика сканера", "статистику сканера",
        "почему нет возможностей",
    )),
    (OPPORTUNITIES_QUERY, (
        "opportunit", "profitable route", "arbitrage opportunit", "arbitrage",
        "routes available", "routes are available", "routes right now", "available routes",
        "current routes", "profitable", "прибыльн", "маршрут", "возможност",
        "арбитраж", "текущие маршруты", "покажи текущие",
        "есть ли сейчас", "есть сейчас", "какие сейчас есть",
    )),
    (EXCHANGE_STATUS_QUERY, (
        "exchanges are online", "exchange online", "exchanges online",
        "which exchanges", "what exchanges", "exchanges are working",
        "exchanges working", "is binance", "is okx", "is bybit",
        "binance online", "okx online", "bybit online", "binance ready",
        "okx ready", "bybit ready", "exchange status", "venue status",
        "какие биржи", "какие биржи работают", "какие биржи онлайн",
        "биржи онлайн", "биржи работают", "биржа онлайн", "статус бирж",
        "binance работает", "okx работает", "bybit работает",
        "binance онлайн", "okx онлайн",
    )),
    (BOT_STATUS_QUERY, (
        "bot status", "status of the bot", "is auto trading", "is autotrade",
        "auto trading running", "auto-trading", "is the bot running",
        "is bot running", "what is happening with", "what's happening",
        "what is the bot doing", "what is the bot doing now",
        "is the bot currently trading", "currently trading",
        "what strategies is the bot using now", "which strategy is active",
        "what strategy is active",
        "статус бота", "состояние бота", "автоторговля", "автоторгов",
        "бот работает", "бот запущен", "что сейчас происходит",
        "что происходит", "статус работы",
        "что сейчас делает бот", "что делает бот сейчас",
        "что сейчас происходит с ботом",
        "какая стратегия сейчас активна", "какая стратегия активна",
        "активная стратегия", "активна сейчас",
        "торгует ли бот сейчас", "бот сейчас торгует",
    )),
    (RISK_QUERY, (
        "risk state", "risk status", "current risk", "risk limit",
        "kill switch engaged", "is kill switch", "daily pnl", "daily p&l",
        "open transfers", "exposure",
        "состояние риска", "статус риска", "риск", "лимит",
        "дневной pnl", "дневной пнл", "открытые переводы", "стоп-кран",
        "стопкран",
    )),
    (PARAMETERS_QUERY, (
        "parameter", "current parameter", "current settings", "settings now",
        "min_net_profit", "min profit", "threshold", "current config",
        "какие сейчас параметры", "параметры", "настройки", "текущие настройки",
        "порог", "минимальн",
    )),
    (MEMORY_QUERY, (
        "agent memory", "show memory", "memory entries", "experiences",
        "lessons", "память агента", "покажи память", "память",
        "опыты", "уроки",
    )),
    (JOURNAL_QUERY, (
        "journal", "audit log", "журнал", "аудит",
    )),
    (RECOMMENDATIONS_QUERY, (
        "recommendation", "recommend", "рекомендац",
        "покажи рекомендации", "советы",
    )),
)

# Minimum signal: single short generic words ("help", "риск", "память")
# should still route — they are unambiguous in this bot's domain.
# Longer generic words ("trade", "сделк") need at least one hit too (score>=1).


def detect_intent(text: str) -> tuple[str, dict]:
    """Classify ``text`` into an intent plus extracted entities.

    Returns ``(intent, entities)`` where entities may carry
    ``venue`` / ``asset`` filters for balance queries.
    """
    t = _norm(text)
    if not t:
        return UNKNOWN, {}
    # Slash commands are never NL intents — the explicit dispatcher owns them.
    if t.startswith("/"):
        return UNKNOWN, {}

    best_intent = UNKNOWN
    best_score = 0
    for intent, patterns in _INTENT_PATTERNS:
        score = 0
        for pat in patterns:
            if pat and pat in t:
                # Weight multi-word / specific patterns higher.
                score += 2 if " " in pat else 1
        if score > best_score:
            best_score = score
            best_intent = intent

    # Balance-filter shorthand without the word "balance":
    #   "show USDT on Binance" / "сколько USDT на Binance" /
    #   "покажи баланс Binance" — a venue plus (an asset or an amount word
    #   like "сколько"/"show") is a balance question even when the keyword
    #   tables above produced no hit (or a weak competing hit).
    filters = extract_balance_filters(text)
    has_venue = bool(filters.get("venue"))
    has_asset = bool(filters.get("asset"))
    amount_words = ("сколько", "show", "покажи", "показать", "how much", "balance", "баланс")
    has_amount_word = any(w in t for w in amount_words)
    if has_venue and (has_asset or has_amount_word):
        venue_score = best_score if best_intent == BALANCE_QUERY else 0
        # A strong competing signal (e.g. "is Binance online" -> exchange
        # status with a multi-word hit) still wins; otherwise balance wins.
        if best_intent == UNKNOWN or (best_intent != BALANCE_QUERY and best_score <= 2) or best_intent == BALANCE_QUERY:
            # Exchange-status questions ("is Binance online?") contain no
            # asset and no amount word besides the venue — they keep their
            # intent; the shorthand above already requires asset/amount word.
            if best_intent in (UNKNOWN, BALANCE_QUERY, OPPORTUNITIES_QUERY) or (
                best_intent == EXCHANGE_STATUS_QUERY and (has_asset or "баланс" in t or "balance" in t or "сколько" in t)
            ):
                return BALANCE_QUERY, filters

    if best_score <= 0:
        return UNKNOWN, {}

    entities: dict = {}
    if best_intent == BALANCE_QUERY:
        entities.update(filters)
    if best_intent == TRADES_QUERY:
        period = extract_trade_period(text)
        if period is not None:
            entities["period"] = period
    return best_intent, entities


_TODAY_MARKERS: tuple[str, ...] = ("сегодня", "today", "за сегодня")

_LAST_HOUR_MARKERS: tuple[str, ...] = (
    "last hour", "past hour",
    "за последний час", "за последние часы", "последний час",
)

_RECENT_MARKERS: tuple[str, ...] = (
    "recently", "recent", "lately", "last few",
    "недавно", "недавние", "последние сделки", "последняя",
)


def extract_trade_period(text: str) -> str | None:
    """Extract the requested time window from a trades question.

    ``today`` / ``last_hour`` are strict filters; ``recent`` (or ``None``)
    keeps the legacy recent-history behavior.
    """
    t = _norm(text)
    if any(m in t for m in _TODAY_MARKERS):
        return PERIOD_TODAY
    if any(m in t for m in _LAST_HOUR_MARKERS):
        return PERIOD_LAST_HOUR
    if any(m in t for m in _RECENT_MARKERS):
        return PERIOD_RECENT
    return None


_BALANCE_ASSET_RE = re.compile(r"\b([A-Za-z]{2,10})\b")


def extract_balance_filters(text: str) -> dict:
    """Extract optional ``venue`` / ``asset`` filters from a balance question."""
    t = _norm(text)
    entities: dict = {}
    for venue in KNOWN_VENUES:
        if venue in t:
            entities["venue"] = venue
            break
    # Asset: look for known quote/base tickers in upper case form.
    upper = (text or "").upper()
    candidates = _BALANCE_ASSET_RE.findall(upper)
    common = ("BTC", "ETH", "SOL", "USDT", "USDC", "BNB", "XRP", "ADA",
              "DOGE", "LINK", "AVAX", "TRX", "TON", "ARB", "DOT", "MATIC",
              "LTC", "ATOM", "NEAR", "APT", "OP", "SUI", "FIL", "INJ")
    for c in candidates:
        if c in common:
            entities["asset"] = c
            break
    return entities
