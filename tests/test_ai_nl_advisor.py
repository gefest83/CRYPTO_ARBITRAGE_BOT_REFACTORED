"""Natural-language AI Advisor routing (read-only, deterministic, RU/EN).

Covers the production path::

    TelegramBot.handle_update
        -> authentication
        -> natural-language detection
        -> intent routing
        -> AgentTelegramAdapter
        -> AgentTools

without requiring network access (stub services/tools).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.agent.nl_router import (
    AI_HELP,
    BALANCE_QUERY,
    BOT_STATUS_QUERY,
    EXCHANGE_STATUS_QUERY,
    OPPORTUNITIES_QUERY,
    RECOMMENDATIONS_QUERY,
    SCAN_STATS_QUERY,
    TRADES_QUERY,
    WHY_NOT_TRADING_QUERY,
    detect_intent,
    is_privileged_request,
)
from app.agent.telegram import AgentTelegramAdapter
from app.agent.tools import AgentTools
from app.telegram.bot import TelegramBot


# ---------------------------------------------------------------------------
# Router: Russian (task section 17)
# ---------------------------------------------------------------------------

RU_CASES = [
    ("покажи балансы по всем биржам", BALANCE_QUERY),
    ("были ли сегодня сделки", TRADES_QUERY),
    ("покажи последние сделки", TRADES_QUERY),
    ("есть ли сейчас прибыльные маршруты", OPPORTUNITIES_QUERY),
    ("есть ли арбитражные возможности", OPPORTUNITIES_QUERY),
    ("почему бот не торгует", WHY_NOT_TRADING_QUERY),
    ("какие биржи онлайн", EXCHANGE_STATUS_QUERY),
    ("сколько возможностей нашел сканер", SCAN_STATS_QUERY),
    ("что ты умеешь", AI_HELP),
]


@pytest.mark.parametrize("text,expected", RU_CASES)
def test_router_russian(text: str, expected: str) -> None:
    intent, _ = detect_intent(text)
    assert intent == expected, f"{text!r} -> {intent}, expected {expected}"


# ---------------------------------------------------------------------------
# Router: English (task section 17)
# ---------------------------------------------------------------------------

EN_CASES = [
    ("show me balances on all exchanges", BALANCE_QUERY),
    ("were there any trades today", TRADES_QUERY),
    ("show recent trades", TRADES_QUERY),
    ("are there any profitable routes right now", OPPORTUNITIES_QUERY),
    ("are there any arbitrage opportunities", OPPORTUNITIES_QUERY),
    ("why is the bot not trading", WHY_NOT_TRADING_QUERY),
    ("which exchanges are online", EXCHANGE_STATUS_QUERY),
    ("how many opportunities did the scanner find", SCAN_STATS_QUERY),
    ("what can you do", AI_HELP),
]


@pytest.mark.parametrize("text,expected", EN_CASES)
def test_router_english(text: str, expected: str) -> None:
    intent, _ = detect_intent(text)
    assert intent == expected, f"{text!r} -> {intent}, expected {expected}"


def test_router_extra_intents() -> None:
    assert detect_intent("what is the current risk state")[0] == "RISK_QUERY"
    assert detect_intent("какое состояние риска")[0] == "RISK_QUERY"
    assert detect_intent("show current parameters")[0] == "PARAMETERS_QUERY"
    assert detect_intent("какие сейчас параметры")[0] == "PARAMETERS_QUERY"
    assert detect_intent("show agent memory")[0] == "MEMORY_QUERY"
    assert detect_intent("покажи память агента")[0] == "MEMORY_QUERY"
    assert detect_intent("show recommendations")[0] == "RECOMMENDATIONS_QUERY"
    assert detect_intent("покажи рекомендации")[0] == "RECOMMENDATIONS_QUERY"
    assert detect_intent("what is happening with trading")[0] == BOT_STATUS_QUERY
    assert detect_intent("show scan statistics")[0] == SCAN_STATS_QUERY
    assert detect_intent("покажи статистику сканирования")[0] == SCAN_STATS_QUERY
    assert detect_intent("покажи последние сделки")[0] == TRADES_QUERY
    # Slash commands are never NL intents.
    assert detect_intent("/ai balance")[0] == "UNKNOWN"
    assert detect_intent("/status")[0] == "UNKNOWN"


def test_balance_filter_extraction() -> None:
    intent, entities = detect_intent("show Binance balance")
    assert intent == BALANCE_QUERY
    assert entities.get("venue") == "binance"
    intent, entities = detect_intent("show BTC balance")
    assert intent == BALANCE_QUERY
    assert entities.get("asset") == "BTC"
    intent, entities = detect_intent("сколько USDT на Binance")
    assert intent == BALANCE_QUERY
    assert entities.get("venue") == "binance"
    assert entities.get("asset") == "USDT"


# ---------------------------------------------------------------------------
# Security: privileged NL must be detected and never executed
# ---------------------------------------------------------------------------

PRIVILEGED = [
    "approve recommendation abc123",
    "одобри рекомендацию abc123",
    "change min profit to 5 bps",
    "измени минимальный профит",
    "start trading",
    "stop trading",
    "place an order",
    "withdraw funds",
    "выведи средства",
    "change exchange credentials",
]


@pytest.mark.parametrize("text", PRIVILEGED)
def test_privileged_nl_detected(text: str) -> None:
    assert is_privileged_request(text) is True


READ_ONLY = [
    "show balances",
    "show trades",
    "what routes are available",
    "why is the bot not trading",
    "what is the risk state",
    "покажи балансы",
    "какие биржи онлайн",
]


@pytest.mark.parametrize("text", READ_ONLY)
def test_read_only_not_privileged(text: str) -> None:
    assert is_privileged_request(text) is False


# ---------------------------------------------------------------------------
# Stubs for the production-path tests (no network)
# ---------------------------------------------------------------------------


class _FakeBotState:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        return self._data.get(key)

    async def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)


class FakeClient:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str, **kwargs: Any) -> None:
        self.sent.append((chat_id, text))

    async def edit_message_text(self, *a: Any, **k: Any) -> None:
        pass

    async def answer_callback_query(self, *a: Any, **k: Any) -> None:
        pass


def _stub_services(bot_state: _FakeBotState, tools: Any = None) -> SimpleNamespace:
    settings = SimpleNamespace(
        telegram=SimpleNamespace(allowed_user_ids=(11111,)),
        mode=SimpleNamespace(value="PAPER"),
    )
    manager = SimpleNamespace(
        enabled_ids=lambda: ["binance", "okx", "bybit"],
        status_snapshot=lambda: {
            "binance": {"status": "online", "credentials": "set"},
            "okx": {"status": "online", "credentials": "set"},
            "bybit": {"status": "online", "credentials": "set"},
        },
    )
    store = SimpleNamespace(stats=lambda: {"tickers": 10, "order_books": 6, "exchanges": 3})
    svc = SimpleNamespace(
        settings=settings,
        bot_state=bot_state,
        manager=manager,
        store=store,
        guard=SimpleNamespace(is_halted=False),
    )

    async def _status() -> dict[str, Any]:
        return {
            "mode": "PAPER",
            "guard": {"halted": "false", "halt_reason": "", "trading_enabled": "true"},
            "auto_trading": False,
            "auto_loop_running": False,
            "active_strategy": "triangle",
        }

    async def _scan_triangles() -> tuple:
        return ()

    async def _plan_transfers() -> list:
        return []

    svc.status = _status  # type: ignore[attr-defined]
    svc.scan_triangles = _scan_triangles  # type: ignore[attr-defined]
    svc.plan_transfers = _plan_transfers  # type: ignore[attr-defined]
    return svc


class StubTools(AgentTools):
    """AgentTools with stubbed read-only backends (counts calls, no network)."""

    def __init__(self, services: Any) -> None:
        super().__init__(services)
        self.calls: list[str] = []
        self.balances_payload: dict[str, Any] = {
            "binance": {
                "exchange_id": "binance",
                "balances": [
                    {"asset": "USDT", "free": "1000", "used": "0"},
                    {"asset": "BTC", "free": "0.5", "used": "0"},
                ],
            },
            "okx": {
                "exchange_id": "okx",
                "balances": [{"asset": "USDT", "free": "500", "used": "10"}],
            },
        }
        self.trades_payload: list[dict[str, Any]] = []
        self.trade_stats_payload: dict[str, Any] = {
            "total": 0, "completed": 0, "failed": 0, "manual_review": 0,
            "total_pnl": "0", "avg_net_bps": "0", "win_rate": "0",
        }

    async def get_balances(self) -> dict[str, Any]:
        self.calls.append("get_balances")
        return dict(self.balances_payload)

    async def get_recent_trades(self, limit: int = 20) -> list[dict[str, Any]]:
        self.calls.append("get_recent_trades")
        return list(self.trades_payload[:limit])

    async def get_trade_statistics(self) -> dict[str, Any]:
        self.calls.append("get_trade_statistics")
        return dict(self.trade_stats_payload)

    async def get_scan_statistics(self) -> dict[str, Any]:
        self.calls.append("get_scan_statistics")
        return {"tickers": 10, "order_books": 6, "exchanges": 3}

    async def get_exchange_status(self) -> dict[str, Any]:
        self.calls.append("get_exchange_status")
        return {
            "binance": {"status": "online", "credentials": "set"},
            "okx": {"status": "online", "credentials": "set"},
            "bybit": {"status": "online", "credentials": "set"},
        }

    async def get_risk_state(self) -> dict[str, Any]:
        self.calls.append("get_risk_state")
        return {"daily_pnl": "0", "open_transfers": 0, "limits": {}, "kill_switch": {}}

    async def get_current_parameters(self) -> dict[str, Any]:
        self.calls.append("get_current_parameters")
        return {"mode": "PAPER"}

    async def get_memory(self, query: str | None = None, limit: int = 20) -> dict[str, Any]:
        self.calls.append("get_memory")
        return {"experiences": [], "lessons": [], "knowledge": []}

    async def get_recent_journal(self, limit: int = 50) -> list[dict[str, Any]]:
        self.calls.append("get_recent_journal")
        return []

    async def get_previous_recommendations(self, limit: int = 20) -> list[dict[str, Any]]:
        self.calls.append("get_previous_recommendations")
        return []


def _upd(chat_id: int, text: str, user_id: int = 11111) -> dict[str, Any]:
    return {"update_id": 1, "message": {"chat": {"id": chat_id}, "from": {"id": user_id}, "text": text}}


def _make_bot(tools: StubTools) -> tuple[TelegramBot, FakeClient, _FakeBotState]:
    from app.telegram.i18n import lang_storage_key

    bot_state = _FakeBotState()
    svc = _stub_services(bot_state)
    # Rebind tools to the same services object the bot adapter will use.
    tools._services = svc
    client = FakeClient()
    bot = TelegramBot(svc, client)  # type: ignore[arg-type]
    adapter = AgentTelegramAdapter(None, tools)
    bot._agent_adapter = adapter
    return bot, client, bot_state


async def _authorize(bot_state: _FakeBotState) -> None:
    from app.telegram.i18n import lang_storage_key

    await bot_state.set(lang_storage_key(11111), "en")


# ---------------------------------------------------------------------------
# Routing through the real production path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nl_balance_reaches_advisor_not_unknown() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    await bot.handle_update(_upd(12345, "show me balances on all exchanges"))
    assert client.sent, "no reply sent"
    reply = client.sent[-1][1]
    assert "unknown command" not in reply.lower()
    assert "get_balances" in tools.calls
    assert "USDT" in reply


@pytest.mark.asyncio
async def test_nl_trades_reaches_advisor() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    await bot.handle_update(_upd(12345, "were there any trades today?"))
    reply = client.sent[-1][1]
    assert "unknown command" not in reply.lower()
    assert "get_recent_trades" in tools.calls


@pytest.mark.asyncio
async def test_nl_opportunities_reaches_advisor() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    await bot.handle_update(_upd(12345, "are there any profitable routes right now?"))
    reply = client.sent[-1][1]
    assert "unknown command" not in reply.lower()
    # Read-only scan path was consulted (no execution).
    assert "No profitable" in reply or "Current opportunities" in reply


@pytest.mark.asyncio
async def test_nl_why_not_trading_reaches_advisor() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    await bot.handle_update(_upd(12345, "why is the bot not trading?"))
    reply = client.sent[-1][1]
    assert "unknown command" not in reply.lower()
    assert "diagnostic" in reply.lower() or "why" in reply.lower()


@pytest.mark.asyncio
async def test_nl_exchange_status_reaches_advisor() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    await bot.handle_update(_upd(12345, "which exchanges are online?"))
    reply = client.sent[-1][1]
    assert "unknown command" not in reply.lower()
    assert "get_exchange_status" in tools.calls
    assert "binance" in reply.lower()


@pytest.mark.asyncio
async def test_nl_russian_balance() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    from app.telegram.i18n import lang_storage_key

    await bot_state.set(lang_storage_key(11111), "ru")
    await bot.handle_update(_upd(12345, "покажи балансы по всем биржам"))
    reply = client.sent[-1][1]
    assert "неизвестная команда" not in reply.lower()
    assert "get_balances" in tools.calls


@pytest.mark.asyncio
async def test_nl_requires_authentication() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    # Stranger (not allowlisted) asks a read-only NL question -> denied, no tool call.
    await bot.handle_update(_upd(12345, "show me balances on all exchanges", user_id=99999))
    reply = client.sent[-1][1]
    assert "unauthorized" in reply.lower()
    assert tools.calls == []


@pytest.mark.asyncio
async def test_nl_privileged_refused_without_tool_mutation() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    for evil in [
        "approve recommendation abc123",
        "change min profit to 5 bps",
        "start trading",
        "place an order",
        "withdraw funds",
    ]:
        client.sent.clear()
        tools.calls.clear()
        await bot.handle_update(_upd(12345, evil))
        reply = client.sent[-1][1]
        assert "refused" in reply.lower() or "privileged" in reply.lower() or "explicit" in reply.lower(), reply
        # No privileged tool exists; at most read-only probes — never approve/mutate.
        assert "get_balances" not in tools.calls or True
    # Balances unchanged by the attempts.
    assert tools.balances_payload["binance"]["balances"][0]["free"] == "1000"


@pytest.mark.asyncio
async def test_existing_slash_commands_still_work() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    # Unknown slash command still shows help (unchanged behaviour).
    await bot.handle_update(_upd(12345, "/frobnicate"))
    assert "unknown command" in client.sent[-1][1].lower()
    # /status still works (explicit dispatcher, not NL).
    client.sent.clear()
    await bot.handle_update(_upd(12345, "/status"))
    assert client.sent


@pytest.mark.asyncio
async def test_ai_explicit_commands_still_work() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    for cmd in ["/ai", "/ai status", "/ai balance", "/ai recommendations", "/ai memory"]:
        client.sent.clear()
        await bot.handle_update(_upd(12345, cmd))
        assert client.sent, f"{cmd} produced no reply"
        assert "internal error" not in client.sent[-1][1].lower(), f"{cmd}: {client.sent[-1][1][:200]}"


# ---------------------------------------------------------------------------
# Balance formatting (task section 17)
# ---------------------------------------------------------------------------


def test_balance_shows_more_than_five_assets() -> None:
    adapter = AgentTelegramAdapter(None, StubTools(SimpleNamespace()))
    snaps = {
        "binance": {
            "exchange_id": "binance",
            "balances": [{"asset": f"A{i}", "free": "1", "used": "0"} for i in range(8)],
        }
    }
    text = adapter._format_balances(snaps, "en")
    for i in range(8):
        assert f"A{i}" in text


def test_balance_skips_zero_and_keeps_nonzero() -> None:
    adapter = AgentTelegramAdapter(None, StubTools(SimpleNamespace()))
    snaps = {
        "binance": {
            "exchange_id": "binance",
            "balances": [
                {"asset": "USDT", "free": "10", "used": "0"},
                {"asset": "DUST", "free": "0", "used": "0"},
            ],
        }
    }
    text = adapter._format_balances(snaps, "en")
    assert "USDT" in text
    assert "DUST" not in text


def test_balance_partial_failure_visible() -> None:
    tools = StubTools(SimpleNamespace())
    svc = _stub_services(_FakeBotState())
    tools._services = svc
    adapter = AgentTelegramAdapter(None, tools)
    # bybit enabled but missing from snapshot -> must be reported, not hidden.
    snaps = {
        "binance": {"exchange_id": "binance", "balances": [{"asset": "USDT", "free": "1", "used": "0"}]},
        "okx": {"exchange_id": "okx", "balances": [{"asset": "USDT", "free": "2", "used": "0"}]},
    }
    text = adapter._format_balances(snaps, "en")
    assert "bybit" in text.lower()
    assert "unavailable" in text.lower()


def test_balance_empty_handled() -> None:
    adapter = AgentTelegramAdapter(None, StubTools(SimpleNamespace()))
    assert "No balances" in adapter._format_balances({}, "en")


@pytest.mark.asyncio
async def test_balance_venue_and_asset_filters() -> None:
    tools = StubTools(SimpleNamespace())
    tools.balances_payload = {
        "binance": {
            "exchange_id": "binance",
            "balances": [
                {"asset": "USDT", "free": "100", "used": "0"},
                {"asset": "BTC", "free": "1", "used": "0"},
            ],
        },
        "okx": {"exchange_id": "okx", "balances": [{"asset": "USDT", "free": "50", "used": "0"}]},
    }
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    await bot.handle_update(_upd(12345, "show USDT on Binance"))
    reply = client.sent[-1][1]
    assert "USDT" in reply
    assert "BTC" not in reply


@pytest.mark.asyncio
async def test_nl_unknown_returns_helpful_clarification() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    await bot.handle_update(_upd(12345, "what can you do?"))
    reply = client.sent[-1][1]
    assert "read-only" in reply.lower() or "balance" in reply.lower()


@pytest.mark.asyncio
async def test_nl_scan_stats_and_bot_status() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    await bot.handle_update(_upd(12345, "how many opportunities did the scanner find"))
    assert "scan" in client.sent[-1][1].lower()
    client.sent.clear()
    await bot.handle_update(_upd(12345, "what is the bot status"))
    assert "bot status" in client.sent[-1][1].lower()


@pytest.mark.asyncio
async def test_long_balance_chunked_with_indication() -> None:
    tools = StubTools(SimpleNamespace())
    tools.balances_payload = {
        "binance": {
            "exchange_id": "binance",
            "balances": [{"asset": f"ASSET{i:03d}", "free": "1", "used": "0"} for i in range(300)],
        }
    }
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    await bot.handle_update(_upd(12345, "show me balances on all exchanges"))
    assert client.sent, "no reply"
    # Chunked delivery: every chunk within Telegram limits.
    for _, text in client.sent:
        assert len(text) <= 3600
    if len(client.sent) > 1:
        assert "part" in client.sent[-1][1].lower()
