"""Hardening for the natural-language AI Advisor.

* BOT_OPERATION_QUERY routes to a real pipeline explanation (RU/EN);
* TRADES_QUERY honors strict time windows (today / last hour);
* diagnostics use cheap runtime state (no expensive fresh scans);
* opportunities run exactly one strategy-scoped calculation, timed;
* the reported threshold comes from the active runtime path;
* privileged NL actions stay refused; existing behavior is preserved.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app.agent.nl_router import (
    BOT_OPERATION_QUERY,
    BOT_STATUS_QUERY,
    TRADES_QUERY,
    detect_intent,
    extract_trade_period,
    is_privileged_request,
)
from app.agent.telegram import AgentTelegramAdapter
from app.telegram.bot import TelegramBot

from tests.test_ai_nl_advisor import FakeClient, StubTools, _authorize, _make_bot, _upd


# ---------------------------------------------------------------------------
# Router: BOT_OPERATION_QUERY (RU/EN) + STATUS separation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "как торгует наш бот?",
        "как работает бот?",
        "какие стратегии использует бот?",
        "как бот выбирает сделки?",
        "как он ищет арбитраж?",
        "как происходит торговый цикл?",
    ],
)
def test_router_bot_operation_ru(text: str) -> None:
    assert detect_intent(text)[0] == BOT_OPERATION_QUERY


@pytest.mark.parametrize(
    "text",
    [
        "how does the bot trade?",
        "how does the bot work?",
        "what strategies does the bot use?",
        "how does it find arbitrage?",
        "how does the trading cycle work?",
        "how are routes selected?",
    ],
)
def test_router_bot_operation_en(text: str) -> None:
    assert detect_intent(text)[0] == BOT_OPERATION_QUERY


@pytest.mark.parametrize(
    "text",
    [
        "что сейчас делает бот?",
        "какая стратегия сейчас активна?",
        "is the bot currently trading?",
        "что сейчас происходит с ботом?",
    ],
)
def test_router_current_state_is_status_not_operation(text: str) -> None:
    assert detect_intent(text)[0] == BOT_STATUS_QUERY


def test_trade_period_entities() -> None:
    assert extract_trade_period("были ли сделки сегодня?") == "today"
    assert extract_trade_period("were there any trades today?") == "today"
    assert extract_trade_period("сделки за последний час") == "last_hour"
    assert extract_trade_period("trades in the last hour") == "last_hour"
    assert extract_trade_period("покажи последние сделки") == "recent"
    assert detect_intent("были ли сделки сегодня?") == (TRADES_QUERY, {"period": "today"})


# ---------------------------------------------------------------------------
# Helpers: trades with controlled timestamps
# ---------------------------------------------------------------------------


def _trade(ts: datetime, route: str = "USDT->BTC->ETH->USDT") -> dict[str, Any]:
    return {
        "strategy": "triangle",
        "route": route,
        "status": "completed",
        "net_profit": "1.5",
        "created_at": ts.isoformat(),
    }


@pytest.mark.asyncio
async def test_today_excludes_old_trades_ru() -> None:
    tools = StubTools(SimpleNamespace())
    now = datetime.now(UTC)
    tools.trades_payload = [
        _trade(now - timedelta(hours=1), "TODAY-ROUTE"),
        _trade(datetime(2026, 9, 3, 12, 0, tzinfo=UTC), "OLD-ROUTE"),
    ]
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    await bot.handle_update(_upd(12345, "были ли сделки сегодня?"))
    reply = client.sent[-1][1]
    assert "TODAY-ROUTE" in reply
    assert "OLD-ROUTE" not in reply
    assert "unknown command" not in reply.lower()


@pytest.mark.asyncio
async def test_today_excludes_old_trades_en() -> None:
    tools = StubTools(SimpleNamespace())
    now = datetime.now(UTC)
    tools.trades_payload = [
        _trade(now - timedelta(minutes=30), "TODAY-ROUTE"),
        _trade(datetime(2026, 9, 3, 12, 0, tzinfo=UTC), "OLD-ROUTE"),
    ]
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    await bot.handle_update(_upd(12345, "were there any trades today?"))
    reply = client.sent[-1][1]
    assert "TODAY-ROUTE" in reply
    assert "OLD-ROUTE" not in reply


@pytest.mark.asyncio
async def test_today_empty_says_so_explicitly() -> None:
    tools = StubTools(SimpleNamespace())
    tools.trades_payload = [_trade(datetime(2026, 9, 3, 12, 0, tzinfo=UTC), "OLD-ROUTE")]
    bot, client, bot_state = _make_bot(tools)
    from app.telegram.i18n import lang_storage_key

    await bot_state.set(lang_storage_key(11111), "ru")
    await bot.handle_update(_upd(12345, "были ли сделки сегодня?"))
    reply = client.sent[-1][1]
    assert "OLD-ROUTE" not in reply
    assert "Сегодня сделок не было" in reply


@pytest.mark.asyncio
async def test_last_hour_window() -> None:
    tools = StubTools(SimpleNamespace())
    now = datetime.now(UTC)
    tools.trades_payload = [
        _trade(now - timedelta(minutes=10), "FRESH-ROUTE"),
        _trade(now - timedelta(hours=5), "STALE-ROUTE"),
    ]
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    await bot.handle_update(_upd(12345, "trades in the last hour"))
    reply = client.sent[-1][1]
    assert "FRESH-ROUTE" in reply
    assert "STALE-ROUTE" not in reply


@pytest.mark.asyncio
async def test_recent_trades_keep_history() -> None:
    tools = StubTools(SimpleNamespace())
    tools.trades_payload = [_trade(datetime(2026, 9, 3, 12, 0, tzinfo=UTC), "OLD-ROUTE")]
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    for text in ("покажи последние сделки", "show recent trades"):
        client.sent.clear()
        await bot.handle_update(_upd(12345, text))
        assert "OLD-ROUTE" in client.sent[-1][1]


# ---------------------------------------------------------------------------
# Bot explanation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bot_operation_returns_real_explanation() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    from app.telegram.i18n import lang_storage_key

    await bot_state.set(lang_storage_key(11111), "ru")
    await bot.handle_update(_upd(12345, "как торгует наш бот?"))
    reply = client.sent[-1][1]
    assert "unknown command" not in reply.lower()
    for marker in ("стратегия", "MarketData", "Scanner", "Risk", "AutoTrader", "DEMO"):
        assert marker.lower() in reply.lower(), f"missing {marker}: {reply[:300]}"


@pytest.mark.asyncio
async def test_bot_operation_en_names_pipeline() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    await bot.handle_update(_upd(12345, "what strategies does the bot use?"))
    reply = client.sent[-1][1]
    assert "triangle" in reply.lower()
    assert "transfer" in reply.lower()


# ---------------------------------------------------------------------------
# Diagnostics use cheap state (no expensive scans)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_why_uses_diagnostic_data() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    # Stub status(): auto-trading OFF -> must surface as a blocker.
    await bot.handle_update(_upd(12345, "почему бот не торгует?"))
    reply = client.sent[-1][1]
    assert "unknown command" not in reply.lower()
    assert "AUTO" in reply.upper() or "автоторгов" in reply.lower()


@pytest.mark.asyncio
async def test_why_does_not_call_expensive_scan() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    svc = tools._services
    calls: list[str] = []

    async def _boom_triangles(*a: Any, **k: Any) -> Any:
        calls.append("scan_triangles")
        raise AssertionError("expensive scan must not run in diagnostics")

    async def _boom_plans(*a: Any, **k: Any) -> Any:
        calls.append("plan_transfers")
        raise AssertionError("expensive planner must not run in diagnostics")

    svc.scan_triangles = _boom_triangles  # type: ignore[attr-defined]
    svc.plan_transfers = _boom_plans  # type: ignore[attr-defined]
    await bot.handle_update(_upd(12345, "why is the bot not trading?"))
    assert calls == []
    assert "threshold" in client.sent[-1][1].lower() or "порог" in client.sent[-1][1].lower()


# ---------------------------------------------------------------------------
# Opportunities: scoped, timed, threshold-consistent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opportunities_scoped_to_active_strategy() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    svc = tools._services
    calls: list[str] = []

    async def _scan(*a: Any, **k: Any) -> tuple:
        calls.append("scan_triangles")
        return ()

    async def _plan(*a: Any, **k: Any) -> list:
        calls.append("plan_transfers")
        raise AssertionError("transfer planner must not run for triangle strategy")

    async def _active() -> str:
        return "triangle"

    svc.scan_triangles = _scan  # type: ignore[attr-defined]
    svc.plan_transfers = _plan  # type: ignore[attr-defined]
    svc.get_active_strategy = _active  # type: ignore[attr-defined]
    await bot.handle_update(_upd(12345, "are there any profitable routes right now?"))
    assert calls == ["scan_triangles"]
    assert "unknown command" not in client.sent[-1][1].lower()


@pytest.mark.asyncio
async def test_opportunities_stale_without_scan() -> None:
    tools = StubTools(SimpleNamespace())

    async def _stats() -> dict[str, Any]:
        return {"tickers": 0, "order_books": 0, "exchanges": 0}

    tools.get_scan_statistics = _stats  # type: ignore[method-assign]
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    svc = tools._services

    async def _boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("no scan on stale market data")

    svc.scan_triangles = _boom  # type: ignore[attr-defined]
    svc.plan_transfers = _boom  # type: ignore[attr-defined]
    await bot.handle_update(_upd(12345, "есть сейчас прибыльные маршруты?"))
    reply = client.sent[-1][1]
    assert "stale" in reply.lower() or "устаревш" in reply.lower() or "неполн" in reply.lower()


@pytest.mark.asyncio
async def test_threshold_matches_active_strategy() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    svc = tools._services
    svc.settings.arbitrage = SimpleNamespace(triangle_min_net_bps="5")
    svc.settings.transfer = SimpleNamespace(min_net_profit_bps="50")
    svc.settings.risk = SimpleNamespace(min_net_profit_bps="10")

    async def _active() -> str:
        return "transfer"

    async def _plan(*a: Any, **k: Any) -> list:
        return []

    svc.get_active_strategy = _active  # type: ignore[attr-defined]
    svc.plan_transfers = _plan  # type: ignore[attr-defined]
    await bot.handle_update(_upd(12345, "есть сейчас прибыльные маршруты?"))
    reply = client.sent[-1][1]
    # Transfer path governs: 50 bps from transfer.min_net_profit_bps — not 5, not 10.
    assert "50" in reply
    assert "transfer.min_net_profit_bps" in reply


# ---------------------------------------------------------------------------
# Performance: lightweight intents never touch scanner/planner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lightweight_intents_skip_scanner() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    svc = tools._services
    calls: list[str] = []

    async def _boom_scan(*a: Any, **k: Any) -> Any:
        calls.append("scan_triangles")
        raise AssertionError("scanner must not run")

    async def _boom_plan(*a: Any, **k: Any) -> Any:
        calls.append("plan_transfers")
        raise AssertionError("planner must not run")

    svc.scan_triangles = _boom_scan  # type: ignore[attr-defined]
    svc.plan_transfers = _boom_plan  # type: ignore[attr-defined]
    for text in [
        "show me balances on all exchanges",
        "were there any trades today?",
        "show recent trades",
        "how many opportunities did the scanner find",
        "which exchanges are online?",
        "what is the bot status",
        "why is the bot not trading?",
        "как торгует наш бот?",
    ]:
        client.sent.clear()
        await bot.handle_update(_upd(12345, text))
        assert client.sent, f"no reply for {text!r}"
        assert "internal error" not in client.sent[-1][1].lower()
    assert calls == []


@pytest.mark.asyncio
async def test_nl_request_timing_logged(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    with caplog.at_level(logging.INFO, logger="cat.agent.telegram"):
        await bot.handle_update(_upd(12345, "show recent trades"))
    records = [r for r in caplog.records if r.msg == "ai_nl_request"]
    assert records, "ai_nl_request timing log missing"
    assert records[0].__dict__.get("intent") == "TRADES_QUERY"
    assert isinstance(records[0].__dict__.get("duration_ms"), int)


# ---------------------------------------------------------------------------
# Security + regression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nl_privileged_still_refused() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    for evil in [
        "approve recommendation abc123",
        "change min profit to 5 bps",
        "start trading",
        "withdraw funds",
    ]:
        client.sent.clear()
        await bot.handle_update(_upd(12345, evil))
        reply = client.sent[-1][1]
        assert "refused" in reply.lower() or "privileged" in reply.lower()
    assert not is_privileged_request("как торгует наш бот?")
    assert not is_privileged_request("были ли сделки сегодня?")


@pytest.mark.asyncio
async def test_explicit_ai_commands_unchanged() -> None:
    tools = StubTools(SimpleNamespace())
    bot, client, bot_state = _make_bot(tools)
    await _authorize(bot_state)
    for cmd in ["/ai", "/ai status", "/ai balance", "/ai recommendations", "/ai memory"]:
        client.sent.clear()
        await bot.handle_update(_upd(12345, cmd))
        assert client.sent, f"{cmd} produced no reply"
        assert "internal error" not in client.sent[-1][1].lower()


@pytest.mark.asyncio
async def test_nine_manual_questions_route_correctly() -> None:
    tools = StubTools(SimpleNamespace())
    tools.trades_payload = [_trade(datetime.now(UTC) - timedelta(minutes=5), "FRESH")]
    bot, client, bot_state = _make_bot(tools)
    from app.telegram.i18n import lang_storage_key

    await bot_state.set(lang_storage_key(11111), "ru")
    for text in [
        "покажи мне балансы по всем биржам",
        "были ли сделки сегодня?",
        "покажи последние сделки",
        "есть сейчас прибыльные маршруты?",
        "почему бот не торгует?",
        "как торгует наш бот?",
        "какие стратегии использует бот?",
        "что сейчас происходит с ботом?",
        "какие биржи онлайн?",
    ]:
        client.sent.clear()
        await bot.handle_update(_upd(12345, text))
        assert client.sent, f"no reply for {text!r}"
        reply = client.sent[-1][1]
        assert "unknown command" not in reply.lower(), text
        assert "неизвестная команда" not in reply.lower(), text
