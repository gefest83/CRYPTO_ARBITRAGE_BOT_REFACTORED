"""Regression for NL routing UX: small-talk and recommendation must use LLM, not fast-path dumps."""
import asyncio
from types import SimpleNamespace

import pytest

from app.agent.providers.base import LLMProvider, LLMRequest, LLMResponse, LLMToolCall, NullProvider
from app.agent.telegram import AgentTelegramAdapter
from app.agent.tools import AgentTools


class StubTools(AgentTools):
    def __init__(self, services):
        super().__init__(services)
        self.calls: list[str] = []

    async def get_balances(self):
        self.calls.append("get_balances")
        return {"binance": {"exchange_id": "binance", "balances": [{"asset": "USDT", "free": "1000", "used": "0"}]}}

    async def get_recent_trades(self, limit=20):
        self.calls.append("get_recent_trades")
        return []

    async def get_trade_statistics(self):
        self.calls.append("get_trade_statistics")
        return {"total": 5, "completed": 2, "failed": 3, "manual_review": 0, "total_pnl": "-10", "avg_net_bps": "-5", "win_rate": "40"}

    async def get_scan_statistics(self):
        self.calls.append("get_scan_statistics")
        return {"tickers": 10, "order_books": 6, "exchanges": 3}

    async def get_exchange_status(self):
        self.calls.append("get_exchange_status")
        return {"binance": {"status": "online", "credentials": "set"}}

    async def get_risk_state(self):
        self.calls.append("get_risk_state")
        return {"daily_pnl": "0", "open_transfers": 0, "limits": {}, "kill_switch": {}}

    async def get_current_parameters(self):
        self.calls.append("get_current_parameters")
        return {"mode": "PAPER"}

    async def get_memory(self, query=None, limit=20):
        self.calls.append("get_memory")
        return {"experiences": [], "lessons": [], "knowledge": []}

    async def get_recent_journal(self, limit=50):
        self.calls.append("get_recent_journal")
        return []

    async def get_previous_recommendations(self, limit=20):
        self.calls.append("get_previous_recommendations")
        return []


class FakeLLM(LLMProvider):
    name = "fake"
    supports_tool_calling = True

    async def complete(self, request: LLMRequest) -> LLMResponse:
        has_tools = request.tools is not None
        user_text = " ".join(m.content for m in request.messages if m.role == "user").lower()
        if "как зовут" in user_text or "привет" in user_text:
            return LLMResponse(content="Привет! Я — AI-советник бота. Готов помочь с балансами, сделками и советами по улучшению.", model="fake")
        if has_tools and len(request.messages) == 2:
            return LLMResponse(
                content="",
                model="fake",
                tool_calls=(
                    LLMToolCall(id="1", name="get_trade_statistics", arguments={}, raw_arguments="{}"),
                    LLMToolCall(id="2", name="get_current_parameters", arguments={}, raw_arguments="{}"),
                    LLMToolCall(id="3", name="get_scan_statistics", arguments={}, raw_arguments="{}"),
                ),
            )
        return LLMResponse(content="Рекомендация: на основе данных — 3 failed из 5, порог 5 bps (arbitrage.triangle_min_net_bps), риск 10 bps.", model="fake")


def _svc():
    svc = SimpleNamespace(
        settings=SimpleNamespace(
            arbitrage=SimpleNamespace(triangle_min_net_bps="5"),
            transfer=SimpleNamespace(min_net_profit_bps="50"),
            risk=SimpleNamespace(min_net_profit_bps="10"),
        ),
        manager=SimpleNamespace(enabled_ids=lambda: ["binance"], status_snapshot=lambda: {}),
        store=SimpleNamespace(stats=lambda: {"tickers": 10, "order_books": 6, "exchanges": 3}),
    )

    async def status():
        return {"mode": "PAPER", "guard": {"halted": "false", "halt_reason": "", "trading_enabled": "true"}, "auto_trading": False, "auto_loop_running": False, "active_strategy": "triangle", "market_data": {"order_books": 6}}

    svc.status = status

    async def get_active():
        return "triangle"

    svc.get_active_strategy = get_active
    return svc


class FakeCollector:
    async def collect(self, query=None, language=None, include_balances=True):
        from app.agent.context import AgentContext

        return AgentContext(recent_trades=[], trade_statistics={}, experiences=[], lessons=[], knowledge_hits=[], risk_state={}, exchange_status={}, current_parameters={}, previous_recommendations=[], journal_analysis=[])


@pytest.mark.asyncio
async def test_small_talk_uses_llm_not_technical_help():
    svc = _svc()
    tools = StubTools(svc)
    from app.agent.core import AgentCore

    core = AgentCore(collector=FakeCollector(), llm=FakeLLM(), audit_repo=None)
    adapter = AgentTelegramAdapter(core, tools)
    ans = await adapter.handle_natural_language("привет, как зовут тебя?", lang="ru")
    # Should be natural, not technical capability list
    assert "Привет!" in ans
    assert "Могу отвечать" not in ans
    assert "read-only" not in ans.lower()
    # Should not have dumped trades
    assert "get_recent_trades" not in tools.calls


@pytest.mark.asyncio
async def test_small_talk_natural_even_without_llm():
    svc = _svc()
    tools = StubTools(svc)
    from app.agent.core import AgentCore

    core = AgentCore(collector=FakeCollector(), llm=NullProvider(), audit_repo=None)
    adapter = AgentTelegramAdapter(core, tools)
    ans = await adapter.handle_natural_language("привет, как зовут тебя?", lang="ru")
    assert "Привет!" in ans
    assert "Могу отвечать" not in ans


@pytest.mark.asyncio
async def test_recommendation_uses_llm_grounded_not_trades_dump():
    svc = _svc()
    tools = StubTools(svc)
    from app.agent.core import AgentCore

    core = AgentCore(collector=FakeCollector(), llm=FakeLLM(), audit_repo=None)
    adapter = AgentTelegramAdapter(core, tools)
    ans = await adapter.handle_natural_language("как улучшить чтоб были сделки?", lang="ru")
    # Must be grounded via tools, not just recent trades dump
    assert "5 bps" in ans or "порог" in ans
    assert "get_trade_statistics" in tools.calls or "get_current_parameters" in tools.calls
    # Must not be the simple trades list
    assert "Recent trades" not in ans
    assert "Последние сделки" not in ans or "порог" in ans  # if it contains trades, it must also be grounded


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "как улучшить, чтоб были сделки?",
        "как улучшить бота, чтоб были сделки?",
        "что мешает боту торговать?",
        "что можно сделать, чтобы было больше сделок?",
    ],
)
async def test_llm_routing_for_analysis_recommendation(query: str):
    svc = _svc()
    tools = StubTools(svc)
    from app.agent.core import AgentCore

    core = AgentCore(collector=FakeCollector(), llm=FakeLLM(), audit_repo=None)
    adapter = AgentTelegramAdapter(core, tools)
    ans = await adapter.handle_natural_language(query, lang="ru")
    # Must go via LLM tool-calling, grounded in tool results, not a plain trades dump
    assert "5 bps" in ans or "порог" in ans, f"not grounded: {ans[:200]}"
    assert len(tools.calls) > 0, "LLM did not use read-only tools"
    assert "get_trade_statistics" in tools.calls or "get_current_parameters" in tools.calls or "get_scan_statistics" in tools.calls
    # Must not be just the 10-trades dump
    assert "Последние сделки" not in ans or "порог" in ans
    assert "Recent trades" not in ans


@pytest.mark.asyncio
async def test_obvious_read_only_still_fast_path():
    svc = _svc()
    tools = StubTools(svc)
    from app.agent.core import AgentCore

    core = AgentCore(collector=FakeCollector(), llm=FakeLLM(), audit_repo=None)
    adapter = AgentTelegramAdapter(core, tools)
    ans = await adapter.handle_natural_language("покажи балансы", lang="ru")
    assert "get_balances" in tools.calls
    assert "USDT" in ans


@pytest.mark.asyncio
async def test_privileged_still_blocked():
    svc = _svc()
    tools = StubTools(svc)
    from app.agent.core import AgentCore

    core = AgentCore(collector=FakeCollector(), llm=FakeLLM(), audit_repo=None)
    adapter = AgentTelegramAdapter(core, tools)
    ans = await adapter.handle_natural_language("зайди на OKX и продай 100 OKB на USDT", lang="ru")
    assert "Отклонено" in ans or "Refused" in ans
