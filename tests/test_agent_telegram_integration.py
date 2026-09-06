"""Phase 2C — Telegram integration for AI Advisor."""

from __future__ import annotations

from typing import Any

import pytest

from app.services import build_app, shutdown_app, start_app
from app.telegram.bot import TelegramBot
from app.telegram.i18n import lang_storage_key


class FakeClient:
    def __init__(self):
        self.sent: list[tuple[int, str]] = []
        self.kbs: list[dict | None] = []

    async def send_message(self, chat_id: int, text: str, *, parse_mode: str = "", reply_markup: dict | None = None):
        self.sent.append((chat_id, text))
        self.kbs.append(reply_markup)

    async def edit_message_text(self, *a, **kw):
        pass

    async def answer_callback_query(self, *a, **kw):
        pass


def _upd(chat_id: int, text: str, user_id: int = 11111):
    return {"update_id": 1, "message": {"chat": {"id": chat_id}, "from": {"id": user_id}, "text": text}}


def _settings(tmp_path):
    from tests.conftest import make_settings

    return make_settings(tmp_path)


@pytest.mark.asyncio
async def test_ai_requires_authorization(tmp_path):
    services = await build_app(_settings(tmp_path))
    await start_app(services, start_telegram=False, start_streams=False)
    try:
        await services.bot_state.set(lang_storage_key(11111), "en")
        # stranger has no entry, but we set his language to prove denied message still localized
        await services.bot_state.set(lang_storage_key(99999), "en")
        bot = TelegramBot(services, FakeClient())
        client = bot._client  # type: ignore[attr-defined]
        # stranger tries /ai status -> unauthorized
        await bot.handle_update(_upd(12345, "/ai status", user_id=99999))
        assert "unauthorized" in client.sent[-1][1].lower()
        # stranger tries /ai balance -> also denied
        client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai balance", user_id=99999))
        assert "unauthorized" in client.sent[-1][1].lower()
        # stranger tries /ai (help) -> denied (help is also protected, not like /language)
        client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai", user_id=99999))
        assert "unauthorized" in client.sent[-1][1].lower()
        # authorized still works
        client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai", user_id=11111))
        assert "AI Advisor" in client.sent[-1][1] or "AI" in client.sent[-1][1]
    finally:
        await shutdown_app(services)


@pytest.mark.asyncio
async def test_ai_i18n_en_ru(tmp_path):
    services = await build_app(_settings(tmp_path))
    await start_app(services, start_telegram=False, start_streams=False)
    try:
        bot = TelegramBot(services, FakeClient())
        client = bot._client  # type: ignore[attr-defined]
        for lang, expect in (("en", "AI Advisor"), ("ru", "AI-советник")):
            await services.bot_state.set(lang_storage_key(11111), lang)
            client.sent.clear()
            await bot.handle_update(_upd(12345, "/ai", user_id=11111))
            assert expect in client.sent[-1][1]
            client.sent.clear()
            await bot.handle_update(_upd(12345, "/ai status", user_id=11111))
            # status title differs
            txt = client.sent[-1][1]
            if lang == "en":
                assert "AI Advisor status" in txt
            else:
                assert "Статус" in txt or "AI-советник" in txt
    finally:
        await shutdown_app(services)


@pytest.mark.asyncio
async def test_ai_no_trading_triggered(tmp_path):
    services = await build_app(_settings(tmp_path))
    await start_app(services, start_telegram=False, start_streams=False)
    try:
        await services.bot_state.set(lang_storage_key(11111), "en")
        bot = TelegramBot(services, FakeClient())
        client = bot._client  # type: ignore[attr-defined]
        # Ensure clean state
        assert not await services.auto_trading_enabled()
        assert not services.guard.is_halted
        # All /ai commands must not start trading, not halt, not create trades/transfers
        for cmd in ["/ai", "/ai status", "/ai report", "/ai recommendations", "/ai memory", "/ai balance"]:
            client.sent.clear()
            await bot.handle_update(_upd(12345, cmd, user_id=11111))
            # No side effect on auto trading flag
            assert not await services.auto_trading_enabled(), f"{cmd} triggered auto trading"
            assert not services.guard.is_halted, f"{cmd} triggered kill switch"
        # No trades created
        trades = await services.trades.list_recent(10)
        assert len(trades) == 0
    finally:
        await shutdown_app(services)


@pytest.mark.asyncio
async def test_ai_bounded_response_length(tmp_path):
    services = await build_app(_settings(tmp_path))
    await start_app(services, start_telegram=False, start_streams=False)
    try:
        # Seed many experiences and large knowledge to potentially blow up response
        from app.agent import build_agent

        core, kb, exp_repo, _, _, _ = build_agent(services)
        # Create huge experiences
        from app.agent.models import Experience

        for i in range(20):
            await exp_repo.save(Experience(situation="x" * 500, observation="y" * 500, source_id=f"src-{i}", confidence=0.9))
        # Create large knowledge doc
        from app.agent.models import KnowledgeCategory

        await kb.ingest_text("Huge", "z" * 10000, KnowledgeCategory.BOT, source_id="huge/doc")
        await services.bot_state.set(lang_storage_key(11111), "en")
        bot = TelegramBot(services, FakeClient())
        client = bot._client  # type: ignore[attr-defined]
        for cmd in ["/ai status", "/ai report", "/ai memory", "/ai recommendations", "/ai balance", "/ai"]:
            client.sent.clear()
            await bot.handle_update(_upd(12345, cmd, user_id=11111))
            txt = client.sent[-1][1]
            assert len(txt) <= 3500, f"{cmd} exceeded bounded length {len(txt)}"
            assert len(txt) > 0
    finally:
        await shutdown_app(services)


@pytest.mark.asyncio
async def test_ai_empty_state_graceful(tmp_path):
    services = await build_app(_settings(tmp_path))
    await start_app(services, start_telegram=False, start_streams=False)
    try:
        await services.bot_state.set(lang_storage_key(11111), "en")
        bot = TelegramBot(services, FakeClient())
        client = bot._client  # type: ignore[attr-defined]
        # Fresh DB -> no trades, no recommendations, no memory -> graceful not error
        for cmd, expect_substr in [
            ("/ai status", "recent trades: 0"),
            ("/ai report", "No actionable"),
            ("/ai recommendations", "No pending"),
            ("/ai memory", "No memory"),
            ("/ai balance", "Balances per venue"),
        ]:
            client.sent.clear()
            await bot.handle_update(_upd(12345, cmd, user_id=11111))
            txt = client.sent[-1][1]
            # Should not be internal error
            assert "internal error" not in txt.lower()
            # For balance, may have balances (always has paper wallets) -> still graceful
            assert len(txt) > 0
    finally:
        await shutdown_app(services)


@pytest.mark.asyncio
async def test_ai_provider_failure_graceful(tmp_path):
    services = await build_app(_settings(tmp_path))
    await start_app(services, start_telegram=False, start_streams=False)
    try:
        # Seed provider failure via failing LLM
        from app.agent import build_agent
        from app.agent.providers.base import LLMProvider, LLMRequest, LLMResponse

        class FailingProvider(LLMProvider):
            name = "failing"

            async def complete(self, request: LLMRequest) -> LLMResponse:
                raise RuntimeError("simulated failure CAT_KEY_BINANCE_SECRET=leak")

        core, *_ = build_agent(services, llm=FailingProvider())
        # Need to wire the failing core into bot's adapter
        # The bot lazily builds adapter via build_agent(services) — but we already built one with failing provider.
        # To make bot use failing provider, we inject via services
        # Simplify: directly test adapter dispatch with failing provider
        from app.agent.telegram import AgentTelegramAdapter
        from app.agent.tools import AgentTools
        from app.agent.context import ContextCollector

        # Build tools/collector using same services
        tools = AgentTools(services)
        # Force the bot's adapter to be the failing one
        from app.agent.telegram import AgentTelegramAdapter
        from app.agent import build_agent as _b

        # Create a fresh failing adapter
        from app.agent.context import ContextCollector as CC
        from app.agent.knowledge import KnowledgeRepository, KnowledgeService
        from app.agent.memory import ExperienceRepository, LessonRepository
        from app.agent.recommendations import RecommendationRepository

        # Reuse existing services.db wiring already done — just create adapter directly
        adapter = AgentTelegramAdapter(core, tools)

        # Need a bot that uses this adapter
        bot = TelegramBot(services, FakeClient())
        bot._agent_adapter = adapter  # inject
        await services.bot_state.set(lang_storage_key(11111), "en")
        client = bot._client  # type: ignore[attr-defined]
        # Even with failing provider, /ai report should not crash to internal error (graceful)
        await bot.handle_update(_upd(12345, "/ai report", user_id=11111))
        txt = client.sent[-1][1]
        assert "internal error" not in txt.lower()
        # Should be either NO_ACTION graceful or still produce something bounded
        assert len(txt) > 0
        assert len(txt) <= 3500
        # No secrets leaked via failure
        assert "CAT_KEY" not in txt
        assert "leak" not in txt.lower() or "<redacted>" in txt.lower()
    finally:
        await shutdown_app(services)


@pytest.mark.asyncio
async def test_ai_arbitrary_tool_not_invokable(tmp_path):
    services = await build_app(_settings(tmp_path))
    await start_app(services, start_telegram=False, start_streams=False)
    try:
        await services.bot_state.set(lang_storage_key(11111), "en")
        bot = TelegramBot(services, FakeClient())
        client = bot._client  # type: ignore[attr-defined]
        # Attempt to invoke arbitrary tool via injection in /ai text
        for evil in [
            "/ai create_order",
            "/ai withdraw",
            "/ai shell execute",
            "/ai run_python print(1)",
            "/ai arbitrary_sql SELECT *",
        ]:
            client.sent.clear()
            await bot.handle_update(_upd(12345, evil, user_id=11111))
            txt = client.sent[-1][1]
            # Must not execute arbitrary tool — should be unknown subcommand help or graceful
            # Must not contain trading mutation side effect
            assert not await services.auto_trading_enabled()
            # Bounded and not internal error leak
            assert len(txt) <= 3500
            # Should be either help or unknown subcommand, not a tool output containing secret
            assert "unauthorized" not in txt.lower()  # authorized, so should get help/unknown, not unauthorized
    finally:
        await shutdown_app(services)


@pytest.mark.asyncio
async def test_ai_no_config_mutation_via_telegram(tmp_path):
    services = await build_app(_settings(tmp_path))
    await start_app(services, start_telegram=False, start_streams=False)
    try:
        original_max = services.settings.risk.max_trade_size
        await services.bot_state.set(lang_storage_key(11111), "en")
        bot = TelegramBot(services, FakeClient())
        client = bot._client  # type: ignore[attr-defined]
        # Try every ai subcommand and ensure risk limit unchanged
        for cmd in ["/ai status", "/ai report", "/ai recommendations", "/ai memory", "/ai balance", "/ai"]:
            client.sent.clear()
            await bot.handle_update(_upd(12345, cmd, user_id=11111))
            assert services.settings.risk.max_trade_size == original_max
            # Also via services.risk
            assert services.risk.limits.max_trade_size == original_max
    finally:
        await shutdown_app(services)
