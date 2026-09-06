"""Phase 4A — runtime preflight (no real OpenRouter credentials).

Verifies:
* CAT_AGENT__* wiring
* startup with missing AI credentials (DEMO)
* /ai graceful when unconfigured
* no secret leakage
* AI failure does not affect trading
* human approval mandatory

No real network calls are made.
"""

from __future__ import annotations

import pathlib
from decimal import Decimal

import pytest
from pydantic import SecretStr

from app.config.settings import AgentSettings, Settings, get_settings, reload_settings
from app.agent.providers import create_provider
from app.agent.providers.base import LLMRequest, LLMMessage, NullProvider, filter_secrets_from_text
from app.agent.providers.openrouter import OpenRouterProvider, OpenRouterError


def test_agent_settings_wired_via_env(monkeypatch):
    # CAT_AGENT__* should map via env_prefix CAT_ + nested delimiter __
    monkeypatch.setenv("CAT_AGENT__PROVIDER", "openrouter")
    monkeypatch.setenv("CAT_AGENT__MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("CAT_AGENT__API_KEY", "sk-test-env")
    monkeypatch.setenv("CAT_AGENT__TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("CAT_AGENT__MAX_RETRIES", "2")
    # Use Settings with env_file=None to force env read
    reload_settings()
    try:
        s = Settings(_env_file=None)
        assert s.agent.provider == "openrouter"
        assert s.agent.model == "openai/gpt-4o-mini"
        assert s.agent.api_key.get_secret_value() == "sk-test-env"
        assert s.agent.timeout_seconds == 12
        assert s.agent.max_retries == 2
        # Factory should produce OpenRouterProvider without network
        p = create_provider(s)
        assert isinstance(p, OpenRouterProvider)
        assert p.model == "openai/gpt-4o-mini"
        # Never log raw key
        assert "sk-test-env" not in str(p.__dict__)
    finally:
        for k in ["CAT_AGENT__PROVIDER", "CAT_AGENT__MODEL", "CAT_AGENT__API_KEY", "CAT_AGENT__TIMEOUT_SECONDS", "CAT_AGENT__MAX_RETRIES"]:
            monkeypatch.delenv(k, raising=False)
        reload_settings()


def test_agent_settings_defaults_no_credentials():
    # Default must be safe: no network, no key, provider null
    s = Settings(_env_file=None, agent=AgentSettings(provider="null", api_key=SecretStr("")))
    assert s.agent.provider == "null"
    assert s.agent.api_key.get_secret_value() == ""
    p = create_provider(s)
    assert isinstance(p, NullProvider)
    # Even when provider openrouter but key empty, construction must not hit network
    s2 = Settings(_env_file=None, agent=AgentSettings(provider="openrouter", api_key=SecretStr(""), model="openai/gpt-4o-mini"))
    p2 = create_provider(s2)
    assert isinstance(p2, OpenRouterProvider)
    # No network on construction — complete will fail closed
    import asyncio

    async def run():
        with pytest.raises(OpenRouterError, match="missing API key"):
            await p2.complete(LLMRequest(messages=(LLMMessage(role="user", content="hi"),)))

    asyncio.run(run())


@pytest.mark.asyncio
async def test_demo_startup_without_ai_credentials(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.config.settings import TradingSettings
    from app.models.enums import TradingMode

    # DEMO mode with AI provider openrouter but NO key — must still start
    base = make_settings(tmp_path)
    settings = base.model_copy(
        update={
            "trading": TradingSettings(mode=TradingMode.DEMO, allow_live=False, base_currency="USDT"),
            "agent": AgentSettings(provider="openrouter", api_key=SecretStr(""), model="openai/gpt-4o-mini"),
        }
    )
    app = await build_app(settings)
    # Mock preflight/market to avoid real network
    from unittest.mock import AsyncMock, patch

    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        with patch.object(app.market, "refresh_order_books", new=AsyncMock()):
            with patch.object(app.market, "start_streams", new=AsyncMock()):
                await start_app(app)
    try:
        # Wire AI advisor (shares DB, no network on construction)
        from app.agent import build_agent

        build_agent(app)
        # Fully functional: status, balances, market data
        status = await app.status()
        assert status["mode"] == "DEMO"
        assert "risk" in status
        balances = await app.balances()
        assert isinstance(balances, dict)
        # Agent should be wired but provider will fail closed on use
        assert app.agent_approval_service is not None
        assert app.agent_audit is not None
        # No trading auto-started
        assert not await app.auto_trading_enabled()
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_ai_commands_graceful_when_unconfigured(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.telegram.bot import TelegramBot
    from app.telegram.i18n import lang_storage_key

    settings = make_settings(tmp_path)
    # Explicitly unconfigured: provider null
    settings = settings.model_copy(update={"agent": AgentSettings(provider="null", api_key=SecretStr(""))})
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        await app.bot_state.set(lang_storage_key(11111), "en")

        class FakeClient:
            def __init__(self):
                self.sent = []

            async def send_message(self, chat_id, text, parse_mode="", reply_markup=None):
                self.sent.append((chat_id, text))

            async def edit_message_text(self, *a, **kw):
                pass

            async def answer_callback_query(self, *a, **kw):
                pass

        def _upd(chat_id, text, user_id=11111):
            return {"update_id": 1, "message": {"chat": {"id": chat_id}, "from": {"id": user_id}, "text": text}}

        bot = TelegramBot(app, FakeClient())
        # /ai status should succeed (reads from tools, no LLM needed)
        await bot.handle_update(_upd(12345, "/ai status", user_id=11111))
        assert "AI Advisor status" in bot._client.sent[-1][1]
        assert "internal error" not in bot._client.sent[-1][1].lower()
        # /ai report with NullProvider (no LLM) should be NO_ACTION graceful, not error
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai report", user_id=11111))
        txt = bot._client.sent[-1][1]
        assert "internal error" not in txt.lower()
        assert len(txt) > 0
        assert "No actionable" in txt or "evidence" in txt.lower() or "no data" in txt.lower()
        # /ai recommendations / memory / balance also graceful empty
        for cmd in ["/ai recommendations", "/ai memory", "/ai balance"]:
            bot._client.sent.clear()
            await bot.handle_update(_upd(12345, cmd, user_id=11111))
            assert "internal error" not in bot._client.sent[-1][1].lower()
            assert len(bot._client.sent[-1][1]) > 0
            assert len(bot._client.sent[-1][1]) <= 3500
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_no_secret_leak_via_provider_and_telegram(tmp_path, caplog):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        # Create provider that will fail with a message containing a fake secret
        class LeakingClient:
            async def post(self, *a, **kw):
                raise RuntimeError("failed with CAT_KEY_BINANCE_SECRET=supersecret123 and CAT_TELEGRAM__BOT_TOKEN=123:ABC and sqlite+aiosqlite:///./data/bot.db")

        p = OpenRouterProvider(api_key=SecretStr("sk-test"), http_client=LeakingClient(), max_retries=0)
        # Direct provider call — exception must be redacted
        with pytest.raises(OpenRouterError) as exc:
            await p.complete(LLMRequest(messages=(LLMMessage(role="user", content="hi CAT_KEY_OKX_SECRET=mysecret"),)))
        msg = str(exc.value)
        assert "supersecret123" not in msg
        assert "CAT_KEY" not in msg or "<redacted>" in msg
        assert "mysecret" not in msg or "<redacted>" in msg

        # Via AgentCore + Telegram: ensure secret in user query does not appear in response
        from app.telegram.bot import TelegramBot
        from app.telegram.i18n import lang_storage_key

        await app.bot_state.set(lang_storage_key(11111), "en")

        # Use a core with the leaking provider
        core, *_ = build_agent(app, llm=p)
        # Seed trades so LLM is actually invoked (otherwise NO_ACTION short-circuits)
        from decimal import Decimal
        from app.models.enums import ArbitrageStrategy, TradeStatus, TradingMode
        from app.models.trade import TradeRecord

        for i in range(6):
            await app.trades.save(
                TradeRecord(
                    strategy=ArbitrageStrategy.TRIANGLE,
                    mode=TradingMode.PAPER,
                    exchange_id="binance",
                    route="r",
                    input_amount=Decimal("1000"),
                    output_amount=Decimal("990"),
                    net_profit=Decimal("-5"),
                    net_profit_bps=Decimal("-50"),
                    status=TradeStatus.FAILED,
                )
            )

        class FakeClient:
            def __init__(self):
                self.sent = []

            async def send_message(self, chat_id, text, parse_mode="", reply_markup=None):
                self.sent.append((chat_id, text))

            async def edit_message_text(self, *a, **kw):
                pass

            async def answer_callback_query(self, *a, **kw):
                pass

        def _upd(chat_id, text, user_id=11111):
            return {"update_id": 1, "message": {"chat": {"id": chat_id}, "from": {"id": user_id}, "text": text}}

        bot = TelegramBot(app, FakeClient())
        # Inject the leaking core via adapter
        from app.agent.telegram import AgentTelegramAdapter
        from app.agent.tools import AgentTools

        tools = AgentTools(app)
        bot._agent_adapter = AgentTelegramAdapter(core, tools, approval_service=app.agent_approval_service)
        # Query containing secrets
        await bot.handle_update(_upd(12345, "/ai report", user_id=11111))
        txt = bot._client.sent[-1][1]
        assert "supersecret123" not in txt
        assert "CAT_KEY" not in txt or "<redacted>" in txt
        assert "mysecret" not in txt or "<redacted>" in txt
        # Also test direct injection via query param that contains DSN
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai status", user_id=11111))
        txt2 = bot._client.sent[-1][1]
        assert "bot.db" not in txt2 or "<redacted>" in txt2

        # Logs must not contain secrets (caplog)
        for rec in caplog.records:
            msg = rec.getMessage()
            assert "supersecret123" not in msg
            assert "mysecret" not in str(rec.args) if rec.args else True
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_ai_failure_does_not_affect_trading_balances_risk(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.providers.base import LLMProvider, LLMRequest, LLMResponse

    class CrashLLM(LLMProvider):
        name = "crash"

        async def complete(self, request: LLMRequest) -> LLMResponse:
            raise RuntimeError("provider outage CAT_KEY_BINANCE_SECRET=should_not_leak")

    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        core, *_ = build_agent(app, llm=CrashLLM())
        from decimal import Decimal
        from app.models.enums import ArbitrageStrategy, TradeStatus, TradingMode
        from app.models.trade import TradeRecord

        for i in range(6):
            await app.trades.save(
                TradeRecord(
                    strategy=ArbitrageStrategy.TRIANGLE,
                    mode=TradingMode.PAPER,
                    exchange_id="binance",
                    route="r",
                    input_amount=Decimal("1000"),
                    output_amount=Decimal("990"),
                    net_profit=Decimal("-5"),
                    net_profit_bps=Decimal("-50"),
                    status=TradeStatus.FAILED,
                )
            )
        from app.agent.core import AgentRequest

        resp = await core.handle(AgentRequest(query="crash test"))
        assert resp is not None
        # Trading state still readable
        status = await app.status()
        assert status["risk"]["open_transfers"] == 0
        balances = await app.balances()
        assert isinstance(balances, dict)
        # Risk limits unchanged
        assert app.settings.risk.max_trade_size == Decimal("1000")
        assert app.risk.limits.max_trade_size == Decimal("1000")
        # No secret leak
        txt = filter_secrets_from_text(str(resp.llm_output or ""))
        assert "should_not_leak" not in txt
    finally:
        from app.services import shutdown_app as _sd

        await _sd(app)


@pytest.mark.asyncio
async def test_human_approval_still_mandatory_preflight(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.approval import ApprovalError

    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size",
            current_value=str(app.settings.risk.max_trade_size),
            proposed_value="900",
            reason="preflight",
            source_id="test",
        )
        # AI tools must not be able to approve
        from app.agent.tools import AgentTools, ToolAccessBlocked

        tools = AgentTools(app)
        assert "approve" not in tools.allowed_tools
        with pytest.raises(ToolAccessBlocked):
            getattr(tools, "approve")
        # Approval without human must fail
        with pytest.raises(ApprovalError):
            await app.agent_approval_service.approve(rec.id, approver="", reason="empty")
        # Correct human approval succeeds
        approved = await app.agent_approval_service.approve(rec.id, approver="human-preflight", reason="ok")
        assert approved.status.value == "approved"
        assert str(app.settings.risk.max_trade_size) == "900"
    finally:
        from app.services import shutdown_app as _sd

        await _sd(app)
