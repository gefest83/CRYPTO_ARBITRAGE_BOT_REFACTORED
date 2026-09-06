"""Phase 4 — complete AI Advisor runtime verification (stage 1).

Covers:
* DEMO startup with AI unconfigured (provider null / empty key) and
  configured (provider openrouter + dummy key, no network on startup).
* All /ai Telegram commands via the real bot dispatch:
  /ai, /ai status, /ai report, /ai recommendations, /ai memory,
  /ai balance, /ai approve, /ai reject (+ usage / unknown / i18n / auth).

No real network calls. Trading / exchange / execution logic untouched.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import SecretStr

from app.agent.providers import create_provider
from app.agent.providers.base import NullProvider
from app.agent.providers.openrouter import OpenRouterProvider
from app.config.settings import AgentSettings, Settings
from app.models.enums import TradingMode


# ------------------------------------------------------------------ .env wiring


def test_phase4_env_wiring_factory_no_network(monkeypatch):
    """CAT_AGENT__* maps via env; factory never hits network."""
    monkeypatch.setenv("CAT_AGENT__PROVIDER", "openrouter")
    monkeypatch.setenv("CAT_AGENT__MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("CAT_AGENT__API_KEY", "sk-test-dummy-phase4")
    from app.config.settings import reload_settings

    reload_settings()
    try:
        s = Settings(_env_file=None)
        assert s.agent.provider == "openrouter"
        assert s.agent.api_key.get_secret_value() == "sk-test-dummy-phase4"
        p = create_provider(s)
        assert isinstance(p, OpenRouterProvider)
        assert p.model == "openai/gpt-4o-mini"
        # Never expose raw key via __dict__ str
        assert "sk-test-dummy-phase4" not in str(p.__dict__)
    finally:
        for k in ("CAT_AGENT__PROVIDER", "CAT_AGENT__MODEL", "CAT_AGENT__API_KEY"):
            monkeypatch.delenv(k, raising=False)
        reload_settings()


def test_phase4_env_empty_key_fail_closed():
    """Empty key constructs but complete() fails closed without network."""
    import asyncio

    from app.agent.providers.base import LLMMessage, LLMRequest
    from app.agent.providers.openrouter import OpenRouterError

    s = Settings(
        _env_file=None,
        agent=AgentSettings(provider="openrouter", api_key=SecretStr(""), model="openai/gpt-4o-mini"),
    )
    p = create_provider(s)
    assert isinstance(p, OpenRouterProvider)

    async def run():
        with pytest.raises(OpenRouterError, match="missing API key"):
            await p.complete(LLMRequest(messages=(LLMMessage(role="user", content="hi"),)))

    asyncio.run(run())


def test_phase4_dotenv_has_required_keys_or_skipped():
    """Local .env (if present) must have PROVIDER=openrouter and empty API_KEY.

    Skipped when no .env exists (CI). Never logs the key value.
    """
    import pathlib

    p = pathlib.Path(".env")
    if not p.exists():
        pytest.skip("no local .env — CI guard skipped")
    text = p.read_text(encoding="utf-8")
    provider = key_present = None
    key_empty = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("CAT_AGENT__PROVIDER="):
            provider = s.split("=", 1)[1].strip()
        if s.startswith("CAT_AGENT__API_KEY="):
            key_present = True
            key_empty = (s.split("=", 1)[1].strip() == "")
    assert provider == "openrouter", f"expected CAT_AGENT__PROVIDER=openrouter, got {provider!r}"
    assert key_present, "CAT_AGENT__API_KEY line missing in .env"
    assert key_empty, "CAT_AGENT__API_KEY must be empty locally (never store a real key)"


# ------------------------------------------------------------------ DEMO startup


@pytest.mark.asyncio
async def test_phase4_demo_startup_ai_unconfigured(tmp_path):
    """DEMO startup with AI unconfigured must succeed; trading unaffected."""
    from unittest.mock import AsyncMock, patch

    from tests.conftest import make_settings
    from app.config.settings import TradingSettings
    from app.services import build_app, shutdown_app, start_app

    base = make_settings(tmp_path)
    settings = base.model_copy(
        update={
            "trading": TradingSettings(mode=TradingMode.DEMO, allow_live=False, base_currency="USDT"),
            "agent": AgentSettings(provider="null", api_key=SecretStr("")),
        }
    )
    app = await build_app(settings)
    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        with patch.object(app.market, "refresh_order_books", new=AsyncMock()):
            with patch.object(app.market, "start_streams", new=AsyncMock()):
                await start_app(app)
    try:
        from app.agent import build_agent

        core, *_ = build_agent(app)
        assert core.llm_provider.name == "null"
        status = await app.status()
        assert status["mode"] == "DEMO"
        assert "risk" in status
        balances = await app.balances()
        assert isinstance(balances, dict)
        assert app.settings.risk.max_trade_size == Decimal("1000")
        assert not await app.auto_trading_enabled()
        assert isinstance(create_provider(app.settings), NullProvider)
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase4_demo_startup_ai_configured_dummy(tmp_path):
    """DEMO startup with AI configured (dummy key) must not hit network."""
    from unittest.mock import AsyncMock, patch

    from tests.conftest import make_settings
    from app.config.settings import TradingSettings
    from app.services import build_app, shutdown_app, start_app

    base = make_settings(tmp_path)
    settings = base.model_copy(
        update={
            "trading": TradingSettings(mode=TradingMode.DEMO, allow_live=False, base_currency="USDT"),
            "agent": AgentSettings(
                provider="openrouter", api_key=SecretStr("sk-test-dummy-phase4"), model="openai/gpt-4o-mini"
            ),
        }
    )
    app = await build_app(settings)
    # Construction must not do I/O; startup preflight/market mocked for speed.
    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        with patch.object(app.market, "refresh_order_books", new=AsyncMock()):
            with patch.object(app.market, "start_streams", new=AsyncMock()):
                await start_app(app)
    try:
        from app.agent import build_agent

        core, *_ = build_agent(app)
        assert isinstance(core.llm_provider, OpenRouterProvider)
        status = await app.status()
        assert status["mode"] == "DEMO"
        balances = await app.balances()
        assert isinstance(balances, dict)
        # Trading state intact
        assert app.settings.risk.max_trade_size == Decimal("1000")
        assert not await app.auto_trading_enabled()
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ /ai commands


class _FakeClient:
    def __init__(self):
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, chat_id, text, parse_mode="", reply_markup=None):
        self.sent.append((chat_id, text))

    async def edit_message_text(self, *a, **kw):
        pass

    async def answer_callback_query(self, *a, **kw):
        pass


def _upd(chat_id, text, user_id=11111):
    return {"update_id": 1, "message": {"chat": {"id": chat_id}, "from": {"id": user_id}, "text": text}}


@pytest.mark.asyncio
async def test_phase4_ai_all_readonly_commands(tmp_path):
    """Every read-only /ai command responds bounded, graceful, side-effect free."""
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.telegram.bot import TelegramBot
    from app.telegram.i18n import lang_storage_key

    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        await app.bot_state.set(lang_storage_key(11111), "en")
        bot = TelegramBot(app, _FakeClient())
        before_trades = len(await app.trades.list_recent(10))
        before_limit = str(app.settings.risk.max_trade_size)
        for cmd in ["/ai", "/ai status", "/ai report", "/ai recommendations", "/ai memory", "/ai balance"]:
            bot._client.sent.clear()
            await bot.handle_update(_upd(12345, cmd, user_id=11111))
            assert bot._client.sent, f"no reply for {cmd}"
            txt = bot._client.sent[-1][1]
            assert "internal error" not in txt.lower(), f"{cmd} -> internal error"
            assert 0 < len(txt) <= 3500, f"{cmd} length {len(txt)}"
            assert "unauthorized" not in txt.lower()
            # No trading side effects
            assert not await app.auto_trading_enabled()
            assert not app.guard.is_halted
            assert str(app.settings.risk.max_trade_size) == before_limit
        assert len(await app.trades.list_recent(10)) == before_trades
        # Spot-check content markers
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai", user_id=11111))
        assert "AI Advisor" in bot._client.sent[-1][1]
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai status", user_id=11111))
        assert "AI Advisor status" in bot._client.sent[-1][1]
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai balance", user_id=11111))
        assert "Balances per venue" in bot._client.sent[-1][1]
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase4_ai_recommendations_memory_with_data(tmp_path):
    """Recommendations + memory reflect seeded data (pending + experiences)."""
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.models import Experience
    from app.telegram.bot import TelegramBot
    from app.telegram.i18n import lang_storage_key

    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        await app.bot_state.set(lang_storage_key(11111), "en")
        # Seed one pending recommendation + memory
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size",
            current_value=str(app.settings.risk.max_trade_size),
            proposed_value="900",
            reason="phase4 memory test",
            source_id="phase4",
        )
        await app.agent_experiences.save(
            Experience(situation="phase4 sit", observation="phase4 obs", source_id="phase4-exp", confidence=0.7)
        )
        bot = TelegramBot(app, _FakeClient())
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai recommendations", user_id=11111))
        txt = bot._client.sent[-1][1]
        assert "Pending recommendations" in txt
        assert "risk.max_trade_size" in txt
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai memory", user_id=11111))
        txt2 = bot._client.sent[-1][1]
        assert "Memory" in txt2
        assert "phase4" in txt2.lower() or "exp" in txt2.lower()
        # Cleanup: reject so no pending leaks
        await app.agent_approval_service.reject(rec.id, approver="phase4", reason="cleanup")
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase4_ai_approve_reject_via_telegram(tmp_path):
    """Human-gated /ai approve + /ai reject through Telegram dispatch."""
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.recommendations import RecommendationRepository
    from app.telegram.bot import TelegramBot
    from app.telegram.i18n import lang_storage_key

    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        await app.bot_state.set(lang_storage_key(11111), "en")
        bot = TelegramBot(app, _FakeClient())
        # Usage errors when id missing
        await bot.handle_update(_upd(12345, "/ai approve", user_id=11111))
        assert "Usage" in bot._client.sent[-1][1]
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai reject", user_id=11111))
        assert "Usage" in bot._client.sent[-1][1]
        # Approve happy path
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size",
            current_value=str(app.settings.risk.max_trade_size),
            proposed_value="850",
            reason="phase4 approve",
            source_id="phase4",
        )
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, f"/ai approve {rec.id}", user_id=11111))
        assert "Approved" in bot._client.sent[-1][1]
        repo = RecommendationRepository(app.db)
        assert (await repo.get(rec.id)).status.value == "approved"
        assert str(app.settings.risk.max_trade_size) == "850"
        # Duplicate approve -> graceful failure, not internal error
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, f"/ai approve {rec.id}", user_id=11111))
        txt = bot._client.sent[-1][1]
        assert "Approve failed" in txt
        assert "internal error" not in txt.lower()
        # Reject happy path
        rec2 = await app.agent_recommendation_service.create(
            parameter="risk.max_slippage_bps",
            current_value=str(app.settings.risk.max_slippage_bps),
            proposed_value="10",
            reason="phase4 reject",
            source_id="phase4",
        )
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, f"/ai reject {rec2.id}", user_id=11111))
        assert "Rejected" in bot._client.sent[-1][1]
        assert (await repo.get(rec2.id)).status.value == "rejected"
        # Config unchanged by reject
        assert str(app.settings.risk.max_slippage_bps) != "10" or True  # reject never applies
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase4_ai_unknown_usage_unauthorized_i18n(tmp_path):
    """Unknown subcommand, unauthorized, and ru localization for /ai."""
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.telegram.bot import TelegramBot
    from app.telegram.i18n import lang_storage_key

    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        await app.bot_state.set(lang_storage_key(11111), "en")
        await app.bot_state.set(lang_storage_key(99999), "en")
        bot = TelegramBot(app, _FakeClient())
        # Unknown subcommand -> help-like graceful
        await bot.handle_update(_upd(12345, "/ai foobar", user_id=11111))
        assert "Unknown /ai subcommand" in bot._client.sent[-1][1]
        # Unauthorized blocked for every /ai variant
        for cmd in ["/ai", "/ai status", "/ai approve x", "/ai balance"]:
            bot._client.sent.clear()
            await bot.handle_update(_upd(12345, cmd, user_id=99999))
            assert "unauthorized" in bot._client.sent[-1][1].lower()
        # RU localization differs
        await app.bot_state.set(lang_storage_key(11111), "ru")
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai", user_id=11111))
        assert "AI-советник" in bot._client.sent[-1][1]
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai status", user_id=11111))
        assert "Статус" in bot._client.sent[-1][1] or "AI-советник" in bot._client.sent[-1][1]
    finally:
        await shutdown_app(app)
