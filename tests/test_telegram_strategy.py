"""Telegram strategy selection for DEMO trading."""

import pathlib
import pytest
from decimal import Decimal

from app.config.settings import TradingSettings
from app.models.enums import TradingMode
from app.services import AppServices, build_app, shutdown_app, start_app
from app.telegram.bot import TelegramBot
from app.telegram.i18n import strategy_storage_key, lang_storage_key, t

AUTHORIZED_USER_ID = 11111
AUTHORIZED_CHAT_ID = 12345
STRANGER_USER_ID = 99999

class FakeClient:
    def __init__(self):
        self.sent: list[tuple[int, str]] = []
        self.kbs: list[dict | None] = []
    async def send_message(self, chat_id: int, text: str, *, parse_mode: str = "", reply_markup: dict | None = None):
        self.sent.append((chat_id, text))
        self.kbs.append(reply_markup)
    async def edit_message_text(self, chat_id: int, message_id: int, text: str, *, reply_markup: dict | None = None):
        pass
    async def answer_callback_query(self, callback_query_id: str, text: str = ""):
        self.last_answer = text

def _bot(services: AppServices):
    c = FakeClient()
    return TelegramBot(services, c), c

def _update(chat_id: int, text: str, *, user_id: int = AUTHORIZED_USER_ID):
    return {"update_id": 1, "message": {"chat": {"id": chat_id}, "from": {"id": user_id}, "text": text}}

def _callback(chat_id: int, user_id: int, data: str):
    return {"update_id": 2, "callback_query": {"id": "cb1", "from": {"id": user_id}, "message": {"chat": {"id": chat_id}, "message_id": 1}, "data": data}}

def _settings(tmp_path):
    from tests.conftest import make_settings
    base = make_settings(tmp_path)
    return base.model_copy(update={"trading": TradingSettings(mode=TradingMode.DEMO, allow_live=False, base_currency="USDT")})

# 1. /strategy shows picker
async def test_strategy_command_shows_picker(tmp_path):
    from unittest.mock import AsyncMock, patch
    settings = _settings(tmp_path)
    services = await build_app(settings)
    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        with patch.object(services.market, "refresh_order_books", new=AsyncMock()):
            with patch.object(services.market, "start_streams", new=AsyncMock()):
                await start_app(services)
    try:
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "en")
        await services.bot_state.delete(strategy_storage_key())
        bot, client = _bot(services)
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/strategy"))
        assert len(client.sent) == 1
        txt = client.sent[0][1]
        assert "Choose strategy" in txt or "Выберите стратегию" in txt
        assert client.kbs[0] is not None
        assert "strategy:triangle" in str(client.kbs[0]) or "Triangle" in str(client.kbs[0])
    finally:
        await shutdown_app(services)

# 2. Selecting Triangle persists
async def test_selecting_triangle_persists(tmp_path):
    from unittest.mock import AsyncMock, patch
    settings = _settings(tmp_path)
    services = await build_app(settings)
    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        with patch.object(services.market, "refresh_order_books", new=AsyncMock()):
            with patch.object(services.market, "start_streams", new=AsyncMock()):
                await start_app(services)
    try:
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "en")
        await services.bot_state.delete(strategy_storage_key())
        bot, client = _bot(services)
        await bot.handle_update(_callback(AUTHORIZED_CHAT_ID, AUTHORIZED_USER_ID, "strategy:triangle"))
        val = await services.bot_state.get(strategy_storage_key())
        assert val == "triangle"
        assert any("Triangle" in s for _, s in client.sent)
    finally:
        await shutdown_app(services)

# 3. Selecting Transfer persists
async def test_selecting_transfer_persists(tmp_path):
    from unittest.mock import AsyncMock, patch
    settings = _settings(tmp_path)
    services = await build_app(settings)
    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        with patch.object(services.market, "refresh_order_books", new=AsyncMock()):
            with patch.object(services.market, "start_streams", new=AsyncMock()):
                await start_app(services)
    try:
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "en")
        bot, client = _bot(services)
        await bot.handle_update(_callback(AUTHORIZED_CHAT_ID, AUTHORIZED_USER_ID, "strategy:transfer"))
        val = await services.bot_state.get(strategy_storage_key())
        assert val == "transfer"
        assert any("Transfer" in s for _, s in client.sent)
    finally:
        await shutdown_app(services)

# 4. /status shows active strategy
async def test_status_shows_active_strategy(tmp_path):
    from unittest.mock import AsyncMock, patch
    settings = _settings(tmp_path)
    services = await build_app(settings)
    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        with patch.object(services.market, "refresh_order_books", new=AsyncMock()):
            with patch.object(services.market, "start_streams", new=AsyncMock()):
                await start_app(services)
    try:
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "en")
        await services.bot_state.set(strategy_storage_key(), "triangle")
        bot, client = _bot(services)
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
        txt = client.sent[-1][1]
        assert "strategy:" in txt.lower()
        assert "triangle" in txt.lower()
        # Change to transfer
        await services.bot_state.set(strategy_storage_key(), "transfer")
        client.sent.clear()
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
        txt2 = client.sent[-1][1]
        assert "transfer" in txt2.lower()
    finally:
        await shutdown_app(services)

# 5. /status in Russian shows translated strategy
async def test_status_shows_strategy_in_russian(tmp_path):
    from unittest.mock import AsyncMock, patch
    settings = _settings(tmp_path)
    services = await build_app(settings)
    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        with patch.object(services.market, "refresh_order_books", new=AsyncMock()):
            with patch.object(services.market, "start_streams", new=AsyncMock()):
                await start_app(services)
    try:
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "ru")
        await services.bot_state.set(strategy_storage_key(), "triangle")
        bot, client = _bot(services)
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
        txt = client.sent[-1][1]
        assert "стратегия" in txt.lower()
    finally:
        await shutdown_app(services)

# 6. /start_trading without strategy now defaults to triangle (for backwards compat) and starts
async def test_start_trading_without_strategy_defaults_to_triangle(tmp_path):
    from unittest.mock import AsyncMock, patch
    settings = _settings(tmp_path)
    services = await build_app(settings)
    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        with patch.object(services.market, "refresh_order_books", new=AsyncMock()):
            with patch.object(services.market, "start_streams", new=AsyncMock()):
                await start_app(services)
    try:
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "en")
        await services.bot_state.delete(strategy_storage_key())
        bot, client = _bot(services)
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
        # Should have started (not shown picker as error, but defaulted)
        # Check that auto_trading is now enabled and strategy is triangle
        assert await services.auto_trading_enabled() is True
        strat = await services.bot_state.get(strategy_storage_key())
        assert strat == "triangle"
        await services.auto_controller.stop()
    finally:
        await shutdown_app(services)

# 7. Authorization: unauthorized cannot change strategy
async def test_unauthorized_cannot_change_strategy(tmp_path):
    from unittest.mock import AsyncMock, patch
    settings = _settings(tmp_path)
    services = await build_app(settings)
    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        with patch.object(services.market, "refresh_order_books", new=AsyncMock()):
            with patch.object(services.market, "start_streams", new=AsyncMock()):
                await start_app(services)
    try:
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "en")
        await services.bot_state.set(strategy_storage_key(), "triangle")
        bot, client = _bot(services)
        # Stranger tries to change to transfer
        await bot.handle_update(_callback(AUTHORIZED_CHAT_ID, STRANGER_USER_ID, "strategy:transfer"))
        # Should still be triangle
        val = await services.bot_state.get(strategy_storage_key())
        assert val == "triangle"
        # Stranger tries /strategy command
        client.sent.clear()
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/strategy", user_id=STRANGER_USER_ID))
        assert "unauthorized" in client.sent[-1][1].lower() or "доступ" in client.sent[-1][1].lower()
    finally:
        await shutdown_app(services)

# 8. Strategy persists across restart
async def test_strategy_persists_across_restart(tmp_path):
    from unittest.mock import AsyncMock, patch
    settings = _settings(tmp_path)
    services_a = await build_app(settings)
    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        with patch.object(services_a.market, "refresh_order_books", new=AsyncMock()):
            with patch.object(services_a.market, "start_streams", new=AsyncMock()):
                await start_app(services_a)
    try:
        await services_a.bot_state.set(strategy_storage_key(), "transfer")
        await shutdown_app(services_a)
        services_b = await build_app(settings)
        with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
            with patch.object(services_b.market, "refresh_order_books", new=AsyncMock()):
                with patch.object(services_b.market, "start_streams", new=AsyncMock()):
                    await start_app(services_b)
        try:
            val = await services_b.bot_state.get(strategy_storage_key())
            assert val == "transfer"
        finally:
            await shutdown_app(services_b)
        services_a = None
    finally:
        if services_a is not None:
            await shutdown_app(services_a)

# 9. /strategy is in COMMANDS and help
async def test_strategy_in_commands_and_help(tmp_path):
    from app.telegram.bot import COMMANDS, HELP_TEXT
    assert "/strategy" in COMMANDS
    assert "/strategy" in HELP_TEXT
    # Help is localized
    from app.telegram.i18n import t
    assert "/strategy" in t("help_text", "en")
    assert "/strategy" in t("help_text", "ru")

# 10. Changing strategy does not affect trading state (does not start/stop)
async def test_strategy_change_does_not_affect_trading(tmp_path):
    from unittest.mock import AsyncMock, patch
    settings = _settings(tmp_path)
    services = await build_app(settings)
    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        with patch.object(services.market, "refresh_order_books", new=AsyncMock()):
            with patch.object(services.market, "start_streams", new=AsyncMock()):
                await start_app(services)
    try:
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "en")
        await services.bot_state.set(strategy_storage_key(), "triangle")
        bot, client = _bot(services)
        # Start trading
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
        assert await services.auto_trading_enabled() is True
        assert services.auto_controller._task is not None
        # Change strategy while running
        await bot.handle_update(_callback(AUTHORIZED_CHAT_ID, AUTHORIZED_USER_ID, "strategy:transfer"))
        # Trading should still be running, not stopped
        assert await services.auto_trading_enabled() is True
        assert services.auto_controller._task is not None
        # Strategy should be transfer now
        assert await services.bot_state.get(strategy_storage_key()) == "transfer"
        await services.auto_controller.stop()
    finally:
        await shutdown_app(services)
