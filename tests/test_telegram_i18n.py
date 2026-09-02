"""Telegram i18n / language-selection tests (16 required).

Covers:
 1. First /start shows picker
 2. Selecting English persists en
 3. Selecting Russian persists ru
 4. Subsequent /start does not show picker
 5. /language en -> ru
 6. ru -> en
 7. /status in English after en
 8. /status in Russian after ru
 9. /help localized
10. /start_trading en calls same service
11. /start_trading ru calls same service
12. /stop_trading unchanged functionally (both langs)
13. /pause and /resume unchanged functionally
14. Authorization unchanged
15. Invalid/missing stored language falls back to picker
16. Language survives reload/restart via persistent storage
"""

from __future__ import annotations

from typing import Any

import pytest

from app.config.settings import Settings, TelegramSettings, TradingSettings
from app.models.enums import TradingMode
from app.services import AppServices, build_app, shutdown_app, start_app
from app.telegram.bot import TelegramBot
from app.telegram.i18n import LANGUAGE_PICKER_PROMPT, lang_storage_key, t

AUTHORIZED_USER_ID = 11111
AUTHORIZED_CHAT_ID = 12345
STRANGER_USER_ID = 99999


class FakeClient:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []
        self.kbs: list[dict | None] = []

    async def send_message(self, chat_id: int, text: str, *, parse_mode: str = "", reply_markup: dict | None = None) -> None:
        self.sent.append((chat_id, text))
        self.kbs.append(reply_markup)

    async def edit_message_text(self, chat_id: int, message_id: int, text: str, *, reply_markup: dict | None = None) -> None:
        pass

    async def answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        self.last_answer = text


def _bot(services: AppServices) -> tuple[TelegramBot, FakeClient]:
    c = FakeClient()
    return TelegramBot(services, c), c


def _update(chat_id: int, text: str, *, user_id: int = AUTHORIZED_USER_ID) -> dict[str, Any]:
    return {"update_id": 1, "message": {"chat": {"id": chat_id}, "from": {"id": user_id}, "text": text}}


def _callback(chat_id: int, user_id: int, lang: str, cb_id: str = "cb1") -> dict[str, Any]:
    return {
        "update_id": 2,
        "callback_query": {
            "id": cb_id,
            "from": {"id": user_id},
            "message": {"chat": {"id": chat_id}, "message_id": 42},
            "data": f"lang:{lang}",
        },
    }


def _settings(tmp_path, mode: TradingMode = TradingMode.PAPER) -> Settings:
    from tests.conftest import make_settings

    base = make_settings(tmp_path)
    if mode is not TradingMode.PAPER:
        base = base.model_copy(update={"trading": TradingSettings(mode=mode, allow_live=False, base_currency="USDT")})
    return base


# 1. First /start shows picker
async def test_first_start_shows_picker(tmp_path) -> None:
    services = await build_app(_settings(tmp_path))
    await start_app(services)
    try:
        # Ensure no language
        await services.bot_state.delete(lang_storage_key(AUTHORIZED_USER_ID))
        bot, client = _bot(services)
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start"))
        assert len(client.sent) == 1
        assert LANGUAGE_PICKER_PROMPT in client.sent[0][1]
        assert client.kbs[0] is not None
        # Check keyboard has both languages
        kb = client.kbs[0]
        assert "inline_keyboard" in kb
        btns = kb["inline_keyboard"][0]
        texts = [b["text"] for b in btns]
        assert any("English" in t for t in texts)
        assert any("Русский" in t for t in texts)
    finally:
        await shutdown_app(services)


# 2. Selecting English persists en
async def test_selecting_english_persists_en(tmp_path) -> None:
    services = await build_app(_settings(tmp_path))
    await start_app(services)
    try:
        await services.bot_state.delete(lang_storage_key(AUTHORIZED_USER_ID))
        bot, client = _bot(services)
        await bot.handle_update(_callback(AUTHORIZED_CHAT_ID, AUTHORIZED_USER_ID, "en"))
        val = await services.bot_state.get(lang_storage_key(AUTHORIZED_USER_ID))
        assert val == "en"
        # Should have confirmation in English
        assert any("Language changed" in s for _, s in client.sent)
    finally:
        await shutdown_app(services)


# 3. Selecting Russian persists ru
async def test_selecting_russian_persists_ru(tmp_path) -> None:
    services = await build_app(_settings(tmp_path))
    await start_app(services)
    try:
        await services.bot_state.delete(lang_storage_key(AUTHORIZED_USER_ID))
        bot, client = _bot(services)
        await bot.handle_update(_callback(AUTHORIZED_CHAT_ID, AUTHORIZED_USER_ID, "ru"))
        val = await services.bot_state.get(lang_storage_key(AUTHORIZED_USER_ID))
        assert val == "ru"
        assert any("Язык" in s for _, s in client.sent)
    finally:
        await shutdown_app(services)


# 4. Subsequent /start does not show picker
async def test_subsequent_start_does_not_show_picker(tmp_path) -> None:
    services = await build_app(_settings(tmp_path))
    await start_app(services)
    try:
        bot, client = _bot(services)
        # Set English
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "en")
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start"))
        assert len(client.sent) == 1
        # Should be help, not picker
        assert LANGUAGE_PICKER_PROMPT not in client.sent[0][1]
        assert "operator commands" in client.sent[0][1].lower() or "crypto-arbitrage-bot" in client.sent[0][1]
        # Now check Russian
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "ru")
        client.sent.clear()
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start"))
        assert LANGUAGE_PICKER_PROMPT not in client.sent[0][1]
        assert "команды" in client.sent[0][1].lower()
    finally:
        await shutdown_app(services)


# 5. /language en -> ru
async def test_language_change_en_to_ru(tmp_path) -> None:
    services = await build_app(_settings(tmp_path))
    await start_app(services)
    try:
        bot, client = _bot(services)
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "en")
        # /language shows picker
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/language"))
        assert LANGUAGE_PICKER_PROMPT in client.sent[-1][1]
        client.sent.clear()
        # Select ru
        await bot.handle_update(_callback(AUTHORIZED_CHAT_ID, AUTHORIZED_USER_ID, "ru"))
        assert await services.bot_state.get(lang_storage_key(AUTHORIZED_USER_ID)) == "ru"
        # Confirmation in Russian
        assert any("Язык" in s for _, s in client.sent)
        # Subsequent /status should be Russian
        client.sent.clear()
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
        txt = client.sent[-1][1]
        assert "режим" in txt.lower() or "биржи" in txt.lower()
    finally:
        await shutdown_app(services)


# 6. /language ru -> en
async def test_language_change_ru_to_en(tmp_path) -> None:
    services = await build_app(_settings(tmp_path))
    await start_app(services)
    try:
        bot, client = _bot(services)
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "ru")
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/language"))
        assert LANGUAGE_PICKER_PROMPT in client.sent[-1][1]
        client.sent.clear()
        await bot.handle_update(_callback(AUTHORIZED_CHAT_ID, AUTHORIZED_USER_ID, "en"))
        assert await services.bot_state.get(lang_storage_key(AUTHORIZED_USER_ID)) == "en"
        assert any("Language changed" in s for _, s in client.sent)
        client.sent.clear()
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
        txt = client.sent[-1][1]
        assert "mode:" in txt.lower()
    finally:
        await shutdown_app(services)


# 7. /status uses English after en
async def test_status_english_after_en(tmp_path) -> None:
    services = await build_app(_settings(tmp_path))
    await start_app(services)
    try:
        bot, client = _bot(services)
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "en")
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
        txt = client.sent[-1][1]
        assert "mode:" in txt.lower()
        assert "kill switch:" in txt.lower()
        assert "auto trading" in txt.lower()
    finally:
        await shutdown_app(services)


# 8. /status uses Russian after ru
async def test_status_russian_after_ru(tmp_path) -> None:
    services = await build_app(_settings(tmp_path))
    await start_app(services)
    try:
        bot, client = _bot(services)
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "ru")
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
        txt = client.sent[-1][1]
        assert "режим" in txt.lower()
        assert "стоп-кран" in txt.lower()
        assert "автоторговля" in txt.lower()
    finally:
        await shutdown_app(services)


# 9. /help is localized
async def test_help_localized(tmp_path) -> None:
    services = await build_app(_settings(tmp_path))
    await start_app(services)
    try:
        bot, client = _bot(services)
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "en")
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/help"))
        en_text = client.sent[-1][1]
        assert "operator commands" in en_text.lower()
        client.sent.clear()
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "ru")
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/help"))
        ru_text = client.sent[-1][1]
        assert "команды оператора" in ru_text.lower()
        assert en_text != ru_text
    finally:
        await shutdown_app(services)


# 10. /start_trading en calls same service
async def test_start_trading_en_calls_service(tmp_path) -> None:
    from unittest.mock import AsyncMock, patch

    settings = _settings(tmp_path, mode=TradingMode.DEMO)
    services = await build_app(settings)
    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        with patch.object(services.market, "refresh_order_books", new=AsyncMock()):
            with patch.object(services.market, "start_streams", new=AsyncMock()):
                await start_app(services)
    try:
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "en")
        bot, client = _bot(services)
        orig_start = services.auto_controller.start
        called = {}

        async def fake_start():
            called["hit"] = True
            return await orig_start()

        services.auto_controller.start = fake_start  # type: ignore
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
        assert called.get("hit") is True
        txt = client.sent[-1][1]
        assert "auto trading" in txt.lower()
        await services.auto_controller.stop()
    finally:
        await shutdown_app(services)


# 11. /start_trading ru calls same service
async def test_start_trading_ru_calls_service(tmp_path) -> None:
    from unittest.mock import AsyncMock, patch

    settings = _settings(tmp_path, mode=TradingMode.DEMO)
    services = await build_app(settings)
    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        with patch.object(services.market, "refresh_order_books", new=AsyncMock()):
            with patch.object(services.market, "start_streams", new=AsyncMock()):
                await start_app(services)
    try:
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "ru")
        bot, client = _bot(services)
        orig_start = services.auto_controller.start
        called = {}

        async def fake_start():
            called["hit"] = True
            return await orig_start()

        services.auto_controller.start = fake_start  # type: ignore
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
        assert called.get("hit") is True
        txt = client.sent[-1][1]
        assert "автоторговля" in txt.lower()
        await services.auto_controller.stop()
    finally:
        await shutdown_app(services)


# 12. /stop_trading unchanged functionally (both langs)
async def test_stop_trading_both_langs(tmp_path) -> None:
    from unittest.mock import AsyncMock, patch

    for lang in ("en", "ru"):
        settings = _settings(tmp_path, mode=TradingMode.DEMO)
        services = await build_app(settings)
        with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
            with patch.object(services.market, "refresh_order_books", new=AsyncMock()):
                with patch.object(services.market, "start_streams", new=AsyncMock()):
                    await start_app(services)
        try:
            await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), lang)
            bot, client = _bot(services)
            # Start then stop
            await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
            client.sent.clear()
            # Track stop call
            orig_stop = services.auto_controller.stop
            called = {}

            async def fake_stop():
                called["hit"] = True
                return await orig_stop()

            services.auto_controller.stop = fake_stop  # type: ignore
            await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/stop_trading"))
            assert called.get("hit") is True
            txt = client.sent[-1][1]
            assert "stopped" in txt.lower() or "остановлена" in txt.lower()
        finally:
            await shutdown_app(services)


# 13. /pause and /resume unchanged functionally
async def test_pause_resume_both_langs(tmp_path) -> None:
    for lang in ("en", "ru"):
        services = await build_app(_settings(tmp_path))
        await start_app(services)
        try:
            await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), lang)
            bot, client = _bot(services)
            # Ensure clean
            if services.guard.is_halted:
                await services.release_kill_switch()
            await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/pause"))
            assert services.guard.is_halted is True
            txt = client.sent[-1][1]
            if lang == "en":
                assert "kill switch" in txt.lower()
            else:
                assert "стоп-кран" in txt.lower()
            client.sent.clear()
            await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/resume"))
            assert services.guard.is_halted is False
            txt2 = client.sent[-1][1]
            if lang == "en":
                assert "kill switch" in txt2.lower()
            else:
                assert "стоп-кран" in txt2.lower()
        finally:
            await shutdown_app(services)


# 14. Authorization unchanged
async def test_authorization_unchanged(tmp_path) -> None:
    services = await build_app(_settings(tmp_path))
    await start_app(services)
    try:
        bot, client = _bot(services)
        # Set language for stranger to ensure denied message is localized but still denied
        await services.bot_state.set(lang_storage_key(STRANGER_USER_ID), "ru")
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "en")
        # Stranger tries protected commands
        for cmd in ("/status", "/pause", "/resume", "/start_trading", "/stop_trading", "/opportunities", "/reconcile"):
            client.sent.clear()
            await bot.handle_update(_update(AUTHORIZED_CHAT_ID, cmd, user_id=STRANGER_USER_ID))
            assert "доступ запрещён" in client.sent[-1][1].lower() or "unauthorized" in client.sent[-1][1].lower()
        # Stranger selecting language must not grant access
        await bot.handle_update(_callback(AUTHORIZED_CHAT_ID, STRANGER_USER_ID, "en"))
        client.sent.clear()
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status", user_id=STRANGER_USER_ID))
        # Still denied (now in English)
        assert "unauthorized" in client.sent[-1][1].lower()
        # Authorized still works
        client.sent.clear()
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status", user_id=AUTHORIZED_USER_ID))
        assert "mode:" in client.sent[-1][1].lower()
    finally:
        await shutdown_app(services)


# 15. Invalid/missing falls back to picker
async def test_invalid_missing_fallback_to_picker(tmp_path) -> None:
    services = await build_app(_settings(tmp_path))
    await start_app(services)
    try:
        bot, client = _bot(services)
        # Missing -> picker for /start
        await services.bot_state.delete(lang_storage_key(AUTHORIZED_USER_ID))
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start"))
        assert LANGUAGE_PICKER_PROMPT in client.sent[-1][1]
        # Invalid value -> picker
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "xx")
        client.sent.clear()
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start"))
        assert LANGUAGE_PICKER_PROMPT in client.sent[-1][1]
        # Also for /status when authorized? Should show picker
        client.sent.clear()
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "invalid")
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
        assert LANGUAGE_PICKER_PROMPT in client.sent[-1][1]
        # Corrupted type (int)
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), 123)  # type: ignore
        client.sent.clear()
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start"))
        assert LANGUAGE_PICKER_PROMPT in client.sent[-1][1]
    finally:
        await shutdown_app(services)


# 16. Language survives reload/restart via persistent storage
async def test_language_survives_restart(tmp_path) -> None:
    settings = _settings(tmp_path)
    services_a = await build_app(settings)
    await start_app(services_a)
    try:
        await services_a.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "ru")
        # Simulate restart: new services with same DB file
        await shutdown_app(services_a)
        # Build new app with same settings (same DB URL)
        services_b = await build_app(settings)
        await start_app(services_b)
        try:
            val = await services_b.bot_state.get(lang_storage_key(AUTHORIZED_USER_ID))
            assert val == "ru"
            bot, client = _bot(services_b)
            await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
            assert "режим" in client.sent[-1][1].lower()
            # Change to en and restart again
            await services_b.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "en")
            await shutdown_app(services_b)
            services_c = await build_app(settings)
            await start_app(services_c)
            try:
                val2 = await services_c.bot_state.get(lang_storage_key(AUTHORIZED_USER_ID))
                assert val2 == "en"
                bot2, client2 = _bot(services_c)
                await bot2.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
                assert "mode:" in client2.sent[-1][1].lower()
            finally:
                await shutdown_app(services_c)
            # Need to prevent double shutdown of services_b (already shut)
            services_b = None  # type: ignore
        finally:
            if services_b is not None:
                await shutdown_app(services_b)
        # Prevent outer finally from double shutdown (services_a already shut)
        services_a = None  # type: ignore
    finally:
        if services_a is not None:
            await shutdown_app(services_a)


# Extra: changing language does not affect trading state
async def test_language_change_does_not_affect_trading(tmp_path) -> None:
    from unittest.mock import AsyncMock, patch

    settings = _settings(tmp_path, mode=TradingMode.DEMO)
    services = await build_app(settings)
    with patch("app.exchanges.preflight.run_demo_preflight", new=AsyncMock()):
        with patch.object(services.market, "refresh_order_books", new=AsyncMock()):
            with patch.object(services.market, "start_streams", new=AsyncMock()):
                await start_app(services)
    try:
        await services.bot_state.set(lang_storage_key(AUTHORIZED_USER_ID), "en")
        bot, client = _bot(services)
        # Start trading
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
        assert await services.auto_trading_enabled() is True
        assert services.auto_controller._task is not None
        # Change language via callback
        client.sent.clear()
        await bot.handle_update(_callback(AUTHORIZED_CHAT_ID, AUTHORIZED_USER_ID, "ru"))
        # Trading must still be running
        assert await services.auto_trading_enabled() is True
        assert services.auto_controller._task is not None
        assert not services.guard.is_halted
        # Change back
        await bot.handle_update(_callback(AUTHORIZED_CHAT_ID, AUTHORIZED_USER_ID, "en"))
        assert await services.auto_trading_enabled() is True
        assert services.auto_controller._task is not None
        await services.auto_controller.stop()
    finally:
        await shutdown_app(services)
