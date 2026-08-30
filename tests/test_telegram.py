"""Telegram bot: command routing, authorization, reuse of AppServices."""

from app.services import AppServices
from app.telegram.bot import COMMANDS, TelegramBot

AUTHORIZED_CHAT = 12345  # the chat allow-listed in tests/conftest.py
STRANGER_CHAT = 42


class FakeClient:
    def __init__(self):
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))


def _bot(services: AppServices) -> tuple[TelegramBot, FakeClient]:
    client = FakeClient()
    return TelegramBot(services, client), client


def _update(chat_id: int, text: str) -> dict:
    return {"update_id": 1, "message": {"chat": {"id": chat_id}, "text": text}}


async def test_unauthorized_chat_is_refused(services: AppServices):
    bot, client = _bot(services)
    await bot.handle_update(_update(STRANGER_CHAT, "/status"))
    assert len(client.sent) == 1
    assert "unauthorized" in client.sent[0][1]


async def test_status_reuses_services(services: AppServices):
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT, "/status"))
    assert len(client.sent) == 1
    text = client.sent[0][1]
    assert "mode: PAPER" in text
    assert "binance" in text
    assert "kill switch: released" in text


async def test_opportunities_command_lists_scans(services: AppServices):
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT, "/opportunities"))
    text = client.sent[0][1]
    assert "triangles:" in text
    assert "transfers:" in text


async def test_triangle_command_executes_via_services(services: AppServices):
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT, "/triangle"))
    text = client.sent[0][1]
    assert "trade " in text or "REJECTED" in text


async def test_transfer_without_args_plans_only(services: AppServices):
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT, "/transfer"))
    text = client.sent[0][1]
    assert "to execute" in text  # plan shown, no execution without args


async def test_transfer_with_args_executes(services: AppServices):
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT, "/transfer AVAX 1"))
    text = client.sent[0][1]
    assert any(marker in text for marker in ("started", "no transfer plans", "REJECTED", "failed"))


async def test_start_stop_auto_toggle(services: AppServices):
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT, "/start_auto"))
    assert "auto trading enabled" in client.sent[0][1]
    assert await services.auto_trading_enabled()
    await bot.handle_update(_update(AUTHORIZED_CHAT, "/stop_auto"))
    assert "auto trading disabled" in client.sent[1][1]
    assert not await services.auto_trading_enabled()
    assert bot._auto_task is None or bot._auto_task.done()


async def test_start_auto_refused_while_kill_switch_engaged(services: AppServices):
    await services.engage_kill_switch("telegram test")
    try:
        bot, client = _bot(services)
        await bot.handle_update(_update(AUTHORIZED_CHAT, "/start_auto"))
        assert "kill switch ENGAGED" in client.sent[0][1]
    finally:
        await services.release_kill_switch()


async def test_unknown_command_shows_help(services: AppServices):
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT, "/frobnicate"))
    assert "unknown command" in client.sent[0][1]
    assert all(command in client.sent[0][1] for command in COMMANDS)


async def test_all_required_commands_exist():
    from app.telegram.bot import _HELP

    for command in COMMANDS:
        assert command in _HELP


async def test_commands_cover_required_surface():
    assert set(COMMANDS) == {
        "/status",
        "/balances",
        "/opportunities",
        "/triangle",
        "/transfer",
        "/trades",
        "/start_auto",
        "/stop_auto",
    }
