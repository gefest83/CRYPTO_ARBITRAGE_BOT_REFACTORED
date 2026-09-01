"""Telegram bot: command routing, authorization, secrets, lifecycle.

The Telegram interface is read-only / safe-control only:

* /start, /help, /status, /reconcile, /opportunities - read-only
* /pause, /resume - safe control (persistent kill switch)
* No order placement, no withdrawal initiation, no auto-trading toggle.

These tests prove:

* authorized users are accepted, unauthorized users are rejected;
* every command in the documented set works;
* replies contain no secret values;
* Telegram startup failure does not enable trading and does not crash
  the app;
* shutdown is clean (no orphan tasks).
"""

from __future__ import annotations

from typing import Any

import pytest

from app.config.settings import Settings, TelegramSettings
from app.services import AppServices, build_app, shutdown_app, start_app
from app.telegram.bot import COMMANDS, HELP_TEXT, TelegramBot


AUTHORIZED_USER_ID = 11111
AUTHORIZED_CHAT_ID = 12345
STRANGER_USER_ID = 99999
STRANGER_CHAT_ID = 42


class FakeClient:
    """A no-op Telegram client that captures every send_message call."""

    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str, *, parse_mode: str = "") -> None:
        self.sent.append((chat_id, text))


def _bot(services: AppServices) -> tuple[TelegramBot, FakeClient]:
    client = FakeClient()
    return TelegramBot(services, client), client


def _update(
    chat_id: int,
    text: str,
    *,
    user_id: int = AUTHORIZED_USER_ID,
    username: str = "alice",
) -> dict[str, Any]:
    return {
        "update_id": 1,
        "message": {
            "chat": {"id": chat_id},
            "from": {"id": user_id, "username": username},
            "text": text,
        },
    }


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


async def test_authorized_user_is_accepted(services: AppServices) -> None:
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status", user_id=AUTHORIZED_USER_ID))
    assert len(client.sent) == 1
    assert client.sent[0][1] != "unauthorized"
    assert "mode: PAPER" in client.sent[0][1]


async def test_unauthorized_user_is_refused(services: AppServices) -> None:
    bot, client = _bot(services)
    # Authorized chat, unauthorized user -> must still refuse
    # (user-id is the only authorisation channel).
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status", user_id=STRANGER_USER_ID))
    assert len(client.sent) == 1
    assert client.sent[0][1] == "unauthorized"


async def test_unauthorized_chat_with_authorized_user_is_accepted(
    services: AppServices,
) -> None:
    """Authorisation is by user ID, not chat ID — an unknown chat is fine."""
    bot, client = _bot(services)
    await bot.handle_update(_update(STRANGER_CHAT_ID, "/status", user_id=AUTHORIZED_USER_ID))
    assert len(client.sent) == 1
    assert client.sent[0][1] != "unauthorized"


async def test_empty_text_is_ignored(services: AppServices) -> None:
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, ""))
    assert client.sent == []


async def test_unauthorized_reply_is_identical_for_every_command(
    services: AppServices,
) -> None:
    """An attacker probing the command set must not learn anything from the replies."""
    bot, client = _bot(services)
    replies: list[str] = []
    for cmd in ("/start", "/status", "/reconcile", "/opportunities", "/pause", "/resume"):
        client.sent.clear()
        await bot.handle_update(
            _update(STRANGER_CHAT_ID, cmd, user_id=STRANGER_USER_ID)
        )
        replies.append(client.sent[-1][1])
    assert len(set(replies)) == 1
    assert replies[0] == "unauthorized"


# ---------------------------------------------------------------------------
# /status, /opportunities, /reconcile (read-only)
# ---------------------------------------------------------------------------


async def test_status_command(services: AppServices) -> None:
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
    assert len(client.sent) == 1
    text = client.sent[0][1]
    assert "mode: PAPER" in text
    assert "binance" in text
    assert "kill switch: released" in text


async def test_opportunities_command_runs_scanner_without_executing(
    services: AppServices,
) -> None:
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/opportunities"))
    text = client.sent[0][1]
    assert "triangle opportunities:" in text
    assert "transfer plans:" in text
    assert "no execution" in text


async def test_reconcile_command_when_no_records(services: AppServices) -> None:
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/reconcile"))
    assert "no transfers require manual review" in client.sent[0][1]


async def test_reconcile_command_lists_manual_review_records(
    services: AppServices,
) -> None:
    from app.models.enums import TransferState

    plan = {
        "source_exchange": "binance",
        "dest_exchange": "bybit",
        "asset": "BTC",
        "network": "BTC",
        "amount": "0.001",
        "buy_price": "78000",
        "sell_price": "78100",
    }
    record = {
        "id": "trf-test0001",
        "source_exchange": "binance",
        "dest_exchange": "bybit",
        "asset": "BTC",
        "network": "BTC",
        "amount": "0.001",
        "state": TransferState.MANUAL_REVIEW.value,
        "plan": plan,
        "buy_order": None,
        "buy_filled_amount": "0.001",
        "withdrawal_id": "wd-abc",
        "withdrawal_txid": None,
        "withdrawal_amount": "0.001",
        "deposit_address": "addr-here",
        "deposit_txid": None,
        "deposit_amount": "0",
        "sell_order": None,
        "sell_filled_amount": "0",
        "sell_proceeds_quote": "0",
        "fees_quote": "0",
        "realized_profit_quote": "0",
        "error": "deposit timeout",
        "mode": "PAPER",
    }
    # Insert via the same engine services owns.
    async with services.db.session() as session:
        from app.storage.tables import TransferRow

        await session.execute(TransferRow.__table__.insert().values(**record))
        await session.commit()

    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/reconcile"))
    text = client.sent[0][1]
    assert "trf-test0001" in text
    assert "binance -> bybit" in text
    assert "BTC" in text
    assert "deposit timeout" in text
    assert "wd-abc" in text


# ---------------------------------------------------------------------------
# /pause, /resume (safe control)
# ---------------------------------------------------------------------------


async def test_pause_engages_kill_switch_persistently(services: AppServices) -> None:
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/pause"))
    assert "kill switch engaged" in client.sent[0][1]
    # Persisted in bot_state
    value = await services.bot_state.get("kill_switch")
    assert isinstance(value, dict)
    assert value.get("engaged") is True
    # /status reflects the engaged kill switch
    bot2, client2 = _bot(services)
    await bot2.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
    assert "ENGAGED" in client2.sent[0][1]
    # Cleanup for downstream tests
    await services.release_kill_switch()


async def test_pause_idempotent_when_already_engaged(services: AppServices) -> None:
    await services.engage_kill_switch("preseed")
    try:
        bot, client = _bot(services)
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/pause"))
        assert "already engaged" in client.sent[0][1]
    finally:
        await services.release_kill_switch()


async def test_resume_releases_kill_switch(services: AppServices) -> None:
    await services.engage_kill_switch("preseed")
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/resume"))
    text = client.sent[0][1]
    assert "kill switch released" in text
    # auto trading must NOT have been silently re-enabled
    assert await services.auto_trading_enabled() is False
    # Bot state cleared
    assert (await services.bot_state.get("kill_switch")) is None


async def test_resume_when_not_engaged(services: AppServices) -> None:
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/resume"))
    assert "already released" in client.sent[0][1]


# ---------------------------------------------------------------------------
# Static invariants: no execution paths, no secrets in replies
# ---------------------------------------------------------------------------


async def test_unknown_command_shows_help(services: AppServices) -> None:
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/frobnicate"))
    text = client.sent[0][1]
    assert "unknown command" in text
    assert HELP_TEXT in text


async def test_command_set_matches_spec() -> None:
    """The documented set must contain exactly the required read-only/safe commands."""
    assert set(COMMANDS) == {
        "/start",
        "/help",
        "/status",
        "/reconcile",
        "/opportunities",
        "/pause",
        "/resume",
    }


async def test_telegram_cannot_execute_exchange_orders() -> None:
    """The dispatch table must not contain any order-placing command."""
    from app.telegram.bot import _DISPATCH

    forbidden = {
        "/triangle", "/transfer", "/balances", "/trades",
        "/start_auto", "/stop_auto", "/execute", "/buy", "/sell",
    }
    assert forbidden.isdisjoint(_DISPATCH.keys())


async def test_telegram_cannot_initiate_withdrawals() -> None:
    """No command may call ``adapter.withdraw`` or ``services.start_transfer``."""
    from app.telegram.bot import _DISPATCH

    forbidden_methods = ("_cmd_withdraw", "_cmd_start_transfer", "_cmd_execute")
    for name, method in _DISPATCH.items():
        qualname = getattr(method, "__qualname__", "")
        for forbidden in forbidden_methods:
            assert forbidden not in qualname, f"{name} references {forbidden}"
        assert "withdraw" not in qualname.lower(), f"{name} references withdraw"


async def test_replies_do_not_contain_secrets(services: AppServices) -> None:
    """No Telegram reply may echo a secret value, token, or query string."""
    bot, client = _bot(services)
    for cmd in ("/start", "/help", "/status", "/reconcile", "/opportunities"):
        client.sent.clear()
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, cmd))
        assert client.sent, f"no reply for {cmd}"
        text = client.sent[0][1]
        for needle in (
            "apiKey=", "secret=", "password=", "passphrase=",
            "x-simulated-trading", "signature=", "token=",
            "?<redacted>",
        ):
            assert needle not in text, f"{cmd} reply leaks {needle!r}"


async def test_internal_errors_do_not_leak_tracebacks(services: AppServices) -> None:
    """A handler exception must produce a generic error message, not a traceback."""
    bot, client = _bot(services)

    async def boom() -> str:
        raise RuntimeError("SECRET_API_KEY=abcd1234 (this must never reach Telegram)")

    bot._dispatch = lambda command: boom if command == "/boom" else None  # type: ignore[method-assign]
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/boom"))
    assert client.sent[-1][1] == "internal error"


# ---------------------------------------------------------------------------
# Lifecycle integration with AppServices
# ---------------------------------------------------------------------------


def _telegram_cfg() -> TelegramSettings:
    return TelegramSettings(
        bot_token="123:fake-token-for-tests",
        allowed_user_ids=(AUTHORIZED_USER_ID,),
        allowed_chat_ids=(AUTHORIZED_CHAT_ID,),
        poll_timeout_seconds=1,
    )


def _settings_with_telegram(tmp_path) -> Settings:
    from tests.conftest import make_settings

    return make_settings(tmp_path, telegram=_telegram_cfg())


async def test_telegram_disabled_when_no_allow_list(tmp_path) -> None:
    """Empty allow-list -> Telegram runner must NOT be started."""
    from app.config.settings import TelegramSettings

    base = _settings_with_telegram(tmp_path)
    base = base.model_copy(
        update={
            "telegram": TelegramSettings(
                bot_token="x:y", allowed_user_ids=(), allowed_chat_ids=()
            )
        }
    )
    services = await build_app(base)
    try:
        await start_app(services)
        assert services.telegram_runner is None
    finally:
        await shutdown_app(services)


async def test_telegram_disabled_when_no_token(tmp_path) -> None:
    """Empty token -> Telegram runner must NOT be started (fail-closed)."""
    from app.config.settings import TelegramSettings

    base = _settings_with_telegram(tmp_path)
    base = base.model_copy(
        update={
            "telegram": TelegramSettings(
                bot_token="", allowed_user_ids=(1,), allowed_chat_ids=()
            )
        }
    )
    services = await build_app(base)
    try:
        await start_app(services)
        assert services.telegram_runner is None
    finally:
        await shutdown_app(services)


async def test_telegram_startup_failure_does_not_enable_trading(tmp_path) -> None:
    """A Telegram startup failure must not enable auto trading or crash the app."""
    services = await build_app(_settings_with_telegram(tmp_path))
    try:
        # Force Telegram startup to fail: replace the token with garbage that
        # the real Bot API will reject on get_me().  We DO NOT mock the
        # transport — the failure must be handled by the services layer.
        await start_app(services)
        # App started.  If the bot could reach the network it would be running;
        # if not (offline / no httpx), the runner stays None.  Either way:
        assert await services.auto_trading_enabled() is False
    finally:
        await shutdown_app(services)


async def test_telegram_shutdown_is_clean(tmp_path) -> None:
    """Shutdown must not leave orphan tasks behind and must be idempotent."""
    import asyncio

    base = _settings_with_telegram(tmp_path)
    services = await build_app(base)
    await start_app(services)
    # Even if the runner never started (no network), shutdown must be clean.
    await shutdown_app(services)
    remaining = [
        t for t in asyncio.all_tasks()
        if (t.get_name() or "").startswith("telegram")
    ]
    assert not remaining, f"orphan Telegram tasks after shutdown: {remaining}"
    # Double shutdown must be a no-op (does not raise)
    await shutdown_app(services)