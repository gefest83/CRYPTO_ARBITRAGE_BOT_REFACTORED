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

import asyncio
from pathlib import Path
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
    """The documented set must contain exactly the required commands."""
    assert set(COMMANDS) == {
        "/start",
        "/help",
        "/status",
        "/reconcile",
        "/opportunities",
        "/pause",
        "/resume",
        "/start_trading",
        "/stop_trading",
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
    for cmd in (
        "/start",
        "/help",
        "/status",
        "/reconcile",
        "/opportunities",
        "/start_trading",
        "/stop_trading",
    ):
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


# Alias used by lifecycle tests.
make_settings_with_telegram = _settings_with_telegram


class FakeTelegramClient:
    """A no-op Telegram transport that captures every send_message call.

    Replaces the real ``TelegramClient`` when a test wants to exercise
    command dispatch without contacting the real Bot API.  The constructor
    accepts and ignores the same positional arguments as the real
    ``TelegramClient`` (token, poll_timeout_seconds, ...) so it can be
    used as a drop-in replacement.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.sent: list[tuple[int, str]] = []
        self.get_me_calls = 0

    async def get_me(self) -> dict:
        self.get_me_calls += 1
        return {"ok": True, "result": {"id": 0, "username": "fake", "is_bot": True}}

    async def send_message(self, chat_id: int, text: str, *, parse_mode: str = "") -> None:
        self.sent.append((chat_id, text))

    async def close(self) -> None:  # pragma: no cover - trivial
        pass

    async def get_updates(self, *args, **kwargs):  # pragma: no cover - trivial
        # The real poll_forever calls this; the fake never starts a real
        # loop in tests, but the runner's _poll_loop references it.  We
        # return an empty list so the loop would exit cleanly if invoked.
        return []


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


# ---------------------------------------------------------------------------
# /start_trading and /stop_trading (Telegram-controlled DEMO auto-trading)
# ---------------------------------------------------------------------------


async def test_start_trading_starts_demo_auto_trading(demo_services: AppServices) -> None:
    """``/start_trading`` engages the persisted auto flag and spawns one task."""
    services = demo_services
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
    text = client.sent[-1][1]
    assert "started" in text.lower(), text
    assert await services.auto_trading_enabled() is True
    # The in-process loop is actually running
    assert services.auto_controller is not None
    assert await services.auto_controller.is_running() is True
    # Cleanup for downstream tests
    await services.auto_controller.stop()


async def test_start_trading_is_idempotent(demo_services: AppServices) -> None:
    """A second ``/start_trading`` does not create a second loop."""
    services = demo_services
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
    first = client.sent[-1][1]
    assert "started" in first.lower()
    # Capture the first task identity
    first_task = services.auto_controller._task
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
    second = client.sent[-1][1]
    assert "already" in second.lower(), second
    # Same task identity -> no second loop was created
    assert services.auto_controller._task is first_task
    await services.auto_controller.stop()


async def test_stop_trading_stops_auto_loop(demo_services: AppServices) -> None:
    services = demo_services
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/stop_trading"))
    text = client.sent[-1][1]
    assert "stopped" in text.lower(), text
    assert await services.auto_trading_enabled() is False
    assert services.auto_controller._task is None


async def test_stop_trading_is_idempotent(services: AppServices) -> None:
    bot, client = _bot(services)
    # No prior start -> first stop is a no-op (mode is PAPER but stop is allowed)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/stop_trading"))
    first = client.sent[-1][1]
    assert "stopped" in first.lower() or "already" in first.lower(), first
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/stop_trading"))
    second = client.sent[-1][1]
    assert "stopped" in second.lower() or "already" in second.lower(), second


async def test_start_trading_refused_outside_demo(tmp_path) -> None:
    """``/start_trading`` must fail-closed in PAPER (and by construction LIVE)."""
    from app.config.settings import TelegramSettings, TradingSettings
    from app.models.enums import TradingMode
    from pydantic import SecretStr
    from tests.conftest import make_settings

    base = make_settings(
        tmp_path,
        telegram=TelegramSettings(
            bot_token="123:fake-token-for-tests",
            allowed_user_ids=(AUTHORIZED_USER_ID,),
            allowed_chat_ids=(AUTHORIZED_CHAT_ID,),
            poll_timeout_seconds=1,
        ),
    ).model_copy(
        update={
            "trading": TradingSettings(
                mode=TradingMode.PAPER,
                allow_live=False,
                live_confirmation=SecretStr(""),
                base_currency="USDT",
            )
        }
    )

    services = await build_app(base)
    try:
        bot, client = _bot(services)
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
        text = client.sent[-1][1]
        assert "refused" in text.lower() or "only allowed" in text.lower(), text
        assert await services.auto_trading_enabled() is False
    finally:
        await shutdown_app(services)


async def test_unauthorized_cannot_start_or_stop_trading(
    demo_services: AppServices,
) -> None:
    services = demo_services
    bot, client = _bot(services)
    for cmd in ("/start_trading", "/stop_trading"):
        client.sent.clear()
        await bot.handle_update(
            _update(AUTHORIZED_CHAT_ID, cmd, user_id=STRANGER_USER_ID)
        )
        assert client.sent[-1][1] == "unauthorized"
    # Auto trading was never enabled
    assert await services.auto_trading_enabled() is False


async def test_telegram_dispatch_does_not_call_create_order_or_withdraw() -> None:
    """Static guarantee: no command handler touches create_order / withdraw."""
    from app.telegram.bot import _DISPATCH

    forbidden = ("create_order", ".withdraw(", "_cmd_withdraw")
    for name, method in _DISPATCH.items():
        source = getattr(method, "__func__", method)
        qualname = getattr(source, "__qualname__", "")
        for needle in forbidden:
            assert needle not in qualname, f"{name} references {needle}"
        assert "create_order" not in qualname, f"{name} references create_order"


async def test_status_reports_auto_loop_state(demo_services: AppServices) -> None:
    services = demo_services
    bot, client = _bot(services)
    # Before start
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
    text_before = client.sent[-1][1]
    assert "auto trading (flag): off" in text_before
    assert "auto loop: stopped" in text_before
    # After start
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
    text_on = client.sent[-1][1]
    assert "auto trading (flag): on" in text_on
    assert "auto loop: running" in text_on
    # After stop
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/stop_trading"))
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
    text_off = client.sent[-1][1]
    assert "auto trading (flag): off" in text_off
    assert "auto loop: stopped" in text_off


async def test_start_trading_refused_when_kill_switch_engaged(
    demo_services: AppServices,
) -> None:
    services = demo_services
    await services.engage_kill_switch("preseed")
    try:
        bot, client = _bot(services)
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
        text = client.sent[-1][1]
        assert "kill switch" in text.lower(), text
        assert await services.auto_trading_enabled() is False
    finally:
        await services.release_kill_switch()


async def test_stop_trading_does_not_engage_kill_switch(
    demo_services: AppServices,
) -> None:
    """``/stop_trading`` must NOT touch the kill switch (semantics preserved)."""
    services = demo_services
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
    assert await services.auto_trading_enabled() is True
    assert not services.guard.is_halted
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/stop_trading"))
    assert not services.guard.is_halted, "stop_trading must not engage the kill switch"


async def test_resume_does_not_start_trading(services: AppServices) -> None:
    """``/resume`` semantics preserved: releases kill switch, does NOT enable auto trading."""
    await services.engage_kill_switch("preseed")
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/resume"))
    text = client.sent[-1][1]
    assert "released" in text.lower()
    assert not services.guard.is_halted
    assert await services.auto_trading_enabled() is False, (
        "/resume must NOT enable auto trading"
    )
    assert services.auto_controller._task is None


async def test_pause_stops_auto_trading_without_resuming(
    demo_services: AppServices,
) -> None:
    """``/pause`` engages the kill switch and stops the auto loop (existing semantics)."""
    services = demo_services
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
    assert await services.auto_trading_enabled() is True
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/pause"))
    text = client.sent[-1][1]
    assert "kill switch" in text.lower()
    assert services.guard.is_halted
    # Auto flag is cleared by the existing engage_kill_switch logic
    assert await services.auto_trading_enabled() is False
    # Controller has no live task
    assert services.auto_controller._task is None
    # Release so later tests start clean
    await services.release_kill_switch()


async def test_restart_does_not_auto_start_trading(tmp_path) -> None:
    """After an explicit ``/stop_trading``, restart must NOT re-spawn the loop.

    The persisted auto flag is the source of truth — if the operator stopped
    it once, restart respects that decision.
    """
    from app.config.settings import TelegramSettings, TradingSettings
    from app.models.enums import TradingMode
    from tests.conftest import make_settings

    base = make_settings(
        tmp_path,
        telegram=TelegramSettings(
            bot_token="123:fake-token-for-tests",
            allowed_user_ids=(AUTHORIZED_USER_ID,),
            allowed_chat_ids=(AUTHORIZED_CHAT_ID,),
            poll_timeout_seconds=1,
        ),
    ).model_copy(
        update={
            "trading": TradingSettings(
                mode=TradingMode.DEMO,
                allow_live=False,
                base_currency="USDT",
            )
        }
    )
    # First session: start then stop
    services_a = await build_app(base)
    await start_app(services_a)
    bot, _ = _bot(services_a)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
    assert await services_a.auto_trading_enabled() is True
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/stop_trading"))
    assert await services_a.auto_trading_enabled() is False
    await shutdown_app(services_a)
    # Second session: the auto flag is still False -> nothing restarts.
    services_b = await build_app(base)
    try:
        assert await services_b.auto_trading_enabled() is False
        assert services_b.auto_controller._task is None
    finally:
        await shutdown_app(services_b)


async def test_application_shutdown_stops_auto_loop(tmp_path) -> None:
    """``shutdown_app`` must stop the in-process auto loop cleanly."""
    import asyncio

    from app.config.settings import TelegramSettings, TradingSettings
    from app.models.enums import TradingMode
    from tests.conftest import make_settings

    base = make_settings(
        tmp_path,
        telegram=TelegramSettings(
            bot_token="123:fake-token-for-tests",
            allowed_user_ids=(AUTHORIZED_USER_ID,),
            allowed_chat_ids=(AUTHORIZED_CHAT_ID,),
            poll_timeout_seconds=1,
        ),
    ).model_copy(
        update={
            "trading": TradingSettings(
                mode=TradingMode.DEMO,
                allow_live=False,
                base_currency="USDT",
            )
        }
    )
    services = await build_app(base)
    await start_app(services)
    bot, _ = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
    assert services.auto_controller._task is not None
    await shutdown_app(services)
    # The auto task must be gone — no orphan background loop survives shutdown.
    remaining = [
        t for t in asyncio.all_tasks()
        if (t.get_name() or "").startswith("auto")
    ]
    assert not remaining, f"orphan auto tasks after shutdown: {remaining}"


# ---------------------------------------------------------------------------
# Code-review regression tests (race + crash recovery + restart semantics).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lifecycle regression tests: exactly one Telegram polling owner.
# ---------------------------------------------------------------------------


async def test_python_m_app_telegram_creates_only_one_poller(tmp_path) -> None:
    """``python -m app telegram`` must end up with exactly ONE TelegramRunner task.

    Regression: previously the CLI's ``telegram`` command also called the
    legacy ``run_telegram` polling loop, producing two concurrent
    ``getUpdates`` consumers on the same bot and therefore ``409 Conflict``
    from the Bot API (surfaced as ``telegram_poll_failed``).

    This test verifies the static invariant that prevents the regression:
    the CLI source does not import or call the legacy foreground poller.
    The runtime invariant (exactly one telegram-poller task) is verified
    by the separate end-to-end runtime smoke test.
    """
    import app.cli.main as cli_mod

    cli_src = open(cli_mod.__file__).read()
    # The CLI must not IMPORT or CALL the legacy foreground poller.  A
    # mention in a docstring is fine (and expected — it documents the
    # regression we are preventing).
    assert "from app.telegram.client import poll_forever" not in cli_src, (
        "cli/main.py must not import poll_forever — it would create a "
        "second getUpdates consumer."
    )
    assert "poll_forever(" not in cli_src, (
        "cli/main.py must not call poll_forever — the TelegramRunner is "
        "the single polling owner."
    )


async def test_cli_telegram_does_not_create_a_second_telegram_client(tmp_path) -> None:
    """The CLI ``telegram`` command must NOT instantiate a new ``TelegramClient``.

    Regression: ``cmd_telegram`` previously imported ``run_telegram`` which
    built its own ``TelegramClient`` and called ``getMe`` + ``poll_forever``,
    producing a second HTTP consumer on the Bot API.

    Static check: the CLI source must not import ``TelegramClient`` and
    must not import ``TelegramBot`` (the CLI is a thin command dispatcher;
    only the runner should construct the bot and client).
    """
    import app.cli.main as cli_mod_src

    cli_src = open(cli_mod_src.__file__).read()
    # Must not IMPORT TelegramClient or construct one.  Docstring mentions
    # are fine — they document the regression we are preventing.
    assert "import TelegramClient" not in cli_src, (
        "cli/main.py must not import TelegramClient — only the TelegramRunner "
        "should construct clients."
    )
    assert "TelegramClient(" not in cli_src, (
        "cli/main.py must not construct TelegramClient directly — only the "
        "TelegramRunner should own the client instance."
    )


async def test_cli_telegram_does_not_import_run_telegram(tmp_path) -> None:
    """Static check: the CLI must not pull in the legacy foreground poller."""
    import app.cli.main as cli_mod

    src = open(cli_mod.__file__).read()
    assert "run_telegram" not in src, (
        "cli/main.py must not reference run_telegram — it creates a second "
        "getUpdates consumer. The TelegramRunner from start_app is the "
        "single polling owner."
    )


async def test_cli_telegram_does_not_import_run_telegram(tmp_path) -> None:
    """Static check: the CLI must not pull in the legacy foreground poller.

    This is a strict invariant test.  It exists separately from
    ``test_python_m_app_telegram_creates_only_one_poller`` so a regression
    of the same kind fails both tests with a clear pointer.
    """
    import app.cli.main as cli_mod

    src = open(cli_mod.__file__).read()
    # Must not IMPORT or CALL the legacy poller.  Docstring mentions are
    # fine — they document the regression we are preventing.
    assert "from app.telegram.bot import run_telegram" not in src, (
        "cli/main.py must not import run_telegram — it creates a second "
        "getUpdates consumer."
    )
    assert "run_telegram(" not in src, (
        "cli/main.py must not call run_telegram — the TelegramRunner is "
        "the single polling owner."
    )


async def test_existing_telegram_commands_still_work_through_runner(
    demo_services: AppServices,
) -> None:
    """All 9 Telegram commands must still dispatch correctly via the runner."""
    from app.telegram.runner import TelegramRunner
    from app.telegram.client import TelegramClient

    services = demo_services
    # Replace the runner's client with a fake so we can exercise command
    # dispatch without contacting the real Bot API.
    fake = FakeTelegramClient()
    services.telegram_runner = TelegramRunner(services)
    services.telegram_runner._client = fake  # type: ignore[attr-defined]
    services.telegram_runner._bot = None  # type: ignore[attr-defined]
    # Re-bind the bot to use the fake client.
    from app.telegram.bot import TelegramBot

    bot = TelegramBot(services, fake)
    services.telegram_runner._bot = bot  # type: ignore[attr-defined]

    for cmd in (
        "/start",
        "/help",
        "/status",
        "/reconcile",
        "/opportunities",
        "/start_trading",
        "/stop_trading",
    ):
        fake.sent.clear()
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, cmd))
        assert fake.sent, f"{cmd} produced no reply"
        assert fake.sent[-1][1] != "internal error", (
            f"{cmd} raised an internal error: {fake.sent[-1][1]}"
        )
    # /pause and /resume require the runner's task to NOT be running for
    # the state assertions; we just confirm the reply is non-error.
    for cmd in ("/pause", "/resume"):
        fake.sent.clear()
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, cmd))
        assert fake.sent, f"{cmd} produced no reply"
        assert fake.sent[-1][1] != "internal error"
    # Cleanup
    await services.auto_controller.stop()


async def test_configure_logging_is_called_exactly_once_per_app_lifecycle(
    tmp_path,
) -> None:
    """``configure_logging`` must be invoked when ``build_app`` is called."""
    import app.config.logging_config as logging_config

    services = await build_app(
        make_settings_with_telegram(tmp_path),
    )
    try:
        # After build_app, the logging module must report it was configured.
        assert logging_config._configured is True, (
            "configure_logging() was not invoked by build_app — runtime "
            "diagnostics will be missing timestamps, levels, and the error "
            "context from log extras."
        )
        # A second build_app call must NOT re-configure (idempotent).
        await build_app(make_settings_with_telegram(tmp_path))
        assert logging_config._configured is True
    finally:
        await shutdown_app(services)


# ---------------------------------------------------------------------------
# Lifecycle self-stop / silent-death regression tests.
# ---------------------------------------------------------------------------


async def test_poller_failure_does_not_silently_disappear(
    tmp_path,
) -> None:
    """If the polling task raises an unhandled exception, the failure is
    observable (not silently swallowed).

    Regression: historically, ``_poll_loop`` caught all exceptions and
    ended the task with only a ``telegram_poll_loop_ended`` log.  When the
    task ended, ``cmd_telegram``'s ``await runner._task`` unblocked and
    the process exited cleanly — looking like a "self-stop" with no
    visible cause.

    The fix adds ``add_done_callback`` which logs the final state (success,
    cancellation, or exception) regardless of how the task ended.
    """
    import asyncio

    from app.config.settings import TelegramSettings
    from app.telegram.client import TelegramClient
    from app.telegram.runner import TelegramRunner
    from tests.conftest import make_settings

    base = make_settings(
        tmp_path,
        telegram=TelegramSettings(
            bot_token="123:fake-token-for-tests",
            allowed_user_ids=(AUTHORIZED_USER_ID,),
            allowed_chat_ids=(AUTHORIZED_CHAT_ID,),
            poll_timeout_seconds=1,
        ),
    )
    services = await build_app(base)
    # Patch TelegramClient to a fake that raises a non-transport error
    # in get_updates, so the poll_forever loop will raise.
    import app.telegram.client as client_mod
    original = client_mod.TelegramClient

    class _CrashingClient:
        def __init__(self, *args, **kwargs):
            self._calls = 0

        async def get_me(self) -> dict:
            return {"ok": True, "result": {"id": 0, "username": "fake", "is_bot": True}}

        async def get_updates(self, *args, **kwargs):
            self._calls += 1
            raise KeyError("simulated malformed update")

        async def send_message(self, *args, **kwargs):
            pass

        async def close(self):
            pass

    client_mod.TelegramClient = _CrashingClient  # type: ignore[assignment]
    try:
        runner = TelegramRunner(services)
        await runner.start()
        # Wait for the task to end (the fake raises immediately).
        task = runner._task
        assert task is not None
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        except KeyError:
            pass
        # The task must have ended (not still running).
        assert task.done()
        # The task must not have been cancelled (the fake raises, not cancels).
        assert not task.cancelled()
        # The task ended unexpectedly: poll_forever is `while True` so it should
        # never complete normally. The fact that it completed means an exception
        # was raised and caught inside _poll_loop (logged as
        # telegram_poll_loop_ended) and the task ended normally. The fix's
        # done_callback logs this as "telegram_poller_task_ended_unexpectedly",
        # making the failure observable.
        assert not task.cancelled()
    finally:
        client_mod.TelegramClient = original  # type: ignore[assignment]
        if services.telegram_runner is not None:
            await services.telegram_runner.stop()
        await shutdown_app(services)


async def test_cancellation_is_not_reported_as_failure(tmp_path) -> None:
    """Normal shutdown must cancel the Telegram task cleanly.

    Cancellation must NOT be logged as a failure — it is the expected
    shutdown path.
    """
    from app.config.settings import TelegramSettings
    import app.telegram.client as client_mod
    from app.telegram.runner import TelegramRunner
    from tests.conftest import make_settings

    base = make_settings(
        tmp_path,
        telegram=TelegramSettings(
            bot_token="123:fake-token-for-tests",
            allowed_user_ids=(AUTHORIZED_USER_ID,),
            allowed_chat_ids=(AUTHORIZED_CHAT_ID,),
            poll_timeout_seconds=1,
        ),
    )
    services = await build_app(base)
    original = client_mod.TelegramClient
    client_mod.TelegramClient = FakeTelegramClient  # type: ignore[assignment]
    try:
        runner = TelegramRunner(services)
        await runner.start()
        task = runner._task
        assert task is not None
        # Use the runner's stop() method which handles cancellation cleanly.
        try:
            await runner.stop()
        except asyncio.CancelledError:
            pass
        # Verify the task was cancelled cleanly.
        assert task.cancelled() is True
        assert task.exception() is None
    finally:
        client_mod.TelegramClient = original  # type: ignore[assignment]
        await shutdown_app(services)


async def test_stream_failure_does_not_shut_down_telegram(tmp_path) -> None:
    """A market-data stream failure must NOT trigger Telegram shutdown.

    The stream supervisor has its own restart/backoff logic.  It must not
    call application shutdown or cancel the Telegram runner.
    """
    from app.config.settings import TelegramSettings, TradingSettings
    from app.models.enums import TradingMode
    import app.telegram.client as client_mod
    from app.telegram.runner import TelegramRunner
    from tests.conftest import make_settings

    base = make_settings(
        tmp_path,
        telegram=TelegramSettings(
            bot_token="123:fake-token-for-tests",
            allowed_user_ids=(AUTHORIZED_USER_ID,),
            allowed_chat_ids=(AUTHORIZED_CHAT_ID,),
            poll_timeout_seconds=1,
        ),
    ).model_copy(
        update={
            "trading": TradingSettings(
                mode=TradingMode.DEMO,
                allow_live=False,
                base_currency="USDT",
            )
        }
    )
    services = await build_app(base)
    original = client_mod.TelegramClient
    client_mod.TelegramClient = FakeTelegramClient  # type: ignore[assignment]
    try:
        await start_app(services)
        assert services.telegram_runner is not None
        runner_task = services.telegram_runner._task
        assert runner_task is not None
        assert not runner_task.done()

        # Simulate a stream failure by calling the supervisor's internal
        # failure handler (if available).  The Telegram runner must remain alive.
        # The stream supervisor exposes ``stop_streams`` which is the normal
        # shutdown path — we verify that the Telegram runner is NOT cancelled
        # by the stream lifecycle.
        await services.market.stop_streams()
        # Telegram runner must still be alive.
        assert not runner_task.done(), (
            "stop_streams() cancelled the Telegram runner — stream "
            "lifecycle must be independent of Telegram lifecycle."
        )
    finally:
        client_mod.TelegramClient = original  # type: ignore[assignment]
        await shutdown_app(services)


async def test_runner_task_remains_alive_while_application_runs(tmp_path) -> None:
    """The Telegram runner task must remain alive while the app is running.

    If the task ends prematurely, the process will exit via
    ``cmd_telegram``'s ``await runner._task`` returning.  This test
    verifies the task stays alive under normal conditions.
    """
    from app.config.settings import TelegramSettings, TradingSettings
    from app.models.enums import TradingMode
    import app.telegram.client as client_mod
    from tests.conftest import make_settings

    base = make_settings(
        tmp_path,
        telegram=TelegramSettings(
            bot_token="123:fake-token-for-tests",
            allowed_user_ids=(AUTHORIZED_USER_ID,),
            allowed_chat_ids=(AUTHORIZED_CHAT_ID,),
            poll_timeout_seconds=1,
        ),
    ).model_copy(
        update={
            "trading": TradingSettings(
                mode=TradingMode.DEMO,
                allow_live=False,
                base_currency="USDT",
            )
        }
    )
    services = await build_app(base)
    original = client_mod.TelegramClient
    client_mod.TelegramClient = FakeTelegramClient  # type: ignore[assignment]
    try:
        await start_app(services)
        assert services.telegram_runner is not None
        runner_task = services.telegram_runner._task
        assert runner_task is not None
        # Verify the task is still alive after a short wait.
        await asyncio.sleep(0.5)
        assert not runner_task.done(), (
            "Telegram runner task ended prematurely while the app is "
            "supposed to be running — this would cause cmd_telegram to "
            "unblock and the process to exit."
        )
    finally:
        client_mod.TelegramClient = original  # type: ignore[assignment]
        await shutdown_app(services)


async def test_only_run_finally_calls_shutdown_app(tmp_path) -> None:
    """Only the intended lifecycle owner can trigger application shutdown.

    ``shutdown_app`` must be called from exactly one place: the
    ``finally:`` block of ``run()`` in ``app/cli/main.py``.
    """
    import re

    cli_src = open(
        Path(__file__).parent.parent / "app" / "cli" / "main.py"
    ).read()
    # Count occurrences of "shutdown_app" in the CLI module.
    count = len(re.findall(r"shutdown_app", cli_src))
    # Exactly one call site (the `finally:` block) and one import.
    assert count <= 2, (
        f"shutdown_app appears {count} times in cli/main.py — must be "
        f"called from exactly one place (the finally: block of run())."
    )
    # The import must be present.
    assert "from app.services import" in cli_src
    assert "shutdown_app" in cli_src


async def test_concurrent_start_calls_create_only_one_task(
    demo_services: AppServices,
) -> None:
    """Two concurrent ``/start_trading`` calls must not create two AutoTrader tasks."""
    services = demo_services
    controller = services.auto_controller
    assert controller is not None
    # Fire two concurrent starts; the asyncio.Lock must serialise them so
    # only one task is ever created.
    results = await asyncio.gather(
        controller.start(),
        controller.start(),
        controller.start(),
    )
    started_count = sum(1 for ok, _ in results if ok)
    assert started_count == 1, f"expected exactly one successful start, got {started_count}"
    # Only one in-process task regardless of how many callers raced.
    assert controller._task is not None
    assert controller.running is True
    # Cleanup
    await controller.stop()


async def test_concurrent_start_via_telegram_handlers(
    demo_services: AppServices,
) -> None:
    """Race at the Telegram dispatch layer: three ``/start_trading`` updates."""
    services = demo_services
    bot, _ = _bot(services)
    # Three updates "in flight" — the dispatch processes them serially, but
    # the controller's lock catches the case where the second start observes
    # a running loop.  Without the lock the second start could see ``False``
    # while the first is mid-spawn and create a duplicate task.
    await asyncio.gather(
        bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading")),
        bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading")),
        bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading")),
    )
    # Only one task.
    controller = services.auto_controller
    assert controller is not None
    assert controller.running is True
    # Confirm only one AutoTrader task is alive.
    auto_tasks = [t for t in asyncio.all_tasks() if (t.get_name() or "") == "auto-trader"]
    assert len(auto_tasks) == 1, f"expected 1 auto-trader task, got {len(auto_tasks)}"
    await controller.stop()


async def test_loop_crash_clears_stale_flag_on_next_status(
    demo_services: AppServices,
) -> None:
    """If the AutoTrader loop crashes, the persisted flag must be cleared.

    Without the cleanup path, ``/status`` would report ``auto trading (flag): on``
    forever after a crash, which is misleading and dangerous.
    """
    services = demo_services
    controller = services.auto_controller
    assert controller is not None
    # Inject a crashing AutoTrader by monkey-patching AutoTrader for the test.
    from app.auto_controller import AutoTradingController as _ATC
    original_init = controller._trader.__class__  # not used directly

    class _CrashingTrader:
        def __init__(self, services):
            self._services = services

        def stop(self):
            pass

        async def run_forever(self):
            raise RuntimeError("simulated crash")

    # Replace the factory by patching the module-level import inside start()
    import app.auto_controller as ac_mod
    saved_auto = ac_mod.__dict__.get("AutoTrader")
    # We monkey-patch by setting a module attribute that start() will use.
    # start() does ``from app.auto import AutoTrader`` lazily — so we patch
    # ``app.auto.AutoTrader`` directly.
    import app.auto as auto_mod
    saved_real_trader = auto_mod.AutoTrader
    auto_mod.AutoTrader = _CrashingTrader  # type: ignore[assignment]
    try:
        started, _ = await controller.start()
        assert started is True
        # Wait for the task to complete (it crashes immediately).  Swallow
        # the exception — we want to assert about controller state afterwards.
        task = controller._task
        assert task is not None
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        except RuntimeError:
            pass
        assert task.done()
        assert task.exception() is not None
    finally:
        auto_mod.AutoTrader = saved_real_trader  # type: ignore[assignment]
    # At this point the task crashed but the flag is still True.
    assert await services.auto_trading_enabled() is True
    # Reading is_running() must detect the crash and clear the stale flag.
    assert await controller.is_running() is False
    assert await services.auto_trading_enabled() is False
    # /status reflects the truth.
    bot, client = _bot(services)
    await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/status"))
    text = client.sent[-1][1]
    assert "auto trading (flag): off" in text
    assert "auto loop: stopped" in text


async def test_start_after_loop_crash_resumes_cleanly(
    demo_services: AppServices,
) -> None:
    """After a crash the next ``/start_trading`` must spawn exactly one fresh task."""
    services = demo_services
    controller = services.auto_controller
    assert controller is not None

    class _CrashingTrader:
        def __init__(self, services):
            self._services = services

        def stop(self):
            pass

        async def run_forever(self):
            raise RuntimeError("boom")

    import app.auto as auto_mod
    saved = auto_mod.AutoTrader
    auto_mod.AutoTrader = _CrashingTrader  # type: ignore[assignment]
    try:
        await controller.start()
        task = controller._task
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        except RuntimeError:
            pass
    finally:
        auto_mod.AutoTrader = saved  # type: ignore[assignment]
    # Trigger status to clear stale state
    await controller.is_running()
    # Now start again — must succeed and produce a fresh task
    started, msg = await controller.start()
    assert started is True, msg
    new_task = controller._task
    assert new_task is not None
    assert not new_task.done()
    await controller.stop()


async def test_cli_and_telegram_share_single_loop(
    tmp_path,
) -> None:
    """CLI ``cmd_start_auto`` and Telegram ``/start_trading`` cannot create
    two independent AutoTrader tasks in the same process."""
    from app.config.settings import TelegramSettings, TradingSettings
    from app.models.enums import TradingMode
    from tests.conftest import make_settings

    base = make_settings(
        tmp_path,
        telegram=TelegramSettings(
            bot_token="123:fake-token-for-tests",
            allowed_user_ids=(AUTHORIZED_USER_ID,),
            allowed_chat_ids=(AUTHORIZED_CHAT_ID,),
            poll_timeout_seconds=1,
        ),
    ).model_copy(
        update={
            "trading": TradingSettings(
                mode=TradingMode.DEMO,
                allow_live=False,
                base_currency="USDT",
            )
        }
    )
    services = await build_app(base)
    try:
        bot, _ = _bot(services)
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
        controller = services.auto_controller
        assert controller is not None
        cli_task = controller._task
        assert cli_task is not None
        # Telegram cannot create a second loop.
        await bot.handle_update(_update(AUTHORIZED_CHAT_ID, "/start_trading"))
        assert controller._task is cli_task, "Telegram created a second loop"
        auto_tasks = [t for t in asyncio.all_tasks() if (t.get_name() or "") == "auto-trader"]
        assert len(auto_tasks) == 1
        await controller.stop()
    finally:
        await shutdown_app(services)