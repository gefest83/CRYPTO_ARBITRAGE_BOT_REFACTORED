"""Telegram operator interface: read-only / safe-control only.

This module is a thin layer over :class:`app.services.AppServices` — the same
services the CLI uses.  No trading logic lives here.

Scope (deliberately restricted)
-------------------------------

The Telegram bot is a **secondary control interface**.  Every command it
exposes is either read-only, a reconciliation query, or a safe state change
(persistent kill switch + auto-trading flag).  It must NOT be able to:

* place exchange orders (no ``/triangle``, no ``/transfer`` execution, no
  ``/start_auto`` / ``/stop_auto`` toggling of the background loop);
* initiate withdrawals;
* expose configuration, credentials, exchange API keys, the database URL or
  raw exception tracebacks to the operator.

The kill-switch + auto-trading flag toggles (commands ``/pause`` / ``/resume``)
go through the existing :class:`app.execution.guard.ExecutionGuard`, which
itself is persisted in :class:`app.storage.repositories.BotStateRepository`
and survives restart.  They do NOT bypass the LIVE confirmation, mode policy
or risk engine — ``/resume`` only restores a flag that the rest of the bot
already enforces.

Security (fail-closed)
----------------------

* only Telegram user IDs in ``CAT_TELEGRAM__ALLOWED_USER_IDS`` (and, for
  back-compat, chats in ``CAT_TELEGRAM__ALLOWED_CHAT_IDS``) may command the
  bot;
* an empty allow-list disables every command;
* the bot token is a :class:`~pydantic.SecretStr` and never appears in logs
  or replies;
* error replies are short, generic and pass through
  :func:`app.exchanges.sanitize.redact_secrets` before being sent.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from app.config.logging_config import get_logger
from app.exchanges.sanitize import redact_secrets
from app.services import AppServices

from .client import TelegramClient, poll_forever

__all__ = ["COMMANDS", "HELP_TEXT", "TelegramBot", "run_telegram"]

logger = get_logger("telegram.bot")


COMMANDS: tuple[str, ...] = (
    "/start",
    "/help",
    "/status",
    "/reconcile",
    "/opportunities",
    "/pause",
    "/resume",
)

HELP_TEXT = (
    "crypto-arbitrage-bot — operator commands:\n"
    "/start        - welcome / help\n"
    "/help         - this message\n"
    "/status       - mode, kill switch, exchanges, transfers, recent trades\n"
    "/reconcile    - list MANUAL_REVIEW transfers (read-only)\n"
    "/opportunities - scan triangles (no execution)\n"
    "/pause        - engage the persistent kill switch (safe)\n"
    "/resume       - release the kill switch (safe)\n"
    "\n"
    "Order placement, transfers and withdrawals are NOT exposed here — "
    "use the CLI for any execution that moves funds."
)


# Maximum number of MANUAL_REVIEW records shown by /reconcile.
_RECONCILE_LIMIT = 20

# Maximum number of triangle / transfer opportunities shown by /opportunities.
_OPPORTUNITY_LIMIT = 5

# Safe message sent to unauthorized chats / users.  Identical wording regardless
# of which command they tried, so an attacker cannot probe the command set.
_DENIED_MESSAGE = "unauthorized"

# Safe message sent on internal errors.  Identical wording regardless of cause.
_INTERNAL_ERROR_MESSAGE = "internal error"

# Maximum reply length — leaves headroom under Telegram's 4096 char limit.
_MAX_REPLY_LENGTH = 3500


class TelegramBot:
    """Stateless command dispatcher backed by :class:`AppServices`.

    The bot holds NO execution state: every command is delegated to
    :class:`AppServices`.  This makes startup/shutdown trivial and means a
    crashed bot can be replaced at any time without losing trading context.
    """

    def __init__(self, services: AppServices, client: TelegramClient) -> None:
        self._services = services
        self._client = client
        self._auto_task: asyncio.Task | None = None

    # ---------------------------------------------------------------- dispatch
    async def handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        user = message.get("from") or {}
        user_id = user.get("id")
        text = str(message.get("text") or "").strip()
        if chat_id is None or not text:
            return
        if not self._authorized(user_id, chat_id):
            # Never reveal whether the chat or the user id was the problem.
            await self._safe_send(chat_id, _DENIED_MESSAGE)
            return
        command, _, _rest = text.partition(" ")
        command = command.split("@")[0].lower()
        handler = self._dispatch(command)
        if handler is None:
            await self._safe_send(chat_id, f"unknown command\n\n{HELP_TEXT}")
            return
        try:
            reply = await handler()
        except Exception as exc:  # noqa: BLE001 - one bad command stops nothing
            logger.error(
                "telegram_command_failed",
                extra={
                    "command": command,
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "error": redact_secrets(str(exc))[:300],
                },
            )
            reply = _INTERNAL_ERROR_MESSAGE
        await self._safe_send(chat_id, reply or "")

    def _dispatch(self, command: str) -> Callable[[], Awaitable[str]] | None:
        method = _DISPATCH.get(command)
        if method is None:
            return None
        # Bind the method to this bot so the dispatch can call it as ``handler()``.
        bound = method.__get__(self, type(self))
        return bound

    def _authorized(self, user_id: int | None, chat_id: int | None) -> bool:
        """Authorisation is by Telegram user ID alone.

        Per the operator-interface spec, only configured user IDs may command
        the bot.  The legacy ``CAT_TELEGRAM__ALLOWED_CHAT_IDS`` list is no
        longer consulted; operators must configure ``CAT_TELEGRAM__ALLOWED_USER_IDS``.
        """
        cfg = self._services.settings.telegram
        return user_id is not None and user_id in cfg.allowed_user_ids

    # ---------------------------------------------------------------- commands
    async def _cmd_start(self) -> str:
        return HELP_TEXT

    async def _cmd_help(self) -> str:
        return HELP_TEXT

    async def _cmd_status(self) -> str:
        status = await self._services.status()
        guard = status["guard"]
        lines: list[str] = [
            f"mode: {status['mode']}",
            f"uptime: {status['uptime_seconds']}s",
            f"trading enabled: {guard['trading_enabled']}",
            "kill switch: "
            + (
                f"ENGAGED ({guard['halt_reason']})"
                if guard["halted"] == "true"
                else "released"
            ),
            f"auto trading: {'on' if status['auto_trading'] else 'off'}",
            "exchanges:",
        ]
        for venue, info in status["exchanges"].items():
            lines.append(f"  {venue}: {info['status']} (keys {info['credentials']})")
        md = status["market_data"]
        lines.append(
            f"market data: {md['order_books']} books / {md['tickers']} tickers "
            f"across {md['exchanges']} venues"
        )
        risk = status["risk"]
        lines.append(f"daily pnl: {risk['daily_pnl']}")
        lines.append(f"open transfers: {risk['open_transfers']}")
        if status["transfers_open"]:
            lines.append("open transfers:")
            for transfer in status["transfers_open"]:
                lines.append(
                    f"  {transfer['route']} {transfer['asset']} "
                    f"{transfer['amount']}: {transfer['state']}"
                )
        if status["recent_trades"]:
            lines.append("recent trades:")
            for trade in status["recent_trades"]:
                lines.append(
                    f"  {trade['strategy']} {trade['route']} "
                    f"{trade['status']} net {trade['net_profit']}"
                )
        return "\n".join(lines)

    async def _cmd_reconcile(self) -> str:
        from app.models.enums import TransferState

        records = await self._services.transfers.list_by_state(
            TransferState.MANUAL_REVIEW.value
        )
        if not records:
            return "no transfers require manual review"
        lines = [f"{len(records)} transfer(s) require manual review:"]
        for record in records[:_RECONCILE_LIMIT]:
            lines.append("")
            lines.append(f"  id        : {record.id}")
            lines.append(f"  state     : {record.state.value}")
            lines.append(
                f"  route     : {record.source_exchange} -> {record.dest_exchange}"
            )
            lines.append(
                f"  asset     : {record.asset} (planned {record.amount})"
            )
            if record.buy_filled_amount and record.buy_filled_amount > 0:
                lines.append(
                    f"  held      : {record.buy_filled_amount} {record.asset} "
                    f"on {record.source_exchange}"
                )
            withdrawal_bits = []
            if record.withdrawal_id:
                withdrawal_bits.append(f"id={record.withdrawal_id}")
            if record.withdrawal_txid:
                withdrawal_bits.append(f"txid={record.withdrawal_txid}")
            if record.withdrawal_amount and record.withdrawal_amount > 0:
                lines.append(
                    f"  withdrawal: {record.withdrawal_amount} {record.asset} "
                    f"on {record.source_exchange} "
                    + ("(" + ", ".join(withdrawal_bits) + ")" if withdrawal_bits else "")
                )
            lines.append(f"  created   : {record.created_at.isoformat()}")
            lines.append(f"  updated   : {record.updated_at.isoformat()}")
            if record.error:
                lines.append(f"  reason    : {record.error}")
        if len(records) > _RECONCILE_LIMIT:
            lines.append("")
            lines.append(
                f"... and {len(records) - _RECONCILE_LIMIT} more (CLI has the full list)"
            )
        return "\n".join(lines)

    async def _cmd_opportunities(self) -> str:
        opportunities = await self._services.scan_triangles()
        plans = await self._services.plan_transfers()
        lines = [
            f"triangle opportunities: {len(opportunities)}",
        ]
        if opportunities:
            for opp in opportunities[:_OPPORTUNITY_LIMIT]:
                lines.append(
                    f"  {opp.direction} net {opp.net_profit_bps} bps "
                    f"notional {opp.size_notional_quote}"
                )
        else:
            lines.append("  (none above configured minimum)")
        lines.append(f"transfer plans: {len(plans)}")
        if plans:
            for plan in plans[:_OPPORTUNITY_LIMIT]:
                lines.append(
                    f"  {plan.source_exchange}->{plan.dest_exchange} {plan.asset} "
                    f"{plan.amount} via {plan.network}: net {plan.net_profit_bps:.1f} bps"
                )
        else:
            lines.append("  (none above configured minimum)")
        lines.append("\n(no execution — view-only)")
        return "\n".join(lines)

    async def _cmd_pause(self) -> str:
        """Engage the persistent kill switch (safe)."""
        if self._services.guard.is_halted:
            return (
                f"kill switch already engaged: {self._services.guard.halt_reason}"
            )
        await self._services.engage_kill_switch("telegram /pause")
        # engage_kill_switch also disables auto trading; mirror the state for
        # operators watching /status.
        await self._services.set_auto_trading(False)
        return "kill switch engaged (persisted across restart)"

    async def _cmd_resume(self) -> str:
        """Release the persistent kill switch (safe)."""
        if not self._services.guard.is_halted:
            return "kill switch already released"
        # The guard is the single source of truth for "may we trade?".  This
        # command restores it but does NOT enable auto trading — operators
        # must do that explicitly through the CLI.
        await self._services.release_kill_switch()
        return (
            "kill switch released — auto trading is still OFF; "
            "start it from the CLI when ready"
        )

    # ---------------------------------------------------------------- send
    async def _safe_send(self, chat_id: int, text: str) -> None:
        """Send ``text`` after redacting secrets and truncating to Telegram limits."""
        cleaned = redact_secrets(text)
        if len(cleaned) > _MAX_REPLY_LENGTH:
            cleaned = cleaned[: _MAX_REPLY_LENGTH - 20] + "... (truncated)"
        try:
            await self._client.send_message(chat_id, cleaned)
        except Exception as exc:  # noqa: BLE001 - send errors must not crash the bot
            logger.warning(
                "telegram_send_failed",
                extra={
                    "chat_id": chat_id,
                    "error": redact_secrets(str(exc))[:200],
                },
            )


_DISPATCH: dict[str, Callable[[TelegramBot], Awaitable[str]]] = {
    "/start": TelegramBot._cmd_start,
    "/help": TelegramBot._cmd_help,
    "/status": TelegramBot._cmd_status,
    "/reconcile": TelegramBot._cmd_reconcile,
    "/opportunities": TelegramBot._cmd_opportunities,
    "/pause": TelegramBot._cmd_pause,
    "/resume": TelegramBot._cmd_resume,
}


async def run_telegram(services: AppServices) -> int:
    """Run the Telegram bot (blocking until cancelled)."""
    settings = services.settings.telegram
    if not settings.is_configured:
        logger.info("telegram_not_configured")
        return 1
    if not settings.has_any_operator:
        logger.info(
            "telegram_allow_list_empty",
            extra={
                "hint": "set CAT_TELEGRAM__ALLOWED_USER_IDS in .env",
            },
        )
        return 1

    client = TelegramClient(
        settings.bot_token.get_secret_value(),
        poll_timeout_seconds=settings.poll_timeout_seconds,
    )
    bot = TelegramBot(services, client)
    me = await client.get_me()
    username = me.get("username") if isinstance(me, dict) else None
    logger.info("telegram_bot_started", extra={"username": username})
    try:
        await poll_forever(client, bot.handle_update)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await client.close()
    return 0