"""Telegram bot: secondary control and monitoring interface.

The handlers are a thin layer over :class:`app.services.AppServices` — the
same services the CLI uses.  No trading logic lives here.

Security (fail-closed):

* only chats listed in ``CAT_TELEGRAM__ALLOWED_CHAT_IDS`` may command the bot;
* an empty allow-list disables all commands (a public bot must never accept
  strangers);
* the bot token is a secret and never appears in logs or replies.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from app.auto import AutoTrader
from app.config.logging_config import get_logger
from app.services import AppServices

from .client import TelegramClient, poll_forever

__all__ = ["TelegramBot", "run_telegram"]

logger = get_logger("telegram.bot")

COMMANDS = (
    "/status",
    "/balances",
    "/opportunities",
    "/triangle",
    "/transfer",
    "/trades",
    "/start_auto",
    "/stop_auto",
)

_HELP = (
    "crypto-arbitrage-bot commands:\n"
    "/status - bot status (mode, exchanges, risk, transfers)\n"
    "/balances - balances per venue\n"
    "/opportunities - scan triangles + transfer plans\n"
    "/triangle - execute the best triangle cycle\n"
    "/transfer [ASSET AMOUNT] - transfer plans; with args executes the best plan\n"
    "/trades - recent trades\n"
    "/start_auto - enable auto trading (runs in this process)\n"
    "/stop_auto - stop auto trading"
)


class TelegramBot:
    def __init__(self, services: AppServices, client: TelegramClient) -> None:
        self._services = services
        self._client = client
        self._auto_task: asyncio.Task | None = None
        self._auto_trader: AutoTrader | None = None

    # ---------------------------------------------------------------- dispatch
    async def handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message") or {}
        chat_id = message.get("chat", {}).get("id")
        text = str(message.get("text") or "").strip()
        if chat_id is None or not text:
            return
        if not self._authorized(chat_id):
            await self._client.send_message(
                chat_id, "unauthorized: this chat is not in CAT_TELEGRAM__ALLOWED_CHAT_IDS"
            )
            return
        command, _, rest = text.partition(" ")
        command = command.split("@")[0].lower()
        handler = {
            "/start": self._cmd_help,
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/balances": self._cmd_balances,
            "/opportunities": self._cmd_opportunities,
            "/triangle": self._cmd_triangle,
            "/transfer": self._cmd_transfer,
            "/trades": self._cmd_trades,
            "/start_auto": self._cmd_start_auto,
            "/stop_auto": self._cmd_stop_auto,
        }.get(command)
        if handler is None:
            await self._client.send_message(chat_id, f"unknown command\n\n{_HELP}")
            return
        try:
            reply = await handler(rest.strip())
        except Exception as exc:  # noqa: BLE001 - report, never crash the bot
            logger.error("telegram_command_failed", extra={"command": command, "error": str(exc)})
            reply = f"command failed: {type(exc).__name__}: {exc}"
        if reply:
            await self._client.send_message(chat_id, reply)

    def _authorized(self, chat_id: int) -> bool:
        allowed = self._services.settings.telegram.allowed_chat_ids
        return chat_id in allowed

    # ---------------------------------------------------------------- commands
    async def _cmd_help(self, rest: str) -> str:
        return _HELP

    async def _cmd_status(self, rest: str) -> str:
        status = await self._services.status()
        guard = status["guard"]
        lines = [
            f"mode: {status['mode']}",
            f"uptime: {status['uptime_seconds']}s",
            f"trading enabled: {guard['trading_enabled']}",
            "kill switch: "
            + (f"ENGAGED ({guard['halt_reason']})" if guard["halted"] == "true" else "released"),
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
        for transfer in status["transfers_open"]:
            lines.append(
                f"  {transfer['route']} {transfer['asset']} "
                f"{transfer['amount']}: {transfer['state']}"
            )
        return "\n".join(lines)

    async def _cmd_balances(self, rest: str) -> str:
        snapshots = await self._services.balances()
        if not snapshots:
            return "no balances available"
        lines = []
        for venue, snapshot in sorted(snapshots.items()):
            lines.append(f"{venue}:")
            shown = 0
            for balance in sorted(snapshot.balances, key=lambda b: b.asset):
                if balance.total <= 0:
                    continue
                lines.append(f"  {balance.asset} free {balance.free} used {balance.used}")
                shown += 1
                if shown >= 8:
                    lines.append("  …")
                    break
        return "\n".join(lines)

    async def _cmd_opportunities(self, rest: str) -> str:
        opportunities = await self._services.scan_triangles()
        plans = await self._services.plan_transfers()
        lines = [f"triangles: {len(opportunities)}"]
        for opportunity in opportunities[:5]:
            lines.append(f"  {opportunity.direction} net {opportunity.net_profit_bps} bps")
        lines.append(f"transfers: {len(plans)}")
        for plan in plans[:5]:
            lines.append(
                f"  {plan.source_exchange}->{plan.dest_exchange} {plan.asset} "
                f"{plan.amount} via {plan.network}: net {plan.net_profit_bps:.1f} bps"
            )
        return "\n".join(lines)

    async def _cmd_triangle(self, rest: str) -> str:
        opportunities = await self._services.scan_triangles()
        if not opportunities:
            return "no triangle opportunities above the configured minimum"
        best = opportunities[0]
        trade, assessment = await self._services.execute_triangle(best)
        if assessment is not None and not assessment.approved:
            reasons = "\n".join(f"  - {reason}" for reason in assessment.reasons)
            return f"REJECTED by risk validation:\n{reasons}"
        return (
            f"trade {trade.id}: {trade.status.value}\n"
            f"route: {trade.route}\n"
            f"in {trade.input_amount} -> out {trade.output_amount}\n"
            f"net {trade.net_profit} ({trade.net_profit_bps} bps)"
            + (f"\nnote: {trade.error}" if trade.error else "")
        )

    async def _cmd_transfer(self, rest: str) -> str:
        asset: str | None = None
        amount: Decimal | None = None
        parts = rest.split()
        if len(parts) >= 1:
            asset = parts[0]
        if len(parts) >= 2:
            try:
                amount = Decimal(parts[1])
            except ArithmeticError:
                return f"invalid amount: {parts[1]}"
        plans = await self._services.plan_transfers(asset=asset, amount=amount)
        if not plans:
            return "no transfer plans above the configured minimum"
        lines = []
        for plan in plans[:5]:
            lines.append(
                f"{plan.source_exchange}->{plan.dest_exchange} {plan.asset} {plan.amount} "
                f"via {plan.network}: net {plan.net_profit_bps:.1f} bps"
            )
        if not asset or not amount:
            lines.append("\nuse /transfer ASSET AMOUNT to execute the best plan")
            return "\n".join(lines)
        plan = plans[0]
        record = await self._services.start_transfer(plan)
        return (
            f"transfer {record.id} started: {record.state.value}\n"
            f"{plan.source_exchange}->{plan.dest_exchange} {plan.asset} {plan.amount} "
            f"via {plan.network}\n"
            "the lifecycle advances in this process and via /status"
        )

    async def _cmd_trades(self, rest: str) -> str:
        trades = await self._services.trades.list_recent(limit=10)
        if not trades:
            return "no trades yet"
        lines = []
        for trade in trades:
            lines.append(
                f"{trade.created_at:%m-%d %H:%M} {trade.strategy.value} "
                f"{trade.route} {trade.status.value} net {trade.net_profit}"
            )
        return "\n".join(lines)

    async def _cmd_start_auto(self, rest: str) -> str:
        if self._services.guard.is_halted:
            return (
                f"kill switch ENGAGED ({self._services.guard.halt_reason}); "
                "release it from the CLI before starting auto trading"
            )
        if self._auto_task is not None and not self._auto_task.done():
            return "auto trading is already running"
        self._services.guard.enable_trading(enabled=True)
        await self._services.set_auto_trading(True)
        self._auto_trader = AutoTrader(self._services)
        self._auto_task = asyncio.create_task(self._auto_trader.run_forever())
        return f"auto trading enabled (mode {self._services.settings.mode.value})"

    async def _cmd_stop_auto(self, rest: str) -> str:
        await self._services.set_auto_trading(False)
        if self._auto_trader is not None:
            self._auto_trader.stop()
        if self._auto_task is not None:
            self._auto_task.cancel()
            self._auto_task = None
        return "auto trading disabled"

    # ---------------------------------------------------------------- lifecycle
    async def background_maintenance(self) -> None:
        """Drive transfer workflows while the bot polls for commands."""
        while True:
            try:
                await self._services.tick_transfers()
            except Exception as exc:  # noqa: BLE001 - keep the maintenance alive
                logger.warning("telegram_tick_failed", extra={"error": str(exc)[:200]})
            await asyncio.sleep(self._services.settings.transfer.poll_interval_seconds)


async def run_telegram(services: AppServices) -> int:
    """Run the Telegram bot (blocking until cancelled)."""
    settings = services.settings.telegram
    if not settings.is_configured:
        print(
            "Telegram is not configured: set CAT_TELEGRAM__BOT_TOKEN "
            "(and CAT_TELEGRAM__ALLOWED_CHAT_IDS) in .env"
        )
        return 1
    if not settings.allowed_chat_ids:
        print(
            "Telegram allow-list is empty: no chat could command the bot. "
            "Set CAT_TELEGRAM__ALLOWED_CHAT_IDS (comma-separated) in .env"
        )
        return 1

    client = TelegramClient(
        settings.bot_token.get_secret_value(),
        poll_timeout_seconds=settings.poll_timeout_seconds,
    )
    bot = TelegramBot(services, client)
    me = await client.get_me()
    logger.info("telegram_bot_started", extra={"username": me.get("username")})
    print(f"Telegram bot @{me.get('username')} polling (Ctrl+C to stop)")

    maintenance = asyncio.create_task(bot.background_maintenance())
    try:
        await poll_forever(client, bot.handle_update)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        maintenance.cancel()
        if bot._auto_task is not None:
            bot._auto_task.cancel()
        await client.close()
    return 0
