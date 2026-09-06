"""Telegram operator interface: read-only / safe-control only.

This module is a thin layer over :class:`app.services.AppServices` — the same
services the CLI uses.  No trading logic lives here.

Scope (deliberately restricted)
-------------------------------

The Telegram bot is a **secondary control interface**.  Every command it
exposes is either read-only, a reconciliation query, a safe state change
(persistent kill switch + auto-trading flag), or — for ``/start_trading`` /
``/stop_trading`` — a wrapper around the existing
:class:`app.auto_controller.AutoTradingController` which is the SINGLE owner
of the background strategy loop.  It must NOT be able to:

* place exchange orders (no ``/triangle``, no ``/transfer`` execution, no
  ``/buy`` / ``/sell`` commands);
* initiate withdrawals;
* expose configuration, credentials, exchange API keys, the database URL or
  raw exception tracebacks to the operator.

The kill-switch + auto-trading flag toggles (commands ``/pause`` / ``/resume``)
go through the existing :class:`app.execution.guard.ExecutionGuard`, which
itself is persisted in :class:`app.storage.repositories.BotStateRepository`
and survives restart.  They do NOT bypass the LIVE confirmation, mode policy
or risk engine — ``/resume`` only restores a flag that the rest of the bot
already enforces.

The ``/start_trading`` command:

* is rejected with an explicit error when `` ``mode is not DEMO;
* goes through the existing :class:`AutoTradingController` which drives the
  existing :class:`app.auto.AutoTrader`;
* the :class:`AutoTrader` calls :meth:`AppServices.execute_triangle` for
  every cycle, which is the ONLY path that may eventually reach the order
  gate, the risk engine, the execution guard and the exchange adapter;
* never calls :meth:`create_order` or :meth:`withdraw` directly from
  Telegram.

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
from .i18n import (
    LANGUAGE_PICKER_KEYBOARD,
    LANGUAGE_PICKER_PROMPT,
    STRATEGY_PICKER_KEYBOARD,
    STRATEGY_PICKER_PROMPT,
    SUPPORTED_LANGUAGES,
    SUPPORTED_STRATEGIES,
    is_supported_strategy,
    lang_storage_key,
    strategy_storage_key,
    t,
)

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
    "/start_trading",
    "/stop_trading",
    "/language",
    "/strategy",
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
    "/start_trading - start the DEMO auto-trading loop (DEMO only)\n"
    "/stop_trading  - stop the auto-trading loop (idempotent)\n"
    "/language     - choose language (English / Русский)\n"
    "/strategy     - choose strategy (Triangle / Transfer)\n"
    "/ai           - AI Advisor (read-only: /ai status/report/recommendations/memory/balance)\n"
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
        # AI Advisor adapter — lazily built so that telegram startup never blocks
        # on agent wiring and never pulls LLM credentials at import time.
        self._agent_adapter: Any | None = None

    # ---------------------------------------------------------------- language
    async def _get_lang(self, user_id: int | None) -> str | None:
        if user_id is None:
            return None
        try:
            val = await self._services.bot_state.get(lang_storage_key(user_id))
        except Exception:
            return None
        if isinstance(val, str) and val in SUPPORTED_LANGUAGES:
            return val
        return None

    async def _set_lang(self, user_id: int, lang: str) -> None:
        if lang not in SUPPORTED_LANGUAGES:
            return
        try:
            await self._services.bot_state.set(lang_storage_key(user_id), lang)
        except Exception as exc:  # noqa: BLE001 - storage failure must not crash bot
            logger.warning("telegram_set_lang_failed", extra={"error": str(exc)[:200]})

    async def _send_picker(self, chat_id: int) -> None:
        try:
            await self._safe_send(chat_id, LANGUAGE_PICKER_PROMPT, reply_markup=LANGUAGE_PICKER_KEYBOARD)
        except Exception:
            pass

    async def _get_strategy(self) -> str | None:
        try:
            val = await self._services.bot_state.get(strategy_storage_key())
        except Exception:
            return None
        if isinstance(val, str) and val in SUPPORTED_STRATEGIES:
            return val
        return None

    async def _set_strategy(self, strategy: str) -> None:
        if strategy not in SUPPORTED_STRATEGIES:
            return
        try:
            await self._services.bot_state.set(strategy_storage_key(), strategy)
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram_set_strategy_failed", extra={"error": str(exc)[:200]})

    async def _send_strategy_picker(self, chat_id: int, lang: str) -> None:
        try:
            await self._safe_send(chat_id, t("strategy_picker_prompt", lang), reply_markup=STRATEGY_PICKER_KEYBOARD)
        except Exception:
            pass

    async def _handle_callback(self, callback: dict[str, Any]) -> None:
        cb_id = callback.get("id")
        data = str(callback.get("data") or "")
        user = callback.get("from") or {}
        user_id = user.get("id")
        message = callback.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        message_id = message.get("message_id")
        # Language selection
        if data.startswith("lang:"):
            lang = data.split(":", 1)[1].strip().lower()
            if lang not in SUPPORTED_LANGUAGES:
                if cb_id:
                    try:
                        await self._client.answer_callback_query(str(cb_id), text="Invalid language")
                    except Exception:
                        pass
                return
            # Persist — language selection never grants trading access
            if isinstance(user_id, int):
                await self._set_lang(user_id, lang)
            # Answer callback in newly selected language
            if cb_id:
                try:
                    await self._client.answer_callback_query(str(cb_id), text=t("callback_language_changed", lang))
                except Exception:
                    pass
            if chat_id is None:
                return
            # Confirm change in newly selected language + show help
            try:
                confirmation = t("language_selected", lang)
                help_text = t("help_text", lang)
                full = f"{confirmation}\n\n{help_text}"
                await self._safe_send(int(chat_id), full)
            except Exception:
                pass
            # Edit original picker message to remove inline keyboard (best-effort)
            if chat_id is not None and message_id is not None:
                try:
                    await self._client.edit_message_text(
                        int(chat_id), int(message_id), text=t("language_picker_chosen", lang)
                    )
                except Exception:
                    pass
            return
        # Strategy selection
        if data.startswith("strategy:"):
            # Strategy changes require authorization — fail-closed if not authorized
            # Extract chat_id for auth check (use user_id and chat_id)
            chat_id_val = chat.get("id")
            # Need to check authorization: only allowed users may change strategy
            # We don't have user_id/chat_id in this scope for auth? Use the ones from callback
            # For safety, check if user_id is authorized; if not, deny
            if not self._authorized(user_id, chat_id_val):
                if cb_id:
                    try:
                        # Use lang for denied message if available, else en
                        lang_for_denied = await self._get_lang(user_id)
                        eff = lang_for_denied if lang_for_denied in SUPPORTED_LANGUAGES else "en"
                        await self._client.answer_callback_query(str(cb_id), text=t("unauthorized", eff))
                    except Exception:
                        pass
                return
            strategy = data.split(":", 1)[1].strip().lower()
            if strategy not in SUPPORTED_STRATEGIES:
                if cb_id:
                    try:
                        await self._client.answer_callback_query(str(cb_id), text="Invalid strategy")
                    except Exception:
                        pass
                return
            await self._set_strategy(strategy)
            # Need lang for response
            lang = await self._get_lang(user_id)
            eff_lang = lang if lang in SUPPORTED_LANGUAGES else "en"
            if cb_id:
                try:
                    await self._client.answer_callback_query(str(cb_id), text=t("callback_strategy_changed", eff_lang))
                except Exception:
                    pass
            if chat_id is None:
                return
            try:
                # Confirmation in selected language
                key = f"strategy_selected_{strategy}"
                confirmation = t(key, eff_lang)
                # Also show current strategy and help
                help_text = t("help_text", eff_lang)
                full = f"{confirmation}\n\n{help_text}"
                await self._safe_send(int(chat_id), full)
            except Exception:
                pass
            if chat_id is not None and message_id is not None:
                try:
                    key_chosen = f"strategy_picker_chosen_{strategy}"
                    await self._client.edit_message_text(
                        int(chat_id), int(message_id), text=t(key_chosen, eff_lang)
                    )
                except Exception:
                    pass
            return
        # Unknown callback data
        if cb_id:
            try:
                await self._client.answer_callback_query(str(cb_id))
            except Exception:
                pass
        return

    # ---------------------------------------------------------------- dispatch
    async def handle_update(self, update: dict[str, Any]) -> None:
        # Callback query path — language selection
        callback = update.get("callback_query")
        if isinstance(callback, dict):
            try:
                await self._handle_callback(callback)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "telegram_callback_failed",
                    extra={"error": redact_secrets(str(exc))[:300]},
                )
            return

        message = update.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        user = message.get("from") or {}
        user_id = user.get("id")
        text = str(message.get("text") or "").strip()
        if chat_id is None or not text:
            return

        # Parse command
        command, _, _rest = text.partition(" ")
        command = command.split("@")[0].lower()

        # Language-sensitive commands: /start and /language
        # They may show the picker before authorization (spec allowance),
        # but must not bypass auth for protected functionality.
        if command == "/start":
            lang = await self._get_lang(user_id)
            if lang is None:
                await self._send_picker(int(chat_id))
                return
            # Language exists — now check authorization for welcome/help
            if not self._authorized(user_id, chat_id):
                await self._safe_send(int(chat_id), t("unauthorized", lang))
                return
            await self._safe_send(int(chat_id), t("help_text", lang))
            return

        if command == "/language":
            # /language always shows picker (even if language already set)
            # No auth gate for the picker itself (may be shown before auth),
            # but actual language change is persisted per user_id and does not
            # grant any trading privileges.
            await self._send_picker(int(chat_id))
            return

        if command == "/strategy":
            # Strategy selection is for authorized operators only (it influences trading)
            lang_for_denied2 = await self._get_lang(user_id)
            eff2 = lang_for_denied2 if lang_for_denied2 in SUPPORTED_LANGUAGES else "en"
            if not self._authorized(user_id, chat_id):
                await self._safe_send(int(chat_id), t("unauthorized", eff2))
                return
            lang2 = await self._get_lang(user_id)
            if lang2 is None:
                await self._send_picker(int(chat_id))
                return
            await self._send_strategy_picker(int(chat_id), lang2)
            return

        # For all other commands: authorization first (fail-closed)
        # Use stored language for denied message localization if available.
        lang_for_denied = await self._get_lang(user_id)
        effective_denied_lang = lang_for_denied if lang_for_denied in SUPPORTED_LANGUAGES else "en"
        if not self._authorized(user_id, chat_id):
            # Never reveal whether the chat or the user id was the problem.
            await self._safe_send(int(chat_id), t("unauthorized", effective_denied_lang))
            return

        # Authorized — ensure language is selected, otherwise show picker
        lang = await self._get_lang(user_id)
        if lang is None:
            await self._send_picker(int(chat_id))
            return

        # ------------------------------------------------------------------
        # AI Advisor — thin, read-only layer (no trading, no mutation)
        # Architecture: Telegram -> AgentTelegramAdapter -> AgentCore ->
        # read-only context / memory / knowledge / analysis
        # ------------------------------------------------------------------
        if command == "/ai":
            # Authorized and language-checked above; delegate full text so that
            # "/ai status", "/ai report", etc. are handled by the adapter.
            try:
                adapter = await self._get_agent_adapter()
                if adapter is None:
                    # Advisor not wired — graceful, localized empty-state handling
                    try:
                        from app.agent.telegram import _ai_t  # type: ignore[import-not-found]

                        reply_ai = _ai_t("ai_not_configured", lang)
                    except Exception:
                        reply_ai = t("internal_error", lang)
                    await self._safe_send(int(chat_id), reply_ai)
                    return
                reply_ai = await adapter.dispatch(text, lang=lang, approver=str(user_id) if user_id is not None else None)
                # Bounded response length is enforced by _safe_send (3500 chars)
                await self._safe_send(int(chat_id), reply_ai)
            except Exception as exc:  # noqa: BLE001 - adapter failure must never crash telegram
                logger.error(
                    "telegram_ai_failed",
                    extra={
                        "command": text[:80],
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "error": redact_secrets(str(exc))[:300],
                    },
                )
                await self._safe_send(int(chat_id), t("internal_error", lang))
            return

        # DEMO strategy gate for /start_trading
        if command == "/start_trading":
            from app.models.enums import TradingMode

            if self._services.settings.mode is TradingMode.DEMO:
                strat = await self._get_strategy()
                if strat is None:
                    # For backwards compat, default to triangle if not chosen.
                    # The operator can explicitly choose via /strategy.
                    await self._set_strategy("triangle")
                    # Also inform but not block start
                    # Continue to handler

        handler = self._dispatch(command)
        if handler is None:
            help_text = t("help_text", lang)
            reply = t("unknown_command", lang, help_text=help_text)
            await self._safe_send(int(chat_id), reply)
            return
        try:
            reply = await handler(lang)
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
            reply = t("internal_error", lang)
        await self._safe_send(int(chat_id), reply or "")

    def _dispatch(self, command: str) -> Callable[[str], Awaitable[str]] | None:
        method = _DISPATCH.get(command)
        if method is None:
            return None
        # Bind the method to this bot so the dispatch can call it as ``handler(lang)``.
        bound = method.__get__(self, type(self))
        return bound  # type: ignore[return-value]

    def _authorized(self, user_id: int | None, chat_id: int | None) -> bool:
        """Authorisation is by Telegram user ID alone.

        Per the operator-interface spec, only configured user IDs may command
        the bot.  The legacy ``CAT_TELEGRAM__ALLOWED_CHAT_IDS`` list is no
        longer consulted; operators must configure ``CAT_TELEGRAM__ALLOWED_USER_IDS``.
        """
        cfg = self._services.settings.telegram
        return user_id is not None and user_id in cfg.allowed_user_ids

    async def _get_agent_adapter(self) -> Any | None:
        """Lazy, fail-closed construction of the AI advisor adapter.

        No network I/O is performed here — only in-memory wiring. The
        provider itself is only contacted when the adapter dispatches an
        ``/ai`` request to ``AgentCore``.
        """
        if self._agent_adapter is not None:
            return self._agent_adapter
        try:
            from app.agent import build_agent
            from app.agent.telegram import AgentTelegramAdapter

            # ``build_agent`` is the single composition root for the advisor;
            # it shares ``self._services.db`` and wires read-only tools.
            core, _kb, _exp, _les, _rec_svc, tools = build_agent(self._services)
            # Approval service is human-gated; expose it to the adapter for
            # ``/ai approve`` / ``/ai reject`` (authorized users only).
            approval = getattr(self._services, "agent_approval_service", None)
            self._agent_adapter = AgentTelegramAdapter(core, tools, approval_service=approval)
            return self._agent_adapter
        except Exception as exc:  # noqa: BLE001 - telegram must never crash on advisor build
            logger.warning(
                "telegram_agent_adapter_build_failed",
                extra={"error": redact_secrets(str(exc))[:200]},
            )
            return None

    # ---------------------------------------------------------------- commands
    async def _cmd_start(self, lang: str) -> str:
        return t("help_text", lang)

    async def _cmd_help(self, lang: str) -> str:
        return t("help_text", lang)

    async def _cmd_status(self, lang: str) -> str:
        status = await self._services.status()
        guard = status["guard"]
        flag_key = "common_on" if status["auto_trading"] else "common_off"
        flag_local = t(flag_key, lang)
        loop_key = "common_running" if status.get("auto_loop_running") else "common_stopped"
        loop_local = t(loop_key, lang)
        # Kill switch line
        if guard["halted"] == "true":
            kill_line = t("status_kill_switch_engaged", lang, reason=guard["halt_reason"])
        else:
            kill_line = t("status_kill_switch_released", lang)
        strat = status.get("active_strategy", "not_set")
        # Translate strategy name if possible
        strat_key = f"strategy_{strat}" if strat in ("triangle", "transfer") else "common_off"
        # Use t for strategy name, fallback to raw
        strat_display = t(strat_key, lang) if strat in ("triangle", "transfer") else str(strat)
        # If translation returned the key itself (missing), use raw
        if strat_display == strat_key:
            strat_display = str(strat)
        lines: list[str] = [
            t("status_mode", lang, mode=status["mode"]),
            t("status_uptime", lang, sec=status["uptime_seconds"]),
            t("status_trading_enabled", lang, val=guard["trading_enabled"]),
            kill_line,
            t("status_auto_trading_flag", lang, flag=flag_local),
            t("status_auto_loop", lang, state=loop_local),
            t("status_strategy", lang, strategy=strat_display),
            t("status_exchanges_header", lang),
        ]
        for venue, info in status["exchanges"].items():
            lines.append(
                t("status_exchange_line", lang, venue=venue, status=info["status"], credentials=info["credentials"])
            )
        md = status["market_data"]
        lines.append(
            t(
                "status_market_data",
                lang,
                order_books=md["order_books"],
                tickers=md["tickers"],
                exchanges=md["exchanges"],
            )
        )
        risk = status["risk"]
        lines.append(t("status_daily_pnl", lang, pnl=risk["daily_pnl"]))
        lines.append(t("status_open_transfers_count", lang, count=risk["open_transfers"]))
        if status["transfers_open"]:
            lines.append(t("status_open_transfers_header", lang))
            for transfer in status["transfers_open"]:
                lines.append(
                    t(
                        "status_open_transfer_line",
                        lang,
                        route=transfer["route"],
                        asset=transfer["asset"],
                        amount=transfer["amount"],
                        state=transfer["state"],
                    )
                )
        if status["recent_trades"]:
            lines.append(t("status_recent_trades_header", lang))
            for trade in status["recent_trades"]:
                lines.append(
                    t(
                        "status_recent_trade_line",
                        lang,
                        strategy=trade["strategy"],
                        route=trade["route"],
                        status=trade["status"],
                        net_profit=trade["net_profit"],
                    )
                )
        return "\n".join(lines)

    async def _cmd_reconcile(self, lang: str) -> str:
        from app.models.enums import TransferState

        records = await self._services.transfers.list_by_state(
            TransferState.MANUAL_REVIEW.value
        )
        if not records:
            return t("reconcile_empty", lang)
        lines = [t("reconcile_header", lang, count=len(records))]
        for record in records[:_RECONCILE_LIMIT]:
            lines.append("")
            lines.append(t("reconcile_id", lang, id=record.id))
            lines.append(t("reconcile_state", lang, state=record.state.value))
            lines.append(
                t(
                    "reconcile_route",
                    lang,
                    route=f"{record.source_exchange} -> {record.dest_exchange}",
                )
            )
            lines.append(
                t("reconcile_asset", lang, asset=record.asset, amount=record.amount)
            )
            if record.buy_filled_amount and record.buy_filled_amount > 0:
                lines.append(
                    t(
                        "reconcile_held",
                        lang,
                        amount=record.buy_filled_amount,
                        asset=record.asset,
                        exchange=record.source_exchange,
                    )
                )
            withdrawal_bits = []
            if record.withdrawal_id:
                withdrawal_bits.append(f"id={record.withdrawal_id}")
            if record.withdrawal_txid:
                withdrawal_bits.append(f"txid={record.withdrawal_txid}")
            if record.withdrawal_amount and record.withdrawal_amount > 0:
                details = ", ".join(withdrawal_bits)
                if details:
                    lines.append(
                        t(
                            "reconcile_withdrawal",
                            lang,
                            amount=record.withdrawal_amount,
                            asset=record.asset,
                            exchange=record.source_exchange,
                            details=details,
                        )
                    )
                else:
                    lines.append(
                        t(
                            "reconcile_withdrawal_no_details",
                            lang,
                            amount=record.withdrawal_amount,
                            asset=record.asset,
                            exchange=record.source_exchange,
                        )
                    )
            lines.append(t("reconcile_created", lang, ts=record.created_at.isoformat()))
            lines.append(t("reconcile_updated", lang, ts=record.updated_at.isoformat()))
            if record.error:
                lines.append(t("reconcile_reason", lang, reason=record.error))
        if len(records) > _RECONCILE_LIMIT:
            lines.append("")
            lines.append(
                t("reconcile_more", lang, remaining=len(records) - _RECONCILE_LIMIT)
            )
        return "\n".join(lines)

    async def _cmd_opportunities(self, lang: str) -> str:
        opportunities = await self._services.scan_triangles()
        plans = await self._services.plan_transfers()
        lines = [
            t("opportunities_triangle", lang, count=len(opportunities)),
        ]
        if opportunities:
            for opp in opportunities[:_OPPORTUNITY_LIMIT]:
                lines.append(
                    t(
                        "opportunities_triangle_line",
                        lang,
                        direction=opp.direction,
                        net_profit_bps=opp.net_profit_bps,
                        size_notional_quote=opp.size_notional_quote,
                    )
                )
        else:
            lines.append(t("opportunities_triangle_none", lang))
        lines.append(t("opportunities_transfer", lang, count=len(plans)))
        if plans:
            for plan in plans[:_OPPORTUNITY_LIMIT]:
                lines.append(
                    t(
                        "opportunities_transfer_line",
                        lang,
                        source=plan.source_exchange,
                        dest=plan.dest_exchange,
                        asset=plan.asset,
                        amount=plan.amount,
                        network=plan.network,
                        net_profit_bps=plan.net_profit_bps,
                    )
                )
        else:
            lines.append(t("opportunities_transfer_none", lang))
        lines.append("")
        lines.append(t("opportunities_view_only", lang))
        return "\n".join(lines)

    async def _cmd_pause(self, lang: str) -> str:
        """Engage the persistent kill switch (safe)."""
        if self._services.guard.is_halted:
            return t("pause_already_engaged", lang, reason=self._services.guard.halt_reason)
        await self._services.engage_kill_switch("telegram /pause")
        # engage_kill_switch already disables the auto flag; tell the
        # controller to stop the in-process loop right now so the kill switch
        # is observed immediately (the loop may be parked in a wait between
        # cycles — we want it gone NOW, not on the next interval tick).
        controller = self._services.auto_controller
        if controller is not None:
            await controller.stop()
        return t("pause_engaged", lang)

    async def _cmd_resume(self, lang: str) -> str:
        """Release the persistent kill switch (safe)."""
        if not self._services.guard.is_halted:
            return t("resume_already_released", lang)
        # The guard is the single source of truth for "may we trade?".  This
        # command restores it but does NOT enable auto trading — operators
        # must do that explicitly through the CLI.
        await self._services.release_kill_switch()
        return t("resume_released", lang)

    async def _cmd_start_trading(self, lang: str) -> str:
        """Start the DEMO auto-trading loop via the existing controller.

        Hard safety gates (in this order):
          1. Mode must be DEMO.  PAPER and LIVE are refused with an explicit
             error — Telegram must NEVER auto-trade in LIVE.
          2. The :class:`AutoTradingController` itself refuses if the kill
             switch is engaged, if the loop is already running, etc.

        The controller drives the existing :class:`AutoTrader`, which calls
        :meth:`AppServices.execute_triangle` for each opportunity.  That path
        is the only one that ever reaches the order gate, the risk engine,
        the execution guard and (in DEMO) the exchange adapter.  Telegram
        never calls :meth:`create_order` or :meth:`withdraw` directly.
        """
        from app.models.enums import TradingMode

        mode = self._services.settings.mode
        if mode is not TradingMode.DEMO:
            return t("start_trading_refused_mode", lang, mode=mode.value)
        controller = self._services.auto_controller
        if controller is None:
            return t("start_trading_no_controller", lang)
        started, message = await controller.start()
        if started:
            return t("start_trading_started", lang, msg=message)
        return t("start_trading_not_started", lang, msg=message)

    async def _cmd_stop_trading(self, lang: str) -> str:
        """Stop the auto-trading loop (idempotent, does NOT touch the kill switch).

        ``/stop_trading`` is the symmetric counterpart of ``/start_trading``.
        It does NOT engage the kill switch — the existing architecture keeps
        ``/pause`` as the kill-switch command.  The controller sets the
        persisted auto flag to ``False`` and asks the in-process task to exit
        cleanly.
        """
        controller = self._services.auto_controller
        if controller is None:
            return t("stop_trading_no_controller", lang)
        stopped, message = await controller.stop()
        if stopped:
            return t("stop_trading_stopped", lang, msg=message)
        return t("stop_trading_already_stopped", lang, msg=message)

    async def _cmd_language(self, lang: str) -> str:
        # This handler is not used directly — /language is intercepted in
        # handle_update to always show the picker.  Kept for dispatch completeness.
        return t("language_picker_prompt", lang)

    async def _cmd_strategy(self, lang: str) -> str:
        # This handler is not used directly — /strategy is intercepted in
        # handle_update to always show the picker.  Kept for dispatch completeness.
        return t("strategy_picker_prompt", lang)

    # ---------------------------------------------------------------- send
    async def _safe_send(
        self, chat_id: int, text: str, reply_markup: dict[str, Any] | None = None
    ) -> None:
        """Send ``text`` after redacting secrets and truncating to Telegram limits."""
        cleaned = redact_secrets(text)
        if len(cleaned) > _MAX_REPLY_LENGTH:
            cleaned = cleaned[: _MAX_REPLY_LENGTH - 20] + "... (truncated)"
        try:
            if reply_markup is not None:
                try:
                    await self._client.send_message(chat_id, cleaned, reply_markup=reply_markup)  # type: ignore[call-arg]
                except TypeError:
                    # Compatibility with FakeClient in tests that does not accept reply_markup
                    await self._client.send_message(chat_id, cleaned)  # type: ignore[call-arg]
            else:
                await self._client.send_message(chat_id, cleaned)  # type: ignore[call-arg]
        except Exception as exc:  # noqa: BLE001 - send errors must not crash the bot
            logger.warning(
                "telegram_send_failed",
                extra={
                    "chat_id": chat_id,
                    "error": redact_secrets(str(exc))[:200],
                },
            )


_DISPATCH: dict[str, Callable[[TelegramBot, str], Awaitable[str]]] = {
    "/start": TelegramBot._cmd_start,
    "/help": TelegramBot._cmd_help,
    "/status": TelegramBot._cmd_status,
    "/reconcile": TelegramBot._cmd_reconcile,
    "/opportunities": TelegramBot._cmd_opportunities,
    "/pause": TelegramBot._cmd_pause,
    "/resume": TelegramBot._cmd_resume,
    "/start_trading": TelegramBot._cmd_start_trading,
    "/stop_trading": TelegramBot._cmd_stop_trading,
    "/language": TelegramBot._cmd_language,
    "/strategy": TelegramBot._cmd_strategy,
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
