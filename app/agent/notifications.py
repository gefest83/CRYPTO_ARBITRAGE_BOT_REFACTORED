"""Phase 6 — event-driven AI Agent Telegram notifications.

Architecture (reuse, no rebuild):

* Event sources are the EXISTING application event records: the journal
  tables (``trades`` / ``transfers``), the append-only ``audit_log`` and the
  live status snapshot (exchange health, market-data freshness). There is no
  new polling service and no hook inside trading/execution/risk/recovery
  code — :meth:`NotificationService.scan` tails those records with
  persistent cursors, and :meth:`NotificationService.notify` sends a single
  event. Operators (or tests) invoke ``scan()``; nothing runs on its own.
* Delivery is asynchronous and non-blocking: every failure path is caught
  and logged, so a Telegram/API outage can never stop or delay trading,
  execution, recovery, risk or market-data services.
* Structured event data is the source of truth. The LLM may format/explain
  validated numbers (:meth:`NotificationService.explain_summary`) but must
  never invent financial values or event details — calculations stay
  deterministic application code.
* Messages are informational only: this module holds no reference to order
  placement, withdrawals, configuration or risk mutation paths.

Safety properties (each covered by focused tests):

* enable/disable master switch + per-event allow-list filtering;
* persistent deduplication (``bot_state``) + in-memory debounce/rate-limit
  for noisy events (market-data, exchange status);
* secret redaction + 3000-char bound on every outbound text;
* recipients limited to configured operator user IDs with per-user language.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import UTC, datetime, time as dtime
from decimal import Decimal
from typing import Any

from app.config.logging_config import get_logger
from app.models.base import DomainModel, utc_now
from pydantic import Field

__all__ = [
    "EVENT_TYPES",
    "NOISY_DEBOUNCE_SECONDS",
    "NotificationEvent",
    "NotificationService",
    "render_notification",
]

logger = get_logger("agent.notifications")

#: All supported event types (also the settings allow-list vocabulary).
EVENT_TYPES: tuple[str, ...] = (
    "trade_completed",
    "trade_failed",
    "order_rejected",
    "execution_degraded",
    "exchange_offline",
    "market_data_stale",
    "recovery_required",
    "risk_event",
    "unusual_slippage",
    "new_recommendation",
    "system_error",
    "daily_summary",
)

#: Per-type debounce windows (seconds). Noisy level-triggered events are
#: held back much longer than one-shot edge events.
NOISY_DEBOUNCE_SECONDS: dict[str, float] = {
    "market_data_stale": 900.0,
    "exchange_offline": 600.0,
    "execution_degraded": 1800.0,
    "system_error": 300.0,
}
_DEFAULT_DEBOUNCE_SECONDS = 60.0

#: bot_state key prefixes (persistent cursors + dedup + daily marker).
_SENT_PREFIX = "agent_notif_sent:"
_PENDING_KEY = "agent_notif_pending"
_CURSOR_TRADES = "agent_notif_cursor_trades"
_CURSOR_AUDIT = "agent_notif_cursor_audit"
_CURSOR_TRANSFERS = "agent_notif_cursor_transfers"
_CURSOR_RECS = "agent_notif_cursor_recs"
_DAILY_PREFIX = "agent_notif_daily:"

_TERMINAL_TRADE_STATUSES: frozenset[str] = frozenset({"completed", "failed", "manual_review"})

# App-audit actions mapped to notification events (tail of the existing log).
_RISK_AUDIT_ACTIONS: frozenset[str] = frozenset({"KILL_SWITCH_ENGAGED", "KILL_SWITCH_RELEASED"})
_FAILED_SUFFIX = "_FAILED"


class NotificationEvent(DomainModel):
    """One structured notification (numbers pre-computed, JSON-safe strings).

    ``dedup_key`` is stable per real-world occurrence (``"<type>:<id>"`` for
    edge events, ``"<type>:<subject>:<UTC-date>"`` for level events) so
    retries/restarts never resend it. ``data`` carries display fields only;
    ``provenance`` carries trade/order/recovery/recommendation IDs.
    """

    event_type: str
    dedup_key: str
    data: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    created_at: datetime = Field(default_factory=utc_now)


# ------------------------------------------------------------------ rendering (en/ru)


def _templates() -> dict[str, dict[str, str]]:
    return {
        "en": {
            "trade_completed": "✅ Trade completed: {strategy} {route} on {exchange} — net {net} ({bps} bps), fees {fees} [trade:{trade_id}]",
            "trade_failed": "❌ Trade failed: {strategy} {route} on {exchange} — {error} [trade:{trade_id}]",
            "order_rejected": "⚠️ Order rejected in trade {trade_id}: {symbol} {side} {status} [trade:{trade_id}]",
            "execution_degraded": "📉 Execution degraded: {failed}/{total} recent trades failed or need review (n={total})",
            "exchange_offline": "🔌 Exchange {exchange} is {status}",
            "market_data_stale": "💤 Market data stale: {stale}/{total} books missing or stale",
            "recovery_required": "🛟 Recovery required: {kind} {ref} is in MANUAL_REVIEW — {error}",
            "risk_event": "🛑 Risk event: {action} — {message}",
            "unusual_slippage": "⚡ Unusual slippage: trade {trade_id} slippage {slippage} bps (limit {limit} bps)",
            "new_recommendation": "💡 New recommendation: {parameter} {old} → {proposed} — {reason} (conf {confidence}) [rec:{rec_id}]",
            "system_error": "🔥 System error: {action} — {message}",
            "daily_summary": (
                "📊 Daily summary ({date}): trades n={n} (completed {completed}, failed {failed}), "
                "PnL {pnl}, avg {avg_bps} bps, win {win_rate}% | "
                "experiences {experiences}, lessons {lessons}, open transfers {open_transfers}"
            ),
        },
        "ru": {
            "trade_completed": "✅ Сделка завершена: {strategy} {route} на {exchange} — итог {net} ({bps} бп), комиссии {fees} [сделка:{trade_id}]",
            "trade_failed": "❌ Сделка не удалась: {strategy} {route} на {exchange} — {error} [сделка:{trade_id}]",
            "order_rejected": "⚠️ Ордер отклонён в сделке {trade_id}: {symbol} {side} {status} [сделка:{trade_id}]",
            "execution_degraded": "📉 Исполнение деградировало: {failed}/{total} недавних сделок неуспешны или на проверке (n={total})",
            "exchange_offline": "🔌 Биржа {exchange} — {status}",
            "market_data_stale": "💤 Маркет-дата устарела: {stale}/{total} стаканов нет или устарели",
            "recovery_required": "🛟 Требуется восстановление: {kind} {ref} в MANUAL_REVIEW — {error}",
            "risk_event": "🛑 Риск-событие: {action} — {message}",
            "unusual_slippage": "⚡ Аномальное проскальзывание: сделка {trade_id}, {slippage} бп (лимит {limit} бп)",
            "new_recommendation": "💡 Новая рекомендация: {parameter} {old} → {proposed} — {reason} (увер {confidence}) [рек:{rec_id}]",
            "system_error": "🔥 Системная ошибка: {action} — {message}",
            "daily_summary": (
                "📊 Дневная сводка ({date}): сделок n={n} (успешно {completed}, неуспешно {failed}), "
                "PnL {pnl}, сред {avg_bps} бп, винрейт {win_rate}% | "
                "опыты {experiences}, уроки {lessons}, открытые трансферы {open_transfers}"
            ),
        },
    }


def render_notification(event: NotificationEvent, lang: str | None) -> str:
    """Render an event to localized text from structured data only."""
    effective = lang if lang in ("en", "ru") else "en"
    template = _templates()[effective].get(event.event_type, _templates()["en"].get(event.event_type, "{event_type}"))
    data = dict(event.data)
    data.setdefault("event_type", event.event_type)
    try:
        text = template.format(**{k: str(v)[:160] for k, v in data.items()})
    except Exception:
        text = f"{event.event_type}: " + ", ".join(f"{k}={v}"[:80] for k, v in list(data.items())[:6])
    from app.agent.telegram import _finalize_telegram

    return _finalize_telegram(text)


# ------------------------------------------------------------------ service


class NotificationService:
    """Event-driven operator notifications over the existing Telegram path.

    Bound to :class:`AppServices` but read-only towards it: journal/audit
    reads, status snapshots and ``bot_state`` cursors only. Sending goes
    through an injected client (tests) or a lazily built
    :class:`TelegramClient` (production, fail-closed when unconfigured).
    """

    def __init__(self, services: Any, *, client: Any | None = None) -> None:  # type: ignore[no-untyped-def]
        self._services = services
        self._client = client
        self._debounced: dict[str, float] = {}
        self._sent_times: deque[float] = deque()

    # ------------------------------------------------------------ single event

    def emit(self, event_type: str, data: dict[str, Any] | None = None,  # type: ignore[no-untyped-def]
             provenance: dict[str, Any] | None = None) -> Any:
        """Fire-and-forget wrapper around :meth:`notify` (never raises)."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        return loop.create_task(self.notify(event_type, data, provenance))

    async def notify(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        *,
        dedup_suffix: str | None = None,
        daily_key: str | None = None,
        extra_text: str | None = None,
        debounce_group: str | None = None,
    ) -> dict[str, Any]:
        """Send one event (filter → dedup → debounce → render → send).

        ``dedup_suffix`` scopes level events per subject+time bucket;
        ``daily_key`` marks the UTC-day idempotency key for the daily summary;
        ``debounce_group`` (default: the dedup key) groups noisy repeats;
        ``extra_text`` appends an (already validated) AI note. Never raises:
        every failure is caught, logged and reported in the returned dict
        (``{"sent": n, "skipped": reason}``).
        """
        event_type = str(event_type or "").strip().lower()
        try:
            if event_type not in EVENT_TYPES:
                return {"sent": 0, "skipped": f"unknown event type: {event_type}"}
            gated = self._gate(event_type)
            if gated is not None:
                return {"sent": 0, "skipped": gated}
            safe_data = {k: self._safe_value(v) for k, v in (data or {}).items()}
            safe_prov = {k: self._safe_value(v) for k, v in (provenance or {}).items()}
            if extra_text:
                safe_data["ai_note"] = str(extra_text)[:300]
            if daily_key is not None:
                day_key = f"{_DAILY_PREFIX}{daily_key}"
                if await self._already_sent(day_key):
                    return {"sent": 0, "skipped": "duplicate"}
                dedup_key = f"{event_type}:daily:{daily_key}"
            elif dedup_suffix is not None:
                safe_data["_scope"] = str(dedup_suffix)[:64]
                dedup_key = f"{event_type}:{dedup_suffix}"
                if await self._already_sent(dedup_key):
                    return {"sent": 0, "skipped": "duplicate"}
            else:
                dedup_key = self._dedup_key(event_type, safe_prov)
                if await self._already_sent(dedup_key):
                    return {"sent": 0, "skipped": "duplicate"}
            event = NotificationEvent(
                event_type=event_type, dedup_key=dedup_key,
                data=safe_data, provenance=safe_prov,
            )
            group = f"{event_type}:{debounce_group or dedup_key}"
            if self._debounced_now(group, event_type):
                return {"sent": 0, "skipped": "debounced"}
            if not self._rate_ok():
                return {"sent": 0, "skipped": "rate_limited"}
            recipients = await self._recipients()
            if not recipients:
                return {"sent": 0, "skipped": "no recipients"}
            client = await self._get_client()
            if client is None:
                return {"sent": 0, "skipped": "telegram not configured"}
            sent = 0
            for chat_id, lang in recipients:
                try:
                    text = render_notification(event, lang)
                    if extra_text and daily_key is not None:
                        text = f"{text}\n🤖 {str(extra_text)[:300]}"
                        from app.agent.telegram import _finalize_telegram

                        text = _finalize_telegram(text)
                    await client.send_message(chat_id, text)
                    sent += 1
                except Exception as exc:  # noqa: BLE001 - one bad recipient stops nothing
                    logger.warning("notification_send_failed", extra={"event": event_type, "error": str(exc)[:200]})
            if sent > 0:
                await self._mark_sent(event.dedup_key)
                if daily_key is not None:
                    await self._mark_sent(day_key)
                self._mark_debounced(group, event_type)
                self._sent_times.append(time.monotonic())
                return {"sent": sent, "dedup_key": event.dedup_key}
            await self._remember_pending(event)
            return {"sent": 0, "skipped": "send failed (queued for retry)"}
        except Exception as exc:  # noqa: BLE001 - notify must never raise
            logger.warning("notification_failed", extra={"event": event_type, "error": str(exc)[:200]})
            return {"sent": 0, "skipped": f"error: {exc}"}

    # ------------------------------------------------------------ scan (event tail)

    async def scan(self) -> dict[str, Any]:
        """Sweep new journal/audit/status events since persistent cursors.

        Retries queued pending sends first, then edge events (trades,
        transfers, audit, recommendations) and level events (exchange health,
        market-data freshness, execution quality). Cursors always advance;
        failed edge sends are queued for retry (bounded).
        """
        report: dict[str, Any] = {"sent": 0, "skipped": {}, "checked": []}
        try:
            if not self._notifications_enabled():
                report["skipped"]["all"] = "disabled"
                return report
            # Pending retries first (idempotent via dedup keys).
            try:
                pending = await self._take_pending(limit=20)
                for item in pending:
                    try:
                        res = await self.notify(str(item.get("event_type", "")),
                                                dict(item.get("data", {}) or {}),
                                                dict(item.get("provenance", {}) or {}))
                        report["sent"] += int(res.get("sent", 0))
                    except Exception:
                        continue
                if pending:
                    report["checked"].append(f"pending:{len(pending)}")
            except Exception:
                pass
            for step in (
                self._scan_trades,
                self._scan_transfers,
                self._scan_audit,
                self._scan_recommendations,
                self._scan_levels,
            ):
                try:
                    res = await step()
                    report["sent"] += int(res.get("sent", 0))
                    if res.get("checked"):
                        report["checked"].append(res["checked"])
                except Exception as exc:  # noqa: BLE001 - one bad source stops nothing
                    logger.warning("notification_scan_step_failed",
                                   extra={"step": getattr(step, "__name__", "?"), "error": str(exc)[:200]})
            return report
        except Exception as exc:  # noqa: BLE001
            logger.warning("notification_scan_failed", extra={"error": str(exc)[:200]})
            report["skipped"]["all"] = f"error: {exc}"
            return report

    async def _scan_trades(self) -> dict[str, Any]:
        services, sent, newest = self._services, 0, None
        try:
            trades = await services.trades.list_recent(limit=50)
        except Exception:
            return {"sent": 0}
        cursor = await self._get_cursor(_CURSOR_TRADES)
        fresh: list[Any] = []
        for t in trades:
            status = t.status.value if hasattr(t.status, "value") else str(t.status)
            if status not in _TERMINAL_TRADE_STATUSES:
                continue
            if cursor and str(t.created_at) <= str(cursor):
                continue
            fresh.append(t)
        slippage_limit = self._slippage_limit_bps()
        for trade in sorted(fresh, key=lambda t: (str(t.created_at), t.id)):
            status = trade.status.value if hasattr(trade.status, "value") else str(trade.status)
            base_data = {
                "strategy": str(trade.strategy.value if hasattr(trade.strategy, "value") else trade.strategy),
                "route": str(trade.route or "?")[:80],
                "exchange": str(trade.exchange_id),
                "net": str(trade.net_profit),
                "bps": str(trade.net_profit_bps),
                "fees": str(trade.fees_quote),
                "trade_id": trade.id,
            }
            if status == "completed":
                res = await self.notify("trade_completed", dict(base_data), {"trade_id": trade.id})
                sent += int(res.get("sent", 0))
            elif status == "failed":
                res = await self.notify(
                    "trade_failed", dict(base_data, error=str(trade.error or "failed")[:160]),
                    {"trade_id": trade.id})
                sent += int(res.get("sent", 0))
            elif status == "manual_review":
                res = await self.notify(
                    "recovery_required",
                    {"kind": "trade", "ref": trade.id, "error": str(trade.error or "manual review")[:160]},
                    {"trade_id": trade.id})
                sent += int(res.get("sent", 0))
            # Rejected legs inside the trade (dedup per trade).
            try:
                rejected = [o for o in (trade.orders or ())
                            if str((o.get("status") or "")) == "rejected"]
            except Exception:
                rejected = []
            for order in rejected[:3]:
                res = await self.notify(
                    "order_rejected",
                    {"trade_id": trade.id, "symbol": str(order.get("symbol", "?"))[:40],
                     "side": str(order.get("side", "?"))[:12], "status": "rejected"},
                    {"trade_id": trade.id, "order_id": str(order.get("id", "?"))})
                sent += int(res.get("sent", 0))
            # Unusual slippage vs the configured risk limit.
            try:
                slip = Decimal(str(trade.slippage_bps))
            except Exception:
                slip = None
            if slip is not None and slippage_limit is not None and slip > slippage_limit:
                res = await self.notify(
                    "unusual_slippage",
                    {"trade_id": trade.id, "slippage": str(trade.slippage_bps), "limit": str(slippage_limit)},
                    {"trade_id": trade.id})
                sent += int(res.get("sent", 0))
            newest = str(trade.created_at)
        if newest is not None:
            await self._set_cursor(_CURSOR_TRADES, newest)
        return {"sent": sent, "checked": f"trades:{len(fresh)}"}

    async def _scan_transfers(self) -> dict[str, Any]:
        services, sent = self._services, 0
        try:
            records = await services.transfers.list_by_state("manual_review")
        except Exception:
            return {"sent": 0}
        cursor = await self._get_cursor(_CURSOR_TRANSFERS)
        fresh = [r for r in records if not cursor or str(r.updated_at) > str(cursor)]
        newest = None
        for record in sorted(fresh, key=lambda r: (str(r.updated_at), r.id))[:20]:
            res = await self.notify(
                "recovery_required",
                {"kind": "transfer",
                 "ref": f"{record.source_exchange}->{record.dest_exchange} {record.asset}",
                 "error": str(record.error or "manual review")[:160]},
                {"transfer_id": record.id})
            sent += int(res.get("sent", 0))
            newest = str(record.updated_at)
        if newest is not None:
            await self._set_cursor(_CURSOR_TRANSFERS, newest)
        return {"sent": sent, "checked": f"transfers:{len(fresh)}"}

    async def _scan_audit(self) -> dict[str, Any]:
        services, sent = self._services, 0
        try:
            entries = await services.audit.list_recent(limit=100)
        except Exception:
            return {"sent": 0}
        cursor = await self._get_cursor(_CURSOR_AUDIT)
        try:
            cursor_id = int(cursor) if cursor is not None else 0
        except (TypeError, ValueError):
            cursor_id = 0
        newest = cursor_id
        for entry in sorted(entries, key=lambda e: (getattr(e, "id", 0) or 0)):
            entry_id = int(getattr(entry, "id", 0) or 0)
            newest = max(newest, entry_id)
            if entry_id <= cursor_id:
                continue
            action = str(getattr(entry, "action", ""))
            message = str(getattr(entry, "message", ""))[:160]
            if action in _RISK_AUDIT_ACTIONS:
                res = await self.notify("risk_event", {"action": action, "message": message}, {})
                sent += int(res.get("sent", 0))
            elif action.endswith(_FAILED_SUFFIX):
                res = await self.notify("system_error", {"action": action, "message": message}, {})
                sent += int(res.get("sent", 0))
        await self._set_cursor(_CURSOR_AUDIT, newest)
        return {"sent": sent, "checked": f"audit:{len(entries)}"}

    async def _scan_recommendations(self) -> dict[str, Any]:
        services, sent = self._services, 0
        repo = getattr(services, "agent_recommendations", None)
        if repo is None or not hasattr(repo, "list_recent"):
            return {"sent": 0}
        try:
            recs = await repo.list_recent(limit=20)
        except Exception:
            return {"sent": 0}
        cursor = await self._get_cursor(_CURSOR_RECS)
        newest = None
        for rec in sorted(recs, key=lambda r: (str(r.created_at), r.id)):
            status = rec.status.value if hasattr(rec.status, "value") else str(rec.status)
            if status != "pending":
                continue
            if cursor and str(rec.created_at) <= str(cursor):
                continue
            res = await self.notify(
                "new_recommendation",
                {"parameter": str(rec.parameter)[:80],
                 "old": str(rec.current_value or rec.old_value or "-")[:40],
                 "proposed": str(rec.proposed_value)[:40],
                 "reason": str(rec.reason or "")[:120],
                 "confidence": f"{float(rec.confidence):.2f}",
                 "rec_id": rec.id},
                {"recommendation_id": rec.id})
            sent += int(res.get("sent", 0))
            newest = str(rec.created_at)
        if newest is not None:
            await self._set_cursor(_CURSOR_RECS, newest)
        return {"sent": sent, "checked": "recommendations"}

    async def _scan_levels(self) -> dict[str, Any]:
        """Level-triggered health events (debounced, hourly dedup per subject)."""
        services, sent, checked = self._services, 0, []
        bucket = datetime.now(UTC).strftime("%Y-%m-%dT%H")
        # Exchange health from the existing status snapshot (read-only).
        try:
            snapshot = services.manager.status_snapshot()
        except Exception:
            snapshot = {}
        for venue, info in (snapshot or {}).items():
            status = str((info or {}).get("status", "unknown"))
            if status in ("offline", "unknown", "maintenance"):
                res = await self.notify(
                    "exchange_offline", {"exchange": str(venue), "status": status}, {},
                    dedup_suffix=f"{venue}:{bucket}", debounce_group=str(venue))
                sent += int(res.get("sent", 0))
        checked.append(f"exchanges:{len(snapshot or {})}")
        # Market-data freshness from the existing store (read-only).
        try:
            stale, total = await self._stale_books()
        except Exception:
            stale, total = 0, 0
        if total > 0 and stale / total >= 0.5:
            res = await self.notify(
                "market_data_stale", {"stale": str(stale), "total": str(total)}, {},
                dedup_suffix=f"books:{bucket}", debounce_group="books")
            sent += int(res.get("sent", 0))
        checked.append(f"books:{stale}/{total}")
        # Execution quality over the recent window (deterministic rule).
        try:
            trades = await services.trades.list_recent(limit=20)
        except Exception:
            trades = []
        terminal = [t for t in trades
                    if (t.status.value if hasattr(t.status, "value") else str(t.status)) in ("completed", "failed", "manual_review")]
        failed = [t for t in terminal
                  if (t.status.value if hasattr(t.status, "value") else str(t.status)) in ("failed", "manual_review")]
        if len(terminal) >= 5 and len(failed) / len(terminal) >= 0.5:
            res = await self.notify(
                "execution_degraded",
                {"failed": str(len(failed)), "total": str(len(terminal))}, {},
                dedup_suffix=f"window:{bucket}", debounce_group="exec-window")
            sent += int(res.get("sent", 0))
        checked.append(f"window:{len(terminal)}")
        return {"sent": sent, "checked": ",".join(checked)}

    async def _stale_books(self) -> tuple[int, int]:
        services = self._services
        store = getattr(services, "store", None)
        watch = getattr(services, "watch_symbols", ()) or ()
        venues = list(services.manager.enabled_ids()) if hasattr(services.manager, "enabled_ids") else []
        total = stale = 0
        for venue in venues:
            for symbol in watch:
                try:
                    book = store.order_book(venue, symbol)
                except Exception:
                    book = None
                total += 1
                if book is None:
                    stale += 1
                    continue
                try:
                    if store.is_stale(store.age_ms(book)):
                        stale += 1
                except Exception:
                    stale += 1
        return stale, total

    # ------------------------------------------------------------ daily summary

    async def daily_summary(self, *, llm: Any | None = None, now: datetime | None = None) -> dict[str, Any]:
        """Deterministic daily summary (numbers computed here, never by the LLM).

        ``llm`` is optional: when provided it may append a short formatted
        explanation of the validated numbers; any LLM failure falls back to
        the template text (isolation).
        """
        from app.agent.journal import aggregate_stats

        now = now or datetime.now(UTC)
        start = datetime.combine(now.date(), dtime.min, tzinfo=UTC)
        services = self._services
        try:
            trades = await services.trades.list_recent(limit=200)
        except Exception:
            trades = []
        day = [t for t in trades if getattr(t, "created_at", None) is not None and t.created_at >= start]
        stats = aggregate_stats([t for t in day])
        try:
            open_transfers = len(await services.transfers.list_open())
        except Exception:
            open_transfers = 0
        try:
            n_exp = await services.agent_experiences.count()
        except Exception:
            n_exp = 0
        try:
            lessons = await services.agent_lessons.list_all()
            n_les = len(lessons)
        except Exception:
            n_les = 0
        summary = {
            "kind": "daily_summary",
            "date": now.date().isoformat(),
            "n": stats["n"],
            "completed": stats["completed"],
            "failed": stats["failed"],
            "pnl": stats["total_pnl"],
            "avg_bps": stats["avg_net_bps"],
            "win_rate": stats["win_rate"],
            "experiences": n_exp,
            "lessons": n_les,
            "open_transfers": open_transfers,
        }
        explanation: str | None = None
        if llm is not None:
            try:
                explanation = await self._explain_with_llm(summary, llm)
            except Exception as exc:  # noqa: BLE001 - LLM failure never breaks the summary
                logger.warning("daily_summary_llm_failed", extra={"error": str(exc)[:200]})
                explanation = None
        return {"summary": summary, "explanation": explanation}

    async def maybe_send_daily(self, *, now: datetime | None = None, llm: Any | None = None) -> dict[str, Any]:
        """Send the daily summary once per UTC day (date-keyed idempotency)."""
        now = now or datetime.now(UTC)
        if not self._notifications_enabled():
            return {"sent": 0, "skipped": "disabled"}
        if not self._daily_enabled():
            return {"sent": 0, "skipped": "daily disabled"}
        day = now.date().isoformat()
        if await self._already_sent(f"{_DAILY_PREFIX}{day}"):
            return {"sent": 0, "skipped": "duplicate"}
        try:
            result = await self.daily_summary(llm=llm, now=now)
        except Exception as exc:  # noqa: BLE001
            return {"sent": 0, "skipped": f"error: {exc}"}
        summary = result["summary"]
        data = dict(summary)
        data.pop("kind", None)
        res = await self.notify("daily_summary", data, {"date": summary["date"]},
                                dedup_suffix=None, daily_key=day,
                                extra_text=result.get("explanation"))
        return res

    async def explain_summary(self, summary: dict[str, Any], llm: Any) -> str | None:
        """LLM formats validated summary numbers (or None on any failure)."""
        try:
            return await self._explain_with_llm(summary, llm)
        except Exception:
            return None

    async def _explain_with_llm(self, summary: dict[str, Any], llm: Any) -> str:
        from app.agent.providers.base import LLMMessage, LLMRequest, filter_secrets_from_text

        numbers = ", ".join(f"{k}={v}" for k, v in summary.items() if k != "kind")
        prompt = (
            "Summarize these validated bot statistics in two sentences. "
            "Do not invent values; use only these numbers: " + numbers[:800]
        )
        response = await llm.complete(LLMRequest(messages=(LLMMessage(role="user", content=prompt),)))
        text = filter_secrets_from_text(response.content or "")[:400].strip()
        if len(text) < 5:
            raise ValueError("empty LLM explanation")
        return text

    # ------------------------------------------------------------ gates & state

    def _notifications_enabled(self) -> bool:
        try:
            return bool(self._services.settings.agent.notifications_enabled)
        except Exception:
            return False

    def _daily_enabled(self) -> bool:
        try:
            return bool(self._services.settings.agent.notifications_daily_enabled)
        except Exception:
            return True

    def _gate(self, event_type: str) -> str | None:
        """Master switch + per-event filter. Returns skip reason or None."""
        try:
            cfg = self._services.settings.agent
        except Exception:
            return "no agent config"
        if not bool(getattr(cfg, "notifications_enabled", False)):
            return "disabled"
        allowed = tuple(getattr(cfg, "notifications_events", ()) or ())
        if allowed and event_type not in allowed:
            return "filtered"
        return None

    def _dedup_key(self, event_type: str, provenance: dict[str, Any] | None) -> str:
        prov = provenance or {}
        for key in ("trade_id", "order_id", "transfer_id", "recommendation_id", "rec_id", "date"):
            value = prov.get(key)
            if value:
                return f"{event_type}:{value}"
        return f"{event_type}:{abs(hash(str(sorted((prov or {}).items())))) & 0xFFFFFFFF}"

    async def _already_sent(self, dedup_key: str) -> bool:
        try:
            return await self._services.bot_state.get(f"{_SENT_PREFIX}{dedup_key}") is not None
        except Exception:
            return False

    async def _mark_sent(self, dedup_key: str) -> None:
        try:
            await self._services.bot_state.set(f"{_SENT_PREFIX}{dedup_key}", utc_now().isoformat())
        except Exception as exc:  # noqa: BLE001 - dedup write must not break sends
            logger.warning("notification_dedup_write_failed", extra={"error": str(exc)[:200]})

    def _debounced_now(self, group: str, event_type: str) -> bool:
        window = NOISY_DEBOUNCE_SECONDS.get(event_type, _DEFAULT_DEBOUNCE_SECONDS)
        last = self._debounced.get(group)
        now = time.monotonic()
        if last is not None and now - last < window:
            return True
        return False

    def _mark_debounced(self, group: str, event_type: str) -> None:
        self._debounced[group] = time.monotonic()
        _ = event_type

    def _rate_ok(self) -> bool:
        try:
            limit = int(self._services.settings.agent.notifications_max_per_hour or 20)
        except Exception:
            limit = 20
        now = time.monotonic()
        while self._sent_times and now - self._sent_times[0] > 3600:
            self._sent_times.popleft()
        return len(self._sent_times) < limit

    async def _recipients(self) -> list[tuple[int, str]]:
        """Authorized operator user IDs with their stored language (default en)."""
        try:
            allowed = tuple(self._services.settings.telegram.allowed_user_ids or ())
        except Exception:
            return []
        if not allowed:
            return []
        from app.telegram.i18n import lang_storage_key

        recipients: list[tuple[int, str]] = []
        for user_id in allowed:
            try:
                lang = await self._services.bot_state.get(lang_storage_key(int(user_id)))
            except Exception:
                lang = None
            recipients.append((int(user_id), lang if lang in ("en", "ru") else "en"))
        return recipients

    async def _get_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        try:
            token = self._services.settings.telegram.bot_token.get_secret_value().strip()
        except Exception:
            return None
        if not token:
            return None
        try:
            from app.telegram.client import TelegramClient

            return TelegramClient(token)
        except Exception:
            return None

    async def _get_cursor(self, key: str) -> Any | None:
        try:
            return await self._services.bot_state.get(key)
        except Exception:
            return None

    async def _set_cursor(self, key: str, value: Any) -> None:
        try:
            await self._services.bot_state.set(key, value)
        except Exception as exc:  # noqa: BLE001
            logger.warning("notification_cursor_write_failed", extra={"error": str(exc)[:200]})

    async def _remember_pending(self, event: NotificationEvent) -> None:
        try:
            raw = await self._services.bot_state.get(_PENDING_KEY) or []
            pending = list(raw) if isinstance(raw, list) else []
            pending.append({"event_type": event.event_type, "data": event.data,
                            "provenance": event.provenance})
            await self._services.bot_state.set(_PENDING_KEY, pending[-20:])
        except Exception:
            pass

    async def _take_pending(self, *, limit: int = 20) -> list[dict[str, Any]]:
        try:
            raw = await self._services.bot_state.get(_PENDING_KEY) or []
            pending = list(raw) if isinstance(raw, list) else []
        except Exception:
            return []
        try:
            await self._services.bot_state.set(_PENDING_KEY, [])
        except Exception:
            pass
        return pending[:limit]

    def _slippage_limit_bps(self) -> Decimal | None:
        try:
            return Decimal(str(self._services.settings.risk.max_slippage_bps))
        except Exception:
            return None

    @staticmethod
    def _safe_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value if not isinstance(value, str) else value[:300]
        return str(value)[:300]
