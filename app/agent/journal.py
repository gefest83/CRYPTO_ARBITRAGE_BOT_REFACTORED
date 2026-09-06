"""Phase 4 — Trading Journal integration (read-only, deterministic).

The AI Agent reaches the persisted trading journal **only** through
:class:`JournalReader`, which sits on top of the existing repositories
(:class:`TradeRepository`, :class:`TransferRepository`,
:class:`AuditLogRepository`). There is:

* no direct database access for the LLM (no SQL strings, no sessions);
* no mutation path (no save/merge/delete — read methods only);
* no secret surface (trade/order/transfer/audit rows carry no credentials;
  outputs project an explicit allow-list of fields).

All financial math (profit, edge, fees, slippage, percentages, PnL,
execution metrics) is computed here, deterministically, from persisted
data. The LLM receives the results as FACTS and must never recompute or
override them — :mod:`app.agent.analysis` keeps them separate from
HYPOTHESES and RECOMMENDATIONS.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from app.config.logging_config import get_logger
from app.models.base import DEC0

__all__ = [
    "JournalReader",
    "MIN_SAMPLE_FOR_CONCLUSIONS",
    "TRADE_ID_RE",
    "aggregate_stats",
    "aggregate_trades",
    "analyze_trade_record",
    "compare_periods",
    "filter_period",
    "missed_opportunities_from_audit",
    "sanitize_trade",
    "split_today_vs_yesterday",
    "summarize_leg",
]

logger = get_logger("agent.journal")

#: Minimum sample size before a statistical result may carry a conclusion.
#: Mirrors the reflection engine's evidence gate (5). Smaller samples are
#: still reported — with ``n`` and ``sufficient: False`` — but never used
#: for conclusions.
MIN_SAMPLE_FOR_CONCLUSIONS = 5

#: Trade-ID detector for queries (``trd-<12 hex>``).
TRADE_ID_RE: re.Pattern[str] = re.compile(r"\btrd-[0-9a-f]{12}\b")

_BPS = Decimal("10000")

# Audit actions that record recovery / failure handling around a trade.
_RECOVERY_ACTION_HINTS: tuple[str, ...] = (
    "RECOVERY",
    "TRIANGLE_EXECUTE",
    "TRIANGLE_RISK_REJECTED",
    "TRANSFER",
    "MANUAL_REVIEW",
    "KILL_SWITCH",
)

# Audit actions that record a *rejected* opportunity (never executed).
_MISSED_ACTION_HINTS: tuple[str, ...] = (
    "TRIANGLE_RISK_REJECTED",
    "OPPORTUNITY",
    "REJECTED",
)


def _dec(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return DEC0


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()  # type: ignore[no-any-return]
        except Exception:
            return str(value)
    return str(value)


# ------------------------------------------------------------------ sanitized views


def summarize_leg(order: dict[str, Any]) -> dict[str, Any]:
    """Project one persisted order dump to its analysis-safe subset."""
    amount = _dec(order.get("amount", 0))
    filled = _dec(order.get("filled_amount", 0))
    fill_ratio = (filled / amount) if amount > DEC0 else DEC0
    return {
        "order_id": str(order.get("id") or order.get("client_order_id") or "?"),
        "symbol": str(order.get("symbol") or "?"),
        "side": str(order.get("side") or "?"),
        "status": str(order.get("status") or "?"),
        "amount": str(amount),
        "filled_amount": str(filled),
        "fill_ratio": str(fill_ratio),
        "price": str(order.get("price") if order.get("price") is not None else "?"),
        "average_price": str(order.get("average_price") if order.get("average_price") is not None else "?"),
        "fee_paid": str(order.get("fee_paid", 0)),
        "fee_currency": order.get("fee_currency"),
        "fills": len(order.get("fills") or ()),
        "error": order.get("error"),
    }


def sanitize_trade(trade: Any) -> dict[str, Any]:
    """Project a :class:`TradeRecord` to the allow-listed journal view.

    Includes completed/failed trades, individual orders + statuses, strategy,
    exchange, route, timestamps, realized values, fees, slippage, duration,
    failure info — and never credentials (the model has none).
    """
    created = getattr(trade, "created_at", None)
    updated = getattr(trade, "updated_at", None)
    duration_s: str | None = None
    try:
        if created is not None and updated is not None:
            duration_s = str((updated - created).total_seconds())
    except Exception:
        duration_s = None
    orders = [summarize_leg(o) for o in (getattr(trade, "orders", ()) or ())]
    return {
        "id": trade.id,
        "strategy": trade.strategy.value if hasattr(trade.strategy, "value") else str(trade.strategy),
        "mode": trade.mode.value if hasattr(trade.mode, "value") else str(trade.mode),
        "exchange_id": trade.exchange_id,
        "route": trade.route,
        "symbols": list(getattr(trade, "symbols", ()) or ()),
        "input_amount": str(trade.input_amount),
        "output_amount": str(trade.output_amount),
        "fees_quote": str(trade.fees_quote),
        "slippage_bps": str(trade.slippage_bps),
        "net_profit": str(trade.net_profit),
        "net_profit_bps": str(trade.net_profit_bps),
        "status": trade.status.value if hasattr(trade.status, "value") else str(trade.status),
        "orders": orders,
        "legs_total": len(orders),
        "legs_filled": sum(1 for o in orders if str(o["status"]) in ("filled",)),
        "error": trade.error,
        "transfer_id": trade.transfer_id,
        "created_at": _iso(created),
        "updated_at": _iso(updated),
        "execution_duration_s": duration_s,
        "provenance": {"journal": "trades", "trade_id": trade.id},
    }


# ------------------------------------------------------------------ single-trade analysis


def analyze_trade_record(
    trade: Any,
    *,
    transfer_plan: dict[str, Any] | None = None,
    audit_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Deterministically analyze one persisted trade (pure function).

    ``transfer_plan`` is the linked :class:`TransferPlan` dump when the trade
    belongs to a transfer workflow (expected edge source). ``audit_refs`` are
    pre-collected related audit entries (failure/recovery info).
    """
    view = sanitize_trade(trade)
    realized_profit = _dec(view["net_profit"])
    realized_bps = _dec(view["net_profit_bps"])

    expected: dict[str, Any] | None = None
    difference: dict[str, Any] | None = None
    if transfer_plan is not None:
        try:
            buy = _dec(transfer_plan.get("buy_price", 0)) * _dec(transfer_plan.get("amount", 0))
            sell = _dec(transfer_plan.get("sell_price", 0)) * _dec(transfer_plan.get("amount", 0))
            exp_net = _dec(transfer_plan.get("net_profit_quote", sell - buy))
            exp_bps = (exp_net / buy * _BPS) if buy > DEC0 else DEC0
            expected = {
                "net_profit": str(exp_net),
                "net_profit_bps": str(exp_bps),
                "source": "transfer_plan",
                "transfer_id": view["transfer_id"],
            }
            difference = {
                "net_profit": str(realized_profit - exp_net),
                "net_profit_bps": str(realized_bps - exp_bps),
            }
        except Exception as exc:  # noqa: BLE001 - never crash analysis
            expected = None
            difference = None
            logger.warning("journal_expected_failed", extra={"trade_id": view["id"], "error": str(exc)[:200]})
    else:
        expected = {
            "net_profit": None,
            "net_profit_bps": None,
            "source": "insufficient_data",
            "reason": "expected edge is not persisted for this trade (triangle opportunities are not journaled)",
        }

    # Fee reconciliation: per-order fee sum vs persisted trade total.
    fee_values: list[Decimal] = []
    for order in getattr(trade, "orders", ()) or ():
        try:
            fee_values.append(_dec(order.get("fee_paid", 0)))
        except Exception:
            continue
    fee_sum = sum(fee_values, DEC0)
    persisted_fees = _dec(view["fees_quote"])

    legs = view["orders"]
    filled_legs = view["legs_filled"]
    outcome = view["status"]
    return {
        "kind": "trade_analysis",
        "trade_id": view["id"],
        "strategy": view["strategy"],
        "exchange_id": view["exchange_id"],
        "route": view["route"],
        "status": outcome,
        "created_at": view["created_at"],
        "updated_at": view["updated_at"],
        "realized": {"net_profit": str(realized_profit), "net_profit_bps": str(realized_bps)},
        "expected": expected,
        "difference": difference,
        "execution": {
            "legs_total": view["legs_total"],
            "legs_filled": filled_legs,
            "legs": legs,
            "execution_duration_s": view["execution_duration_s"],
            "input_amount": view["input_amount"],
            "output_amount": view["output_amount"],
        },
        "fees": {
            "fees_quote": str(persisted_fees),
            "per_order_fee_sum": str(fee_sum),
            "fee_match": fee_sum == persisted_fees,
        },
        "slippage": {"slippage_bps": view["slippage_bps"]},
        "failure": {
            "error": view["error"],
            "recovery": audit_refs or [],
        },
        "outcome": outcome,
        "provenance": {
            "journal": "trades",
            "trade_id": view["id"],
            "transfer_id": view["transfer_id"],
            "order_ids": [str(o.get("order_id")) for o in legs],
        },
    }


# ------------------------------------------------------------------ aggregates


def aggregate_stats(trades: list[Any]) -> dict[str, Any]:
    """Deterministic stats over trade records (or sanitized views)."""
    n = len(trades)
    by_status: dict[str, int] = {}
    pnl = DEC0
    bps_sum = DEC0
    completed = 0
    for t in trades:
        if isinstance(t, dict):
            status = str(t.get("status", "?"))
            net = _dec(t.get("net_profit", 0))
            bps = _dec(t.get("net_profit_bps", 0))
        else:
            status = t.status.value if hasattr(t.status, "value") else str(t.status)
            net = _dec(t.net_profit)
            bps = _dec(t.net_profit_bps)
        by_status[status] = by_status.get(status, 0) + 1
        if status == "completed":
            completed += 1
            pnl += net
            bps_sum += bps
    avg_bps = (bps_sum / completed) if completed else DEC0
    win_rate = (Decimal(completed) / Decimal(n) * Decimal("100")) if n else DEC0
    return {
        "n": n,
        "completed": by_status.get("completed", 0),
        "failed": by_status.get("failed", 0),
        "manual_review": by_status.get("manual_review", 0),
        "executing": by_status.get("executing", 0),
        "total_pnl": str(pnl),
        "avg_net_bps": str(avg_bps),
        "win_rate": str(win_rate),
        "by_status": by_status,
        "sufficient": n >= MIN_SAMPLE_FOR_CONCLUSIONS,
    }


def _group_key(trade: Any, by: str) -> str:
    if isinstance(trade, dict):
        get = trade.get
    else:
        get = lambda k, d=None: getattr(trade, k, d)  # noqa: E731
    if by == "strategy":
        v = get("strategy", "?")
    elif by == "exchange":
        v = get("exchange_id", get("exchange", "?"))
    elif by == "route":
        v = get("route", "?") or "?"
    elif by == "status":
        v = get("status", "?")
    elif by == "day":
        v = str(get("created_at", "?"))[:10]
    elif by == "week":
        v = str(get("created_at", "?"))[:7]
    else:
        v = "all"
    if hasattr(v, "value"):
        v = v.value
    return str(v) if str(v).strip() else "?"


def aggregate_trades(trades: list[Any], *, by: str) -> dict[str, Any]:
    """Aggregate trades by strategy / exchange / route / day / week / status.

    Every group carries its sample size; groups below
    :data:`MIN_SAMPLE_FOR_CONCLUSIONS` are flagged ``sufficient: False`` and
    must not be used for conclusions.
    """
    groups: dict[str, list[Any]] = {}
    for t in trades:
        groups.setdefault(_group_key(t, by), []).append(t)
    result: dict[str, Any] = {}
    for name in sorted(groups):
        stats = aggregate_stats(groups[name])
        result[name] = stats
    return {
        "kind": "aggregation",
        "by": by,
        "groups": result,
        "total_n": len(trades),
        "sufficient": len(trades) >= MIN_SAMPLE_FOR_CONCLUSIONS,
        "min_sample": MIN_SAMPLE_FOR_CONCLUSIONS,
    }


def filter_period(trades: list[Any], *, since: datetime | None, until: datetime | None) -> list[Any]:
    """Filter trades to ``[since, until)`` by ``created_at`` (deterministic)."""
    out: list[Any] = []
    for t in trades:
        if isinstance(t, dict):
            raw = t.get("created_at")
            try:
                ts = datetime.fromisoformat(str(raw)) if raw else None
            except Exception:
                ts = None
        else:
            ts = getattr(t, "created_at", None)
        if ts is None:
            continue
        if since is not None and ts < since:
            continue
        if until is not None and ts >= until:
            continue
        out.append(t)
    return out


def split_today_vs_yesterday(trades: list[Any], *, now: datetime | None = None) -> tuple[list[Any], list[Any]]:
    """Split trades into today vs yesterday (UTC calendar days)."""
    now = now or datetime.now(UTC)
    start_today = datetime.combine(now.date(), time.min, tzinfo=UTC)
    start_yesterday = start_today - timedelta(days=1)
    today = filter_period(trades, since=start_today, until=None)
    yesterday = filter_period(trades, since=start_yesterday, until=start_today)
    return today, yesterday


def compare_periods(
    trades_a: list[Any],
    trades_b: list[Any],
    *,
    label_a: str = "current",
    label_b: str = "previous",
) -> dict[str, Any]:
    """Deterministically compare two periods (deltas are arithmetic FACTS)."""
    stats_a = aggregate_stats(trades_a)
    stats_b = aggregate_stats(trades_b)
    d_pnl = _dec(stats_a["total_pnl"]) - _dec(stats_b["total_pnl"])
    d_bps = _dec(stats_a["avg_net_bps"]) - _dec(stats_b["avg_net_bps"])
    d_win = _dec(stats_a["win_rate"]) - _dec(stats_b["win_rate"])
    sufficient = stats_a["sufficient"] and stats_b["sufficient"]
    if stats_a["n"] == 0 or stats_b["n"] == 0:
        conclusion = "insufficient_data"
    elif not sufficient:
        conclusion = "insufficient_data"
    else:
        direction = "above" if d_pnl > 0 else ("below" if d_pnl < 0 else "equal to")
        conclusion = f"{label_a} realized PnL is {direction} {label_b} by {d_pnl} quote"
    return {
        "kind": "period_comparison",
        "label_a": label_a,
        "label_b": label_b,
        "period_a": stats_a,
        "period_b": stats_b,
        "n_a": stats_a["n"],
        "n_b": stats_b["n"],
        "delta_total_pnl": str(d_pnl),
        "delta_avg_net_bps": str(d_bps),
        "delta_win_rate": str(d_win),
        "sufficient": sufficient,
        "conclusion": conclusion,
    }


def missed_opportunities_from_audit(audit_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Identify missed opportunities **only** from persisted audit records.

    Scanner opportunities are not journaled, so the normal outcome is an
    explicit ``insufficient_data`` result — never invented history.
    """
    refs: list[dict[str, Any]] = []
    for entry in audit_entries:
        action = str(entry.get("action", ""))
        if any(hint in action for hint in _MISSED_ACTION_HINTS):
            refs.append(
                {
                    "ts": entry.get("ts"),
                    "action": action,
                    "message": str(entry.get("message", ""))[:200],
                }
            )
    if not refs:
        return {
            "kind": "missed_opportunities",
            "status": "insufficient_data",
            "n": 0,
            "reason": "no persisted opportunity records in journal/audit (scanner opportunities are not journaled)",
            "refs": [],
        }
    return {
        "kind": "missed_opportunities",
        "status": "observed",
        "n": len(refs),
        "reason": f"{len(refs)} persisted rejection records found; executed trades are the only complete history",
        "refs": refs[:10],
    }


# ------------------------------------------------------------------ read-only service


class JournalReader:
    """Read-only trading-journal access for the AI Agent.

    Wraps the existing repositories; exposes **no** write path (no save /
    merge / delete / SQL / shell). All numbers returned are already-computed
    deterministic results keyed by original trade/order IDs.
    """

    def __init__(self, services: Any) -> None:  # services: AppServices (duck-typed)
        self._services = services

    # ------------------------------------------------------------ retrieval

    async def get_trade(self, trade_id: str) -> dict[str, Any] | None:
        """Sanitized view of one trade, or ``None`` when unknown."""
        trade_id = str(trade_id or "").strip()
        if not trade_id:
            return None
        try:
            trade = await self._services.trades.get(trade_id)
        except Exception as exc:  # noqa: BLE001 - journal must never crash the agent
            logger.warning("journal_get_trade_failed", extra={"trade_id": trade_id[:16], "error": str(exc)[:200]})
            return None
        if trade is None:
            return None
        return sanitize_trade(trade)

    async def list_trades(
        self,
        *,
        limit: int = 50,
        status: str | None = None,
        strategy: str | None = None,
        exchange: str | None = None,
    ) -> list[dict[str, Any]]:
        """Bounded, filterable recent-trade listing (defaults: 50, newest first)."""
        limit = max(1, min(int(limit), 200))
        try:
            trades = await self._services.trades.list_recent(limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("journal_list_trades_failed", extra={"error": str(exc)[:200]})
            return []
        views = [sanitize_trade(t) for t in trades]
        if status is not None:
            views = [v for v in views if v["status"] == status]
        if strategy is not None:
            views = [v for v in views if v["strategy"] == strategy]
        if exchange is not None:
            views = [v for v in views if v["exchange_id"] == exchange]
        return views

    async def transfer_plan(self, transfer_id: str | None) -> dict[str, Any] | None:
        """Public expected-edge source for extraction (read-only)."""
        return await self._transfer_plan_for(transfer_id)

    async def _transfer_plan_for(self, transfer_id: str | None) -> dict[str, Any] | None:
        if not transfer_id:
            return None
        try:
            record = await self._services.transfers.get(transfer_id)
        except Exception:
            return None
        if record is None or getattr(record, "plan", None) is None:
            return None
        try:
            plan = record.plan
            dump = plan.model_dump() if hasattr(plan, "model_dump") else dict(plan)
            buy = _dec(dump.get("buy_price", 0)) * _dec(dump.get("amount", 0))
            sell = _dec(dump.get("sell_price", 0)) * _dec(dump.get("amount", 0))
            buy_fee = buy * _dec(dump.get("buy_fee_bps", 10)) / _BPS
            sell_fee = sell * _dec(dump.get("sell_fee_bps", 10)) / _BPS
            withdraw = _dec(dump.get("withdrawal_fee", 0)) * _dec(dump.get("sell_price", 0)) + _dec(
                dump.get("network_cost_quote", 0)
            )
            net = sell - buy - buy_fee - sell_fee - withdraw
            return {
                "buy_price": str(dump.get("buy_price")),
                "sell_price": str(dump.get("sell_price")),
                "amount": str(dump.get("amount")),
                "net_profit_quote": str(net),
                "transfer_id": record.id,
                "state": record.state.value if hasattr(record.state, "value") else str(record.state),
            }
        except Exception:
            return None

    async def _audit_refs_for(self, trade_id: str, limit: int = 5) -> list[dict[str, Any]]:
        try:
            entries = await self._services.audit.list_recent(limit=100)
        except Exception:
            return []
        refs: list[dict[str, Any]] = []
        for entry in entries:
            action = getattr(entry, "action", "")
            message = getattr(entry, "message", "")
            context = getattr(entry, "context", None) or {}
            haystack = f"{message} {context}"
            if trade_id in haystack or any(h in str(action) for h in _RECOVERY_ACTION_HINTS if trade_id in haystack):
                refs.append(
                    {
                        "ts": _iso(getattr(entry, "ts", None)),
                        "action": str(action),
                        "message": str(message)[:200],
                    }
                )
                if len(refs) >= limit:
                    break
        return refs

    # ------------------------------------------------------------ deterministic tools

    async def analyze_trade(self, trade_id: str) -> dict[str, Any]:
        """Full deterministic analysis of one journaled trade."""
        trade_id = str(trade_id or "").strip()
        if not trade_id:
            return {"kind": "trade_analysis", "status": "insufficient_data", "reason": "empty trade id", "trade_id": trade_id}
        try:
            trade = await self._services.trades.get(trade_id)
        except Exception as exc:  # noqa: BLE001
            return {"kind": "trade_analysis", "status": "insufficient_data", "reason": f"journal read failed: {exc}", "trade_id": trade_id}
        if trade is None:
            return {"kind": "trade_analysis", "status": "insufficient_data", "reason": f"trade not found: {trade_id}", "trade_id": trade_id}
        plan = await self._transfer_plan_for(getattr(trade, "transfer_id", None))
        refs = await self._audit_refs_for(trade.id)
        return analyze_trade_record(trade, transfer_plan=plan, audit_refs=refs)

    async def strategy_performance(self, *, limit: int = 100) -> dict[str, Any]:
        """PnL / win-rate grouped by strategy (with sample sizes)."""
        try:
            trades = await self._services.trades.list_recent(limit=min(int(limit), 200))
        except Exception:
            trades = []
        if not trades:
            return {"kind": "aggregation", "by": "strategy", "groups": {}, "total_n": 0, "sufficient": False,
                    "status": "insufficient_data", "reason": "no journaled trades"}
        return aggregate_trades(trades, by="strategy")

    async def exchange_performance(self, *, limit: int = 100) -> dict[str, Any]:
        """PnL / execution quality grouped by exchange (with sample sizes)."""
        try:
            trades = await self._services.trades.list_recent(limit=min(int(limit), 200))
        except Exception:
            trades = []
        if not trades:
            return {"kind": "aggregation", "by": "exchange", "groups": {}, "total_n": 0, "sufficient": False,
                    "status": "insufficient_data", "reason": "no journaled trades"}
        return aggregate_trades(trades, by="exchange")

    async def route_performance(self, *, limit: int = 100) -> dict[str, Any]:
        """PnL grouped by route (with sample sizes)."""
        try:
            trades = await self._services.trades.list_recent(limit=min(int(limit), 200))
        except Exception:
            trades = []
        if not trades:
            return {"kind": "aggregation", "by": "route", "groups": {}, "total_n": 0, "sufficient": False,
                    "status": "insufficient_data", "reason": "no journaled trades"}
        return aggregate_trades(trades, by="route")

    async def compare_periods(
        self,
        *,
        since_a: datetime | None = None,
        until_a: datetime | None = None,
        since_b: datetime | None = None,
        until_b: datetime | None = None,
        label_a: str = "current",
        label_b: str = "previous",
        limit: int = 200,
    ) -> dict[str, Any]:
        """Compare two explicit periods over journaled trades."""
        try:
            trades = await self._services.trades.list_recent(limit=min(int(limit), 200))
        except Exception:
            trades = []
        period_a = filter_period(trades, since=since_a, until=until_a)
        period_b = filter_period(trades, since=since_b, until=until_b)
        return compare_periods(period_a, period_b, label_a=label_a, label_b=label_b)

    async def compare_today_vs_yesterday(self, *, now: datetime | None = None, limit: int = 200) -> dict[str, Any]:
        """Today vs previous period (UTC calendar days)."""
        try:
            trades = await self._services.trades.list_recent(limit=min(int(limit), 200))
        except Exception:
            trades = []
        today, yesterday = split_today_vs_yesterday(trades, now=now)
        return compare_periods(today, yesterday, label_a="today", label_b="yesterday")

    async def missed_opportunities(self, *, limit: int = 100) -> dict[str, Any]:
        """Persisted rejection records only; ``insufficient_data`` when absent."""
        try:
            entries = await self._services.audit.list_recent(limit=min(int(limit), 200))
        except Exception:
            entries = []
        views = [
            {"ts": _iso(getattr(e, "ts", None)), "action": str(getattr(e, "action", "")),
             "message": str(getattr(e, "message", ""))}
            for e in entries
        ]
        return missed_opportunities_from_audit(views)
