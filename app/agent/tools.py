"""Read-only agent tools.

The advisor is an analytical layer, not a trading engine. Every tool in
this allowlist is **READ ONLY**:

* no order placement, no cancellation, no withdrawal
* no credential exposure (API keys / secrets never leave the adapter)
* no arbitrary SQL / shell / Python execution
* no configuration mutation, no risk-limit mutation

The balance tool delegates to :meth:`AppServices.balances` — the same path the
CLI and Telegram use — so paper wallets vs. real venues and the existing
credential gating are preserved. Secrets are stripped by the adapters
themselves; this layer does not introduce a new credential path.

Enforcement: the class exposes *only* the methods named here. Adding a tool
requires an explicit code change and review; there is no generic
``execute(sql)`` / ``run_python(code)``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.config.logging_config import get_logger

__all__ = ["AgentTools", "ToolAccessBlocked"]

logger = get_logger("agent.tools")


class ToolAccessBlocked(Exception):
    """Raised when code attempts a non-allowlisted tool operation."""

    pass


_ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "get_recent_trades",
        "get_trade_statistics",
        "get_scan_statistics",
        "get_current_parameters",
        "get_risk_state",
        "get_exchange_status",
        "get_balances",
        "get_recent_journal",
        "get_previous_recommendations",
        "get_memory",
    }
)


class AgentTools:
    """Strict read-only tool allowlist backed by :class:`AppServices`.

    Each method is ``async`` and returns sanitized, serialisable data that is
    safe to feed into analysis. API keys, secrets, DSNs and raw ``.env``
    values are never included.
    """

    def __init__(self, services: Any) -> None:
        # ``services`` is typed loosely to avoid an import cycle in tool tests;
        # at runtime it is :class:`app.services.AppServices`.
        self._services = services

    # ------------------------------------------------------------------
    # allow-list introspection (for tests: the advisor must advertise exactly 10 tools)
    # ------------------------------------------------------------------

    @property
    def allowed_tools(self) -> frozenset[str]:
        return _ALLOWED_TOOLS

    def is_allowed(self, name: str) -> bool:
        return name in _ALLOWED_TOOLS

    # ------------------------------------------------------------------
    # trades & statistics
    # ------------------------------------------------------------------

    async def get_recent_trades(self, limit: int = 20) -> list[dict[str, Any]]:
        """Most recent trades (strategy, route, status, net_profit, ...).

        No secrets; no raw orders beyond the already-persisted sanitized
        ``TradeRecord.orders`` dumps (which never contain credentials).
        """
        trades = await self._services.trades.list_recent(limit=limit)
        return [
            {
                "id": t.id,
                "strategy": t.strategy.value if hasattr(t.strategy, "value") else str(t.strategy),
                "mode": t.mode.value if hasattr(t.mode, "value") else str(t.mode),
                "exchange_id": t.exchange_id,
                "route": t.route,
                "symbols": list(t.symbols),
                "input_amount": str(t.input_amount),
                "output_amount": str(t.output_amount),
                "net_profit": str(t.net_profit),
                "net_profit_bps": str(t.net_profit_bps),
                "status": t.status.value if hasattr(t.status, "value") else str(t.status),
                "error": t.error,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in trades
        ]

    async def get_trade_statistics(self) -> dict[str, Any]:
        """Aggregate statistics over recent trades — no mutation."""
        trades = await self._services.trades.list_recent(limit=100)
        if not trades:
            return {
                "total": 0,
                "completed": 0,
                "failed": 0,
                "manual_review": 0,
                "total_pnl": "0",
                "avg_net_bps": "0",
                "win_rate": "0",
            }
        by_status: dict[str, int] = {}
        pnl = Decimal("0")
        bps_sum = Decimal("0")
        completed = 0
        for t in trades:
            key = t.status.value if hasattr(t.status, "value") else str(t.status)
            by_status[key] = by_status.get(key, 0) + 1
            if key == "completed":
                completed += 1
                pnl += t.net_profit if isinstance(t.net_profit, Decimal) else Decimal(str(t.net_profit))
                bps_sum += t.net_profit_bps if isinstance(t.net_profit_bps, Decimal) else Decimal(str(t.net_profit_bps))
        avg_bps = (bps_sum / completed) if completed else Decimal("0")
        win_rate = (Decimal(completed) / Decimal(len(trades)) * Decimal("100")) if trades else Decimal("0")
        return {
            "total": len(trades),
            "completed": by_status.get("completed", 0),
            "failed": by_status.get("failed", 0),
            "manual_review": by_status.get("manual_review", 0),
            "total_pnl": str(pnl),
            "avg_net_bps": str(avg_bps),
            "win_rate": str(win_rate),
            "by_status": by_status,
        }

    async def get_scan_statistics(self) -> dict[str, Any]:
        """Market-data freshness / scan health — read-only view of :class:`MarketDataStore`."""
        store = getattr(self._services, "store", None)
        if store is None or not hasattr(store, "stats"):
            return {"note": "market data store not available"}
        stats = store.stats()
        # Never leak credentials; stats is purely book/ticker counts.
        return dict(stats)

    # ------------------------------------------------------------------
    # parameters / risk / exchanges
    # ------------------------------------------------------------------

    async def get_current_parameters(self) -> dict[str, Any]:
        """Non-sensitive snapshot of current configuration.

        Returns only numeric / enumerated parameters (risk limits, arbitrage
        thresholds, execution timeouts, market-data staleness) — never API keys,
        database URLs, or Telegram tokens.
        """
        settings = self._services.settings
        return {
            "mode": settings.mode.value if hasattr(settings.mode, "value") else str(settings.mode),
            "trading": {
                "mode": settings.trading.mode.value if hasattr(settings.trading.mode, "value") else str(settings.trading.mode),
                "base_currency": settings.trading.base_currency,
            },
            "risk": {
                "max_trade_size": str(settings.risk.max_trade_size),
                "min_net_profit_bps": str(settings.risk.min_net_profit_bps),
                "max_daily_loss": str(settings.risk.max_daily_loss),
                "max_open_transfers": settings.risk.max_open_transfers,
                "max_exchange_exposure": str(settings.risk.max_exchange_exposure),
                "max_asset_exposure": str(settings.risk.max_asset_exposure),
                "max_slippage_bps": str(settings.risk.max_slippage_bps),
                "max_data_age_ms": settings.risk.max_data_age_ms,
            },
            "arbitrage": {
                "enable_triangular": settings.arbitrage.enable_triangular,
                "triangle_assets": list(settings.arbitrage.triangle_assets),
                "triangle_min_net_bps": str(settings.arbitrage.triangle_min_net_bps),
                "max_results": settings.arbitrage.max_results,
            },
            "transfer": {
                "assets": list(settings.transfer.assets[:10]),  # truncated for brevity
                "assets_total": len(settings.transfer.assets),
                "min_net_profit_bps": str(settings.transfer.min_net_profit_bps),
                "min_notional_quote": str(settings.transfer.min_notional_quote),
                "max_notional_quote": str(settings.transfer.max_notional_quote),
            },
            "execution": {
                "leg_timeout_seconds": settings.execution.leg_timeout_seconds,
                "paper_max_slippage_bps": settings.execution.paper_max_slippage_bps,
                "auto_interval_seconds": settings.execution.auto_interval_seconds,
            },
            "market_data": {
                "stale_after_ms": settings.market_data.stale_after_ms,
                "streams_enabled": settings.market_data.streams_enabled,
            },
        }

    async def get_risk_state(self) -> dict[str, Any]:
        """Live risk state (kill switch, daily PnL, open transfers, limits)."""
        risk_state = getattr(self._services, "risk_state", None)
        guard = getattr(self._services, "guard", None)
        risk = getattr(self._services, "risk", None)
        # Prefer the already-cached status() view where possible
        try:
            status = await self._services.status()
            # Return the risk slice plus kill-switch info — sanitized copy
            return {
                "daily_pnl": status.get("risk", {}).get("daily_pnl", "0"),
                "open_transfers": status.get("risk", {}).get("open_transfers", 0),
                "limits": status.get("risk", {}).get("limits", {}),
                "kill_switch": status.get("guard", {}),
            }
        except Exception:
            pass
        # Fallback without status()
        return {
            "daily_pnl": str(getattr(risk_state, "daily_pnl", "0")) if risk_state else "0",
            "open_transfers": getattr(risk_state, "open_transfers", 0) if risk_state else 0,
            "kill_switch_engaged": bool(getattr(guard, "is_halted", False)) if guard else False,
            "limits": {},
        }

    async def get_exchange_status(self) -> dict[str, Any]:
        """Per-venue health (online / degraded / offline / data_only …).

        Credential presence is reduced to the already-sanitized
        ``keys: present | missing`` token from :meth:`ExchangeManager.status_snapshot`;
        no key material is ever returned.
        """
        manager = getattr(self._services, "manager", None)
        if manager is None or not hasattr(manager, "status_snapshot"):
            return {}
        snapshot: dict[str, Any] = manager.status_snapshot()
        # Snapshot is already sanitized (keys = present|missing, no secrets). Return a shallow copy.
        return dict(snapshot)

    # ------------------------------------------------------------------
    # balances (must go through AppServices and must not expose keys)
    # ------------------------------------------------------------------

    async def get_balances(self) -> dict[str, Any]:
        """Current balances per venue via the existing adapter path.

        Uses :meth:`AppServices.balances` which handles PAPER wallets vs. live
        ``fetch_balances`` and isolates per-venue failures. The returned payload
        contains asset/free/used only — never API keys.
        """
        snapshots = await self._services.balances()
        result: dict[str, Any] = {}
        for venue, snap in snapshots.items():
            result[venue] = {
                "exchange_id": snap.exchange_id,
                "updated_at": snap.timestamp.isoformat() if hasattr(snap, "timestamp") and snap.timestamp else None,
                "balances": [
                    # Explicitly project only free/used — never raw adapter response
                    {"asset": b.asset, "free": str(b.free), "used": str(b.used)}
                    for b in getattr(snap, "balances", ())
                ],
            }
        return result

    # ------------------------------------------------------------------
    # journal / memory / recommendations
    # ------------------------------------------------------------------

    async def get_recent_journal(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recent audit log entries (read-only)."""
        audit = getattr(self._services, "audit", None)
        if audit is None or not hasattr(audit, "list_recent"):
            return []
        rows = await audit.list_recent(limit=limit)
        return [
            {
                "ts": r.ts.isoformat() if hasattr(r.ts, "isoformat") else str(r.ts),
                "action": r.action,
                "message": r.message,
                # Context may contain trade ids etc., but never credentials (audit log writes already redact)
                "context": r.context,
            }
            for r in rows
        ]

    async def get_previous_recommendations(self, limit: int = 20) -> list[dict[str, Any]]:
        """Prior advisor recommendations (parameter / old / proposed / evidence / confidence / decision)."""
        # The agent's own repository may or may not be wired; try both paths.
        # 1. If the services object carries an agent_recommendations repo (future wiring), use it.
        repo = getattr(self._services, "agent_recommendations", None)
        if repo is not None and hasattr(repo, "list_recent"):
            recs = await repo.list_recent(limit=limit)
            return [
                {
                    "id": r.id,
                    "parameter": r.parameter,
                    "old_value": r.old_value,
                    "proposed_value": r.proposed_value,
                    "reason": r.reason,
                    "evidence": list(r.evidence),
                    "confidence": r.confidence,
                    "expected_impact": r.expected_impact,
                    "risk": r.risk,
                    "status": r.status.value if hasattr(r.status, "value") else str(r.status),
                    "operator_decision": r.operator_decision,
                    "result": r.result,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in recs
            ]
        # 2. Fallback: no agent repo wired, return empty (advisor has no history yet)
        return []

    async def get_memory(self, query: str | None = None, limit: int = 20) -> dict[str, Any]:
        """Retrieve memory: experiences, lessons, and knowledge hits for ``query``.

        When ``query`` is None, returns the most recent items. When provided,
        performs a substring search across all three stores.
        """
        experiences: list[dict[str, Any]] = []
        lessons: list[dict[str, Any]] = []
        knowledge: list[dict[str, Any]] = []

        exp_repo = getattr(self._services, "agent_experiences", None)
        les_repo = getattr(self._services, "agent_lessons", None)
        kb_repo = getattr(self._services, "agent_knowledge", None)

        if exp_repo is not None:
            try:
                items = await exp_repo.search(query, limit=limit) if query else await exp_repo.list_recent(limit=limit)
                experiences = [
                    {
                        "id": e.id,
                        "situation": e.situation,
                        "observation": e.observation,
                        "decision": e.decision,
                        "result": e.result,
                        "lesson": e.lesson,
                        "confidence": e.confidence,
                        "source_id": e.source_id,
                    }
                    for e in items
                ]
            except Exception:
                pass

        if les_repo is not None:
            try:
                items = await les_repo.search(query, limit=limit) if query else await les_repo.list_recent(limit=limit)
                lessons = [
                    {
                        "id": l.id,
                        "title": l.title,
                        "content": l.content,
                        "pattern": l.pattern,
                        "confidence": l.confidence,
                    }
                    for l in items
                ]
            except Exception:
                pass

        if kb_repo is not None:
            try:
                items = await kb_repo.search(query, limit=limit) if query else await kb_repo.list_all()
                if query is None:
                    items = items[-limit:]  # most recent tail
                knowledge = [
                    {
                        "id": k.id,
                        "title": k.title,
                        "category": k.category.value if hasattr(k.category, "value") else str(k.category),
                        "summary": k.summary,
                        "source_id": k.source_id,
                    }
                    for k in items[:limit]
                ]
            except Exception:
                pass

        return {"experiences": experiences, "lessons": lessons, "knowledge": knowledge}

    # ------------------------------------------------------------------
    # block generic execution
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in _ALLOWED_TOOLS:
            raise ToolAccessBlocked(f"tool '{name}' is not in the read-only allowlist")
        raise AttributeError(name)
