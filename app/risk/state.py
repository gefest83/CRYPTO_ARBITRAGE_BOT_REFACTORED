"""Runtime risk state.

The risk rules need facts that live outside a single opportunity: is the kill
switch engaged, how many transfers are open, what is today's P&L.  This
tracker owns those facts so the runtime can build a complete
:class:`~app.risk.rules.RiskContext` instead of silently defaulting values
(which would disable the rules).

Restart-safety: ``daily_pnl`` and ``open_transfers`` are rebuilt from the
database on startup (trades of today + open transfer records).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from app.models.base import DEC0, utc_now
from app.models.transfer import TransferRecord

__all__ = ["RiskEnvironment", "RiskStateTracker"]


@dataclass(frozen=True, slots=True)
class RiskEnvironment:
    """Immutable snapshot of everything the rules need beyond the trade."""

    kill_switch_engaged: bool = False
    open_transfers: int = 0
    daily_pnl: Decimal = DEC0
    exchange_exposure: Mapping[str, Decimal] = field(default_factory=dict)
    asset_exposure: Mapping[str, Decimal] = field(default_factory=dict)


@dataclass(slots=True)
class RiskStateTracker:
    """Mutable counters maintained by the runtime; execution updates them."""

    open_transfers: int = 0
    daily_pnl: Decimal = DEC0
    _day: date = field(default_factory=lambda: utc_now().date())

    def sync_from_storage(
        self, *, daily_pnl: Decimal, open_transfers: Sequence[TransferRecord]
    ) -> None:
        """Rebuild counters from persisted state (used on startup)."""
        self.daily_pnl = daily_pnl
        self.open_transfers = len(open_transfers)

    def register_realized_pnl(self, amount: Decimal) -> None:
        self._roll_day()
        self.daily_pnl += amount

    def register_transfer_opened(self) -> None:
        self.open_transfers += 1

    def register_transfer_closed(self, *, realized_pnl: Decimal = DEC0) -> None:
        self.open_transfers = max(0, self.open_transfers - 1)
        self.register_realized_pnl(realized_pnl)

    def snapshot(
        self,
        *,
        kill_switch_engaged: bool = False,
        exchange_exposure: Mapping[str, Decimal] | None = None,
        asset_exposure: Mapping[str, Decimal] | None = None,
    ) -> RiskEnvironment:
        self._roll_day()
        return RiskEnvironment(
            kill_switch_engaged=kill_switch_engaged,
            open_transfers=self.open_transfers,
            daily_pnl=self.daily_pnl,
            exchange_exposure=dict(exchange_exposure or {}),
            asset_exposure=dict(asset_exposure or {}),
        )

    def _roll_day(self) -> None:
        today = utc_now().date()
        if today != self._day:
            self._day = today
            self.daily_pnl = DEC0
