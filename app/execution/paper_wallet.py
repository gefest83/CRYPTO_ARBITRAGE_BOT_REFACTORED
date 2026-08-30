"""In-memory paper wallet.

PAPER mode needs somewhere for simulated funds to live: initialised from the
deterministic simulated adapters, then adjusted by every simulated trade so
repeated execution cannot spend the same money twice.  Not persisted across
restarts by design (PAPER funds are synthetic); LIVE/DEMO capital lives on
the exchanges themselves.
"""

from __future__ import annotations

from decimal import Decimal

from app.models.balance import BalanceSnapshot
from app.models.base import DEC0

__all__ = ["PaperWallet"]


class PaperWallet:
    """Tracks free balances by asset; debits fail closed."""

    def __init__(self, balances: dict[str, Decimal] | None = None) -> None:
        self._free: dict[str, Decimal] = dict(balances or {})

    @classmethod
    def from_snapshot(cls, snapshot: BalanceSnapshot) -> PaperWallet:
        return cls({b.asset: b.free for b in snapshot.balances})

    def free(self, asset: str) -> Decimal:
        return self._free.get(asset.strip().upper(), DEC0)

    def credit(self, asset: str, amount: Decimal) -> None:
        if amount <= DEC0:
            return
        key = asset.strip().upper()
        self._free[key] = self._free.get(key, DEC0) + amount

    def debit(self, asset: str, amount: Decimal) -> None:
        """Debit ``amount``; raises when funds are insufficient (fail closed)."""
        if amount <= DEC0:
            return
        key = asset.strip().upper()
        available = self._free.get(key, DEC0)
        if available < amount:
            raise ValueError(
                f"insufficient paper balance for {key}: need {amount}, have {available}"
            )
        self._free[key] = available - amount

    def snapshot(self) -> dict[str, Decimal]:
        return dict(self._free)
