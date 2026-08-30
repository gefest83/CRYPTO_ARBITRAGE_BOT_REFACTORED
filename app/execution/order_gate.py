"""Dynamic order-placement interlock.

The composition root injects exactly one placement policy into the adapter
layer; every ``create_order`` evaluates it *at call time*.

Policy:

* PAPER  — never (paper simulates fills locally, no real orders exist);
* DEMO   — only venues whose adapter actually runs a sandbox/testnet
           (fail-closed: no sandbox flag == no placement);
* LIVE   — only while the kill switch is released and trading is enabled
           (the LIVE mode itself already required the explicit three-flag
           opt-in at configuration load time).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = [
    "OrderGate",
    "live_session_gate",
    "never_place_orders",
    "sandbox_only",
]

#: Same shape as ``app.exchanges.base.OrderGate``, but typed loosely so this
#: policy module does not need to import the exchange layer.
OrderGate = Callable[[Any], bool]


def never_place_orders(adapter: Any) -> bool:
    """PAPER: paper trading simulates fills and must never place orders."""
    return False


def sandbox_only(adapter: Any) -> bool:
    """DEMO: venue-level fail-closed sandbox check."""
    options = getattr(adapter, "options", None)
    return bool(getattr(options, "sandbox", False))


def live_session_gate(guard: Any) -> OrderGate:
    """LIVE: kill switch released and trading flag enabled."""

    def placement_allowed(adapter: Any) -> bool:
        try:
            return bool(guard.trading_enabled) and not guard.is_halted
        except Exception:  # noqa: BLE001 - a broken gate must never open trading
            return False

    return placement_allowed
