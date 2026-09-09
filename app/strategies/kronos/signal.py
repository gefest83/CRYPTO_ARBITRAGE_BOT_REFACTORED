"""Deterministic signal classification — pure function, no I/O."""

from __future__ import annotations

from decimal import Decimal

from app.models.base import DEC0
from app.strategies.kronos.types import Signal

__all__ = ["classify_signal"]

_QUANT = Decimal("0.0001")


def classify_signal(
    *,
    net_bps_1m: Decimal,
    min_edge_bps: Decimal,
    require_5m_confirmation: bool = False,
    net_bps_5m: Decimal | None = None,
) -> tuple[Signal, str]:
    """Map net bps -> BUY/SELL/HOLD deterministically.

    - BUY iff net_bps_1m > min_edge_bps
    - SELL iff net_bps_1m < -min_edge_bps
    - else HOLD
    If 5m confirmation is required, the 5m sign must agree; otherwise HOLD
    with reason containing '5m_confirmation'.
    Threshold is strict > / < (not >=) to avoid flip on equality.
    """
    if min_edge_bps < DEC0:
        min_edge_bps = DEC0

    if net_bps_1m > min_edge_bps:
        candidate = Signal.BUY
    elif net_bps_1m < -min_edge_bps:
        candidate = Signal.SELL
    else:
        return Signal.HOLD, f"below_threshold net={net_bps_1m} min_edge={min_edge_bps}"

    if not require_5m_confirmation:
        return candidate, f"{candidate.lower()}_net={net_bps_1m}_bps"

    if net_bps_5m is None:
        return Signal.HOLD, "5m_confirmation_missing"

    # 5m must have same direction and also clear threshold
    if candidate == Signal.BUY and net_bps_5m > min_edge_bps:
        return Signal.BUY, f"buy_confirmed_1m={net_bps_1m}_5m={net_bps_5m}"
    if candidate == Signal.SELL and net_bps_5m < -min_edge_bps:
        return Signal.SELL, f"sell_confirmed_1m={net_bps_1m}_5m={net_bps_5m}"
    return Signal.HOLD, f"5m_confirmation_failed_1m={net_bps_1m}_5m={net_bps_5m}"
