"""Cost / edge math — pure Decimal, no I/O."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.models.base import DEC0

__all__ = ["CostInputs", "compute_cost_bps", "compute_gross_bps", "compute_net_bps"]

_BPS = Decimal("10000")
_QUANT = Decimal("0.0001")


@dataclass(frozen=True, slots=True)
class CostInputs:
    taker_fee_bps: Decimal = Decimal("10")
    spread_bps: Decimal = Decimal("5")
    slippage_bps: Decimal = Decimal("5")
    cost_model: str = "one_way"  # or round_trip


def compute_gross_bps(last_close: Decimal, pred_close: Decimal) -> Decimal:
    """Gross predicted move in bps: (pred - last)/last * 10000."""
    if last_close <= DEC0 or pred_close <= DEC0:
        return DEC0
    return ((pred_close - last_close) / last_close * _BPS).quantize(_QUANT)


def compute_cost_bps(inputs: CostInputs) -> Decimal:
    """Total cost in bps for the configured model."""
    if inputs.cost_model == "round_trip":
        return (inputs.taker_fee_bps * 2 + inputs.spread_bps + inputs.slippage_bps * 2).quantize(_QUANT)
    # one_way: taker + half spread + slippage
    return (inputs.taker_fee_bps + inputs.spread_bps / 2 + inputs.slippage_bps).quantize(_QUANT)


def compute_net_bps(gross_bps: Decimal, cost_bps: Decimal) -> Decimal:
    """Net edge after costs, never flips sign.

    Costs reduce the magnitude toward zero and clamp at zero so an
    unprofitable long (cost > gross) does not become a synthetic short
    (and vice versa).  ``HOLD`` is then produced by the threshold check.
    """
    if gross_bps > DEC0:
        net = gross_bps - cost_bps
        if net < DEC0:
            net = DEC0
        return net.quantize(_QUANT)
    if gross_bps < DEC0:
        net = gross_bps + cost_bps
        if net > DEC0:
            net = DEC0
        return net.quantize(_QUANT)
    return DEC0
