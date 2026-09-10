"""Cost model: spread + fee + slippage/impact from depth."""

from __future__ import annotations

from decimal import Decimal

__all__ = ["CostModel", "estimate_costs"]

FEE_DIVISOR = Decimal("10000")


class CostModel:
    def __init__(
        self,
        fee_bps: int = 200,
        slippage_per_depth_bps: Decimal = Decimal("0"),
        min_depth: int = 0,
    ) -> None:
        self.fee_bps = fee_bps
        self.slippage_per_depth_bps = slippage_per_depth_bps
        self.min_depth = min_depth

    def fee_cost(self, price: Decimal) -> Decimal:
        # fee is bps of notional; price in [0,1] so fee ≈ price * fee_bps/10000
        return price * Decimal(self.fee_bps) / FEE_DIVISOR

    def spread_cost(self, spread: Decimal | None) -> Decimal:
        if spread is None:
            return Decimal("0")
        # half-spread paid on entry (assume take)
        return spread / Decimal("2")

    def slippage_cost(self, depth: int | None) -> Decimal:
        if depth is None or depth <= 0:
            return Decimal("0")
        if depth < self.min_depth:
            # insufficient liquidity — caller will flag, but cost is not zeroed here
            return Decimal("0")
        # simple: slippage decreases with depth; if per_depth bps provided, invert
        if self.slippage_per_depth_bps == Decimal("0"):
            return Decimal("0")
        # impact bps = slippage_per_depth_bps / depth
        return (self.slippage_per_depth_bps / Decimal(depth) / FEE_DIVISOR)

    def net_edge(self, gross: Decimal, spread: Decimal | None, price: Decimal, depth: int | None) -> Decimal:
        return gross - self.spread_cost(spread) - self.fee_cost(price) - self.slippage_cost(depth)


def estimate_costs(
    gross: Decimal,
    spread: Decimal | None,
    price: Decimal,
    depth: int | None,
    fee_bps: int = 200,
    slippage_per_depth_bps: Decimal = Decimal("0"),
    min_depth: int = 0,
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """Returns (spread_cost, fee_cost, slippage_cost, net)."""
    cm = CostModel(fee_bps, slippage_per_depth_bps, min_depth)
    sc = cm.spread_cost(spread)
    fc = cm.fee_cost(price)
    slc = cm.slippage_cost(depth)
    net = gross - sc - fc - slc
    return sc, fc, slc, net
