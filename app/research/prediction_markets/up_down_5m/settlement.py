"""Binance BTC Up/Down 5m — binary settlement rule (research-only).

Authoritative rules for THIS task (prediction market, NOT spot trading):

* UP wins iff Chainlink BTC/USDT price at market end > price at market start.
* DOWN wins iff end < start.
* Equal (end == start) settles 50/50 (PUSH — each side pays 0.5 per share).
* Resolution source is Chainlink BTC/USDT Top-of-Book mid-price.
* Final price is the close of the 5m candle immediately before market end.

Notes / deviations documented explicitly:
* Polymarket's public BTC Up/Down 5m copy resolves ties as UP (>=).
  Binance per this task resolves ties as PUSH (50/50) — we follow the task spec.
* PUSH payout of 0.5 per share is an assumption: "Equal = 50/50" is
  interpreted as each outstanding share (UP and DOWN) redeeming for 0.5 USDT.
  If the venue instead voids/refunds cost basis, adjust PAYOUT_PUSH only.
* Prices are Decimal only — never float. Comparison is exact Decimal compare.
* This module is pure (no I/O, no trading, no network).
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from app.models.base import DEC0, DEC1, DomainModel, as_decimal

__all__ = [
    "PAYOUT_PUSH",
    "SettlementOutcome",
    "Side",
    "payout_per_share",
    "settle_up_down_5m",
]

PAYOUT_PUSH = Decimal("0.5")


class SettlementOutcome(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    PUSH = "PUSH"  # end == start -> 50/50


class Side(StrEnum):
    UP = "UP"
    DOWN = "DOWN"


def settle_up_down_5m(start_price: Decimal | str | int, end_price: Decimal | str | int) -> SettlementOutcome:
    """Settle one 5m market from Chainlink ToB mid-prices.

    Args:
        start_price: Chainlink BTC/USDT ToB mid at market start.
        end_price: Close of the 5m candle immediately before market end
            (Chainlink BTC/USDT ToB mid).

    Returns:
        UP if end > start, DOWN if end < start, PUSH if equal.

    Raises:
        ValueError: if either price is missing/non-positive.
    """
    start = as_decimal(start_price, field="start_price")
    end = as_decimal(end_price, field="end_price")
    if start <= DEC0 or end <= DEC0:
        raise ValueError(f"Chainlink prices must be positive, got start={start!r} end={end!r}")
    if end > start:
        return SettlementOutcome.UP
    if end < start:
        return SettlementOutcome.DOWN
    return SettlementOutcome.PUSH


def payout_per_share(side: Side | str, outcome: SettlementOutcome | str) -> Decimal:
    """Payout in USDT per 1 share held to settlement.

    Winner gets 1.0, loser gets 0.0, PUSH pays 0.5 to both sides.
    """
    s = Side(str(side).upper())
    o = SettlementOutcome(str(outcome).upper())
    if o == SettlementOutcome.PUSH:
        return PAYOUT_PUSH
    if (s == Side.UP and o == SettlementOutcome.UP) or (s == Side.DOWN and o == SettlementOutcome.DOWN):
        return DEC1
    return DEC0


class SettlementInputs(DomainModel):
    """Explicit settlement inputs for auditability (research replay)."""

    market_id: int
    start_price: Decimal
    end_price: Decimal
    outcome: SettlementOutcome
