"""Research-only market replay for BTC Up/Down 5m (no live trading).

Evaluates entry decisions against the ACTUAL binary settlement outcome
(``settle_up_down_5m``), not against spot-proxy repricing or mid-price drift.

No look-ahead: each decision's ``entry_ts_ms`` must satisfy
``start_ts <= entry_ts < end_ts``; settlement uses only
``start_price`` vs ``end_price`` (Chainlink anchors).

PnL per filled decision (paper shares, USDT):
  cost   = entry_price * size  (+ fee = cost * fee_bps / 10000)
  payout = payout_per_share(side, outcome) * size
  net    = payout - cost - fee
SKIP decisions produce net 0 and do not affect win-rate denominator
(``traded`` counts only UP/DOWN fills).

This module never imports execution/risk/wallet code and never places orders.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field, field_validator

from app.models.base import DEC0, DomainModel, as_decimal
from app.research.prediction_markets.up_down_5m.market import UpDown5mMarket
from app.research.prediction_markets.up_down_5m.settlement import (
    SettlementOutcome,
    Side,
    payout_per_share,
    settle_up_down_5m,
)

__all__ = [
    "ReplayDecision",
    "ReplayFill",
    "ReplaySummary",
    "UpDown5mReplay",
]

_FEE_DIVISOR = Decimal("10000")


class ReplayDecision(DomainModel):
    """One research entry decision for a single 5m market."""

    market_id: int = Field(gt=0)
    side: str  # UP / DOWN / SKIP
    entry_price: Decimal  # contract price paid per share, [0,1]
    size: Decimal = Decimal("1")  # paper shares
    entry_ts_ms: int = Field(ge=0)

    @field_validator("side")
    @classmethod
    def _check_side(cls, v: str) -> str:
        u = v.strip().upper()
        if u not in ("UP", "DOWN", "SKIP"):
            raise ValueError(f"side must be UP/DOWN/SKIP, got {v!r}")
        return u

    @field_validator("entry_price", "size", mode="before")
    @classmethod
    def _dec(cls, v):  # type: ignore[no-untyped-def]
        return as_decimal(v, field="decimal")


class ReplayFill(DomainModel):
    """Settled outcome of one decision (deterministic, auditable)."""

    market_id: int
    side: str
    outcome: SettlementOutcome
    entry_price: Decimal
    size: Decimal
    payout_per_share: Decimal
    cost: Decimal
    fee: Decimal
    payout: Decimal
    net: Decimal
    win: bool | None  # None for SKIP


class ReplaySummary(DomainModel):
    markets: int
    traded: int
    skipped: int
    wins: int
    win_rate: float | None
    total_net: Decimal
    avg_net: Decimal | None
    fills: tuple[ReplayFill, ...]


class UpDown5mReplay(DomainModel):
    """Stateless replay engine (frozen config). Research-only."""

    fee_bps: int = Field(default=200, ge=0)

    def run(
        self,
        markets: list[UpDown5mMarket] | tuple[UpDown5mMarket, ...],
        decisions: list[ReplayDecision] | tuple[ReplayDecision, ...],
    ) -> ReplaySummary:
        by_market: dict[int, UpDown5mMarket] = {}
        for m in markets:
            if m.market_id in by_market:
                raise ValueError(f"duplicate market_id={m.market_id}")
            by_market[m.market_id] = m
        by_decision: dict[int, ReplayDecision] = {}
        for d in decisions:
            if d.market_id in by_decision:
                raise ValueError(f"duplicate decision for market_id={d.market_id}")
            by_decision[d.market_id] = d

        fills: list[ReplayFill] = []
        for mid, market in sorted(by_market.items()):
            dec = by_decision.get(mid)
            if dec is None:
                continue  # no decision -> not scored
            if market.start_price is None or market.end_price is None:
                raise ValueError(f"market_id={mid} missing Chainlink start/end price for settlement")
            outcome = settle_up_down_5m(market.start_price, market.end_price)
            # No look-ahead / window enforcement (research integrity):
            if dec.side != "SKIP":
                if not (market.start_ts_ms <= dec.entry_ts_ms < market.end_ts_ms):
                    raise ValueError(
                        f"market_id={mid} entry_ts={dec.entry_ts_ms} outside OPEN window "
                        f"[{market.start_ts_ms},{market.end_ts_ms})"
                    )
                if dec.entry_price < DEC0 or dec.entry_price > Decimal("1"):
                    raise ValueError(f"market_id={mid} entry_price must be in [0,1]")
                if dec.size <= DEC0:
                    raise ValueError(f"market_id={mid} size must be positive to trade")
            fills.append(self._settle_fill(dec, outcome))

        traded = [f for f in fills if f.side != "SKIP"]
        skipped = [f for f in fills if f.side == "SKIP"]
        wins = [f for f in traded if f.win is True]
        total_net = sum((f.net for f in fills), DEC0)
        avg_net = (total_net / Decimal(len(traded))) if traded else None
        win_rate = (len(wins) / len(traded)) if traded else None
        return ReplaySummary(
            markets=len(by_market),
            traded=len(traded),
            skipped=len(skipped),
            wins=len(wins),
            win_rate=win_rate,
            total_net=total_net,
            avg_net=avg_net,
            fills=tuple(fills),
        )

    def _settle_fill(self, dec: ReplayDecision, outcome: SettlementOutcome) -> ReplayFill:
        if dec.side == "SKIP":
            return ReplayFill(
                market_id=dec.market_id,
                side="SKIP",
                outcome=outcome,
                entry_price=dec.entry_price,
                size=dec.size,
                payout_per_share=DEC0,
                cost=DEC0,
                fee=DEC0,
                payout=DEC0,
                net=DEC0,
                win=None,
            )
        pps = payout_per_share(Side(dec.side), outcome)
        cost = dec.entry_price * dec.size
        fee = cost * Decimal(self.fee_bps) / _FEE_DIVISOR
        payout = pps * dec.size
        net = payout - cost - fee
        won = net > DEC0
        return ReplayFill(
            market_id=dec.market_id,
            side=dec.side,
            outcome=outcome,
            entry_price=dec.entry_price,
            size=dec.size,
            payout_per_share=pps,
            cost=cost,
            fee=fee,
            payout=payout,
            net=net,
            win=won,
        )

    def ensure_research_only(self) -> None:
        """Fail-closed guard — always raises; replay must never trade live."""
        raise RuntimeError("up_down_5m replay is research-only; live trading is disabled (fail-closed)")
