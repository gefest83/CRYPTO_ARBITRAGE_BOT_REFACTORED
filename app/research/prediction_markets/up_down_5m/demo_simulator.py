"""BTC Up/Down 5m DEMO trade simulator — real mechanics, simulated fills.

Frozen signal is NOT reimplemented here: entry/exit decisions come from
:mod:`app.research.prediction_markets.up_down_5m.demo` (frozen +90s/±2bps,
+180s opposite-strong EXIT). This module prices those decisions the way the
venue would fill them:

* entry fill: BUY UP pays the UP ask; BUY DOWN pays ``1 - UP bid``
  (DOWN ask implied from the UP book; venue is not neg-risk).
* EXIT fill: sell UP receives the UP bid; sell DOWN receives ``1 - UP ask``.
* settlement payout per share: winner 1.0, loser 0.0, PUSH 0.5.
* fees: ``fee_bps`` (venue ``feeRateBps``, default 200) on entry cost and on
  EXIT proceeds.
* slippage: ``slippage_per_share`` added to entry cost and subtracted from
  EXIT proceeds (validated-research default 0.005/share), on top of the
  crossed bid/ask spread.
* position size: fixed ``stake_shares`` per entered market (default 10).

Accounting per trade (all per ``stake_shares`` shares):

* staked   = (fill_price + slippage) * shares + entry_fee
* held WIN:  payout = shares (or 0.5*shares on PUSH); net = payout - staked
* EXIT:      salvage = (exit_px - slippage) * shares - exit_fee;
  net = salvage - staked
* gross    = payout/salvage - fill_cost (ex-fee, ex-slippage);
  net      = gross - fees - slippage legs (reported separately).

Cumulative: total gross/net, average net/trade, max drawdown on the realized
net equity curve, ROI = total_net / total_staked.

Simulation-only: no orders, no fund movement. LIVE refused fail-closed.
Never imports execution/risk/recovery/telegram/agent; never touches Chainlink
(the outcome label is supplied by the caller, exactly as the DEMO trader does).
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field, field_validator

from app.models.base import DEC0, DomainModel, as_decimal
from app.research.prediction_markets.up_down_5m.demo import DemoMode, ensure_demo_only
from app.research.prediction_markets.up_down_5m.settlement import SettlementOutcome

__all__ = [
    "DemoFill",
    "DemoSimConfig",
    "SimTrade",
    "SimSummary",
    "ensure_research_only",
    "simulate_trade",
    "summarize",
]


class DemoSimConfig(DomainModel):
    """Simulator parameters (mechanics only; signal stays frozen)."""

    mode: DemoMode = DemoMode.DEMO
    stake_shares: Decimal = Decimal("10")
    fee_bps: int = Field(default=200, ge=0)
    slippage_per_share: Decimal = Decimal("0.005")

    @field_validator("stake_shares", "slippage_per_share", mode="before")
    @classmethod
    def _dec(cls, v):  # type: ignore[no-untyped-def]
        return as_decimal(v, field="sim")


class DemoFill(DomainModel):
    """Real venue contract prices observed read-only at decision instants."""

    market_id: int
    up_bid_entry: Decimal
    up_ask_entry: Decimal
    up_bid_exit: Decimal | None = None
    up_ask_exit: Decimal | None = None
    fee_bps: int = Field(default=200, ge=0)

    @field_validator("up_bid_entry", "up_ask_entry", "up_bid_exit", "up_ask_exit", mode="before")
    @classmethod
    def _dec(cls, v):  # type: ignore[no-untyped-def]
        if v is None:
            return None
        return as_decimal(v, field="price")


class SimTrade(DomainModel):
    """Settled economics of one simulated market lifecycle."""

    market_id: int
    side: str  # UP / DOWN / SKIP
    entry_price: Decimal
    exit_price: Decimal | None = None  # salvage bid (None when held)
    shares: Decimal
    settlement: SettlementOutcome | None = None
    result: str = "SKIP"  # WIN / LOSS / PUSH / EXIT / SKIP / PENDING
    gross: Decimal = DEC0
    fees: Decimal = DEC0
    slippage: Decimal = DEC0
    net: Decimal = DEC0
    staked: Decimal = DEC0


class SimSummary(DomainModel):
    trades: int = 0
    wins: int = 0
    losses: int = 0
    exits: int = 0
    pushes: int = 0
    skips: int = 0
    gross: Decimal = DEC0
    fees: Decimal = DEC0
    slippage: Decimal = DEC0
    total_net: Decimal = DEC0
    avg_net: Decimal | None = None
    total_staked: Decimal = DEC0
    roi: Decimal | None = None
    max_drawdown: Decimal = DEC0


def _fee(amount: Decimal, fee_bps: int) -> Decimal:
    return amount * Decimal(fee_bps) / Decimal("10000")


def simulate_trade(
    *,
    market_id: int,
    side: str,
    fill: DemoFill | None,
    settlement: SettlementOutcome | str | None,
    exited: bool = False,
    config: DemoSimConfig | None = None,
) -> SimTrade:
    """Price one lifecycle. ``side`` UP/DOWN/SKIP; ``exited`` = EXIT filled."""
    cfg = config or DemoSimConfig()
    ensure_demo_only(cfg.mode)
    s = str(side).upper()
    if s == "SKIP" or fill is None:
        return SimTrade(market_id=int(market_id), side="SKIP",
                        entry_price=DEC0, shares=DEC0, settlement=None, result="SKIP")
    n = cfg.stake_shares
    slip = cfg.slippage_per_share
    fee_bps = fill.fee_bps
    if s == "UP":
        fill_px = fill.up_ask_entry
        salvage_px = fill.up_bid_exit if exited else None
    elif s == "DOWN":
        fill_px = Decimal("1") - fill.up_bid_entry
        salvage_px = (Decimal("1") - fill.up_ask_exit) if (exited and fill.up_ask_exit is not None) else None
    else:
        raise ValueError(f"side must be UP/DOWN/SKIP, got {side!r}")
    cost = fill_px * n
    slip_cost = slip * n
    fee_in = _fee(cost + slip_cost, fee_bps)
    staked = cost + slip_cost + fee_in
    if exited:
        if salvage_px is None:
            raise ValueError("exited=True requires exit-side book prices")
        proceeds = salvage_px * n
        slip_out = slip * n
        fee_out = _fee(proceeds, fee_bps)
        salvage = proceeds - slip_out - fee_out
        gross = proceeds - cost
        fees = fee_in + fee_out
        slippage = slip_cost + slip_out
        net = salvage - staked
        out = SettlementOutcome(str(settlement).upper()) if settlement is not None else None
        return SimTrade(
            market_id=int(market_id), side=s, entry_price=fill_px, exit_price=salvage_px,
            shares=n, settlement=out, result="EXIT",
            gross=gross, fees=fees, slippage=slippage, net=net, staked=staked,
        )
    if settlement is None:
        return SimTrade(
            market_id=int(market_id), side=s, entry_price=fill_px, shares=n,
            settlement=None, result="PENDING",
            gross=-cost, fees=fee_in, slippage=slip_cost, net=-staked, staked=staked,
        )
    out = SettlementOutcome(str(settlement).upper())
    if out == SettlementOutcome.PUSH:
        payout = Decimal("0.5") * n
        result = "PUSH"
    elif s == out.value:
        payout = Decimal("1") * n
        result = "WIN"
    else:
        payout = DEC0
        result = "LOSS"
    gross = payout - cost
    return SimTrade(
        market_id=int(market_id), side=s, entry_price=fill_px, shares=n,
        settlement=out, result=result,
        gross=gross, fees=fee_in, slippage=slip_cost,
        net=payout - staked, staked=staked,
    )


def summarize(trades: list[SimTrade] | tuple[SimTrade, ...]) -> SimSummary:
    """Cumulative PnL + drawdown over realized net equity curve (SKIP excluded)."""
    scored = [t for t in trades if t.side != "SKIP" and t.result != "PENDING"]
    gross = sum((t.gross for t in scored), DEC0)
    fees = sum((t.fees for t in scored), DEC0)
    slip = sum((t.slippage for t in scored), DEC0)
    net = sum((t.net for t in scored), DEC0)
    staked = sum((t.staked for t in scored), DEC0)
    cum = DEC0
    peak = DEC0
    dd = DEC0
    for t in trades:
        if t.side == "SKIP" or t.result == "PENDING":
            continue
        cum += t.net
        if cum > peak:
            peak = cum
        gap = peak - cum
        if gap > dd:
            dd = gap
    return SimSummary(
        trades=len(scored),
        wins=sum(1 for t in scored if t.result == "WIN"),
        losses=sum(1 for t in scored if t.result == "LOSS"),
        exits=sum(1 for t in scored if t.result == "EXIT"),
        pushes=sum(1 for t in scored if t.result == "PUSH"),
        skips=sum(1 for t in trades if t.side == "SKIP"),
        gross=gross, fees=fees, slippage=slip, total_net=net,
        avg_net=(net / Decimal(len(scored))) if scored else None,
        total_staked=staked,
        roi=(net / staked) if staked > DEC0 else None,
        max_drawdown=dd,
    )


def ensure_research_only() -> None:
    """Fail-closed guard — always raises; simulator must never trade live."""
    raise RuntimeError("up_down_5m demo simulator is simulation-only; live trading is disabled (fail-closed)")
