"""BTC Up/Down 5m DEMO trader — frozen directional signal (DEMO simulation only).

Frozen validated rules (no optimization here):

* At +90s: drift = (spot_now - spot_start) / spot_start * 10000 (bps).
  Spot inputs are Binance spot reference mids (1s close / bookTicker ToB),
  strictly pre-decision (closeTime rule). Chainlink is NEVER a signal input.
* drift >= +2 bps => BUY UP (DEMO simulation); drift <= -2 bps => BUY DOWN;
  otherwise HOLD (no entry).
* At +180s, same 2 bps rule on fresh pre-decision spot: opposite strong
  signal while holding => EXIT (flatten quickly); otherwise hold until
  settlement (winners never exit early).
* Settlement display uses venue Chainlink anchors ONLY
  (``settle_from_chainlink``); spot reference must never settle.

DEMO-only: simulated fills, no orders, no fund movement, no wallet mutation.
LIVE mode is refused fail-closed (``ensure_demo_only`` raises on LIVE).
This module never imports execution/risk/recovery/telegram/agent code and
never places orders.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import Field, field_validator

from app.models.base import DomainModel, as_decimal
from app.research.prediction_markets.up_down_5m.drift_signal import (
    DriftConfig,
    compute_drift_bps,
    decide_drift,
    manage_with_exit,
)
from app.research.prediction_markets.up_down_5m.live_snapshot import PriceProvenance
from app.research.prediction_markets.up_down_5m.settlement import SettlementOutcome
from app.research.prediction_markets.up_down_5m.signal import Action, Position, Signal

__all__ = [
    "DemoConfig",
    "DemoDecision",
    "DemoMode",
    "decide_demo_entry",
    "decide_demo_exit",
    "ensure_demo_only",
    "ensure_research_only",
    "render_demo_log",
    "run_demo_market",
]

FROZEN_ENTRY_OFFSET_MS = 90_000
FROZEN_EXIT_OFFSET_MS = 180_000
FROZEN_DRIFT_THR_BPS = Decimal("2.0")


class DemoMode(StrEnum):
    PAPER = "PAPER"  # offline simulation
    DEMO = "DEMO"  # venue testnet/demo simulation (still no live orders here)
    LIVE = "LIVE"  # refused fail-closed


class DemoConfig(DomainModel):
    """Frozen DEMO parameters (must match validated research)."""

    mode: DemoMode = DemoMode.DEMO
    entry_offset_ms: int = Field(default=FROZEN_ENTRY_OFFSET_MS, gt=0)
    exit_offset_ms: int = Field(default=FROZEN_EXIT_OFFSET_MS, gt=0)
    drift_thr_bps: Decimal = Decimal("2.0")
    stake_shares: Decimal = Decimal("1")

    @field_validator("drift_thr_bps", "stake_shares", mode="before")
    @classmethod
    def _dec(cls, v):  # type: ignore[no-untyped-def]
        return as_decimal(v, field="demo")


class DemoDecision(DomainModel):
    """One DEMO market lifecycle outcome (simulation, no orders)."""

    market_id: int
    mode: DemoMode
    drift_entry_bps: Decimal | None = None
    signal_entry: Signal = Signal.HOLD
    entry_action: Action = Action.HOLD_POSITION
    drift_exit_bps: Decimal | None = None
    signal_exit: Signal = Signal.HOLD
    exit_action: Action = Action.HOLD_POSITION
    settlement: SettlementOutcome | None = None
    result: str = "SKIP"  # ENTER_UP / ENTER_DOWN / SKIP / EXIT / WIN / LOSS / PUSH / PENDING

    @field_validator("drift_entry_bps", "drift_exit_bps", mode="before")
    @classmethod
    def _dec(cls, v):  # type: ignore[no-untyped-def]
        if v is None:
            return None
        return as_decimal(v, field="drift")


def ensure_demo_only(mode: DemoMode | str) -> None:
    """Fail-closed: DEMO/PAPER simulation only; LIVE is refused."""
    m = DemoMode(str(mode).upper())
    if m == DemoMode.LIVE:
        raise RuntimeError("up_down_5m DEMO refuses LIVE trading (fail-closed; live trading is disabled)")


def ensure_research_only() -> None:
    """Fail-closed guard — always raises; DEMO module must never trade live."""
    raise RuntimeError("up_down_5m DEMO is simulation-only; live trading is disabled (fail-closed)")


def _drift_cfg(cfg: DemoConfig) -> DriftConfig:
    return DriftConfig(
        entry_offset_ms=cfg.entry_offset_ms,
        drift_thr_bps=cfg.drift_thr_bps,
        exit_offset_ms=cfg.exit_offset_ms,
    )


def decide_demo_entry(
    *,
    market_id: int,
    market_start_ms: int,
    market_end_ms: int,
    spot_start: Decimal | str | int | float | None,
    spot_now: Decimal | str | int | float | None,
    provenance: PriceProvenance | str = PriceProvenance.SPOT_REFERENCE,
    config: DemoConfig | None = None,
) -> tuple[Signal, Decimal | None, Action]:
    """Frozen entry decision from SPOT reference only (never Chainlink).

    Raises:
        ValueError: if a Chainlink provenance is supplied as signal input.
    """
    cfg = config or DemoConfig()
    ensure_demo_only(cfg.mode)
    prov = PriceProvenance(str(provenance).lower())
    if prov != PriceProvenance.SPOT_REFERENCE:
        raise ValueError(
            f"refusing signal from provenance={prov.value}: "
            "entry signal accepts Binance spot reference only, never Chainlink"
        )
    dec = lambda v: as_decimal(v, field="spot") if v is not None else None  # noqa: E731
    ref = dec(spot_start)
    now = dec(spot_now)
    drift = compute_drift_bps(now, ref)
    res = decide_drift(
        decision_ts_ms=int(market_start_ms) + int(cfg.entry_offset_ms),
        market_start_ms=int(market_start_ms),
        market_end_ms=int(market_end_ms),
        ref_price=ref,
        mid_now=now,
        config=_drift_cfg(cfg),
    )
    action = manage_with_exit(Position.NONE, res.signal, Signal.HOLD)
    return res.signal, drift, action


def decide_demo_exit(
    *,
    market_id: int,
    market_start_ms: int,
    market_end_ms: int,
    position: Position | str,
    spot_start: Decimal | str | int | float | None,
    spot_now: Decimal | str | int | float | None,
    provenance: PriceProvenance | str = PriceProvenance.SPOT_REFERENCE,
    config: DemoConfig | None = None,
) -> tuple[Signal, Decimal | None, Action]:
    """Frozen exit check from SPOT reference only (never Chainlink)."""
    cfg = config or DemoConfig()
    ensure_demo_only(cfg.mode)
    prov = PriceProvenance(str(provenance).lower())
    if prov != PriceProvenance.SPOT_REFERENCE:
        raise ValueError(
            f"refusing signal from provenance={prov.value}: "
            "exit signal accepts Binance spot reference only, never Chainlink"
        )
    dec = lambda v: as_decimal(v, field="spot") if v is not None else None  # noqa: E731
    res = decide_drift(
        decision_ts_ms=int(market_start_ms) + int(cfg.exit_offset_ms),
        market_start_ms=int(market_start_ms),
        market_end_ms=int(market_end_ms),
        ref_price=dec(spot_start),
        mid_now=dec(spot_now),
        config=_drift_cfg(cfg),
    )
    pos = Position(str(position).upper())
    drift = compute_drift_bps(dec(spot_now), dec(spot_start))
    action = manage_with_exit(pos, Signal.HOLD, res.signal)
    return res.signal, drift, action


def run_demo_market(
    *,
    market_id: int,
    market_start_ms: int,
    market_end_ms: int,
    spot_start: Decimal | str | int | float | None,
    spot_entry: Decimal | str | int | float | None,
    spot_exit: Decimal | str | int | float | None,
    settlement: SettlementOutcome | str | None = None,
    config: DemoConfig | None = None,
) -> DemoDecision:
    """Simulate one full DEMO lifecycle: entry -> exit-check -> settlement.

    Settlement is display-only (venue Chainlink outcome supplied by caller,
    or None => PENDING). No orders, no funds move.
    """
    cfg = config or DemoConfig()
    ensure_demo_only(cfg.mode)
    sig_e, drift_e, act_e = decide_demo_entry(
        market_id=market_id, market_start_ms=market_start_ms, market_end_ms=market_end_ms,
        spot_start=spot_start, spot_now=spot_entry, config=cfg,
    )
    position = Position.NONE
    if act_e == Action.ENTER_UP:
        position = Position.LONG_UP
    elif act_e == Action.ENTER_DOWN:
        position = Position.LONG_DOWN
    if position == Position.NONE:
        out = SettlementOutcome(str(settlement).upper()) if settlement is not None else None
        return DemoDecision(
            market_id=int(market_id), mode=cfg.mode,
            drift_entry_bps=drift_e, signal_entry=sig_e, entry_action=act_e,
            drift_exit_bps=None, signal_exit=Signal.HOLD,
            exit_action=Action.HOLD_POSITION,
            settlement=out, result="SKIP",
        )
    sig_x, drift_x, act_x = decide_demo_exit(
        market_id=market_id, market_start_ms=market_start_ms, market_end_ms=market_end_ms,
        position=position, spot_start=spot_start, spot_now=spot_exit, config=cfg,
    )
    if act_x == Action.EXIT:
        out = SettlementOutcome(str(settlement).upper()) if settlement is not None else None
        return DemoDecision(
            market_id=int(market_id), mode=cfg.mode,
            drift_entry_bps=drift_e, signal_entry=sig_e, entry_action=act_e,
            drift_exit_bps=drift_x, signal_exit=sig_x, exit_action=act_x,
            settlement=out, result="EXIT",
        )
    if settlement is None:
        return DemoDecision(
            market_id=int(market_id), mode=cfg.mode,
            drift_entry_bps=drift_e, signal_entry=sig_e, entry_action=act_e,
            drift_exit_bps=drift_x, signal_exit=sig_x, exit_action=act_x,
            settlement=None, result="PENDING",
        )
    out = SettlementOutcome(str(settlement).upper())
    side = "UP" if position == Position.LONG_UP else "DOWN"
    if out == SettlementOutcome.PUSH:
        result = "PUSH"
    elif side == out.value:
        result = "WIN"
    else:
        result = "LOSS"
    return DemoDecision(
        market_id=int(market_id), mode=cfg.mode,
        drift_entry_bps=drift_e, signal_entry=sig_e, entry_action=act_e,
        drift_exit_bps=drift_x, signal_exit=sig_x, exit_action=act_x,
        settlement=out, result=result,
    )


def render_demo_log(d: DemoDecision) -> str:
    """Single structured DEMO log line (no secrets, no orders)."""
    de = f"{d.drift_entry_bps}" if d.drift_entry_bps is not None else "n/a"
    dx = f"{d.drift_exit_bps}" if d.drift_exit_bps is not None else "n/a"
    st = d.settlement.value if d.settlement is not None else "PENDING"
    return (
        f"DEMO up_down_5m market_id={d.market_id} mode={d.mode.value} "
        f"signal={d.signal_entry.value} drift_entry_bps={de} entry={d.entry_action.value} "
        f"exit_signal={d.signal_exit.value} drift_exit_bps={dx} exit={d.exit_action.value} "
        f"settlement={st} result={d.result} live_trading=disabled"
    )
