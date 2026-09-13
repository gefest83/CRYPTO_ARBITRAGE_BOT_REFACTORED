"""BTC Up/Down 5m PROFIT signal — spot-drift persistence (research-only).

Strategy (directional only, no mispricing, no profit-taking, no Chainlink signal):

* At ``t = market_start + entry_offset_ms`` (default +90s) compute pre-decision
  spot drift ``(mid_now - ref) / ref * 10000`` in bps, where ``ref`` is the
  Binance 1s close at/before market start and ``mid_now`` the close at/before
  ``t`` (closeTime rule — strictly no look-ahead).
* ``drift >= +thr``  => Signal UP   => immediately BUY UP at the venue UP
  chance/100 observed at/before ``t`` (real contract price, pre-decision).
* ``drift <= -thr``  => Signal DOWN => immediately BUY DOWN at ``1 - UP``.
* Otherwise HOLD (no entry).
* Position policy: flat + UP/DOWN => ENTER; holding + opposite strong drift
  at ``exit_offset_ms`` (default +180s, same ``thr``) => EXIT at the
  then-current contract price (salvage, minimizes loss); same/HOLD => hold
  until settlement and collect the $1 payout (never exit winners early).
* Settlement is the venue-resolved UP/DOWN (Predict.fun WON/LOST +
  venue-recorded Chainlink anchors). Chainlink is NEVER a signal input —
  only the settlement label. Contract price is NEVER compared to fair value
  (no mispricing); it is only the execution cost and exit salvage.

Defaults (``DriftConfig``): entry +90s, thr 2.0 bps, exit +180s, fee 200 bps,
slippage 0.005/share. Selected as the simplest robust plateau:
single feature, two timings, one threshold. Research report:
docs/up_down_5m_profit_report.md (49 settleable markets: 48 decisive + 1 PUSH).

Research-only: pure functions, no I/O, no trading, no Chainlink import.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field, field_validator

from app.models.base import DEC0, DomainModel, as_decimal
from app.research.prediction_markets.up_down_5m.signal import Action, Position, Signal

__all__ = [
    "DriftConfig",
    "DriftResult",
    "backtest_drift",
    "compute_drift_bps",
    "decide_drift",
    "ensure_research_only",
    "exit_net",
    "hold_net",
    "manage_with_exit",
]

_BPS = Decimal("10000")
_FEE_DIV = Decimal("10000")


class DriftConfig(DomainModel):
    """Exact parameters (frozen)."""

    entry_offset_ms: int = Field(default=90_000, gt=0)
    drift_thr_bps: Decimal = Decimal("2.0")
    exit_offset_ms: int = Field(default=180_000, gt=0)
    fee_bps: int = Field(default=200, ge=0)
    slippage_per_share: Decimal = Decimal("0.005")
    min_elapsed_ms: int = Field(default=60_000, ge=0)
    min_remaining_ms: int = Field(default=120_000, ge=0)

    @field_validator("drift_thr_bps", "slippage_per_share", mode="before")
    @classmethod
    def _dec(cls, v):  # type: ignore[no-untyped-def]
        return as_decimal(v, field="threshold")


class DriftResult(DomainModel):
    signal: Signal
    drift_bps: Decimal | None = None
    detail: str = ""


def compute_drift_bps(mid_now: Decimal | None, ref: Decimal | None) -> Decimal | None:
    """Spot drift in bps (None when inputs missing/non-positive)."""
    if mid_now is None or ref is None:
        return None
    if ref <= DEC0 or mid_now <= DEC0:
        return None
    return (mid_now - ref) / ref * _BPS


def decide_drift(
    *,
    decision_ts_ms: int,
    market_start_ms: int,
    market_end_ms: int,
    ref_price: Decimal | str | int | None,
    mid_now: Decimal | str | int | None,
    config: DriftConfig | None = None,
) -> DriftResult:
    """Deterministic UP/DOWN/HOLD from pre-decision spot drift only."""
    cfg = config or DriftConfig()
    dec = lambda v: as_decimal(v, field="feature") if v is not None else None  # noqa: E731
    ref = dec(ref_price)
    now = dec(mid_now)
    elapsed = int(decision_ts_ms) - int(market_start_ms)
    remaining = int(market_end_ms) - int(decision_ts_ms)
    if ref is None or now is None:
        return DriftResult(signal=Signal.HOLD, drift_bps=None, detail="no-ref")
    if elapsed < cfg.min_elapsed_ms:
        return DriftResult(signal=Signal.HOLD, drift_bps=None, detail="warming")
    if remaining < cfg.min_remaining_ms:
        return DriftResult(signal=Signal.HOLD, drift_bps=None, detail="too-late")
    d = compute_drift_bps(now, ref)
    if d is None:
        return DriftResult(signal=Signal.HOLD, drift_bps=None, detail="bad-price")
    thr = cfg.drift_thr_bps
    if d >= thr:
        return DriftResult(signal=Signal.UP, drift_bps=d, detail=f"drift={d}>=+{thr}")
    if d <= -thr:
        return DriftResult(signal=Signal.DOWN, drift_bps=d, detail=f"drift={d}<=-{thr}")
    return DriftResult(signal=Signal.HOLD, drift_bps=d, detail=f"weak|{d}|<{thr}")


def manage_with_exit(
    position: Position | str,
    entry_signal: Signal | str,
    exit_signal: Signal | str | None = None,
) -> Action:
    """Entry now; fast EXIT only on opposite strong signal; else hold.

    * flat + UP/DOWN => ENTER_UP/ENTER_DOWN (buy immediately);
    * holding + opposite non-HOLD signal => EXIT (flatten quickly);
    * otherwise HOLD_POSITION (never exit winners early).
    """
    pos = Position(str(position).upper())
    ent = Signal(str(entry_signal).upper())
    ext = Signal(str(exit_signal).upper()) if exit_signal is not None else Signal.HOLD
    if pos == Position.NONE:
        if ent == Signal.UP:
            return Action.ENTER_UP
        if ent == Signal.DOWN:
            return Action.ENTER_DOWN
        return Action.HOLD_POSITION
    if pos == Position.LONG_UP:
        return Action.EXIT if ext == Signal.DOWN else Action.HOLD_POSITION
    return Action.EXIT if ext == Signal.UP else Action.HOLD_POSITION


def _cost(side: str, up_price: Decimal, cfg: DriftConfig) -> Decimal:
    base = up_price if str(side).upper() == "UP" else (Decimal("1") - up_price)
    return base + cfg.slippage_per_share


def _fee(amount: Decimal, cfg: DriftConfig) -> Decimal:
    return amount * Decimal(cfg.fee_bps) / _FEE_DIV


def hold_net(side: str, up_price: Decimal, outcome: str, cfg: DriftConfig | None = None) -> Decimal:
    """Hold-to-settlement net per 1 share (fee + slippage included)."""
    c = cfg or DriftConfig()
    cost = _cost(side, up_price, c)
    fee = _fee(cost, c)
    o = str(outcome).upper()
    s = str(side).upper()
    if o == "PUSH":
        payout = Decimal("0.5")
    elif s == o:
        payout = Decimal("1")
    else:
        payout = DEC0
    return payout - cost - fee


def exit_net(
    side: str,
    entry_up: Decimal,
    exit_up: Decimal,
    cfg: DriftConfig | None = None,
) -> Decimal:
    """Reversal-exit net: salvage at exit price minus entry cost and both fees."""
    c = cfg or DriftConfig()
    s = str(side).upper()
    cost = _cost(s, entry_up, c)
    salvage = exit_up if s == "UP" else (Decimal("1") - exit_up)
    return salvage - cost - _fee(cost, c) - _fee(salvage, c) - c.slippage_per_share


def backtest_drift(markets, config: DriftConfig | None = None):  # type: ignore[no-untyped-def]
    """Chronological hold-to-settlement backtest over ValidatedMarket list.

    Returns dict with trades, coverage, wins, win_rate, total_net, avg_net,
    avg_win, avg_loss, expectancy, max_drawdown, per-market nets. No I/O.
    """
    from app.research.prediction_markets.up_down_5m.validation_dataset import (
        build_signal_features,
        contract_price_at,
    )

    cfg = config or DriftConfig()
    nets: list[Decimal] = []
    trades = 0
    wins = 0
    gross_win = Decimal("0")
    gross_loss = Decimal("0")
    n_win = 0
    n_loss = 0
    for m in sorted(markets, key=lambda x: x.start_ts_ms):
        t = m.start_ts_ms + cfg.entry_offset_ms
        f = build_signal_features(m, t)
        r = decide_drift(
            decision_ts_ms=t, market_start_ms=m.start_ts_ms, market_end_ms=m.end_ts_ms,
            ref_price=f.ref_price, mid_now=f.mid_now, config=cfg,
        )
        # exit check (reversal at exit_offset) — scored as salvage when triggered
        up0 = contract_price_at(m.contract_series, t)
        if r.signal == Signal.HOLD or up0 is None:
            nets.append(DEC0)
            continue
        side = r.signal.value
        t1 = m.start_ts_ms + cfg.exit_offset_ms
        f1 = build_signal_features(m, t1)
        r1 = decide_drift(
            decision_ts_ms=t1, market_start_ms=m.start_ts_ms, market_end_ms=m.end_ts_ms,
            ref_price=f1.ref_price, mid_now=f1.mid_now, config=cfg,
        )
        up1 = contract_price_at(m.contract_series, t1)
        opposed = (r1.signal != Signal.HOLD and r1.signal != r.signal and up1 is not None)
        if opposed:
            assert up1 is not None
            n = exit_net(side, up0, up1, cfg)
        else:
            n = hold_net(side, up0, m.outcome.value, cfg)
        nets.append(n)
        trades += 1
        if n > DEC0:
            wins += 1
            gross_win += n
            n_win += 1
        elif n < DEC0:
            gross_loss += -n
            n_loss += 1
    total = sum(nets, DEC0)
    cum = DEC0
    peak = DEC0
    max_dd = DEC0
    for n in nets:
        cum += n
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    return {
        "markets": len(markets),
        "trades": trades,
        "coverage": (trades / len(markets)) if markets else 0.0,
        "wins": wins,
        "win_rate": (wins / trades) if trades else None,
        "total_net": total,
        "avg_net": (total / Decimal(trades)) if trades else None,
        "avg_win": (gross_win / Decimal(n_win)) if n_win else None,
        "avg_loss": (-gross_loss / Decimal(n_loss)) if n_loss else None,
        "expectancy": (total / Decimal(trades)) if trades else None,
        "max_drawdown": max_dd,
        "nets": tuple(nets),
    }


def ensure_research_only() -> None:
    """Fail-closed guard — always raises; profit signal must never trade live."""
    raise RuntimeError("up_down_5m drift signal is research-only; live trading is disabled (fail-closed)")
