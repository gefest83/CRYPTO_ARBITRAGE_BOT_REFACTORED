"""ETH Up/Down 5m EARLY-ENTRY flow signal — taker-imbalance at +20s (research-only).

ETH-specific companion to the frozen BTC ``early_flow_signal``. The BTC
strategy and its tests are untouched; all entry/exit logic here lives in
:class:`EthFlowConfig` / :func:`decide_eth_flow` / :func:`decide_eth_exit_drift`
so ETH can be tuned independently later.

Framework reused from BTC (same market lifecycle, same execution/settlement
infrastructure — no technical difference required because every input used
here is symbol-parameterized in the existing project):

* signal klines: :func:`validation_dataset.fetch_klines` accepts any symbol;
  ETH uses ``ETHUSDT`` 1s klines with the identical fields
  (close / volume_base / taker_buy_base) and the identical closeTime rule
  (strictly pre-decision, no look-ahead);
* lifecycle: shared :class:`Signal` / :class:`Position` / :class:`Action`
  plus :class:`SettlementOutcome` (UP/DOWN/PUSH payouts are symbol-agnostic);
* fills: the same simulated-fill mechanics (UP ask for UP entry,
  ``1 - UP bid`` for DOWN entry, mirrored salvage on EXIT) and the same
  fee/slippage accounting apply to ETH books unchanged.

Strategy (spot-only, no Chainlink signal, no fair-value/mispricing, no
Kronos, no probability models, no ML, no edge estimates):

* At ``t = market_start + 20_000`` compute pre-decision ETH taker flow
  imbalance over the prior 30s:
  ``flow = (taker_buy - taker_sell) / total`` in ``[-1, 1]``.
* ``flow >= +0.25`` => UP (buy UP at venue UP chance/100 at/before ``t``);
  ``flow <= -0.25`` => DOWN (buy DOWN at ``1 - UP``); else HOLD.
* Position policy: flat + UP/DOWN => ENTER; holding + opposite ETH spot
  drift at +90s/+180s beyond 2.0bps => EXIT at the then-current contract
  price (salvage); same/HOLD => hold until settlement
  ($1 win, $0 loss, $0.5 PUSH). Winners are never exited early by design
  (exit fires only on opposite drift).
* Settlement is the venue-resolved UP/DOWN. Chainlink is NEVER a signal
  input — only the settlement label. Contract price is NEVER compared to
  fair value; it is only the execution cost and exit salvage.

Initial parameters (``EthFlowConfig``): entry +20s, flow_thr 0.25,
flow window 30s, exit offsets (90s, 180s), exit drift thr 2.0bps,
fee 200bps, slippage 0.005/share. These mirror the BTC starting point for
comparability and are explicitly tunable per-ETH later without touching BTC.

Research/DEMO only: pure functions, no I/O, no trading, no withdrawals,
no Chainlink import. LIVE trading is refused fail-closed.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field, field_validator

from app.models.base import DEC0, DomainModel, as_decimal
from app.research.prediction_markets.up_down_5m.signal import Action, Position, Signal

__all__ = [
    "ETH_SPOT_SYMBOL",
    "EthFlowConfig",
    "EthFlowResult",
    "backtest_eth_flow",
    "compute_drift_bps",
    "compute_flow_imb",
    "decide_eth_exit_drift",
    "decide_eth_flow",
    "ensure_research_only",
    "exit_net",
    "hold_net",
    "manage_with_exit",
]

#: Spot symbol this strategy reads. The only symbol-level difference vs BTC.
ETH_SPOT_SYMBOL = "ETHUSDT"

_BPS = Decimal("10000")
_FEE_DIV = Decimal("10000")


class EthFlowConfig(DomainModel):
    """Exact ETH parameters (independent of the frozen BTC config)."""

    spot_symbol: str = Field(default=ETH_SPOT_SYMBOL)
    entry_offset_ms: int = Field(default=20_000, gt=0)
    flow_thr: Decimal = Decimal("0.25")
    flow_window_ms: int = Field(default=30_000, gt=0)
    exit_offsets_ms: tuple[int, ...] = (90_000, 180_000)
    exit_drift_thr_bps: Decimal = Decimal("2.0")
    fee_bps: int = Field(default=200, ge=0)
    slippage_per_share: Decimal = Decimal("0.005")

    @field_validator("flow_thr", "exit_drift_thr_bps", "slippage_per_share", mode="before")
    @classmethod
    def _dec(cls, v):  # type: ignore[no-untyped-def]
        return as_decimal(v, field="threshold")


class EthFlowResult(DomainModel):
    signal: Signal
    flow_imb: Decimal | None = None
    detail: str = ""


def compute_flow_imb(taker_buy_base: Decimal | None, volume_base: Decimal | None) -> Decimal | None:
    """Signed taker imbalance in [-1, 1] (None when volume missing/non-positive)."""
    if taker_buy_base is None or volume_base is None:
        return None
    if volume_base <= DEC0:
        return None
    return (taker_buy_base - (volume_base - taker_buy_base)) / volume_base


def compute_drift_bps(mid_now: Decimal | None, ref: Decimal | None) -> Decimal | None:
    if mid_now is None or ref is None:
        return None
    if ref <= DEC0 or mid_now <= DEC0:
        return None
    return (mid_now - ref) / ref * _BPS


def decide_eth_flow(
    *,
    flow_imb: Decimal | str | int | float | None,
    config: EthFlowConfig | None = None,
) -> EthFlowResult:
    """Deterministic UP/DOWN/HOLD from pre-decision ETH taker flow only."""
    cfg = config or EthFlowConfig()
    f = as_decimal(flow_imb, field="feature") if flow_imb is not None else None
    if f is None:
        return EthFlowResult(signal=Signal.HOLD, flow_imb=None, detail="no-flow")
    if f >= cfg.flow_thr:
        return EthFlowResult(signal=Signal.UP, flow_imb=f, detail=f"flow={f}>=+{cfg.flow_thr}")
    if f <= -cfg.flow_thr:
        return EthFlowResult(signal=Signal.DOWN, flow_imb=f, detail=f"flow={f}<=-{cfg.flow_thr}")
    return EthFlowResult(signal=Signal.HOLD, flow_imb=f, detail=f"weak|{f}|<{cfg.flow_thr}")


def decide_eth_exit_drift(
    *,
    side: str,
    drift_bps: Decimal | str | int | None,
    config: EthFlowConfig | None = None,
) -> bool:
    """True when ETH spot drift opposes the held side beyond the exit threshold."""
    cfg = config or EthFlowConfig()
    d = as_decimal(drift_bps, field="feature") if drift_bps is not None else None
    if d is None:
        return False
    s = str(side).upper()
    if s == "UP":
        return d <= -cfg.exit_drift_thr_bps
    if s == "DOWN":
        return d >= cfg.exit_drift_thr_bps
    return False


def manage_with_exit(
    position: Position | str,
    entry_signal: Signal | str,
    exit_triggered: bool = False,
) -> Action:
    """Entry now; fast EXIT only when spot drift reverses; else hold to settlement."""
    pos = Position(str(position).upper())
    ent = Signal(str(entry_signal).upper())
    if pos == Position.NONE:
        if ent == Signal.UP:
            return Action.ENTER_UP
        if ent == Signal.DOWN:
            return Action.ENTER_DOWN
        return Action.HOLD_POSITION
    if pos == Position.LONG_UP:
        return Action.EXIT if exit_triggered else Action.HOLD_POSITION
    if pos == Position.LONG_DOWN:
        return Action.EXIT if exit_triggered else Action.HOLD_POSITION
    return Action.HOLD_POSITION


def _cost(side: str, up_price: Decimal, cfg: EthFlowConfig) -> Decimal:
    base = up_price if str(side).upper() == "UP" else (Decimal("1") - up_price)
    return base + cfg.slippage_per_share


def _fee(amount: Decimal, cfg: EthFlowConfig) -> Decimal:
    return amount * Decimal(cfg.fee_bps) / _FEE_DIV


def hold_net(
    side: str, up_price: Decimal, outcome: str, cfg: EthFlowConfig | None = None
) -> Decimal:
    """Hold-to-settlement net per 1 share (fee + slippage included)."""
    c = cfg or EthFlowConfig()
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
    cfg: EthFlowConfig | None = None,
) -> Decimal:
    """Reversal-exit net: salvage at exit price minus entry cost and both fees."""
    c = cfg or EthFlowConfig()
    s = str(side).upper()
    cost = _cost(s, entry_up, c)
    salvage = exit_up if s == "UP" else (Decimal("1") - exit_up)
    return salvage - cost - _fee(cost, c) - _fee(salvage, c) - c.slippage_per_share


def backtest_eth_flow(markets, config: EthFlowConfig | None = None):  # type: ignore[no-untyped-def]
    """Chronological backtest over ValidatedMarket list (no I/O).

    Identical mechanics to the BTC early-flow backtest, parameterized by
    :class:`EthFlowConfig` (entry/exit offsets, flow window, thresholds).
    Callers supply ETH markets (ETHUSDT klines + ETH contract series);
    all prices via ``contract_price_at`` (strictly pre-decision, no
    look-ahead). Returns dict with trades, coverage, wins, win_rate,
    total_net, avg_net, expectancy, max_drawdown, exits, nets. No I/O.
    """
    import bisect

    from app.research.prediction_markets.up_down_5m.validation_dataset import contract_price_at

    cfg = config or EthFlowConfig()
    nets: list[Decimal] = []
    trades = 0
    wins = 0
    exits = 0

    for m in sorted(markets, key=lambda x: x.start_ts_ms):
        kl = list(m.klines)
        cts = [k.close_ts_ms for k in kl]

        def _close_at(ts_ms: int):  # type: ignore[no-untyped-def]
            i = bisect.bisect_right(cts, int(ts_ms)) - 1
            return kl[i] if i >= 0 else None

        ref = _close_at(m.start_ts_ms)
        t = m.start_ts_ms + cfg.entry_offset_ms
        lo, hi = int(t) - cfg.flow_window_ms, int(t)
        buy = vol = Decimal("0")
        for k in kl:
            if lo < k.close_ts_ms <= hi:
                buy += k.taker_buy_base
                vol += k.volume_base
        flow = compute_flow_imb(buy, vol) if vol > 0 else None
        sig = decide_eth_flow(flow_imb=flow, config=cfg).signal
        up0 = contract_price_at(m.contract_series, t)
        if sig.value == "HOLD" or up0 is None:
            nets.append(DEC0)
            continue
        side = sig.value
        exited_up = None
        for eo in cfg.exit_offsets_ms:
            if eo <= cfg.entry_offset_ms:
                continue
            te = m.start_ts_ms + int(eo)
            now = _close_at(te)
            d = compute_drift_bps(
                now.close if now else None, ref.close if ref else None
            )
            if decide_eth_exit_drift(side=side, drift_bps=d, config=cfg):
                up1 = contract_price_at(m.contract_series, te)
                if up1 is not None:
                    exited_up = up1
                    break
        if exited_up is not None:
            n = exit_net(side, up0, exited_up, cfg)
            exits += 1
        else:
            n = hold_net(side, up0, m.outcome.value, cfg)
        nets.append(n)
        trades += 1
        if n > DEC0:
            wins += 1
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
        "expectancy": (total / Decimal(trades)) if trades else None,
        "max_drawdown": max_dd,
        "exits": exits,
        "nets": tuple(nets),
    }


def ensure_research_only() -> None:
    """Fail-closed guard — always raises; ETH flow signal must never trade live."""
    raise RuntimeError("up_down_5m ETH flow signal is research-only; live trading is disabled")
