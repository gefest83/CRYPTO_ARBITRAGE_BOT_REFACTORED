"""BTC Up/Down 5m entry signal — deterministic UP/DOWN/HOLD (research-only).

Uses ONLY pre-decision BTC market data (never Chainlink, never the future):

* ``drift_bps`` — spot mid vs the market's reference price (spot mid at
  market start; Chainlink is the settlement source, never a signal input);
* ``momentum_bps`` / ``accel_bps`` — short-term momentum and its change;
* ``flow_imb`` — signed taker-volume imbalance from spot trades
  (Binance ``m`` flag: maker-buyer => taker sell);
* ``book_imb`` — prediction-book top-depth imbalance
  (assumption, documented: the single Predict.fun book is the UP-side book);
* ``time_remaining_ms`` — gating (no entries while warming up / too late).

Rule (2-of-4 vote, all thresholds explicit in :class:`SignalConfig`):
drift, momentum(+accel confirm), flow and book each vote +1/-1/0 through
deadband thresholds; sum >= +2 => UP, <= -2 => DOWN, else HOLD.

Position policy (:func:`manage`): flat + UP => ENTER_UP (BUY UP),
flat + DOWN => ENTER_DOWN (BUY DOWN); opposing signal while holding =>
EXIT (reversal, flatten quickly); same/HOLD => HOLD_POSITION (hold until
settlement). :func:`exit_to_skip` maps an EXIT onto a replay SKIP so the
existing settlement replay scores entries only.

Research-only: pure functions, no I/O, no trading, no Chainlink import.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import Field, field_validator

from app.models.base import DomainModel, as_decimal

__all__ = [
    "Action",
    "Position",
    "Signal",
    "SignalConfig",
    "SignalFeatures",
    "SignalResult",
    "compute_features",
    "ensure_research_only",
    "evaluate",
    "exit_to_skip",
    "manage",
]

_BPS = Decimal("10000")
_VOTE_THRESHOLD = 2


class Signal(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    HOLD = "HOLD"


class Position(StrEnum):
    NONE = "NONE"
    LONG_UP = "LONG_UP"
    LONG_DOWN = "LONG_DOWN"


class Action(StrEnum):
    ENTER_UP = "ENTER_UP"  # BUY UP now
    ENTER_DOWN = "ENTER_DOWN"  # BUY DOWN now
    HOLD_POSITION = "HOLD_POSITION"  # hold until settlement
    EXIT = "EXIT"  # reversal: flatten quickly, minimal loss


class SignalConfig(DomainModel):
    """All thresholds explicit. Defaults from multiday threshold research
    (scripts/run_up_down_5m_threshold_research.py,
    data/research/up_down_5m_thresholds.json): 135 labeled 5m windows,
    spot-drift proxy label (NOT settlement), best coverage>=20% =
    drift 8bps / mom 5bps / flow 0.10 / book 0.05 at 64% accuracy,
    37% coverage (n=50)."""

    drift_entry_bps: Decimal = Decimal("8")
    momentum_bps: Decimal = Decimal("5")
    accel_tolerance_bps: Decimal = Decimal("12")
    flow_threshold: Decimal = Decimal("0.10")
    book_threshold: Decimal = Decimal("0.05")
    momentum_window_ms: int = Field(default=30_000, gt=0)
    accel_window_ms: int = Field(default=60_000, gt=0)
    flow_window_ms: int = Field(default=60_000, gt=0)
    min_elapsed_ms: int = Field(default=60_000, ge=0)
    min_remaining_ms: int = Field(default=120_000, ge=0)

    @field_validator(
        "drift_entry_bps", "momentum_bps", "accel_tolerance_bps",
        "flow_threshold", "book_threshold", mode="before",
    )
    @classmethod
    def _dec(cls, v):  # type: ignore[no-untyped-def]
        return as_decimal(v, field="threshold")


class SignalFeatures(DomainModel):
    """Pre-decision features only (all optional; missing => abstain vote)."""

    decision_ts_ms: int = Field(ge=0)
    ref_price: Decimal | None = None
    mid_now: Decimal | None = None
    mid_momentum_ago: Decimal | None = None
    mid_accel_ago: Decimal | None = None
    flow_imb: Decimal | None = None  # [-1, 1], + = taker-buy pressure
    book_imb: Decimal | None = None  # [-1, 1], + = bid-side depth
    time_remaining_ms: int | None = None
    elapsed_ms: int | None = None

    @field_validator(
        "ref_price", "mid_now", "mid_momentum_ago", "mid_accel_ago",
        "flow_imb", "book_imb", mode="before",
    )
    @classmethod
    def _dec(cls, v):  # type: ignore[no-untyped-def]
        if v is None:
            return None
        return as_decimal(v, field="feature")


class SignalResult(DomainModel):
    signal: Signal
    votes: int
    detail: str


def _bps(now: Decimal, base: Decimal) -> Decimal | None:
    if base <= 0:
        return None
    return (now - base) / base * _BPS


def compute_features(
    *,
    decision_ts_ms: int,
    market_start_ms: int,
    market_end_ms: int,
    ref_price: Decimal | str | int | None,
    mid_now: Decimal | str | int | None,
    mid_momentum_ago: Decimal | str | int | None = None,
    mid_accel_ago: Decimal | str | int | None = None,
    flow_imb: Decimal | str | int | float | None = None,
    book_imb: Decimal | str | int | float | None = None,
) -> SignalFeatures:
    """Assemble features; caller guarantees every input is dated <= decision."""
    dec = lambda v: as_decimal(v, field="feature") if v is not None else None  # noqa: E731
    return SignalFeatures(
        decision_ts_ms=int(decision_ts_ms),
        ref_price=dec(ref_price),
        mid_now=dec(mid_now),
        mid_momentum_ago=dec(mid_momentum_ago),
        mid_accel_ago=dec(mid_accel_ago),
        flow_imb=dec(flow_imb),
        book_imb=dec(book_imb),
        time_remaining_ms=int(market_end_ms) - int(decision_ts_ms),
        elapsed_ms=int(decision_ts_ms) - int(market_start_ms),
    )


def _vote(value: Decimal | None, thr: Decimal) -> int:
    if value is None:
        return 0
    if value >= thr:
        return 1
    if value <= -thr:
        return -1
    return 0


def evaluate(features: SignalFeatures, config: SignalConfig | None = None) -> SignalResult:
    """Deterministic UP/DOWN/HOLD from pre-decision features only."""
    cfg = config or SignalConfig()
    elapsed = features.elapsed_ms if features.elapsed_ms is not None else 0
    remaining = features.time_remaining_ms if features.time_remaining_ms is not None else 0
    if features.ref_price is None or features.mid_now is None:
        return SignalResult(signal=Signal.HOLD, votes=0, detail="no-ref")
    if elapsed < cfg.min_elapsed_ms:
        return SignalResult(signal=Signal.HOLD, votes=0, detail="warming")
    if remaining < cfg.min_remaining_ms:
        return SignalResult(signal=Signal.HOLD, votes=0, detail="too-late")

    drift = _bps(features.mid_now, features.ref_price)
    mom = _bps(features.mid_now, features.mid_momentum_ago) if features.mid_momentum_ago is not None else None
    mom_prev = (
        _bps(features.mid_momentum_ago, features.mid_accel_ago)
        if features.mid_momentum_ago is not None and features.mid_accel_ago is not None
        else None
    )
    accel = mom - mom_prev if mom is not None and mom_prev is not None else None

    votes = 0
    parts: list[str] = []
    v = _vote(drift, cfg.drift_entry_bps)
    votes += v
    parts.append(f"drift={drift} v={v}")
    if mom is None:
        parts.append("mom=missing v=0")
    elif (mom >= cfg.momentum_bps and (accel is None or accel >= -cfg.accel_tolerance_bps)) or (
        mom <= -cfg.momentum_bps and (accel is None or accel <= cfg.accel_tolerance_bps)
    ):
        mv = 1 if mom > 0 else -1
        votes += mv
        parts.append(f"mom={mom} accel={accel} v={mv}")
    else:
        parts.append(f"mom={mom} accel={accel} v=0")
    v = _vote(features.flow_imb, cfg.flow_threshold)
    votes += v
    parts.append(f"flow={features.flow_imb} v={v}")
    v = _vote(features.book_imb, cfg.book_threshold)
    votes += v
    parts.append(f"book={features.book_imb} v={v}")

    if votes >= _VOTE_THRESHOLD:
        return SignalResult(signal=Signal.UP, votes=votes, detail=";".join(parts))
    if votes <= -_VOTE_THRESHOLD:
        return SignalResult(signal=Signal.DOWN, votes=votes, detail=";".join(parts))
    return SignalResult(signal=Signal.HOLD, votes=votes, detail=";".join(parts))


def manage(position: Position | str, signal: Signal | str) -> Action:
    """Position policy: immediate entry, fast exit on reversal, else hold."""
    pos = Position(str(position).upper())
    sig = Signal(str(signal).upper())
    if pos == Position.NONE:
        if sig == Signal.UP:
            return Action.ENTER_UP
        if sig == Signal.DOWN:
            return Action.ENTER_DOWN
        return Action.HOLD_POSITION
    if pos == Position.LONG_UP:
        return Action.EXIT if sig == Signal.DOWN else Action.HOLD_POSITION
    return Action.EXIT if sig == Signal.UP else Action.HOLD_POSITION


def exit_to_skip(market_id: int, entry_ts_ms: int):  # type: ignore[no-untyped-def]
    """Map an EXIT (flattened, no settlement exposure) onto a replay SKIP."""
    from app.research.prediction_markets.up_down_5m.replay import ReplayDecision

    return ReplayDecision(
        market_id=int(market_id), side="SKIP", entry_price=Decimal("0.5"),
        size=Decimal("1"), entry_ts_ms=int(entry_ts_ms),
    )


def ensure_research_only() -> None:
    """Fail-closed guard — always raises; signal must never trade live."""
    raise RuntimeError("up_down_5m signal is research-only; live trading is disabled (fail-closed)")
