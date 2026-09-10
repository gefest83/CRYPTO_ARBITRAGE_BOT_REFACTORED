"""Backtest immutable models."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import Field, field_validator

from app.models.base import DomainModel, as_decimal

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "GroupKey",
    "GroupStats",
    "ImpulseEvent",
    "OutcomeMetrics",
]


class GroupKey(StrEnum):
    BTC = "BTC"
    ETH = "ETH"
    M5 = "5m"
    M15 = "15m"
    YES = "YES"
    NO = "NO"


class BacktestConfig(DomainModel):
    """Deterministic config — all thresholds explicit."""

    # impulse detection
    impulse_lookback_ms: int = Field(default=1000, ge=100)
    impulse_thresholds_bps: tuple[int, ...] = (5, 10, 20, 50)  # evaluated separately
    # evaluation window after t0
    eval_window_ms: int = Field(default=10000, ge=100)
    # cost model
    fee_bps: int = Field(default=200, ge=0)
    # slippage: estimated impact bps per unit depth if depth known
    slippage_per_depth_bps: Decimal = Field(default=Decimal("0"))
    # minimum depth required to consider executable
    min_depth: int = Field(default=0, ge=0)
    # liquidity buckets (by depth or notional) — boundaries
    liquidity_buckets: tuple[str, ...] = ("low", "mid", "high")
    # time-to-resolution buckets (ms thresholds)
    ttr_buckets_ms: tuple[int, ...] = (60_000, 120_000, 300_000)  # <1m, 1-2m, 2-5m, >5m
    # stale/out-of-order handling already done by collector; analyzer skips expired markets
    min_observations: int = Field(default=30, ge=1, description="minimum dataset size to claim significance")


class ImpulseEvent(DomainModel):
    """Deterministic spot impulse at t0."""

    t0_ms: int = Field(ge=0)
    symbol: str  # BTCUSDT / ETHUSDT
    spot_mid_before: Decimal
    spot_mid_at_t0: Decimal
    impulse_bps: Decimal  # (mid_at - mid_before)/mid_before * 10000
    threshold_bps: int
    direction: int  # +1 up, -1 down
    # anchor for PM evaluation — initial PM price at or before t0
    market_id: int
    token_id: str
    outcome: str  # YES / NO
    duration: str  # 5m / 15m
    initial_price: Decimal  # PM price at or before t0
    initial_spread: Decimal | None = None
    initial_spread_bps: Decimal | None = None
    depth: int | None = None
    time_to_resolution_ms: int | None = None
    liquidity_bucket: str | None = None
    ttr_bucket: str | None = None

    @field_validator("symbol")
    @classmethod
    def _check_sym(cls, v: str) -> str:
        u = v.strip().upper()
        if u.replace("USDT", "") not in ("BTC", "ETH"):
            raise ValueError(f"BTC/ETH only, got {v!r}")
        return u


class OutcomeMetrics(DomainModel):
    """Per-event outcome measured strictly after t0 (evaluator sees future, signal did not)."""

    event: ImpulseEvent
    # repricing
    repricing_lag_ms: int | None = None  # time from t0 to first favorable move beyond spread
    max_favorable: Decimal | None = None  # MFE in price units (0-1)
    max_adverse: Decimal | None = None  # MAE
    time_to_max_favorable_ms: int | None = None
    # costs
    gross_edge: Decimal | None = None  # max_favorable (or last) in price units
    spread_cost: Decimal | None = None
    fee_cost: Decimal | None = None
    slippage_cost: Decimal | None = None
    net_edge: Decimal | None = None  # gross - spread - fee - slippage
    win: bool | None = None  # net_edge > 0
    # liquidity / sufficiency flags
    insufficient_liquidity: bool = False
    is_expired: bool = False

    @field_validator("max_favorable", "max_adverse", "gross_edge", "net_edge", mode="before")
    @classmethod
    def _dec(cls, v):  # type: ignore[no-untyped-def]
        if v is None:
            return None
        return as_decimal(v, field="decimal")


class GroupStats(DomainModel):
    group: str  # e.g. "BTC|5m|YES|threshold=10|liquidity=high"
    sample_count: int
    signal_count: int
    median_lag_ms: float | None = None
    mean_lag_ms: float | None = None
    p25_lag_ms: float | None = None
    p75_lag_ms: float | None = None
    p95_lag_ms: float | None = None
    gross_edge_mean: Decimal | None = None
    net_edge_mean: Decimal | None = None
    win_rate: float | None = None
    avg_return: Decimal | None = None
    mfe_mean: Decimal | None = None
    mae_mean: Decimal | None = None
    # distribution helpers
    gross_edge_p50: Decimal | None = None
    net_edge_p50: Decimal | None = None


class BacktestResult(DomainModel):
    """Deterministic full result set."""

    config: BacktestConfig
    dataset_size: int  # total observations
    coverage: dict[str, int]  # counts by symbol/duration etc
    group_stats: tuple[GroupStats, ...]
    all_metrics: tuple[OutcomeMetrics, ...]
    has_evidence: bool
    evidence_notes: str
    limitations: str
