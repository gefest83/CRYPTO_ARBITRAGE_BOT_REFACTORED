"""Immutable collector models — all frozen, extra=forbid.

BTC/ETH only, 5m/15m only, BNB excluded at validation.
Timestamps: integer ms since epoch, UTC, deterministic (no floats).
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import Field, computed_field, field_validator

from app.models.base import DEC0, DomainModel, as_decimal

__all__ = [
    "CollectorStats",
    "PredictionOrderbookObservation",
    "PredictionTradeObservation",
    "RawObservation",
    "SpotObservation",
    "SynchronizedObservation",
]


class SourceKind(StrEnum):
    SPOT_ORDERBOOK = "spot_orderbook"
    SPOT_TRADE = "spot_trade"
    PREDICTION_ORDERBOOK = "prediction_orderbook"
    PREDICTION_TRADE = "prediction_trade"


class RawObservation(DomainModel):
    """Base raw observation persisted verbatim."""

    captured_at_ms: int = Field(ge=0, description="collector wall clock ms")
    source: SourceKind
    symbol: str  # BTCUSDT / ETHUSDT
    market_id: int | None = None
    token_id: str | None = None
    market_topic_id: int | None = None
    duration: str | None = None  # 5m / 15m
    exchange_ts_ms: int | None = None  # venue event timestamp
    update_ts_ms: int | None = None  # prediction WS updateTimestampMs
    sequence: int | None = None
    resolution_ms: int | None = None  # market endDate
    time_to_resolution_ms: int | None = None  # resolution_ms - exchange_ts_ms
    raw: dict | None = None  # verbatim payload for replay

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, v: str) -> str:
        u = v.strip().upper()
        base = u.replace("USDT", "")
        if base not in ("BTC", "ETH"):
            raise ValueError(f"collector scope is BTC/ETH only, got {v!r} (BNB excluded)")
        if u not in ("BTCUSDT", "ETHUSDT", "BTC", "ETH"):
            raise ValueError(f"invalid symbol {v!r}")
        return u

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_expired(self) -> bool:
        if self.resolution_ms is None or self.exchange_ts_ms is None:
            return False
        return self.exchange_ts_ms >= self.resolution_ms


class SpotObservation(RawObservation):
    """Binance spot bid/ask + trade for BTC/ETH."""

    source: SourceKind = SourceKind.SPOT_ORDERBOOK
    bid: Decimal | None = None
    ask: Decimal | None = None
    bid_qty: Decimal | None = None
    ask_qty: Decimal | None = None
    last_price: Decimal | None = None
    last_qty: Decimal | None = None
    volume: Decimal | None = None  # keep generic

    @field_validator("bid", "ask", "bid_qty", "ask_qty", "last_price", "last_qty", mode="before")
    @classmethod
    def _dec(cls, v):  # type: ignore[no-untyped-def]
        if v is None:
            return None
        return as_decimal(v, field="decimal")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread_bps(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        mid = (self.bid + self.ask) / Decimal("2")
        if mid <= DEC0:
            return None
        return (self.ask - self.bid) / mid * Decimal("10000")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def depth(self) -> int | None:
        # simplified: 1 level shown; collector stores full book in raw
        if self.bid is not None and self.ask is not None:
            return 1
        return None


class PredictionOrderbookObservation(RawObservation):
    """Prediction YES/NO orderbook update (WS pref, REST fallback)."""

    source: SourceKind = SourceKind.PREDICTION_ORDERBOOK
    outcome: str | None = None  # YES / NO
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None
    bid_size: Decimal | None = None
    ask_size: Decimal | None = None
    bids: tuple[tuple[Decimal, Decimal], ...] = ()
    asks: tuple[tuple[Decimal, Decimal], ...] = ()
    last_trade_price: Decimal | None = None
    # time_to_resolution_ms already in base

    @field_validator("best_bid", "best_ask", "bid_size", "ask_size", "last_trade_price", mode="before")
    @classmethod
    def _dec2(cls, v):  # type: ignore[no-untyped-def]
        if v is None:
            return None
        return as_decimal(v, field="decimal")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread_bps(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        mid = (self.best_bid + self.best_ask) / Decimal("2")
        if mid <= DEC0:
            return None
        return (self.best_ask - self.best_bid) / mid * Decimal("10000")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def depth(self) -> int:
        return max(len(self.bids), len(self.asks))


class PredictionTradeObservation(RawObservation):
    """Prediction trade (lastTradePrice or WS trade)."""

    source: SourceKind = SourceKind.PREDICTION_TRADE
    price: Decimal
    size: Decimal | None = None
    side: str | None = None  # BUY/SELL if known

    @field_validator("price", mode="before")
    @classmethod
    def _price(cls, v):  # type: ignore[no-untyped-def]
        return as_decimal(v, field="price")

    @field_validator("size", mode="before")
    @classmethod
    def _size(cls, v):  # type: ignore[no-untyped-def]
        if v is None:
            return None
        return as_decimal(v, field="size")


class SynchronizedObservation(DomainModel):
    """Paired spot + prediction observation for backtesting (deterministic)."""

    spot: SpotObservation
    prediction: PredictionOrderbookObservation | PredictionTradeObservation
    delta_ms: int  # prediction.update_ts_ms - spot.exchange_ts_ms
    synchronized_at_ms: int  # collector time of pairing
    time_to_resolution_ms: int | None = None


class CollectorStats(DomainModel):
    total_observations: int = 0
    spot_count: int = 0
    prediction_ob_count: int = 0
    prediction_trade_count: int = 0
    duplicates_dropped: int = 0
    out_of_order_dropped: int = 0
    stale_dropped: int = 0
    expired_dropped: int = 0
    gaps_detected: int = 0
    reconnects: int = 0
    rest_snapshots: int = 0
    last_sequence: dict[str, int] = Field(default_factory=dict)
