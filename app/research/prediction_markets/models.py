"""Immutable internal models for prediction-market research.

All models are frozen + extra=forbid via DomainModel.
No mutation, no BNB, BTC/ETH 5m+15m only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import Field, computed_field, field_validator

from app.models.base import DEC0, DomainModel, as_decimal

__all__ = [
    "HistoricalAvailability",
    "MarketDuration",
    "MarketIdentifiers",
    "NormalizedMarket",
    "OutcomeSnapshot",
    "PredictionConstraints",
    "PredictionMarketType",
    "QuoteInspection",
    "ResearchUniverse",
    "TimestampQuality",
    "UpdateFrequency",
]


class MarketDuration(StrEnum):
    M5 = "5m"
    M15 = "15m"
    UNKNOWN = "unknown"

    @property
    def milliseconds(self) -> int | None:
        return {"5m": 300_000, "15m": 900_000}.get(self.value)


class PredictionMarketType(StrEnum):
    """BTC/ETH 5m/15m up/down market types discovered.

    Binance chartType is CRYPTO_UP_DOWN; duration is inferred from
    endDate - startDate (5m vs 15m).  Spot price feed is BINANCE.
    """

    BTC_5M = "BTC_5m"
    BTC_15M = "BTC_15m"
    ETH_5M = "ETH_5m"
    ETH_15M = "ETH_15m"


# Scope guard — only BTC/ETH supported
ALLOWED_SYMBOLS = frozenset({"BTCUSDT", "BTC", "ETHUSDT", "ETH"})
ALLOWED_DURATIONS = frozenset({MarketDuration.M5, MarketDuration.M15})


def _infer_duration(start_ms: int | None, end_ms: int | None) -> MarketDuration:
    if start_ms is None or end_ms is None:
        return MarketDuration.UNKNOWN
    diff = end_ms - start_ms
    if diff == 300_000:
        return MarketDuration.M5
    if diff == 900_000:
        return MarketDuration.M15
    # allow small clock skew (tolerance 5s)
    if 295_000 <= diff <= 305_000:
        return MarketDuration.M5
    if 895_000 <= diff <= 905_000:
        return MarketDuration.M15
    return MarketDuration.UNKNOWN


class MarketIdentifiers(DomainModel):
    """Resolved identifiers for one prediction market."""

    market_topic_id: int = Field(gt=0)
    market_id: int = Field(gt=0)
    token_id: str
    slug: str
    vendor: str = "PREDICT_FUN"
    chain_id: str = "56"
    symbol: str  # BTCUSDT / ETHUSDT
    duration: MarketDuration
    market_type: PredictionMarketType

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        u = v.strip().upper()
        if u not in {"BTCUSDT", "ETHUSDT", "BTC", "ETH"}:
            raise ValueError(f"symbol {v!r} not in research scope (BTC/ETH only)")
        return u

    @field_validator("token_id")
    @classmethod
    def _validate_token(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tokenId must be non-empty")
        return v.strip()


class OutcomeSnapshot(DomainModel):
    """Single YES/NO outcome with price, liquidity, orderbook."""

    name: str  # YES / NO
    token_id: str
    price: Decimal  # [0,1]
    chance: Decimal | None = None
    index: int | None = None
    # Orderbook (may be None if not fetched)
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None
    bid_size: Decimal | None = None
    ask_size: Decimal | None = None
    bids: tuple[tuple[Decimal, Decimal], ...] = ()
    asks: tuple[tuple[Decimal, Decimal], ...] = ()
    last_trade_price: Decimal | None = None
    timestamp_ms: int | None = None

    @field_validator("price", mode="before")
    @classmethod
    def _price_decimal(cls, v):  # type: ignore[no-untyped-def]
        return as_decimal(v, field="price")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mid(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / Decimal("2")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread_bps(self) -> Decimal | None:
        mid = self.mid
        if mid is None or mid <= DEC0 or self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_ask - self.best_bid) / mid * Decimal("10000")


class NormalizedMarket(DomainModel):
    """Fully normalized prediction market (immutable)."""

    identifiers: MarketIdentifiers
    slug: str
    title: str
    question: str | None = None
    symbol: str
    duration: MarketDuration
    market_type: PredictionMarketType
    start_ms: int | None = None
    end_ms: int | None = None
    resolution_ms: int | None = None
    status: str | None = None
    trading_status: str | None = None
    liquidity: Decimal | None = None
    volume: Decimal | None = None
    participant_count: int | None = None
    fee_rate_bps: int | None = None
    outcomes: tuple[OutcomeSnapshot, ...] = ()
    raw_topic: dict | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_active(self) -> bool:
        return (self.trading_status or "").upper() == "OPEN"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def yes_outcome(self) -> OutcomeSnapshot | None:
        for o in self.outcomes:
            if o.name.upper() == "YES":
                return o
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def no_outcome(self) -> OutcomeSnapshot | None:
        for o in self.outcomes:
            if o.name.upper() == "NO":
                return o
        return None


class PredictionConstraints(DomainModel):
    """Execution constraints and fees (no order placement)."""

    fee_rate_bps: int = 200  # Binance default
    slippage_bps: int = 1200
    min_order_usdt: Decimal = Decimal("1.5")
    price_precision: int = 2
    collateral: str = "USDT"
    chain_id: str = "56"
    vendor: str = "PREDICT_FUN"
    supports_market_orders: bool = True
    supports_limit_orders: bool = True
    settlement_payout: Decimal = Decimal("1")  # winner gets 1 USDT per share
    notes: str = "All Binance prediction trades require signed SAPI; funds via MPC or CEX transfer."


class TimestampQuality(DomainModel):
    """Timestamp characteristics of prediction market feeds."""

    has_exchange_timestamp: bool
    has_update_timestamp_ms: bool
    has_start_end_resolution: bool
    granularity_ms: int | None = None  # e.g. 1s candles vs ms
    ordering_guaranteed: bool = False  # Binance WS: no strict ordering
    notes: str = ""


class UpdateFrequency(DomainModel):
    """Measured update frequency (requires live sampling; report expectation when mocked)."""

    expected_interval_ms: int | None = None
    measured_interval_ms: float | None = None
    sample_count: int = 0
    ws_latency_expected_ms: int = 200  # Binance docs: <200ms after upstream
    rest_weight_per_call: int = 200
    is_active_only: bool = True  # WS pushes active markets only


class HistoricalAvailability(DomainModel):
    """Historical data available for backtesting."""

    has_ohlcv: bool = False
    has_trade_history: bool = False
    has_orderbook_history: bool = False
    has_ticker_history: bool = False
    binance_provides_history: bool = False  # Binance SAPI has no historical candles
    predict_fun_has_history: bool = False
    notes: str = ""


class QuoteInspection(DomainModel):
    """Result of inspecting the official quote API without placing orders."""

    endpoint: str = "/sapi/v1/w3w/wallet/prediction/trade/get-quote"
    method: str = "POST"
    required_fields: tuple[str, ...] = (
        "walletAddress",
        "tokenId",
        "side",
        "amountIn",
        "orderType",
        "slippageBps",
    )
    optional_fields: tuple[str, ...] = ("chainId", "priceLimit", "feeRateBps")
    response_fields: tuple[str, ...] = (
        "quoteId",
        "averagePrice",
        "priceImpact",
        "feeAmount",
        "amountOut",
        "expireAt",
        "slippageBps",
        "feeRateBps",
    )
    is_read_only: bool = True
    does_not_place_order: bool = True
    notes: str = "Quote inspection is read-only; placing an order requires separate signed call with quoteId."


class ResearchUniverse(DomainModel):
    """Complete research snapshot returned by the investigator."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    markets: tuple[NormalizedMarket, ...] = ()
    market_types_found: tuple[PredictionMarketType, ...] = ()
    identifiers: tuple[MarketIdentifiers, ...] = ()
    constraints: PredictionConstraints = Field(default_factory=PredictionConstraints)
    timestamp_quality: TimestampQuality = Field(default_factory=lambda: TimestampQuality(
        has_exchange_timestamp=True,
        has_update_timestamp_ms=True,
        has_start_end_resolution=True,
        granularity_ms=1,
        ordering_guaranteed=False,
    ))
    update_frequency: UpdateFrequency = Field(default_factory=UpdateFrequency)
    historical: HistoricalAvailability = Field(default_factory=HistoricalAvailability)
    quote_inspection: QuoteInspection = Field(default_factory=QuoteInspection)
    hypothesis_testable: bool = False
    hypothesis_notes: str = ""
