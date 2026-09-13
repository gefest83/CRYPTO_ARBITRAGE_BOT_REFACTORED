"""Binance BTC Up/Down 5m — market lifecycle + data model (research-only).

Market lifecycle (derived from existing SAPI discovery code + task spec):

1. DISCOVERED: topic appears via ``market/list`` (l1Category=crypto) with
   ``startDate``/``endDate`` (ms epoch), ``symbol`` BTCUSDT, duration 5m
   (``endDate - startDate == 300_000``, tolerance 5s for clock skew).
2. OPEN: ``tradingStatus == OPEN`` and ``start_ts <= now < end_ts``.
   UP/DOWN contract prices come from ``order-book`` per tokenId
   (best bid/ask in [0,1]) + ``last-trade-price``; executable price is
   inspected read-only via ``trade/get-quote`` (never places orders here).
3. CLOSED: ``now >= end_ts`` — quoting/trading halted. No new positions.
4. SETTLED: ``end_price`` (Chainlink BTC/USDT ToB mid = close of the 5m
   candle immediately before market end) and ``start_price`` (Chainlink
   ToB mid at market start) are both known -> binary outcome via
   :func:`settle_up_down_5m`. Winner shares redeem 1.0, loser 0.0,
   equal (PUSH) 0.5/0.5.

What this model stores per 5m market (task field list):
  market_id, start_ts_ms, end_ts_ms, start_price (Chainlink mid at start),
  current_price (latest Chainlink mid observed), up_bid/ask, down_bid/ask,
  position (shares held per side), settlement (outcome once known).

Trading mechanics available to us (read-only in research):
  SAPI signed GETs: market/list, market/search, market/detail,
  order-book (per vendor/marketId/tokenId), last-trade-price.
  Read-only POST: trade/get-quote (quoteId/averagePrice/fee/expiry).
  Account: position/list, batch-redeem (documented, never invoked here).
  WS: SAPI orderbook push (signed) + Predict.fun WS + spot WS for reference.
  Fees: fee_rate_bps default 200, slippage tolerance default 1200 bps,
  collateral USDT, payout 1 USDT per winning share.

DEPRECATED / IGNORED previous approach:
  ``app.research.prediction_markets.backtest`` (Phase 3 spot-impulse ->
  repricing-lag with ``BacktestAnalyzer``/``detect_impulses``) is a
  spot-proxy probability model. It does NOT know binary settlement and
  must NOT be used for Up/Down edge evaluation. It is left untouched for
  history but ignored by this package — all new work uses settlement-based
  replay in :mod:`app.research.prediction_markets.up_down_5m.replay`.

Research-only: no order placement, no wallet mutation, no live trading.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import Field, computed_field, field_validator

from app.models.base import DEC0, DEC1, DomainModel, as_decimal
from app.research.prediction_markets.up_down_5m.settlement import (
    SettlementOutcome,
    Side,
    settle_up_down_5m,
)

__all__ = [
    "FIVE_MIN_MS",
    "MarketState",
    "UpDown5mMarket",
]

FIVE_MIN_MS = 300_000
_DURATION_TOLERANCE_MS = 5_000


class MarketState(StrEnum):
    UPCOMING = "UPCOMING"  # now < start_ts
    OPEN = "OPEN"  # start_ts <= now < end_ts
    CLOSED = "CLOSED"  # now >= end_ts, settlement inputs incomplete
    SETTLED = "SETTLED"  # outcome computed from start_price + end_price


class UpDown5mMarket(DomainModel):
    """One BTC Up/Down 5m market snapshot (immutable)."""

    market_id: int = Field(gt=0)
    start_ts_ms: int = Field(ge=0)
    end_ts_ms: int = Field(ge=0)
    # Chainlink BTC/USDT ToB mid at market start (the "price to beat").
    start_price: Decimal | None = None
    # Latest Chainlink BTC/USDT ToB mid observed (reference, NOT settlement).
    current_price: Decimal | None = None
    # Chainlink ToB mid = close of 5m candle immediately before market end.
    end_price: Decimal | None = None
    # UP/DOWN contract order-book top (prices in [0,1] USDT per share).
    up_bid: Decimal | None = None
    up_ask: Decimal | None = None
    down_bid: Decimal | None = None
    down_ask: Decimal | None = None
    # Research position: shares held to settlement (paper only).
    position_up: Decimal = Decimal("0")
    position_down: Decimal = Decimal("0")
    # Settlement outcome once computed (None until CLOSED->SETTLED).
    settlement: SettlementOutcome | None = None

    @field_validator("end_ts_ms")
    @classmethod
    def _check_window(cls, v: int, info) -> int:  # type: ignore[no-untyped-def]
        start = (info.data or {}).get("start_ts_ms")
        if start is not None:
            diff = v - int(start)
            if abs(diff - FIVE_MIN_MS) > _DURATION_TOLERANCE_MS:
                raise ValueError(f"5m market window must be 300_000ms ±5s, got diff={diff}")
            if v <= int(start):
                raise ValueError("end_ts_ms must be after start_ts_ms")
        return v

    @field_validator(
        "start_price", "current_price", "end_price",
        "up_bid", "up_ask", "down_bid", "down_ask",
        "position_up", "position_down", mode="before",
    )
    @classmethod
    def _dec(cls, v):  # type: ignore[no-untyped-def]
        if v is None:
            return None
        return as_decimal(v, field="decimal")

    @field_validator("up_bid", "up_ask", "down_bid", "down_ask")
    @classmethod
    def _check_contract_range(cls, v: Decimal | None) -> Decimal | None:
        if v is None:
            return None
        if v < DEC0 or v > DEC1:
            raise ValueError(f"contract price must be in [0,1], got {v!r}")
        return v

    # ---------- derived ----------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_ms(self) -> int:
        return self.end_ts_ms - self.start_ts_ms

    @computed_field  # type: ignore[prop-decorator]
    @property
    def up_mid(self) -> Decimal | None:
        if self.up_bid is None or self.up_ask is None:
            return None
        return (self.up_bid + self.up_ask) / Decimal("2")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def down_mid(self) -> Decimal | None:
        if self.down_bid is None or self.down_ask is None:
            return None
        return (self.down_bid + self.down_ask) / Decimal("2")

    # ---------- lifecycle ----------
    def state_at(self, now_ms: int) -> MarketState:
        """State at a given wall-clock ms (deterministic, no I/O)."""
        if self.settlement is not None:
            return MarketState.SETTLED
        if int(now_ms) < self.start_ts_ms:
            return MarketState.UPCOMING
        if int(now_ms) < self.end_ts_ms:
            return MarketState.OPEN
        # now >= end: settled iff both Chainlink anchors known AND they agree
        # with the stored settlement (or settlement computable).
        if self.start_price is not None and self.end_price is not None:
            return MarketState.SETTLED
        return MarketState.CLOSED

    def can_trade_at(self, now_ms: int) -> bool:
        """True only while OPEN. Quotes after end or before start are rejected."""
        return self.state_at(int(now_ms)) == MarketState.OPEN

    def settle(self) -> UpDown5mMarket:
        """Return a SETTLED copy using start_price/end_price.

        Raises:
            ValueError: if either Chainlink anchor is missing/non-positive.
        """
        if self.start_price is None or self.end_price is None:
            raise ValueError("settlement requires both start_price and end_price (Chainlink ToB mids)")
        outcome = settle_up_down_5m(self.start_price, self.end_price)
        return self.model_copy(update={"settlement": outcome, "current_price": self.end_price})

    def with_position(self, side: Side | str, shares: Decimal | str | int) -> UpDown5mMarket:
        """Return a copy with paper position added (research replay only)."""
        s = Side(str(side).upper())
        qty = as_decimal(shares, field="shares")
        if qty < DEC0:
            raise ValueError("position shares must be >= 0")
        if s == Side.UP:
            return self.model_copy(update={"position_up": self.position_up + qty})
        return self.model_copy(update={"position_down": self.position_down + qty})
