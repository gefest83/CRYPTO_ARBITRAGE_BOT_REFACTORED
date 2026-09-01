"""Domain enumerations shared by every layer."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

__all__ = [
    "ArbitrageStrategy",
    "ExchangeStatus",
    "HealthStatus",
    "MarketType",
    "OpportunityStatus",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "RiskLevel",
    "StreamKind",
    "TimeInForce",
    "TradeStatus",
    "TradingMode",
    "TransferState",
]


class TradingMode(StrEnum):
    """Execution environment: PAPER (simulated), DEMO (testnet), LIVE (real)."""

    PAPER = "PAPER"
    DEMO = "DEMO"
    LIVE = "LIVE"


class ExchangeStatus(StrEnum):
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    DISABLED = "disabled"
    UNKNOWN = "unknown"
    #: Public data flows, private calls are stopped — the venue's API keys
    #: were not accepted (check testnet/demo key mode).
    DATA_ONLY = "data_only"


class MarketType(StrEnum):
    SPOT = "spot"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> OrderSide:
        return OrderSide.SELL if self is OrderSide.BUY else OrderSide.BUY


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(StrEnum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class OrderStatus(StrEnum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    #: Placement/cancellation exceeded the configured leg timeout; recovery
    #: must query the exchange before the outcome is believed.
    TIMEOUT = "timeout"
    #: The outcome could not be classified (adapter raised something
    #: unexpected).  Recovery must query the exchange before believing it.
    UNKNOWN = "unknown"
    #: Recovery could not establish the order's final state on the venue.
    #: The order MUST NOT be treated as rejected (it may have filled) and
    #: MUST NOT be retried — a human resolves it.  Fail-closed status.
    MANUAL_REVIEW = "manual_review"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_ORDER_STATUSES


class TradeStatus(StrEnum):
    """Lifecycle of one executed (or attempted) arbitrage trade."""

    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    MANUAL_REVIEW = "manual_review"


class TransferState(StrEnum):
    """Lifecycle of a cross-exchange transfer arbitrage workflow.

    The workflow is sequential and long-running (minutes to hours):

        CREATED -> BUY_SUBMITTED -> BUY_FILLED -> WITHDRAW_SUBMITTED ->
        WITHDRAW_PENDING -> TRANSFER_IN_PROGRESS -> DEPOSIT_DETECTED ->
        SELL_SUBMITTED -> COMPLETED

    Any step may move the record to FAILED (safe, understood error) or
    MANUAL_REVIEW (unknown state that a human must resolve).
    """

    CREATED = "created"
    BUY_SUBMITTED = "buy_submitted"
    BUY_FILLED = "buy_filled"
    WITHDRAW_SUBMITTED = "withdraw_submitted"
    WITHDRAW_PENDING = "withdraw_pending"
    TRANSFER_IN_PROGRESS = "transfer_in_progress"
    DEPOSIT_DETECTED = "deposit_detected"
    SELL_SUBMITTED = "sell_submitted"
    COMPLETED = "completed"
    FAILED = "failed"
    MANUAL_REVIEW = "manual_review"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_TRANSFER_STATES

    @property
    def is_open(self) -> bool:
        return not self.is_terminal


_TERMINAL_TRANSFER_STATES = frozenset(
    {TransferState.COMPLETED, TransferState.FAILED, TransferState.MANUAL_REVIEW}
)


class ArbitrageStrategy(StrEnum):
    """The only two strategies this bot implements."""

    TRIANGLE = "triangle"
    TRANSFER = "transfer"


class OpportunityStatus(StrEnum):
    DETECTED = "detected"
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTING = "executing"
    COMPLETED = "completed"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StreamKind(StrEnum):
    TICKER = "ticker"
    ORDER_BOOK = "order_book"


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

    @property
    def severity(self) -> int:
        return _HEALTH_SEVERITY[self]

    @classmethod
    def worst(cls, statuses: Iterable[HealthStatus]) -> HealthStatus:
        """Aggregate several statuses into the most severe one."""
        items = list(statuses)
        if not items:
            return cls.UNKNOWN
        return max(items, key=lambda status: status.severity)


_TERMINAL_ORDER_STATUSES = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
        OrderStatus.TIMEOUT,
        # No further automated action is taken on a MANUAL_REVIEW order.
        OrderStatus.MANUAL_REVIEW,
    }
)

#: Statuses that make a venue-reported ``filled_amount`` real fill evidence
#: (an order the venue itself terminated — the amount will not grow).
_CONFIRMED_FILL_STATUSES = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
        OrderStatus.REJECTED,
    }
)

#: Statuses whose final outcome could NOT be established.  Such orders must
#: never be retried and never treated as rejected — they may have filled.
_UNCONFIRMED_OUTCOME_STATUSES = frozenset(
    {
        OrderStatus.PENDING,
        OrderStatus.OPEN,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.TIMEOUT,
        OrderStatus.UNKNOWN,
        OrderStatus.MANUAL_REVIEW,
    }
)

_HEALTH_SEVERITY: dict[HealthStatus, int] = {
    HealthStatus.HEALTHY: 0,
    HealthStatus.UNKNOWN: 1,
    HealthStatus.DEGRADED: 2,
    HealthStatus.UNHEALTHY: 3,
}
