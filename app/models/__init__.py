"""Domain models shared across the application."""

from app.models.arbitrage import ArbitrageLeg, ArbitrageOpportunity, ProfitBreakdown
from app.models.balance import Balance, BalanceSnapshot
from app.models.base import DEC0, DEC1, DomainModel, as_decimal, utc_now
from app.models.enums import (
    ArbitrageStrategy,
    ExchangeStatus,
    HealthStatus,
    MarketType,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskLevel,
    StreamKind,
    TimeInForce,
    TradeStatus,
    TradingMode,
    TransferState,
)
from app.models.exchange import Exchange, ExchangeCapabilities, ExchangeHealth
from app.models.health import ComponentHealth, HealthReport, StreamStatus
from app.models.market import Market, MarketFees, MarketLimits, MarketPrecision
from app.models.market_data import (
    ExecutionEstimate,
    OrderBook,
    OrderBookLevel,
    Ticker,
    Trade,
)
from app.models.order import Fill, Order, OrderRequest
from app.models.risk import RiskAssessment, RiskLimits, RiskViolation
from app.models.symbol import Symbol
from app.models.trade import TradeRecord
from app.models.transfer import TransferPlan, TransferRecord

__all__ = [
    "DEC0",
    "DEC1",
    "ArbitrageLeg",
    "ArbitrageOpportunity",
    "ArbitrageStrategy",
    "Balance",
    "BalanceSnapshot",
    "ComponentHealth",
    "DomainModel",
    "Exchange",
    "ExchangeCapabilities",
    "ExchangeHealth",
    "ExchangeStatus",
    "ExecutionEstimate",
    "Fill",
    "HealthReport",
    "HealthStatus",
    "Market",
    "MarketData",
    "MarketFees",
    "MarketLimits",
    "MarketPrecision",
    "MarketType",
    "Order",
    "OrderBook",
    "OrderBookLevel",
    "OrderRequest",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "ProfitBreakdown",
    "RiskAssessment",
    "RiskLevel",
    "RiskLimits",
    "RiskViolation",
    "StreamKind",
    "StreamStatus",
    "Symbol",
    "Ticker",
    "TimeInForce",
    "Trade",
    "TradeRecord",
    "TradeStatus",
    "TradingMode",
    "TransferPlan",
    "TransferRecord",
    "TransferState",
    "as_decimal",
    "utc_now",
]
