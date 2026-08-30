"""Exchange layer: unified adapter interface for Binance, OKX and Bybit."""

from app.exchanges.base import (
    AdapterOptions,
    BaseExchangeAdapter,
    ExchangeAdapter,
    OrderGate,
)
from app.exchanges.credentials import (
    CredentialsProvider,
    EnvCredentialsProvider,
    ExchangeCredentials,
    InMemoryCredentialsProvider,
)
from app.exchanges.manager import ExchangeManager
from app.exchanges.registry import AdapterRegistry, AdapterRequest

__all__ = [
    "AdapterOptions",
    "AdapterRegistry",
    "AdapterRequest",
    "BaseExchangeAdapter",
    "CredentialsProvider",
    "EnvCredentialsProvider",
    "ExchangeAdapter",
    "ExchangeCredentials",
    "ExchangeManager",
    "InMemoryCredentialsProvider",
    "OrderGate",
]
