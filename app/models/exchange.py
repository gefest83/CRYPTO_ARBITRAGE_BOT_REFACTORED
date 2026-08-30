"""Exchange domain models: venue entries, capabilities and health."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator

from app.models.base import DomainModel, utc_now
from app.models.enums import ExchangeStatus

__all__ = ["Exchange", "ExchangeCapabilities", "ExchangeHealth"]


class ExchangeCapabilities(DomainModel):
    """What an adapter can actually do (declared, spot-only surface)."""

    spot: bool = False
    fetch_markets: bool = False
    fetch_ticker: bool = False
    fetch_order_book: bool = False
    fetch_trading_fees: bool = False
    fetch_balance: bool = False
    create_order: bool = False
    cancel_order: bool = False
    withdraw: bool = False
    deposit_address: bool = False
    fetch_deposits: bool = False
    fetch_withdrawals: bool = False
    fetch_withdrawal_networks: bool = False
    watch_ticker: bool = False
    watch_order_book: bool = False

    @property
    def is_market_data_ready(self) -> bool:
        return self.fetch_markets and self.fetch_order_book and self.fetch_ticker

    @property
    def is_trading_ready(self) -> bool:
        return self.create_order and self.cancel_order and self.fetch_balance

    @property
    def is_transfer_ready(self) -> bool:
        return self.fetch_withdrawal_networks and self.deposit_address


class Exchange(DomainModel):
    """A venue in the bot universe (binance / okx / bybit)."""

    id: str
    name: str
    adapter: str = "ccxt"
    enabled: bool = True
    status: ExchangeStatus = ExchangeStatus.UNKNOWN
    capabilities: ExchangeCapabilities = ExchangeCapabilities()
    has_credentials: bool = False
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("id", mode="before")
    @classmethod
    def _normalise_id(cls, value: str) -> str:
        return str(value).strip().lower()

    @property
    def is_usable(self) -> bool:
        return self.enabled and self.status not in (
            ExchangeStatus.OFFLINE,
            ExchangeStatus.DISABLED,
            ExchangeStatus.MAINTENANCE,
        )

    def with_status(self, status: ExchangeStatus) -> Exchange:
        return self.model_copy(update={"status": status, "updated_at": utc_now()})

    def with_enabled(self, enabled: bool) -> Exchange:
        return self.model_copy(update={"enabled": enabled, "updated_at": utc_now()})

    def with_capabilities(self, capabilities: ExchangeCapabilities) -> Exchange:
        return self.model_copy(update={"capabilities": capabilities, "updated_at": utc_now()})


class ExchangeHealth(DomainModel):
    """Rolling health snapshot of one venue."""

    exchange_id: str
    status: ExchangeStatus = ExchangeStatus.UNKNOWN
    rest_latency_ms: float | None = None
    ws_latency_ms: float | None = None
    consecutive_failures: int = 0
    last_error: str | None = None
    checked_at: datetime = Field(default_factory=utc_now)

    @property
    def accepts_new_trades(self) -> bool:
        return self.status in (ExchangeStatus.ONLINE, ExchangeStatus.DEGRADED)
