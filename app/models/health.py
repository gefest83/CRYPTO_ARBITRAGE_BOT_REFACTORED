"""Health-check models used by the status command and Telegram /status."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.models.base import DomainModel, utc_now
from app.models.enums import HealthStatus, StreamKind, TradingMode

__all__ = ["ComponentHealth", "HealthReport", "StreamStatus"]


class ComponentHealth(DomainModel):
    """Health of one subsystem (database, exchange adapter, market data, ...)."""

    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str | None = None
    latency_ms: float | None = None
    checked_at: datetime = Field(default_factory=utc_now)
    details: dict[str, str] = Field(default_factory=dict)

    @property
    def is_operational(self) -> bool:
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


class HealthReport(DomainModel):
    """Aggregated bot health."""

    status: HealthStatus = HealthStatus.UNKNOWN
    version: str = "0.0.0"
    mode: TradingMode = TradingMode.PAPER
    uptime_seconds: float = 0.0
    components: tuple[ComponentHealth, ...] = ()
    generated_at: datetime = Field(default_factory=utc_now)

    @property
    def is_healthy(self) -> bool:
        return self.status is HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def component(self, name: str) -> ComponentHealth | None:
        return next((c for c in self.components if c.name == name), None)


class StreamStatus(DomainModel):
    """Status of a single supervised watch stream."""

    exchange_id: str
    symbol: str
    kind: StreamKind
    market_type: str = "spot"
    status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0
    last_event_at: datetime | None = None
    last_error: str | None = None
    restarted_at: datetime | None = None
    total_restarts: int = 0

    @property
    def is_healthy(self) -> bool:
        return self.status is HealthStatus.HEALTHY

    @property
    def is_degraded(self) -> bool:
        return self.status is HealthStatus.DEGRADED
