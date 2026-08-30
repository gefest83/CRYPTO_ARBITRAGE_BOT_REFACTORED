"""ExchangeManager: runtime owner of the fixed venue universe.

Responsibilities (deliberately small):

* build one adapter per enabled venue (binance / okx / bybit), lazily, with
  per-venue locking and failure isolation;
* PAPER mode uses the deterministic simulated adapter by default;
* network circuit breaker per venue: after N consecutive failures the venue
  is skipped for a cooldown, then allowed one probe (half-open);
* auth gate: a venue that rejects our keys becomes DATA_ONLY — public market
  data keeps flowing, private calls stop — and never trips the breaker.
"""

from __future__ import annotations

from typing import Any

from app.clock import SystemClock
from app.config.logging_config import get_logger
from app.config.settings import ExchangeSettings, Settings
from app.errors import ConfigurationError
from app.models.enums import ExchangeStatus, TradingMode
from app.models.exchange import Exchange, ExchangeHealth

from .base import AdapterOptions, BaseExchangeAdapter, OrderGate
from .ccxt_adapter import CCXTAdapter
from .credentials import CredentialsProvider, EnvCredentialsProvider
from .profiles import resolve_profile
from .sanitize import redact_secrets
from .simulated import SimulatedExchangeAdapter

__all__ = ["ExchangeManager"]

logger = get_logger("exchanges.manager")


class ExchangeManager:
    """Owns adapter instances for the configured venues."""

    def __init__(
        self,
        settings: Settings,
        *,
        credentials: CredentialsProvider | None = None,
        order_gate: OrderGate | None = None,
    ) -> None:
        self._settings = settings
        self._exchanges_cfg: ExchangeSettings = settings.exchanges
        self._credentials = credentials or EnvCredentialsProvider()
        self._order_gate = order_gate
        self._clock = SystemClock()

        self._exchanges: dict[str, Exchange] = {}
        self._adapters: dict[str, BaseExchangeAdapter] = {}
        self._health: dict[str, ExchangeHealth] = {}

        self._breaker_failures: dict[str, int] = {}
        self._breaker_opened_at_ms: dict[str, int] = {}
        self._auth_blocked: dict[str, str] = {}

        for venue_id in self._exchanges_cfg.enabled:
            profile = resolve_profile(venue_id)  # fail-closed on unknown ids
            simulated = self._use_simulated()
            self._exchanges[profile.id] = Exchange(
                id=profile.id,
                name=profile.display_name,
                adapter="simulated" if simulated else "ccxt",
                # Simulated venues need no keys; real adapters report the truth.
                has_credentials=True if simulated else self._credentials.has(profile.id),
            )

    # ---------------------------------------------------------------- universe
    def _use_simulated(self) -> bool:
        return self._settings.mode is TradingMode.PAPER and self._exchanges_cfg.simulate_in_paper

    @property
    def exchanges(self) -> tuple[Exchange, ...]:
        return tuple(self._exchanges.values())

    def enabled_ids(self) -> tuple[str, ...]:
        return tuple(self._exchanges)

    def exchange(self, exchange_id: str) -> Exchange | None:
        return self._exchanges.get(exchange_id.strip().lower())

    def adapter(self, exchange_id: str) -> BaseExchangeAdapter:
        """Lazily created, cached adapter; failure marks the venue offline."""
        key = exchange_id.strip().lower()
        if key not in self._exchanges:
            raise ConfigurationError(f"exchange '{exchange_id}' is not enabled", exchange_id=key)
        cached = self._adapters.get(key)
        if cached is not None:
            return cached
        return self._build_adapter(key)

    def _build_adapter(self, key: str) -> BaseExchangeAdapter:
        exchange = self._exchanges[key]
        # DEMO routes the ccxt client to the venue's demo/testnet environment;
        # LIVE talks to production endpoints; PAPER never places orders at all.
        options = AdapterOptions(
            timeout_seconds=self._exchanges_cfg.request_timeout_seconds,
            order_book_depth=self._exchanges_cfg.order_book_depth,
            sandbox=self._settings.mode is TradingMode.DEMO,
            enable_rate_limit=True,
            quote_currencies=self._exchanges_cfg.quote_currencies,
        )
        credentials = self._credentials.get(key)
        if self._use_simulated():
            adapter: BaseExchangeAdapter = SimulatedExchangeAdapter(
                exchange, credentials=None, options=options, order_gate=self._order_gate
            )
        else:
            adapter = CCXTAdapter(
                exchange,
                credentials=credentials,
                options=options,
                order_gate=self._order_gate,
                allow_withdrawals=(
                    self._settings.mode is TradingMode.LIVE
                    and self._settings.trading.allow_live_withdrawals
                ),
            )
        self._adapters[key] = adapter
        logger.info(
            "exchange_adapter_created",
            extra={"exchange_id": key, "adapter": adapter.adapter_name},
        )
        return adapter

    async def open_all(self) -> None:
        """Open every enabled adapter; failures are isolated per venue."""
        for venue_id in self.enabled_ids():
            try:
                adapter = self.adapter(venue_id)
                await adapter.open()
                capabilities = adapter.capabilities
                self._exchanges[venue_id] = (
                    self._exchanges[venue_id]
                    .with_capabilities(capabilities)
                    .with_status(ExchangeStatus.ONLINE)
                )
            except Exception as exc:  # noqa: BLE001 - one dead venue must not sink startup
                self._mark_offline(self._exchanges[venue_id], str(exc))

    async def close(self) -> None:
        errors: list[str] = []
        for venue_id, adapter in list(self._adapters.items()):
            try:
                await adapter.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must stay quiet
                errors.append(f"{venue_id}: {exc}")
        self._adapters.clear()
        if errors:
            logger.warning("exchange_close_failed", extra={"errors": "; ".join(errors)})

    async def health_all(self) -> dict[str, ExchangeHealth]:
        """Probe every open adapter; never raises."""
        for venue_id in self.enabled_ids():
            adapter = self._adapters.get(venue_id)
            if adapter is None:
                continue
            try:
                health = await adapter.health()
            except Exception as exc:  # noqa: BLE001 - health must never raise
                health = ExchangeHealth(
                    exchange_id=venue_id,
                    status=ExchangeStatus.OFFLINE,
                    last_error=redact_secrets(str(exc))[:300],
                )
            self._health[venue_id] = health
        return dict(self._health)

    # ---------------------------------------------------------------- breaker
    def record_success(self, exchange_id: str) -> None:
        key = exchange_id.strip().lower()
        self._breaker_failures.pop(key, None)
        self._breaker_opened_at_ms.pop(key, None)

    def record_failure(self, exchange_id: str) -> None:
        """One hard failure; N consecutive failures open the breaker."""
        key = exchange_id.strip().lower()
        self._breaker_failures[key] = self._breaker_failures.get(key, 0) + 1
        if self._breaker_failures[key] >= self._exchanges_cfg.breaker_failure_threshold:
            self._breaker_opened_at_ms[key] = self._clock.monotonic_ms()
            logger.warning(
                "venue_breaker_open",
                extra={
                    "exchange_id": key,
                    "failures": self._breaker_failures[key],
                    "cooldown_seconds": self._exchanges_cfg.breaker_cooldown_seconds,
                },
            )

    def record_rate_limit(self, exchange_id: str, retry_after_seconds: float | None) -> None:
        """RATE class: open the breaker for max(cooldown, Retry-After)."""
        key = exchange_id.strip().lower()
        cooldown = float(self._exchanges_cfg.breaker_cooldown_seconds)
        wait = max(cooldown, float(retry_after_seconds or 0.0))
        self._breaker_opened_at_ms[key] = self._clock.monotonic_ms()
        # Rate-limit trips are transient: keep the counter below the threshold
        # so a single 429 does not look like repeated hard failures.
        self._breaker_failures[key] = max(self._exchanges_cfg.breaker_failure_threshold - 1, 1)
        logger.warning(
            "venue_rate_limited",
            extra={"exchange_id": key, "retry_after_seconds": round(wait, 3)},
        )

    def is_breaker_open(self, exchange_id: str) -> bool:
        """Whether calls to this venue are currently skipped.

        Half-open semantics: after the cooldown elapses the venue is allowed
        again — a further failure re-opens it with a fresh window, a success
        clears the counter completely.
        """
        key = exchange_id.strip().lower()
        opened_at = self._breaker_opened_at_ms.get(key)
        if opened_at is None:
            return False
        cooldown_ms = int(self._exchanges_cfg.breaker_cooldown_seconds * 1000)
        if self._clock.monotonic_ms() - opened_at >= cooldown_ms:
            self._breaker_opened_at_ms.pop(key, None)
            return False
        return True

    # ---------------------------------------------------------------- auth gate
    def record_auth_failure(self, exchange_id: str, reason: str) -> None:
        """The venue rejected our keys: stop private calls immediately.

        This never touches the network breaker and never marks the venue
        OFFLINE — public data keeps flowing; the venue shows DATA_ONLY.
        """
        key = exchange_id.strip().lower()
        first_time = key not in self._auth_blocked
        self._auth_blocked[key] = reason
        self._breaker_failures.pop(key, None)
        logger.warning(
            "venue_auth_blocked",
            extra={"exchange_id": key, "reason": redact_secrets(reason), "first_time": first_time},
        )
        if key in self._exchanges:
            self._exchanges[key] = self._exchanges[key].with_status(ExchangeStatus.DATA_ONLY)

    def record_private_success(self, exchange_id: str) -> None:
        """A private call succeeded: the keys work again — lift the auth block."""
        key = exchange_id.strip().lower()
        if self._auth_blocked.pop(key, None) is not None:
            logger.info("venue_auth_cleared", extra={"exchange_id": key})
            if key in self._exchanges and self._exchanges[key].status is ExchangeStatus.DATA_ONLY:
                self._exchanges[key] = self._exchanges[key].with_status(ExchangeStatus.ONLINE)

    def is_private_blocked(self, exchange_id: str) -> bool:
        """Whether private calls for this venue are stopped (keys rejected)."""
        return exchange_id.strip().lower() in self._auth_blocked

    # ---------------------------------------------------------------- status
    def _mark_offline(self, exchange: Exchange, error: str) -> None:
        safe_error = redact_secrets(error)[:300]
        self._exchanges[exchange.id] = exchange.with_status(ExchangeStatus.OFFLINE)
        previous = self._health.get(exchange.id)
        self._health[exchange.id] = ExchangeHealth(
            exchange_id=exchange.id,
            status=ExchangeStatus.OFFLINE,
            last_error=safe_error,
            consecutive_failures=previous.consecutive_failures + 1 if previous else 1,
        )
        logger.warning("exchange_offline", extra={"exchange_id": exchange.id, "detail": safe_error})

    def status_snapshot(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for venue_id, exchange in self._exchanges.items():
            out[venue_id] = {
                "name": exchange.name,
                "adapter": exchange.adapter,
                "status": exchange.status.value,
                "credentials": "set" if exchange.has_credentials else "empty",
                "breaker_open": self.is_breaker_open(venue_id),
                "private_blocked": self.is_private_blocked(venue_id),
                "spot": exchange.capabilities.spot,
            }
        return out
