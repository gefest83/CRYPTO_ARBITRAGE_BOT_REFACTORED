"""Application settings.

All values come from environment variables / ``.env`` with the ``CAT_`` prefix
and ``__`` as the nesting delimiter, e.g. ``CAT_TRADING__MODE=DEMO``.

Design notes
------------
* Trading mode (PAPER/DEMO/LIVE) lives here; the *policy* of what each mode is
  allowed to do lives in :mod:`app.config.modes`.
* Secrets are wrapped in :class:`~pydantic.SecretStr` so they never leak into
  logs or terminal output.
* LIVE mode requires three explicit opt-ins (mode, allow_live, confirmation
  phrase) and fails configuration loading otherwise.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.errors import ConfigurationError
from app.models.enums import TradingMode
from app.models.risk import RiskLimits

__all__ = [
    "AppSettings",
    "ArbitrageSettings",
    "DatabaseSettings",
    "ExchangeSettings",
    "ExecutionSettings",
    "LoggingSettings",
    "MarketDataSettings",
    "RiskSettings",
    "Settings",
    "TelegramSettings",
    "TradingSettings",
    "TransferSettings",
    "get_settings",
    "reload_settings",
]

LIVE_CONFIRMATION_PHRASE = "I UNDERSTAND THE RISK"

#: Single source of the risk-limit defaults (see :class:`RiskSettings`).
_RISK_DEFAULTS = RiskLimits()

#: Assets whose ``*/USDT`` pairs (and cross pairs between them) are watched.
DEFAULT_TRIANGLE_ASSETS: tuple[str, ...] = (
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "ADA",
    "DOGE",
    "LINK",
    "AVAX",
    "TRX",
)

#: Assets considered for transfer arbitrage by default.
#: Top 50 by liquidity (market-cap / volume) where spot trading + withdrawal
#: networks exist on Binance, OKX and Bybit.  The runtime still filters at
#: plan time via `load_markets` / `fetch_withdrawal_networks` – this tuple is
#: the *candidate* universe, not a guarantee of availability on every venue.
DEFAULT_TRANSFER_ASSETS: tuple[str, ...] = (
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "ADA",
    "DOGE",
    "LINK",
    "AVAX",
    "TRX",
    "LTC",
    "DOT",
    "BCH",
    "UNI",
    "ATOM",
    "ETC",
    "XLM",
    "FIL",
    "HBAR",
    "APT",
    "NEAR",
    "ARB",
    "OP",
    "SUI",
    "TAO",
    "RNDR",
    "MATIC",
    "TON",
    "SHIB",
    "PEPE",
    "AAVE",
    "ENA",
    "WIF",
    "FLOKI",
    "BONK",
    "TIA",
    "INJ",
    "IMX",
    "MKR",
    "GRT",
    "STX",
    "FET",
    "AR",
    "SEI",
    "JUP",
    "ONDO",
    "STRK",
    "WLD",
    "PYTH",
    "JTO",
)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AppSettings(_Section):
    name: str = "Crypto Arbitrage Bot"
    environment: Literal["development", "production"] = "development"
    data_dir: Path = Path("data")


class TradingSettings(_Section):
    """PAPER / DEMO / LIVE selection plus the LIVE safety interlocks."""

    mode: TradingMode = TradingMode.PAPER
    allow_live: bool = False
    live_confirmation: SecretStr = SecretStr("")
    #: LIVE withdrawals (transfer arbitrage) need their own explicit opt-in on
    #: top of LIVE trading itself.
    allow_live_withdrawals: bool = False
    base_currency: str = "USDT"

    @field_validator("base_currency")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def _validate_live(self) -> TradingSettings:
        if self.mode is not TradingMode.LIVE:
            if self.allow_live:
                raise ValueError("CAT_TRADING__ALLOW_LIVE=true is only meaningful with MODE=LIVE")
            return self
        if not self.allow_live:
            raise ValueError("LIVE mode requires CAT_TRADING__ALLOW_LIVE=true (explicit opt-in)")
        if self.live_confirmation.get_secret_value().strip().upper() != LIVE_CONFIRMATION_PHRASE:
            raise ValueError(
                f"LIVE mode requires CAT_TRADING__LIVE_CONFIRMATION='{LIVE_CONFIRMATION_PHRASE}'"
            )
        return self


class DatabaseSettings(_Section):
    url: str = "sqlite+aiosqlite:///./data/bot.db"
    echo: bool = False
    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=10, ge=0)
    connect_timeout_seconds: float = Field(default=10.0, gt=0)
    trade_retention_days: int = Field(default=90, ge=0)
    audit_retention_days: int = Field(default=90, ge=0)
    #: ``create_all`` on startup (SQLite). The bot's schema is small and owned
    #: end-to-end, so this is the default everywhere; production operators who
    #: prefer explicit control may disable it and manage the schema themselves.
    auto_create_schema: bool = True

    @property
    def is_sqlite(self) -> bool:
        return self.url.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        return self.url.startswith("postgresql")

    @property
    def safe_url(self) -> str:
        """DSN without credentials — the only form allowed to leave the process."""
        parsed = urlsplit(self.url)
        if not parsed.netloc:
            return self.url
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        user = f"{parsed.username}:***@" if parsed.username else ""
        return urlunsplit((parsed.scheme, f"{user}{host}", parsed.path, "", ""))


class ExchangeSettings(_Section):
    """Fixed three-venue universe plus transport behaviour."""

    #: The only supported venues; resolved against the venue profiles.
    enabled: tuple[str, ...] = ("binance", "okx", "bybit")
    quote_currencies: tuple[str, ...] = ("USDT",)
    default_adapter: str = "ccxt"
    #: PAPER mode runs on the deterministic simulated adapter by default, so
    #: the bot is fully usable offline and without API keys.
    simulate_in_paper: bool = True
    request_timeout_seconds: float = Field(default=10.0, gt=0)
    order_book_depth: int = Field(default=25, ge=1, le=1000)
    #: Circuit breaker: after this many consecutive failed calls a venue is
    #: skipped entirely until the cooldown elapses (then allowed to retry).
    breaker_failure_threshold: int = Field(default=5, ge=1)
    breaker_cooldown_seconds: float = Field(default=60.0, ge=0)

    @field_validator("enabled", "quote_currencies", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("enabled", mode="after")
    @classmethod
    def _normalise_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(v.strip().lower() for v in values if v.strip()))

    @field_validator("quote_currencies", mode="after")
    @classmethod
    def _normalise_quotes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(v.strip().upper() for v in values if v.strip()))


class RiskSettings(_Section):
    """Environment overrides for :class:`~app.models.risk.RiskLimits`.

    Defaults are taken from the domain model, so the numbers exist in exactly
    one place and the two copies cannot drift.
    """

    max_trade_size: Decimal = _RISK_DEFAULTS.max_trade_size
    min_net_profit_bps: Decimal = _RISK_DEFAULTS.min_net_profit_bps
    max_daily_loss: Decimal = _RISK_DEFAULTS.max_daily_loss
    max_open_transfers: int = Field(default=_RISK_DEFAULTS.max_open_transfers, ge=0)
    max_exchange_exposure: Decimal = _RISK_DEFAULTS.max_exchange_exposure
    max_asset_exposure: Decimal = _RISK_DEFAULTS.max_asset_exposure
    max_slippage_bps: Decimal = _RISK_DEFAULTS.max_slippage_bps
    max_data_age_ms: int = Field(default=_RISK_DEFAULTS.max_data_age_ms, ge=100)

    def to_limits(self) -> RiskLimits:
        """Project settings onto the domain model used by the risk engine."""
        return RiskLimits(**self.model_dump())


class MarketDataSettings(_Section):
    stale_after_ms: int = Field(default=2000, ge=100)
    #: Concurrent in-flight REST requests per venue.
    max_requests_per_exchange: int = Field(default=4, ge=1, le=64)
    #: Concurrent in-flight REST requests across all venues.
    max_parallel_requests: int = Field(default=16, ge=1, le=256)

    # WebSocket stream supervision
    streams_enabled: bool = True
    max_streams_per_exchange: int = Field(default=48, ge=1, le=128)
    max_parallel_streams: int = Field(default=256, ge=1, le=1024)
    stream_backoff_base_seconds: float = Field(default=0.5, gt=0)
    stream_backoff_max_seconds: float = Field(default=30.0, gt=0)
    stream_backoff_jitter: float = Field(default=0.25, ge=0, le=1)
    #: After this many restarts without a single delivered event the stream is
    #: suspended and REST refresh takes over its symbols (fail-safe default).
    #: ``0`` disables quarantine (retry forever).
    stream_quarantine_restarts: int = Field(default=5, ge=0)


class ArbitrageSettings(_Section):
    """Triangular arbitrage scanning configuration."""

    enable_triangular: bool = True
    #: Assets that may start/end a triangle (USDT→X→Y→USDT).
    triangle_assets: tuple[str, ...] = DEFAULT_TRIANGLE_ASSETS
    #: Default notional (quote) for scanning.
    default_notional_quote: Decimal = Field(default=Decimal("1000"), gt=0)
    #: Minimum net cycle profit (bps) after fees and slippage.
    triangle_min_net_bps: Decimal = Field(default=Decimal("5"), ge=0)
    #: Per-leg VWAP slippage tolerance (bps); a leg above it kills the ring.
    triangle_max_leg_slippage_bps: Decimal = Field(default=Decimal("15"), ge=0)
    #: Notional quote spent on the first leg of every cycle.
    triangle_max_notional_quote: Decimal = Field(default=Decimal("500"), gt=0)
    #: Max results per scan.
    max_results: int = Field(default=20, ge=1, le=200)
    #: Scan TTL in milliseconds.
    scan_ttl_ms: int = Field(default=1500, ge=100)
    #: Require fresh data for scanning.
    require_fresh_data: bool = True

    @field_validator("triangle_assets", mode="before")
    @classmethod
    def _split_triangle_assets(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("triangle_assets", mode="after")
    @classmethod
    def _normalise_assets(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(v.strip().upper() for v in values if v.strip()))


class TransferSettings(_Section):
    """Transfer arbitrage configuration."""

    #: Assets considered when scanning for transfer opportunities.
    assets: tuple[str, ...] = DEFAULT_TRANSFER_ASSETS
    #: Minimum net profit (bps) for a transfer plan to be executable.
    min_net_profit_bps: Decimal = Field(default=Decimal("50"), ge=0)
    #: Default amount (in the transferred asset) when not specified explicitly.
    #: Kept for explicit `amount` overrides; the auto-sizer uses notional below.
    default_amount: Decimal = Field(default=Decimal("1"), gt=0)
    #: Upper bound for amount when auto-sizing from a quote budget.
    max_amount: Decimal = Field(default=Decimal("10000"), gt=0)
    #: Minimum quote budget spent on the buy leg of one transfer.
    min_notional_quote: Decimal = Field(default=Decimal("100"), gt=0)
    #: Max quote budget spent on the buy leg of one transfer.
    max_notional_quote: Decimal = Field(default=Decimal("500"), gt=0)
    #: Maximum allowed gross price divergence between source and
    #: destination venues (bps).  `gross = (sell_bid - buy_ask)/buy_ask*10000`.
    #: Rejects absurd cross-venue quotes (e.g. STRK 0.026 vs 312).
    max_transfer_gross_divergence_bps: Decimal = Field(
        default=Decimal("5000"), ge=0
    )
    #: How long to wait for a deposit before flagging MANUAL_REVIEW (seconds).
    deposit_timeout_seconds: int = Field(default=3600, ge=60)
    #: Poll interval while tracking an in-flight transfer (seconds).
    poll_interval_seconds: float = Field(default=30.0, ge=1)
    #: PAPER/DEMO: simulated blockchain confirmation delay (seconds).
    simulated_transfer_seconds: float = Field(default=30.0, ge=0)

    @field_validator("assets", mode="before")
    @classmethod
    def _split_assets(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("assets", mode="after")
    @classmethod
    def _normalise_assets(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(v.strip().upper() for v in values if v.strip()))


class ExecutionSettings(_Section):
    """Execution engine configuration."""

    #: PAPER execution is always available; DEMO/LIVE additionally require the
    #: trading-mode gates (see app.config.modes and app.execution.order_gate).
    leg_timeout_seconds: int = Field(default=10, ge=1, le=60)
    #: Maximum slippage tolerance for simulated (PAPER) fills (bps).
    paper_max_slippage_bps: int = Field(default=50, ge=0, le=500)
    #: Minimum net profit (bps) re-verified at execution time; below it the
    #: trade is abandoned before any leg is placed.
    min_net_at_execute_bps: int = Field(default=0, ge=0, le=1000)
    #: How many opportunities the auto trader may execute per cycle.
    auto_max_per_cycle: int = Field(default=1, ge=1, le=10)
    #: Pause between auto-trading cycles (seconds).
    auto_interval_seconds: float = Field(default=5.0, ge=0.5)
    #: Auto trader scan-notional (quote) used for triangle scanning.
    auto_notional_quote: Decimal = Field(default=Decimal("1000"), gt=0)


class TelegramSettings(_Section):
    """Telegram bot configuration (secondary control interface)."""

    #: Bot token from @BotFather; empty disables the Telegram interface.
    bot_token: SecretStr = SecretStr("")
    #: Telegram user IDs allowed to control the bot. Empty list = bot refuses
    #: every command (fail-closed: a public bot must never accept strangers).
    allowed_user_ids: tuple[int, ...] = ()
    #: Legacy chat-id allow-list, retained for back-compat. The bot accepts
    #: commands from any of these chats OR any of the user IDs in
    #: :attr:`allowed_user_ids`.
    allowed_chat_ids: tuple[int, ...] = ()
    poll_timeout_seconds: int = Field(default=30, ge=1, le=120)

    @field_validator("allowed_user_ids", "allowed_chat_ids", mode="before")
    @classmethod
    def _parse_allowed_ids(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token.get_secret_value().strip())

    @property
    def has_any_operator(self) -> bool:
        """True when at least one user ID or chat ID may command the bot."""
        return bool(self.allowed_user_ids or self.allowed_chat_ids)


class LoggingSettings(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_logs: bool = False
    file: Path | None = None

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, value: str) -> str:
        return str(value).strip().upper()


class Settings(BaseSettings):
    """Root settings object; build via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="CAT_",
        env_nested_delimiter="__",
        # Project-local .env only: never reach outside the repository for
        # configuration (a stray parent-directory .env must not leak in).
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
    )

    app: AppSettings = AppSettings()
    trading: TradingSettings = TradingSettings()
    database: DatabaseSettings = DatabaseSettings()
    exchanges: ExchangeSettings = ExchangeSettings()
    risk: RiskSettings = RiskSettings()
    market_data: MarketDataSettings = MarketDataSettings()
    arbitrage: ArbitrageSettings = ArbitrageSettings()
    transfer: TransferSettings = TransferSettings()
    execution: ExecutionSettings = ExecutionSettings()
    telegram: TelegramSettings = TelegramSettings()
    logging: LoggingSettings = LoggingSettings()

    @property
    def mode(self) -> TradingMode:
        return self.trading.mode

    @model_validator(mode="after")
    def _validate_production(self) -> Settings:
        """Fail fast on configurations that are unsafe outside development."""
        if self.app.environment != "production":
            return self
        if self.database.is_sqlite:
            raise ValueError("production requires PostgreSQL, not SQLite")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor (single source of truth for the process)."""
    try:
        return Settings()
    except Exception as exc:  # pragma: no cover - surfaced as a fatal config error
        raise ConfigurationError(f"invalid configuration: {exc}") from exc


def reload_settings() -> Settings:
    """Drop the cache and re-read the environment (used by tests)."""
    get_settings.cache_clear()
    return get_settings()
