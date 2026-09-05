"""CCXT-backed adapter for Binance, OKX and Bybit.

One adapter class serves every ccxt venue: the exchange id selects the ccxt
class at runtime through :mod:`.profiles`, capabilities are detected from
``exchange.has``, and everything is translated into the bot's own domain
models.

DEMO routing (fail-closed by construction):

* ``binance`` — dedicated demo endpoints (``urls['demo']``); ``set_sandbox_mode``
  is NOT used (it would point at the separate testnet account system).
* ``okx`` — the ``x-simulated-trading: 1`` header on private requests.
* ``bybit`` — demo host (api-demo.bybit.com) serves private calls only; public
  market data stays on production.
* unknown/unexpected ccxt URL tables refuse to start rather than silently
  routing signed calls to production.

``ccxt`` is imported lazily so neither the core nor the tests require the
package to be installed.  WebSocket streaming uses ``ccxt.pro`` when available
and degrades to REST polling otherwise.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from app.config.logging_config import get_logger
from app.errors import (
    CapabilityNotSupportedError,
    ConfigurationError,
    ExchangeError,
    ExchangeUnavailableError,
    ExecutionDisabledError,
    RateLimitError,
    TimeoutError,
    VenueAuthError,
)
from app.models.balance import Balance, BalanceSnapshot
from app.models.base import DEC0, utc_now
from app.models.enums import (
    ExchangeStatus,
    MarketType,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from app.models.exchange import Exchange, ExchangeCapabilities, ExchangeHealth
from app.models.market import Market, MarketFees, MarketLimits, MarketPrecision
from app.models.market_data import OrderBook, OrderBookLevel, Ticker
from app.models.order import Fill, Order, OrderRequest
from app.models.symbol import Symbol
from app.models.transfer import DepositAddress, TransferTx, WithdrawalNetwork

from .base import AdapterOptions, BaseExchangeAdapter, OrderGate
from .credentials import ExchangeCredentials
from .profiles import VenueProfile, ccxt_candidate_ids, resolve_profile
from .sanitize import REDACTED, redact_secrets

__all__ = [
    "CCXTAdapter",
    "build_ccxt_adapter",
    "classify_ccxt_exception",
    "installed_ccxt_ids",
    "preload_ccxt",
    "reset_ccxt_registry_cache",
    "resolve_ccxt_id",
]

logger = get_logger("exchanges.ccxt")

#: Resolved lazily once per process: importing ccxt pulls in 100+ modules.
_CCXT_MODULE: Any | None = None


def preload_ccxt() -> bool:
    """Import ccxt eagerly (call from a worker thread during startup).

    Returns ``True`` when the module is available.  Doing this at startup keeps
    a ~1 s blocking import out of the trading path.
    """
    try:
        CCXTAdapter._load_module()
    except ExchangeUnavailableError:
        logger.warning("ccxt_not_installed")
        return False
    return True


# --------------------------------------------------------------------------
# Installed-ccxt registry resolution (no hardcoded internal→ccxt renames).
# --------------------------------------------------------------------------
_CCXT_REGISTRY: frozenset[str] | None = None


def installed_ccxt_ids() -> frozenset[str]:
    """Lower-case ids of the *really installed* ccxt module (cached once)."""
    global _CCXT_REGISTRY
    if _CCXT_REGISTRY is None:
        try:
            import ccxt
        except Exception:  # noqa: BLE001 - any registry failure means "unknown"
            _CCXT_REGISTRY = frozenset()
        else:
            _CCXT_REGISTRY = frozenset(
                str(entry).lower() for entry in getattr(ccxt, "exchanges", ())
            )
    return _CCXT_REGISTRY


def reset_ccxt_registry_cache() -> None:
    """Test hook: forget the cached registry snapshot."""
    global _CCXT_REGISTRY
    _CCXT_REGISTRY = None


def resolve_ccxt_id(profile: VenueProfile) -> str:
    """The ccxt class name to build for this venue, checked against reality.

    The exact declared id wins when the installed ccxt ships it; otherwise the
    profile's aliases are tried in declaration order; when none exists the
    venue fails closed instead of being bound to a dead class name.
    """
    installed = installed_ccxt_ids()
    if not installed:
        return profile.ccxt_id
    for candidate in ccxt_candidate_ids(profile):
        if candidate in installed:
            return candidate
    tried = ", ".join(ccxt_candidate_ids(profile))
    raise ConfigurationError(
        f"installed ccxt has none of ({tried}) for venue '{profile.id}'; "
        "update the venue profile or the installed ccxt version",
        exchange_id=profile.id,
    )


_ORDER_STATUSES: Mapping[str, OrderStatus] = {
    "open": OrderStatus.OPEN,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "cancelled": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    # Bybit DEMO raw statuses (CCXT may return these before normalization)
    "new": OrderStatus.OPEN,
    "created": OrderStatus.OPEN,
    "pending": OrderStatus.PENDING,
    "partiallyfilled": OrderStatus.PARTIALLY_FILLED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
}


def classify_ccxt_exception(exc: BaseException) -> type[ExchangeError]:
    """Map a ccxt exception to the bot's error class by MRO class names.

    This function does not import ccxt; it matches on ``__class__.__name__``
    and the MRO chain, so it works without ccxt installed and is fully testable
    with duck-typed exception classes.
    """
    names = [cls.__name__ for cls in type(exc).__mro__]

    if "AuthenticationError" in names or "PermissionDenied" in names:
        return VenueAuthError
    if "InvalidNonce" in names:
        return VenueAuthError
    if "RateLimitExceeded" in names or "DDoSProtection" in names:
        return RateLimitError
    if "RequestTimeout" in names:
        return TimeoutError
    # A symbol the venue does not list is a permanent, symbol-level fact —
    # never evidence of a network outage.
    if "BadSymbol" in names:
        return CapabilityNotSupportedError
    if "NotSupported" in names:
        return CapabilityNotSupportedError
    if "ExchangeNotAvailable" in names or "OnMaintenance" in names:
        return ExchangeUnavailableError
    if "NetworkError" in names:
        return ExchangeUnavailableError
    if "ExchangeError" in names:
        return ExchangeError
    if "BaseError" in names:
        return ExchangeError
    return ExchangeError


def _dec(value: Any, default: Decimal | None = None) -> Decimal | None:
    if value is None or value == "":
        return default
    try:
        return Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError):
        return default


def _venue_error_code(exc: BaseException) -> str:
    """Best-effort venue-side error code from a ccxt exception."""
    for attr in ("code", "errcode", "retCode", "http_status", "statusCode"):
        value = getattr(exc, attr, None)
        if value is not None and str(value).strip():
            return str(value)
    args = getattr(exc, "args", ())
    if args and isinstance(args[0], int | float | str) and str(args[0]).strip():
        token = str(args[0]).split(" ", 1)[0][:24]
        return token
    return ""


def _ts(value: Any) -> datetime:
    if isinstance(value, int | float) and value > 0:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    return utc_now()


#: Configuration keys whose values are credentials (H-11).  The config dict
#: handed to the ccxt constructor is a plain dict as far as ccxt is
#: concerned, but its ``repr``/``str`` never expose the secret values.
_SECRET_CONFIG_KEYS = ("apiKey", "secret", "password", "uid")


class _SecretConfig(dict):
    """A dict whose textual forms redact credential values (H-11).

    ccxt consumes the config through the normal dict API (``deep_extend``
    copies the real values into the client), so authentication behaviour is
    unchanged — but accidentally logging / repr'ing / formatting the config
    structure can no longer leak the secrets it carries.
    """

    def _redacted_items(self):
        return [
            (key, REDACTED if key in _SECRET_CONFIG_KEYS else value)
            for key, value in self.items()
        ]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._redacted_items()!r})"

    def __str__(self) -> str:
        return repr(self)

    def __format__(self, format_spec: str) -> str:
        return format(repr(self), format_spec)


class CCXTAdapter(BaseExchangeAdapter):
    """Adapter that speaks ccxt on behalf of Binance, OKX or Bybit."""

    adapter_name = "ccxt"

    def __init__(
        self,
        exchange: Exchange,
        *,
        credentials: ExchangeCredentials | None = None,
        options: AdapterOptions | None = None,
        order_gate: OrderGate | None = None,
        client: Any | None = None,
        allow_withdrawals: bool = False,
    ) -> None:
        super().__init__(exchange, credentials=credentials, options=options, order_gate=order_gate)
        self._client: Any | None = client
        self._profile: VenueProfile | None = None
        self._markets: tuple[Market, ...] = ()
        self._supports_ws = False
        self._currencies: dict[str, Any] | None = None
        # Withdrawals are real money movements: the adapter refuses them
        # unless the composition root explicitly armed it (LIVE + opt-in).
        self._allow_withdrawals = allow_withdrawals

    @property
    def profile(self) -> VenueProfile:
        """Resolved venue profile (fail-closed: unknown ids never instantiate)."""
        if self._profile is None:
            self._profile = resolve_profile(self.id)
        return self._profile

    # ---------------------------------------------------------------- lifecycle
    async def open(self) -> None:
        client = self._client if self._client is not None else self._instantiate()
        if self._options.sandbox:
            self._apply_sandbox_routing(client)
        self._client = client
        self._capabilities = self._detect_capabilities(client)
        self._supports_ws = bool(getattr(client, "has", {}).get("watchOrderBook"))

    def _apply_sandbox_routing(self, client: Any) -> None:
        """Route a sandbox client for this venue's demo semantics.

        * ``demo_private_only`` (bybit): public market data stays on
          production, private sections are overridden to the demo host —
          ``set_sandbox_mode`` is *not* used (it would point everything at the
          separate testnet account system where demo keys are invalid).
        * ``demo_replaces_sandbox`` (binance): the demo environment uses
          completely separate endpoints; ``set_sandbox_mode`` is *not* used.
        * ``demo_headers`` only (okx): demo is header-only on production
          (``x-simulated-trading: 1``) — ``set_sandbox_mode`` must NOT be
          called (it would route to testnet where demo keys are invalid).
          ``AdapterOptions.sandbox`` stays True so the order gate remains active.
        * everyone else: plain ``set_sandbox_mode(True)`` plus optional
          header overrides.
        """
        profile = self.profile
        if profile.demo_private_only or profile.demo_replaces_sandbox:
            self._apply_demo_routing(client)
            return
        # OKX demo: header-only on production — never use set_sandbox_mode
        if profile.demo_headers:
            self._apply_demo_routing(client)
            return
        self._enable_sandbox(client)
        self._apply_demo_routing(client)

    def _override_private_demo_hosts(self, client: Any, base_url: str) -> None:
        """Point every non-public URL section at the demo host, fail-closed.

        The demo keys must never be able to reach production private endpoints:
        if the URL table does not look like ccxt's sectioned ``urls['api']``
        dict with string hosts, the adapter refuses to start instead of
        silently routing signed calls to production.
        """
        urls = getattr(client, "urls", None)
        api = urls.get("api") if isinstance(urls, dict) else None
        if not isinstance(api, dict) or not any(
            isinstance(value, str) and value for value in api.values()
        ):
            raise ExchangeUnavailableError(
                f"{self.id}: cannot route DEMO private calls to {base_url} "
                "(unexpected ccxt URL table); refusing to send demo keys to "
                "production endpoints",
                exchange_id=self.id,
            )
        overridden: list[str] = []
        new_api = dict(api)
        for section, value in api.items():
            if isinstance(value, str) and value and section != "public":
                new_api[section] = base_url
                overridden.append(section)
        if not overridden:
            raise ExchangeUnavailableError(
                f"{self.id}: no private URL sections found to point at {base_url}",
                exchange_id=self.id,
            )
        if isinstance(urls, dict):
            urls["api"] = new_api
        logger.info(
            "demo_routing_private_override",
            extra={"exchange_id": self.id, "base_url": base_url, "sections": ",".join(overridden)},
        )

    def _apply_demo_routing(self, client: Any) -> None:
        """Venue-specific DEMO routing for private calls."""
        profile = self.profile
        if not self._options.sandbox:
            return
        if profile.demo_private_only:
            base_url = profile.demo_base_url
            if not base_url:
                raise ExchangeUnavailableError(
                    f"{self.id} declares split demo routing without a demo base URL",
                    exchange_id=self.id,
                )
            self._override_private_demo_hosts(client, base_url)
            # The demo host refuses PRIVATE asset/currency endpoints
            # (retCode 10032 "Demo trading are not supported"), and ccxt calls
            # one of them implicitly on the first public request
            # (load_markets -> load_currencies).  Markets themselves load from
            # PUBLIC endpoints and stay on.
            has_map = getattr(client, "has", None)
            if isinstance(has_map, dict):
                has_map["fetchCurrencies"] = False
        elif profile.demo_replaces_sandbox or profile.demo_has_url_section:
            # Binance: the demo host does not serve all sapi endpoints; the
            # native ``enable_demo_trading`` helper swaps urls['api'] ->
            # urls['demo'] AND sets the flag that skips sapi-only calls.
            has_map = getattr(client, "has", None)
            if isinstance(has_map, dict):
                has_map["fetchCurrencies"] = False
            enable_demo = getattr(client, "enable_demo_trading", None)
            if callable(enable_demo):
                enable_demo(True)
                logger.info(
                    "demo_routing_url_section_copy",
                    extra={"exchange_id": self.id, "sections": "native_enable_demo_trading"},
                )
            else:
                demo_urls = getattr(client, "urls", {}).get("demo")
                if isinstance(demo_urls, dict) and demo_urls:
                    client.urls["api"] = dict(demo_urls)
                    options = getattr(client, "options", None)
                    if isinstance(options, dict):
                        options["enableDemoTrading"] = True
                    logger.info(
                        "demo_routing_url_section_copy",
                        extra={"exchange_id": self.id, "sections": ",".join(demo_urls.keys())},
                    )
                else:
                    raise ExchangeUnavailableError(
                        f"{self.id} declares demo_has_url_section but no demo urls found",
                        exchange_id=self.id,
                    )
        if profile.demo_headers:
            # OKX DEMO: public market data must not require private currencies endpoint.
            # CCXT OKX load_markets -> fetch_currencies is private and DEMO rejects it
            # with 50038 "unavailable in demo trading". Disable the capability
            # DEMO-only via the existing has flag — normal OKX keeps it.
            has_map = getattr(client, "has", None)
            if isinstance(has_map, dict):
                has_map["fetchCurrencies"] = False
            headers = getattr(client, "headers", None)
            if not isinstance(headers, dict):
                headers = {}
                client.headers = headers
            headers.update(dict(profile.demo_headers))
            logger.info(
                "demo_routing_headers",
                extra={
                    "exchange_id": self.id,
                    "headers": ",".join(name for name, _value in profile.demo_headers),
                },
            )

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        closer = getattr(client, "close", None)
        if closer is None:
            return
        try:
            result = closer()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:  # noqa: BLE001 - shutdown must stay quiet
            logger.warning("ccxt_close_failed", extra={"exchange_id": self.id, "error": str(exc)})

    def _base_config(self) -> dict[str, Any]:
        # H-11: a redacting dict — ccxt reads the real values through the
        # normal dict API, but the structure itself can never leak them via
        # str()/repr()/logging.
        config: dict[str, Any] = _SecretConfig(
            {
                "enableRateLimit": self._options.enable_rate_limit,
                "timeout": int(self._options.timeout_seconds * 1000),
            }
        )
        if self._credentials is not None and not self._credentials.is_empty:
            config["apiKey"] = self._credentials.api_key
            config["secret"] = self._credentials.secret
            if self._credentials.password:
                config["password"] = self._credentials.password
            if self._credentials.uid:
                config["uid"] = self._credentials.uid
        return config

    def _instantiate(self) -> Any:
        """Build a ccxt client for this venue."""
        profile = self.profile  # raises ConfigurationError for unknown ids
        config = self._base_config()
        config["defaultType"] = "spot"

        module = self._load_module()
        ccxt_class = resolve_ccxt_id(profile)
        factory = getattr(module, ccxt_class, None)
        if factory is None:
            raise ExchangeUnavailableError(
                f"ccxt has no exchange '{ccxt_class}' (venue '{self.id}')",
                exchange_id=self.id,
            )
        return factory(config)

    @staticmethod
    def _load_module() -> Any:
        """Resolve (and cache) the ccxt module; prefers ``ccxt.pro`` for WebSocket."""
        global _CCXT_MODULE
        if _CCXT_MODULE is not None:
            return _CCXT_MODULE
        try:  # ccxt.pro gives WebSocket support where available
            import ccxt.pro as ccxt_pro

            _CCXT_MODULE = ccxt_pro
            return _CCXT_MODULE
        except ImportError:
            pass
        try:
            import ccxt.async_support as ccxt_async

            _CCXT_MODULE = ccxt_async
            return _CCXT_MODULE
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ExchangeUnavailableError(
                "ccxt is not installed (pip install 'ccxt>=4.4')"
            ) from exc

    def _enable_sandbox(self, client: Any) -> None:
        """Sandbox activation must never fail silently.

        Failing open would send DEMO-mode orders to the production venue with
        real credentials, so an unsupported testnet marks the venue unusable.
        """
        setter = getattr(client, "set_sandbox_mode", None)
        if setter is None:
            raise ExchangeUnavailableError(
                f"{self.id} does not support sandbox/testnet mode", exchange_id=self.id
            )
        try:
            setter(True)
        except Exception as exc:
            raise ExchangeUnavailableError(
                f"{self.id} sandbox mode unavailable: {type(exc).__name__}", exchange_id=self.id
            ) from exc

    @staticmethod
    def _detect_capabilities(client: Any) -> ExchangeCapabilities:
        has: Mapping[str, Any] = getattr(client, "has", {}) or {}

        def flag(name: str) -> bool:
            return bool(has.get(name))

        return ExchangeCapabilities(
            spot=True,
            fetch_markets=flag("fetchMarkets") or True,
            fetch_ticker=flag("fetchTicker"),
            fetch_order_book=flag("fetchOrderBook"),
            fetch_trading_fees=flag("fetchTradingFee") or flag("fetchTradingFees"),
            fetch_balance=flag("fetchBalance"),
            create_order=flag("createOrder"),
            cancel_order=flag("cancelOrder"),
            withdraw=flag("withdraw"),
            deposit_address=flag("fetchDepositAddress"),
            fetch_deposits=flag("fetchDeposits"),
            fetch_withdrawals=flag("fetchWithdrawals"),
            fetch_withdrawal_networks=flag("fetchCurrencies"),
            watch_ticker=flag("watchTicker"),
            watch_order_book=flag("watchOrderBook"),
        )

    def _require_client(self) -> Any:
        if self._client is None:
            raise ExchangeUnavailableError(
                f"adapter for '{self.id}' is not opened", exchange_id=self.id
            )
        return self._client

    # ---------------------------------------------------------------- routing
    def _native_symbol(self, symbol: Symbol) -> str:
        """Unified CCXT symbol for REST/WS calls (e.g. BTC/USDT).

        CCXT expects unified symbols (``BTC/USDT``), not exchange-native ids
        (``BTCUSDT``).  The native id is kept in ``Market.native_symbol`` only
        for diagnostics; the unified name is passed to every CCXT method to
        avoid ``BadSymbol`` on Binance/OKX/Bybit.
        """
        return symbol.name

    def _secret_values(self) -> tuple[str, ...]:
        if self._credentials is None:
            return ()
        return tuple(
            value
            for value in (
                self._credentials.api_key,
                self._credentials.secret,
                self._credentials.password,
                self._credentials.uid,
            )
            if value
        )

    def _redact(self, text: str) -> str:
        return redact_secrets(text, extra=self._secret_values())

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        client = self._require_client()
        return await self._call_on(client, method, *args, **kwargs)

    async def _call_on(self, client: Any, method: str, *args: Any) -> Any:
        func = getattr(client, method, None)
        if func is None:
            raise self._unsupported(method)
        try:
            return await func(*args)
        except (CapabilityNotSupportedError, ExchangeError):
            raise
        except Exception as exc:
            # ccxt puts the signed request URL into the exception text and some
            # venues sign GETs via query params, so the raw text may contain the
            # API key and signature. It is redacted for logs and never returned.
            logger.warning(
                "ccxt_call_failed",
                extra={
                    "exchange_id": self.id,
                    "method": method,
                    "detail": self._redact(f"{type(exc).__name__}: {exc}")[:500],
                },
            )
            error_cls = classify_ccxt_exception(exc)
            message = f"{self.id}.{method} failed ({type(exc).__name__})"
            context: dict[str, Any] = {"method": method}
            venue_code = _venue_error_code(exc)
            if venue_code:
                context["venue_code"] = venue_code
            if error_cls is CapabilityNotSupportedError:
                raise error_cls(message, exchange_id=self.id, capability=method) from exc
            if error_cls is VenueAuthError:
                raise error_cls(message, exchange_id=self.id, **context) from exc
            if error_cls is ExchangeUnavailableError:
                raise error_cls(message, exchange_id=self.id) from exc
            if error_cls is RateLimitError:
                retry_after = getattr(exc, "retry_after", None)
                if retry_after is not None:
                    context["retry_after"] = retry_after
                raise error_cls(message, exchange_id=self.id, **context) from exc
            if error_cls is TimeoutError:
                raise error_cls(message, exchange_id=self.id, method=method) from exc
            raise error_cls(message, exchange_id=self.id, **context) from exc

    # ---------------------------------------------------------------- market data
    async def load_markets(self) -> tuple[Market, ...]:
        raw = await self._call("load_markets")
        markets: list[Market] = []
        for payload in (raw or {}).values():
            market = self._to_market(payload)
            if market is not None:
                markets.append(market)
        self._markets = tuple(markets)
        return self._markets

    def _to_market(self, payload: Mapping[str, Any]) -> Market | None:
        symbol_text = payload.get("symbol")
        base, quote = payload.get("base"), payload.get("quote")
        if not symbol_text or not base or not quote:
            return None
        raw_type = str(payload.get("type") or "spot").lower()
        if raw_type != "spot":
            return None
        limits = payload.get("limits") or {}
        amount_limits = limits.get("amount") or {}
        cost_limits = limits.get("cost") or {}
        price_limits = limits.get("price") or {}
        precision = payload.get("precision") or {}
        price_tick = _dec(precision.get("price"))
        amount_step = _dec(precision.get("amount"))
        price_decimals = None
        amount_decimals = None
        if price_tick is not None and 0 < price_tick < 1:
            pass
        elif price_tick is not None:
            price_decimals = _int(price_tick)
            price_tick = None
        if amount_step is not None and 0 < amount_step < 1:
            pass
        elif amount_step is not None:
            amount_decimals = _int(amount_step)
            amount_step = None
        return Market(
            exchange_id=self.id,
            symbol=Symbol(base=str(base), quote=str(quote)),
            market_type=MarketType.SPOT,
            native_symbol=str(payload.get("id") or symbol_text),
            active=bool(payload.get("active", True)),
            fees=MarketFees(
                maker_bps=(_dec(payload.get("maker"), DEC0) or DEC0) * 10000,
                taker_bps=(_dec(payload.get("taker"), DEC0) or DEC0) * 10000,
                is_account_specific=False,
            ),
            limits=MarketLimits(
                min_amount=_dec(amount_limits.get("min")),
                max_amount=_dec(amount_limits.get("max")),
                min_cost=_dec(cost_limits.get("min")),
                min_price=_dec(price_limits.get("min")),
                max_price=_dec(price_limits.get("max")),
            ),
            precision=MarketPrecision(
                price=price_decimals,
                amount=amount_decimals,
                price_tick=price_tick,
                amount_step=amount_step,
            ),
        )

    async def fetch_ticker(self, symbol: Symbol) -> Ticker:
        raw = await self._call("fetch_ticker", self._native_symbol(symbol))
        return self._to_ticker(raw, symbol)

    def _to_ticker(self, raw: Mapping[str, Any], symbol: Symbol) -> Ticker:
        return Ticker(
            exchange_id=self.id,
            symbol=symbol,
            bid=_dec(raw.get("bid")),
            ask=_dec(raw.get("ask")),
            last=_dec(raw.get("last") or raw.get("close")),
            bid_volume=_dec(raw.get("bidVolume")),
            ask_volume=_dec(raw.get("askVolume")),
            volume_24h=_dec(raw.get("quoteVolume") or raw.get("baseVolume")),
            timestamp=_ts(raw.get("timestamp")),
            received_at=utc_now(),
        )

    async def fetch_order_book(self, symbol: Symbol, *, depth: int | None = None) -> OrderBook:
        limit = depth or self._options.order_book_depth
        # ccxt's unified signature is ``fetch_order_book(symbol, limit=None,
        # params={})`` — the depth must be the *second* argument.
        raw = await self._call("fetch_order_book", self._native_symbol(symbol), limit)
        return self._to_order_book(raw, symbol)

    def _to_order_book(self, raw: Mapping[str, Any], symbol: Symbol) -> OrderBook:
        return OrderBook(
            exchange_id=self.id,
            symbol=symbol,
            bids=_levels(raw.get("bids")),
            asks=_levels(raw.get("asks")),
            timestamp=_ts(raw.get("timestamp")),
            received_at=utc_now(),
            sequence=_int(raw.get("nonce")),
        )

    async def fetch_trading_fees(self, symbol: Symbol) -> MarketFees:
        if self._capabilities.fetch_trading_fees:
            try:
                raw = await self._call("fetch_trading_fee", self._native_symbol(symbol))
                return MarketFees(
                    maker_bps=(_dec(raw.get("maker"), DEC0) or DEC0) * 10000,
                    taker_bps=(_dec(raw.get("taker"), DEC0) or DEC0) * 10000,
                    is_account_specific=True,
                )
            except (CapabilityNotSupportedError, ExchangeError):
                pass
        market = next((m for m in self._markets if m.symbol == symbol), None)
        if market is None:
            raise self._unsupported("fetch_trading_fees")
        return market.fees

    # ---------------------------------------------------------------- account
    async def fetch_balances(self) -> BalanceSnapshot:
        raw = await self._call("fetch_balance")
        free = raw.get("free") or {}
        used = raw.get("used") or {}
        assets = {str(a) for a in (*free.keys(), *used.keys())}
        balances = tuple(
            Balance(
                exchange_id=self.id,
                asset=asset,
                free=_dec(free.get(asset), DEC0) or DEC0,
                used=_dec(used.get(asset), DEC0) or DEC0,
            )
            for asset in sorted(assets)
        )
        return BalanceSnapshot(exchange_id=self.id, balances=balances, timestamp=utc_now())

    # ---------------------------------------------------------------- orders
    def _order_payload(self, request: OrderRequest) -> dict[str, Any]:
        params: dict[str, Any] = {"clientOrderId": request.client_order_id}
        if request.order_type is OrderType.LIMIT and request.time_in_force is not TimeInForce.GTC:
            params["timeInForce"] = str(request.time_in_force.value).upper()
        params.update(dict(request.metadata))
        return params

    async def create_order(self, request: OrderRequest) -> Order:
        if not self.can_place_orders():
            raise ExecutionDisabledError(
                "order placement refused by the dynamic order gate "
                "(mode/session does not allow real orders right now)",
                exchange_id=self.id,
            )
        params = self._order_payload(request)
        raw = await self._call(
            "create_order",
            self._native_symbol(request.symbol),
            request.order_type.value,
            request.side.value,
            float(request.amount),
            float(request.price) if request.price is not None else None,
            params,
        )
        return self._to_order(raw, request.symbol)

    async def cancel_order(self, order_id: str, *, symbol: Symbol) -> Order:
        raw = await self._call("cancel_order", order_id, self._native_symbol(symbol))
        return self._to_order(raw, symbol)

    async def fetch_order(self, order_id: str, *, symbol: Symbol) -> Order:
        # Bybit DEMO: fetchOrder for closed orders requires params["acknowledged"]=True
        # (CCXT raises ArgumentsRequired if not set). Try normal first, then retry
        # with acknowledged for Bybit, without breaking other venues.
        try:
            raw = await self._call("fetch_order", order_id, self._native_symbol(symbol))
        except ExchangeError as exc:
            # Check both wrapper and cause for Bybit acknowledged hint
            msg = str(exc).lower()
            cause_msg = str(exc.__cause__).lower() if exc.__cause__ else ""
            combined = f"{msg} {cause_msg}"
            if self.id == "bybit" and ("acknowledged" in combined or "fetchorder" in combined or "last 500" in combined):
                try:
                    raw = await self._call(
                        "fetch_order", order_id, self._native_symbol(symbol), {"acknowledged": True}
                    )
                except Exception:
                    raise exc from None
            else:
                raise
        return self._to_order(raw, symbol)

    async def fetch_open_orders(self, *, symbol: Symbol | None = None) -> tuple[Order, ...]:
        raw = await self._call("fetch_open_orders", self._native_symbol(symbol) if symbol else None)
        orders: list[Order] = []
        for item in raw or ():
            item_symbol = item.get("symbol")
            parsed = Symbol.parse(str(item_symbol)) if item_symbol else symbol
            if parsed is None:
                continue
            orders.append(self._to_order(item, parsed))
        return tuple(orders)

    def _to_order(self, raw: Mapping[str, Any], symbol: Symbol) -> Order:
        status = _ORDER_STATUSES.get(str(raw.get("status") or "").lower(), OrderStatus.PENDING)
        amount = _dec(raw.get("amount"), DEC0) or DEC0
        filled = _dec(raw.get("filled"), DEC0) or DEC0
        if status is OrderStatus.OPEN and DEC0 < filled < amount:
            status = OrderStatus.PARTIALLY_FILLED
        fee = raw.get("fee") or {}
        trades = raw.get("trades") or ()
        return Order(
            exchange_id=self.id,
            exchange_order_id=str(raw.get("id")) if raw.get("id") is not None else None,
            client_order_id=raw.get("clientOrderId"),
            symbol=symbol,
            side=OrderSide(str(raw.get("side") or "buy")),
            order_type=OrderType(str(raw.get("type") or "limit")),
            status=status,
            amount=amount,
            filled_amount=filled,
            price=_dec(raw.get("price")),
            average_price=_dec(raw.get("average")),
            fee_paid=_dec(fee.get("cost"), DEC0) or DEC0,
            fee_currency=fee.get("currency"),
            fills=tuple(
                Fill(
                    fill_id=str(t.get("id")) if t.get("id") is not None else None,
                    price=_dec(t.get("price"), DEC0) or DEC0,
                    amount=_dec(t.get("amount"), DEC0) or DEC0,
                    fee=_dec((t.get("fee") or {}).get("cost"), DEC0) or DEC0,
                    fee_currency=(t.get("fee") or {}).get("currency"),
                    timestamp=_ts(t.get("timestamp")),
                )
                for t in trades
                if isinstance(t, Mapping)
            ),
            created_at=_ts(raw.get("timestamp")),
            updated_at=utc_now(),
        )

    # ---------------------------------------------------------------- transfers
    async def _load_currencies(self) -> dict[str, Any]:
        """Fetch (and cache) the ccxt currency table, including networks."""
        if self._currencies is not None:
            return self._currencies
        raw = await self._call("fetch_currencies")
        self._currencies = dict(raw or {})
        return self._currencies

    async def fetch_withdrawal_networks(self, asset: str) -> tuple[WithdrawalNetwork, ...]:
        code = asset.strip().upper()
        # DEMO: demo hosts disable fetchCurrencies — use deterministic
        # simulated networks directly. DEMO withdrawals are simulated anyway.
        if self._options.sandbox:
            try:
                from app.exchanges.simulated import _NETWORKS as _SIM_NETS

                sim = _SIM_NETS.get(code)
                if sim:
                    return tuple(
                        WithdrawalNetwork(
                            network=name,
                            network_code=unified,
                            withdraw_enabled=True,
                            deposit_enabled=True,
                            withdrawal_fee=Decimal(fee),
                            withdrawal_min=Decimal(fee) * Decimal("10"),
                        )
                        for name, unified, fee in sim
                    )
                return (
                    WithdrawalNetwork(
                        network=code,
                        network_code=code,
                        withdraw_enabled=True,
                        deposit_enabled=True,
                        withdrawal_fee=DEC0,
                    ),
                )
            except Exception:
                pass
        currencies = await self._load_currencies()
        currency = currencies.get(code)
        if currency is None:
            raise CapabilityNotSupportedError(
                f"{self.id} has no currency data for {code}",
                exchange_id=self.id,
                capability="fetch_withdrawal_networks",
            )
        networks: list[WithdrawalNetwork] = []
        raw_networks = currency.get("networks") or {}
        for key, entry in raw_networks.items():
            if not isinstance(entry, Mapping):
                continue
            limits = entry.get("limits") or {}
            withdraw_limits = limits.get("withdraw") or {}
            deposit_limits = limits.get("deposit") or {}
            networks.append(
                WithdrawalNetwork(
                    network=str(entry.get("network") or key),
                    network_code=str(entry.get("code") or key) if entry.get("code") else None,
                    withdraw_enabled=bool(entry.get("withdraw", entry.get("active", True))),
                    deposit_enabled=bool(entry.get("deposit", entry.get("active", True))),
                    withdrawal_fee=_dec(entry.get("fee"), DEC0) or DEC0,
                    withdrawal_min=_dec(withdraw_limits.get("min"), DEC0) or DEC0,
                    deposit_min=_dec(deposit_limits.get("min"), DEC0) or DEC0,
                )
            )
        if not networks:
            # Single-network asset without a networks table: the currency row
            # itself carries the fee.
            networks.append(
                WithdrawalNetwork(
                    network=code,
                    network_code=code,
                    withdraw_enabled=bool(currency.get("withdraw", currency.get("active", True))),
                    deposit_enabled=bool(currency.get("deposit", currency.get("active", True))),
                    withdrawal_fee=_dec(currency.get("fee"), DEC0) or DEC0,
                )
            )
        return tuple(networks)

    async def fetch_deposit_address(
        self, asset: str, *, network: str | None = None
    ) -> DepositAddress:
        # DEMO: use deterministic simulated address directly (real demo host
        # often fails for deposit address and withdrawals are simulated anyway).
        if self._options.sandbox:
            try:
                import hashlib

                code = asset.strip().upper()
                net = network or code
                seed = f"{self.id}:{code}:{net}:addr".encode()
                digest = hashlib.blake2b(seed, digest_size=10).hexdigest()
                addr = f"sim-{code.lower()}-{net.lower()}-{digest[:20]}"
                # Provide memo for tag-required assets so orchestrator's memo check passes
                from app.strategies.transfer.orchestrator import _MEMO_REQUIRED_ASSETS

                memo_val = "123456" if code in _MEMO_REQUIRED_ASSETS else None
                return DepositAddress(address=addr, memo=memo_val, network=network)
            except Exception:
                pass
        params: dict[str, Any] = {}
        if network:
            params["network"] = network
        raw = await self._call("fetch_deposit_address", asset.strip().upper(), params)
        address = raw.get("address") if isinstance(raw, Mapping) else None
        if not address:
            raise ExchangeError(f"{self.id} returned no deposit address", exchange_id=self.id)
        return DepositAddress(
            address=str(address),
            memo=str(raw["tag"]) if raw.get("tag") else None,
            network=network,
        )

    async def fetch_deposits(self, asset: str, *, limit: int = 20) -> tuple[TransferTx, ...]:
        raw = await self._call("fetch_deposits", asset.strip().upper(), None, limit)
        return tuple(self._to_tx(item, "deposit") for item in (raw or ()))

    async def fetch_withdrawals(self, asset: str, *, limit: int = 20) -> tuple[TransferTx, ...]:
        raw = await self._call("fetch_withdrawals", asset.strip().upper(), None, limit)
        return tuple(self._to_tx(item, "withdrawal") for item in (raw or ()))

    async def withdraw(
        self,
        asset: str,
        amount: str,
        address: str,
        *,
        memo: str | None = None,
        network: str | None = None,
    ) -> TransferTx:
        if not self._allow_withdrawals:
            raise ExecutionDisabledError(
                f"{self.id}: withdrawals are disabled (LIVE withdrawals require "
                "CAT_TRADING__ALLOW_LIVE_WITHDRAWALS=true and LIVE mode)",
                exchange_id=self.id,
            )
        if not self.can_place_orders():
            raise ExecutionDisabledError(
                "withdrawal refused by the dynamic order gate",
                exchange_id=self.id,
            )
        params: dict[str, Any] = {}
        if network:
            params["network"] = network
        raw = await self._call(
            "withdraw",
            asset.strip().upper(),
            amount,
            address,
            memo,
            params,
        )
        return self._to_tx(raw, "withdrawal")

    @staticmethod
    def _to_tx(raw: Any, direction: str) -> TransferTx:
        item = raw if isinstance(raw, Mapping) else {}
        return TransferTx(
            direction=direction,  # type: ignore[arg-type]
            asset=str(item.get("currency") or item.get("asset") or ""),
            network=item.get("network"),
            amount=_dec(item.get("amount"), DEC0) or DEC0,
            fee=_dec(item.get("fee"), DEC0) or DEC0,
            status=str(item.get("status") or "pending").lower(),
            txid=item.get("txid") or item.get("txId"),
            address=item.get("address"),
            timestamp=_ts(item.get("timestamp")) if item.get("timestamp") else None,
        )

    # ---------------------------------------------------------------- streaming
    async def watch_ticker(  # type: ignore[override]
        self, symbol: Symbol
    ) -> AsyncIterator[Ticker]:
        client = self._require_client()
        native = self._native_symbol(symbol)
        interval = max(0.1, self._options.timeout_seconds / 10)
        if self._capabilities.watch_ticker and hasattr(client, "watch_ticker"):
            while True:
                raw = await client.watch_ticker(native)
                yield self._to_ticker(raw, symbol)
        else:  # REST polling fallback keeps the pipeline uniform
            while True:
                yield await self.fetch_ticker(symbol)
                await asyncio.sleep(interval)

    async def watch_order_book(  # type: ignore[override]
        self, symbol: Symbol
    ) -> AsyncIterator[OrderBook]:
        client = self._require_client()
        native = self._native_symbol(symbol)
        interval = max(0.1, self._options.timeout_seconds / 10)
        if self._capabilities.watch_order_book and hasattr(client, "watch_order_book"):
            # Bybit spot WS allows only 1/50/200/1000 — REST depth 25 is invalid for WS.
            ws_depth = 50 if self.id == "bybit" else self._options.order_book_depth
            while True:
                raw = await client.watch_order_book(native, ws_depth)
                yield self._to_order_book(raw, symbol)
        else:
            while True:
                yield await self.fetch_order_book(symbol)
                await asyncio.sleep(interval)

    # ---------------------------------------------------------------- health
    async def ping(self) -> float:
        client = self._require_client()
        started = asyncio.get_running_loop().time()
        method = "fetch_time" if hasattr(client, "fetch_time") else "fetch_status"
        await self._call_on(client, method)
        return (asyncio.get_running_loop().time() - started) * 1000.0

    async def health(self) -> ExchangeHealth:
        try:
            latency = await self.ping()
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return ExchangeHealth(
                exchange_id=self.id,
                status=ExchangeStatus.OFFLINE,
                last_error=self._redact(f"{type(exc).__name__}: {exc}")[:300],
                consecutive_failures=1,
            )
        return ExchangeHealth(
            exchange_id=self.id,
            status=ExchangeStatus.ONLINE,
            rest_latency_ms=latency,
            ws_latency_ms=None if not self._supports_ws else latency,
        )


def _levels(raw: Any) -> tuple[OrderBookLevel, ...]:
    levels: list[OrderBookLevel] = []
    for item in raw or ():
        if not isinstance(item, list | tuple) or len(item) < 2:
            continue
        price, amount = _dec(item[0]), _dec(item[1])
        if price is None or amount is None or amount <= DEC0:
            continue
        levels.append(OrderBookLevel(price=price, amount=amount))
    return tuple(levels)


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def build_ccxt_adapter(request: Any) -> CCXTAdapter:
    """Registry factory for :class:`CCXTAdapter`."""
    return CCXTAdapter(
        request.exchange,
        credentials=request.credentials,
        options=request.options,
        order_gate=getattr(request, "order_gate", None),
        allow_withdrawals=getattr(request, "allow_withdrawals", False),
    )
