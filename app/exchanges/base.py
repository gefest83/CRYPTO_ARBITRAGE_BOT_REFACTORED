"""The single exchange interface every venue adapter implements.

Two artefacts live here:

* :class:`ExchangeAdapter` — a ``Protocol`` describing the contract the rest
  of the bot programs against (market data, balances, orders, transfers).
* :class:`BaseExchangeAdapter` — a base class whose every method raises
  :class:`~app.errors.CapabilityNotSupportedError` by default.  A venue that
  cannot provide some capability simply does not override it, and the bot
  degrades gracefully instead of crashing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.errors import CapabilityNotSupportedError
from app.models.balance import BalanceSnapshot
from app.models.exchange import Exchange, ExchangeCapabilities, ExchangeHealth
from app.models.market import Market, MarketFees
from app.models.market_data import OrderBook, Ticker
from app.models.order import Order, OrderRequest
from app.models.symbol import Symbol
from app.models.transfer import DepositAddress, TransferTx, WithdrawalNetwork

from .credentials import ExchangeCredentials

__all__ = [
    "AdapterOptions",
    "BaseExchangeAdapter",
    "ExchangeAdapter",
    "OrderGate",
]

#: Dynamic order-placement decision made *at call time*, never a static flag:
#: the gate receives the adapter and answers whether ``create_order`` may run
#: right now (PAPER: never; DEMO: sandbox venues only; LIVE: armed + allowed).
OrderGate = Callable[["BaseExchangeAdapter"], bool]


@dataclass(frozen=True, slots=True)
class AdapterOptions:
    """Transport-level options shared by all adapters."""

    timeout_seconds: float = 10.0
    order_book_depth: int = 25
    sandbox: bool = False
    enable_rate_limit: bool = True
    quote_currencies: tuple[str, ...] = ("USDT",)


@runtime_checkable
class ExchangeAdapter(Protocol):
    """Uniform venue contract. Implementations are created by the registry."""

    # --- identity -----------------------------------------------------------
    @property
    def exchange(self) -> Exchange: ...

    @property
    def id(self) -> str: ...

    @property
    def capabilities(self) -> ExchangeCapabilities: ...

    # --- lifecycle ----------------------------------------------------------
    async def open(self) -> None: ...

    async def close(self) -> None: ...

    # --- market data --------------------------------------------------------
    async def load_markets(self) -> tuple[Market, ...]: ...

    async def fetch_ticker(self, symbol: Symbol) -> Ticker: ...

    async def fetch_order_book(self, symbol: Symbol, *, depth: int | None = None) -> OrderBook: ...

    async def fetch_trading_fees(self, symbol: Symbol) -> MarketFees: ...

    # --- account ------------------------------------------------------------
    async def fetch_balances(self) -> BalanceSnapshot: ...

    # --- orders -------------------------------------------------------------
    async def create_order(self, request: OrderRequest) -> Order: ...

    async def cancel_order(self, order_id: str, *, symbol: Symbol) -> Order: ...

    async def fetch_order(self, order_id: str, *, symbol: Symbol) -> Order: ...

    async def fetch_open_orders(self, *, symbol: Symbol | None = None) -> tuple[Order, ...]: ...

    # --- transfers ----------------------------------------------------------
    async def fetch_withdrawal_networks(self, asset: str) -> tuple[WithdrawalNetwork, ...]: ...

    async def fetch_deposit_address(
        self, asset: str, *, network: str | None = None
    ) -> DepositAddress: ...

    async def fetch_deposits(self, asset: str, *, limit: int = 20) -> tuple[TransferTx, ...]: ...

    async def fetch_withdrawals(self, asset: str, *, limit: int = 20) -> tuple[TransferTx, ...]: ...

    async def withdraw(
        self,
        asset: str,
        amount: str,
        address: str,
        *,
        memo: str | None = None,
        network: str | None = None,
    ) -> TransferTx: ...

    # --- streaming ----------------------------------------------------------
    def watch_ticker(self, symbol: Symbol) -> AsyncIterator[Ticker]: ...

    def watch_order_book(self, symbol: Symbol) -> AsyncIterator[OrderBook]: ...

    # --- health -------------------------------------------------------------
    async def ping(self) -> float: ...

    async def health(self) -> ExchangeHealth: ...


class BaseExchangeAdapter:
    """Capability-gated base implementation.

    Subclasses override only what the venue really supports and declare it in
    :attr:`capabilities`.  Every non-overridden method raises
    :class:`~app.errors.CapabilityNotSupportedError` instead of failing
    obscurely.
    """

    adapter_name: str = "base"

    def __init__(
        self,
        exchange: Exchange,
        *,
        credentials: ExchangeCredentials | None = None,
        options: AdapterOptions | None = None,
        order_gate: OrderGate | None = None,
    ) -> None:
        self._exchange = exchange
        self._credentials = credentials
        self._options = options or AdapterOptions()
        self._order_gate = order_gate
        self._capabilities = exchange.capabilities

    # --- identity -----------------------------------------------------------
    @property
    def exchange(self) -> Exchange:
        return self._exchange

    @property
    def id(self) -> str:
        return self._exchange.id

    @property
    def name(self) -> str:
        return self._exchange.name

    @property
    def options(self) -> AdapterOptions:
        return self._options

    @property
    def capabilities(self) -> ExchangeCapabilities:
        return self._capabilities

    @property
    def has_credentials(self) -> bool:
        return self._credentials is not None and not self._credentials.is_empty

    def can_place_orders(self) -> bool:
        """Dynamic interlock evaluated at ``create_order`` time.

        Fail-closed by construction: without an injected gate, or if the gate
        itself blows up, placement is refused.
        """
        if self._order_gate is None:
            return False
        try:
            return bool(self._order_gate(self))
        except Exception:  # noqa: BLE001 - a broken gate must never open trading
            return False

    # --- lifecycle ----------------------------------------------------------
    async def open(self) -> None:
        """Establish transports. Default: nothing to do."""

    async def close(self) -> None:
        """Release transports. Default: nothing to do."""

    async def __aenter__(self) -> BaseExchangeAdapter:
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # --- capability guard ---------------------------------------------------
    def _unsupported(self, capability: str) -> CapabilityNotSupportedError:
        return CapabilityNotSupportedError(
            f"{self.id} does not support '{capability}'",
            exchange_id=self.id,
            capability=capability,
        )

    # --- market data --------------------------------------------------------
    async def load_markets(self) -> tuple[Market, ...]:
        raise self._unsupported("load_markets")

    async def fetch_ticker(self, symbol: Symbol) -> Ticker:
        raise self._unsupported("fetch_ticker")

    async def fetch_order_book(self, symbol: Symbol, *, depth: int | None = None) -> OrderBook:
        raise self._unsupported("fetch_order_book")

    async def fetch_trading_fees(self, symbol: Symbol) -> MarketFees:
        raise self._unsupported("fetch_trading_fees")

    # --- account ------------------------------------------------------------
    async def fetch_balances(self) -> BalanceSnapshot:
        raise self._unsupported("fetch_balances")

    # --- orders -------------------------------------------------------------
    async def create_order(self, request: OrderRequest) -> Order:
        raise self._unsupported("create_order")

    async def cancel_order(self, order_id: str, *, symbol: Symbol) -> Order:
        raise self._unsupported("cancel_order")

    async def fetch_order(self, order_id: str, *, symbol: Symbol) -> Order:
        raise self._unsupported("fetch_order")

    async def fetch_open_orders(self, *, symbol: Symbol | None = None) -> tuple[Order, ...]:
        raise self._unsupported("fetch_open_orders")

    # --- transfers ----------------------------------------------------------
    async def fetch_withdrawal_networks(self, asset: str) -> tuple[WithdrawalNetwork, ...]:
        raise self._unsupported("fetch_withdrawal_networks")

    async def fetch_deposit_address(
        self, asset: str, *, network: str | None = None
    ) -> DepositAddress:
        raise self._unsupported("fetch_deposit_address")

    async def fetch_deposits(self, asset: str, *, limit: int = 20) -> tuple[TransferTx, ...]:
        raise self._unsupported("fetch_deposits")

    async def fetch_withdrawals(self, asset: str, *, limit: int = 20) -> tuple[TransferTx, ...]:
        raise self._unsupported("fetch_withdrawals")

    async def withdraw(
        self,
        asset: str,
        amount: str,
        address: str,
        *,
        memo: str | None = None,
        network: str | None = None,
    ) -> TransferTx:
        raise self._unsupported("withdraw")

    # --- streaming ----------------------------------------------------------
    def watch_ticker(self, symbol: Symbol) -> AsyncIterator[Ticker]:
        raise self._unsupported("watch_ticker")

    def watch_order_book(self, symbol: Symbol) -> AsyncIterator[OrderBook]:
        raise self._unsupported("watch_order_book")

    # --- health -------------------------------------------------------------
    async def ping(self) -> float:
        raise self._unsupported("ping")

    async def health(self) -> ExchangeHealth:
        """Default health probe based on :meth:`ping`."""
        from app.models.enums import ExchangeStatus

        try:
            latency = await self.ping()
        except CapabilityNotSupportedError:
            return ExchangeHealth(exchange_id=self.id, status=ExchangeStatus.UNKNOWN)
        except Exception as exc:  # noqa: BLE001 - health must never raise
            from .sanitize import redact_secrets

            return ExchangeHealth(
                exchange_id=self.id,
                status=ExchangeStatus.OFFLINE,
                last_error=redact_secrets(str(exc))[:500],
                consecutive_failures=1,
            )
        return ExchangeHealth(
            exchange_id=self.id, status=ExchangeStatus.ONLINE, rest_latency_ms=latency
        )

    # --- helpers ------------------------------------------------------------
    def filter_quotes(self, markets: Sequence[Market]) -> tuple[Market, ...]:
        quotes = set(self._options.quote_currencies)
        if not quotes:
            return tuple(markets)
        return tuple(m for m in markets if m.symbol.quote in quotes)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{type(self).__name__}(id={self.id!r}, adapter={self.adapter_name!r})"
