"""Market data service: WebSocket-first quotes with REST fallback.

* Streams (``watch_ticker`` / ``watch_order_book`` via the adapters) feed the
  store continuously and are supervised with restart + backoff + quarantine.
* REST refresh covers every pair not served by a *delivering* stream.
* One failing venue degrades only its own data (per-venue isolation).
* A stream that silently stops delivering is detected by the inactivity
  watchdog: its pairs revert to REST refresh instead of going stale.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass

from app.config.logging_config import get_logger
from app.config.settings import MarketDataSettings
from app.errors import (
    CapabilityNotSupportedError,
    RateLimitError,
    TerminalError,
    VenueAuthError,
)
from app.exchanges.base import BaseExchangeAdapter
from app.exchanges.manager import ExchangeManager
from app.market_data.store import MarketDataStore
from app.market_data.streams import StreamSpec, StreamSupervisor
from app.models.base import utc_now
from app.models.enums import HealthStatus, StreamKind
from app.models.market_data import OrderBook, Ticker
from app.models.symbol import Symbol

__all__ = ["MarketDataService", "RefreshOutcome"]

logger = get_logger("market_data.service")

#: A stream whose last event is older than this is no longer trusted to cover
#: its pair; REST refresh takes over (fixes silently-stalled-stream starvation).
STREAM_INACTIVITY_SECONDS = 30.0


def _ticker_stream_factory(
    adapter: BaseExchangeAdapter, symbol: Symbol
) -> Callable[[], AsyncIterator[Ticker]]:
    """Zero-arg iterator factory for the supervisor (binds venue/symbol now)."""

    async def iterator() -> AsyncIterator[Ticker]:
        async for item in adapter.watch_ticker(symbol):
            yield item

    return iterator


def _book_stream_factory(
    adapter: BaseExchangeAdapter, symbol: Symbol
) -> Callable[[], AsyncIterator[OrderBook]]:
    async def iterator() -> AsyncIterator[OrderBook]:
        async for item in adapter.watch_order_book(symbol):
            yield item

    return iterator


@dataclass(frozen=True, slots=True)
class RefreshOutcome:
    """Result of a refresh round; ``failures`` never raise upstream."""

    requested: int = 0
    succeeded: int = 0
    failures: tuple[tuple[str, str], ...] = ()

    @property
    def failed(self) -> int:
        return len(self.failures)

    @property
    def is_complete(self) -> bool:
        return self.requested == self.succeeded


class MarketDataService:
    """Coordinates exchange adapters, the store and stream supervision."""

    def __init__(
        self,
        *,
        manager: ExchangeManager,
        store: MarketDataStore,
        config: MarketDataSettings,
    ) -> None:
        self._manager = manager
        self._store = store
        self._config = config

        # (venue, symbol) pairs the venue itself declared unsupported (ccxt
        # BadSymbol).  A missing market is a permanent fact — re-asking every
        # refresh round only burns rate limit and floods the log.
        self._unsupported_pairs: set[tuple[str, str]] = set()

        self._stream_supervisor: StreamSupervisor | None = None
        if config.streams_enabled:
            self._stream_supervisor = StreamSupervisor(
                backoff_base_seconds=config.stream_backoff_base_seconds,
                backoff_max_seconds=config.stream_backoff_max_seconds,
                backoff_jitter=config.stream_backoff_jitter,
                max_streams_per_exchange=config.max_streams_per_exchange,
                max_parallel_streams=config.max_parallel_streams,
                quarantine_restarts=config.stream_quarantine_restarts,
                quarantine_after_seconds=45.0,
            )

    @property
    def store(self) -> MarketDataStore:
        return self._store

    @property
    def stream_supervisor(self) -> StreamSupervisor | None:
        return self._stream_supervisor

    # ---------------------------------------------------------------- streams
    async def start_streams(
        self,
        symbols: Sequence[Symbol],
        *,
        exchange_ids: Sequence[str] | None = None,
    ) -> None:
        """Start supervised watch streams for the given symbols/venues.

        The supervisor limits are admission limits: a spec that does not fit
        the global/per-venue budget is *not* registered — it stays
        REST-refreshable instead of occupying a semaphore slot forever.
        """
        if self._stream_supervisor is None:
            logger.info("streams_disabled", extra={"reason": "streams_enabled=false in config"})
            return

        supervisor = self._stream_supervisor
        venues = tuple(exchange_ids or self._manager.enabled_ids())

        for venue in venues:
            if self._manager.is_breaker_open(venue):
                continue
            for symbol in symbols:
                if supervisor.registered_count() >= self._config.max_parallel_streams:
                    logger.info(
                        "streams_global_budget_exhausted",
                        extra={"limit": self._config.max_parallel_streams},
                    )
                    return
                adapter = self._manager.adapter(venue)
                has_ticker_stream = adapter.capabilities.watch_ticker
                has_book_stream = adapter.capabilities.watch_order_book
                if not has_ticker_stream and not has_book_stream:
                    continue

                per_venue_budget = (
                    self._config.max_streams_per_exchange - supervisor.registered_count(venue)
                )
                if per_venue_budget <= 0:
                    break  # this venue is fully streamed; move to the next venue

                if has_ticker_stream:
                    spec = StreamSpec(exchange_id=venue, symbol=symbol.name, kind=StreamKind.TICKER)
                    if not supervisor.is_running(spec):
                        await supervisor.start_stream(
                            spec,
                            _ticker_stream_factory(adapter, symbol),
                            self._make_stream_handler(StreamKind.TICKER),
                        )
                        per_venue_budget -= 1

                if has_book_stream and per_venue_budget > 0:
                    spec = StreamSpec(
                        exchange_id=venue, symbol=symbol.name, kind=StreamKind.ORDER_BOOK
                    )
                    if not supervisor.is_running(spec):
                        await supervisor.start_stream(
                            spec,
                            _book_stream_factory(adapter, symbol),
                            self._make_stream_handler(StreamKind.ORDER_BOOK),
                        )
                        per_venue_budget -= 1

    def _make_stream_handler(self, kind: StreamKind) -> Callable[[object], object]:
        """Handler that puts stream events into the store."""

        def handler(event: object) -> None:
            if kind == StreamKind.TICKER:
                self._store.put_ticker(event)  # type: ignore[arg-type]
            else:
                self._store.put_order_book(event)  # type: ignore[arg-type]

        return handler

    async def stop_streams(self) -> None:
        """Stop all supervised streams."""
        if self._stream_supervisor is not None:
            await self._stream_supervisor.stop_all()

    def get_stream_covered_pairs(self, kind: StreamKind | None = None) -> set[tuple[str, str]]:
        """``(exchange_id, symbol)`` pairs served by a *delivering* stream.

        A pair is covered only when its stream actually delivers data *and*
        delivered recently: specs still waiting on a semaphore report
        ``UNKNOWN`` and do not count, and a stream whose last event is older
        than the inactivity watchdog threshold no longer covers its pair
        (REST refresh takes over until it delivers again).
        """
        if self._stream_supervisor is None:
            return set()
        now = utc_now()
        covered: set[tuple[str, str]] = set()
        for status in self._stream_supervisor.all_statuses():
            if kind is not None and status.kind != kind:
                continue
            if status.status is HealthStatus.UNKNOWN:
                continue
            if (
                status.last_event_at is not None
                and (now - status.last_event_at).total_seconds() > STREAM_INACTIVITY_SECONDS
            ):
                continue
            covered.add((status.exchange_id, status.symbol))
        return covered

    def stream_statuses(self) -> tuple[object, ...]:
        if self._stream_supervisor is None:
            return ()
        return self._stream_supervisor.all_statuses()

    # ---------------------------------------------------------------- REST
    async def refresh_tickers(
        self, symbols: Sequence[Symbol], *, exchange_ids: Sequence[str] | None = None
    ) -> RefreshOutcome:
        return await self._refresh(symbols, exchange_ids, kind="ticker")

    async def refresh_order_books(
        self, symbols: Sequence[Symbol], *, exchange_ids: Sequence[str] | None = None
    ) -> RefreshOutcome:
        return await self._refresh(symbols, exchange_ids, kind="order_book")

    async def _refresh(
        self,
        symbols: Sequence[Symbol],
        exchange_ids: Sequence[str] | None,
        *,
        kind: str,
    ) -> RefreshOutcome:
        # Skip (venue, symbol) pairs that an actively delivering stream covers.
        stream_kind = StreamKind.TICKER if kind == "ticker" else StreamKind.ORDER_BOOK
        covered_pairs = self.get_stream_covered_pairs(stream_kind)

        all_venues = tuple(exchange_ids or self._manager.enabled_ids())
        # Circuit breaker: a venue tripped by consecutive failures is skipped
        # entirely — it cannot tax the refresh round any more.
        all_venues = tuple(
            venue for venue in all_venues if not self._manager.is_breaker_open(venue)
        )
        targets = [
            (venue, symbol)
            for venue in all_venues
            for symbol in symbols
            if (venue, symbol.name) not in covered_pairs
            and (venue, symbol.name) not in self._unsupported_pairs
        ]
        if not targets:
            return RefreshOutcome()

        global_limit = asyncio.Semaphore(max(1, self._config.max_parallel_requests))
        per_venue: dict[str, asyncio.Semaphore] = {
            venue: asyncio.Semaphore(max(1, self._config.max_requests_per_exchange))
            for venue in all_venues
        }
        failures: list[tuple[str, str]] = []
        succeeded = 0

        async def fetch(venue: str, symbol: Symbol) -> None:
            nonlocal succeeded
            async with global_limit, per_venue[venue]:
                try:
                    adapter = self._manager.adapter(venue)
                    if kind == "ticker":
                        await self._fetch_ticker(adapter, venue, symbol)
                    else:
                        await self._fetch_book(adapter, venue, symbol)
                except CapabilityNotSupportedError as exc:
                    # The venue does not list this market (BadSymbol) or lacks
                    # the capability.  Permanent — cache the pair and never
                    # count it against the network breaker.
                    self._remember_unsupported_pair(venue, symbol.name, exc)
                    return
                except VenueAuthError as exc:
                    # An auth problem seen on a PUBLIC market-data call is
                    # reported honestly but must not poison private-call state
                    # nor trip the network breaker (public data needs no keys).
                    failures.append((venue, exc.message))
                    return
                except RateLimitError as exc:
                    retry_after = exc.context.get("retry_after")
                    self._manager.record_rate_limit(
                        venue, float(retry_after) if retry_after is not None else None
                    )
                    failures.append((venue, exc.message))
                    return
                except TerminalError as exc:
                    failures.append((venue, exc.message))
                    self._manager.record_failure(venue)
                    return
                except Exception as exc:  # noqa: BLE001 - isolate venue failures
                    failures.append((venue, str(exc)))
                    self._manager.record_failure(venue)
                    return
                succeeded += 1
                self._manager.record_success(venue)

        await asyncio.gather(*(fetch(venue, symbol) for venue, symbol in targets))

        if failures:
            logger.info(
                "market_data_refresh_partial",
                extra={"kind": kind, "failed": len(failures), "ok": succeeded},
            )
        return RefreshOutcome(requested=len(targets), succeeded=succeeded, failures=tuple(failures))

    async def _fetch_ticker(self, adapter: BaseExchangeAdapter, venue: str, symbol: Symbol) -> None:
        ticker = await adapter.fetch_ticker(symbol)
        self._store.put_ticker(ticker)

    async def _fetch_book(self, adapter: BaseExchangeAdapter, venue: str, symbol: Symbol) -> None:
        # Depth comes from the adapter's own AdapterOptions (exchange settings).
        book = await adapter.fetch_order_book(symbol)
        self._store.put_order_book(book)

    def _remember_unsupported_pair(self, venue: str, symbol_name: str, exc: Exception) -> None:
        """Cache a venue-declared unsupported (venue, symbol) pair once."""
        key = (venue, symbol_name)
        if key in self._unsupported_pairs:
            return
        self._unsupported_pairs.add(key)
        logger.info(
            "market_symbol_unsupported",
            extra={"exchange_id": venue, "symbol": symbol_name, "reason": str(exc)[:160]},
        )

    # ---------------------------------------------------------------- reads
    def tickers(self, symbol: Symbol, *, fresh_only: bool = False) -> tuple[Ticker, ...]:
        return self._store.tickers_for(
            symbol, exchange_ids=self._manager.enabled_ids(), fresh_only=fresh_only
        )

    def order_books(self, symbol: Symbol, *, fresh_only: bool = False) -> tuple[OrderBook, ...]:
        return self._store.order_books_for(
            symbol, exchange_ids=self._manager.enabled_ids(), fresh_only=fresh_only
        )

    def order_book(self, exchange_id: str, symbol: Symbol) -> OrderBook | None:
        return self._store.order_book(exchange_id, symbol)

    def ticker(self, exchange_id: str, symbol: Symbol) -> Ticker | None:
        return self._store.ticker(exchange_id, symbol)
