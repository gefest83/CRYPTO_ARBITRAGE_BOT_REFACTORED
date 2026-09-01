"""In-memory market data store with explicit freshness tracking.

Every snapshot carries its own timestamps, so the store can answer "how old is
this quote?" — a hard requirement for arbitrage: a stale price is a fake spread.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.clock import Clock, SystemClock
from app.models.enums import MarketType
from app.models.market_data import OrderBook, Ticker
from app.models.symbol import Symbol

__all__ = ["MarketDataStore", "QuoteKey"]

#: A snapshot timestamped materially in the future (beyond this tolerance)
#: is treated as stale: clock skew of a few seconds between a venue and the
#: local clock is normal, but a far-future timestamp (replay, tampering or a
#: broken venue clock) must never pass freshness validation (H-9).
_FUTURE_TOLERANCE_MS = 10_000.0


@dataclass(frozen=True, slots=True)
class QuoteKey:
    """Identity of a quote stream: venue + instrument + market type."""

    exchange_id: str
    symbol: str
    market_type: MarketType

    @classmethod
    def of(cls, exchange_id: str, symbol: Symbol, market_type: MarketType) -> QuoteKey:
        return cls(exchange_id=exchange_id.lower(), symbol=symbol.name, market_type=market_type)


class MarketDataStore:
    """Latest-value cache for tickers and order books."""

    def __init__(self, *, stale_after_ms: int = 2000, clock: Clock | None = None) -> None:
        self._stale_after_ms = stale_after_ms
        self._clock = clock or SystemClock()
        self._tickers: dict[QuoteKey, Ticker] = {}
        self._books: dict[QuoteKey, OrderBook] = {}

    # ---------------------------------------------------------------- writes
    def put_ticker(self, ticker: Ticker) -> None:
        self._tickers[QuoteKey.of(ticker.exchange_id, ticker.symbol, ticker.market_type)] = ticker

    def put_order_book(self, book: OrderBook) -> None:
        """Cache a book snapshot; H-10: an older snapshot never replaces a
        newer one (out-of-order websocket replays / reconnect duplicates are
        dropped).  Equal timestamps are allowed (same snapshot, refreshed
        content)."""
        key = QuoteKey.of(book.exchange_id, book.symbol, book.market_type)
        existing = self._books.get(key)
        if existing is not None and book.timestamp < existing.timestamp:
            return
        self._books[key] = book

    def clear(self, *, exchange_id: str | None = None) -> None:
        if exchange_id is None:
            self._tickers.clear()
            self._books.clear()
            return
        needle = exchange_id.lower()
        for storage in (self._tickers, self._books):
            for key in [k for k in storage if k.exchange_id == needle]:
                storage.pop(key, None)

    # ---------------------------------------------------------------- reads
    def ticker(self, exchange_id: str, symbol: Symbol) -> Ticker | None:
        return self._tickers.get(QuoteKey.of(exchange_id, symbol, MarketType.SPOT))

    def order_book(self, exchange_id: str, symbol: Symbol) -> OrderBook | None:
        return self._books.get(QuoteKey.of(exchange_id, symbol, MarketType.SPOT))

    def tickers_for(
        self,
        symbol: Symbol,
        *,
        exchange_ids: Iterable[str] | None = None,
        fresh_only: bool = False,
    ) -> tuple[Ticker, ...]:
        allowed = {e.lower() for e in exchange_ids} if exchange_ids is not None else None
        return tuple(
            ticker
            for key, ticker in self._tickers.items()
            if key.symbol == symbol.name
            and (allowed is None or key.exchange_id in allowed)
            and (not fresh_only or self.is_fresh(self._age_ms(ticker)))
        )

    def order_books_for(
        self,
        symbol: Symbol,
        *,
        exchange_ids: Iterable[str] | None = None,
        fresh_only: bool = False,
    ) -> tuple[OrderBook, ...]:
        allowed = {e.lower() for e in exchange_ids} if exchange_ids is not None else None
        return tuple(
            book
            for key, book in self._books.items()
            if key.symbol == symbol.name
            and (allowed is None or key.exchange_id in allowed)
            and (not fresh_only or self.is_fresh(self._age_ms(book)))
        )

    def books(self, *, exchange_id: str | None = None) -> dict[str, OrderBook]:
        """Cached spot books of one venue (or all venues) keyed by symbol name.

        The triangular scanner walks a venue's whole spot graph, so it needs
        per-venue access without knowing symbol names up front.
        """
        needle = exchange_id.lower() if exchange_id else None
        out: dict[str, OrderBook] = {}
        for key, book in self._books.items():
            if needle is not None and key.exchange_id != needle:
                continue
            out[key.symbol] = book
        return out

    def all_tickers(self) -> tuple[Ticker, ...]:
        """Every cached ticker (single pass; used for price maps)."""
        return tuple(self._tickers.values())

    def symbols(self) -> tuple[str, ...]:
        keys = (*self._tickers.keys(), *self._books.keys())
        return tuple(dict.fromkeys(key.symbol for key in keys))

    def exchange_ids(self) -> tuple[str, ...]:
        keys = (*self._tickers.keys(), *self._books.keys())
        return tuple(dict.fromkeys(key.exchange_id for key in keys))

    # ---------------------------------------------------------------- freshness
    @property
    def stale_after_ms(self) -> int:
        return self._stale_after_ms

    def _age_ms(self, quote: Ticker | OrderBook) -> float:
        """Snapshot age relative to the injected clock.

        H-9: a timestamp materially in the future is reported as infinitely
        old (stale) instead of clamping to a fresh age of 0 — a future
        timestamp must not bypass freshness validation.
        """
        raw = (self._clock.now() - quote.timestamp).total_seconds() * 1000.0
        if raw < -_FUTURE_TOLERANCE_MS:
            return float("inf")
        return max(0.0, raw)

    def age_ms(self, quote: Ticker | OrderBook) -> float:
        """Public snapshot age (used by scanners that iterate cached books)."""
        return self._age_ms(quote)

    def is_fresh(self, age_ms: float) -> bool:
        return age_ms <= self._stale_after_ms

    def is_stale(self, age_ms: float) -> bool:
        return not self.is_fresh(age_ms)

    def stats(self) -> dict[str, int]:
        return {
            "tickers": len(self._tickers),
            "order_books": len(self._books),
            "exchanges": len(self.exchange_ids()),
        }
