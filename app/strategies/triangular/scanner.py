"""Triangular arbitrage scanner.

Finds 3-leg spot cycles ``USDT → X → Y → USDT`` *inside one venue* and prices
every leg with a depth-aware VWAP walk in trade direction:

    net = Π(leg rates) × (1 − taker)³ − 1   → bps

The middle leg uses whichever cross book exists (``Y/X`` to buy Y with X, or
``X/Y`` to sell X for Y); both orientations are equivalent economically.  A
cycle is reported only when every leg stays inside the per-leg slippage
tolerance, the notional cap holds and the measured net clears
``triangle_min_net_bps``.

The scanner receives the venue list through an injected provider and reads the
market-data store — no venue names here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from app.clock import Clock, SystemClock
from app.config.logging_config import get_logger
from app.config.settings import ArbitrageSettings
from app.execution.precision import (
    PrecisionProvider,
    StaticPrecisionProvider,
    round_step_down,
)
from app.market_data.order_book import estimate_for_base_amount, estimate_for_quote_amount
from app.market_data.store import MarketDataStore
from app.models.arbitrage import ArbitrageLeg, ArbitrageOpportunity, ProfitBreakdown
from app.models.base import DEC0
from app.models.enums import ArbitrageStrategy, MarketType, OrderSide
from app.models.market_data import OrderBook
from app.models.symbol import Symbol

from .fees import FeeProvider, ScanRequest, StaticFeeProvider, VenueProvider

__all__ = ["TriangleRoute", "TriangularScanner"]

logger = get_logger("strategies.triangular")

_BPS = Decimal("10000")
_QUANTUM = Decimal("0.00000001")
_USDT = "USDT"


@dataclass(frozen=True, slots=True)
class _CycleWalk:
    """Successful forward pass over the three legs (rounded amounts included)."""

    est1: object
    amt1: Decimal
    est2: object
    amt2: Decimal
    est3: object
    amt3_pre_fee: Decimal
    proceeds: Decimal


#: Estimates must fill essentially completely for a ring to be usable.
_FULL_FILL = Decimal("0.999")

#: Sizing loop bound: shrink-and-retry at most this many times.
_MAX_SIZING_PASSES = 8


@dataclass(frozen=True, slots=True)
class TriangleRoute:
    """One candidate cycle: three books traded in a fixed order."""

    exchange_id: str
    #: Symbols in trade order; the first and last are always ``*/USDT``.
    symbols: tuple[Symbol, Symbol, Symbol]
    sides: tuple[OrderSide, OrderSide, OrderSide]
    #: The middle book that connects the two USDT-quoted assets.
    leg2_book: OrderBook


class TriangularScanner:
    """Depth-aware 3-leg spot cycle detector per venue."""

    def __init__(
        self,
        *,
        store: MarketDataStore,
        venues: VenueProvider,
        fees: FeeProvider | None = None,
        settings: ArbitrageSettings | None = None,
        clock: Clock | None = None,
        stale_after_ms: int = 2000,
        precision: PrecisionProvider | None = None,
    ) -> None:
        self._store = store
        self._venues = venues
        self._fees = fees or StaticFeeProvider()
        self._settings = settings or ArbitrageSettings()
        self._clock = clock or SystemClock()
        self._stale_after_ms = stale_after_ms
        # Instrument steps for sizing and order rounding.
        self._precision: PrecisionProvider = precision or StaticPrecisionProvider({}, {})

    # ------------------------------------------------------------------ public
    async def scan(self, request: ScanRequest) -> tuple[ArbitrageOpportunity, ...]:
        venues = tuple(dict.fromkeys(v.lower() for v in self._venues()))
        wanted = {s.name for s in request.symbols}
        opportunities: list[ArbitrageOpportunity] = []
        for venue in venues:
            try:
                opportunities.extend(self._scan_venue(venue, request, wanted))
            except Exception as exc:  # noqa: BLE001 - one broken venue must not sink the scan
                logger.warning(
                    "triangular_scan_venue_failed", extra={"venue": venue, "error": str(exc)}
                )
        opportunities.sort(key=lambda item: item.net_profit_bps, reverse=True)
        return tuple(opportunities[: request.max_results])

    # ------------------------------------------------------------------ private
    def _scan_venue(
        self,
        venue: str,
        request: ScanRequest,
        wanted: set[str],
    ) -> list[ArbitrageOpportunity]:
        cached = self._venue_spot_books(venue, request.require_fresh_data)
        if len(cached) < 2:
            return []

        usdt_books: dict[str, OrderBook] = {}
        crosses: dict[tuple[str, str], OrderBook] = {}
        for name, book in cached.items():
            symbol = book.symbol
            if symbol.quote == _USDT:
                # The watchlist governs which assets may start/end a cycle;
                # middle cross books are always eligible.
                if wanted and name not in wanted:
                    continue
                usdt_books[symbol.base] = book
            elif symbol.base == _USDT:
                if wanted and name not in wanted:
                    continue
                # Inverse-quoted USDT pair (rare); treat as base asset too.
                usdt_books[symbol.quote] = book
            else:
                crosses[(symbol.base, symbol.quote)] = book

        if len(usdt_books) < 1:
            return []

        notional = min(request.notional_quote, self._settings.triangle_max_notional_quote)
        min_net = (
            request.min_net_profit_bps
            if request.min_net_profit_bps is not None
            else self._settings.triangle_min_net_bps
        )

        found: list[ArbitrageOpportunity] = []
        for first_base, leg1_book in usdt_books.items():
            for second_base, leg3_book in usdt_books.items():
                if first_base == second_base:
                    continue
                route = self._middle_edge(first_base, second_base, crosses)
                if route is None:
                    continue
                try:
                    opportunity = self._evaluate(
                        venue=venue,
                        route=route,
                        leg1_book=leg1_book,
                        leg2_book=route.leg2_book,
                        leg3_book=leg3_book,
                        notional=notional,
                        ttl_ms=request.ttl_ms,
                        require_fresh=request.require_fresh_data,
                    )
                except Exception as exc:  # noqa: BLE001 - contained per ring
                    logger.debug(
                        "triangular_ring_failed",
                        extra={
                            "venue": venue,
                            "ring": f"{first_base}->{second_base}",
                            "error": str(exc),
                        },
                    )
                    continue
                if opportunity is None:
                    continue
                if opportunity.net_profit_bps >= min_net:
                    found.append(opportunity)

        found.sort(key=lambda item: item.net_profit_bps, reverse=True)
        return found

    @staticmethod
    def _middle_edge(
        first: str, second: str, crosses: dict[tuple[str, str], OrderBook]
    ) -> TriangleRoute | None:
        """Build the middle leg between ``first`` (held) and ``second`` (wanted)."""
        sym_first_usdt = Symbol(base=first, quote=_USDT)
        sym_second_usdt = Symbol(base=second, quote=_USDT)
        # Prefer buying `second` paying `first` (book second/first).
        book = crosses.get((second, first))
        if book is not None:
            return TriangleRoute(
                exchange_id=book.exchange_id,
                symbols=(sym_first_usdt, book.symbol, sym_second_usdt),
                sides=(OrderSide.BUY, OrderSide.BUY, OrderSide.SELL),
                leg2_book=book,
            )
        # Otherwise sell `first` receiving `second` (book first/second).
        book = crosses.get((first, second))
        if book is not None:
            return TriangleRoute(
                exchange_id=book.exchange_id,
                symbols=(sym_first_usdt, book.symbol, sym_second_usdt),
                sides=(OrderSide.BUY, OrderSide.SELL, OrderSide.SELL),
                leg2_book=book,
            )
        return None

    def _evaluate(
        self,
        *,
        venue: str,
        route: TriangleRoute,
        leg1_book: OrderBook,
        leg2_book: OrderBook,
        leg3_book: OrderBook,
        notional: Decimal,
        ttl_ms: int = 1500,
        require_fresh: bool = True,
    ) -> ArbitrageOpportunity | None:
        """Size the cycle so EVERY book can carry its leg.

        Backward feasibility by iteration: walk the cycle; when any book cannot
        fill its whole leg, shrink the starting notional by that book's
        supported/requested ratio and retry.  Between attempts every amount is
        rounded DOWN onto its market's ``amount_step`` and the chain is
        re-walked from the rounded value, so the published route is placeable
        verbatim (D5).
        """
        max_age = self._stale_after_ms
        if require_fresh:
            for book in (leg1_book, leg2_book, leg3_book):
                if (self._clock.now() - book.timestamp).total_seconds() * 1000.0 > max_age:
                    return None

        taker1 = self._fees.fees_for(venue, route.symbols[0], MarketType.SPOT).taker_bps / _BPS
        taker2 = self._fees.fees_for(venue, route.symbols[1], MarketType.SPOT).taker_bps / _BPS
        taker3 = self._fees.fees_for(venue, route.symbols[2], MarketType.SPOT).taker_bps / _BPS
        filters_list = [self._precision.filters_for(venue, symbol.name) for symbol in route.symbols]
        # Venue floors (D5): a cycle that can only exist below any market's
        # minimum notional / amount is not a signal at all.
        entry_floor = next(
            (f.min_cost for f in (filters_list[0],) if f is not None and f.min_cost is not None),
            None,
        )

        def _step(idx: int) -> Decimal | None:
            f = filters_list[idx]
            return f.amount_step if f is not None else None

        q0 = notional
        walk: _CycleWalk | None = None
        for _ in range(_MAX_SIZING_PASSES):
            if entry_floor is not None and q0 < entry_floor:
                return None
            outcome = self._walk_cycle(
                route=route,
                books=(leg1_book, leg2_book, leg3_book),
                q0=q0,
                takers=(taker1, taker2, taker3),
                steps=(_step(0), _step(1), _step(2)),
            )
            if isinstance(outcome, Decimal):
                # Shrink to what the binding book supports and retry.
                if outcome <= DEC0:
                    return None
                q0 = (q0 * outcome * Decimal("0.999")).quantize(_QUANTUM)
                continue
            walk = outcome
            break
        if walk is None or walk.est3.quote_amount <= DEC0:
            return None

        slippage_total = (
            walk.est1.slippage_bps + walk.est2.slippage_bps + walk.est3.slippage_bps
        ).quantize(Decimal("0.0001"))
        max_slip = self._settings.triangle_max_leg_slippage_bps
        if any(
            slip > max_slip
            for slip in (
                walk.est1.slippage_bps,
                walk.est2.slippage_bps,
                walk.est3.slippage_bps,
            )
        ):
            return None

        spend = walk.est1.quote_amount
        if spend <= DEC0:
            return None
        net_bps = ((walk.proceeds / spend - 1) * _BPS).quantize(Decimal("0.0001"))
        fees_bps = ((taker1 + taker2 + taker3) * _BPS).quantize(Decimal("0.0001"))
        gross_bps = (net_bps + fees_bps + slippage_total).quantize(Decimal("0.0001"))
        profit = ProfitBreakdown(
            notional_quote=spend.quantize(_QUANTUM),
            gross_spread_bps=gross_bps,
            trading_fees_bps=fees_bps,
            slippage_bps=slippage_total,
        )

        legs = (
            ArbitrageLeg(
                exchange_id=venue,
                symbol=route.symbols[0],
                market_type=MarketType.SPOT,
                side=OrderSide.BUY,
                price=walk.est1.average_price,
                amount=walk.amt1,
                fee_bps=taker1 * _BPS,
                estimate=walk.est1,
            ),
            ArbitrageLeg(
                exchange_id=venue,
                symbol=route.symbols[1],
                market_type=MarketType.SPOT,
                side=route.sides[1],
                price=walk.est2.average_price,
                amount=walk.amt2,
                fee_bps=taker2 * _BPS,
                estimate=walk.est2,
            ),
            ArbitrageLeg(
                exchange_id=venue,
                symbol=route.symbols[2],
                market_type=MarketType.SPOT,
                side=OrderSide.SELL,
                price=walk.est3.average_price,
                amount=walk.amt3_pre_fee,
                fee_bps=taker3 * _BPS,
                estimate=walk.est3,
            ),
        )

        detected_at = self._clock.now()
        route_text = f"{_USDT}->{route.symbols[0].base}->{route.symbols[2].base}->{_USDT} @ {venue}"
        return ArbitrageOpportunity(
            strategy=ArbitrageStrategy.TRIANGLE,
            symbol=route.symbols[0],
            buy_leg=legs[0],
            sell_leg=legs[2],
            legs_route=legs,
            profit=profit,
            size_notional_quote=spend.quantize(_QUANTUM),
            max_notional_quote=min(
                _side_notional(leg1_book, OrderSide.BUY),
                _side_notional(leg2_book, route.sides[1]),
                _side_notional(leg3_book, OrderSide.SELL),
            ),
            direction=route_text,
            data_age_ms=max(
                (self._clock.now() - book.timestamp).total_seconds() * 1000.0
                for book in (leg1_book, leg2_book, leg3_book)
            ),
            latency_ms=max(book.transport_latency_ms for book in (leg1_book, leg2_book, leg3_book)),
            detected_at=detected_at,
            expires_at=detected_at + timedelta(milliseconds=ttl_ms),
            notes=(route_text,),
        )

    def _walk_cycle(
        self,
        *,
        route: TriangleRoute,
        books: tuple[OrderBook, OrderBook, OrderBook],
        q0: Decimal,
        takers: tuple[Decimal, Decimal, Decimal],
        steps: tuple[Decimal | None, Decimal | None, Decimal | None],
    ) -> Decimal | _CycleWalk:
        """One forward pass; a bare Decimal is the shrink factor to retry with."""
        book1, book2, book3 = books

        est1 = estimate_for_quote_amount(book1, OrderSide.BUY, q0)
        if est1.fill_ratio < _FULL_FILL:
            supported = sum((level.notional for level in book1.asks), DEC0)
            return supported / q0 if q0 > DEC0 else DEC0
        amt1 = round_step_down(est1.filled_amount * (1 - takers[0]), steps[0])
        if amt1 <= DEC0:
            return DEC0

        if route.sides[1] is OrderSide.BUY:
            est2 = estimate_for_quote_amount(book2, OrderSide.BUY, amt1)
            needed2 = amt1
            supported2 = sum((level.notional for level in book2.asks), DEC0)
        else:
            est2 = estimate_for_base_amount(book2, OrderSide.SELL, amt1)
            needed2 = amt1
            supported2 = sum((level.amount for level in book2.bids), DEC0)
        if est2.fill_ratio < _FULL_FILL:
            return supported2 / needed2 if needed2 > DEC0 else DEC0
        held_y_raw = (
            est2.filled_amount * (1 - takers[1])
            if route.sides[1] is OrderSide.BUY
            else est2.quote_amount * (1 - takers[1])
        )
        amt2 = round_step_down(held_y_raw, steps[1])
        if amt2 <= DEC0:
            return DEC0

        est3 = estimate_for_base_amount(book3, OrderSide.SELL, amt2)
        if est3.fill_ratio < _FULL_FILL:
            supported3 = sum((level.amount for level in book3.bids), DEC0)
            return supported3 / amt2 if amt2 > DEC0 else DEC0
        proceeds = (est3.quote_amount * (1 - takers[2])).quantize(_QUANTUM)

        return _CycleWalk(
            est1=est1,
            amt1=amt1,
            est2=est2,
            amt2=amt2,
            est3=est3,
            amt3_pre_fee=amt2,
            proceeds=proceeds,
        )

    def _venue_spot_books(self, venue: str, fresh_only: bool) -> dict[str, OrderBook]:
        """Every cached SPOT book of one venue keyed by symbol name."""
        out: dict[str, OrderBook] = {}
        for name, book in self._store.books(exchange_id=venue).items():
            if fresh_only and self._store.is_stale(self._store.age_ms(book)):
                continue
            out[name] = book
        return out


def _side_notional(book: OrderBook, side: OrderSide) -> Decimal:
    levels = book.side(side)
    return sum((level.notional for level in levels), DEC0)
