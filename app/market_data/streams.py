"""Stream supervision for real-time market data.

Supervises async iterators from exchange adapters (watch_ticker, watch_order_book)
with automatic restart, exponential backoff + jitter, and resubscription.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.clock import Clock, SystemClock
from app.config.logging_config import get_logger
from app.models.enums import HealthStatus, StreamKind
from app.models.health import StreamStatus

__all__ = ["Sleeper", "StreamSpec", "StreamSupervisor"]

logger = get_logger("core.market_data.streams")


@dataclass(frozen=True, slots=True)
class StreamSpec:
    """Specification of a stream to supervise."""

    exchange_id: str
    symbol: str
    kind: StreamKind
    market_type: str = "spot"


Sleeper = Callable[[float], Awaitable[None]]


def failure_log_level(consecutive_failures: int, *, repeat_limit: int) -> int:
    """P10: the first attempts are WARNING; the same failure repeating is DEBUG."""
    return logging.WARNING if consecutive_failures <= max(0, repeat_limit) else logging.DEBUG


def reconnect_log_level(total_restarts: int, *, repeat_limit: int) -> int:
    """P10: the first reconnect notes are INFO; later ones are DEBUG."""
    return logging.INFO if total_restarts <= max(0, repeat_limit) else logging.DEBUG


#: Exception class names that mean "this (venue, symbol) pair will never work"
#: — e.g. ccxt BadSymbol for a market the venue does not list. Retrying such
#: a stream 5x over 45s only burns WS slots / rate limit and floods the log;
#: the stream must suspend immediately and let REST mark the pair unsupported.
_PERMANENT_STREAM_ERROR_NAMES = frozenset({"BadSymbol", "NotSupported", "ArgumentsRequired"})


def is_permanent_stream_error(exc: BaseException) -> bool:
    """True when `exc` (or any chained cause) is a permanent listing fact."""
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        names = {cls.__name__ for cls in type(current).__mro__}
        if names & _PERMANENT_STREAM_ERROR_NAMES:
            return True
        # ccxt phrases the same fact as plain ExchangeError text.
        message = str(current).lower()
        if (
            "does not have market symbol" in message
            or "symbol not found" in message
            # OKX WS: one delisted instId (e.g. MATIC-USDT after the POL
            # migration) poisons the shared connection — every stream on it
            # fails with code 60018 "doesn't exist". The culprit pair will
            # never subscribe; it must be marked unsupported, not retried.
            or "doesn't exist" in message
            or "does not exist" in message
            or ("subscribe failed" in message and "wrong url or channel" in message)
        ):
            return True
        cause = getattr(current, "__cause__", None)
        if isinstance(cause, BaseException):
            stack.append(cause)
        context = getattr(current, "__context__", None)
        if isinstance(context, BaseException):
            stack.append(context)
    return False


def extract_ws_culprit_symbol(exc: BaseException) -> str | None:
    """Unified ``BASE/QUOTE`` named by a WS subscription error, if any.

    OKX multiplexes all subscriptions over one connection, so a single bad
    instId (``instId:MATIC-USDT``) raises on *every* stream sharing the
    connection — including healthy ones like BTC/USDT. The error text names
    the real culprit; callers use it to mark exactly that pair unsupported
    instead of penalising the innocent stream that happened to surface it.
    """
    import re

    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    pattern = re.compile(r"instId\s*:\s*([A-Za-z0-9]+)-([A-Za-z0-9]+)", re.IGNORECASE)
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        match = pattern.search(str(current))
        if match:
            base, quote = match.group(1).upper(), match.group(2).upper()
            # OKX uses USDT/USDC/BTC/ETH as quote; default to USDT shape.
            return f"{base}/{quote}"
        cause = getattr(current, "__cause__", None)
        if isinstance(cause, BaseException):
            stack.append(cause)
        context = getattr(current, "__context__", None)
        if isinstance(context, BaseException):
            stack.append(context)
    return None


@dataclass(slots=True)
class _StreamTask:
    """Internal state for a supervised stream."""

    spec: StreamSpec
    task: asyncio.Task[None] | None = None
    iterator: AsyncIterator[Any] | None = None
    backoff_seconds: float = 0.0
    consecutive_failures: int = 0
    total_restarts: int = 0
    last_event_at: datetime | None = None
    last_error: str | None = None
    status: HealthStatus = HealthStatus.UNKNOWN
    stopped: bool = False
    restarted_at: datetime | None = None
    #: P10: whether this stream ever delivered a single event.  A stream that
    #: never delivered anything across many restarts (e.g. a venue whose
    #: testnet/sandbox endpoint has no WebSocket at all) is quarantined instead
    #: of retrying forever.
    ever_delivered: bool = False
    suspended: bool = False
    #: Monotonic birth time of the supervised stream (quarantine grace window).
    started_at_ms: float = 0.0


class StreamSupervisor:
    """Supervises multiple market data streams with restart/backoff/resubscribe."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        sleeper: Sleeper | None = None,
        backoff_base_seconds: float = 0.5,
        backoff_max_seconds: float = 30.0,
        backoff_jitter: float = 0.25,
        max_streams_per_exchange: int = 4,
        max_parallel_streams: int = 16,
        #: P10 WS strategy: after this many restarts without a single delivered
        #: event the stream is suspended (no more retries, REST refresh takes
        #: over the symbol).  ``0`` disables quarantine — chaos tests and any
        #: caller that wants unconditional retries keep the old behaviour.
        quarantine_restarts: int = 0,
        #: Minimum stream age before the quarantine may fire: a stream that is
        #: still warming up (early lives failing before the first snapshot)
        #: must not be killed by the restart count alone.
        quarantine_after_seconds: float = 45.0,
        #: stream_failed / stream_reconnecting stay at WARNING/INFO for the
        #: first attempts of a stream, then drop to DEBUG so a dead endpoint
        #: cannot flood the operator log.
        log_repeat_limit: int = 3,
    ) -> None:
        self._clock = clock or SystemClock()
        self._sleeper = sleeper or (lambda seconds: asyncio.sleep(seconds))
        self._backoff_base = backoff_base_seconds
        self._backoff_max = backoff_max_seconds
        self._backoff_jitter = backoff_jitter
        self._max_per_exchange = max_streams_per_exchange
        self._max_parallel = max_parallel_streams
        self._quarantine_restarts = quarantine_restarts
        self._quarantine_after_ms = quarantine_after_seconds * 1000.0
        self._log_repeat_limit = log_repeat_limit

        self._streams: dict[str, _StreamTask] = {}
        self._semaphore = asyncio.Semaphore(max_parallel_streams)
        self._per_exchange_semaphores: dict[str, asyncio.Semaphore] = {}
        self._running = False

    def _get_exchange_semaphore(self, exchange_id: str) -> asyncio.Semaphore:
        sem = self._per_exchange_semaphores.get(exchange_id)
        if sem is None:
            sem = asyncio.Semaphore(self._max_per_exchange)
            self._per_exchange_semaphores[exchange_id] = sem
        return sem

    def _stream_key(self, spec: StreamSpec) -> str:
        return f"{spec.exchange_id}:{spec.symbol}:{spec.kind.value}:{spec.market_type}"

    def _next_backoff(self, current: float) -> float:
        """Exponential backoff with jitter, capped at max."""
        next_val = self._backoff_base if current <= 0 else min(current * 2, self._backoff_max)
        jitter = next_val * self._backoff_jitter * random.uniform(-1, 1)
        return min(max(next_val + jitter, self._backoff_base), self._backoff_max)

    async def start_stream(
        self,
        spec: StreamSpec,
        iterator_factory: Callable[[], AsyncIterator[Any]],
        on_event: Callable[[Any], Awaitable[None]] | Callable[[Any], None],
    ) -> None:
        """Start supervising a stream.

        Args:
            spec: Stream specification (exchange, symbol, kind).
            iterator_factory: Zero-arg callable that returns a fresh async iterator.
            on_event: Callback for each event yielded by the iterator.
        """
        key = self._stream_key(spec)
        existing = self._streams.get(key)
        if existing is not None:
            if existing.suspended:
                # Quarantine lifted: a fresh registration replaces the
                # suspended attempt with clean counters.
                self._streams.pop(key, None)
            else:
                logger.warning("stream_already_running", extra={"stream": key})
                return

        task_state = _StreamTask(spec=spec, started_at_ms=self._clock.monotonic_ms())
        self._streams[key] = task_state

        async def _run() -> None:
            await self._run_stream(task_state, iterator_factory, on_event)

        task_state.task = asyncio.create_task(_run())
        logger.info(
            "stream_started",
            extra={
                "exchange_id": spec.exchange_id,
                "symbol": spec.symbol,
                "kind": spec.kind.value,
                "market_type": spec.market_type,
            },
        )

    async def _run_stream(
        self,
        state: _StreamTask,
        iterator_factory: Callable[[], AsyncIterator[Any]],
        on_event: Callable[[Any], Awaitable[None]] | Callable[[Any], None],
    ) -> None:
        """Main supervision loop for a single stream.

        C-5: semaphore acquisition is guarded — a cancellation (or any other
        error) between acquiring the global permit and the exchange permit
        releases what was already acquired.  Without this, a cancelled stream
        waiting on its exchange semaphore would leak a global permit forever.
        """
        while not state.stopped:
            acquired_global = False
            acquired_exchange = False
            ex_sem: asyncio.Semaphore | None = None
            try:
                await self._semaphore.acquire()
                acquired_global = True
                ex_sem = self._get_exchange_semaphore(state.spec.exchange_id)
                await ex_sem.acquire()
                acquired_exchange = True
            except BaseException:
                # CancelledError (and any acquisition failure) must not leak
                # permits that were already taken.
                if acquired_exchange and ex_sem is not None:
                    ex_sem.release()
                if acquired_global:
                    self._semaphore.release()
                raise

            try:
                iterator = iterator_factory()
                state.iterator = iterator
                state.status = HealthStatus.HEALTHY

                async for event in iterator:
                    if state.stopped:
                        break
                    state.last_event_at = self._clock.now()
                    state.ever_delivered = True
                    if asyncio.iscoroutinefunction(on_event):
                        await on_event(event)
                    else:
                        on_event(event)
                    state.consecutive_failures = 0

            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - any stream failure triggers restart
                # Permanent listing facts (BadSymbol / unknown market) never
                # become healthy by retrying: suspend immediately so the slot
                # is freed and REST can mark the pair unsupported once.
                # Shared-connection poisoning (OKX: one bad instId fails every
                # stream on the connection) is the exception: when the error
                # names a DIFFERENT pair than this stream, this stream is
                # innocent — back off and retry while the culprit's own stream
                # suspends itself permanently.
                if is_permanent_stream_error(exc):
                    culprit = extract_ws_culprit_symbol(exc)
                    if culprit is not None and culprit != state.spec.symbol:
                        state.consecutive_failures += 1
                        state.last_error = str(exc)[:300]
                        state.status = (
                            HealthStatus.DEGRADED
                            if state.consecutive_failures < 3
                            else HealthStatus.UNHEALTHY
                        )
                        logger.log(
                            failure_log_level(
                                state.consecutive_failures,
                                repeat_limit=self._log_repeat_limit,
                            ),
                            "stream_poisoned_by_other_symbol",
                            extra={
                                "exchange_id": state.spec.exchange_id,
                                "symbol": state.spec.symbol,
                                "kind": state.spec.kind.value,
                                "culprit_symbol": culprit,
                                "error": str(exc)[:200],
                                "consecutive_failures": state.consecutive_failures,
                            },
                        )
                    else:
                        state.consecutive_failures += 1
                        state.last_error = str(exc)[:300]
                        state.suspended = True
                        state.status = HealthStatus.UNKNOWN
                        logger.info(
                            "stream_suspended_unsupported",
                            extra={
                                "exchange_id": state.spec.exchange_id,
                                "symbol": state.spec.symbol,
                                "kind": state.spec.kind.value,
                                "market_type": state.spec.market_type,
                                "last_error": state.last_error,
                                "culprit_symbol": culprit or state.spec.symbol,
                                "reason": "venue does not list this market; REST marks it unsupported",
                            },
                        )
                        break
                else:
                    state.consecutive_failures += 1
                    state.last_error = str(exc)
                    if state.consecutive_failures < 3:
                        state.status = HealthStatus.DEGRADED
                    else:
                        state.status = HealthStatus.UNHEALTHY
                    logger.log(
                        failure_log_level(
                            state.consecutive_failures, repeat_limit=self._log_repeat_limit
                        ),
                        "stream_failed",
                        extra={
                            "exchange_id": state.spec.exchange_id,
                            "symbol": state.spec.symbol,
                            "kind": state.spec.kind.value,
                            "error": str(exc),
                            "consecutive_failures": state.consecutive_failures,
                        },
                    )
            finally:
                iterator_to_close = state.iterator
                state.iterator = None
                if iterator_to_close is not None:
                    aclose = getattr(iterator_to_close, "aclose", None)
                    if callable(aclose):
                        try:
                            await aclose()
                        except Exception:
                            pass
                if ex_sem is not None:
                    ex_sem.release()
                self._semaphore.release()

            if state.stopped:
                break

            # Backoff before restart
            state.backoff_seconds = self._next_backoff(state.backoff_seconds)
            state.total_restarts += 1
            state.restarted_at = self._clock.now()
            state.status = HealthStatus.DEGRADED

            # P10 WS strategy: a stream that never delivered a single event in
            # its lifetime (typical for venues whose testnet has no WebSocket)
            # stops retrying once it has burned both the restart budget AND the
            # grace window.  Status falls back to UNKNOWN so the REST refresh
            # path covers the symbol again.
            stream_age_ms = self._clock.monotonic_ms() - state.started_at_ms
            if (
                self._quarantine_restarts > 0
                and not state.ever_delivered
                and state.total_restarts >= self._quarantine_restarts
                and stream_age_ms >= self._quarantine_after_ms
            ):
                state.suspended = True
                state.status = HealthStatus.UNKNOWN
                logger.warning(
                    "stream_suspended",
                    extra={
                        "exchange_id": state.spec.exchange_id,
                        "symbol": state.spec.symbol,
                        "kind": state.spec.kind.value,
                        "market_type": state.spec.market_type,
                        "restarts_without_events": state.total_restarts,
                        "last_error": state.last_error,
                        "reason": "no events ever delivered; REST refresh takes over",
                    },
                )
                break

            logger.log(
                reconnect_log_level(
                    total_restarts=state.total_restarts, repeat_limit=self._log_repeat_limit
                ),
                "stream_reconnecting",
                extra={
                    "exchange_id": state.spec.exchange_id,
                    "symbol": state.spec.symbol,
                    "kind": state.spec.kind.value,
                    "backoff_seconds": round(state.backoff_seconds, 2),
                    "total_restarts": state.total_restarts,
                    "last_error": (state.last_error or "")[:160],
                },
            )

            try:
                await self._sleeper(state.backoff_seconds)
            except asyncio.CancelledError:
                break

        state.status = HealthStatus.UNKNOWN
        logger.info(
            "stream_stopped",
            extra={
                "exchange_id": state.spec.exchange_id,
                "symbol": state.spec.symbol,
                "kind": state.spec.kind.value,
                "total_restarts": state.total_restarts,
            },
        )

    async def stop_stream(self, spec: StreamSpec) -> None:
        """Stop supervising a specific stream."""
        key = self._stream_key(spec)
        state = self._streams.pop(key, None)
        if state is None:
            return
        state.stopped = True
        if state.task is not None:
            state.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await state.task
        logger.info(
            "stream_stopped",
            extra={
                "exchange_id": spec.exchange_id,
                "symbol": spec.symbol,
                "kind": spec.kind.value,
            },
        )

    async def stop_all(self) -> None:
        """Stop all supervised streams."""
        keys = list(self._streams.keys())
        for key in keys:
            state = self._streams.get(key)
            if state:
                await self.stop_stream(state.spec)
        self._streams.clear()
        self._per_exchange_semaphores.clear()

    def get_status(self, spec: StreamSpec) -> StreamStatus | None:
        """Get current status of a supervised stream."""
        state = self._streams.get(self._stream_key(spec))
        if state is None:
            return None
        return self._status_of(state)

    def all_statuses(self) -> tuple[StreamStatus, ...]:
        """Get statuses of all supervised streams."""
        return tuple(self._status_of(state) for state in self._streams.values())

    @staticmethod
    def _status_of(state: _StreamTask) -> StreamStatus:
        """Project internal state onto the public status model.

        ``market_type`` must be carried through: the REST refresh path uses these
        statuses to decide which (venue, symbol) pairs a stream already covers,
        and a spot stream does not cover the perp book.
        """
        return StreamStatus(
            exchange_id=state.spec.exchange_id,
            symbol=state.spec.symbol,
            kind=state.spec.kind,
            market_type=state.spec.market_type,
            status=state.status,
            consecutive_failures=state.consecutive_failures,
            last_event_at=state.last_event_at,
            last_error=state.last_error,
            restarted_at=state.restarted_at,
            total_restarts=state.total_restarts,
        )

    def is_running(self, spec: StreamSpec) -> bool:
        state = self._streams.get(self._stream_key(spec))
        return state is not None and not state.stopped and not state.suspended

    def registered_count(self, exchange_id: str | None = None) -> int:
        """Number of live (supervised) streams, optionally for one venue.

        Registration is admission: a registered stream owns a slot even while it
        waits on a semaphore.  Suspended (quarantined) streams own nothing and
        free their slot for a fresh attempt.  Callers use this to cap how many
        specs they feed in, so no stream is ever registered just to starve.
        """
        if exchange_id is None:
            return sum(1 for s in self._streams.values() if not s.suspended)
        return sum(
            1
            for s in self._streams.values()
            if s.spec.exchange_id == exchange_id and not s.suspended
        )

    @property
    def active_count(self) -> int:
        return sum(1 for s in self._streams.values() if not s.stopped and not s.suspended)
