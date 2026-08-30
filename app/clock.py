"""Injectable clock — keeps time-dependent logic testable and deterministic."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from app.models.base import utc_now

__all__ = ["Clock", "FixedClock", "SystemClock", "VirtualClock"]


@runtime_checkable
class Clock(Protocol):
    """Source of the current UTC time."""

    def now(self) -> datetime: ...

    def monotonic_ms(self) -> float: ...


class VirtualClock:
    """Event-time clock for deterministic pipelines.

    ``now()`` only moves when :meth:`advance_to` is called with a later
    timestamp, so pipeline logic (staleness, ages, TTLs) follows the
    recording's timeline instead of the wall clock.
    """

    __slots__ = ("_current", "_origin")

    def __init__(self, start: datetime) -> None:
        self._current = start
        self._origin = start

    @property
    def current(self) -> datetime:
        return self._current

    def now(self) -> datetime:
        return self._current

    def monotonic_ms(self) -> float:
        return (self._current - self._origin).total_seconds() * 1000.0

    def advance_to(self, ts: datetime) -> None:
        """Move to ``ts``; never moves backwards."""
        if ts > self._current:
            self._current = ts


@dataclass(frozen=True, slots=True)
class SystemClock:
    """Default production clock."""

    def now(self) -> datetime:
        return utc_now()

    def monotonic_ms(self) -> float:
        from time import perf_counter

        return perf_counter() * 1000.0


@dataclass(slots=True)
class FixedClock:
    """Deterministic clock for tests; advance it manually."""

    current: datetime = field(default_factory=utc_now)
    elapsed_ms: float = 0.0

    def now(self) -> datetime:
        return self.current

    def monotonic_ms(self) -> float:
        return self.elapsed_ms

    def advance(self, *, seconds: float = 0.0, ms: float = 0.0) -> None:
        delta = seconds * 1000.0 + ms
        self.current = self.current + timedelta(milliseconds=delta)
        self.elapsed_ms += delta
