"""Adapter registry: maps an adapter name to a factory.

The arbitrage core never imports a concrete venue class.  It asks the registry
for an adapter by ``Exchange.adapter``, which makes adding/removing venues a
configuration concern.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from app.errors import ConfigurationError
from app.models.exchange import Exchange

from .base import AdapterOptions, BaseExchangeAdapter, OrderGate
from .credentials import ExchangeCredentials

__all__ = ["AdapterFactory", "AdapterRegistry", "AdapterRequest"]


@dataclass(frozen=True, slots=True)
class AdapterRequest:
    """Everything a factory needs to build an adapter."""

    exchange: Exchange
    options: AdapterOptions
    credentials: ExchangeCredentials | None = None
    order_gate: OrderGate | None = None


AdapterFactory = Callable[[AdapterRequest], BaseExchangeAdapter]


class AdapterRegistry:
    """Name -> factory mapping with explicit registration (no import side effects)."""

    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, name: str, factory: AdapterFactory, *, override: bool = False) -> None:
        key = name.strip().lower()
        if not key:
            raise ConfigurationError("adapter name must not be empty")
        if key in self._factories and not override:
            raise ConfigurationError(f"adapter '{key}' is already registered")
        self._factories[key] = factory

    def unregister(self, name: str) -> None:
        self._factories.pop(name.strip().lower(), None)

    def factory(self, name: str) -> AdapterFactory:
        try:
            return self._factories[name.strip().lower()]
        except KeyError as exc:
            raise ConfigurationError(
                f"unknown adapter '{name}'; registered: {', '.join(self.names()) or 'none'}",
                adapter=name,
            ) from exc

    def create(self, request: AdapterRequest) -> BaseExchangeAdapter:
        return self.factory(request.exchange.adapter)(request)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.strip().lower() in self._factories

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._factories)
