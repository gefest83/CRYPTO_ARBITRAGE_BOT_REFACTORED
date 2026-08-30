"""Exchange credentials handling.

Credentials never travel through domain models or API responses; only the boolean
``Exchange.has_credentials`` flag is exposed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "CredentialsProvider",
    "EnvCredentialsProvider",
    "ExchangeCredentials",
    "InMemoryCredentialsProvider",
]

ENV_TEMPLATE = "CAT_KEY_{exchange}_{field}"

#: Values parsed from the operator's project-local ``.env`` (same lookup order
#: as :class:`~app.config.settings.Settings`).  pydantic-settings merges them
#: into the settings model but never into ``os.environ``, so the credentials
#: provider consults them as a fallback — otherwise a bot launched from any
#: working directory would silently run keyless.
_DOTENV_CACHE: dict[str, str] | None = None


def _dotenv_values() -> dict[str, str]:
    global _DOTENV_CACHE
    if _DOTENV_CACHE is None:
        values: dict[str, str] = {}
        try:
            from dotenv import dotenv_values

            for key, value in (dotenv_values(".env") or {}).items():
                if value:
                    values[str(key)] = str(value)
        except Exception:  # noqa: BLE001 - a missing/unreadable file means "no fallback"
            pass
        _DOTENV_CACHE = values
    return _DOTENV_CACHE


def reset_dotenv_cache() -> None:
    """Test hook: forget the parsed .env snapshot (cwd may have changed)."""
    global _DOTENV_CACHE
    _DOTENV_CACHE = None


@dataclass(frozen=True, slots=True)
class ExchangeCredentials:
    """API credentials of a single exchange account."""

    api_key: str = ""
    secret: str = ""
    password: str = ""
    uid: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.api_key and self.secret)

    def __repr__(self) -> str:  # never leak secrets into logs/tracebacks
        state = "empty" if self.is_empty else "set"
        return f"ExchangeCredentials({state})"

    __str__ = __repr__


@runtime_checkable
class CredentialsProvider(Protocol):
    def get(self, exchange_id: str) -> ExchangeCredentials | None: ...

    def has(self, exchange_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class EnvCredentialsProvider:
    """Reads ``CAT_KEY_<EXCHANGE>_APIKEY`` / ``_SECRET`` / ``_PASSWORD`` / ``_UID``."""

    def get(self, exchange_id: str) -> ExchangeCredentials | None:
        key = self._env(exchange_id, "APIKEY")
        secret = self._env(exchange_id, "SECRET")
        if not key or not secret:
            return None
        password = self._env(exchange_id, "PASSWORD") or self._env(exchange_id, "PASSPHRASE")
        return ExchangeCredentials(
            api_key=key,
            secret=secret,
            password=password,
            uid=self._env(exchange_id, "UID"),
        )

    def has(self, exchange_id: str) -> bool:
        return self.get(exchange_id) is not None

    @staticmethod
    def _env(exchange_id: str, field: str) -> str:
        name = ENV_TEMPLATE.format(exchange=exchange_id.strip().upper(), field=field)
        value = os.getenv(name)
        if not value:
            value = _dotenv_values().get(name, "")
        return (value or "").strip()


@dataclass(frozen=True, slots=True)
class InMemoryCredentialsProvider:
    """Used by PAPER mode and tests."""

    credentials: dict[str, ExchangeCredentials]

    def get(self, exchange_id: str) -> ExchangeCredentials | None:
        return self.credentials.get(exchange_id.strip().lower())

    def has(self, exchange_id: str) -> bool:
        return self.get(exchange_id) is not None
