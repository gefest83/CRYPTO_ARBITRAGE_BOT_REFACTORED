"""Shared error kernel.

This module is intentionally dependency-free: every layer may import it and it
must never import anything from :mod:`app`.  Errors carry a stable ``code`` so
the API layer can map them to HTTP responses without ``isinstance`` chains
spread across the codebase.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CapabilityNotSupportedError",
    "ConfigurationError",
    "DiscoveryError",
    "ExchangeError",
    "ExchangeUnavailableError",
    "ExecutionDisabledError",
    "NotFoundError",
    "RateLimitError",
    "RiskViolationError",
    "TerminalError",
    "TimeoutError",
    "ValidationError",
    "VenueAuthError",
]


class TerminalError(Exception):
    """Base class for all domain/application errors of the terminal."""

    code: str = "terminal_error"
    http_status: int = 500

    def __init__(self, message: str, /, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context

    def to_dict(self) -> dict[str, Any]:
        """Wire representation consumed by the API error handler."""
        return {"code": self.code, "message": self.message, "context": self.context}

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


class ConfigurationError(TerminalError):
    """Invalid or missing configuration."""

    code = "configuration_error"
    http_status = 500


class ValidationError(TerminalError):
    """Input failed domain validation."""

    code = "validation_error"
    http_status = 422


class NotFoundError(TerminalError):
    """Requested entity does not exist."""

    code = "not_found"
    http_status = 404


class ExchangeError(TerminalError):
    """Generic exchange-side failure."""

    code = "exchange_error"
    http_status = 502


class ExchangeUnavailableError(ExchangeError):
    """Exchange is offline, in maintenance or otherwise unusable right now.

    A single unavailable exchange must never break the terminal: callers are
    expected to degrade gracefully and keep the remaining venues running.
    """

    code = "exchange_unavailable"
    http_status = 503


class RateLimitError(ExchangeError):
    """Rate limit exceeded (HTTP 429 / ccxt RateLimitExceeded / DDoSProtection)."""

    code = "rate_limit_exceeded"
    http_status = 429


class VenueAuthError(ExchangeError):
    """The venue rejected the API keys (auth/permission/nonce).

    A credentials problem, not a network one —
    it must never open the venue's network circuit breaker nor silence the
    public market data.  The venue keeps serving public data; only private
    calls are stopped immediately.
    """

    code = "venue_auth_error"
    http_status = 502


class TimeoutError(ExchangeError):
    """Request timeout (ccxt RequestTimeout)."""

    code = "timeout"
    http_status = 504


class CapabilityNotSupportedError(ExchangeError):
    """The exchange adapter does not implement the requested capability."""

    code = "capability_not_supported"
    http_status = 501


class DiscoveryError(TerminalError):
    """Exchange discovery source failed (e.g. CoinMarketCap unreachable)."""

    code = "discovery_error"
    http_status = 503


class RiskViolationError(TerminalError):
    """A risk rule blocked the requested action."""

    code = "risk_violation"
    http_status = 409


class ExecutionDisabledError(TerminalError):
    """Order execution is disabled by mode policy or by the kill switch."""

    code = "execution_disabled"
    http_status = 409
