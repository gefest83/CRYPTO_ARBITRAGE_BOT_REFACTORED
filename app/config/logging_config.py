"""Logging setup: structured JSON for production, readable console for dev.

A ``correlation_id`` context variable is attached to every record so that a
request (later: an arbitrage operation) can be traced end-to-end across modules.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from contextvars import ContextVar
from typing import Any

from app.config.settings import LoggingSettings

__all__ = [
    "bind_correlation_id",
    "configure_logging",
    "current_correlation_id",
    "get_logger",
]

correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)

_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
        "correlation_id",
    }
)

_configured = False


def bind_correlation_id(value: str | None) -> None:
    """Attach a correlation id to the current async context."""
    correlation_id_var.set(value)


def current_correlation_id() -> str | None:
    return correlation_id_var.get()


class CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """Single-line JSON records, suitable for log shipping."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = _jsonable(value)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    default_format = "%(asctime)s %(levelname)-8s [%(correlation_id)s] %(name)s: %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.default_format, datefmt="%H:%M:%S")


def _jsonable(value: Any) -> Any:
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    return str(value)


def configure_logging(config: LoggingSettings | None = None, *, force: bool = False) -> None:
    """Configure the root logger. Idempotent unless ``force=True``."""
    global _configured
    if _configured and not force:
        return

    config = config or LoggingSettings()
    formatter: logging.Formatter = JsonFormatter() if config.json_logs else ConsoleFormatter()

    handlers: list[logging.Handler] = []
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    handlers.append(stream)

    if config.file is not None:
        config.file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            config.file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    correlation_filter = CorrelationFilter()
    for handler in handlers:
        handler.addFilter(correlation_filter)
        root.addHandler(handler)
    root.setLevel(config.level)

    # Uvicorn duplicates records through its own handlers; let the root own output.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(noisy)
        logger.handlers.clear()
        logger.propagate = True

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Namespaced logger (``cat.<module>``).

    Note: never pass ``extra={"message": ...}`` — ``message`` and other
    :class:`logging.LogRecord` attributes are reserved (use ``detail`` instead).
    """
    return logging.getLogger(name if name.startswith("cat") else f"cat.{name}")
