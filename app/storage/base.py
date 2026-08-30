"""SQLAlchemy declarative base and portable column types.

A single naming convention keeps future Alembic migrations deterministic across
SQLite and PostgreSQL.

Two type decorators exist because SQLite does not support the semantics this
domain needs:

* :class:`MoneyDecimal` — SQLite has no native ``NUMERIC``, so SQLAlchemy would
  round-trip money through ``float`` and silently lose precision.  Values are
  stored as exact decimal strings there and as ``NUMERIC(38, 18)`` on PostgreSQL.
* :class:`UtcDateTime` — SQLite drops ``tzinfo``, so timestamps would come back
  naive and be interpreted as local time by clients.  Everything is normalised to
  timezone-aware UTC on the way in and out.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any

from sqlalchemy import DateTime, Dialect, MetaData, Numeric, String
from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy.types import TypeDecorator

__all__ = ["MONEY", "UTC_DATETIME", "Base", "Money", "MoneyDecimal", "ShortStr", "UtcDateTime"]

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

_PRECISION = 38
_SCALE = 18


class MoneyDecimal(TypeDecorator[Decimal]):
    """Exact decimal storage on every supported backend."""

    impl = Numeric(_PRECISION, _SCALE)
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "sqlite":
            return dialect.type_descriptor(String(80))
        return dialect.type_descriptor(Numeric(_PRECISION, _SCALE))

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        if dialect.name == "sqlite":
            return format(decimal_value, "f")
        return decimal_value

    def process_result_value(self, value: Any, dialect: Dialect) -> Decimal | None:
        if value is None:
            return None
        return value if isinstance(value, Decimal) else Decimal(str(value))


class UtcDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC timestamps, including on SQLite."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):  # pragma: no cover - defensive
            raise TypeError(f"expected datetime, got {type(value)!r}")
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


MONEY = MoneyDecimal()
UTC_DATETIME = UtcDateTime()
SHORT_STR = String(64)

Money = Annotated[Decimal, mapped_column(MONEY)]
ShortStr = Annotated[str, mapped_column(SHORT_STR)]


class Base(DeclarativeBase):
    """Common declarative base for every table."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
