"""Base primitives for domain models.

All domain models are immutable (``frozen=True``) pydantic v2 models with
``extra="forbid"``.  Immutability keeps the arbitrage/risk pipeline predictable:
stages transform snapshots instead of mutating shared state.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.errors import ValidationError

__all__ = ["DEC0", "DEC1", "DomainModel", "as_decimal", "utc_now"]

DEC0 = Decimal("0")
DEC1 = Decimal("1")


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp (never use naive datetimes in the domain)."""
    return datetime.now(UTC)


def as_decimal(value: Any, *, field: str = "value") -> Decimal:
    """Convert ``value`` to :class:`~decimal.Decimal` without float artefacts."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValidationError(f"{field} is not a valid decimal: {value!r}") from exc


class DomainModel(BaseModel):
    """Immutable, strictly validated domain value object."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_default=True,
    )
