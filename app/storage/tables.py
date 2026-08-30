"""SQLAlchemy table definitions.

Five tables cover the whole persistence surface of the bot:

* ``trades``       — executed (or attempted) arbitrage trades
* ``transfers``    — transfer-arbitrage lifecycle records (restart-safe)
* ``balances``     — last known balance snapshot per exchange/asset
* ``audit_log``    — append-only event log (errors, execution, recovery)
* ``bot_state``    — small key/value store (kill switch, auto-trading flag)
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, Boolean, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import utc_now
from app.storage.base import MONEY, SHORT_STR, UTC_DATETIME, Base

__all__ = [
    "AuditLogRow",
    "BalanceRow",
    "BotStateRow",
    "TradeRow",
    "TransferRow",
    "purge_old_rows",
]


class TradeRow(Base):
    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    strategy: Mapped[str] = mapped_column(String(16))
    mode: Mapped[str] = mapped_column(String(8))
    exchange_id: Mapped[str] = mapped_column(String(16))
    route: Mapped[str] = mapped_column(String(128), default="")
    symbols: Mapped[list[str]] = mapped_column(JSON, default=list)
    input_amount: Mapped[Decimal] = mapped_column(MONEY)
    output_amount: Mapped[Decimal] = mapped_column(MONEY)
    fees_quote: Mapped[Decimal] = mapped_column(MONEY)
    slippage_bps: Mapped[Decimal] = mapped_column(MONEY)
    net_profit: Mapped[Decimal] = mapped_column(MONEY)
    net_profit_bps: Mapped[Decimal] = mapped_column(MONEY)
    status: Mapped[str] = mapped_column(String(16))
    orders: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    transfer_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)

    __table_args__ = (
        Index("ix_trades_created_at", "created_at"),
        Index("ix_trades_status", "status"),
    )


class TransferRow(Base):
    __tablename__ = "transfers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    source_exchange: Mapped[str] = mapped_column(String(16))
    dest_exchange: Mapped[str] = mapped_column(String(16))
    asset: Mapped[str] = mapped_column(SHORT_STR)
    network: Mapped[str] = mapped_column(SHORT_STR)
    amount: Mapped[Decimal] = mapped_column(MONEY)
    state: Mapped[str] = mapped_column(String(24))
    plan: Mapped[dict[str, Any]] = mapped_column(JSON)
    buy_order: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    buy_filled_amount: Mapped[Decimal] = mapped_column(MONEY, default=Decimal(0))
    withdrawal_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    withdrawal_txid: Mapped[str | None] = mapped_column(String(128), nullable=True)
    withdrawal_amount: Mapped[Decimal] = mapped_column(MONEY, default=Decimal(0))
    deposit_address: Mapped[str | None] = mapped_column(String(128), nullable=True)
    deposit_txid: Mapped[str | None] = mapped_column(String(128), nullable=True)
    deposit_amount: Mapped[Decimal] = mapped_column(MONEY, default=Decimal(0))
    sell_order: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    sell_filled_amount: Mapped[Decimal] = mapped_column(MONEY, default=Decimal(0))
    sell_proceeds_quote: Mapped[Decimal] = mapped_column(MONEY, default=Decimal(0))
    fees_quote: Mapped[Decimal] = mapped_column(MONEY, default=Decimal(0))
    realized_profit_quote: Mapped[Decimal] = mapped_column(MONEY, default=Decimal(0))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    mode: Mapped[str] = mapped_column(String(8), default="PAPER")
    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)

    __table_args__ = (
        Index("ix_transfers_state", "state"),
        Index("ix_transfers_created_at", "created_at"),
    )


class BalanceRow(Base):
    __tablename__ = "balances"

    exchange_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    asset: Mapped[str] = mapped_column(SHORT_STR, primary_key=True)
    free: Mapped[Decimal] = mapped_column(MONEY)
    used: Mapped[Decimal] = mapped_column(MONEY)
    updated_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)


class AuditLogRow(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now, index=True)
    action: Mapped[str] = mapped_column(String(64))
    message: Mapped[str] = mapped_column(Text, default="")
    context: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class BotStateRow(Base):
    __tablename__ = "bot_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(UTC_DATETIME, default=utc_now)


async def purge_old_rows(session: Any, *, trade_days: int, audit_days: int) -> tuple[int, int]:
    """Delete rows older than the retention windows; returns (trades, audits)."""
    from datetime import timedelta

    now = utc_now()
    trades_deleted = 0
    audits_deleted = 0
    if trade_days > 0:
        cutoff = now - timedelta(days=trade_days)
        result = await session.execute(
            TradeRow.__table__.delete().where(TradeRow.created_at < cutoff)  # type: ignore[arg-type]
        )
        trades_deleted = int(result.rowcount or 0)
    if audit_days > 0:
        cutoff = now - timedelta(days=audit_days)
        result = await session.execute(
            AuditLogRow.__table__.delete().where(AuditLogRow.ts < cutoff)  # type: ignore[arg-type]
        )
        audits_deleted = int(result.rowcount or 0)
    return trades_deleted, audits_deleted
