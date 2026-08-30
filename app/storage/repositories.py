"""Repositories: persistence of trades, transfers, balances, audit, bot state.

Domain models are pydantic; rows are SQLAlchemy.  Conversion happens only
here, so the rest of the application never touches the ORM.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, time
from decimal import Decimal
from typing import Any

from sqlalchemy import delete, func, select

from app.config.logging_config import get_logger
from app.models.balance import Balance, BalanceSnapshot
from app.models.base import DEC0, utc_now
from app.models.enums import TradeStatus, TransferState
from app.models.trade import TradeRecord
from app.models.transfer import TransferPlan, TransferRecord
from app.storage.engine import Database
from app.storage.tables import (
    AuditLogRow,
    BalanceRow,
    BotStateRow,
    TradeRow,
    TransferRow,
    purge_old_rows,
)

__all__ = [
    "AuditLogRepository",
    "BalanceRepository",
    "BotStateRepository",
    "TradeRepository",
    "TransferRepository",
]

logger = get_logger("storage")

_OPEN_TRANSFER_STATES = tuple(state.value for state in TransferState if state.is_open)


def _dump_json(model: Any) -> dict[str, Any]:
    """Model -> JSON-safe dict (Decimal -> str, datetimes -> ISO).

    Computed fields are excluded: they are derived values that pydantic refuses
    to re-validate (``extra="forbid"``) and are recomputed on load anyway.
    """
    computed = set(model.model_computed_fields)
    return json.loads(model.model_dump_json(exclude=computed))


class TradeRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, trade: TradeRecord) -> TradeRecord:
        row = TradeRow(
            id=trade.id,
            strategy=trade.strategy.value,
            mode=trade.mode.value,
            exchange_id=trade.exchange_id,
            route=trade.route,
            symbols=list(trade.symbols),
            input_amount=trade.input_amount,
            output_amount=trade.output_amount,
            fees_quote=trade.fees_quote,
            slippage_bps=trade.slippage_bps,
            net_profit=trade.net_profit,
            net_profit_bps=trade.net_profit_bps,
            status=trade.status.value,
            orders=[dict(o) for o in trade.orders],
            error=trade.error,
            transfer_id=trade.transfer_id,
            created_at=trade.created_at,
            updated_at=trade.updated_at,
        )
        async with self._db.session() as session:
            await session.merge(row)
        return trade

    async def get(self, trade_id: str) -> TradeRecord | None:
        async with self._db.session() as session:
            row = await session.get(TradeRow, trade_id)
            return self._to_record(row) if row else None

    async def list_recent(self, limit: int = 20) -> list[TradeRecord]:
        async with self._db.session() as session:
            result = await session.execute(
                select(TradeRow).order_by(TradeRow.created_at.desc()).limit(limit)
            )
            return [self._to_record(row) for row in result.scalars()]

    async def realized_pnl_since(self, since: datetime) -> Decimal:
        """Net profit of completed trades since ``since`` (UTC)."""
        async with self._db.session() as session:
            result = await session.execute(
                select(func.coalesce(func.sum(TradeRow.net_profit), Decimal(0))).where(
                    TradeRow.status == TradeStatus.COMPLETED.value,
                    TradeRow.created_at >= since,
                )
            )
            value = result.scalar_one()
            return value if isinstance(value, Decimal) else Decimal(str(value or 0))

    async def realized_pnl_today(self) -> Decimal:
        start = datetime.combine(datetime.now(UTC).date(), time.min, tzinfo=UTC)
        return await self.realized_pnl_since(start)

    @staticmethod
    def _to_record(row: TradeRow) -> TradeRecord:
        return TradeRecord(
            id=row.id,
            strategy=row.strategy,  # type: ignore[arg-type]
            mode=row.mode,  # type: ignore[arg-type]
            exchange_id=row.exchange_id,
            route=row.route,
            symbols=tuple(row.symbols),
            input_amount=row.input_amount,
            output_amount=row.output_amount,
            fees_quote=row.fees_quote,
            slippage_bps=row.slippage_bps,
            net_profit=row.net_profit,
            net_profit_bps=row.net_profit_bps,
            status=row.status,  # type: ignore[arg-type]
            orders=tuple(row.orders or ()),
            error=row.error,
            transfer_id=row.transfer_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class TransferRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, record: TransferRecord) -> TransferRecord:
        row = TransferRow(
            id=record.id,
            source_exchange=record.source_exchange,
            dest_exchange=record.dest_exchange,
            asset=record.asset,
            network=record.network,
            amount=record.amount,
            state=record.state.value,
            plan=_dump_json(record.plan),
            buy_order=record.buy_order,
            buy_filled_amount=record.buy_filled_amount,
            withdrawal_id=record.withdrawal_id,
            withdrawal_txid=record.withdrawal_txid,
            withdrawal_amount=record.withdrawal_amount,
            deposit_address=record.deposit_address,
            deposit_txid=record.deposit_txid,
            deposit_amount=record.deposit_amount,
            sell_order=record.sell_order,
            sell_filled_amount=record.sell_filled_amount,
            sell_proceeds_quote=record.sell_proceeds_quote,
            fees_quote=record.fees_quote,
            realized_profit_quote=record.realized_profit_quote,
            error=record.error,
            mode=record.mode,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
        async with self._db.session() as session:
            await session.merge(row)
        return record

    async def get(self, transfer_id: str) -> TransferRecord | None:
        async with self._db.session() as session:
            row = await session.get(TransferRow, transfer_id)
            return self._to_record(row) if row else None

    async def list_open(self) -> list[TransferRecord]:
        """All transfers in a non-terminal state (resumed on startup)."""
        async with self._db.session() as session:
            result = await session.execute(
                select(TransferRow)
                .where(TransferRow.state.in_(_OPEN_TRANSFER_STATES))
                .order_by(TransferRow.created_at)
            )
            return [self._to_record(row) for row in result.scalars()]

    async def count_open(self) -> int:
        async with self._db.session() as session:
            result = await session.execute(
                select(func.count())
                .select_from(TransferRow)
                .where(TransferRow.state.in_(_OPEN_TRANSFER_STATES))
            )
            return int(result.scalar_one())

    async def list_recent(self, limit: int = 20) -> list[TransferRecord]:
        async with self._db.session() as session:
            result = await session.execute(
                select(TransferRow).order_by(TransferRow.created_at.desc()).limit(limit)
            )
            return [self._to_record(row) for row in result.scalars()]

    @staticmethod
    def _to_record(row: TransferRow) -> TransferRecord:
        return TransferRecord(
            id=row.id,
            source_exchange=row.source_exchange,
            dest_exchange=row.dest_exchange,
            asset=row.asset,
            network=row.network,
            amount=row.amount,
            state=row.state,  # type: ignore[arg-type]
            plan=TransferPlan.model_validate(row.plan),
            buy_order=row.buy_order,
            buy_filled_amount=row.buy_filled_amount,
            withdrawal_id=row.withdrawal_id,
            withdrawal_txid=row.withdrawal_txid,
            withdrawal_amount=row.withdrawal_amount,
            deposit_address=row.deposit_address,
            deposit_txid=row.deposit_txid,
            deposit_amount=row.deposit_amount,
            sell_order=row.sell_order,
            sell_filled_amount=row.sell_filled_amount,
            sell_proceeds_quote=row.sell_proceeds_quote,
            fees_quote=row.fees_quote,
            realized_profit_quote=row.realized_profit_quote,
            error=row.error,
            mode=row.mode,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class BalanceRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def save_snapshot(self, snapshot: BalanceSnapshot) -> None:
        async with self._db.session() as session:
            await session.execute(
                delete(BalanceRow).where(BalanceRow.exchange_id == snapshot.exchange_id)
            )
            for balance in snapshot.balances:
                if balance.is_empty:
                    continue
                session.add(
                    BalanceRow(
                        exchange_id=snapshot.exchange_id,
                        asset=balance.asset,
                        free=balance.free,
                        used=balance.used,
                        updated_at=snapshot.timestamp,
                    )
                )

    async def load_all(self) -> dict[str, BalanceSnapshot]:
        """Last persisted balances keyed by exchange id."""
        collected: dict[str, list[Balance]] = {}
        timestamps: dict[str, Any] = {}
        async with self._db.session() as session:
            result = await session.execute(select(BalanceRow))
            rows = result.scalars().all()
        for row in rows:
            balances = collected.setdefault(row.exchange_id, [])
            balances.append(
                Balance(exchange_id=row.exchange_id, asset=row.asset, free=row.free, used=row.used)
            )
            timestamps[row.exchange_id] = row.updated_at
        return {
            exchange_id: BalanceSnapshot(
                exchange_id=exchange_id,
                balances=tuple(balances),
                timestamp=timestamps[exchange_id],
            )
            for exchange_id, balances in collected.items()
        }

    async def free_of(self, exchange_id: str, asset: str) -> Decimal:
        async with self._db.session() as session:
            result = await session.execute(
                select(BalanceRow.free).where(
                    BalanceRow.exchange_id == exchange_id, BalanceRow.asset == asset
                )
            )
            value = result.scalar_one_or_none()
            return value if value is not None else DEC0


class AuditLogRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def log(
        self, action: str, message: str = "", context: dict[str, Any] | None = None
    ) -> None:
        row = AuditLogRow(ts=utc_now(), action=action, message=message[:2000], context=context)
        try:
            async with self._db.session() as session:
                session.add(row)
        except Exception as exc:  # noqa: BLE001 - auditing must never break trading
            logger.error("audit_log_write_failed", extra={"action": action, "error": str(exc)})

    async def list_recent(self, limit: int = 50) -> list[AuditLogRow]:
        async with self._db.session() as session:
            result = await session.execute(
                select(AuditLogRow)
                .order_by(AuditLogRow.ts.desc(), AuditLogRow.id.desc())
                .limit(limit)
            )
            return list(result.scalars())


class BotStateRepository:
    """Tiny key/value store for runtime flags that survive restarts."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def get(self, key: str) -> Any | None:
        async with self._db.session() as session:
            row = await session.get(BotStateRow, key)
            return row.value if row is not None else None

    async def set(self, key: str, value: Any) -> None:
        row = BotStateRow(key=key, value=value, updated_at=utc_now())
        async with self._db.session() as session:
            await session.merge(row)

    async def delete(self, key: str) -> None:
        async with self._db.session() as session:
            await session.execute(delete(BotStateRow).where(BotStateRow.key == key))


async def purge_history(db: Database, *, trade_days: int, audit_days: int) -> tuple[int, int]:
    """Retention sweep; returns (trades_deleted, audits_deleted)."""
    async with db.session() as session:
        return await purge_old_rows(session, trade_days=trade_days, audit_days=audit_days)
