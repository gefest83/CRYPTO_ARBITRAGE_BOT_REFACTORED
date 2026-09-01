"""Async engine / session management.

Local development runs on SQLite (``sqlite+aiosqlite``), production on
PostgreSQL (``postgresql+asyncpg``).  The only difference is the URL: pool
options that SQLite rejects are filtered out here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import func, inspect, select
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config.logging_config import get_logger
from app.config.settings import DatabaseSettings
from app.errors import ConfigurationError

__all__ = ["Database"]

logger = get_logger("database")


class Database:
    """Owns the engine and session factory for the process."""

    def __init__(self, config: DatabaseSettings) -> None:
        self._config = config
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None

    # ---------------------------------------------------------------- lifecycle
    @property
    def config(self) -> DatabaseSettings:
        return self._config

    @property
    def is_started(self) -> bool:
        return self._engine is not None

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise ConfigurationError("database is not started")
        return self._engine

    def start(self) -> AsyncEngine:
        if self._engine is not None:
            return self._engine
        self._prepare_sqlite_directory()
        self._engine = create_async_engine(self._config.url, **self._engine_kwargs())
        self._sessionmaker = async_sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )
        logger.info("database_started", extra={"dialect": self._dialect()})
        return self._engine

    async def dispose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            logger.info("database_disposed", extra={"dialect": self._dialect()})
        self._engine = None
        self._sessionmaker = None

    async def create_schema(self) -> None:
        """Create tables when missing; rename a drifted SQLite file to ``*.bak``.

        If the physical file drifted from the models (an existing table is
        missing columns the metadata requires — typically a database from an
        older checkout), it is renamed to ``*.bak`` and a fresh schema is
        created instead of failing mid-query at runtime.

        H-14 fail-closed: the rename never happens while the file still
        carries open lifecycle state (open transfers or trades interrupted
        mid-execution) — destroying their recovery history would silently
        abandon in-flight funds.  Startup refuses instead.
        """
        from app.storage import tables  # noqa: F401  (import registers the mappers)
        from app.storage.base import Base

        engine = self.start()
        if self._config.is_sqlite and self._sqlite_path() is not None:
            drift = await self._detect_sqlite_drift(engine)
            if drift:
                open_rows = await self._count_open_lifecycle_rows(engine)
                if open_rows > 0:
                    raise ConfigurationError(
                        f"refusing to rebuild drifted SQLite database "
                        f"'{self._sqlite_path()}': {open_rows} open transfer/trade "
                        "record(s) would lose their recovery history — close or "
                        "resolve them (or migrate the file manually) before "
                        "restarting"
                    )
                await self._rebuild_drifted_file(engine, drift)
                engine = self.start()
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        logger.info("database_schema_ready", extra={"tables": len(Base.metadata.tables)})

    async def _count_open_lifecycle_rows(self, engine: AsyncEngine) -> int:
        """Open transfer workflows + trades interrupted mid-execution.

        These rows are the only record of in-flight funds; a drifted file
        containing any of them must not be rotated away (H-14).
        """
        from app.models.enums import TradeStatus, TransferState
        from app.storage.tables import TradeRow, TransferRow

        open_states = tuple(state.value for state in TransferState if state.is_open)

        def _count(sync_connection: Connection) -> int:
            inspector = inspect(sync_connection)
            total = 0
            if inspector.has_table(TransferRow.__tablename__):
                columns = {
                    column["name"] for column in inspector.get_columns(TransferRow.__tablename__)
                }
                if "state" in columns:
                    total += sync_connection.execute(
                        select(func.count())
                        .select_from(TransferRow)
                        .where(TransferRow.state.in_(open_states))
                    ).scalar_one()
                else:
                    # Cannot read the lifecycle column: if the table holds any
                    # rows we cannot establish that none are open — fail closed.
                    total += sync_connection.execute(
                        select(func.count()).select_from(TransferRow)
                    ).scalar_one()
            if inspector.has_table(TradeRow.__tablename__):
                columns = {
                    column["name"] for column in inspector.get_columns(TradeRow.__tablename__)
                }
                if "status" in columns:
                    total += sync_connection.execute(
                        select(func.count())
                        .select_from(TradeRow)
                        .where(TradeRow.status == TradeStatus.EXECUTING.value)
                    ).scalar_one()
                elif sync_connection.execute(
                    select(func.count()).select_from(TradeRow)
                ).scalar_one():
                    total += 1
            return int(total)

        async with engine.connect() as connection:
            return await connection.run_sync(_count)

    async def _detect_sqlite_drift(self, engine: AsyncEngine) -> dict[str, set[str]]:
        """Existing tables whose columns do not satisfy the current metadata."""
        from app.storage.base import Base

        def _compare(sync_connection: Connection) -> dict[str, set[str]]:
            inspector = inspect(sync_connection)
            drift: dict[str, set[str]] = {}
            for table in Base.metadata.sorted_tables:
                if not inspector.has_table(table.name):
                    continue  # absent tables are created fresh by create_all
                actual = {column["name"] for column in inspector.get_columns(table.name)}
                missing = {column.name for column in table.columns} - actual
                if missing:
                    drift[table.name] = missing
            return drift

        async with engine.connect() as connection:
            return await connection.run_sync(_compare)

    async def _rebuild_drifted_file(self, engine: AsyncEngine, drift: dict[str, set[str]]) -> None:
        """Rename the drifted database to ``*.bak``; the caller re-creates it."""
        path = self._sqlite_path()
        assert path is not None
        # The pool keeps open handles to the file; release them first — renaming
        # an open file fails on Windows.
        await engine.dispose()
        self._engine = None
        self._sessionmaker = None

        backup = Path(f"{path}.bak")
        if backup.exists():
            backup.unlink()
        path.rename(backup)
        # Stale WAL/SHM sidecars belong to the old file and must not be
        # replayed into the fresh database.
        for sidecar in (Path(f"{path}-wal"), Path(f"{path}-shm")):
            sidecar.unlink(missing_ok=True)

        details = "; ".join(
            f"{table}: missing {', '.join(sorted(columns))}"
            for table, columns in sorted(drift.items())
        )
        logger.warning(
            "schema_drift_detected_recreated",
            extra={"backup": str(backup), "detail": details},
        )

    # ---------------------------------------------------------------- sessions
    @property
    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        if self._sessionmaker is None:
            self.start()
        assert self._sessionmaker is not None
        return self._sessionmaker

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Transactional scope: commit on success, rollback on error."""
        async with self.sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def ping(self) -> bool:
        from sqlalchemy import text

        async with self.sessionmaker() as session:
            await session.execute(text("SELECT 1"))
        return True

    # ---------------------------------------------------------------- private
    def _dialect(self) -> str:
        return self._config.url.split(":", 1)[0]

    def _sqlite_path(self) -> Path | None:
        """Filesystem path of a SQLite URL, or ``None`` for other dialects."""
        if not self._config.is_sqlite:
            return None
        _, _, tail = self._config.url.partition(":///")
        return Path(tail) if tail and tail != ":memory:" else None

    def _engine_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"echo": self._config.echo, "future": True}
        if self._config.is_sqlite:
            # SQLite (aiosqlite) does not accept server-side pool sizing.
            return kwargs
        kwargs.update(
            pool_size=self._config.pool_size,
            max_overflow=self._config.max_overflow,
            pool_pre_ping=True,
        )
        return kwargs

    def _prepare_sqlite_directory(self) -> None:
        path = self._sqlite_path()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
