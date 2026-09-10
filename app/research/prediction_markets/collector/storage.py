"""Research-friendly persistence: JSONL + SQLite (deterministic, queryable).

Writes to:
  data/research/prediction_markets/observations/YYYY-MM-DD.jsonl  (raw, append, one JSON per line)
  SQLite table `observations` (same rows, indexed for backtesting)

For tests, pass :memory: or a temp path / in-memory list via `memory_only=True`.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
from datetime import UTC, datetime
from typing import Any

from app.research.prediction_markets.collector.models import RawObservation

__all__ = ["CollectorStore", "OBSERVATIONS_DDL"]

OBSERVATIONS_DDL = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at_ms INTEGER NOT NULL,
    source TEXT NOT NULL,
    symbol TEXT NOT NULL,
    market_id INTEGER,
    token_id TEXT,
    market_topic_id INTEGER,
    duration TEXT,
    exchange_ts_ms INTEGER,
    update_ts_ms INTEGER,
    sequence INTEGER,
    resolution_ms INTEGER,
    time_to_resolution_ms INTEGER,
    bid TEXT,
    ask TEXT,
    spread TEXT,
    spread_bps TEXT,
    depth INTEGER,
    price TEXT,
    outcome TEXT,
    raw_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_symbol_ts ON observations(symbol, exchange_ts_ms);
CREATE INDEX IF NOT EXISTS idx_obs_market_token ON observations(market_id, token_id);
CREATE INDEX IF NOT EXISTS idx_obs_source ON observations(source);
CREATE INDEX IF NOT EXISTS idx_obs_captured ON observations(captured_at_ms);
"""


def _obs_to_row(obs: RawObservation) -> dict[str, Any]:
    # extract common fields safely
    bid = getattr(obs, "bid", None) or getattr(obs, "best_bid", None)
    ask = getattr(obs, "ask", None) or getattr(obs, "best_ask", None)
    spread = getattr(obs, "spread", None)
    spread_bps = getattr(obs, "spread_bps", None)
    depth = getattr(obs, "depth", None)
    if callable(depth):
        depth = None
    price = getattr(obs, "price", None) or getattr(obs, "last_price", None) or getattr(obs, "last_trade_price", None)
    outcome = getattr(obs, "outcome", None)
    return {
        "captured_at_ms": obs.captured_at_ms,
        "source": obs.source.value if hasattr(obs.source, "value") else str(obs.source),
        "symbol": obs.symbol,
        "market_id": obs.market_id,
        "token_id": obs.token_id,
        "market_topic_id": obs.market_topic_id,
        "duration": obs.duration,
        "exchange_ts_ms": obs.exchange_ts_ms,
        "update_ts_ms": obs.update_ts_ms,
        "sequence": obs.sequence,
        "resolution_ms": obs.resolution_ms,
        "time_to_resolution_ms": obs.time_to_resolution_ms,
        "bid": str(bid) if bid is not None else None,
        "ask": str(ask) if ask is not None else None,
        "spread": str(spread) if spread is not None else None,
        "spread_bps": str(spread_bps) if spread_bps is not None else None,
        "depth": int(depth) if isinstance(depth, int) else None,
        "price": str(price) if price is not None else None,
        "outcome": outcome,
        "raw_json": json.dumps(obs.raw, default=str) if obs.raw is not None else None,
    }


class CollectorStore:
    """Dual JSONL + SQLite store. Deterministic ordering by captured_at_ms, then exchange_ts_ms."""

    def __init__(
        self,
        base_dir: str | pathlib.Path = "data/research/prediction_markets",
        sqlite_path: str | pathlib.Path | None = None,
        memory_only: bool = False,
    ) -> None:
        self.memory_only = memory_only
        self.base_dir = pathlib.Path(base_dir)
        self.sqlite_path = pathlib.Path(sqlite_path) if sqlite_path is not None else self.base_dir / "collector.db"
        self._memory: list[RawObservation] = []
        self._conn: sqlite3.Connection | None = None
        if not memory_only:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            (self.base_dir / "observations").mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
            self._conn.executescript(OBSERVATIONS_DDL)
            self._conn.commit()
        else:
            # still allow sqlite :memory: if sqlite_path == :memory:
            if sqlite_path == ":memory:":
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._conn.executescript(OBSERVATIONS_DDL)
                self._conn.commit()
            else:
                self._conn = None

    def write(self, obs: RawObservation) -> None:
        self._memory.append(obs)
        row = _obs_to_row(obs)
        if self._conn is not None:
            self._conn.execute(
                """
                INSERT INTO observations
                (captured_at_ms, source, symbol, market_id, token_id, market_topic_id, duration,
                 exchange_ts_ms, update_ts_ms, sequence, resolution_ms, time_to_resolution_ms,
                 bid, ask, spread, spread_bps, depth, price, outcome, raw_json)
                VALUES (:captured_at_ms, :source, :symbol, :market_id, :token_id, :market_topic_id, :duration,
                        :exchange_ts_ms, :update_ts_ms, :sequence, :resolution_ms, :time_to_resolution_ms,
                        :bid, :ask, :spread, :spread_bps, :depth, :price, :outcome, :raw_json)
                """,
                row,
            )
            self._conn.commit()
        if not self.memory_only:
            day = datetime.fromtimestamp(obs.captured_at_ms / 1000, tz=UTC).strftime("%Y-%m-%d")
            path = self.base_dir / "observations" / f"{day}.jsonl"
            with open(path, "a", encoding="utf-8") as fh:
                # full observation as JSON (including computed fields via model_dump)
                fh.write(json.dumps(obs.model_dump(mode="json"), default=str) + "\n")

    def count(self) -> int:
        if self._conn is not None:
            cur = self._conn.execute("SELECT COUNT(*) FROM observations")
            return int(cur.fetchone()[0])
        return len(self._memory)

    def query(
        self,
        symbol: str | None = None,
        market_id: int | None = None,
        source: str | None = None,
        limit: int = 100,
        order: str = "exchange_ts_ms",
    ) -> list[dict[str, Any]]:
        allowed_order = {"captured_at_ms", "exchange_ts_ms", "update_ts_ms", "id"}
        if order not in allowed_order:
            order = "exchange_ts_ms"
        if self._conn is None:
            # in-memory filter (deterministic)
            items = self._memory
            if symbol is not None:
                items = [o for o in items if o.symbol == symbol]
            if market_id is not None:
                items = [o for o in items if o.market_id == market_id]
            if source is not None:
                items = [o for o in items if str(o.source.value) == source or str(o.source) == source]
            # sort deterministically
            items = sorted(items, key=lambda o: (getattr(o, order) or 0, o.captured_at_ms))
            return [o.model_dump(mode="json") for o in items[:limit]]
        where = []
        params: dict[str, Any] = {}
        if symbol is not None:
            where.append("symbol = :symbol")
            params["symbol"] = symbol
        if market_id is not None:
            where.append("market_id = :market_id")
            params["market_id"] = market_id
        if source is not None:
            where.append("source = :source")
            params["source"] = source
        sql = "SELECT * FROM observations"
        if where:
            sql += " WHERE " + " AND ".join(where)
        # deterministic: secondary sort by captured_at_ms
        sql += f" ORDER BY {order} ASC, captured_at_ms ASC LIMIT :limit"
        params["limit"] = limit
        cur = self._conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def all_observations(self) -> list[RawObservation]:
        return list(self._memory)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
