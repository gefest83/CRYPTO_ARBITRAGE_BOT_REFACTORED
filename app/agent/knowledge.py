"""Knowledge Base: ingestion and retrieval of project documentation.

The initial Knowledge Base is populated from *existing repository material*:

* README.md
* docs/ARCHITECTURE.md, docs/OPERATIONS.md
* strategy documentation (triangular scanner, transfer planner/networks/orchestrator)
* configuration descriptions (.env.example, app/config/settings.py)
* risk rules (app/risk/rules.py), execution rules (app/execution/*),
  recovery (app/recovery/*), exchange integration (app/exchanges/*),
  market-data (app/market_data/*)

No random web scraping is performed and no user-provided documents are
required. Every ingested document preserves ``source_type`` / ``source_id``
/ ``version`` / ``status`` so provenance is never lost.
"""

from __future__ import annotations

import pathlib
from typing import Any

from sqlalchemy import select

from app.agent.models import KnowledgeCategory, KnowledgeDocument, SourceType
from app.agent.tables import AgentKnowledgeRow
from app.config.logging_config import get_logger
from app.storage.engine import Database

__all__ = ["KnowledgeRepository", "KnowledgeService"]

logger = get_logger("agent.knowledge")


# ------------------------------------------------------------------ mapping


def _doc_to_row(doc: KnowledgeDocument) -> AgentKnowledgeRow:
    return AgentKnowledgeRow(
        id=doc.id,
        title=doc.title,
        category=doc.category.value if isinstance(doc.category, KnowledgeCategory) else str(doc.category),
        content=doc.content,
        summary=doc.summary,
        tags=list(doc.tags),
        extra_metadata=None,
        source_type=doc.source_type,
        source_id=doc.source_id,
        version=doc.version,
        status=doc.status,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


def _row_to_doc(row: AgentKnowledgeRow) -> KnowledgeDocument:
    return KnowledgeDocument(
        id=row.id,
        title=row.title,
        category=row.category,  # validated by pydantic coercion
        content=row.content,
        summary=row.summary,
        tags=tuple(row.tags or ()),
        source_type=row.source_type,
        source_id=row.source_id,
        version=row.version,
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


# ------------------------------------------------------------------ repository


class KnowledgeRepository:
    """Persistence for :class:`KnowledgeDocument`."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def save(self, doc: KnowledgeDocument) -> KnowledgeDocument:
        row = _doc_to_row(doc)
        async with self._db.session() as session:
            await session.merge(row)
        return doc

    async def get(self, doc_id: str) -> KnowledgeDocument | None:
        async with self._db.session() as session:
            row = await session.get(AgentKnowledgeRow, doc_id)
            return _row_to_doc(row) if row else None

    async def get_by_source(self, source_id: str) -> KnowledgeDocument | None:
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentKnowledgeRow).where(AgentKnowledgeRow.source_id == source_id)
            )
            row = result.scalars().first()
            return _row_to_doc(row) if row else None

    async def list_all(self) -> list[KnowledgeDocument]:
        async with self._db.session() as session:
            result = await session.execute(select(AgentKnowledgeRow).order_by(AgentKnowledgeRow.created_at))
            return [_row_to_doc(r) for r in result.scalars()]

    async def list_by_category(self, category: str | KnowledgeCategory) -> list[KnowledgeDocument]:
        cat = category.value if isinstance(category, KnowledgeCategory) else str(category).lower()
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentKnowledgeRow).where(AgentKnowledgeRow.category == cat).order_by(AgentKnowledgeRow.created_at)
            )
            return [_row_to_doc(r) for r in result.scalars()]

    async def search(self, query: str, limit: int = 20) -> list[KnowledgeDocument]:
        """Substring search over title/content/tags.

        Phase 1 uses LIKE; embeddings can be added later without changing the
        interface.
        """
        pattern = f"%{query}%"
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentKnowledgeRow)
                .where(
                    (AgentKnowledgeRow.title.like(pattern))
                    | (AgentKnowledgeRow.content.like(pattern))
                    | (AgentKnowledgeRow.summary.like(pattern))  # type: ignore[arg-type]
                )
                .order_by(AgentKnowledgeRow.updated_at.desc())
                .limit(limit)
            )
            return [_row_to_doc(r) for r in result.scalars()]

    async def count(self) -> int:
        from sqlalchemy import func

        async with self._db.session() as session:
            result = await session.execute(select(func.count()).select_from(AgentKnowledgeRow))
            return int(result.scalar_one())


# ------------------------------------------------------------------ service


# Mapping of known repo files to knowledge categories.
# This is the allow-list for Phase 1 ingestion — no web content.
_KNOWN_SOURCES: tuple[tuple[str, KnowledgeCategory, str], ...] = (
    ("README.md", KnowledgeCategory.BOT, "Project overview, quick start, modes, safety"),
    ("docs/ARCHITECTURE.md", KnowledgeCategory.BOT, "Module layout and design principles"),
    ("docs/OPERATIONS.md", KnowledgeCategory.BOT, "Runbook: lifecycle, kill switch, recovery"),
    (".env.example", KnowledgeCategory.BOT, "Configuration reference (annotated)"),
    ("app/config/settings.py", KnowledgeCategory.BOT, "Settings schema and validation"),
    ("app/config/modes.py", KnowledgeCategory.BOT, "Trading mode policy (PAPER/DEMO/LIVE)"),
    ("app/risk/rules.py", KnowledgeCategory.TRADING, "Risk rules (fail-closed)"),
    ("app/risk/engine.py", KnowledgeCategory.TRADING, "Risk engine evaluation"),
    ("app/execution/guard.py", KnowledgeCategory.TRADING, "Kill switch / execution guard"),
    ("app/execution/order_gate.py", KnowledgeCategory.TRADING, "Order gate per mode"),
    ("app/execution/paper_wallet.py", KnowledgeCategory.TRADING, "Paper wallet (PAPER simulation)"),
    ("app/execution/fill_simulator.py", KnowledgeCategory.TRADING, "Fill simulation"),
    ("app/recovery/recovery.py", KnowledgeCategory.TRADING, "Execution recovery logic"),
    ("app/strategies/triangular/scanner.py", KnowledgeCategory.TRADING, "Triangular scanner (depth-aware VWAP)"),
    ("app/strategies/triangular/executor.py", KnowledgeCategory.TRADING, "Triangle executor (3-leg sequential)"),
    ("app/strategies/transfer/planner.py", KnowledgeCategory.TRADING, "Transfer planner (economics)"),
    ("app/strategies/transfer/networks.py", KnowledgeCategory.TRADING, "Withdrawal network validation"),
    ("app/strategies/transfer/orchestrator.py", KnowledgeCategory.TRADING, "Transfer orchestrator lifecycle"),
    ("app/exchanges/profiles.py", KnowledgeCategory.EXCHANGE, "Exchange profiles (Binance/OKX/Bybit)"),
    ("app/exchanges/manager.py", KnowledgeCategory.EXCHANGE, "Exchange manager (breaker + auth gate)"),
    ("app/exchanges/ccxt_adapter.py", KnowledgeCategory.EXCHANGE, "CCXT adapter (live venues)"),
    ("app/exchanges/simulated.py", KnowledgeCategory.EXCHANGE, "Simulated exchange (PAPER)"),
    ("app/market_data/store.py", KnowledgeCategory.RESEARCH, "Market-data store (order books, freshness)"),
    ("app/market_data/service.py", KnowledgeCategory.RESEARCH, "Market-data service (REST + WS)"),
    ("app/market_data/streams.py", KnowledgeCategory.RESEARCH, "WebSocket stream supervision"),
)


class KnowledgeService:
    """High-level knowledge operations: ingest, retrieve, search.

    All ingestion preserves ``source_type`` / ``source_id`` / ``version`` so
    that the advisor can cite where a piece of knowledge came from.
    """

    def __init__(self, repository: KnowledgeRepository) -> None:
        self._repo = repository

    @property
    def repository(self) -> KnowledgeRepository:
        return self._repo

    # ---------------------------------------------------------------- ingestion

    async def ingest_file(
        self,
        path: pathlib.Path | str,
        category: KnowledgeCategory | str,
        *,
        title: str | None = None,
        source_type: str = SourceType.DOCUMENT.value,
    ) -> KnowledgeDocument | None:
        """Read ``path`` from disk and persist it as a knowledge document.

        Returns ``None`` when the file does not exist (ingestion is best-effort).
        The document content is capped at 20_000 characters to avoid bloating
        the SQLite file while still retaining the load-bearing details for
        analysis.
        """
        p = pathlib.Path(path)
        if not p.exists() or not p.is_file():
            logger.debug("knowledge_ingest_skip_missing", extra={"path": str(p)})
            return None
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - file read best-effort
            logger.warning("knowledge_ingest_read_failed", extra={"path": str(p), "error": str(exc)[:200]})
            return None
        # Truncate to keep the DB bounded but preserve the head (titles + intro matter most)
        if len(text) > 20_000:
            text = text[:20_000] + "\n\n... (truncated, source_id holds full path)"
        cat = category if isinstance(category, KnowledgeCategory) else KnowledgeCategory(str(category).lower())
        doc = KnowledgeDocument(
            title=title or p.name,
            category=cat,
            content=text,
            summary=text[:500].replace("\n", " ").strip() if text else None,
            source_type=source_type,
            source_id=str(p.as_posix()),
            version=1,
            status="active",
        )
        # Upsert by source_id: if an older version exists, bump version.
        existing = await self._repo.get_by_source(doc.source_id)
        if existing is not None:
            doc = doc.model_copy(update={"id": existing.id, "version": existing.version + 1})
        await self._repo.save(doc)
        logger.info("knowledge_ingested", extra={"source_id": doc.source_id, "category": cat.value, "chars": len(text)})
        return doc

    async def ingest_text(
        self,
        title: str,
        content: str,
        category: KnowledgeCategory | str,
        *,
        source_id: str,
        source_type: str = SourceType.DOCUMENT.value,
    ) -> KnowledgeDocument:
        """Ingest an in-memory text blob (e.g. a strategy summary)."""
        cat = category if isinstance(category, KnowledgeCategory) else KnowledgeCategory(str(category).lower())
        doc = KnowledgeDocument(
            title=title,
            category=cat,
            content=content[:20_000],
            summary=content[:500].replace("\n", " ").strip(),
            source_type=source_type,
            source_id=source_id,
            version=1,
            status="active",
        )
        existing = await self._repo.get_by_source(source_id)
        if existing is not None:
            doc = doc.model_copy(update={"id": existing.id, "version": existing.version + 1})
        await self._repo.save(doc)
        return doc

    async def ingest_repository(
        self,
        root: pathlib.Path | str = ".",
        *,
        max_files: int | None = None,
    ) -> list[KnowledgeDocument]:
        """Populate the knowledge base from the existing repository documentation.

        Iterates :data:`_KNOWN_SOURCES` under ``root`` and ingests each file
        that exists. No web scraping is performed; only files that are already
        part of the checkout are considered.
        """
        base = pathlib.Path(root)
        ingested: list[KnowledgeDocument] = []
        sources = _KNOWN_SOURCES[:max_files] if max_files else _KNOWN_SOURCES
        for rel, category, title in sources:
            p = base / rel
            doc = await self.ingest_file(p, category, title=title, source_type=SourceType.REPOSITORY.value)
            if doc is not None:
                ingested.append(doc)
        logger.info("knowledge_repository_ingested", extra={"count": len(ingested), "root": str(base)})
        return ingested

    # ---------------------------------------------------------------- retrieval

    async def get_all(self) -> list[KnowledgeDocument]:
        return await self._repo.list_all()

    async def get_by_category(self, category: str | KnowledgeCategory) -> list[KnowledgeDocument]:
        return await self._repo.list_by_category(category)

    async def search(self, query: str, limit: int = 20) -> list[KnowledgeDocument]:
        return await self._repo.search(query, limit=limit)

    async def get_recent(self, limit: int = 20) -> list[KnowledgeDocument]:
        all_docs = await self._repo.list_all()
        # list_all is ordered by created_at asc; reverse for most recent
        return list(reversed(all_docs))[:limit]
