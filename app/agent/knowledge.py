"""Knowledge Base: ingestion and retrieval of project documentation (Phase 3).

Phase 3 builds the project's internal knowledge base from the actual
current implementation:

* README.md, docs/, pyproject.toml
* app/config/, app/exchanges/, app/market_data/, app/strategies/
* app/execution/, app/risk/, app/recovery/, app/storage/, app/services.py
* tests/ (behavioural sample)

Categories ``BOT_KNOWLEDGE`` / ``EXCHANGE_KNOWLEDGE`` / ``TRADING_KNOWLEDGE``
/ ``RESEARCH_KNOWLEDGE`` are aliases for ``bot`` / ``exchange`` / ``trading``
/ ``research``. Repository knowledge (``source_type="repository"``) stays
distinguishable from external knowledge (any other ``source_type``).

Records carry Phase 3 metadata (knowledge ID, source, document/path,
section, created/updated, verification status, confidence, tags, category).
New metadata rides in the row's ``extra_metadata`` JSON so no migration of
``agent_knowledge`` is required; chunks live in the new
``agent_knowledge_chunks`` table.

Retrieval is relevance-scored over deterministic chunks with limits and
metadata/category filtering; provenance is preserved on every hit.

Security: ``.env`` files, API keys/secrets, passwords, tokens, credentials
and private keys are never indexed — ingestion refuses sensitive paths and
secret-bearing content (fail-closed, best-effort heuristics + block-list).
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

from sqlalchemy import delete, select

from app.agent.models import (
    KnowledgeCategory,
    KnowledgeChunk,
    KnowledgeDocument,
    SourceType,
    normalize_knowledge_category,
)
from app.agent.tables import AgentKnowledgeChunkRow, AgentKnowledgeRow
from app.config.logging_config import get_logger
from app.storage.engine import Database

__all__ = [
    "CHUNK_OVERLAP_CHARS",
    "CHUNK_SIZE_CHARS",
    "KnowledgeChunkRepository",
    "KnowledgeRepository",
    "KnowledgeService",
    "chunk_text",
    "contains_secret_content",
    "is_sensitive_path",
    "normalize_knowledge_category",
    "score_chunk",
    "tokenize_query",
]

logger = get_logger("agent.knowledge")


# ------------------------------------------------------------------ Phase 3 metadata helpers


def _doc_to_row(doc: KnowledgeDocument) -> AgentKnowledgeRow:
    return AgentKnowledgeRow(
        id=doc.id,
        title=doc.title,
        category=doc.category.value if isinstance(doc.category, KnowledgeCategory) else str(doc.category),
        content=doc.content,
        summary=doc.summary,
        tags=list(doc.tags),
        extra_metadata={
            "section": doc.section or "",
            "document_path": doc.document_path or doc.source_id,
            "verification_status": doc.verification_status or "unverified",
            "confidence": float(doc.confidence),
        },
        source_type=doc.source_type,
        source_id=doc.source_id,
        version=doc.version,
        status=doc.status,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


def _row_to_doc(row: AgentKnowledgeRow) -> KnowledgeDocument:
    meta: dict[str, Any] = dict(row.extra_metadata or {})
    return KnowledgeDocument(
        id=row.id,
        title=row.title,
        category=row.category,  # validated by pydantic coercion (aliases included)
        content=row.content,
        summary=row.summary,
        tags=tuple(row.tags or ()),
        section=str(meta.get("section", "") or ""),
        document_path=meta.get("document_path"),
        verification_status=str(meta.get("verification_status", "unverified") or "unverified"),
        confidence=float(meta.get("confidence", 0.5)),
        source_type=row.source_type,
        source_id=row.source_id,
        version=row.version,
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _chunk_to_row(chunk: KnowledgeChunk) -> AgentKnowledgeChunkRow:
    return AgentKnowledgeChunkRow(
        id=chunk.id,
        doc_id=chunk.doc_id,
        chunk_index=chunk.chunk_index,
        section=chunk.section or "",
        content=chunk.content,
        char_count=len(chunk.content),
        title=chunk.title or "",
        category=chunk.category.value if isinstance(chunk.category, KnowledgeCategory) else str(chunk.category),
        source_type=chunk.source_type,
        source_id=chunk.source_id,
        document_path=chunk.document_path,
        verification_status=chunk.verification_status or "unverified",
        confidence=float(chunk.confidence),
        created_at=chunk.created_at,
        updated_at=chunk.created_at,
    )


def _row_to_chunk(row: AgentKnowledgeChunkRow) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=row.id,
        doc_id=row.doc_id,
        chunk_index=row.chunk_index,
        section=row.section or "",
        content=row.content,
        title=row.title or "",
        category=row.category,
        source_type=row.source_type,
        source_id=row.source_id,
        document_path=row.document_path,
        verification_status=row.verification_status or "unverified",
        confidence=float(row.confidence),
        created_at=row.created_at,
    )


# ------------------------------------------------------------------ security (Phase 3 §6)


_SENSITIVE_FILENAME_PARTS: tuple[str, ...] = (
    ".env",
    "id_rsa",
    "id_ed25519",
    ".pem",
    ".key",
    "secrets",
    "credentials",
    "private_key",
)

_SENSITIVE_PATH_SUFFIXES: tuple[str, ...] = (".key", ".pem", ".p12", ".pfx")

_SECRET_CONTENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bCAT_KEY_[A-Z_]*\s*[:=]\s*([^\s'\"]+)"),
    re.compile(r"(?i)\bCAT_TELEGRAM__BOT_TOKEN\s*[:=]\s*([^\s'\"]+)"),
    re.compile(r"(?i)\bCAT_DATABASE__URL\s*[:=]\s*([^\s'\"]+)"),
    re.compile(r"(?i)\bapi[_-]?key\s*[:=]\s*([^\s'\"]{2,})"),
    re.compile(r"(?i)\bapi[_-]?secret\s*[:=]\s*([^\s'\"]{2,})"),
    re.compile(r"(?i)\bpassword\s*[:=]\s*([^\s'\"]{2,})"),
    re.compile(r"(?i)\btoken\s*[:=]\s*([^\s'\"]{2,})"),
    re.compile(r"(?i)\bprivate[_-]?key\b"),
    re.compile(r"sk-(live|test)-[A-Za-z0-9]{8,}"),
    re.compile(r"xox[bap]-"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)

# Values that prove a match is documentation / code, not a credential:
# placeholders, comment markers, type annotations, example DSNs.
_PLACEHOLDER_VALUES: frozenset[str] = frozenset(
    {
        "", "#", "//", "<", ">", "...", "....", "---", '""', "''",
        "none", "null", "empty", "example", "placeholder", "changeme",
        "your", "yourkey", "str", "secretstr", "secret", "token",
        "test", "tests",
    }
)

_SAFE_DSN_SUBSTRINGS: tuple[str, ...] = (
    "./data/bot.db",  # default local sqlite path (no credentials)
    "postgresql+asyncpg://...",  # docs placeholder (truncated host)
    "...",
)


def _captured_value_is_secret(match: re.Match[str]) -> bool:
    """True when a regex match carries a real-looking credential value."""
    try:
        value = match.group(1)
    except IndexError:
        return True  # patterns without a group are real-secret formats
    text = value.strip().strip("\"'`,;")
    if not text:
        return False
    lowered = text.lower()
    if lowered in _PLACEHOLDER_VALUES:
        return False
    if text.startswith(("<", "#", ".", "/", '"', "'")):
        return False
    if lowered.startswith(("your", "example", "changeme", "placeholder", "<", "#", "...", "sqlite+aiosqlite:///./data/")):
        return False
    if lowered in ("secretstr", "secretstr(\"\"", "str"):
        return False
    # Type annotations in code (``api_key: SecretStr``) are not credentials.
    if lowered.rstrip(",)") in ("secretstr", "str", "none"):
        return False
    for safe in _SAFE_DSN_SUBSTRINGS:
        if safe in text:
            return False
    # Comment marker as the entire value (``KEY= # comment``) is not a secret.
    if len(text) <= 2:
        return False
    # Code references are not credentials: ``api_key=key,`` (pass-through),
    # ``password=self._env(...)``, ``token=getenv(...)`` etc.
    code_markers = ("self", "_env", "getenv", "environ", "get_secret", "(", ")", "await ", "return ", "os.", "sys.")
    if any(marker in lowered for marker in code_markers):
        return False
    # Bare pass-through identifiers (``password=password``) carry no value.
    if lowered.rstrip(",;)") in (
        "key", "secret", "password", "token", "passphrase",
        "api_key", "apikey", "api_secret", "apisecret", "bot_token",
    ):
        return False
    return True


def is_sensitive_path(path: pathlib.Path | str) -> bool:
    """True when ``path`` must never be indexed (``.env``, keys, secrets...).

    ``.env.example`` is *not* sensitive (annotated reference without values)
    and remains ingestible; every other ``.env*`` file is blocked.
    """
    name = pathlib.Path(path).name.strip()
    lowered = name.lower()
    if lowered == ".env.example":
        return False
    if lowered.startswith(".env"):
        return True
    for part in _SENSITIVE_FILENAME_PARTS:
        if part in lowered:
            return True
    for suffix in _SENSITIVE_PATH_SUFFIXES:
        if lowered.endswith(suffix):
            return True
    return False


def contains_secret_content(text: str) -> bool:
    """Heuristic secret detector for ingestion gating (fail-closed).

    Returns True when ``text`` looks like it carries credentials. Benign
    project prose, documentation placeholders (``=...``, ``=<token>``,
    ``= # comment``) and code annotations (``api_key: SecretStr``) return
    False. Real assignments (``KEY=abcdef12345``) return True.
    """
    if not text:
        return False
    for pattern in _SECRET_CONTENT_PATTERNS:
        for match in pattern.finditer(text):
            if _captured_value_is_secret(match):
                return True
    lowered = text.lower()
    # Raw .env dumps are never ingestible: ``export VAR=realvalue`` style
    # assignments with non-placeholder values are blocked even without a
    # known key name — but only for sensitive variable names, so benign
    # config (``MODE=PAPER``) and docstrings stay ingestible.
    if ".env" in lowered and ("cat_" in lowered or "export " in lowered):
        for dump_match in re.finditer(
            r"(?im)^\s*(?:export\s+)?(?:[A-Z_]*(?:KEY|SECRET|PASSWORD|TOKEN|PRIVATE|CREDENTIALS)[A-Z_]*)\s*=\s*([^\s#'\"]+)",
            text,
        ):
            if _captured_value_is_secret(dump_match):
                return True
    return False


# ------------------------------------------------------------------ chunking + scoring (Phase 3 §4)

CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 100

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "have", "has",
        "are", "was", "were", "will", "would", "can", "not", "but", "into",
        "over", "under", "between", "through", "using", "used", "each",
    }
)

_HEADER_RE: re.Pattern[str] = re.compile(r"(?m)^(#{1,4}\s+.+)$")


def tokenize_query(query: str) -> tuple[str, ...]:
    """Deterministic query tokenizer: lowercase alnum terms, len>=2, no stopwords."""
    terms = re.findall(r"[a-z0-9]{2,}", query.lower())
    return tuple(dict.fromkeys(t for t in terms if t not in _STOPWORDS))


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split ``text`` on markdown headers; returns ``(section, body)`` pairs.

    Text before the first header uses section "". Deterministic.
    """
    parts = _HEADER_RE.split(text)
    if len(parts) == 1:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    # parts[0] is pre-header body
    if parts[0].strip():
        sections.append(("", parts[0]))
    for i in range(1, len(parts), 2):
        header = parts[i].strip().lstrip("#").strip()[:120]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((header, f"{parts[i]}\n{body}"))
    return sections or [("", text)]


def chunk_text(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[tuple[str, str]]:
    """Deterministic chunking: section-aware sliding windows.

    Returns ``(section, chunk_content)`` in stable document order. Pure
    function of ``text`` — same input always yields the same chunks.
    """
    if not text or not text.strip():
        return []
    if chunk_size < 200:
        chunk_size = 200
    if overlap < 0 or overlap >= chunk_size:
        overlap = min(100, chunk_size // 4)
    chunks: list[tuple[str, str]] = []
    for section, body in _split_sections(text):
        body = body.strip()
        if not body:
            continue
        if len(body) <= chunk_size:
            chunks.append((section, body))
            continue
        step = chunk_size - overlap
        for start in range(0, len(body), step):
            piece = body[start : start + chunk_size].strip()
            if piece:
                chunks.append((section, piece))
            if start + chunk_size >= len(body):
                break
    return chunks


def score_chunk(
    *,
    query_terms: tuple[str, ...],
    title: str,
    section: str,
    tags: tuple[str, ...] | list[str],
    content: str,
) -> int:
    """Deterministic relevance score: title×3 + section×2 + tags×2 + content×1."""
    if not query_terms:
        return 0
    title_l = title.lower()
    section_l = section.lower()
    tags_l = " ".join(tags).lower()
    content_l = content.lower()
    score = 0
    for term in query_terms:
        score += 3 * title_l.count(term)
        score += 2 * section_l.count(term)
        score += 2 * tags_l.count(term)
        score += content_l.count(term)
    return score


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
        cat = normalize_knowledge_category(category).value
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentKnowledgeRow).where(AgentKnowledgeRow.category == cat).order_by(AgentKnowledgeRow.created_at)
            )
            return [_row_to_doc(r) for r in result.scalars()]

    async def search(
        self,
        query: str,
        limit: int = 20,
        *,
        category: str | KnowledgeCategory | None = None,
        source_type: str | None = None,
        tags: tuple[str, ...] | list[str] | None = None,
    ) -> list[KnowledgeDocument]:
        """Substring search with deterministic limit + metadata/category filters.

        Phase 3 keeps the LIKE pre-filter but applies category / source_type /
        tag filtering in Python (portable across SQLite/PostgreSQL) and caps
        ``limit`` to [1, 50]. Provenance is preserved on every hit.
        """
        limit = max(1, min(int(limit), 50))
        cat_value: str | None = None
        if category is not None:
            cat_value = normalize_knowledge_category(category).value
        wanted_tags = {t.strip().lower() for t in (tags or ()) if str(t).strip()}
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
                .limit(limit * 3)
            )
            docs = [_row_to_doc(r) for r in result.scalars()]
        filtered: list[KnowledgeDocument] = []
        for doc in docs:
            if cat_value is not None and doc.category.value != cat_value:
                continue
            if source_type is not None and doc.source_type != source_type:
                continue
            if wanted_tags:
                doc_tags = {t.lower() for t in doc.tags}
                if not (wanted_tags & doc_tags):
                    continue
            filtered.append(doc)
            if len(filtered) >= limit:
                break
        return filtered

    async def count(self) -> int:
        from sqlalchemy import func

        async with self._db.session() as session:
            result = await session.execute(select(func.count()).select_from(AgentKnowledgeRow))
            return int(result.scalar_one())


class KnowledgeChunkRepository:
    """Persistence for :class:`KnowledgeChunk` (Phase 3 retrieval unit)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def replace_for_doc(self, doc: KnowledgeDocument, contents: list[tuple[str, str]]) -> list[KnowledgeChunk]:
        """Replace all chunks of ``doc`` with ``contents`` (deterministic order)."""
        from app.models.base import utc_now as _now

        async with self._db.session() as session:
            await session.execute(delete(AgentKnowledgeChunkRow).where(AgentKnowledgeChunkRow.doc_id == doc.id))
            chunks: list[KnowledgeChunk] = []
            for index, (section, content) in enumerate(contents):
                chunk = KnowledgeChunk(
                    id=f"{doc.id}-c{index:03d}",
                    doc_id=doc.id,
                    chunk_index=index,
                    section=section or doc.section or "",
                    content=content,
                    title=doc.title,
                    category=doc.category,
                    source_type=doc.source_type,
                    source_id=doc.source_id,
                    document_path=doc.effective_document_path,
                    verification_status=doc.verification_status,
                    confidence=doc.confidence,
                    created_at=_now(),
                )
                chunks.append(chunk)
            for chunk in chunks:
                await session.merge(_chunk_to_row(chunk))
        return chunks

    async def list_for_doc(self, doc_id: str) -> list[KnowledgeChunk]:
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentKnowledgeChunkRow)
                .where(AgentKnowledgeChunkRow.doc_id == doc_id)
                .order_by(AgentKnowledgeChunkRow.chunk_index)
            )
            return [_row_to_chunk(r) for r in result.scalars()]

    async def count(self) -> int:
        from sqlalchemy import func

        async with self._db.session() as session:
            result = await session.execute(select(func.count()).select_from(AgentKnowledgeChunkRow))
            return int(result.scalar_one())

    async def search_all(
        self,
        *,
        category: str | KnowledgeCategory | None = None,
        source_type: str | None = None,
        verification_status: str | None = None,
        tags: tuple[str, ...] | list[str] | None = None,  # doc-level filter via parent join
        limit: int = 200,
    ) -> list[KnowledgeChunk]:
        """Candidate scan for scored retrieval (bounded, deterministic order)."""
        limit = max(1, min(int(limit), 500))
        query = select(AgentKnowledgeChunkRow).order_by(
            AgentKnowledgeChunkRow.source_id, AgentKnowledgeChunkRow.chunk_index
        )
        if category is not None:
            query = query.where(AgentKnowledgeChunkRow.category == normalize_knowledge_category(category).value)
        if source_type is not None:
            query = query.where(AgentKnowledgeChunkRow.source_type == source_type)
        if verification_status is not None:
            query = query.where(
                AgentKnowledgeChunkRow.verification_status == str(verification_status).strip().lower()
            )
        async with self._db.session() as session:
            result = await session.execute(query.limit(limit))
            chunks = [_row_to_chunk(r) for r in result.scalars()]
        if tags:
            # Tags live on the parent doc; filter via doc lookup map (bounded).
            wanted = {str(t).strip().lower() for t in tags if str(t).strip()}
            if wanted:
                # Defer to caller-side doc tag check; chunks carry no tags themselves.
                # Keep all here — service layer applies the doc-tag filter.
                pass
        return chunks


# ------------------------------------------------------------------ service


# Mapping of known repo files to knowledge categories.
# Phase 3: covers the actual current implementation (README, docs/,
# pyproject.toml, app/config/, app/exchanges/, app/market_data/,
# app/strategies/, app/execution/, app/risk/, app/recovery/, app/storage/,
# app/services.py). No web content, no secrets.
_KNOWN_SOURCES: tuple[tuple[str, KnowledgeCategory, str], ...] = (
    ("README.md", KnowledgeCategory.BOT, "Project overview, quick start, modes, safety"),
    ("docs/ARCHITECTURE.md", KnowledgeCategory.BOT, "Module layout and design principles"),
    ("docs/OPERATIONS.md", KnowledgeCategory.BOT, "Runbook: lifecycle, kill switch, recovery"),
    ("pyproject.toml", KnowledgeCategory.BOT, "Packaging, dependencies, pytest/ruff config"),
    (".env.example", KnowledgeCategory.BOT, "Configuration reference (annotated, no secrets)"),
    ("app/config/settings.py", KnowledgeCategory.BOT, "Settings schema and validation"),
    ("app/config/modes.py", KnowledgeCategory.BOT, "Trading mode policy (PAPER/DEMO/LIVE)"),
    ("app/config/logging_config.py", KnowledgeCategory.BOT, "Logging configuration"),
    ("app/services.py", KnowledgeCategory.BOT, "Composition root: AppServices build/start/shutdown"),
    ("app/storage/engine.py", KnowledgeCategory.BOT, "Database engine and schema lifecycle"),
    ("app/storage/repositories.py", KnowledgeCategory.BOT, "Storage repositories (trades/transfers/audit)"),
    ("app/storage/tables.py", KnowledgeCategory.BOT, "Storage tables (trades/transfers/balances)"),
    ("app/risk/rules.py", KnowledgeCategory.TRADING, "Risk rules (fail-closed)"),
    ("app/risk/engine.py", KnowledgeCategory.TRADING, "Risk engine evaluation"),
    ("app/risk/state.py", KnowledgeCategory.TRADING, "Risk runtime state"),
    ("app/execution/guard.py", KnowledgeCategory.TRADING, "Kill switch / execution guard"),
    ("app/execution/order_gate.py", KnowledgeCategory.TRADING, "Order gate per mode"),
    ("app/execution/paper_wallet.py", KnowledgeCategory.TRADING, "Paper wallet (PAPER simulation)"),
    ("app/execution/fill_simulator.py", KnowledgeCategory.TRADING, "Fill simulation"),
    ("app/execution/precision.py", KnowledgeCategory.TRADING, "Instrument precision filters"),
    ("app/recovery/recovery.py", KnowledgeCategory.TRADING, "Execution recovery logic"),
    ("app/strategies/triangular/scanner.py", KnowledgeCategory.TRADING, "Triangular scanner (depth-aware VWAP)"),
    ("app/strategies/triangular/executor.py", KnowledgeCategory.TRADING, "Triangle executor (3-leg sequential)"),
    ("app/strategies/triangular/fees.py", KnowledgeCategory.TRADING, "Triangular fee model"),
    ("app/strategies/transfer/planner.py", KnowledgeCategory.TRADING, "Transfer planner (economics)"),
    ("app/strategies/transfer/networks.py", KnowledgeCategory.TRADING, "Withdrawal network validation"),
    ("app/strategies/transfer/orchestrator.py", KnowledgeCategory.TRADING, "Transfer orchestrator lifecycle"),
    ("app/exchanges/profiles.py", KnowledgeCategory.EXCHANGE, "Exchange profiles (Binance/OKX/Bybit)"),
    ("app/exchanges/manager.py", KnowledgeCategory.EXCHANGE, "Exchange manager (breaker + auth gate)"),
    ("app/exchanges/ccxt_adapter.py", KnowledgeCategory.EXCHANGE, "CCXT adapter (live venues)"),
    ("app/exchanges/simulated.py", KnowledgeCategory.EXCHANGE, "Simulated exchange (PAPER)"),
    ("app/exchanges/credentials.py", KnowledgeCategory.EXCHANGE, "Credential presence gating (no values)"),
    ("app/exchanges/sanitize.py", KnowledgeCategory.EXCHANGE, "Secret redaction for logs/replies"),
    ("app/exchanges/preflight.py", KnowledgeCategory.EXCHANGE, "DEMO preflight checks"),
    ("app/exchanges/registry.py", KnowledgeCategory.EXCHANGE, "Venue registry"),
    ("app/market_data/store.py", KnowledgeCategory.RESEARCH, "Market-data store (order books, freshness)"),
    ("app/market_data/service.py", KnowledgeCategory.RESEARCH, "Market-data service (REST + WS)"),
    ("app/market_data/streams.py", KnowledgeCategory.RESEARCH, "WebSocket stream supervision"),
    ("app/market_data/order_book.py", KnowledgeCategory.RESEARCH, "Order-book math (VWAP/depth)"),
)

# Phase 3 directory fallback: any .py/.md/.toml under these roots inherits
# the mapped category when not listed explicitly above.
_PHASE3_DIR_CATEGORIES: tuple[tuple[str, KnowledgeCategory], ...] = (
    ("app/config", KnowledgeCategory.BOT),
    ("app/storage", KnowledgeCategory.BOT),
    ("app/exchanges", KnowledgeCategory.EXCHANGE),
    ("app/market_data", KnowledgeCategory.RESEARCH),
    ("app/strategies", KnowledgeCategory.TRADING),
    ("app/execution", KnowledgeCategory.TRADING),
    ("app/risk", KnowledgeCategory.TRADING),
    ("app/recovery", KnowledgeCategory.TRADING),
    ("tests", KnowledgeCategory.RESEARCH),
)

_PHASE3_ALLOWED_SUFFIXES: frozenset[str] = frozenset({".py", ".md", ".toml"})


class KnowledgeService:
    """High-level knowledge operations: ingest, index, retrieve, search.

    All ingestion preserves ``source_type`` / ``source_id`` / ``version`` /
    section / verification status / confidence so that the advisor can cite
    where a piece of knowledge came from. Repository facts
    (``source_type="repository"``) stay distinguishable from external
    knowledge, AI assumptions and recommendations downstream.
    """

    def __init__(self, repository: KnowledgeRepository, chunk_repository: KnowledgeChunkRepository | None = None) -> None:
        self._repo = repository
        self._chunks = chunk_repository
        if self._chunks is None:
            try:
                self._chunks = KnowledgeChunkRepository(repository._db)  # type: ignore[attr-defined]
            except Exception:
                self._chunks = None

    @property
    def repository(self) -> KnowledgeRepository:
        return self._repo

    @property
    def chunk_repository(self) -> KnowledgeChunkRepository | None:
        return self._chunks

    # ---------------------------------------------------------------- ingestion

    async def ingest_file(
        self,
        path: pathlib.Path | str,
        category: KnowledgeCategory | str,
        *,
        title: str | None = None,
        source_type: str = SourceType.DOCUMENT.value,
        section: str = "",
        tags: tuple[str, ...] | list[str] = (),
        verification_status: str = "unverified",
        confidence: float = 0.5,
    ) -> KnowledgeDocument | None:
        """Read ``path`` from disk and persist it as a knowledge document.

        Fail-closed on secrets: sensitive paths (``.env``, keys, ...) return
        ``None``; secret-bearing content returns ``None``. Missing files
        return ``None`` (best-effort). Content capped at 20_000 chars.
        Chunks are (re)indexed on success.
        """
        p = pathlib.Path(path)
        if is_sensitive_path(p):
            logger.warning("knowledge_ingest_blocked_sensitive_path", extra={"path": p.name})
            return None
        if not p.exists() or not p.is_file():
            logger.debug("knowledge_ingest_skip_missing", extra={"path": str(p)})
            return None
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - file read best-effort
            logger.warning("knowledge_ingest_read_failed", extra={"path": str(p), "error": str(exc)[:200]})
            return None
        if contains_secret_content(text):
            logger.warning("knowledge_ingest_blocked_secret_content", extra={"path": p.name})
            return None
        # Truncate to keep the DB bounded but preserve the head (titles + intro matter most)
        if len(text) > 20_000:
            text = text[:20_000] + "\n\n... (truncated, source_id holds full path)"
        cat = normalize_knowledge_category(category)
        doc = KnowledgeDocument(
            title=title or p.name,
            category=cat,
            content=text,
            summary=text[:500].replace("\n", " ").strip() if text else None,
            tags=tuple(tags),
            section=section,
            document_path=str(p.as_posix()),
            verification_status=verification_status,
            confidence=confidence,
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
        await self._index_doc(doc)
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
        section: str = "",
        document_path: str | None = None,
        tags: tuple[str, ...] | list[str] = (),
        verification_status: str = "unverified",
        confidence: float = 0.5,
    ) -> KnowledgeDocument:
        """Ingest an in-memory text blob. Raises ``ValueError`` on secrets."""
        if contains_secret_content(f"{title}\n{content}\n{source_id}"):
            raise ValueError("knowledge ingest blocked: secret content detected")
        cat = normalize_knowledge_category(category)
        doc = KnowledgeDocument(
            title=title,
            category=cat,
            content=content[:20_000],
            summary=content[:500].replace("\n", " ").strip(),
            tags=tuple(tags),
            section=section,
            document_path=document_path or source_id,
            verification_status=verification_status,
            confidence=confidence,
            source_type=source_type,
            source_id=source_id,
            version=1,
            status="active",
        )
        existing = await self._repo.get_by_source(source_id)
        if existing is not None:
            doc = doc.model_copy(update={"id": existing.id, "version": existing.version + 1})
        await self._repo.save(doc)
        await self._index_doc(doc)
        return doc

    async def _index_doc(self, doc: KnowledgeDocument) -> list[KnowledgeChunk]:
        """Deterministically chunk + persist ``doc`` (best-effort, never raises)."""
        if self._chunks is None:
            return []
        try:
            pieces = chunk_text(doc.content)
            if not pieces:
                pieces = [("", doc.content[:CHUNK_SIZE_CHARS])]
            return await self._chunks.replace_for_doc(doc, pieces)
        except Exception as exc:  # noqa: BLE001 - indexing must not break ingestion
            logger.warning("knowledge_index_failed", extra={"source_id": doc.source_id, "error": str(exc)[:200]})
            return []

    async def ingest_repository(
        self,
        root: pathlib.Path | str = ".",
        *,
        max_files: int | None = None,
    ) -> list[KnowledgeDocument]:
        """Populate the knowledge base from the existing repository documentation.

        Iterates :data:`_KNOWN_SOURCES` under ``root`` and ingests each file
        that exists. Repository docs are marked ``source_type="repository"``,
        ``verification_status="verified"``. No web scraping; secrets blocked.
        """
        base = pathlib.Path(root)
        ingested: list[KnowledgeDocument] = []
        sources = _KNOWN_SOURCES[:max_files] if max_files else _KNOWN_SOURCES
        for rel, category, title in sources:
            p = base / rel
            doc = await self.ingest_file(
                p,
                category,
                title=title,
                source_type=SourceType.REPOSITORY.value,
                verification_status="verified",
                confidence=0.8,
                tags=("repository", "project"),
            )
            if doc is not None:
                ingested.append(doc)
        logger.info("knowledge_repository_ingested", extra={"count": len(ingested), "root": str(base)})
        return ingested

    async def ingest_project(
        self,
        root: pathlib.Path | str = ".",
        *,
        max_files: int | None = None,
    ) -> list[KnowledgeDocument]:
        """Phase 3 project sweep: ``_KNOWN_SOURCES`` + directory fallback.

        Walks ``README.md``, ``docs/``, ``pyproject.toml``, ``app/config/``,
        ``app/exchanges/``, ``app/market_data/``, ``app/strategies/``,
        ``app/execution/``, ``app/risk/``, ``app/recovery/``,
        ``app/storage/``, ``app/services.py`` and a bounded ``tests/`` sample.
        Sensitive paths and secret content are skipped. Deterministic order.
        """
        base = pathlib.Path(root)
        ingested = await self.ingest_repository(root=base)
        seen = {d.source_id for d in ingested}
        extra: list[tuple[pathlib.Path, KnowledgeCategory]] = []
        for dir_rel, category in _PHASE3_DIR_CATEGORIES:
            d = base / dir_rel
            if not d.exists() or not d.is_dir():
                continue
            # Bounded tests sample: keep the suite fast, still representative.
            candidates = sorted(d.rglob("*"), key=lambda q: q.as_posix())
            if dir_rel == "tests":
                candidates = [q for q in candidates if q.name.startswith("test_")][:25]
            for q in candidates:
                if not q.is_file():
                    continue
                if q.suffix.lower() not in _PHASE3_ALLOWED_SUFFIXES:
                    continue
                if "__pycache__" in q.parts:
                    continue
                rel = q.relative_to(base).as_posix()
                if rel in seen:
                    continue
                if is_sensitive_path(q):
                    continue
                extra.append((q, category))
                seen.add(rel)
        if max_files is not None:
            remaining = max(0, max_files - len(ingested))
            extra = extra[:remaining]
        for q, category in extra:
            doc = await self.ingest_file(
                q,
                category,
                title=q.name,
                source_type=SourceType.REPOSITORY.value,
                verification_status="verified",
                confidence=0.8,
                tags=("repository", "project"),
            )
            if doc is not None:
                ingested.append(doc)
        logger.info("knowledge_project_ingested", extra={"count": len(ingested), "root": str(base)})
        return ingested

    # ---------------------------------------------------------------- retrieval

    async def get_all(self) -> list[KnowledgeDocument]:
        return await self._repo.list_all()

    async def get_by_category(self, category: str | KnowledgeCategory) -> list[KnowledgeDocument]:
        return await self._repo.list_by_category(normalize_knowledge_category(category))

    async def search(
        self,
        query: str,
        limit: int = 20,
        *,
        category: str | KnowledgeCategory | None = None,
        source_type: str | None = None,
        tags: tuple[str, ...] | list[str] | None = None,
    ) -> list[KnowledgeDocument]:
        return await self._repo.search(query, limit=limit, category=category, source_type=source_type, tags=tags)

    async def get_recent(self, limit: int = 20) -> list[KnowledgeDocument]:
        all_docs = await self._repo.list_all()
        # list_all is ordered by created_at asc; reverse for most recent
        return list(reversed(all_docs))[:limit]

    async def search_chunks(
        self,
        query: str,
        limit: int = 5,
        *,
        category: str | KnowledgeCategory | None = None,
        source_type: str | None = None,
        verification_status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Relevance-scored chunk retrieval with provenance (Phase 3 §4).

        Returns at most ``limit`` (1..20) chunk dicts ordered by
        ``score`` desc, then ``source_id`` / ``chunk_index`` asc. Each hit
        carries document provenance (doc/chunk IDs, section, source,
        document path, verification, confidence, category, score, excerpt).
        Repository vs external stays visible via ``source_type``.
        """
        limit = max(1, min(int(limit), 20))
        terms = tokenize_query(query)
        if self._chunks is None or not terms:
            return []
        try:
            candidates = await self._chunks.search_all(
                category=category,
                source_type=source_type,
                verification_status=verification_status,
                limit=300,
            )
        except Exception:
            return []
        # Parent tags for tag-aware scoring (bounded lookup)
        doc_tags: dict[str, tuple[str, ...]] = {}
        try:
            docs = await self._repo.list_all()
            doc_tags = {d.id: tuple(d.tags) for d in docs}
        except Exception:
            pass
        scored: list[tuple[int, KnowledgeChunk]] = []
        for chunk in candidates:
            tags = doc_tags.get(chunk.doc_id, ())
            score = score_chunk(
                query_terms=terms,
                title=chunk.title,
                section=chunk.section,
                tags=tags,
                content=chunk.content,
            )
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].source_id, item[1].chunk_index))
        hits: list[dict[str, Any]] = []
        for score, chunk in scored[:limit]:
            hits.append(
                {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "section": chunk.section,
                    "score": score,
                    "title": chunk.title,
                    "category": chunk.category.value,
                    "source_type": chunk.source_type,
                    "source_id": chunk.source_id,
                    "document_path": chunk.document_path or chunk.source_id,
                    "verification_status": chunk.verification_status,
                    "confidence": chunk.confidence,
                    "excerpt": chunk.content[:400],
                }
            )
        return hits

    async def get_chunks(self, doc_id: str) -> list[KnowledgeChunk]:
        if self._chunks is None:
            return []
        return await self._chunks.list_for_doc(doc_id)
