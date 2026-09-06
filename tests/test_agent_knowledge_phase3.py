"""Phase 3 — Knowledge Base focused tests.

Covers: repository ingestion, deterministic chunking, indexing, relevance
retrieval, metadata, category separation (+BOT_KNOWLEDGE aliases),
provenance, persistence/reload, secret exclusion, AI context integration.

Read-only w.r.t. trading/execution/risk/recovery/exchanges.
"""

from __future__ import annotations

import pathlib

import pytest

from app.agent.knowledge import (
    chunk_text,
    contains_secret_content,
    is_sensitive_path,
    score_chunk,
    tokenize_query,
)
from app.agent.models import (
    BOT_KNOWLEDGE,
    EXCHANGE_KNOWLEDGE,
    RESEARCH_KNOWLEDGE,
    TRADING_KNOWLEDGE,
    KnowledgeCategory,
    normalize_knowledge_category,
)
from app.config.settings import DatabaseSettings
from app.storage.engine import Database


async def _memory_db(tmp_path: pathlib.Path) -> Database:
    db = Database(DatabaseSettings(url=f"sqlite+aiosqlite:///{tmp_path / 'kb3.db'}"))
    await db.create_schema()
    return db


def _svc(db: Database):
    from app.agent.knowledge import KnowledgeRepository, KnowledgeService

    return KnowledgeService(KnowledgeRepository(db))


# ------------------------------------------------------------------ categories


def test_phase3_category_aliases():
    assert normalize_knowledge_category("BOT_KNOWLEDGE") is KnowledgeCategory.BOT
    assert normalize_knowledge_category("bot_knowledge") is KnowledgeCategory.BOT
    assert normalize_knowledge_category("BOT") is KnowledgeCategory.BOT
    assert normalize_knowledge_category("exchange_knowledge") is KnowledgeCategory.EXCHANGE
    assert normalize_knowledge_category("TRADING_KNOWLEDGE") is KnowledgeCategory.TRADING
    assert normalize_knowledge_category("research_knowledge") is KnowledgeCategory.RESEARCH
    assert BOT_KNOWLEDGE is KnowledgeCategory.BOT
    assert EXCHANGE_KNOWLEDGE is KnowledgeCategory.EXCHANGE
    assert TRADING_KNOWLEDGE is KnowledgeCategory.TRADING
    assert RESEARCH_KNOWLEDGE is KnowledgeCategory.RESEARCH
    with pytest.raises(ValueError):
        normalize_knowledge_category("unknown_knowledge")


# ------------------------------------------------------------------ chunking


def test_phase3_chunking_deterministic():
    text = "# Risk\n" + ("risk rule fail-closed kill switch. " * 60) + "\n# Execution\n" + ("guard order gate. " * 40)
    first = chunk_text(text)
    second = chunk_text(text)
    assert first == second
    assert len(first) >= 3
    sections = {s for s, _ in first}
    assert "Risk" in sections and "Execution" in sections
    for _, content in first:
        assert 0 < len(content) <= 800
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_phase3_tokenize_and_score():
    terms = tokenize_query("Risk Kill-Switch rules!")
    assert "risk" in terms
    assert "the" not in tokenize_query("the risk")
    s_title = score_chunk(query_terms=("risk",), title="Risk rules", section="", tags=(), content="unrelated")
    s_body = score_chunk(query_terms=("risk",), title="Other", section="", tags=(), content="risk risk risk")
    # title weight (3) vs content count: title hit must outscore single content hit
    s_single = score_chunk(query_terms=("risk",), title="Other", section="", tags=(), content="risk")
    assert s_title > s_single
    assert s_body > s_single
    assert score_chunk(query_terms=(), title="Risk", section="", tags=(), content="risk") == 0


# ------------------------------------------------------------------ ingestion + metadata


@pytest.mark.asyncio
async def test_phase3_repository_ingestion_covers_required_sources(tmp_path):
    db = await _memory_db(tmp_path)
    try:
        svc = _svc(db)
        ingested = await svc.ingest_repository(root=pathlib.Path("."))
        by_id = {d.source_id: d for d in ingested}
        for required in (
            "README.md",
            "docs/ARCHITECTURE.md",
            "docs/OPERATIONS.md",
            "pyproject.toml",
            "app/config/settings.py",
            "app/services.py",
            "app/risk/rules.py",
            "app/execution/guard.py",
            "app/recovery/recovery.py",
            "app/exchanges/manager.py",
            "app/market_data/store.py",
        ):
            assert any(s.endswith(required) for s in by_id), f"missing {required}"
        # Every record carries full Phase 3 metadata + provenance
        for doc in ingested:
            assert doc.id.startswith("kd-")
            assert doc.source_type == "repository"
            assert doc.source_id
            assert doc.effective_document_path
            assert doc.created_at is not None and doc.updated_at is not None
            assert doc.verification_status == "verified"
            assert 0.0 <= doc.confidence <= 1.0
            assert doc.category in (
                KnowledgeCategory.BOT,
                KnowledgeCategory.EXCHANGE,
                KnowledgeCategory.TRADING,
                KnowledgeCategory.RESEARCH,
            )
            assert doc.source.startswith("repository:")
    finally:
        await db.dispose()


@pytest.mark.asyncio
async def test_phase3_project_sweep_covers_all_required_roots(tmp_path):
    db = await _memory_db(tmp_path)
    try:
        svc = _svc(db)
        ingested = await svc.ingest_project(root=pathlib.Path("."), max_files=400)
        sources = {d.source_id for d in ingested}
        for root in (
            "app/config", "app/exchanges", "app/market_data", "app/strategies",
            "app/execution", "app/risk", "app/recovery", "app/storage",
        ):
            assert any(s.startswith(root) or s == root for s in sources), f"root uncovered: {root}"
        assert any(s.endswith("app/services.py") for s in sources)
        assert any(s == "pyproject.toml" or s.endswith("pyproject.toml") for s in sources)
    finally:
        await db.dispose()


@pytest.mark.asyncio
async def test_phase3_indexing_and_relevance(tmp_path):
    db = await _memory_db(tmp_path)
    try:
        svc = _svc(db)
        await svc.ingest_text(
            "Kill switch guard", "The execution guard engages the kill switch and blocks trading.",
            "TRADING_KNOWLEDGE", source_id="repo/guard", source_type="repository",
            verification_status="verified", confidence=0.9, tags=("safety",),
        )
        await svc.ingest_text(
            "Order book depth", "Market data order books stream tickers and depth.",
            RESEARCH_KNOWLEDGE, source_id="repo/market", source_type="repository",
            verification_status="verified", confidence=0.7, tags=("market",),
        )
        # Chunks indexed for both docs
        assert await svc.chunk_repository.count() >= 2
        chunks = await svc.get_chunks((await svc.repository.get_by_source("repo/guard")).id)
        assert len(chunks) >= 1
        assert chunks[0].source_id == "repo/guard"
        # Relevance: kill-switch query ranks guard first
        hits = await svc.search_chunks("kill switch guard", limit=5)
        assert hits
        assert hits[0]["source_id"] == "repo/guard"
        assert hits[0]["score"] > 0
        assert all(h["score"] >= hits[-1]["score"] for h in hits)  # desc order
        # Limits + category filter
        assert len(await svc.search_chunks("kill switch", limit=1)) == 1
        trading_hits = await svc.search_chunks("kill switch", limit=5, category="TRADING_KNOWLEDGE")
        assert all(h["category"] == "trading" for h in trading_hits)
        # Document search honors category filter too
        docs = await svc.search("kill", limit=5, category=TRADING_KNOWLEDGE)
        assert all(d.category is KnowledgeCategory.TRADING for d in docs)
    finally:
        await db.dispose()


@pytest.mark.asyncio
async def test_phase3_category_separation_and_provenance(tmp_path):
    db = await _memory_db(tmp_path)
    try:
        svc = _svc(db)
        await svc.ingest_text("Repo bot", "bot behaviour cli", "BOT_KNOWLEDGE",
                              source_id="repo/bot", source_type="repository")
        await svc.ingest_text("External research", "bot behaviour cli", "BOT_KNOWLEDGE",
                              source_id="https://example.com/ext", source_type="document")
        bot_docs = await svc.get_by_category("BOT_KNOWLEDGE")
        assert {d.source_id for d in bot_docs} >= {"repo/bot", "https://example.com/ext"}
        # Repository vs external distinguishable by source_type
        by_type = {d.source_id: d.source_type for d in bot_docs}
        assert by_type["repo/bot"] == "repository"
        assert by_type["https://example.com/ext"] == "document"
        # Exchange category stays separate
        await svc.ingest_text("Venue", "binance profile", EXCHANGE_KNOWLEDGE, source_id="repo/venue")
        assert all(d.category is KnowledgeCategory.BOT for d in await svc.get_by_category(BOT_KNOWLEDGE))
        assert any(d.source_id == "repo/venue" for d in await svc.get_by_category("EXCHANGE_KNOWLEDGE"))
    finally:
        await db.dispose()


@pytest.mark.asyncio
async def test_phase3_persistence_reload(tmp_path):
    from app.agent.knowledge import KnowledgeRepository, KnowledgeService

    url = f"sqlite+aiosqlite:///{tmp_path / 'persist.db'}"
    db = Database(DatabaseSettings(url=url))
    await db.create_schema()
    try:
        svc = KnowledgeService(KnowledgeRepository(db))
        doc = await svc.ingest_text("Persist me", "kill switch content " * 30, TRADING_KNOWLEDGE,
                                    source_id="repo/persist", source_type="repository",
                                    section="Safety", verification_status="verified", confidence=0.77,
                                    tags=("reload",))
        doc_id, chunk_n = doc.id, await svc.chunk_repository.count()
        assert chunk_n >= 1
    finally:
        await db.dispose()
    db2 = Database(DatabaseSettings(url=url))
    await db2.create_schema()
    try:
        svc2 = KnowledgeService(KnowledgeRepository(db2))
        loaded = await svc2.repository.get(doc_id)
        assert loaded is not None
        assert loaded.section == "Safety"
        assert loaded.verification_status == "verified"
        assert loaded.confidence == pytest.approx(0.77)
        assert loaded.tags == ("reload",)
        assert (await svc2.chunk_repository.count()) == chunk_n
        assert await svc2.get_chunks(doc_id)
    finally:
        await db2.dispose()


# ------------------------------------------------------------------ security


def test_phase3_sensitive_paths():
    assert is_sensitive_path(".env")
    assert is_sensitive_path("config/.env")
    assert is_sensitive_path("id_rsa")
    assert is_sensitive_path("api_private.key")
    assert is_sensitive_path("wallet.pem")
    assert not is_sensitive_path(".env.example")
    assert not is_sensitive_path("README.md")
    assert not is_sensitive_path("app/risk/rules.py")


def test_phase3_secret_content_detector():
    assert contains_secret_content("CAT_KEY_BINANCE_SECRET=supersecret123")
    assert contains_secret_content("CAT_TELEGRAM__BOT_TOKEN=123:ABC")
    assert contains_secret_content("api_key=abcd1234")
    assert contains_secret_content("password=hunter2secret")
    assert not contains_secret_content("The kill switch blocks trading when risk limits breach.")
    assert not contains_secret_content("risk rule fail-closed guard")


@pytest.mark.asyncio
async def test_phase3_secret_exclusion_on_ingest(tmp_path):
    db = await _memory_db(tmp_path)
    try:
        svc = _svc(db)
        # Sensitive path refused
        env_file = tmp_path / ".env"
        env_file.write_text("CAT_KEY_BINANCE_SECRET=real\n", encoding="utf-8")
        assert await svc.ingest_file(env_file, KnowledgeCategory.BOT) is None
        # Secret content refused (file + text)
        leak = tmp_path / "leak.md"
        leak.write_text("deploy notes\nCAT_KEY_OKX_SECRET=topsecret\n", encoding="utf-8")
        assert await svc.ingest_file(leak, KnowledgeCategory.BOT) is None
        with pytest.raises(ValueError, match="secret"):
            await svc.ingest_text("t", "api_key=abcd1234", KnowledgeCategory.BOT, source_id="x")
        # .env.example stays ingestible (annotated reference, no values)
        assert await svc.ingest_file(pathlib.Path(".env.example"), KnowledgeCategory.BOT) is not None
        # Nothing secret-bearing persisted
        for doc in await svc.get_all():
            assert not contains_secret_content(doc.title + "\n" + doc.content)
            assert not is_sensitive_path(doc.source_id) or doc.source_id.endswith(".env.example")
    finally:
        await db.dispose()


# ------------------------------------------------------------------ AI context integration


@pytest.mark.asyncio
async def test_phase3_ai_context_integration(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        _, kb, *_ = build_agent(app)
        await kb.ingest_text(
            "Kill switch runbook", "When the kill switch engages, trading halts and release is manual.",
            TRADING_KNOWLEDGE, source_id="repo/runbook", source_type="repository",
            section="Safety", verification_status="verified", confidence=0.9, tags=("safety",),
        )
        await kb.ingest_text(
            "Outside blog", "Some external opinion about kill switches.",
            TRADING_KNOWLEDGE, source_id="https://example.com/blog", source_type="document",
            verification_status="unverified", confidence=0.3,
        )
        from app.agent.core import AgentRequest

        core = app.agent_knowledge_service  # service wired; build core path via adapter below
        assert core is not None
        # Collect context with a query: chunk hits carry full provenance
        from app.agent.context import ContextCollector
        from app.agent.tools import AgentTools

        collector = ContextCollector(
            AgentTools(app),
            knowledge_service=kb,
            experience_repo=app.agent_experiences,
            lesson_repo=app.agent_lessons,
            recommendation_repo=app.agent_recommendations,
        )
        ctx = await collector.collect(query="kill switch engages trading halts")
        assert ctx.knowledge_hits, "expected relevant knowledge, not the whole repo"
        assert len(ctx.knowledge_hits) <= 3
        top = ctx.knowledge_hits[0]
        for field in ("source_id", "source_type", "document_path", "section",
                      "verification_status", "confidence", "score", "excerpt"):
            assert field in top, f"provenance field missing: {field}"
        assert top["source_id"] == "repo/runbook"  # relevant repo chunk first
        # Analysis keeps repo facts / external facts / hypotheses / recommendations distinct
        from app.agent.analysis import AnalysisEngine

        engine = AnalysisEngine()
        analysis = engine.build(context=ctx, reflection=None, llm_output="maybe the guard is slow")
        facts_joined = "\n".join(analysis.facts)
        assert "repo-fact" in facts_joined
        assert "repo/runbook" in facts_joined
        assert "LLM hypothesis" in "\n".join(analysis.hypotheses)
        assert analysis.recommendations == ()
        # Full core pipeline preserves provenance end-to-end
        from app.agent import build_agent as _build

        core2, *_ = _build(app)
        resp = await core2.handle(AgentRequest(query="kill switch engages trading halts"))
        assert resp.context.knowledge_hits
        assert resp.context.knowledge_hits[0]["source_id"] == "repo/runbook"
    finally:
        await shutdown_app(app)
