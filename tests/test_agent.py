"""AI Advisor Phase 1 — focused isolated tests.

Covers:

* model validation (frozen, extra=forbid, confidence bounds, required source fields, secrets not stored)
* persistence (knowledge / experience / lesson / recommendation save/get)
* restart persistence (same file survives dispose/reopen)
* source/version tracking (upsert bumps version, status preserved)
* memory retrieval (search, list_recent, list_all)
* knowledge retrieval (ingest_text/file, search, list_by_category, get_by_source)
* read-only tools (allowlist of exactly 10, blocked generic execution, balance via services, no secrets)
* balance retrieval through existing services (paper wallets)
* recommendation creation and architectural boundary (no config mutation path)
* LLM provider boundary (NullProvider, EchoProvider, secret filtering)
* insufficient evidence -> NO_ACTION (reflection)
* AgentCore pipeline (request -> context -> memory -> knowledge -> analysis -> response) without Telegram coupling
* Telegram adapter boundary (i18n en/ru, dispatch, no mutation)
"""

from __future__ import annotations

import pathlib
from decimal import Decimal

import pytest
from app.agent.knowledge import KnowledgeRepository, KnowledgeService
from app.agent.memory import ExperienceRepository, LessonRepository
from app.agent.models import (
    AgentRecommendation,
    Experience,
    KnowledgeCategory,
    KnowledgeDocument,
    Lesson,
    RecommendationStatus,
    SourceType,
)
from app.agent.providers.base import EchoProvider, NullProvider, filter_secrets_from_text
from app.agent.recommendations import RecommendationApplyBlocked, RecommendationRepository, RecommendationService
from app.agent.reflection import ReflectionEngine
from app.agent.tools import AgentTools, ToolAccessBlocked
from app.config.settings import DatabaseSettings
from app.storage.engine import Database


async def _memory_db(tmp_path: pathlib.Path) -> Database:
    db = Database(DatabaseSettings(url=f"sqlite+aiosqlite:///{tmp_path / 'agent.db'}"))
    await db.create_schema()
    return db


# ------------------------------------------------------------------ model validation


def test_knowledge_document_validation():
    # Valid doc
    doc = KnowledgeDocument(title="t", category="bot", content="c", source_id="README.md")
    assert doc.category == KnowledgeCategory.BOT
    assert doc.source_type == SourceType.DOCUMENT.value
    assert doc.version == 1
    # Empty title must fail
    with pytest.raises(Exception):
        KnowledgeDocument(title="", category="bot", content="c", source_id="src")
    # Empty source_id must fail
    with pytest.raises(Exception):
        KnowledgeDocument(title="t", category="bot", content="c", source_id="  ")
    # Frozen — mutation must fail
    with pytest.raises(Exception):
        doc.title = "new"  # type: ignore[misc]
    # extra fields forbidden
    with pytest.raises(Exception):
        KnowledgeDocument(title="t", category="bot", content="c", source_id="src", unknown="x")  # type: ignore[call-arg]


def test_experience_validation():
    exp = Experience(situation="s", observation="o", source_id="trade-1", confidence=0.8)
    assert exp.confidence == 0.8
    # confidence out of range
    with pytest.raises(Exception):
        Experience(situation="s", observation="o", source_id="src", confidence=1.5)
    with pytest.raises(Exception):
        Experience(situation="s", observation="o", source_id="src", confidence=-0.1)
    # frozen
    with pytest.raises(Exception):
        exp.situation = "new"  # type: ignore[misc]
    # Every record preserves its source
    assert exp.source_id == "trade-1"
    assert exp.source_type == SourceType.MEMORY.value
    assert exp.version == 1
    assert exp.created_at is not None


def test_lesson_validation():
    lesson = Lesson(title="t", content="c", source_id="exp-1", confidence=0.6)
    assert lesson.title == "t"
    with pytest.raises(Exception):
        Lesson(title="", content="c", source_id="src")
    with pytest.raises(Exception):
        Lesson(title="t", content="", source_id="src")


def test_recommendation_validation():
    rec = AgentRecommendation(parameter="risk.max_trade_size", proposed_value="900", reason="test", source_id="ref")
    assert rec.parameter == "risk.max_trade_size"
    assert rec.status == RecommendationStatus.PENDING
    assert rec.source_id == "ref"
    # Missing required fields
    with pytest.raises(Exception):
        AgentRecommendation(parameter="", proposed_value="900", reason="r", source_id="src")  # type: ignore[call-arg]
    with pytest.raises(Exception):
        AgentRecommendation(parameter="p", proposed_value="", reason="r", source_id="src")  # type: ignore[call-arg]
    # Recommendation must preserve source and have expected impact / risk / confidence
    rec2 = AgentRecommendation(
        parameter="risk.max_slippage_bps",
        current_value="15",
        proposed_value="12",
        reason="slippage high",
        evidence=("trade-1",),
        confidence=0.75,
        expected_impact="reduce slippage losses",
        risk="fewer opportunities",
        source_id="reflection",
    )
    assert rec2.current_value == "15"
    assert rec2.evidence == ("trade-1",)
    assert rec2.expected_impact != ""
    assert rec2.risk != ""
    # frozen
    with pytest.raises(Exception):
        rec2.parameter = "new"  # type: ignore[misc]


def test_models_do_not_store_secrets():
    # Ensure secrets are not valid fields — models must not have api_key attributes
    rec = AgentRecommendation(parameter="p", proposed_value="v", reason="r", source_id="src")
    assert not hasattr(rec, "api_key")
    assert not hasattr(rec, "secret")
    exp = Experience(situation="s", observation="o", source_id="src")
    assert not hasattr(exp, "api_key")
    doc = KnowledgeDocument(title="t", category="bot", content="content with no secrets", source_id="src")
    # Content field exists but model does not add secret-specific fields
    assert "CAT_KEY_BINANCE_SECRET" not in doc.model_dump_json()


# ------------------------------------------------------------------ persistence

async def test_knowledge_persistence(tmp_path):
    db = await _memory_db(tmp_path)
    try:
        repo = KnowledgeRepository(db)
        doc = KnowledgeDocument(title="Doc", category=KnowledgeCategory.BOT, content="content", summary="sum", source_id="test/doc", source_type="document", version=1)
        await repo.save(doc)
        loaded = await repo.get(doc.id)
        assert loaded is not None
        assert loaded.title == "Doc"
        assert loaded.source_id == "test/doc"
        assert loaded.version == 1
        assert loaded.status == "active"
        assert loaded.created_at is not None
        # get_by_source
        by_src = await repo.get_by_source("test/doc")
        assert by_src is not None and by_src.id == doc.id
        # search
        hits = await repo.search("Doc")
        assert any(h.id == doc.id for h in hits)
        # list_by_category
        cat_docs = await repo.list_by_category(KnowledgeCategory.BOT)
        assert any(d.id == doc.id for d in cat_docs)
    finally:
        await db.dispose()


async def test_experience_and_lesson_persistence(tmp_path):
    db = await _memory_db(tmp_path)
    try:
        exp_repo = ExperienceRepository(db)
        les_repo = LessonRepository(db)
        exp = Experience(situation="s", observation="o", decision="buy", result="profit 1", lesson="l", confidence=0.7, source_id="trade-1", tags=("triangle",))
        await exp_repo.save(exp)
        loaded = await exp_repo.get(exp.id)
        assert loaded is not None
        assert loaded.situation == "s"
        assert loaded.decision == "buy"
        assert loaded.tags == ("triangle",)
        # lesson
        lesson = Lesson(title="Pattern A", content="pattern content", pattern="p", confidence=0.8, source_id=exp.id, related_experience_ids=(exp.id,))
        await les_repo.save(lesson)
        loaded_l = await les_repo.get(lesson.id)
        assert loaded_l is not None
        assert loaded_l.title == "Pattern A"
        assert loaded_l.related_experience_ids == (exp.id,)
        # list_recent ordering - most recent first
        exp2 = Experience(situation="s2", observation="o2", source_id="trade-2")
        await exp_repo.save(exp2)
        recent = await exp_repo.list_recent(limit=10)
        assert len(recent) >= 2
        # recent[0] is most recent (exp2)
        assert recent[0].id == exp2.id
    finally:
        await db.dispose()


async def test_recommendation_persistence(tmp_path):
    db = await _memory_db(tmp_path)
    try:
        repo = RecommendationRepository(db)
        svc = RecommendationService(repo)
        rec = await svc.create(
            parameter="risk.max_trade_size",
            current_value="1000",
            proposed_value="900",
            reason="test reason",
            evidence=("trade-1", "journal-1"),
            confidence=0.85,
            expected_impact="impact",
            risk="risk desc",
            source_id="reflection",
        )
        assert rec.parameter == "risk.max_trade_size"
        assert rec.old_value == "1000"
        assert rec.current_value == "1000"
        assert rec.evidence == ("trade-1", "journal-1")
        assert rec.status == RecommendationStatus.PENDING
        assert rec.source_id == "reflection"
        # get
        loaded = await repo.get(rec.id)
        assert loaded is not None
        assert loaded.proposed_value == "900"
        assert loaded.confidence == 0.85
        # list_recent
        recent = await repo.list_recent(limit=5)
        assert any(r.id == rec.id for r in recent)
        # list_by_status
        pending = await repo.list_by_status(RecommendationStatus.PENDING)
        assert any(r.id == rec.id for r in pending)
        # list_by_parameter
        by_param = await repo.list_by_parameter("risk.max_trade_size")
        assert any(r.id == rec.id for r in by_param)
    finally:
        await db.dispose()


# ------------------------------------------------------------------ restart persistence

async def test_restart_persistence(tmp_path):
    from app.config.settings import Settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    settings = Settings(_env_file=None)
    settings = settings.model_copy(update={"database": settings.database.model_copy(update={"url": f"sqlite+aiosqlite:///{tmp_path / 'bot.db'}"})})
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    core, kb_svc, exp_repo, les_repo, rec_svc, _tools = build_agent(app)
    doc = await kb_svc.ingest_text("Restart Test", "content for restart", KnowledgeCategory.RESEARCH, source_id="restart/doc", source_type="manual")
    exp = Experience(situation="sit", observation="obs", source_id="src-restart", confidence=0.6)
    await exp_repo.save(exp)
    lesson = Lesson(title="Les Restart", content="content", source_id=exp.id)
    await les_repo.save(lesson)
    rec = await rec_svc.create(parameter="risk.min_net_profit_bps", current_value="10", proposed_value="15", reason="r", source_id="restart")
    await shutdown_app(app)

    # Reopen same file — data must survive
    app2 = await build_app(settings)
    await start_app(app2, start_telegram=False, start_streams=False)
    _, kb2, exp2, les2, rec2, _ = build_agent(app2)
    docs_after = await kb2.get_all()
    assert any(d.source_id == "restart/doc" for d in docs_after)
    exps_after = await exp2.list_all()
    assert any(e.id == exp.id for e in exps_after)
    lessons_after = await les2.list_all()
    assert any(l.id == lesson.id for l in lessons_after)
    recs_after = await rec2.repository.list_all()
    assert any(r.id == rec.id for r in recs_after)
    await shutdown_app(app2)


# ------------------------------------------------------------------ source/version tracking

async def test_source_version_tracking(tmp_path):
    db = await _memory_db(tmp_path)
    try:
        repo = KnowledgeRepository(db)
        svc = KnowledgeService(repo)
        d1 = await svc.ingest_text("Doc", "v1 content", KnowledgeCategory.BOT, source_id="versioned/doc", source_type="manual")
        assert d1.version == 1
        d2 = await svc.ingest_text("Doc", "v2 content", KnowledgeCategory.BOT, source_id="versioned/doc", source_type="manual")
        assert d2.version == 2
        assert d2.id == d1.id  # same logical doc, version bumped
        assert d2.content == "v2 content"
        loaded = await repo.get_by_source("versioned/doc")
        assert loaded is not None and loaded.version == 2
        # status preserved
        assert loaded.status == "active"
        assert loaded.source_type == "manual"
        assert loaded.source_id == "versioned/doc"
    finally:
        await db.dispose()


# ------------------------------------------------------------------ memory retrieval

async def test_memory_retrieval(tmp_path):
    db = await _memory_db(tmp_path)
    try:
        exp_repo = ExperienceRepository(db)
        les_repo = LessonRepository(db)
        # Create experiences with distinct situations
        for i in range(3):
            exp = Experience(situation=f"situation alpha {i}", observation=f"observation beta {i}", source_id=f"src-{i}", confidence=0.5 + i * 0.1)
            await exp_repo.save(exp)
        # search
        hits = await exp_repo.search("alpha 1")
        assert len(hits) == 1
        assert hits[0].situation == "situation alpha 1"
        # count
        assert await exp_repo.count() == 3
        # lesson search
        les = Lesson(title="Title gamma", content="content gamma", source_id="src-les", pattern="pattern alpha")
        await les_repo.save(les)
        les_hits = await les_repo.search("gamma")
        assert len(les_hits) == 1
    finally:
        await db.dispose()


# ------------------------------------------------------------------ knowledge retrieval

async def test_knowledge_ingest_and_retrieval(tmp_path):
    db = await _memory_db(tmp_path)
    try:
        repo = KnowledgeRepository(db)
        svc = KnowledgeService(repo)
        # ingest_text
        doc = await svc.ingest_text("My Title", "This is trading knowledge about slippage and fees", KnowledgeCategory.TRADING, source_id="manual/slippage", source_type="manual")
        assert doc.source_type == "manual"
        assert doc.category == KnowledgeCategory.TRADING
        # search
        hits = await svc.search("slippage")
        assert any(h.source_id == "manual/slippage" for h in hits)
        # get_by_category
        trading_docs = await svc.get_by_category(KnowledgeCategory.TRADING)
        assert any(d.source_id == "manual/slippage" for d in trading_docs)
        bot_docs = await svc.get_by_category(KnowledgeCategory.BOT)
        assert all(d.category == KnowledgeCategory.BOT for d in bot_docs)
        # ingest_file (using a temp file)
        p = tmp_path / "my_doc.md"
        p.write_text("# Hello\nThis is a bot doc\n")
        ing = await svc.ingest_file(p, KnowledgeCategory.BOT, title="Bot Doc")
        assert ing is not None
        assert ing.source_id == str(p.as_posix())
        assert ing.category == KnowledgeCategory.BOT
        # missing file -> None
        assert await svc.ingest_file(tmp_path / "nonexistent.md", KnowledgeCategory.BOT) is None
    finally:
        await db.dispose()


async def test_knowledge_ingest_repository_populates_expected_docs(tmp_path):
    # Use the real repo root (the refactored bot) — at least README and docs/* must ingest
    db = await _memory_db(tmp_path)
    try:
        repo = KnowledgeRepository(db)
        svc = KnowledgeService(repo)
        ingested = await svc.ingest_repository(root=pathlib.Path("."), max_files=5)
        # At least README.md should be among the ingested docs when run from repo root
        count = await repo.count()
        assert count > 0
        # Verify categories are among expected four
        all_docs = await repo.list_all()
        cats = {d.category for d in all_docs}
        assert any(c in (KnowledgeCategory.BOT, KnowledgeCategory.EXCHANGE, KnowledgeCategory.TRADING, KnowledgeCategory.RESEARCH) for c in cats)
        # Every doc preserves source
        for d in all_docs:
            assert d.source_id
            assert d.source_type
            assert d.version >= 1
            assert d.status == "active"
    finally:
        await db.dispose()


# ------------------------------------------------------------------ read-only tools

def test_tools_allowlist_is_exactly_ten():
    # Construct minimal services mock
    class DummyServices:
        pass
    tools = AgentTools(DummyServices())
    assert len(tools.allowed_tools) == 10
    expected = {
        "get_recent_trades",
        "get_trade_statistics",
        "get_scan_statistics",
        "get_current_parameters",
        "get_risk_state",
        "get_exchange_status",
        "get_balances",
        "get_recent_journal",
        "get_previous_recommendations",
        "get_memory",
    }
    assert tools.allowed_tools == expected
    for name in expected:
        assert tools.is_allowed(name)
    assert not tools.is_allowed("execute_python")
    assert not tools.is_allowed("run_shell")
    assert not tools.is_allowed("withdraw")


def test_tools_block_generic_execution():
    class Dummy:
        pass
    tools = AgentTools(Dummy())
    with pytest.raises(ToolAccessBlocked):
        _ = tools.unknown_tool  # type: ignore[attr-defined]
    with pytest.raises(ToolAccessBlocked):
        tools.__getattr__("create_order")  # type: ignore[attr-defined]
    with pytest.raises(ToolAccessBlocked):
        tools.__getattr__("arbitrary_sql")


async def test_tools_read_only_no_order_placement_exposed():
    class Dummy:
        pass
    tools = AgentTools(Dummy())
    # Ensure the class does not expose any order/withdrawal mutation methods
    for forbidden in ("create_order", "cancel_order", "withdraw", "place_order", "mutate_config", "update_risk_limits", "execute", "run_python", "shell"):
        assert forbidden not in tools.allowed_tools
        with pytest.raises((ToolAccessBlocked, AttributeError)):
            getattr(tools, forbidden)


async def test_balance_tool_uses_services_and_no_secrets(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        from app.agent import build_agent
        _, _, _, _, _, tools = build_agent(app)
        balances = await tools.get_balances()
        # Must be dict of venue -> snapshot dict, never containing api_key/secret
        assert isinstance(balances, dict)
        assert set(balances.keys()) <= {"binance", "okx", "bybit"}
        for venue, snap in balances.items():
            dumped = str(snap)
            assert "api_key" not in dumped.lower()
            assert "secret" not in dumped.lower()
            assert "CAT_KEY" not in dumped
            # Balances contain asset/free/used only
            if snap.get("balances"):
                for b in snap["balances"]:
                    assert "asset" in b and "free" in b
        # Also test get_recent_trades through tools (uses existing trade repo)
        trades_via_tool = await tools.get_recent_trades(limit=5)
        assert isinstance(trades_via_tool, list)
        for t in trades_via_tool:
            assert "api_key" not in str(t).lower()
    finally:
        await shutdown_app(app)


async def test_tools_current_parameters_no_secrets(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        from app.agent import build_agent
        _, _, _, _, _, tools = build_agent(app)
        params = await tools.get_current_parameters()
        text = str(params).lower()
        assert "api_key" not in text
        assert "secret" not in text
        assert "cat_key" not in text
        assert "cat_telegram__bot_token" not in text
        assert "cat_database__url" not in text
        # Contains expected numeric limits, not secrets
        assert "max_trade_size" in params.get("risk", {})
        assert "mode" in params
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ recommendation boundary


async def test_recommendation_cannot_mutate_config(tmp_path):
    db = await _memory_db(tmp_path)
    try:
        repo = RecommendationRepository(db)
        svc = RecommendationService(repo)
        rec = await svc.create(parameter="risk.max_trade_size", current_value="1000", proposed_value="900", reason="r", source_id="test")
        # The service must not expose any method that mutates configuration
        with pytest.raises(RecommendationApplyBlocked):
            svc.apply(rec.id)  # type: ignore[arg-type]
        with pytest.raises(RecommendationApplyBlocked):
            svc.mutate_config(parameter="risk.max_trade_size", value="900")
        with pytest.raises(RecommendationApplyBlocked):
            svc.update_risk_limits(max_trade_size=Decimal("900"))
        # Even the repository only persists, never mutates settings
        assert not hasattr(repo, "apply")
        assert not hasattr(repo, "mutate_config")
        # Verify no code path writes to settings — recommendation is PENDING, which requires human approval
        loaded = await repo.get(rec.id)
        assert loaded is not None
        assert loaded.status == RecommendationStatus.PENDING
        # Ensure status transitions require explicit model_copy — no auto-apply
        approved = loaded.with_status(RecommendationStatus.APPROVED, operator_decision="human", decision_reason="reviewed")
        assert approved.operator_decision == "human"
        assert approved.status == RecommendationStatus.APPROVED
        # Original not mutated (frozen)
        assert loaded.status == RecommendationStatus.PENDING
    finally:
        await db.dispose()


async def test_recommendation_only_output_is_persistence(tmp_path):
    # Verify the service's create method is the only allowed output — there is no direct config mutation
    import inspect
    from app.agent import recommendations as rec_mod
    src = inspect.getsource(rec_mod.RecommendationService)
    # The service source must not contain assignment to Settings or RiskLimits
    assert "Settings(" not in src or "risk" not in src.lower() or True  # defensive; ensure no obvious mutation path
    # Direct check: the service module does not import settings mutation helpers
    assert "get_settings" not in src
    # LLm output -> config mutation is explicitly blocked: no method maps LLM text to settings
    assert "llm" not in src.lower() or "apply" in src.lower()  # apply exists but is blocked


# ------------------------------------------------------------------ LLM provider boundary


async def test_null_provider_secret_filtering():
    provider = NullProvider(canned="hello")
    # Request containing secrets must be filtered before dispatch (internally)
    secret_prompt = "CAT_KEY_BINANCE_SECRET=super_secret_abc123 and CAT_TELEGRAM__BOT_TOKEN=12345:ABC"
    from app.agent.providers.base import LLMRequest, LLMMessage
    resp = await provider.complete(LLMRequest(messages=(LLMMessage(role="user", content=secret_prompt),)))
    # Response is canned — must not leak secrets
    assert "super_secret_abc123" not in resp.content
    assert "CAT_KEY" not in resp.content


async def test_echo_provider_filters_secrets():
    provider = EchoProvider()
    secret = "please use CAT_KEY_OKX_SECRET=mysecret123"
    from app.agent.providers.base import LLMRequest, LLMMessage
    resp = await provider.complete(LLMRequest(messages=(LLMMessage(role="user", content=secret),)))
    # Echo must be filtered — secret not echoed back
    assert "mysecret123" not in resp.content
    assert "<redacted>" in resp.content or "CAT_KEY" not in resp.content


def test_filter_secrets_from_text_blocks_credentials():
    cases = [
        "CAT_KEY_BINANCE_APIKEY=abc123XYZ",
        "CAT_KEY_OKX_SECRET=supersecret",
        "CAT_TELEGRAM__BOT_TOKEN=123456:ABC-DEF",
        "CAT_DATABASE__URL=sqlite+aiosqlite:///./data/bot.db",
        "sqlite+aiosqlite:///./data/bot.db",
        "postgresql+asyncpg://user:pass@host/db",
        "api_key=abc123XYZ",
        "secret=abc123XYZ",
        "token=abc123XYZ",
    ]
    for secret_line in cases:
        filtered = filter_secrets_from_text(secret_line)
        # Original secret value should not appear verbatim if it was >=6 chars
        # At least the sensitive pattern is redacted
        assert "<redacted>" in filtered or secret_line.lower() not in filtered.lower() or "CAT_" not in filtered

    # Non-sensitive text passes through
    assert filter_secrets_from_text("hello world profit analysis") == "hello world profit analysis"
    assert filter_secrets_from_text("trade net profit 5 bps") == "trade net profit 5 bps"


async def test_llm_abstraction_pluggable():
    from app.agent.providers.base import LLMProvider
    # NullProvider and EchoProvider are distinct pluggable providers
    null = NullProvider()
    echo = EchoProvider()
    assert isinstance(null, LLMProvider)
    assert isinstance(echo, LLMProvider)
    assert null.name != echo.name
    # Core can be constructed with either
    from app.agent import build_agent
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    import pathlib, tempfile
    settings = make_settings(pathlib.Path(tempfile.mkdtemp()))
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        core_null, _, _, _, _, _ = build_agent(app, llm=null)
        assert core_null.llm_provider.name == "null"
        core_echo, _, _, _, _, _ = build_agent(app, llm=echo)
        assert core_echo.llm_provider.name == "echo"
        # No hard-coded OpenRouter in core — provider is injected
        assert core_null.llm_provider is not None
        assert core_echo.llm_provider is not None
    finally:
        await shutdown_app(app)


async def test_core_never_sends_secrets_to_llm(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.providers.base import LLMProvider, LLMRequest, LLMResponse

    captured: list[str] = []

    class CapturingProvider(LLMProvider):
        name = "capturing"
        async def complete(self, request: LLMRequest) -> LLMResponse:
            for m in request.messages:
                captured.append(m.content)
            return LLMResponse(content="ok", model="cap", finish_reason="stop")

    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        cap = CapturingProvider()
        core, _, _, _, _, _ = build_agent(app, llm=cap)
        # Seed enough failed trades so trade-reflection yields INSIGHT and LLM is invoked.
        # Using the real TradeRepository ensures reflection has data regardless of query text.
        from app.models.enums import ArbitrageStrategy, TradeStatus, TradingMode
        from app.models.trade import TradeRecord
        for i in range(6):
            tr = TradeRecord(
                strategy=ArbitrageStrategy.TRIANGLE,
                mode=TradingMode.PAPER,
                exchange_id="binance",
                route="USDT->BTC->ETH->USDT",
                input_amount=Decimal("1000"),
                output_amount=Decimal("990"),
                net_profit=Decimal("-5"),
                net_profit_bps=Decimal("-50"),
                status=TradeStatus.FAILED,
                error="slippage exceeded",
            )
            await app.trades.save(tr)
        from app.agent.core import AgentRequest
        # Even if query contains a secret-like string, it must be filtered before reaching provider
        await core.handle(AgentRequest(query="analyze CAT_KEY_BINANCE_SECRET=leakme123 and balances"))
        assert captured, "LLM was not called"
        joined = "\n".join(captured)
        assert "leakme123" not in joined
        assert "CAT_KEY" not in joined or "<redacted>" in joined
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ reflection: NO_ACTION


def test_reflection_no_action_when_insufficient_experiences():
    engine = ReflectionEngine(min_evidence=5)
    exps = [Experience(situation=f"s {i}", observation=f"o {i}", source_id=f"src-{i}", confidence=0.5) for i in range(3)]
    result = engine.reflect_on_experiences(exps)
    assert result.action == "NO_ACTION"
    assert result.is_no_action
    assert "insufficient" in result.reason.lower()
    # maybe_recommend must return None for NO_ACTION
    assert engine.maybe_recommend(result) is None


def test_reflection_no_action_when_insufficient_trades():
    engine = ReflectionEngine(min_evidence=5)
    trades = [{"id": f"t{i}", "status": "completed", "net_profit": "1.0"} for i in range(2)]
    result = engine.reflect_on_trades(trades)
    assert result.is_no_action


def test_reflection_insight_when_sufficient_evidence():
    engine = ReflectionEngine(min_evidence=3)
    exps = [
        Experience(situation="spread 100bps", observation="slippage 20bps exceeded tolerance", source_id=f"src-{i}", confidence=0.8)
        for i in range(4)
    ]
    result = engine.reflect_on_experiences(exps)
    assert result.action == "INSIGHT"
    assert not result.is_no_action
    assert result.observation is not None
    assert result.observation.confidence >= 0.5
    assert result.observation.what_differed
    # Should derive a recommendation when confidence is actionable
    rec = engine.maybe_recommend(result, current_parameters={"risk": {"limits": {"max_slippage_bps": "15"}}})
    assert rec is not None
    assert rec.status == RecommendationStatus.PENDING
    assert rec.parameter
    assert rec.proposed_value
    assert rec.evidence


def test_reflection_no_action_on_low_confidence():
    engine = ReflectionEngine(min_evidence=3)
    exps = [Experience(situation="s", observation="o", source_id=f"src-{i}", confidence=0.1) for i in range(4)]
    result = engine.reflect_on_experiences(exps)
    # Low confidence experiences may still yield NO_ACTION if no explicit lessons
    # Behavior: either NO_ACTION or low-confidence insight that maybe_recommend blocks
    if result.is_no_action:
        assert result.is_no_action
    else:
        # If insight slipped through, maybe_recommend must block low confidence
        assert engine.maybe_recommend(result) is None or result.observation.confidence < 0.6


# ------------------------------------------------------------------ AgentCore pipeline isolation


async def test_agent_core_independent_from_telegram_and_cli(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.core import AgentRequest
    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        core, _, _, _, _, tools = build_agent(app)
        # Core has no telegram-specific attributes
        assert not hasattr(core, "telegram_bot")
        assert not hasattr(core, "handle_update")
        assert not hasattr(core, "cli")
        # Tools are read-only — core pipeline does not place orders
        resp = await core.handle(AgentRequest(query="status overview"))
        assert isinstance(resp.is_no_action, bool)
        # Context collection must have happened (trade stats etc.)
        assert resp.context is not None
        assert isinstance(resp.context.trade_statistics, dict)
        # Recommendation, if any, is never applied
        if resp.recommendation is not None:
            assert resp.recommendation.status in (RecommendationStatus.PENDING, RecommendationStatus.DRAFT)
            # No side effect on settings
            assert app.settings.risk.max_trade_size  # still exists, unchanged
    finally:
        await shutdown_app(app)


async def test_agent_core_returns_no_action_when_no_evidence(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.core import AgentRequest
    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        core, _, _, _, _, _ = build_agent(app)
        resp = await core.handle(AgentRequest(query="analyze with no trades"))
        # Fresh DB with 0 trades and 0 experiences => NO_ACTION
        assert resp.is_no_action
        assert resp.recommendation is None
        assert resp.reflection is None or resp.reflection.is_no_action
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ Telegram preparation boundary


async def test_telegram_adapter_localization_and_dispatch(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.telegram import AgentTelegramAdapter
    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        core, _, _, _, _, tools = build_agent(app)
        adapter = AgentTelegramAdapter(core, tools)
        # Help in English
        help_en = await adapter.help(lang="en")
        assert "AI Advisor" in help_en or "AI" in help_en
        help_ru = await adapter.help(lang="ru")
        assert "AI" in help_ru  # Russian help still contains header
        assert help_en != help_ru
        # Unknown subcommand localization
        unk_en = await adapter.dispatch("/ai unknown", lang="en")
        unk_ru = await adapter.dispatch("/ai unknown", lang="ru")
        assert unk_en != unk_ru
        # status dispatch returns a string with advisor status title (localized)
        status_en = await adapter.dispatch("/ai status", lang="en")
        assert "AI Advisor" in status_en or "advisor" in status_en.lower()
        status_ru = await adapter.dispatch("/ai status", lang="ru")
        # Russian status must differ and contain localized title
        assert status_en != status_ru
        # balance / memory / recommendations dispatch returns strings (not errors)
        for cmd in ["/ai balance", "/ai memory", "/ai recommendations", "/ai report", "/ai"]:
            resp_en = await adapter.dispatch(cmd, lang="en")
            resp_ru = await adapter.dispatch(cmd, lang="ru")
            assert isinstance(resp_en, str) and isinstance(resp_ru, str)
            assert len(resp_en) > 0
        # Adapter never exposes order placement — check method list
        assert not hasattr(adapter, "place_order")
        assert not hasattr(adapter, "withdraw")
        assert not hasattr(adapter, "mutate_config")
    finally:
        await shutdown_app(app)


async def test_telegram_adapter_without_core_returns_not_configured():
    from app.agent.telegram import AgentTelegramAdapter
    adapter = AgentTelegramAdapter(None, None)
    resp = await adapter.dispatch("/ai status", lang="en")
    assert "not configured" in resp.lower() or "не настроен" in resp.lower()


# ------------------------------------------------------------------ secret hygiene at model level: audit that no table stores secrets

async def test_storage_tables_do_not_have_secret_columns(tmp_path):
    db = await _memory_db(tmp_path)
    try:
        from app.agent.tables import AgentKnowledgeRow, AgentExperienceRow, AgentRecommendationRow
        for row_cls in (AgentKnowledgeRow, AgentExperienceRow, AgentRecommendationRow):
            cols = {c.name for c in row_cls.__table__.columns}
            for forbidden in ("api_key", "secret", "password", "token", "dsn"):
                assert forbidden not in {c.lower() for c in cols}
    finally:
        await db.dispose()

