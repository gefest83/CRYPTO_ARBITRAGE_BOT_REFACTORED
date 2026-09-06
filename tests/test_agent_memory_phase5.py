"""Phase 5 — Memory / Reflection focused tests.

Covers: experience extraction (+missing-value explicitness), persistence and
restart, FACT/OBSERVATION/HYPOTHESIS/RECOMMENDATION separation (no LLM
auto-promotion), confidence, sample size, provenance, freshness/decay,
contradiction detection (both retained), versioning (auditable history),
post-trade / N-trade / daily / weekly reflection, retrieval + context
integration (memory distinct from journal facts), insufficient-data
handling, security boundaries, LLM failure isolation.

Trading/execution/risk/recovery behavior is never modified.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.agent.extraction import (
    aggregate_to_lesson,
    compute_lesson_confidence,
    detect_contradiction,
    extract_experience,
    outcome_direction,
    record_llm_hypothesis,
    validate_experience,
)
from app.agent.journal import MIN_SAMPLE_FOR_CONCLUSIONS, sanitize_trade
from app.agent.memory import (
    MemoryLabelRepository,
    decayed_confidence,
    freshness_weight,
    memory_age_days,
    memory_freshness,
    revise_lesson,
)
from app.agent.models import Experience, LearningType, Lesson, SourceType
from app.models.enums import ArbitrageStrategy, TradeStatus, TradingMode
from app.models.trade import TradeRecord


def _trade_view(**overrides) -> dict:
    trade = TradeRecord(
        strategy=overrides.get("strategy", ArbitrageStrategy.TRIANGLE),
        mode=TradingMode.PAPER,
        exchange_id=overrides.get("exchange", "binance"),
        route=overrides.get("route", "USDT->BTC->ETH->USDT"),
        input_amount=Decimal("1000"),
        output_amount=Decimal("1005"),
        fees_quote=overrides.get("fees", Decimal("1")),
        slippage_bps=overrides.get("slippage", Decimal("10")),
        net_profit=overrides.get("net", Decimal("5")),
        net_profit_bps=overrides.get("bps", Decimal("50")),
        status=overrides.get("status", TradeStatus.COMPLETED),
        orders=overrides.get("orders", ()),
        error=overrides.get("error"),
        transfer_id=overrides.get("transfer_id"),
    )
    return sanitize_trade(trade)


def _exp(**overrides) -> Experience:
    base = {"situation": "s", "observation": "o", "source_id": "trd-abc123456789",
            "evidence": ("trd-abc123456789",), "confidence": 0.7}
    base.update(overrides)
    return Experience(**base)


# ------------------------------------------------------------------ extraction


def test_phase5_extraction_preserves_journal_fields():
    view = _trade_view()
    exp = extract_experience(view)
    assert exp.source_id == view["id"]
    assert exp.evidence == (view["id"],)
    assert exp.source_type == SourceType.TRADE.value
    assert "binance" in exp.situation and "triangle" in exp.situation
    assert "5" in exp.situation and "USDT->BTC->ETH->USDT" in exp.situation
    assert "fees=1" in exp.situation and "slippage=10" in exp.situation
    assert "completed" in exp.observation and "completed" in (exp.result or "")
    assert exp.confidence == 0.7
    assert validate_experience(exp) == []


def test_phase5_extraction_missing_values_explicit():
    view = _trade_view()
    view["transfer_id"] = None
    exp = extract_experience(view, transfer_plan=None, market_conditions=None)
    assert "n/a" in exp.situation  # expected edge + market conditions
    assert "no legs journaled" in exp.observation
    assert "failure=none" in exp.observation
    # Failed trade keeps its reason and lower confidence.
    failed = _trade_view(status=TradeStatus.FAILED, error="venue timeout", net=Decimal("-7"))
    exp_f = extract_experience(failed)
    assert "venue timeout" in exp_f.observation
    assert exp_f.confidence == 0.6
    # Transfer plan feeds the expected edge explicitly.
    plan = {"net_profit_quote": "20.94", "buy_price": "3000", "sell_price": "3030", "amount": "1"}
    exp_p = extract_experience(view, transfer_plan=plan)
    assert "20.94" in exp_p.situation


def test_phase5_validation():
    assert validate_experience(_exp()) == []
    broken = Experience.model_construct(situation="  ", observation="o", source_id="x",
                                        evidence=(), confidence=0.7)
    errors = validate_experience(broken)
    assert any("situation" in e for e in errors)
    assert any("evidence" in e for e in errors)


# ------------------------------------------------------------------ types: no LLM promotion


def test_phase5_learning_types_exist():
    assert {t.value for t in LearningType} == {"fact", "observation", "hypothesis", "recommendation"}


@pytest.mark.asyncio
async def test_phase5_hypothesis_never_becomes_fact(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        n_exp_before = await app.agent_experiences.count()
        result = await record_llm_hypothesis(app.agent_audit, "venue X is rigged", source="test", query="q")
        assert result["learning_type"] == "hypothesis"
        # Audit-only sink: memory tables untouched.
        assert await app.agent_experiences.count() == n_exp_before
        events = await app.agent_audit.list_recent(limit=5)
        assert any(e.event_type == "llm_hypothesis" for e in events)
        # No promotion path exists anywhere in the agent package.
        import pathlib as _pl

        for path in _pl.Path("app/agent").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "promote_to_fact" not in text
            assert "hypothesis_to_fact" not in text
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ confidence + sample size


def test_phase5_lesson_confidence_and_sample_gate():
    exps = [_exp(confidence=0.8) for _ in range(6)]
    assert compute_lesson_confidence(exps) == pytest.approx(0.8)
    assert aggregate_to_lesson("failure", exps, pattern="p") is not None
    lesson = aggregate_to_lesson("failure", exps, pattern="p")
    assert lesson is not None and len(lesson.related_experience_ids) == 6
    assert lesson.confidence <= 0.9
    # Insufficient sample → explicit None + capped confidence.
    small = [_exp(confidence=0.9) for _ in range(3)]
    assert aggregate_to_lesson("failure", small, pattern="p") is None
    assert compute_lesson_confidence(small) <= 0.44
    assert compute_lesson_confidence([]) == 0.0
    assert MIN_SAMPLE_FOR_CONCLUSIONS == 5


@pytest.mark.asyncio
async def test_phase5_persistence_and_restart(tmp_path):
    from app.config.settings import DatabaseSettings
    from app.storage.engine import Database
    from app.agent.memory import ExperienceRepository, LessonRepository, FeedbackRepository

    url = f"sqlite+aiosqlite:///{tmp_path / 'mem5.db'}"
    db = Database(DatabaseSettings(url=url))
    await db.create_schema()
    try:
        exp_repo, les_repo, fdb_repo = ExperienceRepository(db), LessonRepository(db), FeedbackRepository(db)
        label_repo = MemoryLabelRepository(db)
        exp = await exp_repo.save(_exp(source_id="trd-restart00"))
        les = aggregate_to_lesson("profit", [_exp(source_id=f"trd-{i:012d}") for i in range(5)], pattern="p")
        assert les is not None
        await les_repo.save(les)
        await label_repo.set_label(target_type="experience", target_id=exp.id,
                                   learning_type=LearningType.FACT, sample_size=1, direction="positive")
        await fdb_repo.submit(target_type="lesson", target_id=les.id, rating=1,
                              comment="useful", approver="human")
        exp_id, les_id = exp.id, les.id
    finally:
        await db.dispose()
    db2 = Database(DatabaseSettings(url=url))
    await db2.create_schema()
    try:
        exp_repo2, les_repo2 = ExperienceRepository(db2), LessonRepository(db2)
        label_repo2, fdb_repo2 = MemoryLabelRepository(db2), FeedbackRepository(db2)
        assert (await exp_repo2.get(exp_id)) is not None
        assert (await les_repo2.get(les_id)) is not None
        label = await label_repo2.get_label("experience", exp_id)
        assert label is not None and label["learning_type"] == "fact" and label["sample_size"] == 1
        score = await fdb_repo2.score_for_target("lesson", les_id)
        assert score == {"n": 1, "score": 1, "up": 1, "down": 0}
    finally:
        await db2.dispose()


# ------------------------------------------------------------------ freshness / decay


def test_phase5_freshness_decay_math():
    assert freshness_weight(0.0) == 1.0
    assert freshness_weight(30.0) == pytest.approx(0.5)
    assert freshness_weight(60.0) == pytest.approx(0.25)
    assert decayed_confidence(0.8, 30.0) == pytest.approx(0.4)
    assert memory_age_days(None) == 0.0
    bundle = memory_freshness(datetime.now(UTC), 0.8)
    assert bundle["weight"] == pytest.approx(1.0, abs=0.01)
    assert set(bundle) == {"age_days", "weight", "confidence", "decayed_confidence"}
    old = memory_freshness(datetime.now(UTC) - timedelta(days=60), 0.8)
    assert old["weight"] == pytest.approx(0.25, abs=0.01)


@pytest.mark.asyncio
async def test_phase5_decay_never_deletes(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        from app.agent import build_agent

        build_agent(app)
        old = Experience(situation="old s", observation="old o", source_id="trd-old000001",
                         confidence=0.9,
                         created_at=datetime.now(UTC) - timedelta(days=90),
                         updated_at=datetime.now(UTC) - timedelta(days=90))
        await app.agent_experiences.save(old)
        n_before = await app.agent_experiences.count()
        # Decay only re-ranks retrieval; the record survives.
        assert decayed_confidence(0.9, 90.0) < 0.9
        assert await app.agent_experiences.count() == n_before
        assert await app.agent_experiences.get(old.id) is not None
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ contradiction + versioning


def test_phase5_contradiction_detection_pure():
    conflict = detect_contradiction(
        existing_direction="positive", new_avg_net_bps="-30", new_fail_rate=0.7,
        existing_lesson_id="les-old", new_evidence_ids=["trd-1"],
    )
    assert conflict is not None
    assert conflict["resolution"] == "retained_both"
    assert conflict["existing_lesson_id"] == "les-old"
    assert conflict["existing_direction"] == "positive"
    assert conflict["new_direction"] == "negative"
    conflict2 = detect_contradiction(
        existing_direction="negative", new_avg_net_bps="60", new_fail_rate=0.0,
        existing_lesson_id="les-old", new_evidence_ids=["trd-2"],
    )
    assert conflict2 is not None and conflict2["new_direction"] == "positive"
    # Same direction or mixed → no conflict.
    assert detect_contradiction(existing_direction="negative", new_avg_net_bps="-10",
                                new_fail_rate=0.6, existing_lesson_id="x", new_evidence_ids=[]) is None
    assert detect_contradiction(existing_direction="positive", new_avg_net_bps="10",
                                new_fail_rate=0.4, existing_lesson_id="x", new_evidence_ids=[]) is None
    assert outcome_direction("-5", 0.1) == "negative"
    assert outcome_direction("10", 0.1) == "positive"
    assert outcome_direction("10", 0.4) == "mixed"


@pytest.mark.asyncio
async def test_phase5_contradiction_persisted_both_retained(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        old = Lesson(title="Pattern: failure", content="venues fail", pattern="p",
                     confidence=0.7, evidence=("trd-old000001",),
                     related_experience_ids=("exp-old000001",), source_id="phase5:test")
        await app.agent_lessons.save(old)
        await app.agent_memory_labels.set_label(target_type="lesson", target_id=old.id,
                                                learning_type=LearningType.OBSERVATION,
                                                sample_size=6, direction="negative")
        # New opposing evidence → conflict persisted, nothing overwritten.
        conflict = detect_contradiction(
            existing_direction="negative", new_avg_net_bps="60", new_fail_rate=0.0,
            existing_lesson_id=old.id, new_evidence_ids=["trd-new000001"],
        )
        assert conflict is not None
        from app.agent.audit import AgentAuditEvent

        await app.agent_audit.log(AgentAuditEvent(
            event_type="contradiction", evidence_count=1, action="CONFLICT",
            details=conflict, source_type="reflection", source_id="test",
        ))
        new = Lesson(title="Pattern: failure", content="venues recovered", pattern="p2",
                     confidence=0.65, evidence=("trd-new000001",),
                     related_experience_ids=("exp-new000001",), source_id="phase5:test")
        await app.agent_lessons.save(new)
        assert (await app.agent_lessons.get(old.id)) is not None
        assert (await app.agent_lessons.get(new.id)) is not None
        events = await app.agent_audit.list_recent(limit=10)
        stored = [e for e in events if e.event_type == "contradiction"]
        assert stored and stored[0].details["existing_lesson_id"] == old.id
        assert stored[0].details["resolution"] == "retained_both"
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase5_versioning(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        les = Lesson(title="Pattern: profit", content="v1", pattern="p",
                     confidence=0.6, evidence=("trd-0000000001",),
                     related_experience_ids=("exp-0000000001",), source_id="phase5:test")
        await app.agent_lessons.save(les)
        updated = await revise_lesson(
            app.agent_lessons, app.agent_lesson_history, les.id,
            new_evidence_ids=["trd-0000000002"], new_experience_ids=["exp-0000000002"],
            reason="new week of evidence",
        )
        assert updated.version == 2
        assert set(updated.evidence) == {"trd-0000000001", "trd-0000000002"}
        versions = await app.agent_lesson_history.get_versions(les.id)
        assert len(versions) == 1
        assert versions[0]["version"] == 1
        assert versions[0]["snapshot"]["content"] == "v1"
        assert versions[0]["superseded_by"] == f"{les.id}@v2"
        assert "week" in versions[0]["reason"]
        with pytest.raises(KeyError):
            await revise_lesson(app.agent_lessons, app.agent_lesson_history, "les-missing")
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ reflection triggers


async def _seed_trades(app, n, *, status=TradeStatus.COMPLETED, net=Decimal("5"), exchange="binance"):
    ids = []
    for _ in range(n):
        t = await app.trades.save(TradeRecord(
            strategy=ArbitrageStrategy.TRIANGLE, mode=TradingMode.PAPER, exchange_id=exchange,
            route="USDT->BTC->ETH->USDT", input_amount=Decimal("1000"),
            output_amount=Decimal("1000") + net, fees_quote=Decimal("1"),
            slippage_bps=Decimal("10"), net_profit=net,
            net_profit_bps=Decimal("50") if net > 0 else Decimal("-50"),
            status=status, error=None if status == TradeStatus.COMPLETED else "timeout",
        ))
        ids.append(t.id)
    return ids


@pytest.mark.asyncio
async def test_phase5_post_trade_reflection(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        await _seed_trades(app, 3)
        report = await app.agent_reflection.run_due(app)
        assert "post_trade" in report["triggered"]
        assert len(report["experiences_created"]) == 3
        assert await app.agent_experiences.count() == 3
        # Restart-safe cursor: second run finds nothing new.
        report2 = await app.agent_reflection.run_due(app)
        assert "post_trade" not in report2["triggered"]
        assert await app.agent_experiences.count() == 3
        # Labels carry FACT type + provenance for journal-derived items.
        label = await app.agent_memory_labels.get_label("experience", report["experiences_created"][0])
        assert label is not None and label["learning_type"] == "fact" and label["sample_size"] == 1
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase5_n_trade_reflection(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    base = make_settings(tmp_path)
    settings = base.model_copy(update={"agent": base.agent.model_copy(update={"reflection_n_trades": 6})})
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        # Below threshold → skipped with reason.
        await _seed_trades(app, 3)
        report = await app.agent_reflection.run_due(app)
        assert "n_trade" not in report["triggered"]
        assert len(await app.agent_lessons.list_all()) == 0
        # At threshold → aggregate lesson from validated evidence.
        await _seed_trades(app, 3)
        report2 = await app.agent_reflection.run_due(app)
        assert "n_trade" in report2["triggered"]
        assert report2["lessons_created"]
        lesson = await app.agent_lessons.get(report2["lessons_created"][0])
        # The engine caps lesson evidence at its top-5 ids by design.
        assert lesson is not None and 1 <= len(lesson.related_experience_ids) <= 6
        label = await app.agent_memory_labels.get_label("lesson", lesson.id)
        assert label is not None and label["sample_size"] == 6
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase5_daily_reflection(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        await _seed_trades(app, 6, status=TradeStatus.FAILED, net=Decimal("-5"), exchange="okx")
        await app.agent_reflection.run_due(app)  # post-trade extraction
        # Force the daily trigger due (last run long ago).
        await app.bot_state.set("agent_reflection_last_daily",
                                (datetime.now(UTC) - timedelta(hours=30)).isoformat())
        report = await app.agent_reflection.run_due(app)
        assert "daily" in report["triggered"]
        assert report["lessons_created"]
        # Fresh cursor → skipped.
        report2 = await app.agent_reflection.run_due(app)
        assert "daily" not in report2["triggered"]
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase5_weekly_reflection(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        await _seed_trades(app, 6, status=TradeStatus.FAILED, net=Decimal("-5"))
        await app.agent_reflection.run_due(app)
        # Disable daily to isolate the weekly trigger.
        app.settings = app.settings.model_copy(update={
            "agent": app.settings.agent.model_copy(update={"reflection_daily_enabled": False})})
        await app.bot_state.set("agent_reflection_last_weekly",
                                (datetime.now(UTC) - timedelta(hours=200)).isoformat())
        report = await app.agent_reflection.run_due(app)
        assert "weekly" in report["triggered"]
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase5_reflection_configurable_and_llm_free(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.providers.base import LLMProvider, LLMRequest, LLMResponse

    class CrashLLM(LLMProvider):
        name = "crash"

        async def complete(self, request: LLMRequest) -> LLMResponse:
            raise AssertionError("scheduler must never call the LLM")

    base = make_settings(tmp_path)
    settings = base.model_copy(update={"agent": base.agent.model_copy(update={
        "reflection_n_trades": 4, "reflection_daily_hours": 12.0, "reflection_weekly_hours": 100.0,
    })})
    assert settings.agent.reflection_n_trades == 4
    assert settings.agent.reflection_daily_hours == 12.0
    assert settings.agent.reflection_weekly_hours == 100.0
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app, llm=CrashLLM())
        assert not hasattr(app.agent_reflection, "_llm")
        await _seed_trades(app, 4)
        report = await app.agent_reflection.run_due(app)  # would raise via CrashLLM if LLM were used
        assert "post_trade" in report["triggered"]
        # Disabled master switch short-circuits everything.
        app.settings = app.settings.model_copy(update={
            "agent": app.settings.agent.model_copy(update={"reflection_enabled": False})})
        report2 = await app.agent_reflection.run_due(app)
        assert report2["triggered"] == [] and report2["skipped"]["all"] == "reflection_enabled=false"
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ retrieval / context


@pytest.mark.asyncio
async def test_phase5_memory_retrieval_context(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.context import ContextCollector
    from app.agent.tools import AgentTools

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        fresh = await app.agent_experiences.save(_exp(source_id="trd-fresh00001",
                                                             situation="fresh slippage s",
                                                             observation="fresh slippage o"))
        await app.agent_memory_labels.set_label(target_type="experience", target_id=fresh.id,
                                                learning_type=LearningType.FACT,
                                                sample_size=1, direction="positive")
        old = Experience(situation="old slippage s", observation="old slippage o",
                         source_id="trd-old00000001", confidence=0.9,
                         created_at=datetime.now(UTC) - timedelta(days=60),
                         updated_at=datetime.now(UTC) - timedelta(days=60))
        await app.agent_experiences.save(old)
        await app.agent_memory_labels.set_label(target_type="experience", target_id=old.id,
                                                learning_type=LearningType.OBSERVATION,
                                                sample_size=1, direction="negative")
        collector = ContextCollector(AgentTools(app),
                                     experience_repo=app.agent_experiences,
                                     lesson_repo=app.agent_lessons,
                                     recommendation_repo=app.agent_recommendations,
                                     journal_reader=app.agent_journal,
                                     label_repository=app.agent_memory_labels)
        ctx = await collector.collect(query="slippage")
        assert ctx.experiences
        first = ctx.experiences[0]
        for field in ("learning_type", "sample_size", "freshness", "decayed_confidence",
                      "source_id", "evidence", "created_at"):
            assert field in first, f"missing retrieval field: {field}"
        # Decayed ranking: fresh 0.5 outranks 60-day-old 0.9 (0.225).
        assert first["id"] == fresh.id
        assert first["freshness"]["weight"] == pytest.approx(1.0, abs=0.01)
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase5_memory_distinct_from_journal_facts(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.analysis import AnalysisEngine

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        ids = await _seed_trades(app, 6)
        exp = _exp(source_id=ids[0], situation=f"trade {ids[0]} fees observed",
                   observation=f"fees on trade {ids[0]} within tolerance")
        await app.agent_experiences.save(exp)
        from app.agent.context import ContextCollector
        from app.agent.tools import AgentTools

        collector = ContextCollector(AgentTools(app),
                                     experience_repo=app.agent_experiences,
                                     lesson_repo=app.agent_lessons,
                                     recommendation_repo=app.agent_recommendations,
                                     journal_reader=app.agent_journal,
                                     label_repository=app.agent_memory_labels)
        ctx = await collector.collect(query=f"trade {ids[0]} fees")
        analysis = AnalysisEngine().build(context=ctx, reflection=None, llm_output=None)
        facts = "\n".join(analysis.facts)
        assert "journal-fact" in facts  # authoritative current journal truth
        assert "memory-observation" in facts  # past memory, clearly labeled
        assert "memory-lesson" in facts or "memory-observation" in facts
        # Memory lines never masquerade as journal facts.
        for line in analysis.facts:
            if line.startswith("memory-"):
                assert "journal-fact" not in line
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase5_insufficient_data(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        # Empty journal → scheduler reports skips, not conclusions.
        report = await app.agent_reflection.run_due(app)
        assert report["triggered"] == []
        assert "post_trade" in report["skipped"]
        assert aggregate_to_lesson("x", [_exp()], pattern="p") is None
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ security


@pytest.mark.asyncio
async def test_phase5_security_boundaries(tmp_path):
    import pathlib as _pl

    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.providers.base import LLMProvider, LLMRequest, LLMResponse
    from app.agent.tools import AgentTools

    # Analysis layer (extraction/reflection) has no DB-session/SQL/shell surface.
    for name in ("extraction.py", "reflection.py"):
        text = (_pl.Path("app/agent") / name).read_text(encoding="utf-8")
        for forbidden in ("session.merge", "session.add", "session.delete", ".execute(",
                          "os.system", "subprocess", "eval(", "exec(", "open("):
            assert forbidden not in text, f"{name} contains {forbidden}"
    # Persistence layer uses parameterized ORM only (no raw SQL text).
    mem_text = (_pl.Path("app/agent/memory.py")).read_text(encoding="utf-8")
    assert "text(" not in mem_text and "os.system" not in mem_text

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        from app.agent.core import AgentRequest

        class LyingLLM(LLMProvider):
            name = "lying"

            async def complete(self, request: LLMRequest) -> LLMResponse:
                return LLMResponse(content="All trades print money.")

        core, *_ = build_agent(app, llm=LyingLLM())
        n_exp_before = await app.agent_experiences.count()
        n_les_before = len(await app.agent_lessons.list_all())
        # LLM traffic never writes memory: counts unchanged after handling.
        await core.handle(AgentRequest(query="status overview"))
        assert await app.agent_experiences.count() == n_exp_before
        assert len(await app.agent_lessons.list_all()) == n_les_before
        # Tools allowlist unchanged; feedback requires a human approver.
        assert len(AgentTools(app).allowed_tools) == 10
        with pytest.raises(Exception):
            await app.agent_feedback.submit(target_type="lesson", target_id="les-x",
                                            rating=1, approver="  ")
        with pytest.raises(Exception):
            await app.agent_feedback.submit(target_type="lesson", target_id="les-x",
                                            rating=5, approver="human")
    finally:
        await shutdown_app(app)
