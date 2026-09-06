"""Phase 7 — Recommendation Engine focused tests.

Covers: evidence-based generation, evidence/confidence/sample size,
persistence (+restart), deduplication, contradiction + insufficient-data
handling, lifecycle CREATED → REVIEWED → APPROVED/REJECTED, full audit
trail, Telegram /ai recommendations output, new_recommendation notification
integration, LLM failure isolation, and proof that generation never mutates
trading/config/risk state. Phase 8 approval workflow is NOT implemented.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.agent.models import RecommendationStatus
from app.models.enums import ArbitrageStrategy, TradeStatus, TradingMode
from app.models.trade import TradeRecord


def _trade(**overrides) -> TradeRecord:
    base = {
        "strategy": ArbitrageStrategy.TRIANGLE,
        "mode": TradingMode.PAPER,
        "exchange_id": "binance",
        "route": "USDT->BTC->ETH->USDT",
        "input_amount": Decimal("1000"),
        "output_amount": Decimal("1005"),
        "fees_quote": Decimal("1"),
        "slippage_bps": Decimal("10"),
        "net_profit": Decimal("5"),
        "net_profit_bps": Decimal("50"),
        "status": TradeStatus.COMPLETED,
        "orders": (),
        "error": None,
    }
    base.update(overrides)
    return TradeRecord(**base)


async def _wired_app(tmp_path, **agent_overrides):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    base = make_settings(tmp_path)
    if agent_overrides:
        base = base.model_copy(update={"agent": base.agent.model_copy(update=agent_overrides)})
    app = await build_app(base)
    await start_app(app, start_telegram=False, start_streams=False)
    build_agent(app)
    return app, shutdown_app


# ------------------------------------------------------------------ generation


@pytest.mark.asyncio
async def test_phase7_generation_slippage_rule(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        ids = []
        for _ in range(6):
            t = await app.trades.save(_trade(slippage_bps=Decimal("40")))
            ids.append(t.id)
        report = await app.agent_recommender.generate()
        assert len(report["created"]) == 1
        rec = await app.agent_recommendation_service.repository.get(report["created"][0])
        assert rec is not None
        assert rec.parameter == "risk.max_slippage_bps"
        assert rec.status == RecommendationStatus.PENDING
        # Evidence = supporting trade ids; sample size derived from evidence.
        assert set(ids) <= set(rec.evidence)
        assert rec.sample_size >= 5
        assert 0.0 <= rec.confidence <= 0.85
        assert rec.source_type == "system" and rec.source_id.startswith("phase7:")
        assert rec.created_at is not None and rec.updated_at is not None
        # Conservative proposal within the allowlist, below the live limit.
        assert Decimal(rec.proposed_value) < Decimal("15")
        assert Decimal("1") <= Decimal(rec.proposed_value) <= Decimal("100")
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase7_generation_failure_rule(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        for _ in range(3):
            await app.trades.save(_trade())
        for _ in range(4):
            await app.trades.save(_trade(status=TradeStatus.FAILED, net_profit=Decimal("-5"),
                                         net_profit_bps=Decimal("-50"), error="timeout"))
        report = await app.agent_recommender.generate(limit=5)
        created = [await app.agent_recommendation_service.repository.get(rid) for rid in report["created"]]
        by_param = {r.parameter: r for r in created if r is not None}
        assert "risk.max_trade_size" in by_param
        rec = by_param["risk.max_trade_size"]
        assert Decimal(rec.proposed_value) == Decimal("900")  # 1000 * 0.9, deterministic
        assert rec.sample_size >= 5
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase7_insufficient_data(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        await app.trades.save(_trade())
        await app.trades.save(_trade())
        report = await app.agent_recommender.generate()
        assert report["created"] == [] and report["deduplicated"] == []
        assert report["skipped"]
        assert all("insufficient_data" in reason for reason in report["skipped"].values())
        # Empty journal → explicit journal skip.
        from tests.conftest import make_settings as _ms
        from app.services import build_app as _build, shutdown_app as _shut, start_app as _start
        from app.agent import build_agent as _wire

        app2 = await _build(_ms(tmp_path / "empty"))
        await _start(app2, start_telegram=False, start_streams=False)
        try:
            _wire(app2)
            report2 = await app2.agent_recommender.generate()
            assert report2["created"] == []
            assert "insufficient_data" in report2["skipped"].get("journal", "")
        finally:
            await _shut(app2)
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase7_persistence_and_restart(tmp_path):
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from tests.conftest import make_settings

    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("40")))
        report = await app.agent_recommender.generate()
        rec_id = report["created"][0]
    finally:
        await shutdown_app(app)
    app2 = await build_app(settings)
    await start_app(app2, start_telegram=False, start_streams=False)
    try:
        build_agent(app2)
        loaded = await app2.agent_recommendation_service.repository.get(rec_id)
        assert loaded is not None
        assert loaded.status == RecommendationStatus.PENDING
        assert loaded.sample_size >= 5 and loaded.evidence
    finally:
        await shutdown_app(app2)


@pytest.mark.asyncio
async def test_phase7_deduplication(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("40")))
        first = await app.agent_recommender.generate()
        assert len(first["created"]) == 1
        second = await app.agent_recommender.generate()
        assert second["created"] == []
        assert second["deduplicated"] == first["created"]
        all_recs = await app.agent_recommendation_service.repository.list_all()
        assert len(all_recs) == 1
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase7_contradiction_retains_both(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("40")))
        first = await app.agent_recommender.generate()
        existing_id = first["created"][0]
        # Human-style conflicting row for the same parameter, other value.
        manual = await app.agent_recommendation_service.create(
            parameter="risk.max_slippage_bps", current_value="15", proposed_value="20",
            reason="operator experiment", source_id="manual-test",
        )
        # New evidence round proposes the engine value again → dedup hits the
        # first row (same value), manual row retained untouched.
        again = await app.agent_recommender.generate()
        assert again["deduplicated"] == [existing_id]
        assert (await app.agent_recommendation_service.repository.get(manual.id)) is not None
        # Opposing engine value → both retained + conflict audit event.
        await app.agent_recommendation_service.repository.save(
            (await app.agent_recommendation_service.repository.get(existing_id)).with_status(
                RecommendationStatus.REJECTED, operator_decision="rejected by test"))
        report = await app.agent_recommender.generate()
        assert len(report["created"]) == 1
        events = await app.agent_audit.list_recent(limit=20)
        conflicts = [e for e in events if e.event_type == "recommendation_conflict"]
        assert conflicts and conflicts[0].details["resolution"] == "retained_both"
    finally:
        await shutdown(app)


# ------------------------------------------------------------------ lifecycle


@pytest.mark.asyncio
async def test_phase7_lifecycle_reviewed_flow(tmp_path):
    from app.agent.approval import ApprovalError

    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value="1000", proposed_value="900",
            reason="lifecycle", source_id="phase7-test",
        )
        assert rec.status == RecommendationStatus.PENDING  # CREATED
        reviewed = await app.agent_approval_service.review(rec.id, approver="alice", reason="looks sane")
        assert reviewed.status == RecommendationStatus.REVIEWED
        assert reviewed.operator_decision == "reviewed by alice"
        approved = await app.agent_approval_service.approve(rec.id, approver="alice", reason="go")
        assert approved.status == RecommendationStatus.APPROVED
        # Double decision rejected; review-after-decision rejected.
        with pytest.raises(ApprovalError, match="not PENDING"):
            await app.agent_approval_service.approve(rec.id, approver="alice")
        with pytest.raises(ApprovalError, match="not PENDING"):
            await app.agent_approval_service.review(rec.id, approver="alice")
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase7_review_validation_and_reject(tmp_path):
    from app.agent.approval import ApprovalError

    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_slippage_bps", current_value="15", proposed_value="12",
            reason="review me", source_id="phase7-test",
        )
        with pytest.raises(ApprovalError):
            await app.agent_approval_service.review(rec.id, approver="  ")
        with pytest.raises(ApprovalError):
            await app.agent_approval_service.review("rec-missing", approver="bob")
        reviewed = await app.agent_approval_service.review(rec.id, approver="bob")
        assert reviewed.status == RecommendationStatus.REVIEWED
        rejected = await app.agent_approval_service.reject(rec.id, approver="bob", reason="no")
        assert rejected.status == RecommendationStatus.REJECTED
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase7_audit_trail(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("40")))
        report = await app.agent_recommender.generate()
        rec_id = report["created"][0]
        await app.agent_approval_service.review(rec_id, approver="carol", reason="triage")
        await app.agent_approval_service.approve(rec_id, approver="carol", reason="ok")
        events = await app.agent_audit.list_recent(limit=20)
        by_type = {e.event_type for e in events}
        assert "recommendation_created" in by_type
        assert "recommendation_reviewed" in by_type
        created = next(e for e in events if e.event_type == "recommendation_created")
        assert created.details["recommendation_id"] == rec_id
        assert created.confidence is not None and created.evidence_count >= 5
    finally:
        await shutdown(app)


# ------------------------------------------------------------------ integrations


@pytest.mark.asyncio
async def test_phase7_telegram_recommendations(tmp_path):
    from app.telegram.bot import TelegramBot
    from app.telegram.i18n import lang_storage_key

    app, shutdown = await _wired_app(tmp_path)
    try:
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("40")))
        report = await app.agent_recommender.generate()
        rec_id = report["created"][0]
        await app.bot_state.set(lang_storage_key(11111), "en")

        class FakeClient:
            def __init__(self):
                self.sent = []

            async def send_message(self, chat_id, text, parse_mode="", reply_markup=None):
                self.sent.append((chat_id, text))

            async def edit_message_text(self, *a, **kw):
                pass

            async def answer_callback_query(self, *a, **kw):
                pass

        def _upd(chat_id, text, user_id=11111):
            return {"update_id": 1, "message": {"chat": {"id": chat_id}, "from": {"id": user_id}, "text": text}}

        bot = TelegramBot(app, FakeClient())
        await bot.handle_update(_upd(12345, "/ai recommendations", user_id=11111))
        txt = bot._client.sent[-1][1]
        assert "Pending recommendations" in txt
        assert "risk.max_slippage_bps" in txt and "[n=" in txt
        # Reviewed items stay visible with their status.
        await app.agent_approval_service.review(rec_id, approver="dave")
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai recommendations", user_id=11111))
        txt2 = bot._client.sent[-1][1]
        assert "risk.max_slippage_bps" in txt2 and "reviewed" in txt2
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase7_notification_integration(tmp_path):
    from app.agent.notifications import NotificationService

    app, shutdown = await _wired_app(tmp_path, notifications_enabled=True)
    try:
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("40")))

        class FakeClient:
            def __init__(self):
                self.sent = []

            async def send_message(self, chat_id, text, parse_mode="", reply_markup=None):
                self.sent.append((chat_id, text))

        fake = FakeClient()
        app.agent_notifications = NotificationService(app, client=fake)
        report = await app.agent_recommender.generate()
        assert report["created"] and report["notified"] == report["created"]
        texts = " | ".join(t for _, t in fake.sent)
        assert report["created"][0] in texts and "risk.max_slippage_bps" in texts
        # Disabled layer → generation works, notification skipped.
        app.settings = app.settings.model_copy(update={
            "agent": app.settings.agent.model_copy(update={"notifications_enabled": False})})
        for _ in range(6):
            await app.trades.save(_trade(status=TradeStatus.FAILED, net_profit=Decimal("-5"),
                                         net_profit_bps=Decimal("-50"), error="x"))
        report2 = await app.agent_recommender.generate(limit=5)
        assert report2["notified"] == []
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase7_llm_explains_but_never_decides(tmp_path):
    from app.agent.providers.base import LLMProvider, LLMRequest, LLMResponse

    class Explainer(LLMProvider):
        name = "explainer"

        async def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(content="The data shows persistent slippage pressure.")

    class CrashLLM(LLMProvider):
        name = "crash"

        async def complete(self, request: LLMRequest) -> LLMResponse:
            raise RuntimeError("llm down")

    app, shutdown = await _wired_app(tmp_path)
    try:
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("40")))
        ok = await app.agent_recommender.generate(llm=Explainer())
        rec = await app.agent_recommendation_service.repository.get(ok["created"][0])
        assert rec is not None and "AI note" in rec.reason
        assert "15" in rec.reason and "40" in rec.reason  # computed numbers survive
        # Crashing LLM → template reason, same computed values, no raise.
        for _ in range(6):
            await app.trades.save(_trade(status=TradeStatus.FAILED, net_profit=Decimal("-5"),
                                         net_profit_bps=Decimal("-50"), error="x"))
        crashed = await app.agent_recommender.generate(limit=5, llm=CrashLLM())
        assert crashed["created"]
        rec2 = await app.agent_recommendation_service.repository.get(crashed["created"][0])
        assert rec2 is not None and "AI note" not in rec2.reason
        assert rec2.parameter == "risk.max_trade_size" and rec2.proposed_value == "900"
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase7_no_mutation_proof(tmp_path):
    from app.agent.recommendations import RecommendationApplyBlocked

    app, shutdown = await _wired_app(tmp_path)
    try:
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("40")))
        limits_before = str(app.settings.risk.max_trade_size)
        slip_before = str(app.settings.risk.max_slippage_bps)
        n_before = len(await app.trades.list_recent(50))
        halted_before = app.guard.is_halted
        report = await app.agent_recommender.generate()
        assert report["created"]
        # Nothing moved: config, risk engine, guard, journal identical.
        assert str(app.settings.risk.max_trade_size) == limits_before
        assert str(app.settings.risk.max_slippage_bps) == slip_before
        assert str(app.risk.limits.max_slippage_bps) == slip_before
        assert len(await app.trades.list_recent(50)) == n_before
        assert app.guard.is_halted is halted_before
        assert (await app.agent_recommendation_service.repository.get(report["created"][0])).status == RecommendationStatus.PENDING
        # Direct apply paths stay blocked.
        with pytest.raises(RecommendationApplyBlocked):
            app.agent_recommendation_service.apply(report["created"][0])
        with pytest.raises(RecommendationApplyBlocked):
            app.agent_recommendation_service.mutate_config(parameter="risk.max_trade_size", value="1")
        # Engine exposes no executive surface.
        for forbidden in ("approve", "apply", "mutate_config", "update_risk_limits",
                          "create_order", "withdraw", "execute", "set_config", "shell"):
            assert not hasattr(app.agent_recommender, forbidden), forbidden
    finally:
        await shutdown(app)


def test_phase7_sample_size_property():
    from app.agent.models import AgentRecommendation

    rec = AgentRecommendation(parameter="risk.max_trade_size", proposed_value="900",
                              reason="r", source_id="s", evidence=("a", "b", "c"))
    assert rec.sample_size == 3
    assert AgentRecommendation(parameter="p", proposed_value="v", reason="r", source_id="s").sample_size == 0

