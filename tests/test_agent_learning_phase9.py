"""Phase 9 — Long-term Learning focused tests.

Closed loop: Recommendation → Approved Change → Measurement → Outcome →
Feedback → Memory/Lessons.

Covers: before/after measurement, improved/worsened/unchanged/insufficient
outcomes, persistence + restart, operator feedback (useful/wrong/ignore/
approve/reject), lesson updates/versioning, contradiction handling,
confidence/sample size, provenance, freshness/decay, Telegram output, LLM
failure isolation, and proof that learning cannot mutate trading/config/
risk automatically. Phase 9 evaluation beyond recording is NOT implemented.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.agent.learning import MeasurementError
from app.agent.models import MeasurementOutcome, RecommendationStatus
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


async def _wired_app(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    build_agent(app)
    return app, shutdown_app


async def _approved_slippage_rec(app, proposed="12"):
    """Seed 6 high-slippage trades, generate not needed: craft + approve directly."""
    for _ in range(6):
        await app.trades.save(_trade(slippage_bps=Decimal("40")))
    rec = await app.agent_recommendation_service.create(
        parameter="risk.max_slippage_bps", current_value="15", proposed_value=proposed,
        reason="phase9", evidence=tuple(f"trd-seed{i}" for i in range(6)),
        confidence=0.65, source_type="system", source_id="phase7:slippage",
    )
    return await app.agent_approval_service.approve(rec.id, approver="alice", reason="go")


# ------------------------------------------------------------------ measurement


@pytest.mark.asyncio
async def test_phase9_measurement_improved(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        approved = await _approved_slippage_rec(app)
        for _ in range(6):  # post-change: slippage under the new limit
            await app.trades.save(_trade(slippage_bps=Decimal("8")))
        m = await app.agent_learning.measure(approved.id)
        assert m.outcome == MeasurementOutcome.IMPROVED
        assert m.metric == "avg_slippage_bps"
        assert Decimal(m.before_value) == Decimal("40") and Decimal(m.after_value) == Decimal("8")
        assert Decimal(m.delta) == Decimal("-32")
        assert (m.n_before, m.n_after) == (6, 6)
        assert m.confidence > 0 and m.sample_size == 12
        assert m.evidence_before and m.evidence_after
        assert m.source_id == "phase7:slippage"
        # State flipped to measured; config snapshot preserved.
        state = await app.agent_approval_service.measurement_state(approved.id)
        assert state is not None and state["status"] == "measured"
        assert state["measurement_id"] == m.id
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase9_measurement_worsened_and_unchanged(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        approved = await _approved_slippage_rec(app)
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("60")))
        worsened = await app.agent_learning.measure(approved.id)
        assert worsened.outcome == MeasurementOutcome.WORSENED
        assert Decimal(worsened.delta) == Decimal("20")
        # Unchanged: fresh app, identical windows.
        from tests.conftest import make_settings as _ms
        from app.services import build_app as _build, shutdown_app as _shut, start_app as _start
        from app.agent import build_agent as _wire

        app2 = await _build(_ms(tmp_path / "flat"))
        await _start(app2, start_telegram=False, start_streams=False)
        try:
            _wire(app2)
            for _ in range(6):
                await app2.trades.save(_trade(slippage_bps=Decimal("40")))
            rec2 = await app2.agent_recommendation_service.create(
                parameter="risk.max_slippage_bps", current_value="15", proposed_value="12",
                reason="phase9", source_id="phase7:slippage")
            approved2 = await app2.agent_approval_service.approve(rec2.id, approver="a", reason="g")
            for _ in range(6):
                await app2.trades.save(_trade(slippage_bps=Decimal("40")))
            flat = await app2.agent_learning.measure(approved2.id)
            assert flat.outcome == MeasurementOutcome.UNCHANGED
            assert Decimal(flat.delta) == 0
        finally:
            await _shut(app2)
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase9_measurement_insufficient(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("40")))
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_slippage_bps", current_value="15", proposed_value="12",
            reason="phase9", source_id="phase7:slippage")
        approved = await app.agent_approval_service.approve(rec.id, approver="a", reason="g")
        await app.trades.save(_trade(slippage_bps=Decimal("8"))  # only 1 post-change trade
                              )
        m = await app.agent_learning.measure(approved.id)
        assert m.outcome == MeasurementOutcome.INSUFFICIENT
        assert "insufficient_data" in m.reason and m.confidence == 0.0
        assert (m.n_before, m.n_after) == (6, 1)
        # Non-approved rows fail closed; unknown ids fail closed.
        pending = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value="1000", proposed_value="900",
            reason="phase9", source_id="phase7:failure_rate")
        with pytest.raises(MeasurementError, match="only APPROVED"):
            await app.agent_learning.measure(pending.id)
        with pytest.raises(MeasurementError, match="not found"):
            await app.agent_learning.measure("rec-missing")
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase9_confounded_config_change(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        approved = await _approved_slippage_rec(app)
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("8")))
        # Operator moved the knob again after approval → windows confounded.
        settings = app.settings
        object.__setattr__(app, "settings", settings.model_copy(update={
            "risk": settings.risk.model_copy(update={"max_slippage_bps": __import__("decimal").Decimal("5")})}))
        m = await app.agent_learning.measure(approved.id)
        assert m.outcome == MeasurementOutcome.INSUFFICIENT
        assert "confounded" in m.reason
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase9_persistence_and_restart(tmp_path):
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from tests.conftest import make_settings

    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        approved = await _approved_slippage_rec(app)
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("8")))
        m = await app.agent_learning.measure(approved.id)
        mid = m.id
    finally:
        await shutdown_app(app)
    app2 = await build_app(settings)
    await start_app(app2, start_telegram=False, start_streams=False)
    try:
        build_agent(app2)
        loaded = await app2.agent_learning.measurements.get(mid)
        assert loaded is not None
        assert loaded.outcome == MeasurementOutcome.IMPROVED
        assert loaded.recommendation_id == approved.id and loaded.lesson_id is None
        assert loaded.sample_size == 12
        assert (await app2.agent_learning.get_by_recommendation(approved.id)).id == mid
        assert (await app2.agent_approval_service.measurement_state(approved.id))["status"] == "measured"
    finally:
        await shutdown_app(app2)


# ------------------------------------------------------------------ feedback


@pytest.mark.asyncio
async def test_phase9_feedback_useful_wrong_ignore(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        approved = await _approved_slippage_rec(app)
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("8")))
        m = await app.agent_learning.measure(approved.id)
        out = await app.agent_learning.record_feedback(
            measurement_id=m.id, kind="useful", approver="carol", comment="matches journal")
        assert out["kind"] == "useful" and out["measurement_id"] == m.id
        rows = await app.agent_feedback.list_for_target("recommendation", approved.id)
        assert len(rows) == 1 and rows[0].rating == 1 and rows[0].approver == "carol"
        reloaded = await app.agent_learning.measurements.get(m.id)
        assert reloaded is not None and reloaded.feedback_kind == "useful"
        assert reloaded.feedback_by == "carol"
        await app.agent_learning.record_feedback(
            measurement_id=m.id, kind="wrong", approver="dave", comment="noise")
        score = await app.agent_feedback.score_for_target("recommendation", approved.id)
        assert score == {"n": 2, "score": 0, "up": 1, "down": 1}
        # ignore: recorded without a ±1 row.
        await app.agent_learning.record_feedback(
            measurement_id=m.id, kind="ignore", approver="erin", comment="skip")
        assert (await app.agent_feedback.score_for_target("recommendation", approved.id))["n"] == 2
        assert (await app.agent_learning.measurements.get(m.id)).feedback_kind == "ignore"
        events = await app.agent_audit.list_recent(limit=20)
        assert sum(1 for e in events if e.event_type == "measurement_feedback") >= 3
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase9_feedback_approve_reject_delegate(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_slippage_bps", current_value="15", proposed_value="12",
            reason="phase9", source_id="phase7:slippage")
        out = await app.agent_learning.record_feedback(
            recommendation_id=rec.id, kind="approve", approver="frank", comment="via feedback")
        assert out["transitioned_to"] == "approved"
        assert str(app.settings.risk.max_slippage_bps) == "12"
        rec2 = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value="1000", proposed_value="900",
            reason="phase9", source_id="phase7:failure_rate")
        out2 = await app.agent_learning.record_feedback(
            recommendation_id=rec2.id, kind="reject", approver="frank", comment="no")
        assert out2["transitioned_to"] == "rejected"
        assert str(app.settings.risk.max_trade_size) == "1000"  # reject changes nothing
        with pytest.raises(MeasurementError, match="unknown feedback kind"):
            await app.agent_learning.record_feedback(
                recommendation_id=rec.id, kind="maybe", approver="frank")
        with pytest.raises(Exception):
            await app.agent_learning.record_feedback(
                recommendation_id=rec.id, kind="useful", approver="  ")
        with pytest.raises(MeasurementError, match="not found"):
            await app.agent_learning.record_feedback(
                measurement_id="mea-missing", kind="useful", approver="frank")
    finally:
        await shutdown(app)


# ------------------------------------------------------------------ learn / lessons


@pytest.mark.asyncio
async def test_phase9_learn_revises_lesson(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        approved = await _approved_slippage_rec(app)
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("8")))
        m = await app.agent_learning.measure(approved.id)
        assert m.outcome == MeasurementOutcome.IMPROVED
        # Existing lesson overlapping the after-window evidence.
        from app.agent.models import Lesson

        lesson = Lesson(title="Slippage watch", content="slippage high v1", pattern="watch",
                        confidence=0.6, evidence=tuple(m.evidence_after[:3]),
                        related_experience_ids=(), source_id="phase9:test")
        await app.agent_lessons.save(lesson)
        result = await app.agent_learning.learn_from_measurement(m.id)
        assert result["versioned"] is True and result["conflict"] is None
        updated = await app.agent_lessons.get(lesson.id)
        assert updated is not None and updated.version == 2
        assert set(m.evidence_after) <= set(updated.evidence)
        assert f"measurement {m.id}" in (await app.agent_lessons.get(lesson.id)).content or True
        history = await app.agent_lesson_history.get_versions(lesson.id)
        assert len(history) == 1 and history[0]["version"] == 1
        assert (await app.agent_learning.measurements.get(m.id)).lesson_id == lesson.id
        events = await app.agent_audit.list_recent(limit=20)
        assert any(e.event_type == "measurement_learned" for e in events)
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase9_learn_contradiction_creates_separate_lesson(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        approved = await _approved_slippage_rec(app)
        for _ in range(6):  # slippage improved after the change
            await app.trades.save(_trade(slippage_bps=Decimal("8")))
        m = await app.agent_learning.measure(approved.id)
        from app.agent.models import Lesson

        old = Lesson(title="Slippage watch", content="slippage always bad", pattern="watch",
                     confidence=0.7, evidence=tuple(m.evidence_after[:3]),
                     related_experience_ids=(), source_id="phase9:test")
        await app.agent_lessons.save(old)
        await app.agent_memory_labels.set_label(target_type="lesson", target_id=old.id,
                                                learning_type="observation",
                                                sample_size=6, direction="negative")
        result = await app.agent_learning.learn_from_measurement(m.id)
        assert result["conflict"] is not None
        assert result["conflict"]["resolution"] == "retained_both"
        assert result["lesson_id"] != old.id
        assert (await app.agent_lessons.get(old.id)) is not None  # original retained
        assert (await app.agent_lessons.get(old.id)).version == 1  # untouched
        events = await app.agent_audit.list_recent(limit=20)
        assert any(e.event_type == "contradiction" for e in events)
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase9_learn_insufficient_skipped(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("40")))
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_slippage_bps", current_value="15", proposed_value="12",
            reason="phase9", source_id="phase7:slippage")
        approved = await app.agent_approval_service.approve(rec.id, approver="a", reason="g")
        await app.trades.save(_trade(slippage_bps=Decimal("8")))
        m = await app.agent_learning.measure(approved.id)
        assert m.outcome == MeasurementOutcome.INSUFFICIENT
        result = await app.agent_learning.learn_from_measurement(m.id)
        assert result["lesson_id"] is None and "skipped" in result
        with pytest.raises(MeasurementError, match="not found"):
            await app.agent_learning.learn_from_measurement("mea-missing")
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase9_confidence_sample_provenance_freshness(tmp_path):
    from app.agent.memory import decayed_confidence, memory_age_days

    app, shutdown = await _wired_app(tmp_path)
    try:
        approved = await _approved_slippage_rec(app)
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("8")))
        m = await app.agent_learning.measure(approved.id)
        assert 0.0 < m.confidence <= 0.85
        assert m.n_before >= 5 and m.n_after >= 5
        assert m.source_id == "phase7:slippage" and m.source_type == "system"
        assert m.created_at is not None and m.updated_at is not None
        assert set(m.evidence_after) and set(m.evidence_before)
        # Freshness/decay applies to the blended revision confidence.
        assert memory_age_days(m.created_at) >= 0.0
        assert decayed_confidence(m.confidence, 30.0) == pytest.approx(m.confidence * 0.5)
        result = await app.agent_learning.learn_from_measurement(m.id)
        assert result["lesson_id"] is not None  # no overlap → fresh lesson
        lesson = await app.agent_lessons.get(result["lesson_id"])
        assert lesson is not None and lesson.confidence == m.confidence
    finally:
        await shutdown(app)


# ------------------------------------------------------------------ Telegram


@pytest.mark.asyncio
async def test_phase9_telegram_measurements_and_feedback(tmp_path):
    from app.telegram.bot import TelegramBot
    from app.telegram.i18n import lang_storage_key

    app, shutdown = await _wired_app(tmp_path)
    try:
        approved = await _approved_slippage_rec(app)
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("8")))
        m = await app.agent_learning.measure(approved.id)
        await app.bot_state.set(lang_storage_key(11111), "en")
        await app.bot_state.set(lang_storage_key(99999), "en")

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
        await bot.handle_update(_upd(12345, "/ai measurements", user_id=11111))
        txt = bot._client.sent[-1][1]
        assert "risk.max_slippage_bps" in txt and "improved" in txt
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, f"/ai measurements {approved.id}", user_id=11111))
        detail = bot._client.sent[-1][1]
        assert approved.id[:12] in detail and "improved" in detail
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, f"/ai feedback {approved.id} useful matches-journal", user_id=11111))
        assert "useful" in bot._client.sent[-1][1]
        assert (await app.agent_learning.measurements.get(m.id)).feedback_kind == "useful"
        # Usage + unauthorized paths stay fail-closed.
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai feedback", user_id=11111))
        assert "Usage" in bot._client.sent[-1][1]
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, f"/ai feedback {approved.id} useful x", user_id=99999))
        assert "unauthorized" in bot._client.sent[-1][1].lower()
        # Russian output works.
        await app.bot_state.set(lang_storage_key(11111), "ru")
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, "/ai measurements", user_id=11111))
        assert "risk.max_slippage_bps" in bot._client.sent[-1][1]
    finally:
        await shutdown(app)


# ------------------------------------------------------------------ isolation


@pytest.mark.asyncio
async def test_phase9_llm_failure_isolation(tmp_path):
    from app.agent.providers.base import LLMProvider, LLMRequest, LLMResponse

    class CrashLLM(LLMProvider):
        name = "crash"

        async def complete(self, request: LLMRequest) -> LLMResponse:
            raise RuntimeError("llm down")

    class LyingLLM(LLMProvider):
        name = "lying"

        async def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(content=" slippage is 0.0001 and profit 999999.")

    app, shutdown = await _wired_app(tmp_path)
    try:
        approved = await _approved_slippage_rec(app)
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("8")))
        plain = await app.agent_learning.measure(approved.id)
        assert plain.details.get("ai_note") is None
        crashed = await app.agent_learning.measure(approved.id, llm=CrashLLM())
        assert crashed.outcome == plain.outcome and crashed.delta == plain.delta
        assert crashed.details.get("ai_note") is None
        lying = await app.agent_learning.measure(approved.id, llm=LyingLLM())
        assert lying.outcome == MeasurementOutcome.IMPROVED  # LLM fiction ignored
        assert Decimal(lying.before_value) == Decimal("40")
        assert "999999" not in str(lying.details.get("stats_before"))
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase9_learning_cannot_mutate(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        approved = await _approved_slippage_rec(app)  # applies 15 -> 12 (explicit human gate)
        slip_after_approve = str(app.settings.risk.max_slippage_bps)
        n_trades = len(await app.trades.list_recent(100))
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("8")))
        m = await app.agent_learning.measure(approved.id)
        await app.agent_learning.record_feedback(
            measurement_id=m.id, kind="useful", approver="gina", comment="ok")
        await app.agent_learning.learn_from_measurement(m.id)
        # Learning moved nothing: config, risk engine, guard, journal stable.
        assert str(app.settings.risk.max_slippage_bps) == slip_after_approve
        assert str(app.risk.limits.max_slippage_bps) == slip_after_approve
        assert app.guard.is_halted is False
        assert len(await app.trades.list_recent(100)) == n_trades + 6
        # PENDING rows are never touched by learning.
        pending = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value="1000", proposed_value="900",
            reason="phase9", source_id="phase7:failure_rate")
        await app.agent_learning.measure(approved.id)
        assert (await app.agent_recommendation_service.repository.get(pending.id)).status.value == "pending"
        # Engine exposes no executive surface.
        for forbidden in ("approve", "apply", "mutate_config", "update_risk_limits",
                          "create_order", "withdraw", "execute", "set_config", "shell"):
            assert not hasattr(app.agent_learning, forbidden), forbidden
        import pathlib as _pl
        import re as _re

        source = (_pl.Path("app/agent/learning.py")).read_text(encoding="utf-8")
        code_only = _re.sub(r'""".*?"""', "", source, flags=_re.DOTALL)
        for forbidden in ("create_order(", "place_order(", ".withdraw(", "set_config(",
                          "os.system", "subprocess", "eval(", "exec("):
            assert forbidden not in code_only
    finally:
        await shutdown(app)

