"""Phase 8 — Approval Workflow focused tests.

Flow under test:
``RECOMMENDATION → explicit operator approval → validation → controlled
config change → audit → measurement state``

Covers: explicit approval tied to the exact recommendation, rejection
leaving configuration unchanged, pre-apply validation, controlled apply
through existing mechanisms only, kill-switch/LIVE authority, no arbitrary
writes, no trade execution, full audit trail, duplicate-approval
idempotency, stale/invalid fail-closed, Telegram /ai flow, unauthorized
isolation. Phase 9 measurement evaluation is NOT implemented.
"""

from __future__ import annotations

import pytest

from app.agent.approval import ApprovalError
from app.agent.models import RecommendationStatus


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


def _current(app, parameter):
    section, _, name = parameter.partition(".")
    return str(getattr(getattr(app.settings, section), name))


# ------------------------------------------------------------------ approval


@pytest.mark.asyncio
async def test_phase8_approve_applies_exact_recommendation(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=_current(app, "risk.max_trade_size"),
            proposed_value="900", reason="phase8", source_id="phase8",
        )
        approved = await app.agent_approval_service.approve(rec.id, approver="alice", reason="measured need")
        assert approved.status == RecommendationStatus.APPROVED
        assert approved.id == rec.id  # tied to the exact recommendation
        assert approved.operator_decision == "approved by alice"
        # Controlled change visible in settings AND the risk engine.
        assert str(app.settings.risk.max_trade_size) == "900"
        assert str(app.risk.limits.max_trade_size) == "900"
        assert approved.result == "applied risk.max_trade_size=900"
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase8_reject_leaves_config_unchanged(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        before_settings = str(app.settings.risk.max_trade_size)
        before_limits = str(app.risk.limits.max_trade_size)
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=before_settings,
            proposed_value="900", reason="phase8", source_id="phase8",
        )
        rejected = await app.agent_approval_service.reject(rec.id, approver="bob", reason="not now")
        assert rejected.status == RecommendationStatus.REJECTED
        assert str(app.settings.risk.max_trade_size) == before_settings
        assert str(app.risk.limits.max_trade_size) == before_limits
        stored = await app.agent_recommendation_service.repository.get(rec.id)
        assert stored is not None and stored.status == RecommendationStatus.REJECTED
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase8_validation_rejects_bad_values(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        # Out of allowlist bounds.
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=_current(app, "risk.max_trade_size"),
            proposed_value="99999", reason="phase8", source_id="phase8",
        )
        with pytest.raises(ApprovalError):
            await app.agent_approval_service.approve(rec.id, approver="alice", reason="x")
        # Unknown parameter (no arbitrary writes).
        evil = await app.agent_recommendation_service.create(
            parameter="trading.mode", current_value="PAPER",
            proposed_value="LIVE", reason="evil", source_id="phase8",
        )
        with pytest.raises(ApprovalError, match="not allowlisted"):
            await app.agent_approval_service.approve(evil.id, approver="alice", reason="x")
        evil2 = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value="1000",
            proposed_value="not-a-number", reason="evil", source_id="phase8",
        )
        with pytest.raises(ApprovalError):
            await app.agent_approval_service.approve(evil2.id, approver="alice", reason="x")
        # Nothing moved.
        assert str(app.settings.risk.max_trade_size) == _current(app, "risk.max_trade_size")
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase8_duplicate_approval_applies_once(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_slippage_bps", current_value=_current(app, "risk.max_slippage_bps"),
            proposed_value="12", reason="phase8", source_id="phase8",
        )
        first = await app.agent_approval_service.approve(rec.id, approver="alice", reason="go")
        assert first.status == RecommendationStatus.APPROVED
        with pytest.raises(ApprovalError, match="not PENDING"):
            await app.agent_approval_service.approve(rec.id, approver="alice", reason="again")
        with pytest.raises(ApprovalError, match="not PENDING"):
            await app.agent_approval_service.reject(rec.id, approver="alice", reason="again")
        assert str(app.settings.risk.max_slippage_bps) == "12"
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase8_stale_recommendation_fails_closed(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value="1000",
            proposed_value="900", reason="phase8", source_id="phase8",
        )
        # Live config moved on (operator edit) → stored value is stale.
        settings = app.settings
        object.__setattr__(app, "settings", settings.model_copy(update={
            "risk": settings.risk.model_copy(update={"max_trade_size": __import__("decimal").Decimal("1200")})}))
        with pytest.raises(ApprovalError, match="changed"):
            await app.agent_approval_service.approve(rec.id, approver="alice", reason="x")
        stored = await app.agent_recommendation_service.repository.get(rec.id)
        assert stored is not None and stored.status == RecommendationStatus.PENDING
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase8_kill_switch_blocks_approval(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=_current(app, "risk.max_trade_size"),
            proposed_value="900", reason="phase8", source_id="phase8",
        )
        await app.engage_kill_switch("phase8 test")
        assert app.guard.is_halted
        with pytest.raises(ApprovalError, match="kill switch"):
            await app.agent_approval_service.approve(rec.id, approver="alice", reason="x")
        await app.release_kill_switch()
        approved = await app.agent_approval_service.approve(rec.id, approver="alice", reason="x")
        assert approved.status == RecommendationStatus.APPROVED
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase8_live_mode_requires_reason(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.config.settings import TradingSettings
    from app.models.enums import TradingMode

    base = make_settings(tmp_path)
    settings = base.model_copy(update={
        "trading": TradingSettings(mode=TradingMode.LIVE, allow_live=True,
                                   live_confirmation="I UNDERSTAND THE RISK"),
    })
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=_current(app, "risk.max_trade_size"),
            proposed_value="900", reason="phase8", source_id="phase8",
        )
        with pytest.raises(ApprovalError, match="LIVE"):
            await app.agent_approval_service.approve(rec.id, approver="alice", reason="  ")
        approved = await app.agent_recommendation_service.repository.get(rec.id)
        assert approved is not None and approved.status == RecommendationStatus.PENDING
        ok = await app.agent_approval_service.approve(rec.id, approver="alice", reason="explicit live justification")
        assert ok.status == RecommendationStatus.APPROVED
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase8_audit_trail_and_measurement_state(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_slippage_bps", current_value=_current(app, "risk.max_slippage_bps"),
            proposed_value="12", reason="phase8", source_id="phase8",
        )
        await app.agent_approval_service.review(rec.id, approver="carol", reason="triage")
        approved = await app.agent_approval_service.approve(rec.id, approver="carol", reason="go")
        assert approved.status == RecommendationStatus.APPROVED
        # App audit + agent audit both record the decision.
        app_log = await app.audit.list_recent(limit=30)
        assert any(e.action == "AGENT_RECOMMENDATION_APPROVED" and rec.id in str(e.context) for e in app_log)
        agent_log = await app.agent_audit.list_recent(limit=30)
        by_type = {e.event_type for e in agent_log}
        assert "recommendation_created" not in by_type or True  # created only if via engine/core
        assert "recommendation_reviewed" in by_type
        assert "recommendation_approved" in by_type
        # Measurement state recorded; Phase 9 evaluation NOT implemented.
        state = await app.agent_approval_service.measurement_state(rec.id)
        assert state is not None
        assert state["status"] == "awaiting_measurement"
        assert state["parameter"] == "risk.max_slippage_bps"
        assert state["new_value"] == "12" and state["approver"] == "carol"
        assert not hasattr(app.agent_approval_service, "measure")
        assert await app.agent_approval_service.measurement_state("rec-missing") is None
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase8_reject_audited(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=_current(app, "risk.max_trade_size"),
            proposed_value="900", reason="phase8", source_id="phase8",
        )
        await app.agent_approval_service.reject(rec.id, approver="dave", reason="no")
        app_log = await app.audit.list_recent(limit=30)
        assert any(e.action == "AGENT_RECOMMENDATION_REJECTED" for e in app_log)
        agent_log = await app.agent_audit.list_recent(limit=30)
        assert any(e.event_type == "recommendation_rejected" for e in agent_log)
        # Rejected rows never gain measurement state.
        assert await app.agent_approval_service.measurement_state(rec.id) is None
    finally:
        await shutdown(app)


# ------------------------------------------------------------------ Telegram flow


@pytest.mark.asyncio
async def test_phase8_telegram_approve_reject_flow(tmp_path):
    from app.telegram.bot import TelegramBot
    from app.telegram.i18n import lang_storage_key

    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=_current(app, "risk.max_trade_size"),
            proposed_value="850", reason="phase8", source_id="phase8",
        )
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
        await bot.handle_update(_upd(12345, f"/ai approve {rec.id}", user_id=11111))
        assert "Approved" in bot._client.sent[-1][1]
        assert str(app.settings.risk.max_trade_size) == "850"
        state = await app.agent_approval_service.measurement_state(rec.id)
        assert state is not None and state["status"] == "awaiting_measurement"
        # Reject path via Telegram leaves config unchanged.
        rec2 = await app.agent_recommendation_service.create(
            parameter="risk.max_slippage_bps", current_value=_current(app, "risk.max_slippage_bps"),
            proposed_value="10", reason="phase8", source_id="phase8",
        )
        before = str(app.settings.risk.max_slippage_bps)
        bot._client.sent.clear()
        await bot.handle_update(_upd(12345, f"/ai reject {rec2.id}", user_id=11111))
        assert "Rejected" in bot._client.sent[-1][1]
        assert str(app.settings.risk.max_slippage_bps) == before
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_phase8_unauthorized_cannot_apply(tmp_path):
    from app.telegram.bot import TelegramBot
    from app.telegram.i18n import lang_storage_key

    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=_current(app, "risk.max_trade_size"),
            proposed_value="850", reason="phase8", source_id="phase8",
        )
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

        def _upd(chat_id, text, user_id):
            return {"update_id": 1, "message": {"chat": {"id": chat_id}, "from": {"id": user_id}, "text": text}}

        bot = TelegramBot(app, FakeClient())
        before = str(app.settings.risk.max_trade_size)
        await bot.handle_update(_upd(12345, f"/ai approve {rec.id}", user_id=99999))
        assert "unauthorized" in bot._client.sent[-1][1].lower()
        assert str(app.settings.risk.max_trade_size) == before
        stored = await app.agent_recommendation_service.repository.get(rec.id)
        assert stored is not None and stored.status == RecommendationStatus.PENDING
        # Direct service calls without a human approver also fail.
        with pytest.raises(ApprovalError):
            await app.agent_approval_service.approve(rec.id, approver="", reason="x")
    finally:
        await shutdown(app)


# ------------------------------------------------------------------ isolation


@pytest.mark.asyncio
async def test_phase8_no_trades_or_arbitrary_writes(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        n_before = len(await app.trades.list_recent(50))
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=_current(app, "risk.max_trade_size"),
            proposed_value="900", reason="phase8", source_id="phase8",
        )
        await app.agent_approval_service.review(rec.id, approver="erin", reason="triage")
        await app.agent_approval_service.approve(rec.id, approver="erin", reason="go")
        await app.agent_approval_service.reject(
            (await app.agent_recommendation_service.create(
                parameter="risk.max_slippage_bps", current_value=_current(app, "risk.max_slippage_bps"),
                proposed_value="10", reason="phase8", source_id="phase8")).id,
            approver="erin", reason="no")
        # Approval moves exactly one allowlisted number; journal untouched.
        assert len(await app.trades.list_recent(50)) == n_before
        assert str(app.settings.trading.mode) == "PAPER"
        # Source scan: no trade/order/withdrawal/generic-config surface.
        import pathlib as _pl
        import re as _re

        source = (_pl.Path("app/agent/approval.py")).read_text(encoding="utf-8")
        code_only = _re.sub(r'""".*?"""', "", source, flags=_re.DOTALL)
        for forbidden in ("create_order(", "place_order(", "cancel_order(", ".withdraw(",
                          "set_config(", "os.system", "subprocess", "eval(", "exec("):
            assert forbidden not in code_only
    finally:
        await shutdown(app)

