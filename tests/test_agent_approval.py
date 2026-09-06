"""Phase 2D — human-gated recommendation lifecycle tests."""

from __future__ import annotations

import asyncio
import pathlib
from decimal import Decimal

import pytest

from app.agent.approval import ALLOWLIST, ApprovalError, RecommendationApprovalService
from app.agent.models import AgentRecommendation, RecommendationStatus


async def _make_services(tmp_path: pathlib.Path):
    from tests.conftest import make_settings
    from app.services import build_app, start_app

    settings = make_settings(tmp_path)
    services = await build_app(settings)
    await start_app(services, start_telegram=False, start_streams=False)
    # wire agent
    from app.agent import build_agent

    build_agent(services)
    return services


@pytest.mark.asyncio
async def test_approval_happy_path_and_audit(tmp_path):
    services = await _make_services(tmp_path)
    try:
        rec_svc = services.agent_recommendation_service
        approval = services.agent_approval_service
        # create a valid recommendation
        rec = await rec_svc.create(
            parameter="risk.max_trade_size",
            current_value=str(services.settings.risk.max_trade_size),
            proposed_value="900",
            reason="test approve",
            evidence=("t1",),
            confidence=0.8,
            source_id="test",
        )
        assert rec.status == RecommendationStatus.PENDING
        # approve as human
        approved = await approval.approve(rec.id, approver="human-1", reason="looks good")
        assert approved.status == RecommendationStatus.APPROVED
        assert approved.operator_decision == "approved by human-1"
        assert approved.result == "applied risk.max_trade_size=900"
        # config actually changed (allowlisted)
        assert str(services.settings.risk.max_trade_size) == "900"
        # risk engine also updated
        assert str(services.risk.limits.max_trade_size) == "900"
        # audit log written
        logs = await services.audit.list_recent(20)
        assert any("AGENT_RECOMMENDATION_APPROVED" in r.action for r in logs)
    finally:
        from app.services import shutdown_app

        await shutdown_app(services)


@pytest.mark.asyncio
async def test_double_approval_rejected(tmp_path):
    services = await _make_services(tmp_path)
    try:
        rec_svc = services.agent_recommendation_service
        approval = services.agent_approval_service
        rec = await rec_svc.create(
            parameter="risk.max_trade_size",
            current_value=str(services.settings.risk.max_trade_size),
            proposed_value="800",
            reason="double",
            source_id="test",
        )
        await approval.approve(rec.id, approver="alice", reason="first")
        with pytest.raises(ApprovalError, match="not PENDING"):
            await approval.approve(rec.id, approver="alice", reason="second")
        # Ensure config not double-applied (still 800)
        assert str(services.settings.risk.max_trade_size) == "800"
    finally:
        from app.services import shutdown_app

        await shutdown_app(services)


@pytest.mark.asyncio
async def test_invalid_parameter_rejected(tmp_path):
    services = await _make_services(tmp_path)
    try:
        rec_svc = services.agent_recommendation_service
        approval = services.agent_approval_service
        rec = await rec_svc.create(
            parameter="risk.unknown_param",
            current_value="1",
            proposed_value="2",
            reason="invalid",
            source_id="test",
        )
        with pytest.raises(ApprovalError, match="not allowlisted"):
            await approval.approve(rec.id, approver="human", reason="try")
        # status still pending
        from app.agent.recommendations import RecommendationRepository

        repo = RecommendationRepository(services.db)
        loaded = await repo.get(rec.id)
        assert loaded is not None and loaded.status == RecommendationStatus.PENDING
    finally:
        from app.services import shutdown_app

        await shutdown_app(services)


@pytest.mark.asyncio
async def test_invalid_value_out_of_bounds(tmp_path):
    services = await _make_services(tmp_path)
    try:
        rec_svc = services.agent_recommendation_service
        approval = services.agent_approval_service
        # max_trade_size bounds 10..5000, try 99999
        rec = await rec_svc.create(
            parameter="risk.max_trade_size",
            current_value=str(services.settings.risk.max_trade_size),
            proposed_value="99999",
            reason="out of bounds",
            source_id="test",
        )
        with pytest.raises(ApprovalError, match="out of bounds"):
            await approval.approve(rec.id, approver="human", reason="try")
        # also test int param with float string
        rec2 = await rec_svc.create(
            parameter="risk.max_data_age_ms",
            current_value=str(services.settings.risk.max_data_age_ms),
            proposed_value="12.5",  # int expected float string -> invalid
            reason="invalid int",
            source_id="test",
        )
        with pytest.raises(ApprovalError, match="invalid int"):
            await approval.approve(rec2.id, approver="human", reason="try")
    finally:
        from app.services import shutdown_app

        await shutdown_app(services)


@pytest.mark.asyncio
async def test_stale_recommendation_rejected_when_config_changed(tmp_path):
    services = await _make_services(tmp_path)
    try:
        rec_svc = services.agent_recommendation_service
        approval = services.agent_approval_service
        # Create rec with current 1000
        rec = await rec_svc.create(
            parameter="risk.max_trade_size",
            current_value="1000",
            proposed_value="900",
            reason="stale",
            source_id="test",
        )
        # Simulate config changed externally before approval (e.g., manual edit)
        # Change live setting to 1200 (different from stored 1000)
        settings = services.settings
        new_settings = settings.model_copy(update={"risk": settings.risk.model_copy(update={"max_trade_size": Decimal("1200")})})
        object.__setattr__(services, "settings", new_settings)
        # Approval should fail stale check
        with pytest.raises(ApprovalError, match="current config changed"):
            await approval.approve(rec.id, approver="human", reason="try")
    finally:
        from app.services import shutdown_app

        await shutdown_app(services)


@pytest.mark.asyncio
async def test_already_applied_rejected(tmp_path):
    # Same as double approval but after restart
    services = await _make_services(tmp_path)
    try:
        rec_svc = services.agent_recommendation_service
        approval = services.agent_approval_service
        rec = await rec_svc.create(
            parameter="risk.min_net_profit_bps",
            current_value=str(services.settings.risk.min_net_profit_bps),
            proposed_value="15",
            reason="already applied",
            source_id="test",
        )
        await approval.approve(rec.id, approver="human", reason="first")
        with pytest.raises(ApprovalError):
            await approval.approve(rec.id, approver="human", reason="second")
    finally:
        from app.services import shutdown_app

        await shutdown_app(services)


@pytest.mark.asyncio
async def test_restart_between_pending_and_approve(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, start_app, shutdown_app
    from app.agent import build_agent

    settings = make_settings(tmp_path)
    services = await build_app(settings)
    await start_app(services, start_telegram=False, start_streams=False)
    build_agent(services)
    rec_svc = services.agent_recommendation_service
    rec = await rec_svc.create(
        parameter="risk.max_slippage_bps",
        current_value=str(services.settings.risk.max_slippage_bps),
        proposed_value="12",
        reason="restart test",
        source_id="test",
    )
    rec_id = rec.id
    expected_current = str(services.settings.risk.max_slippage_bps)
    await shutdown_app(services)

    # Restart with same DB file
    services2 = await build_app(settings)
    await start_app(services2, start_telegram=False, start_streams=False)
    try:
        build_agent(services2)
        approval2 = services2.agent_approval_service
        # Recommendation still pending after restart
        from app.agent.recommendations import RecommendationRepository

        repo = RecommendationRepository(services2.db)
        loaded = await repo.get(rec_id)
        assert loaded is not None and loaded.status == RecommendationStatus.PENDING
        # Approve now
        approved = await approval2.approve(rec_id, approver="human-restart", reason="after restart")
        assert approved.status == RecommendationStatus.APPROVED
        assert str(services2.settings.risk.max_slippage_bps) == "12"
    finally:
        await shutdown_app(services2)


@pytest.mark.asyncio
async def test_failure_during_apply_keeps_pending(tmp_path):
    services = await _make_services(tmp_path)
    try:
        rec_svc = services.agent_recommendation_service
        approval = services.agent_approval_service
        # Create valid rec, but monkey-patch _apply_value to fail
        rec = await rec_svc.create(
            parameter="risk.max_trade_size",
            current_value=str(services.settings.risk.max_trade_size),
            proposed_value="800",
            reason="apply failure",
            source_id="test",
        )
        original_apply = approval._apply_value

        def failing_apply(param, value):
            raise RuntimeError("simulated apply crash")

        approval._apply_value = failing_apply  # type: ignore[assignment]
        with pytest.raises(ApprovalError, match="apply failed"):
            await approval.approve(rec.id, approver="human", reason="try")
        # Should remain pending
        from app.agent.recommendations import RecommendationRepository

        repo = RecommendationRepository(services.db)
        loaded = await repo.get(rec.id)
        assert loaded is not None and loaded.status == RecommendationStatus.PENDING
        approval._apply_value = original_apply
    finally:
        from app.services import shutdown_app

        await shutdown_app(services)


@pytest.mark.asyncio
async def test_audit_failure_does_not_rollback_approval(tmp_path):
    services = await _make_services(tmp_path)
    try:
        rec_svc = services.agent_recommendation_service
        approval = services.agent_approval_service
        rec = await rec_svc.create(
            parameter="risk.max_trade_size",
            current_value=str(services.settings.risk.max_trade_size),
            proposed_value="800",
            reason="audit fail",
            source_id="test",
        )
        # Make audit.log fail
        original_log = services.audit.log

        async def failing_log(*a, **kw):
            raise RuntimeError("audit crash")

        services.audit.log = failing_log  # type: ignore[assignment]
        approved = await approval.approve(rec.id, approver="human", reason="audit fail test")
        assert approved.status == RecommendationStatus.APPROVED
        assert str(services.settings.risk.max_trade_size) == "800"
        services.audit.log = original_log
    finally:
        from app.services import shutdown_app

        await shutdown_app(services)


@pytest.mark.asyncio
async def test_unauthorized_telegram_approval_blocked(tmp_path):
    services = await _make_services(tmp_path)
    try:
        rec_svc = services.agent_recommendation_service
        rec = await rec_svc.create(
            parameter="risk.max_trade_size",
            current_value=str(services.settings.risk.max_trade_size),
            proposed_value="800",
            reason="telegram auth",
            source_id="test",
        )
        # Bot with allow-list 11111 only; 99999 is stranger
        from app.telegram.bot import TelegramBot
        from app.telegram.i18n import lang_storage_key

        await services.bot_state.set(lang_storage_key(11111), "en")
        await services.bot_state.set(lang_storage_key(99999), "en")

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

        bot = TelegramBot(services, FakeClient())
        client = bot._client  # type: ignore[attr-defined]
        # unauthorized approve
        await bot.handle_update(_upd(12345, f"/ai approve {rec.id}", user_id=99999))
        assert "unauthorized" in client.sent[-1][1].lower()
        # ensure not applied
        from app.agent.recommendations import RecommendationRepository

        repo = RecommendationRepository(services.db)
        loaded = await repo.get(rec.id)
        assert loaded is not None and loaded.status == RecommendationStatus.PENDING
        assert str(services.settings.risk.max_trade_size) != "800"
        # authorized approve succeeds
        client.sent.clear()
        await bot.handle_update(_upd(12345, f"/ai approve {rec.id}", user_id=11111))
        txt = client.sent[-1][1]
        assert "Approved" in txt or "Одобрено" in txt
        loaded2 = await repo.get(rec.id)
        assert loaded2 is not None and loaded2.status == RecommendationStatus.APPROVED
    finally:
        from app.services import shutdown_app

        await shutdown_app(services)


@pytest.mark.asyncio
async def test_concurrent_approvals_only_one_succeeds(tmp_path):
    services = await _make_services(tmp_path)
    try:
        rec_svc = services.agent_recommendation_service
        approval = services.agent_approval_service
        rec = await rec_svc.create(
            parameter="risk.max_data_age_ms",
            current_value=str(services.settings.risk.max_data_age_ms),
            proposed_value="2000",
            reason="concurrent",
            source_id="test",
        )

        async def try_approve(name: str):
            try:
                return await approval.approve(rec.id, approver=name, reason="concurrent")
            except ApprovalError as e:
                return e

        results = await asyncio.gather(try_approve("alice"), try_approve("bob"))
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]
        # Exactly one should succeed, one should fail (duplicate or stale due to live config change)
        assert len(successes) == 1
        assert len(failures) == 1
        msg = str(failures[0])
        assert ("not PENDING" in msg or "current config changed" in msg or "concurrent" in msg)
    finally:
        from app.services import shutdown_app

        await shutdown_app(services)


@pytest.mark.asyncio
async def test_risk_limits_require_explicit_validation(tmp_path):
    services = await _make_services(tmp_path)
    try:
        rec_svc = services.agent_recommendation_service
        approval = services.agent_approval_service
        # Try to approve a risk param with valid value -> should succeed via explicit risk path
        rec = await rec_svc.create(
            parameter="risk.max_open_transfers",
            current_value=str(services.settings.risk.max_open_transfers),
            proposed_value="5",
            reason="risk explicit",
            source_id="test",
        )
        approved = await approval.approve(rec.id, approver="human", reason="risk")
        assert approved.status == RecommendationStatus.APPROVED
        assert str(services.settings.risk.max_open_transfers) == "5"
        # Ensure no generic set_config exists on approval service
        assert not hasattr(approval, "set_config")
        assert not hasattr(approval, "mutate_config")
    finally:
        from app.services import shutdown_app

        await shutdown_app(services)


@pytest.mark.asyncio
async def test_no_approval_via_ai_tools(tmp_path):
    services = await _make_services(tmp_path)
    try:
        from app.agent.tools import AgentTools, ToolAccessBlocked

        tools = AgentTools(services)
        # AI tools must not expose approval
        assert "approve" not in tools.allowed_tools
        assert "set_config" not in tools.allowed_tools
        # Even via __getattr__ should raise
        with pytest.raises(ToolAccessBlocked):
            getattr(tools, "approve")
        with pytest.raises(ToolAccessBlocked):
            getattr(tools, "set_config")
    finally:
        from app.services import shutdown_app

        await shutdown_app(services)
