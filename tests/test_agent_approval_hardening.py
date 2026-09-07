"""Phase 8 hardening — crash-safe, persistent, idempotent, fail-closed approval.

Covers the three P8/P9 acceptance blockers:

1. Approved values persist across restart (durable ``bot_state``
   ``agent_config:<param>`` keys, re-applied on startup).
2. Approval/apply is atomic + idempotent (claim-before-apply, concurrent
   and duplicate approvals apply exactly once, DB failure applies nothing).
3. RiskEngine synchronization fails closed (never swallowed, never a false
   APPROVED, never silently divergent risk state).
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent.approval import ApprovalError, config_key
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


# ------------------------------------------------------------------ persistence


@pytest.mark.asyncio
async def test_hardening_approved_value_persists_and_restarts(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=_current(app, "risk.max_trade_size"),
            proposed_value="900", reason="hardening", source_id="hardening",
        )
        approved = await app.agent_approval_service.approve(rec.id, approver="alice", reason="go")
        assert approved.status == RecommendationStatus.APPROVED
        # Durable record exists alongside the in-memory change.
        assert await app.bot_state.get(config_key("risk.max_trade_size")) == "900"
        assert str(app.settings.risk.max_trade_size) == "900"
        assert str(app.risk.limits.max_trade_size) == "900"
    finally:
        await shutdown(app)

    # Restart: rebuild services from the SAME database file.
    app2 = await build_app(make_settings(tmp_path))
    await start_app(app2, start_telegram=False, start_streams=False)
    try:
        build_agent(app2)
        assert str(app2.settings.risk.max_trade_size) == "900"
        assert str(app2.risk.limits.max_trade_size) == "900"
        assert await app2.bot_state.get(config_key("risk.max_trade_size")) == "900"
        # Phase 9 confound check still sees the live approved value.
        assert app2.agent_learning._live_param("risk.max_trade_size") == "900"
    finally:
        await shutdown_app(app2)


@pytest.mark.asyncio
async def test_hardening_non_risk_param_persists_and_restarts(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="execution.leg_timeout_seconds",
            current_value=_current(app, "execution.leg_timeout_seconds"),
            proposed_value="25", reason="hardening", source_id="hardening",
        )
        await app.agent_approval_service.approve(rec.id, approver="alice", reason="go")
        assert await app.bot_state.get(config_key("execution.leg_timeout_seconds")) == "25"
    finally:
        await shutdown(app)

    app2 = await build_app(make_settings(tmp_path))
    await start_app(app2, start_telegram=False, start_streams=False)
    try:
        build_agent(app2)
        assert str(app2.settings.execution.leg_timeout_seconds) == "25"
    finally:
        await shutdown_app(app2)


# ------------------------------------------------------------------ idempotency


@pytest.mark.asyncio
async def test_hardening_duplicate_approval_applies_once(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_slippage_bps", current_value=_current(app, "risk.max_slippage_bps"),
            proposed_value="12", reason="hardening", source_id="hardening",
        )
        service = app.agent_approval_service
        calls = 0
        original = service._apply_value

        def counting(param, value):
            nonlocal calls
            calls += 1
            return original(param, value)

        service._apply_value = counting  # type: ignore[method-assign]
        try:
            first = await service.approve(rec.id, approver="alice", reason="go")
            assert first.status == RecommendationStatus.APPROVED
            with pytest.raises(ApprovalError):
                await service.approve(rec.id, approver="alice", reason="again")
        finally:
            service._apply_value = original  # type: ignore[method-assign]
        assert calls == 1
        stored = await app.agent_recommendation_service.repository.get(rec.id)
        assert stored is not None and stored.status == RecommendationStatus.APPROVED
        assert str(app.settings.risk.max_slippage_bps) == "12"
        assert str(app.risk.limits.max_slippage_bps) == "12"
        # Exactly one APPROVED audit for this recommendation.
        app_log = await app.audit.list_recent(limit=50)
        approved_marks = [
            e for e in app_log
            if e.action == "AGENT_RECOMMENDATION_APPROVED" and rec.id in str(e.context)
        ]
        assert len(approved_marks) == 1
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_hardening_concurrent_approve_single_winner(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_slippage_bps", current_value=_current(app, "risk.max_slippage_bps"),
            proposed_value="12", reason="hardening", source_id="hardening",
        )
        service = app.agent_approval_service
        calls = 0
        original = service._apply_value

        def counting(param, value):
            nonlocal calls
            calls += 1
            return original(param, value)

        service._apply_value = counting  # type: ignore[method-assign]
        try:
            results = await asyncio.gather(
                service.approve(rec.id, approver="alice", reason="go"),
                service.approve(rec.id, approver="bob", reason="go"),
                return_exceptions=True,
            )
        finally:
            service._apply_value = original  # type: ignore[method-assign]
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]
        assert len(successes) == 1
        assert len(failures) == 1 and isinstance(failures[0], ApprovalError)
        assert calls == 1
        stored = await app.agent_recommendation_service.repository.get(rec.id)
        assert stored is not None and stored.status == RecommendationStatus.APPROVED
        assert str(app.settings.risk.max_slippage_bps) == "12"
        assert str(app.risk.limits.max_slippage_bps) == "12"
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_hardening_approve_reject_race_single_terminal(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        before = _current(app, "risk.max_trade_size")
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=before,
            proposed_value="900", reason="hardening", source_id="hardening",
        )
        service = app.agent_approval_service
        results = await asyncio.gather(
            service.approve(rec.id, approver="alice", reason="go"),
            service.reject(rec.id, approver="bob", reason="no"),
            return_exceptions=True,
        )
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]
        assert len(successes) == 1
        assert len(failures) == 1 and isinstance(failures[0], ApprovalError)
        stored = await app.agent_recommendation_service.repository.get(rec.id)
        assert stored is not None
        assert stored.status in (RecommendationStatus.APPROVED, RecommendationStatus.REJECTED)
        # Final configuration matches the winning operation only.
        if stored.status == RecommendationStatus.APPROVED:
            assert str(app.settings.risk.max_trade_size) == "900"
            assert str(app.risk.limits.max_trade_size) == "900"
        else:
            assert str(app.settings.risk.max_trade_size) == before
            assert str(app.risk.limits.max_trade_size) == before
    finally:
        await shutdown(app)


# ------------------------------------------------------------------ failure modes


@pytest.mark.asyncio
async def test_hardening_db_failure_before_apply_changes_nothing(tmp_path):
    from contextlib import asynccontextmanager

    app, shutdown = await _wired_app(tmp_path)
    try:
        before = _current(app, "risk.max_trade_size")
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=before,
            proposed_value="900", reason="hardening", source_id="hardening",
        )
        service = app.agent_approval_service
        original_session = service._db.session
        entries = 0

        @asynccontextmanager
        async def failing_session():
            nonlocal entries
            entries += 1
            if entries >= 2:  # fail the claim transaction (after validation read)
                raise ApprovalError("injected durable failure")
            async with original_session() as session:
                yield session

        service._db.session = failing_session  # type: ignore[method-assign]
        try:
            with pytest.raises(ApprovalError):
                await service.approve(rec.id, approver="alice", reason="go")
        finally:
            service._db.session = original_session  # type: ignore[method-assign]
        # No configuration moved anywhere.
        assert str(app.settings.risk.max_trade_size) == before
        assert str(app.risk.limits.max_trade_size) == before
        assert await app.bot_state.get(config_key("risk.max_trade_size")) is None
        stored = await app.agent_recommendation_service.repository.get(rec.id)
        assert stored is not None and stored.status == RecommendationStatus.PENDING
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_hardening_config_persistence_failure_fails_closed(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        before = _current(app, "risk.max_trade_size")
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=before,
            proposed_value="900", reason="hardening", source_id="hardening",
        )
        service = app.agent_approval_service

        async def broken_persist(session, param, value_str):
            raise RuntimeError("injected persistence failure")

        service._persist_in_session = broken_persist  # type: ignore[method-assign]
        with pytest.raises(ApprovalError):
            await service.approve(rec.id, approver="alice", reason="go")
        # Fail-closed: no APPROVED, runtime and durable state untouched.
        stored = await app.agent_recommendation_service.repository.get(rec.id)
        assert stored is not None and stored.status != RecommendationStatus.APPROVED
        assert str(app.settings.risk.max_trade_size) == before
        assert str(app.risk.limits.max_trade_size) == before
        assert await app.bot_state.get(config_key("risk.max_trade_size")) is None
        # No false APPROVED audit for this recommendation.
        app_log = await app.audit.list_recent(limit=50)
        assert not any(
            e.action == "AGENT_RECOMMENDATION_APPROVED" and rec.id in str(e.context)
            for e in app_log
        )
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_hardening_risk_sync_failure_fails_closed(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        before = _current(app, "risk.max_slippage_bps")
        before_limits = str(app.risk.limits.max_slippage_bps)
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_slippage_bps", current_value=before,
            proposed_value="12", reason="hardening", source_id="hardening",
        )
        service = app.agent_approval_service

        def broken_apply(param, value):
            raise RuntimeError("injected risk synchronization failure")

        service._apply_value = broken_apply  # type: ignore[method-assign]
        with pytest.raises(ApprovalError):
            await service.approve(rec.id, approver="alice", reason="go")
        # Fail-closed: rolled back everywhere, exception not swallowed.
        stored = await app.agent_recommendation_service.repository.get(rec.id)
        assert stored is not None and stored.status != RecommendationStatus.APPROVED
        assert str(app.settings.risk.max_slippage_bps) == before
        assert str(app.risk.limits.max_slippage_bps) == before_limits
        assert await app.bot_state.get(config_key("risk.max_slippage_bps")) is None
        app_log = await app.audit.list_recent(limit=50)
        assert not any(
            e.action == "AGENT_RECOMMENDATION_APPROVED" and rec.id in str(e.context)
            for e in app_log
        )
        assert any(e.action == "AGENT_RECOMMENDATION_APPROVAL_FAILED" for e in app_log)
    finally:
        await shutdown(app)


@pytest.mark.asyncio
async def test_hardening_missing_risk_engine_fails_closed(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        before = _current(app, "risk.max_trade_size")
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=before,
            proposed_value="900", reason="hardening", source_id="hardening",
        )
        service = app.agent_approval_service
        saved_risk = app.risk
        object.__setattr__(app, "risk", None)
        try:
            with pytest.raises(ApprovalError):
                await service.approve(rec.id, approver="alice", reason="go")
        finally:
            object.__setattr__(app, "risk", saved_risk)
        stored = await app.agent_recommendation_service.repository.get(rec.id)
        assert stored is not None and stored.status != RecommendationStatus.APPROVED
        assert str(app.settings.risk.max_trade_size) == before
    finally:
        await shutdown(app)
