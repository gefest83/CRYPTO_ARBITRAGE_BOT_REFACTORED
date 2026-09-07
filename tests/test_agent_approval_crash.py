"""Phase 8 crash-consistency — deterministic recovery from durable state alone.

Authoritative durable state (existing ``bot_state`` table, same database):

* ``agent_config:<param>`` — validated desired value;
* ``agent_apply:<id>`` — ``applying`` / ``applied`` / ``failed`` record.

Every test below simulates a crash at a different boundary and proves the
invariant: at every process boundary, durable config, recommendation state,
runtime settings and RiskEngine converge to the same authoritative config.

A test that only asserts ``ApprovalError`` is insufficient here — each test
verifies actual DB / durable / settings / risk state.
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent.approval import (
    APPLY_STATUS_APPLIED,
    APPLY_STATUS_APPLYING,
    APPLY_STATUS_FAILED,
    ApprovalError,
    apply_key,
    config_key,
)
from app.agent.models import RecommendationStatus


class _SimulatedCrash(BaseException):
    """Process crash: bypasses every ``except Exception`` handler by construction."""


async def _wired_app(tmp_path, **agent_overrides):
    from app.agent import build_agent
    from app.services import build_app, shutdown_app, start_app
    from tests.conftest import make_settings

    base = make_settings(tmp_path)
    if agent_overrides:
        base = base.model_copy(update={"agent": base.agent.model_copy(update=agent_overrides)})
    app = await build_app(base)
    await start_app(app, start_telegram=False, start_streams=False)
    build_agent(app)
    return app, shutdown_app


async def _restart(tmp_path, shutdown, app):
    """Simulate a process restart: dispose everything, rebuild from the same DB file."""
    from app.agent import build_agent
    from app.services import build_app, shutdown_app, start_app
    from tests.conftest import make_settings

    await shutdown(app)
    app2 = await build_app(make_settings(tmp_path))
    await start_app(app2, start_telegram=False, start_streams=False)
    build_agent(app2)
    return app2, shutdown_app


def _current(app, parameter):
    section, _, name = parameter.partition(".")
    return str(getattr(getattr(app.settings, section), name))


def _risk_limit(app, parameter):
    return str(getattr(app.risk.limits, parameter.partition(".")[2]))


# ------------------------------------------------------------------ Test A


@pytest.mark.asyncio
async def test_crash_a_before_durable_commit_changes_nothing(tmp_path):
    """Crash before the claim transaction commits: no approval, no config."""
    from contextlib import asynccontextmanager

    app, shutdown = await _wired_app(tmp_path)
    try:
        before = _current(app, "risk.max_trade_size")
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=before,
            proposed_value="900", reason="crash", source_id="crash",
        )
        service = app.agent_approval_service
        original_session = service._db.session
        entries = 0

        @asynccontextmanager
        async def failing_session():
            nonlocal entries
            entries += 1
            if entries >= 2:  # crash inside the claim transaction
                raise _SimulatedCrash("crash before durable commit")
            async with original_session() as session:
                yield session

        service._db.session = failing_session  # type: ignore[method-assign]
        try:
            with pytest.raises(_SimulatedCrash):
                await service.approve(rec.id, approver="alice", reason="go")
        finally:
            service._db.session = original_session  # type: ignore[method-assign]
        # Actual state: nothing durably decided, nothing applied anywhere.
        stored = await app.agent_recommendation_service.repository.get(rec.id)
        assert stored is not None and stored.status == RecommendationStatus.PENDING
        assert await app.bot_state.get(config_key("risk.max_trade_size")) is None
        assert await app.bot_state.get(apply_key(rec.id)) is None
        assert _current(app, "risk.max_trade_size") == before
        assert _risk_limit(app, "risk.max_trade_size") == before
    finally:
        await shutdown(app)


# ------------------------------------------------------------------ Test B


@pytest.mark.asyncio
async def test_crash_b_after_durable_claim_recovers_on_restart(tmp_path):
    """Crash after commit, before runtime sync: restart converges deterministically."""
    app, shutdown = await _wired_app(tmp_path)
    try:
        before = _current(app, "risk.max_trade_size")
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=before,
            proposed_value="900", reason="crash", source_id="crash",
        )
        service = app.agent_approval_service

        def crashing_apply(param, value):
            raise _SimulatedCrash("crash after durable commit, before runtime sync")

        service._apply_value = crashing_apply  # type: ignore[method-assign]
        with pytest.raises(_SimulatedCrash):
            await service.approve(rec.id, approver="alice", reason="go")
        # Interrupted durable state, runtime untouched (the dangerous window).
        stored = await app.agent_recommendation_service.repository.get(rec.id)
        assert stored is not None and stored.status == RecommendationStatus.APPROVED
        assert await app.bot_state.get(config_key("risk.max_trade_size")) == "900"
        record = await app.bot_state.get(apply_key(rec.id))
        assert isinstance(record, dict) and record["status"] == APPLY_STATUS_APPLYING
        assert _current(app, "risk.max_trade_size") == before
        assert _risk_limit(app, "risk.max_trade_size") == before
    finally:
        pass  # no shutdown: the "crashed" process never cleans up

    # Restart: reconciliation must finish the interrupted application.
    app2, shutdown_app2 = await _restart(tmp_path, shutdown, app)
    try:
        assert _current(app2, "risk.max_trade_size") == "900"
        assert _risk_limit(app2, "risk.max_trade_size") == "900"
        assert await app2.bot_state.get(config_key("risk.max_trade_size")) == "900"
        record2 = await app2.bot_state.get(apply_key(rec.id))
        assert isinstance(record2, dict) and record2["status"] == APPLY_STATUS_APPLIED
        # Recovery was audited explicitly.
        app_log = await app2.audit.list_recent(limit=50)
        assert any(
            e.action == "AGENT_RECOMMENDATION_APPLY_RECOVERED" and rec.id in str(e.context)
            for e in app_log
        )
        # Duplicate approval after recovery still cannot re-apply.
        with pytest.raises(ApprovalError):
            await app2.agent_approval_service.approve(rec.id, approver="alice", reason="again")
        assert _current(app2, "risk.max_trade_size") == "900"
    finally:
        await shutdown_app2(app2)


# ------------------------------------------------------------------ Test C


@pytest.mark.asyncio
async def test_crash_c_during_risk_sync_recovers_on_restart(tmp_path):
    """Crash mid-sync (settings NEW, risk OLD): never silently accepted, restart converges."""
    from decimal import Decimal

    app, shutdown = await _wired_app(tmp_path)
    try:
        before = _current(app, "risk.max_slippage_bps")
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_slippage_bps", current_value=before,
            proposed_value="12", reason="crash", source_id="crash",
        )
        service = app.agent_approval_service

        def partial_apply(param, value):
            settings = app.settings
            object.__setattr__(app, "settings", settings.model_copy(update={
                "risk": settings.risk.model_copy(update={"max_slippage_bps": Decimal("12")})}))
            raise _SimulatedCrash("crash between settings and RiskEngine update")

        service._apply_value = partial_apply  # type: ignore[method-assign]
        with pytest.raises(_SimulatedCrash):
            await service.approve(rec.id, approver="alice", reason="go")
        # The dangerous partial state exists pre-restart (not silently accepted later).
        assert _current(app, "risk.max_slippage_bps") == "12"
        assert _risk_limit(app, "risk.max_slippage_bps") == before
    finally:
        pass  # crashed process never cleans up

    app2, shutdown_app2 = await _restart(tmp_path, shutdown, app)
    try:
        assert _current(app2, "risk.max_slippage_bps") == "12"
        assert _risk_limit(app2, "risk.max_slippage_bps") == "12"
        record2 = await app2.bot_state.get(apply_key(rec.id))
        assert isinstance(record2, dict) and record2["status"] == APPLY_STATUS_APPLIED
    finally:
        await shutdown_app2(app2)


# ------------------------------------------------------------------ Test D


@pytest.mark.asyncio
async def test_crash_d_stale_runtime_reconciles_to_desired(tmp_path):
    """persistent=NEW with settings/risk OLD converges fully on startup."""
    app, shutdown = await _wired_app(tmp_path)
    try:
        await app.bot_state.set(config_key("risk.max_daily_loss"), "500")
        assert _current(app, "risk.max_daily_loss") != "500"
        assert _risk_limit(app, "risk.max_daily_loss") != "500"
    finally:
        pass

    app2, shutdown_app2 = await _restart(tmp_path, shutdown, app)
    try:
        assert await app2.bot_state.get(config_key("risk.max_daily_loss")) == "500"
        assert _current(app2, "risk.max_daily_loss") == "500"
        assert _risk_limit(app2, "risk.max_daily_loss") == "500"
    finally:
        await shutdown_app2(app2)


# ------------------------------------------------------------------ Test E


@pytest.mark.asyncio
async def test_crash_e_corrupt_durable_state_fails_closed(tmp_path):
    """Invalid persisted desired config refuses startup instead of unsafe defaults."""
    from app.agent.approval import reconcile_approved_config
    from app.services import build_app
    from tests.conftest import make_settings

    app, shutdown = await _wired_app(tmp_path)
    try:
        before = _current(app, "risk.max_trade_size")
        await app.bot_state.set(config_key("risk.max_trade_size"), "99999")  # out of bounds
        # Direct reconciliation fails closed with an explicit error.
        with pytest.raises(ApprovalError, match="invalid"):
            await reconcile_approved_config(app)
        # The corrupt value was never applied to runtime.
        assert _current(app, "risk.max_trade_size") == before
        assert _risk_limit(app, "risk.max_trade_size") == before
    finally:
        await shutdown(app)

    # A fresh boot with the corrupt key refuses startup (no leak: db disposed).
    with pytest.raises(ApprovalError, match="invalid"):
        await build_app(make_settings(tmp_path))


# ------------------------------------------------------------------ Test F


@pytest.mark.asyncio
async def test_crash_f_duplicate_approval_after_restart_applies_once(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_slippage_bps", current_value=_current(app, "risk.max_slippage_bps"),
            proposed_value="12", reason="crash", source_id="crash",
        )
        approved = await app.agent_approval_service.approve(rec.id, approver="alice", reason="go")
        assert approved.status == RecommendationStatus.APPROVED
        record = await app.bot_state.get(apply_key(rec.id))
        assert isinstance(record, dict) and record["status"] == APPLY_STATUS_APPLIED
    finally:
        pass

    app2, shutdown_app2 = await _restart(tmp_path, shutdown, app)
    try:
        with pytest.raises(ApprovalError):
            await app2.agent_approval_service.approve(rec.id, approver="alice", reason="again")
        # Exactly one application, one APPROVED audit, converged state.
        assert _current(app2, "risk.max_slippage_bps") == "12"
        assert _risk_limit(app2, "risk.max_slippage_bps") == "12"
        app_log = await app2.audit.list_recent(limit=50)
        assert len([e for e in app_log
                    if e.action == "AGENT_RECOMMENDATION_APPROVED" and rec.id in str(e.context)]) == 1
    finally:
        await shutdown_app2(app2)


# ------------------------------------------------------------------ Test G


@pytest.mark.asyncio
async def test_crash_g_concurrent_approve_reject_single_terminal(tmp_path):
    app, shutdown = await _wired_app(tmp_path)
    try:
        before = _current(app, "risk.max_trade_size")
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=before,
            proposed_value="900", reason="crash", source_id="crash",
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
        if stored.status == RecommendationStatus.APPROVED:
            assert _current(app, "risk.max_trade_size") == "900"
            assert _risk_limit(app, "risk.max_trade_size") == "900"
            record = await app.bot_state.get(apply_key(rec.id))
            assert isinstance(record, dict) and record["status"] == APPLY_STATUS_APPLIED
        else:
            assert stored.status == RecommendationStatus.REJECTED
            assert _current(app, "risk.max_trade_size") == before
            assert _risk_limit(app, "risk.max_trade_size") == before
            assert await app.bot_state.get(apply_key(rec.id)) is None
    finally:
        await shutdown(app)


# ------------------------------------------------------------------ Test H


@pytest.mark.asyncio
async def test_crash_h_phase9_measures_restored_value_after_restart(tmp_path):
    """Phase 9 still observes the approved effective value after a restart."""
    from decimal import Decimal

    from app.agent.models import MeasurementOutcome
    from app.models.enums import ArbitrageStrategy, TradeStatus, TradingMode
    from app.models.trade import TradeRecord

    def _trade(**overrides):
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

    app, shutdown = await _wired_app(tmp_path)
    try:
        for _ in range(6):
            await app.trades.save(_trade(slippage_bps=Decimal("40")))
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_slippage_bps", current_value="15", proposed_value="12",
            reason="crash", source_id="phase7:slippage",
        )
        approved = await app.agent_approval_service.approve(rec.id, approver="alice", reason="go")
        assert approved.status == RecommendationStatus.APPROVED
    finally:
        pass

    app2, shutdown_app2 = await _restart(tmp_path, shutdown, app)
    try:
        # Restored effective config is what measurement must observe.
        assert _current(app2, "risk.max_slippage_bps") == "12"
        assert _risk_limit(app2, "risk.max_slippage_bps") == "12"
        for _ in range(6):
            await app2.trades.save(_trade(slippage_bps=Decimal("8")))
        m = await app2.agent_learning.measure(rec.id)
        # IMPROVED (not confounded/INSUFFICIENT) proves the durable-backed
        # approved value survived the restart and stayed effective.
        assert m.outcome == MeasurementOutcome.IMPROVED
        assert m.new_value == "12"
        assert Decimal(m.before_value) == Decimal("40")
        assert Decimal(m.after_value) == Decimal("8")
    finally:
        await shutdown_app2(app2)


# ------------------------------------------------------------------ failed-mark recovery


@pytest.mark.asyncio
async def test_crash_failed_mark_recovers_on_restart(tmp_path):
    """A durably FAILED application converges on restart via desired config."""
    app, shutdown = await _wired_app(tmp_path)
    try:
        before = _current(app, "risk.max_trade_size")
        rec = await app.agent_recommendation_service.create(
            parameter="risk.max_trade_size", current_value=before,
            proposed_value="900", reason="crash", source_id="crash",
        )
        service = app.agent_approval_service

        def broken_apply(param, value):
            raise RuntimeError("injected sync failure")

        service._apply_value = broken_apply  # type: ignore[method-assign]
        with pytest.raises(ApprovalError):
            await service.approve(rec.id, approver="alice", reason="go")
        record = await app.bot_state.get(apply_key(rec.id))
        assert isinstance(record, dict) and record["status"] == APPLY_STATUS_FAILED
        assert _current(app, "risk.max_trade_size") == before
    finally:
        pass

    app2, shutdown_app2 = await _restart(tmp_path, shutdown, app)
    try:
        # Desired config is authoritative: restart applies what approval granted.
        assert _current(app2, "risk.max_trade_size") == "900"
        assert _risk_limit(app2, "risk.max_trade_size") == "900"
        record2 = await app2.bot_state.get(apply_key(rec.id))
        assert isinstance(record2, dict) and record2["status"] == APPLY_STATUS_APPLIED
    finally:
        await shutdown_app2(app2)
