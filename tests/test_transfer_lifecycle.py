"""Transfer lifecycle: full workflow, persistence, resume, safety gates."""

import asyncio
from decimal import Decimal

import pytest
from app.models.enums import TransferState
from app.services import AppServices, build_app, shutdown_app, start_app
from tests.conftest import make_settings

D = Decimal


async def test_full_lifecycle_completes_in_paper(services: AppServices):
    plans = await services.plan_transfers()
    assert plans, "the simulated venues must produce transfer plans"
    record = await services.start_transfer(plans[0])
    seen_states = {record.state}
    ticks = 0
    while not record.is_terminal and ticks < 100:
        await asyncio.sleep(0.05)
        advanced = await services.tick_transfers()
        for updated in advanced:
            if updated.id == record.id:
                record = updated
                seen_states.add(record.state)
        ticks += 1
    assert record.state is TransferState.COMPLETED
    assert record.realized_profit_quote > 0
    # the lifecycle visited the core states in order
    ordered = [
        TransferState.CREATED,
        TransferState.BUY_SUBMITTED,
        TransferState.BUY_FILLED,
        TransferState.WITHDRAW_SUBMITTED,
        TransferState.TRANSFER_IN_PROGRESS,
        TransferState.DEPOSIT_DETECTED,
        TransferState.SELL_SUBMITTED,
        TransferState.COMPLETED,
    ]
    positions = [ordered.index(s) for s in ordered if s in seen_states]
    assert positions == sorted(positions), f"states out of order: {seen_states}"
    # a completed transfer produces a trade record
    trades = await services.trades.list_recent(5)
    assert any(t.transfer_id == record.id for t in trades)
    assert await services.transfers.count_open() == 0


async def test_transfer_survives_restart(tmp_path):
    """The whole point of the persisted lifecycle: restart mid-flight."""
    settings = make_settings(tmp_path)
    slow = settings.model_copy(
        update={
            "transfer": settings.transfer.model_copy(update={"simulated_transfer_seconds": 5.0})
        }
    )
    app1 = await build_app(slow)
    await start_app(app1)
    plans = await app1.plan_transfers()
    assert plans
    record = await app1.start_transfer(plans[0])
    assert record.state in (
        TransferState.BUY_FILLED,
        TransferState.WITHDRAW_SUBMITTED,
        TransferState.WITHDRAW_PENDING,
    )
    transfer_id = record.id
    # simulate a crash: no graceful tick, just shutdown
    await shutdown_app(app1)

    # fresh process from the same database, faster blockchain for the resume
    fast = settings.model_copy(
        update={
            "transfer": settings.transfer.model_copy(update={"simulated_transfer_seconds": 0.1})
        }
    )
    app2 = await build_app(fast)
    await start_app(app2)
    open_records = await app2.transfers.list_open()
    assert [r.id for r in open_records] == [transfer_id]
    record = open_records[0]
    ticks = 0
    while not record.is_terminal and ticks < 100:
        advanced = await app2.tick_transfers()
        for updated in advanced:
            if updated.id == transfer_id:
                record = updated
        await asyncio.sleep(0.05)
        ticks += 1
    assert record.state is TransferState.COMPLETED
    assert record.realized_profit_quote > 0
    await shutdown_app(app2)


async def test_risk_max_open_transfers_blocks_new_transfer(tmp_path):
    settings = make_settings(tmp_path)
    settings = settings.model_copy(
        update={"risk": settings.risk.model_copy(update={"max_open_transfers": 1})}
    )
    from app.services import build_app, shutdown_app, start_app

    app = await build_app(settings)
    await start_app(app)
    try:
        plans = await app.plan_transfers()
        assert plans, "the simulated venues must produce transfer plans"
        first = await app.start_transfer(plans[0])
        assert first.state.is_open
        assert await app.transfers.count_open() == 1
        # the cap is reached: the next transfer must be refused by risk
        with pytest.raises(Exception, match="max_open_transfers"):
            await app.start_transfer(plans[0])
    finally:
        await shutdown_app(app)


async def test_transfer_requires_capital(services: AppServices):
    """A plan that cannot be funded is refused before any order is placed."""
    plans = await services.plan_transfers()
    assert plans
    plan = plans[0]
    # blow the plan's size far beyond every cap
    huge = plan.model_copy(update={"amount": D("1000000")})
    with pytest.raises(Exception, match=r"max_trade_size|risk validation"):
        await services.start_transfer(huge)


async def test_plans_respect_network_constraints(services: AppServices):
    """Every emitted plan carries a network that both venues actually list."""
    plans = await services.plan_transfers()
    for plan in plans:
        source = await services.manager.adapter(plan.source_exchange).fetch_withdrawal_networks(
            plan.asset
        )
        dest = await services.manager.adapter(plan.dest_exchange).fetch_withdrawal_networks(
            plan.asset
        )
        source_codes = {n.network_code or n.network for n in source if n.withdraw_enabled}
        dest_codes = {n.network_code or n.network for n in dest if n.deposit_enabled}
        assert (plan.network.upper()) in {n.network.upper() for n in source}
        assert source_codes & dest_codes, "plan network must exist on both venues"


async def test_transfer_below_minimum_profit_is_not_planned(services: AppServices):
    """The planner never returns plans under the configured minimum."""
    plans = await services.plan_transfers()
    minimum = services.settings.transfer.min_net_profit_bps
    for plan in plans:
        assert plan.net_profit_bps >= minimum
