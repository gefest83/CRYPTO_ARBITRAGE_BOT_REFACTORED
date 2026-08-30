"""Triangle execution in PAPER mode (integration through AppServices)."""

from decimal import Decimal

import pytest
from app.errors import ExecutionDisabledError
from app.services import AppServices

D = Decimal


async def test_paper_triangle_executes_profitably(services: AppServices):
    opportunities = await services.scan_triangles()
    assert opportunities, "the simulated venues must produce triangle rings"
    best = opportunities[0]
    trade, assessment = await services.execute_triangle(best)
    assert assessment is not None and assessment.approved
    assert trade.status.value == "completed"
    assert trade.input_amount > 0
    assert trade.output_amount > 0
    assert len(trade.orders) == 3  # all three legs recorded
    assert trade.net_profit > 0
    assert trade.net_profit_bps > 0


async def test_risk_rejects_oversized_trade(services: AppServices, tmp_path):
    opportunities = await services.scan_triangles(notional_quote=D("10"))
    assert opportunities
    # Force a notional far above the risk limit (default 1000).
    oversized = opportunities[0].model_copy(
        update={"size_notional_quote": D("999999")},
    )
    trade, assessment = await services.execute_triangle(oversized)
    assert assessment is not None and not assessment.approved
    assert trade.status.value == "failed"
    assert any("max_trade_size" in v.rule for v in assessment.violations)


async def test_risk_rejects_below_minimum_profit(services: AppServices):
    opportunities = await services.scan_triangles()
    assert opportunities
    weak = opportunities[-1].model_copy(
        update={"profit": opportunities[-1].profit.model_copy(update={"gross_spread_bps": D("0")})}
    )
    trade, assessment = await services.execute_triangle(weak)
    assert assessment is not None and not assessment.approved
    assert trade.status.value == "failed"


async def test_kill_switch_blocks_execution_before_any_order(services: AppServices):
    await services.engage_kill_switch("test kill")
    try:
        opportunities = await services.scan_triangles()
        assert opportunities
        with pytest.raises(ExecutionDisabledError):
            await services.execute_triangle(opportunities[0])
    finally:
        await services.release_kill_switch()


async def test_kill_switch_persists_across_restart(services: AppServices, tmp_path):
    await services.engage_kill_switch("persist me")
    from app.services import build_app, shutdown_app, start_app

    # rebuild a fresh app from the same database (simulated restart)
    settings = services.settings
    await shutdown_app(services)
    app2 = await build_app(settings.model_copy(update={"database": settings.database}))
    await start_app(app2)
    try:
        assert app2.guard.is_halted
        assert app2.guard.halt_reason == "persist me"
    finally:
        await app2.release_kill_switch()
        await shutdown_app(app2)


async def test_stale_market_data_refuses_execution(services: AppServices):
    """Books older than the staleness window must not be traded against."""
    opportunities = await services.scan_triangles()
    assert opportunities
    # Age every cached book beyond the window.
    from datetime import timedelta

    for key in list(services.store._books.keys()):
        book = services.store._books[key]
        services.store._books[key] = book.model_copy(
            update={"timestamp": book.timestamp - timedelta(seconds=30)}
        )
    trade, assessment = await services.execute_triangle(opportunities[0])
    if assessment is not None and assessment.approved:
        # risk passed (data age checked from the opportunity, which is fresh);
        # the executor must still refuse per-leg on stale books
        assert trade.status.value == "failed"
        assert "stale" in (trade.error or "").lower()
    else:
        assert any("max_data_age" in v.rule for v in assessment.violations)


async def test_paper_wallet_debits_and_credits(services: AppServices):
    wallet = services.paper_wallets["binance"]
    before = wallet.free("USDT")
    wallet.debit("USDT", D("100"))
    assert wallet.free("USDT") == before - D("100")
    wallet.credit("USDT", D("50"))
    assert wallet.free("USDT") == before - D("50")
    with pytest.raises(ValueError):
        wallet.debit("USDT", wallet.free("USDT") + D("1"))
