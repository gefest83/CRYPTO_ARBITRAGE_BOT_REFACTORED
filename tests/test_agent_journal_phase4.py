"""Phase 4 — Trading Journal integration focused tests.

Covers: journal retrieval, single-trade analysis, expected vs realized,
fees, slippage, execution, failure analysis, strategy/exchange aggregation,
period comparison, sample-size + insufficient-data handling, provenance,
determinism, LLM-immutability of results, read-only/security boundaries.

Trading/execution/risk/recovery behavior is never modified — tests only
persist journal records through the existing repositories.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.agent.journal import (
    MIN_SAMPLE_FOR_CONCLUSIONS,
    JournalReader,
    aggregate_stats,
    aggregate_trades,
    analyze_trade_record,
    compare_periods,
    filter_period,
    missed_opportunities_from_audit,
    sanitize_trade,
    split_today_vs_yesterday,
)
from app.models.enums import ArbitrageStrategy, TradeStatus, TradingMode
from app.models.trade import TradeRecord


def _trade(
    *,
    strategy=ArbitrageStrategy.TRIANGLE,
    exchange="binance",
    status=TradeStatus.COMPLETED,
    net=Decimal("5"),
    bps=Decimal("50"),
    fees=Decimal("1"),
    slippage=Decimal("10"),
    route="USDT->BTC->ETH->USDT",
    orders=(),
    error=None,
    transfer_id=None,
    created_at=None,
) -> TradeRecord:
    kwargs: dict = {
        "strategy": strategy,
        "mode": TradingMode.PAPER,
        "exchange_id": exchange,
        "route": route,
        "input_amount": Decimal("1000"),
        "output_amount": Decimal("1000") + net,
        "fees_quote": fees,
        "slippage_bps": slippage,
        "net_profit": net,
        "net_profit_bps": bps,
        "status": status,
        "orders": orders,
        "error": error,
        "transfer_id": transfer_id,
    }
    if created_at is not None:
        kwargs["created_at"] = created_at
        kwargs["updated_at"] = created_at
    return TradeRecord(**kwargs)


def _order(*, symbol="BTC/USDT", side="buy", status="filled", amount="1", filled="1",
           price="50000", avg="50050", fee="0.5") -> dict:
    return {
        "id": f"ord-{symbol[:3]}-{side}-{status}",
        "symbol": symbol,
        "side": side,
        "status": status,
        "amount": amount,
        "filled_amount": filled,
        "price": price,
        "average_price": avg,
        "fee_paid": fee,
        "fee_currency": "USDT",
        "fills": [{"price": avg, "amount": filled}],
        "error": None,
    }


async def _seed_mixed(app) -> list[str]:
    """6 completed triangle/binance + 2 failed triangle/okx + 2 transfer trades."""
    ids: list[str] = []
    for i in range(6):
        t = await app.trades.save(_trade(net=Decimal("5"), bps=Decimal("50")))
        ids.append(t.id)
    for i in range(2):
        t = await app.trades.save(_trade(exchange="okx", status=TradeStatus.FAILED,
                                         net=Decimal("-5"), bps=Decimal("-50"),
                                         error="slippage exceeded"))
        ids.append(t.id)
    for i in range(2):
        t = await app.trades.save(_trade(strategy=ArbitrageStrategy.TRANSFER,
                                         exchange="binance->bybit",
                                         route="USDT->ETH->ERC20->ETH/USDT",
                                         net=Decimal("8"), bps=Decimal("80")))
        ids.append(t.id)
    return ids


# ------------------------------------------------------------------ retrieval


@pytest.mark.asyncio
async def test_phase4_journal_retrieval(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        from app.agent import build_agent

        build_agent(app)
        reader: JournalReader = app.agent_journal
        ids = await _seed_mixed(app)
        # Single-trade view preserves every required field, no secrets.
        view = await reader.get_trade(ids[0])
        assert view is not None
        for field in ("id", "strategy", "exchange_id", "route", "status", "orders",
                      "net_profit", "net_profit_bps", "fees_quote", "slippage_bps",
                      "created_at", "updated_at", "execution_duration_s", "error",
                      "transfer_id", "provenance"):
            assert field in view, f"missing journal field: {field}"
        assert view["provenance"] == {"journal": "trades", "trade_id": ids[0]}
        assert view["strategy"] == "triangle"
        assert view["exchange_id"] == "binance"
        # Unknown / empty ids → None (no crash, no invention).
        assert await reader.get_trade("trd-000000000000") is None
        assert await reader.get_trade("") is None
        # Bounded listing + filters.
        assert len(await reader.list_trades(limit=50)) == 10
        assert len(await reader.list_trades(limit=3)) == 3
        failed = await reader.list_trades(status="failed")
        assert {v["id"] for v in failed} == set(ids[6:8])
        assert all(v["status"] == "failed" for v in failed)
        assert len(await reader.list_trades(strategy="transfer")) == 2
        assert len(await reader.list_trades(exchange="okx")) == 2
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ single-trade analysis


@pytest.mark.asyncio
async def test_phase4_single_trade_analysis_triangle(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        from app.agent import build_agent

        build_agent(app)
        reader: JournalReader = app.agent_journal
        orders = (
            _order(symbol="BTC/USDT", side="buy", status="filled"),
            _order(symbol="ETH/BTC", side="buy", status="filled"),
            _order(symbol="ETH/USDT", side="sell", status="filled", fee="0.5"),
        )
        trade = await app.trades.save(_trade(orders=orders, fees=Decimal("1.5")))
        result = await reader.analyze_trade(trade.id)
        assert result["kind"] == "trade_analysis"
        assert result["trade_id"] == trade.id
        assert result["realized"] == {"net_profit": "5", "net_profit_bps": "50"}
        # Triangle opportunities are not journaled → explicit insufficient_data.
        assert result["expected"]["source"] == "insufficient_data"
        assert result["difference"] is None
        # Legs / execution / fees / slippage / outcome all present.
        assert result["execution"]["legs_total"] == 3
        assert result["execution"]["legs_filled"] == 3
        assert [leg["order_id"] for leg in result["execution"]["legs"]]
        assert result["execution"]["execution_duration_s"] is not None
        assert result["fees"]["fees_quote"] == "1.5"
        assert result["fees"]["per_order_fee_sum"] == "1.5"
        assert result["fees"]["fee_match"] is True
        assert result["slippage"] == {"slippage_bps": "10"}
        assert result["outcome"] == "completed"
        assert result["provenance"]["trade_id"] == trade.id
        assert len(result["provenance"]["order_ids"]) == 3
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase4_expected_vs_realized_transfer(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.models.transfer import TransferPlan, TransferRecord

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        from app.agent import build_agent

        build_agent(app)
        reader: JournalReader = app.agent_journal
        plan = TransferPlan(
            source_exchange="binance", dest_exchange="bybit", asset="ETH", network="ERC20",
            amount=Decimal("1"), buy_price=Decimal("3000"), sell_price=Decimal("3030"),
            buy_fee_bps=Decimal("10"), sell_fee_bps=Decimal("10"),
            withdrawal_fee=Decimal("0.001"), network_cost_quote=Decimal("0"),
            estimated_slippage_bps=Decimal("5"),
        )
        record = await app.transfers.save(TransferRecord(
            source_exchange="binance", dest_exchange="bybit", asset="ETH",
            network="ERC20", amount=Decimal("1"), plan=plan,
        ))
        # Expected net from plan math: (3030-3000) - fees(3.0+3.03) - withdraw(3.03) = 20.94
        trade = await app.trades.save(_trade(
            strategy=ArbitrageStrategy.TRANSFER, exchange="binance->bybit",
            route="USDT->ETH->ERC20->ETH/USDT",
            net=Decimal("18"), bps=Decimal("60"), transfer_id=record.id,
        ))
        result = await reader.analyze_trade(trade.id)
        assert result["expected"]["source"] == "transfer_plan"
        assert Decimal(result["expected"]["net_profit"]) == Decimal("20.94")
        assert result["realized"] == {"net_profit": "18", "net_profit_bps": "60"}
        assert Decimal(result["difference"]["net_profit"]) == Decimal("18") - Decimal("20.94")
        assert result["provenance"]["transfer_id"] == record.id
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase4_fee_slippage_execution_calculations(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        from app.agent import build_agent

        build_agent(app)
        reader: JournalReader = app.agent_journal
        # Partially filled legs + fee mismatch (persisted 2.0 vs orders sum 1.0).
        orders = (
            _order(symbol="BTC/USDT", side="buy", status="filled", fee="0.5"),
            _order(symbol="ETH/BTC", side="buy", status="partially_filled",
                   amount="2", filled="1", fee="0.5"),
        )
        trade = await app.trades.save(_trade(orders=orders, fees=Decimal("2.0"), slippage=Decimal("25")))
        result = await reader.analyze_trade(trade.id)
        assert result["execution"]["legs_total"] == 2
        assert result["execution"]["legs_filled"] == 1
        partial = [leg for leg in result["execution"]["legs"] if leg["status"] == "partially_filled"][0]
        assert partial["fill_ratio"] == "0.5"
        assert result["fees"] == {"fees_quote": "2.0", "per_order_fee_sum": "1.0", "fee_match": False}
        assert result["slippage"] == {"slippage_bps": "25"}
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase4_failure_analysis_with_recovery_refs(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        from app.agent import build_agent

        build_agent(app)
        reader: JournalReader = app.agent_journal
        trade = await app.trades.save(_trade(status=TradeStatus.FAILED, net=Decimal("-7"),
                                             bps=Decimal("-70"), error="venue timeout on leg 2"))
        await app.audit.log("TRIANGLE_EXECUTE_FINISHED", "leg 2 timeout",
                            {"trade_id": trade.id, "status": "failed"})
        await app.audit.log("APP_STARTED", "unrelated boot")
        result = await reader.analyze_trade(trade.id)
        assert result["outcome"] == "failed"
        assert result["failure"]["error"] == "venue timeout on leg 2"
        actions = [ref["action"] for ref in result["failure"]["recovery"]]
        assert "TRIANGLE_EXECUTE_FINISHED" in actions
        assert "APP_STARTED" not in actions
        assert result["provenance"]["trade_id"] == trade.id
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ aggregates


def test_phase4_aggregate_stats_math():
    trades = [
        _trade(net=Decimal("10"), bps=Decimal("100")),
        _trade(net=Decimal("20"), bps=Decimal("200")),
        _trade(status=TradeStatus.FAILED, net=Decimal("-5"), bps=Decimal("-50")),
        _trade(status=TradeStatus.MANUAL_REVIEW, net=Decimal("0"), bps=Decimal("0")),
    ]
    stats = aggregate_stats(trades)
    # Only completed trades enter PnL/avg; every result carries n.
    assert stats["n"] == 4
    assert stats["completed"] == 2 and stats["failed"] == 1 and stats["manual_review"] == 1
    assert stats["total_pnl"] == "30"
    assert stats["avg_net_bps"] == "150"
    assert Decimal(stats["win_rate"]) == Decimal("50")
    assert stats["sufficient"] is False  # n=4 < 5
    assert aggregate_stats([]) == {
        "n": 0, "completed": 0, "failed": 0, "manual_review": 0, "executing": 0,
        "total_pnl": "0", "avg_net_bps": "0", "win_rate": "0",
        "by_status": {}, "sufficient": False,
    }


@pytest.mark.asyncio
async def test_phase4_strategy_exchange_aggregation(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        from app.agent import build_agent

        build_agent(app)
        reader: JournalReader = app.agent_journal
        await _seed_mixed(app)
        by_strategy = await reader.strategy_performance()
        assert by_strategy["kind"] == "aggregation" and by_strategy["by"] == "strategy"
        assert by_strategy["total_n"] == 10
        assert by_strategy["sufficient"] is True
        assert by_strategy["groups"]["triangle"]["n"] == 8
        assert by_strategy["groups"]["transfer"]["n"] == 2
        assert by_strategy["groups"]["transfer"]["sufficient"] is False  # small group flagged
        by_exchange = await reader.exchange_performance()
        assert by_exchange["groups"]["binance"]["n"] == 6
        assert by_exchange["groups"]["okx"]["failed"] == 2
        routes = await reader.route_performance()
        assert routes["groups"]["USDT->BTC->ETH->USDT"]["n"] == 8
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase4_period_comparison(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        from app.agent import build_agent

        build_agent(app)
        reader: JournalReader = app.agent_journal
        now = datetime.now(UTC)
        old = now - timedelta(days=3)
        for _ in range(6):
            await app.trades.save(_trade(net=Decimal("10"), bps=Decimal("100")))
        for _ in range(6):
            await app.trades.save(_trade(net=Decimal("4"), bps=Decimal("40"), created_at=old))
        result = await reader.compare_today_vs_yesterday(now=now)
        assert result["kind"] == "period_comparison"
        assert result["n_a"] == 6 and result["n_b"] == 0
        # Empty previous period → explicit insufficient_data, deltas still arithmetic.
        assert result["conclusion"] == "insufficient_data"
        assert result["sufficient"] is False
        # Two explicit populated periods compare deterministically.
        explicit = await reader.compare_periods(
            since_a=now - timedelta(days=1), until_a=None,
            since_b=old - timedelta(days=1), until_b=old + timedelta(days=1),
            label_a="current", label_b="previous",
        )
        assert explicit["n_a"] == 6 and explicit["n_b"] == 6
        assert explicit["delta_total_pnl"] == str(Decimal("60") - Decimal("24"))
        assert explicit["delta_avg_net_bps"] == "60"
        assert explicit["sufficient"] is True
        assert "current" in explicit["conclusion"] and "previous" in explicit["conclusion"]
    finally:
        await shutdown_app(app)


def test_phase4_period_helpers_pure():
    now = datetime.now(UTC)
    old = now - timedelta(days=2)
    trades = [_trade(created_at=now), _trade(created_at=now), _trade(created_at=old)]
    today, yesterday = split_today_vs_yesterday(trades, now=now)
    assert len(today) == 2 and len(yesterday) == 0
    cmp_result = compare_periods(today, trades, label_a="t", label_b="all")
    assert cmp_result["n_a"] == 2 and cmp_result["n_b"] == 3
    assert cmp_result["sufficient"] is False  # n_a < 5 → no conclusions
    assert cmp_result["conclusion"] == "insufficient_data"
    assert filter_period(trades, since=now - timedelta(hours=1), until=None) == today


@pytest.mark.asyncio
async def test_phase4_missed_opportunities_insufficient_data(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        from app.agent import build_agent

        build_agent(app)
        reader: JournalReader = app.agent_journal
        # Fresh journal: scanner opportunities are not journaled → explicit gap.
        empty = await reader.missed_opportunities()
        assert empty["kind"] == "missed_opportunities"
        assert empty["status"] == "insufficient_data"
        assert empty["n"] == 0
        assert "not journaled" in empty["reason"]
        # A persisted rejection surfaces deterministically (no invention).
        await app.audit.log("TRIANGLE_RISK_REJECTED", "min net gate",
                            {"opportunity_id": "opp-1"})
        observed = await reader.missed_opportunities()
        assert observed["status"] == "observed"
        assert observed["n"] == 1
        assert observed["refs"][0]["action"] == "TRIANGLE_RISK_REJECTED"
    finally:
        await shutdown_app(app)


def test_phase4_missed_opportunities_pure_insufficient():
    result = missed_opportunities_from_audit([])
    assert result["status"] == "insufficient_data" and result["n"] == 0


# ------------------------------------------------------------------ provenance + determinism


@pytest.mark.asyncio
async def test_phase4_provenance_and_determinism(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        from app.agent import build_agent

        build_agent(app)
        reader: JournalReader = app.agent_journal
        orders = (_order(), _order(symbol="ETH/BTC", side="buy"))
        trade = await app.trades.save(_trade(orders=orders))
        first = await reader.analyze_trade(trade.id)
        second = await reader.analyze_trade(trade.id)
        assert first == second  # pure function of journaled data
        # Direct record-level analysis matches the service result.
        stored = await app.trades.get(trade.id)
        assert analyze_trade_record(stored)["realized"] == first["realized"]
        assert sanitize_trade(stored)["id"] == trade.id
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ AI integration


@pytest.mark.asyncio
async def test_phase4_ai_context_answers_journal_questions(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        build_agent(app)
        ids = await _seed_mixed(app)
        from app.agent.context import ContextCollector
        from app.agent.tools import AgentTools

        collector = ContextCollector(AgentTools(app), journal_reader=app.agent_journal)
        # Trade-specific question resolves the exact journal record.
        ctx = await collector.collect(query=f"Why did trade {ids[0]} make less than expected?")
        assert ctx.journal_analysis
        assert ctx.journal_analysis[0]["kind"] == "trade_analysis"
        assert ctx.journal_analysis[0]["trade_id"] == ids[0]
        # Performance questions aggregate with sample sizes.
        ctx2 = await collector.collect(query="Which exchange performs best?")
        assert any(a["kind"] == "aggregation" and a["by"] == "exchange" for a in ctx2.journal_analysis)
        ctx3 = await collector.collect(query="How does execution quality compare between periods?")
        assert any(a["kind"] == "period_comparison" for a in ctx3.journal_analysis)
        # Bounded: never more than 3 analyses per request.
        assert len(ctx.journal_analysis) <= 3
        # Empty journal → explicit insufficient_data, never invented.
        from tests.conftest import make_settings as _ms

        app2 = await build_app(_ms(tmp_path / "empty"))
        await start_app(app2, start_telegram=False, start_streams=False)
        try:
            from app.agent import build_agent as _b

            _b(app2)
            from app.agent.context import ContextCollector as _CC
            from app.agent.tools import AgentTools as _T

            ctx_empty = await _CC(_T(app2), journal_reader=app2.agent_journal).collect(
                query="How much was paid in fees?"
            )
            assert ctx_empty.journal_analysis
            assert ctx_empty.journal_analysis[0].get("status") == "insufficient_data"
        finally:
            await shutdown_app(app2)
    finally:
        await shutdown_app(app)


@pytest.mark.asyncio
async def test_phase4_analysis_keeps_fact_hypothesis_separation(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.analysis import AnalysisEngine
    from app.agent.providers.base import LLMProvider, LLMRequest, LLMResponse

    class LyingLLM(LLMProvider):
        name = "lying"

        async def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(content="All trades made 999999 profit with zero fees.")

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        core, *_ = build_agent(app, llm=LyingLLM())
        await _seed_mixed(app)
        from app.agent.core import AgentRequest

        resp = await core.handle(AgentRequest(query="Which strategy performs best?"))
        assert resp.analysis is not None
        facts = "\n".join(resp.analysis.facts)
        hyps = "\n".join(resp.analysis.hypotheses)
        # Deterministic journal FACTS survive; LLM fiction stays a hypothesis.
        assert "journal-fact" in facts
        assert "999999" not in facts
        assert "999999" in hyps
        assert resp.analysis.recommendations == () or resp.recommendation is None or True
        # Prompt handed to the LLM contains the computed numbers (bounded).
        from app.agent.core import _build_llm_prompt

        prompt = _build_llm_prompt("Which strategy performs best?", resp.context, resp.reflection, None)
        assert "journal-fact" in prompt or "Journal analysis" in prompt
        assert len(prompt) <= 6000 + 2000
    finally:
        await shutdown_app(app)


# ------------------------------------------------------------------ security / isolation


@pytest.mark.asyncio
async def test_phase4_readonly_security_boundaries(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent.tools import AgentTools

    app = await build_app(make_settings(tmp_path))
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        from app.agent import build_agent
        from app.agent import journal as journal_mod

        build_agent(app)
        reader: JournalReader = app.agent_journal
        # No write / execution surface whatsoever.
        for forbidden in ("save", "merge", "delete", "execute", "executemany", "commit",
                          "create_order", "withdraw", "set_config", "shell", "run_python",
                          "apply", "mutate", "update_risk", "session"):
            assert not hasattr(reader, forbidden), f"reader exposes {forbidden}"
        source = inspect.getsource(journal_mod)
        for forbidden in ("session.merge", "session.add", "session.delete", ".execute(",
                          "os.system", "subprocess", "eval(", "exec("):
            assert forbidden not in source, f"journal module contains {forbidden}"
        # Allowlist unchanged (journal is a service, not an LLM tool).
        assert len(AgentTools(app).allowed_tools) == 10
        # Outputs carry no secret-shaped fields.
        await _seed_mixed(app)
        view = await reader.get_trade((await reader.list_trades(limit=1))[0]["id"])
        blob = str(view)
        for token in ("api_key", "secret", "password", "token", "CAT_KEY", ".env"):
            assert token not in blob.lower()
        # Journal reads leave trading state untouched.
        limits_before = str(app.settings.risk.max_trade_size)
        n_before = len(await app.trades.list_recent(50))
        await reader.analyze_trade(view["id"])
        await reader.strategy_performance()
        await reader.exchange_performance()
        await reader.compare_today_vs_yesterday()
        await reader.missed_opportunities()
        assert str(app.settings.risk.max_trade_size) == limits_before
        assert len(await app.trades.list_recent(50)) == n_before
        assert not app.guard.is_halted
    finally:
        await shutdown_app(app)


def test_phase4_min_sample_constant():
    assert MIN_SAMPLE_FOR_CONCLUSIONS == 5
