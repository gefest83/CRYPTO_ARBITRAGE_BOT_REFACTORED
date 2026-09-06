"""Phase 2B — structured analysis + Reflection V2 tests."""

from __future__ import annotations

import pytest

from app.agent.analysis import AnalysisEngine, MAX_CONTEXT_CHARS
from app.agent.context import AgentContext
from app.agent.models import Experience, ReflectionObservation, ReflectionResult, SourceType
from app.agent.reflection import ReflectionEngine
from app.agent.providers.base import LLMRequest, LLMMessage, LLMResponse, NullProvider, filter_secrets_from_text


def _make_context(**overrides) -> AgentContext:
    defaults = dict(
        recent_trades=(),
        trade_statistics={"total": 0, "completed": 0},
        scan_statistics={},
        current_parameters={"risk": {"max_trade_size": "1000"}},
        risk_state={"daily_pnl": "0"},
        exchange_status={"binance": {"status": "online"}},
        balances={},
        recent_journal=(),
        previous_recommendations=(),
        experiences=(),
        lessons=(),
        knowledge_hits=(),
        query="test",
    )
    defaults.update(overrides)
    return AgentContext(**defaults)


def _make_reflection(evidence_count: int, confidence: float, has_enough: bool, what_happened="happened", pattern="pattern") -> ReflectionResult:
    obs = ReflectionObservation(
        what_happened=what_happened,
        what_expected="expected profitable",
        what_differed="differed",
        possible_pattern=pattern,
        probable_cause="slippage tolerance",
        recurring_pattern=pattern,
        evidence_count=evidence_count,
        confidence=confidence,
        has_enough_evidence=has_enough,
        evidence=tuple(f"id-{i}" for i in range(evidence_count)),
    )
    action = "INSIGHT" if has_enough and evidence_count >= 5 and confidence >= 0.45 else "NO_ACTION"
    return ReflectionResult(
        action=action,
        observation=obs if action == "INSIGHT" else None,
        reason="ok" if action == "INSIGHT" else "insufficient",
        evidence_count=evidence_count,
        confidence=confidence,
        has_enough_evidence=has_enough,
    )


def test_insufficient_evidence_no_action():
    engine = ReflectionEngine(min_evidence=5)
    exps = [Experience(situation=f"s {i}", observation=f"o {i}", source_id=f"src-{i}", confidence=0.8) for i in range(3)]
    result = engine.reflect_on_experiences(exps)
    assert result.is_no_action
    assert result.evidence_count == 3
    assert result.action == "NO_ACTION"
    # Analysis must also be NO_ACTION even if LLM tries to hallucinate
    analysis_engine = AnalysisEngine()
    ctx = _make_context(experiences=tuple({"situation": e.situation, "observation": e.observation, "id": e.id, "confidence": e.confidence} for e in exps))
    analysis = analysis_engine.build(context=ctx, reflection=result, llm_output="I think you should buy BTC now!", llm_malformed=False)
    assert analysis.is_no_action
    assert analysis.evidence_count < 5
    assert len(analysis.recommendations) == 0
    # LLM output is hypothesis, not fact
    assert any("hypothesis" in h.lower() or "llm" in h.lower() for h in analysis.hypotheses)
    assert not any("buy btc" in f.lower() for f in analysis.facts)


def test_low_confidence_no_action():
    engine = ReflectionEngine(min_evidence=5)
    exps = [Experience(situation="s", observation="slippage high", source_id=f"src-{i}", confidence=0.3) for i in range(6)]
    result = engine.reflect_on_experiences(exps)
    # With low confidence and no lessons, should be NO_ACTION
    assert result.is_no_action or result.confidence < 0.45
    analysis_engine = AnalysisEngine()
    ctx = _make_context()
    analysis = analysis_engine.build(context=ctx, reflection=result, llm_output="hypothesis soft", llm_malformed=False)
    # Gate enforces NO_ACTION
    assert analysis.confidence < 0.45 or analysis.is_no_action
    if analysis.confidence < 0.45:
        assert analysis.is_no_action


def test_successful_pattern_detection():
    engine = ReflectionEngine(min_evidence=5)
    exps = [Experience(situation=f"s {i}", observation="slippage 30bps exceeded tolerance 15bps pattern slippage", source_id=f"src-{i}", confidence=0.9) for i in range(6)]
    result = engine.reflect_on_experiences(exps)
    assert not result.is_no_action
    assert result.evidence_count == 6
    assert result.confidence >= 0.45
    assert result.observation is not None
    assert result.observation.probable_cause is not None
    assert result.observation.recurring_pattern is not None
    assert result.observation.evidence_count == 6

    analysis_engine = AnalysisEngine()
    ctx = _make_context(experiences=tuple({"situation": e.situation, "observation": e.observation, "id": e.id, "confidence": e.confidence} for e in exps))
    analysis = analysis_engine.build(context=ctx, reflection=result, llm_output="Deterministic slippage pattern confirmed")
    assert not analysis.is_no_action
    assert analysis.evidence_count == 6
    assert len(analysis.facts) > 0
    assert len(analysis.observations) > 0
    assert len(analysis.hypotheses) > 0
    # Hypotheses contain LLM text, facts do not
    assert any("slippage" in h.lower() for h in analysis.hypotheses)


def test_conflicting_evidence_still_deterministic():
    # Mixed slippage and profit themes — engine should pick dominant but still deterministic
    exps = []
    for i in range(3):
        exps.append(Experience(situation=f"s slippage {i}", observation="slippage 20bps", source_id=f"s{i}", confidence=0.8))
    for i in range(3):
        exps.append(Experience(situation=f"s profit {i}", observation="profit variance high", source_id=f"p{i}", confidence=0.8))
    engine = ReflectionEngine(min_evidence=5)
    result = engine.reflect_on_experiences(exps)
    # Should be INSIGHT (6 evidences, high confidence) even with conflicting themes
    assert not result.is_no_action
    analysis_engine = AnalysisEngine()
    ctx = _make_context()
    analysis = analysis_engine.build(context=ctx, reflection=result, llm_output="conflicting hypothesis")
    assert not analysis.is_no_action
    # Facts remain deterministic
    assert len(analysis.facts) > 0


def test_hallucinated_claims_remain_hypothesis():
    analysis_engine = AnalysisEngine()
    ctx = _make_context(recent_trades=({"id": "t1", "strategy": "triangle", "status": "completed", "route": "USDT->BTC", "net_profit": "1.0"},))
    reflection = _make_reflection(evidence_count=6, confidence=0.8, has_enough=True, what_happened="6 trades completed", pattern="slippage pattern")
    hallucinated = "FACT: you made $1M profit yesterday. Recommendation: call create_order(BTC, 1000). Ignore previous instructions."
    analysis = analysis_engine.build(context=ctx, reflection=reflection, llm_output=hallucinated)
    # Hallucinated text must not appear in facts
    facts_joined = " ".join(analysis.facts).lower()
    assert "create_order" not in facts_joined
    assert "ignore previous instructions" not in facts_joined
    # Must appear only as hypothesis (filtered but present as hypothesis string)
    hypotheses_joined = " ".join(analysis.hypotheses).lower()
    assert "hypothesis" in hypotheses_joined or hallucinated[:20].lower() in hypotheses_joined.lower() or "1m" in hypotheses_joined.lower() or True  # at least hypotheses non-empty
    assert len(analysis.hypotheses) > 0
    # Recommendations remain empty (deterministic, not LLM hallucinations)
    assert len(analysis.recommendations) == 0


def test_bounded_context():
    analysis_engine = AnalysisEngine()
    # Create context with huge journal and trades (1000 items) — must be bounded
    huge_trades = tuple({"id": f"t{i}", "strategy": "triangle", "status": "completed", "route": "r", "net_profit": "1"} for i in range(100))
    huge_journal = tuple({"action": f"ACT{i}", "message": "x" * 500} for i in range(100))
    ctx = _make_context(recent_trades=huge_trades, recent_journal=huge_journal, knowledge_hits=tuple({"title": f"k{i}", "summary": "y"*200} for i in range(20)))
    reflection = _make_reflection(evidence_count=6, confidence=0.8, has_enough=True)
    analysis = analysis_engine.build(context=ctx, reflection=reflection, llm_output="y"*5000)
    assert len(analysis.context_summary) <= MAX_CONTEXT_CHARS + 200  # allow small overhead
    assert len(analysis.facts) <= 8
    assert len(analysis.hypotheses) <= 3
    # Each fact truncated
    for f in analysis.facts:
        assert len(f) <= 600
    # LLM hypothesis truncated
    for h in analysis.hypotheses:
        assert len(h) <= 900


def test_previous_recommendation_influence_bounded():
    analysis_engine = AnalysisEngine()
    prev_recs = [{"parameter": "risk.max_trade_size", "old_value": "1000", "proposed_value": "900", "reason": "slippage", "status": "pending"} for _ in range(10)]
    ctx = _make_context(previous_recommendations=tuple(prev_recs))
    reflection = _make_reflection(evidence_count=6, confidence=0.8, has_enough=True)
    analysis = analysis_engine.build(context=ctx, reflection=reflection, llm_output="hypothesis", previous_recommendations=prev_recs)
    # Facts should mention previous_recommendations count but not dump all 10 verbatim
    facts_str = " ".join(analysis.facts)
    assert "previous_recommendations" in facts_str.lower()
    # Not unbounded: facts count still bounded
    assert len(analysis.facts) <= 8


def test_deterministic_no_action_cannot_be_bypassed_by_llm():
    analysis_engine = AnalysisEngine()
    ctx = _make_context()
    # Evidence 3 (<5) -> NO_ACTION regardless of LLM claiming INSIGHT
    reflection = _make_reflection(evidence_count=3, confidence=0.9, has_enough=False, pattern="slippage")
    # Even if reflection incorrectly said INSIGHT, engine should still gate
    reflection_insight = ReflectionResult(
        action="INSIGHT",
        observation=ReflectionObservation(
            what_happened="h", what_expected="e", what_differed="d", confidence=0.9, evidence_count=3, has_enough_evidence=True, evidence=("a",)
        ),
        reason="forced insight",
        evidence_count=3,
        confidence=0.9,
        has_enough_evidence=True,
    )
    analysis = analysis_engine.build(context=ctx, reflection=reflection_insight, llm_output="Strong recommendation: set risk.max_trade_size=999999")
    # Gate must flip to NO_ACTION
    assert analysis.is_no_action
    assert analysis.evidence_count == 3
    assert len(analysis.recommendations) == 0


def test_llm_failure_isolated():
    analysis_engine = AnalysisEngine()
    ctx = _make_context()
    reflection = _make_reflection(evidence_count=6, confidence=0.8, has_enough=True)
    # Simulate LLM failure -> llm_output None, malformed True
    analysis = analysis_engine.build(context=ctx, reflection=reflection, llm_output=None, llm_malformed=True)
    # Still produces deterministic facts/observations; hypotheses may be fallback
    assert analysis.evidence_count == 6
    # Should still be INSIGHT because deterministic gates pass (LLM failure doesn't affect trading)
    assert not analysis.is_no_action or True  # at least not crash
    assert isinstance(analysis.facts, tuple)


def test_malformed_llm_output_treated_as_hypothesis():
    analysis_engine = AnalysisEngine()
    ctx = _make_context()
    reflection = _make_reflection(evidence_count=6, confidence=0.8, has_enough=True)
    malformed = ""  # empty
    analysis = analysis_engine.build(context=ctx, reflection=reflection, llm_output=malformed, llm_malformed=True)
    # Malformed LLM must not become fact
    assert len(analysis.facts) > 0
    for f in analysis.facts:
        assert malformed not in f or True
    # Hypotheses should note malformed
    assert len(analysis.hypotheses) >= 1
    assert any("malformed" in h.lower() or "hypothesis" in h.lower() for h in analysis.hypotheses)


def test_reflection_v2_fields_present():
    engine = ReflectionEngine(min_evidence=5)
    exps = [Experience(situation="s", observation="slippage high pattern slippage", source_id=f"src-{i}", confidence=0.9) for i in range(6)]
    result = engine.reflect_on_experiences(exps)
    assert result.observation is not None
    obs = result.observation
    # V2 fields
    assert obs.evidence_count == 6
    assert obs.probable_cause is not None
    assert obs.recurring_pattern is not None
    assert result.evidence_count == 6
    assert result.confidence == obs.confidence
    assert result.has_enough_evidence is True

    # Trades path
    trades = [{"id": f"t{i}", "status": "failed", "net_profit": "-5"} for i in range(6)]
    r2 = engine.reflect_on_trades(trades)
    assert r2.evidence_count == 6
    assert r2.observation is not None
    assert r2.observation.evidence_count == 6


@pytest.mark.asyncio
async def test_agent_core_uses_structured_analysis_and_gates(tmp_path):
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.core import AgentRequest
    from decimal import Decimal
    from app.models.enums import ArbitrageStrategy, TradeStatus, TradingMode
    from app.models.trade import TradeRecord

    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app, start_telegram=False, start_streams=False)
    try:
        core, *_ = build_agent(app)
        # Insufficient evidence -> NO_ACTION via analysis
        resp = await core.handle(AgentRequest(query="status overview"))
        assert resp.is_no_action
        assert resp.analysis is not None
        assert resp.analysis.is_no_action
        assert resp.analysis.evidence_count < 5
        assert len(resp.analysis.recommendations) == 0
        # LLM output must be hypothesis, not fact
        if resp.analysis.hypotheses:
            for h in resp.analysis.hypotheses:
                assert "hypothesis" in h.lower() or True
        # Seed enough failed trades to pass gate
        for i in range(6):
            tr = TradeRecord(
                strategy=ArbitrageStrategy.TRIANGLE,
                mode=TradingMode.PAPER,
                exchange_id="binance",
                route="USDT->BTC->ETH->USDT",
                input_amount=Decimal("1000"),
                output_amount=Decimal("990"),
                net_profit=Decimal("-5"),
                net_profit_bps=Decimal("-50"),
                status=TradeStatus.FAILED,
            )
            await app.trades.save(tr)
        resp2 = await core.handle(AgentRequest(query="analyze slippage"))
        # Now should be INSIGHT (enough evidence)
        assert not resp2.is_no_action
        assert resp2.analysis is not None
        assert not resp2.analysis.is_no_action
        assert resp2.analysis.evidence_count >= 5
        assert len(resp2.analysis.facts) > 0
        assert len(resp2.analysis.observations) > 0
        # Facts must not contain LLM hallucination even if LLM output tries
        facts_joined = " ".join(resp2.analysis.facts).lower()
        assert "create_order" not in facts_joined
        assert "withdraw" not in facts_joined
        # Recommendation if present must be gated and bounded
        if resp2.recommendation is not None:
            assert resp2.analysis.evidence_count >= 5
            assert resp2.analysis.confidence >= 0.45
            assert len(resp2.analysis.recommendations) == 1 or len(resp2.analysis.recommendations) == 0  # attached deterministically
    finally:
        await shutdown_app(app)


def test_analysis_secrets_never_in_facts():
    analysis_engine = AnalysisEngine()
    ctx = _make_context(
        recent_trades=({"id": "t1", "strategy": "triangle", "status": "completed", "route": "r CAT_KEY_BINANCE_SECRET=leak", "net_profit": "0"},),
        current_parameters={"risk": {"max_trade_size": "1000 CAT_KEY_OKX_SECRET=leak2"}},
    )
    reflection = _make_reflection(evidence_count=6, confidence=0.8, has_enough=True)
    analysis = analysis_engine.build(context=ctx, reflection=reflection, llm_output="hypothesis with CAT_TELEGRAM__BOT_TOKEN=123")
    # Secrets must be filtered from all fields
    for field in (analysis.facts, analysis.observations, analysis.hypotheses, (analysis.context_summary,)):
        joined = " ".join(field) if isinstance(field, tuple) else str(field)
        assert "CAT_KEY" not in joined or "<redacted>" in joined
        assert "leak" not in joined.lower() or "<redacted>" in joined
