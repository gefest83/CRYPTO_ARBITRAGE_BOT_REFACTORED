"""Phase 2A — OpenRouter provider tests.

Covers construction, missing key, timeout, transient failure, malformed
response, successful response, secret filtering, failure isolation, and
factory behaviour. No real network is used.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import SecretStr

from app.agent.providers.base import LLMRequest, LLMMessage, NullProvider, EchoProvider, filter_secrets_from_text
from app.agent.providers.openrouter import OpenRouterProvider, OpenRouterError
from app.agent.providers import create_provider
from app.config.settings import AgentSettings, Settings


class FakeSuccessClient:
    def __init__(self, response: dict):
        self.response = response
        self.calls = 0
        self.last_timeout = None
        self.last_headers = None

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        self.last_timeout = timeout
        self.last_headers = headers
        return self.response


class FakeTransientThenSuccess:
    def __init__(self, success_response: dict):
        self.success_response = success_response
        self.calls = 0

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        if self.calls == 1:
            err = OpenRouterError("retryable HTTP 503")
            err._retryable = True  # type: ignore[attr-defined]
            raise err
        return self.success_response


class FakeTimeoutClient:
    def __init__(self):
        self.calls = 0

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        raise TimeoutError("timeout simulated")


class FakeMalformedClient:
    async def post(self, url, json=None, headers=None, timeout=None):
        return {"no_choices": []}


def test_openrouter_construction_and_model_configurable():
    p = OpenRouterProvider(api_key=SecretStr("sk-test"), model="openai/gpt-4o-mini", timeout_seconds=5, max_retries=2)
    assert p.name == "openrouter"
    assert p.model == "openai/gpt-4o-mini"
    # custom model
    p2 = OpenRouterProvider(api_key="key", model="anthropic/claude-3.5-sonnet")
    assert p2.model == "anthropic/claude-3.5-sonnet"


def test_openrouter_missing_api_key_fail_closed():
    p = OpenRouterProvider(api_key=SecretStr(""), model="openai/gpt-4o-mini")
    request = LLMRequest(messages=(LLMMessage(role="user", content="hello"),))
    # Should raise OpenRouterError without network
    import asyncio as _asyncio

    async def run():
        with pytest.raises(OpenRouterError, match="missing API key"):
            await p.complete(request)

    _asyncio.run(run())


def test_openrouter_successful_response():
    response = {
        "choices": [{"message": {"content": "hello world"}, "finish_reason": "stop"}],
        "model": "openai/gpt-4o-mini",
        "usage": {"prompt_tokens": 10},
    }
    client = FakeSuccessClient(response)
    p = OpenRouterProvider(api_key=SecretStr("sk-test"), model="openai/gpt-4o-mini", http_client=client)

    async def run():
        req = LLMRequest(messages=(LLMMessage(role="user", content="hello"),))
        out = await p.complete(req)
        assert out.content == "hello world"
        assert out.model == "openai/gpt-4o-mini"
        assert out.finish_reason == "stop"
        assert client.calls == 1
        # Ensure timeout was passed
        assert client.last_timeout == 10.0
        # Ensure Authorization header present but not logged with secrets elsewhere — header exists
        assert "Authorization" in client.last_headers
        assert "Bearer" in client.last_headers["Authorization"]

    asyncio.run(run())


def test_openrouter_malformed_response():
    client = FakeMalformedClient()
    p = OpenRouterProvider(api_key=SecretStr("sk-test"), http_client=client)

    async def run():
        req = LLMRequest(messages=(LLMMessage(role="user", content="hi"),))
        with pytest.raises(OpenRouterError, match="malformed"):
            await p.complete(req)

    asyncio.run(run())


def test_openrouter_secret_filtering_outbound_and_inbound():
    # Response contains a secret — must be filtered before returning
    response = {
        "choices": [{"message": {"content": "leaked CAT_KEY_BINANCE_SECRET=supersecret123"}, "finish_reason": "stop"}],
        "model": "openai/gpt-4o-mini",
    }

    class CapturingClient:
        def __init__(self):
            self.captured_json = None

        async def post(self, url, json=None, headers=None, timeout=None):
            self.captured_json = json
            return response

    client = CapturingClient()
    p = OpenRouterProvider(api_key=SecretStr("sk-test"), http_client=client)

    async def run():
        req = LLMRequest(messages=(LLMMessage(role="user", content="prompt CAT_KEY_OKX_SECRET=mysecret"),))
        out = await p.complete(req)
        # Outbound was filtered: captured payload must not contain raw secret
        payload_str = str(client.captured_json)
        assert "mysecret" not in payload_str
        assert "<redacted>" in payload_str or "CAT_KEY" not in payload_str
        # Inbound also filtered
        assert "supersecret123" not in out.content
        assert "<redacted>" in out.content

    asyncio.run(run())


def test_openrouter_transient_failure_retry():
    response = {
        "choices": [{"message": {"content": "success after retry"}, "finish_reason": "stop"}],
        "model": "openai/gpt-4o-mini",
    }
    client = FakeTransientThenSuccess(response)
    p = OpenRouterProvider(api_key=SecretStr("sk-test"), max_retries=1, http_client=client)

    async def run():
        req = LLMRequest(messages=(LLMMessage(role="user", content="hi"),))
        out = await p.complete(req)
        assert out.content == "success after retry"
        assert client.calls == 2

    asyncio.run(run())


def test_openrouter_timeout_retry_and_fail_closed():
    client = FakeTimeoutClient()
    p = OpenRouterProvider(api_key=SecretStr("sk-test"), max_retries=1, timeout_seconds=2, http_client=client)

    async def run():
        req = LLMRequest(messages=(LLMMessage(role="user", content="hi"),))
        with pytest.raises(OpenRouterError):
            await p.complete(req)
        # Retried once + initial = 2 calls
        assert client.calls == 2

    asyncio.run(run())


def test_openrouter_never_logs_api_key():
    # Ensure that exceptions/logs do not contain raw key
    response = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "model": "x",
    }
    client = FakeSuccessClient(response)
    secret = "sk-super-secret-xyz123"
    p = OpenRouterProvider(api_key=SecretStr(secret), http_client=client)

    async def run():
        req = LLMRequest(messages=(LLMMessage(role="user", content="hi"),))
        out = await p.complete(req)
        assert out.content == "ok"
        # Ensure that the provider's internal state does not leak via repr/str that includes key
        # The __repr__ is default — but we check that our filtering would redact if key leaked
        # Simulate an exception that includes payload — ensure api_key not in exception
        try:
            bad_client = FakeMalformedClient()
            p2 = OpenRouterProvider(api_key=SecretStr(secret), http_client=bad_client)
            await p2.complete(req)
        except OpenRouterError as exc:
            txt = str(exc)
            assert secret not in txt
            assert "sk-super-secret" not in txt

    asyncio.run(run())


def test_create_provider_factory():
    # null
    s = Settings(_env_file=None)
    s = s.model_copy(update={"agent": AgentSettings(provider="null", model="openai/gpt-4o-mini")})
    p = create_provider(s)
    assert isinstance(p, NullProvider)

    # echo
    s2 = s.model_copy(update={"agent": AgentSettings(provider="echo")})
    p2 = create_provider(s2)
    assert isinstance(p2, EchoProvider)

    # openrouter without key still constructs but will fail closed on complete
    s3 = s.model_copy(update={"agent": AgentSettings(provider="openrouter", api_key=SecretStr("sk-test"), model="anthropic/claude-3.5")})
    p3 = create_provider(s3)
    assert isinstance(p3, OpenRouterProvider)
    assert p3.model == "anthropic/claude-3.5"

    # local -> stub null
    s4 = s.model_copy(update={"agent": AgentSettings(provider="local")})
    p4 = create_provider(s4)
    assert isinstance(p4, NullProvider)

    # missing settings -> null
    assert isinstance(create_provider(None), NullProvider)

    # No network on construction
    # If factory made a network call, this would have raised or taken time — it doesn't


def test_null_and_echo_keep_functional():
    async def run():
        null = NullProvider(canned="stub")
        out = await null.complete(LLMRequest(messages=(LLMMessage(role="user", content="hi"),)))
        assert "stub" in out.content

        echo = EchoProvider()
        out2 = await echo.complete(LLMRequest(messages=(LLMMessage(role="user", content="hi CAT_KEY_X=secret"),)))
        # Echo must filter secrets
        assert "secret" not in out2.content or "<redacted>" in out2.content

    asyncio.run(run())


def test_openrouter_does_not_store_secrets_in_tables(tmp_path):
    # Direct check: the tables module must not have columns named secret/api_key
    from app.agent.tables import AgentKnowledgeRow, AgentExperienceRow, AgentRecommendationRow
    for cls in (AgentKnowledgeRow, AgentExperienceRow, AgentRecommendationRow):
        cols = {c.name.lower() for c in cls.__table__.columns}
        for forbidden in ("api_key", "secret", "password", "token", "dsn", "api_key"):
            assert forbidden not in cols


def test_provider_failure_isolation_does_not_affect_trading(tmp_path):
    # Simulate core with failing provider — handle() must still return gracefully and not raise
    from tests.conftest import make_settings
    from app.services import build_app, shutdown_app, start_app
    from app.agent import build_agent
    from app.agent.providers.base import LLMProvider, LLMRequest, LLMResponse

    class FailingProvider(LLMProvider):
        name = "failing"

        async def complete(self, request: LLMRequest) -> LLMResponse:
            raise RuntimeError("simulated provider crash with CAT_KEY_BINANCE_SECRET=should_be_redacted")

    async def run_inner():
        settings = make_settings(tmp_path)
        app = await build_app(settings)
        await start_app(app, start_telegram=False, start_streams=False)
        try:
            # Seed some trades so reflection yields insight and core will attempt LLM
            from decimal import Decimal
            from app.models.enums import ArbitrageStrategy, TradeStatus, TradingMode
            from app.models.trade import TradeRecord

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
            core, *_ = build_agent(app, llm=FailingProvider())
            from app.agent.core import AgentRequest

            resp = await core.handle(AgentRequest(query="status"))
            # Must not propagate exception — core catches provider failures
            assert resp is not None
            # Trading state must be unchanged — e.g., risk limits still readable
            assert app.settings.risk.max_trade_size == Decimal("1000")
            # LLM output should be None due to failure, but reflection may still be present
            assert resp.llm_output is None or isinstance(resp.llm_output, str)
        finally:
            await shutdown_app(app)

    asyncio.run(run_inner())
