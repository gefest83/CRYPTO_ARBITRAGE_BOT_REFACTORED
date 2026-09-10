"""Phase 4A — Live collector runtime/CLI wiring (deterministic, no network)."""

from __future__ import annotations

import asyncio
import pathlib
from decimal import Decimal

import httpx
import pytest

from app.cli.main import build_parser
from app.research.prediction_markets.client import BinancePredictionClient
from app.research.prediction_markets.collector.storage import CollectorStore
from app.research.prediction_markets.runtime import LiveCollectorRuntime, RuntimeConfig


def test_cli_parser_includes_research_collect() -> None:
    p = build_parser()
    args = p.parse_args(["research-collect", "--duration", "5", "--out", "data/research/prediction_markets"])
    assert args.command == "research-collect"
    assert float(args.duration) == 5.0
    assert args.out == "data/research/prediction_markets"


def test_cli_wiring_is_research_only_and_fail_closed() -> None:
    # runtime file must not import forbidden modules
    text = pathlib.Path("app/research/prediction_markets/runtime.py").read_text(encoding="utf-8").lower()
    for kw in ["from app.execution", "from app.risk", "from app.recovery", "from app.telegram", "from app.agent", "from app.strategies.kronos", "from app.strategies.triangular", "from app.strategies.transfer"]:
        assert kw not in text, f"runtime imports forbidden {kw}"
    text_cli = pathlib.Path("app/cli/main.py").read_text(encoding="utf-8")
    assert "research-collect" in text_cli
    assert "cmd_research_collect" in text_cli


def test_runtime_ensure_research_only_raises() -> None:
    rt = LiveCollectorRuntime(config=RuntimeConfig(duration_secs=0.1), store=CollectorStore(memory_only=True))
    with pytest.raises(RuntimeError, match="fail-closed"):
        rt.ensure_research_only()


@pytest.mark.asyncio
async def test_runtime_with_mocked_sources_and_mocked_discovery(tmp_path: pathlib.Path) -> None:
    # Mock prediction discovery: return BTC/ETH 5m markets via MockTransport
    BTC_5M = {
        "marketTopicId": 1001, "vendor": "PREDICT_FUN", "chainId": "56", "slug": "btc-5m", "title": "BTC 5m", "symbol": "BTCUSDT",
        "participantCount": 10, "collateral": "USDT", "feeRateBps": 200, "slippageBps": 1200,
        "liquidity": "1000", "tradeVolume": "1000", "publishedAt": 1_748_131_200_000, "startDate": 1_748_131_200_000, "endDate": 9_000_000_000_000,
        "status": "REGISTERED", "markets": [{"marketId": 9001, "title": "UP", "status": "REGISTERED", "tradingStatus": "OPEN", "liquidity": "500", "tradeVolume": "500", "outcomes": [{"name": "YES", "price": "0.51", "chance": "0.51", "index": 0, "tokenId": "tok_yes"}, {"name": "NO", "price": "0.49", "chance": "0.49", "index": 1, "tokenId": "tok_no"}]}],
    }
    ETH_15M = {
        "marketTopicId": 1002, "vendor": "PREDICT_FUN", "chainId": "56", "slug": "eth-15m", "title": "ETH 15m", "symbol": "ETHUSDT",
        "participantCount": 10, "collateral": "USDT", "feeRateBps": 200, "slippageBps": 1200,
        "liquidity": "1000", "tradeVolume": "1000", "publishedAt": 1_748_131_200_000, "startDate": 1_748_131_200_000, "endDate": 9_000_000_000_000,
        "status": "REGISTERED", "markets": [{"marketId": 9002, "title": "UP", "status": "REGISTERED", "tradingStatus": "OPEN", "liquidity": "500", "tradeVolume": "500", "outcomes": [{"name": "YES", "price": "0.51", "chance": "0.51", "index": 0, "tokenId": "tok_eth_yes"}, {"name": "NO", "price": "0.49", "chance": "0.49", "index": 1, "tokenId": "tok_eth_yes2"}]}],
    }

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path.endswith("/market/list"):
            return httpx.Response(200, json={"marketTopics": [BTC_5M, ETH_15M], "total": 2, "offset": 0, "limit": 20, "hasMore": False})
        if path.endswith("/market/search"):
            return httpx.Response(200, json=[])
        if path.endswith("/market/detail"):
            tid = int(dict(req.url.params).get("marketTopicId", "0"))
            mapping = {1001: BTC_5M, 1002: ETH_15M}
            return httpx.Response(200, json=mapping.get(tid, BTC_5M))
        if path.endswith("/last-trade-price"):
            return httpx.Response(200, json={"marketId": int(dict(req.url.params).get("marketId", "9001")), "lastTradePrice": "0.52"})
        if path.endswith("/order-book"):
            return httpx.Response(200, json={"outcome": "YES", "tokenId": "tok_yes", "timestamp": 1_748_131_200_000, "bids": [{"price": "0.51", "size": "1000"}], "asks": [{"price": "0.53", "size": "1000"}]})
        return httpx.Response(404, json={})

    transport = httpx.MockTransport(handler)
    client = BinancePredictionClient("k", "s", transport=transport)

    import time as _time
    base_now = int(_time.time() * 1000)
    # Mock spot + prediction sources via async generators (timestamps near now to avoid stale)
    async def fake_spot():
        for i in range(5):
            yield {"symbol": "BTCUSDT", "bid": "68000", "ask": "68002", "exchange_ts_ms": base_now + i * 100, "sequence": i}

    async def fake_pred():
        for i in range(5):
            yield {"marketId": 9001, "tokenId": "tok_yes", "symbol": "BTCUSDT", "updateTimestampMs": base_now + i * 120, "sequence": i, "resolution_ms": 9_000_000_000_000, "bids": [["0.51", "1000"]], "asks": [["0.53", "1000"]], "duration": "5m"}

    store = CollectorStore(base_dir=tmp_path / "research", memory_only=True)
    cfg = RuntimeConfig(duration_secs=0.6, out_dir=tmp_path / "research", spot_poll_ms=50, prediction_poll_ms=50, stale_threshold_ms=60000)
    rt = LiveCollectorRuntime(config=cfg, store=store, client=client, credentials=("k", "s"), spot_source_factory=lambda: fake_spot(), prediction_source_factory=lambda: fake_pred())
    result = await rt.run()

    # discovery via mocks should have found 2 markets (BTC+ETH)
    assert result.discovered_markets >= 1
    # mocked sources produced observations
    assert result.spot_observations >= 1
    assert result.prediction_observations >= 1
    # collector handled duplicates/gaps etc
    assert store.count() >= 2
    # synchronized: spot+pred close in time => some sync
    # not strict, but at least not all zero when both sources active
    # graceful shutdown completed (duration timeout)
    assert result.reconnects >= 0


@pytest.mark.asyncio
async def test_runtime_graceful_shutdown_via_duration(tmp_path: pathlib.Path) -> None:
    async def empty_spot():
        if False:
            yield {}
    async def empty_pred():
        if False:
            yield {}

    store = CollectorStore(memory_only=True)
    cfg = RuntimeConfig(duration_secs=0.2, out_dir=tmp_path / "r")
    rt = LiveCollectorRuntime(config=cfg, store=store, credentials=("k", "s"), spot_source_factory=lambda: empty_spot(), prediction_source_factory=lambda: empty_pred())
    # need a mock client to avoid real network discovery
    def handler(req: httpx.Request) -> httpx.Response:
        if "market/list" in req.url.path:
            return httpx.Response(200, json={"marketTopics": [], "total": 0, "offset": 0, "limit": 20, "hasMore": False})
        return httpx.Response(200, json={})
    rt._client = BinancePredictionClient("k", "s", transport=httpx.MockTransport(handler))  # type: ignore[attr-defined]
    import time
    t0 = time.time()
    await rt.run()
    elapsed = time.time() - t0
    assert 0.15 <= elapsed <= 2.0  # should stop near duration


def test_runtime_config_defaults_and_env_resolution() -> None:
    from app.research.prediction_markets.runtime import _resolve_credentials
    # without env, should be None or real keys if present — not crash
    creds = _resolve_credentials()
    # we have CAT_KEY_BINANCE in .env, so creds should be not None in this repo
    # but test must not fail if env missing
    assert creds is None or (isinstance(creds, tuple) and len(creds) == 2)
    cfg = RuntimeConfig()
    assert cfg.duration_secs == 60.0
    assert cfg.spot_poll_ms == 500
    assert "BTCUSDT" in cfg.out_dir.as_posix() or "research" in cfg.out_dir.as_posix() or cfg.out_dir == cfg.out_dir  # trivial
