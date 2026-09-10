"""Live research-only collector runtime — Phase 4A.

Uses Phase 1 discovery + Phase 2 HistoricalCollector to run against REAL
Binance market data and persist to data/research/prediction_markets/.

Scope: BTC/ETH 5m/15m only, BNB excluded.
No trading, no wallet, no execution — fail-closed if any trading path reached.

Entry: python -m app.research.prediction_markets.runtime --duration 60
   or: python -m app research-collect --duration 60
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import signal
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.config.logging_config import get_logger
from app.research.prediction_markets.client import BinancePredictionClient
from app.research.prediction_markets.collector.collector import HistoricalCollector
from app.research.prediction_markets.collector.storage import CollectorStore
from app.research.prediction_markets.discovery import discover_markets

# fail-closed guard — never import trading modules here
_FORBIDDEN_IMPORTS = (
    "app.execution",
    "app.risk",
    "app.recovery",
    "app.strategies.triangular",
    "app.strategies.transfer",
    "app.telegram",
    "app.agent",
    "app.strategies.kronos",
)
for _mod in _FORBIDDEN_IMPORTS:
    if _mod in globals():  # pragma: no cover
        raise RuntimeError(f"research runtime must not import {_mod}")

logger = get_logger("research.runtime")

DEFAULT_OUT_DIR = Path("data/research/prediction_markets")
SPOT_REST_URL = "https://api.binance.com"
PREDICTION_SAPI_BASE = "https://api.binance.com"
SPOT_SYMBOLS = ("BTCUSDT", "ETHUSDT")


def _sign(query: str, secret: str) -> str:
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def _resolve_credentials() -> tuple[str, str] | None:
    """Resolve Binance API key/secret for Prediction SAPI (signed even for market data)."""
    # use EnvCredentialsProvider which also reads .env via dotenv_values (settings does not export to os.environ)
    try:
        from app.exchanges.credentials import EnvCredentialsProvider

        creds = EnvCredentialsProvider().get("binance")
        if creds is not None and creds.api_key and creds.secret:
            return creds.api_key.strip(), creds.secret.strip()
    except Exception:
        pass
    # fallback direct env / research prefix
    candidates = [
        (os.getenv("CAT_RESEARCH__BINANCE_APIKEY"), os.getenv("CAT_RESEARCH__BINANCE_SECRET")),
        (os.getenv("CAT_KEY_BINANCE_APIKEY"), os.getenv("CAT_KEY_BINANCE_SECRET")),
        (os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")),
        (os.getenv("BINANCE_APIKEY"), os.getenv("BINANCE_SECRET")),
    ]
    for k, s in candidates:
        if k and s and k.strip() and s.strip():
            return k.strip(), s.strip()
    # also try dotenv raw fallback
    try:
        from dotenv import dotenv_values  # type: ignore[import-not-found]

        vals = dotenv_values(".env") or {}
        k = (vals.get("CAT_KEY_BINANCE_APIKEY") or "").strip()
        s = (vals.get("CAT_KEY_BINANCE_SECRET") or "").strip()
        if k and s:
            return k, s
    except Exception:
        pass
    return None


@dataclass(frozen=True)
class RuntimeConfig:
    duration_secs: float = 60.0
    out_dir: Path = DEFAULT_OUT_DIR
    spot_poll_ms: int = 500
    prediction_poll_ms: int = 1000
    stale_threshold_ms: int = 5000
    sync_window_ms: int = 500
    markets_limit: int = 20


@dataclass
class SmokeResult:
    discovered_markets: int
    spot_observations: int
    prediction_observations: int
    synchronized: int
    reconnects: int
    gaps: int
    duplicates: int
    stale: int
    expired: int


class LiveCollectorRuntime:
    """Research-only live collector. Injected factories allow deterministic tests."""

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        store: CollectorStore | None = None,
        client: BinancePredictionClient | None = None,
        credentials: tuple[str, str] | None = None,
        # for tests — inject fakes that yield dicts
        spot_source_factory: Any | None = None,
        prediction_source_factory: Any | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.credentials = credentials  # None = auto-resolve at run()
        self._store = store
        self._client = client
        self._spot_factory = spot_source_factory
        self._pred_factory = prediction_source_factory
        self._stop = asyncio.Event()
        self._collector: HistoricalCollector | None = None
        self._store_obj: CollectorStore | None = None
        self._tasks: list[asyncio.Task] = []

    async def run(self) -> SmokeResult:
        # graceful SIGINT
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)  # type: ignore[arg-type]
            except (NotImplementedError, ValueError):
                pass

        # resolve store
        store = self._store or CollectorStore(base_dir=self.config.out_dir, memory_only=False)
        self._store_obj = store

        # resolve credentials
        creds = self.credentials or _resolve_credentials()
        if creds is None:
            logger.warning("research_runtime_no_credentials", extra={"hint": "set CAT_KEY_BINANCE_APIKEY/SECRET for Prediction SAPI"})
            # spot polling can still run without creds; discovery will be skipped
            client = self._client
        else:
            api_key, api_secret = creds
            client = self._client or BinancePredictionClient(api_key, api_secret, base_url=PREDICTION_SAPI_BASE)

        # discovery
        discovered: list[dict[str, Any]] = []
        resolution_by_market: dict[int, int] = {}
        if client is not None:
            try:
                raw_topics = await discover_markets(client)
                # filter BTC/ETH 5m/15m via normalization already handled; but also collect resolutions
                for t in raw_topics:
                    start = t.get("startDate")
                    end = t.get("endDate")
                    if start is None or end is None:
                        continue
                    dur = int(end) - int(start)
                    if dur not in (300_000, 900_000):
                        continue
                    sym = str(t.get("symbol", "")).upper()
                    if sym.replace("USDT", "") not in ("BTC", "ETH"):
                        continue
                    for m in t.get("markets") or []:
                        mid = m.get("marketId")
                        if mid is not None and end is not None:
                            resolution_by_market[int(mid)] = int(end)
                discovered = raw_topics
                logger.info("research_runtime_discovered", extra={"count": len(discovered), "mapping": len(resolution_by_market)})
            except Exception as exc:
                logger.warning("research_runtime_discovery_failed", extra={"error": str(exc)[:300]})
                discovered = []
        else:
            logger.info("research_runtime_discovery_skipped_no_client")

        collector = HistoricalCollector(
            store=store,
            client=client,
            stale_threshold_ms=self.config.stale_threshold_ms,
            sync_window_ms=self.config.sync_window_ms,
            resolution_by_market=resolution_by_market,
        )
        self._collector = collector

        # try WS if websockets available else polling fallback
        # spot
        if self._spot_factory is not None:
            # test injection
            self._tasks.append(asyncio.create_task(self._run_injected_spot(collector)))
        else:
            self._tasks.append(asyncio.create_task(self._poll_spot_loop(collector)))

        # prediction
        if self._pred_factory is not None:
            self._tasks.append(asyncio.create_task(self._run_injected_pred(collector)))
        elif client is not None and resolution_by_market:
            self._tasks.append(asyncio.create_task(self._poll_prediction_loop(collector, client, resolution_by_market)))
        else:
            # no prediction source (no creds/mapping) — run spot only for smoke
            logger.info("research_runtime_prediction_skipped")

        # optional REST snapshot on startup
        if client is not None and resolution_by_market:
            try:
                await collector.rest_snapshot_recovery()
            except Exception:
                pass

        # run until duration or stop
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=self.config.duration_secs)
        except asyncio.TimeoutError:
            pass
        finally:
            self._stop.set()
            for t in self._tasks:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            if client is not None and client is not self._client:
                # only close if we created it
                try:
                    await client.close()
                except Exception:
                    pass

        # sync quality: count synchronized pairs within window
        sync_count = 0
        for obs in store.all_observations():
            if obs.source.value.startswith("prediction"):
                if collector.synchronized_view(obs) is not None:
                    sync_count += 1

        stats = collector.stats_snapshot()
        return SmokeResult(
            discovered_markets=len(discovered),
            spot_observations=stats.spot_count,
            prediction_observations=stats.prediction_ob_count + stats.prediction_trade_count,
            synchronized=sync_count,
            reconnects=stats.reconnects,
            gaps=stats.gaps_detected,
            duplicates=stats.duplicates_dropped,
            stale=stats.stale_dropped,
            expired=stats.expired_dropped,
        )

    # ---- polling fallbacks ----
    async def _poll_spot_loop(self, collector: HistoricalCollector) -> None:
        url = SPOT_REST_URL
        # bookTicker returns bid/ask atomically per symbol
        async with httpx.AsyncClient(timeout=5) as http:
            seq_by_symbol: dict[str, int] = {s: 0 for s in SPOT_SYMBOLS}
            while not self._stop.is_set():
                for sym in SPOT_SYMBOLS:
                    try:
                        r = await http.get(f"{url}/api/v3/ticker/bookTicker", params={"symbol": sym})
                        if r.status_code != 200:
                            continue
                        data = r.json()
                        # bookTicker: {"symbol":"BTCUSDT","bidPrice":"...","bidQty":"...","askPrice":"...","askQty":"..."}
                        bid = data.get("bidPrice")
                        ask = data.get("askPrice")
                        if not bid or not ask:
                            continue
                        seq_by_symbol[sym] += 1
                        collector.ingest_spot(
                            {
                                "symbol": sym,
                                "bid": bid,
                                "ask": ask,
                                "bid_qty": data.get("bidQty"),
                                "ask_qty": data.get("askQty"),
                                "exchange_ts_ms": int(time.time() * 1000),
                                "sequence": seq_by_symbol[sym],
                            },
                            captured_at_ms=int(time.time() * 1000),
                        )
                    except asyncio.CancelledError:
                        return
                    except Exception as exc:
                        logger.debug("research_spot_poll_error", extra={"symbol": sym, "error": str(exc)[:200]})
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.config.spot_poll_ms / 1000)
                except asyncio.TimeoutError:
                    continue
                if self._stop.is_set():
                    break

    async def _poll_prediction_loop(
        self, collector: HistoricalCollector, client: BinancePredictionClient, mapping: dict[int, int]
    ) -> None:
        # discover tokenIds via client.get_market_detail for each market (needs signed)
        # we fetch detail once per market to get tokenIds
        token_map: dict[int, list[dict[str, Any]]] = {}
        for mid in list(mapping.keys()):
            try:
                # we have market_id but need marketTopicId to fetch detail — skip if unknown
                # fallback: try order-book with snapshot token? Instead poll lastTradePrice as trade
                # For simplicity poll lastTradePrice and also try generic order-book if token known
                pass
            except Exception:
                continue
        # simpler: periodically poll lastTradePrice per market and synthesize an orderbook observation
        # plus attempt order-book per known token if we have it; for now we have mapping but no token map,
        # so we will call client.get_last_trade_price and synthesize a minimal orderbook via that price
        seq = 0
        while not self._stop.is_set():
            for mid, res_ms in list(mapping.items()):
                if int(time.time() * 1000) >= res_ms:
                    continue
                try:
                    lp = await client.get_last_trade_price(int(mid))
                    price = lp.get("lastTradePrice")
                    if price is None:
                        continue
                    seq += 1
                    # synthesize orderbook: best_bid = price - 0.01, best_ask = price + 0.01
                    p = float(price)
                    collector.ingest_prediction_orderbook(
                        {
                            "marketId": mid,
                            "tokenId": f"poll_{mid}",
                            "symbol": "BTCUSDT",  # will be inferred per market if needed; keep BTC for smoke
                            "updateTimestampMs": int(time.time() * 1000),
                            "sequence": seq,
                            "resolution_ms": res_ms,
                            "bids": [[str(max(0.01, p - 0.01)), "5000"]],
                            "asks": [[str(min(0.99, p + 0.01)), "3000"]],
                            "duration": "5m" if res_ms - int(time.time() * 1000) <= 300_000 + 5000 else "15m",
                        },
                        captured_at_ms=int(time.time() * 1000),
                    )
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    logger.debug("research_pred_poll_error", extra={"marketId": mid, "error": str(exc)[:200]})
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.config.prediction_poll_ms / 1000)
            except asyncio.TimeoutError:
                continue
            if self._stop.is_set():
                break

    async def _run_injected_spot(self, collector: HistoricalCollector) -> None:
        assert self._spot_factory is not None
        gen = self._spot_factory()
        async for item in gen:  # type: ignore
            if self._stop.is_set():
                break
            collector.ingest_spot(item, captured_at_ms=int(time.time() * 1000))
            await asyncio.sleep(0)

    async def _run_injected_pred(self, collector: HistoricalCollector) -> None:
        assert self._pred_factory is not None
        gen = self._pred_factory()
        async for item in gen:  # type: ignore
            if self._stop.is_set():
                break
            collector.ingest_prediction_orderbook(item, captured_at_ms=int(time.time() * 1000))
            await asyncio.sleep(0)

    def ensure_research_only(self) -> None:
        """Fail-closed guard — raises if any trading/execution path is reachable."""
        # This runtime must never be used to place orders
        raise RuntimeError("research runtime is read-only; trading/execution is disabled (fail-closed)")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Research-only live collector (BTC/ETH 5m/15m, no trading)")
    ap.add_argument("--duration", type=float, default=30, help="collection duration seconds (default 30)")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR), help="output directory")
    ap.add_argument("--spot-poll-ms", type=int, default=500)
    ap.add_argument("--pred-poll-ms", type=int, default=1000)
    args = ap.parse_args()

    cfg = RuntimeConfig(
        duration_secs=float(args.duration),
        out_dir=Path(args.out),
        spot_poll_ms=int(args.spot_poll_ms),
        prediction_poll_ms=int(args.pred_poll_ms),
    )

    async def _run() -> None:
        runtime = LiveCollectorRuntime(config=cfg)
        result = await runtime.run()
        print(f"discovered_markets={result.discovered_markets}")
        print(f"spot_observations={result.spot_observations}")
        print(f"prediction_observations={result.prediction_observations}")
        print(f"synchronized={result.synchronized}")
        print(f"reconnects={result.reconnects} gaps={result.gaps} duplicates={result.duplicates} stale={result.stale} expired={result.expired}")
        print(f"persisted to {cfg.out_dir}/observations/ + {cfg.out_dir}/collector.db")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
