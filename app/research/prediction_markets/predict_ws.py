"""Predict.fun WebSocket client — research only, real-time orderbook for BTC/ETH 5m/15m.

Uses wss://ws.predict.fun/ws with CAT_RESEARCH__PREDICT_API_KEY (x-api-key).
Scope: BTC/ETH only, BNB excluded, 5m/15m only. No trading.

This module is deliberately dependency-light: if ``websockets`` is not installed,
the client falls back to REST polling (caller should handle). For tests, the
websockets.connect call is mockable.

Subscription topics (exact, documented per https://dev.predict.fun/subscription-topics and Rust SDK):
- Predict.fun: wss://ws.predict.fun/ws, no auth for orderbook, topic `predictOrderbook/{marketId}`
  RPC: {"requestId": 0, "method": "subscribe", "params": ["predictOrderbook/123"]}
  One topic per request. marketIds are active BTC/ETH 5m/15m child ids from REST list_categories (BNB excluded).

Timestamps:
- captured_at_ms = int(time.time()*1000) at local receive (exact, not fabricated)
- exchange_ts_ms / updateTimestampMs = from payload (updateTimestampMs, timestamp, E) preserved verbatim
- Do NOT fabricate exchange timestamps if missing -> keep None
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

from app.config.logging_config import get_logger
from app.research.prediction_markets import endpoints as ep
from app.research.prediction_markets.collector.collector import HistoricalCollector
from app.research.prediction_markets.predict_client import PredictFunClient

logger = get_logger("research.predict_ws")

PREDICT_WS_URL = ep.PREDICT_FUN_WS_BASE_MAINNET
# Exact subscription channel used
PREDICT_WS_CHANNEL = ep.PREDICT_FUN_WS_CHANNEL_ORDERBOOK
# Backoff for reconnect
BACKOFF_BASE = 0.5
BACKOFF_MAX = 30.0


def _iso_to_ms(s: str | None) -> int | None:
    if not s:
        return None
    try:
        from datetime import datetime as _dt
        d = _dt.fromisoformat(s.replace("Z", "+00:00"))
        return int(d.timestamp() * 1000)
    except Exception:
        return None


def _filter_btc_eth_5m_15m(categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter categories to BTC/ETH 5m/15m, BNB excluded. Returns list of market entries.

    For live collection, returns the most recent BTC/ETH 5m/15m markets (up to 20) that have
    orderbooks, regardless of start time, to ensure WS has data. Time filtering is handled
    by the collector's stale/expiry checks, not discovery.
    """
    active: list[dict[str, Any]] = []
    for cat in categories:
        slug = str(cat.get("slug", "")).lower()
        if "bnb" in slug:
            continue
        if not ("btc" in slug or "eth" in slug):
            continue
        if not ("5m" in slug or "15m" in slug or "5-min" in slug or "15-min" in slug):
            tags = cat.get("tags", []) or []
            tag_names = " ".join(str(t.get("name", "")).lower() for t in tags)
            if "5 min" not in tag_names and "15 min" not in tag_names:
                continue
        vd = cat.get("variantData") or {}
        symbol = vd.get("priceFeedSymbol") or ("BTCUSDT" if "btc" in slug else "ETHUSDT")
        # Check 15m before 5m to avoid substring false positive ("15m" contains "5m")
        if "15m" in slug or "15-min" in slug:
            duration = "15m"
        elif "5m" in slug or "5-min" in slug:
            duration = "5m"
        else:
            tag_names = " ".join(str(t.get("name", "")).lower() for t in cat.get("tags", []) or [])
            if "15 min" in tag_names:
                duration = "15m"
            elif "5 min" in tag_names:
                duration = "5m"
            else:
                duration = "5m"
        res_ms = _iso_to_ms(cat.get("endsAt"))
        for m in cat.get("markets") or []:
            mid = m.get("id")
            if mid is None:
                continue
            active.append({
                "market_id": int(mid),
                "symbol": symbol,
                "duration": duration,
                "resolution_ms": res_ms,
                "slug": slug,
                "category": cat,
                "market": m,
            })
            if len(active) >= 20:
                break
        if len(active) >= 20:
            break
    # No fallback to far-future markets — if no active, return empty and let caller retry on next discovery
    return active


def build_predict_subscription(market_ids: list[int]) -> list[dict[str, Any]]:
    """Exact subscription messages used for Predict.fun WS — per official docs.

    Official: wss://ws.predict.fun/ws, no auth for orderbook, topic `predictOrderbook/{marketId}`,
    RPC: {"requestId": 0, "method": "subscribe", "params": ["predictOrderbook/123"]}
    One topic per request (see dev.predict.fun/subscription-topics).
    Returns list of JSON-RPC subscribe messages, one per marketId (BTC/ETH 5m/15m, BNB excluded).
    """
    # Deduplicate and sort for determinism
    unique = sorted(set(market_ids))
    return [
        {"requestId": idx, "method": "subscribe", "params": [f"predictOrderbook/{mid}"]}
        for idx, mid in enumerate(unique)
    ]


def build_predict_subscription_single(market_id: int, request_id: int = 0) -> dict[str, Any]:
    """Single topic subscription — matches documented RPC format."""
    return {"requestId": request_id, "method": "subscribe", "params": [f"predictOrderbook/{market_id}"]}


def parse_predict_ws_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a Predict.fun WS orderbook message into collector ingest format.

    Official format (dev.predict.fun/subscription-topics):
    - Push: {"type":"M","topic":"predictOrderbook/123","data":{"marketId":123,"updateTimestampMs":..., "bids":[[p,q]], "asks":..., "version":1, ...}}
    - Ack:  {"type":"R","requestId":0,"success":true}
    Also handles legacy direct: {"marketId":..., "bids":..., "asks":..., "updateTimestampMs":...}
    Preserves exchange/update timestamps, do not fabricate.
    """
    if not isinstance(msg, dict):
        return None
    # Handle subscription ack / ping/pong / heartbeat
    if msg.get("type") == "R":  # request response
        return None
    if msg.get("type") == "pong" or msg.get("event") == "pong":
        return None
    if msg.get("event") == "subscribed" or msg.get("result") == "subscribed":
        return None
    # Official push is type M with topic predictOrderbook/{id}
    topic = msg.get("topic", "")
    if isinstance(topic, str) and topic.startswith("predictOrderbook/"):
        # Extract marketId from topic if not in data
        try:
            topic_mid = int(topic.split("/")[-1])
        except Exception:
            topic_mid = None
        data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
        # Fallback to whole msg if data empty
        if not data:
            data = msg
        # Inject marketId from topic if missing
        if topic_mid is not None and "marketId" not in data and "market_id" not in data:
            data["marketId"] = topic_mid
    else:
        # Fallback: data may be directly in msg or in msg["data"]
        data = msg.get("data") if isinstance(msg.get("data"), dict) else msg
        # Some wrappers use "payload" or direct
        if not data or ("marketId" not in data and "bids" not in data and "asks" not in data):
            # Try to handle raw orderbook without wrapper
            if "marketId" in msg or "bids" in msg:
                data = msg
            else:
                return None
    # Normalize marketId
    if "marketId" not in data and "market_id" in data:
        data["marketId"] = data["market_id"]
    market_id = data.get("marketId") or data.get("market_id") or data.get("id")
    if market_id is None:
        # Try topic
        if topic and "/" in topic:
            try:
                market_id = int(topic.split("/")[-1])
            except Exception:
                pass
    if market_id is None:
        return None

    # Extract bids/asks in various formats
    bids = data.get("bids") or msg.get("bids") or []
    asks = data.get("asks") or msg.get("asks") or []
    # Some payloads use bestBid/bestAsk or orderCount
    update_ts = data.get("updateTimestampMs") or data.get("update_ts_ms") or data.get("timestamp") or data.get("ts") or data.get("E") or data.get("T") or data.get("version")
    sequence = data.get("sequence") or data.get("seq") or data.get("u") or data.get("lastUpdateId") or data.get("version")
    symbol = data.get("symbol") or data.get("priceFeedSymbol") or ""
    duration = data.get("duration")
    resolution_ms = data.get("resolution_ms") or data.get("endDate") or data.get("resolutionMs")

    # Normalize - preserve, do not fabricate if missing
    # Return raw dict for collector; collector will handle symbol scoping and timestamps
    out: dict[str, Any] = {
        "marketId": int(market_id) if market_id is not None else None,
        "tokenId": str(data.get("tokenId") or data.get("token_id") or f"predict_{market_id}"),
        "symbol": str(symbol).upper() if symbol else "",
        "updateTimestampMs": int(update_ts) if update_ts is not None else None,
        "sequence": int(sequence) if sequence is not None else None,
        "bids": bids,
        "asks": asks,
        "duration": duration,
        "resolution_ms": int(resolution_ms) if resolution_ms is not None else None,
        "marketTopicId": data.get("marketTopicId"),
        "outcome": data.get("outcome"),
        "raw": msg,  # preserve full
    }
    # Clean None symbol to allow collector to infer
    if not out["symbol"]:
        out.pop("symbol", None)
    return out


class PredictFunWSClient:
    """Real-time Predict.fun WS orderbook for BTC/ETH 5m/15m.

    Usage:
        client = PredictFunWSClient(token, collector, rest_client)
        await client.run(stop_event)  # handles discovery, subscribe, reconnect, recovery
    """

    def __init__(
        self,
        token: str,
        collector: HistoricalCollector,
        rest_client: PredictFunClient | None = None,
        ws_url: str = PREDICT_WS_URL,
        connect_factory: Callable | None = None,
    ) -> None:
        self.token = token
        self.collector = collector
        self.rest_client = rest_client
        self.ws_url = ws_url
        self._connect_factory = connect_factory  # for tests: mock websockets.connect
        self._stop = asyncio.Event()
        self._active_markets: list[dict[str, Any]] = []

    async def discover_markets(self) -> list[dict[str, Any]]:
        """Discover BTC/ETH 5m/15m markets via REST, BNB excluded."""
        if self.rest_client is None:
            # Create temporary client for discovery
            from app.research.prediction_markets.predict_client import PredictFunClient as PFC
            # need token
            client = PFC(self.token)
            try:
                cats = await client.list_categories(limit=50)
                items = cats.get("data", []) if isinstance(cats, dict) else []
                return _filter_btc_eth_5m_15m(items)
            finally:
                await client.close()
        else:
            cats = await self.rest_client.list_categories(limit=50)
            items = cats.get("data", []) if isinstance(cats, dict) else []
            return _filter_btc_eth_5m_15m(items)

    def build_subscription(self, market_ids: list[int]) -> list[dict[str, Any]]:
        return build_predict_subscription(market_ids)

    async def _connect(self):
        """Connect with x-api-key header; mockable via connect_factory."""
        import websockets  # type: ignore[import-not-found]

        url = self.ws_url
        # Some deployments require token as query param as well
        # Preserve header for official spec
        headers = {"x-api-key": self.token}
        factory = self._connect_factory or websockets.connect
        # Try with extra_headers (websockets >=12)
        try:
            return await factory(url, extra_headers=headers)
        except TypeError:
            # Fallback for older websockets that use additional_headers
            try:
                return await factory(url, additional_headers=headers)
            except TypeError:
                # Last fallback: query param
                sep = "&" if "?" in url else "?"
                url_q = f"{url}{sep}apiKey={self.token}"
                return await factory(url_q)

    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Main loop: discovery -> subscribe -> listen with reconnect and REST recovery."""
        if stop is not None:
            self._stop = stop
        # Initial discovery
        try:
            self._active_markets = await self.discover_markets()
            logger.info("predict_ws_discovered", extra={"count": len(self._active_markets)})
        except Exception as exc:
            logger.warning("predict_ws_discovery_failed", extra={"error": str(exc)[:300]})
            self._active_markets = []

        backoff = BACKOFF_BASE
        while not self._stop.is_set():
            ws = None
            try:
                ws = await self._connect()
                # Subscribe — one topic per request per official docs (predictOrderbook/{marketId})
                market_ids = [m["market_id"] for m in self._active_markets]
                if market_ids:
                    subs = self.build_subscription(market_ids)
                    for sub in subs:
                        await ws.send(json.dumps(sub))
                    logger.info("predict_ws_subscribed", extra={"marketIds": market_ids, "channel": PREDICT_WS_CHANNEL, "count": len(subs)})
                else:
                    logger.info("predict_ws_no_markets_to_subscribe")

                # Listen loop
                backoff = BACKOFF_BASE  # reset on successful connect
                async for raw_msg in ws:
                    if self._stop.is_set():
                        break
                    captured = int(time.time() * 1000)
                    try:
                        msg = json.loads(raw_msg) if isinstance(raw_msg, (str, bytes)) else raw_msg
                    except Exception:
                        continue
                    # Handle ping
                    if isinstance(msg, dict) and msg.get("type") == "ping":
                        try:
                            await ws.send(json.dumps({"type": "pong"}))
                        except Exception:
                            pass
                        continue
                    parsed = parse_predict_ws_message(msg)
                    if parsed is None:
                        continue
                    # Fill symbol/duration/resolution from active list if missing
                    if not parsed.get("symbol") or not parsed.get("duration") or not parsed.get("resolution_ms"):
                        for am in self._active_markets:
                            if am["market_id"] == parsed.get("marketId"):
                                if not parsed.get("symbol"):
                                    parsed["symbol"] = am["symbol"]
                                if not parsed.get("duration"):
                                    parsed["duration"] = am["duration"]
                                if not parsed.get("resolution_ms") and am.get("resolution_ms"):
                                    parsed["resolution_ms"] = am["resolution_ms"]
                                break
                    # BNB excluded check already done at discovery, but double-check
                    sym = str(parsed.get("symbol", "")).upper().replace("USDT", "")
                    if sym not in ("BTC", "ETH") and sym != "":
                        continue
                    # Ingest with exact captured time, preserved exchange timestamps
                    self.collector.ingest_prediction_orderbook(parsed, captured_at_ms=captured)

                    # Gap/reconnect handling: sequence gaps are counted inside collector
                    # No look-ahead

            except asyncio.CancelledError:
                break
            except Exception as exc:
                # Reconnect with backoff, count reconnect, try REST snapshot recovery
                self.collector.on_reconnect(f"predict_ws_error:{str(exc)[:100]}")
                logger.warning("predict_ws_reconnect", extra={"error": str(exc)[:300], "backoff": backoff})
                try:
                    # REST snapshot recovery for active markets
                    if self._active_markets:
                        # Refresh discovery in case new markets appeared
                        try:
                            fresh = await self.discover_markets()
                            if fresh:
                                self._active_markets = fresh
                        except Exception:
                            pass
                        await self.collector.rest_snapshot_recovery()
                except Exception:
                    pass
                # Backoff
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, BACKOFF_MAX)
                continue
            finally:
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                if self._stop.is_set():
                    break
            # If loop exited without stop, reconnect after backoff
            if not self._stop.is_set():
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, BACKOFF_MAX)

        logger.info("predict_ws_stopped")
