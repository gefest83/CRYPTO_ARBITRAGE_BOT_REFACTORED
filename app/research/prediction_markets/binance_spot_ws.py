"""Binance Spot WebSocket client — research only, real-time BTC/ETH bookTicker.

Uses wss://stream.binance.com:9443/ws (or combined stream) — public, no auth.
Preserves exact local receive time and exchange event time, does not fabricate.

Subscription topics (exact, documented):
- ["btcusdt@bookTicker","ethusdt@bookTicker"] via {"method":"SUBSCRIBE","params":[...],"id":1}

Timestamps:
- captured_at_ms = int(time.time()*1000) at WS receive
- exchange_ts_ms = event time E (or T) from Binance payload, preserved
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

from app.config.logging_config import get_logger
from app.research.prediction_markets import endpoints as ep
from app.research.prediction_markets.collector.collector import HistoricalCollector

logger = get_logger("research.binance_spot_ws")

BINANCE_WS_URL = ep.BINANCE_SPOT_WS_BASE
BINANCE_WS_TOPICS = ep.BINANCE_SPOT_SUBSCRIPTION_TOPICS
# Alternative combined stream URL for single connection
BINANCE_WS_COMBINED_URL = ep.BINANCE_SPOT_WS_COMBINED

BACKOFF_BASE = 0.5
BACKOFF_MAX = 30.0


def build_spot_subscription(topics: list[str] | None = None) -> dict[str, Any]:
    """Exact subscription message for Binance Spot WS."""
    return {
        "method": "SUBSCRIBE",
        "params": topics or BINANCE_WS_TOPICS,
        "id": 1,
    }


def parse_spot_ws_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Parse Binance Spot bookTicker WS message into collector ingest format.

    Expected payloads:
    - BookTicker: {"e":"bookTicker","E":123456789,"s":"BTCUSDT","b":"68000.00","B":"0.5","a":"68000.50","A":"0.6"}
    - Combined stream wraps: {"stream":"btcusdt@bookTicker","data":{...}}
    - Trade: {"e":"trade","E":..., "s":"BTCUSDT","p":"68000.00","q":"0.1"}

    Preserves exchange timestamps, does not fabricate.
    Returns dict with symbol, bid, ask, bid_qty, ask_qty, exchange_ts_ms, sequence (if any).
    """
    if not isinstance(msg, dict):
        return None
    # Handle combined stream wrapper
    if "data" in msg and isinstance(msg["data"], dict) and "e" in msg["data"]:
        msg = msg["data"]
    # Also handle result ack
    if msg.get("result") is None and msg.get("id") == 1 and "e" not in msg:
        return None  # subscription ack

    event = msg.get("e")
    # bookTicker and trade are relevant; ignore other events
    if event not in (None, "bookTicker", "trade", "aggTrade"):
        # For bookTicker without e field (some streams), still handle if has s/b/a
        if not ("s" in msg and "b" in msg and "a" in msg):
            return None

    symbol = str(msg.get("s") or msg.get("symbol") or "").upper()
    if not symbol:
        return None
    # BTC/ETH only check - caller also filters, but double-check
    base = symbol.replace("USDT", "")
    if base not in ("BTC", "ETH"):
        return None

    exchange_ts = msg.get("E") or msg.get("T") or msg.get("eventTime") or msg.get("timestamp") or msg.get("ts")
    # Do not fabricate if missing -> keep None
    try:
        exchange_ts_ms = int(exchange_ts) if exchange_ts is not None else None
    except Exception:
        exchange_ts_ms = None

    # Sequence / updateId for gap detection
    sequence = msg.get("u") or msg.get("U") or msg.get("lastUpdateId") or msg.get("seq") or msg.get("sequence")

    # Extract bid/ask
    bid = msg.get("b") or msg.get("bidPrice") or msg.get("bid")
    ask = msg.get("a") or msg.get("askPrice") or msg.get("ask")
    bid_qty = msg.get("B") or msg.get("bidQty") or msg.get("bid_qty")
    ask_qty = msg.get("A") or msg.get("askQty") or msg.get("ask_qty")
    last_price = msg.get("p") or msg.get("price") or msg.get("lastPrice")

    out: dict[str, Any] = {
        "symbol": symbol,
        "bid": str(bid) if bid is not None else None,
        "ask": str(ask) if ask is not None else None,
        "bid_qty": str(bid_qty) if bid_qty is not None else None,
        "ask_qty": str(ask_qty) if ask_qty is not None else None,
        "exchange_ts_ms": exchange_ts_ms,
        "sequence": int(sequence) if sequence is not None else None,
        "last_price": str(last_price) if last_price is not None else None,
        "is_trade": event in ("trade", "aggTrade"),
        "raw": msg,
    }
    # Clean None bid/ask for trade events where only price exists
    if out["bid"] is None and out["ask"] is None and out["last_price"] is None:
        return None
    return out


class BinanceSpotWSClient:
    """Real-time Binance Spot bookTicker for BTCUSDT/ETHUSDT."""

    def __init__(
        self,
        collector: HistoricalCollector,
        ws_url: str = BINANCE_WS_URL,
        topics: list[str] | None = None,
        connect_factory: Callable | None = None,
    ) -> None:
        self.collector = collector
        self.ws_url = ws_url
        self.topics = topics or BINANCE_WS_TOPICS
        self._connect_factory = connect_factory
        self._stop = asyncio.Event()

    def build_subscription(self, topics: list[str] | None = None) -> dict[str, Any]:
        return build_spot_subscription(topics or self.topics)

    async def _connect(self):
        import websockets  # type: ignore[import-not-found]

        factory = self._connect_factory or websockets.connect
        # For Binance, no auth headers needed
        return await factory(self.ws_url)

    async def run(self, stop: asyncio.Event | None = None) -> None:
        if stop is not None:
            self._stop = stop
        backoff = BACKOFF_BASE
        while not self._stop.is_set():
            ws = None
            try:
                ws = await self._connect()
                sub = self.build_subscription()
                await ws.send(json.dumps(sub))
                logger.info("binance_spot_ws_subscribed", extra={"topics": self.topics})

                backoff = BACKOFF_BASE
                async for raw_msg in ws:
                    if self._stop.is_set():
                        break
                    captured = int(time.time() * 1000)
                    try:
                        msg = json.loads(raw_msg) if isinstance(raw_msg, (str, bytes)) else raw_msg
                    except Exception:
                        continue
                    parsed = parse_spot_ws_message(msg)
                    if parsed is None:
                        continue
                    # Ingest with exact captured time, preserved exchange time
                    self.collector.ingest_spot(parsed, captured_at_ms=captured)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.collector.on_reconnect(f"binance_spot_ws_error:{str(exc)[:100]}")
                logger.warning("binance_spot_ws_reconnect", extra={"error": str(exc)[:300], "backoff": backoff})
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
            if not self._stop.is_set():
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, BACKOFF_MAX)

        logger.info("binance_spot_ws_stopped")
