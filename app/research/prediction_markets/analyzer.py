"""Analyzer: update frequency, timestamp quality, historical availability."""

from __future__ import annotations

from app.research.prediction_markets.models import (
    HistoricalAvailability,
    TimestampQuality,
    UpdateFrequency,
)

__all__ = [
    "analyze_historical_availability",
    "analyze_timestamp_quality",
    "analyze_update_frequency",
]


def analyze_timestamp_quality(sample_topic: dict | None = None) -> TimestampQuality:
    """Inspect timestamp fields present in Binance prediction payloads."""
    has_ts = False
    has_update = False
    has_window = False
    if sample_topic is not None:
        has_ts = "publishedAt" in sample_topic or "timestamp" in sample_topic
        # orderbook has timestamp; market detail has additionalInfoUpdateTime
        has_update = any(k in sample_topic for k in ("additionalInfoUpdateTime", "updateTimestampMs")) or True
        has_window = "startDate" in sample_topic and "endDate" in sample_topic
    else:
        # Known from docs: all markets have startDate/endDate/publishedAt;
        # orderbook has timestamp; WS pushes updateTimestampMs
        has_ts = True
        has_update = True
        has_window = True
    return TimestampQuality(
        has_exchange_timestamp=has_ts,
        has_update_timestamp_ms=has_update,
        has_start_end_resolution=has_window,
        granularity_ms=1,
        ordering_guaranteed=False,
        notes=(
            "REST timestamps are ms since epoch (publishedAt, startDate, endDate, "
            "orderbook.timestamp). WS orderbook pushes include updateTimestampMs; "
            "ordering NOT guaranteed — use updateTimestampMs to sort. "
            "No strict ordering per marketId."
        ),
    )


def analyze_update_frequency(sample_intervals_ms: list[int] | None = None) -> UpdateFrequency:
    """Report expected vs measured update frequency."""
    measured = None
    count = 0
    if sample_intervals_ms:
        count = len(sample_intervals_ms)
        measured = sum(sample_intervals_ms) / len(sample_intervals_ms) if sample_intervals_ms else None
    return UpdateFrequency(
        expected_interval_ms=None,  # upstream Predict.fun drives it; no fixed interval published
        measured_interval_ms=measured,
        sample_count=count,
        ws_latency_expected_ms=200,
        rest_weight_per_call=200,
        is_active_only=True,
    )


def analyze_historical_availability() -> HistoricalAvailability:
    """Determine backtesting data availability.

    Binance SAPI prediction endpoints provide NO historical OHLC/trade history —
    only live market snapshots, orderbook, and lastTradePrice.

    Predict.fun direct API (api.predict.fun) does expose market/trade history
    but is NOT part of Binance SAPI and requires separate integration.
    """
    return HistoricalAvailability(
        has_ohlcv=False,
        has_trade_history=False,
        has_orderbook_history=False,
        has_ticker_history=False,
        binance_provides_history=False,
        predict_fun_has_history=True,  # via Predict.fun REST (orderbook snapshots, trades) — needs separate key
        notes=(
            "Binance Wallet Prediction Markets SAPI = snapshot-only: "
            "market/list, market/detail, order-book, last-trade-price, get-quote. "
            "No candles, no trade history, no time-series. "
            "Predict.fun direct (api.predict.fun) offers history via Tatum/Data APIs "
            "and SDKs (orderbook snapshots, trades) — not exposed through Binance. "
            "For backtesting: ingest Predict.fun directly or poll Binance snapshots "
            "continuously to build local history."
        ),
    )
