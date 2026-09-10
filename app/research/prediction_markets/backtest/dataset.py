"""Dataset loader from Phase 2 CollectorStore (deterministic, offline)."""

from __future__ import annotations

import json
import pathlib
from decimal import Decimal
from typing import Any

from app.research.prediction_markets.collector.models import (
    PredictionOrderbookObservation,
    SpotObservation,
    SourceKind,
)
from app.research.prediction_markets.collector.storage import CollectorStore

__all__ = ["load_dataset", "build_synthetic_dataset"]


def load_dataset(store: CollectorStore) -> tuple[list[SpotObservation], list[PredictionOrderbookObservation]]:
    """Load and split observations; deterministic sort, BNB excluded already at collector but re-filter."""
    spots: list[SpotObservation] = []
    preds: list[PredictionOrderbookObservation] = []
    for obs in store.all_observations():
        # re-validate scope
        if obs.symbol.replace("USDT", "") not in ("BTC", "ETH"):
            continue
        if obs.is_expired:
            continue
        if isinstance(obs, SpotObservation):
            spots.append(obs)
        elif isinstance(obs, PredictionOrderbookObservation):
            # only 5m/15m
            if obs.duration is not None and obs.duration not in ("5m", "15m"):
                continue
            preds.append(obs)
        else:
            # ignore PredictionTradeObservation for main analyzer (uses orderbook)
            continue
    spots.sort(key=lambda o: (o.exchange_ts_ms or 0, o.captured_at_ms))
    preds.sort(key=lambda o: (o.update_ts_ms or o.exchange_ts_ms or 0, o.captured_at_ms))
    return spots, preds


def build_synthetic_dataset(
    scenario: str = "valid_repricing",
    symbol: str = "BTCUSDT",
    duration: str = "5m",
    market_id: int = 9001,
    fee_bps: int = 200,
) -> tuple[list[SpotObservation], list[PredictionOrderbookObservation]]:
    """Helper for tests/demos — generates minimal synthetic observations for each scenario."""

    T0 = 1_748_131_200_000
    RES = T0 + 300_000 if duration == "5m" else T0 + 900_000

    def spot(ts: int, bid: str, ask: str) -> SpotObservation:
        return SpotObservation(
            captured_at_ms=ts + 5,
            source=SourceKind.SPOT_ORDERBOOK,
            symbol=symbol,
            exchange_ts_ms=ts,
            bid=Decimal(bid),
            ask=Decimal(ask),
            raw={"symbol": symbol, "bid": bid, "ask": ask, "exchange_ts_ms": ts},
        )

    def pred(ts: int, bid: str, ask: str, outcome: str = "YES") -> PredictionOrderbookObservation:
        bb = Decimal(bid)
        ba = Decimal(ask)
        return PredictionOrderbookObservation(
            captured_at_ms=ts + 5,
            source=SourceKind.PREDICTION_ORDERBOOK,
            symbol=symbol,
            market_id=market_id,
            token_id=f"tok_{market_id}_{outcome.lower()}",
            duration=duration,
            exchange_ts_ms=ts,
            update_ts_ms=ts,
            sequence=ts,
            resolution_ms=RES,
            time_to_resolution_ms=RES - ts,
            outcome=outcome,
            best_bid=bb,
            best_ask=ba,
            bids=((bb, Decimal("5000")),),
            asks=((ba, Decimal("3000")),),
            raw={"marketId": market_id, "tokenId": f"tok_{market_id}_{outcome.lower()}", "updateTimestampMs": ts, "bids": [[bid, "5000"]], "asks": [[ask, "3000"]]},
        )

    # baseline spots: quiet then impulse
    if scenario == "valid_repricing":
        spots = [
            spot(T0 + 0, "68000", "68002"),
            spot(T0 + 1000, "68200", "68202"),  # +~29 bps impulse (up)
        ]
        preds = [
            pred(T0 + 500, "0.51", "0.53", "YES"),  # before t0 (1000)
            pred(T0 + 1200, "0.51", "0.53", "YES"),  # still flat
            pred(T0 + 1500, "0.54", "0.56", "YES"),  # reprices +0.03 after 500ms
            pred(T0 + 2000, "0.55", "0.57", "YES"),
        ]
        return spots, preds
    if scenario == "no_repricing":
        spots = [spot(T0 + 0, "68000", "68002"), spot(T0 + 1000, "68200", "68202")]
        preds = [pred(T0 + 500, "0.51", "0.53"), pred(T0 + 1200, "0.51", "0.53"), pred(T0 + 5000, "0.51", "0.53")]
        return spots, preds
    if scenario == "delayed_repricing":
        spots = [spot(T0 + 0, "68000", "68002"), spot(T0 + 1000, "68200", "68202")]
        preds = [pred(T0 + 500, "0.51", "0.53"), pred(T0 + 8000, "0.54", "0.56")]  # 7000ms lag
        return spots, preds
    if scenario == "negative_repricing":
        spots = [spot(T0 + 0, "68000", "68002"), spot(T0 + 1000, "68200", "68202")]
        preds = [pred(T0 + 500, "0.51", "0.53"), pred(T0 + 1500, "0.48", "0.50")]  # adverse
        return spots, preds
    if scenario == "spread_fee_impact":
        spots = [spot(T0 + 0, "68000", "68002"), spot(T0 + 1000, "68200", "68202")]
        # wide spread 0.04, fee 200bps -> net negative even with +0.03
        preds = [pred(T0 + 500, "0.49", "0.53"), pred(T0 + 1500, "0.52", "0.56")]  # gross 0.03 but spread 0.04 half=0.02
        return spots, preds
    if scenario == "insufficient_liquidity":
        spots = [spot(T0 + 0, "68000", "68002"), spot(T0 + 1000, "68200", "68202")]
        p1 = pred(T0 + 500, "0.51", "0.53")
        # shallow depth
        p1 = p1.model_copy(update={"bids": ((Decimal("0.51"), Decimal("1")),), "asks": ((Decimal("0.53"), Decimal("1")),)})
        p2 = pred(T0 + 1500, "0.54", "0.56")
        p2 = p2.model_copy(update={"bids": ((Decimal("0.54"), Decimal("1")),), "asks": ((Decimal("0.56"), Decimal("1")),)})
        return spots, [p1, p2]
    if scenario == "expired":
        spots = [spot(T0 + 0, "68000", "68002"), spot(T0 + 1000, "68200", "68202")]
        # pred after expiry
        exp_pred = PredictionOrderbookObservation(
            captured_at_ms=RES + 100,
            source=SourceKind.PREDICTION_ORDERBOOK,
            symbol=symbol,
            market_id=market_id,
            token_id="tok",
            duration=duration,
            exchange_ts_ms=RES + 50,
            update_ts_ms=RES + 50,
            resolution_ms=RES,
            time_to_resolution_ms=RES - (RES + 50),
            outcome="YES",
            best_bid=Decimal("0.51"),
            best_ask=Decimal("0.53"),
            bids=((Decimal("0.51"), Decimal("5000")),),
            asks=((Decimal("0.53"), Decimal("3000")),),
        )
        return spots, [pred(T0 + 500, "0.51", "0.53"), exp_pred]
    if scenario == "out_of_order":
        spots = [spot(T0 + 1000, "68200", "68202"), spot(T0 + 0, "68000", "68002")]  # out-of-order, analyzer sorts
        preds = [pred(T0 + 1500, "0.54", "0.56"), pred(T0 + 500, "0.51", "0.53")]
        return spots, preds
    # default
    return build_synthetic_dataset("valid_repricing", symbol, duration, market_id, fee_bps)
