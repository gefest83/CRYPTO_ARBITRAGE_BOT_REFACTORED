"""Forecast model interface + deterministic mock (offline, no torch)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Protocol, Sequence

from app.strategies.kronos.types import Candle, ForecastBar

__all__ = ["KronosForecastModel", "MockKronosPredictor"]


class KronosForecastModel(Protocol):
    """Predictor contract — real KronosPredictor will implement this later."""

    def predict(self, candles: Sequence[Candle], pred_len: int) -> Sequence[ForecastBar]:
        """Return exactly pred_len forecast bars, strictly after last candle. No lookahead."""
        ...


@dataclass(frozen=True, slots=True)
class MockKronosPredictor:
    """Deterministic mock — no torch, no network.

    Produces a flat or drifting forecast based on injected params.
    - drift_bps_per_bar: basis points added to close per predicted bar
    - noise_seed: if set, uses hashlib to produce deterministic per-bar noise
    - fixed_closes: if provided, returns exactly those closes (for tests)
    """

    drift_bps_per_bar: Decimal = Decimal("0")
    fixed_closes: tuple[Decimal, ...] | None = None
    # optional: return same close for all bars if drift 0

    def predict(self, candles: Sequence[Candle], pred_len: int) -> Sequence[ForecastBar]:
        if not candles:
            raise ValueError("no candles for prediction")
        if pred_len <= 0:
            raise ValueError("pred_len must be >0")
        last = candles[-1]
        base_close = last.close
        base_ts = last.timestamp
        if base_ts.tzinfo is None:
            base_ts = base_ts.replace(tzinfo=timezone.utc)

        out: list[ForecastBar] = []
        if self.fixed_closes is not None:
            if len(self.fixed_closes) != pred_len:
                raise ValueError(f"fixed_closes len {len(self.fixed_closes)} != pred_len {pred_len}")
            for i, close in enumerate(self.fixed_closes):
                ts = base_ts + timedelta(minutes=i + 1)
                out.append(
                    ForecastBar(
                        timestamp=ts,
                        open=close,
                        high=close,
                        low=close,
                        close=close,
                        volume=Decimal("0"),
                    )
                )
            return tuple(out)

        # drift mode
        drift = self.drift_bps_per_bar
        for i in range(pred_len):
            # step i+1
            factor = Decimal("1") + drift * Decimal(str(i + 1)) / Decimal("10000")
            close = (base_close * factor).quantize(Decimal("0.00000001"))
            ts = base_ts + timedelta(minutes=i + 1)
            out.append(
                ForecastBar(
                    timestamp=ts,
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=Decimal("0"),
                )
            )
        return tuple(out)
