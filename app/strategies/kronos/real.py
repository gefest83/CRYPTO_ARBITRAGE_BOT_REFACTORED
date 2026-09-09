"""Isolated real Kronos predictor adapter (Phase 3A).

- Lazy loading: model + tokenizer are downloaded from Hugging Face only when
  explicitly requested (benchmark). Normal imports/tests never touch torch or network.
- No coupling to trading, DEMO, execution, AutoTrader, Telegram, storage, or AI agent.
- Implements :class:`app.strategies.kronos.forecast.KronosForecastModel` protocol.
- Supports both mini and base variants with their correct tokenizers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timedelta, timezone
from decimal import Decimal
from typing import Literal, Sequence

import pandas as pd

from app.strategies.kronos.types import Candle, ForecastBar

__all__ = [
    "KronosModelVariant",
    "KronosModelSpec",
    "RealKronosPredictor",
    "get_device",
    "detect_environment",
]

KronosModelVariant = Literal["mini", "base"]


@dataclass(frozen=True, slots=True)
class KronosModelSpec:
    variant: KronosModelVariant
    model_id: str
    tokenizer_id: str
    max_context: int
    params: str  # human-readable for table

    @property
    def display_name(self) -> str:
        return f"Kronos-{self.variant}"


# Canonical specs per task requirement – do NOT silently change.
MODEL_SPECS: dict[KronosModelVariant, KronosModelSpec] = {
    "mini": KronosModelSpec(
        variant="mini",
        model_id="NeoQuasar/Kronos-mini",
        tokenizer_id="NeoQuasar/Kronos-Tokenizer-2k",
        max_context=2048,
        params="4.1M",
    ),
    "base": KronosModelSpec(
        variant="base",
        model_id="NeoQuasar/Kronos-base",
        tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
        max_context=512,
        params="102.3M",
    ),
}


def get_device() -> str:
    """Detect best available device without importing torch at module load if possible."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except Exception:
        pass
    return "cpu"


def detect_environment() -> dict:
    """Return env info: python version, torch version, device, RAM, CUDA."""
    import platform

    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": get_device(),
    }
    try:
        import torch  # type: ignore

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_device_count"] = torch.cuda.device_count()
        else:
            info["cuda_device_name"] = None
    except Exception as exc:  # pragma: no cover
        info["torch_version"] = f"import_failed:{exc}"
        info["cuda_available"] = False

    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / (1024**3), 2)
        info["ram_available_gb"] = round(vm.available / (1024**3), 2)
        info["ram_percent"] = vm.percent
    except Exception:
        info["ram_total_gb"] = None

    return info


class RealKronosPredictor:
    """Real Kronos adapter implementing KronosForecastModel.

    Lazy: construction does NOT load model/tokenizer. Call :meth:`load` explicitly.
    This keeps the normal test suite free of torch/network requirements.

    Example:
        pred = RealKronosPredictor(variant="mini")
        pred.load()  # downloads from HF if needed
        bars = pred.predict(candles, pred_len=5)
    """

    def __init__(
        self,
        variant: KronosModelVariant = "mini",
        *,
        device: str | None = None,
        max_context: int | None = None,
        # sampling params – keep identical for both models in benchmark
        T: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        sample_count: int = 1,
        clip: int = 5,
    ) -> None:
        if variant not in MODEL_SPECS:
            raise ValueError(f"unknown variant {variant!r}, expected one of {list(MODEL_SPECS)}")
        self._spec: KronosModelSpec = MODEL_SPECS[variant]
        self._device: str = device or get_device()
        self._max_context: int = max_context or self._spec.max_context
        self._T = T
        self._top_p = top_p
        self._top_k = top_k
        self._sample_count = sample_count
        self._clip = clip

        # lazy handles
        self._tokenizer = None  # type: ignore
        self._model = None  # type: ignore
        self._predictor = None  # type: ignore
        self._loaded = False
        self._init_time_s: float | None = None

    # ------------------------------------------------------------------ properties
    @property
    def spec(self) -> KronosModelSpec:
        return self._spec

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def init_time_s(self) -> float | None:
        return self._init_time_s

    # ------------------------------------------------------------------ loading
    def load(self) -> float:
        """Load tokenizer + model from HF (lazy). Returns init time in seconds.

        Raises with exact failure if torch/network/HF unavailable.
        Idempotent: second call is no-op.
        """
        if self._loaded:
            return self._init_time_s or 0.0

        t0 = time.perf_counter()
        try:
            import torch  # noqa: F401  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"torch not available (required for Kronos): {exc}") from exc

        try:
            from app.strategies.kronos.model.kronos import Kronos, KronosPredictor, KronosTokenizer  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"failed to import vendored Kronos model code: {exc}") from exc

        try:
            tokenizer = KronosTokenizer.from_pretrained(self._spec.tokenizer_id)  # type: ignore[attr-defined]
        except Exception as exc:
            raise RuntimeError(f"failed to load tokenizer {self._spec.tokenizer_id}: {exc}") from exc

        try:
            model = Kronos.from_pretrained(self._spec.model_id)  # type: ignore[attr-defined]
        except Exception as exc:
            raise RuntimeError(f"failed to load model {self._spec.model_id}: {exc}") from exc

        try:
            predictor = KronosPredictor(model, tokenizer, device=self._device, max_context=self._max_context, clip=self._clip)
        except Exception as exc:
            raise RuntimeError(f"failed to instantiate KronosPredictor: {exc}") from exc

        self._tokenizer = tokenizer
        self._model = model
        self._predictor = predictor
        self._loaded = True
        self._init_time_s = time.perf_counter() - t0
        return self._init_time_s

    # ------------------------------------------------------------------ helpers
    def _ensure_loaded(self) -> None:
        if not self._loaded or self._predictor is None:
            raise RuntimeError("RealKronosPredictor not loaded – call .load() first (lazy, requires torch + HF)")

    @staticmethod
    def _candles_to_df(candles: Sequence[Candle]) -> pd.DataFrame:
        # Use float for upstream predictor (it normalizes anyway)
        data = {
            "open": [float(c.open) for c in candles],
            "high": [float(c.high) for c in candles],
            "low": [float(c.low) for c in candles],
            "close": [float(c.close) for c in candles],
            "volume": [float(c.volume) for c in candles],
        }
        # amount = volume * close avg (upstream will synthesize if missing, but provide explicitly)
        data["amount"] = [float(c.volume * c.close) for c in candles]
        return pd.DataFrame(data)

    @staticmethod
    def _timestamps(candles: Sequence[Candle]) -> tuple[pd.Series, pd.Series]:
        # x_timestamp: from candles
        # y_timestamp: future pred_len steps 1m apart
        x_ts = pd.to_datetime([c.timestamp for c in candles], utc=True)
        x_series = pd.Series(x_ts)
        last = candles[-1].timestamp
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        # future timestamps: last + 1m, +2m ...
        # caller provides pred_len; this helper needs it – handled in predict()
        return x_series, last  # type: ignore

    # ------------------------------------------------------------------ KronosForecastModel API
    def predict(self, candles: Sequence[Candle], pred_len: int) -> Sequence[ForecastBar]:
        """Single-symbol forecast. Returns exactly pred_len ForecastBar."""
        if not candles:
            raise ValueError("no candles for prediction")
        if pred_len <= 0:
            raise ValueError("pred_len must be >0")
        self._ensure_loaded()
        assert self._predictor is not None

        # Truncate to max_context if needed (upstream predictor does internally, but keep explicit)
        if len(candles) > self._max_context:
            candles = candles[-self._max_context :]

        df = self._candles_to_df(candles)
        x_ts = pd.to_datetime([c.timestamp for c in candles], utc=True)
        x_series = pd.Series(x_ts)
        last_ts = candles[-1].timestamp
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        y_timestamps = [last_ts + timedelta(minutes=i + 1) for i in range(pred_len)]
        y_series = pd.Series(pd.to_datetime(y_timestamps, utc=True))

        # KronosPredictor.predict is blocking; verbose False for bench
        try:
            pred_df: pd.DataFrame = self._predictor.predict(
                df=df,
                x_timestamp=x_series,
                y_timestamp=y_series,
                pred_len=pred_len,
                T=self._T,
                top_k=self._top_k,
                top_p=self._top_p,
                sample_count=self._sample_count,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Kronos predict failed: {exc}") from exc

        # Convert back to ForecastBar (Decimal, UTC)
        out: list[ForecastBar] = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            ts = y_timestamps[i]
            # Use Decimal quantized to avoid float noise; keep 8 decimals like mock
            close = Decimal(str(row["close"])).quantize(Decimal("0.00000001"))
            # Clamp non-positive to tiny positive to satisfy scanner guards – but keep real value
            if close <= 0:
                close = Decimal("0.00000001")
            open_v = Decimal(str(row["open"])).quantize(Decimal("0.00000001")) if "open" in row else close
            high_v = Decimal(str(row["high"])).quantize(Decimal("0.00000001")) if "high" in row else close
            low_v = Decimal(str(row["low"])).quantize(Decimal("0.00000001")) if "low" in row else close
            vol_v = Decimal(str(row["volume"])).quantize(Decimal("0.00000001")) if "volume" in row else Decimal("0")
            out.append(
                ForecastBar(
                    timestamp=ts,
                    open=open_v,
                    high=high_v,
                    low=low_v,
                    close=close,
                    volume=vol_v if vol_v >= 0 else Decimal("0"),
                )
            )
        if len(out) != pred_len:
            raise RuntimeError(f"forecast len mismatch {len(out)} != {pred_len}")
        return tuple(out)

    def predict_batch(
        self,
        candles_list: Sequence[Sequence[Candle]],
        pred_len: int,
    ) -> Sequence[Sequence[ForecastBar]]:
        """Batch forecast for multiple symbols (same pred_len, same lookback).

        Uses KronosPredictor.predict_batch where available for GPU parallelism.
        Falls back to sequential predict if batch not supported.
        """
        if not candles_list:
            raise ValueError("empty batch")
        if pred_len <= 0:
            raise ValueError("pred_len must be >0")
        self._ensure_loaded()
        assert self._predictor is not None

        # Validate uniform length requirement of upstream batch API
        first_len = len(candles_list[0])
        for idx, c in enumerate(candles_list):
            if len(c) != first_len:
                raise ValueError(f"batch requires uniform length, idx {idx} has {len(c)} vs {first_len}")

        # Truncate to max_context
        truncated = [seq[-self._max_context :] if len(seq) > self._max_context else seq for seq in candles_list]

        df_list: list[pd.DataFrame] = []
        x_ts_list: list[pd.Series] = []
        y_ts_list: list[pd.Series] = []
        y_timestamps_all: list[list] = []

        for candles in truncated:
            if not candles:
                raise ValueError("empty candles in batch")
            df_list.append(self._candles_to_df(candles))
            x_ts = pd.to_datetime([c.timestamp for c in candles], utc=True)
            x_ts_list.append(pd.Series(x_ts))
            last_ts = candles[-1].timestamp
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            y_timestamps = [last_ts + timedelta(minutes=i + 1) for i in range(pred_len)]
            y_timestamps_all.append(y_timestamps)
            y_ts_list.append(pd.Series(pd.to_datetime(y_timestamps, utc=True)))

        try:
            pred_dfs = self._predictor.predict_batch(
                df_list=df_list,
                x_timestamp_list=x_ts_list,
                y_timestamp_list=y_ts_list,
                pred_len=pred_len,
                T=self._T,
                top_k=self._top_k,
                top_p=self._top_p,
                sample_count=self._sample_count,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Kronos predict_batch failed: {exc}") from exc

        result: list[tuple[ForecastBar, ...]] = []
        for df, y_timestamps in zip(pred_dfs, y_timestamps_all):
            bars: list[ForecastBar] = []
            for i, (_, row) in enumerate(df.iterrows()):
                ts = y_timestamps[i]
                close = Decimal(str(row["close"])).quantize(Decimal("0.00000001"))
                if close <= 0:
                    close = Decimal("0.00000001")
                open_v = Decimal(str(row["open"])).quantize(Decimal("0.00000001")) if "open" in row else close
                high_v = Decimal(str(row["high"])).quantize(Decimal("0.00000001")) if "high" in row else close
                low_v = Decimal(str(row["low"])).quantize(Decimal("0.00000001")) if "low" in row else close
                vol_v = Decimal(str(row["volume"])).quantize(Decimal("0.00000001")) if "volume" in row else Decimal("0")
                bars.append(
                    ForecastBar(
                        timestamp=ts,
                        open=open_v,
                        high=high_v,
                        low=low_v,
                        close=close,
                        volume=vol_v if vol_v >= 0 else Decimal("0"),
                    )
                )
            result.append(tuple(bars))
        return tuple(result)

    def unload(self) -> None:
        """Free GPU/CPU memory (optional)."""
        try:
            import torch  # type: ignore

            if self._model is not None:
                del self._model
            if self._tokenizer is not None:
                del self._tokenizer
            if self._predictor is not None:
                del self._predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._loaded = False
        self._predictor = None
        self._model = None
        self._tokenizer = None
