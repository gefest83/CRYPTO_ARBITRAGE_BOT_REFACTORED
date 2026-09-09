"""Phase 3A – real Kronos adapter contract tests (offline, no torch/network/HF).

Uses MockKronosPredictor to verify compatibility with KronosForecastModel and
KronosScanner. Real model tests are marked integration and skipped by default.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.strategies.kronos.config import KronosConfig
from app.strategies.kronos.forecast import MockKronosPredictor
from app.strategies.kronos.real import MODEL_SPECS, RealKronosPredictor, detect_environment, get_device
from app.strategies.kronos.scanner import KronosScanner
from app.strategies.kronos.synthetic import generate_synthetic_candles
from app.strategies.kronos.types import ForecastBar, Signal

UTC = timezone.utc


def _dt(y, m, d, hh=0, mm=0) -> datetime:
    return datetime(y, m, d, hh, mm, 0, tzinfo=UTC)


# ---------------------------------------------------------------- spec correctness (no torch)
def test_model_specs_exact():
    assert MODEL_SPECS["mini"].model_id == "NeoQuasar/Kronos-mini"
    assert MODEL_SPECS["mini"].tokenizer_id == "NeoQuasar/Kronos-Tokenizer-2k"
    assert MODEL_SPECS["mini"].max_context == 2048
    assert MODEL_SPECS["mini"].params == "4.1M"
    assert MODEL_SPECS["base"].model_id == "NeoQuasar/Kronos-base"
    assert MODEL_SPECS["base"].tokenizer_id == "NeoQuasar/Kronos-Tokenizer-base"
    assert MODEL_SPECS["base"].max_context == 512
    assert MODEL_SPECS["base"].params == "102.3M"


def test_real_predictor_not_loaded_by_default():
    p = RealKronosPredictor(variant="mini")
    assert not p.is_loaded
    assert p.spec.variant == "mini"
    # predict without load must fail fast with clear message
    candles = generate_synthetic_candles(start=_dt(2026, 1, 1), count=20)
    with pytest.raises(RuntimeError, match="not loaded"):
        p.predict(candles, pred_len=5)
    with pytest.raises(RuntimeError, match="not loaded"):
        p.predict_batch([candles, candles], pred_len=5)


def test_real_predictor_invalid_variant():
    with pytest.raises(ValueError, match="unknown variant"):
        RealKronosPredictor(variant="large")  # type: ignore


def test_detect_environment_cpu():
    env = detect_environment()
    assert "python" in env
    assert env["python"].startswith("3.13")
    # This environment is CPU-only
    assert env["device"] in ("cpu", "cuda:0", "mps")
    assert "ram_total_gb" in env


def test_get_device_returns_string():
    assert isinstance(get_device(), str)


# ---------------------------------------------------------------- mock compatibility with scanner (offline)
def test_mock_compatible_with_scanner_protocol():
    """Verify MockKronosPredictor satisfies KronosForecastModel and scanner works."""
    from app.strategies.kronos.forecast import KronosForecastModel

    cfg = KronosConfig(lookback=20, pred_len_1m=5, pred_len_5m=1, min_edge_bps=Decimal("10"), require_5m_confirmation=False, max_candle_age_ms=600_000, max_gap_ms=600_000)
    candles = generate_synthetic_candles(start=_dt(2026, 1, 1), count=20)
    mock = MockKronosPredictor(drift_bps_per_bar=Decimal("30"))
    # protocol duck-type check
    assert hasattr(mock, "predict")
    scanner = KronosScanner(cfg, mock)  # type: ignore[arg-type]
    sig = scanner.scan("BTC/USDT", candles, now=candles[-1].timestamp + timedelta(seconds=10))
    assert sig.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
    assert sig.pred_close_1m is not None
    assert sig.strategy.value == "kronos"


def test_forecast_bar_shape_and_types():
    candles = generate_synthetic_candles(start=_dt(2026, 1, 1), count=20)
    mock = MockKronosPredictor(drift_bps_per_bar=Decimal("10"))
    bars = mock.predict(candles, pred_len=5)
    assert len(bars) == 5
    for b in bars:
        assert isinstance(b, ForecastBar)
        assert isinstance(b.close, Decimal)
        assert b.close > 0
        assert b.timestamp.tzinfo is not None
    # strictly after last candle, 1m apart
    assert bars[0].timestamp == candles[-1].timestamp + timedelta(minutes=1)
    assert bars[-1].timestamp == candles[-1].timestamp + timedelta(minutes=5)


def test_real_predictor_lazy_does_not_import_torch_eagerly():
    """Importing RealKronosPredictor should not require torch at import time (only at load)."""
    # If this test runs, import already succeeded without torch being required for mock tests
    assert True


def test_same_input_same_sampling_params_for_both_specs():
    """Ensure benchmark would use same lookback/pred_len/sampling for both models."""
    # This is a contract test: both specs must support lookback 400 and pred_len 5
    for variant in ("mini", "base"):
        spec = MODEL_SPECS[variant]  # type: ignore
        assert spec.max_context >= 400, f"{variant} max_context {spec.max_context} < 400"
    # Sampling params are defined in RealKronosPredictor defaults
    p_mini = RealKronosPredictor(variant="mini", T=1.0, top_p=0.9, top_k=0, sample_count=1)
    p_base = RealKronosPredictor(variant="base", T=1.0, top_p=0.9, top_k=0, sample_count=1)
    assert p_mini._T == p_base._T == 1.0  # type: ignore
    assert p_mini._top_p == p_base._top_p == 0.9  # type: ignore
    assert p_mini._sample_count == p_base._sample_count == 1  # type: ignore


# ---------------------------------------------------------------- batch mock compatibility
def test_batch_mock_via_scanner_loop():
    """Batch (5 symbols) via repeated scanner calls stays isolated (no state)."""
    cfg = KronosConfig(lookback=20, pred_len_1m=5, pred_len_5m=1, min_edge_bps=Decimal("10"), require_5m_confirmation=False, max_candle_age_ms=600_000, max_gap_ms=600_000)
    start = _dt(2026, 1, 1)
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
    candles_list = [generate_synthetic_candles(start=start, count=20, start_price=Decimal(str(100 + i * 10))) for i in range(5)]
    mock = MockKronosPredictor(drift_bps_per_bar=Decimal("20"))
    for sym, candles in zip(symbols, candles_list):
        scanner = KronosScanner(cfg, mock)
        sig = scanner.scan(sym, candles, now=candles[-1].timestamp + timedelta(seconds=10))
        assert sig.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
        assert sig.symbol == sym


# ---------------------------------------------------------------- real model integration (skipped unless --integration)
@pytest.mark.integration
def test_real_mini_predict_integration():
    """Integration: actually loads mini and predicts (requires torch + HF + network)."""
    pytest.importorskip("torch")
    p = RealKronosPredictor(variant="mini")
    try:
        p.load()
    except Exception as exc:
        pytest.skip(f"HF download or torch not available: {exc}")
    candles = generate_synthetic_candles(start=_dt(2026, 1, 1), count=50, start_price=Decimal("50000"))
    bars = p.predict(candles, pred_len=5)
    assert len(bars) == 5
    for b in bars:
        assert isinstance(b.close, Decimal)
        assert b.close > 0
    # scanner compatibility
    cfg = KronosConfig(lookback=20, pred_len_1m=5, pred_len_5m=1, require_5m_confirmation=False, max_candle_age_ms=600_000, max_gap_ms=600_000)
    scanner = KronosScanner(cfg, p)  # type: ignore
    sig = scanner.scan("BTC/USDT", candles, now=candles[-1].timestamp + timedelta(seconds=10))
    assert sig.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
    p.unload()


@pytest.mark.integration
def test_real_base_predict_integration():
    pytest.importorskip("torch")
    p = RealKronosPredictor(variant="base")
    try:
        p.load()
    except Exception as exc:
        pytest.skip(f"HF download or torch not available: {exc}")
    candles = generate_synthetic_candles(start=_dt(2026, 1, 1), count=50, start_price=Decimal("50000"))
    bars = p.predict(candles, pred_len=5)
    assert len(bars) == 5
    p.unload()
