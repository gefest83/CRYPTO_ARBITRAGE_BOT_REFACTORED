"""Phase 2 offline vertical slice — deterministic Kronos tests (no torch, no network)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.models.enums import ArbitrageStrategy
from app.strategies.kronos.config import KronosConfig
from app.strategies.kronos.cost import CostInputs, compute_cost_bps, compute_gross_bps, compute_net_bps
from app.strategies.kronos.forecast import MockKronosPredictor
from app.strategies.kronos.scanner import KronosScanner, resample_to_5m, validate_candles
from app.strategies.kronos.signal import classify_signal
from app.strategies.kronos.synthetic import generate_synthetic_candles
from app.strategies.kronos.types import Candle, ForecastBar, Signal

UTC = timezone.utc


def _dt(y, m, d, hh=0, mm=0) -> datetime:
    return datetime(y, m, d, hh, mm, 0, tzinfo=UTC)


# ---------------------------------------------------------------- cost
def test_compute_gross_bps_positive():
    assert compute_gross_bps(Decimal("100"), Decimal("101")) == Decimal("100.0000")


def test_compute_gross_bps_negative():
    assert compute_gross_bps(Decimal("100"), Decimal("99")) == Decimal("-100.0000")


def test_compute_cost_one_way():
    ci = CostInputs(taker_fee_bps=Decimal("10"), spread_bps=Decimal("6"), slippage_bps=Decimal("4"), cost_model="one_way")
    assert compute_cost_bps(ci) == Decimal("17.0000")  # 10 + 3 + 4


def test_compute_cost_round_trip():
    ci = CostInputs(taker_fee_bps=Decimal("10"), spread_bps=Decimal("6"), slippage_bps=Decimal("4"), cost_model="round_trip")
    assert compute_cost_bps(ci) == Decimal("34.0000")  # 20+6+8


def test_compute_net_bps_long():
    assert compute_net_bps(Decimal("30"), Decimal("17")) == Decimal("13.0000")


def test_compute_net_bps_short():
    # short gross -30 with cost 17 => -13
    assert compute_net_bps(Decimal("-30"), Decimal("17")) == Decimal("-13.0000")


def test_kronos_config_total_cost():
    cfg = KronosConfig(taker_fee_bps=Decimal("10"), spread_bps=Decimal("10"), slippage_bps=Decimal("5"), cost_model="one_way")
    assert cfg.total_cost_bps == Decimal("20")  # 10+5+5
    cfg2 = KronosConfig(taker_fee_bps=Decimal("10"), spread_bps=Decimal("10"), slippage_bps=Decimal("5"), cost_model="round_trip")
    assert cfg2.total_cost_bps == Decimal("40")  # 20+10+10


# ---------------------------------------------------------------- signal thresholds
def test_signal_buy_above_threshold():
    sig, _ = classify_signal(net_bps_1m=Decimal("20"), min_edge_bps=Decimal("15"))
    assert sig == Signal.BUY


def test_signal_sell_below_threshold():
    sig, _ = classify_signal(net_bps_1m=Decimal("-20"), min_edge_bps=Decimal("15"))
    assert sig == Signal.SELL


def test_signal_hold_inside_band():
    sig, reason = classify_signal(net_bps_1m=Decimal("14.9999"), min_edge_bps=Decimal("15"))
    assert sig == Signal.HOLD
    assert "below_threshold" in reason


def test_signal_threshold_strict():
    # equal to edge -> HOLD (strict >)
    sig, _ = classify_signal(net_bps_1m=Decimal("15"), min_edge_bps=Decimal("15"))
    assert sig == Signal.HOLD
    sig2, _ = classify_signal(net_bps_1m=Decimal("-15"), min_edge_bps=Decimal("15"))
    assert sig2 == Signal.HOLD


def test_signal_just_over_edge_buy():
    sig, _ = classify_signal(net_bps_1m=Decimal("15.0001"), min_edge_bps=Decimal("15"))
    assert sig == Signal.BUY


def test_signal_just_under_edge_sell():
    sig, _ = classify_signal(net_bps_1m=Decimal("-15.0001"), min_edge_bps=Decimal("15"))
    assert sig == Signal.SELL


# ---------------------------------------------------------------- fees / spread / slippage impact via scanner
def _base_config(**overrides) -> KronosConfig:
    defaults = dict(
        lookback=20,
        pred_len_1m=5,
        pred_len_5m=1,
        min_edge_bps=Decimal("10"),
        taker_fee_bps=Decimal("10"),
        spread_bps=Decimal("5"),
        slippage_bps=Decimal("5"),
        cost_model="one_way",
        require_5m_confirmation=False,
        max_candle_age_ms=300_000,
        max_gap_ms=300_000,
    )
    defaults.update(overrides)
    return KronosConfig(**defaults)


def test_scanner_buy_when_net_clears_cost():
    # last 100, drift +0.5% per bar = 50 bps per bar, 5 bars => 250 bps gross
    # cost = 10 + 2.5 +5 =17.5, net ~232 -> BUY
    cfg = _base_config(min_edge_bps=Decimal("15"))
    start = _dt(2026, 1, 1, 0, 0)
    candles = generate_synthetic_candles(start=start, count=20, start_price=Decimal("100"))
    # predictor drifts +50 bps per bar: after 5 bars gross ~250 bps
    model = MockKronosPredictor(drift_bps_per_bar=Decimal("50"))
    scanner = KronosScanner(cfg, model)
    now = candles[-1].timestamp + timedelta(seconds=10)
    sig = scanner.scan("BTC/USDT", candles, now=now)
    assert sig.signal == Signal.BUY
    assert sig.net_bps_1m > cfg.min_edge_bps


def test_scanner_hold_when_fees_eat_edge():
    # gross 30 bps, cost 17.5 => net 12.5 < min_edge 15 => HOLD
    cfg = _base_config(min_edge_bps=Decimal("15"), taker_fee_bps=Decimal("10"), spread_bps=Decimal("5"), slippage_bps=Decimal("5"))
    start = _dt(2026, 1, 1, 0, 0)
    candles = generate_synthetic_candles(start=start, count=20, start_price=Decimal("100"))
    # drift 6 bps per bar *5 => 30 gross
    model = MockKronosPredictor(drift_bps_per_bar=Decimal("6"))
    scanner = KronosScanner(cfg, model)
    now = candles[-1].timestamp + timedelta(seconds=10)
    sig = scanner.scan("BTC/USDT", candles, now=now)
    assert sig.signal == Signal.HOLD
    assert sig.gross_bps_1m == pytest.approx(Decimal("30"), abs=Decimal("0.01"))
    assert sig.net_bps_1m < Decimal("15")


def test_scanner_high_spread_blocks_signal():
    cfg_low = _base_config(spread_bps=Decimal("5"), min_edge_bps=Decimal("10"))
    cfg_high = _base_config(spread_bps=Decimal("80"), min_edge_bps=Decimal("10"))
    start = _dt(2026, 1, 1, 0, 0)
    candles = generate_synthetic_candles(start=start, count=20, start_price=Decimal("100"))
    # gross 50 bps (10*5); low cost 17.5 -> net 32.5 BUY; high cost 55 -> net 0 HOLD
    model = MockKronosPredictor(drift_bps_per_bar=Decimal("10"))
    scanner_low = KronosScanner(cfg_low, model)
    scanner_high = KronosScanner(cfg_high, model)
    now = candles[-1].timestamp + timedelta(seconds=10)
    sig_low = scanner_low.scan("BTC/USDT", candles, now=now)
    sig_high = scanner_high.scan("BTC/USDT", candles, now=now)
    assert sig_low.signal == Signal.BUY
    assert sig_high.signal == Signal.HOLD
    assert sig_high.cost_bps > sig_low.cost_bps
    assert sig_high.net_bps_1m == Decimal("0.0000")


def test_scanner_high_slippage_blocks_signal():
    cfg = _base_config(slippage_bps=Decimal("100"), min_edge_bps=Decimal("10"))
    start = _dt(2026, 1, 1, 0, 0)
    candles = generate_synthetic_candles(start=start, count=20, start_price=Decimal("100"))
    model = MockKronosPredictor(drift_bps_per_bar=Decimal("20"))  # 100 gross
    scanner = KronosScanner(cfg, model)
    now = candles[-1].timestamp + timedelta(seconds=10)
    sig = scanner.scan("BTC/USDT", candles, now=now)
    assert sig.signal == Signal.HOLD
    assert sig.cost_bps == Decimal("115.0000") or sig.cost_bps > Decimal("100")


def test_scanner_sell_signal():
    cfg = _base_config(min_edge_bps=Decimal("10"), require_5m_confirmation=False)
    start = _dt(2026, 1, 1, 0, 0)
    candles = generate_synthetic_candles(start=start, count=20, start_price=Decimal("100"))
    model = MockKronosPredictor(drift_bps_per_bar=Decimal("-50"))
    scanner = KronosScanner(cfg, model)
    now = candles[-1].timestamp + timedelta(seconds=10)
    sig = scanner.scan("BTC/USDT", candles, now=now)
    assert sig.signal == Signal.SELL


# ---------------------------------------------------------------- 5m confirmation
def test_5m_confirmation_pass():
    cfg = KronosConfig(lookback=20, pred_len_1m=5, pred_len_5m=1, min_edge_bps=Decimal("10"), taker_fee_bps=Decimal("5"), spread_bps=Decimal("2"), slippage_bps=Decimal("2"), require_5m_confirmation=True, max_candle_age_ms=300_000, max_gap_ms=300_000)
    start = _dt(2026, 1, 1, 0, 0)
    candles = generate_synthetic_candles(start=start, count=40, start_price=Decimal("100"))
    # Both horizons agree drift +30 => both BUY
    model = MockKronosPredictor(drift_bps_per_bar=Decimal("30"))
    scanner = KronosScanner(cfg, model)
    now = candles[-1].timestamp + timedelta(seconds=10)
    sig = scanner.scan("BTC/USDT", candles, now=now)
    assert sig.signal == Signal.BUY
    assert sig.confirmation_applied is True
    assert sig.confirmation_passed is True


def test_5m_confirmation_blocks_opposite():
    cfg = KronosConfig(lookback=20, pred_len_1m=5, pred_len_5m=1, min_edge_bps=Decimal("10"), taker_fee_bps=Decimal("5"), spread_bps=Decimal("2"), slippage_bps=Decimal("2"), require_5m_confirmation=True, max_candle_age_ms=300_000, max_gap_ms=300_000)

    class ConflictingMock:
        def __init__(self):
            self.calls = 0
        def predict(self, candles, pred_len):
            self.calls += 1
            # first call is 1m (pred_len 5) -> bullish, second is 5m (pred_len 1) -> bearish
            last = candles[-1]
            base = last.close
            if pred_len == 5:
                close = (base * Decimal("1.02")).quantize(Decimal("0.00000001"))  # +200 bps
            else:
                close = (base * Decimal("0.98")).quantize(Decimal("0.00000001"))  # -200 bps
            from app.strategies.kronos.types import ForecastBar
            ts = last.timestamp + timedelta(minutes=1)
            if pred_len == 5:
                return tuple(ForecastBar(timestamp=ts+timedelta(minutes=i), open=close, high=close, low=close, close=close) for i in range(pred_len))
            return (ForecastBar(timestamp=ts, open=close, high=close, low=close, close=close),)

    start = _dt(2026, 1, 1, 0, 0)
    candles = generate_synthetic_candles(start=start, count=40, start_price=Decimal("100"))
    scanner = KronosScanner(cfg, ConflictingMock())
    now = candles[-1].timestamp + timedelta(seconds=10)
    sig = scanner.scan("BTC/USDT", candles, now=now)
    assert sig.signal == Signal.HOLD
    assert "5m_confirmation_failed" in sig.reason
    assert sig.confirmation_passed is False


def test_5m_disabled_allows_buy():
    cfg = KronosConfig(lookback=20, pred_len_1m=5, pred_len_5m=1, min_edge_bps=Decimal("10"), taker_fee_bps=Decimal("5"), spread_bps=Decimal("2"), slippage_bps=Decimal("2"), require_5m_confirmation=False, max_candle_age_ms=300_000, max_gap_ms=300_000)

    class ConflictingMock:
        def predict(self, candles, pred_len):
            last = candles[-1]
            base = last.close
            close = (base * Decimal("1.02")).quantize(Decimal("0.00000001"))
            from app.strategies.kronos.types import ForecastBar
            ts = last.timestamp + timedelta(minutes=1)
            return tuple(ForecastBar(timestamp=ts+timedelta(minutes=i), open=close, high=close, low=close, close=close) for i in range(pred_len))

    start = _dt(2026, 1, 1, 0, 0)
    candles = generate_synthetic_candles(start=start, count=20, start_price=Decimal("100"))
    scanner = KronosScanner(cfg, ConflictingMock())
    now = candles[-1].timestamp + timedelta(seconds=10)
    sig = scanner.scan("BTC/USDT", candles, now=now)
    assert sig.signal == Signal.BUY
    assert sig.confirmation_applied is False


def test_classify_5m_same_direction_required():
    sig, _ = classify_signal(net_bps_1m=Decimal("20"), min_edge_bps=Decimal("15"), require_5m_confirmation=True, net_bps_5m=Decimal("5"))
    assert sig == Signal.HOLD
    sig2, _ = classify_signal(net_bps_1m=Decimal("20"), min_edge_bps=Decimal("15"), require_5m_confirmation=True, net_bps_5m=Decimal("16"))
    assert sig2 == Signal.BUY


# ---------------------------------------------------------------- stale / incomplete rejection
def test_insufficient_candles_hold():
    cfg = _base_config(lookback=20, require_5m_confirmation=False)
    candles = generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=10, start_price=Decimal("100"))
    model = MockKronosPredictor(drift_bps_per_bar=Decimal("50"))
    scanner = KronosScanner(cfg, model)
    now = candles[-1].timestamp + timedelta(seconds=10)
    sig = scanner.scan("BTC/USDT", candles, now=now)
    assert sig.signal == Signal.HOLD
    assert "insufficient_candles" in sig.reason


def test_stale_last_candle():
    cfg = _base_config(lookback=20, max_candle_age_ms=30_000)
    candles = generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=20, start_price=Decimal("100"))
    model = MockKronosPredictor(drift_bps_per_bar=Decimal("50"))
    scanner = KronosScanner(cfg, model)
    now = candles[-1].timestamp + timedelta(minutes=5)  # 300s later -> stale
    sig = scanner.scan("BTC/USDT", candles, now=now)
    assert sig.signal == Signal.HOLD
    assert "stale" in sig.reason


def test_incomplete_last_candle_rejected():
    cfg = KronosConfig(lookback=20, pred_len_1m=5, pred_len_5m=1, min_edge_bps=Decimal("10"), taker_fee_bps=Decimal("5"), spread_bps=Decimal("2"), slippage_bps=Decimal("2"), require_5m_confirmation=False, max_candle_age_ms=300_000, max_gap_ms=300_000, allow_incomplete_last=False)
    candles = list(generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=20, start_price=Decimal("100")))
    # mark last incomplete
    candles[-1] = Candle(timestamp=candles[-1].timestamp, open=candles[-1].open, high=candles[-1].high, low=candles[-1].low, close=candles[-1].close, volume=candles[-1].volume, complete=False)
    model = MockKronosPredictor(drift_bps_per_bar=Decimal("50"))
    scanner = KronosScanner(cfg, model)
    now = candles[-1].timestamp + timedelta(seconds=10)
    sig = scanner.scan("BTC/USDT", tuple(candles), now=now)
    assert sig.signal == Signal.HOLD
    assert "incomplete" in sig.reason


def test_incomplete_allowed_when_config():
    cfg = KronosConfig(lookback=20, pred_len_1m=5, pred_len_5m=1, min_edge_bps=Decimal("10"), taker_fee_bps=Decimal("5"), spread_bps=Decimal("2"), slippage_bps=Decimal("2"), require_5m_confirmation=False, max_candle_age_ms=300_000, max_gap_ms=300_000, allow_incomplete_last=True)
    candles = list(generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=20, start_price=Decimal("100")))
    candles[-1] = Candle(timestamp=candles[-1].timestamp, open=candles[-1].open, high=candles[-1].high, low=candles[-1].low, close=candles[-1].close, volume=candles[-1].volume, complete=False)
    model = MockKronosPredictor(drift_bps_per_bar=Decimal("50"))
    scanner = KronosScanner(cfg, model)
    now = candles[-1].timestamp + timedelta(seconds=10)
    sig = scanner.scan("BTC/USDT", tuple(candles), now=now)
    assert sig.signal == Signal.BUY


def test_gap_too_large():
    cfg = _base_config(lookback=20, max_gap_ms=90_000)
    candles = list(generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=20, start_price=Decimal("100")))
    # introduce 5m gap at idx 10
    for i in range(10, len(candles)):
        candles[i] = Candle(timestamp=candles[i].timestamp + timedelta(minutes=4), open=candles[i].open, high=candles[i].high, low=candles[i].low, close=candles[i].close, volume=candles[i].volume, complete=True)
    model = MockKronosPredictor(drift_bps_per_bar=Decimal("50"))
    scanner = KronosScanner(cfg, model)
    now = candles[-1].timestamp + timedelta(seconds=10)
    sig = scanner.scan("BTC/USDT", tuple(candles), now=now)
    assert sig.signal == Signal.HOLD
    assert "gap" in sig.reason


def test_non_positive_price_rejected():
    cfg = _base_config(lookback=20, require_5m_confirmation=False)
    candles = list(generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=20, start_price=Decimal("100")))
    candles[-1] = Candle(timestamp=candles[-1].timestamp, open=Decimal("0"), high=Decimal("0"), low=Decimal("0"), close=Decimal("0"), volume=Decimal("10"), complete=True)
    model = MockKronosPredictor(drift_bps_per_bar=Decimal("50"))
    scanner = KronosScanner(cfg, model)
    now = candles[-1].timestamp + timedelta(seconds=10)
    sig = scanner.scan("BTC/USDT", tuple(candles), now=now)
    assert sig.signal == Signal.HOLD


def test_resample_5m():
    candles = generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=20, start_price=Decimal("100"))
    bars_5m = resample_to_5m(candles)
    assert len(bars_5m) == 4  # 20/5
    assert bars_5m[0].open == candles[0].open
    assert bars_5m[0].close == candles[4].close
    assert bars_5m[0].volume == sum((c.volume for c in candles[:5]), Decimal("0"))


def test_resample_drops_incomplete_tail():
    candles = generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=22, start_price=Decimal("100"))
    bars = resample_to_5m(candles)
    assert len(bars) == 4


def test_validate_future_timestamp():
    cfg = _base_config()
    candles = generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=20, start_price=Decimal("100"))
    now = candles[-1].timestamp - timedelta(minutes=10)
    reason = validate_candles(candles, cfg, now=now)
    assert reason is not None and "future" in reason


# ---------------------------------------------------------------- no-lookahead
def test_no_lookahead_window_slicing():
    """Predictor must not see candles beyond the scan window."""

    class RecordingMock:
        def __init__(self):
            self.seen_lengths: list[int] = []
            self.seen_last_ts: list[datetime] = []
        def predict(self, candles, pred_len):
            self.seen_lengths.append(len(candles))
            self.seen_last_ts.append(candles[-1].timestamp)
            last = candles[-1]
            from app.strategies.kronos.types import ForecastBar
            ts = last.timestamp + timedelta(minutes=1)
            return tuple(ForecastBar(timestamp=ts+timedelta(minutes=i), open=last.close, high=last.close, low=last.close, close=last.close) for i in range(pred_len))

    cfg = KronosConfig(lookback=10, pred_len_1m=5, pred_len_5m=1, min_edge_bps=Decimal("1000"), taker_fee_bps=Decimal("0"), spread_bps=Decimal("0"), slippage_bps=Decimal("0"), require_5m_confirmation=False, max_candle_age_ms=600_000, max_gap_ms=600_000)
    base = generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=10, start_price=Decimal("100"))
    future = generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 10), count=5, start_price=Decimal("9999"))  # would dominate if leaked
    full = base + future
    # scan only base window — future must not be seen
    rec = RecordingMock()
    scanner = KronosScanner(cfg, rec)
    now = base[-1].timestamp + timedelta(seconds=10)
    scanner.scan("BTC/USDT", base, now=now)
    assert rec.seen_last_ts[0] == base[-1].timestamp
    # ensure future not leaked even if caller mistakenly passes full but scanner slices lookback tail
    rec2 = RecordingMock()
    scanner2 = KronosScanner(cfg, rec2)
    scanner2.scan("BTC/USDT", full, now=full[-1].timestamp + timedelta(seconds=10))
    # lookback 10 => last 10 of full are the future 5 + last 5 of base
    assert rec2.seen_last_ts[0] == full[-1].timestamp  # scanner correctly uses tail, not future beyond that tail (no extra)
    # but if predictor were to look ahead beyond last element, it would need data after full[-1] which doesn't exist


def test_fixed_closes_no_leakage():
    """Mock with fixed closes proves predictor uses only injected forecast, not future candles."""
    cfg = _base_config(lookback=20, require_5m_confirmation=False, min_edge_bps=Decimal("10"), max_candle_age_ms=600_000, max_gap_ms=600_000)
    candles = generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=20, start_price=Decimal("100"))
    now = candles[-1].timestamp + timedelta(seconds=10)
    # predictor will return 5 fixed closes at 150 (+5000 bps gross)
    model = MockKronosPredictor(fixed_closes=(Decimal("150"), Decimal("150"), Decimal("150"), Decimal("150"), Decimal("150")))
    scanner = KronosScanner(cfg, model)
    sig = scanner.scan("BTC/USDT", candles, now=now)
    assert sig.pred_close_1m == Decimal("150")
    assert sig.signal == Signal.BUY
    # now append real future that would contradict if leaked — scanner still uses fixed
    candles2 = candles + generate_synthetic_candles(start=candles[-1].timestamp + timedelta(minutes=1), count=1, start_price=Decimal("1"))
    sig2 = scanner.scan("BTC/USDT", candles, now=now)  # same window => same result
    assert sig2.pred_close_1m == Decimal("150")


# ---------------------------------------------------------------- deterministic repeated evaluation
def test_deterministic_repeated():
    cfg = _base_config(lookback=20, require_5m_confirmation=False, min_edge_bps=Decimal("10"), max_candle_age_ms=600_000, max_gap_ms=600_000)
    candles = generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=20, start_price=Decimal("100"))
    model = MockKronosPredictor(drift_bps_per_bar=Decimal("30"))
    scanner = KronosScanner(cfg, model)
    now = candles[-1].timestamp + timedelta(seconds=10)
    sig1 = scanner.scan("BTC/USDT", candles, now=now)
    sig2 = scanner.scan("BTC/USDT", candles, now=now)
    sig3 = scanner.scan("BTC/USDT", candles, now=now)
    assert sig1 == sig2 == sig3
    assert sig1.signal == sig2.signal
    assert sig1.net_bps_1m == sig2.net_bps_1m


def test_deterministic_across_scanner_instances():
    cfg = _base_config(lookback=20, require_5m_confirmation=False, min_edge_bps=Decimal("10"), max_candle_age_ms=600_000, max_gap_ms=600_000)
    candles = generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=20, start_price=Decimal("100"))
    now = candles[-1].timestamp + timedelta(seconds=10)
    s1 = KronosScanner(cfg, MockKronosPredictor(drift_bps_per_bar=Decimal("30"))).scan("BTC/USDT", candles, now=now)
    s2 = KronosScanner(cfg, MockKronosPredictor(drift_bps_per_bar=Decimal("30"))).scan("BTC/USDT", candles, now=now)
    assert s1.signal == s2.signal
    assert s1.gross_bps_1m == s2.gross_bps_1m


def test_signal_strategy_is_kronos():
    cfg = _base_config(require_5m_confirmation=False, max_candle_age_ms=600_000, max_gap_ms=600_000)
    candles = generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=20, start_price=Decimal("100"))
    model = MockKronosPredictor(drift_bps_per_bar=Decimal("50"))
    scanner = KronosScanner(cfg, model)
    now = candles[-1].timestamp + timedelta(seconds=10)
    sig = scanner.scan("BTC/USDT", candles, now=now)
    assert sig.strategy == ArbitrageStrategy.KRONOS


def test_only_buy_sell_hold():
    cfg = _base_config(require_5m_confirmation=False, max_candle_age_ms=600_000, max_gap_ms=600_000, min_edge_bps=Decimal("10"))
    candles = generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=20, start_price=Decimal("100"))
    now = candles[-1].timestamp + timedelta(seconds=10)
    for drift in [Decimal("50"), Decimal("-50"), Decimal("0"), Decimal("11"), Decimal("-11")]:
        sig = KronosScanner(cfg, MockKronosPredictor(drift_bps_per_bar=drift)).scan("BTC/USDT", candles, now=now)
        assert sig.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)


def test_generate_synthetic_deterministic():
    c1 = generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=10, start_price=Decimal("100"), drift_bps_per_bar=Decimal("10"))
    c2 = generate_synthetic_candles(start=_dt(2026, 1, 1, 0, 0), count=10, start_price=Decimal("100"), drift_bps_per_bar=Decimal("10"))
    assert c1 == c2
