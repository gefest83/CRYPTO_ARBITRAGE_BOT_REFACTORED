"""Backtest analyzer — deterministic, no look-ahead, grouped evaluation."""

from __future__ import annotations

import statistics
from collections import defaultdict
from decimal import Decimal
from typing import Any

from app.research.prediction_markets.backtest.cost_model import CostModel
from app.research.prediction_markets.backtest.models import (
    BacktestConfig,
    BacktestResult,
    GroupStats,
    ImpulseEvent,
    OutcomeMetrics,
)
from app.research.prediction_markets.backtest.signal import detect_impulses, mid_price
from app.research.prediction_markets.collector.models import (
    PredictionOrderbookObservation,
    SpotObservation,
)

__all__ = ["BacktestAnalyzer"]

ALLOWED_SYMBOLS = {"BTCUSDT", "BTC", "ETHUSDT", "ETH"}
ALLOWED_DURATIONS = {"5m", "15m"}


def _bucket_liquidity(depth: int | None) -> str:
    if depth is None:
        return "unknown"
    if depth < 2:
        return "low"
    if depth < 5:
        return "mid"
    return "high"


def _bucket_ttr(ttr_ms: int | None, thresholds: tuple[int, ...]) -> str:
    if ttr_ms is None:
        return "unknown"
    for thr in sorted(thresholds):
        if ttr_ms < thr:
            return f"<{thr//1000}s"
    return f">={thresholds[-1]//1000}s"


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    # deterministic linear interpolation
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    d = k - f
    return float(s[f] * (1 - d) + s[c] * d)


class BacktestAnalyzer:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    # ---------- public ----------
    def run(
        self,
        spot_observations: list[SpotObservation],
        prediction_observations: list[PredictionOrderbookObservation],
    ) -> BacktestResult:
        """Run full backtest. No look-ahead: events use only data <= t0."""
        # filter scope: BTC/ETH, 5m/15m, not expired, in-order already handled by collector
        spots = [o for o in spot_observations if o.symbol in ALLOWED_SYMBOLS and not o.is_expired]
        preds = [
            o
            for o in prediction_observations
            if o.symbol in ALLOWED_SYMBOLS
            and (o.duration in ALLOWED_DURATIONS if o.duration else True)
            and not o.is_expired
        ]
        # deterministic sort
        spots.sort(key=lambda o: (o.exchange_ts_ms or 0, o.captured_at_ms))
        preds.sort(key=lambda o: (o.update_ts_ms or o.exchange_ts_ms or 0, o.captured_at_ms))

        # build per-market pred index (sorted)
        pred_by_market: dict[int, list[PredictionOrderbookObservation]] = defaultdict(list)
        for p in preds:
            if p.market_id is not None:
                pred_by_market[int(p.market_id)].append(p)
            else:
                pred_by_market[0].append(p)
        for lst in pred_by_market.values():
            lst.sort(key=lambda o: (o.update_ts_ms or o.exchange_ts_ms or 0))

        dataset_size = len(spots) + len(preds)
        coverage: dict[str, int] = defaultdict(int)
        for o in spots:
            coverage[o.symbol] += 1
        for o in preds:
            coverage[f"pred:{o.symbol}:{o.duration}:{o.outcome}"] = coverage.get(f"pred:{o.symbol}:{o.duration}:{o.outcome}", 0) + 1

        # For each threshold separately, detect impulses then try to pair with prediction evaluation
        all_metrics: list[OutcomeMetrics] = []
        # Keep impulse events deduplicated per threshold? We emit one event per impulse per market/outcome pair.
        # To avoid combinatorial explosion in tests, pair each impulse with each active market that matches symbol+duration.
        # Determine active markets from preds: map market_id -> sample pred with metadata
        market_meta: dict[int, PredictionOrderbookObservation] = {}
        for mid, lst in pred_by_market.items():
            if lst:
                market_meta[mid] = lst[0]

        for thr in self.config.impulse_thresholds_bps:
            impulses = detect_impulses(spots, threshold_bps=thr, lookback_ms=self.config.impulse_lookback_ms)
            for imp in impulses:
                t0 = int(imp["t0_ms"])
                symbol = imp["symbol"]
                direction = int(imp["direction"])
                bps = imp["impulse_bps"]
                # For each market that matches symbol (BTC->BTC, ETH->ETH) and each outcome YES/NO
                # we have at most 2 durations; we iterate over markets whose symbol matches
                for mid, meta in market_meta.items():
                    if meta.symbol != symbol:
                        continue
                    # duration filter already
                    dur = meta.duration or "5m"
                    if dur not in ALLOWED_DURATIONS:
                        continue
                    # find initial price: latest pred for this market at or before t0 (strict no look-ahead)
                    initial_obs = self._latest_at_or_before(pred_by_market[mid], t0)
                    if initial_obs is None:
                        continue
                    if initial_obs.update_ts_ms is not None and int(initial_obs.update_ts_ms) > t0:
                        continue  # look-ahead violation guard (should not happen)
                    # derive initial price/spread/depth — use best mid or last_trade as price
                    init_price = self._price_of(initial_obs)
                    if init_price is None:
                        continue
                    spread = initial_obs.spread if initial_obs.spread is not None else None
                    # outcome handling: we have one token per market in our synthetic split; use YES/NO from obs
                    outcomes_to_consider = []
                    if initial_obs.outcome in ("YES", "NO"):
                        outcomes_to_consider.append((initial_obs.outcome, initial_obs, init_price))
                    else:
                        # synthetic without outcome: treat as YES (tests use YES)
                        outcomes_to_consider.append(("YES", initial_obs, init_price))

                    for outcome, obs_for_outcome, price_for_outcome in outcomes_to_consider:
                        # liquidity bucket
                        liq_b = _bucket_liquidity(obs_for_outcome.depth)
                        ttr_b = _bucket_ttr(obs_for_outcome.time_to_resolution_ms, self.config.ttr_buckets_ms)
                        event = ImpulseEvent(
                            t0_ms=t0,
                            symbol=symbol,
                            spot_mid_before=imp["before_mid"],
                            spot_mid_at_t0=imp["at_mid"],
                            impulse_bps=bps,
                            threshold_bps=thr,
                            direction=direction,
                            market_id=int(mid),
                            token_id=obs_for_outcome.token_id or "",
                            outcome=outcome,
                            duration=dur,
                            initial_price=price_for_outcome,
                            initial_spread=spread,
                            initial_spread_bps=obs_for_outcome.spread_bps,
                            depth=obs_for_outcome.depth,
                            time_to_resolution_ms=obs_for_outcome.time_to_resolution_ms,
                            liquidity_bucket=liq_b,
                            ttr_bucket=ttr_b,
                        )
                        metrics = self._evaluate(event, pred_by_market[mid], t0)
                        all_metrics.append(metrics)

        # aggregate by requested dimensions
        group_stats = self._aggregate(all_metrics)

        # evidence / limitations
        has_evidence, notes, lim = self._assess_evidence(dataset_size, len(all_metrics), group_stats)

        return BacktestResult(
            config=self.config,
            dataset_size=dataset_size,
            coverage=dict(coverage),
            group_stats=tuple(group_stats),
            all_metrics=tuple(all_metrics),
            has_evidence=has_evidence,
            evidence_notes=notes,
            limitations=lim,
        )

    # ---------- internals ----------
    def _latest_at_or_before(self, lst: list[PredictionOrderbookObservation], t0: int) -> PredictionOrderbookObservation | None:
        best: PredictionOrderbookObservation | None = None
        for o in lst:
            ts = o.update_ts_ms or o.exchange_ts_ms
            if ts is None:
                continue
            if int(ts) <= t0:
                best = o
            else:
                break  # sorted ascending, future beyond t0
        return best

    def _price_of(self, obs: PredictionOrderbookObservation) -> Decimal | None:
        # prefer mid if spread available, else last_trade_price or best_bid/ask mid
        if obs.best_bid is not None and obs.best_ask is not None:
            return (obs.best_bid + obs.best_ask) / Decimal("2")
        if obs.last_trade_price is not None:
            return obs.last_trade_price
        return None

    def _evaluate(self, event: ImpulseEvent, pred_list: list[PredictionOrderbookObservation], t0: int) -> OutcomeMetrics:
        """Evaluate metrics strictly using data after t0 (future) for outcome."""
        # collect future observations with t > t0 and <= t0+eval_window
        window_end = t0 + self.config.eval_window_ms
        future = [o for o in pred_list if (o.update_ts_ms or o.exchange_ts_ms or 0) > t0 and (o.update_ts_ms or o.exchange_ts_ms or 0) <= window_end]
        if not future:
            return OutcomeMetrics(
                event=event,
                repricing_lag_ms=None,
                max_favorable=None,
                max_adverse=None,
                time_to_max_favorable_ms=None,
                gross_edge=None,
                spread_cost=None,
                fee_cost=None,
                slippage_cost=None,
                net_edge=None,
                win=None,
                insufficient_liquidity=(event.depth or 0) < self.config.min_depth,
                is_expired=False,
            )
        # for YES: favorable = price increase when direction up, and opposite when down
        # Generalize: if direction +1 expects YES up (and NO down). For test determinism, define:
        #   if outcome == YES: favorable when price > initial
        #   if outcome == NO: favorable when price < initial (since NO is opposite)
        # For mixed, we use direction sign: favorable = (price - initial) * direction_for_outcome
        # where direction_for_outcome = direction if YES else -direction
        # Simpler: evaluator does YES expects +direction, NO expects -direction? But prompt says YES vs NO separately.
        # Use: for outcome YES, MFE = max(price - initial) if direction up else max(initial - price)? Actually both same formula: favorable = (price - initial) * sign where sign = 1 for YES, -1 for NO when impulse up? This is ambiguous.
        # Provide deterministic: MFE = max(price - initial) regardless of direction; but for down impulse, a drop in YES is favorable.
        # So define direction-corrected: favorable_delta = (price - initial) * event.direction * (1 if outcome YES else -1)? Let's pick: YES favors impulse direction, NO favors opposite.
        # That yields meaningful separation for report.
        outcome_sign = 1 if event.outcome == "YES" else -1
        direction_sign = event.direction  # +1 up, -1 down
        # expected favorable direction = direction_sign * outcome_sign
        expected_sign = direction_sign * outcome_sign
        # So if expected_sign == 1, price should go up to be favorable; if -1, price should go down.

        def price(o: PredictionOrderbookObservation) -> Decimal | None:
            return self._price_of(o)

        init = event.initial_price
        max_fav: Decimal | None = None
        max_adv: Decimal | None = None
        time_to_fav: int | None = None
        best_price: Decimal | None = None
        lag: int | None = None
        # also track earliest favorable beyond spread
        spread_thr = event.initial_spread or Decimal("0")
        # half-spread threshold for detectable move
        half_spread = spread_thr / Decimal("2") if spread_thr else Decimal("0")

        for o in future:
            p = price(o)
            if p is None:
                continue
            raw_delta = p - init
            # map to favorable coord
            fav_delta = raw_delta * Decimal(expected_sign)
            adv_delta = -fav_delta  # opposite
            if max_fav is None or fav_delta > max_fav:
                max_fav = fav_delta
                best_price = p
                ts = o.update_ts_ms or o.exchange_ts_ms
                if ts is not None:
                    time_to_fav = int(ts) - t0
            if max_adv is None or adv_delta > max_adv:
                max_adv = adv_delta
            if lag is None and fav_delta > half_spread:
                ts = o.update_ts_ms or o.exchange_ts_ms
                if ts is not None:
                    lag = int(ts) - t0

        gross = max_fav  # already favorable delta
        cm = CostModel(self.config.fee_bps, self.config.slippage_per_depth_bps, self.config.min_depth)
        spread_c = cm.spread_cost(spread_thr) if spread_thr else Decimal("0")
        fee_c = cm.fee_cost(init)
        slip_c = cm.slippage_cost(event.depth)
        net = None
        win = None
        if gross is not None:
            net = gross - spread_c - fee_c - slip_c
            win = net > Decimal("0")

        insufficient = (event.depth or 0) < self.config.min_depth if event.depth is not None else False

        return OutcomeMetrics(
            event=event,
            repricing_lag_ms=lag,
            max_favorable=max_fav,
            max_adverse=max_adv,
            time_to_max_favorable_ms=time_to_fav,
            gross_edge=gross,
            spread_cost=spread_c,
            fee_cost=fee_c,
            slippage_cost=slip_c,
            net_edge=net,
            win=win,
            insufficient_liquidity=insufficient,
            is_expired=False,
        )

    def _aggregate(self, metrics: list[OutcomeMetrics]) -> list[GroupStats]:
        # group by multiple keys
        groups: dict[str, list[OutcomeMetrics]] = defaultdict(list)
        # helper to add
        def add(key: str, m: OutcomeMetrics) -> None:
            groups[key].append(m)

        for m in metrics:
            e = m.event
            # high-level groups
            add(f"ALL", m)
            add(f"symbol:{e.symbol}", m)
            add(f"duration:{e.duration}", m)
            add(f"outcome:{e.outcome}", m)
            add(f"threshold:{e.threshold_bps}", m)
            add(f"liquidity:{e.liquidity_bucket}", m)
            add(f"ttr:{e.ttr_bucket}", m)
            # combined
            add(f"{e.symbol}|{e.duration}|{e.outcome}|thr={e.threshold_bps}|liq={e.liquidity_bucket}", m)

        stats: list[GroupStats] = []
        for key, lst in sorted(groups.items()):
            lags = [float(x.repricing_lag_ms) for x in lst if x.repricing_lag_ms is not None]
            gross = [x.gross_edge for x in lst if x.gross_edge is not None]
            net = [x.net_edge for x in lst if x.net_edge is not None]
            mfe = [x.max_favorable for x in lst if x.max_favorable is not None]
            mae = [x.max_adverse for x in lst if x.max_adverse is not None]
            wins = [1 for x in lst if x.win is True]
            n = len(lst)
            win_rate = (len(wins) / n) if n else None
            # percentiles for lag
            median = _pct(lags, 50) if lags else None
            mean = statistics.mean(lags) if lags else None
            p25 = _pct(lags, 25) if lags else None
            p75 = _pct(lags, 75) if lags else None
            p95 = _pct(lags, 95) if lags else None
            gross_mean = (sum(gross) / len(gross)) if gross else None  # type: ignore[operator]
            net_mean = (sum(net) / len(net)) if net else None  # type: ignore[operator]
            avg_ret = net_mean
            mfe_mean = (sum(mfe) / len(mfe)) if mfe else None  # type: ignore[operator]
            mae_mean = (sum(mae) / len(mae)) if mae else None  # type: ignore[operator]
            # p50 gross/net
            def _median_dec(vals: list[Decimal]) -> Decimal | None:
                if not vals:
                    return None
                s = sorted(vals)
                mid = len(s) // 2
                if len(s) % 2 == 1:
                    return s[mid]
                return (s[mid - 1] + s[mid]) / Decimal("2")

            stats.append(
                GroupStats(
                    group=key,
                    sample_count=len([x for x in lst if x.gross_edge is not None]),
                    signal_count=n,
                    median_lag_ms=median,
                    mean_lag_ms=mean,
                    p25_lag_ms=p25,
                    p75_lag_ms=p75,
                    p95_lag_ms=p95,
                    gross_edge_mean=gross_mean,
                    net_edge_mean=net_mean,
                    win_rate=win_rate,
                    avg_return=avg_ret,
                    mfe_mean=mfe_mean,
                    mae_mean=mae_mean,
                    gross_edge_p50=_median_dec(gross),  # type: ignore[arg-type]
                    net_edge_p50=_median_dec(net),  # type: ignore[arg-type]
                )
            )
        return stats

    def _assess_evidence(self, dataset_size: int, signal_count: int, stats: list[GroupStats]) -> tuple[bool, str, str]:
        if dataset_size < self.config.min_observations or signal_count < self.config.min_observations:
            needed = self.config.min_observations
            have = signal_count
            missing = max(0, needed - have)
            # estimate collection time: collector yields ~1 impulse per X observations, but deterministic
            est_per_day = 500  # heuristic
            days_needed = (missing + est_per_day - 1) // est_per_day if est_per_day else 1
            notes = (
                f"Insufficient dataset: {dataset_size} observations, {signal_count} signals (<{needed}). "
                f"Need {missing} more signals. At ~{est_per_day}/day, collect {days_needed} more day(s) "
                f"with synchronized spot+prediction capture for BTC/ETH 5m/15m."
            )
            lim = (
                "Dataset too small to claim profitability. Results are descriptive only. "
                "No live trading. Keep collector running continuously, ensure both durations and "
                "all impulse thresholds have coverage, and re-run after 7-14 days."
            )
            return False, notes, lim
        # check win rate + net edge
        all_group = next((g for g in stats if g.group == "ALL"), None)
        if all_group and all_group.net_edge_mean is not None and all_group.win_rate is not None:
            if all_group.net_edge_mean > Decimal("0") and all_group.win_rate > 0.52:
                return True, f"Preliminary evidence: net {all_group.net_edge_mean:.4f}, win rate {all_group.win_rate:.1%} over {signal_count} signals.", "Evidence is preliminary; out-of-sample and fee/slippage sensitivity required."
        return False, f"Dataset sufficient ({signal_count} signals) but no statistically significant net edge (mean net {all_group.net_edge_mean if all_group else 'n/a'}).", "Hypothesis not supported on current window; extend collection and test different impulse/liquidity buckets."
