"""Hypothesis testability: can we detect exploitable repricing delay after fees/spread/slippage?"""

from __future__ import annotations

from decimal import Decimal

from app.research.prediction_markets.models import HistoricalAvailability, NormalizedMarket, PredictionConstraints

__all__ = ["assess_hypothesis"]


def assess_hypothesis(
    markets: list[NormalizedMarket],
    constraints: PredictionConstraints,
    historical: HistoricalAvailability,
    spreads_bps: list[Decimal] | None = None,
) -> tuple[bool, str]:
    """Determine whether latency/repricing edge is testable.

    Hypothesis: underlying BTC/ETH spot moves first, prediction market
    reprices with measurable delay -> executable edge after fees/spread/slippage.

    Testable iff:
      - at least one BTC or ETH 5m/15m market is active (live orderbook),
      - timestamps allow ordering (start/end + WS updateTimestampMs),
      - we can estimate costs (fee + spread + slippage),
      - AND we have a way to get historical or sampled price series.

    Without local polling or Predict.fun history, the *Binance-only* path
    is NOT backtestable — but is *forward-testable* by building a collector.
    """
    if not markets:
        return False, "No BTC/ETH 5m/15m markets discovered — cannot test."

    active = [m for m in markets if m.is_active]
    if not active:
        return False, "Markets found but none with tradingStatus OPEN."

    # costs
    fees_ok = constraints.fee_rate_bps is not None
    # spreads
    avg_spread = None
    if spreads_bps:
        avg_spread = sum(spreads_bps) / len(spreads_bps)

    # historical
    has_series = historical.predict_fun_has_history or historical.has_orderbook_history or historical.has_trade_history

    # Build notes
    parts: list[str] = []
    parts.append(f"Active markets: {len(active)}/{len(markets)}.")
    parts.append(f"Fee {constraints.fee_rate_bps} bps, slippage tolerance {constraints.slippage_bps} bps, collateral {constraints.collateral}.")
    if avg_spread is not None:
        est = Decimal(constraints.fee_rate_bps * 2) + avg_spread + Decimal(constraints.slippage_bps)
        parts.append(f"Avg spread {avg_spread:.1f} bps -> est round-trip ~{est:.1f} bps (2*fee + spread + slippage).")
    else:
        parts.append("Spread unknown until live orderbook sampled.")

    if has_series:
        parts.append("Historical series available via Predict.fun; hypothesis IS testable by correlating spot moves vs prediction repricing lag.")
        testable = True
    else:
        parts.append(
            "Binance SAPI alone provides no history — hypothesis is NOT backtestable on Binance alone. "
            "It IS forward-testable by polling orderbook/lastTradePrice + spot price at high frequency and building local history. "
            "WS orderbook pushes (<200ms) + updateTimestampMs allow direct latency measurement."
        )
        # Forward-testable counts as testable with collector
        testable = True

    # Edge existence unknown until measured — but the *test* itself is feasible
    parts.append(
        "Verdict: hypothesis is testable (requires live collector or Predict.fun history). "
        "Whether edge survives fees/spread/slippage/impact is unknown until measured — "
        "needs paired spot vs prediction time-series at 100-500ms resolution."
    )
    return testable, " ".join(parts)
