"""Binance BTC Up/Down 5m strategy foundation (research-only).

Canonical prediction-market path — NOT spot trading:

* :mod:`app.research.prediction_markets.up_down_5m.settlement` — binary
  Chainlink settlement rule (UP iff end > start, DOWN iff end < start,
  PUSH 50/50 on equal).
* :mod:`app.research.prediction_markets.up_down_5m.market` — 5m market
  data model + lifecycle (UPCOMING -> OPEN -> CLOSED -> SETTLED).
* :mod:`app.research.prediction_markets.up_down_5m.replay` — research-only
  replay evaluating decisions against actual binary settlement.
* :mod:`app.research.prediction_markets.up_down_5m.live_snapshot` — wire
  one live BTC 5m market from SAPI data onto the model, with honest
  Chainlink-vs-reference provenance (read-only, no trading).

Supersedes the spot-proxy probability approach in
``app.research.prediction_markets.backtest`` (impulse -> repricing lag),
which is ignored for Up/Down edge evaluation.

No live trading in this package (fail-closed).
"""

from app.research.prediction_markets.up_down_5m.live_snapshot import (
    FieldCoverage,
    LiveUpDown5mSnapshot,
    NoLiveMarketError,
    PriceProvenance,
    build_live_snapshot,
    fetch_one_btc_5m_snapshot,
    render_snapshot,
    required_field_coverage,
    settlement_preview,
)
from app.research.prediction_markets.up_down_5m.market import (
    FIVE_MIN_MS,
    MarketState,
    UpDown5mMarket,
)
from app.research.prediction_markets.up_down_5m.replay import (
    ReplayDecision,
    ReplayFill,
    ReplaySummary,
    UpDown5mReplay,
)
from app.research.prediction_markets.up_down_5m.settlement import (
    PAYOUT_PUSH,
    SettlementOutcome,
    Side,
    payout_per_share,
    settle_up_down_5m,
)

__all__ = [
    "FIVE_MIN_MS",
    "PAYOUT_PUSH",
    "FieldCoverage",
    "LiveUpDown5mSnapshot",
    "MarketState",
    "NoLiveMarketError",
    "PriceProvenance",
    "ReplayDecision",
    "ReplayFill",
    "ReplaySummary",
    "SettlementOutcome",
    "Side",
    "UpDown5mMarket",
    "UpDown5mReplay",
    "build_live_snapshot",
    "fetch_one_btc_5m_snapshot",
    "payout_per_share",
    "render_snapshot",
    "required_field_coverage",
    "settlement_preview",
    "settle_up_down_5m",
]
