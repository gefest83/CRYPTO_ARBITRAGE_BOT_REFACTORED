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
* :mod:`app.research.prediction_markets.up_down_5m.chainlink_feed` —
  official Chainlink Data Streams resolution feed (auth client, v3 decode,
  start price / final 5m close math, CHAINLINK attach; read-only).

Supersedes the spot-proxy probability approach in
``app.research.prediction_markets.backtest`` (impulse -> repricing lag),
which is ignored for Up/Down edge evaluation. Binance spot is never a
settlement substitute: only ``CHAINLINK`` provenance may settle.

No live trading in this package (fail-closed).
"""

from app.research.prediction_markets.up_down_5m.chainlink_feed import (
    BTC_USD_CEX_V3_FEED_ID,
    CHAINLINK_DATAENGINE_MAINNET,
    REPORT_DECIMALS,
    ChainlinkFeedClient,
    ChainlinkFeedError,
    ChainlinkReport,
    FiveMinuteCandle,
    ResolutionInputs,
    attach_chainlink_resolution,
    build_5m_candles,
    build_auth_headers,
    decode_v3_report,
    final_close,
    market_start_price,
    resolve_chainlink_credentials,
    resolve_market,
    tob_mid,
)
from app.research.prediction_markets.up_down_5m.live_snapshot import (
    VENUE_CHAINLINK,
    FieldCoverage,
    LiveUpDown5mSnapshot,
    NoLiveMarketError,
    PriceProvenance,
    build_live_snapshot,
    fetch_one_btc_5m_snapshot,
    render_snapshot,
    required_field_coverage,
    settle_from_chainlink,
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
    "BTC_USD_CEX_V3_FEED_ID",
    "CHAINLINK_DATAENGINE_MAINNET",
    "FIVE_MIN_MS",
    "PAYOUT_PUSH",
    "REPORT_DECIMALS",
    "VENUE_CHAINLINK",
    "ChainlinkFeedClient",
    "ChainlinkFeedError",
    "ChainlinkReport",
    "FieldCoverage",
    "FiveMinuteCandle",
    "LiveUpDown5mSnapshot",
    "MarketState",
    "NoLiveMarketError",
    "PriceProvenance",
    "ReplayDecision",
    "ReplayFill",
    "ReplaySummary",
    "ResolutionInputs",
    "SettlementOutcome",
    "Side",
    "UpDown5mMarket",
    "UpDown5mReplay",
    "attach_chainlink_resolution",
    "build_5m_candles",
    "build_auth_headers",
    "build_live_snapshot",
    "decode_v3_report",
    "fetch_one_btc_5m_snapshot",
    "final_close",
    "market_start_price",
    "payout_per_share",
    "render_snapshot",
    "required_field_coverage",
    "resolve_chainlink_credentials",
    "resolve_market",
    "settle_from_chainlink",
    "settlement_preview",
    "settle_up_down_5m",
    "tob_mid",
]
