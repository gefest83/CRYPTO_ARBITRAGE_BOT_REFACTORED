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
* :mod:`app.research.prediction_markets.up_down_5m.signal` — deterministic
  UP/DOWN/HOLD entry signal from pre-decision data only (drift, momentum/
  acceleration, trade flow, book imbalance, time gating) + enter/exit
  policy (immediate entry, fast exit on reversal, hold to settlement).
* :mod:`app.research.prediction_markets.up_down_5m.validation_dataset` —
  valid research dataset: exact windows, full-window BTC series (public
  klines backfill), in-window contract prices (venue timeseries), actual
  venue-resolved outcomes; evaluates the frozen signal (never modifies it).

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
from app.research.prediction_markets.up_down_5m.validation_dataset import (
    BINANCE_KLINES_URL,
    Kline,
    ValidatedMarket,
    ValidationResult,
    build_signal_features,
    contract_price_at,
    fetch_klines,
    parse_contract_series,
    parse_klines,
    parse_venue_outcome,
    parse_window_from_slug,
    run_validation,
)
from app.research.prediction_markets.up_down_5m.signal import (
    Action,
    Position,
    Signal,
    SignalConfig,
    SignalFeatures,
    SignalResult,
    compute_features,
    ensure_research_only as ensure_signal_research_only,
    evaluate,
    exit_to_skip,
    manage,
)

__all__ = [
    "BINANCE_KLINES_URL",
    "BTC_USD_CEX_V3_FEED_ID",
    "CHAINLINK_DATAENGINE_MAINNET",
    "FIVE_MIN_MS",
    "PAYOUT_PUSH",
    "REPORT_DECIMALS",
    "VENUE_CHAINLINK",
    "Action",
    "ChainlinkFeedClient",
    "ChainlinkFeedError",
    "ChainlinkReport",
    "FieldCoverage",
    "FiveMinuteCandle",
    "Kline",
    "LiveUpDown5mSnapshot",
    "MarketState",
    "NoLiveMarketError",
    "Position",
    "PriceProvenance",
    "ReplayDecision",
    "ReplayFill",
    "ReplaySummary",
    "ResolutionInputs",
    "SettlementOutcome",
    "Side",
    "Signal",
    "SignalConfig",
    "SignalFeatures",
    "SignalResult",
    "UpDown5mMarket",
    "UpDown5mReplay",
    "ValidatedMarket",
    "ValidationResult",
    "attach_chainlink_resolution",
    "build_5m_candles",
    "build_auth_headers",
    "build_live_snapshot",
    "compute_features",
    "contract_price_at",
    "decode_v3_report",
    "ensure_signal_research_only",
    "evaluate",
    "exit_to_skip",
    "fetch_klines",
    "fetch_one_btc_5m_snapshot",
    "final_close",
    "manage",
    "market_start_price",
    "payout_per_share",
    "parse_contract_series",
    "parse_klines",
    "parse_venue_outcome",
    "parse_window_from_slug",
    "render_snapshot",
    "required_field_coverage",
    "resolve_chainlink_credentials",
    "resolve_market",
    "run_validation",
    "settle_from_chainlink",
    "settlement_preview",
    "settle_up_down_5m",
    "tob_mid",
]
