"""Research-only Chainlink BTC/USDT resolution feed for BTC Up/Down 5m.

Official source: Chainlink Data Streams, BTC/USD CexPrice feed, report
schema v3 (Crypto Advanced), REST ``https://api.dataengine.chain.link``.
Verified 2026-09-13 against the live unauthenticated discovery catalog:
exactly one live BTC/USD crypto feed exists —

  feedId 0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8
  (``Bitcoin | CexPrice | schema V3 | status live``).

No separate Top-of-Book feed is listed in the public catalog, so the v3
``bid``/``ask`` fields (liquidity-weighted impact prices) are the closest
official ToB proxy; the mid ``(bid+ask)/2`` is the resolution input. This
substitution is recorded in code and snapshot notes, not hidden.

Access reality (probed, read-only):
  * ``GET /api/v1/discovery`` — public, no auth. Works.
  * ``GET /api/v1/reports[/latest|/page|/bulk]`` — requires HMAC-SHA256
    auth (``Authorization`` = API key, ``X-Authorization-Timestamp`` ms,
    ``X-Authorization-Signature-SHA256`` over
    ``METHOD FULL_PATH BODY_HASH API_KEY TIMESTAMP``). Without credentials
    the server returns 400 missing ``UserId/Timestamp/HmacSignature``.
    No Chainlink credentials exist in this environment, so live historical
    Chainlink data is BLOCKED (see module docstring note + commit report).

What this module provides regardless (pure + mocked-testable):
  * standards-compliant auth-header builder,
  * v3 ``fullReport`` blob decoder (documented field order; verified on
    synthetic blobs — live decode unverified while credentials are absent),
  * deterministic resolution math: market start price, 5m candles, final
    5m close, attach to :class:`UpDown5mMarket` with ``CHAINLINK``
    provenance,
  * a hard guard: settlement from anything but Chainlink provenance raises.

Research-only, read-only. No trading. No long-running collection: callers
fetch one bounded page of reports per market (``page_reports``), never a
streaming subscription.

Report value scale assumption (documented): Data Streams prices are
fixed-point with 18 decimals (``Decimal(raw) / 1e18``), per Chainlink
docs/SDKs. If the venue ever serves a different scale, only
``REPORT_DECIMALS`` changes.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from decimal import Decimal
from typing import Any

import httpx

from app.config.logging_config import get_logger
from app.models.base import DomainModel, as_decimal
from app.research.prediction_markets.up_down_5m.market import UpDown5mMarket

__all__ = [
    "BTC_USD_CEX_V3_FEED_ID",
    "CHAINLINK_DATAENGINE_MAINNET",
    "FIVE_MIN_MS",
    "REPORT_DECIMALS",
    "ChainlinkFeedClient",
    "ChainlinkFeedError",
    "ChainlinkReport",
    "FiveMinuteCandle",
    "ResolutionInputs",
    "attach_chainlink_resolution",
    "build_5m_candles",
    "build_auth_headers",
    "decode_v3_report",
    "final_close",
    "market_start_price",
    "resolve_chainlink_credentials",
    "tob_mid",
]

logger = get_logger("research.chainlink_feed")

CHAINLINK_DATAENGINE_MAINNET = "https://api.dataengine.chain.link"

#: Live BTC/USD CexPrice v3 feed (from public discovery catalog, 2026-09-13).
BTC_USD_CEX_V3_FEED_ID = (
    "0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8"
)

REPORT_DECIMALS = 18
_DECIMAL_DIVISOR = Decimal(10) ** REPORT_DECIMALS
FIVE_MIN_MS = 300_000

# v3 fullReport layout: 9 ABI words (bytes32) in documented schema order.
# feedId | validFrom | observations | nativeFee | linkFee | expiresAt |
# price | bid | ask. price/bid/ask are signed (two's complement, 32B word).
_V3_WORDS = 9
_V3_PRICE_WORD = 6
_V3_BID_WORD = 7
_V3_ASK_WORD = 8


class ChainlinkFeedError(RuntimeError):
    """Raised on Chainlink Data Streams API failure; safe fields only."""

    def __init__(self, message: str, *, status: int | None = None, code: Any | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def _safe_error(response: httpx.Response) -> Any | None:
    try:
        data = response.json()
        if isinstance(data, dict):
            return data.get("code") or data.get("error")
    except Exception:
        pass
    return None


def resolve_chainlink_credentials() -> tuple[str, str] | None:
    """Resolve Data Streams key/secret from env/.env. Never prints values."""
    pairs = (
        (os.getenv("CHAINLINK_DATA_STREAMS_KEY"), os.getenv("CHAINLINK_DATA_STREAMS_SECRET")),
        (os.getenv("CHAINLINK_API_KEY"), os.getenv("CHAINLINK_API_SECRET")),
    )
    for k, s in pairs:
        if k and s and k.strip() and s.strip():
            return k.strip(), s.strip()
    try:
        from dotenv import dotenv_values  # type: ignore[import-not-found]

        vals = dotenv_values(".env") or {}
        for k_name, s_name in (
            ("CHAINLINK_DATA_STREAMS_KEY", "CHAINLINK_DATA_STREAMS_SECRET"),
            ("CHAINLINK_API_KEY", "CHAINLINK_API_SECRET"),
        ):
            k = (vals.get(k_name) or "").strip()
            s = (vals.get(s_name) or "").strip()
            if k and s:
                return k, s
    except Exception:
        pass
    return None


def build_auth_headers(
    api_key: str,
    api_secret: str,
    method: str,
    full_path: str,
    body: bytes = b"",
    timestamp_ms: int | None = None,
) -> dict[str, str]:
    """Build the three Data Streams auth headers (official HMAC scheme).

    String to sign: ``METHOD FULL_PATH BODY_HASH API_KEY TIMESTAMP``
    (single spaces), BODY_HASH = hex(SHA-256(body)), empty body for GET.
    """
    ts = int(timestamp_ms) if timestamp_ms is not None else int(time.time() * 1000)
    body_hash = hashlib.sha256(body).hexdigest()
    to_sign = f"{method.upper()} {full_path} {body_hash} {api_key} {ts}"
    sig = hmac.new(api_secret.encode(), to_sign.encode(), hashlib.sha256).hexdigest()
    return {
        "Authorization": api_key,
        "X-Authorization-Timestamp": str(ts),
        "X-Authorization-Signature-SHA256": sig,
    }


class ChainlinkReport(DomainModel):
    """One decoded v3 report: official bid/ask with ToB mid."""

    feed_id: str
    observations_ts_ms: int
    valid_from_ts_ms: int
    bid: Decimal
    ask: Decimal
    benchmark: Decimal | None = None

    @property
    def mid(self) -> Decimal:
        return tob_mid(self.bid, self.ask)


def _word_to_int(word: bytes) -> int:
    """ABI int (sign-extended 32B word) -> int."""
    return int.from_bytes(word, byteorder="big", signed=True)


def decode_v3_report(report: dict[str, Any], decimals: int = REPORT_DECIMALS) -> ChainlinkReport:
    """Decode a ``/reports`` envelope with a v3 ``fullReport`` blob.

    Envelope timestamps are seconds -> ms. Blob words are 32 bytes each in
    documented v3 order; price/bid/ask scaled by ``10**decimals``.
    """
    try:
        feed_id = str(report["feedID"])
        valid_from_ms = int(report["validFromTimestamp"]) * 1000
        obs_ms = int(report["observationsTimestamp"]) * 1000
        blob = str(report["fullReport"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ChainlinkFeedError(f"malformed report envelope: {exc}") from exc
    hexs = blob[2:] if blob.startswith(("0x", "0X")) else blob
    try:
        raw = bytes.fromhex(hexs)
    except ValueError as exc:
        raise ChainlinkFeedError(f"fullReport is not hex: {exc}") from exc
    if len(raw) < _V3_WORDS * 32:
        raise ChainlinkFeedError(f"fullReport too short for v3: {len(raw)} bytes")
    words = [raw[i * 32:(i + 1) * 32] for i in range(_V3_WORDS)]
    divisor = Decimal(10) ** decimals
    price = Decimal(_word_to_int(words[_V3_PRICE_WORD])) / divisor
    bid = Decimal(_word_to_int(words[_V3_BID_WORD])) / divisor
    ask = Decimal(_word_to_int(words[_V3_ASK_WORD])) / divisor
    if bid <= 0 or ask <= 0 or bid > ask:
        raise ChainlinkFeedError(f"v3 bid/ask inconsistent: bid={bid} ask={ask}")
    return ChainlinkReport(
        feed_id=feed_id,
        observations_ts_ms=obs_ms,
        valid_from_ts_ms=valid_from_ms,
        bid=bid,
        ask=ask,
        benchmark=price,
    )


class ChainlinkFeedClient:
    """Read-only Data Streams REST client (reports need credentials)."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = CHAINLINK_DATAENGINE_MAINNET,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, transport=transport, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> ChainlinkFeedClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _headers(self, method: str, full_path: str) -> dict[str, str]:
        if not self.api_key or not self.api_secret:
            raise ChainlinkFeedError(
                "Chainlink credentials absent: reports endpoints require an API key+secret "
                "(CHAINLINK_DATA_STREAMS_KEY/SECRET). Discovery is the only public route."
            )
        return build_auth_headers(self.api_key, self.api_secret, method, full_path)

    async def _get(self, path: str, params: dict[str, Any], *, auth: bool) -> Any:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        full_path = path + (f"?{query}" if query else "")
        headers = self._headers("GET", full_path) if auth else {}
        r = await self._client.get(path, params=params, headers=headers)
        if r.status_code != 200:
            code = _safe_error(r)
            logger.warning("chainlink_feed_error", extra={"http_status": r.status_code, "code": code, "endpoint": path})
            raise ChainlinkFeedError(
                f"GET {path} failed {r.status_code} code={code!r} detail={r.text[:200]!r}",
                status=r.status_code,
                code=code,
            )
        return r.json()

    async def discovery(self, **filters: Any) -> list[dict[str, Any]]:
        """Public catalog (no auth). Returns feed entries."""
        data = await self._get("/api/v1/discovery", dict(filters), auth=False)
        if isinstance(data, dict):
            return data.get("feeds") or []
        return data if isinstance(data, list) else []

    async def get_report(self, feed_id: str, timestamp_sec: int) -> ChainlinkReport:
        data = await self._get("/api/v1/reports", {"feedID": feed_id, "timestamp": timestamp_sec}, auth=True)
        return decode_v3_report(data["report"])

    async def get_latest_report(self, feed_id: str) -> ChainlinkReport:
        data = await self._get("/api/v1/reports/latest", {"feedID": feed_id}, auth=True)
        return decode_v3_report(data["report"])

    async def page_reports(self, feed_id: str, start_sec: int, limit: int = 100) -> list[ChainlinkReport]:
        """One bounded page of sequential reports (no streaming, no loop)."""
        data = await self._get(
            "/api/v1/reports/page", {"feedID": feed_id, "startTimestamp": start_sec, "limit": limit}, auth=True
        )
        reports = data.get("reports") if isinstance(data, dict) else data
        if not isinstance(reports, list):
            raise ChainlinkFeedError(f"unexpected page shape: {str(data)[:200]!r}")
        out = [decode_v3_report(rep) for rep in reports]
        out.sort(key=lambda x: x.observations_ts_ms)
        return out

    def ensure_research_only(self) -> None:
        raise RuntimeError("chainlink feed is research-only; live trading is disabled (fail-closed)")


# ---------- deterministic resolution math (pure) ----------

def tob_mid(bid: Decimal | str | int, ask: Decimal | str | int) -> Decimal:
    """Top-of-Book mid from official bid/ask. Exact Decimal, no float."""
    b = as_decimal(bid, field="bid")
    a = as_decimal(ask, field="ask")
    if b <= 0 or a <= 0 or b > a:
        raise ValueError(f"inconsistent ToB book: bid={b} ask={a}")
    return (b + a) / Decimal("2")


def market_start_price(reports: list[ChainlinkReport], start_ts_ms: int) -> ChainlinkReport:
    """Market start price = latest report with observations at/before start.

    Deterministic; raises if no report covers the start (fail-closed rather
    than forward-filling from the future).
    """
    cands = [r for r in reports if r.observations_ts_ms <= int(start_ts_ms)]
    if not cands:
        raise ValueError(f"no Chainlink report at/before market start {start_ts_ms}")
    return max(cands, key=lambda r: r.observations_ts_ms)


def final_close(reports: list[ChainlinkReport], end_ts_ms: int) -> ChainlinkReport:
    """Final price = close of the 5m candle immediately before market end.

    Implemented as the latest report strictly before ``end_ts_ms`` (the
    last Chainlink observation of the terminal 5m candle). Raises if none.
    """
    cands = [r for r in reports if r.observations_ts_ms < int(end_ts_ms)]
    if not cands:
        raise ValueError(f"no Chainlink report before market end {end_ts_ms}")
    return max(cands, key=lambda r: r.observations_ts_ms)


class FiveMinuteCandle(DomainModel):
    """5m candle of Chainlink ToB mids (floor-aligned, UTC)."""

    open_ms: int
    close_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    samples: int


def build_5m_candles(reports: list[ChainlinkReport]) -> list[FiveMinuteCandle]:
    """Group report mids into floor-aligned 5m candles (deterministic)."""
    buckets: dict[int, list[ChainlinkReport]] = {}
    for r in sorted(reports, key=lambda x: x.observations_ts_ms):
        buckets.setdefault((r.observations_ts_ms // FIVE_MIN_MS) * FIVE_MIN_MS, []).append(r)
    candles: list[FiveMinuteCandle] = []
    for open_ms in sorted(buckets):
        rs = buckets[open_ms]
        mids = [r.mid for r in rs]
        candles.append(
            FiveMinuteCandle(
                open_ms=open_ms,
                close_ms=open_ms + FIVE_MIN_MS,
                open=mids[0],
                high=max(mids),
                low=min(mids),
                close=mids[-1],
                samples=len(rs),
            )
        )
    return candles


class ResolutionInputs(DomainModel):
    """Auditable Chainlink anchors for one market."""

    market_id: int
    feed_id: str = BTC_USD_CEX_V3_FEED_ID
    start_ts_ms: int
    end_ts_ms: int
    start_price: Decimal
    end_price: Decimal
    start_report_ts_ms: int
    end_report_ts_ms: int


def resolve_market(
    reports: list[ChainlinkReport],
    market_id: int,
    start_ts_ms: int,
    end_ts_ms: int,
    feed_id: str = BTC_USD_CEX_V3_FEED_ID,
) -> ResolutionInputs:
    """Compute exact resolution inputs from a bounded report set."""
    start_rep = market_start_price(reports, start_ts_ms)
    end_rep = final_close(reports, end_ts_ms)
    return ResolutionInputs(
        market_id=int(market_id),
        feed_id=feed_id,
        start_ts_ms=int(start_ts_ms),
        end_ts_ms=int(end_ts_ms),
        start_price=start_rep.mid,
        end_price=end_rep.mid,
        start_report_ts_ms=start_rep.observations_ts_ms,
        end_report_ts_ms=end_rep.observations_ts_ms,
    )


def attach_chainlink_resolution(
    market: UpDown5mMarket,
    resolution: ResolutionInputs,
    *,
    current_price: Decimal | str | int | None = None,
) -> UpDown5mMarket:
    """Attach Chainlink anchors to the market (returns a copy).

    Raises if the market id/window disagrees with the resolution inputs —
    anchors must never be attached to the wrong market.
    """
    if int(market.market_id) != int(resolution.market_id):
        raise ValueError(f"market mismatch: {market.market_id} != {resolution.market_id}")
    if int(market.start_ts_ms) != int(resolution.start_ts_ms) or int(market.end_ts_ms) != int(resolution.end_ts_ms):
        raise ValueError("resolution window disagrees with market window")
    return market.model_copy(
        update={
            "start_price": resolution.start_price,
            "end_price": resolution.end_price,
            "current_price": as_decimal(current_price, field="current_price") if current_price is not None else resolution.end_price,
        }
    )
