"""Binance Prediction Markets REST client (read-only, signed, mockable).

No order placement, no transfers, no wallet mutations.
All calls are signed SAPI GETs except get-quote (POST) which is inspected
without submitting an order.

Injectable httpx.AsyncClient or transport for deterministic tests.
"""

from __future__ import annotations

import hashlib
import hmac
import time
import urllib.parse
from typing import Any

import httpx

from app.config.logging_config import get_logger
from app.research.prediction_markets import endpoints as ep

__all__ = ["BinancePredictionClient", "SignedRequestError"]


class SignedRequestError(RuntimeError):
    """Raised on non-200 from Prediction SAPI; carries safe fields only."""

    def __init__(self, message: str, *, status: int | None = None, code: int | str | None = None, binance_msg: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.binance_msg = binance_msg

    @property
    def safe_message(self) -> str:
        return str(self.args[0]) if self.args else ""


logger = get_logger("research.prediction_client")


def _safe_binance_error(response: httpx.Response) -> tuple[int | str | None, str | None]:
    """Extract Binance code/msg without touching secrets; never logs headers/keys."""
    try:
        data = response.json()
        if isinstance(data, dict):
            code = data.get("code")
            msg = data.get("msg") or data.get("message") or data.get("error")
            if msg is not None:
                msg = str(msg)[:500]
            return code, msg
    except Exception:
        pass
    return None, None


def _redacted_log(status: int | None, code: int | str | None, msg: str | None, path: str) -> None:
    # safe: only status, binance code, binance msg, endpoint path
    # never log api key, secret, signature, Authorization, query params
    logger.warning(
        "prediction_api_error",
        extra={
            "http_status": status,
            "binance_code": code,
            "binance_msg": (msg or "")[:500],
            "endpoint": path,
        },
    )


def _sign(query: str, secret: str) -> str:
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


class BinancePredictionClient:
    """Read-only client for Binance Wallet Prediction Markets SAPI."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = ep.BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            transport=transport,
            timeout=timeout,
            headers={"X-MBX-APIKEY": api_key},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> BinancePredictionClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _signed_params(self, params: dict[str, Any]) -> dict[str, Any]:
        p = {k: v for k, v in params.items() if v is not None}
        p["timestamp"] = int(time.time() * 1000)
        # canonical query string sorted by key
        qs = urllib.parse.urlencode(sorted(p.items()), doseq=True)
        sig = _sign(qs, self.api_secret)
        p["signature"] = sig
        return p

    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        sp = self._signed_params(params)
        r = await self._client.get(path, params=sp)
        if r.status_code != 200:
            code, msg = _safe_binance_error(r)
            _redacted_log(r.status_code, code, msg, path)
            safe_msg = f"GET {path} failed {r.status_code} code={code!r} msg={msg!r}"
            raise SignedRequestError(safe_msg, status=r.status_code, code=code, binance_msg=msg)
        return r.json()

    async def _post(self, path: str, params: dict[str, Any]) -> Any:
        sp = self._signed_params(params)
        r = await self._client.post(path, params=sp)
        if r.status_code != 200:
            code, msg = _safe_binance_error(r)
            _redacted_log(r.status_code, code, msg, path)
            safe_msg = f"POST {path} failed {r.status_code} code={code!r} msg={msg!r}"
            raise SignedRequestError(safe_msg, status=r.status_code, code=code, binance_msg=msg)
        return r.json()

    # --- Market data ---

    async def list_categories(self) -> dict:
        return await self._get(ep.ENDPOINT_CATEGORY_LIST, {})

    async def list_markets(
        self,
        l1_category: str | None = None,
        l2_category: str | None = None,
        sort_by: str | None = None,
        order_by: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> dict:
        return await self._get(
            ep.ENDPOINT_MARKET_LIST,
            {"l1Category": l1_category, "l2Category": l2_category, "sortBy": sort_by, "orderBy": order_by, "offset": offset, "limit": limit},
        )

    async def search_markets(self, query: str, top_k: int = 20) -> list[dict]:
        data = await self._get(ep.ENDPOINT_MARKET_SEARCH, {"query": query, "topK": top_k})
        return data if isinstance(data, list) else data.get("marketTopics", data)

    async def get_market_detail(self, market_topic_id: int) -> dict:
        return await self._get(ep.ENDPOINT_MARKET_DETAIL, {"marketTopicId": market_topic_id})

    async def get_order_book(self, vendor: str, market_id: int, token_id: str) -> dict:
        return await self._get(ep.ENDPOINT_ORDER_BOOK, {"vendor": vendor, "marketId": market_id, "tokenId": token_id})

    async def get_last_trade_price(self, market_id: int) -> dict:
        return await self._get(ep.ENDPOINT_LAST_TRADE_PRICE, {"marketId": market_id})

    async def list_wallets(self) -> dict:
        return await self._get(ep.ENDPOINT_WALLET_LIST, {})

    # --- Quote inspection (read-only, no order placed) ---

    async def get_quote_inspection(
        self,
        wallet_address: str,
        token_id: str,
        side: str,
        amount_in_wei: str,
        order_type: str = "MARKET",
        slippage_bps: int = 1200,
        chain_id: str = "56",
        price_limit: str | None = None,
    ) -> dict:
        """Inspect quote API — returns pricing, fee, impact, expiry. Does NOT place order."""
        params: dict[str, Any] = {
            "walletAddress": wallet_address,
            "tokenId": token_id,
            "side": side,
            "amountIn": amount_in_wei,
            "orderType": order_type,
            "slippageBps": slippage_bps,
            "chainId": chain_id,
        }
        if price_limit is not None:
            params["priceLimit"] = price_limit
        return await self._post(ep.ENDPOINT_GET_QUOTE, params)

    def quote_spec(self) -> dict:
        """Return quote endpoint spec without network call (for reporting)."""
        return {
            "endpoint": ep.ENDPOINT_GET_QUOTE,
            "method": "POST",
            "required": ["walletAddress", "tokenId", "side", "amountIn", "orderType", "slippageBps"],
            "optional": ["chainId", "priceLimit", "feeRateBps", "fundingSource"],
            "response": ["quoteId", "averagePrice", "priceImpact", "feeAmount", "amountOut", "expireAt"],
            "read_only": True,
            "places_order": False,
        }
