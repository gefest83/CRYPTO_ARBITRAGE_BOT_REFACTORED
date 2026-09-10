"""Predict.fun Mainnet API client — research only.

Uses x-api-key (CAT_RESEARCH__PREDICT_API_KEY) via EnvCredentials/dotenv fallback.
Read-only: no order placement, no wallet creation, no fund transfers.

Reuses helpers from existing research code where appropriate (base URL, safe logging pattern).
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from app.config.logging_config import get_logger
from app.research.prediction_markets import endpoints as ep

__all__ = ["PredictFunClient", "PredictFunError", "resolve_predict_token"]


class PredictFunError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, code: Any | None = None, binance_msg: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.binance_msg = binance_msg


logger = get_logger("research.predict_client")


def _safe_predict_error(resp: httpx.Response) -> tuple[Any | None, str | None]:
    try:
        data = resp.json()
        if isinstance(data, dict):
            return data.get("code"), (data.get("message") or data.get("error") or data.get("msg"))
    except Exception:
        pass
    return None, None


def _redacted_log(status: int | None, code: Any | None, msg: str | None, path: str) -> None:
    # never log api key / secret
    logger.warning(
        "predict_api_error",
        extra={"http_status": status, "predict_code": code, "predict_msg": (msg or "")[:500], "endpoint": path},
    )


def resolve_predict_token() -> str | None:
    """Resolve Predict.fun API token from env/.env without printing it."""
    # via EnvCredentials pattern: check os env then dotenv
    for key in ("CAT_RESEARCH__PREDICT_API_KEY", "PREDICT_API_KEY", "PREDICT_FUN_API_KEY"):
        v = os.getenv(key)
        if v and v.strip():
            return v.strip()
    try:
        from dotenv import dotenv_values

        vals = dotenv_values(".env") or {}
        for k in ("CAT_RESEARCH__PREDICT_API_KEY", "PREDICT_API_KEY", "PREDICT_FUN_API_KEY"):
            v = (vals.get(k) or "").strip()
            if v:
                return v
    except Exception:
        pass
    return None


class PredictFunClient:
    """Read-only Predict.fun Mainnet client (x-api-key)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = ep.PREDICT_FUN_BASE_MAINNET,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            transport=transport,
            timeout=timeout,
            headers={"x-api-key": api_key},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> PredictFunClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        r = await self._client.get(path, params=params or {})
        if r.status_code != 200:
            code, msg = _safe_predict_error(r)
            _redacted_log(r.status_code, code, msg, path)
            safe = f"GET {path} failed {r.status_code} code={code!r} msg={msg!r}"
            raise PredictFunError(safe, status=r.status_code, code=code, binance_msg=msg)
        return r.json()

    # --- Markets ---
    async def list_markets(self, limit: int = 20, cursor: str | None = None, **extra: Any) -> dict:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        params.update(extra)
        return await self._get("/v1/markets", params)

    async def list_categories(self, limit: int = 20, cursor: str | None = None) -> dict:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self._get("/v1/categories", params)

    async def get_market(self, market_id: int) -> dict:
        return await self._get(f"/v1/markets/{market_id}")

    # --- Timeseries ---
    async def get_timeseries(
        self,
        market_id: int,
        metric: str = "chance",
        from_sec: int | None = None,
        to_sec: int | None = None,
        resolution: str | None = None,
        limit: int | None = None,
        after: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"metric": metric}
        if from_sec is not None:
            params["from"] = from_sec
        if to_sec is not None:
            params["to"] = to_sec
        if resolution is not None:
            params["resolution"] = resolution
        if limit is not None:
            params["limit"] = limit
        if after is not None:
            params["after"] = after
        return await self._get(f"/v1/markets/{market_id}/timeseries", params)

    async def get_orderbook(self, market_id: int) -> dict:
        return await self._get(f"/v1/markets/{market_id}/orderbook")

    async def get_stats(self, market_id: int) -> dict:
        return await self._get(f"/v1/markets/{market_id}/stats")

    async def get_last_sale(self, market_id: int) -> dict:
        return await self._get(f"/v1/markets/{market_id}/last-sale")

    async def get_quote(self, market_id: int) -> dict:
        return await self._get(f"/v1/markets/{market_id}/quote")
