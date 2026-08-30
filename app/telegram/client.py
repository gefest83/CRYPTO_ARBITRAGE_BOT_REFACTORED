"""Minimal async Telegram Bot API client (long polling).

Deliberately dependency-light: the only transport is httpx, imported lazily
so the bot degrades gracefully (disabled with a clear message) when httpx is
not installed.  No framework, no webhook server — ``getUpdates`` long polling
is enough for a control interface.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.config.logging_config import get_logger

__all__ = ["TelegramClient", "TelegramTransportError"]

logger = get_logger("telegram.client")

_API_BASE = "https://api.telegram.org"
#: Telegram truncates message bodies at 4096 characters.
MAX_MESSAGE_LENGTH = 4000


class TelegramTransportError(Exception):
    """The Bot API could not be reached (the bot keeps polling)."""


class TelegramClient:
    def __init__(self, bot_token: str, *, poll_timeout_seconds: int = 30) -> None:
        self._token = bot_token.strip()
        self._poll_timeout = poll_timeout_seconds
        self._session: Any | None = None

    @property
    def base_url(self) -> str:
        return f"{_API_BASE}/bot{self._token}"

    async def _ensure_session(self) -> Any:
        if self._session is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise TelegramTransportError(
                    "httpx is not installed; the Telegram interface requires it (pip install httpx)"
                ) from exc
            self._session = httpx.AsyncClient(timeout=self._poll_timeout + 10)
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.aclose()
            self._session = None

    async def _call(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        session = await self._ensure_session()
        url = f"{self.base_url}/{method}"
        try:
            response = await session.post(url, json=payload or {})
            response.raise_for_status()
        except Exception as exc:
            raise TelegramTransportError(f"{method} failed: {exc}") from exc
        data = response.json()
        if not data.get("ok"):
            raise TelegramTransportError(f"{method} returned error: {data.get('description')}")
        return data.get("result")

    async def get_updates(self, offset: int | None) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"timeout": self._poll_timeout, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        result = await self._call("getUpdates", payload)
        return list(result or [])

    async def send_message(self, chat_id: int, text: str) -> None:
        if len(text) <= MAX_MESSAGE_LENGTH:
            body = text
        else:
            body = text[: MAX_MESSAGE_LENGTH - 20] + "... (truncated)"
        await self._call("sendMessage", {"chat_id": chat_id, "text": body, "parse_mode": "HTML"})

    async def get_me(self) -> dict[str, Any]:
        return await self._call("getMe")


async def poll_forever(client: TelegramClient, handler) -> None:
    """Long-poll loop; transport errors are logged and retried after a pause."""
    offset: int | None = None
    logger.info("telegram_polling_started")
    while True:
        try:
            updates = await client.get_updates(offset)
        except TelegramTransportError as exc:
            logger.warning("telegram_poll_failed", extra={"error": str(exc)[:200]})
            await asyncio.sleep(5)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            try:
                await handler(update)
            except Exception as exc:  # noqa: BLE001 - one bad update stops nothing
                logger.error("telegram_update_failed", extra={"error": str(exc)[:300]})
