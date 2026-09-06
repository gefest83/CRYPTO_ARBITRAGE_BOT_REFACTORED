"""OpenRouter provider — production LLM via OpenRouter API.

The provider is strictly read-only and never touches trading execution.
All prompts and responses are filtered via :func:`filter_secrets_from_text`
before leaving/entering the process. Credentials are never stored in
agent tables and never appear in logs/exceptions/telegram.

Bounded behaviour:
* ``timeout_seconds`` caps each HTTP call.
* ``max_retries`` caps transient retries (429 / 502-504 / timeout / network).
* Fail-closed: any error returns via exception that ``AgentCore`` catches;
  trading execution is unaffected.

No network request is made at construction — only when ``complete()`` is
called.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import SecretStr

from app.agent.providers.base import LLMProvider, LLMRequest, LLMResponse, filter_secrets_from_text
from app.config.logging_config import get_logger
from app.exchanges.sanitize import redact_secrets

__all__ = ["OpenRouterProvider", "OpenRouterError"]

logger = get_logger("agent.provider.openrouter")


class OpenRouterError(Exception):
    """Provider failure (fail-closed — trading unaffected)."""

    pass


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenRouterProvider(LLMProvider):
    """OpenRouter Chat Completions provider.

    ``model`` is fully configurable (e.g. ``openai/gpt-4o-mini``,
    ``anthropic/claude-3.5-sonnet``). The API key is held as a
    :class:`SecretStr` and never logged.
    """

    name = "openrouter"

    def __init__(
        self,
        *,
        api_key: SecretStr | str,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 10.0,
        max_retries: int = 1,
        temperature: float = 0.2,
        max_tokens: int = 800,
        referer: str | None = None,
        title: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        # Never store raw key in plain attribute name that could be dumped
        self._api_key = api_key if isinstance(api_key, SecretStr) else SecretStr(str(api_key))
        self._model_default = model.strip() if model else "openai/gpt-4o-mini"
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout_seconds)
        self._max_retries = int(max_retries)
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._referer = referer
        self._title = title
        # Optional injected client for tests (must expose ``post`` async)
        self._http_client = http_client

    @property
    def model(self) -> str:
        return self._model_default

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # Defensive: missing API key -> fail closed without network
        raw_key = self._api_key.get_secret_value().strip() if isinstance(self._api_key, SecretStr) else str(self._api_key).strip()
        if not raw_key:
            raise OpenRouterError("missing API key — provider not configured")

        # Filter outbound messages before any network
        safe_messages = []
        for msg in request.messages:
            safe_content = filter_secrets_from_text(msg.content)
            safe_messages.append({"role": msg.role, "content": safe_content})

        # Use request model override or default
        model = request.model or self._model_default
        payload: dict[str, Any] = {
            "model": model,
            "messages": safe_messages,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        else:
            payload["temperature"] = self._temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        elif self._max_tokens:
            payload["max_tokens"] = self._max_tokens

        headers: dict[str, str] = {
            "Authorization": f"Bearer {raw_key}",
            "Content-Type": "application/json",
        }
        if self._referer:
            headers["HTTP-Referer"] = self._referer
        if self._title:
            headers["X-Title"] = self._title

        url = f"{self._base_url}/chat/completions"

        # Retry loop for transient failures
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response_content = await self._post(url, headers, payload)
                # Parse successful response
                return self._parse_response(response_content, model)
            except OpenRouterError as exc:
                # Non-retryable (e.g. missing key, malformed) -> fail immediately
                last_exc = exc
                # Only retry on retryable status/timeout if flagged
                if getattr(exc, "_retryable", False) and attempt < self._max_retries:
                    backoff = 0.2 * (2**attempt)
                    await asyncio.sleep(backoff)
                    continue
                raise
            except Exception as exc:  # noqa: BLE001 - network/timeout
                last_exc = exc
                # Treat as retryable if we have retries left
                is_retryable = True
                if attempt < self._max_retries and is_retryable:
                    backoff = 0.2 * (2**attempt)
                    await asyncio.sleep(backoff)
                    continue
                # Fail closed — redact exception before raising
                safe_msg = redact_secrets(str(exc))
                safe_msg = filter_secrets_from_text(safe_msg)
                logger.warning("openrouter_provider_failed", extra={"attempt": attempt, "error": safe_msg[:300]})
                raise OpenRouterError(f"provider failure: {safe_msg[:200]}") from exc

        # Should not reach here
        raise OpenRouterError(f"provider failed after retries: {last_exc}")

    async def _post(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        """POST ``payload`` to ``url`` and return parsed JSON.

        Separated for test injection (via ``http_client``) and retry handling.
        """
        # Injected fake client for tests
        if self._http_client is not None:
            # Fake client contract: await client.post(url, json=payload, headers=headers, timeout=...)
            # It should return a dict or raise.
            try:
                result = await self._http_client.post(url, json=payload, headers=headers, timeout=self._timeout)
                if isinstance(result, dict):
                    return result
                # Assume httpx-like response with .json()
                if hasattr(result, "json"):
                    data = result.json() if not asyncio.iscoroutinefunction(result.json) else await result.json()
                    # Check status
                    status = getattr(result, "status_code", 200)
                    if status in _RETRYABLE_STATUS:
                        err = OpenRouterError(f"retryable HTTP {status}")
                        err._retryable = True  # type: ignore[attr-defined]
                        raise err
                    if status >= 400:
                        # Redact body before raising
                        safe_body = redact_secrets(str(data)[:500])
                        raise OpenRouterError(f"HTTP {status}: {safe_body[:200]}")
                    return data
                return dict(result)
            except OpenRouterError:
                raise
            except Exception as exc:
                # Check if its httpx Timeout etc — treat as retryable
                msg = str(exc).lower()
                if "timeout" in msg or "429" in msg or "502" in msg or "503" in msg or "504" in msg:
                    err = OpenRouterError(f"transient failure: {exc}")
                    err._retryable = True  # type: ignore[attr-defined]
                    raise err
                raise

        # Real httpx path
        try:
            import httpx  # local import so tests without network still import module
        except ImportError as exc:
            raise OpenRouterError("httpx not available") from exc

        # Never log headers (contains Bearer) — redacted
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code in _RETRYABLE_STATUS:
                    err = OpenRouterError(f"retryable HTTP {resp.status_code}")
                    err._retryable = True  # type: ignore[attr-defined]
                    raise err
                if resp.status_code >= 400:
                    # Do not include raw body with potential secrets
                    safe_body = redact_secrets(resp.text[:500])
                    safe_body = filter_secrets_from_text(safe_body)
                    raise OpenRouterError(f"HTTP {resp.status_code}: {safe_body[:300]}")
                try:
                    data = resp.json()
                except Exception as exc:
                    raise OpenRouterError(f"malformed JSON response: {exc}") from exc
                return data
        except OpenRouterError:
            raise
        except Exception as exc:
            # httpx timeout / network
            msg = str(exc).lower()
            retryable = "timeout" in msg or "connect" in msg or "read" in msg
            if retryable:
                err = OpenRouterError(f"transient network failure: {exc}")
                err._retryable = True  # type: ignore[attr-defined]
                raise err
            raise

    def _parse_response(self, data: dict[str, Any], model: str) -> LLMResponse:
        """Parse OpenRouter JSON envelope; fail closed on malformed."""
        try:
            # Expected shape: {"choices": [{"message": {"content": "..."}}], "model": "...", "usage": {...}}
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise OpenRouterError("malformed response: missing choices")
            first = choices[0]
            if not isinstance(first, dict):
                raise OpenRouterError("malformed response: choice not a dict")
            # OpenRouter may use .message.content or .text
            msg = first.get("message") or first
            if isinstance(msg, dict):
                content = msg.get("content")
                if content is None:
                    # Some models return text field
                    content = msg.get("text", "")
            else:
                content = str(msg)
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = str(content)
            # Usage may be absent
            usage = data.get("usage", {})
            # Secret filtering on inbound content
            safe_content = filter_secrets_from_text(content)
            return LLMResponse(
                content=safe_content,
                model=data.get("model", model),
                finish_reason=first.get("finish_reason"),
                usage=dict(usage) if isinstance(usage, dict) else {},
                raw=data,
            )
        except OpenRouterError:
            raise
        except Exception as exc:
            raise OpenRouterError(f"malformed response: {exc}") from exc
