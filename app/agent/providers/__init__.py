"""Agent LLM provider package.

Factory :func:`create_provider` builds the configured provider from
:class:`app.config.settings.Settings` without performing any network I/O.
"""

from __future__ import annotations

from typing import Any

from app.agent.providers.base import (
    EchoProvider,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    NullProvider,
    filter_secrets_from_text,
)

__all__ = [
    "EchoProvider",
    "LLMMessage",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "NullProvider",
    "OpenRouterProvider",
    "create_provider",
    "filter_secrets_from_text",
]

# Lazy import so that importing this package never pulls httpx unless openrouter is used.
try:
    from app.agent.providers.openrouter import OpenRouterProvider  # noqa: F401
except Exception:  # pragma: no cover
    OpenRouterProvider = None  # type: ignore[assignment]


def create_provider(settings: Any | None = None, *, http_client: Any | None = None) -> LLMProvider:
    """Build the configured provider without network I/O.

    ``settings`` may be ``None`` (returns ``NullProvider``), a
    ``Settings`` instance, or any object exposing ``agent`` with the same
    fields. The function never logs the API key.
    """
    if settings is None:
        return NullProvider()

    agent_cfg = getattr(settings, "agent", None)
    if agent_cfg is None:
        return NullProvider()

    provider_name = str(getattr(agent_cfg, "provider", "null") or "null").strip().lower()
    if provider_name == "echo":
        return EchoProvider()
    if provider_name == "openrouter":
        if OpenRouterProvider is None:
            return NullProvider()
        # Pull values via getattr to stay compatible with test fakes
        from pydantic import SecretStr

        api_key = getattr(agent_cfg, "api_key", SecretStr(""))
        model = str(getattr(agent_cfg, "model", "openai/gpt-4o-mini") or "openai/gpt-4o-mini")
        base_url = str(getattr(agent_cfg, "base_url", "https://openrouter.ai/api/v1") or "https://openrouter.ai/api/v1")
        timeout = float(getattr(agent_cfg, "timeout_seconds", 10.0) or 10.0)
        retries = int(getattr(agent_cfg, "max_retries", 1) or 1)
        temperature = float(getattr(agent_cfg, "temperature", 0.2) or 0.2)
        max_tokens = int(getattr(agent_cfg, "max_tokens", 800) or 800)
        referer = getattr(agent_cfg, "referer", None)
        title = getattr(agent_cfg, "title", None)
        max_prompt_chars = int(getattr(agent_cfg, "max_prompt_chars", 6000) or 6000)
        max_response_chars = int(getattr(agent_cfg, "max_response_chars", 4000) or 4000)
        rate_limit_per_minute = int(getattr(agent_cfg, "rate_limit_per_minute", 10) or 10)
        rate_limit_per_hour = int(getattr(agent_cfg, "rate_limit_per_hour", 60) or 60)
        budget_max = int(getattr(agent_cfg, "budget_max_requests_per_day", 200) or 200)
        circuit_threshold = int(getattr(agent_cfg, "circuit_failure_threshold", 5) or 5)
        circuit_cooldown = float(getattr(agent_cfg, "circuit_cooldown_seconds", 60.0) or 60.0)
        return OpenRouterProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout,
            max_retries=retries,
            temperature=temperature,
            max_tokens=max_tokens,
            referer=referer,
            title=title,
            http_client=http_client,
            max_prompt_chars=max_prompt_chars,
            max_response_chars=max_response_chars,
            rate_limit_per_minute=rate_limit_per_minute,
            rate_limit_per_hour=rate_limit_per_hour,
            budget_max_requests_per_day=budget_max,
            circuit_failure_threshold=circuit_threshold,
            circuit_cooldown_seconds=circuit_cooldown,
        )
    if provider_name == "local":
        # Future local provider — currently behaves as null (no network)
        return NullProvider(canned="Local provider not yet implemented — stub.")
    # default
    return NullProvider()
