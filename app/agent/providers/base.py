"""LLM provider abstraction.

The advisor must not be coupled to any single LLM vendor (OpenRouter, local
model, future providers). The :class:`LLMProvider` interface is the only thing
the core talks to. Concrete providers (``OpenRouterProvider``,
``LocalModelProvider``) can be slotted in later without touching
``AgentCore``.

Every request/response is filtered for secrets before it leaves the process
or is persisted. The block-list covers:

* ``CAT_KEY_*`` exchange API keys / secrets / passphrases
* Telegram bot tokens (``CAT_TELEGRAM__BOT_TOKEN``)
* database DSNs / credentials
* raw ``.env`` content
* any value that looks like an API key / secret / token in free text

Phase 1 does not require a fully working external API integration if the
existing configuration does not supply an LLM endpoint; the abstraction and
the safe request/response boundary are what matter.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from typing import Any

from app.config.logging_config import get_logger
from app.exchanges.sanitize import redact_secrets

__all__ = [
    "LLMMessage",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "NullProvider",
    "EchoProvider",
    "filter_secrets_from_text",
    "SAFE_CONTENT_PATTERN",
]

logger = get_logger("agent.provider")


# ------------------------------------------------------------------ secret filtering

# Heuristic secret patterns that must never be sent to an LLM.
# The base sanitizer (redact_secrets) already strips api_key / secret / token
# pairs and query strings; additional patterns guard against pasting raw .env
# or DSN fragments.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bCAT_KEY_[A-Z_]*\s*[:=]\s*[^\s]+"),
    re.compile(r"(?i)\bCAT_TELEGRAM__BOT_TOKEN\s*[:=]\s*[^\s]+"),
    re.compile(r"(?i)\bCAT_DATABASE__URL\s*[:=]\s*[^\s]+"),
    re.compile(r"sqlite\+aiosqlite://[^\s]+"),
    re.compile(r"postgresql\+asyncpg://[^\s]+"),
    re.compile(r"bot\.db"),
)

# Generic fallback: strings that simply contain these substrings are treated as sensitive
# when they appear in LLM-bound text (the whole line is redacted).
_SENSITIVE_SUBSTRINGS = (
    "api_key",
    "api secret",
    "CAT_KEY",
    "CAT_TELEGRAM",
    "CAT_DATABASE",
    ".env",
)

SAFE_CONTENT_PATTERN = re.compile(r".*")  # placeholder for external validation


def filter_secrets_from_text(text: str, *, extra: tuple[str, ...] = ()) -> str:
    """Redact secrets from ``text`` before it is sent to any LLM.

    The function is deliberately *conservative*: when in doubt, redact.
    It first runs the existing :func:`app.exchanges.sanitize.redact_secrets`
    (API-key / signature / query-string stripping) and then applies the
    advisor-specific patterns above.
    """
    cleaned = redact_secrets(text, extra=extra)
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub("<redacted>", cleaned)
    # Line-level fallback: any line mentioning a sensitive substring is redacted entirely
    lines: list[str] = []
    for line in cleaned.splitlines():
        low = line.lower()
        if any(sub.lower() in low for sub in _SENSITIVE_SUBSTRINGS) and "<redacted>" not in line:
            lines.append("<redacted>")
        else:
            lines.append(line)
    return "\n".join(lines)


# ------------------------------------------------------------------ data models


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: str  # system | user | assistant
    content: str


@dataclass(frozen=True, slots=True)
class LLMRequest:
    messages: tuple[LLMMessage, ...]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    model: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


# ------------------------------------------------------------------ abstract provider


class LLMProvider(abc.ABC):
    """Abstract LLM provider — the only integration point the core uses.

    Future providers (OpenRouter, local model) implement :meth:`complete`.
    The core is responsible for calling :func:`filter_secrets_from_text` on
    every outbound prompt; providers must also filter inbound content before
    returning it (defence in depth).
    """

    name: str = "base"

    @abc.abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send ``request`` to the LLM and return the filtered response."""
        ...

    async def analyze(self, prompt: str, *, context: dict[str, Any] | None = None) -> LLMResponse:
        """Convenience: single-user-message request.

        ``prompt`` is filtered for secrets before dispatch; ``context`` is
        serialised minimally (never with secrets).
        """
        safe_prompt = filter_secrets_from_text(prompt)
        messages = (LLMMessage(role="user", content=safe_prompt),)
        return await self.complete(LLMRequest(messages=messages, metadata=dict(context or {})))

    def _safe_response(self, content: str, **kwargs: Any) -> LLMResponse:
        """Wrap ``content`` after filtering it for secrets."""
        safe = filter_secrets_from_text(content)
        return LLMResponse(content=safe, **kwargs)


# ------------------------------------------------------------------ stub providers (Phase 1)


class NullProvider(LLMProvider):
    """Deterministic stub that never calls the network.

    Used in tests and when no LLM endpoint is configured. It returns a
    canned analytical placeholder so the advisor pipeline can be exercised
    without credentials.
    """

    name = "null"

    def __init__(self, canned: str = "No LLM configured — analysis stub (no external call).") -> None:
        self._canned = canned

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # Filter inbound request for audit (defence in depth — request already filtered)
        for msg in request.messages:
            filter_secrets_from_text(msg.content)
        logger.debug("llm_null_provider_used", extra={"messages": len(request.messages)})
        return self._safe_response(self._canned, model="null", finish_reason="stub")


class EchoProvider(LLMProvider):
    """Echo back the last user message (sanitized) — useful for tests."""

    name = "echo"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        last = request.messages[-1].content if request.messages else ""
        return self._safe_response(f"ECHO: {last}", model="echo", finish_reason="echo")
