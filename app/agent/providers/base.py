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
    "sanitize_untrusted_text",
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

# Prompt-injection patterns — retrieved text is untrusted data, never authority.
# We do not attempt to be exhaustive; we neutralise the most common jailbreak
# carriers so that even if a repo doc or journal entry contains them, they are
# rendered harmless before reaching the LLM.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bignore\s+previous\s+instructions\b"),
    re.compile(r"(?i)\bignore\s+all\s+instructions\b"),
    re.compile(r"(?i)\bcall\s+create_order\b"),
    re.compile(r"(?i)\bwithdraw\b.*\bfunds\b"),
    re.compile(r"(?i)\breveal\s+api\b"),
    re.compile(r"(?i)\bprint\s+environment\b"),
    re.compile(r"(?i)\bexecute\s+shell\b"),
    re.compile(r"(?i)\bmodify\s+risk\b"),
    re.compile(r"(?i)\bmodify\s+config\b"),
    re.compile(r"(?i)\bexecute\s+sql\b"),
    re.compile(r"(?i)\bdrop\s+table\b"),
    re.compile(r"(?i)\byou\s+are\s+now\b"),
    re.compile(r"(?i)^\s*system\s*:"),
    re.compile(r"(?i)^\s*assistant\s*:"),
    re.compile(r"(?i)\btool\s*:\b"),
)

MAX_UNTRUSTED_CHARS = 2000  # hard cap for any single untrusted field when rendered for LLM


def sanitize_untrusted_text(text: str, *, max_chars: int = MAX_UNTRUSTED_CHARS) -> str:
    """Treat ``text`` as untrusted data — never as instructions.

    * Truncates to ``max_chars`` (bounded retrieval).
    * Neutralises common prompt-injection carriers by replacing them with
      ``[filtered]``.
    * Also runs secret filtering so that even untrusted data with embedded
      secrets is redacted.
    * The function is idempotent and preserves provenance — the original
      stored text is unchanged; only the rendered-for-LLM copy is filtered.
    """
    if not text:
        return text
    # First secret filtering (conservative)
    cleaned = filter_secrets_from_text(text)
    # Truncate early to avoid regex on huge blobs
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 20] + "... (truncated)"
    # Neutralise injection patterns
    for pat in _INJECTION_PATTERNS:
        cleaned = pat.sub("[filtered]", cleaned)
    # Escape markdown code fences that could be used to hide instructions
    cleaned = cleaned.replace("```", "` `[filtered]` `")
    return cleaned


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
class LLMToolCall:
    """Parsed tool call from an LLM (native function calling)."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str | None = None


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: str  # system | user | assistant | tool
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: tuple[LLMToolCall, ...] | None = None


@dataclass(frozen=True, slots=True)
class LLMRequest:
    messages: tuple[LLMMessage, ...]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tools: tuple[dict[str, Any], ...] | None = None
    tool_choice: str | None = None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    model: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    tool_calls: tuple[LLMToolCall, ...] | None = None


# ------------------------------------------------------------------ abstract provider


class LLMProvider(abc.ABC):
    """Abstract LLM provider — the only integration point the core uses.

    Future providers (OpenRouter, local model) implement :meth:`complete`.
    The core is responsible for calling :func:`filter_secrets_from_text` on
    every outbound prompt; providers must also filter inbound content before
    returning it (defence in depth).
    """

    name: str = "base"
    supports_tool_calling: bool = False

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
    supports_tool_calling = False

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
    supports_tool_calling = False

    async def complete(self, request: LLMRequest) -> LLMResponse:
        last = request.messages[-1].content if request.messages else ""
        return self._safe_response(f"ECHO: {last}", model="echo", finish_reason="echo")
