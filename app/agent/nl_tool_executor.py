"""Strict LLM tool executor for natural-language requests.

Allowlist: exactly the 10 read-only AgentTools.
Never exposes approve/reject/trade/order/withdraw/config/start/stop/execution tools.
Never imports execution, transfer-execution, recovery, or AutoTrader modules.

Every tool result is NOT trusted — caller must pass it through
sanitize_untrusted_text + filter_secrets before feeding back to LLM.
"""

from __future__ import annotations

import json
from typing import Any

from app.agent.providers.base import filter_secrets_from_text, sanitize_untrusted_text
from app.agent.tools import ToolAccessBlocked

__all__ = [
    "ALLOWED_NL_TOOLS",
    "NL_TOOL_DEFINITIONS",
    "MAX_NL_TOOL_CALLS",
    "validate_tool_call",
    "NLToolExecutor",
]

MAX_NL_TOOL_CALLS = 5

ALLOWED_NL_TOOLS: frozenset[str] = frozenset(
    {
        "get_recent_trades",
        "get_trade_statistics",
        "get_scan_statistics",
        "get_current_parameters",
        "get_risk_state",
        "get_exchange_status",
        "get_balances",
        "get_recent_journal",
        "get_previous_recommendations",
        "get_memory",
    }
)

# JSON Schema definitions for native tool calling (OpenAI/OpenRouter format)
NL_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "get_recent_trades",
            "description": "Get recent trades (read-only, no execution).",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trade_statistics",
            "description": "Aggregate trade statistics (read-only).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_scan_statistics",
            "description": "Market data freshness / scan health (read-only).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_parameters",
            "description": "Current non-sensitive configuration parameters (read-only).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_state",
            "description": "Live risk state (read-only).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_status",
            "description": "Per-venue health (read-only, no credentials).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balances",
            "description": "Current balances per venue (read-only, no keys).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_journal",
            "description": "Recent audit log entries (read-only).",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_previous_recommendations",
            "description": "Prior advisor recommendations (read-only).",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 20}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory",
            "description": "Retrieve memory experiences/lessons/knowledge (read-only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "additionalProperties": False,
            },
        },
    },
)


def validate_tool_call(name: str, arguments: Any) -> dict[str, Any]:
    """Validate tool name and arguments, raising ToolAccessBlocked on violation."""
    if name not in ALLOWED_NL_TOOLS:
        raise ToolAccessBlocked(f"tool '{name}' is not in the read-only allowlist")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolAccessBlocked(f"tool '{name}' arguments must be an object")
    # Per-tool argument validation (strict)
    if name == "get_recent_trades":
        if "limit" in arguments:
            v = arguments["limit"]
            if not isinstance(v, int) or not (1 <= v <= 100):
                raise ToolAccessBlocked("get_recent_trades limit must be int 1..100")
        # strip unknown keys already enforced by schema, but double-check
        for k in list(arguments.keys()):
            if k not in ("limit",):
                raise ToolAccessBlocked(f"unknown argument '{k}' for {name}")
        return {"limit": int(arguments.get("limit", 20))}
    if name == "get_recent_journal":
        if "limit" in arguments:
            v = arguments["limit"]
            if not isinstance(v, int) or not (1 <= v <= 50):
                raise ToolAccessBlocked("get_recent_journal limit must be int 1..50")
        for k in list(arguments.keys()):
            if k not in ("limit",):
                raise ToolAccessBlocked(f"unknown argument '{k}' for {name}")
        return {"limit": int(arguments.get("limit", 50))}
    if name == "get_previous_recommendations":
        if "limit" in arguments:
            v = arguments["limit"]
            if not isinstance(v, int) or not (1 <= v <= 20):
                raise ToolAccessBlocked("get_previous_recommendations limit must be int 1..20")
        for k in list(arguments.keys()):
            if k not in ("limit",):
                raise ToolAccessBlocked(f"unknown argument '{k}' for {name}")
        return {"limit": int(arguments.get("limit", 20))}
    if name == "get_memory":
        out: dict[str, Any] = {}
        if "query" in arguments:
            q = arguments["query"]
            if q is not None and not isinstance(q, str):
                raise ToolAccessBlocked("get_memory query must be string or null")
            out["query"] = str(q)[:500] if q is not None else None
        if "limit" in arguments:
            v = arguments["limit"]
            if not isinstance(v, int) or not (1 <= v <= 20):
                raise ToolAccessBlocked("get_memory limit must be int 1..20")
            out["limit"] = int(v)
        # check unknown
        for k in list(arguments.keys()):
            if k not in ("query", "limit"):
                raise ToolAccessBlocked(f"unknown argument '{k}' for {name}")
        return out
    # no-arg tools
    if arguments:
        raise ToolAccessBlocked(f"tool '{name}' takes no arguments")
    return {}


class NLToolExecutor:
    """Executes allowlisted read-only tools via AgentTools."""

    def __init__(self, tools: Any) -> None:
        self._tools = tools
        self._calls = 0

    @property
    def calls_made(self) -> int:
        return self._calls

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute one validated tool call and return sanitized JSON string."""
        if name not in ALLOWED_NL_TOOLS:
            raise ToolAccessBlocked(f"tool '{name}' is not in the read-only allowlist")
        if self._calls >= MAX_NL_TOOL_CALLS:
            raise ToolAccessBlocked(f"tool budget exhausted ({MAX_NL_TOOL_CALLS})")
        # Dispatch — only allowlisted methods
        method = getattr(self._tools, name, None)
        if method is None:
            raise ToolAccessBlocked(f"tool '{name}' not found")
        # Call with validated kwargs
        if name in ("get_recent_trades", "get_recent_journal", "get_previous_recommendations"):
            result = await method(limit=arguments.get("limit", 20 if name != "get_recent_journal" else 50))
        elif name == "get_memory":
            kwargs: dict[str, Any] = {}
            if "query" in arguments:
                kwargs["query"] = arguments["query"]
            if "limit" in arguments:
                kwargs["limit"] = arguments["limit"]
            result = await method(**kwargs)
        else:
            result = await method()
        self._calls += 1
        # Serialize result with size bound, then sanitize before returning
        try:
            raw = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            raw = str(result)
        if len(raw) > 8000:
            raw = raw[:8000] + "... (truncated)"
        # Caller will also sanitize before LLM; defense in depth here too
        cleaned = sanitize_untrusted_text(filter_secrets_from_text(raw), max_chars=4000)
        return cleaned
