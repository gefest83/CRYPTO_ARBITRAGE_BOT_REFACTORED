"""Execution guard: mode policy + kill switch.

Every order path must call :meth:`ExecutionGuard.ensure_can_trade` first.
Centralising the check means "can we trade right now?" has exactly one answer.

The kill switch:

* stops ALL new execution immediately when engaged;
* is persisted in the ``bot_state`` table, so it survives restarts — engaging
  it and rebooting the bot must never silently re-enable trading;
* must be released explicitly by the operator.
"""

from __future__ import annotations

from typing import Any

from app.config.logging_config import get_logger
from app.config.modes import ModePolicy
from app.errors import ExecutionDisabledError
from app.models.base import utc_now
from app.models.enums import TradingMode
from app.storage.repositories import BotStateRepository

__all__ = ["ExecutionGuard"]

logger = get_logger("execution.guard")

_KILL_SWITCH_KEY = "kill_switch"


class ExecutionGuard:
    """Holds the trading interlocks of the running bot."""

    def __init__(self, policy: ModePolicy, *, trading_enabled: bool = False) -> None:
        self._policy = policy
        self._trading_enabled = trading_enabled
        self._halt_reason: str | None = None
        self._halted_at: str | None = None

    # ---------------------------------------------------------------- state
    @property
    def policy(self) -> ModePolicy:
        return self._policy

    @property
    def mode(self) -> TradingMode:
        return self._policy.mode

    @property
    def trading_enabled(self) -> bool:
        return self._trading_enabled

    @property
    def is_halted(self) -> bool:
        return self._halt_reason is not None

    @property
    def halt_reason(self) -> str | None:
        return self._halt_reason

    # ---------------------------------------------------------------- checks
    def ensure_can_trade(self) -> None:
        """The hard gate: raise when the kill switch is engaged.

        Used by every execution path (explicit operator commands included) —
        the kill switch stops ALL new execution immediately.  The auto-trading
        flag is NOT checked here: it governs the background loop, not the
        operator's explicit commands.
        """
        if self.is_halted:
            raise ExecutionDisabledError(
                f"kill switch engaged: {self._halt_reason}", mode=str(self.mode)
            )

    def ensure_auto_trading(self) -> None:
        """The background-loop gate: kill switch released AND auto flag on."""
        if self.is_halted:
            raise ExecutionDisabledError(
                f"kill switch engaged: {self._halt_reason}", mode=str(self.mode)
            )
        if not self._trading_enabled:
            raise ExecutionDisabledError("auto trading is disabled", mode=str(self.mode))

    def can_trade(self) -> bool:
        try:
            self.ensure_can_trade()
        except ExecutionDisabledError:
            return False
        return True

    def enable_trading(self, *, enabled: bool = True) -> None:
        self._trading_enabled = enabled
        logger.warning("trading_flag_changed", extra={"enabled": enabled, "mode": str(self.mode)})

    # ---------------------------------------------------------------- kill switch
    def engage_kill_switch(self, reason: str) -> None:
        """Engage the kill switch: no new execution from this moment on."""
        self._halt_reason = reason
        self._halted_at = utc_now().isoformat()
        logger.critical("kill_switch_engaged", extra={"reason": reason, "mode": str(self.mode)})

    def release_kill_switch(self) -> None:
        logger.warning("kill_switch_released", extra={"previous_reason": self._halt_reason})
        self._halt_reason = None
        self._halted_at = None

    # ---------------------------------------------------------------- persistence
    async def persist(self, repo: BotStateRepository) -> None:
        """Persist the kill switch so it survives restarts (fail-closed)."""
        if self.is_halted:
            await repo.set(
                _KILL_SWITCH_KEY,
                {"engaged": True, "reason": self._halt_reason, "halted_at": self._halted_at},
            )
        else:
            await repo.delete(_KILL_SWITCH_KEY)

    async def restore(self, repo: BotStateRepository) -> None:
        """Restore an engaged kill switch from storage (startup)."""
        value = await repo.get(_KILL_SWITCH_KEY)
        if isinstance(value, dict) and value.get("engaged"):
            self._halt_reason = str(value.get("reason") or "engaged before restart")
            self._halted_at = value.get("halted_at")
            logger.warning(
                "kill_switch_restored",
                extra={"reason": self._halt_reason, "halted_at": self._halted_at},
            )

    def status(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode),
            "trading_enabled": str(self._trading_enabled).lower(),
            "sends_orders_to_exchange": str(self._policy.sends_orders_to_exchange).lower(),
            "halted": str(self.is_halted).lower(),
            "halt_reason": self._halt_reason or "",
            "halted_at": self._halted_at or "",
        }
