"""Trading mode policy: what PAPER / DEMO / LIVE are allowed to do.

Keeping the policy separate from :class:`~app.config.settings.Settings` means the
execution layer never inspects raw configuration flags — it asks the policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from app.errors import ConfigurationError
from app.models.enums import TradingMode

__all__ = ["MODE_POLICIES", "ModePolicy", "policy_for"]


@dataclass(frozen=True, slots=True)
class ModePolicy:
    """Immutable capability matrix of a trading mode."""

    mode: TradingMode
    description: str
    sends_orders_to_exchange: bool
    uses_real_funds: bool
    uses_testnet: bool
    requires_credentials: bool
    requires_explicit_confirmation: bool

    @property
    def is_simulated(self) -> bool:
        return not self.sends_orders_to_exchange

    @property
    def is_live(self) -> bool:
        return self.mode is TradingMode.LIVE


MODE_POLICIES: Mapping[TradingMode, ModePolicy] = MappingProxyType(
    {
        TradingMode.PAPER: ModePolicy(
            mode=TradingMode.PAPER,
            description="Real market data, fully simulated execution. No exchange orders.",
            sends_orders_to_exchange=False,
            uses_real_funds=False,
            uses_testnet=False,
            requires_credentials=False,
            requires_explicit_confirmation=False,
        ),
        TradingMode.DEMO: ModePolicy(
            mode=TradingMode.DEMO,
            description="Exchange testnet/demo accounts. Real API calls, no real funds.",
            sends_orders_to_exchange=True,
            uses_real_funds=False,
            uses_testnet=True,
            requires_credentials=True,
            requires_explicit_confirmation=False,
        ),
        TradingMode.LIVE: ModePolicy(
            mode=TradingMode.LIVE,
            description="Production trading with real capital. Requires explicit opt-in.",
            sends_orders_to_exchange=True,
            uses_real_funds=True,
            uses_testnet=False,
            requires_credentials=True,
            requires_explicit_confirmation=True,
        ),
    }
)


def policy_for(mode: TradingMode) -> ModePolicy:
    try:
        return MODE_POLICIES[mode]
    except KeyError as exc:  # pragma: no cover - unreachable with the enum
        raise ConfigurationError(f"unknown trading mode: {mode!r}") from exc
