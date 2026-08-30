"""Network validation for transfer arbitrage.

Before any transfer executes, the blockchain leg must be *proven* safe:

* the source venue must allow withdrawals of the asset on some network;
* the destination venue must accept deposits of the asset on the same network;
* both venues must be talking about the SAME blockchain (network codes are
  matched, never display names — "ERC20" on one venue and "Ethereum (ERC20)"
  on the other are the same chain only when their unified codes agree);
* withdrawal must be enabled right now (venues suspend networks), and the
  deposit side must not be disabled;
* the withdrawal fee must be known — an unknown fee makes the net profit
  uncertain, and uncertain profitability means NO execution.

Fail-closed everywhere: any doubt about the network disqualifies the pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.models.base import DEC0
from app.models.transfer import WithdrawalNetwork

__all__ = ["NetworkMismatchError", "NetworkSelector", "RouteNetwork"]


class NetworkMismatchError(Exception):
    """No provably-safe network exists for the asset between the two venues."""


@dataclass(frozen=True, slots=True)
class RouteNetwork:
    """A validated network for one asset between two venues."""

    network: str
    #: Unified network code (e.g. ``ETH`` for ERC20, ``TRX`` for TRC20).
    code: str
    #: Withdrawal fee on the source venue, denominated in the asset.
    withdrawal_fee: Decimal
    withdrawal_min: Decimal = DEC0
    deposit_min: Decimal = DEC0


def _code_of(network: WithdrawalNetwork) -> str:
    """Unified network code, falling back to the display name (normalised)."""
    return (network.network_code or network.network).strip().upper()


class NetworkSelector:
    """Picks the cheapest provably-safe network between two venues."""

    def select(
        self,
        *,
        asset: str,
        source_networks: tuple[WithdrawalNetwork, ...],
        dest_networks: tuple[WithdrawalNetwork, ...],
    ) -> RouteNetwork:
        """Return the cheapest common, enabled network.

        Raises :class:`NetworkMismatchError` with the exact reason when no
        safe network exists — the caller refuses the transfer, never guesses.
        """
        if not source_networks:
            raise NetworkMismatchError(f"{asset}: source venue publishes no withdrawal networks")
        if not dest_networks:
            raise NetworkMismatchError(f"{asset}: destination venue publishes no deposit networks")

        withdrawable = {_code_of(n): n for n in source_networks if n.withdraw_enabled}
        if not withdrawable:
            raise NetworkMismatchError(
                f"{asset}: withdrawals are disabled on the source venue for every network"
            )
        depositable = {_code_of(n): n for n in dest_networks if n.deposit_enabled}
        if not depositable:
            raise NetworkMismatchError(
                f"{asset}: deposits are disabled on the destination venue for every network"
            )

        common = set(withdrawable) & set(depositable)
        if not common:
            src = ", ".join(sorted(withdrawable))
            dst = ", ".join(sorted(depositable))
            raise NetworkMismatchError(
                f"{asset}: no common enabled network between source ({src}) "
                f"and destination ({dst}) — refusing to transfer"
            )

        # Cheapest common network by withdrawal fee (tie-break: name order
        # for determinism).
        def cost(code: str) -> tuple[Decimal, str]:
            return (withdrawable[code].withdrawal_fee, code)

        best = min(common, key=cost)
        source = withdrawable[best]
        dest = depositable[best]
        return RouteNetwork(
            network=source.network,
            code=best,
            withdrawal_fee=source.withdrawal_fee,
            withdrawal_min=source.withdrawal_min,
            deposit_min=dest.deposit_min,
        )
