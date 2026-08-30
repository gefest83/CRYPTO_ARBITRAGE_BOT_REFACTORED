"""Transfer arbitrage models: plan (math) and record (persisted lifecycle).

A transfer arbitrage is a long-running sequential workflow::

    source exchange          blockchain network           destination exchange
    BUY asset       --->     WITHDRAW + TRANSFER   --->    DEPOSIT detected
                                                             SELL asset

Unlike triangular arbitrage this is *not* latency sensitive; correctness comes
from persisting every step so the bot can be restarted mid-transfer without
losing state.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, computed_field, field_validator

from app.models.base import DEC0, DomainModel, utc_now
from app.models.enums import TransferState

__all__ = [
    "DepositAddress",
    "TransferPlan",
    "TransferRecord",
    "TransferTx",
    "WithdrawalNetwork",
]

_BPS = Decimal("10000")


class DepositAddress(DomainModel):
    """Deposit address of one asset on one venue, with optional memo/tag."""

    address: str
    #: Memo/tag/destination tag (XRP, XLM, EOS ...).  Withdrawals to an
    #: address that requires a memo without sending it lose funds.
    memo: str | None = None
    network: str | None = None


class WithdrawalNetwork(DomainModel):
    """One blockchain network of an asset on one venue (ccxt currency data)."""

    network: str
    #: Unified network code, e.g. ``ETH`` for ``ERC20`` (used to match the
    #: same blockchain across venues).
    network_code: str | None = None
    withdraw_enabled: bool = True
    deposit_enabled: bool = True
    #: Withdrawal fee, denominated in the transferred asset.
    withdrawal_fee: Decimal = DEC0
    #: Minimum withdrawable amount (asset).
    withdrawal_min: Decimal = DEC0
    #: Minimum deposit amount (asset).
    deposit_min: Decimal = DEC0
    #: Typical confirmation time estimate, if the venue provides one.
    ETA_minutes: int | None = None

    @field_validator("network", "network_code", mode="before")
    @classmethod
    def _normalise(cls, value: str | None) -> str | None:
        return str(value).strip().upper() if value else None


class TransferTx(DomainModel):
    """One deposit or withdrawal as reported by a venue."""

    direction: Literal["deposit", "withdrawal"]
    asset: str
    network: str | None = None
    amount: Decimal = DEC0
    fee: Decimal = DEC0
    #: Venue-side status text ("pending", "confirmed", "failed", ...).
    status: str = "pending"
    txid: str | None = None
    address: str | None = None
    timestamp: datetime | None = None

    @field_validator("asset", mode="before")
    @classmethod
    def _upper(cls, value: str) -> str:
        return str(value).strip().upper()

    @property
    def is_confirmed(self) -> bool:
        return self.status.lower() in ("confirmed", "success", "ok", "complete")


class TransferPlan(DomainModel):
    """Fully evaluated transfer arbitrage economics (all Decimal, quote = USDT).

    ``buy_price``/``sell_price`` are the *executable* prices: ask on the source
    venue (we buy) and bid on the destination venue (we sell), depth-derived
    where possible.
    """

    source_exchange: str
    dest_exchange: str
    asset: str
    network: str
    amount: Decimal
    buy_price: Decimal
    sell_price: Decimal
    buy_fee_bps: Decimal = Decimal("10")
    sell_fee_bps: Decimal = Decimal("10")
    #: Withdrawal fee, denominated in the transferred asset.
    withdrawal_fee: Decimal = DEC0
    #: Additional network cost estimate in quote currency (gas etc.), when the
    #: adapter can provide it.
    network_cost_quote: Decimal = DEC0
    #: Estimated execution slippage (bps) already included via depth pricing;
    #: kept separately for reporting.
    estimated_slippage_bps: Decimal = DEC0

    @property
    def buy_cost_quote(self) -> Decimal:
        """Quote spent on the buy leg (price x amount, before fees)."""
        return self.buy_price * self.amount

    @property
    def sell_proceeds_quote(self) -> Decimal:
        """Quote received on the sell leg (price x amount, before fees)."""
        return self.sell_price * self.amount

    @property
    def trading_fees_quote(self) -> Decimal:
        buy_fee = self.buy_cost_quote * self.buy_fee_bps / _BPS
        sell_fee = self.sell_proceeds_quote * self.sell_fee_bps / _BPS
        return buy_fee + sell_fee

    @property
    def withdrawal_cost_quote(self) -> Decimal:
        """Withdrawal fee valued at the sell-side price plus network costs."""
        return self.withdrawal_fee * self.sell_price + self.network_cost_quote

    @property
    def total_fees_quote(self) -> Decimal:
        return self.trading_fees_quote + self.withdrawal_cost_quote

    @property
    def gross_profit_quote(self) -> Decimal:
        return self.sell_proceeds_quote - self.buy_cost_quote

    @property
    def net_profit_quote(self) -> Decimal:
        return self.gross_profit_quote - self.total_fees_quote

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_profit_bps(self) -> Decimal:
        if self.buy_cost_quote <= DEC0:
            return DEC0
        return self.net_profit_quote / self.buy_cost_quote * _BPS

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread_bps(self) -> Decimal:
        """Raw price spread between venues (before all costs)."""
        if self.buy_price <= DEC0:
            return DEC0
        return (self.sell_price - self.buy_price) / self.buy_price * _BPS


class TransferRecord(DomainModel):
    """Persisted lifecycle state of one transfer arbitrage execution."""

    id: str = Field(default_factory=lambda: f"trf-{uuid.uuid4().hex[:12]}")
    source_exchange: str
    dest_exchange: str
    asset: str
    network: str
    amount: Decimal
    state: TransferState = TransferState.CREATED
    plan: TransferPlan

    # --- buy leg ---
    #: Serialised :class:`~app.models.order.Order` of the buy leg, if any.
    buy_order: dict[str, Any] | None = None
    buy_filled_amount: Decimal = DEC0

    # --- withdrawal / transfer ---
    withdrawal_id: str | None = None
    withdrawal_txid: str | None = None
    #: Amount actually sent after the withdrawal fee was deducted.
    withdrawal_amount: Decimal = DEC0

    # --- deposit ---
    deposit_address: str | None = None
    deposit_txid: str | None = None
    #: Amount credited on the destination venue.
    deposit_amount: Decimal = DEC0

    # --- sell leg ---
    sell_order: dict[str, Any] | None = None
    sell_filled_amount: Decimal = DEC0
    sell_proceeds_quote: Decimal = DEC0

    # --- outcome ---
    fees_quote: Decimal = DEC0
    realized_profit_quote: Decimal = DEC0
    error: str | None = None

    mode: str = "PAPER"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal

    def with_state(
        self,
        state: TransferState,
        *,
        error: str | None = None,
        **updates: Any,
    ) -> TransferRecord:
        """Immutable transition helper; stamps ``updated_at``."""
        data = dict(updates)
        data["state"] = state
        data["updated_at"] = utc_now()
        if error is not None:
            data["error"] = error
        return self.model_copy(update=data)
