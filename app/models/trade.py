"""Executed-trade records (persisted via app.storage)."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import Field

from app.models.base import DEC0, DomainModel, utc_now
from app.models.enums import ArbitrageStrategy, TradeStatus, TradingMode

__all__ = ["TradeRecord"]


class TradeRecord(DomainModel):
    """One executed (or attempted) arbitrage trade, strategy-agnostic.

    * triangle: one venue, three legs; ``input_amount``/``output_amount`` are
      quote currency (USDT in / USDT out).
    * transfer: recorded on completion of the whole workflow by the
      orchestrator (the transfer table owns the lifecycle).
    """

    id: str = Field(default_factory=lambda: f"trd-{uuid.uuid4().hex[:12]}")
    strategy: ArbitrageStrategy
    mode: TradingMode = TradingMode.PAPER
    exchange_id: str
    #: Ordered route description, e.g. ``USDT->BTC->ETH->USDT``.
    route: str = ""
    symbols: tuple[str, ...] = ()
    input_amount: Decimal = DEC0
    output_amount: Decimal = DEC0
    fees_quote: Decimal = DEC0
    slippage_bps: Decimal = DEC0
    net_profit: Decimal = DEC0
    net_profit_bps: Decimal = DEC0
    status: TradeStatus = TradeStatus.EXECUTING
    #: Serialised orders of every leg (list of Order model dumps).
    orders: tuple[dict[str, Any], ...] = ()
    error: str | None = None
    transfer_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            TradeStatus.COMPLETED,
            TradeStatus.FAILED,
            TradeStatus.MANUAL_REVIEW,
        )

    def with_status(self, status: TradeStatus, **updates: Any) -> TradeRecord:
        data = dict(updates)
        data["status"] = status
        data["updated_at"] = utc_now()
        return self.model_copy(update=data)
