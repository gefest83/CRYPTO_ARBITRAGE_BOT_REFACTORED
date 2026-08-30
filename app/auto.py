"""Auto-trading loop.

Runs inside ``start_auto`` (CLI) and the Telegram bot process:

1. stop when the kill switch engages or the auto flag is cleared;
2. keep market data fresh;
3. scan for triangle cycles, risk-validate and execute the best ones
   (bounded by ``auto_max_per_cycle``);
4. advance open transfer workflows;
5. cool down and repeat.

The loop observes the persisted auto flag and kill switch, so ``stop_auto``
from another terminal (or Telegram) stops it immediately.
"""

from __future__ import annotations

import asyncio
import contextlib

from app.config.logging_config import get_logger
from app.services import AppServices

__all__ = ["AutoTrader"]

logger = get_logger("auto")


class AutoTrader:
    def __init__(self, services: AppServices) -> None:
        self._services = services
        self._stop_event = asyncio.Event()

    def stop(self) -> None:
        self._stop_event.set()

    async def run_forever(self) -> None:
        services = self._services
        settings = services.settings
        logger.info(
            "auto_trader_started",
            extra={
                "mode": str(settings.mode),
                "interval": settings.execution.auto_interval_seconds,
            },
        )
        try:
            while not self._stop_event.is_set():
                if not await services.auto_trading_enabled() or services.guard.is_halted:
                    logger.info("auto_trader_stopping", extra={"reason": "disabled or halted"})
                    break
                try:
                    await self._cycle()
                except Exception as exc:  # noqa: BLE001 - the loop must survive
                    logger.error("auto_cycle_error", extra={"error": str(exc)[:300]})
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=settings.execution.auto_interval_seconds,
                    )
        finally:
            services.guard.enable_trading(enabled=False)
            logger.info("auto_trader_stopped")

    async def _cycle(self) -> None:
        services = self._services
        services.guard.ensure_auto_trading()
        opportunities = await services.scan_triangles(
            notional_quote=services.settings.execution.auto_notional_quote
        )
        for executed, opportunity in enumerate(opportunities):
            if executed >= services.settings.execution.auto_max_per_cycle:
                break
            if services.guard.is_halted:
                break
            trade, assessment = await services.execute_triangle(opportunity)
            logger.info(
                "auto_triangle_executed",
                extra={
                    "trade_id": trade.id,
                    "status": trade.status.value,
                    "net_profit": str(trade.net_profit),
                    "violations": len(assessment.violations) if assessment else 0,
                },
            )
        await services.tick_transfers()
