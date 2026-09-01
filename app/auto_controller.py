"""AutoTradingController: single owner of the AutoTrader background loop.

The controller is the ONLY entry point that starts or stops the background
strategy loop.  Both the CLI ``start_auto`` / ``stop_auto`` commands and the
Telegram ``/start_trading`` / ``/stop_trading`` commands go through it — there
is no other path to the :class:`AutoTrader`.

Design rules
------------

* The persisted auto flag (see :meth:`AppServices.set_auto_trading`) is the
  single source of truth for "should the loop be running?".  The controller
  observes it and keeps the in-process task aligned.
* :meth:`start` is idempotent — a second call when the loop is already
  running is a no-op (no second task is ever created).
* :meth:`stop` is idempotent — stopping when nothing is running is a no-op.
* :meth:`stop` does NOT touch the kill switch (unless the caller explicitly
  asked for it via :meth:`AppServices.engage_kill_switch`).
* :meth:`shutdown` is called by :func:`app.services.shutdown_app` and is the
  only path that may cancel the task in an orderly way during process exit.
* The loop is DEMO/PAPER neutral: the :class:`AutoTrader` decides what to
  actually do.  The controller does not place orders directly — it never
  calls :meth:`create_order` or :meth:`withdraw`.  All execution flows
  through :meth:`AppServices.execute_triangle`, which itself goes through the
  :class:`RiskEngine` and the :class:`ExecutionGuard`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.config.logging_config import get_logger

if TYPE_CHECKING:
    from app.services import AppServices

__all__ = ["AutoTradingController"]

logger = get_logger("auto.controller")


class AutoTradingController:
    """Owns the :class:`AutoTrader` background task for one :class:`AppServices`."""

    def __init__(self, services: AppServices) -> None:
        self._services = services
        self._task: asyncio.Task | None = None
        self._trader = None  # type: ignore[assignment]
        # Serialises concurrent calls to ``start`` so two handlers racing for
        # the loop cannot both observe ``_task is None`` and create a second
        # task.  Created lazily on first use (an asyncio.Lock must be bound
        # to a running loop, which may not exist at __init__ time during
        # ``build_app``).
        self._start_lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        return self._start_lock

    # ------------------------------------------------------------ status
    @property
    def running(self) -> bool:
        """Whether the in-process loop task is currently alive."""
        return self._task is not None and not self._task.done()

    async def is_running(self) -> bool:
        """The loop is "running" when the task is alive AND the persisted flag is on.

        Side-effect: if the previous task crashed (so the task reference is
        finished but the flag is still True), the persisted flag is cleared
        here too so ``/status`` and the flag stay in sync.  The clear is
        idempotent and safe to call from any status / read-only path.
        """
        if not self.running:
            # If the previous task crashed, clean up the stale flag so the
            # operator-visible state is honest.
            if self._task is not None and self._task.done():
                if (
                    not self._task.cancelled()
                    and self._task.exception() is not None
                ):
                    logger.error(
                        "auto_trader_crashed_detected_on_status",
                        extra={
                            "error": str(self._task.exception())[:300],
                        },
                    )
                self._task = None
                self._trader = None
                await self._services.set_auto_trading(False)
            return False
        return await self._services.auto_trading_enabled()

    # ------------------------------------------------------------ control
    async def start(self) -> tuple[bool, str]:
        """Start the loop.

        Returns ``(started, message)``.  When ``started`` is ``False`` the
        caller must surface ``message`` to the operator; nothing was changed.

        Concurrency: protected by ``_start_lock`` so two concurrent callers
        cannot both observe ``running is False`` and create two tasks.
        """
        services = self._services
        async with self._get_lock():
            if services.guard.is_halted:
                return False, (
                    f"kill switch engaged ({services.guard.halt_reason}); "
                    "release it first via /resume"
                )
            # Idempotency: a second start with the loop alive is a no-op.
            if self.running:
                return False, "auto trading already running"
            # A previous task may have crashed (exception) or been cleanly
            # stopped: in either case the in-process reference is left and the
            # persisted auto flag may be stale.  Detect and clear before
            # spawning a fresh task so ``auto_trading_enabled()`` reflects
            # reality.
            if self._task is not None and self._task.done():
                if not self._task.cancelled() and self._task.exception() is not None:
                    logger.error(
                        "auto_trader_crashed",
                        extra={
                            "error": str(self._task.exception())[:300],
                        },
                    )
                # Drop the stale references and reset the persisted flag so
                # ``is_running()`` and ``/status`` agree.
                self._task = None
                self._trader = None
                await services.set_auto_trading(False)
            # Persist the flag FIRST so a crash between flag-set and task-spawn
            # still leaves the operator-visible state consistent.
            await services.set_auto_trading(True)
            from app.auto import AutoTrader

            self._trader = AutoTrader(services)
            self._task = asyncio.create_task(
                self._trader.run_forever(), name="auto-trader"
            )
            logger.info(
                "auto_controller_started",
                extra={"mode": str(services.settings.mode)},
            )
            return True, "auto trading started"

    async def stop(self) -> tuple[bool, str]:
        """Stop the loop without touching the kill switch.

        Returns ``(stopped, message)``.  The persisted auto flag is cleared
        so the loop will not be auto-restarted by ``start_app``.

        Always cleans up any stale task reference and the persisted flag,
        even when the loop has already exited on its own (e.g. kill switch
        flipped the flag and the AutoTrader broke out before ``stop`` was
        called).  Idempotent: ``stopped=False`` and a clear message are
        returned when nothing was running to begin with.
        """
        services = self._services
        was_running = self.running or await services.auto_trading_enabled()
        await services.set_auto_trading(False)
        # If the loop is parked in ``_stop_event.wait()`` or scanning for the
        # next opportunity, telling it to stop lets it exit within one
        # ``auto_interval_seconds`` window.
        if self._trader is not None:
            self._trader.stop()
        task = self._task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await asyncio.shield(task)
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001 - shutdown must stay quiet
                pass
        # Always clear references so a future ``start`` is a fresh task and
        # ``running`` correctly reports False.
        self._task = None
        self._trader = None
        logger.info("auto_controller_stopped")
        if was_running:
            return True, "auto trading stopped"
        return False, "auto trading already stopped"

    # ------------------------------------------------------------ lifecycle
    async def shutdown(self) -> None:
        """Called by :func:`app.services.shutdown_app`.

        Idempotent.  Does not raise.  Does not flip the kill switch.
        """
        try:
            await self.stop()
        except Exception as exc:  # noqa: BLE001 - shutdown must stay quiet
            logger.warning(
                "auto_controller_shutdown_failed",
                extra={"error": str(exc)[:200]},
            )