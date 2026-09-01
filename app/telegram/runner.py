"""Background Telegram runner: owns the polling task, starts/stops cleanly.

Separating the runner from :mod:`app.telegram.bot` keeps the bot itself
focused on command dispatch — the runner exists purely to manage the
asyncio task lifecycle so :func:`app.services.start_app` and
:func:`app.services.shutdown_app` never leave orphan tasks behind.

Start is non-blocking: the runner spawns the polling task and returns
immediately.  ``stop`` cancels the task and waits for it (with a bounded
grace period) so application shutdown is deterministic.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from app.config.logging_config import get_logger

if TYPE_CHECKING:
    from app.services import AppServices
    from app.telegram.bot import TelegramBot
    from app.telegram.client import TelegramClient

__all__ = ["TelegramRunner"]

logger = get_logger("telegram.runner")

# Maximum time ``stop`` waits for the polling task to wind down.  Keeps
# application shutdown bounded even when the Bot API is unresponsive.
_STOP_GRACE_SECONDS = 5.0


class TelegramRunner:
    """Owns the lifecycle of the Telegram polling task.

    Construction never touches the network.  ``start`` builds the client and
    spawns the polling task; ``stop`` cancels and joins it.
    """

    def __init__(self, services: AppServices) -> None:
        self._services = services
        self._task: asyncio.Task | None = None
        self._client: TelegramClient | None = None
        self._bot: TelegramBot | None = None

    @property
    def started(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Build the client + bot, then spawn the polling task.  Non-blocking."""
        if self.started:
            return
        from app.telegram.bot import TelegramBot
        from app.telegram.client import TelegramClient

        cfg = self._services.settings.telegram
        client = TelegramClient(
            cfg.bot_token.get_secret_value(),
            poll_timeout_seconds=cfg.poll_timeout_seconds,
        )
        bot = TelegramBot(self._services, client)
        # ``get_me`` runs once before polling so a bad token is surfaced here
        # rather than silently failing on every poll iteration.  It does NOT
        # gate trading — a failure is reported and re-raised to the caller
        # which logs and continues without Telegram.
        try:
            me = await client.get_me()
        except Exception:
            await client.close()
            raise
        username = me.get("username") if isinstance(me, dict) else None
        logger.info("telegram_runner_started", extra={"username": username})
        self._client = client
        self._bot = bot
        self._task = asyncio.create_task(
            self._poll_loop(), name="telegram-poller"
        )

    async def _poll_loop(self) -> None:
        """The background polling loop.  Errors are caught; one bad poll never dies."""
        from app.telegram.client import poll_forever

        assert self._client is not None and self._bot is not None
        try:
            await poll_forever(self._client, self._bot.handle_update)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - bot must never propagate to the app
            logger.warning(
                "telegram_poll_loop_ended",
                extra={"error": str(exc)[:200]},
            )

    async def stop(self) -> None:
        """Cancel the polling task and close the client.  Idempotent."""
        task = self._task
        client: TelegramClient | None = self._client
        self._task = None
        self._client = None
        self._bot = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(task), timeout=_STOP_GRACE_SECONDS
                )
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as exc:  # noqa: BLE001 - shutdown must stay quiet
                logger.warning(
                    "telegram_task_join_failed",
                    extra={"error": str(exc)[:200]},
                )
        if client is not None:
            try:
                await client.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must stay quiet
                logger.warning(
                    "telegram_client_close_failed",
                    extra={"error": str(exc)[:200]},
                )
        if task is not None or client is not None:
            logger.info("telegram_runner_stopped")