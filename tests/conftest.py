"""Shared fixtures: fast paper-mode services against a temporary database."""

from __future__ import annotations

import asyncio
import pathlib
from collections.abc import AsyncIterator

import pytest
from app.config.settings import Settings
from app.services import AppServices, build_app, shutdown_app, start_app


def make_settings(tmp: pathlib.Path, **overrides) -> Settings:
    base = Settings(_env_file=None)
    return base.model_copy(
        update={
            "database": base.database.model_copy(
                update={"url": f"sqlite+aiosqlite:///{tmp / 'bot.db'}"}
            ),
            # Small universe keeps the suite fast.
            "arbitrage": base.arbitrage.model_copy(
                update={"triangle_assets": ("BTC", "ETH", "SOL")}
            ),
            "transfer": base.transfer.model_copy(
                update={
                    # AVAX/DOGE carry cheap withdrawal fees in the simulation,
                    # so plans above the minimum actually exist for tests.
                    "assets": ("AVAX", "DOGE"),
                    "simulated_transfer_seconds": 0.1,
                    "poll_interval_seconds": 0.05,
                    "default_amount": __import__("decimal").Decimal("1"),
                }
            ),
            "market_data": base.market_data.model_copy(update={"streams_enabled": False}),
            # telegram tests chat id (also proves the allow-list wiring)
            "telegram": base.telegram.model_copy(update={"allowed_chat_ids": (12345,)}),
            **overrides,
        }
    )


@pytest.fixture()
def settings(tmp_path: pathlib.Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture()
async def services(tmp_path: pathlib.Path) -> AsyncIterator[AppServices]:
    app = await build_app(make_settings(tmp_path))
    await start_app(app)
    try:
        yield app
    finally:
        await shutdown_app(app)


@pytest.fixture()
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()
