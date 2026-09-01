"""Stream supervision: semaphore cancellation must never leak permits (C-5).

A stream acquires the GLOBAL concurrency permit and then its exchange permit.
If the task is cancelled between the two acquisitions (or while waiting for
the second one), every permit already taken must be released — otherwise each
cancelled stream permanently reduces the global stream budget.
"""

from __future__ import annotations

import asyncio

from app.market_data.streams import StreamSpec, StreamSupervisor
from app.models.enums import StreamKind

SPEC_A = StreamSpec("binance", "ETH/USDT", StreamKind.ORDER_BOOK)
SPEC_B = StreamSpec("binance", "BTC/USDT", StreamKind.ORDER_BOOK)
SPEC_C = StreamSpec("okx", "ETH/USDT", StreamKind.ORDER_BOOK)


def _feed(tag: str):
    """Iterator factory yielding ``tag`` forever (holds its permits)."""

    async def _gen():
        while True:
            yield tag
            await asyncio.sleep(0.005)

    return _gen


async def _wait_for_tag(received: list[str], tag: str, timeout: float = 2.0) -> bool:
    for _ in range(int(timeout / 0.01)):
        if tag in received:
            return True
        await asyncio.sleep(0.01)
    return False


async def test_cancel_during_exchange_acquire_releases_global_permit():
    """Cancellation while waiting on the exchange semaphore must release the
    global permit that was already acquired."""
    supervisor = StreamSupervisor(max_parallel_streams=2, max_streams_per_exchange=1)
    received: list[str] = []
    try:
        # A holds binance(1/1) + global(1/2) and keeps streaming.
        await supervisor.start_stream(SPEC_A, _feed("a"), received.append)
        assert await _wait_for_tag(received, "a")

        # B (same exchange) acquires the LAST global permit and then blocks
        # on the binance semaphore forever.
        await supervisor.start_stream(SPEC_B, _feed("b"), received.append)
        await asyncio.sleep(0.05)
        assert "b" not in received

        # Cancel B while it waits on the exchange permit.
        await supervisor.stop_stream(SPEC_B)

        # The freed global permit must now be available: C (a different
        # exchange) acquires global + okx permits and delivers.  If B had
        # leaked the global permit, C would starve (2/2 held forever).
        await supervisor.start_stream(SPEC_C, _feed("c"), received.append)
        assert await _wait_for_tag(received, "c"), (
            "global semaphore permit leaked by cancellation during "
            "exchange-semaphore acquisition"
        )
    finally:
        await supervisor.stop_all()


async def test_cancel_during_global_acquire_keeps_semaphore_usable():
    """Cancellation while waiting on the depleted global semaphore must not
    consume (or otherwise corrupt) the permit budget."""
    supervisor = StreamSupervisor(max_parallel_streams=1, max_streams_per_exchange=4)
    received: list[str] = []
    try:
        # A holds the only global permit.
        await supervisor.start_stream(SPEC_A, _feed("a"), received.append)
        assert await _wait_for_tag(received, "a")

        # B blocks on the global semaphore (never acquires it).
        await supervisor.start_stream(SPEC_B, _feed("b"), received.append)
        await asyncio.sleep(0.05)
        assert "b" not in received

        # Cancel B mid-acquisition, then release A's permit.
        await supervisor.stop_stream(SPEC_B)
        await supervisor.stop_stream(SPEC_A)

        # C must be able to acquire the single global permit and deliver.
        await supervisor.start_stream(SPEC_C, _feed("c"), received.append)
        assert await _wait_for_tag(received, "c"), (
            "global semaphore budget corrupted by cancellation during "
            "global acquisition"
        )
    finally:
        await supervisor.stop_all()


async def test_cancelled_stream_releases_both_permits_it_held():
    """A stream cancelled while actively streaming releases global AND
    exchange permits (no double release, no leak)."""
    supervisor = StreamSupervisor(max_parallel_streams=1, max_streams_per_exchange=1)
    received: list[str] = []
    try:
        await supervisor.start_stream(SPEC_A, _feed("a"), received.append)
        assert await _wait_for_tag(received, "a")
        await supervisor.stop_stream(SPEC_A)
        # Both permits are back: a stream on the SAME exchange and symbol
        # slot can start again and deliver.
        await supervisor.start_stream(SPEC_A, _feed("a2"), received.append)
        assert await _wait_for_tag(received, "a2")
    finally:
        await supervisor.stop_all()
