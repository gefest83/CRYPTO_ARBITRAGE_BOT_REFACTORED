"""Execution recovery: partial fills, timeouts, unknown orders, escalation."""

from decimal import Decimal

from app.models.enums import OrderSide, OrderStatus, OrderType
from app.models.order import Order
from app.models.symbol import Symbol
from app.recovery import ExecutionRecovery

D = Decimal


class FakeAdapter:
    def __init__(self, fetched=None, open_orders=(), fail=False):
        self._fetched = fetched
        self._open = open_orders
        self._fail = fail

    async def fetch_order(self, order_id, *, symbol):
        if self._fail:
            raise RuntimeError("venue down")
        return self._fetched

    async def fetch_open_orders(self, *, symbol=None):
        if self._fail:
            raise RuntimeError("venue down")
        return self._open


def _order(status: OrderStatus, **kw) -> Order:
    params = {
        "exchange_id": "binance",
        "symbol": Symbol.parse("ETH/USDT"),
        "side": OrderSide.BUY,
        "order_type": OrderType.MARKET,
        "amount": D("1"),
        "status": status,
    }
    params.update(kw)
    return Order(**params)


async def test_partial_fill_resolved_from_venue():
    recovery = ExecutionRecovery()
    local = _order(OrderStatus.PARTIALLY_FILLED, exchange_order_id="E1", filled_amount=D("0.4"))
    fetched = _order(
        OrderStatus.FILLED,
        exchange_order_id="E1",
        filled_amount=D("1"),
        average_price=D("2000"),
    )
    resolved = await recovery.recover(local, FakeAdapter(fetched=fetched))
    assert resolved.status is OrderStatus.FILLED
    assert resolved.filled_amount == D("1")


async def test_partial_fill_stays_partial_when_venue_agrees():
    recovery = ExecutionRecovery()
    local = _order(OrderStatus.PARTIALLY_FILLED, exchange_order_id="E1", filled_amount=D("0.4"))
    fetched = _order(OrderStatus.PARTIALLY_FILLED, exchange_order_id="E1", filled_amount=D("0.6"))
    resolved = await recovery.recover(local, FakeAdapter(fetched=fetched))
    assert resolved.status is OrderStatus.PARTIALLY_FILLED
    assert resolved.filled_amount == D("0.6")


async def test_timeout_resolved_by_exchange_order_id():
    recovery = ExecutionRecovery()
    local = _order(OrderStatus.TIMEOUT, exchange_order_id="E9")
    fetched = _order(OrderStatus.FILLED, exchange_order_id="E9", filled_amount=D("1"))
    resolved = await recovery.recover(local, FakeAdapter(fetched=fetched))
    assert resolved.status is OrderStatus.FILLED
    assert resolved.filled_amount == D("1")


async def test_timeout_resolved_via_client_order_id_in_open_orders():
    recovery = ExecutionRecovery()
    local = _order(OrderStatus.TIMEOUT, client_order_id="C7")
    open_order = _order(
        OrderStatus.PARTIALLY_FILLED,
        exchange_order_id="E9",
        client_order_id="C7",
        filled_amount=D("0.5"),
    )
    resolved = await recovery.recover(local, FakeAdapter(open_orders=(open_order,)))
    assert resolved.status is OrderStatus.PARTIALLY_FILLED
    assert resolved.exchange_order_id == "E9"  # identity merged from the venue


async def test_unknown_order_not_found_escalates_to_manual_review():
    recovery = ExecutionRecovery()
    local = _order(OrderStatus.UNKNOWN, client_order_id="C8")
    resolved = await recovery.recover(local, FakeAdapter())
    assert "manual review" in (resolved.error or "")
    assert resolved.status is OrderStatus.REJECTED


async def test_venue_unreachable_escalates_to_manual_review():
    recovery = ExecutionRecovery()
    local = _order(OrderStatus.TIMEOUT, client_order_id="C9")
    resolved = await recovery.recover(local, FakeAdapter(fail=True))
    assert "manual review" in (resolved.error or "")


async def test_rejected_order_passes_through_unchanged():
    recovery = ExecutionRecovery()
    rejected = _order(OrderStatus.REJECTED, error="insufficient balance")
    assert await recovery.recover(rejected, FakeAdapter()) is rejected


async def test_venue_overfill_is_clamped():
    """A venue reporting more fill than ordered is never trusted verbatim."""
    recovery = ExecutionRecovery()
    local = _order(OrderStatus.PARTIALLY_FILLED, exchange_order_id="E1", filled_amount=D("0.4"))
    fetched = _order(OrderStatus.FILLED, exchange_order_id="E1", filled_amount=D("7"))
    resolved = await recovery.recover(local, FakeAdapter(fetched=fetched))
    assert resolved.filled_amount == D("1")
