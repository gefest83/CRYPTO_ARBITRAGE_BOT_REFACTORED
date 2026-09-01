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
    """B-2/H-5: 'not found on venue' is UNCONFIRMED, never a proven rejection."""
    recovery = ExecutionRecovery()
    local = _order(OrderStatus.UNKNOWN, client_order_id="C8")
    resolved = await recovery.recover(local, FakeAdapter())
    assert "manual review" in (resolved.error or "")
    assert resolved.status is OrderStatus.MANUAL_REVIEW


async def test_venue_unreachable_escalates_to_manual_review():
    recovery = ExecutionRecovery()
    local = _order(OrderStatus.TIMEOUT, client_order_id="C9")
    resolved = await recovery.recover(local, FakeAdapter(fail=True))
    assert "manual review" in (resolved.error or "")
    assert resolved.status is OrderStatus.MANUAL_REVIEW


async def test_timeout_unresolved_never_rejected():
    """B-2: an unreachable venue must not be reported as a proven rejection."""
    recovery = ExecutionRecovery()
    local = _order(OrderStatus.TIMEOUT, client_order_id="C10")
    resolved = await recovery.recover(local, FakeAdapter(fail=True))
    assert resolved.status is not OrderStatus.REJECTED
    assert resolved.status is OrderStatus.MANUAL_REVIEW
    assert resolved.filled_amount == D("0")


async def test_venue_confirms_not_filled_is_rejected():
    """CONFIRMED_NOT_FILLED: a venue terminal zero-fill status stays rejected."""
    recovery = ExecutionRecovery()
    local = _order(OrderStatus.TIMEOUT, exchange_order_id="E20")
    fetched = _order(
        OrderStatus.CANCELED, exchange_order_id="E20", filled_amount=D("0")
    )
    resolved = await recovery.recover(local, FakeAdapter(fetched=fetched))
    assert resolved.status is OrderStatus.CANCELED
    assert resolved.filled_amount == D("0")


async def test_venue_open_zero_fill_is_manual_review():
    """A live order with zero fill may still fill — never claim REJECTED."""
    recovery = ExecutionRecovery()
    local = _order(OrderStatus.TIMEOUT, exchange_order_id="E21")
    fetched = _order(OrderStatus.OPEN, exchange_order_id="E21", filled_amount=D("0"))
    resolved = await recovery.recover(local, FakeAdapter(fetched=fetched))
    assert resolved.status is OrderStatus.MANUAL_REVIEW


async def test_pending_intent_resolved_from_open_orders():
    """H-2/H-3: a persisted pre-submission intent is resolvable by client id."""
    recovery = ExecutionRecovery()
    local = _order(OrderStatus.PENDING, client_order_id="C11", filled_amount=D("0"))
    matched = _order(
        OrderStatus.FILLED,
        exchange_order_id="E22",
        client_order_id="C11",
        filled_amount=D("1"),
        average_price=D("2000"),
    )
    resolved = await recovery.recover(local, FakeAdapter(open_orders=(matched,)))
    assert resolved.status is OrderStatus.FILLED
    assert resolved.filled_amount == D("1")
    assert resolved.exchange_order_id == "E22"


async def test_pending_intent_not_found_is_manual_review():
    """A pending intent that cannot be found is UNCONFIRMED — manual review."""
    recovery = ExecutionRecovery()
    local = _order(OrderStatus.PENDING, client_order_id="C12")
    resolved = await recovery.recover(local, FakeAdapter())
    assert resolved.status is OrderStatus.MANUAL_REVIEW


async def test_timeout_recovered_to_filled_is_a_real_fill():
    """B-1 prerequisite: recovery returns FILLED with the venue's fill data."""
    recovery = ExecutionRecovery()
    local = _order(
        OrderStatus.TIMEOUT, exchange_order_id="E23", amount=D("2")
    )
    fetched = _order(
        OrderStatus.FILLED,
        exchange_order_id="E23",
        filled_amount=D("2"),
        average_price=D("2001"),
    )
    resolved = await recovery.recover(local, FakeAdapter(fetched=fetched))
    assert resolved.status is OrderStatus.FILLED
    assert resolved.filled_amount == D("2")
    assert resolved.average_price == D("2001")


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


async def test_canceled_with_partial_fill_keeps_partial_amount():
    """A venue terminal status with a partial fill keeps the filled amount."""
    recovery = ExecutionRecovery()
    local = _order(OrderStatus.TIMEOUT, exchange_order_id="E24", amount=D("2"))
    fetched = _order(
        OrderStatus.CANCELED,
        exchange_order_id="E24",
        filled_amount=D("0.5"),
        average_price=D("2000"),
    )
    resolved = await recovery.recover(local, FakeAdapter(fetched=fetched))
    assert resolved.status is OrderStatus.CANCELED
    assert resolved.filled_amount == D("0.5")
