"""DEMO fail-closed behaviour of the ccxt adapter (duck-typed client, no network)."""

from decimal import Decimal

import pytest
from app.errors import ExchangeUnavailableError, ExecutionDisabledError
from app.exchanges.ccxt_adapter import CCXTAdapter
from app.models.enums import OrderSide, OrderType
from app.models.exchange import Exchange
from app.models.order import OrderRequest
from app.models.symbol import Symbol


class DuckClient:
    """Duck-typed ccxt client for offline tests."""

    def __init__(self):
        self.urls = {
            "api": {"public": "https://api.example.com", "private": "https://api.example.com"}
        }
        self.has = {"fetchTicker": True, "fetchOrderBook": True, "createOrder": True}
        self.headers = {}
        self.options = {}
        self.calls = []
        self.sandbox_mode = None

    def set_sandbox_mode(self, enabled: bool) -> None:
        self.sandbox_mode = enabled

    async def fetch_ticker(self, symbol):
        self.calls.append(("fetch_ticker", symbol))
        return {"symbol": symbol, "bid": 100, "ask": 101, "last": 100.5, "timestamp": None}

    async def create_order(self, *args):
        self.calls.append(("create_order", args))
        return {
            "id": "E1",
            "status": "closed",
            "side": "buy",
            "type": "market",
            "amount": 1,
            "filled": 1,
            "price": 100,
        }


def _adapter(client: DuckClient, *, sandbox: bool, gate=None) -> CCXTAdapter:
    from app.exchanges.base import AdapterOptions

    exchange = Exchange(id="okx", name="OKX")
    options = AdapterOptions(sandbox=sandbox, timeout_seconds=5.0)
    return CCXTAdapter(exchange, options=options, order_gate=gate, client=client)


async def test_demo_mode_without_sandbox_support_fails_closed():
    """A venue without a testnet must never accept private calls in DEMO."""
    # gate: a venue with no sandbox flag cannot place orders
    from app.execution.order_gate import sandbox_only

    adapter = _adapter(DuckClient(), sandbox=False, gate=sandbox_only)
    await adapter.open()
    request = OrderRequest(
        exchange_id="okx",
        symbol=Symbol.parse("ETH/USDT"),
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("1"),
    )
    with pytest.raises(ExecutionDisabledError):
        await adapter.create_order(request)


async def test_market_data_still_flows_without_private_calls():
    client = DuckClient()
    adapter = _adapter(client, sandbox=False)
    await adapter.open()
    ticker = await adapter.fetch_ticker(Symbol.parse("ETH/USDT"))
    assert str(ticker.bid) == "100"
    assert str(ticker.ask) == "101"
    assert ("fetch_ticker", "ETH/USDT") in client.calls


async def test_sandbox_routing_okx_sets_demo_header():
    client = DuckClient()
    adapter = _adapter(client, sandbox=True)
    await adapter.open()  # okx demo routing: x-simulated-trading header
    assert client.headers.get("x-simulated-trading") == "1"


async def test_sandbox_routing_bybit_requires_url_table():
    client = DuckClient()
    exchange = Exchange(id="bybit", name="Bybit")
    from app.exchanges.base import AdapterOptions

    adapter = CCXTAdapter(exchange, options=AdapterOptions(sandbox=True), client=client)
    # bybit is demo_private_only: every non-public URL section must be
    # overridable to the demo host or the adapter refuses to start.
    client.urls = {"api": {"public": "https://api.bybit.com", "private": "https://api.bybit.com"}}
    await adapter.open()
    assert client.urls["api"]["private"] == "https://api-demo.bybit.com"
    assert client.urls["api"]["public"] == "https://api.bybit.com"  # public stays

    # an unexpected URL table must fail CLOSED (never route demo keys to prod)
    broken = DuckClient()
    broken.urls = {"api": "https://api.bybit.com"}
    adapter2 = CCXTAdapter(exchange, options=AdapterOptions(sandbox=True), client=broken)
    with pytest.raises(ExchangeUnavailableError):
        await adapter2.open()


async def test_withdrawals_disabled_by_default():
    client = DuckClient()
    adapter = _adapter(client, sandbox=True)
    await adapter.open()
    with pytest.raises(ExecutionDisabledError):
        await adapter.withdraw("SOL", "5", "addr", network="SOL")
