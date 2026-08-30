"""Deterministic simulated exchange adapter (PAPER mode, zero network).

Purpose: run the whole bot — market data, triangular scanning, paper
execution, transfer planning — without network access or API keys.  Prices are
derived from a stable hash of ``exchange_id`` + symbol, so the three venues
show slightly different quotes: enough to exercise spread/depth logic,
triangular rings and transfer opportunities in development and tests.

Determinism guarantees:

* every pseudo-random value comes from :func:`_unit_offset`, a blake2b hash —
  never from ``random`` — so runs are reproducible across processes;
* venue divergence (``±48 bps`` default) creates cross-venue price spreads;
* cross pairs (``ETH/BTC``) carry a cyclical ``±80 bps`` skew vs their USDT
  legs so 3-leg rings do not multiply out to 1 and the triangular scanner has
  real (simulated) inefficiencies to find.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Mapping, Sequence
from decimal import Decimal

from app.models.balance import Balance, BalanceSnapshot
from app.models.base import DEC0, utc_now
from app.models.enums import ExchangeStatus, MarketType
from app.models.exchange import Exchange, ExchangeCapabilities, ExchangeHealth
from app.models.market import Market, MarketFees, MarketLimits, MarketPrecision
from app.models.market_data import OrderBook, OrderBookLevel, Ticker
from app.models.symbol import Symbol
from app.models.transfer import DepositAddress, TransferTx, WithdrawalNetwork

from .base import AdapterOptions, BaseExchangeAdapter, OrderGate
from .credentials import ExchangeCredentials

__all__ = [
    "DEFAULT_SEED_PRICES",
    "NETWORK_WITHDRAWAL_FEES",
    "SimulatedExchangeAdapter",
    "build_simulated_adapter",
]

#: Reference mid prices used as the simulation seed (USDT quoted).
DEFAULT_SEED_PRICES: Mapping[str, Decimal] = {
    "BTC/USDT": Decimal("116420"),
    "ETH/USDT": Decimal("4180"),
    "SOL/USDT": Decimal("214.5"),
    "XRP/USDT": Decimal("2.94"),
    "BNB/USDT": Decimal("1035"),
    "ADA/USDT": Decimal("0.92"),
    "DOGE/USDT": Decimal("0.24"),
    "LINK/USDT": Decimal("22.4"),
    "AVAX/USDT": Decimal("38.6"),
    "TRX/USDT": Decimal("0.31"),
}

#: Seed order of assets — cross pairs are listed in this order (ETH/BTC, not
#: BTC/ETH), mirroring how real venues list the more liquid base first.
_SEED_ASSETS: tuple[str, ...] = tuple(name.split("/")[0] for name in DEFAULT_SEED_PRICES)

_SIMULATED_CAPABILITIES = ExchangeCapabilities(
    spot=True,
    fetch_markets=True,
    fetch_ticker=True,
    fetch_order_book=True,
    fetch_trading_fees=True,
    fetch_balance=True,
    withdraw=True,
    deposit_address=True,
    fetch_deposits=True,
    fetch_withdrawals=True,
    fetch_withdrawal_networks=True,
    watch_ticker=True,
    watch_order_book=True,
)

#: Deterministic network table per asset: (display network, unified code, fee).
_NETWORKS: Mapping[str, tuple[tuple[str, str, str], ...]] = {
    "USDT": (
        ("TRC20", "TRX", "1.20"),
        ("ERC20", "ETH", "5.80"),
        ("BEP20", "BSC", "0.65"),
    ),
    "BTC": (("BTC", "BTC", "0.00021"),),
    "ETH": (("ERC20", "ETH", "0.0035"),),
    "SOL": (("SOL", "SOL", "0.010"),),
    "XRP": (("XRP", "XRP", "0.25"),),
    "BNB": (("BEP20", "BSC", "0.0008"),),
    "ADA": (("ADA", "ADA", "1.0"),),
    "DOGE": (("DOGE", "DOGE", "4.0"),),
    "LINK": (("ERC20", "ETH", "0.85"),),
    "AVAX": (("AVAX", "AVAX", "0.01"),),
    "TRX": (("TRC20", "TRX", "1.0"),),
}

#: Backwards-compatible withdrawal fee lookup keyed by display network.
NETWORK_WITHDRAWAL_FEES: dict[str, str] = {
    network: fee for networks in _NETWORKS.values() for network, _code, fee in networks
}

#: Venue divergence band in bps (price offsets seeded per venue+symbol).
#: Sized so cross-venue spreads realistically exceed transfer costs (trading
#: fees + withdrawal fee); intra-venue triangular rings are driven by the
#: cross-pair skew below and are independent of this value.
_DIVERGENCE_BPS = Decimal("120")
#: Half-spread of the synthetic book in bps.
_SPREAD_BPS = Decimal("4")
#: Cyclical cross-pair mispricing vs the USDT legs in bps.
_TRIANGLE_SKEW_BPS = Decimal("80")

#: Deterministic per-venue seed inventory: (base, spread) per asset.  Covers
#: every seeded asset so paper execution is fundable for the whole watchlist.
_SEED_INVENTORY: Mapping[str, tuple[Decimal, Decimal]] = {
    "USDT": (Decimal("5000"), Decimal("2000")),
    "BTC": (Decimal("0.05"), Decimal("0.02")),
    "ETH": (Decimal("2"), Decimal("1")),
    "SOL": (Decimal("50"), Decimal("20")),
    "BNB": (Decimal("2"), Decimal("1")),
    "XRP": (Decimal("600"), Decimal("300")),
    "ADA": (Decimal("2000"), Decimal("1000")),
    "DOGE": (Decimal("8000"), Decimal("4000")),
    "LINK": (Decimal("80"), Decimal("40")),
    "AVAX": (Decimal("45"), Decimal("20")),
    "TRX": (Decimal("6000"), Decimal("3000")),
}

_BALANCE_QUANTA: Mapping[str, Decimal] = {
    "USDT": Decimal("0.01"),
    "BTC": Decimal("0.00000001"),
    "ETH": Decimal("0.000001"),
    "SOL": Decimal("0.001"),
    "BNB": Decimal("0.0001"),
    "XRP": Decimal("0.01"),
    "ADA": Decimal("0.1"),
    "DOGE": Decimal("1"),
    "LINK": Decimal("0.0001"),
    "AVAX": Decimal("0.001"),
    "TRX": Decimal("1"),
}


def _unit_offset(*parts: str) -> Decimal:
    """Stable pseudo-random value in ``[-1, 1]`` (independent of PYTHONHASHSEED)."""
    digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / float(1 << 64)
    return Decimal(str(round(value * 2 - 1, 6)))


class SimulatedExchangeAdapter(BaseExchangeAdapter):
    """In-memory venue with deterministic quotes and balances."""

    adapter_name = "simulated"

    def __init__(
        self,
        exchange: Exchange,
        *,
        credentials: ExchangeCredentials | None = None,
        options: AdapterOptions | None = None,
        order_gate: OrderGate | None = None,
        seed_prices: Mapping[str, Decimal] | None = None,
        symbols: Sequence[str] | None = None,
        divergence_bps: Decimal = _DIVERGENCE_BPS,
        spread_bps: Decimal = _SPREAD_BPS,
    ) -> None:
        super().__init__(exchange, credentials=credentials, options=options, order_gate=order_gate)
        self._seed_prices = dict(seed_prices or DEFAULT_SEED_PRICES)
        self._symbols = tuple(symbols or self._seed_prices.keys())
        self._divergence_bps = divergence_bps
        self._spread_bps = spread_bps
        self._capabilities = _SIMULATED_CAPABILITIES.model_copy()
        self._opened = False
        #: Simulated on-chain transfers initiated through :meth:`withdraw`.
        self._withdrawals: list[TransferTx] = []
        self._deposits: list[TransferTx] = []

    # ---------------------------------------------------------------- lifecycle
    async def open(self) -> None:
        self._opened = True

    async def close(self) -> None:
        self._opened = False

    # ---------------------------------------------------------------- pricing
    def triangular_skew_bps(self, base: str, quote: str) -> Decimal:
        """Deterministic cyclical mispricing of ``base/quote`` vs the USDT legs.

        Seeded per *unordered* pair and oriented by the name order, so
        ``skew(Y, X) == -skew(X, Y)`` and cross mids stay mutually consistent
        while 3-cycles over distinct pairs do not cancel.
        """
        first, second = sorted((base, quote))
        offset = _unit_offset(self.id, "tri", first, second)
        magnitude = offset * _TRIANGLE_SKEW_BPS
        return magnitude if base <= quote else -magnitude

    def mid_price(self, symbol: Symbol) -> Decimal:
        """Deterministic price for a spot symbol.

        Cross pairs (``ETH/BTC``) are derived from the venue's own USDT mids
        times ``(1 + triangular_skew)``.  Symbols outside the seed table get a
        synthetic level derived from the symbol itself.
        """
        seed = self._seed_prices.get(symbol.name)
        if seed is None and symbol.quote != "USDT":
            leg_base = self.mid_price(Symbol(base=symbol.base, quote="USDT"))
            leg_quote = self.mid_price(Symbol(base=symbol.quote, quote="USDT"))
            skew = self.triangular_skew_bps(symbol.base, symbol.quote) / Decimal("10000")
            return (leg_base / leg_quote * (1 + skew)).quantize(Decimal("0.00000001"))
        if seed is None:
            seed = (
                Decimal("10") + abs(_unit_offset("seed", symbol.name)) * Decimal("990")
            ).quantize(Decimal("0.01"))
        offset_bps = _unit_offset(self.id, symbol.name) * self._divergence_bps
        return (seed * (1 + offset_bps / Decimal("10000"))).quantize(Decimal("0.00000001"))

    def _book_sides(
        self, symbol: Symbol, depth: int
    ) -> tuple[tuple[OrderBookLevel, ...], tuple[OrderBookLevel, ...]]:
        mid = self.mid_price(symbol)
        half_spread = mid * self._spread_bps / Decimal("20000")
        step = mid * Decimal("0.0001")
        # ~5k quote per level: deep enough to trade, shallow enough that a large
        # order really walks the book (so slippage math is exercised).
        base_amount = (Decimal("5000") / mid).quantize(Decimal("0.00000001"))
        bids = tuple(
            OrderBookLevel(
                price=(mid - half_spread - step * level).quantize(Decimal("0.00000001")),
                amount=(base_amount * (1 + Decimal(level) / 4)).quantize(Decimal("0.00000001")),
            )
            for level in range(depth)
        )
        asks = tuple(
            OrderBookLevel(
                price=(mid + half_spread + step * level).quantize(Decimal("0.00000001")),
                amount=(base_amount * (1 + Decimal(level) / 4)).quantize(Decimal("0.00000001")),
            )
            for level in range(depth)
        )
        return bids, asks

    # ---------------------------------------------------------------- market data
    @staticmethod
    def precision_for(symbol: Symbol) -> MarketPrecision:
        """Magnitude-appropriate tick/step per instrument.

        Price tick and amount step follow the price magnitude the way real
        venues configure them (BTC ticks at cents, DOGE amounts in whole
        units), so order rounding has realistic granularity.
        """
        seed = DEFAULT_SEED_PRICES.get(symbol.name)
        if seed is None and symbol.quote != "USDT":
            leg_base = DEFAULT_SEED_PRICES.get(f"{symbol.base}/USDT")
            leg_quote = DEFAULT_SEED_PRICES.get(f"{symbol.quote}/USDT")
            seed = leg_base / leg_quote if leg_base is not None and leg_quote is not None else None
        reference = abs(seed) if seed is not None else Decimal("100")
        if reference >= Decimal("1000"):
            price_decimals, amount_decimals = 2, 5
        elif reference >= Decimal("10"):
            price_decimals, amount_decimals = 2, 4
        elif reference >= Decimal("1"):
            price_decimals, amount_decimals = 4, 3
        elif reference >= Decimal("0.01"):
            price_decimals, amount_decimals = 6, 1
        else:
            price_decimals, amount_decimals = 8, 0
        return MarketPrecision(
            price=price_decimals,
            amount=amount_decimals,
            price_tick=Decimal(1).scaleb(-price_decimals),
            amount_step=Decimal(1).scaleb(-amount_decimals),
        )

    def _market(self, name: str) -> Market:
        symbol = Symbol.parse(name)
        native = symbol.name.replace("/", "")
        return Market(
            exchange_id=self.id,
            symbol=symbol,
            market_type=MarketType.SPOT,
            native_symbol=native,
            active=True,
            fees=MarketFees(maker_bps=Decimal("8"), taker_bps=Decimal("10")),
            limits=MarketLimits(min_amount=Decimal("0.0001"), min_cost=Decimal("5")),
            precision=self.precision_for(symbol),
        )

    async def load_markets(self) -> tuple[Market, ...]:
        """USDT pairs plus one canonical cross pair per asset pair.

        Cross pairs follow the real-venue convention — the more liquid asset
        is the quote: ``ETH/BTC``, ``SOL/BTC``, ``SOL/ETH``, ...
        """
        names = list(self._symbols)
        assets = [name.split("/")[0] for name in self._symbols if name.endswith("/USDT")]
        for i, quote_asset in enumerate(assets):
            for base_asset in assets[i + 1 :]:
                names.append(f"{base_asset}/{quote_asset}")
        return tuple(self._market(name) for name in names)

    async def fetch_ticker(self, symbol: Symbol) -> Ticker:
        bids, asks = self._book_sides(symbol, 1)
        mid = self.mid_price(symbol)
        return Ticker(
            exchange_id=self.id,
            symbol=symbol,
            bid=bids[0].price,
            ask=asks[0].price,
            last=mid,
            bid_volume=bids[0].amount,
            ask_volume=asks[0].amount,
            volume_24h=Decimal("1000000"),
            timestamp=utc_now(),
            received_at=utc_now(),
        )

    async def fetch_order_book(self, symbol: Symbol, *, depth: int | None = None) -> OrderBook:
        levels = max(1, depth or self._options.order_book_depth)
        bids, asks = self._book_sides(symbol, levels)
        return OrderBook(
            exchange_id=self.id,
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=utc_now(),
            received_at=utc_now(),
        )

    async def fetch_trading_fees(self, symbol: Symbol) -> MarketFees:
        return MarketFees(maker_bps=Decimal("8"), taker_bps=Decimal("10"), is_account_specific=True)

    # ---------------------------------------------------------------- account
    async def fetch_balances(self) -> BalanceSnapshot:
        balances = tuple(
            Balance(
                exchange_id=self.id,
                asset=asset,
                free=max(
                    DEC0,
                    (base + _unit_offset(self.id, asset) * spread).quantize(
                        _BALANCE_QUANTA.get(asset, Decimal("0.0001"))
                    ),
                ),
            )
            for asset, (base, spread) in _SEED_INVENTORY.items()
        )
        return BalanceSnapshot(exchange_id=self.id, balances=balances, timestamp=utc_now())

    # ---------------------------------------------------------------- transfers
    def simulated_address(self, asset: str, network: str | None = None) -> str:
        networks = _NETWORKS.get(asset.upper(), (("MAIN", asset.upper(), "0"),))
        net = network or networks[0][0]
        seed = f"{self.id}:{asset}:{net}:addr".encode()
        digest = hashlib.blake2b(seed, digest_size=10).hexdigest()
        return f"sim-{asset.lower()}-{net.lower()}-{digest[:20]}"

    async def fetch_withdrawal_networks(self, asset: str) -> tuple[WithdrawalNetwork, ...]:
        code = asset.strip().upper()
        networks = _NETWORKS.get(code)
        if not networks:
            return ()
        return tuple(
            WithdrawalNetwork(
                network=name,
                network_code=unified,
                withdraw_enabled=True,
                deposit_enabled=True,
                withdrawal_fee=Decimal(fee),
                withdrawal_min=Decimal(fee) * Decimal("10"),
            )
            for name, unified, fee in networks
        )

    async def fetch_deposit_address(
        self, asset: str, *, network: str | None = None
    ) -> DepositAddress:
        return DepositAddress(address=self.simulated_address(asset, network), network=network)

    async def fetch_deposits(self, asset: str, *, limit: int = 20) -> tuple[TransferTx, ...]:
        code = asset.strip().upper()
        return tuple(tx for tx in self._deposits if tx.asset == code)[:limit]

    async def fetch_withdrawals(self, asset: str, *, limit: int = 20) -> tuple[TransferTx, ...]:
        code = asset.strip().upper()
        return tuple(tx for tx in self._withdrawals if tx.asset == code)[:limit]

    async def withdraw(
        self,
        asset: str,
        amount: str,
        address: str,
        *,
        memo: str | None = None,
        network: str | None = None,
    ) -> TransferTx:
        """Record a simulated withdrawal (PAPER mode never moves real funds)."""
        code = asset.strip().upper()
        networks = _NETWORKS.get(code, (("MAIN", code, "0"),))
        net = network or networks[0][0]
        fee = Decimal(next((f for n, _c, f in networks if n == net), "0"))
        tx = TransferTx(
            direction="withdrawal",
            asset=code,
            network=net,
            amount=Decimal(amount),
            fee=fee,
            status="pending",
            txid=f"sim-wd-{len(self._withdrawals)}",
            address=address,
            timestamp=utc_now(),
        )
        self._withdrawals.append(tx)
        # The matching deposit arrives on this venue's ledger after the
        # orchestrator confirms the simulated transfer — mirror it here so
        # deposit detection has something realistic to poll.
        self._deposits.append(tx.model_copy(update={"direction": "deposit", "status": "confirmed"}))
        return tx

    def confirm_withdrawal(self, txid: str) -> None:
        """Test/demo hook: mark a pending simulated withdrawal as confirmed."""
        for index, tx in enumerate(self._withdrawals):
            if tx.txid == txid:
                self._withdrawals[index] = tx.model_copy(update={"status": "confirmed"})

    # ---------------------------------------------------------------- streaming
    async def watch_ticker(  # type: ignore[override]
        self, symbol: Symbol
    ) -> AsyncIterator[Ticker]:
        while True:
            yield await self.fetch_ticker(symbol)
            await asyncio.sleep(0.5)

    async def watch_order_book(  # type: ignore[override]
        self, symbol: Symbol
    ) -> AsyncIterator[OrderBook]:
        while True:
            yield await self.fetch_order_book(symbol)
            await asyncio.sleep(0.5)

    # ---------------------------------------------------------------- health
    async def ping(self) -> float:
        return float(abs(_unit_offset(self.id, "ping")) * 20 + 5)

    async def health(self) -> ExchangeHealth:
        latency = await self.ping()
        return ExchangeHealth(
            exchange_id=self.id,
            status=ExchangeStatus.ONLINE if self._opened else ExchangeStatus.UNKNOWN,
            rest_latency_ms=latency,
            ws_latency_ms=latency / 2,
        )


def build_simulated_adapter(request: object) -> SimulatedExchangeAdapter:
    """Registry factory for :class:`SimulatedExchangeAdapter`.

    Venue-specific seeding stays on the module defaults on purpose: the whole
    universe must be reproducible from the venue id alone.
    """
    exchange = request.exchange  # type: ignore[attr-defined]
    return SimulatedExchangeAdapter(
        exchange,
        credentials=request.credentials,  # type: ignore[attr-defined]
        options=request.options,  # type: ignore[attr-defined]
        order_gate=request.order_gate,  # type: ignore[attr-defined]
    )
