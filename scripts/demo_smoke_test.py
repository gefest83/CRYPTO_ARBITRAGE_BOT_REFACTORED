"""Manual DEMO smoke test — market BUY 5 USDT BTC/USDT → verify → SELL.

DEMO-only: refuses to run in LIVE (fail-closed).
Does NOT run automatically in pytest — invoke manually (Python 3.13 required):

    py -3.13 scripts/demo_smoke_test.py --exchange binance --symbol BTC/USDT --quote 5

Requires:
- CAT_TRADING__MODE=DEMO
- Demo API keys in .env (CAT_KEY_*)

Steps per venue:
1. fetch_ticker / fetch_order_book (market data)
2. market BUY 5 USDT worth of BTC/USDT → verify FILLED, filled_amount, average_price, fee
3. SELL the acquired BTC → verify final balance
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

# Ensure workspace root on path
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from app.config.settings import get_settings
from app.models.enums import TradingMode
from app.models.symbol import Symbol
from app.models.order import OrderRequest
from app.models.enums import OrderSide, OrderType


def _refuse_if_live() -> None:
    # LIVE is never allowed — handle both clean LIVE settings and invalid LIVE
    # config (which raises ValidationError) as refuse.
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001 - LIVE mis-configuration is still LIVE attempt
        msg = str(exc).lower()
        if "live" in msg:
            print("REFUSED: manual DEMO smoke test must never run in LIVE mode", file=sys.stderr)
            print(f"Configuration error (LIVE guard): {exc}", file=sys.stderr)
            sys.exit(2)
        raise
    if settings.mode is TradingMode.LIVE:
        print("REFUSED: manual DEMO smoke test must never run in LIVE mode", file=sys.stderr)
        print(f"Current mode: {settings.mode} — aborting", file=sys.stderr)
        sys.exit(2)
    if settings.mode is not TradingMode.DEMO:
        print(f"WARNING: expected MODE=DEMO, got {settings.mode} — continuing but not recommended", file=sys.stderr)
    if settings.trading.allow_live_withdrawals:
        print("REFUSED: withdrawals must remain disabled for smoke test", file=sys.stderr)
        sys.exit(2)


def _minimum_valid_qty(filters, reference_price: Decimal) -> tuple[Decimal | None, Decimal | None]:
    """Smallest amount satisfying step/min_amount/min_cost for reference_price.

    Returns (qty, quote) or (None,None) if filters is None.
    Used to report minimum executable notional when 5 USDT is too small.
    """
    if filters is None:
        return None, None
    from decimal import ROUND_CEILING

    step = filters.amount_step
    # Start from min_amount and min_cost
    candidates: list[Decimal] = []
    if filters.min_amount is not None:
        candidates.append(filters.min_amount)
    if filters.min_cost is not None and reference_price > Decimal("0"):
        # ceil(min_cost / price / step) * step  if step, else min_cost/price
        raw_needed = filters.min_cost / reference_price
        if step is not None and step > Decimal("0"):
            steps = (raw_needed / step).to_integral_value(rounding=ROUND_CEILING)
            if steps < 1:
                steps = Decimal("1")
            candidates.append((steps * step).quantize(Decimal("0.00000001")))
        else:
            candidates.append(raw_needed.quantize(Decimal("0.00000001")))
    # Also ensure step alignment for candidates
    if not candidates:
        return None, None
    qty = max(candidates)
    # Align to step (ceil)
    if step is not None and step > Decimal("0"):
        steps = (qty / step).to_integral_value(rounding=ROUND_CEILING)
        qty = (steps * step).quantize(Decimal("0.00000001"))
    else:
        qty = qty.quantize(Decimal("0.00000001"))
    quote = (qty * reference_price).quantize(Decimal("0.00000001"))
    return qty, quote


async def _wait_for_fill(adapter, order, symbol, timeout: float = 10.0):
    """Poll fetch_order / fetch_open_orders until FILLED or timeout.

    Uses existing recovery semantics where practical: if order is already
    FILLED/CLOSED return immediately; otherwise poll fetch_order by exchange
    order ID, fallback to fetch_open_orders by clientOrderId. No second
    create_order is ever issued (idempotent).
    """
    from app.models.enums import OrderStatus
    from app.recovery import ExecutionRecovery

    # Already filled
    if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED) and order.filled_amount > Decimal("0"):
        return order
    # For terminal REJECTED, nothing to poll
    if order.status == OrderStatus.REJECTED:
        return order

    # For timeout/unknown/partially_filled use existing recovery once
    if order.status in (OrderStatus.TIMEOUT, OrderStatus.UNKNOWN, OrderStatus.PARTIALLY_FILLED):
        try:
            recovery = ExecutionRecovery()
            recovered = await recovery.recover(order, adapter)
            if recovered.filled_amount > Decimal("0") and recovered.status in (OrderStatus.FILLED, OrderStatus.CANCELED):
                return recovered
            order = recovered
        except Exception:
            pass

    # For pending/open/partially_filled or filled==0, poll fetch_order
    # Bybit DEMO may return pending with filled=0 initially; poll up to timeout
    start = asyncio.get_running_loop().time()
    last = order
    while asyncio.get_running_loop().time() - start < timeout:
        # If already filled, stop
        if last.status in (OrderStatus.FILLED, OrderStatus.CANCELED) and last.filled_amount > Decimal("0"):
            break
        # Try direct fetch by exchange order ID
        try:
            if last.exchange_order_id:
                fetched = await adapter.fetch_order(last.exchange_order_id, symbol=symbol)
                last = fetched
                if fetched.status in (OrderStatus.FILLED, OrderStatus.CANCELED) and fetched.filled_amount > Decimal("0"):
                    break
                if fetched.filled_amount > Decimal("0") and fetched.status == OrderStatus.PARTIALLY_FILLED:
                    # Keep polling for more fill
                    pass
        except Exception:
            pass
        # Fallback: check open orders by clientOrderId
        try:
            if last.client_order_id:
                opens = await adapter.fetch_open_orders(symbol=symbol)
                for o in opens:
                    if o.client_order_id == last.client_order_id:
                        last = o
                        break
        except Exception:
            pass
        await asyncio.sleep(0.5)
        # If still pending and we have no update, continue polling
        # Also try one more fetch_order as last attempt
        try:
            if last.exchange_order_id:
                fetched = await adapter.fetch_order(last.exchange_order_id, symbol=symbol)
                last = fetched
                if fetched.status in (OrderStatus.FILLED, OrderStatus.CANCELED) and fetched.filled_amount > Decimal("0"):
                    break
        except Exception:
            pass
    return last


async def _smoke_one(exchange_id: str, symbol_text: str, quote_amount: Decimal) -> None:
    from app.services import build_app, shutdown_app, start_app

    settings = get_settings()
    print(f"\n=== {exchange_id} {symbol_text} {quote_amount} USDT (mode={settings.mode}) ===")

    services = await build_app(settings)
    await start_app(services)

    try:
        if exchange_id not in services.manager.enabled_ids():
            print(f"SKIP: {exchange_id} not enabled")
            return

        adapter = services.manager.adapter(exchange_id)
        symbol = Symbol.parse(symbol_text)

        # Preflight is already run in start_app for DEMO, but explicitly verify
        print(f"[{exchange_id}] fetching ticker...")
        ticker = await adapter.fetch_ticker(symbol)
        assert ticker.bid is not None or ticker.ask is not None, "ticker empty"
        print(f"  ticker bid={ticker.bid} ask={ticker.ask}")

        print(f"[{exchange_id}] fetching order book...")
        book = await adapter.fetch_order_book(symbol)
        assert book.bids and book.asks, "book empty"
        print(f"  book best bid={book.bids[0].price} ask={book.asks[0].price}")

        # Check that symbol mapping uses unified (no BadSymbol)
        print(f"[{exchange_id}] verifying symbol mapping unified...")
        unified = adapter._native_symbol(symbol)  # type: ignore[attr-defined]
        assert unified == symbol_text, f"_native_symbol returned {unified!r} != {symbol_text!r}"
        print(f"  _native_symbol OK: {unified}")

        # Check precision fallback for BTC/USDT and ETH/USDT
        filt = services._precision if hasattr(services, "_precision") else None
        # Actually executor holds precision
        prec = services.executor._precision  # type: ignore[attr-defined]
        if prec is not None:
            for probe in ("BTC/USDT", "ETH/USDT"):
                f = prec.filters_for(exchange_id, probe)
                status = f"amount_step={f.amount_step} min_amount={f.min_amount} min_cost={f.min_cost}" if f else "None"
                print(f"  precision {probe}: {status}")
                if exchange_id == "binance" and probe in ("BTC/USDT", "ETH/USDT"):
                    assert f is not None, f"binance {probe} precision missing (fallback failed)"
                    assert f.amount_step is not None, f"binance {probe} amount_step missing"

        # Check fee provider is real for DEMO
        fees = services.scanner._fees  # type: ignore[attr-defined]
        fee = fees.fees_for(exchange_id, symbol, None)  # type: ignore[arg-type]
        print(f"  fee taker_bps={fee.taker_bps} (account={fee.is_account_specific})")

        # Now real DEMO order — only if --execute flag
        if not _smoke_one.execute:
            print(f"[{exchange_id}] dry-run: skipping real order (use --execute to place)")
            return

        # Balance before
        bal_before = await adapter.fetch_balances()
        usdt_before = bal_before.free_of("USDT")
        base_before = bal_before.free_of(symbol.base)
        print(f"  balance before: USDT {usdt_before} {symbol.base} {base_before}")

        # MARKET BUY — sizing via exchange filters (same as triangular executor)
        raw_qty = (quote_amount / book.asks[0].price).quantize(Decimal("0.00000001"))
        # Local filter check before any network call (mirrors TriangleExecutor._fill_leg)
        prec = services.executor._precision  # type: ignore[attr-defined]
        filters = prec.filters_for(exchange_id, symbol.name) if prec is not None else None
        qty_estimate = raw_qty
        if filters is not None:
            from app.execution.precision import apply_filters

            _tmp_req = OrderRequest(
                exchange_id=exchange_id,
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                amount=raw_qty,
            )
            _, _reason = apply_filters(_tmp_req, filters, reference_price=book.asks[0].price)
            if _reason is not None:
                # Compute minimum valid quantity that satisfies step + min_amount + min_cost
                min_qty, min_quote = _minimum_valid_qty(filters, book.asks[0].price)
                msg = (
                    f"Requested quote {quote_amount} USDT cannot satisfy {symbol.name} "
                    f"minimum notional after amount-step rounding; "
                    f"minimum executable quote is approximately {min_quote} USDT "
                    f"({min_qty} {symbol.base})."
                )
                print(f"[{exchange_id}] {msg}")
                # For smoke test designed to do smallest valid DEMO order, adjust explicitly
                # (not silent — we printed the adjustment)
                if min_qty is not None and min_qty > Decimal("0"):
                    print(f"[{exchange_id}] Adjusting to minimum valid amount {min_qty} {symbol.base} (~{min_quote} USDT) for DEMO smoke test.")
                    qty_estimate = min_qty
                    quote_amount = min_quote  # keep accounting consistent
                else:
                    print(f"[{exchange_id}] Local sizing failed ({_reason}) — aborting before create_order.")
                    return
                # Re-validate adjusted amount
                _adj_req = OrderRequest(
                    exchange_id=exchange_id,
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    amount=qty_estimate,
                )
                _, _adj_reason = apply_filters(_adj_req, filters, reference_price=book.asks[0].price)
                if _adj_reason is not None:
                    print(f"[{exchange_id}] Adjusted amount still invalid ({_adj_reason}) — aborting.")
                    return
        print(f"[{exchange_id}] MARKET BUY {qty_estimate} {symbol.base} (~{quote_amount} USDT)...")
        req = OrderRequest(
            exchange_id=exchange_id,
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=qty_estimate,
        )
        order = await adapter.create_order(req)
        print(f"  order id={order.exchange_order_id} status={order.status} filled={order.filled_amount} avg={order.average_price} fee={order.fee_paid} {order.fee_currency}")
        # Bybit DEMO may return pending — poll up to ~10s using existing recovery semantics, no duplicate order
        if order.status.value not in ("filled", "closed", "FILLED") or order.filled_amount <= Decimal("0"):
            print(f"  order pending/open, polling fetch_order for up to 10s...")
            order = await _wait_for_fill(adapter, order, symbol, timeout=10.0)
            print(f"  after polling: id={order.exchange_order_id} status={order.status} filled={order.filled_amount} avg={order.average_price} fee={order.fee_paid}")
            if order.status.value not in ("filled", "closed", "FILLED") or order.filled_amount <= Decimal("0"):
                print(f"  BUY not FILLED after polling: status={order.status} filled={order.filled_amount} — STOP (no retry, no duplicate order)")
                return
        assert order.filled_amount > Decimal("0"), "BUY filled_amount 0"
        assert order.average_price is not None and order.average_price > Decimal("0"), "BUY average_price missing"
        # Fee may be 0 on demo for small amount, but should be present field
        print(f"  BUY verified: filled={order.filled_amount} avg={order.average_price} fee={order.fee_paid}")

        # Verify via fetch_order
        try:
            fetched = await adapter.fetch_order(order.exchange_order_id or order.id, symbol=symbol)
            print(f"  fetch_order status={fetched.status} filled={fetched.filled_amount}")
        except Exception as exc:
            print(f"  fetch_order failed: {exc}")

        # SELL the acquired BTC — same async handling
        sell_amt = order.filled_amount
        print(f"[{exchange_id}] MARKET SELL {sell_amt} {symbol.base}...")
        # Re-fetch book for fresh price
        book2 = await adapter.fetch_order_book(symbol)
        req2 = OrderRequest(
            exchange_id=exchange_id,
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            amount=sell_amt,
        )
        order2 = await adapter.create_order(req2)
        print(f"  sell id={order2.exchange_order_id} status={order2.status} filled={order2.filled_amount} avg={order2.average_price} fee={order2.fee_paid}")
        if order2.status.value not in ("filled", "closed", "FILLED") or order2.filled_amount <= Decimal("0"):
            print(f"  sell pending/open, polling fetch_order for up to 10s...")
            order2 = await _wait_for_fill(adapter, order2, symbol, timeout=10.0)
            print(f"  after polling SELL: id={order2.exchange_order_id} status={order2.status} filled={order2.filled_amount} avg={order2.average_price} fee={order2.fee_paid}")
            if order2.status.value not in ("filled", "closed", "FILLED") or order2.filled_amount <= Decimal("0"):
                print(f"  SELL not FILLED after polling: status={order2.status} filled={order2.filled_amount} — STOP (no retry)")
                return
        assert order2.filled_amount > Decimal("0")

        bal_after = await adapter.fetch_balances()
        usdt_after = bal_after.free_of("USDT")
        print(f"  balance after: USDT {usdt_after} {symbol.base} {bal_after.free_of(symbol.base)}")
        print(f"[{exchange_id}] smoke test PASSED")

    finally:
        await shutdown_app(services)


_smoke_one.execute = False  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual DEMO smoke test (DEMO-only)")
    parser.add_argument("--exchange", default="binance", help="binance|okx|bybit or 'all'")
    parser.add_argument("--symbol", default="BTC/USDT", help="e.g. BTC/USDT")
    parser.add_argument("--quote", default="5", help="quote amount USDT for BUY (default 5)")
    parser.add_argument("--execute", action="store_true", help="actually place DEMO orders (default dry-run)")
    args = parser.parse_args()

    _refuse_if_live()

    if args.execute:
        print("*** REAL DEMO ORDERS WILL BE PLACED (--execute) ***")
        confirm = input("Type 'DEMO' to confirm: ")
        if confirm.strip() != "DEMO":
            print("Aborted")
            sys.exit(1)
        _smoke_one.execute = True  # type: ignore[attr-defined]

    async def run():
        if args.exchange == "all":
            for ex in ("binance", "okx", "bybit"):
                await _smoke_one(ex, args.symbol, Decimal(args.quote))
        else:
            await _smoke_one(args.exchange, args.symbol, Decimal(args.quote))

    asyncio.run(run())


if __name__ == "__main__":
    main()
