"""CLI: the primary interface of the bot.

Commands (``python -m app <command>``):

    scan          scan all venues for triangle + transfer opportunities
    triangle      scan and EXECUTE the best triangle cycle
    transfer      plan transfer arbitrage (``--asset/--amount/--execute``)
    balances      show balances per venue
    trades        show recent trades
    status        show bot status (mode, exchanges, risk, transfers)
    reconcile     list transfers stuck in MANUAL_REVIEW (read-only)
    start_auto    run the auto-trading loop (Ctrl+C to stop)
    stop_auto     stop auto trading (``--kill`` engages the kill switch)
    telegram      run the Telegram bot (same services as the CLI)

Every command prints structured terminal output.  No trading logic lives
here — everything goes through :class:`app.services.AppServices`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from decimal import Decimal
from typing import Any

from app.config.logging_config import get_logger
from app.errors import TerminalError
from app.models.enums import TransferState
from app.services import AppServices, build_app, shutdown_app, start_app

__all__ = ["main"]

logger = get_logger("cli")

COMMANDS = (
    "scan",
    "triangle",
    "transfer",
    "balances",
    "trades",
    "status",
    "reconcile",
    "start_auto",
    "stop_auto",
    "telegram",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arbbot", description="Crypto Arbitrage Bot (triangular + transfer)"
    )
    parser.add_argument("command", choices=COMMANDS, help="command to run")
    parser.add_argument("--notional", type=Decimal, default=None, help="scan notional (quote)")
    parser.add_argument("--asset", default=None, help="transfer: asset to move")
    parser.add_argument("--amount", type=Decimal, default=None, help="transfer: asset amount")
    parser.add_argument("--source", default=None, help="transfer: source venue")
    parser.add_argument("--dest", default=None, help="transfer: destination venue")
    parser.add_argument("--execute", action="store_true", help="transfer: execute the best plan")
    parser.add_argument(
        "--wait", action="store_true", help="transfer: drive the lifecycle until terminal state"
    )
    parser.add_argument("--limit", type=int, default=20, help="listing limit")
    parser.add_argument(
        "--reconcile-limit",
        type=int,
        default=100,
        help="reconcile: max number of MANUAL_REVIEW transfers to display",
    )
    parser.add_argument("--kill", action="store_true", help="stop_auto: engage the kill switch")
    parser.add_argument("--reason", default="engaged from CLI", help="stop_auto --kill: reason")
    parser.add_argument(
        "--release", action="store_true", help="start_auto: release an engaged kill switch first"
    )
    return parser


# ---------------------------------------------------------------- output
def _line(label: str, value: Any, indent: int = 0) -> None:
    print(f"{'  ' * indent}{label:<20}{value}")


def _print_opportunities(opportunities) -> None:
    if not opportunities:
        print("no triangle opportunities above the configured minimum")
        return
    for opportunity in opportunities:
        print(
            f"  {opportunity.direction:<38} "
            f"net {opportunity.net_profit_bps:>8.2f} bps  "
            f"notional {opportunity.size_notional_quote}  "
            f"age {opportunity.data_age_ms:.0f} ms"
        )


def _print_plans(plans) -> None:
    if not plans:
        print("no transfer plans above the configured minimum")
        return
    for plan in plans:
        print(
            f"  {plan.source_exchange}->{plan.dest_exchange:<8} {plan.asset:<6} "
            f"{plan.amount:<14} via {plan.network:<8} "
            f"net {plan.net_profit_bps:>8.2f} bps  ({plan.net_profit_quote:.4f} quote)"
        )


def _print_transfers(records) -> None:
    if not records:
        print("no transfers")
        return
    for record in records:
        line = (
            f"  {record.id}  {record.source_exchange}->{record.dest_exchange} "
            f"{record.asset} {record.amount} via {record.network}: {record.state.value}"
        )
        if record.realized_profit_quote:
            line += f"  realized {record.realized_profit_quote}"
        if record.error:
            line += f"  [{record.error[:80]}]"
        print(line)


def _print_manual_review_record(record) -> None:
    """One MANUAL_REVIEW reconciliation block.

    The block answers the operator's six questions at a glance:

      1. which transfer is stuck      -> record.id + state
      2. asset and quantity           -> asset + amount / buy_filled_amount
      3. which exchange holds it      -> source / dest
      4. withdrawal ID / TXID         -> withdrawal_id / withdrawal_txid
      5. why did the bot stop         -> error
      6. when did it happen           -> created_at / updated_at
    """
    print(f"  transfer_id : {record.id}")
    print(f"  state       : {record.state.value}")
    print(f"  route       : {record.source_exchange} -> {record.dest_exchange}")
    print(f"  asset       : {record.asset}  (planned {record.amount})")
    if record.buy_filled_amount and record.buy_filled_amount > 0:
        print(
            f"  held        : {record.buy_filled_amount} {record.asset} "
            f"on {record.source_exchange}"
        )
    if record.withdrawal_amount and record.withdrawal_amount > 0:
        print(
            f"  withdrawal  : {record.withdrawal_amount} {record.asset} "
            f"on {record.source_exchange} (id={record.withdrawal_id or '-'}"
        )
    if record.withdrawal_txid:
        print(f"                 txid={record.withdrawal_txid})")
    elif record.withdrawal_id and not record.withdrawal_txid:
        print(")")
    if record.deposit_address:
        print(f"  deposit addr: {record.deposit_address}")
    print(f"  created_at  : {record.created_at.isoformat()}")
    print(f"  updated_at  : {record.updated_at.isoformat()}")
    if record.error:
        print(f"  reason      : {record.error}")


# ---------------------------------------------------------------- commands
async def cmd_scan(services: AppServices, args: argparse.Namespace) -> int:
    print(f"=== SCAN (mode {services.settings.mode.value}) ===")
    opportunities = await services.scan_triangles(notional_quote=args.notional)
    print(f"triangle opportunities: {len(opportunities)}")
    _print_opportunities(opportunities)
    print()
    plans = await services.plan_transfers()
    print(f"transfer plans: {len(plans)}")
    _print_plans(plans)
    return 0


async def cmd_triangle(services: AppServices, args: argparse.Namespace) -> int:
    print(f"=== TRIANGLE (mode {services.settings.mode.value}) ===")
    opportunities = await services.scan_triangles(notional_quote=args.notional)
    _print_opportunities(opportunities)
    if not opportunities:
        return 0
    best = opportunities[0]
    print(f"\nexecuting best: {best.direction} (net {best.net_profit_bps} bps)")
    trade, assessment = await services.execute_triangle(best)
    if assessment is not None and not assessment.approved:
        print("REJECTED by risk validation:")
        for reason in assessment.reasons:
            print(f"  - {reason}")
    print(
        f"trade {trade.id}: {trade.status.value}  "
        f"in {trade.input_amount}  out {trade.output_amount}  "
        f"fees {trade.fees_quote}  net {trade.net_profit} ({trade.net_profit_bps} bps)"
    )
    if trade.error:
        print(f"  note: {trade.error}")
    return 0 if trade.status.value == "completed" else 1


async def cmd_transfer(services: AppServices, args: argparse.Namespace) -> int:
    print(f"=== TRANSFER (mode {services.settings.mode.value}) ===")
    plans = await services.plan_transfers(
        asset=args.asset, amount=args.amount, source=args.source, dest=args.dest
    )
    _print_plans(plans)
    if not plans:
        return 0
    if not args.execute:
        print("\n(dry run: pass --execute to run the best plan)")
        return 0
    plan = plans[0]
    print(
        f"\nexecuting best plan: {plan.source_exchange}->{plan.dest_exchange} "
        f"{plan.asset} {plan.amount} via {plan.network}"
    )
    record = await services.start_transfer(plan)
    print(f"transfer {record.id}: {record.state.value}")
    if record.error:
        print(f"  error: {record.error}")
    if args.wait:
        print("driving lifecycle until terminal state (Ctrl+C to leave it running)...")
        while not record.is_terminal:
            await asyncio.sleep(services.settings.transfer.poll_interval_seconds)
            advanced = await services.tick_transfers()
            for updated in advanced:
                if updated.id == record.id:
                    record = updated
            print(f"  state: {record.state.value}")
        print(
            f"final: {record.state.value}  realized {record.realized_profit_quote} "
            f"(planned {record.plan.net_profit_quote:.4f})"
        )
        if record.error:
            print(f"  error: {record.error}")
    else:
        print("(lifecycle continues in the background of start_auto / telegram)")
    return 0 if record.state.value != "failed" else 1


async def cmd_balances(services: AppServices, args: argparse.Namespace) -> int:
    print(f"=== BALANCES (mode {services.settings.mode.value}) ===")
    snapshots = await services.balances()
    if not snapshots:
        print("no balances available")
        return 1
    for venue, snapshot in sorted(snapshots.items()):
        print(f"{venue}:")
        shown = 0
        for balance in sorted(snapshot.balances, key=lambda b: b.asset):
            if balance.total <= 0:
                continue
            print(f"  {balance.asset:<8} free {balance.free:<20} used {balance.used}")
            shown += 1
            if shown >= args.limit:
                print("  ...")
                break
    return 0


async def cmd_trades(services: AppServices, args: argparse.Namespace) -> int:
    print("=== RECENT TRADES ===")
    trades = await services.trades.list_recent(limit=args.limit)
    if not trades:
        print("no trades yet")
        return 0
    for trade in trades:
        print(
            f"  {trade.created_at:%Y-%m-%d %H:%M:%S}  {trade.strategy.value:<9} "
            f"{trade.route:<40} {trade.status.value:<14} "
            f"net {trade.net_profit:>12} ({trade.net_profit_bps:>8} bps)"
        )
        if trade.error:
            print(f"      {trade.error[:110]}")
    return 0


async def cmd_status(services: AppServices, args: argparse.Namespace) -> int:
    print("=== STATUS ===")
    status = await services.status()
    _line("mode", status["mode"])
    _line("uptime", f"{status['uptime_seconds']}s")
    guard = status["guard"]
    _line("trading enabled", guard["trading_enabled"])
    _line(
        "kill switch",
        f"ENGAGED: {guard['halt_reason']}" if guard["halted"] == "true" else "released",
    )
    _line("auto trading", "on" if status["auto_trading"] else "off")
    print("exchanges:")
    for venue, info in status["exchanges"].items():
        print(
            f"  {venue:<10} {info['status']:<10} adapter={info['adapter']:<10} "
            f"keys={info['credentials']}"
        )
    print("market data:")
    for key, value in status["market_data"].items():
        _line(key, value, indent=1)
    print("risk:")
    risk = status["risk"]
    _line("daily pnl", risk["daily_pnl"], indent=1)
    _line("open transfers", risk["open_transfers"], indent=1)
    _line("max trade size", risk["limits"]["max_trade_size"], indent=1)
    _line("min net profit", f"{risk['limits']['min_net_profit_bps']} bps", indent=1)
    if status["transfers_open"]:
        print("open transfers:")
        _print_transfers([type("T", (), dict(t)) for t in status["transfers_open"]])
    if status["recent_trades"]:
        print("recent trades:")
        for trade in status["recent_trades"]:
            print(
                f"  {trade['strategy']:<9} {trade['route']:<40} "
                f"{trade['status']:<14} net {trade['net_profit']}"
            )
    return 0


async def cmd_reconcile(services: AppServices, args: argparse.Namespace) -> int:
    """List every transfer currently in MANUAL_REVIEW (read-only).

    Pure SELECT against the transfers table.  Does NOT call any exchange
    adapter, does NOT tick the orchestrator, does NOT mutate any record,
    does NOT acquire a trading lock, does NOT engage the kill switch.
    An operator can run this safely while the bot is live.
    """
    print(f"=== MANUAL RECONCILIATION (mode {services.settings.mode.value}) ===")
    limit = max(1, int(getattr(args, "reconcile_limit", 100) or 100))
    records = await services.transfers.list_by_state(TransferState.MANUAL_REVIEW.value)
    if not records:
        print("No transfers require manual review.")
        return 0
    print(f"{len(records)} transfer(s) require manual review:\n")
    shown = 0
    for record in records:
        if shown >= limit:
            remaining = len(records) - shown
            print(f"... and {remaining} more (use --reconcile-limit to show more)")
            break
        _print_manual_review_record(record)
        print()
        shown += 1
    return 0


async def cmd_start_auto(services: AppServices, args: argparse.Namespace) -> int:
    print(f"=== AUTO TRADING (mode {services.settings.mode.value}) ===")
    if services.guard.is_halted:
        if not args.release:
            print(
                f"kill switch is ENGAGED ({services.guard.halt_reason}).\n"
                "Release it explicitly with:  python -m app start_auto --release"
            )
            return 1
        await services.release_kill_switch()
        print("kill switch released")
    # Use the AutoTradingController — same entry point as the Telegram
    # ``/start_trading`` command — so CLI and Telegram cannot create two
    # independent auto loops in the same process.
    started, message = await services.auto_controller.start()
    if not started:
        print(message)
        return 1
    print(f"{message} — Ctrl+C to stop")
    # Wait for either an external stop signal (the controller's task), or the
    # operator pressing Ctrl+C.  The controller is the single owner of the
    # loop: pressing Ctrl+C asks the loop to stop and waits for it to exit.
    controller = services.auto_controller
    assert controller is not None
    try:
        task = controller._task
        if task is not None:
            await asyncio.shield(task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await controller.stop()
    return 0


async def cmd_stop_auto(services: AppServices, args: argparse.Namespace) -> int:
    print("=== STOP AUTO ===")
    if args.kill:
        await services.engage_kill_switch(args.reason)
        print(f"auto trading disabled; kill switch ENGAGED ({args.reason})")
        return 0
    # Use the controller so we share state with Telegram; the controller is
    # idempotent and never raises.
    stopped, message = await services.auto_controller.stop()
    print(message)
    return 0


async def cmd_telegram(services: AppServices, args: argparse.Namespace) -> int:
    from app.telegram.bot import run_telegram

    return await run_telegram(services)


_HANDLERS = {
    "scan": cmd_scan,
    "triangle": cmd_triangle,
    "transfer": cmd_transfer,
    "balances": cmd_balances,
    "trades": cmd_trades,
    "status": cmd_status,
    "reconcile": cmd_reconcile,
    "start_auto": cmd_start_auto,
    "stop_auto": cmd_stop_auto,
    "telegram": cmd_telegram,
}


async def run(args: argparse.Namespace) -> int:
    services = await build_app()
    try:
        await start_app(services)
        handler = _HANDLERS[args.command]
        return await handler(services, args)
    finally:
        await shutdown_app(services)


def main() -> None:
    # Windows consoles often default to a legacy code page; make every
    # command's output safe to print regardless of the terminal encoding.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(run(args)))
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130) from None
    except TerminalError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
