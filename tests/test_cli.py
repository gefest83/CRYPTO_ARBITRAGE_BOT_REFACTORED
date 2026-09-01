"""CLI: parser surface and command behaviour through the real paper services."""

from decimal import Decimal

from app.cli.main import build_parser, cmd_balances, cmd_status, cmd_trades
from app.services import AppServices


def test_parser_accepts_all_required_commands():
    parser = build_parser()
    for command in (
        "scan",
        "triangle",
        "transfer",
        "balances",
        "trades",
        "status",
        "reconcile",
        "start_auto",
        "stop_auto",
    ):
        args = parser.parse_args([command])
        assert args.command == command


def test_parser_transfer_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "transfer",
            "--asset",
            "SOL",
            "--amount",
            "2.5",
            "--source",
            "binance",
            "--dest",
            "okx",
            "--execute",
        ]
    )
    assert args.asset == "SOL"
    assert args.amount == Decimal("2.5")
    assert args.source == "binance"
    assert args.dest == "okx"
    assert args.execute is True


async def test_status_command_prints_structured_output(services: AppServices, capsys):
    code = await cmd_status(services, build_parser().parse_args(["status"]))
    assert code == 0
    out = capsys.readouterr().out
    assert "=== STATUS ===" in out
    assert "mode" in out and "PAPER" in out
    assert "binance" in out and "okx" in out and "bybit" in out
    assert "kill switch" in out
    assert "daily pnl" in out


async def test_balances_command_lists_venues(services: AppServices, capsys):
    code = await cmd_balances(services, build_parser().parse_args(["balances"]))
    assert code == 0
    out = capsys.readouterr().out
    assert "binance:" in out
    assert "USDT" in out


async def test_trades_command_handles_empty_history(services: AppServices, capsys):
    code = await cmd_trades(services, build_parser().parse_args(["trades"]))
    assert code == 0
    assert "no trades yet" in capsys.readouterr().out


async def test_scan_and_triangle_commands(services: AppServices, capsys):
    from app.cli.main import cmd_scan, cmd_triangle

    code = await cmd_scan(services, build_parser().parse_args(["scan"]))
    assert code == 0
    out = capsys.readouterr().out
    assert "triangle opportunities" in out
    assert "transfer plans" in out

    code = await cmd_triangle(services, build_parser().parse_args(["triangle"]))
    out = capsys.readouterr().out
    # the best ring either executes or is honestly rejected — both are valid
    assert "executing best" in out
    assert code in (0, 1)


async def test_transfer_command_dry_run_then_wait(services: AppServices, capsys):
    from app.cli.main import cmd_transfer

    args = build_parser().parse_args(["transfer"])
    code = await cmd_transfer(services, args)
    assert code == 0
    assert "dry run" in capsys.readouterr().out

    args = build_parser().parse_args(["transfer", "--execute", "--wait"])
    code = await cmd_transfer(services, args)
    out = capsys.readouterr().out
    assert "executing best plan" in out
    assert "final:" in out


async def test_stop_auto_and_kill_switch(tmp_path, services: AppServices, capsys):
    from app.cli.main import cmd_start_auto, cmd_stop_auto

    # stop_auto without --kill only clears the flag
    code = await cmd_stop_auto(services, build_parser().parse_args(["stop_auto"]))
    assert code == 0
    assert not await services.auto_trading_enabled()

    # stop_auto --kill engages the persistent kill switch
    code = await cmd_stop_auto(
        services, build_parser().parse_args(["stop_auto", "--kill", "--reason", "cli test"])
    )
    assert code == 0
    assert services.guard.is_halted

    # start_auto refuses while halted (no --release)
    code = await cmd_start_auto(services, build_parser().parse_args(["start_auto"]))
    assert code == 1
    assert "ENGAGED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# TEST GAP #6: manual reconciliation CLI command.  The ``reconcile`` command
# must be strictly read-only: it only queries the transfers table for rows
# in MANUAL_REVIEW, prints a per-record reconciliation block, and never
# touches the exchange adapters, the orchestrator, balances, or risk state.
# ---------------------------------------------------------------------------


class _StrictlyReadOnlyAdapter:
    """Exchange adapter that fails on every trading-side method.

    The ``reconcile`` command must never call any of these; if the CLI
    ever does, the test fails loudly.
    """

    def __init__(self) -> None:
        self.create_order_calls = 0
        self.withdraw_calls = 0
        self.cancel_order_calls = 0

    async def create_order(self, *args, **kwargs):  # pragma: no cover
        self.create_order_calls += 1
        raise AssertionError("reconcile command must not call create_order")

    async def withdraw(self, *args, **kwargs):  # pragma: no cover
        self.withdraw_calls += 1
        raise AssertionError("reconcile command must not call withdraw")

    async def cancel_order(self, *args, **kwargs):  # pragma: no cover
        self.cancel_order_calls += 1
        raise AssertionError("reconcile command must not call cancel_order")


async def test_reconcile_command_lists_manual_review_transfers(tmp_path, capsys):
    """TEST GAP #6 / Scenario A: the ``reconcile`` command surfaces every
    MANUAL_REVIEW transfer with the full reconciliation payload.

    Two MANUAL_REVIEW records are persisted with different assets,
    exchanges, reasons, and withdrawal ids.  The CLI must print every
    required reconciliation field for each one.
    """
    from decimal import Decimal as _D
    from app.cli.main import cmd_reconcile
    from app.models.enums import TransferState
    from app.models.transfer import TransferPlan, TransferRecord
    from app.services import build_app, shutdown_app, start_app
    from tests.conftest import make_settings

    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app)
    try:
        transfers = app.transfers

        record_a = TransferRecord(
            source_exchange="binance",
            dest_exchange="okx",
            asset="ETH",
            network="SIM",
            amount=_D("5"),
            plan=TransferPlan(
                source_exchange="binance",
                dest_exchange="okx",
                asset="ETH",
                network="SIM",
                amount=_D("5"),
                buy_price=_D("100"),
                sell_price=_D("105"),
            ),
            state=TransferState.MANUAL_REVIEW,
            buy_filled_amount=_D("5"),
            withdrawal_id="wd-A-1",
            withdrawal_txid="wd-A-1",
            withdrawal_amount=_D("4.99"),
            deposit_address="0xDEST-A",
            error="withdrawal failed: venue error; purchased 5 ETH remains on binance",
        )
        record_b = TransferRecord(
            source_exchange="binance",
            dest_exchange="bybit",
            asset="XRP",
            network="XRP",
            amount=_D("100"),
            plan=TransferPlan(
                source_exchange="binance",
                dest_exchange="bybit",
                asset="XRP",
                network="XRP",
                amount=_D("100"),
                buy_price=_D("1"),
                sell_price=_D("1.01"),
            ),
            state=TransferState.MANUAL_REVIEW,
            buy_filled_amount=_D("100"),
            withdrawal_id="wd-B-7",
            withdrawal_txid=None,
            withdrawal_amount=_D("99.75"),
            error="destination venue returned no memo/tag for the XRP deposit address",
        )
        await transfers.save(record_a)
        await transfers.save(record_b)

        pre_a = await transfers.get(record_a.id)
        pre_b = await transfers.get(record_b.id)

        code = await cmd_reconcile(app, build_parser().parse_args(["reconcile"]))
        assert code == 0
        out = capsys.readouterr().out

        # Header + count.
        assert "=== MANUAL RECONCILIATION" in out
        assert "2 transfer(s) require manual review" in out

        # Both transfer ids surface.
        assert record_a.id in out
        assert record_b.id in out

        # Per-record reconciliation fields (the six questions).
        # 1. which transfer is stuck -> id + state
        assert "manual_review" in out
        # 2. asset and quantity
        assert "ETH" in out and "XRP" in out
        # 3. source / destination
        assert "binance -> okx" in out
        assert "binance -> bybit" in out
        # 4. withdrawal id / txid
        assert "wd-A-1" in out and "wd-B-7" in out
        # 5. reason
        assert "purchased 5 ETH remains on binance" in out
        assert "no memo/tag for the XRP deposit address" in out
        # 6. timestamps
        assert "created_at" in out
        assert "updated_at" in out

        # Post-condition: the DB rows are byte-equivalent to the snapshot.
        post_a = await transfers.get(record_a.id)
        post_b = await transfers.get(record_b.id)
        assert post_a is not None and pre_a is not None
        assert post_b is not None and pre_b is not None
        assert post_a.model_dump() == pre_a.model_dump(), (
            "reconcile must not mutate a transfer record"
        )
        assert post_b.model_dump() == pre_b.model_dump()
    finally:
        await shutdown_app(app)


async def test_reconcile_command_is_strictly_read_only(tmp_path, capsys):
    """TEST GAP #6 / Scenario B: ``reconcile`` must NOT touch the
    exchange adapters, must NOT call recovery, must NOT mutate any
    transfer, must NOT change balances or risk state.

    The exchange manager is replaced with a ``_StrictlyReadOnlyAdapter``
    that raises ``AssertionError`` on any trading-side call.
    """
    from decimal import Decimal as _D
    from app.cli.main import cmd_reconcile
    from app.exchanges.manager import ExchangeManager
    from app.models.enums import TransferState
    from app.models.transfer import TransferPlan, TransferRecord
    from app.services import build_app, shutdown_app, start_app
    from tests.conftest import make_settings

    settings = make_settings(tmp_path)
    app = await build_app(settings)
    await start_app(app)
    try:
        transfers = app.transfers
        record = TransferRecord(
            source_exchange="binance",
            dest_exchange="okx",
            asset="ETH",
            network="SIM",
            amount=_D("5"),
            plan=TransferPlan(
                source_exchange="binance",
                dest_exchange="okx",
                asset="ETH",
                network="SIM",
                amount=_D("5"),
                buy_price=_D("100"),
                sell_price=_D("105"),
            ),
            state=TransferState.MANUAL_REVIEW,
            buy_filled_amount=_D("5"),
            withdrawal_id="wd-RO-1",
            withdrawal_amount=_D("4.99"),
            error="withdrawal failed: venue error; purchased 5 ETH remains on binance",
        )
        await transfers.save(record)

        # Snapshot: row contents, daily pnl, open-transfers count, guard state.
        pre_row = await transfers.get(record.id)
        pre_daily = app.risk_state.daily_pnl
        pre_open = app.risk_state.open_transfers
        pre_halted = app.guard.is_halted
        pre_kill_reason = app.guard.halt_reason

        readonly = _StrictlyReadOnlyAdapter()
        original_manager = app.manager
        ro = ExchangeManager.__new__(ExchangeManager)
        # Copy over the bound state from the real manager so the rest of
        # the app keeps working (the CLI must not actually need any of
        # this, but we preserve the invariant).
        for attr in (
            "_settings",
            "_exchanges_cfg",
            "_credentials",
            "_order_gate",
            "_clock",
            "_exchanges",
            "_health",
            "_breaker_failures",
            "_breaker_opened_at_ms",
            "_auth_blocked",
        ):
            try:
                setattr(ro, attr, getattr(original_manager, attr))
            except AttributeError:
                pass
        ro._adapters = {
            "binance": readonly,
            "okx": readonly,
            "bybit": readonly,
        }
        app.manager = ro

        code = await cmd_reconcile(app, build_parser().parse_args(["reconcile"]))
        out = capsys.readouterr().out
        assert code == 0
        assert "MANUAL RECONCILIATION" in out
        assert record.id in out

        # The read-only adapter was never asked to do anything.
        assert readonly.create_order_calls == 0
        assert readonly.withdraw_calls == 0
        assert readonly.cancel_order_calls == 0

        # The persisted row is unchanged.
        post_row = await transfers.get(record.id)
        assert post_row is not None and pre_row is not None
        assert post_row.model_dump() == pre_row.model_dump(), (
            "reconcile must not mutate any transfer row"
        )

        # Risk state and guard state are unchanged (the CLI does not
        # touch either).
        assert app.risk_state.daily_pnl == pre_daily
        assert app.risk_state.open_transfers == pre_open
        assert app.guard.is_halted == pre_halted
        assert app.guard.halt_reason == pre_kill_reason

        # Restore the real manager.
        app.manager = original_manager
    finally:
        await shutdown_app(app)


async def test_reconcile_command_handles_empty_state(services, capsys):
    """TEST GAP #6 / Scenario C: when no MANUAL_REVIEW transfers exist,
    the command must print a clear empty-state message.
    """
    from app.cli.main import cmd_reconcile
    from app.models.enums import TransferState

    pre = await services.transfers.list_by_state(TransferState.MANUAL_REVIEW.value)
    assert pre == [], (
        f"services fixture must start with no MANUAL_REVIEW rows; got {pre}"
    )

    code = await cmd_reconcile(
        services, build_parser().parse_args(["reconcile"])
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "No transfers require manual review." in out
