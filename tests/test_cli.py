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
