# CRYPTO ARBITRAGE BOT

A CLI-first, Telegram-controlled cryptocurrency arbitrage bot with exactly two
strategies on exactly three exchanges:

- **Triangular arbitrage** — depth-aware 3-leg spot cycles (`USDT -> X -> Y -> USDT`)
  inside ONE venue
- **Transfer arbitrage** — a persisted, restart-safe cross-venue workflow
  (`BUY -> WITHDRAW -> TRANSFER -> DEPOSIT -> SELL`)

Venues: **Binance**, **OKX**, **Bybit**.

```
python -m app scan        # scan triangles + transfer plans
python -m app triangle    # execute the best triangle cycle
python -m app transfer    # plan transfers (--execute to run one)
python -m app balances    # balances per venue
python -m app trades      # recent trades
python -m app status      # mode, exchanges, risk, open transfers
python -m app start_auto  # auto-trading loop (Ctrl+C stops)
python -m app stop_auto   # stop auto trading (--kill engages the kill switch)
python -m app telegram    # run the Telegram bot (same services)
```

## Quick start (PAPER, zero configuration)

```bash
python -m pip install -e .            # + .[exchanges] for real venues
python -m app status                  # builds the database, connects
python -m app scan                    # finds simulated opportunities
python -m app triangle                # executes one simulated cycle
python -m app transfer --execute --wait   # full simulated transfer lifecycle
pytest                                # 100+ tests, no network needed
```

PAPER mode runs on deterministic simulated venues by default: books, balances
and network fees are reproducible from the venue id, so the whole bot works
offline and without API keys.

## Execution modes

| Mode   | Orders             | Funds       | Notes                                        |
|--------|--------------------|-------------|----------------------------------------------|
| PAPER  | simulated locally  | paper wallet| default; deterministic and testable          |
| DEMO   | venue demo/testnet | demo funds  | fail-closed demo routing (per-venue)         |
| LIVE   | production         | real capital| requires triple opt-in + withdrawal opt-in   |

DEMO routing is venue-specific and never falls back to production endpoints:
Binance uses its dedicated demo endpoints, OKX uses the
`x-simulated-trading` header, Bybit routes private calls to `api-demo.bybit.com`.
A venue whose URL table cannot be verified **refuses to start**.

LIVE additionally requires `CAT_TRADING__ALLOW_LIVE=true` and
`CAT_TRADING__LIVE_CONFIRMATION="I UNDERSTAND THE RISK"` — the configuration
itself refuses to load otherwise. Live withdrawals require
`CAT_TRADING__ALLOW_LIVE_WITHDRAWALS=true` on top of that.

## Safety systems

- **Risk engine** (fail-closed): max trade size, minimum net profit, max daily
  loss, max open transfers, exchange/asset exposure caps, slippage and
  market-data-age limits. Every execution passes validation before any order.
- **Kill switch**: `python -m app stop_auto --kill` stops all new execution
  instantly and persists across restarts; explicit release required
  (`start_auto --release`).
- **Execution recovery**: REJECTED / PARTIALLY_FILLED / TIMEOUT / UNKNOWN
  orders are re-queried from the venue; unresolvable outcomes escalate to
  MANUAL_REVIEW — never silently ignored.
- **Stale-data guard**: books older than the configured age are never traded.
- **Secret hygiene**: credentials are read from `CAT_KEY_*` env vars, redacted
  from every log line and never appear in errors.

## Telegram (secondary interface)

```bash
CAT_TELEGRAM__BOT_TOKEN=... CAT_TELEGRAM__ALLOWED_CHAT_IDS=12345 python -m app telegram
```

Commands: `/status /balances /opportunities /triangle /transfer /trades
/start_auto /stop_auto`.  Handlers reuse the same application services as the
CLI — no duplicated trading logic.  Chats not on the allow-list are refused.

## Configuration

Copy `.env.example` to `.env`.  Everything is a `CAT_`-prefixed environment
variable (pydantic-settings); see the file for the full annotated reference.

## Documentation

- `docs/ARCHITECTURE.md` — module layout and design decisions
- `docs/OPERATIONS.md` — runbook: modes, lifecycle, kill switch, recovery

## Tests

```bash
pytest          # full suite: strategies, risk, recovery, storage, lifecycle,
                # restart recovery, CLI and Telegram — all offline
```
