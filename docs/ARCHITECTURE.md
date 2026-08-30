# Architecture

```
app/
├── cli/            argparse entry point (python -m app <command>)
├── telegram/       Bot API long-poll client + handlers (interface only)
├── exchanges/      unified adapters: ccxt (real venues), simulated (PAPER),
│                   manager (breaker + auth gate), profiles, credentials
├── market_data/    WebSocket-first store/service + order-book math
├── strategies/
│   ├── triangular/ depth-aware scanner + sequential 3-leg executor
│   └── transfer/   planner, network validation, lifecycle orchestrator
├── execution/      order gates, kill-switch guard, fill simulator,
│                   paper wallet, precision filters
├── risk/           rule engine (fail-closed) + runtime state
├── recovery/       REJECTED / PARTIALLY_FILLED / TIMEOUT / UNKNOWN resolver
├── storage/        SQLAlchemy async: trades, transfers, balances, audit log,
│                   bot state
├── models/         frozen pydantic domain models
├── config/         settings (pydantic-settings), mode policy, logging
├── services.py     AppServices: the single composition root
├── auto.py         auto-trading loop
├── clock.py        injectable clock (deterministic tests)
└── errors.py       typed error hierarchy
```

## Principles

1. **One composition root.** `app/services.py` builds `AppServices`; the CLI
   and Telegram are thin interfaces over it. No trading logic lives in either
   interface layer.
2. **Fail closed everywhere.** Unknown venues refuse to instantiate; DEMO
   routing refuses unverifiable URL tables; a venue without a testnet is
   DATA-ONLY; risk rules that crash block the trade; withdrawals require a
   separate LIVE opt-in; an empty Telegram allow-list refuses all commands.
3. **Decimal for money.** Every price, amount, fee and P&L figure is
   `Decimal`. The fill simulator charges fees in the *received* currency so
   legs never mix currencies; reports convert cross-currency fees explicitly.
4. **Freshness is a hard gate.** Books carry timestamps; the store answers
   "how old is this?"; scanners filter stale data; executors re-check every
   book immediately before use and refuse to trade on it.
5. **State survives restarts.** Transfers persist every lifecycle step before
   the next one runs; `resume()` picks them up on boot. The kill switch and
   auto-trading flag persist in `bot_state`. Daily P&L is rebuilt from the
   trades table.

## Strategy: triangular arbitrage

`strategies/triangular/scanner.py` (preserved from the original codebase)
walks a venue's whole spot graph: USDT-quoted books may start/end a cycle and
cross books (e.g. `ETH/BTC`) connect them in either orientation. Each leg is
priced with a depth-aware VWAP walk in trade direction; the walk rounds
amounts **down** onto instrument steps so the published route is placeable
verbatim. A cycle is reported only when every leg stays inside the per-leg
slippage tolerance and the net clears `triangle_min_net_bps` after
`(1 - taker)^3` fees.

`executor.py` runs the three legs sequentially:

- PAPER: each leg is filled by `execution/fill_simulator.py` against the live
  book snapshot; funds move in per-venue paper wallets;
- DEMO/LIVE: real market orders through the adapter with per-leg timeouts;
  uncertain outcomes go through `recovery/`.

Before any leg, a non-destructive preview of the whole cycle must clear
`min_net_at_execute_bps` — if profitability is uncertain, nothing is placed.
Mid-cycle failures unwind what is held back to USDT; a failed unwind lands in
MANUAL_REVIEW.

## Strategy: transfer arbitrage

A long-running workflow, deliberately NOT a blocking function:

```
CREATED -> BUY_SUBMITTED -> BUY_FILLED -> WITHDRAW_SUBMITTED ->
WITHDRAW_PENDING -> TRANSFER_IN_PROGRESS -> DEPOSIT_DETECTED ->
SELL_SUBMITTED -> COMPLETED        (FAILED / MANUAL_REVIEW from any step)
```

- `planner.py` — pure economics: ask/bid depth pricing, trading fees,
  withdrawal fee, network costs, minimum-profit gate.
- `networks.py` — the safety layer: networks are matched by **unified code**
  (never display name), withdrawals and deposits must be enabled on both
  sides, and the cheapest common network wins. No provably-safe network means
  no transfer — no guessing.
- `orchestrator.py` — the state machine. `start()` executes buy + withdrawal
  submission; `tick()` advances every open record (called by the auto loop,
  the Telegram maintenance loop, or `transfer --wait`). PAPER/DEMO simulate
  the blockchain leg (DEMO clearly flags this — testnets cannot move real
  funds between exchanges); LIVE performs real withdrawals.

## Market data

`market_data/streams.py` supervises WebSocket watch streams with restart +
exponential backoff + quarantine (a stream that never delivers hands its
symbols back to REST). The service's REST refresh covers every pair not
served by a *delivering* stream, and an inactivity watchdog revokes coverage
from streams that silently stop delivering. Venue failures are isolated; the
per-venue circuit breaker skips tripped venues entirely.

## Storage

Five tables (`storage/tables.py`): `trades`, `transfers`, `balances`,
`audit_log`, `bot_state`. SQLite by default (`create_all` with drift
detection — a legacy file is renamed `.bak`, never corrupted), PostgreSQL for
production. Repositories translate between pydantic domain models and rows;
nothing else touches the ORM. Money columns use exact decimal storage
(strings on SQLite, NUMERIC on PostgreSQL).

## Risk

`risk/rules.py` holds one pure rule per protection; the engine evaluates all
of them and converts a crashing rule into a CRITICAL violation (fail closed).
The runtime builds `RiskContext` from live state: kill switch, today's
realised P&L (rebuilt from the trades table at startup), open transfers and
exposure maps derived from in-flight transfers.

## Modes

`config/modes.py` defines the capability matrix; `execution/order_gate.py`
turns it into per-adapter placement policies evaluated at order time:

- PAPER → `never_place_orders`
- DEMO → `sandbox_only` (no sandbox flag == no placement)
- LIVE → enabled guard + released kill switch
