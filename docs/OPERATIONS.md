# Operations Runbook

## Daily operation

```bash
python -m app status      # first thing: mode, exchanges, kill switch, risk
python -m app scan        # what's out there right now
python -m app start_auto  # auto-trade triangles + drive transfers (Ctrl+C stops)
```

Auto trading also advances open transfer workflows every cycle. Transfers can
be driven without auto trading via `python -m app transfer --execute --wait`
or by running the Telegram bot (its maintenance loop ticks transfers).

## Modes

### PAPER (default)

Fully simulated execution against (simulated) order books; deterministic
balances per venue; the blockchain leg of transfers is simulated with
`CAT_TRANSFER__SIMULATED_TRANSFER_SECONDS`. Nothing can lose money.

### DEMO

Real API calls against each venue's demo environment using **demo keys**:

| Venue   | Routing                                                  |
|---------|----------------------------------------------------------|
| Binance | dedicated demo endpoints (`urls['demo']`)                |
| OKX     | `x-simulated-trading: 1` header on every private call    |
| Bybit   | private calls to `api-demo.bybit.com`, public stays prod |

The transfer blockchain leg stays **simulated** in DEMO (testnets cannot move
real funds between exchanges) — this is recorded on the transfer and in the
audit log. A venue that cannot prove its demo routing refuses to start.

### LIVE (explicit triple opt-in)

```
CAT_TRADING__MODE=LIVE
CAT_TRADING__ALLOW_LIVE=true
CAT_TRADING__LIVE_CONFIRMATION="I UNDERSTAND THE RISK"
```

Configuration loading **fails** without all three. Live withdrawals (transfer
arbitrage) additionally require `CAT_TRADING__ALLOW_LIVE_WITHDRAWALS=true`.
Re-calibrate every risk limit to your real capital before going live.

## The kill switch

Engage: `python -m app stop_auto --kill --reason "..."` (or breach the daily
loss limit — it engages automatically).

- stops ALL new execution immediately (every leg re-checks it);
- persists across restarts — rebooting the bot does NOT re-enable trading;
- blocks `start_auto` until explicitly released.

Release: `python -m app start_auto --release` (then Ctrl+C the loop, or let
it run). Release is a deliberate operator action by design.

## Transfer lifecycle

States: `CREATED → BUY_SUBMITTED → BUY_FILLED → WITHDRAW_SUBMITTED →
WITHDRAW_PENDING → TRANSFER_IN_PROGRESS → DEPOSIT_DETECTED → SELL_SUBMITTED →
COMPLETED`, with `FAILED` (understood error) and `MANUAL_REVIEW` (uncertain
state) from any step.

Inspect with `python -m app status` (open transfers) or `python -m app
transfer --asset X --amount N` (planning only). Restarting the app mid-flight
is safe: open transfers are resumed from the database on startup.

`MANUAL_REVIEW` means the bot could not establish the truth (withdrawal
stuck past the deposit timeout, sell leg unrecoverable, venue unreachable
during recovery). Check the venue's order/withdrawal history, resolve
manually, then record the outcome — the transfer stays flagged until then.

## Execution recovery

The recovery component handles the four problematic outcomes:

- **REJECTED** — final; nothing to recover.
- **PARTIALLY_FILLED** — the venue is re-queried for the authoritative fill;
  triangles continue with what is actually held.
- **TIMEOUT / UNKNOWN** — the order is looked up by exchange id, then by
  client id in the open-orders list. Not found anywhere → marked
  never-placed (manual review note). Venue unreachable after 3 attempts →
  manual review.

## Risk limits (fail-closed)

| Limit                    | Env var                              | Default |
|--------------------------|--------------------------------------|---------|
| Max trade size           | `CAT_RISK__MAX_TRADE_SIZE`           | 1000    |
| Min net profit           | `CAT_RISK__MIN_NET_PROFIT_BPS`       | 10 bps  |
| Max daily loss           | `CAT_RISK__MAX_DAILY_LOSS`           | 250     |
| Max open transfers       | `CAT_RISK__MAX_OPEN_TRANSFERS`       | 3       |
| Max exchange exposure    | `CAT_RISK__MAX_EXCHANGE_EXPOSURE`    | 150000  |
| Max asset exposure       | `CAT_RISK__MAX_ASSET_EXPOSURE`       | 750000  |
| Max slippage             | `CAT_RISK__MAX_SLIPPAGE_BPS`         | 15 bps  |
| Max market-data age      | `CAT_RISK__MAX_DATA_AGE_MS`          | 2500 ms |

## Telegram

```
CAT_TELEGRAM__BOT_TOKEN=<token from @BotFather>
CAT_TELEGRAM__ALLOWED_CHAT_IDS=<your chat id>
python -m app telegram
```

Commands: `/status /balances /opportunities /triangle /transfer /trades
/start_auto /stop_auto`. `/transfer ASSET AMOUNT` executes a plan;
`/transfer` alone only lists plans. Only allow-listed chats may command the
bot — with an empty list every command is refused (fail-closed).

## Credentials

Per venue: `CAT_KEY_<VENUE>_APIKEY`, `CAT_KEY_<VENUE>_SECRET`, and
`CAT_KEY_<VENUE>_PASSWORD` for OKX's passphrase. Credentials are redacted
from logs, never stored in the database, and never echoed by the CLI.

## Database

SQLite at `data/bot.db` by default; schema is created automatically, and a
file with a drifted (old) schema is renamed `.bak` and rebuilt. For
PostgreSQL set `CAT_DATABASE__URL=postgresql+asyncpg://...`. Retention:
`CAT_DATABASE__TRADE_RETENTION_DAYS` / `CAT_DATABASE__AUDIT_RETENTION_DAYS`.

## Troubleshooting

- **No opportunities**: check `status` market-data counts; raise watchlist or
  loosen minimums; in PAPER the simulated venues always carry edges.
- **"kill switch engaged"** in every command: release it deliberately
  (`start_auto --release`).
- **Venue shows DATA_ONLY**: its keys were rejected (usually demo keys on a
  production profile or vice versa) — public data still flows, private calls
  are blocked until valid keys are provided.
- **Transfer stuck in TRANSFER_IN_PROGRESS**: past
  `DEPOSIT_TIMEOUT_SECONDS` it escalates to MANUAL_REVIEW automatically.
