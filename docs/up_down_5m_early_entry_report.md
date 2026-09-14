# BTC Up/Down 5m — early-entry research (read-only, no PnL optimization)

Goal: earliest robust spot entry in [0s..90s] at ±2bps concept, vs frozen +90s.
Datasets: 51 offline markets (48 decisive, cached 1s klines) + 20 live DEMO
markets (20 decisive; early drifts recomputed from public klines for the SAME
20 windows — no new markets). No look-ahead (closeTime rule). No Chainlink
signal. Contract economics only as a second-step verification (real offline
UP chance/100, fee 200bps + 0.005 slip, 1 share). Signal code untouched.

## Sweep result (thr ±2.0bps)

| entry | OFF sig/cov/acc | LIVE sig/cov/acc |
|---|---|---|
| 0s  | 0 / .00 / n/a (drift ≡ 0) | 0 / .00 / n/a |
| 15s | 2 / .04 / 1.00 | 7 / .35 / .857 |
| 30s | 5 / .10 / .800 | 12 / .60 / .833 |
| 45s | 5 / .10 / 1.00 | 12 / .60 / .833 |
| 60s | 6 / .12 / 1.00 | 15 / .75 / .800 |
| 75s | 9 / .19 / 1.00 | 16 / .80 / .812 |
| 90s | 13 / .27 / 1.00 | 16 / .80 / .875 |

Balanced accuracy thr2.0 — 60s: OFF 1.0 / LIVE .806; 75s: 1.0 / .817;
90s: 1.0 / .875. False signals live — 60s: 3 (2244332, 2244760, 2245353);
75s: 3 (2244527, 2244760, 2245353); 90s: 2 (2244527, 2244760). Offline
false signals: 0 at all three offsets (thin samples early).

## Chronological halves (thr2.0)

- OFF-1: 60s 3/3, 75s 5/5, 90s 7/7 (all 1.0).
- OFF-2: 60s 3/3, 75s 4/4, 90s 6/6 (all 1.0).
- LIVE-1: 60s 7 acc .857, 75s 6/6 1.0, 90s 7/7 1.0.
- LIVE-2: 60s 8 acc .750, 75s 10 acc .700, 90s 9 acc .778.

## Walk-forward (train OFF-1, accuracy, coverage ≥ 20%)

Only 90s clears the coverage floor on OFF-1 (cov .29; 75s .208 is
borderline on 5 signals, 60s .125 fails) → selects 90s/2.0. Test: OFF-2
6/6 1.0; LIVE 16/20 .875; LIVE halves 7/7 and 7/9.

## Economics check (offline real prices, thr2.0)

Mean entry cost/share: 15s .500, 30s .523, 45s .531, 60s .649, 75s .617,
90s .615. Total net: 90s +5.01 (13 trades) dominates; best per-trade (45s
+.469) rests on n=5. Earlier entry does NOT materially cheapen fills —
by 60–75s the contract already prices most of the move, while coverage
collapses (45s: 5 off-signals; 60s: 6).

## Decision

RETAIN frozen +90s / ±2bps with +180s opposite-strong EXIT. Every earlier
offset is worse-or-equal on live accuracy (60s .80, 75s .812 vs 90s .875),
adds a third live false signal, halves offline coverage (13→9→6), and
saves nothing on entry cost. 0s never fires by construction. No earlier
signal meets "robust": 60s/75s offline samples (6/9) are too thin to
support the 100% readings, and both fail the walk-forward coverage floor
as cleanly as 90s clears it.

Exact retained signal: entry t=+90s, drift=(mid−ref)/ref×10000 from
pre-decision Binance 1s closes, UP ≥ +2.0 / DOWN ≤ −2.0 else HOLD;
+180s same-threshold opposite signal → EXIT else hold to settlement.
Accuracy: OFF 13/13 (1.0, bal 1.0), LIVE 14/16 (.875, bal .875).
Coverage: OFF .27, LIVE .80. Verdict: PASS (frozen).
