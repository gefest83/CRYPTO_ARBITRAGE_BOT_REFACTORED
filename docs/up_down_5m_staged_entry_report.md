# BTC Up/Down 5m — staged pre-positioning research (read-only, no PnL optimization)

Question: enter EARLIER than +90s (probe) while keeping frozen +90s confirmation?
Frozen +90s signal itself UNCHANGED throughout. No Chainlink signal. No
fair-value/mispricing (probe/confirm use spot drift only; prices used solely
to MEASURE fills, never to decide). +180s EXIT concept kept (applies to final
positions). Datasets: 51 offline (48 decisive) + 20 live DEMO (20 decisive);
no new collection (live early drifts recomputed once from public klines for
the same 20 windows in prior research; reused here).

## Candidates (0.5u probe + 0.5u add on confirm; scratch probe if unconfirmed)

| probe | OFF conf W-L / scratch | LIVE conf W-L / scratch |
|---|---|---|
| frozen 90s-only (1.0u) | 13, 13-0 / 0 | 16, 14-2 / 0 |
| 60s/1.0 | 5, 5-0 / 7 | 12, 11-1 / 5 |
| 60s/1.5 | 5, 5-0 / 2 | 12, 11-1 / 4 |
| 60s/2.0 | 5, 5-0 / 1 | 12, 11-1 / 3 |
| 45s/1.5 | 5, 5-0 / 1 | 10, 10-0 / 5 |
| 75s/1.5 | 9, 9-0 / 5 | 15, 13-2 / 3 |
| 60s/1.5 + expanding-move filter | 4, 4-0 / 1 | 11, 10-1 / 2 |

Chrono halves probe45/1.5 (best-looking): OFF 2/2 + 3/3, LIVE 4/4 + 6/6 —
perfect but n≤6 per half; LIVE-1 needed 4 scratches to bank 4 confirmed
(50% churn). Probe60/1.5 halves: OFF 2/2 + 3/3, LIVE 5/5 + 7/6-1.

## Why every staged variant is rejected

1. Confirmation does all the work: staged confirms strictly FEWER positions
   than frozen (OFF 13→5, LIVE 16→10..12) while keeping ≥1 live loss in all
   but the thinnest config. No variant raises confirmed accuracy above frozen
   on samples that matter (probe45's 10/10 has Wilson LB ~72%, overlapping
   frozen's 14/16; built from 6 tried configs = selection luck).
2. Economics fail on measurement: paired offline fills show 60s and 90s legs
   cost IDENTICALLY (Δ=0.000 on all 5 confirmed markets — 1-minute price
   resolution), so blending cannot cheapen entry with existing data; staged
   offline net is 1.54–3.23 on 5–9 confirmed vs frozen 5.01 on 13, plus
   scratch bleed. Live early-leg prices don't exist (books only at 90s/180s),
   so savings are unverifiable where they would matter.
3. False-signal churn rises: 1–7 offline and 1–5 live scratches per config vs
   frozen's zero; stricter exits already proven harmful in prior research.

## Decision

RETAIN frozen 90s-only 1.0u entry (UP ≥ +2 / DOWN ≤ −2 else HOLD; +180s
opposite-strong EXIT; hold winners). Coverage OFF .27 / LIVE .80; accuracy
OFF 13/13 (1.0) / LIVE 14/16 (.875). Best staged idea (probe 45s/1.5 +
90s confirm) documented above but NOT adopted — complexity + churn +
   selection risk for no measurable gain. Verdict: PASS (frozen retained).
