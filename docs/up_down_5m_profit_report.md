# BTC Up/Down 5m — drift-persistence profit report (research-only)
Markets: 49 settleable (48 decisive UP/DOWN + 1 PUSH). Chronological, no look-ahead (closeTime rule), fee 200bps + 0.005 slippage/share.
## Exact signal rules (frozen `DriftConfig`)
- entry_offset_ms=90000, drift_thr_bps=2.0, exit_offset_ms=180000, fee_bps=200, slippage=0.005, gates warming=60000/too-late=120000
- drift_bps=(mid_now-ref)/ref*10000; ref=Binance 1s close at/before start, mid=close at/before entry (pre-decision only, never Chainlink).
- drift>=+thr => UP (buy UP at UP chance/100); drift<=-thr => DOWN (buy DOWN at 1-UP); else HOLD.
- reversal EXIT: same thr at exit_offset with opposite sign => exit at then-current contract price; else HOLD until settlement ($1 win, $0 loss, $0.5 PUSH).
- No fair-value/mispricing, no profit-taking, no Chainlink signal.
## Full-sample backtest (hold+exit, 49 markets)
- trades=13 coverage=0.265 wins=13 win_rate=1.0 total_net=5.0083 avg_net=0.3852538461538461538461538462 avg_win=0.3852538461538461538461538462 avg_loss=None max_dd=0
## Walk-forward (train selects thr on train only, test next 12)
- split=24 train_thr=1.0 train(net=3.4358,tr=16,wr=0.8125) -> test(tr=5,wr=0.6,net=0.1491)
- split=36 train_thr=1.0 train(net=3.5849,tr=21,wr=0.7619047619047619) -> test(tr=5,wr=0.8,net=0.9553)
## Split-half (frozen thr 2.0): first(net=2.6089,tr=7,wr=1.0) second(net=2.3994,tr=6,wr=1.0)
## Baselines @90s: always-UP(net=2.7909,wr=0.571) always-DOWN(net=-4.2707,wr=0.408)
## Sensitivity (thr @90s)
- thr=1.0: tr=27 wr=0.7777777777777778 net=4.9945 dd=1.3018
- thr=1.5: tr=18 wr=0.8333333333333334 net=4.2424 dd=1.1390
- thr=2.0: tr=13 wr=1.0 net=5.0083 dd=0
- thr=2.5: tr=8 wr=1.0 net=3.0224 dd=0
## Sensitivity (offset @thr2.0)
- off=60000: tr=6 wr=1.0 net=2.1036
- off=90000: tr=13 wr=1.0 net=5.0083
- off=120000: tr=16 wr=0.875 net=2.3618
- off=180000: tr=20 wr=0.85 net=-0.2176
## Permutation (frozen rule, 2000 shuffles): p(net>=obs)=0.0000 (0/2000), obs_net=5.0083
## Overfitting notes
- Single feature / single threshold / two timings (simplest plateau: thr 1.0-2.5 all profitable @90s).
- LOO: removing any 1 of 13 trades keeps 13->12 trades at 100% wr.
- Limitation: n=13 trades over ~4h single-regime window, UP-biased sample (28U/20D); true win-rate 95% Wilson LB ~75% still implies positive expectancy at avg payoff, but out-of-regime performance is unproven. Size accordingly.
## Verdict: PASS
