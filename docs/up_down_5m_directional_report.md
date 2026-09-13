# BTC Up/Down 5m — directional re-validation (research-only, no PnL)
Dataset: exact 51 report markets = 51 total, 48 decisive (UP=28, DOWN=20), PUSH=3 excluded from accuracy. No fees/slippage/contract-price/PnL. No Chainlink signal.
## Current strategy (frozen)
- entry +90s; drift_bps=(mid-start)/start*10000 (Binance 1s closeTime, pre-decision); UP if drift>=+2, DOWN if drift<=-2, else HOLD; exit check +180s same thr: opposite=>EXIT else hold to settlement.
- signals=13 coverage51=0.255 coverage_dec=0.271 acc=1.0 up_acc=1.0 down_acc=1.0 bal_acc=1.0
## Threshold sweep @+90s
- thr=1.0: sig=27 cov51=0.529 acc=0.8148148148148148 up=0.8666666666666667 dn=0.75 bal=0.8083333333333333
- thr=1.5: sig=18 cov51=0.353 acc=0.8888888888888888 up=0.8888888888888888 dn=0.8888888888888888 bal=0.8888888888888888
- thr=2.0: sig=13 cov51=0.255 acc=1.0 up=1.0 dn=1.0 bal=1.0
- thr=2.5: sig=8 cov51=0.157 acc=1.0 up=1.0 dn=1.0 bal=1.0
- thr=3.0: sig=4 cov51=0.078 acc=1.0 up=1.0 dn=1.0 bal=1.0
## Entry-time sweep @thr2.0
- off=60000: sig=6 cov51=0.118 acc=1.0 bal=1.0
- off=90000: sig=13 cov51=0.255 acc=1.0 bal=1.0
- off=120000: sig=16 cov51=0.314 acc=0.875 bal=0.8888888888888888
- off=180000: sig=20 cov51=0.392 acc=0.85 bal=0.8636363636363636
## Chronological break-tests (frozen +90s/2.0)
- half1 n=24: sig=7 acc=1.0 bal=1.0
- half2 n=24: sig=6 acc=1.0 bal=1.0
- q1 n=12: sig=5 acc=1.0
- q2 n=12: sig=2 acc=1.0
- q3 n=12: sig=1 acc=1.0
- q4 n=12: sig=5 acc=1.0
## UP/DOWN-balanced subsets (20U+20D, 5 seeds)
- seed0: sig=12 acc=1.0 bal=1.0
- seed1: sig=12 acc=1.0 bal=1.0
- seed2: sig=12 acc=1.0 bal=1.0
- seed3: sig=11 acc=1.0 bal=1.0
- seed4: sig=11 acc=1.0 bal=1.0
## Rolling 24-decisive windows
- win[0:24]: sig=7 acc=1.0
- win[6:30]: sig=5 acc=1.0
- win[12:36]: sig=3 acc=1.0
- win[18:42]: sig=5 acc=1.0
- win[24:48]: sig=6 acc=1.0
## Walk-forward (train picks thr by accuracy, cov>=20%; test next 12)
- split=24 pick_thr=2.0 train(sig=7,acc=1.0) -> test(sig=1,acc=1.0,bal=None)
- split=36 pick_thr=2.0 train(sig=8,acc=1.0) -> test(sig=5,acc=1.0,bal=1.0)
## Frozen walk-forward (+90s/2.0, test next 12)
- split=24: test sig=1 acc=1.0
- split=36: test sig=5 acc=1.0
## Leave-one-out (frozen)
- LOO acc range 1.000-1.000 (all n-1 subsets)
## Removal of best trades (largest |drift| correct first)
- drop top1: sig=12 acc=1.0
- drop top3: sig=10 acc=1.0
- drop top5: sig=8 acc=1.0
## Exit consistency (+180s reversal check on the 13 signals)
- reversals=0/13, reversals_on_losers=0 (0 losers total)
## Permutation (frozen, 2000 shuffles, accuracy statistic)
- p(acc>=obs)=0.0000 (0/2000), obs_acc=1.0
## Worst subset result: half1 acc=1.0
## Verdict: PASS
Note: concept unchanged (single-feature drift); no improvement tested beats +90s/2.0 on reliable accuracy (thr1.5 higher coverage but lower acc; later entries higher coverage but lower/down-biased accuracy).
