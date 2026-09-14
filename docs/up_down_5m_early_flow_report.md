# BTC Up/Down 5m — early-flow +20s report (research-only)
Markets: 51 with full-window klines+series (48 decisive UP/DOWN + 3 PUSH). Chronological, no look-ahead (closeTime rule), fee 200bps + 0.005 slippage/share.
## Exact signal rules (frozen `EarlyFlowConfig`)
- entry_offset_ms=20000, flow_thr=0.25, flow_window_ms=30000, exit_offsets_ms=[90000, 180000], exit_drift_thr_bps=2.0, fee_bps=200, slippage=0.005
- flow=(taker_buy-(vol-taker_buy))/vol over prior 30s from Binance 1s klines (pre-decision only, never Chainlink); flow>=+thr => UP (buy UP at UP chance/100); flow<=-thr => DOWN (buy DOWN at 1-UP); else HOLD.
- reversal EXIT: spot drift (mid-ref)/ref*10000 at +90s/+180s opposes held side beyond 2.0bps => exit at then-current contract price; else HOLD until settlement ($1 win, $0 loss, $0.5 PUSH). Winners never exited early in sample (2 exits, both loss-reducing).
- No Chainlink signal, no fair-value/mispricing (contract price only as execution cost/salvage).
## Full-sample backtest
- all: trades=40 coverage=0.784 wins=28 win_rate=0.700 total_net=8.6468 avg_net=0.21617 max_dd=1.1118 exits=2
- decisive-only: trades=38 coverage=0.792 win_rate=0.737 total_net=8.6872 max_dd=1.1118
## Chronological regimes (frozen)
- half1(net=3.6932,tr=18,wr=0.667) half2(net=4.9536,tr=22,wr=0.727)
- q1(net=1.4001,tr=7,wr=0.571)
- q2(net=1.7980,tr=10,wr=0.700)
- q3(net=2.3233,tr=9,wr=0.778)
- q4(net=2.6862,tr=12,wr=0.750)
- roll[0:24](net=3.1981,tr=17,wr=0.647)
- roll[6:30](net=3.6821,tr=17,wr=0.706)
- roll[12:36](net=4.1213,tr=19,wr=0.737)
- roll[18:42](net=3.9891,tr=21,wr=0.714)
- roll[24:48](net=5.0095,tr=21,wr=0.762)
- balanced20U20D s=0(net=5.8186,tr=32,wr=0.688)
- balanced20U20D s=1(net=7.8594,tr=32,wr=0.750)
- balanced20U20D s=2(net=5.8386,tr=30,wr=0.700)
- balanced20U20D s=3(net=7.7778,tr=32,wr=0.750)
- balanced20U20D s=4(net=6.3439,tr=31,wr=0.710)
## Walk-forward (frozen, test next 12)
- split=24: test(net=2.3233,tr=9,wr=0.778)
- split=36: test(net=2.6862,tr=12,wr=0.750)
## Walk-forward (train picks flow_thr, test next 12)
- split=24 pick=0.10 train(net=3.8116,tr=20) -> test(net=1.2115,tr=11,wr=0.636)
- split=36 pick=0.25 train(net=5.5214,tr=26) -> test(net=2.6862,tr=12,wr=0.750)
## Baselines @20s: always-UP(net=2.9545,wr=0.549) always-DOWN(net=-4.4947,wr=0.392)
## Sensitivity (flow_thr)
- thr=0.10: tr=45 wr=0.667 net=8.1485
- thr=0.15: tr=43 wr=0.651 net=7.1889
- thr=0.25: tr=40 wr=0.700 net=8.6468
- thr=0.40: tr=36 wr=0.722 net=8.2174
## Sensitivity (entry offset @thr0.25)
- off=15000: tr=41 wr=0.585 net=4.4379
- off=20000: tr=40 wr=0.700 net=8.6468
- off=30000: tr=42 wr=0.667 net=7.5328
- off=45000: tr=42 wr=0.667 net=7.9938
- off=60000: tr=41 wr=0.634 net=3.4577
## LOO net range 8.1211-9.2741 (n=51)
- drop-top1: net=8.1211 tr=39 wr=0.692
- drop-top3: net=7.0901 tr=37 wr=0.676
- drop-top5: net=6.0999 tr=35 wr=0.657
## Permutation (frozen, 2000 shuffles): p(net>=obs)=0.0015 (3/2000), obs_net=8.6468
## Fee stress
- fee200+slip0.005: net=8.6468 wr=0.700
- fee300+slip0.01: net=8.2192 wr=0.700
- fee400+slip0.01: net=8.0056 wr=0.700
- fee500+slip0.015: net=7.57200 wr=0.700
## Verdict: PASS
