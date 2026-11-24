# Quote, accumulate, actively reduce — bounded pilot

Frozen before replay: `7626e320f`. USER-directed follow-up to maker3 sizing/inventory audit. Same three consumed dashboard markets; opening-only first-batch rule retained throughout. Four complete feedback arms, no threshold tuning after the outputs. Twelve ledgers; the three PASSIVE_100 controls exactly reproduce the accepted OPENING_ONLY fills, orders, commands, requests, cash, position and modes.

100shares/level uses cap200, soft100/release50. 220shares/level uses cap440, soft220/release110. The220 sensitivity scales risk limits too; it is not a pure order-size effect. Both use two levels and the same pressure/structural/entry rules. Maker3 private cap is unknown; these are our experimental risk limits.

Active arm: when abs(client-known inventory)>=soft, submit an IOC toward the release level, capped at one order size and reservation room. Preserve reducing-side resting orders; do not bypass cancel-pending/own-cross protections. Re-evaluate changed state, no fixed cooldown. IOC fills pay saved registry200bps * min(p,1-p), collateral-equivalent fee; passive fee0 as before, no rebates.

| Market | Passive100 | Take100 | Passive220 | Take220 |
|---|---:|---:|---:|---:|
| 3702960 | $+5.41 | $-27.53 | $+44.78 | $-18.32 |
| 3789484 | $+9.73 | $-18.63 | $+66.40 | $-6.08 |
| 3877472 | $-136.46 | $-125.59 | $-272.19 | $-293.75 |
| **Pooled** | **$-121.32** | **$-171.75** | **$-161.00** | **$-318.15** |

Values are cash + remaining inventory at the common298s END mark, minus modeled fees. They are NOT binary settlement P&L, live achievable returns or untouched validation.

| Arm | Completed FIFO turnover, gross USD | Residual, gross USD | Taker fees USD | Mean absolute inventory | Taker shares / filled IOC events |
|---|---:|---:|---:|---:|---:|
| PASSIVE_100 | -100.27 | -21.05 | 0.00 | 79.1 | 0.0 / 0 |
| TAKE_100 | -158.51 | +1.79 | 15.03 | 53.6 | 2987.1 / 53 |
| PASSIVE_220 | -139.83 | -21.17 | 0.00 | 149.7 | 0.0 / 0 |
| TAKE_220 | -293.49 | -6.66 | 18.00 | 115.3 | 3617.8 / 36 |

**Computed reading:** {"all_pooled_negative": true, "taking_worse_pooled_both_sizes": true}.

At size100, takes change net value by$-50.43: gross completed-turnover change$-58.24, marked-residual change$+22.85, less taker fees. Mean inventory falls: True; 1/3 markets improve.
At size220, takes change net value by$-157.15: gross completed-turnover change$-153.65, marked-residual change$+14.51, less taker fees. Mean inventory falls: True; 0/3 markets improve.

Both active variants remain worse before fees too; fees alone do not explain the gap. The exposure reduction changes the subsequent passive fills and closing opportunities. This experiment rejects this simple automatic soft-limit-to-release rule on these selected examples; it does not show that selective maker3-style takes cannot work. The 220 passive result also mixes a higher cap/soft target with size and is still pooled negative. No larger-size or active-take promotion follows.

**Execution reliability and limits.** All124 requests independently reconstructed from hash-bound venue frames/trades. Same10ms arrival,100ms fill/result notice; limit from received best, no worse-price sweep, execution at venue best only if fresh/valid and available. Net-depth budget prevents reusing unchanged displayed liquidity; intervening public trades reduce availability. Positive net depth changes can replenish. This can miss gross replenishment and does not model reactions/market impact: historical external flow remains exogenous. It is a conservative liquidity convention, not a mathematical profit bound or measured exchange execution. Native token fee balances are not modeled; fees are charged as cash-equivalent cost.

Fifteen synthetic control groups cover both directions, latency, partial/empty/refused/self-cross fills, replenishment/public-volume depletion, delayed notices, reserve bounds, hidden-fill risk, unchanged-state retry and FIFO/fees. Independent checking covers all12 ledgers and6 corruptions; separate raw-tape execution checker covers124requests and6 availability corruptions. The first independent checker counted post-only-refused orders as self-cross liquidity; corrected checker and raw refused-order diagnostic are retained (no strategy or replay changed). All actual filled takes reduce absolute venue inventory in these twelve runs; the synthetic hidden-fill case demonstrates this is not guaranteed by a decision based on known I.

**Next priority:** do not tune another inventory threshold from these results. Explain the costly active closures against the passive continuation on the same saved paths, and compare the known-state situations with maker3’s small/selective trims. The missing rule is the economic choice between paying to reduce now and continuing to seek passive offsetting fills. The larger maker3 exposure range alone is not a reason to expand our cap. Keep whole-book turnover/residual and costs separate, and preserve the opening-only scope.

Double-fee stress, same fills: TAKE_100 $-186.78, TAKE_220 $-336.16. This is an accounting stress, not another feedback run.

Resource charge upper: 110CPU seconds of 720; all failures and checks included. Standing USER small-CPU authorization; no push or live use.
