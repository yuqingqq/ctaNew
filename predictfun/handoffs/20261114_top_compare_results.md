# TOP-COMPARE: our current ladder against the top makers, on the same trades (results, 2026-11-14)

Session ce68ebe0. USER: "It will be a much larger cap, besides, how is current metrics compared with top makers".

Script `research/pf_top_compare.py` + `run_top_compare.sh` (6cf0d150, committed before its run). Run 08:39-08:41Z, claude slot. **POST HOC
and descriptive**: no readings, no verdicts.

## What is compared

- **Ours (simulated).** JOINT-LADDER v3's lanes, read from its saved fills with no re-run:
  - LDG (pull every rung on the guard) and FB25 (keep rule); OFB2 (perfect foresight) as the ceiling;
  - both sides, rungs 0-5 x 30 sh, caps hard 10 and none (USER: the real cap will be much larger);
  - fill clocks M 0 / 50 / 250, without fills on orders < 100 ms old (C1).
- **Theirs (real).** Every maker leg in the same windows' first 60 s on the 9 consumed days (1,573,290 legs). Makers are ranked per day by
  shares (GAP / B12's rule): top 5, ranks 6-20, rest.
- **The same for both:**
  - legs aligned to the simulator's anchors (checked on every window-side);
  - GAP's depth classifier, the one v3 used for our fills;
  - the value sg x (mid(anchor + 1 s or 5 s) - price), from the same mid series;
  - WS / E2 by taker order.
- Our 18 WS cells reconcile with v3's printed table.

## Results

**Per day** (first minute, 9 days):

| Who | Shares/day | $/day at 1 s | c/sh 1 s | c/sh 5 s | WS-timed shares/day | WS $/day | WS c/sh |
|---|---|---|---|---|---|---|---|
| Top 5 (real) | 851,396 | +3,610 | +0.424 | +0.317 | 596,108 | +3,471 | +0.582 |
| Top 5, per maker | 170,279 | +722 | +0.424 | +0.317 | 119,222 | +694 | +0.582 |
| Ranks 6-20 (real) | 790,142 | +2,813 | +0.356 | +0.332 | 496,485 | +2,884 | +0.581 |
| Rest (real) | 818,209 | -3,483 | -0.426 | -0.559 | 474,332 | -94 | -0.020 |
| Ours LDG, hard 10, M 0 | 32,898 | -2 | -0.005 | -0.066 | 19,220 | +74 | +0.383 |
| Ours FB25, hard 10, M 0 | 45,144 | -3 | -0.008 | -0.139 | 26,153 | +147 | +0.562 |
| Ours LDG, hard 10, M 50 | 36,387 | -101 | -0.278 | -0.338 | 20,677 | +47 | +0.226 |
| Ours FB25, hard 10, M 50 | 51,588 | -179 | -0.346 | -0.471 | 28,531 | +100 | +0.351 |
| Ours LDG, none, M 50 | 37,509 | -92 | -0.245 | -0.310 | 21,434 | +54 | +0.251 |
| Ours FB25, none, M 50 | 54,377 | -155 | -0.286 | -0.427 | 30,337 | +120 | +0.397 |
| Ours OFB2 (perfect), hard 10, M 50 | 59,212 | +295 | +0.498 | +0.635 | 37,181 | +335 | +0.900 |
| Ours LDG, hard 10, M 250 | 66,467 | -1,348 | -2.028 | -1.939 | 39,773 | -675 | -1.697 |
| Ours FB25, hard 10, M 250 | 106,366 | -2,306 | -2.168 | -2.091 | 61,896 | -1,100 | -1.777 |

**By depth at the fill** (WS-timed, 1 s; ours at M 50, hard 10; share of the group's WS shares, c/sh, $/day):

| Depth | Top 5 | Ranks 6-20 | Rest | Ours LDG | Ours FB25 | Ours OFB2 |
|---|---|---|---|---|---|---|
| At the best | 55.5%, **+0.378**, +$1,251 | 63.2%, +0.499 | 69.6%, -0.090 | 63.6%, **+0.004**, +$0.5 | 43.9%, +0.063, +$7.9 | 59.8%, +0.651 |
| 1 tick behind | 24.2%, **+0.559**, +$805 | 19.8%, +0.576 | 15.3%, -0.146 | 20.0%, **+0.522**, +$21.6 | 27.5%, +0.273, +$21.4 | 22.4%, +1.038 |
| 2+ behind | 17.9%, **+1.254**, +$1,340 | 12.2%, +1.076 | 11.4%, +0.519 | 11.4%, **+1.414**, +$33.3 | 24.6%, +1.151, +$80.6 | 15.3%, +1.959 |
| Inside the spread | 2.4%, +0.520 | 4.9%, +0.427 | 3.7%, +0.160 | 5.0%, -0.842 | 4.0%, -0.843 | 2.5%, -0.827 |

**Theirs by our guard ladder's state at the leg** (lag 50 ms; ON = our ladder pulls, OFF = it rests; WS-timed, 1 s):

| Who | ON: share, c/sh, $/day | OFF: share, c/sh, $/day |
|---|---|---|
| Top 5 | 51.5%, +0.281, +$862 | 48.5%, +0.901, +$2,607 |
| Ranks 6-20 | 36.7%, +0.061, +$111 | 63.3%, +0.882, +$2,773 |
| Rest | 51.6%, -0.697, -$1,708 | 48.3%, +0.704, +$1,611 |

**Theirs by trade type** (WS-timed): the top 5's at-best fills are 82.3% STOP (+0.479 c/sh) and 17.7% THROUGH (-0.087).
- Ours at the best: 37% STOP in BEST-GAP's rung-0 lane. v3's saved fills cannot be split by trade type.
- 1 tick behind, the top 5 are 66.8% STOP; 2+ behind, 65.1%.

## What it means

1. **Volume.**
   - The top 5 fill 851,396 sh a day in the first minute, 170,279 per maker.
   - Our whole ladder fills 36,387 (LDG) to 51,588 (FB25) at M 50, hard 10, on orders >= 100 ms old.
   - Replay volumes are inflated (C4), so the real gap is larger.
2. **Per share on WS-timed trades** at M 50, we are behind:
   - top 5 +0.582 c/sh;
   - LDG +0.226 and FB25 +0.351 (hard 10).
   - At M 0 FB25 is level with them (+0.562); at M 250 we lose -1.7.
3. **The gap is at the best.**
   - **At the best:** our fills earn +0.004 c/sh against their +0.378. Their at-best fills are 82% trades that stop at their level. Ours are
     mostly sweeps (BEST-GAP: queue place).
   - **Behind the best:** our per-share value matches theirs. 1 tick behind, LDG +0.522 against +0.559; 2+ behind, +1.414 against +1.254.
   - **FB25 is worse 1 tick behind (+0.273):** the orders it keeps are run over more often.
4. **Over all trades we lose and they do not.**
   - Their c/sh drops from +0.582 (WS-timed) to +0.424 (all trades). Ours drops from +0.226 to -0.278 (LDG) and from +0.351 to -0.346 (FB25).
   - The difference is the E2-timed trades: the replay's weakest timing (C12), and where our simulated fills lose most.
5. **When they earn.**
   - $2,607 of the top 5's $3,471 a day on WS-timed trades comes while our guard ladder rests (+0.901 c/sh).
   - While it pulls, they still make +0.281 c/sh on half their volume. The rest of the market loses -0.697 there.
   - Our guard separates good from bad flow for everyone. The top 5 are filled better than the rest even in the moments we pull.

## Limits

- Ours is simulated (rule M unvalidated, B1a; inflated volumes, C4); theirs is real (C9).
- Consumed days (C8); first minute only.
- Signer ids are per day: our own account, if it traded on these days, is among the ranked makers.
- Fees and rebates are not counted.
- No trade-type split for ours: v3's fills have no leg index.

Outputs:
- `data/predict_fun/research/top_compare_20261114/`: top_compare.txt, theirs_*.parquet, day_*.json, logs.
- Copies (txt / json / logs) in `reviews/top_compare_20261114/`.
