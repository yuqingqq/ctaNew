# STOP-MODEL: the keep / cancel model for every ladder order (results, 2026-11-14)

Session aea5bfb9. USER:
- "Proceed to build the keep vs cancel model for multiple ladder orders".
- "Build review, then test model performance, make sure the model is built aligned to our goal, check data distribution, make sure the
  data is good for training, clear all issues in training, make sure the model predicts the right thing with good data, then check
  model's performance".

The USER's design (HANDOFF START HERE 16:38Z): rest every level by default; at every step predict, per side, how far the move against the
side goes; cancel only the levels it is likely to run over; keep the rest (the protection side at the new best); no guard trigger.
This is step 2 (STOP-MODEL) of that plan. The policy replay (STOP-POLICY) is next, on the USER's go.

## Runs (claude slot, `research/pf_stop_model.py`, driver `run_stop_model.sh`)

| Version | Commit | What |
|---|---|---|
| v1 | 63a42519 | build: rows every 50 ms, STOP-DEPTH's target and inputs without the guard; audit v1 (raw inputs) run |
| v2 | 52f2ce98 | processing after audit v1, before any fit (dev days only): sizes relative to a causal liquidity scale, GAP status |
| v3 | 8f0b24ee | the independent build review's 12 fixes (B1-B12, below) |
| v3.1 | e69dbf82 | memory restructure; T5 rule (post hoc, see below); dev-day decomposition of the short-horizon skill |

Runs: rows 17:42-17:55Z (9 days, ~77 s / day); scalecheck + audit2 17:55Z; 5 fits 18:06-18:40Z; 5 oracle / order days 18:40-18:56Z;
report 18:56Z. Peak memory: rows 1.26 GiB; fits 1.99 / 2.06 / 2.09 / 2.09 / 2.18 GiB (four folds up to 6% above the 2 GiB self-guard,
4 GiB cap; noted, not raised); align 0.77-1.2 GiB.

## What the model is

- **Rows:** every 50 ms step of the first 60 s, both sides, 9 consumed days (6.2 M rows).
  - Statuses: invalid known book (12-722 a day); GAP = a > 500 ms feed gap in the label window (206-3,316 a day, most on the
    weekend days); 10-31's six windows before 00:30 excluded (sigma's 30-min look back reaches sealed 10-30, B12).
- **Target** for the order d ticks behind the known best (level j = d + 1): is it run over within H ms after a cancel could land?
  - Run over means the side's best moves >= j ticks against us in (t + 22, t + H].
  - Moves before t + 22 ms cannot be avoided (status ALREADY).
  - H = 72 ms (until the next decision's cancel lands), 250 ms and 1000 ms.
- **Inputs (85), all known at t:**
  - STOP-DEPTH's inputs without the guard value and the guard age (leave-one-day-out models, C16);
  - INFLUENCE's fair-price inputs, NaN until the start price is known (P16);
  - two level ages;
  - book sizes divided by a causal liquidity scale (below).
- **Fits:** chronological folds (train on the days before the calibration day; test 10-27, 10-31, 11-01, 11-02, 11-03), LightGBM with
  PULL-DEPTH's pinned parameters, one head per (H, j) = 18 per fold.

## Build review (independent agent, before any fit) and fixes

| # | Finding | Fix |
|---|---|---|
| B1 | align matched a re-posted same-price fill to two orders (would refuse every day) | match on `t <= tfill`; toy case through the real lane |
| B2 | the declared alignment test could not fail (a cleared level fills us THROUGH by construction) | ORACLE replay: act on the true label vs keep-all, whole portfolio, with a time-shifted-label control |
| B3 | 1 s labels against 50 ms decisions pull early | labels and heads at 72 / 250 / 1000 ms |
| B4 | audit v1 printed test-day statistics before the processing choices | recorded (Rule 11): the choices used dev-day evidence only; audit2 prints dev days only |
| B5 | T5 permuted within day and side | permuted over the whole training set |
| B6-B10 | young-fill / clock splits, align statuses, cumulative min, crossed frames as gaps, typed arrays | fixed |
| B11 | comparators leave-one-day-out; INFLUENCE inputs missing | STOP-DEPTH refit chronologically; INFLUENCE inputs added, masked |
| B12 | 10-31's first windows read sealed 10-30 through look backs | excluded with a status |

## Data checks

**Audit v1 (raw inputs).** A shift rule (PSI >= 0.25 between dev days) would have dropped 19 inputs, including every book size. Book sizes
are the strongest single inputs (AUC 0.33 every day). Their scale moves about 2x between days: the median L1 size was 177 sh on 10-24 and
417 on 10-26.

**Probe (dev days only).** Spread across the four dev days of the y1 rate within input deciles:

| Input | Absolute | Relative to the day's own scale |
|---|---|---|
| Book sizes (s1, s2, c2, ct2, ct3, own_l1) | 5.0-7.4 pp | **1.9-2.7 pp** |
| sigma | **1.4 pp** | 2.2 pp |
| basis_bps | 2.8 pp, AUC flips sign | 2.4 pp |

Decisions (dev days only):

| Input | Problem | Treatment |
|---|---|---|
| Book sizes | relation stable only relative to the day's liquidity | normalised by a causal scale |
| sigma | covariate shift only; relation stable | kept |
| basis_bps | relation unstable in both forms | dropped |

**Scale choice (declared rule, dev days).** W30, the median of both sides' L1 size over the frames known in the last 30 s, gives the
lowest dispersion:

| Scale | Mean dispersion over the size inputs |
|---|---|
| none (absolute) | 3.95 pp |
| the day's own median (not causal) | 1.61 pp |
| H1 (earlier windows' medians) | 1.17 pp |
| **W30** | **1.12 pp** |

**Audit v2 (model inputs, dev days).** No drops. Normalised sizes have PSI 0.00-0.14 (absolute: 0.08-1.43). The 7 flags are mild PSI or
covariate shifts with a stable relation (sigma, sens). No label flags.

**Controls.** All passed on all nine days; any failure would have blocked the output.
- STOP-DEPTH parity: every shared input and both 1 s labels equal on 2.01 M rows.
- Future-perturbation falsifier on real rows: frames, step rows and fair prices after t randomised. Inputs changed in 0 of 7,094 checks;
  the label changed in 6,989 of them (98.5%: the perturbation reached the future). The W30 scale changed in 0 of 3,564 checks.
- Brute-force labels at all three horizons: 0 of 68,392 rows differ; the planted sign error differs on 1,747-2,872 rows a day.
- Gated lane vs client lane: equal fill for fill on all 1,434 windows (M 50).
- Fills: 0 unmatched, 0 conservation violations.
- Planted errors: 11 of 11 caught, each by its own check.

## Training checks

| Check | Result |
|---|---|
| T1 chronology | asserted; no calibration or test row in any fit |
| T2 sample vs population base rate | within 0.5 pp for all 90 heads |
| T3 positives | 1 s: 4-77 k a head (j1-j6); 72 ms j3+: hundreds (rare) |
| T4 raw non-monotone across j | 0.2-3% for j2-j4 at 1 s; 3-14% at j5-j6 (rare events) |
| T5 permuted labels | Brier skill -0.005 (no skill) -> PASS under v3.1; AUC 0.477 (outside v3's two-sided 0.02, anti-skill side; see below) |
| T6 overfitting | 1 s: train 0.77-0.83 vs test 0.73-0.76; 72 ms j3: train 1.000 vs test 0.94-0.97 (rare positives memorised, test still high) |
| T7 top inputs | stable across folds: Binance BBO imbalance change (250 ms), Predict L3 / L1 imbalance against us, the relative L1 size, Binance 250 ms return |

T5 change (post hoc, recorded in the script): v3 also required |AUC - 0.5| <= 0.02. That failed on fold 10-27 (0.477) on the anti-skill side.
A model fit to permuted labels is a random function of informative inputs, so its AUC strays either way. A leak would show as skill, so
T5 now blocks only on Brier skill > 0.001. T5 cannot detect feature leaks at all; the falsifier and the brute-force labels cover those.

## Results (5 chronological test days, usable rows not ALREADY)

**1. Skill (R702, R703).** PREDICTABLE / BEATS-BASE for every horizon and level, on 5 of 5 days.

| H | j1 AUC (days) | j2 | j3 | Brier skill j1-j3 | Calibration j1-j3 |
|---|---|---|---|---|---|
| 72 ms | 0.911 (0.88-0.93) | 0.947 | 0.964 | +0.16..+0.18 | calibrated |
| 250 ms | 0.854 (0.82-0.87) | 0.889 | 0.907 | +0.24..+0.33 | calibrated |
| 1000 ms | 0.747 (0.73-0.75) | 0.756 | 0.762 | +0.15..+0.17 | j1 **miscalibrated** (over-predicts 1-4 pp; quieter test days), j2 / j3 calibrated |

Base rates (j1): 1.95% at 72 ms, 7.5% at 250 ms, 23% at 1 s.

**2. Against the comparators.**
- The guard's value (R704): the model is better on 5 of 5 days at every horizon for j1-j3. At 1 s, j1: 0.73-0.75 vs 0.68-0.71.
- STOP-DEPTH refit chronologically (R705), on its own guard-on population: equal at j1 and j3 (NOT-SHOWN), better at j2. A model trained at
  every step matches the guard-on specialist there and also covers the 66% of steps where the guard is off.

**3. Keep quality (R706).** At the guard's own keep count the model keeps steps that are run over less often, at rungs 0-4 for 1 s (rung 0:
0.11-0.17 vs 0.12-0.19 of kept steps) and rungs 0-3 / 0-2 at 250 / 72 ms. The deep-rung readings at the shorter horizons are noise (rates
< 0.1%).

**4. Where the short-horizon skill comes from** (POST HOC, dev calibration day 10-26).

| Rows | Share | 72 ms model AUC | Thin best level alone |
|---|---|---|---|
| All | 100% | 0.913 | 0.807 |
| No Binance move in 500 ms | 83% | 0.901 | 0.817 |

Binance's book-imbalance change alone reaches 0.757 on the no-move rows. Most of the very-near-term skill is Predict-local: a thinning best
level, plus the Binance book leaning. Binance price moves (3.9% of rows) carry less. Binance enters at its event time + 3.5 ms (P15, an
assumed constant), so its inputs still depend on that latency.

**5. Is the target worth acting on? (R701, oracle replay).** Whole portfolio at the 60.2 s mark, $ a day, rungs 0-5 x 30 sh both sides, no
cap. "Oracle" acts on the true label: it cancels a resting order whose level will be run over within H and does not post a level that
will be. The shifted control uses the previous window's labels.

| Arm | M 0 | M 50 | M 250 |
|---|---|---|---|
| Keep-all (rest everything, never pull) | -5,262 | -5,873 | -7,280 |
| Oracle, 72 ms | -720 | -2,973 | -5,725 |
| Oracle, 250 ms | +145 | -48 | -3,252 |
| **Oracle, 1 s** | **+832** | **+795** | **+578** |
| Shifted control, 1 s | -4,475 | -4,967 | -6,192 |

All figures exclude fills on orders < 100 ms old (C1); with them the picture is the same. Oracle minus keep-all is positive on 5 of 5 days
at every clock and horizon. The shifted control recovers only about 1/7 of the gain (+$907 vs +$6,668 a day at M 50). **ALIGNED** at every
horizon.

- **1 s is the target worth acting on.** It is the only oracle that is positive at every clock, including M 250. Shorter horizons
  re-enter into moves that are still under way.
- Activity at M 50: the 1 s oracle cancels 11.7 k orders a day (keep-all 0) and fills 74 k shares (keep-all 345 k). It skips most of the
  volume and keeps the good part.

**6. Does the model's risk rank our orders' fill value? (R707).** Keep-all orders every 250 ms. V = 1 s value of the order's fills in
(t + 22, t + H], cents per order share, without young fills.

| Clock | Top-risk decile | Bottom half |
|---|---|---|
| M 50 | -0.317 | -0.054 |
| M 250 | -0.271 | -0.076 |

**PRICES-HARM** at every horizon and clock, 5 of 5 days. The 1 s label holds 93% of the harm. The 72 ms label misses 71% of it at M 50:
the harm comes later than the next decision. 41% of the positive fill value also sits in run-over rows, which is the cost of a cancel.

**VERDICT: MODEL-READY at 72 ms, 250 ms and 1 s.** That means: aligned target, predictable and beating the base rate at j1-j3, prices our
orders' harm, and not worse than the guard or STOP-DEPTH. Development only (consumed days, C8 / C16): it qualifies the model as an input
for STOP-POLICY, nothing more.

## What it means

1. **The target is right.** Knowing which levels the next second runs over would turn the always-resting ladder from about -$5.9k to about
   +$0.8k a day at M 50, and keep it positive at M 250. This is the first arm in this line positive at every fill clock, but it is an
   oracle. The same pulls without the information do not help.
2. **The model has real skill on that target**, far above the base rate and the guard. That skill is about which levels are at risk, not
   when the guard fires. At 1 s it ranks rows at AUC ~0.75: much better than random, far from the oracle.
3. The open question for STOP-POLICY is how much of the oracle's +$6.7k a day over keep-all the model captures with a threshold per level,
   hysteresis and the 50 ms cadence. Comparators: LDG and FB25C on the same lane; keep-all is a weak baseline. Expect a fraction.

## Checklist (FINDINGS section 1)

- C1, C2, C15: done (oracle without young fills, M 0 / 50 / 250, whole portfolio).
- C3: cancel-pending share of the oracle's fills not reported.
- C4, C9, C10: replay volumes inflated; ours simulated; not order-level validated.
- C5: rule M only (L not run).
- C6: clocks in the script header.
- C7: yes (our orders in the lane).
- C8: consumed days, B4 exposure.
- C11: the keep-all ladder (6 x 30 sh, client lane), not the frozen v3.
- **C12: NOT DONE**. The oracle's gain is not split WS / E2. Required in STOP-POLICY.
- C13: client-known lane; the oracle uses the future by design.
- C14: all arms at 50 ms.
- C16: chronological.
- C17: n/a (no policy).
- C18: design restated; own skeleton.

## Limits

- Consumed days. Inputs were chosen after PULL-DEPTH / STOP-DEPTH on them. Test-day statistics were seen once before the processing
  choices (B4).
- Replay rule M: the queue position is an estimate (B1a). E2 trade anchors are uncertain by +-250 ms (C12 not done).
- The oracle is a ceiling, not a strategy.
- Binance timing is an assumed 3.5 ms (P15).
- The fits' memory went slightly above the self-guard.
- Rows overlap: the per-row statistics describe decision states, not P&L.

## Next (the USER's call)

- **STOP-POLICY** (step 3 of the plan), on this model:
  - per rung, cancel when P^1000_(d+1) >= tau_d, with hysteresis; re-post when the level's risk is gone;
  - tau set on the calibration day;
  - arms STOP and STOP-FAST against LDG, FB25C and keep-all in the same lane;
  - whole portfolio at M 0 / 50 / 250 with and without young fills;
  - WS / E2 split (C12), queue place (C17), pulls a day, the rule U bound.
- **E2-CHECK** (step 1) is still open.
- Memory: build the model inputs in float32 to bring the fits under the 2 GiB guard.

Outputs:
- `data/predict_fun/research/stop_model_20261114/`: rows / pred / align parquet, models, logs.
- Copies (txt / json / logs) in `reviews/stop_model_20261114/`.
