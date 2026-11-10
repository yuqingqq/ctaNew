# B3: bounded soft prediction exits — 2026-11-09

The narrow exit band improves **conditional quote continuity**, but adds exposure while the original guard is off. It is a research candidate, not evidence of higher trading P&L or lower inventory variance. Keep the predeclared 0.01-cent band as the primary candidate for further evaluation; do not select the widest band from this mechanical table.

Producer as-of **2026-11-09 16:01:32 UTC**, artifact verification **16:02:15 UTC**. Population: **220 consumed Nov8 windows**, first 60 seconds, **279,059 original decisions**, **13,077 neutral reference touch-pull actions**, zero excluded windows. One partial UTC day: no confidence interval or independent validation. Nov9–13 remain unread.

## Exact guard evidence

The earlier reconstructed-score contradiction (12,044 / 137,325 wanted lane updates, 8.77%) prevented reliable soft-pull attribution. The new exporter replays captured native inputs with the captured models, parameters and basis mode. It reproduces **679,873 v6 outputs and 582,739 basis outputs exactly**, then reconciles every one of the 279,059 first-minute decisions against the earlier intent artifact. All 137,325 wanted lane updates satisfy the exact native guard checks. This resolves the evidence problem for this cohort; it does not isolate which reconstructed feature/clock caused each earlier disagreement.

A second observer splits scheduled work at every due native event. Its original decision and book exports are byte-identical to the first replay, while adding **1,249,610 read-only state observations** over 0–65 seconds (604,872 v6; 644,738 basis). The study uses only observations before 60 seconds. This matters: a lower exit line can cross on a guard update that did not cross the old line and therefore generated no old decision.

The native worktree is isolated at `366d4192af219eef568ae17c530e4303e6f4ca45`. Its only library change is a generated read-only input accessor; removing that addition reproduces the original source exactly. Every emitted factual output still matches the capture. The shared live Rust repository and frozen research engine were not changed by this study.

## Policy fixed before results

All arms use the prior incremental sizing planner, with the same 40 / 22 / 95 ms create / accepted-cancel / pending-create-cancel assumptions, 1 s cancellation reservation grace, 150-share side cap and three pending creates per side.

At an original touch quote-on to quote-off transition, an arm may keep an **already accepted, uncancelled base order of at most five shares** at its existing price when:

- `line - delta < exact native v <= line`, with minute-zero line 0.0492 cents;
- the native quote function, changing only the soft pull/rung lines, still permits that price and quantity under the same factual state;
- the quote does not duplicate a currently desired rung.

The hold lasts at most **100 ms**. A lower-line crossing, missing/stale guard, Binance guard, hold, or native price/size eligibility failure ends it sooner. The original hard predicates remain in effect **conditional on recorded inventory/order state**. No new order is opened during the guard-off grace, no cancellation is undone, and no rung order receives this grace. Extra native probes only cancel retained orders. Normal quote-on events resume ordinary price/size rules. An independent timer cancels at the 100 ms deadline even between original decisions.

The primary band was **0.01 cents**; 0.005 / 0.02 / 0.04 were declared sensitivity arms. Four new policies; cumulative multiplicity 16 including the preceding sizing/entry/re-entry work. Source freeze `2d235fd4`, declaration `4c69c4d1`.

## Mechanical comparison

All rows have n=220 windows as of 16:01:32 UTC. Changes are relative to the fixed incremental reference. “Aligned” means time active at the original desired price while that original lane wants a quote. “Off-target” includes guard-off and stale/cancel-pending occupancy; it is not booked inventory.

| Exit band, cents | Requests | Request change | Original aligned time change | Off-target time change | Active share-time change | Reserved share-time change |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 65,211 | — | — | — | — | — |
| 0.005 | 62,319 | -4.43% | +1.61% | +18.09% | +1.89% | -1.57% |
| **0.01, primary** | **60,651** | **-6.99%** | **+2.65%** | **+33.87%** | **+3.40%** | **-2.42%** |
| 0.02 | 59,289 | -9.08% | +3.59% | +56.46% | +5.35% | -3.01% |
| 0.04 | 58,856 | -9.75% | +3.93% | +72.00% | +6.54% | -3.12% |

For the primary band:

- Original desired-price coverage rises **76.17% → 78.18%**. It gains 113.72 aligned lane-seconds and loses 3.56 elsewhere, for a net **110.16 lane-seconds** over the cohort. This is recovered original coverage, not just counting newly allowed guard-off time.
- Off-target occupancy rises **704.97 → 943.75 lane-seconds**, an extra **1.09 lane-seconds per 60-second window**, summed across lanes.
- **4,783 / 13,077** touch pulls retain a small accepted order. **2,376 / 4,783 (49.68%)** reopen at the same price within the grace. The remaining retained actions end at the deadline (1,371), lower line (795), native price/size guard (235), or the window boundary (6).
- **790** cancellations occur at added native probes. Ignoring intervening guard updates would miss these exits.
- Mean oldest active quote age rises **515 → 568 ms**. This does not establish a better queue rank or fill rate.
- v6 (n=122): requests **-7.46%**, aligned time **+2.67%**. Basis (n=98): **-6.50% / +2.62%**. These are two runs of the same consumed day, not independent day validation.

Unlike the earlier re-entry delays, all tested exit bands improve aggregate original-price coverage while reducing requests. Wider bands add progressively more off-target exposure for smaller additional coverage gains. Lower reservation usage reflects fewer cancelled tickets awaiting release; it **does not establish lower inventory risk**, especially with active share-time increasing.

## Future price-move diagnostic

Labels are signed future midpoint changes from the decision midpoint, excluding initial spread. Up bid uses +change, Down ask uses -change. Native book frames must be at or before the label time and no more than 250 ms old at both endpoints. Every stale or missing endpoint remains a status. These are **unweighted action diagnostics**, not fill markouts, trade returns, settlement attribution or P&L.

For primary-band retained actions (n=4,783, same as-of):

| Horizon | Valid labels | Excluded status | Mean signed move, cents/share | Matched-control mean | Adverse fraction |
|---|---:|---|---:|---:|---:|
| +1 s, primary | 4,766 | stale frame: 17 | +0.361 | +0.387 | 6.46% |
| +5 s, secondary | 4,759 | stale frame: 24 | +0.638 | +0.671 | 26.10% |

Controls use 500 predeclared fixed-seed draws from all neutral reference touch pulls with valid labels, matched on action count within run / UTC hour / side / guard-distance bin / price bin / spread bin. Selected actions remain in the control pool. The controls have slightly stronger average drift: the positive raw moves do not demonstrate incremental predictive skill. This descriptive proxy cannot choose a trading policy or price the risk of the extra exposure; actual fills are selected differently from all quote actions, and both horizons exceed the maximum grace period.

## Verification and limits

- **26 controls pass**, including retained-order positive, hard/lower-line exits, deadline inside an event gap, no new order or cancellation undo, pending-create refusal, rung duplication, zero-band exact planner parity, and stale/mismatched-book label refusal.
- Original baseline matches the previous incremental artifact on 18 metrics per window. Counts are unchanged; split integration has a predeclared numeric tolerance of 1e-4 absolute / 1e-11 relative.
- Saved artifacts reconcile **1,100 window-arm rows and 65,385 action-arm rows**. Neutral pull identities regenerate from the native source; all future-price labels and 500-draw diagnostics reconcile. Coverage gain-minus-loss identities hold. A deliberately corrupted request total is refused.
- No hypothetical fills, counterfactual inventory/rung-state feedback, throttle, post-only rejection or live latency distribution. Historical trade-clock gate remains closed. No P&L, fill-rate, queue-rank, settlement or inventory-variance result.
- This compares against the earlier **research incremental planner**. The separately completed Rust A2 implementation `70b965b` was not used as the counterfactual order manager. Integration with A2's paired-order, retry and fill rules still requires its B4 parity work.
- Hard guards are preserved as functions of captured state; a different fill path would produce different state. The result must not be represented as a live safety or deployment check.
- Next economic step needs calibrated/exact fills or a controlled forward experiment. Do not loosen exit rules live based on the table. Inventory-aware grace (only on the reducing side) is a possible separately declared follow-up, not tested or selected here. Symmetric hysteresis, size-line hysteresis, EMA, rung retention and longer grace remain open.

## Artifacts and resource use

Branch `research/predict-soft-pull-20261109`; checkout `/tmp/cta-soft-pull-20261109`.

- Producer / policy / checker: `research/pf_soft_pull_study.py`, `pf_soft_pull.py`, `pf_soft_pull_verify.py`.
- Exact native exporter / observer: `research/native_guard_evidence.rs`, `pf_native_guard.py`, `pf_native_guard_verify.py`, `pf_native_grid.py`, `pf_native_grid_verify.py`.
- Small committed receipts: `reviews/native_guard_20261109/` and `reviews/soft_pull_20261109/`.
- Data and logs: `/home/yuqing/ctaNew/data/predict_fun/research/native_guard_20261109/` (~693 MiB, including bounded prefixes and isolated native build) and `soft_pull_20261109/` (~1.2 MiB). Large artifacts are not committed.
- All substantive jobs ran serially through `run_light.sh`, one core, 4 GiB address-space cap. Study: **1,134 MiB peak RSS, 67.93 s**; artifact verifier: **401 MiB, 8.78 s**. All jobs finished.

Recorded run commands, from this branch (declaration/output paths refuse overwrite):

```bash
python3 -m predictfun.research.pf_native_guard declare
python3 -m predictfun.research.pf_native_guard build
# prepare / replay v6, then basis; native guard verifier follows
python3 -m predictfun.research.pf_native_grid declare
python3 -m predictfun.research.pf_native_grid build
# replay v6, then basis; grid verifier follows
python3 -m predictfun.research.pf_soft_pull_study declare
python3 -m unittest predictfun.research.tests.test_soft_pull predictfun.research.tests.test_quote_capture predictfun.research.tests.test_reentry_study
python3 -m predictfun.research.pf_soft_pull_study run
python3 -m predictfun.research.pf_soft_pull_verify
```

Use `run_light.sh` for each substantive command. Exact source commits and hashes, bounded-input hashes, observer parity and output hashes are in the receipts. A fresh reproduction needs a clean isolated native worktree at the pinned commit and fresh versioned output paths; do not rerun original-source checks against the accessor-instrumented worktree or overwrite the completed artifacts.
