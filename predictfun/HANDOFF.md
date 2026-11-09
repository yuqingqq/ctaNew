# P-2026-005 HANDOFF

## 2026-11-09 — inflight inventory-reservation experiment RUNNING

- User authorized testing all in-flight quantities to avoid unintended reverse exposure. Branch
  `research/predict-inflight-budget-20261109`, checkout `/tmp/cta-inflight-budget-20261109`, declared source `f0e15415`.
  Data/declaration/logs: `/home/yuqing/ctaNew/data/predict_fun/research/inflight_budget_20261109/`.
- Two new arms: `reduce_reserved` preserves original adding policy and caps aggregate reducing reservations to known
  inventory; `both_reserved` adds the prior aggregate 10-share adding budget. Resting, pending-create and pending-cancel
  remaining quantities counted across all lanes, no opposite-side credit. Keep eligible same-price partial orders.
  Submission recheck and minimum lots/notional apply. Original guards/RO60/S240/latencies unchanged.
- New-arm cancellation reservations persist through confirmation plus the simulated 100ms fill-report delay +.001ms:
  no cumulative execution quantity is provided with simulator cancel confirmations, so early release could reuse room
  before a late fill is reported. Original reference arms retain their exact behavior; this grace is part of the treatment.
- All nine consumed dates / 2592 markets per arm/timing, J/X. Controls: baseline and prior reserved10. Fresh first-date
  references plus eight-market smoke; hash-checked prior references on other days. Fixed guards trained on eight dates:
  exploratory in-sample. Uncapped loss budget, 150-share backstop, rebates excluded. No forward/live changes.
- 31 preflight controls pass. Declared 2000 day null + 2000 day bootstrap draws, 18 comparisons: reduce_reserved-baseline,
  both_reserved-reserved10, both_reserved-reduce_reserved x J/X x settlement/+1s/inventory-square. All-market full-path
  inventory, zero crossings, FIFO matched/residual PnL and volume metrics. Each both_reserved fill asserts |actual net|<=10.
- Batch started 07:01Z; single light process/core, 4GiB address-space cap. Progress 07:21Z: all nine J replays
  complete, all inventory/path checks passing; X first date starts next. Day-level PnL is mixed; await full report.
  Protected Nov9–13 untouched.
- Supplemental mechanism audit prepared 07:31Z after seeing daily PnL: classify reducing intent at exact send timestamps
  and count any subsequently opening fill quantity, including fills from flat. Bound same-time notification ambiguity;
  all markets, positive/known-bad controls, separate receipt. No additional policy or inference selection.

## 2026-11-09 — entry inventory-budget experiment COMPLETE

- As-of **2026-11-09 06:12:40Z**; supplemental peak trace **06:18:57Z**. Nine consumed dates (Oct23–27,
  Oct31–Nov3), **2,592 markets per arm/timing**, all J/X runs complete. Guards trained on eight dates:
  exploratory in-sample, not independent validation. Uncapped total-loss budget, existing 150-share exposure backstop,
  rebates excluded. Branch `research/predict-entry-budget-20261109`, checkout `/tmp/cta-entry-budget-20261109`;
  declared source `f7fc7162`, tested source/state `f2867829`. Data/logs:
  `/home/yuqing/ctaNew/data/predict_fun/research/entry_budget_20261109/`.
- `uniform10` caps new inventory-adding orders at 10 shares per order (existing 5-share targets remain 5).
  `reserved10` caps adding orders to the remaining 10-share same-side budget after known inventory and all known
  outstanding reservations, across all lanes. Adding means side-sign times known net >=0. Reducing sizes/guards
  unchanged, even if an order could cross zero. Both new arms keep eligible same-price adding orders when known
  remainder fits desired size/room, without upsizing; pending creates/cancels reserve capacity until terminal knowledge.
  Lots/minimum notional and a submission-time budget invariant apply. Baseline unchanged. No urgency/top-up/restart.

| Timing / arm | Settlement $ | +1s markout $ | Daily PnL SD $ | Mean residual shares | Mean time-weighted inventory squared |
|---|---:|---:|---:|---:|---:|
| J baseline | 514.06 | 602.83 | 134.51 | 6.557 | 168.162 |
| J uniform10 | 425.59 | 395.52 | 74.58 | 4.025 | 58.855 |
| J reserved10 | 309.74 | 237.96 | 41.74 | 2.515 | 30.388 |
| X baseline | 317.63 | 463.56 | 103.09 | 5.720 | 144.533 |
| X uniform10 | 281.63 | 292.05 | 64.93 | 3.475 | 46.731 |
| X reserved10 | 253.05 | 176.21 | 43.28 | 2.252 | 25.295 |

- Uniform cap retains **82.8% / 88.7% J/X** of observed settlement profit, with daily SD **44.6% / 37.0% lower**,
  mean residual inventory **38.6% / 39.2% lower**, and filled volume **29.7% / 32.0% lower**.
  Reserved budget retains **60.3% / 79.7%**, with SD **69.0% / 58.0% lower**, residual **61.6% / 60.6% lower**,
  volume **52.5% / 53.4% lower**. Inventory-square falls 65.0%/67.7% uniform, 81.9%/82.5% reserved.
  These describe the risk/volume tradeoff, not improved execution at matched activity or calibrated outcome variance.
- Both new arms reduce inventory-square AND total +1s value on every day at both timings. Paired day inference:
  2,000 shared sign flips + 2,000 day bootstrap draws; 18 comparisons (three arm pairs x J/X x settlement/+1s/q²).
  Settlement mean-difference intervals all cross zero; no reliable expected-profit ordering. All q²/+1s comparisons
  have max-T p=0.005997, conditional on these consumed days. Daily SD/tail/drawdown figures are descriptive.
- J/X maximum drawdown: baseline **-$293.73 / -$249.33**, uniform **-$148.76 / -$119.37**, reserved
  **-$79.60 / -$77.39**. Worst day: **-$53.12 / -$128.60**, **-$27.31 / -$59.83**, **-$30.04 / -$45.96**.
  Reserved is not uniformly better on every tail metric (J worst day worse than uniform).
- Spread capture has NOT improved: +1s value per 100 filled shares J/X falls from **1.7246 / 1.6162** to
  **1.6088 / 1.4964** uniform and **1.4324 / 1.3178** reserved. FIFO matched-trade profit J/X is
  **$134.02 / $173.54** baseline, **-$8.48 / $24.64** uniform, **-$33.20 / $24.31** reserved.
  FIFO matching includes intervening price drift, not pure spread; residual FIFO attribution is not carry-at-240.
  Both new arms share queue retention, so their comparison with baseline combines retention and size effects;
  reserved-versus-uniform isolates reservation sizing with common mechanics, but is not turnover matched.
- **Not a hard 10-share limit:** instantaneous peak absolute inventory J/X is **75 / 65** baseline,
  **80 / 48.5** for BOTH new arms. Post-hoc trace covers both tied global peaks (2 paths / 15,552 total paths,
  no exclusions): Oct26 J market 3818191, initial +10 inventory, then 90 sell shares filled through 10.546209s,
  reaching -80. Every sell order was submitted while known inventory was +10 (therefore classified reducing);
  50 sell shares filled while cancellation was pending. Send times match exact placement events; known inventory
  reconstructed with the recorded 100ms fill delay, no event-priority ties. Terminal inventory is flat.
  This confirms the remaining exposure route: unchanged reducing orders can reverse inventory, with several
  outstanding/cancel-pending orders across lanes. A hard limit must budget those zero-crossing portions too.
- **25 preflight controls PASS**; eight-market baseline smoke and all 18 full day/timing reconciliation checks PASS.
  Fresh Oct23 baseline at both timings exactly matches reference fills/requests/windows; other baselines reused
  with hashes. Zero excluded markets, missing valuations, or missing bounded Binance hour files. All input/source
  and per-run adapter identities verified. Peak diagnostic also passes positive cross-zero and known-bad missing-event /
  wrong-peak refusal controls. Peak replay RSS ~1.6 GiB, one core; no jobs remain running.
- Main receipt `reviews/entry_budget_20261109/receipt.json`; supplemental explanatory receipt
  `reviews/entry_budget_20261109/peak_diagnostic.json` (main receipt unchanged), reproduced by `pf_entry_peak.py`.
  Simplest per-order cap offers better observed profit retention; reservation budget offers greater average risk
  reduction. Neither establishes improved spread capture or a safe hard position ceiling. Next narrow experiment:
  reserve full potential exposure including reducing-order zero crossings while retaining useful reduction liquidity.
  No live code/settings, frozen artifacts, forward-race changes, or protected Nov9–13 data reads.

## 2026-11-09 — spread capture / inventory-risk review

- User asks how to improve spread capture while lowering inventory variance. Post-hoc descriptive risk calculation from
  the completed receipts (data as-of 04:15:18Z), nine consumed days / 2,592 markets per arm/timing, no exclusions/missing values:
  daily settlement PnL sample SD baseline -> urgency **134.5108 -> 127.9100 J**, **103.0877 -> 97.8480 X** (~5% lower).
  Worst day **-53.1211 -> -53.2061 J**, **-128.5982 -> -99.4829 X**. J also cuts a large winning day; X improvement remains
  concentrated on Nov2. No interval or significance claim. Receipt: `reviews/urgency_history_20261109/risk_diagnostic.json`.
- Code review: CAP10 is a threshold on current known net, not a reservation-aware 10-share bound on the next order.
  If |known net| <10, a 25-share adding order can still pass, and each ladder lane checks the same pre-fill position.
  Previously tested HCAP clips against the separate 150-share backstop; it did not test a strategic 10-share risk budget.
- Priorities to investigate: size/withhold inventory-increasing quotes against potential same-side fills across all lanes
  (including pending creates/cancels); retain effective, executable reduction sizes; isolate queue retention at matched
  target exposure; then test selective new MM cycles after returning near flat (RO60 currently prevents them after 60s).
  Budget any price concession for reduction against the risk removed; existing guards remain, no blanket late aggression.
- Evaluation should separate matched-pair trading value, adverse-selection markout, unpaired carry, and downside risk.
  Compare at matched trading activity where possible so merely trading less is not called better execution. No new variant,
  threshold sweep, live change or forward data read launched in this review.

## 2026-11-09 — earlier urgency check COMPLETE

- As-of **2026-11-09 04:15:18Z**. All four earlier complete compatible dates (Oct 23–26), **1,152 markets per arm/timing**,
  replayed at J/X; baseline versus the exact previous RPRICE rule. Branch `research/predict-urgency-history-20261109`,
  checkout `/tmp/cta-urgency-history-20261109`, declared source `10893e49`, tested source/state `a94a0dd7`.
  Receipt: `predictfun/reviews/urgency_history_20261109/receipt.json`; data/logs under
  `/home/yuqing/ctaNew/data/predict_fun/research/urgency_history_20261109/`. No study job remains running.
- These are previously consumed guard-training days. Urgency was selected after the prior four-candidate comparison;
  this is an earlier historical consistency check, not independent validation. Exact policy source remains `f014efbe...`:
  one-cent reducing-side improvement into the spread, 180<=t<240, |known inventory| >=10, original guards retained.
  Uncapped total-loss budget; existing 150-share exposure cap; rebates excluded. No parameter/model changes.

| Settlement PnL change versus baseline ($) | J | X |
|---|---:|---:|
| Oct 23 | +1.3408 | +3.4780 |
| Oct 24 | +6.0191 | -0.4361 |
| Oct 25 | -27.0320 | +0.3366 |
| Oct 26 | -1.0178 | -0.4778 |
| Earlier four days | -20.6899 | +2.9007 |
| Previous five days | +10.9085 | +25.7631 |
| All nine days, descriptive | -9.7814 | +28.6638 |

- Earlier baseline -> urgency settlement: **$691.6287 -> $670.9389 J**, **$433.5722 -> $436.4729 X**. Both timings
  improve on only 2/4 earlier days. Earlier +1s changes **-$0.0742 / -$0.6977**. Mark-at-240 changes
  **-$6.6740 / -$2.4265**; carry changes **-$14.0159 / +$5.3272**. Thus the small positive X result comes from carry,
  while both measured short-term and stop-mark value decline on these earlier days.
- Mean residual inventory falls **6.9581 -> 6.8818 shares J**, **6.2460 -> 6.2148 X**. Economically reducing volume rises
  **87.8751 / 35.9633 shares** over all 1,152 earlier markets. Time-weighted absolute inventory over 60–300s falls
  **7.5130 -> 7.4737 J**, **6.7139 -> 6.6960 X**. The reduction is real but small and not a reliable PnL gain.
- Combined nine days / **2,592 markets per arm/timing**: baseline -> urgency **$514.0559 -> $504.2745 J**,
  **$317.6259 -> $346.2897 X**. +1s delta **-$2.0719 / +$0.1535**; carry delta **-$12.7697 / +$25.0251**.
  J improves 6/9 days but Oct25 dominates the losses. X improves only 3/9 (one unchanged); remove Nov2's +$29.1153
  and its pooled gain becomes **-$0.4515**. The sign and concentration depend on timing and individual settlement days.
- Earlier inference: 2,000 predeclared shared UTC-day sign flips, four comparisons (J/X x settlement/+1s); all max-T
  family p values 1.0. **No confidence intervals on four days**; nine-day pooled points are descriptive after candidate
  selection, with no pooled significance claim. Prior four candidates / 16 comparisons retained in the declaration.
- All **22 controls pass**, including positive/refusal controls for scope, source identity, event identity and day nulls.
  Fresh Oct23 baseline parity exact at J/X; other baseline artifacts copied from the prior verified study with hashes.
  All eight day/timing runs have exact pre-180 fills/requests and per-market settlement/carry/inventory reconciliation.
  Zero excluded markets or missing valuations, zero missing bounded Binance hour files, zero ambiguous filled-order keys.
  Final receipt copy and source hashes verified. Peak replay RSS ~1.6 GiB on one core; ~493 GiB disk free.
- Clarification of previous stress interpretation, retained in this declaration: Nov3 urgency's settlement delta is
  identical with/without p99 cancel reservations at each timing. That negative day does not show incremental sensitivity
  to this delay stress. No additional stress sweep was used here.
- Conclusion: earlier data weakens the case for a general urgency PnL improvement. Keep this rule as research; do not
  enable it globally from these results. It slightly reduces inventory but can surrender favorable subsequent price moves
  and settlement carry. No live change, new freeze, forward-race change, or protected Nov9–13 data read.

## 2026-11-09 — passive inventory experiment COMPLETE

- As-of **2026-11-09 03:53:11Z**. All declared replays, reconciliation and 18 source controls passed. Isolated branch
  `research/predict-passive-inventory-20261109`, checkout `/tmp/cta-passive-inventory-20261109`, tested source `983a3b4f`.
  Receipt: `predictfun/reviews/passive_inventory_20261109/receipt.json`; retained data/logs under
  `/home/yuqing/ctaNew/data/predict_fun/research/passive_inventory_20261109/`. No study job remains running.
- Five consumed dates: Oct 27, Oct 31, Nov 1–3, **1,440 markets per arm/timing**; same fixed guards, trained on four of
  these five dates. This is in-sample policy feasibility, not independent validation or a live-return estimate. J uses
  transaction-frame jitter; X uses transactions shifted -50 ms and max activation queue. Main total-loss budget uncapped;
  separate 150-share exposure cap unchanged, rebates excluded. Baseline settlement **-$177.5728 / -$115.9463 J/X**;
  baseline +1s markout **$261.9645 / $183.1157** (a diagnostic, not realized spread PnL).
- Same-price queue retention after 60s keeps an eligible order only when its known remainder plus other reducing-side
  reservations fits known inventory. It can also retain a smaller order instead of upsizing. Small reduction caps desired
  size at 5 shares; combined may preserve an older larger original order whose remaining quantity fits 5. Urgency improves
  the reducing quote one cent into the spread during 180–240s for |known net| >=10. Existing guards must admit the original
  quote; their score is not claimed calibrated to the improved price. Price concession enters actual fill PnL.

| Change | Settlement delta J / X ($) | +1s delta J / X ($) | Mean residual shares J / X |
|---|---:|---:|---:|
| Baseline | 0 / 0 | 0 / 0 | 6.2363 / 5.2992 |
| Keep queue | -10.7803 / +11.1000 | -0.3580 / +0.1500 | 6.1357 / 5.2506 |
| Five-share reduction | -22.1177 / +17.7284 | -25.9673 / -21.8333 | 6.8862 / 5.8172 |
| Late urgency | +10.9085 / +25.7631 | -1.9977 / +0.8512 | 6.2272 / 5.2945 |
| Combined | -21.0283 / +19.2284 | -26.1020 / -22.3833 | 6.8674 / 5.8172 |

- Queue retention lowers place requests **2.5687% / 1.9782%**, and residual inventory **1.61% / 0.92%**, but settlement
  improves only 1/5 J and 2/5 X days. It remains an operational-efficiency candidate, not demonstrated PnL improvement.
- Five-share reduction cuts economically reducing volume **45.94% / 50.72%** and raises mean residual inventory
  **10.42% / 9.77%**. MIN_USD withholding decisions rise **81,403 -> 91,065 J**, **76,790 -> 86,246 X**; these are repeated
  decisions, not distinct orders. Both smaller fills and the existing $1 minimum limit reduction; no causal attribution
  of the full loss to either mechanism. Among the same pre-60 inventory population (674 J / 579 X markets), the fraction
  ending below one share falls **16.32% -> 9.35% J**, **12.95% -> 6.56% X**. Do not adopt a blanket five-share cap.
- Urgency improves 4/5 J days but only 1/5 X days (one unchanged). Its X gain disappears without Nov 2: **-$3.3522**.
  J gain decomposes to mark-at-240 **+$9.6623** / carry **+$1.2462**; X **+$6.0652 / +$19.6979**. Mean residual inventory
  falls only **0.0091 / 0.0048 shares**. It does not establish stronger spread capture or materially better inventory control.
- Declared 2,000 shared UTC-day sign flips + 2,000 day bootstraps, 16 comparisons. Urgency settlement family-adjusted p
  **0.5727 J / 1.0 X**. J pointwise bootstrap mean/day interval is positive, but coarse five-day randomization and family
  inference do not support a robust positive. No new parameter sweep or candidate adoption after outcomes.
- One-day Nov 3 stress: **288 markets per arm/timing**, cancel reservations delayed to at least measured p99 **143.3536 ms**,
  physical removal unchanged. Settlement deltas J/X: queue **+6.2466 / +0.3500**, small **+0.6981 / -8.5854**, urgency
  **-0.0850 / -2.9689**, combined **+1.8131 / -10.3354**. No interval; reservation sensitivity, not calibrated full OMS.
- Checks: exact baseline J/X smoke and fresh first-day baseline parity (also fresh Oct31 J); remaining main baseline
  artifacts reused with hashes/origin recorded; stress baseline always fresh. All pre-60 fill/request traces equal across
  arms; fill/settlement/carry and inventory paths reconcile per market. Zero excluded markets/missing valuations; no missing
  bounded Binance hour files. Guard withdrawals, pending reservations, lot/notional checks and stop remain active.
- Declaration v3 supersedes v2's filled-order diagnostic assumption: legitimate duplicate creation timestamps caused a
  checker stop. PnL/policies unchanged; counts now use immutable price/size lower bounds and sent-request/tranche upper
  bounds. Final observed bounds collapse: one duplicate create key per main arm/timing, zero ambiguous filled keys. Original
  failure log/declarations retained. Synthetic ambiguous-order positive and corrupted-identity refusal controls pass.
- All source/input/artifact identities recorded; receipt copy and source hashes verified. Peak replay RSS about 1.3 GiB,
  one core, ~493 GiB disk free. Frozen artifacts, live code/settings and protected Nov 9–13 forward dates unchanged/unread.
- Next: avoid the blanket five-share cap. Queue retention merits efficiency work; urgency needs independent evidence.
  Any further size policy should explicitly account for executable minimum notional and reduction throughput before a new
  predeclared test. Current results do not justify promoting any candidate to live.

## 2026-11-08 — sizing and settlement probability study COMPLETE

- As-of **2026-11-08 17:11:07Z**. All declared jobs and controls finished; no research job left running. Isolated branch
  `research/predict-sizing-settlement-20261108`, checkout `/tmp/cta-sizing-settlement-20261108`, tested source `beccfc63`.
  Full receipt: `predictfun/reviews/sizing_settlement_20261108/receipt.json`; retained inputs/outputs and logs under
  `/home/yuqing/ctaNew/data/predict_fun/research/sizing_settlement_20261108/`. This entry supersedes earlier running statuses.
- Main experiment: nine consumed days (Oct 23–27, Oct 31–Nov 3), 2,592 markets per arm/timing; baseline, capacity clipping,
  size hysteresis, combined; J/X; $150/uncapped total-loss budget. Fixed deployed guards were trained on eight of these days:
  **in-sample strategy comparison**, not a live-profit estimate. Continuous historical-clock basis, lot/notional checks,
  known-state reservations and refusal backoff are shared. No missing +1s or stop valuations in these results.
- Main settlement $ J/X: baseline **514.0559 / 317.6259**, capacity identical; hysteresis/combined **550.5360 / 339.7488**.
  Hysteresis place requests fall **7.7464% / 7.3397%**. +1s value changes **-10.9858 / -8.8662**; mark-at-240s changes
  **-68.4191 / -94.4639**; carry improves **104.8992 / 116.5868**. Thus settlement gains are from leftover inventory,
  not improved measured short-term value. Only 4/9 days improve at each timing; excluding Nov 2 changes the gain to
  **-92.5270 / -73.4372**. Observed drawdown falls, but no robust PnL benefit established and no candidate adopted.
- $150 is explicitly a **test scenario**, not a permanent user constraint. All 216 main capped/uncapped win/fill/request
  comparisons are exact. The separate 150-share exposure cap stays in both. Budget credits simulated window-end settlement
  and skips unstudied gaps; no live path/settlement-confirmation parity claim. Rebates excluded and never replenish budget.
- Cancellation diagnostic: same 105 Tokyo-5 windows, 13,775 cancels, 13,766 matched acknowledgements / 9 missing; completed
  delay p50 20.67 ms, p99 143.35 ms, max 14.97 s. Uniform-p99 reservation stress on Nov 3, 288 markets per arm, keeps original
  physical removal timing; fresh zero-PnL budget. Capacity lowers refusals but adds no PnL: uncapped baseline/capacity J
  -45.1963/-45.2518, X both -11.7216. With hysteresis, capacity J -45.2269 -> -49.6016, X unchanged -68.7496.
  The $150 budget changes some stress paths (both beneficial and detrimental). One day, no interval; sensitivity only,
  not calibrated OMS simulation. Rare longer/missing acknowledgements and full watchdog/OMS behavior remain unmodelled.
- Settlement forecast feasibility: initial train Oct 23–25; six chronological consumed test days, **1,728 markets /
  13,352 eligible neutral decisions**. Excluded: 453 missing market prices, 19 start-price-not-known. Each fit uses earlier
  venue outcomes known before the test day; no fill-selected population. Day-balanced Brier (lower better): market
  **0.189047**, Brownian/basis **0.192590**, calibrated market **0.189106**, market+physical logistic **0.190309**,
  fixed market-residual LightGBM **0.195683**. No alternative beats the market; nonlinear is worse on every test day.
  Added nonlinear family is explicitly post-initial-result exploration, not an untouched test or tuning search.
- Cost-aware shadow: 26,704 hypothetical 25-share Up/Down opportunities per model (separate alternatives, not strategy PnL).
  24,398 eligible; 1,854 insufficient arrival depth; 442 below notional; 6 stale; 4 invalid books. Calibrated-market mean by day
  +$0.01275 per opportunity, day bootstrap interval [-$0.11371,+$0.13730]; other models negative. No reliable value gain.
  500 matched action randomizations and 2,000 day null/bootstrap draws. Expanded reports supersede initial inference:
  max-T now uses sign-invariant RMS scaling; original point forecasts/valuations retained unchanged. See declaration v3
  and nonlinear_declaration_v3.json; five probability models / ten forecast comparisons, four shadow comparisons.
- Controls passed: 10 source tests (positive/refusal controls); eight-window exact frozen-engine parity; full 288-window,
  eight-arm cached/direct comparison exact across 24 artifacts; per-market fill-to-settlement reconciliation throughout;
  source/input hashes, label causality, probability oracle/anti-oracle, prior-offset and inventory-value controls.
  Peak observed replay/control RSS about 1.6 GiB, one core; ~499 GiB free disk. No large data added to git.
- Next: treat hysteresis as an efficiency candidate only; probability forecasts remain research outputs. Keep short-term
  adverse-selection guards separate from settlement valuation. Compare expected payout against executable exit value and
  inventory risk, not merely P(settle=1)>0.5. More independent future data and full execution calibration are needed before
  adopting a new controller. Live code/settings, frozen artifacts and the Nov 9–13 forward race are unchanged/unread.

## 2026-11-08 — continuous basis agrees with Rust on captured Tokyo inputs

- User requested the missing research-side boundary check. Source commit `4f7cf752` adds
  `pf_basis_continuous.py` (one sample per ordered raw reference event) and
  `pf_step5_features_continuous.py` (pinned original feature formulas, separate `_basis_v1` rows).
  `PF_BASIS_SAMPLES` must name the verified sample artifact. Original feature/strategy/freeze files are unchanged.
- All 105 complete Tokyo-5 windows, Nov 7 17:40Z to Nov 8 02:25Z: zero feature differences on 494,637 rows,
  zero guard differences (498,422 eligible keys including 3,785 start-withholds), zero conditional quote differences
  on 537,056 decisions. First 70 s after UTC hours: 12,189 model rows and 12,558 decisions, all equal.
  Original frozen research is a deliberate control: 2,198 differing feature rows and 72 differing quote decisions,
  all in that boundary interval. As-of 2026-11-08 13:04:44Z; receipt `reviews/continuous_basis_20261108/receipt.json`.
- Rust tools: pmmmrust branch `review/pfmm-continuous-research-parity-20261108`, source `4d757be`;
  continuous runtime is unchanged from `pfmm-ladder-v3-basis-tokyo-1`. Legacy replay still matches all 637,754 outputs.
  Evidence: `/home/yuqing/pmmm-data/review-exports/continuous-parity-20261108/` (bounded sequential jobs, peak 807 MiB).
- Scope: same captured known-time input stream and captured legacy inventory/reservations. No changed-request
  OMS/fill/PnL simulation, model refit, research freeze or deployment. UTC known-hour CSV adapter; other clocks/groupings
  are different inputs. Next: explicitly select these new rows for continuous-version research; full strategy effects remain unmeasured.
- Nov 7–8 are consumed diagnostic days (including the earlier live comparison); Nov 9–13 remains untouched and
  the already frozen v3 forward procedure remains unchanged. The original frozen v3 is NOT the new continuous variant.

## 2026-11-07 08:20Z (session 62f3ce33) - Codex v3 port review complete (verified)

- pmmmrust 3366e61 (receipt docs/receipts/20261107_pfmm_ladder_v3_review.json; tested code 173f53c). It names our receipt by its
  current hash; its flip audit equals ours (4 / 3 of 2592 windows, <= 5.04 sh). No launch, deployment, tag or forward read.
- Not done yet: Tokyo shadow / real-venue test of v3. Live = user's call. Forward block 11-09..11-13 still pending (after 11-14 02Z).

## 2026-11-07 08:02Z (session 62f3ce33) - Codex v3 port verified; v3 receipt's no-flip sentence is wrong

- Codex ported v3 (pmmmrust b85a401, ab40361); 11-03 J / X trace hashes = i2J / i2X (checked here). Port fixes (min 1 sh with the
  1 USDT check; expiry +120 s grace) match dev.predict.fun's place-orders rules (read here). No live launch.
- v3 receipt's 'the position never flips' is wrong (STATUS same time): flips occur in the replay in 4 / 3 of 2592 windows (J / X),
  <= 5 sh, mostly via cancel-pending orders; already inside the in-sample and forward numbers. Receipt not edited.
- Open (user): a correcting receipt for the description only, and/or a v4 with a hard no-flip guard (new replay + freeze + days).

## 2026-11-07 06:18Z (session 62f3ce33) - pm_ladder_v3 FROZEN (v1 + CAP10 + RO60 + S240)

- Receipt predictfun/freeze/pm_ladder_v3/RECEIPT.md, code commit 4ecaca69; engine parity PASS (STATUS 06:18Z).
- Forward check: own block 11-09..11-13; run `predictfun/research/run_forward_v3.sh` by hand after 2026-11-14 02:00Z (no timer).
  The 11-04..11-08 race (run_forward.sh after 11-09 02:00Z) is unchanged; v3 is not in it.
- Live port would need: 10-share add limit on known net (touch and rung), 60-240 s reduce-only phase (reducing side only, size <=
  |known net|, no rung, minute 1-3 lines, all guards), stop at 240 s, guard values to 240 s (port not checked here). Live = user's call.

## 2026-11-07 04:34Z (session 62f3ce33) - CAP10 + RO60-240 tested (in-sample)

- pf_inv2.py / run_inv2.sh -> step4/report_inv2.json, checks PASS (STATUS 04:34Z). Same +1 s trading profit as v1; day sd -37 / -48%,
  worst day and max drawdown about a third of v1's, $150-cap halts 2 / 0%. Settle +$25-30/day vs v1, not resolved.
- RO60+S290 with the line is not runnable (no guard values after 240 s); would need a model refit to 300 s.
- Open (user): freeze CAP10+RO60-240 for untouched days or declare it into the 11-04..11-08 race before the 11-09 02:00Z run; live
  port would need CAP10 and the 60-240 s reduce-only phase with the minute 1-3 lines (port not checked here).

## 2026-11-07 03:40Z (session 62f3ce33) - taker hedge of the leftover

- pf_hedge_check.py -> step4/report_hedge_v1.json (STATUS 03:40Z). Buying Down = selling Up on Predict (one book); passive sell-down
  already does it. Taking at the stop costs ~1.2 c/sh at 60 s, ~0.95 c/sh at 240 s; a full hedge removes most of the risk and about
  two thirds of v1's 60 s trading profit. Partial (cut to 25 sh) costs little; CAP10 + H25 and RO60-240 + H25 are the in-sample sweet
  spots. Not tested: a hedge during the window (changes later decisions: needs an engine token and a replay).
- Open (user): candidate(s) to freeze / declare before the 11-09 02:00Z forward run.

## 2026-11-07 03:04Z (session 62f3ce33) - spread capture by arm

- pf_spread_capture.py -> step4/report_spread_capture_v1.json (STATUS 03:04Z). All arms capture the same ~1.2 c/sh spread; the pull
  line is the edge (1 s move +0.28 / +0.09 c/sh vs real top-5 makers -0.85). Arms that let the reducing side ignore the line lose
  that edge on those fills. RO60-240 (sell down to 240 s, line kept) is the only arm adding +1 s $ at both timings after the 20-way
  adjustment (+10.1 / +6.0 $/day). CAP10 cuts risk but costs +1 s volume.
- Open (user): which arm, if any, to freeze for untouched days or declare into the 11-04..11-08 race before 11-09 02:00Z
  (candidates by this view: RO60-240; by risk: CAP10).

## 2026-11-07 02:56Z (session 62f3ce33) - CAP10 attribution (Codex), verified

- Commits 96d97a31 / 13e36621; report step4/cap10_attribution_20261107/report.json. Totals, robustness stats and p-values recomputed
  here and equal. CAP10's mark@stop equals v1's because less spread is offset by less adverse drift; its settle gain is carry, last 4
  days only, p 0.42 / 0.65 raw, ~1.0 after selection. Risk lower 9/9 days. STATUS 02:56Z.
- Open (user): freeze CAP10 for untouched days, or declare it into the 11-04..11-08 race before the 11-09 02:00Z run, or neither.

## 2026-11-07 02:37Z (session 62f3ce33) - INCSIZE 2-day check DONE (Codex), verified

- Committed 73ba8594 / 2dcb5985 / 4c84e17c; report step4/incremental_20261107/report.json. v1 parity 4/4 byte-identical, totals
  recomputed here. Requests -4.4 / -4.7%, mark@60 s +6.0 / -0.5 $ (J / X, 2 days), settle gain is carry. Nothing to adopt; the slot is
  free (no Codex job running). STATUS 02:37Z.
- Supersedes the 02:27Z "in progress" note: the Codex files may now be edited normally (commit the change; the result cites 2dcb5985).

## 2026-11-07 02:27Z (session 62f3ce33) - Codex handoff: INCSIZE variant (running elsewhere)

- Codex built an opt-in INCSIZE order-management variant (pf_step4_ladder_incremental.py, builder pf_ladder_incremental_build.py,
  planner pf_incremental_size.py; receipt predictfun/reviews/incremental_size_20261106/receipt.json, 192 source-only tests). Hashes
  verified 02:27Z; frozen engine and race files unchanged. Details: STATUS 02:27Z.
- IN PROGRESS (Codex, not this seat): v1 vs v1+INCSIZE on 10-24 and 11-01, J and X, one run_light job at a time. Do not launch
  duplicate jobs or edit the 6 Codex files until Codex sends the result artifact.
- Watch out: the Codex files are untracked; commit them (with the result's run tags) before citing any number. Two consumed days =
  point estimates, no interval. Entering the forward race would need a superseding race receipt before the 11-09 02:00Z run.

## 2026-11-06 17:15Z (session 62f3ce33) - D1 inventory mechanisms on v1 (in-sample)

- Engine pf_step4_ladder_inv.py (generated, pf_ladder_inv_build.py; frozen engine untouched), runner run_inv.sh, checks
  pf_inv_check.py (PASS), report step4/report_inv_v1.json; STATUS 17:15Z has the table. No arm beats v1 with an interval excluding 0.
- CAP10 (stop adding at 10 sh) halves max drawdown and cuts $150-cap halts 27 -> 11% (J), 10 -> 3% (X); mark@stop about equal.
  Selling down after 60 s with the line exempted (RO60-240+RX5, D1ALL) halves day sd but loses 35-50% of settle; skew and the $ limit do
  ~nothing.
- Open (user): add CAP10 (or another arm) to the forward race as a declared candidate (superseding race receipt) before the
  11-04..11-08 read; capped replay with the port's halt rule; run_forward.sh after 11-09 02:00Z.

## 2026-11-06 15:23Z (session 62f3ce33) - $150 loss cap vs round-i simulation (read-only)

- Frozen v1 replay has no loss cap; the live $150 cap binds (refuses places) within 1 day from ~54% of simulated starts and the
  uncapped path reaches the port halt within 1 day from 27% (J) / 10% (X). Report step4/report_cap150_round_i.json; STATUS 15:23Z.
- Open (user decision): capped replay of v1 on the 9 round-i days with the port's halt rule, to price the cap. Needs a generated
  engine variant (frozen engine untouched).

## 2026-11-06 09:23Z (session 62f3ce33) - FROZEN pm_ladder_v2; forward race v1 + v2 declared

- Receipts: predictfun/freeze/pm_ladder_v2/RECEIPT.md, predictfun/freeze/forward_20261104_08/RECEIPT.md (supersedes v1's forward
  procedure; v1 rule unchanged). Never edit either; changes = new version.
- NEXT: after 2026-11-09 02:00Z run predictfun/research/run_forward.sh (sequential run_light jobs; ~2-3 h), then read
  step4/report_forward_20261104_08.json. Nothing may read 11-04+ before that. The +1 s vs top-5 comparison is computed after the rule.
- Live port: v2 would need a daily refit on this box + daily delivery of 2 models + 1 lines file; not in scope until the user decides.

## 2026-11-06 08:14Z (session 62f3ce33) - line stability + week split (read-only)

- Fixed pull line quotes 6-8% of minute-0 time on weekdays, 9-18% on weekends. Both arms earned far less in week 2 (10-31..11-03)
  than week 1 (10-23..27), same weekdays: candidate J +92.6 -> +8.4, X +76.9 -> -0.4 $/day. v1 may fail its forward check if this
  persists. Open user decision: a rolling-refit v2 raced beside v1 (must be frozen before any 11-04+ read).

## 2026-11-06 07:04Z (session 62f3ce33) - port: lockstep equal; guard values 1,727,227 / 1,728,000 bit-equal

- v1 known issue: the guard models split on bn_flow noise (|thr| < 1e-9); the 773 rows differ by cumsum residuals only. v2 item: round
  the flow features before training. Nothing to change in v1. Port next: live wiring (pfmm feedback, holds, shadow).

## 2026-11-06 06:26Z (session 62f3ce33) - lockstep trace for the exec port DONE

- data/predict_fun/research/step4/trace_pm_ladder_v1/README.md: 9 traced windows (8 x 11-03 J, 1 x 10-25 X own-cross), byte-identity with
  round i PASS. Regenerate: pf_ladder_trace_build.py, then pf_step4_ladder_trace.py with PF_TRACE_DIR / PF_TRACE_MIDS; check with
  pf_ladder_trace_check.py. The port session (ctanew-5c) builds model / feature parity itself; it may ask for more windows or fixtures.
- The 377-share queue figure was not live (replay definition, void L2c arm). No live queue measurement exists.

## 2026-11-06 06:09Z (session 62f3ce33) - FROZEN pm_ladder_v1; handed to the exec-port coding session

- Receipt predictfun/freeze/pm_ladder_v1/RECEIPT.md (never edit; changes = v2 receipt). Code commit 8d5b02a0; artifacts + receipt in the
  next commit. Builder pf_freeze_pm_ladder_v1.py (guard models saved; parity with round i exact).
- Forward check declared in the receipt: 11-04..11-08, PASS iff candidate sum > 0 at J and X and candidate - control > 0 at J and X.
  Needs: step1 / features / guard values for those days rebuilt with the frozen commit (frozen models, no refit), then ladder replays.
  11-08 completes 11-09 00Z. Not started.
- Hand-off message to the coding session ctanew-5c (tmux rust): implement pm_ladder_v1; earlier research numbers VOID (sent after
  the freeze commit; the send is recorded in STATUS).

## 2026-11-06 05:09Z (session 62f3ce33) - refresh decomposition (section 59) + ROUND-i SUMMARY (section 60)

- pf_mm_decomp_fw.py: section 57's decomposition on the refreshed (fwJ / fwX) and grid runs; all checks PASS. Refresh trades a little
  spread for less 1 s adverse selection at 5/25, net within noise; vs top 5 unchanged; last minute still loses after 1 s.
- Section 60 = the round-i summary (levels, lever table, 7 findings, standing candidate). Read it first for the current picture.
- Standing candidate (NOT frozen): 5/25 S60 per-minute lines + ladder, V0 + GS on the 50 ms grid; control today S60.
- Open for the user: forward replay (11-04+) of a frozen candidate; telling the exec-port session its research numbers are VOID.

## 2026-11-06 04:17Z (session 62f3ce33) - whole-window feed refresh DONE (docs section 58): adds nothing past 65 s

- Every-frame guard refresh over 0-240 s vs the 50 ms grid, fixed live engine round i, in-sample, 5/25 and 20/100, J and X, S60..S240.
  Inputs parity and gate PASS. No MM interval above 0 except s5/S60 at X (the known first-65 s result); 5/25 +1 s c/sh slightly higher
  in every minute (direction only), fewer shares filled; 11-03 worse in 14/16. Not adopted.
- Not measured: post-1 s loss by minute on the refreshed runs (pf_mm_decomp could be pointed at the fwJ / fwX dirs).
- Open for the user: forward replay (11-04+) of a frozen fixed-engine candidate; telling the exec-port session its research numbers are VOID.

## 2026-11-06 02:41Z (session 62f3ce33) - whole-window feed refresh RUNNING (declared STATUS 02:38Z) [done 04:15Z, see above]

- Correction filed: the round-i fresh-refresh result is 2 of 18 comparisons excluding 0 (today arm, bound timing, +12 $ / day), not 1.
- Running: research/run_fresh240.sh inputs -> ctrl -> fullJ -> fullX (log data/predict_fun/research/logs/run_fresh240.out, ~2.5 h).
  Every-frame guard rows over 0-240 s (rows_rl_d2_pk_ev240; first 65 s = the 65 s build, checked on 10-24), preds_tf60 -> preds_t60kF
  (merge refuses unless its first 65 s = preds_t60kE). Arms S60..S240 x 5/25 AV, 20/100 A1, J and X, tags fwJ / fwX; gate fw_ctrl_cmp
  (S60 refreshed-240 = refreshed-65, S240 refreshed != grid). Then read step4/report_fresh240.json, write docs section 58.

## 2026-11-06 02:11Z (session 62f3ce33) - pre-fix logs deleted; MM vs settlement + top makers + window diagnosis DONE (docs section 57)

- USER "2 first": 914 pre-fix replay logs deleted (kept 289: round-i / invwin / repair_smoke_v5 + data / model / market-only logs).
- pf_mm_decomp.py splits every fill into spread / informed (1 s) / drift (to stop) / carry (stop -> outcome) and compares +1 s per minute
  with the top makers' real fills; reconciles to the engine on every arm-day. Result: MM edge is spread minus 1 s adverse selection; the
  settlement part is outcome noise; longer windows add spread but lose it to adverse selection and drift; the last minute loses after 1 s
  for every maker band and every arm of ours.
- Open for the user: forward replay (11-04+) of a frozen fixed-engine candidate; telling the exec-port session its research numbers are VOID.

## 2026-11-05 21:52Z (session 62f3ce33) - inventory x window DONE (docs section 56): neither lever adds money

- Fixed live engine round i, in-sample, 5/25 and 20/100, J and X timing. Gate PASS (never-trigger controls equal, CAP5 falsifier differs);
  s5 S60 = ladder today (V0 + GS) 9 / 9 days. 36 comparisons per timing: none excludes 0 on mark@stop or settle. CAP / reduce-only cut
  carry sd on every read day (risk only); longer windows raise settle point estimates (carry), wide intervals; bound 20/100 +1 s
  markout < 0 for W >= 120 s. Neither lever adopted.
- Open for the user: forward replay (11-04+) of a frozen fixed-engine candidate (per-minute lines + ladder family is the only
  in-sample positive at both timings, section 55); whether to delete the logs that still print pre-fix numbers; telling the exec-port
  session its research numbers are VOID.

## 2026-11-05 17:35Z (session 62f3ce33) - round-i ladder DONE (docs section 55); inventory x window RUNNING

- Fixed engines, in-sample: per-minute lines + ladder positive at both timings (pm+ladder +63.6 J / +52.0 X $ / day; ladder adds
  above 0 at both); 11-03 negative almost everywhere. No forward day on the fixed engines yet.
- Running: run_invwin.sh ctrl -> fullJ -> fullX (log data/predict_fun/research/logs/run_invwin.out), ~2.5 h; then read
  step4/report_invwin.json and write docs section 56.
- Open for the user: forward replay (11-04+) of a frozen fixed-engine candidate - the only out-of-sample check available.

## 2026-11-05 17:20Z (session 62f3ce33) - PRE-FIX RESULTS CLEARED; C2 stopped; inventory x window test declared

- USER: clear the results from before the fix and use only the fixed engine. Deleted 163 pre-fix replay outputs (4.8 GB; list in STATUS
  17:17Z), stopped C2 (pf_step4.py never got the fixes) -> C1 / C2 VOID; docs/EXPLORATION banner + 33 VOID / 12 PARTLY VOID sections.
  EVERY replay number in the HANDOFF entries below (before 15:30Z) is VOID - do not quote or compare against it. kJ / kX are gone:
  the round-i baseline is kJi / kXi (C1-C4 S60, 8 read days $ / day: J 49.6 / 140.2 / 96.3 / 45.7, X -0.4 / 20.8 / 29.3 / 17.9).
- Declared (STATUS this time): run_invwin.sh ctrl -> fullJ -> fullX + pf_invwin_report.py (inventory CAP / reduce-only x window
  60-240 s, sizes 5/25 AV and 20/100 A1, both timings). Starts after the round-i ladder chain ends (fullX + report).
- Logs in data/predict_fun/research/logs still print pre-fix numbers; they are not results.

## 2026-11-05 15:30Z (session 62f3ce33) - R7 / R8 fixes verified; round-i chain RUNNING

- User fixed R7 / R8 and asked for a final review, then the tests. Review (diff vs /tmp/predictfun-repair-loop-20261105/*.before) + synthetic
  check `reviews/simulation_20261105/check_fixes_v4.py` PASS, falsifiers fire on the pre-fix files (STATUS 15:30Z). Minor non-split path note
  in STATUS (no planned run uses it).
- Running: run_ladder2.sh stages fix1 -> view -> fix2 -> par -> smoke -> fullJ -> fullX with round suffix i (kJoi, viewi_*, kJi / kXi, pari,
  ownoffi, smki, lJi / lXi, report_ladder2_i.json). Log data/predict_fun/research/logs/run_ladder2_i.out.
- Next: read fix1 PARITY (must PASS), view PASS, ownoffi own_cross > 0, kJi / kXi (corrected C1-C4), report_ladder2_i.json,
  docs section 55. (Done 17:0xZ: all gates PASS.) Watch out: C2's driver launches 11-05 at ~11-06 02:10Z via run_light, which refuses while a pf_ job runs - the chain
  (~2.5 h) must be done by then.

## 2026-11-05 15:15Z - repair/check loop COMPLETE: reported R7/R8 cleared; 145 checks + integration smoke pass

- User requested an autonomous repair loop. Superseding receipt: `reviews/simulation_20261105/receipt_v5.json`. Re-run the guarded
  regression gate with `research/run_simulation_checks.sh`; tests live in `research/tests/test_simulation_causality.py`.
- BOTH engines now keep client-visible cancel reservations independent of physical fills, including canceled pending creates.
  Reservations release on observable terminal acknowledgements. The known-view loss cap also uses notified cash/inventory and
  client-visible open quantities. Physical matching/accounting and explicit physical regression controls are preserved.
- `pf_view_check.py` refuses missing/empty/mismatched run manifests, incomplete arms/files, wrong cohorts, invalid event data and
  counts, and invalid notification delays; complete zero-fill arms remain valid. Its limited early-replace diagnostic is documented;
  full causal request-stream checks (including cancel-pending fills and throttling) are the source-only regression gate.
- Repair passes: 143 pass / 2 fixture failures (own-cross fixture corrected), then 145/145 pass twice. Earlier fixed/off controls
  pass. Coverage includes both sides, partial/full fills, notification delays, request pressure, loss caps and sequential variants.
- Integration only: first 8 already-read 10-24 windows, tag `repairloop_v5`: live and ladder base fills + requests identical; engaged
  ladder accounting / own-cross checks pass; the checker reads both real smoke runs. Comparator corruption and missing-run controls
  pass. Source in `reviews/simulation_20261105/repair_smoke_v5.py`; results in its JSON. Peak RSS 1,432 MiB, no forward days read.
- No confirmed reported code findings remain under these gates. Full economics / ladder grids still need fresh output tags on the
  repaired bytes; earlier kJh / kXh / lJ / lX are not revalidated. Fill calibration / own impact remain model limits.

## 2026-11-05 14:59Z (session 66f8e5c3) - round-3 chain done, SUPERSEDED by review 4 (R7 / R8); HOLD until the user hands back

- Done: chain br79kqi74 on the round-3 engines (kJh / kXh, parh PASS, ownoffh own_cross 1, smoke, lJ / lX + report_ladder2.json:
  recon 0 / 324, cross-check vs kJh C4 equal on 9 days). Every number in it is provenance only: R7 (throttle open_n reads the
  matcher's remaining size of cancel-pending orders) affects every arm, base included (all use XV). Do not quote kJh / kXh / lJ / lX.
- Hold: user said not to run tests now; they clear R7 / R8 themselves, then hand back. Pointers for R7 in STATUS this time.
- Next when handed back, in order: (1) fix1 regression with all old switches incl. PF_POLICYVIEW=physical (must reproduce the old
  throttle exactly); (2) view check with the fixed checker (must refuse missing / empty / mismatched runs); (3) a throttle
  look-ahead check that fires on the physical view and is 0 on the known view; (4) run_ladder2.sh fix2 -> par -> smoke -> fullJ ->
  fullX, then read kJ* / kX* vs kJ / kX (corrected C1-C4 baseline) and report_ladder2.json, write docs section 55.
- Watch out: run_ladder2.sh hardcodes tags kJh / kXh / parh / ownoffh and lJ / lX, and pf_ladder2_report.py:24 cross-checks against
  kJh - bump the tags (e.g. kJi ...) and LIVEKJ together, or the rerun overwrites lJ / lX and checks against the stale kJh.
  Results stay in-sample (lines and guard fit on the 8 read days; 11-03 beside); compare with the negative ladder test docs 36-38.
  Freeze / forward replay of C4 on 11-04..08 still waits for the user.

## 2026-11-05 14:45Z - review 4: direct rung leak fixed; throttle leak in BOTH engines + checker refusal OPEN

- Supersedes v3: `reviews/simulation_20261105/receipt_v4.json` + `check_v4.py`, via `research/run_light.sh`. Source-only synthetic
  review plus temporary synthetic parquet fixtures; no market data read, no simulator changes by reviewer; history impact unmeasured.
- All earlier fixed/off controls pass. Original R6 rung path is fixed, including full guard stack and L_FILL=1000; physical-view
  control still fires. R6 remains incomplete through R7 below.
- P1 R7 BOTH engines: `open_n` still counts cancel-pending orders via physical `rem`. With the known view + actual XV/44 ms answer
  cap, hidden fill at 343 ms (notification 443 ms; cancel effect 380 ms) enables a placement at 353 ms. Identical public-input no-fill
  branch does not place then; all newly received public frames are stamped before 343 ms. Without XV the pre-notification requests
  match. Use policy-visible lifecycle state for request reservation/authorization, including cancel-pending orders.
- P2 R8: actual `pf_view_check.py` main reports PASS / exit 0 for a nonexistent known run (physical positive fixture flags one early
  replacement). Existing known-bad fixture correctly fails. Refuse missing/empty/mismatched artifacts; compare full causal request
  streams, including cancels and cancel-pending fills, rather than relying only on its current early-replace count.

## 2026-11-05 12:40Z - review 3: prior fixes pass; ladder reacts to unreported fills (R6 OPEN)

- Supersedes review v2: `reviews/simulation_20261105/receipt_v3.json` + `check_v3.py` (run through `research/run_light.sh`). Source-only
  synthetic review, no market data read by checker, no simulator changes by reviewer; historical impact unmeasured.
- Previous R1 / R2 / R4 / R5 counterexamples now pass their fixed/off controls; missing/stale guard withholding passes with GS.
- P1 R6, ladder: a rung fills at 90 ms, notification due 190 ms. At 100 ms, on a frame stamped 85 ms, the engine places a new touch
  order + rung. Identical public-input no-fill control sends neither. L_FILL=1000 also sends at 100 ms. Reproduced under the full
  active MICRO / AV / six guards / V0+GS+LD2+XNA+XMIN1+XV+S60 stack with positive depth and flat Binance inputs.
- Cause: matcher removes `orders[lane]` immediately; `cur_order()` reads that physical state inside `desired()`, before K arrives.
  Separate policy-visible own-order state from physical matching state and update it on received acknowledgements by order id.
  Delaying knet alone does not enforce the information boundary. Add a causal equality test before re-evaluating ladder comparisons.

## 2026-11-05 11:38Z - follow-up simulation review: prior fixes pass; atomic sweep bug + rejection recovery OPEN

- Superseding receipt: `reviews/simulation_20261105/receipt_v2.json`; reproducer `check_remaining.py` in the same folder, via
  `research/run_light.sh`. Source-only synthetic inputs, no market data read, no engine changes by reviewer; history impact unmeasured.
- Verified prior price-priority and own-cross fixes with their regression/off controls; GS withholds missing/stale guard coverage.
- P1 R4, BOTH engines: trade tuples sort ascending by price at a shared timestamp, replaying a sell sweep worst price first.
  Synthetic 100 sh at 50 + 5 at 49, our bid 10 at 50 behind 100: engine fills 10, maximum valid fill 5. Better-price-earlier,
  single aggregated leg, and mirrored ask controls each fill 5. Retain taker identity; replay atomic sweeps in matching order.
- P2 R5, BOTH engines: venue-book post-only refusal leaves the rejected intent installed (own-cross refusal resets it). When
  recovered known state wants the same quote, no retry is sent. Reproduced with coalesced knowledge times; no-refusal control fills.
- Current kJf / kXf / ladder reruns still contain R4. Fix + rerun before relying on their precise economics; parity alone misses it.

## 2026-11-05 11:22Z (session 66f8e5c3) - review bugs fixed in both engines; chain fix2 -> par -> smoke -> fullJ -> fullX RUNNING

- Fill priority among our own orders, post-only vs our own resting orders, GS strict guard (STATUS this time). Regression gate
  PASS. Earlier numbers carry the fill-priority bug until kJf / kXf report the impact; quote no earlier $ figure without it.
- Next: read fix2 impact (kJf / kXf vs kJ / kX), parity parf, own-cross falsifier (tag ownoff must count own_cross > 0),
  report_ladder2.json, docs section 55.

## 2026-11-05 11:14Z (session 66f8e5c3) - ladder 2 / improve / per-minute lines / fresh refresh RUNNING (STATUS declaration 11:00Z)

- Done: fresh-refresh predictions preds_t60kE (every-frame rows 0-65 s + 50 ms grid; same models, thresholds equal); ladder
  engine pf_step4_ladder.py (parity with pf_step4_live PASS on 10-24 / 11-01, controls PMzero / AMc006 / LDinf / IMinf equal,
  falsifier ok); LH (rung hold) added after the smoke showed guard-driven rung churn; parity re-run in the chain.
- Running: run_ladder2.sh par -> smoke -> fullJ (8 arms x 2 prediction sets x 9 days, then pf_ladder2_report). Next: fullX
  (bound timing) and docs section 55.
- Watch out: in-sample (lines and guard from the same 8 days); prior negative ladder test docs 36-38. run_ladder.sh is the
  11-03 driver (restored verbatim after an accidental overwrite at 10:58Z); the new driver is run_ladder2.sh.

## 2026-11-05 11:12Z - simulation code review: two defects + guard-policy risk OPEN

- Review receipt + synthetic reproducer: `reviews/simulation_20261105/receipt.json` and `check_engine.py`. No simulator code changed;
  the reproducer reads source only, no market data. Historical frequency / P&L impact NOT measured. Run via `research/run_light.sh`.
- P1: both live and new ladder engines, with ladder tokens off, fill cancel-pending orders before better-priced replacements
  (old bid 49, replacement 50, both live: a sweep fills 49). Sort all fillable orders by price then activation time.
- P1: new ladder post-only / own-cross checks omit cancel-pending orders. Synthetic live bid 51 / pending-cancel ask 51 coexists
  while `own_cross` reports 0. Include every fillable own order in activation checks and diagnostics.
- P2 policy risk: missing / stale per-side guard predictions permit quoting; fresh adverse-value control withholds correctly.
  Verify whether this fallback is intended; report unavailable inputs, and withhold if guard coverage is required. Synthetic
  positive controls and unsupported split/amend refusal passed.
- Next: fix these paths and rerun affected comparisons before relying on precise replay economics. Passing parity does not test
  these properties. A successful reproducer run confirms these behaviors are present; it is not a correctness certificate.

## 2026-11-05 08:33Z (session 66f8e5c3) - per-minute guard percentiles + their fill P&L (descriptive; STATUS 07:27Z, 08:23Z, 08:33Z)

- Per-minute percentile values of the guard: pf_minute_pct.py. Fill P&L by percentile bucket: pf_minute_pnl.py (kJ fills, no replay).
- Minute 1: the money is in the top ~10% of the guard (8/8 or 7/8 days); p70-p90 about 0; fills below the line (pull in flight) lose.
  A minute-1 pull at ~p90 is a hypothesis read off in-sample fills (path effects ignored), not a result.
- Exec-port session's queue-ahead measurement copied to data/predict_fun/research/ext/queue_ahead_execport_20261105/.
- Freeze / forward replay still waits for the user.

## 2026-11-05 06:22Z (session 66f8e5c3) - rebuild 3 DONE (docs §54): edge small, recent days <= 0; longer / larger does not add

- Central (real key + within-frame time): C1 50, C2 155, C3 106, C4 46 $/day (8 read days, no rebate); none passes all declared
  predicates (all fail the E2-only -300 ms stress, a far-tail bound). Latest four days negative on average for every candidate.
- Window x size: nothing adds; longer stops add carry variance, not fill quality.
- C3 freeze recommendation withdrawn. Open for the user: stop here, forward-replay C4 on 11-04..08 (same replay limits), or a
  capped live pilot via the implementation session to measure own fills (the only way to settle book-timed trade timing + impact).

## 2026-11-05 03:05Z (session 66f8e5c3) - second review found timing defects; rebuild 3 (STATUS 03:01Z)

- Verified: the transaction key merged different taker orders (42% of keys on 10-24 12-18Z); trades sit at frame stamps up to one
  ~50 ms frame after the match; the replay padded Binance / books into 11-04 (read record in STATUS). The old money numbers (§53) are
  not reliable; C3 / C4 / C1 / C2 must be re-measured. The window / size grid (rbw / rbwX) was stopped, superseded.
- New: pf_txmap.py (taker order per leg from REST), engine PF_TXKEY=tk / PF_TXJIT=frame + input bounds + code hash in summary,
  pf_badfill_grid_tx PF_TXKEY=tk. Chain run_review3.sh: txmap 9 days -> grid_tk -> guard refit preds_t60 -> preds_t60k -> replays kJ
  (central) / kX (bound) on 14 arms, kS / kE / kM on C1-C4 -> pf_review3_report.py. Predicates in STATUS 03:01Z.
- No freeze, commit or implementation message before the user chooses; validation = 11-04..11-08 forward days of a frozen version.

## 2026-11-05 01:50Z (session 66f8e5c3) - second review + window / size grid RUNNING (STATUS 01:49Z)

- Second independent code review (agent, read-only) running. Fix checks done here (start withhold 0 early fills vs control 39-106;
  pending fix touches only pf_pend_*; tx-time instrument 39,351 -> 0 multi-time transactions, engine REFUSES otherwise).
- run_review2.sh waits for C2's 11-04 run (02:10Z), then: stress rbtxE (E2-only -300 ms), rbtxS50 (-50 ms); window x size grid rbw
  (S60..S240 x AV 5/25, 20/100, 40/200) + stress rbwX; then pf_window_size_report.py. ~1.7 h.
- 11-04+ are forward days: not read (no freeze yet). Freeze still waits for the user's version choice.

## 2026-11-04 15:40Z (session 66f8e5c3) - reliability review DONE (docs §53); freeze waits for the user's choice

- After every correction (one time per taker transaction, start withhold, pending features per frame, tx-timed labels + refit, XV,
  queue bound, 25 ms shift) C1 / C3 / C4 pass all declared predicates; C2 fails top-5 per share. Money about half of first reported
  (C3 138 $/day mark, no rebate). 11-03 (only unchosen day) ~0 for all. Forward days 11-04+ decide.
- Freeze path when the user picks: commit predictfun/ (untracked today), run pf_guard_freeze.py <name> 60 at that commit (selftest on
  11-03 preds_h60), commit predictfun/frozen/<name>/ + a receipt (arms, live rules, forward pass rule, multiplicity), then message
  the implementation session (exec port: needs V0 per-side guard with 2 text models + thresholds, AV size, start withhold, concurrent
  replace, XV throttle; today it runs sequential replace at 1 place/s).
- Engine options added today (default off): PF_TXTIME, PF_TXSHIFT, PF_QACT; size kind AV. Feature option PF_PEND_KNOWN=frame (_pk).

## 2026-11-04 13:45Z (session 66f8e5c3) - reliability review RUNNING before a freeze (STATUS 13:41Z)

- User: review the result, then freeze a version and tell the implementation session. run_review.sh (log run_review.out):
  pf_step5_guard_preds_wf (walk-forward guard, preds_w60), pf_preds_startknown (withhold until the start price is known: first
  predictMarketChanged patch, Tokyo clock -> preds_g60k / preds_w60k), replays tag rv (C1 5/25, C2 20/100, C3 AV 20/100 v>=0.06;
  each +/- XV). Then pf_review_report.py (predicates R_START / R_WF / R_DAYS / R_TOP5). An independent code-review agent runs too.
- Freeze builder ready, not run: pf_guard_freeze.py <name> <k> (refuses unless the pipeline is committed; selftest on 11-03).
  predictfun/ is entirely untracked in git today: the freeze commit adds it (check: no secrets found by a pattern scan, 2.9 MB).
- Implementation side: pmmmrust docs/PREDICT_FUN_PLAN.md (exec port: sequential replace + 1 place/s today; the candidate assumes
  concurrent replace + venue throttle XV - a change to flag). Sessions: "Predict fun - Tokyo" (Remote Control) / pmmmrust.
- The 50 / 70 size chain (run_size_dyn.sh) was stopped at 13:40Z after gzd50_20261102; skip logic resumes it.

## 2026-11-04 12:07Z (session 66f8e5c3) - size / stop / dynamic size at pull 60% DONE (docs §52); 50 / 70 RUNNING

- Size scales $ almost linearly at unchanged quality (S60 20/100: mark@stop 355 $/day vs 125, 8/8) - but no own impact in the
  replay and 100 sh = 37% of the touch. Longer stops: not reliable (CI spans 0; 11-03 worse). AV guard-value sizing: best per share,
  beats fixed at matched shares at v >= 0.06, but less total $ than fixed 100. A3 rejected; A1 = fixed hi in practice.
- run_size_dyn.sh continues (gz / gzd at k = 50, 70), final report size_stop_report_all. Open ideas (not declared): skip 60-120 s
  only; reduce-only after the stop to cut carry; own-impact model before trusting 50-100 sh; 11-04 unseen check after C2.
- Watch out: pausing the chain = kill -STOP its bash between runs (the report can't win the slot otherwise); never pkill -f a
  pattern that is in your own command line.

## 2026-11-04 10:38Z (session 66f8e5c3) - size x stop grid + dynamic sizes RUNNING (STATUS 10:32Z, 10:37Z)

- run_size_dyn.sh (setsid; log data/predict_fun/research/logs/run_size_dyn.out) took over from run_size_stop.sh after its first day-run;
  finished day-runs are skipped. Dynamic arms tag gzd (fixed 25/50/100, AV guard-value size, A3 depth), new engine size kind AV (additive).
- grid: run_size_stop.sh (stopped): trained guard V0 at k = 60, then 50 / 70; stop 60 / 120
  / 180 / 240 s x size A1:5:25:0 / A1:10:50:0 / A1:20:100:0; 8 read days + 11-03, tag gz. pf_size_stop_report.py runs after each k
  (non-fatal). Parity: gz S60 A1:5:25:0 must equal the gd run. Caveat to carry: the fill model has no own impact (our order is never
  shown to others), so large sizes read optimistic; compare hi size to the displayed touch.

## 2026-11-04 09:13Z (session 66f8e5c3) - step-back variants DONE: keep withholding (docs §50)

- One tick back (SBV) and prediction-priced (SBP) both fill far more and badly (+1 s -0.35 / -0.20 c/sh at 50% vs withhold +0.34).
  Best rule so far: trained per-side guard (V0), withhold, pull 50% (most dollars) to 60-70% (top-maker quality). To do: 11-04 as
  the second unseen day (after C2 at 11-05 02:10Z: pf_step5_features_rl, pf_step5_c3_preds apply, pf_step5_guard_preds with
  PF_GUARD_KS=50,60,70 and the day, replay arms).

## 2026-11-04 08:22Z (session 66f8e5c3) - pull sweep 10-70% DONE (docs §50)

- Trained per-side guard (preds_g<k>, engine token V0) reaches the top-maker quality at a 60-70% pull (bad 9-10%, ratio 33-42%,
  +0.53..+0.79 c/sh); total mark-at-stop peaks at 50% (+920 / 8 days vs +355 now). 11-03 (unseen) agrees in direction. Running:
  one-tick step-back (SBV, run_guard_stepback.sh) then prediction-priced step-back (SBP, run_guard_predstep.sh). Next: 11-04 as a
  second unseen day once C2 has processed it (after 11-05 02:30Z; build its features_rl_d2 + preds_c3 + preds_g<k>).

## 2026-11-04 05:59Z (session 66f8e5c3) - guard: trained M1 is the one to keep; latency inputs add nothing (docs §50)

- Matched-pull control: M1 guard beats the current guard's input at the same pull (+0.15 c/sh at 30%, 8 days; +0.30 on 11-03).
  Latency-aware inputs (frame age, Binance move since the frame) add nothing to it. Best so far: bad 13.5%, ratio 64%, +0.14 c/sh
  (target <= ~10%, <= ~45-50%, ~+0.3). Next candidates: pull 40 / 50%, anchor as quote centre, 11-04 as a second unseen day.

## 2026-11-04 05:25Z (session 66f8e5c3) - anchors tested; the lead-adjusted one passes (docs §50)

- Binance-only fair values (p_bn, recalibrated, fitted A5) fail; Predict's late price + the Binance move since that frame (A6..A8)
  passes: Predict makes ~1.1x the adjustment within 1 s and keeps it. Stage 3 guard: +0.14 c/sh at a 30% pull, matched-pull control
  (pf_step5_baseguard_preds, run_guard_matched.sh) running. Next: dp / A8 into the guard and as the quote centre.

## 2026-11-04 04:22Z (session 66f8e5c3) - stage 1 (anchor) and stage 2 screen DONE (docs §50)

- No Binance fair value beats Predict's own (late) price on settlement or leads it; Predict book signals add ~0.003 AUC to the
  guard model; everything we record tops out at AUC ~0.83 vs 0.78 now. Open (user decision): a lead-adjusted anchor (Predict's
  late price plus the Binance move since that frame), stage 3 replay of the best guard, or conclude.

## 2026-11-04 03:59Z (session 66f8e5c3) - research flow set (docs §50)

- Stage 1 anchor test next (pf_anchor_eval.py), then stage 2 predictor (pf_badfill_info2 running is its first screen), stage 3
  combined replay vs the top-maker target (bad <= ~10%, bad$/good$ <= ~45-50%), then fresh days 11-03+.

## 2026-11-04 03:03Z (session 66f8e5c3) - 10-venue big-order check DONE (docs §49)

- pf_big_prints.py / pf_big_prints_split.py: 1 bad fill in 11 has a big order elsewhere in time for our cancel (half of those off
  Binance); most big orders come with or after our bad fills. Not checked: Polymarket's parallel BTC 5-min markets (P-2026-003
  data, ask first), Chainlink's own fast stream.
- Watch out: data_collectors/fetch_public_trades.py writes rows sorted by (time, price); order by id before any tick test.

## 2026-11-04 02:59Z (session 66f8e5c3) - anchor sweep cut + information test DONE (docs §48)

- Sweep: R^2 does not rank anchors (synthetic 0.15 >> real model at 0.15); cut at the calibration check.
- pf_badfill_info.py: current features rank bad exposures at AUC 0.78 (guard input) / 0.83 (trained model); outside feeds add
  ~0.5%; ideal 0.999. New public trade data for 7 more venues in data/exchange_public/ (README); pf_big_prints.py (10 venues) next.

## 2026-11-04 01:29Z (session 66f8e5c3) - ideal-anchor test DONE (docs §46)

- preds_oracle (realised 1 s / 250 ms moves) through the unchanged replay: strongly positive every day; the real model
  (R^2 0.11-0.20) is break-even in the first minute. The anchor's quality is the lever.
- Next candidate (user decision): a degraded-anchor sweep (realised move blended with noise to R^2 0.2..0.9) to find the
  quality at which quoting pays, then ask whether any feed can reach it.

## 2026-11-04 01:00Z (session 66f8e5c3) - request limit verified + replace compare DONE (docs §44-45)

- Predict's headers: create/remove 40/s + 500/min, reads 500/min, one budget per key (pmmmrust plan still says 240/min).
- pf_replace_compare.py: parallel replace helps a little (+1 s -0.17 -> -0.10 c/sh) but the swept share stays ~65%;
  queue position and the late feed are the cause.

## 2026-11-04 00:28Z (session 66f8e5c3) - re-post anatomy DONE (docs §43)

- pf_repost_anatomy.py: fast re-posts are makers sending cancels and new orders in parallel within one ~50 ms publication,
  several orders at once; not aggressive-order remainders. Our live plan is sequential + 1 place/s and sees the feed ~53 ms
  late.

## 2026-11-03 23:58Z (session 66f8e5c3) - cross-venue check DONE (docs §42)

- pf_cross_venue.py, pf_sweep_reversion.py; Binance spot aggTrades for the 8 days in data/binance_spot_vision/ (public,
  checksums OK). Bad sweeps are informed (they last) but no recorded feed leads them by much; makers pull on Binance perp.
- READ RECORD (STATUS 23:51Z): io.hours_of(day, 1, 1) padding loads the Binance tape of 10-28 00Z / 10-30 23Z (for 10-27 /
  10-31) and 11-03 00Z (for 11-02), since the project's first scripts. Recorded, no harm under the 10-22 rule; fix pf_io
  (refuse sealed hours) only after C2 finishes, since pf_io is a C2 dependency.
- Open (user decision): Polymarket parallel BTC 5-min markets as the sweepers' source (P-2026-003 data: ask first); other
  exchanges (OKX / Bybit public trade archives); a Binance-perp-flow guard in the replay.

## 2026-11-03 23:31Z (session 66f8e5c3) - speed race DONE (docs §41)

- pf_speed_race.py, pf_sweep_precursor.py, pf_vacate_gap.py. Binance-led risk: we are fast enough. Most bad fills: no
  Binance move and no warning visible within our 75 ms blind spot. Front: freed prices are re-posted in the same book
  update in 62% of cases; we could be first in ~14%.
- Next candidates (user decision): what the sweepers know (do bad sweeps follow the settlement reference feed we record,
  data/predict_reference?); replay a wider Binance pull (all ladder levels on 0.5 bp); a live latency probe (our real
  Predict feed / order path vs the 13 ms feed minimum).
- Watch out: pf_speed_race peaked at 2006 MiB RSS (guard 2048); split by day before adding anything to it.

## 2026-11-03 23:05Z (session 66f8e5c3) - fill mix vs the other makers DONE (docs §40; corrects §38)

- pf_fill_mix.py (transaction sweep rule, key validated 98.4% by pf_match_type.py), pf_skip_match.py. Our quoting is not
  the gap: at our own price we match the top makers. We lose on mix (64-73% swept vs 18-21%) and on which sweeps reach us
  (ours far more informed). Both are speed / queue position. Watch out: the book "cleared" rule under-counts sweeps; use
  the transaction rule.

## 2026-11-03 22:26Z (session 66f8e5c3) - maker behaviour DONE (docs §39)

- pf_maker_behavior.py: no queue held from listing; fills come at levels formed in the last 0.1-10 s, mostly at levels born
  as a new best; the front of a new level is traded within 1 s in 55-72%. Top 5 are not over-represented at the front; they
  fill deeper, lose less when picked off, and end windows nearly flat, with small trading P&L per share.
- Natural next checks (user decision): who creates new best levels and how fast after a Binance move (could we be the
  creator at our latency?); liquidity rewards (T5) as the top makers' income.

## 2026-11-03 22:12Z (session 66f8e5c3) - hold check DONE (docs §38; corrects §37)

- pf_hold_check.py: most of the size resting behind the best at the open is cancelled within 10 s (lower bound >= 53-71%);
  top makers are swept far less than our ladder (22% vs 74%). §37's "built at listing" was read from an upper bound: void.
- Open: how the top makers reach the front (speed to new price levels vs pulling before sweeps) - a possible next check
  from the maker legs; true queue position of a resting order needs a live test.

## 2026-11-03 21:48Z (session 66f8e5c3) - queue-priority diagnosis DONE (docs §37)

- pf_queue_priority.py (market side) and pf_ladder.py PF_ORDLOG=1 (order log; summary parity with lad1 checked by
  pf_ladder_orders_report.py). We are at the back because the incumbents post at listing (~4.7 min before the open) and
  stay; our ladder re-centres and pulls (median order life 0.76 s). Next is a user decision (pre-window persistent ladder
  test, small live test, or stop).

## 2026-11-03 16:36Z (session 66f8e5c3) - ladder test DONE, negative (docs §35-36)

- pf_ladder.py (multi-order replay, selftest 6/6). All ladder arms lose on mark-at-stop; our deep orders sit behind the
  incumbents' ladders and fill mostly in sweeps. Untested structural idea: a persistent ladder placed at market
  registration (pre-window), never re-centred. Otherwise the replay finds no robust maker edge at our latency and queue
  position; a small live test is the only way to learn real queue dynamics.

## 2026-11-03 11:20Z (session 66f8e5c3) - O4 DONE, C3 2 days (docs §25-26)

- C3 (11-01/02): O3's edge did not appear (P3 -156, +1 s -0.52 c/sh). O4 (sweep protection, 8 days, no hold-out): nothing
  adopted; P3 +1,462 / 8 d with +1 s mark ~0. Tuning inside this strategy family is at the noise floor.
- Open options: real-fill calibration (small live test, user decision); liquidity rewards (T5, unverified); a sweep /
  regime model (new research); keep C2 / C3 running on new days (11-03 from 11-04 ~02:30Z).

## 2026-11-03 09:59Z (session 66f8e5c3) - round O3 DONE (docs §24)

- At the measured Tokyo clock + decision 2 ms: quote only the first 60 s of each window (S60) + stand back one tick (BKA)
  (+ Q025 250 ms guard, B2W100) -> combined +1,617 (seq) / +1,806 (conc) over six read days, 5/6 days, +1 s edge positive.
- Next: fresh-day confirmation C3 (declare + freeze before reading). Pipeline for new days needs features_rl d2, model A
  pred_1000 (model_c2_y1000 = the all-six-days fit of the same spec) and a y250 model fit on all six days' rows_rl_d2.

## 2026-11-03 08:05Z (session 66f8e5c3) - latency test L2c DONE (docs §23)

- Measured split latencies (order live 40 / resting cancel 22 / pending cancel 95 after create / fills known 100 / decision
  10 ms): live-like C2-P loses in every unthrottled and venue-throttled arm; v6 (sequential, 1/s) positive but adverse fills
  and model-sensitive. Levers: order-live time, then our decision time. Replay support for a live go: none at this latency.
- pf_step4_live.py sha 1edab4e7 (split handler + pending-cancel + fill-known settings); tokyo20 / L1 paths unchanged.

## 2026-11-03 05:51Z (session 66f8e5c3) - latency test L1 DONE (docs §22)

- Tokyo probe (pmmmrust fe073d1): no removal lock; limits 40/s + 500/min; Predict feed 53 ms median + 10 ms decision + 22 ms
  order. Replay at that clock (scenario tokyoreal; pf_clock_rl, pf_step5_features_rl, pf_step5_lodo_rl): edge mostly gone
  (+240..600$ / 6 d, +1 s mark <= 0). Live go is not supported by the replay at real latency.
- Next if asked: latency decomposition (which component costs most: Predict feed, our 10 ms, order 22 ms); strategy
  changes aimed at stale quotes (wider / one tick back, stronger guard) tuned at the Tokyo clock, confirmed on fresh days.
- pf_step4_live.py changed (sha 224fde6d): new scenario + XV only; tokyo20 arms reproduce round 2 to the cent.

## 2026-11-02 12:35Z (session 66f8e5c3) - user decision: live = sequential replace + 1 place/s

- Relayed to the exec-port session (it owns pmmmrust and re-freezes). Reference rule = pf_step4_live.py XP<lambda> + XT1
  (sha256 c0b2ba80e9e29621); expected outputs in step4/live_..._livedev4 (and livedev2 for lambda 0). FROZEN as pmmmrust
  pfmm-c2p-tokyo-v2 (code e77a376), now tag v3 (same code + runbook + scripts/pf_tokyo.sh, which reads the §19 grid). Waiting for: the probe's lambda (read off docs §19/§20) and rate headers. Do not edit pf_step4_live.py without
  telling the exec port (its parity fixture pins this hash).

## 2026-11-02 09:37Z (session 66f8e5c3) - live round 2 DONE (docs §17)

- Sequential lock (XS) and the exec-port throttle (XT) added to pf_step4_live.py; `run_live_dev2.sh`, `pf_live_report2.py`.
  Lock >= 200 ms loses in either design; throttle at L = 0 keeps +958 (1/s) / +2,056 (2/s) of +3,067. Sent to the exec port.
- Waiting on the exec port's Tokyo probe (user's step): lambda = removalLockedUntil - acceptance time. Read it off the
  placement-relative grid (docs §19): break-even 163 ms unthrottled / 204 ms at 1 place/s / 180 ms at 2/s; 100 ms keeps
  +522..+1,471$/6 d; 500 ms loses everywhere. (§18 = lock on every cancel, the worst case, break-even ~75 ms.)
  Concurrent design (frozen Tokyo version) in §20: at 1 place/s it is worse than sequential at every lambda <= 200 ms
  (break-even 138 vs 204 ms); unthrottled it is better.

## 2026-11-02 09:02Z (session 66f8e5c3) - live-deviation tests DONE (docs §16)

- For the pmmmrust exec port (C2-P frozen there): `research/pf_step4_live.py` (copy; live rules as X tokens, parity to the
  cent), `run_live_dev.sh`, `pf_live_report.py`. Removal lock and request rate dominate; no amend / min 1 USD minor; the
  150$ cap can stick at -149.96 without halting. Results sent to the exec-port session.
- Next if asked: sequential-lock design (replace only after the cancel lands); a request-budget (throttle) replay.
- pf_step4_live with 14 arms peaked 1,994 MiB (guard 2,048): run fewer arms per process.

## 2026-11-02 02:01Z (session 66f8e5c3) - C2 RUNNING

**In progress:** confirmation C2 (STATUS.yml `confirmation_C2`, declared 02:01Z before any 11-01+ content was read).
- Candidates: P = D1 stack + M025 + A1 5..25 (primary), S = combo G1+G5+M025 + A1 5..25; references R0-R5, C1c.
- Model: `research/pf_step5_c2.py` fit on the six read days (sha be6e916e, 4.02M rows, peak 1.1 GiB).
- Driver: `research/run_c2_auto.sh` (setsid nohup, pid in `data/predict_fun/research/c2/auto.pid`) runs
  `run_c2_day.sh <day>` after the next day's 01Z hour closes (+10 min), appends `c2/admission.jsonl`, and after 5 ADMITTED
  days runs `pf_c2_report` -> `c2/report_c2.json`. Heartbeat `c2/auto.heartbeat`, log `c2/auto.log`.
- Earliest verdict ~11-06 02:30Z (days 11-01..11-05). A FAILED step stops the driver: look at `logs/c2*_<day>.log`;
  any code fix = superseding declaration C2 v2 in STATUS.yml (keep v1), then remove that day's FAILED record and restart.

**Watch out for:** do not change arms, rules, code, model or the pass rule while C2 runs (hashes in the declaration).
The driver only uses run_light.sh (core 8, one process); P-2026-003 must not pin its jobs to core 8.

## 2026-11-01 05:41Z (session 66f8e5c3)

**Done** (write-up: `docs/EXPLORATION_2026-11-01.md` §1-9; exploratory; dev days 10-23..25 fully read)
- T1-T3: sources reconciled (REST volumes; WS pending = USD at YES price; executedAt rounded); plain baseline loses.
- T4: Tokyo latency (user: ~20 ms) -> scenario tokyo20; every D1 rule helps alone (3/3 days); full MICRO stack break-even.
  Untimed-trade bracket narrowed by the book-drop estimator E2 (median |err| 0-49 ms); three timing models agree.
- T6: LightGBM (pinned, `research/pf_model_spec.py`), leave-one-day-out: 1 s mid-move R2 0.14-0.18, 250 ms 0.38-0.48.
  D1 stack + model guard th 0.5c = +$237 / 3 days (+0.24 c/sh, 3/3 days, all timing modes) vs D1 stack -$5.

- C1 confirmation (06:26Z) PASSES its declared rule on 10-26/27/31: D1 stack + model guard +$190.67 (mid; early +148,
  late +226), beats D1 alone on 3/3 days (+102/+86/+87). Small: +0.27 c/share, ~$0.22/window at 5 shares.

- 09:39Z WS-vs-REST check (10-23, docs §11): WS pending is an aggregate, not a trade list; 99.7% of REST volume shows in
  WS, but 8.2% of WS increases have no REST trade at that price within +-1 s -> REST completeness OPEN (<= 9.4%).

- 10:51Z on-chain reconciliation (docs §12): REST complete and exact on 6 slices of 10-23; executedAt = floor(block time);
  orphans = taker orders listed at their signed limit price. Scripts `research/pf_chain_*.py` (public RPC, read-only).

- 13:26Z round O1 (docs §13): size scales P&L, per-share +1 s edge flat (~0.25 c); 6-day model no P&L gain; declared combo
  MICRO+G1+G5+M025 +497 dev / +680 2nd look (inventory uncapped). Scripts pf_o1_*.py, run_opt.sh, run_o1_combo.sh.

- 14:45Z round O2 (docs §14): replay loop fix (in-flight replacement after a full fill now cancelled; restated numbers within
  ~$12, C1 still passes). Adaptive size: model-based (A1) best per-share edge; adopted B1+A6 and B2+A1. Script pf_o2_report.py.

- 15:58Z code review + fill model v2 (docs §15): queue double-count fixed (material), S270/S300 invalid (no predictions
  after 240 s), smaller fixes; PF_FILLMODEL=v1 reproduces old numbers. v2: C1 still passes (+158, margin halved); B2+A1 adopted.
  Use v2 (default) for everything from here; pf_check_fills.py on every new output.

**Next (proposed)**
- Confirmation C2 on fresh days (11-01 onward, complete UTC days; check P-2026-003 has not reserved them first): model fit
  on all 6 read days; candidates B2+A1 (D1+M025, model-sized 5..25) and B1+A1 (combo, model-sized), fixed-size references, fill model v2; declared before reading.
- Optional: on-chain block-ms timing for untimed fills (10-31's 29%); extend the REST-vs-chain check to other days.
- Latency sensitivity of the candidate (Predict 20 -> 58 ms 'tape' scenario; order latency 20 -> 50/100 ms).
- Size sensitivity (5 -> 10/25 shares: queue position and the inventory caps bind differently).
- Liquidity-reward terms (T5): hourlyRate 3000 / 6c / 100 sh could dwarf the +$64/day gross.
- 10-31 data note: 29% of volume untimed (refills within a frame?); check before using later days.

**Watch out for**
- One process at a time via `research/run_light.sh` (sets ARROW_DEFAULT_MEMORY_POOL=system; the 4 GiB AS cap killed one
  model run before). Peaks: step 1 ~1.0 GiB, step 4 ~1.5 GiB, features ~1.7 GiB, model ~1.4 GiB.
- 10-28..10-30: do not read (Binance tape sealed for P-2026-003's order-flow test). Read days now: 10-22 (partial), 10-23..27, 10-31.
- Results are gross at 5 shares with a modelled fill (FIFO behind displayed size, no own impact); no real-fill calibration.
