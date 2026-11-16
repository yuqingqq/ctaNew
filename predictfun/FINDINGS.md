<!-- STRATEGY_SCORECARD_20261116 -->
## 2026-11-16 Common strategy scorecard: frozen, replay accounting in progress

User requested clear strategy names and common spread/markout/settlement
metrics after STOP-EV model diagnostics were confused with strategy P&L.
Source 7f0ab3a7, freeze 819e59fe. No fit, threshold or policy change.
Ten existing cases: KA/QR/MD1000/B2/EV at40/22, EV/arrival forecast/future
entry at58/37, EV atzero orders and atzero feeds+orders. All50ms review,
M50/queueM, six30-share rungs/side; same five consumed dates/1,434 windows.
New canonical strategy IDs: KEEP_ALL, FILL_GUARD, SWEEP_GUARD, ORDER_EV,
STOP_EV, ARRIVAL_EV, FUTURE_ENTRY; clock and model versions separate.
Legacy QR250 with recalibrated thresholds is not the fixed QR50 comparator.

Gross per-fill spread capture = signed pre-fill-mid execution edge; midpoint
movement + spread = markout at100/500/1000/5000ms. Marks use modeled fill
clock (anchor minus50ms), while old anchor markouts also reconcile. Terminal
P&L is cash+inventory at60.2s; settlement P&L holds only that remaining
inventory to recorded binary resolution. Settlement carry is their difference.
Every prior window must reproduce its old P&L, shares, orders, pulls, fills,
maximum inventory, rung exposure, age/source partitions and anchor markouts.
Positive/known-bad accounting controls pass. No policy promotion or inference.

Pipeline in isolated /tmp/cta-scorecard-20261116, branch research/predict-scorecard-20261116.
Single Codex/core8 slot. Ten-case smoke parity and independent fill-ledger
reconciliation passed. Five complete dates now running serially; session11092.
Logs /tmp/pf-scorecard-YYYYMMDD.log; final report follows automatically.
Output data/predict_fun/research/strategy_scorecard_20261116_v1/; freeze
predictfun/reviews/strategy_scorecard_20261116/freeze_v1.json.
<!-- /STRATEGY_SCORECARD_20261116 -->

<!-- ENTRY_ARRIVAL_20261116 -->
## 2026-11-16 Entry arrival forecast complete: better score error, worse trading value

As of 2026-11-16T03:35:42.541848+00:00; nine consumed build dates / 387,535 neutral
placement generations. Test dates 20261027, 20261031, 20261101, 20261102,
20261103: 203,153 neutral placements, 203,114 scored targets; 191,442
supported, 11,672 activation rejections retained at zero, 39 missing-book
statuses. Economic replay: 1,434 first-minute windows, six Oct31 boundary
exclusions, all fills. This is chronological development evidence, not
untouched/full-day/live validation.

Same frozen v3 teachers as the positive zero-delay example, no teacher
optimization. Separate bid/ask LightGBM residual forecasts from 83 current
fields plus current score; exact current absolute price at activation t+58ms.
One recipe / ten side-fold fits / no threshold search. Train only on each
teacher fold's original earlier days, calibration date unused. Replay at
place58/cancel37, notice100, review50, M50 queueM, six rungs x30/side;
existing cancellation guard unchanged. No cancellation forecast in this test.

Pooled marked value: existing $+159.56, future-informed entry
$+488.18, causal forecast $+26.81; forecast change
$-132.74, better on 1/5 dates. Cents/share
+0.3312 / +0.7135 / +0.1011.
Forecast teacher MAE 3.8204 -> 2.9280 c/order; weighted
wrong-sign teacher value 38676.08 -> 14797.43 c.
These are proxy score errors, not realized profit or proof of useful entries.

Post-hoc matched-post-count current-score comparator: forecast ranking wins
5/5 dates, but selected teacher value is negative on 3/5; pooled
-1339.89 c. Always-wait teacher disagreement 13457.55 c is lower than the
forecast. Fixed first12 registry windows/date (60 total): forecast-selected
submissions average +0.2108 c predicted versus -0.2582 c arrival teacher;
75.42% have negative arrival teacher scores (not realized loss frequency).
Guard pulls 8,684/11,085 still-exposed forecast orders at first review versus
3,885/6,819 existing. Full replay posts 245,475 vs189,968 orders and fills
26,519 vs48,173 shares. Selected-state calibration and entry/guard disagreement
remain issues; their causal PnL attribution is not established. WS-timed and
age>=100ms attribution also underperform; E2 clocks alone do not explain it.

Checks pass: 1,434 baseline window reproductions, all five uncached trajectory
controls, lane/terminal accounting, 15 teacher identities, all saved prediction
rederivations, exact fixed-price/58ms target, 63,850 client-state comparisons,
positive and known-bad controls. Student-only 12-quote CPU inference p95
0.1313ms (500 calls), not measured end-to-end latency or added replay delay.
The arrival diagnostic is nondeployable, not the whole zero-delay policy or
an attainable upper bound; zero rejection scores can trigger futile requests.
M0/M250, queueL, B1a weighting and untouched validation are not tested here.
No strategy promotion. Next experiment: repair selected-state post/wait
calibration and value consistency with the continuing guard before teacher
optimization; this next experiment has not been run.

Source 1b94ec80; superseding freeze 3ce50516 (v1 memory stop before results,
same recipe; incomplete v1 retained). Results/audit 459a94a0, merged into
mm-research as 33ee3ea3. Review: handoffs/20261116_entry_arrival_review.md.
Receipts: reviews/entry_arrival_20261116/results_v1/report.json and audit_v1.json.
Data/models: data/predict_fun/research/entry_arrival_20261116_v2/.
<!-- /ENTRY_ARRIVAL_20261116 -->

<!-- EV_LAG_GAP_20261116 -->
## 2026-11-16 Recorded latency materially weakens full EV; anticipation remains untested

Complete as of 2026-11-16T01:57:35.556172+00:00: same 1,434 first-minute windows on
five consumed dates (10-27, 10-31, 11-01..03), six October 31 boundary
exclusions, all fills. Source 06fb7d77, freeze 37d61a9d, results f54795de,
integration 193e0440. Full v3 EV only; placement/cancel 40/22 versus the
previously recorded medians 58/37 ms. All other clocks, models, M50 tape,
50 ms review/fresh-entry grid and queue assumptions fixed. No new fits.

40/22: $+397.10, +0.9259c/share, 42,890 shares.
58/37: $+159.56, +0.3312c/share, 48,173 shares.
Value falls 59.82% despite
12.32% more shares. Four dates positive, but
best date 10-27 contributes $+200.40; other four
sum to $-40.85. All dates retained. Prior
same-policy full modeled zero delay +$820.92: gap to medians +$661.36.
These are simulated terminal marks, not whole-day/live/untouched profits.

Verified mechanism: an order activates at 40 ms and first gets the EV guard
at 50 ms (age 10); at 58 ms it first gets the guard at 100 ms (age 42);
zero placement gets first review at 50 ms (age 50). This establishes grid
interaction, not its PnL share. Pending EV review is absent. Timing/queue
assumptions and jitter remain unvalidated by this median-only diagnostic.

Correction to earlier priority: selected inventory losses did not establish
inventory control as the biggest lever. Baseline terminal minus +1s mark
is +$46.50; medians -$92.21; full zero -$148.48. Mark differences are not
executable exit gains. LGBM architecture is not shown to be the main gap.
The keep-value target still doubles as fresh-entry value and omits priced
queue continuation. Its contribution to losses has not been isolated.

Next focused test: bounded per-order act-now vs act-50/100ms-later value,
separate post and cancel, realistic clocks held fixed; explicitly noncausal
lookahead diagnostic then causal forecast. Zero latency proves opportunity,
not predictability. No new forecast or oracle was tested here.

Every baseline window reproduced; 6188 independent
state comparisons and 10 trajectory controls
pass, all source/input identities and accounting checked, planted errors
refused. Review: handoffs/20261116_ev_lag_gap_review.md. Evidence:
reviews/ev_lag_gap_20261116/results_v1/report.json and interpretation_v1.json.
<!-- /EV_LAG_GAP_20261116 -->

<!-- STOP_EV_LATENCY_20261115 -->
## 2026-11-15 Latency matters; full EV earns selectively, churn and inventory remain

COMPLETE as of 2026-11-15T18:58:55.527073+00:00: 1,434 first-minute windows on five
consumed dates, 28,680 rule-clock replays, all fills included. Six established
October 31 boundary windows excluded. Source c7bc4aca, freeze 30b92397,
results 26803af4, integration ebce46a3. Corrected v3 EV/B2/QR heads; saved v6
MD benchmark. Fresh gates and review every 50 ms, 6 x 30 shares per side,
no cap or retry timer. No new fits, threshold search or policy promotion.

Five-date EV trading value: existing replay delays +$397.10 / +0.9259c per
share / 42,890 shares; zero order/notice delay +$642.44 / +0.9252c / 69,439;
zero modeled feed + order delay +$820.92 / +0.9612c / 85,402. Positive dates
4/5, 5/5 and 3/5. Full zero helps 3/5 dates and hurts the last two; its gain
is chiefly more captured volume (1.991x), not much more value/share (1.038x).
The hypothesis that the model cannot earn even without latency is not
supported by this replay. It is not live or untouched validation.

Same-environment rules: existing-delay B2 -$219.70, QR -$818.39, MD1000
-$805.51; full zero B2 -$265.18, QR -$0.19, MD1000 -$65.16. EV/B2 control
supports further use of market inputs, but endogenous exposures differ;
no matched-retention causal skill claim. QR keeps prior fixed thresholds.
Trade-time M0 sensitivity takes full-zero EV to +$913.24, QR to +$478.74,
MD1000 to +$306.08: physical timestamp/queue review remains material.

EV +1s markout is positive on every date in all clocks. Full-zero Nov 2/3
+1s values +$132.45/+$225.88 become -$8.37/-$38.42 at the terminal mark;
later price moves erase short-horizon gains. Maximum net inventory reaches
1,579.71 shares without a cap. Queue retention is still poor: pulls/orders
92.98% -> 92.40%; mean rung exposure 2.82% -> 3.69%, touch rung 2.77% ->
4.18%. These are per-rung measures, not any-order uptime. At cutoff zero,
EV sign is conditional-value sign whenever P(fill)>0: tiny expected losses
can trigger pulls without pricing surrendered queue position.

292,978 independent sampled state comparisons, 60 cached/uncached trajectory
comparisons, all lane/terminal/input checks, five frozen source/fit checks
and 20-cell independent reconciliation pass; planted errors refused.
One core, 4,559.55 s, peak RSS 1,364.02 MiB. Interpretation v4 supersedes
v3's verification summary with computed artifact checks; numbers unchanged.
Next: separate entry and resting-order targets; price queue/re-entry value
and inventory exposure, while the other session verifies physical replay.
No fixed cooldown implied. Review: handoffs/20261115_stop_ev_latency_review.md.
Evidence: reviews/stop_ev_latency_20261115/report.json, attribution_v3.json,
interpretation_v4.json and receipt_*_v3.json.
<!-- /STOP_EV_LATENCY_20261115 -->


<!-- STOP_EV_DYNAMICS_SCALE_20261115 -->
## 2026-11-15 Dynamics exist; marginal scaling does not explain most severe misses

As of 17:01:13Z, nine consumed dates / 149,658 certified actions profiled;
five existing test folds / 81,630 actions, 7,620 fills, 1,014 severe outcomes.
83 current inputs + 22 predefined context/history descriptors; 40 severe
separation diagnostics, all reported. Source 73203849, freeze 79a11db1,
results integration eccbf302. No fits, new policy or protected data.

Current inputs include net OFI, depth/price changes, level ages and relative
activity. They omit much of gross flow and fixed-price replenishment paths.
Zero infinite current fields. None of 87 raw-kept severe cases has a finite
input beyond prior training min/max; only 6 cross a training 0.1/99.9% tail.
This does not rule out joint/regime shift, missingness or timing issues.
Largest absolute 1% of outcomes accounts for 65.45-82.02% of test EV squared
error; v2 L2 already weights tails heavily. Do not trim real losses.

Extra cached histories show modest/unstable severe separation after matching
order state and score bands. Depletion/(remaining+shown) has inverse AUC
0.428, 2,748 matched fills; not a cancellation rule. Existing relative queue
AUC 0.585, 4,083 matched fills. Strict subset has 421 fills / 23 severe;
conditional-history comparisons often cover only 5 fills. Fixed-price 1s
history available for 48,695/81,630 actions. Proposed net-flow/median-net-flow
normalization is ill-conditioned (up to 1.25e15), rejected; it is NOT a current
model input. Next: stable gross-flow/replenishment/short-volatility block,
scale-support indicators, paired ablation and severe-risk calibration.

Positive/refusal controls and source/prediction identities pass. One core,
69 s, 619 MiB. Review: handoffs/20261115_stop_ev_dynamics_scale_review.md;
full evidence: reviews/stop_ev_dynamics_20261115/report_v1.json. All results
descriptive on consumed data; no established model or portfolio improvement.
<!-- /STOP_EV_DYNAMICS_SCALE_20261115 -->

<!-- STOP_EV_ECONOMIC_LAYER_20261115 -->
## 2026-11-15 Per-order economic correction avoids some losses; no tested variant is profitable

As of 16:29:13Z, five consumed test dates / 81,630 primary neutral actions
(7,620 filled), saved LGBM heads unchanged. Four earlier-reserved-day
calibration layers plus raw, three fixed keep-margin sensitivities = 15
variants. Corrections use ALL actions, including no-fills; no tail trimming.
Source 1183afef, freeze f6b82170, integration 78d5b0cd.

Raw nonnegative set: 15,681 actions / 1,121 fills, forecast +$136.56 versus
-$117.57 local value. Good/bad fills 545/573; mean +34.63/-53.45 c/order.
At-price +$64.55, through-price -$182.12. Filled-value term is 91.48% of
retained-set overprediction (identity, not causal attribution).

Side x queue x edge correction at zero cutoff: 9,846 actions / 810 fills,
-$24.67; preserves 79.78% of raw good value, 42.79% less gross loss. At equal
retention counts it beats raw by $25.93, positive 5/5 dates. Highlighted after
inspection, not promoted. It removes 40/87 raw-kept calibration-defined tail
cases ($83.37) but misses 47 ($85.69). Order + pressure gives -$37.64 and
-$0.21 vs order-only at its own counts. This pressure representation adds
little; no general information-ceiling claim.

All 15 variants retain negative pooled local value. Primary pressure + 0.1c
margin retains 59,153 actions / -$811.32; fails declared screen (positive
kept value 0/5, vs matched order-only 3/5), so no new sequential replay.
Margins are sensitivities, not measured queue value. Strict subset: raw
zero-cutoff +$16.76 on 3,368; order correction +$10.67 on 2,251. No uptime,
churn, portfolio or untouched-validation improvement claimed. The existing
activity-only replay is separately reconciled; young-order losses are age
attribution, not a measured causal churn cost.

Both saved heads and calibration reload exact; identity/chronology/label
checks and planted controls pass, known-bad inputs refused. One core, 14s,
324 MiB. Prepared consumed data only. Review:
handoffs/20261115_stop_ev_economic_layer_review.md; evidence:
reviews/stop_ev_economic_layer_20261115/interpretation_v1.json. Remaining
question: improve severe-loss discrimination or measure queue continuation
value; arbitrary pressure penalties/keep margins have not repaired the edge.
<!-- /STOP_EV_ECONOMIC_LAYER_20261115 -->

<!-- FILL_INPUT_SEPARATION_20261115 -->
## 2026-11-15 Inputs distinguish harmful fills modestly; pressure and queue deserve the next check

As of 15:59:26Z, five consumed dates / 81,630 certified-estimated actions:
7,620 filled (2,561 good, 5,027 bad, 32 zero). No new fit or horizon change.
All 83 inputs independently rebuilt from stored event-derived rows and known
order state over 265,373 actions / nine dates: 22,025,959 fields exact,
including times/signs/scaling. Collector tape was not rebuilt; physical feed
and estimated trade timing remain unresolved by this reconstruction check.

Good/bad conditional AUC, matching day/side/rung/edge/fill-probability bands:
queue ahead 0.587 (6,428 covered fills), Binance depth OFI 0.594 and Predict
OFI 0.580 (6,519 each), fair deficit 0.504 (5,323). Adding hour preserves the
queue/OFI associations. Strict timing-invariant evidence is weaker for OFI
(AUC 0.525/0.530, 413 nonzero fills); queue AUC 0.596 (411). All 88 scores
reported; no significance or information-ceiling claim. Near-zero filtering
barely changes the main readings.

Four frozen all-action rules, with exact matched-count analytic expectations:
fair-price-below-order has -$83.38 incremental avoidance; adverse fair move
exceeding edge +$37.88 (2,095 flagged), thin adverse book +$67.00 (10,034).
The latter two have positive incremental avoidance on 5/5 dates, descriptive
only. Every rule leaves negative unflagged local value. No policy promoted.
Next input question: pressure x queue x quote cushion, with freshness and
persistence; the shorter-horizon proposal is not established as the best test.
The other session's committed replay reviews every 250 ms (50 ms grid), so a
future incremental-exposure target must use actual review cadence.

Source af087fdb, freeze 331a8279; integration 4266da46. Controls pass and
known-bad inputs refused. One core / 152 s / 794 MiB. Review:
handoffs/20261115_fill_input_separation_review.md; evidence:
reviews/fill_input_separation_20261115/readings_v1.json. Consumed data only;
no protected data, model fitting, active guard or other-session job changes.
<!-- /FILL_INPUT_SEPARATION_20261115 -->

<!-- STOP_EV_MODEL_CHOICE_20261115 -->
## 2026-11-15 Controlled model comparison favors retaining the hurdle benchmark

As of 15:31:29Z, five consumed test days / 81,630 certified-estimated actions
(7,620 filled): four setups, each raw + one reserved-day mean-offset
calibration; eight variants, no parameter search. Original LGBM hurdle versus
direct LGBM, at-price + through-price LGBM and Extra Trees (one fixed setup).
Same inputs/labels/folds; control predictions and training matrices exact.
Hurdle MSE beats all three alternatives on all five days. At identical group
counts (17,585 selected actions), local value is -$148.73 / -$233.01 /
-$262.55 / -$155.98 respectively; order-only -$328.35. Hurdle and Extra Trees
beat order-only at matched counts all five days, but no raw setup is profitable.

Fill AUC 0.809-0.841; conditional filled-value R2 0.0007-0.0318. Calibration
reduces hurdle mean optimism 1.033 -> 0.320 c/action, but its 4,791 positive
selections lose $41.37. Calibrated Extra Trees makes $5.97 on only 1,115
selected actions (1.37%); this is not measured quote uptime or a promoted
replacement. No controlled evidence here for switching model families.

Target review: agreed one-second neutral local EV is coherent, but does not
isolate urgency before the next 50 ms recheck/cancel (about t+72 ms under
22 ms cancel latency). A shorter incremental-exposure target with good-fill
value, longer risk context and queue-priority costs is a PROPOSED next test;
no horizon or active policy changed. The current zero cutoff is diagnostic,
not the agreed stage-3 default-rest policy with a negative pull margin.

Qualification to the newer E2IZE interpretation: re-timing observed-WS groups
leaves actual E2 groups unchanged. Small mean effect in the selected invariant
subset does not establish timestamp validity for WS-missing trades. It is a
useful negative control, not proof that all E2-associated losses are physical.

Source d32b0ca1, freeze 266f3ee2; merge 10ca931a. All heads save/reload
exactly, positive controls pass and corrupted inputs refused. One core, 176 s,
519 MiB peak; prepared consumed data only. Review:
handoffs/20261115_stop_ev_model_choice_review.md; evidence:
reviews/stop_ev_model_choice_20261115/{report_v1,interpretation_v1}.json.
No new model promoted, protected data read, live code or T5v2 changes.
<!-- /STOP_EV_MODEL_CHOICE_20261115 -->

<!-- STOP_EV_TAIL_DISTRIBUTION_20261115 -->
## 2026-11-15 Tail losses and the training/forward gap independently checked

Distribution as of 15:05:27Z: 137,703 original-model test actions over five
consumed days. Tail threshold fixed to each training first percentile:
1,779 test tail actions; highest-risk 10% contains 39.1% (Huber) / 39.4% (L2),
versus 12.2% / 12.5% expected after matching day/hour/side/rung. Descriptive
ranking only, not a deployable threshold or proof those losses are avoidable.
V2 comparator supersedes v1 by also matching hour; both receipts retained.

Test fill frequency falls every fold; filled size rises modestly but cost
per filled share worsens. Side/rung/price/sigma matching leaves worse
within-cell outcomes all five days. Matching filled size/source additionally
leaves 76-94% of the last four days' matched gap within groups. These are
coarse descriptive controls, not causal identification. Marginal out-of-range
features affect 0.062-4.391% of test actions; not wholesale unseen ranges,
but joint support and unmeasured regime changes remain unresolved.

Saved v2 heads independently replayed at 15:05:47Z (81,630 certified test
actions): all keys, labels, matrices and predictions exact. Training IC
0.204-0.224 is IN-SAMPLE; intermediate reserved-day IC 0.080-0.133 and test
0.081-0.129. The code saves the reserved day's predictions but fits no value
recalibrator. V2 positive set: 15,681 actions, -$117.57 local value, loses
on four of five days. Positive gain vs keep-all is not profitable kept value.

Keep tails in the EV target; conditional expected tail cost matters. Physical
E2 timing, representative conditional coverage and forward calibration are
still unresolved. This does not select a new model or run the other session's
pending T5v2. Controls pass, three prediction corruptions refused. Sources
23eb4c3e / freeze 60732c8e; integration 694ae73e.
Review: handoffs/20261115_stop_ev_tail_distribution_review.md
Evidence: reviews/stop_ev_distribution_20261115/ (report_v2,
generalization_v1, interpretation_v1). No raw/protected reads or new fits.
<!-- /STOP_EV_TAIL_DISTRIBUTION_20261115 -->

<!-- STOP_EV_PERFORMANCE_CAUSES_20261115 -->
## 2026-11-15 Controlled cause analysis confirms loss-objective bias and remaining adverse-fill risk

As of 14:45:39Z, five consumed test days / 137,703 actions: original fold
predictions reproduced exactly. Changing ONLY Huber -> squared error reduces
average overprediction 1.985 -> 0.692 c/action (65.2%, better all days). On the
original top decile (13,771 actions), optimism improves only 19.4%. Both heads'
own positive sets lose: Huber 9,727 actions -$55.41; squared error 17,558 actions
-$82.76. No replacement model selected. This isolates objective choice on v1
labels; it does NOT evaluate the newly committed certified-population v2.

For the original kept set, 93.4% of residual decomposition is filled-value error
(an accounting identity, not independent causal attribution). At-price +$85.12
vs through-price -$140.54. Worst 1% of all actions contribute 50.5% of gross
losses. Test filled-value means are worse than their training samples all five
folds; this is outcome shift, not proof of unseen input support.

Timing audit at 14:46:05Z verifies nine-day E2SEP scenario means -4.159 / -3.053 /
-2.147 c at -250 / 0 / +250 ms, n=268,756 / 265,373 / 266,011. Populations change;
these are not paired causal effects or physical timing truth. All 18,911 strict
nominal actions retain keys/times/values under both shifts. Four model-data and
three timing-audit corruptions refused; saved model heads reload identically.

Source c69a6431, freeze d36e022b, merge 9a642ab7. Review:
handoffs/20261115_stop_ev_performance_causes.md; evidence:
reviews/stop_ev_causes_20261115/ (report_v1, interpretation_v1, timing_v1).
No active pipeline changes or protected raw reads. Other session's b80765a1
landed during this work and addresses target/certificate/validation issues in
code; this study does not establish that v2 fixes the remaining losses.
<!-- /STOP_EV_PERFORMANCE_CAUSES_20261115 -->

<!-- STOP_EV_LATEST_COMMIT_REVIEW_20261115 -->
## 2026-11-15 Model and E2SEP review: validity/target issues remain

Reviewed stage-2 commits through f3fb7e3b and new E2SEP 0b7e30cd. Saved-file
check as of 14:35:33Z reproduces published metrics (source 14:00:57Z; five
consumed test days, 137,703 actions); 3 planted corruptions refused. No model,
policy, label or running E2SEP source changed by this review.

High-priority findings: (1) Huber estimates a robust location, not the mean
required by EV; top decile n=13,771 forecasts +0.4345 c but realizes -0.5266 c.
(2) 56,073 / 137,703 model test actions lack the new estimated-timing
certificate; certified overlap values match exactly. Even 993 / 4,572 reported
strict-WS actions lack the missing-timing certificate (not measured errors).
(3) Oct 31 / Nov 3 fail the original T5 gate; the report reinterprets them under
a post hoc one-sided gate, with only 10 draws rather than the required 200.
(4) E2SEP's future WS-REACHED group does not remove prior E2 queue dependence:
a synthetic earlier E2 shift changes its local value +5 c -> 0 at 14:36:46Z.

Additional issues: R950 compares changing populations; E2SEP source manifest
omits its own builder; exact joins omit decision time/uniqueness; model fitting
saves predictions but not fitted heads or input hashes. Keep MODEL-READY as
exploratory ranking output, not proof that an EV-driven policy is ready.
Read handoffs/20261115_stop_ev_latest_commit_review.md before integration.
Evidence: reviews/stop_ev_model_commit_review_20261115/receipt_v2.json and
e2sep_v1.json; review merge af9478e7. Existing experiments remain untouched.
<!-- /STOP_EV_LATEST_COMMIT_REVIEW_20261115 -->

<!-- STOP_EV_TIMING_COVERAGE_20261115 -->
## 2026-11-15 Timing coverage improved with a side-prefix certificate; historical audit passes

As of 14:02:20Z, nine consumed days / 393,154 neutral probes per variant:
strict eligible 2,809 -> 18,911; conditional E2 point-anchor eligible
72,255 -> 149,658. Development Oct 23–26 / 187,384 probes: strict
969 -> 7,337; estimated 30,076 -> 68,028. These are label coverage counts,
not quote uptime, physical timing validation or model-performance results.
All 75,064 baseline label/feature comparisons match; 9 synthetic checks pass;
5 saved-artifact corruptions are refused. Sources/raw inputs and artifact
hashes reconcile. Full exclusions and bid/ask/day counts are in audit_v2.json.

The filter proves missing/E2 events cannot affect any accepted focal-side order
before the label horizon, including earlier queue drain, partial fills and
order births. Opposite-side independence holds only for the uncapped neutral
reference with fixed external books. Missing timestamps stay unbounded;
strict E2 +/-250 ms bounds remain assumptions. Prior uncertainty and selected
coverage remain unresolved. No actual exchange fill claim.

CAUTION for the active model session: future-only/focal-price filtering can
miss earlier state changes (three synthetic counterexamples). Its 265,373
admissions include 115,715 lacking this certificate; this is NOT a measured
mislabel rate. Keep that user-approved population as a qualified sensitivity.
No active model sources, inputs or fitting jobs were changed here.

New labels: data/predict_fun/research/stop_ev_timing_certified_20261115_v1/.
Do not use certificate/status columns as model inputs. A subsequent experiment
needs an exact decision-time market join and explicit status/timing adapter;
these exports omit k/hour. See the handoff for integration/reproduction.
Review: handoffs/20261115_stop_ev_timing_coverage.md.
Evidence: reviews/stop_ev_timing_20261115/current.json (audit_v2 extends v1).
Builder f0682872; freeze 3bb6773f; checker 1e6c71d2; result merge 79e8f13d.
<!-- /STOP_EV_TIMING_COVERAGE_20261115 -->

<!-- STOP_EV_ACCOUNTING_STATUS_20261115 -->
## 2026-11-15 Historical STOP-EV primary accounting audited: no new defect found

As of 13:37:11Z, all 393,154 saved neutral probes were audited in both timing
variants across nine consumed dates. The 72,255 eligible estimated-timing
labels match E9 counts; day receipts have zero E9 mismatches/ledger failures.
Saved quantity, residual, component, normalization, null-status and unique-key
checks pass. Source identities match. Five planted corruptions are refused.

Strict development coverage remains only 969 / 187,384 probes; estimated timing
has 30,076. Timing/selection and actual fill realism remain open. E7's older
single-order diagnostics (including their mark freshness convention) must not
replace primary order labels. Raw maker identities behind equal-content trade
flags were not independently re-read in this audit. No model/performance claim.
Review: handoffs/20261115_stop_ev_accounting_status.md; artifact:
reviews/stop_ev_accounting_status_20261115/receipt_v1.json. Checker c677eef8.
<!-- /STOP_EV_ACCOUNTING_STATUS_20261115 -->

<!-- STOP_EV_ACCOUNTING_FIX_20261115 -->
## 2026-11-15 STOP-EV accounting fixed and integrated; synthetic checks pass

User authorized resolving the accounting issues. `pf_stop_ev_accounting.py`
uses the corrected joint neutral replay's per-order tranches, shared leg
quantity and partial remainder. It excludes common cancellation-latency fills,
marks each later tranche using valid raw quotes, separates client-known inputs
from outcomes, retains invalid timing/book/mark data as statuses, and evaluates
unique actions with matched nulls. The one-second neutral continuation is
explicit; policy decisions can still repeat every 50 ms without a retry timer.

Source c04370d6, code integration f373310b, receipts integrated bf57feff.
All 16 accounting checks passed as of 2026-11-15T12:54:55Z. Builder integration
passed at 12:55:32Z on 06f80178, catching all nine planted errors. The other
session's builder now uses the adapter for primary neutral-generation labels,
independent E9 reconciliation, explicit statuses and source-hash guards.
See handoffs/20261115_stop_ev_accounting_fix.md and
reviews/stop_ev_accounting_20261115/receipt_v2.json (v1 retained).

Next: validate historical coverage/exclusion counts and the known-time market
feature join, then fit on regenerated labels. These checks establish accounting
correctness on synthetic cases; no historical or economic improvement claim.
The pre-existing entry experiment was resumed and its runner records all stages
complete at 12:54:58Z; its performance report has not been reviewed in this fix.
<!-- /STOP_EV_ACCOUNTING_FIX_20261115 -->

<!-- STOP_EV_PREBUILD_REVIEW_20261115 -->
## 2026-11-15 Revised Stop EV design has label issues before fitting

Review of cae5c88f sections 7-8: direct local EV and data-first staging are
improvements, but window-average markouts can erase front/back queue value;
any through leg still assigns a full residual fill; one-second exposure versus
50-ms continuation is undefined; partial remaining size is unsupported; and
the decision gate still uses unlike percentages/overlapping states. Top-maker
signs must be descriptive comparisons, not data sanity pass conditions.

No STOP-EV builder/dataset/model was visible at this review snapshot; actual
implementation quality is unverified. Pooled side-aligned training with a side
indicator is a permissible design choice, subject to per-side EV checks.
New work must explicitly import the corrected queue lane, preserve causal q/Z,
propagate missing timing to action validity and rebuild affected references.
See handoffs/20261115_stop_ev_prebuild_review.md and
reviews/stop_ev_review_20261115/prebuild_v2.json. Review only; no running source
or the owning session's design was changed.
<!-- /STOP_EV_PREBUILD_REVIEW_20261115 -->

<!-- QUEUE_INITIALIZATION_FIX_20261115 -->
## 2026-11-15 Activation queue initialization fixed for new runs

New entry/STOP engines use the last snapshot at or before activation; later
additions no longer inflate initial queue ahead. Existing queue, partial fills,
repost acknowledgements and unknown-depth scenarios are preserved. The new
pf_queue_activation_replay.py commands explicitly route to corrected engines
and use separate output namespaces. Historical commands and running frozen
experiments retain the old behavior; no completed result is relabelled.

Source/integration 07f287be / a815aa8a. All 12 synthetic tests pass as of
2026-11-15T12:08:34Z, both sides and both engines, including known-bad legacy
controls and command routing. No historical performance rerun or refit yet.
Retraining needs regenerated neutral references and labels. See
handoffs/20261115_queue_initialization_fix.md and
reviews/queue_initialization_20261115/current.json for commands and evidence.
<!-- /QUEUE_INITIALIZATION_FIX_20261115 -->

<!-- STOP_EV_REVIEW_20261115 -->
## 2026-11-15 Stop EV design reviewed before implementation

Read-only review of the other session's design and latest completed STOP-PLAY
v6 receipts. Economic objective is better aligned with profitable queue
retention, but implementation needs quantity conservation, joint fill-value
labels, a distinction between book movement and execution, causal q/remaining-Z
inputs, cancellation-prefix accounting and action-level net-value evaluation.
Recommend one direct signed local keep-value head per side first. Entry remains
separate and evaluates the back of the queue; current resting priority can make
KEEP attractive when a new POST is not. No model or running source changed.

Review: handoffs/20261115_stop_ev_review.md; artifact identities and day-receipt
reconciliation: reviews/stop_ev_review_20261115/evidence.json. The review records
the completed result as of 11:24:12Z and qualifies its consumed-data, no-cap,
queue/timing assumptions. Original draft remains unchanged for its owning session.
<!-- /STOP_EV_REVIEW_20261115 -->

<!-- ENTRY_LOCAL_PREDICTION_20261115 -->
## 2026-11-15 Local entry prediction complete; strategy replay running

User requested a simpler model that avoids recursive downstream targets.
Compact single-order local value is fitted: 36 causal inputs, separate sides,
DIRECT one head versus FACTOR fill fraction plus fill-weighted movement. Count
focal fills over 5 s (bounded by session end), mark each tranche at fill+1 s;
these are measurements, not retry/cancel timers. Entry re-evaluates every 50 ms.
Separate frozen v6 keeper in replay; primitive labels still use neutral KEEP.

Evidence as of 2026-11-15T10:53:36Z: 27,669 eligible actions over seven consumed
label dates / 2,010 windows. Statuses: eligible 27,669, unsampled 65,889,
unreached 2,281, sealed 251 (six excluded windows). Chronological evaluation:
19,306 actions (bid 9,671 / ask 9,635), five consumed dates Oct 27/31, Nov 1/2/3.
DIRECT pooled error reduction versus edge/price/rung baseline is 2.0148% bid /
1.5084% ask; R2 versus prior training mean 0.00535 / 0.01109; both beat baseline
on 5/5 dates. FACTOR error reductions 0.9847% / 0.9242%. Gains are modest;
the changed target and denser evaluation prevent direct attribution versus the
old 9,871-action terminal study. No economic or forward-validation claim yet.

Model source/freeze 0d1b70e9 / cc929f01; runtime source/freeze c32f4197 / 9433f91b.
Checkpoint fc6208d1; integration 85de7ba4. All saved model/prediction/row hashes
verified. Faster inference passed 57,918 prediction comparisons and 24 complete
replay equivalence cases as of 11:02:51Z. Old slow replay explicitly superseded;
labels/models unchanged, prior artifacts retained. Full 30-share strategy replay
and 500-draw matched local tests finish automatically in core8.

Canonical reviews/entry_local_20261115/current.json; details
handoffs/20261115_entry_local_results.md. Authoritative status:
data/predict_fun/research/entry_local_20261115_v4/runner.json; final output
report/result.json under that directory. Service pf-entry-local-v4-20261115.service.
This is a checkpoint: inspect the runner before treating it as current status.
<!-- /ENTRY_LOCAL_PREDICTION_20261115 -->

<!-- ENTRY_REPAIR_COMPLETE_20261115 -->
## 2026-11-15 08:34:59 UTC Entry coverage and history study complete

All stages complete. Eligible labels 13,911 -> 27,669 over seven consumed days;
all 13,911 original features/targets/probe times match. Statuses: eligible
27,669, unsampled 65,889, unreached 2,281, sealed 251 (six windows). Fixed test:
9,871 actions, 4,990 bid /4,881 ask on five consumed chronological dates.

Dense all-history timing MSE improves 2.27% bid /2.38% ask versus sparse;
R2 remains -0.0557 /-0.0134 versus training means. Recent-two timing MSE improves
another 0.50% /1.14% pooled, but only two of four nontrivial days per side;
absolute POST favors all history. Zero of six 500-draw matched economic checks
pass; all selected absolute POST means negative. No promotion or forward claim.

Interpretation: sampling repair helps modestly; simply lengthening training is
not supported as the fix. Neutral KEEP continuation and terminal portfolio
spillovers remain target-design gaps relative to the intended selective keeper.
Capacity/regularization unchanged; causes not uniquely identified. The larger
size results were neutral baseline economics, not a combined learned policy.

Evidence 5a26b127; handoffs/20261115_entry_repair_results.md; canonical
reviews/entry_repair_20261115/current.json now points to completion.json and
full_result.json. All 60 model identities and 49 frozen sources verified;
all stages rc0, max full-run worker RSS 859 MiB. Older owned action-depth driver
resumed. Next experiment should freeze keeper/target alignment and model changes
separately, preserving this dense-data control; no new model launched here.
<!-- /ENTRY_REPAIR_COMPLETE_20261115 -->

<!-- ENTRY_REPAIR_SIZES_20261115 -->
## 2026-11-15 08:19 UTC Entry size results and denser coverage

Neutral POST/KEEP sizing completed: seven consumed dates, 2,010 included
windows, six sealed-lookback windows excluded, same 1,500-share cap. Original
queue assumption at 30/50/100 shares: mean resting time 15.33/15.89/17.02 s;
reposts 71,719/67,128/57,525; gross cents per filled share -1.850/-1.878/-1.917.
Both larger sizes worsen gross total P&L on all seven days under both queue
assumptions. Conservative queue treatment preserves the direction. Hard-cap
blocked decisions increase sharply with size; this includes cap/inventory
effects, not a pure partial-fill queue experiment. These are neutral replays,
not learned WAIT/POST or selective-keeper results; no fees/rewards/own impact.

Six completed dense days: eligible actions 11,918 -> 23,486; later generations
1,630 -> 13,198; all 11,918 old eligible labels/features/timestamps match exactly.
Statuses: eligible 23,486; unsampled 55,819; unreached 1,967; sealed 251. Last day
November 3 building after main driver resumed 08:19:03 UTC; fixed-evaluation
model fits follow. No new prediction-performance claim yet.

Evidence 383cd85b; handoffs/20261115_entry_size_early_results.md;
reviews/entry_repair_20261115/sizes_current.json. Source/inputs frozen separately
for this already-declared summary; full experiment unchanged.
<!-- /ENTRY_REPAIR_SIZES_20261115 -->

<!-- ENTRY_REPAIR_VALIDITY_20261115 -->
## 2026-11-15 07:10 UTC Entry validity complete and coverage build running

Seven consumed days complete: 2,010 included market windows, six sealed-lookback
windows excluded. All 3,982 predeclared action checks reproduce saved targets,
probe times and features; zero missing grid rows. All 30/50/100-share neutral
replays and both queue scenarios finished; size economics are not yet aggregated.
Maximum clock-shift difference from admissible parts is 0.004638671875 ms; the
retrospective clock and fill assumptions remain qualified.

Denser label construction is running (first day 224/288 windows at this snapshot).
No new fit completed yet. Current process remains healthy; the later training
history comparison is queued. Checkpoint evidence 3cd077a6:
reviews/entry_repair_20261115/validity_checkpoint.json. The running status is
still data/predict_fun/research/entry_repair_20261115_v2/runner.json.
<!-- /ENTRY_REPAIR_VALIDITY_20261115 -->

<!-- ENTRY_REPAIR_20261115 -->
## 2026-11-15 Entry repair and larger size study running

User accepted loader/label validity → representative coverage → controlled
training history → model changes, and requested larger sizes. Implemented and
frozen at ce6e645e / edca5c96: strict pre-read loader filtering; old-target and
feature reconciliation; denser later sampling preserving old ages; fixed-test
sparse/dense/history comparisons; 30/50/100-share neutral paths at the same
1,500-share cap and two unknown-queue assumptions. Eight synthetic controls and
the two-window smoke pass (28 eligible actions, all 10 old eligible actions
exact). Full study running in pf-entry-repair-20261115.service, core 8.

Current: reviews/entry_repair_20261115/current.json and
handoffs/20261115_entry_repair_progress.md. Runner:
data/predict_fun/research/entry_repair_20261115_v2/runner.json.
No performance/promotion claim. Clock reconstruction, simulator queue truth,
selective-keeper labels and size-specific learned entry models remain open.
Older owned supervisor 2900451 held; durable runner resumes it on exit.
<!-- /ENTRY_REPAIR_20261115 -->

<!-- ENTRY_TRAINING_ISSUES_20261115 -->
## 2026-11-15 — complete known entry-training issue register and coverage repair specification

User asks data versus training-window cause, repairability, and all current
training issues. Review only: no new labels, models or replay. Snapshot remains
2026-11-15T05:04:52Z, 9,871 evaluation actions on five consumed dates.
Coverage is mainly a dataset/sampling and admissible-date issue, not a rolling
window cap: initial generations sampled 1/2, later 1/32; only 2–6 comparable
training days. Proposed density comparison uses later 1/4, unchanged original
probe ages and evaluation actions, per-row probabilities and correct 2/4 train
versus 2/32 original evaluation weights. Specification only, not implemented.

15 issues span coverage/state scope; loader/upstream lineage; neutral-KEEP
continuation and terminal noise; unknown-depth queue and execution assumptions;
missing observations/proxy signals; model support/generalization; controlled
learning curves and full economic/forward validation. Two code paths verified:
entry builders do not enable SKIP_FOREIGN_PARTS before reads; place() can turn
unknown beyond-five-level queue into zero. The latter's effect on labels/P&L is
unmeasured. Neither establishes that future values entered eligible features.
Missingness-aware inputs do not fix execution-label assumptions. Downstream
future checks do not certify every upstream precomputed field.

Read-boundary disclosure: directory inventory exposed forward-family date names;
no forward file content/outcome was opened in this review. No fresh-validation
eligibility claim. Details and exact source bindings in the receipt.
Review: `handoffs/20261115_entry_training_issues_review.md`.
Receipt: `reviews/entry_training_issues_20261115/current.json`; evidence 7a4aa584.
Next: fix/qualify loader and label assumptions, freeze keeper scope, build denser
representative labels, isolate training-length effects, then model/portfolio tests.
<!-- /ENTRY_TRAINING_ISSUES_20261115 -->

<!-- ENTRY_TRAINING_SUPPORT_20261115 -->
## 2026-11-15 — training-window interpretation checked; cause not established

User asks whether representation is cleared and short training explains failure.
Known coordinate/missingness changes are implemented in CONDITION, not promoted;
missing observations and adequacy of conditional prediction remain unresolved.
Read-only review of the 2026-11-15T05:04:52Z snapshot, 9,871 evaluation actions on
five consumed dates. No new candidate, fit or learning-curve experiment.

Actual expanding training sets contain 2–6 labeled days; per-side training rows
1,975–6,068; weight-concentration effective rows 522.6–1,587.1; later-entry
training rows 284–847. Effective counts discount unequal weights only, not time
or market dependence. All available comparable earlier labels are used; there
is no short rolling cap discarding a larger same-target dataset. Different
folds evaluate different days, so their non-monotone scores cannot identify a
training-size effect. Thirty-second feature coverage is a separate limitation.

Review: `handoffs/20261115_entry_training_support_review.md`.
Receipt: `reviews/entry_training_support_20261115/current.json`; evidence b46f6c7c.
Next discriminating test: fixed evaluation actions, nested earlier training
histories, unchanged representation/targets/learner; denser later-entry sampling
as a separate comparison. No claim that short history is the sole cause.
<!-- /ENTRY_TRAINING_SUPPORT_20261115 -->

<!-- ENTRY_REPRESENTATION_20261115 -->
## 2026-11-15 — condition-preserving entry representation COMPLETE

User: check data and make economically different conditions separable. Completed
2026-11-15T05:04:52Z: 9,871 evaluation probes (4,990 bid / 4,881 ask), 1,434
windows on five consumed dates Oct 27/31 and Nov 1/2/3; all seven label days
13,911 probes / 2,010 windows. Two new candidates, one exact saved RAW control.
RELATIVE replaces 47 public size coordinates while preserving absolute liquidity
and own risk; CONDITION adds 26 history-support, volatility and fair-response
features (351 total). All original rows, labels, weights and model params stay
unchanged. Actual history support reconciles to the saved liquidity scale.

CONDITION timing squared error improves 1.43% bid (5/5 days), worsens 0.95% ask
(0/5 better). RELATIVE alone improves 0.36% bid, worsens 0.73% ask. All arms
remain below prior-mean and price/rung baselines; 0/6 matched economic checks
pass (500 draws, six-way adjustment). Absolute-POST improvement differs from
timing improvement; no policy promotion or forward-validation claim.

Actual coverage: 7,857/9,871 partial liquidity histories, 4,758 unknown start
references, 2,031 unavailable complete 250ms candidate histories. Later entries
are 1,323 rows (13.40%) but 71.23% of inclusion-weight mass. Representation
exposes these support limits without manufacturing missing observations.
Seven synthetic checks, 2,010 future checks, 60 saved model artifacts verified.
Peak worker RSS 859 MiB; completed service exits 0. Initial synthetic assertion
failure preserved; corrected before any historical build/fit. Older size
verification completed and its resumed supervisor exited.

Source a6ee5803, authoritative freeze 2523293e / freeze_v2.json, evidence
5b287084 (summary line endings af777b65). Review:
`handoffs/20261115_entry_representation_review.md`; current receipt:
`reviews/entry_representation_20261115/current.json`.
Next: component-level investigation using explicit condition/support states;
side-specific evaluation and representative later-entry data remain necessary.
Component study not started; WAIT/POST and cancel/keep remain separate.
<!-- /ENTRY_REPRESENTATION_20261115 -->

<!-- ENTRY_NORMALIZATION_20261115 -->
## 2026-11-15 — entry normalization audit COMPLETE; component study not started

User requested normalization review before the next component check. As of
2026-11-15T04:33:04Z: 13,911 eligible label probes; 9,871 evaluation probes on
five consumed dates, Oct 27/31 and Nov 1/2/3. No new candidates or model refits.
All audited units/signs/denominators/target reconciliations pass; all 10 saved
cleanup preprocessors reproduce exactly and test transforms leave state intact.

Liquidity-relative public sizes reduce marginal train/test KS in 178/180
feature/side/fold comparisons, median 0.1322 to 0.0484. This is distribution
alignment, not prediction improvement. Earlier cleanup uses unweighted scaling
statistics despite weighted model fitting. At least 7,815/9,871 actions (79.17%)
cannot have a full nominal 30-second liquidity history. Cleanup loses an extreme
indicator on 15 evaluation actions; 14 duplicate divergences overlap those 15;
zero missing-indicator losses observed. Cleanup defects do not directly affect
the separate raw 325-input information study. Earlier cleanup changed multiple
things together, so its failure does not isolate normalization's effect.

Five synthetic controls pass; supervised audit exits 0; size supervisor resumed.
Source 3db21aec, freeze 0bb344c6, evidence edc910b4. Full evidence and limitations:
`handoffs/20261115_entry_normalization_review.md` and
`reviews/entry_normalization_20261115/current.json`.
Recommended next: controlled normalization-only comparison with unchanged rows,
targets, folds, weights and pinned model parameters. No fixes or promotion yet;
WAIT/POST entry remains separate from cancel/keep. Consumed development only.
<!-- /ENTRY_NORMALIZATION_20261115 -->

<!-- ENTRY_INFORMATION_20261115 -->
## 2026-11-15 — entry-information inventory and contribution measurements COMPLETE

User: list available information and measure influence on the wait/post target.
Entry stays wait/post; cancel/keep remains a separate model. Completed as of
2026-11-15T03:52:15Z: 9,871 evaluation probes (4,990 bid / 4,881 ask), 1,434
windows on five consumed dates, Oct 27/31 and Nov 1/2/3. Seven label days contain
13,911 probes / 2,010 windows. No forward validation or policy promotion.

325 named inputs (284 existing + 41 transforms); complete field inventory and
1,300 feature/side/target screens retained. Nine fixed family probes compare
incremental histories/geometry against current book, contract and order state.
For POST-minus-WAIT, Predict history/pending flow reduces bid squared error by
1.25%; level dynamics 0.73%; fair-value dynamics 0.37%. No addition improves
aggregate ask timing error. Recovered upstream fields worsen timing on both
sides, as does ALL. Every pooled model R2 remains below the prior-mean baseline;
0/18 matched local economic checks pass (500 draws each, multiplicity adjusted).
Weak marginal relationships exist; they do not establish causal or tradable
influence. The four missing inputs alone do not explain the earlier failures.

Six synthetic checks and 2,010 historical future checks pass; 180 saved model
schemas/digests verified; peak worker RSS 902 MiB. Source c6621973, authoritative
freeze 1acaf302 / freeze_v2.json, evidence d97f0b29; merged dd702519 into mm-research.
Completed worker exited 0; older size supervisor resumed. Prior launch attempts
preserved, no historical builds/fits in those attempts. Labels unchanged under
neutral KEEP: separate cancellation benefits are not credited in this study.

Results: `handoffs/20261115_entry_information_results.md`.
Inventory: `handoffs/20261115_entry_information_inventory.md`.
Scores/receipt: `reviews/entry_information_20261115/current.json`.
Next research should use the measured side-specific contributions, not assume
that a larger feature set solves timing. All dates are consumed development.
<!-- /ENTRY_INFORMATION_20261115 -->

<!-- WAIT_POST_STATE_RESULT_20261115 -->
## 2026-11-15 — ladder state and cleanup fail economic gates (QUALIFIED development result)

User: add missing inputs, review the model and improve it; diagnose poor first
prediction and noise filtering. Completed asof2026-11-15T02:59:00Z. Five consumed
test dates Oct27/31,Nov1/2/3:9871 probes (4990bid/4881ask),1434 windows.
Source99173056/freeze42f61bf1; noise20ba3773/068a4382;
cleanup70217846/c9fc924e (authoritative clean_freeze_v2.json metadata correction);
evidenceeeab2486. Merged into mm-research. Results:
`handoffs/20261115_wait_post_state_results.md` and
`reviews/wait_post_state_20261115/current.json`.

Implemented client-known per-rung prices/remaining/fills/ages/pending status,
empty generations and causal public queue estimates in labels and inference.
Added first-order/remaining-portfolio target components with exact reconciliation.
Repaired unavailable start-price gap in inputs and frozen continuation;994of13911
rebuilt eligible labels changed. Six sealed-lookback windows explicitly excluded.
Nineteen synthetic checks pass;2010 historical feature falsifiers;120 fitted
artifacts verified. Three supervised jobs exited0; old size queue resumed.

Five candidates (BASE,LADDER,COMPONENTS,CLEAN_TREE,CLEAN_RIDGE). All pooled POST
and timing R2 remain below the prior-mean baseline.0/10 full local economic
side/candidate tests pass,500 matched draws each; all selected absolute POST
means negative. Rich inputs slightly improve absolute-value error but not timing.
Train-only cleanup and fixed ridge do not improve aggregate error. Cleanup's160
raw fields become318-332 columns with indicators, not fewer encoded columns.

First-fold diagnosis (4040 train/1992 test probes) supports sparse/tail-heavy
labels and overfitting. Failures persist on ordinary-outcome/fresher subsets and
after oracle intercept removal. Filtering alone is not established as the fix;
real adverse outcomes must remain. Main remaining gap: neutral KEEP/terminal
portfolio targets versus intended selective keeper, plus limited later-generation
and joint-waiting support. Relabel under the actual frozen keeper; learn near-term
order influence separately; expand representative states. No model promoted,
portfolio replay/acceptance or forward validation. Old frozen results preserved.
<!-- /WAIT_POST_STATE_RESULT_20261115 -->

<!-- WAIT_POST_STATE_NOISE_20261115 -->
## 2026-11-15 — first-fold state-model noise diagnosis (post hoc, QUALIFIED)

As of02:35:02Z,4040 training probes (Oct25/26),1992 evaluation probes (Oct27).
New ladder inputs improve training fit without reliably improving the first
later-day fit. Labels are sparse/tail-heavy; the remaining portfolio/replacement
contribution has more variance than the first order's contribution. Failures
persist on ordinary-outcome and fresher-data diagnostic subsets and after
oracle intercept correction. This does not prove an irreducible noise floor.
Do not remove real adverse outcomes as noise. Train-only compact preprocessing,
representative later-generation/joint-waiting data, and relabeling under the
intended selective keeper remain necessary research directions. Full comparison and cleanup complete;0/10 economic gates pass across five
candidates. The diagnosis itself added no candidate. No promotion. Source20ba3773,
freeze068a4382, evidence004c7ddc, now merged into mm-research;
reviews/wait_post_state_20261115/noise/{summary,result}.json. Three diagnostic
controls pass, positive cases and known-bad refusals; exact saved inference parity.
<!-- /WAIT_POST_STATE_NOISE_20261115 -->

# P-2026-005 findings register

<!-- WAIT_POST_SIGNAL_DATA_20261114 -->
## 2026-11-14 — distribution, processing and sampling correction COMPLETE

USER: filter noise, generate useful signals, check train/test distribution and
process the data. Complete asof **2026-11-14T16:57:36Z**, five consumed development dates
(Oct27,Oct31,Nov1/2/3), **9,922 held-out action probes /1,440 market windows**.
Evidence: `reviews/wait_post_signal_data_20261114/summary.json` and
`completion.json`; phase1 audit/probes and sampling result retain every cell.
Source04d94d1b/freeze7d177ca0; sampling follow-up3145fc9f/freezea357cbc4;
completed evidence560686a4. Five new candidates total, no forward validation.

The evidence does NOT support "all test data was unseen". The declared market
nearest-distance check flags **605/9922 (6.1%)** outside
its training-calibration95% threshold. Inside that market-support proxy, entry
selection still loses. In coarse order-state bins with >=5 training examples,
**5377 probes /1006 selected** have realized selected POST value
**-2.389c per nominal share**; POST and timing predictions
still lose to prior means. These support checks are descriptive and do not rule
out changing conditional relationships or hidden market regimes.

Of49487 actual VALUE posts, 33086 use unseen joint bins,
but only **3** exceed the min/max of the FIVE audited state
variables (known position, own/other exposure, idle age, price change while
empty). This is new combinations of familiar values, not proof all market
inputs are familiar. Runtime market-feature vectors were not exhaustively
stored; biased saved inference samples were not substituted.

The training builder samples first generations with probability1/2 and later
ones1/32: **16x different inclusion probability**. Initial generations are
84.3..86.7% of training probes; runtime generation share is
33.6% (policy-dependent, a different population). Implemented inverse-inclusion
weights2/32 in both fitting AND evaluation. Weighted initial training share
becomes25.1..29.0%; training effective n is only
522.6..1595.8 per side/fold. Weighting corrects known sampling,
not absent joint states or unreached probes, and does not force runtime shares.

Implemented/tested train-only volatility/spread/lot normalization, missing and
extreme flags, constant/duplicate removal, direct timing-difference learning,
and a fixed regularized linear baseline. Labels and rare losses remain raw;
no test-row removal, test-fitted transforms or outcome-selected training.
Three initial variants plus two sampling-corrected variants were declared and
all reported. The regularized variant reduces prediction error but remains
below the prior-mean baseline overall. No full economic test passes:
**0/8 unweighted side/arm tests** and **0/12 weighted side/arm tests**,
including controls. Every null has500 draws; weighted controls additionally
match initial/later generation and selected weight mass. These are two views
of overlapping candidates, not20 independent experiments.

Fourteen synthetic checks passed (positive controls + refusals),57 frozen
source/artifact files verified,70 fitted LGBM files bound in
`model_artifacts.json` (linear parameters in phase1 receipt). Both supervised
jobs exited0; old size queue resumed. Source exclusions:8361
untimed legs /105159.20 shares; no missing feature-grid rows.

Next data change: label more later-generation and jointly waiting ladder states
on the neutral action reference, with client-known prices/remaining quantities
of the other resting orders. Reweighting scarce examples cannot create that
information. Evaluate signals incrementally to simple price/rung baselines and
use the actual intended frozen keeper/continuation for combined entry labels.
No new portfolio/forward claim; useful economic prediction is still unproven.
<!-- /WAIT_POST_SIGNAL_DATA_20261114 -->

<!-- WAIT_POST_WHY_20261114 -->
## 2026-11-14 — wait/post failure diagnosis COMPLETE

USER: "It looks the model performs really bad, check why." Complete post hoc
explanation as of **2026-11-14T16:21:11Z**, source33925207/freeze3ac6a3b4.
Primary evidence: `reviews/wait_post_diagnosis_20261114/v3/diagnosis.json`;
`raw_report.json` and `completion.json` bind the calculations and artifacts.
Five complete consumed dates, **1440 market windows** and
**9922 held-out neutral-reference probes**;
first60s/window, common60.2s terminal, M50, six rungs/side,30shares/order,
cap1500. Gross simulation, no new candidate, selection or forward validation.

Confirmed generalization failure: POST, WAIT and their difference each lose to
both prior-training-mean and fixed price/rung baselines in **10/10 side-day
cells**. POST trainR2 0.227..0.289,
testR2 -0.113..-0.023; timing rankIC
-0.034..0.055.
Selected local POST predictions vs actual: bid+3.139c vs-0.836c, ask+3.794c
vs-3.656c per nominal share. Original500-draw matched tests fail on both sides.
This failure exists before the closed-loop state shift, not merely calibration.
The timing target is68..73% zero; its largest1% has48..59% of squared-target
magnitude (not residual loss). POST/WAIT target correlation0.861..0.946.

VALUE fill accounting: spread **$5,814.22**, subsequent marked
move **$-17,860.68**, total **$-12,046.47**.
Per filled share, VALUE-1.867c versus ALWAYS-1.676c and TIMER-1.766c;
volume/value decomposition is descriptive, not a matched selection experiment.
The crude guard-exit sensitivity reduces loss on5/5dates,
but causes **14.41x placements**, with only
21.8% of VALUE's filled quantity. It uses the SAME
entry heads trained under neutral KEEP and a historical LODO guard, so it is
not a correctly relabeled combined candidate or evidence of selective keeper
skill. Naive entry/exit disagreement produces severe churn.

At actual VALUE entries, 66.9% of49487
POSTs lie in unseen coarse inventory/exposure/wait-age bins, versus
16.4% of9922 neutral labeled test probes. This is a
support proxy, not causal attribution. Generations advance after posting and
terminal waits are censored: eventual-POST rates do NOT prove failure to abstain.
Code inspection also finds a target/input gap: targets include whole-ladder
interactions, but inputs lack the other resting orders' prices, ages and
remaining quantities. No causal cost is assigned to that omission yet.

Eight diagnostic tests passed, including planted errors;47 frozen files
unchanged; **4320 original market/arm reconciliations**, plus all-fill terminal
accounting. Source exclusions:8361 untimed legs /
105159.20 shares; no missing feature-grid rows or excluded
registry windows. Exact-best/near presence excludes improving quotes and says
nothing about front queue residence. Failed diagnostic attempts1/2 remain
preserved; currentattempt3 completed, service exited, older size queue resumed.

Next: test direct POST-minus-WAIT prediction while retaining positive absolute
POST value; add client-known ladder/joint-waiting state support; label under
the actual frozen order-specific keeper/continuation. Gate prediction skill
before another broad strategy replay. These dates remain development-only;
validation needs later untouched complete days. Larger model/size or fixed
retry timers are not established remedies.


Follow-up to "prediction is totally reverse?" (2026-11-14T16:26:50.436761+00:00; same9,922
probes over5dates, measurements asof2026-11-14T16:21:11Z):
`reviews/wait_post_diagnosis_20261114/v3/sign_followup.json`. Actual-source
terminal accounting passes four bid/ask favorable/adverse sign controls and
refuses four corrupt-position cases. POST target is POST-minus-NEVER, not
its inverse. POST rank correlations are positive in6/10side-day cells and
negative in4/10, range-0.016..+0.082: weak/mixed, no consistent inversion.
Both negative-forecast and nonnegative-forecast cohorts lose on average on
both sides. Those cohorts include all probes, not just the selected POST
subset. This supports overconfident/noisy estimates plus selection of
optimistic errors, not simply flipping the score. No inverted policy was
evaluated, no new skill claim, and the controls do not exclude every other
possible data/model/simulator defect.
<!-- /WAIT_POST_WHY_20261114 -->

<!-- WAIT_POST_V2_20261114 -->
## 2026-11-14 15:01Z — wait/post v2 target fixes complete; five-day candidate REJECTED

USER: "We fix the model first, align the target and fix the issues. Then check
 the performance." Source **990b4e99**, freeze **e4a6ad83**, fix merged
**1ee92421**. Complete evidence **33cce3bf**, merged **3a057b9c** into
`mm-research`. Current result: `reviews/wait_post_v2_20261114/performance.json`;
`raw_report.json`, `daily.json`, `completion.json` bind and reconcile the artifacts.
This supersedes the interim `first_day.json` as the full-study status.

Target repair: POST and WAIT share the same frozen learned continuation, with
both valued against leaving the selected rung empty. WAIT can keep waiting;
POST requires positive absolute value and value at least as high as WAIT.
One delayed probe per sampled neutral generation adds waiting-state examples;
sampling ages are not retry timers. Raw whole-portfolio targets price shared
ladder displacement. Bid/ask POST/WAIT heads are separate, with shared inventory.
No fees/rebates or hyperparameter changes. Neutral KEEP with engine safety is
fixed throughout; combined learned keep/pull performance is not claimed.

16 synthetic checks pass, including causal prefix/feature parity, repeated
waiting, better later entry, shared-fill displacement, chronology, constant-head
refusals and matched-control falsifiers. All fitted heads have actual splits and
varying predictions; the final fold trains11969 actions on six earlier days.
The old constant-ask/small-support defect is repaired, without establishing skill.

Performance as of **2026-11-14T14:57:44Z**: five COMPLETE consumed development dates
(Oct27, Oct31, Nov1/2/3), **1440 market windows**, first60s of each market,
common60.2s terminal. M50 gross simulated terminal cash plus inventory:

| Entry arm; same neutral keeper | PNL, $ | Without <100ms-fill contributions, $ |
|---|---:|---:|
| VALUE | -12,046.47 | -11,850.13 |
| ADMISSION | -13,440.61 | -12,975.81 |
| ALWAYS | -13,832.57 | -13,161.58 |
| SCHEDULED | -11,509.31 | -11,617.62 |
| TIMER | -11,066.00 | -11,197.08 |
| FLAT | +0.00 | +0.00 |

VALUE loses on **5/5 dates at every M0/M50/M250 clock**, also without young-fill
contributions. It beats ALWAYS on4/5 M50 dates but TIMER on only1/5. Its
inventory-square exposure is **5.67% above TIMER**. Both-side quote presence
is **91.535%**, meaning at least one live order
per side, including deep rungs; no at-best/front-of-queue claim.

Pooled local probe diagnostics, cents per nominal30-share order share:

| Side | Eligible probes | Selected POST | Predicted absolute POST | Actual absolute POST | Actual POST-minus-WAIT | Matched-null p95 | Required test |
|---|---:|---:|---:|---:|---:|---:|---|
| bid | 5017 | 954 | +3.139 | -0.836 | -0.184 | +0.355 | FAIL |
| ask | 4905 | 889 | +3.794 | -3.656 | +0.109 | +0.180 | FAIL |

Each local null has500 selections matched on day/side/hour/rung and action count.
Both fail. These are counterfactual labeled states, not the realized policy's
exact entries; local labels are never summed as portfolio PNL. Absolute POST
value is negative for selected probes on both sides despite positive forecasts.
The remaining defect is unreliable admission-value prediction; joint waiting-
state distribution shift is a plausible concern, not a proven sole cause.

Acceptance FALSE: absolute profit, required relative comparisons, inventory risk
and both local tests fail. Only10/24 relative comparison cells pass. The200-draw
portfolio null was explicitly **skipped_basic_screen_failed**, as predeclared;
no incomplete/null-skipped result can accept. UTC-day intervals in the raw report
are descriptive; these consumed dates are not forward validation.

Source exclusions: **8361 untimed trade legs /105159.20 shares**; zero excluded
registry markets, no missing feature-grid rows. M0/M50/M250 shift public trade
anchors while the decision grid stays50ms. The young-fill cut removes those
legs' PNL contributions on the saved path; it is not a new execution replay.
Queue/fill assumptions remain uncalibrated, and terminal marks are not settlement
or realized live PNL. Guard comparators inherit historical LODO inputs.

Completion verified:42 frozen files,5 eval receipts,6 model receipts,9 label
receipts; model hashes/tree structures, daily sums, comparison predicates and
local means reconciled. Service `pf-wait-post-v2-20261114` exited0; older size
queue1576587 resumed with identity checked. No v2 worker remains running.
Raw models/data: `data/predict_fun/research/wait_post_v2_20261114_v1/full/`.

Next research: audit absolute-admission calibration and joint waiting-state
coverage under the frozen learned keeper before integrating entry with keep/pull.
Integration changes continuation values and requires new labels and replay.
No timer or threshold adjustment was selected from these outcomes.
<!-- /WAIT_POST_V2_20261114 -->

<!-- ORDER_HARM_CLIENT50_FIX_20261114 -->
Integration: merged into `mm-research` on 2026-11-14 10:16:29 UTC; no push or research run.

## 2026-11-14 — ORDER-HARM client-state / 50 ms repair complete in code

User requested the fix after the 09:47 review. Source **32da4014**, branch
`research/predict-order-harm-fix-20261114` (checkout `/tmp/cta-order-harm-fix-20261114`);
verified files mirrored into this shared workspace. [Repair record](handoffs/20261114_order_harm_client50_fix.md).
Labels and policy now share the client-state engine; unseen fills cannot end
label decisions before notice. Model/label decisions are every **50 ms**.
Portfolio cash and inventory are marked at **60.2 s**, after the 60 s stop and
cancellation tail; every run reconciles the complete fill value to cash plus
inventory. Frozen JOINT-LADDER v3 is unchanged; old 60 s values remain only for
regression. Fresh client50 directories and artifact-contract checks prevent
reuse of old labels/models. Re-entry cooldown and execution delays unchanged.
Verification as of **10:07:17 UTC**: 9 targeted tests, 12 default-engine parity
cases, 5 joint / 4 target / 5 policy planted defects caught, existing synthetic
selftests and shell syntax pass; zero shared historical-data read attempts.
Receipt: `reviews/order_harm_client50_fix_20261114/verification.json`.
No historical experiment or model fit launched. The earlier no-run instruction
remains in force. Next research step, when separately authorized, is the repaired
label/fit driver followed by policy replay. Upstream guard/FRONT-PULL LODO,
fill-model calibration and a real 50 ms inference benchmark remain unresolved.
<!-- /ORDER_HARM_CLIENT50_FIX_20261114 -->

USER 2026-11-12: "Can we still keep all our findings tracked, otherwise these misfindings happen again and again".

The program's memory of what is known, what is broken and what every new result must be checked against. HANDOFF.md is the
chronological log; this file is the standing state.

**How to use it.**
1. Read it at session start, after HANDOFF.md.
2. Before reporting any number, go through §1. A report that skips an item says which one and why.
3. When a study finishes, add or update its line in §3 in the same commit as its report.
4. When a later check changes an earlier finding, change its status here and point to the check. Never delete a line. Also put a
   note at the top of the old report.

Statuses:
- **HOLDS:** no later check against it.
- **QUALIFIED:** holds with the stated limit.
- **VOID:** a defect invalidates it.
- **SUPERSEDED:** replaced by a later result.
- **OPEN:** not yet tested.

## 1. Checklist for every replay result

| # | Rule | Why (source) |
|---|---|---|
| C1 | Split profit by the order's age at the fill (< 100 / 100-500 / >= 500 ms). Report the headline without fills on orders < 100 ms old and B1a-weighted, beside the headline | Replay fills on orders < 100 ms old were 8-12% real on live's own orders (B1a). 73-77% of v3's and QR2's replay profit is there (AGE) |
| C2 | Report fill clocks M 0 / M 50 / M 250, and say whether the claim holds at M 250 | No constant trade-time shift is valid; signs flip between clocks (B1a, review 2026-11-10) |
| C3 | Report the cancel-pending share of fills, and its $ | The replay credits 2.5x live's cancel-pending shares (B1a); 42% of the frozen v3's replay shares are cancel-pending (FV3) |
| C4 | Treat replay volumes and $ totals as inflated | The replay credits 1.5-2x live's shares (B1a R24) |
| C5 | Rule M primary, L shown; U is a front bound only | M is unvalidated; a fifth of live's fills contradict the shown-size queue (B1a) |
| C6 | Every decision may use only information observable at its time. Write out observation, decision and placement clocks: feed >= 13.2 ms (median ~53), decision 2 ms, placement 40 ms. To model faster placement, change that leg explicitly. Never schedule a price-dependent decision before its information exists | QR's fast entries decided before the level existed (review 2026-11-12 10:42Z, 9053263d) |
| C7 | A whole-market markout is not our order's value: replay one order with queue position before recommending | Pro rata flow reversed the sign once (2026-11-10) |
| C8 | Name the days used. Parameters chosen on seen days are post hoc. A pass is a candidate; validation needs a freeze and >= 5 untouched days | Consumed days: 10-23..27, 10-31, 11-01..03 (9 replay days), 11-07, 11-08. Never read 11-04..06 or 11-09..13 (forward blocks) |
| C9 | When ours is simulated and theirs is real, say so in the table | GAP |
| C10 | No replay number is validated at the order level; only a live run with the A4b capture can validate one | B1a; USER D2 |
| C11 | Name the strategy. Lane J-v3 (one 5-sh order per side, v3's PM90 gate on the 50 ms grid) is NOT the frozen pm_ladder v3 (MICRO side choice, G1-G6 guards, LD2 rung, 25-sh size, CAP10, XMIN1, XV, RO60 / S240, event-driven). Test any candidate on the frozen engine (`pf_step4_ladder_v3.py`) too | Pipeline review 2026-11-12, S1 |
| C12 | Split C1's age table by trade-anchor source (WS-timed / E2-estimated); E2 anchors are uncertain by about +-250 ms | Pipeline review D4 |
| C13 | Every decision in a replay lane uses CLIENT-KNOWN state (own orders: refusal known at activation, fills +100 ms, cancel confirmation at landing +100 ms; caps count every client-live share from submission). Ship a decision-invariance test: change every venue event after a cut; the decision log up to the cut must be identical (with planted leaks that must fail) | USER review 2026-11-14: JOINT-LADDER v2's cap released capacity on a refusal 40 ms before it was known; the S4 rest lane reads venue state (P21) |
| C14 | Compare keep / pull rules at the SAME decision cadence: a rule that reconsiders every 250 ms also cancels less often than one that reconsiders every 50 ms. Run the incumbent at the candidate's cadence (and print the cadence effect alone) | USER review 2026-11-14 of the ORDER-HARM POLICY design (250 ms model arms against FB25 at 50 ms) |
| C15 | Acceptance is a WHOLE-PORTFOLIO criterion declared before the run: every fill on every trade (WS and E2) marked to the window's end, at every fill clock (M 0 / 50 / 250), with and without fills on orders < 100 ms old, against every comparator, pooled and by day. A reading on a favorable subset (WS-timed fills, one clock) is a diagnostic, never acceptance | USER review 2026-11-14: "BETTER / PAYS" on WS fills while broader coverage or timing stress performs poorly |
| C16 | Models and thresholds are fit chronologically (train on earlier days, calibrate on the next, test on later days; comparators refit the same way), and an improvement needs a frozen forward test on >= 5 untouched days after that. Leave-one-day-out fits and thresholds set on all days are development only | USER review 2026-11-14 (thresholds from all nine days; LODO training used later dates) |
| C17 | A keep share is not quote availability. Judge a keep / pull policy on the whole-portfolio value AND the time with live quotes (every rung, during the guard, at the best), whether resting orders survive the guard, and whether more GOOD at-best fills (the taker order stopped at our price) are captured - the evidence for closing the maker gap | USER review 2026-11-14: the guard still blocks new orders and re-entry, then the cooldown, so a policy can keep more orders yet spend too long without quotes |
| C18 | Test a design in its OWN skeleton, and before rejecting it, find where its loss sits (trade source WS / E2, fill clock, timing class); a loss concentrated on E2-timed trades is checked against the E2 anchors' error (P12) before any conclusion. Restate the USER's design at the top of every plan and check the lane against it | COURSE REVIEW 2026-11-14: the USER's stop-price design was tested only inside the guard ladder and rejected on E2-timed losses that were never investigated |

## 2. Data and code pitfalls (each has bitten once)

| # | Pitfall | Fix |
|---|---|---|
| P1 | Step-1 signer ids are per day (order of first appearance), not global | Rank makers per day; a maker cannot be followed across days; our own account is unknown on the replay days |
| P2 | Live cancels sent before the ack can make B13c's re-place match pair two orders both ways | Walk chains only to strictly earlier accepts (QR part C, 4756b27d) |
| P3 | The venue's book arrives in ~50 ms snapshots (p10 / p50 / p90 41 / 50 / 58 ms after a birth frame) | "At the birth" has 50 ms resolution; the raise rule adds the next snapshot's joiners |
| P4 | Research hours padded by +-1 h reach sealed days | Enforce day rules in the loader; exclude boundary windows with a status |
| P5 | Proxy tuple keys merge distinct taker orders | Group by the event's own id (taker hash / legs_tk) |
| P6 | Cancel-pending orders still trade | Include them in price priority, self-cross checks and invariants |
| P7 | Results from before an engine fix | Void; report only fixed-engine numbers |
| P8 | STATUS.yml `updated:` must stay a block scalar | A quoted scalar broke the file (repaired 2026-11-12) |
| P9 | Forward data-day paths printed in a session count as reads | Aggregate at family level; never name the paths |
| P10 | Tie windows (start = end price, paid 50/50) are recorded as "Up" in the registry (`pf_step1.py:57-59`): 14 on the 9 days | Settlement-valued numbers need ties at 50 (fix proposed) |
| P11 | Step-1 day folders include the adjacent days' boundary hour files (sealed 10-28 / 10-30, forward 11-04); the loader keeps frames from ws - 5 s | Filter frames by stamp in the loader with a status; never list or read those files |
| P12 | Trade anchors: E2 estimates (about a third of first-minute shares) differ from WS times by IQR about -250 / +150 ms; a taker order's minimum time can come from a mismatched leg | See C12 |
| P13 | The sim2 family serves our own orders in time order (the frozen engine: best price first) and lets two of them take one trade leg | Matters when two of our orders overlap (CS fast paths: 1-7% of shares) |
| P14 | 10-31's taker ids: the step-1 and txmap signer indexes disagree (13% of first-minute shares unmapped) | Taker-order analyses on 10-31 are weaker |
| P15 | Binance is known at event time + 3.5 ms (a Tokyo-like constant); Binance-side delay spikes are not modelled | Sensitivity once a Tokyo busy-hour capture exists |
| P16 | Raw features (`gap`, `p_bn`) use the window's start price from tau = ws; only the guard value is masked (v = -1) until the price is known | Any model that reads raw features must apply the start-known mask itself |
| P17 | A rank AUC over rows that include NaN scores returns NaN without warning (FA's R341 on 5 of 9 days) | Drop or count NaN rows before any AUC / rank statistic; a NaN reading is an instrument failure, not a result |
| P18 | An order placed at a price beyond the 5 shown levels gets 0 sh queue ahead (sim2 / sim4 place() reads the unknown level as 0), so it starts at the front (NEXT-PRICE posthoc2: rung 5 at 0 sh on 97.8% of its births) | Exclude or flag such orders in queue statistics; in the guard ladder rung 5 is ~0% of at-best fill shares (QUEUE-PLACE, M 50) and sweep fills do not depend on the queue, so earlier ladder results are barely touched (not re-run) |
| P19 | The replay simulator's queue (a0 raised to the next frame up to 100 ms after activation; rule M's cap by frame stamps) uses information after the decision: it is the fill model, never a model input (ORDER-HARM v1 fed it; USER review 2026-11-14) | Build order inputs from frames known by the decision time (pf_order_harm_v2.queue_known) and ship a future-perturbation falsifier on real rows |
| P20 | A soft cap on the KNOWN position does not bound exposure: one sweep fills several resting orders before it reacts (JOINT-LADDER v1: ~2x the cap at p95) | Enforce caps with a pre-trade check on known position + every share able to fill, incl. unseen fills (pf_joint_ladder_v3; v2's hard mode released capacity on a refusal not yet known, P21); run conservation / accounting checks on every run, not only parity on non-interacting cases |
| P21 | Replay-lane decisions read VENUE state the client cannot know yet. (a) JOINT-LADDER v2's hard cap dropped a refused order from the cap's count at submission, 40 ms before the refusal is known, so a replacement could be placed at once (USER review 2026-11-14; conservation checks pass: a timing error, not volume). (b) The S4 rest lane (sim2 'rest', sim_arm, sim4: every per-rung study, ORDER-HARM's pull branch, JOINT-LADDER v1 / v2) decides "our order rests" from the venue's remaining size and the held price from the venue's live orders, so a fill not yet seen changes decisions up to 100 ms early. Size in the joint lane (JOINT-LADDER v3 R615, client-known minus venue-state decisions): LDG / FB25 -0.9 to 0.0 $/day at M 0 / 50 (-0.3 to +5.7 at M 250), the oracle -2.7 to -8.2 | Decide on client-known state: an order is live to the client from submission until its refusal is known (activation), its full fill is known (+100 ms) or its cancel is confirmed (landing +100 ms); caps count every client-live share (pf_joint_ladder_v3). Ship a decision-invariance test (C13). Earlier S4 results not re-run (size above); ORDER-HARM v3's labels now run on the client-state lane (pf_order_harm_client_lane, USER 32da4014) |
| P22 | A cap statistic printed as "max true exposure" measured the largest absolute position only (JOINT-LADDER v3's header; its 'cap' check compared the position with the cap). Position <= exposure, so a position check cannot show that a cap on exposure holds | Measure exposure directly: the position plus every share of ours that can still fill, from submission until the cancel lands or the order is fully filled (pf_joint_ladder_v3_exposure.max_exposure), with a positive control that overshoots (soft 2) and a planted position-only statistic |
| P23 | ORDER-HARM v2's keep branch was the never-pull lane, which also places replacement orders while the guard is on; keeping an order does not (the policy lane blocks placement during the guard). Its label also kept blind for 2 s while the policy reconsiders every 250 ms (USER review 2026-11-14) | Label on the policy's own lane rules (keep-all with the guard as the entry line) and cadence, with the option to cancel later (pf_order_harm_v3) |
| P24 | A fitted-Q model scored against its own bootstrapped target (the target contains the model's next-state prediction) shows skill by construction (caught in ORDER-HARM v3 before its first result) | Score against a realized, model-free outcome (keep-all minus pull) |
| P25 | One-step fitted-Q backups (target T0 + max(0, A_hat(next))) move the end-of-episode value back ONE decision per iteration: with 3 iterations only the last 3 decisions see it - 150 ms at 50 ms decisions, 750 ms at 250 ms. A synthetic 2 s episode worth +1 c/sh to keep read "pull" at its first decision (review of ORDER-HARM v3, 2026-11-14) | Policy iteration on realized continuation returns from a keep-all start: G_j = T0_j + [the policy keeps at the next decision] G_(j+1) (pf_order_harm_v3.continuation_targets, a8e6d473); print the decisions that change between iterations |
| P26 | A 5x finer decision step makes 5x the label rows: ORDER-HARM v3 at 50 ms holds ~1.6 M rows a day (v2: up to 322 k at 250 ms, peak 1.6 GiB); float64 rows plus a vstack copy pass the 2 GiB self-guard, and so do full-day reads in the fit and parity stages | Size memory before a cadence change: rows written in batches with the model inputs as float32, compact keys, sampling and prediction batch by batch (a8e6d473) |
| P27 | A "60 s mark" that counts fills up to 61 s values the fills after the stop (cancel-pending executions to 60.2 s) at the 60 s mid, before they happen: JOINT-LADDER v3's "all fills marked at 60 s" column and ORDER-HARM POLICY v2's first R652 (found by the USER's review 2026-11-14; v3's column not re-run, a few fills a window) | Mark cash plus inventory after every cancellation has landed, refuse post-terminal fills, reconcile per run (pf_order_harm_client_lane.portfolio_terminal, 60.2 s, USER 32da4014) |
| P28 | Leave-one-day-out inside an iterated fit is not out of fold: held-out day A shapes the other days' targets through earlier models trained on A, which then train the model predicting A - from the second iteration A reaches its own decisions (USER review 2026-11-14, reproduced with a synthetic learner through the actual loop; it does not put the test day into training) | Nested exclusion: every day's decisions come from a chain that never saw it (pf_order_harm_v3.chain_model, 06e534e8); falsifier: shift a held-out day's targets, its decisions must not move |
| P29 | The winning arm of one study became the skeleton of every later one: DEEP-PULL's LDG (pull every rung whenever the guard fires; it beat pulling only at the best) framed the label lanes, the decision scope (only inside guard episodes), the benchmark and the re-post rule from 11-13 to 11-14, although the guard is a sweep-cost alarm on a quantile line (MODEL AUDIT) | A benchmark is a comparator, never the frame (C18); the pull line's direction is in HANDOFF START HERE (2026-11-14) |
| P30 | A distribution-shift rule (PSI) drops inputs whose relation to the target is stable relative to a scale: STOP-MODEL's audit v1 would have dropped all 17 book sizes (the strongest inputs; the day's liquidity scale moves ~2x) and sigma (a covariate shift with a stable relation) | Before dropping, measure the relation's stability across dev days (rate of y by input decile minus the day's base rate); normalise scale-shifted inputs by a causal scale (W30: both sides' L1 size over the frames known in the last 30 s); drop only on an unstable relation |
| P31 | A permuted-label model's AUC strays from 0.5 on informative inputs (0.477 with Brier skill -0.005, STOP-MODEL fold 10-27): a two-sided AUC tolerance refuses a clean pipeline, and a permuted-label control cannot see a feature leak at all (the labels it learns from carry no signal) | Judge it one-sided on skill (Brier skill <= 0); cover feature leaks with future-perturbation falsifiers on real rows and brute-force labels |
| P32 | A "harm by label" alignment test can hold by construction: y = "our level was cleared" implies a THROUGH fill under rule M, so "run-over rows lose" cannot fail (STOP-MODEL R701 v1, build review B2) | Test a target by acting on it: an oracle replay against keep-all on the whole portfolio, with a time-shifted-label control at the same pull rate |
| P33 | A model trained and scored on a status population answers a question the policy cannot ask: STOP-MODEL v3.1's heads used only rows where the best had not already moved j ticks before the cancel could land (D0 < j); a policy acts on every state and cannot know D0, and there v3.1 predicted half the risk at 72 ms (3.8% vs 7.2%; 0.70 vs 0.99 on those rows at 1 s) | Train and score a decision model on the decision population (every known state); keep a status only as a diagnostic split (diag Q3, 2026-11-15) |
| P34 | Whole-day pandas reads fragment the heap: each ~110-column x 690k-row read added ~300 MB that was never returned, and the 2 GiB self-guard stopped a 28-min run with no output (STOP-MODEL DIAG 11-15) | Read the needed columns one at a time (pyarrow), build float32 model matrices column by column, malloc_trim before every guard check |
| P35 | A day's step-1 books folder holds the adjacent days' boundary hours (sealed 10-28 00Z in 10-27's, sealed 10-30 23Z in 10-31's, forward-block 11-04 00Z in 11-03's); pf_kp_s4.load_day and step 5 read every part file and filter the frames afterwards (none reached a row of an included window) | Select part files by hour BEFORE reading (pf_kp_s4.SKIP_FOREIGN_PARTS, opt-in, skips counted; STOP-MODEL turns it on); step 5's per-day clock shift is still computed over the folder's frames |
| P36 | A falsifier that perturbs the future but reuses precomputed upstream rows cannot see an upstream leak: STOP-MODEL's covered the 25 model-side inputs, not the 60 from step 5 (code review D1, 2026-11-15) | Recompute upstream inputs from raw data known by t at sampled steps and compare, with planted look-aheads that must fail (review s5check K1: 23 inputs, 100% match) |
| P37 | The pinned LightGBM parameters (since PULL-DEPTH; no L2) overfit rare heads: lambda_l2 = 10 cut the deep heads' log loss 9-15% and left level 1 unchanged (STOP-MODEL reg R3, 2026-11-15) | Check regularization on a dev fold when heads have few positives; flag any change of pinned parameters |
| P38 | Step 5's fair price is biased in the first minute of every HH:00 window: its per-hour Binance passes include each hour's 10 s carry, so the relay prints there are counted in two passes and the 60 s median relay basis is off for steps within 60 s after an hour boundary (24 of 288 windows a day, ~8% of rows; on those rows gap_adv / pbn_dist p99 3.1 c, max 5.6 c; dfv250 p99 0.6 c). Found by recomputing every step-5 input from raw data (STOP-MODEL s5check2, 2026-11-15; 8 of 9 days matched 100% because the sample missed HH:00 windows). A biased statistic, not a look-ahead. Every consumer of rows_rl_d2_pk's p_bn / gap inherits it (incl. the entry model's fair-price inputs) | Count each relay print once; STOP-MODEL's fair_fix applies the corrected basis as a delta on step 5's p_bn (other rows unchanged bit for bit); step 5 itself is shared and not rebuilt. Always sample the boundary cases (HH:00 windows) in a recomputation check |
| P39 | A "timing-clean" group defined by the trades that reach an order in its OWN future window is not timing-independent: earlier uncertain trades drain its queue, change which generation is resting, or split later legs across our rungs differently (Codex counterexamples; E2SEP v1's future-WS-reached group moved -2.5 / -1.1 / -2.9 c under common E2 shifts of -250 / 0 / +250 ms) | Certify against the PATH: an uncertain event is harmless only if it cannot reach any accepted order of the side before the label ends (pf_stop_ev_timing_certificate); compare fixed identities across shifts |
| P40 | A robust (Huber) value head is not an expected value, so its product with P(fill) is not EV and a zero threshold on it is biased (STOP-EV v1: the top prediction decile +0.43 c predicted, -0.53 c realized); and LightGBM's Huber starts from the label mean and barely moves on a homogeneous group, so an aggregate check cannot tell it from squared error | Use a mean target for decisions on value; test the fitted objective and group means where the robust location and the mean differ (two groups, one with outliers) |
| P41 | A local-gain reading "value saved by the pulled actions vs keep-all" is positive whenever the population loses on average, so it cannot fail (STOP-EV v1 / v2 R903 read LOCAL-GAIN while every prediction decile lost); comparing two rules at their own pull shares ranks the one that pulls more | Read the KEPT set's own value (> 0) beside a matched random-keep null at the same count; compare baselines at the same kept count (R903' / R904') |
| P42 | A cadence policy whose entry gate is evaluated only at decision steps and held otherwise lets a rung it just pulled re-post at the next 50 ms step (STOP-EV-PLAY v1: every pull re-posted within ~150 ms at the back of the queue, unchecked) | A pulled rung stays empty until the next decision step; every rung's gate is evaluated fresh at each decision step; test with a stub head that pulls every resting order and count re-posts before the next decision (pf_stop_ev_play2 selftest 3) |
| P43 | A random control matched on how OFTEN an arm acts but not WHERE it acts compares placement, not timing: STOP-EV-PLAY v2's RV took QR's pull and closed-gate rates per window and side, but QR quoted rungs 0-1 0% of the time and RV 26-28% (even 0.26-0.34 on every rung), so "QR beats RV" (R984) can come from avoiding the touch rather than from when it pulls | Match the control per rung as well (window x side x rung x clock) and print the arm's and the control's uptime per rung beside the reading |

## 3. Findings

### A. Is the replay right?

| Finding | Status | Source |
|---|---|---|
| The replay is not validated order by order. It credits 1.47-1.96x live's shares, at precision 0.03-0.25. Orders < 100 ms old are 8-12% real (M 0 / M 50); no constant trade shift is valid. Live's real young fills were not adverse (+0.05 c/sh) | HOLDS (one day, the old churning engine) | B1a, 20261110_b1a_live_orders.md |
| 73-77% of v3's and QR2's replay profit is on orders < 100 ms old. Without them v3 is $3.7 a day at M 50 ($2.5 B1a-weighted); orders >= 500 ms earn about 0; M 250 about 0 | HOLDS | AGE, 20261112_age_recheck_results.md |
| Replay results before round i (2026-11-05) | VOID (USER) | STATUS prefix_results |

### B. v3 and the guard

| Finding | Status | Source |
|---|---|---|
| The frozen pm_ladder v3 (its own engine, 9 replay days, whole window): $60 / $45 a day at 1 s (J / X timing), settlement $99 / $65 with ties at 50; 79-83% of the 1 s profit is on orders < 100 ms old (about $10 a day without them, $9-10 B1a-weighted); orders >= 500 ms lose slightly; 42% of its filled shares are on cancel-pending orders; at the 250 ms fill clock it loses $40 a day (1 of 9 days positive) | QUALIFIED: young and cancel-pending fills unvalidated (B1a); parameters chosen on these days | FV3, 20261112_fv3_results.md |
| A faster order path on the frozen v3 (20 / 10 ms against 40): +$2.0-2.6 / +$4.6-4.7 a day without young fills, 8-9 of 9 days at J and X; not at the 250 ms clock | HOLDS (replay; the speeds are settings, not measurements) | FV3 R332 / R333 |
| The frozen v3 has no inventory price skew (SK / RX / RL / A4 are not in its arm); its inventory logic only withholds quotes: CAP10 (10 sh net), G6 (50 sh, binds only without CAP10), RO60 (reduce-only from 60 s). Only CAP10 touches the first minute: it withholds about a sixth of v3's shares there; without it +$2.7 (J, 8 of 9 days) / +$1.2 (X, 4 of 9) a day w/o young fills, settlement -$20 / -$10 (4 of 9 days above: noise). The J-v3 timing studies carry no inventory logic. Timing base for the frozen engine: first 60 s, CAP10 and G6 off, venue rules kept | QUALIFIED: post hoc, in-sample (CAP10 / RO60 chosen on these days); no arm without G6 exists | INV-RULES, 20261112_inv_rules_results.md |
| v3 without its quoting limits (CAP10, G6, RO60 off; first minute): +$4.5 (J, 8 of 9 days) / +$3.4 (X, 7 of 9) a day w/o young fills, settlement lower (more inventory carried). Improving one tick inside a >= 2-tick spread (IM95) moves us toward the front (filled shares with <= 5 sh ahead 18.7% vs 5.0%, median queue ahead 208 vs 308 sh) but adds only +$1.6 / +$0.4 a day w/o young (6 of 9 days), $0 at the 250 ms clock: being first alone barely pays | QUALIFIED: replay rule M (B1a), consumed days | TB, 20261112_tb_results.md |
| Public tape (no fill model): no missing information in any feed we have. On top of the guard's inputs, Predict's trade tape (incl. a causal taker reputation), book cancels / adds / level age and Coinbase / OKX / Bybit add <= +0.002 AUC for 'the best moves against the side in (t + 24, t + 274] ms'. The guard's inputs fitted on that event reach AUC 0.87 (activity 0.60: they track harm when aimed at it); 29% of those moves happen before a cancel can land (speed lever); the avoidable rest is predicted at AUC 0.84 (9 of 9 days 0.82-0.86), mainly by the guard's own v (53% of gain); among trades that arrive, clearing vs standing at 0.82. Pulling in 11% / 23% of moments catches 50% / 70% of the avoidable level losses | HOLDS (descriptive, consumed days; AVOID / ops post hoc; AUC is not money: the pull's cost is the queue place) | MOVE, 20261112_move_results.md |
| Which live feed carries the avoidable-harm prediction (post hoc, MOVE-AVOID feeds; level still there at t + 24, goes in (t + 24, t + 274] ms): Binance alone AUC 0.762, Predict's book alone 0.767, both 0.841; the guard value v adds nothing on top (-0.0003; v alone 0.779); each feed adds about +0.08 to the other, so a pull needs both | HOLDS (consumed days, first minute, post hoc, public tape) | 20261112_move_results.md; 50227d6e |
| Rest at the best by default and pull on P (the level goes in 24-274 ms) (FRONT-PULL, J-v3 rest lane, 9 days, first minute): beats matched random pulls on 9 / 9 days but loses to the guard trigger v <= -0.10 on 0 / 9 (-$48 vs -$20 a day w/o young at M 50; -1.18 vs -0.37 c/sh) and to J-v3 (-$52 a day); more pulling worsens each remaining fill (top-10% line -0.64, top-50% -1.30 c/sh; swept 48% -> 63%); a front-tier line adds +$1.3 a day (8 / 9); with nothing ahead (rule U) the same policies earn +$171-246 a day: the queue model decides the sign; fills during our cancel 6% of shares (-$6.5 a day) | HOLDS (replay rule M, B1a; L not run; consumed days, design after MOVE) | 20261113_front_pull_results.md; df320ba3 |
| The top makers' order shape in our replay (FRONT-PULL-SIZE, S4 rest lane, 9 days, first minute): size does not change per-share value (FPQ -1.18 / -1.19 / -1.18 c/sh w/o young at 5 / 30 / 100 sh, M 50; R410 NOT-SHOWN), so losses scale (FPQ-30 -$234, FPQ-100 -$562 a day); a 6-rung ladder at 30 sh loses on 9 / 9 days (LFPQ-30 -$5,095 a day, R411 LADDER-HURTS; deep rungs 68% swept, negative even at rule U); our kept 30-100 sh orders are filled 3+ times (41-61%) but 60% of their shares come from level-taking trades against the top makers' ~16%; guard trigger stays the best pull; at rule U single-level resting + guard earns +$213 / +$689 / +$1,203 a day at 5 / 30 / 100 sh: the queue model decides | HOLDS (replay rule M, B1a; L not run; consumed days, design after ORDERS) | 20261113_front_pull_size_results.md; b8e7592a |
| Why real fills behind the best pay and our rungs lose (DEPTH-TIMING: same days, tape and value function; WS-timed exact complete trades): behind the best the top 5 earn +0.89 c/sh at 1 s and the field +0.58 (rising with depth: +0.58 / +0.98 / +1.77 at 1 / 2 / 3+ ticks), our never-pulled 30-sh rungs -1.10; in the same tercile of our P our fills are worth about theirs (low P STOP +1.41 vs +1.45), but half our deep shares fill at high P (-1.35 / -3.64 STOP / THROUGH) against 19% of theirs (R422 PRESENT-WHEN-SAFE; R420 WITHIN-TYPE 73%, 9 / 9 days; R421 64% vs 37% stops): it is the timing of presence / pulls at every rung, not the replay | HOLDS (replay rule M for ours; 62% of our fills excluded as not WS-timed / complete, same rule as real) | 20261113_depth_timing_results.md; 586efcba |
| Pulling every ladder rung when risk is high (DEEP-PULL, 30 sh x 6 rungs, 9 days, first minute): the high-P share of our deep fills falls from 54% to 13% (top makers 19%) and the ladder gains +$4,033 a day over pull-at-the-best (9 / 9 days, R430 DEEP-PULL-HELPS), but no arm is positive (R431 0 of 3; R432 0 of 6 vs J-v3): LD90 -$1,062, LDG (guard trigger on every rung) -$137 at M 50 (-$16 at M 0; deep rungs -0.33 c/sh), +$781 at the front bound; deep shares still 61-69% swept vs the top makers' 36% with ~75k pulls a day: the remaining gap is queue place at the deep levels (each pull re-queues) | HOLDS (replay rule M, B1a; consumed days, design after DEPTH-TIMING) | 20261113_deep_pull_results.md; ffc7e475 |
| Fewer / stickier pulls on the guard ladder (KEEP-PLACE: strong-signal-only, near-best-only, immediate re-entry): none helps (R440 KPS / KPN HURTS 9 / 9 days, KPF NOT-SHOWN; R441 0 of 3, R442 0 of 6); the pulls are needed (near-best-only lets deep fills at high P back to 67%), and our 60-68% swept share does not fall with a third of the pulls: our new orders join behind queues that already exist. Gap review: size, persistence, levels, high-risk exposure, two-sided closed in replay; open = which trades fill us at a level (rung 0 48% / deep 63% level-taking vs the top makers' 20% / 36%), speed / fill clock, taking | HOLDS (replay rule M; consumed days, tuning variants) | 20261113_keep_place_and_gap_review.md; 0a5a6ee6 |
| The guard ladder's loss is on estimated-time trades (TIMING-SPLIT, C12): LDG (30 sh x 6 rungs, guard on every rung) earns +$54 a day w/o young at M 50 (+0.24 c/sh) and +$86 at M 0 on WS-timed trades (7 / 9 days, R451 WS-POSITIVE) and loses -$190 on E2-timed ones (46% of its shares; R450 E2-DRIVEN); its WS-timed fills behind the best earn +0.92 c/sh against the top 5's +0.86; real makers' E2-timed fills are also much lower (top 5 at the best +0.38 -> -0.08); at the best our WS aged fills +0.05 vs +0.38; at M 250 even WS loses (-$936). Qualifies FRONT-PULL-SIZE / DEEP-PULL / KEEP-PLACE totals | HOLDS (replay rule M; consumed days; WS-only is a selection) | 20261113_timing_split_results.md; 298390f7 |
| The gap at the best price is queue place (BEST-GAP, exactly-timed trades, rung 0 with guard pulls, 30 sh): top 5 +0.405 c/sh, ours +0.007 (gap +0.398); their at-best fills are 80% trades that stop at their level, ours 37% (63% level-taking); within type ours are close (STOP +0.47 vs +0.53). Levers: queue place (rule U bound) +0.660 (166%, largest on 9 / 9 days; R464 MAIN: QUEUE), latency FAST +0.111 (28%; ZERO +0.133), size +0.003 (1%), timing -0.133 (we already sit at the best at lower risk than they do) | HOLDS (replay rule M; U / ZERO bounds; consumed days) | 20261113_best_gap_results.md; a4767fc1 |
| Our queue place at the best is lost at entry, not by re-places (QUEUE-PLACE, guard ladder rung 0, M 50, exact trades): of the queue ahead at our fills (14.5 sh) 22% is our own re-places (80% guard pulls; R480 BEHIND-FROM-ENTRY, 7 / 9 days) and 78% was ahead before our first order at that price went live - mostly LATE (40%; R481 LATE-ENTRY, 8 / 9): our first order arrives a median 359 ms after the level becomes the best (5% within 97 ms), when only 12% of the eventual queue was there at the birth. Post hoc: at the birth we are OUT (guard pull + 250 ms clear wait, 60%) or HELD at the old price (38%). Orders already at the price before it became the best (PRE-KEPT) earn +0.20 c/sh (rung 0 +0.29) at 57-63% STOP vs +0.03 / -0.13 and 33-35% for joins after the birth (R482 BEING-THERE-FIRST-PAYS, 7 / 9); only 17% of our at-best shares. Top 5 at-best fills come 0.84 s into a level's life vs ours 2.1 s, same ~60% fallback mix | HOLDS (replay rule M; consumed days; WHY-LATE post hoc) | 20261113_queue_place_results.md; bad57131 / 2f7143ea |
| Pull only the prices at risk (PULL-DEPTH, guard ladder): at guard-on steps the move within 1 s is predictable out of day (AUC 0.75 for >= 1-3 ticks; R490 PREDICTABLE) and most signals are false alarms (60% of onsets: no move in 1 s; 20% one tick). Pulling a rung only if P(move >= its depth) >= 0.10 / 0.25 beats pulling every rung on WS-timed trades: +$23 / +$44 a day at M 50 (+$29 / +$60 at M 0), 9/9 and 8/9 days (R491 HELPS), 39-59% fewer pulls; the gain is behind the best (+$62 -> +$83 / +$103), not at it; fixed-depth keeping loses $600-1,000 a day; on E2 trades PD25 loses more, over all trades only PD10 beats LDG. Two sides after a signal: the hit side's new best already holds 128 sh at its birth (the top 5 take 42% of its fills at +0.30 c/sh vs field -0.11; front proven resting before the signal 18% vs 10%; R493 PROPORTIONAL 1.10); the other side's new best starts at 54 sh and fills in ~250 ms, fills in its first 100 ms earn +0.96 vs +0.45 c/sh (R494 EARLY-PAYS 9/9); the top 5 do not win that race (20% of first fills); our FAST path gets 30% more shares there, per share 6/9 (R495 NOT-SHOWN) | HOLDS at M 0 / M 50 on WS trades; FLIPS at M 250 (FUTURE-BEST post hoc: PD10 -$172, PD25 -$453 a day vs LDG on WS trades, 0/9 days); P model inherits PULL-DEPTH's nested-fold leak (FRONT-PULL score fitted with the held-out day), known-frame move target and missing-frame zeros (other session's review 20261113_pull_depth_model_integration_review.md; magnitude unmeasured) (replay rule M; consumed days; five arms) | 20261113_pull_depth_results.md; 6fa5182e / 7f5f6aa1; 20261113_future_best_results.md |
| Keep the protection side at the future best (FUTURE-BEST, guard ladder, the USER's rule): while the guard is on, pull an order only if the move is likely to go THROUGH its price (P(move >= d + 1) >= tau, PULL-DEPTH's model and tau) and keep the order at the predicted landing price. On WS-timed trades it beats PULL-DEPTH (FB10 +$18, FB25 +$27 a day at M 50; +$21 / +$48 at M 0; 8/9 and 7/9 days, R500 BETTER) and pulling everything (+$41 / +$71, R501 HELPS); the gain is on orders >= 500 ms, mostly behind the best. It does not hold at M 250 (every keeping arm, PD included, below LDG on 9/9 days) or over all trades (FB10 +$4 vs LDG, -$13 vs PD10; FB25 -$134 vs LDG; E2 losses grow). At the hit side's new best FB25 puts us there first (64% of our shares from orders resting before it became the best vs 16%), but those orders are run over half the time (50% STOP, PRE -0.22 c/sh; R502 NOT-SHOWN). With the realized move (OFB, perfect foresight; a diagnostic, not a bound) the same rule makes +$378 a day more than LDG on WS trades and +0.90 c/sh at 88% STOP on the new best (top 5 +0.30, 82% STOP): the rule is limited by predicting where the move stops | FB10 / FB25 gain NOT robust (flips at M 250, all trades); P model inherits PULL-DEPTH's nested-fold leak (FRONT-PULL score fitted with the held-out day), known-frame move target and missing-frame zeros (other session's review 20261113_pull_depth_model_integration_review.md; magnitude unmeasured); 30-fill day rule checked after review (7b5eced4, min 657, labels unchanged); replay rule M; consumed days | 20261113_future_best_results.md; 037ed9de / 6f13eb38 |
| The top makers handle the two sides differently around a move (SIDE-MOVES + post hoc 6e49d5ff, real WS legs with maker / order hash; fills only: public match records carry no order creation or cancel time). Protection side (the side the price moves into: asks in a pump): their orders are mostly ALREADY at the next price before the move: in the trade that pushes the price there they hold 49% / 47% (pump / dump) of the volume filled at that price (their at-best share 34%); >= 16-17% of their later fills there are from orders proven there before (field 9%, 9/9 days); first in its queue (median 0 sh filled ahead vs 6-10); +0.30 / +0.29 c/sh there (field -0.05 / -0.18). Makers quoting the old best are filled at the new best 2.75x as often as the field (15.4% vs 5.6%), which can be a ladder order rather than a requote; run over at the old best proportionally (R513). Other side (the side the price moves away from: bids in a pump): about one in three quote the better price where it trades enough to see it (top 5 32% vs field 28%, ROBUST-CHECK K4; the 6% vs 4-5% over all events, R510, is a visibility artifact); their old order is still there and filled again more often than others' (18.4-18.5% vs 6.9-7.4%; with >= 100 sh traded there after the move 24% vs 9%; R511 9/9), so at least a quarter stay and the rest are gone (cancelled or used up, not separable); kept fills earn +0.52 / +0.64 c/sh about 1.2-1.4 s after the move (field -0.04 / +0.30); when they join the new price they are first (R512 9/9). Same in pumps and dumps and at calm moves: how they quote, not a reaction to a signal. Ours (simulated LDG rung 0): protection side no order at the birth 89%; other side our guard pulls the held order (MOVE-DOWN) RESTATED 2026-11-13 16:15Z by MOVE-CASES: the next-price share was measured only at the price where the move STOPPED; at prices run over in bigger moves the top 5 are also heavily filled (43% of that volume in big trends) and lose -3.6 c/sh there; being at the next price is not an edge by itself. | HOLDS (descriptive; consumed days; fills only). Corrected 2026-11-13 15:13Z: the first version said the top 5 "MOVE down" on the protection side and "mostly KEEP" on the other side | 20261113_side_moves_results.md; e2626a18 / 36637b1f / 6e49d5ff |
| Reaching the protection side's new best faster after a move loses; a faster path in general helps (MOVE-DOWN, rung 0 of the guard ladder). Today our order is there in 35% of events, live 347 ms after the birth (guard still on at the birth 83%, clears 236 ms later, then 250 ms + 40 ms). Entering at once when the move happens (MD) gets us there in 60% at 128 ms, with the fast path (MDF) 78% at 46 ms (R520 FASTER 9/9), but still behind 161-222 sh shown, filled when the move continues (31-37% STOP, -0.66 to -0.70 c/sh; R522 LOWER 9/9): rung 0 -$444 / -$398 a day vs LDG at M 50, every clock (R521 HURTS). The top makers' place there (0 sh filled ahead) comes from being in the queue at or before the birth, not from speed after it. Today's rule with the fast path (FAST, 5 ms order path + fastest feed) helps at every clock: +$19 / +$22 / +$12 a day at M 0 / 50 / 250, 9/9 each, all trades +$44 (R521 HELPS; first gain here that holds at M 250). Other side: our held old-best order is pulled by our own guard on that side within 1 s 52% of the time (drift 11%, still live 34%); the top makers keep theirs | HOLDS (replay rule M; consumed days; rung 0; FAST path hypothetical; MDP arm uninformative: entry not gated, churned) | 20261113_move_down_results.md; 3695d17c |
| The conclusions about the two sides and speed hold under stricter tests (ROBUST-CHECK v1 1ae29bdb, v2 9b218820 after an independent code review; post hoc aaa04b61). Side the price moves into: in taker orders that cross from the old best into the next price the top 5 hold 50.7% of the volume at the next price vs 28-29% at the old best (usual 33.9%; 9/9 days, pump and dump); their orders lead the new best's queue (ratio 1.37 by first fill, 1.19 by first trade, 9/9); proven there >= 250 ms before the move 6.6-6.9% vs 4.1-4.4% (8/9). Side the price moves away from: where the new price trades >= 100 sh, 32% of top makers at the old best also quote it vs 28% of the field (about one in three, not rare; the 6% over all events was a visibility artifact); per order, 24-25% of their old orders fill again when >= 300 sh trade there (field 8-9%; not used up 46% vs 11-12%; 9/9), kept fills +0.53 / +0.60 c/sh at 1 s (9/9). Speed: early entry at the new best after the move, corrected, loses $384-477 a day vs the same path at every clock (9/9); gated by the model it adds nothing at M 0 / 50 and loses $32 at M 250; the fast path on the full ladder +$27 / +$31 a day WS (M 0 / 50, 9/9), +$22 at M 250 (8/9), about +$70 over all trades. Review fixes moved no number materially (0.1% invalid events; hash coverage top 5 99.8% vs field 96.1%) RESTATED 2026-11-13 16:15Z by MOVE-CASES: the next-price share was measured only at the price where the move STOPPED; at prices run over in bigger moves the top 5 are also heavily filled (43% of that volume in big trends) and lose -3.6 c/sh there; being at the next price is not an edge by itself. | HOLDS as measured, C1 restated (replay rule M for ours; fills only for theirs; consumed days; K4 passes its 33% line narrowly) | 20261113_robust_check_results.md; 1ae29bdb / aaa04b61 / 9b218820 |
| Around a move the top makers take adverse selection on the side the price moves into and are paid by the other side (MOVE-CASES, real WS fills valued at 5 s; protection-side events after a guard onset). Cases: 1 tick holds 15.9%, 1 tick reverts 19.1%, 2 ticks 20.9%, big (>= 3 ticks in 2 s) trends 25.8%, partial 4.2%, reverts 14.1%; pumps = dumps. Big trends: run over at the old best (31.5% of the volume, -5.9 c/sh) and at every price on the way (42.9% of the run-over volume, above their usual 33.9%, 9/9 days, R540 THICK-WHERE-RUN-OVER; -3.6 c/sh), stop at a price where they are thickest (58.8%); bids filled during the pump +7.4 / +4.7 c/sh; ask side -$5,924 / day, bid side +$6,168, net +$245 (R541: they lose there 9/9, less than the field 9/9). Big reverts: the same asks win (+$4,394), bids lose (-$1,102), net +$3,292. Small moves +$0.8k to +$2.0k per case. All cases +$7,965 / day (+0.52 c/sh, 9/9, R544); field +$7,103 (+0.30 c/sh) on 1.5x the volume; per share the top 5 are better by about 0.2-0.7 c/sh in almost every segment. As takers they lean into big moves a little (4.9% of the buying vs 3.5%, R542 9/9). Ours (simulated LDG): two-sided like them (away / into volume 0.94 vs 0.90), +0.15 c/sh vs +0.52, worst in big trends (-$33 / day) | HOLDS (descriptive; fills only; 5 s marks; consumed days) | 20261113_move_cases_results.md; 0dd3eb0c / a8cb3141 |
| The top makers avoid TRADES, we avoid PERIODS (LEARN-PULL, public tape: front orders at the best proven by a sole-maker stop, followed to the next trade at that price within 3 s; 193,153 observations). On the same intervals the top 5 pull 27.8% vs our guard 59.8% (2.2x; every interval bucket). They pull 47.0% before trades that go THROUGH their price vs 26.9% before trades that stop (1.74, 9/9 days, R550 AVOID-ADVERSE); a front fill is worth +0.50 c/sh on a stopping trade and +0.07 even when the trade goes through. Where we pull and they stay (38.6% of intervals) the skipped front fill was worth +0.25 c/sh (8/9 days, R552 OUR-EXTRA-PULLS-COST); where both pull, +0.01. Their pulls follow the level's own state (P 18% -> 47% by quintile, thin queue 42% vs thick 19%, own side depleting) and hardly Binance imbalance (26 -> 33%) or volatility (30 -> 27%), which drive ours (35 -> 87%, 48 -> 67%); model AUC 0.78 (v alone 0.59, inventory 0.53; R555 inventory NOT-SHOWN, 1.10). In big moves they quote throughout (top-5 fills at prices run over in 73% of slower moves, 29% of the fastest sweeps) and their share rises with depth (36 -> 44%): no backing off. Their low pull rate rests on being at the front; FRONT-PULL showed their kind of rule loses for us at our queue position (rule M) and wins with nothing ahead (rule U) | HOLDS (fills only; price-time assumed; 5 s marks; ambiguous intervals excluded; consumed days) | 20261113_learn_pull_results.md; e3ad37ff / d4adcdef / 245fbc4c |
| Why the top makers pull (LEARN-PULL-TARGET, post hoc on LEARN-PULL's 78,294 top-5 front-order observations): a trade taking their level is only part of it. Trades that go THROUGH their price are 4% of the next trades there (gone before 37-47% of them); among trades that STOP at their price they are gone 1.77x as often before a fill that loses at 1 s as before one that wins (39.9% vs 22.5%; 1.26 at 5 s, 9/9 days, R557 PULLS-TRACK-FILL-LOSS). Per trade the fills they kept were worth +0.56 c at 1 s and the ones they skipped -0.12; the trade's kind accounts for 6% of that gap (9/9; declared volume-weighted R558 GAP-MOSTLY-WITHIN-KIND, 7/9). Their pulls line up with a ~1 s fill-value target (gains and losses), not the sweep alone. Our guard's label is the sweep alone (pf_badfill_grid_tx: the 1 s loss if the best is swept within 0.5 s, else 0; no credit for good fills; one line at the front and the back). At their pull count our guard value / P keep at least as much per trade as their pulls (+0.58 / +0.61 vs +0.55 c at 1 s, 9/9; post hoc, our signals read 100 ms before the trade); the declared $-weighted comparison is VOID (R560: their pulls above the random p95 on 3/9 days; the volume at the price depends on their own presence). The difference is how often we pull and where our orders sit, not a better signal on their side | HOLDS (post hoc; fills only; GONE includes moves; their pull can break the level; consumed days) | 20261114_learn_pull_target_results.md; a325538c / acbdb7f7 |
| Where we are when the next price becomes the best (NEXT-PRICE, guard ladder, 9 days, first minute; 154,740 fallback births at M 50, e.g. 50c goes and 49c becomes the best): our order is at that price only 7.8% of the time; 59.9% the guard pulled the order we had there a median 0.56 s before (it had rested ~0.4 s); 29.5% we had none there for 5 s (88% with the guard on at the birth). Late because we repost, not on first arrival (R570 LATE-BY-REPOST: 66% vs 32% of the misses, 9/9 days; the same at M 0 / 250). We come back 434 ms after the birth, after the level is gone 72% of the time. Kept, the pulled orders would have had a median 80 sh ahead at the birth (15% at the front) against 469 sh for our next order on the same births (rungs 0-4, post hoc; declared R571 NOT-SHOWN, 2/9 days); orders that were there (age 0.8 s) sit behind ~190 sh (2% front): the queue formed before them. The top makers hold the front of a price that comes from behind: small first trades there are 45% theirs (x1.36 their at-best share, 9/9, R572 THEY-HOLD-THE-FRONT); taker orders running through the old best into it fill a top-5 maker 62% of the time (resting there before). At new better prices they are under their share (x0.66) | HOLDS (replay rule M for ours; fills only for theirs; consumed days; queue pairing post hoc) | 20261114_next_price_results.md; 89b45cac / 2626caae / 0eaa311b |
| How far a danger move goes, and how well we can predict where it stops (STOP-DEPTH, guard-on steps, 9 days, first minute; 1.83 M steps with nothing moved yet; clean target on venue frames after t + 22 ms): within 1 s 59% no move, 20% 1 tick, 8% 2, 13% 3+; a move that starts stops at 1 tick 48% of the time. Its size follows the Binance move against us: with the top third of 250 ms Binance moves against us (2% of steps) 68% of moves go 3+ ticks and 15% stop at 1; with no Binance move (93% of steps) 51% stop at 1 (post hoc). A cleaned and extended model (no fitted inputs; signed directional features, size within 1-5 ticks, recent best moves) beats PULL-DEPTH's on >= 7/9 days but only marginally (AUC 0.725 vs 0.721 for 1 tick, 0.733 vs 0.731 for 2, 0.738 vs 0.736 for 3; R580 PREDICTABLE, R581 BETTER, R582 FEATURES-HELP) and is well calibrated; at FB25's keep count for the order one tick behind it runs over 14.6% of kept steps vs 14.7% (R583 NOT-SHOWN, 6/9), against 0% (and 28% stopped there) with the realized move. The stop is not much more predictable from our inputs: the into-side gain stays FUTURE-BEST's | HOLDS (offline, no replay; our guard's population; 1 s horizon; consumed days, inputs chosen after PULL-DEPTH / FUTURE-BEST) | 20261114_stop_depth_results.md; ebec8cd1 / 4adf3d32 |
| The influence of a Binance move, in price units (INFLUENCE, 9 days, first minute; target reset to the signal's onset and the whole move): when the Binance-implied fair price p_bn moves >= 0.5 c against a side in 250 ms (34,813 events), the Predict mid follows about one tick per cent (0.9-1.25 at 1 s), 70-80% of it within 250 ms, and it stays at 3 s (R590 FOLLOWS, 9/9); a >= 4 c move moves the best >= 3 ticks 85% of the time. But only 2.9% of our guard's onsets come with such a move; at the other 97% (Predict-local signals) the move does not depend on the fair price: 50% no move within 2 s, moves stop at 1 tick 41-44% of the time; with a >= 2 c fair-price move 13% no move and 52% go 5+ ticks. The fair-price gap's level hardly matters (Spearman +0.04-0.08, R591 FOLLOWS). After a guard onset the farthest move within 2 s: 50% none, 21% 1 tick, 10% 2, 19% 3+; peaks take a median 0.6-1.1 s. A model rebuilt on the onset target beats STOP-DEPTH's step model there (AUC 0.701 / 0.685 / 0.675 vs 0.697 / 0.679 / 0.667 for 1 / 2 / 3 ticks, 8/9 days, R593 BETTER), the influence inputs add nothing beyond the Binance inputs (R592 NOT-SHOWN), and the keep decision for the order one tick behind barely moves (24.0% vs 24.1% of kept onsets run over, lower on 9/9, R594) | HOLDS (offline; p_bn basis / sigma as step 5; first minute; our guard's onsets; consumed days) | 20261114_influence_results.md; 19e52dfb |
| Is the move target right, and do the rebuilt models help in the replay (TARGET-TEST, FUTURE-BEST's harness unchanged, 9 days, first minute, WS $/day w/o young at 1 s): perfect knowledge of the move over 1 s or 2 s is worth the same (OFB1 +432, OFB2 +435 vs LDG +54 at M 50; R600 NOT-SHOWN), 3 s is worse (R600b 2S-BETTER, 0/9): the horizon is not the problem. INFLUENCE's onset model applied at every guard step (FBF25 +109) does not beat the old model (FB25 +124; R601 NOT-SHOWN, M 0 0/9 days higher) though it beats LDG at M 0 / 50 (R602 HELPS); STOP-DEPTH's model equals the old one (+120); the influence rule loses (IR -219; R603 HURTS: kept orders run over by Predict-local moves, -0.36 c/sh at the best). Every keep arm turns negative against LDG at M 250 (R604 FLIPS-AT-M250, 0/9) and over all trades (only FBF10 +$10, 7/9). USER's gaps apply: rung 0 blanket-pulled, onset model applied off its population, rungs independent (no shared position) | HOLDS (replay rule M; consumed days; FUTURE-BEST's simplifications) | 20261114_target_test_results.md; 54408bcd |
| The guard ladder with both sides in one lane, a shared position and a cap = K x order size (JOINT-LADDER, 9 days, first minute, WS $/day w/o young at 1 s): keeping (FB25) still beats pulling everything (LDG) at M 0 / 50 at every cap (R611 PAYS: +$41 a day at v3's soft cap 2 x size, +$66 uncapped, at M 50; 7-9 of 9 days) and loses at M 250 (0/9). With perfect foresight of the 2 s move keeping beats LDG by $240-310 a day at every cap AND every fill clock (9/9): the strategy holds under shared inventory; prediction is the limit. The cap at 2 x size costs $22-92 a day at M 0 / 50 (helps at M 250), binds 25-30% of steps, and sweeps overshoot it to ~2x (p95 120 sh for a 60-sh cap). Over all trades and with fills held to 60 s both LDG and FB25 lose (FB25 cap 2: -$167, -$201); only the oracle is positive. Shared legs and price collisions cost the oracle ~$70 a day vs the independent rungs; LDG / FB25 about unchanged. Controls: no-cap parity on non-interacting window-sides (897) and, post hoc, independent-mode parity on all 15,552 checks of 3 days (0 differ) | QUALIFIED by JOINT-LADDER v3: decisions re-run on client-known state move LDG / FB25 by under $1 a day at M 0 / 50 and the oracle by -$3 to -$8 (R615), so the WS numbers stand; but over all trades FB25 is below LDG at M 50 on 7 of 9 days (post hoc PH1) and the WS gain sits in 10-23..10-27 (PH3); a hard cap that counts every resting share is much tighter than this soft cap (replay rule M; consumed days; settlement one outcome per window) | 20261114_joint_ladder_results.md; 0ded5c49 / 879fe15a / 0e60b8e4; 20261114_joint_ladder_v3_results.md |
| Probability and severity of harm per resting order after the cancel could land (ORDER-HARM v2, after the USER's review voided v1: look-ahead in queue inputs, summed overlapping labels, unequal branch ends; 2.29 M decision states of the guard ladder every 250 ms while the guard is on, rung 0 included, M 50): within 2 s a kept order is harmed 18.9% of the time (rung 0 32%, rung 5 11%), -3.1 c per share when harmed; keep minus pull-and-stay-out is negative at every rung (-0.68 rung 0 to -0.21 rung 5; -0.40 all) and worse the longer the guard has been on. The harm is predictable from known information (P(harm) AUC 0.753, 9/9 days >= 0.73, R621 PREDICTABLE; DELTA rank correlation +0.151, 9/9, R620 SKILL; fair-price inputs among the top), but even the best predicted decile realizes -0.06: in this replay no predicted state makes keeping through a signal pay on average. The label keeps blind for 2 s (an adaptive policy can leave later); the economic answer is the policy replay on the repaired joint engine. Controls: future-perturbation falsifier 114,383 rows 0 differ; reference path = sim_arm; censoring | QUALIFIED (USER review 2026-11-14): its keep branch placed replacements during the guard (P23) and kept blind for 2 s while the policy reconsiders every 250 ms; its models are leave-one-day-out. For the keep / pull decision it is superseded by ORDER-HARM v3 (written, not run). The harm statistics stand as descriptions (replay rule M; overlapping rows: per-day diagnostics only; consumed days) | 20261114_order_harm_v2_results.md; abf5638e |
| The joint ladder on client-known decisions with hard pre-trade caps (JOINT-LADDER v3, after the USER's review voided v2: its hard cap released capacity on a refusal 40 ms before it was known; 9 days, first minute, WS $/day w/o young at 1 s): no decision changes when only future venue events change (6,912 checks, 9.8 M decisions, 0 differ; five planted leaks caught). A hard cap counts every resting share, so 2 x size allows two 30-sh orders a side (LDG +$14 vs +$32 under the soft cap at M 50); measured post hoc, true exposure never passes a hard cap (0 of 69,984 runs; soft 2 reached 420 sh for a 60-sh cap) and reaches it briefly in most windows even at 10 x. Keeping (FB25) beats LDG at M 0 / 50 where the cap has room (R613 PAYS at hard 5 / 10: +$33 / +$54 at M 50, 7/9; +$46 / +$73 at M 0, 9/9; NOT-SHOWN at hard 2) and loses at M 250 (0/9); the oracle pays at every cap and clock (+$61 to +$289 a day, 9/9). Post hoc: over all trades FB25 is below LDG at M 50 on 7 of 9 days (hard 10 -$77 a day; on E2-timed trades below on 9/9 days at every cap and clock), and the WS gains sit in 10-23..10-27 (FB25 hard 10 +$164 vs +$20 a day on 10-31..11-03; LDG +$80 vs +$4.5). USER: the real cap will be much larger (multiple ladders) | HOLDS (replay rule M; consumed days; post hoc readings labelled) | 20261114_joint_ladder_v3_results.md; eb735795 / 273bbe8f / 87ebe59c |
| Our current ladder against the top makers on the same trades (TOP-COMPARE, post hoc, descriptive; 9 days, first minute; ours simulated JOINT-LADDER v3 hard 10 / none without young fills, theirs real): the top 5 fill 851,396 sh a day (170,279 per maker) at +0.424 c/sh at 1 s (+0.582 WS-timed), +$3,610 a day; our ladder at M 50 fills 36,387 sh at -0.278 (+0.226 WS) for LDG and 51,588 at -0.346 (+0.351 WS) for FB25. On WS-timed trades our per-share value matches theirs behind the best (1 tick: LDG +0.522 vs +0.559; 2+: +1.414 vs +1.254) but not at the best (+0.004 vs +0.378; their at-best fills 82% STOP, ours 37% in BEST-GAP); over all trades ours turn negative while theirs stay +0.424 (the E2-timed trades). $2,607 of the top 5's $3,471 WS dollars a day come while our guard ladder rests (+0.901 c/sh); while it pulls they still make +0.281 on half their volume, where the rest of the market loses -0.697 | HOLDS (descriptive; ours simulated, rule M, C4 / C9; consumed days; first minute) | 20261114_top_compare_results.md; 6cf0d150 |
| Keep now with the option to cancel later, learned per order (ORDER-HARM v3: 50 ms labels on the client-known lane, policy iteration on a nested out-of-fold chain, chronological folds; ORDER-HARM POLICY v2: the joint ladder replayed with a scheduled re-post at the first step after the guard clears; 5 consumed test days, first minute): the model ranks (R661 SKILL: IC +0.137 pooled, > 0 on 5 / 5 days) but is NOT ACCEPTED on the whole portfolio (R652 REJECT for both arms at every cap). Keep iff A_hat > 0 (OHQ0) beats FB25 at M 50 / 250 (+$100 to +$527 a day, >= 4 / 5 days) but not at M 0 (3 / 5), and loses to pulling everything (LDG) at M 0 w/o young (-$41, 1 / 5) and at M 250 (-$482 to -$667, 0 / 5); at FB25's keep share the model picks worse orders than FB25 (OHQM fails 12 / 12). More good at-best fills (R655 MORE, +$9-16 a day at M 50) but fewer onset orders kept through the guard (R654 FEWER) and less time at the best than LDG (0.34 vs 0.42 of 50 ms steps): R656 NOT-YET. Every non-ceiling arm loses on the whole portfolio at M 50 (LDG -$172 a day) and heavily at M 250; the realized-move ceiling makes +$358 at M 50. Keep share follows the level and the cushion in front (size at better prices); the volatility tercile moves it by <= 0.07; the policy iteration is not converged (21% of decisions flip at iteration 3) | NOT ACCEPTED (replay rule M, C4 / C9; consumed days; development only) | 20261114_order_harm_v3_policy_v2_results.md; c5f4557c / 9ee5cc1b |
| Why the keep-vs-cancel model loses, against the top makers (ORDER-HARM WHY, POST HOC, descriptive; POLICY v2's 5 test days, hard 50; our fills reconcile row for row with POLICY v2; the top makers real): (1) at M 50 the model swaps losses against LDG - it saves $99 a day on orders re-posted right after the guard clears and loses $93 on orders it keeps through the guard (net +$6); at M 250 kept orders lose $1,358 a day against a +$929 re-post saving; the kept-order loss sits on E2-timed trades (M 50 KEPT-ON: E2 -$116, WS +$23 a day). (2) Not a timing-of-presence problem: against the top 5 our state mix (our guard phase x fair-price move x depth) is BETTER at M 0 / 50 (filled less while the guard is on: 26% vs 48% of WS shares) and the whole gap is inside the same state (W2 WITHIN-STATE 24 / 24); a third to a half of it is fewer STOP fills (queue place: at the best 80-85% of their shares are STOP vs our 14-62%), the rest same-type fills that are still worse (WT SAME-TYPE-WORSE 11 / 12); the gap grows steeply with our reaction delay (LDG WS +0.37 / +0.18 / -1.96 c/sh at M 0 / 50 / 250 vs their +0.55). (3) On its labels the model's keeps beat pull-and-re-post on 5 / 5 days (LABEL-OK) but both branches lose (the pull branch -0.25 c/sh): G > 0 means the lesser loss; it keeps through visible moves when the cushion is thick (MISSES-MOVES 4 / 5 days; SEEN -$34 a day at M 50) and cannot see moves after its decision (LATER -$34 at M 50, -$1,101 at M 250) | HOLDS (post hoc, descriptive; replay rule M, C4 / C9; consumed days; E2 anchors +-250 ms, C12) | 20261114_order_harm_why_results.md; 38995519 / a83af1f2 |
| Are the pull models aligned with our goal (MODEL AUDIT, POST HOC, descriptive; 9 consumed days, first 60 s)? The training procedure is sound, the TARGETS are not: (1) the guard (pm_ladder_v1, frozen) predicts only the 1 s markout of being swept through the best within 500 ms (88.6% of its targets exactly 0, 9.2% negative, 2.2% positive; mean -0.226 c/sh), with no credit for fills that stop at our price or for queue place, and pulls on a QUANTILE line (c60 - 0.10: predicted sweep cost <= -0.186 c/sh), not a break-even: it is on in 33.9% of 50 ms steps, 54.5 alarms per side-minute, on spells median 100 ms, off 150 ms, and every alarm pulls every rung (G1; G2 label NOT-COST-ONLY by its declared rule); (2) ORDER-HARM v3's label sees no cost of pulling in most episodes: keep-all = pull within the horizon in 66-87% of rung-episodes by rung (rungs 2-5 84.8%, K1 FREE-PULLS; 55-77% with a 5 s tail), 94.6% of its pull decisions have realized G = 0 (K3), 39% of pulls sit within 0.01 c/sh of the threshold, and an order kept at the onset meets a later pull in 42% / 66% of 0.35-1 s / > 1 s episodes (K2; flip rate 8.1%, STABLE by its rule): neither target contains the value of the queue place | HOLDS (post hoc; code + consumed days) | 20261114_model_audit_results.md; a1b88d8b |
| The keep / cancel model for every ladder order (STOP-MODEL, the USER's stop-price design: per side and level, is the order run over within H after a cancel could land; every 50 ms step, 9 consumed days, chronological folds, test 10-27 / 10-31 / 11-01..03; 85 known-time inputs, book sizes relative to a causal liquidity scale): predictable far above the base rate at every horizon (j1 AUC 0.91 / 0.85 / 0.75 at H 72 / 250 / 1000 ms, Brier skill +0.16 / +0.24 / +0.17, 5 / 5 days; j2-j3 higher), better than the guard's value at every horizon (5 / 5) and equal to STOP-DEPTH refit chronologically on its guard-on rows (better at j2); at the guard's keep count fewer kept orders are run over (rungs 0-4 at 1 s). The target is worth acting on (R701 oracle replay, whole portfolio, w/o young fills): acting on the true 1 s label turns the always-resting 6 x 30 sh ladder from -$5.9k to +$0.8k a day at M 50 (+$0.8k at M 0, +$0.6k at M 250), 5 / 5 days at every clock; the time-shifted control recovers ~1/7 of the gain; shorter targets are worth less (250 ms ~0, 72 ms still negative). The model's top-risk decile marks our orders' harm (V -0.32 vs -0.05 c per order share at M 50, 5 / 5). Short-horizon skill is mostly Predict-local (thin best level; post hoc). VERDICT MODEL-READY at every H | QUALIFIED: consumed days (test-day statistics seen once before the processing choices, B4); replay rule M; oracle gain not split WS / E2 (C12); the oracle is a ceiling; Binance at event + 3.5 ms (P15); 1 s j1 over-predicts 1-4 pp; no policy run yet (STOP-POLICY next) | 20261114_stop_model_results.md; 63a42519 / 52f2ce98 / 8f0b24ee / e69dbf82 |
| STOP-MODEL v4 after the USER's model / data check and input review (2026-11-15; decisions on the dev days 10-23..26; 5 chronological test days): trained and scored on every decision state (v3.1 excluded the ~4% already moving through level 1 and there predicted half the 72 ms risk), 180k rows a day, lambda_l2 = 10 (the pinned parameters overfit the rare deep heads), level projection, 61 inputs (exact duplicates and 20 within-side raw copies dropped; nothing normalized). On every usable row: j1 AUC 0.929 / 0.883 / 0.774 (H 72 / 250 / 1000 ms), Brier skill +0.34 / +0.35 / +0.23; log loss vs v3.1 -11% / -1.2% / -0.1% at j1 and -45% / -10% / -1% at j6; better than the guard and than STOP-DEPTH refit chronologically at j1-j3. Inputs: step-5 inputs recomputed from raw data known by t match 100% (planted look-aheads fail); every input group needed (Binance 27-35% of SHAP, book sizes 19-31%, Predict state 16-23%; removing any worsens all 9 tested heads); imbalance carries the model, flows are weak; VERDICT MODEL-READY | QUALIFIED: consumed days; H1000 j1 over-predicts up to 4.5 pp; lambda_l2 changes the pinned parameters; extra order-flow inputs (Predict OFI, Binance depth) under test; no policy run yet (STOP-PLAY queued) | 20261115_stop_model_input_review.md; fd7f56c5 .. 9220fefc |
| The harm-only keep rule replayed (STOP-PLAY v6, 2026-11-15; the v6 sweep model's own pull / entry decisions in the joint client lane, 6 rungs x 30 sh a side, (tau_on, r) chosen per fold on its calibration day, 5 chronological test days, whole portfolio): the 1 s model loses on 5 of 5 days - M 50 without young fills -$135 a day with the queue beyond the 5th shown level = 0 and -$115 with it = inf (keep-all -$5,874 / -$5,884, the guard LDG -$173 / -$176, the true-label oracle +$795 / +$710); it beats LDG on 3 of 5 days (R802 NOT-SHOWN); M 250 -$1,306 / -$1,303 (LDG -$1,710). The unknown-queue bound changes little. The rule is not selective: at the guard's pull share it avoids 72% of the harm and gives up 75% of the good fill value; surviving orders earn +0.007 c per order share against -1.29 c when run over | HOLDS (old queue initialization with the next-frame uplift; consumed days; replay rule M; the design is superseded by STOP-EV) | reviews/stop_model_20261114/stop_play_v6*.txt; harm-capture check (design section 1) |
| STOP-EV stage 1 data (2026-11-15; 9 consumed days, first 60 s; labels through the Codex accounting contract): every data check passes on 9 / 9 days - keys = the STOP-MODEL rows, every first-minute leg used or untimed (0 unaccounted), types / visible sizes / marks / components 0 mismatches against raw-file recomputations, the single-order rule equals the corrected lane on 5.12M clean order-steps, 72k eligible actions recompute exactly. The label population is set by timing: an E2 leg near the path refuses 99.5% of probes under ws_strict (969 of 187,384 dev probes eligible), the whole-window untimed rule refuses half of all probes (53-85% of windows hold an untimed leg); all_anchors_sensitivity keeps 16%. Over 1 s a 30 sh order loses on average at every rung (all_anchors dev mean -2.45 c per action, 88.9% zero; losses from through-price fills) and even a front order at the best loses (-4.9 c; STOP legs at the best +0.14 c/sh, sweeps -2.35); a third of front fills are E2-anchored | SUPERSEDED (REVIEW 2026-11-15: the counts are pre-rebuild - after the untimed-rule rebuild 68% of probes are eligible; that rule and the strict-WS / certificate populations are selected by trades after the decision, so their means describe quieter states; '0 unaccounted' and '5.12M clean steps' are near-vacuous as cited) | reviews/stop_ev_20261115/stop_ev_data.txt, untimed_rule_counts.json |
| The STOP-EV keep model (2026-11-15; hurdle P(fill) x E[value | fill], Huber; v6's 71 market inputs at the action's own step + client-known order inputs; one probe per neutral order generation, E2 point times, the USER's untimed rule; 5 chronological test days, 137,703 actions; value = the order's own fills in [t + 22 ms, t + 1 s] at fill time + 1 s): it ranks - realized mean by prediction quintile -10.4 / -3.6 / -1.8 / -0.8 / -0.5 c per 30 sh order, IC +0.134 (> 0 on 5 / 5 days, both sides) against +0.087 for the order-state-only baseline (above it on 5 / 5) - but no state is worth keeping at 1 s: every prediction decile loses on average (the top decile realizes -0.53 c), so keep-iff-positive pulls 93% of orders (local gain +$4,628 over 5 days vs keep-all, matched null p 0.002) and its decisions are not better than the order-state baseline's (3 / 5 days). On the strictly WS-timed test actions (4,572) the mean value is -0.12 c and the IC -0.009: the losses and the skill sit in windows with E2-timed trades | SUPERSEDED by the v3 row below (NOT-READY on the corrected labels); QUALIFIED (REVIEW 2026-11-15: R903 cannot fail when every day loses - pulling EVERYTHING beats the model's keep rule, v1 $4,683 vs $4,628, v2 $2,719 vs $2,602; the 'strictly WS-timed' subset is selected by later trades; v1's T5 FAILED as declared; MODEL-READY is not supported) | reviews/stop_ev_model_20261115/stop_ev_model.txt |
| Do the STOP-EV losses on E2-timed trades depend on their assumed time (E2SEP v1, 2026-11-15; stage 1's labels rebuilt with every E2-anchored leg shifted -250 / 0 / +250 ms, the zero shift reproducing stage 1 exactly on 9 / 9 days)? Strongly: actions reached by E2-timed trades average -26.3 / -19.5 / -8.2 c per 30 sh order over 1 s (9 / 9 days the same order), versus -2.5 / -1.1 / -2.9 for the future-WS-reached group; at equal reach activity E2-reached actions lose 10.9 c more (9 / 9 days) | QUALIFIED (the groups follow only the order's future window, P39, and each shift's population is rebuilt - common-offset scenario means, not fixed identities; superseded by E2SEP v2's certificate groups when it runs) | reviews/stop_ev_e2sep_20261115/e2sep.txt |
| Is the replay's loss on E2-timed trades made by the E2 timing method (E2IZE, 2026-11-15; every WS-timed trade re-timed by its own E2 book-time estimate, stage 1's labels rebuilt, fixed identities)? No: on the certified timing-invariant actions (WS times only) the value moves +0.175 -> +0.161 c per 30 sh order over 1 s (-0.01, mixed signs over 9 days: E2-TIMING-NEUTRAL), and the E2 estimator is unbiased at the median on 1.01M WS-timed cells (median error 0 ms; p10 / p90 about -400 / +300 ms; 8-14% later and 12-16% earlier than 250 ms). The losses sit in states that hold E2-timed trades, which lose 13.8 c more than timing-invariant ones at equal reach activity: real adverse flow where the trade feed misses events, within the replay's assumptions | QUALIFIED (REVIEW 2026-11-15: the invariant group is certificate-selected using trades after the decision - quiet states by construction; low power: only 2.2% of them change at all; WS-timed trades may differ in kind) | reviews/stop_ev_e2ize_20261115/e2ize.txt, reviews/stop_ev_e2sep_20261115/e2sep_v2.txt |
| "Rest when quiet, pull when the fill model sees activity" (STOP-EV-PLAY, 2026-11-15; the v2 fill head - the model's real skill, AUC 0.82-0.86 - on the lane's own client-known order state pulls a resting order when P(fill) > theta and blocks re-entry while a fresh order's P(fill) >= r x theta, every 250 ms; corrected queue lane for every arm; 6 rungs x 30 sh; theta chosen on each calibration day; 5 chronological test days, whole portfolio): it beats keep-all (M 50 w/o young -$1,371 vs -$5,812 a day) and its shifted-market control (-$3,786: INFORMED), but LOSES to the guard (-$144) and to the harm-only sweep model (-$129) on 5 / 5 days at M 50 and at M 250 (-$3,218 vs -$1,673 / -$1,303). Every fold chose the most aggressive threshold of the grid (theta 0.10): quoting less is better. An activity signal pulls good fills with bad ones, and its 72.7k pulls a day re-post young orders that lose $1,188 a day on fills < 100 ms old; the sweep model targets the loss itself (the level being run over) and keeps rung 0 mostly out (uptime 0.09 vs QR 0.30) | SUPERSEDED by the STOP-EV-PLAY v2 row below (NOT-PROFITABLE); VOID AS RUN (REVIEW 2026-11-15: QR's re-entry gate was stale - every pull was re-posted within ~150 ms without the declared check, so the churn and young-fill losses are the defect; QRS fed the model the previous window's mid and scale in the ORDER inputs, so R983 is not an information test; QR decided every 250 ms against 50 ms incumbents; theta at the grid floor on 5/5 folds; and the $ ranking is the inverse of the volume ranking - per share every arm loses) | reviews/stop_ev_play_20261115/stop_ev_play.txt |
| REVIEW of the STOP-EV pipeline (2026-11-15, four independent read-only reviews: labels, model, replay, premises; USER: "Review the model thoroughly, from data to model, check if there are bugs first then check how to optimize"): no leak, no join or arithmetic error; label clocks, signs, queue rule, provenance, client-known inputs, folds and the evaluation arithmetic are sound; marking at the 60.2 s mid is sound (settlement - mid -0.35 +- 1.21 c, unpredictable); the exact-content duplicate legs are distinct on-chain fills (one maker's equal clips; already in step 1's (transaction, maker order hash)-keyed table). DEFECTS: (1) outcome-selected populations - the untimed rule refuses probes whose own next second holds an untimed trade at their price (the refused 2% carry 13-24% of the day's keep-all loss), the certificate and strict-WS subsets drop states by later trades; (2) the QR replay's stale re-entry gate, corrupted QRS order inputs, cadence mismatch, theta grid floor; (3) readings that cannot fail (R903, R952v2's sign rule, '0 unaccounted') or compare unlike pull shares (R904), a T5v2 gate that cannot see a feature leak and centres on exactly 0. PREMISES that can flip the best arms' sign: $ a day ranks arms by how little they trade (per share every arm loses); replay latency 40 / 22 ms vs measured medians 58 / 37 ms; unknown queue = 0 (29% of MD1000's filled shares); queue rule M vs U; cancel-pending fills ~2.5x live; missed fills from orders that would cross ours (cancelled as 'cross'); liquidity rewards excluded (100 sh minimum: a 30 sh ladder never qualifies) | DEFECTS CONFIRMED; premises open | handoffs (HANDOFF 2026-11-15 17:00Z); reviewer check scripts in the session scratchpad |
| STOP-EV labels without outcome selection (2026-11-15, REVIEW fix; USER: "Can we fix the critical issues and let the results be correct first"): untimed legs are imputed into the replay tape at their taker group's earliest exec second + the day's median anchor offset (E2-class, +-1500 ms) instead of refusing the probes they could reach (that refusal was selected by the probes' own next-second fills). 9 / 9 days pass E1-E9; 16,154 imputed legs, anchors equal an independent recomputation; 270,767 eligible probes (all_anchors_sensitivity; exclusions: not resting 114,547, feed-gap coverage 4,655, not reached 3,789, stale mark 6) | HOLDS (data; consumed days; imputed times are E2-class estimates) | reviews/stop_ev_20261115/stop_ev_data_imp.txt |
| The STOP-EV keep model v3 on those labels (2026-11-15; hurdle P(fill) x E[value | fill] by squared error; v6's 71 market + 12 client-known order inputs; every eligible status; 5 chronological test days, 140,487 test actions, mean -4.10 c per 30-sh order over 1 s): it RANKS - IC +0.138 (order-only B2 +0.087), > 0 on 5 / 5 days (day-cluster 95% +0.105..+0.162), within-window permutation p 0.001, both sides, quintiles -12.67 / -4.06 / -2.36 / -1.15 / -0.28 c - but no predicted subset earns: keep iff pred > 0 keeps 9.9% and loses $46 over 5 days (-0.33 c per kept order; random keeps of the same count lose $483, p 0.001; B2 at the same count worse on 4 / 5 days); the top decile is predicted +0.93 c and realizes -0.33 c | NOT-READY (R903' no positive keep, R905 miscalibrated; development only: consumed days, replay rule M, a local 1 s value under a neutral continuation) | reviews/stop_ev_model_20261115/stop_ev_model_v3.txt |
| STOP-EV-PLAY v2: "rest when quiet, pull when the fill model sees activity" (QR) replayed after the REVIEW fixes (2026-11-15; v3's fill head on the lane's client-known order state; decisions every 250 ms, a pulled rung empty until the next decision, every rung's entry gate fresh at each decision; corrected queue lane, 6 rungs x 30 sh, the imputed-untimed tape; theta 0.02-0.30 and MD1000 re-chosen on each calibration day; 5 chronological test days, whole portfolio w/o young fills, $ a day M 50 / M 250, c / share at M 50): NO arm is profitable - QR -186 / -516 (-2.40 c/sh on 7.8k sh a day), keep-all -5,845 / -7,260 (-1.78 c), the guard LDG -179 / -1,713 (-0.40 c on 44k), harm-only MD1000 -141 / -1,364 (-0.92 c on 15k), random pulls at QR's rates RV -1,186 / -1,541, shifted-market QRS -622 / -823; at QR's 250 ms cadence the incumbents lose more (LDG250 -1,113, MD250 -708). QR beats keep-all, QRS (INFORMED) and RV, but not LDG (-$7.5, 2 / 5 days) or MD1000 (-$45, 2 / 5) at M 50; at M 250 it beats both on 5 / 5 days because it barely quotes. Its theta sits at the grid floor on 5 / 5 folds with the calibration $ monotone (less resting is always better: the family's limit is not quoting); it never quotes rungs 0-1, so its smaller loss is volume, not better fills (the worst arm per share). LDG earns on WS-timed trades (+$33 a day) and loses on E2-timed ones (-$211). Measured latency 58 / 37 ms worsens every arm; unknown queue = inf changes little | NOT-PROFITABLE (R985); R980 BEATS-KEEP-ALL, R981 / R982 NOT-SHOWN, R983 INFORMED, R984 BEATS-RANDOM QUALIFIED (RV matched per window and side, not per rung - QR sits on deep rungs, RV evenly - so it can be placement, not timing: P43); development only: consumed days, replay rule M, E2 point anchors, imputed untimed times | reviews/stop_ev_play2_20261115/stop_ev_play2.txt |
| The move after a fill at the best: BTC explains a third (out-of-day R^2 Binance 0.34, other venues 0.38) but none of the top makers' edge (2-3%); their edge (+0.21 c at 250 ms) is that fewer of their fills are in trades that take the whole level (57% goes with the book's reaction); in fills that leave the level standing there is no gap (-0.005 c) | HOLDS (descriptive; WS-timed fills only) | MOVE Part A |
| Lane J-v3 (v3's quote line only, NOT the frozen pm_ladder v3, C11) is the only positive simple replay arm (+1.03 c/sh, $14.7 a day at M 50 on the 9 days). The frozen v3 is under a forward test with its own engine | QUALIFIED: 75% of J-v3's value is young-order fills (AGE); every "v3" in S4 / QR / QR2 / AGE / GAP / CS means J-v3 | S4, QR2, AGE, pipeline review S1 |
| The top makers' edge is not in our guard's states | HOLDS | B12 |
| A pressure-feature guard does not change the guard | HOLDS | B14 |
| A queue-aware guard target equals the live guard; adding pressure is better at 1 s, worse at 60 s; not promoted | HOLDS | B16 |
| The guard is on (stay out) in ~90% of resting moments; 94% of its alarms see no harm within its own 500 ms horizon, but after those "false" alarms a resting order loses -1.7 to -1.9 c per moment over the next 4.5 s (9 of 9 days), twice the loss after no-alarm moments: the alarms are mostly early, not false. It separates activity (AUC 0.67-0.72) better than harm (0.60-0.65) | HOLDS (replay, never-pull path, orders >= 100 ms old; the AUC result is post hoc after an instrument fix) | FA, 20261112_false_alarm_results.md |

### C. Resting by default and pulling

| Finding | Status | Source |
|---|---|---|
| Orders held in advance gain no priority (live ~82 ms with the guard, 288-355 sh ahead) | HOLDS | B20 |
| Resting at the touch by default and pulling on a signal loses at every fill clock (eight arms) | HOLDS (11-08) | B21 |
| Zero own latency does not rescue resting; it helps the gated v3 (+0.25 c/sh) | QUALIFIED: the v3 gain is young-order exposed (C1) | B21b |
| A perfect keep / pull decision makes resting pay on 11-08 | HOLDS as a perfect-foresight ceiling only | S1 |
| A learned keep / pull model has skill (9 of 9 days) but its policy loses to the guard trigger and v3; the label is mostly re-entry luck | HOLDS as a negative; its KP numbers are biased in KP's favour (start price before known, P16; fold leakage through v; no decision during the unknown-fill window) | S4, S4 diagnostic, pipeline review F1 / F3 / S3 |
| Default-pull loop (Codex): no candidate keeps trading value at no higher inventory; a learned pull beats random but not the guard margin; a joint fill model improves calibration, not selection | HOLDS | 20261112_default_pull_review.md, 20261112_fill_value_review.md, 20261112_joint_fill_review.md |
| 250 ms danger-pull design | OPEN, never run | bbf0112d |

### D. Queue priority and speed

| Finding | Status | Source |
|---|---|---|
| A new best level holds ~52 sh at birth and ~155 sh 100 ms later; no tested rule buys priority at our latency | HOLDS (11-08) | review 6.5 |
| The queue ahead of our orders mostly joins after we could have been there (v3 41%, live 44%); latency 16% / 27%, re-places 15% / 14% | QUALIFIED: a model-based split (fixed 97 ms boundary, ghost queues, snapshot allowance) | QR parts A / C; review 10:42Z |
| v3's fills on young levels / first orders earn more than on old levels / re-places (+1.59 / +1.16 against about 0, 9 of 9 days) | QUALIFIED: young levels mean young orders (C1); not re-checked by level age | QR |
| QR speed sweep at d = 0 / 15 / 30 ms | **VOID**: the decision is scheduled before the level exists (C6) | QR part B; review 10:42Z |
| QR speed sweep at d = 55 / 97 / 150 ms | QUALIFIED: causal (55 ms assumes the 15.2 ms minimum feed; its decision is 0.2 ms early, immaterial); young orders (C1) | QR part B |
| Joining earlier at today's speed (no old levels, keep through flicker) does not beat v3's $/day | HOLDS | QR2 R290 |
| Keeping orders through guard flicker is worse per share | HOLDS | QR2 K / K2 |
| The 500 ms age cut: same $/day, better per share, less M 250 loss | QUALIFIED: its gain is young-only (AGE R312 GAIN-IS-YOUNG) | QR2, AGE |
| v3 + joining new levels at 30 ms (F30, Y+F30, QR2-fast) beats v3 (+8.7 $/day; +2.6 without young fills) | **VOID** (C6); SUPERSEDED by CS's causal speed test | QR2, AGE R312 / R313; CS |
| With causal clocks, v3 plus event-driven joins at new levels gains only with a faster order path: 10-20 ms paths +1.5 to +4.4 $/day without young fills (M 50, 8-9 of 9 days); a faster feed with today's 40 ms path gains nothing; slowest gaining arm = today's feed + 20 ms path; at M 250 everything still loses; joining at births beats random times | HOLDS (replay, on J-v3 not the frozen v3 (C11); young fills about 70% of the headline; mechanism not established; pending-cancel window ruled out post hoc; own-order overlap fills (P13) are 1-7% of shares and excluding them shrinks the gains by up to 14%, still 8-9 of 9 days) | CS, 20261112_causal_speed_results.md, pf_cs_overlap.py |
| The live engine churns: at-best orders cancelled 23-38 ms after going live, 82.5% re-placed; guard flicker | HOLDS | B13c / B13d |
| The live soft pull is a churn control with value about unchanged; merged (pmmmrust 0c3b115), not deployed | HOLDS | B15 |

### E. The top makers

| Finding | Status | Source |
|---|---|---|
| Their edge is not queue place, not Binance and not trend. It is earned within 250 ms of the trade report, on Predict.fun itself; they stay after being hit; about a quarter is their own taking; it holds to 60 s and settlement | HOLDS (mostly 11-08; B12g 9 days); B12g's settlement values carry the tie error (P10, 14 windows) | B12-B12g, B13b, B18 |
| The top makers' exits before level-taking trades (EXIT, public tape, 9 days, first minute; front orders proven by a sole-maker stop <= 1 s before): they do not leave more often than other makers (GONE 51% vs 46-62%; gap-matched -0.8 points, R390 NOT-SHOWN) but they leave at the right moments - GONE 28% -> 80% from the bottom to the top quintile of our P against 42% -> 63% for the field (9 / 9 days, every gap bin); the choice is made 100-1000 ms before the trade, not in the last 100 ms; they are 42% of first-in-queue volume against 34% of level-taken volume, 52 sh per leg when a level is taken while they are in it | HOLDS (public tape, consumed days; R392 post hoc robust) | 20261113_exit_results.md; 11fb2adc / 5abc1d96 |
| How the top makers work, order by order (ORDERS: raw match records' order hashes joined to step 1's legs, 9 days, first minute): they rest larger orders and keep them through fills - 25.8% of their orders filled by >= 3 trades vs 10.0% (R400, 9 / 9 days), 15.7% first in the queue at >= 2 trades vs 4.9% (R401, 9 / 9), median 30 sh filled per order (p90 100) vs 20 / 10; they do NOT create new levels more than their share (20.7% of creators vs 24.4% of at-best fills, R402 PROPORTIONAL); 6 prices per side per minute (p90 11), two-sided in 89% of windows, end |net| / gross 27%; their takes (12.6% of maker volume) mostly add to their position (R404 DIRECTIONAL at the line, 39.6% reducing). EXIT holds on exact, complete taker orders (EXIT-v2: 29% -> 79% vs 44% -> 64%, 9 / 9) | HOLDS (public tape, consumed days; sizes and lives are lower bounds from fills) | 20261113_orders_results.md; 9f8ec338 |
| Per share, our simulated J-v3 beats their real at-best fills (+1.03 against +0.23) | QUALIFIED by AGE: MIXED without young fills (8 / 6 of 9 days); J-v3, not the frozen v3 (C11) | GAP, AGE R314 |
| They trade 345x our shares; 60% of their money is made while v3's guard is OFF (+0.15 c/sh on 459k sh a day); only 17% of our fills share a taker order with them | HOLDS (theirs real) | GAP |

### F. Live and execution

| Finding | Status | Source |
|---|---|---|
| Live is not running (USER 2026-11-10); live = the user's call | HOLDS | HANDOFF |
| Slow cancels (0.5-5 s) 50-60 s into the window caused 11-08's first-minute loss | HOLDS; handed to the live session | B12b |
| Maker fee 0 in public match records; rebate unverified | OPEN | STATUS maker_fee_public, rebate |

### G. Pipeline review (2026-11-12)

| Finding | Status | Source |
|---|---|---|
| Three independent stage reviews (data / time alignment, features / guard, simulator / valuation): two ERRORs (tie outcomes P10; S4's `gap` before the start price is known P16) and the framing risk that "v3" in the replays is J-v3 (C11); the rest are RISK / NOTE items now in §2 | HOLDS; fixes proposed (user's call) | 20261112_pipeline_review.md |

## 4. Only live can answer

- Whether our young orders really fill, and at what value (C1).
- The fill clock: when trades match relative to their report (C2).
- Our real place in the queue (C5).
- All three need a live run with the A4b capture (USER D2).
