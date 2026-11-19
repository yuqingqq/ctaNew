# First-day pipeline review — 2026-11-19

**Not ready to accept as a correct full pipeline.** The immediate blocker is a
reproduced future-information dependency in POST inputs. Two additional checks
can accept forecasts that are unsuitable for the controller. This review does
not establish that a fitted model loses money: no completed benchmark or policy
result was available at the final source check, **04:47:42 UTC**.

Reviewed exporter commit: **8e57fd8969ed6a57194b0487c3fb51b584a6e36b**.
The exporter was committed during this review, byte-identical to the tested
snapshot. The benchmark, policy and scenario integration were work in progress:
the exact **04:42:06 UTC** snapshot is recorded in [snapshot.json](snapshot.json)
and [snapshot.patch](snapshot.patch), based on full commit
**e96321df503662ae5b15844a5a090c1df2242043** in an isolated checkout. This is not
a freeze of a committed complete pipeline.

[Probe](probe.py), [receipt](receipt.json), [selftests](selftests.txt), and
[source follow-up](source_followup.json). Source line references below refer to
that snapshot. Later changes to benchmark memory handling and policy output
labels are retained in [source_followup.patch](source_followup.patch); the
functions supporting findings 2–3 are unchanged, verified by AST comparison.
The new memory path and revised policy selftest were not rerun. No development
code or other-session artifacts were modified by this review.

## Findings, ranked by severity

### 1. P1 — POST queue inputs reveal whether a future update arrives before activation

`predictfun/research/pf_pipeline_export.py:97` locates the frame at activation,
which is 58 ms after submission. Lines 104–112 use whether that frame is already
at or before the decision to choose queue values versus missing values.
`order_block:188` applies this to the POST row at submission as well as the
later RETAIN rows. Consequently the POST row reveals a future arrival event.

**Reproduction:** identical known history at decision 0, identical book values,
activation at +58 ms. Move only the next book receipt from +30 to +80 ms.
Five POST inputs change: `o_q_rel`, `o_q0_rel`, `o_q_frac`, `o_qunk`, `fp_queue`.
The unknown flag changes from 1 to 0. Current shown quantity remains 12, and the
later +100 ms live state agrees. A same-time quantity change is detected as a
positive control. This is a dependency on information unavailable at submission,
even though future price values are not directly copied into the feature.

This reaches the saved data: the hash-matched Oct27 export contains **28,038
eligible POST rows**, of which the unknown flag is 0 on **3,936** and 1 on
**24,102**. These counts establish that the varying input is present; they do
not measure its contribution to a trained model's score.

The existing X5 future perturbation calls only the market-feature builder
(`causal_check:151–157`, invocation at 300); it does not exercise the assembled
POST own-order inputs. `known_parity:203–213` compares against the same
activation-based convention in `AC.known_state`, so agreement cannot establish
causality before activation. Both existing exporter and policy synthetic
selftests pass despite this counterexample.

**Required correction:** define fresh/pending POST state using information
available at the decision, without consulting the eventual activation-frame
index. Test the complete feature vector against future changes in both values
and receipt times/arrivals. Supersede the affected export and any POST fits that
used it. This finding does not by itself invalidate the later live RETAIN
features or the engine-reconciled outcome labels.

### 2. P2 — contract-rejected forecasts still drive actions and the parity gate can pass

`pf_pipeline_policy.py:93–100` catches a contract `ValueError`, increments the
refused count and skips the forecast. `controls:143–145` accepts P0f whenever
the remaining comparable rows agree; it does not require the rejected rows to
be handled consistently. Meanwhile `triggers_of:68–72` uses every raw `J_V` to
choose cancellations, including forecasts the contract rejected.

**Reproduction:** two synthetic RETAIN forecasts, one valid and one implying an
impossible endpoint mid of -150 cents. The contract parity result is **1
compared, 1 refused, 1 agreement**. P0f accepts it, while the trigger builder
pulls **both** orders. Valid-only and invalid-only controls establish that the
contract itself accepts/refuses the expected inputs.

The benchmark can produce incompatible moments: `pf_pipeline_bench.py:126–130`
clips conditional quantity but leaves the separately regressed `E[q*move]`
unconstrained. B5 checks probability and quantity bounds only (194–195).
This review has not measured the occurrence of invalid forecasts in fitted
outputs; it proves that the current validation can miss them.

**Required correction:** validate moment coherence and use the same accepted
forecast/action path for policy generation and parity. A rejected forecast must
have an explicit, accounted fallback or refuse publication; it cannot disappear
from validation while remaining in the executed rule. Include a mixed valid /
invalid batch in the falsifiers, not just an entirely invalid batch.

### 3. P2 — the mapping gate can PASS when every zero-threshold decision has the wrong value sign

`pf_pipeline_bench.py:227–230` gates only on total quantity calibration and
value ranking versus F. It ignores Q2's movement amount/sign check and absolute
value calibration. The policy, however, acts on the **zero crossing** of the
predicted value. Ranking is insensitive to a constant offset.

**Reproduction:** 20 synthetic outcomes with true values from -10 to -1 cents,
correct quantity predictions, and value predictions shifted upward by 20 cents.
Mean prediction is **+14.5 cents**, mean outcome **-5.5 cents**. Value rank is
perfect, above the supplied F control. Both POST and RETAIN mapping gates say
**PASS**, despite Q2 saying `MOVE-TERM-NOT-SHOWN` and every predicted value
having the wrong sign. A quantity-miscalibrated control correctly fails the
existing gate. These are constructed checker cases, not market results.

**Required correction:** distinguish a rank/quantity diagnostic from an
economic-mapping gate. Before attributing controller performance to calibrated
values, require predeclared value-level/sign and loss-amount diagnostics near
the action boundary, with action-level, participation-matched economic controls.
Do not tune those criteria after seeing Oct27 and call Oct27 validation.

## Scope gaps: what the current experiment would actually test

- **Cancellation only.** `triggers_of:70` selects RETAIN rows. Reversing a POST
  forecast's sign leaves the policy unchanged in the independent probe. All
  neutral JOIN opportunities still post; there is no learned POST/WAIT
  admission, re-entry or inventory controller. The header's "whole pipeline"
  description should therefore mean an initial prediction-to-cancellation
  experiment, not the full proposed market-making strategy.
- **The scenario table holds decisions fixed.** `pf_pipeline_policy.py:161–164`
  constructs one baseline prediction/trigger table and reuses it in all 19
  scenarios. The header discloses this limitation. It is a fixed-decision
  execution sensitivity test, not a replay of the same state-dependent model
  under each scenario. Public information receipt times need not move, but
  execution-dependent own-order information can: with a 100 ms notice delay,
  a five-share fill at +280 versus +80 ms produces remaining quantities 30
  versus 25 at decision +300 ms. A causal state-dependent forecast can then
  choose different actions. The probe demonstrates this possibility; it does
  not claim the current learned model actually flips. Recompute scenario-local
  state and predictions before claiming controller robustness across clocks,
  or retain the narrower fixed-decision interpretation.
- **Overlapping RETAIN diagnostics are per decision.** The benchmark explicitly
  discloses this. They are not independent action returns or portfolio P&L.
  The single first-pull replay is the appropriate place to count action value;
  it still lacks sequential repost, inventory management and quote-uptime tests.

## What is aligned, and what was verified

The source has useful foundations: neutral JOIN opportunities, explicit no-fill
and exclusion outcomes, separate POST/RETAIN and bid/ask models, a common
birth+5-second END, latency-dependent removable fills, and the joint
`E[q*move]` mapping rather than a product of marginal means. Chronology is
declared as Oct23–25 training and Oct27 development evaluation, with pinned
parameters and mechanical/fair/direct-value controls. These design choices do
not overcome the causality and validation findings above.

The bounded review ran through **PF_SLOT=codex / run_light.sh**, using absolute
script paths: **70.34 seconds, 728 MiB, exit 0**. It ran the frozen exporter and
policy selftests, five independent counterexamples/scope checks, and a bounded
saved Oct27 POST-column scan. It verified all 12 exported part hashes and the
exporter source identity. No full training, raw tape experiment, benchmark
training selftest, or new untouched day was run.

Saved export as-of **2026-11-19T04:40:00.751625Z**: **29,820 POST rows**,
with statuses 28,038 eligible, 1,300 refused post-only, 460 missing endpoint,
22 stale future book. The receipt records **1,404,872 RETAIN rows** (1,404,540
eligible and 332 feature unavailable); the review's bounded scan independently
recounted POST, not every RETAIN row. The exporter receipt reports X1–X6 passing;
the review did not rerun the five raw-day exports or independently revalue all
labels. Passing those checks is compatible with finding 1 because X5's coverage
is incomplete.

Oct23–27 are consumed development days, not untouched validation. Oct26 is
excluded from this benchmark's fitting/evaluation but was already exported; it
is not globally unread. No confidence interval or profitability claim follows
from this one-day exercise. Queue/fill realism, timing assumptions, cancellation
refusals and the distinction between local END marks and settlement/portfolio
P&L remain limitations.

## Prior findings and one next priority

The previous [score-result review](../20261119_5578d96f/report.md) found a P2
PULL-ALL output-label issue. The later working policy now writes both absolute
PULL-ALL value and improvement, and the other session has edited the variants
formatter similarly. The old result artifacts still require superseding
publication before that finding is closed at the artifact level. Previously
resolved accounting/readiness findings are not reopened by this review; this
is a new feature-time and model/controller validation boundary.

**One next priority: repair and supersede the POST feature exporter before
accepting first-day fitted results.** The acceptance test is that identical
history through submission produces identical complete POST inputs despite
different future update arrivals/values, while a relevant same-time change
still changes the vector. Publish new hashes/receipts and rebuild affected POST
fits. Preserve RETAIN/label evidence separately unless a correction changes
those objects. Findings 2–3 must also be addressed before a mapping or controller
PASS can be treated as an economic gate; the development session owns the fixes.

Review only. No scheduler restart, push, merge, implementation changes or
external messaging.
