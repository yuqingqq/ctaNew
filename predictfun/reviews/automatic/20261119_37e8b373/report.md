# Corrected pipeline results reviewed — 2026-11-19

Target commit: **37e8b373d5ef29beb605573c82d0842dba320a89**, including the
preceding fixes in 54e25790. The pinned committed source was read and exercised
in an isolated checkout. Other-session changes were preserved.

**The saved baseline values reproduce. The original POST look-ahead and the
previous two counterexamples are fixed. The learned cancellation rule still
does not consistently beat first-opportunity PULL-ALL.** A remaining moment
coherence bug occurs in saved predictions but changes no executed trigger or
baseline value on these two days. Two statements in the result report need
narrowing before they guide the next experiment.

[Verification](verify.py), [receipt](receipt.json), [impact follow-up](followup.py),
[follow-up evidence](followup.json), [selftest output](selftests.txt).

## Findings, ranked by severity

### 1. P2 — zero expected quantity can still carry nonzero expected fill value

`predictfun/research/pf_pipeline_bench.py:133–140` clips conditional quantity to
zero but leaves the separately regressed `Eqm` unchanged. When `Eq == 0`, the
implied movement is set to zero and the forecast can be marked `ok=True` even
though `Eqm != 0`. For nonnegative fill quantity, zero expected quantity implies
zero quantity-weighted movement. These moments cannot describe a coherent joint
distribution.

`pf_pipeline_policy.py:105–109` similarly converts such moments to a zero-size,
zero-movement atom, losing the original nonzero value. Its controller action can
then disagree with the raw-value rule. The original impossible-endpoint example
is fixed, but this boundary remains uncovered by the sampled parity gate.

- Independent synthetic case: conditional quantity -1 clips to zero; P=0.5,
  Eqm=-100 cents, `ok=True`. The raw rule pulls, while the reconstructed contract
  forecast has value zero and keeps. Valid/invalid controls for the prior case
  behave correctly.
- Saved Oct26: **23 affected RETAIN rows / 9 orders**, maximum absolute Eqm
  **36.5011 cents**. Oct27: **15 rows / 6 orders**, maximum **5.2075 cents**.
  No affected POST row on either day. These counts are over all saved predictions
  (as-of Nov19 07:39:08.652623Z), not independent action samples.
- **Impact is zero in these runs:** every affected row occurs after that order's
  first model pull. Rejecting all affected rows changes **0 first triggers** and
  **0 cents** of baseline value on either day. This bug is not the explanation
  for the reported losses.

Require coherent zero-quantity moments and a round-trip check that reconstructing
the forecast preserves both moments and mapped value. Action-only parity is
insufficient: positive impossible values can agree on KEEP despite differing in
value. Add both signs to the falsifiers before a new candidate run.

### 2. P2 — the result report misattributes the difference from PULL-ALL

`predictfun/reviews/pipeline_one_day_v2_20261119/report.md:79–81` attributes the
whole +1,950 / -570 cent difference to the approximately 1% of orders never
pulled. Independent per-order fill accounting gives:

| Contribution versus PULL-ALL, cents | Oct26 | Oct27 |
|---|---:|---:|
| Pull at first live step | 0.00 | 0.00 |
| Pull later | +954.33 (1,040 orders) | -821.85 (831 orders) |
| Never model-pull, with RETAIN predictions | +995.36 (292 orders) | +251.74 (255 orders) |
| No RETAIN prediction/opportunity | 0.00 (168 orders) | 0.00 (157 orders) |
| Total difference | **+1,949.69** | **-570.11** |

On Oct27 the never-pulled subset contributes positively **relative to PULL-ALL**;
the delayed-pull subset drives the negative difference. Its absolute value is
still negative. First-step groups contain 26,942 / 26,795 orders and have
identical outcomes under the two policies. Publish this attribution in a
superseding report instead of directing a fix at the wrong subset. The total
values and comparisons themselves are correct.

### 3. P2 — the entry conclusion compares different continuation policies

The same report at lines 82–83 concludes that no model-selected posting avoids
loss and the problem is entering, not cancellation. The POST benchmark target is
the neutral order's value to END under structural exits **without model pulls**
(`pf_pipeline_export.py:169–179`, `pf_pipeline_bench.py:117–124`). The cancellation
experiment has a different continuation. Its results do not establish that entry
must fail with that continuation, or that size/price redesign is the required
next step.

For the **existing, unchanged J_ok and POST-value > 0 selection**, a saved-fill
accounting join gives the following local values in cents:

| Same selected POST opportunities | Oct26, n=2,502 | Oct27, n=1,975 |
|---|---:|---:|
| Neutral continuation, the POST label | -14,865.52 | -11,642.45 |
| With frozen MODEL-J cancellation | +276.93 | -823.53 |
| With first-opportunity PULL-ALL | -745.56 | -929.51 |
| No entry | 0 | 0 |

This is **post-hoc accounting on consumed data**, not a newly validated entry
strategy or a full portfolio replay. No threshold or model was fitted by the
review. The simulator treats these orders independently, so the join does not
establish shared-volume, inventory or repost behavior. It does show why the
neutral POST label alone cannot settle the entry-plus-cancellation question.
The defensible conclusion is that the current POST predictor fails its declared
neutral-continuation value target; entry with a frozen cancellation policy still
needs its own aligned evaluation.

## What the results establish

Baseline **M50 / activation 58 ms / cancellation 37 ms / fill notice 100 ms**;
one 30-share best-price JOIN opportunity every 5 seconds per window-side. Values
below are **cents marked to the common +5-second END**, not settlement P&L or a
spread-capture metric comparable directly with top makers.

| Policy | Oct26 | Oct27 |
|---|---:|---:|
| KEEP | -416,660.83 | -443,038.45 |
| A+0 fair-edge cancellation | -152,452.89 | -181,822.91 |
| MODEL-J learned cancellation | **-21,780.81** | **-23,916.10** |
| First-opportunity PULL-ALL | -23,730.50 | -23,345.99 |

The learned rule pulls 27,982 / 27,626 orders, approximately 99% of orders with
RETAIN predictions. Of those pulls, 96.3% / 97.0% occur at the first live step.
It beats A+0 in 17/19 timing scenarios on each day, but beats PULL-ALL in only
15/19 on Oct26 and 3/19 on Oct27. These are overlapping alternative assumptions,
not independent observations or an economic probability distribution. Do not
use the report's sum over 19 scenarios as expected return.

The early exposure explains most remaining absolute loss:

| MODEL-J realized value by fill time, cents | Oct26 | Oct27 |
|---|---:|---:|
| Before the first model decision at +100 ms | -15,892.56 | -13,988.06 |
| +100 to first possible cancellation effect at +137 ms | -7,837.94 | -9,357.93 |
| At/after +137 ms | +1,949.69 | -570.11 |

Given this fixed decision grid and latency, improving a cancellation decision
made at +100 ms cannot remove earlier fills. Entry selection, quote design or
faster effective action can affect that exposure, but this experiment has not
compared those alternatives. It does not establish that latency alone is the
cause or that a larger order will solve it.

**The main forecast miss on favoured POSTs is the fill-associated movement.**
Their predicted values average approximately +4 cents, versus realized neutral
POST values of -5.94 / -5.89 cents. Decomposing prediction minus outcome using
the exact mapped identity, Oct26's error is +25,336 cents in the joint movement
term versus +179 cents in the spread-times-quantity term; Oct27 is +19,546 versus
-79 cents. This is conditional on the published positive selection and does not
measure global feature importance. Quantity calibration or better fill AUC alone
does not repair this value error.

RETAIN ranks value beyond the fair-edge control and its new economic gate passes
on the declared Oct26 gate day. The corresponding boundary-sign check fails on
Oct27. Both dates were already consumed by earlier research; "gate day" does
not mean untouched validation. The neutral table includes hypothetical decisions
after the model would already have pulled. Restricting diagnostics to rows
reached before/at its first pull gives 52,413 / 49,805 rows, compared with
1,269,738 / 1,404,540 neutral RETAIN rows. On Oct27, reached coherent KEEP rows
have mean neutral retained value +1.41 cents on bids (12,710 rows) and -6.38 on
asks (9,112 rows). These overlapping decision diagnostics identify a useful
place to inspect; the per-order table above remains the economic comparison.

## Previous findings, tests and limits

- **POST causality:** original full-vector future-arrival example now passes,
  the same-time control changes inputs, and re-enabling the old convention
  reintroduces the leak. Exporter selftest passes. Saved v2 evaluation exports
  and their source hashes match; the development comparison reports all RETAIN
  rows unchanged and only the five POST queue features changed. This review
  inspected that comparison, not reran all five days' full-column comparison.
- **Rejected forecast bypass:** original mixed valid/impossible-endpoint batch
  now produces one pull and accounts for the refusal; a falsely accepted flag
  fails parity. The additional zero-quantity case remains finding 1.
- **Shifted-value false PASS:** the former +20-cent example now fails the economic
  gate, while a correctly signed/calibrated positive control passes. The old
  ranking/quantity gate is correctly labelled a diagnostic. This closes that
  counterexample, not all questions about economic calibration or generalization.
- **PULL-ALL labelling:** corrected columns distinguish improvement from total
  value in the new policy outputs. Earlier variants v2 publication also contains
  the corrected fields. No delta-as-profit interpretation is used here.
- **Scope:** fixed decisions across timing scenarios, cancellation-only policy,
  no learned admission/re-entry/inventory. This is now disclosed clearly.

Bounded checks through **PF_SLOT=codex / run_light.sh**: main verification
98.75 seconds / 1,605 MiB; impact follow-up 5.91 seconds / 1,595 MiB; both exit 0.
Exporter and policy synthetic selftests pass. The benchmark training selftest and
mutation suites were not rerun; their saved logs were inspected. All 12 source
manifest entries, 44 saved model files, both prediction files, both evaluation
export manifests/parts and published policy JSON/CSV/text identities were checked.
The review recomputed J's eight kind/side/day diagnostic and economic cells,
76 policy/scenario CSV rows, first triggers, baseline values from saved individual
fills, and the impact/attribution above. This is not an independent replay of all
19 raw-tape scenarios or a fresh model refit.

Populations and as-of:

| Day | Original opportunities | Eligible | Post-only refusals | Missing endpoint | Stale future book | Evaluated POST rows |
|---|---:|---:|---:|---:|---:|---:|
| Oct26 | 30,242 | 28,442 | 1,390 | 374 | 36 | 29,832 |
| Oct27 | 29,820 | 28,038 | 1,300 | 460 | 22 | 29,338 |

Benchmark as-of Nov19 **07:39:08.652623Z**; policy outputs **07:39:21.127033Z** /
**07:39:33.598761Z**; exports **07:23:49.445752Z** / **07:30:44.484555Z**.
Underlying keep-path receipts Nov18 **02:59:27.356546Z** /
**03:01:23.638978Z**. Model training used consumed Oct23–25; Oct27 is explicitly
post hoc. No raw tape, new fit or untouched-day access in this review. Two days
support point estimates only, with no day-cluster interval. Queue/fill realism,
execution-clock assumptions, cancellation refusals and portfolio accounting
remain outside what these checks establish.

## One next priority

**Evaluate entry using a frozen downstream cancellation policy and a common
economic target.** After repairing the small coherence boundary, keep the current
size, price, thresholds and model fixed for this diagnostic. Compare POST versus
no entry under MODEL-J and PULL-ALL, decompose loss before the earliest effective
pull versus continuation value, and assess quantity/movement calibration on the
orders actually selected. Include later-pull versus never-pull attribution and
bid/ask splits. This makes the next training target and controller benchmark
agree before adding new order designs or tuning thresholds. Declare any changed
candidate before further evaluation; these seen days cannot become validation.

Review only; the development session owns fixes. No scheduler restart, push,
merge or implementation change.
