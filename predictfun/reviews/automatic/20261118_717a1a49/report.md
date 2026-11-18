# Manual keep/pull results review — 2026-11-18

Review as of 2026-11-18 03:12 UTC. **Pinned target:
`717a1a49f27320def34cb466ca73b7325daf8ea3`.** Reviewed the full results/state diff
from `532d2a79fd0a825a0727c98ed203c6d944ae13e2`, using the detached checkout
`/tmp/predictfun-review-717a1a49`. No implementation changed in that range.

**No new accounting or implementation discrepancy found in the bounded checks.
The result shows local selection at M50, but it is not a profitable or
clock-robust policy.** The next priority is to diagnose the M50/M250 difference
on paired orders and decision states before tuning prediction.

## Findings by priority

### Main unresolved gate: selection reverses under the M250 fill clock

The declared A+0 rule pulls at the first live decision with negative signed
fair edge. It is a fixed rule, not a newly trained LightGBM model. At M50 it
beats the declared STATES comparison; at M250 it fails both K2 and K3 and
exceeds the random mean on only one of five days. The stored trajectory null
remains NOT-EVALUABLE, correctly: every draw has unmatched pulls. This is a
confirmed limitation reported by the commit, not a new failure of its guard.

Pinned evidence: `predictfun/reviews/keep_pull_20261117/keep_pull.txt:29-47`
and `:68-74`; sampler and dispatch in `pf_keep_pull.py:459-475` and `:551-560`.
All unmatched comparisons retain null trajectory K2/K3 in the emitted JSON.

M250 shifts the assumed **trade timestamps**; it is not 250 ms cancellation
latency. The change affects simulated fills, full-fill notices, eligible
decision states and control populations. Actual score pulls fall from 70,373
at M50 to 69,462 at M250. Thus the difference is not a comparison of exactly
the same executed decisions with a different markout label.

I computed a bounded, post-hoc source decomposition from the saved fills.
For each stratum/step, the STATES sampler includes an eligible order with
probability `number_of_pulls / number_of_live_orders`. Weighting its removable
fills by that probability gives the exact expected state-null improvement,
without rerunning the full Monte Carlo report.

The following is **score improvement minus exact expected STATES improvement**,
in cents summed across the five consumed days. It is a local selection
diagnostic, not realized portfolio P&L or a source-specific significance test.

| Fill source | M50 excess, cents | M250 excess, cents |
|---|---:|---:|
| WebSocket-timed | +130,295 | +20,298 |
| E2-estimated | +21,380 | -27,294 |
| Imputed | -428 | -1,427 |
| Mixed sources at one timestamp | +83,031 | -18,702 |
| Total | +234,279 | -27,124 |

The exact state means are 935,392 cents at M50 and 1,051,920 at M250; the
committed 500-draw sample means are 936,736 and 1,051,187 respectively. The
clock reversal therefore persists using the analytical expectation. It occurs
on both sides in aggregate: bid excess +120,263 to -7,525 cents, ask excess
+114,017 to -19,599 cents.

The deterioration is **not confined to E2-estimated timestamps**: the
WebSocket contribution also shrinks substantially, and mixed-source fills
reverse. This does not prove a timestamp bug or a model defect. It narrows the
next diagnostic: match the order/decision populations and separate changed
fills from changed eligibility/control weights. Do not select only the
favorable WebSocket subset as policy acceptance.

### Absolute value remains negative; improvement is not profit

Population: 143,134 eligible resting orders from 151,228 counted states on
consumed Oct 23–27; the remaining 6,040 are post-only refusals and 2,054 have
declared path exclusions. Result as of **2026-11-18 03:01:27 UTC**.

At M50 / 37 ms cancellation latency, summing the order branches' cash plus
inventory marked at the common five-second endpoint gives:

| Rule | Improvement over KEEP | Absolute marked value |
|---|---:|---:|
| KEEP | $0 | -$19,461.63 |
| Declared fair-edge score A+0 | +$11,696.71 | -$7,764.92 |
| PULL-ALL at the first live decision | +$18,412.43 | -$1,049.20 |
| Perfect-foresight oracle | +$33,008.77 | +$13,547.14 |

These are sums of isolated simulated order values, not a validated portfolio
return, settlement P&L, or fill-relative markout. The source report labels its
positive numbers as improvements (`keep_pull.txt:11-16`); retain that distinction
when summarizing `STATUS.yml`'s stakes. PULL-ALL acts more often than the score,
so its higher total is evidence about exposure, not superior matched ranking.

The score is still negative at zero cancellation latency (-$7,131.82), at M250
(-$13,937.91), and without first-100-ms fills (-$6,557.81). In the no-young-fill
view PULL-ALL becomes exactly zero because removal at +137 ms precedes the
included fills at age >=100 ms after +58 ms activation. That is a consequence
of the filter and timing, not evidence of a profitable trading strategy.

### Interpretation limit: the oracle does not measure timely causal detection

The 94% avoidable-loss figure and the 91,555-cent same-decision 0/37 ms gap
reproduce. Every executed oracle pull in the saved-data reconstruction occurs
at step 2, the first live decision (+100 ms), across all reported variants.
One price and one endpoint mark make every fill of a given order have the same
sign. Knowing the endpoint sign lets the oracle pull losers immediately.

This is exactly the ceiling declared in `pf_keep_pull.py:38-44`, not a model
that discovers adverse flow later in an order's life. The small cancellation
gap does not quantify how soon a usable signal becomes observable, nor changes
to feed/placement latency, queue position or re-entry. The latency-priority
inference at `predictfun/HANDOFF.md:8` must remain scoped to this fixed setup.
Eliminating the existing score's 37 ms cancel delay is insufficient here, but
the run does not establish that better causal timing is unimportant generally.

## Accounting and verification

`verify_results.py` checks the pinned source against recorded hashes and reads
only existing derived order/fill tables. On every day it verifies original JOIN
reconciliation and ledger status identity, regenerates the primary threshold
decisions independently from fair inputs, and reproduces all reported policy
outcome components and oracle/pull-all latency curves. Direct fill summation
also reproduces the primary score at M50 and M250. Live-decision counts and
missing-input counts match: 6,935,675 live states, 579 without a causal fair.

The previously repaired ledger accounting and unmatched-acceptance findings
remain resolved at their stated scope. Reports, receipts and logs match their
committed/live copies; all five full-day order/fill hashes and source hashes
match. Recorded synthetic and mutation suites are passing; those full suites
were not rerun. The bounded check includes positive arithmetic controls, a
known-bad value mismatch, state-expectation controls, and the existing reading
and unmatched-exhaustion controls.

Final check: **6.00 seconds, 507 MiB peak RSS**, sequentially through
`PF_SLOT=codex predictfun/research/run_light.sh`. No full report rerun, tape
experiment, training, or development implementation change. Full Monte Carlo
draws/quantiles were not regenerated; their dispatch/predicates and the exact
state-mean diagnostic were checked. Raw venue-time/anchor correctness and live
fill validity were not established by this review.

Evidence: `receipt_v2.json`, `verification.log`, `verify_results.py`.
The v2 receipt supersedes v1 only to omit uncomputed score fields at latencies
without declared score evaluation; every evaluated result is unchanged. The
old receipt remains as provenance. All days are consumed development data;
no untouched validation, new confidence interval or strategy promotion.

## One next priority

**Perform a paired M50/M250 fill-clock and eligibility audit on the same saved
orders and decision opportunities.** Attribute the lost selection advantage
to changed fill timing/quantity versus changed client eligibility and state-null
weights, by side and source, retaining E2, imputed and mixed fills in the total.
The basic source split is already supplied above. This gate should establish
whether the economic target/control remains stable enough to justify model
tuning. Development owns any implementation or further tape work.

This was a manual review. The removed schedule remains removed.
