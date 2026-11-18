# Manual paired-clock result review — 2026-11-18

Pinned target: **1a1abbaa0e9b572a6d71c9150aa530a11f5b7e6f**.
Reviewed the full change from e0067ca654056b235424e6e4bd3cf637789f2dc6,
including the new audit source, results and state changes, in detached checkout
`/tmp/predictfun-review-1a1abbaa`. Another session owns development.

**The paired audit reproduces. No new arithmetic or implementation defect was
found in these checks. It closes the previous attribution question: changing
the simulated fills dominates changing eligibility, and the rule's selection
advantage still reverses on common decision opportunities. The next priority
is to identify or explicitly bound fill-versus-cancel ordering, retaining all
economic consequences, before tuning a model.**

## Findings and remaining gates, in priority order

### 1. Main unresolved gate: the cancellation-value labels depend on the fill clock

Population: 151,228 counted states, 143,134 eligible resting orders, 6,040
post-only refusals and 2,054 declared path exclusions on consumed Oct 23–27.
Result as of **2026-11-18T03:40:50.672573+00:00**; review verification as of
2026-11-18T04:03:01.454329+00:00. No untouched validation was used.

The two clocks' common risk sets produce the same **69,421 executed pulls**.
On those opportunities, the declared A+0 rule's excess over the exact expected
STATES comparison is:

| Fill-clock assumption | Excess, cents | Days with positive excess |
|---|---:|---:|
| M50 | +171,416 | 5 / 5 |
| M250 | -27,154 | 1 / 5 |

These are local decision-state comparisons, not absolute profits, portfolio
P&L or a newly trained model. The negative result does not prove that no other
causal predictor can work.

The full diagonal change is -261,403 cents: -234,132 (89.6%) is attributed to
the fill scenario and -27,271 (10.4%) to client eligibility/null weights by the
declared two-path Shapley decomposition. Eligibility matters but does not
explain the sign reversal. All six cells and the side/source decompositions
reproduce independently.

Evidence at the target: `predictfun/research/pf_keep_pull_clock_audit.py:188`
(common eligibility and crossed scenarios), `:119` (decomposition), and
`predictfun/reviews/keep_pull_clock_audit_20261118/clock_audit.txt:5` and `:23`.

### 2. Important interpretation limit: the fill factor includes changed quantities

The label `FILL-TIMING-DOMINATES` names a change of the entire saved fill table,
not a pure timestamp perturbation holding executions fixed. `fills_of` uses
each clock's quantities as well as its timestamps
(`pf_keep_pull_clock_audit.py:59-64`). The upstream replay moves trade anchors
relative to an unchanged book (`pf_queue_place_test.py:80-86`), which can alter
queue depletion and execution.

An additional saved-table check finds **24,881 / 143,134 orders (17.4%)** have
different total filled quantity between M50 and M250. This is a descriptive
check on the same consumed population/as-of, not new validation. Thus the
result supports **fill-scenario sensitivity**, not a finding that real-world
losses are caused solely by timestamp error, nor proof that either fixed
offset is the true matching clock. The 200 ms scenario difference is not, by
itself, an empirically established bound on every source's timing error.

### 3. Next-step risk: dropping near-cancel fills does not repair the economic target

The handoff proposes a target that values only fills beyond the clock's
uncertainty after cancellation (`predictfun/HANDOFF.md:9`). That can be a
declared diagnostic, but it cannot replace the full keep-versus-pull value
without retaining the omitted fills' gains and losses. Those fills may be
exactly the ones a cancellation avoids or fails to avoid.

In fact, subtracting the reported first-200-ms contribution on the COMMON
opportunities leaves excess **-65,705 cents at M50 and -56,532 at M250**. This
is arithmetic on the published result, not a newly selected policy test.
There is no evidence here that merely excluding that interval rescues A+0.
Evidence: `pf_keep_pull_clock_audit.py:67-83` (near-window definition) and the
COMMON rows of `clock_audit.json`. All sources must stay in total accounting;
the favorable WS-only contribution is not an admissible policy-selection gate.

## Prior review resolution

- **Paired fill/eligibility attribution:** completed and independently verified.
- **Improvement versus absolute value:** the STATUS stakes now distinguish them;
  the prior handoff latency claim has an explicit fixed-setup qualification.
- **Ledger exclusions and unmatched trajectory acceptance:** remain resolved at
  their previously verified scope; this commit does not alter those engines.
- **Live fill validity and clock-robust selection:** still unestablished. No
  profitability or promotion follows from passing this diagnostic.

## Verification and limits

`verify_audit.py` verifies the committed source hash, dependency hashes, saved
result copies, five source receipts and all order/fill parquet hashes. It
checks status counts and refuses any fill whose order is outside the eligible
population. It regenerates A+0 from fair inputs and client eligibility from
fill-notice/structural times.

An independent per-order/step probability calculation reproduces every fill's
score, exact null, and near-window contributions for all six crossed scenarios
on every day; all pooled/day/side/source totals and Shapley terms agree. The
pinned selftest passes (20,000 synthetic draws, brute-force comparison, planted
fill/eligibility cases). Independent controls check exact cancellation and
200-ms boundaries and refuse a deliberately wrong boundary value.

Final bounded check: **19.33 s, 413 MiB peak RSS**, through `PF_SLOT=codex`
`predictfun/research/run_light.sh`. Only saved derived data were read. No raw
tape, training, development-source edits, new strategy, full report run or
untouched validation. The first harness attempt stopped because it looked for
the copied run log in the data directory; the corrected harness checks its
repository location. That setup error is preserved in
`verification_initial_path_error.log`; it is not a research-result failure.

Evidence: `receipt_v1.json`, `verification.log`, `verify_audit.py`. Source and
result are committed together; this review treats the work as a consumed-data
diagnostic, not independently proven preregistration or forward acceptance.

## One concrete next priority

**Build a source-aware audit of fill-versus-cancel ordering for the same neutral
decision population.** Link each relevant trade/fill to its original identity
and distinguish observed source/receipt times from inferred match times and
simulated cancel landing. Report whether timing establishes that the fill is
before cancellation, after cancellation, or ambiguous. Carry ambiguous fills
into the full value accounting with explicitly justified uncertainty; report
scenario ranges where genuine bounds cannot be established. Do not silently
drop near fills, assume M250 is ground truth, or fit an offset to improve P&L.

The deliverable is evidence about how much keep/pull value has an identified
sign under supported timing assumptions, by side/source/day with unknowns
retained. Where existing captures lack matching-time precision, name that
missing evidence rather than claiming the clock is calibrated. This is the
gate before model tuning; development owns it. No live execution is part of
this review. The removed schedule remains removed.
