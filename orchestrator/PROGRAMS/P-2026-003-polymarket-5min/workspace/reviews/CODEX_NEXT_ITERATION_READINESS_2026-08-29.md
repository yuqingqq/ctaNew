# Codex next-iteration readiness review — 2026-08-29

**Audit tip:** `3e6eb381a152a39a0698479a14dbea5b219fe326`  
**Scope:** research sequencing after the R-293 fragment diagnostic and before
the first post-O1 complete UTC day  
**Result-bearing work performed:** none; no fit, score, threshold choice, or
forward-day read

## Decision

The next bounded hypothesis is **still Iteration 011 conditional signed-value
decomposition**, frozen by R-232 and Amendment A1. **Do not open a new feature,
model, budget, or Iteration 012 scoreboard.** The positive fragment diagnostic
was preregistered as weak comfort only and cannot select the 10% budget, a new
arm, or a revised hypothesis.

**Iteration 011 fit/score HOLD remains maintained.** Its production runner is
byte-identical to the exact code reviewed at `a63d717`, where the hold was
maintained. The four executed round-3 blockers therefore remain current; two
were reconfirmed directly at this tip below.

This is a readiness/sequence finding, not a new candidate and not another
multiplicity event.

## Current-state evidence

| Artifact/check | Current evidence |
|---|---|
| `phase2_iter011.py` | SHA-256 `5a5327c8b43b6297f950fdad186a1da4228ae312f7978a9affea28add328cd5e`, exactly equal to `a63d717` |
| `phase2_iter011_run.py` | SHA-256 `ca4dc7c624371ad114bb83e1fbc0d2ae60c4f5281763c73083e7e513403bbccc`, exactly equal to `a63d717` |
| incumbent in real `main()` | AST call count: load `0`, composed apply `0`, hazard apply `0` |
| Q4 in real `main()` | one candidate-only `q4_economics()` call; no incumbent argument |
| failed-cell survivor | an executed `NO_INCUMBENT_COUNTERPART` Q2 cell with `p=0.001` receives Holm `0.024`, is marked surviving, and enters `surviving_cells` |
| post-O1 forward evidence | no complete post-O1 UTC day exists yet; earliest possible day is 08-30 after a successful boundary and close-time gate |

Exact byte identity makes the prior executed round-3 counterexamples current
evidence rather than a memory-based carry:

- duplicating rows inside an unchanged action moved Q1 AUC from `1.0` to
  `0.00990099` while reported `n_actions` stayed 2;
- a 200-row, one-action fixture passed the 100-observation fitting floor;
- Q3 computed no slope intervals;
- the runner returned `AGGREGATION_UNDECLARED` rather than implementing the
  later user rulings for Q3 and the coin axis.

## Why suite-green is not readiness

The runner contains correct isolated incumbent loaders and appliers, but its
real `main()` never invokes them. It computes candidate Q4 economics alone.
The component capability therefore still does not reach the result-bearing
path.

`assemble_family()` also derives survival solely from adjusted p-value. It
does not conjunct the cell's governed status/gates, so a named failed cell can
become a discovery. A denominator fixed at 24 is necessary but insufficient:
the survivor predicate itself must require every frozen leg of that cell.

The current report path computes row-level AUC/Brier/slope and merely reports a
deduplicated action count. Generation weights are computed on the all-row
population and then subset, so a conditional head can give one generation
half the mass of another. The action population must govern fitting weights,
power floor, metrics, nulls, and reported `n`, not just the label printed beside
row metrics.

Finally, Amendment A1 now contains the user's R-306 rulings—BTC-only
adjudication and Q3 conjunction plus worse-side—but the unchanged runner still
collapses all available coins and describes Q3 aggregation as undeclared. It
also takes the minimum of whichever point slopes are present, allowing one
side to carry the report and computing no ruled interval.

## Minimum-safe sequence

### 1. One correctness/identity cycle; not a hypothesis

Bundle the already-recorded identity-moving defects before any fit so the
pipeline pays for one attributable rebind rather than several sequential
identity changes:

1. make `encode_row` fail closed on a missing guarded field instead of encoding
   it as confidently present zero;
2. make action evaluation order-independent at the actual frozen threshold,
   with tie counts measured at that threshold rather than nominal top-k;
3. make the result-bearing builder expose a real selftest/CLI and refuse an
   unknown mode instead of launching a build;
4. perform the already-queued annotation-merge wiring, manifest re-stamp,
   tranche-persistence decision, and one-sided increment-null supersession in
   the same governed cycle where applicable;
5. prove semantic changes and identity changes separately, then re-gate/rebind
   in-band. No old receipt is edited.

These repairs consume no candidate slot and may use synthetic/reproduction
tests. They may not inspect a new outcome and then choose their semantics.

### 2. Close Iteration 011 in a non-fit batch

Before any number exists:

1. load and verify the frozen incumbent once per coin in real `main()`; apply
   its hazard head to Q1 and composed value to Q4 on the identical actions;
2. predeclare and implement one action-level reduction, then use it consistently
   for head-specific weights, power floors, metrics, matched controls, and `n`;
3. make joint survival the conjunction of status, every frozen head gate, and
   Holm—not Holm alone;
4. implement R-306 exactly: BTC alone occupies the 24 adjudicated cells; ETH is
   fully reported but never adjudicated; Q3 requires both slope legs and uses
   the worse side under the ruled interval semantics;
5. add production-path seams for each item and request another **non-fit**
   review. Green helper tests alone do not release the hold.

### 3. Only after explicit release, run Iteration 011 once

- Fit and score exactly the two frozen arms: composed-linear and
  composed-LGBM.
- Preserve the fixed 24-cell family and pinned parameters; no sweep.
- Use only the preregistered development populations and declare them
  development, never validation.
- Do not choose a budget, threshold, arm, or feature from the output and then
  reuse that population.
- Report all four heads and both coins even when null/underpowered; adjudicate
  BTC only as ruled.
- No automatic race entry or strategy promotion follows from a positive.

### 4. Forward and integrated strategy work stays separate

The first post-O1 day may accrue only after the stamped Aug-30 deployment and
its close-time day gate. Validation remains per coin at `G >= 5` complete
untouched UTC days. Below that: point estimates only, no interval or promotion.

Iteration 011 is a model decomposition, not integrated market-making
performance. The later stateful replay must still include, on identical neutral
opportunities, `QR_CANCEL_HOLD_X_SKEW` as the queue-realistic baseline and
`QR_SKEW_ONLY` as the required comparator, with no same-price zero-queue/front
assumption and no policy-generated training path.

## Coordinator action

Keep 011 dark and dispatch the correctness/identity cycle plus the four
non-fit closures above. Do not interpret the waiting time for post-O1 data as
permission to start another scored family: code correctness and frozen-protocol
wiring are the available work that advances the same hypothesis without
consuming evidence.
