# Queue continuity and the cost of bad fills — USER amendment

2026-11-22. Parent review commit: **3e84ca6d9e8a957e14ec97f5dbafe48a83f2026d**.
Inspected execution source/result: **85127b3dfbdfb8689159fe8b9e76b447324b7cfc**.

USER: “We need to identify the cases when bad fills are necessary to get the
front queue for good fills, and purely bad fills.”

This amends the required outputs and interpretation of
[FEASIBLE-CANCEL-HEADROOM-v1](../20261122_async_fast_review/report.md).
It stays ONE development round and ONE combined review. No new experiment,
fit, outcome table or policy is run by this amendment. The development session
owns implementation; fold this into the pending freeze or a declared amendment
before its new output. Do not discard or silently change any already-frozen run.

## Question and distinction

The unit is a **resting order generation and a feasible decision on its
remaining life**, not an isolated losing fill. Measure which losing fills
come with sufficient later good fills to justify retaining that order, and
which can be removed with no or insufficient sacrificed good value.

Bad fills do not mechanically buy queue priority. In the inspected replay,
same-price matching consumes external ahead before our quantity fills, and
the residual order retains its original priority. Through-price matching
ignores that queue and zeros ahead (`pf_branch_engine.py:153-174`). Resting
continuously preserves exposure to both good flow and adverse flow. An own
fill is not evidence that accepting a loss was needed to advance the queue.
Own orders at the same price also have price/time priority; external ahead
zero is not a universal all-orders front certificate.

The current saved-data test can establish **a realized trade-off conditional
on one permanent cancellation and the recorded continuation**. It cannot
establish that accepting a bad fill was necessary under every alternative:
cancelling and reposting might still capture later good fills. That needs a
same-start-state KEEP versus PULL/re-entry counterfactual with actual queue
reset, fills, notices and inventory feedback. Such a replay is not implicitly
authorized or replaced by this saved-data audit.

## Required additions to the existing audit

Keep its population, source identities, feasible live KEEP callbacks, 10 ms
effect, strict timing, one-cut oracle and no-replay scope unchanged. CURRENT
ASYNC is primary, FRONT ASYNC separate. All generations/statuses stay present.

For each allowed cut at decision d, let S(d) be that generation's recorded
fills at or after the hypothetical cancel effect. Using the declared eligible
+1 s per-fill values v, report:

- **G(d) = sum max(v, 0)**: good value lost if cancelling there.
- **L(d) = -sum min(v, 0)**: bad value avoided.
- **KEEP-minus-cancel local value = G(d) - L(d)** on that suffix.
- Positive/negative/zero fill records and shares, not just net value.
- The already-realized prefix separately. Past bad fills are sunk at d;
  do not charge them again to decide whether retaining now is worthwhile.

Use accounting tolerance 1e-6 c for meaningful summed value, with exact
zero counts alongside it. These are descriptive classes, not trading rules:

| Future recorded exposure | Definition | Permitted interpretation |
|---|---|---|
| Purely bad observed suffix | L > tolerance, G <= tolerance, complete +1 s marks on that suffix | That feasible cancellation removes losses with no meaningful recorded good-fill value sacrificed. |
| Compensated mixed suffix | G, L > tolerance and G-L > tolerance | Keeping that recorded suffix earns more good value than bad cost. |
| Uncompensated mixed suffix | G, L > tolerance and L-G > tolerance | Later good fills exist but do not repay the bad cost; distinguish from purely bad. |
| Mixed near tie | G, L > tolerance and abs(G-L) <= tolerance | No meaningful local net advantage at that cut. |
| Good-only / near-zero / no fill | Remaining exhaustive cases | Keep these denominators; absence of a loss is not a success signal. |
| Incomplete valuation | Any suffix fill lacks its declared +1 s value | Show observed components, but do not call the entire suffix purely bad or compensated. |
| No feasible callback | No recorded live KEEP can be changed | Constrained by this diagnostic's action set, not proof of universally unavoidable loss. |

“Complete” refers only to valuation of the **recorded** continuation. Always
show the actual generation end cause (filled out, recorded cancellation,
refusal, END/open). A recorded cancellation truncates the possible later
benefit of KEEP; no-positive-before-that-cancellation is not evidence of
no-positive-ever. No-fill and refused orders are not bad-fill cases.

### Order sequence and the queue connection

Make the following concrete cases visible with exact event and callback times:

1. **Bad then good on the same surviving generation.** Was there a feasible
   cancel before the bad group? If so, a permanent cancel avoiding it would
   also remove later good fills. State whether their net value compensates
   the losses, without calling that evidence of queue necessity.
2. **Good then bad.** Was there an actual live KEEP callback whose effect lies
   after the good group and at/before the bad group? If so, the recorded good
   fills can be kept while cutting the bad suffix under this local test.
3. **Interleaved or simultaneous good/bad fills.** Respect indivisible
   execution times and the full suffix. No hindsight removal of selected
   negative tranches while retaining later positives at unchanged priority.
4. **Only bad observed fills, or too little later good value.** Keep pure and
   mixed-net-negative cases separate; both are candidate exposures to avoid,
   but only the first sacrifices no recorded good value.

Sequence categories may overlap across a long order. Do not sum them as
disjoint populations or recount one fill at every callback. Economic totals
must reconcile once per generation at (a) the first eligible callback and
(b) the existing oracle-selected cut/KEEP. Other callback rows are a labeled
decision-state panel, not independent samples. For a small casebook, use a
declared deterministic key/rank per category, not handpicked persuasive wins.

Keep same-gid continuity distinct from a replacement at the same price, and
external queue distinct from own price/time ranking. Carry submitted size,
remaining quantity with its observation clock, order age, activation ahead,
first-review ahead and any exact-time queue record actually available.
QueueObs saves activation/first-review/pull observations, not a full queue
trajectory (`pf_ladder_pressure.py:235-254`); no future observation, invented
interpolation or “bad fill improved queue” inference. Unknown stays unknown.
FRONT ASYNC is a separate artificial-priority reference, not a matched proof
that a particular CURRENT fill purchased queue value.

### Outputs, controls and decision use

- Report G/L, net, fill/share counts, feasible effects, end causes and missing
  marks by day, side and class. Include concentration and the share of losses
  in pure, compensated and uncompensated mixed suffixes at each declared
  anchor. Do not combine repeated decision rows or the two arms.
- Keep +1 s as the existing primary diagnostic. For the **same cut**, report
  5 s / 30 s and END sensitivity from saved marks only, with native eligible
  quantities and exclusions. Compare horizon signs on an identical common
  cohort before claiming a loss recovers. Do not reselect cuts by horizon.
- Retain the parent's unchanged-path, suffix and boundary reconciliation.
  Add hand cases: bad-then-good with net gain, bad-then-good with net loss,
  good-then-bad with and without a feasible intervening effect, simultaneous
  fills, a missing later mark, and a new gid at the same price. A known-bad
  classification must fail its named check. These fit the same batch.
- Do not train on these outcome-defined groups or present hindsight classes
  as predictable states. Any later predictor must use decision-time inputs,
  all eligible exposures and a causal evaluation design under its declared
  population. The audit's immediate purpose is to say **what economic
  discrimination is worth learning**, including how costly false cancels are.

The saved-data answer must use “linked under the recorded keep-versus-
permanent-cancel choice,” not “necessary to earn front priority.” A claim
of actual queue dependence remains **unidentified** until re-entry is
compared. Conversely, a purely bad suffix on this finite path is a candidate
for avoidance, not proof it was detectable before the fills.

## Scope and resource disposition

This supersedes any reading of the parent plan that would label every bad
prefix before later good fills as an unavoidable queue-acquisition cost.
It clarifies the same audit; it does not start an extra tuning/review cycle.
No strategy implementation, live use, new day or replay is authorized here.

The parent's development cap **0.05 core-hours**, review reserve **0.02h**,
and cumulative **2.50h** ceiling remain. Count this small planning/check
overhead in the cumulative ledger. If the expanded tables cannot fit,
prioritize the reconciled G/L classes and feasible sequence cases; defer
ancillary casebook formatting, never validity checks. No additional USER
permission is needed for this saved-data clarification within those caps.
Any queue re-entry experiment needs a concrete bounded design and resource
decision first. POSITIVE-PRESSURE-GUARD-v1 remains stopped.
