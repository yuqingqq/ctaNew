# Queue-fill classification — independent saved-data results

2026-11-22, as of this receipt. User-requested classification, frozen at
**2a8cde50** before the run; input receipt pinned to
**85127b3dfbdfb8689159fe8b9e76b447324b7cfc**. This is review evidence for the
same pending development round, not a new strategy or candidate test.

## What was classified

Every submitted generation on the same **19 complete markets from consumed
Oct 24–27**: **5,832 CURRENT_ASYNC_FAST10** orders and **6,303 FRONT_ASYNC_FAST10**
orders, kept separate. One row per generation, no selection on whether the
hindsight rule cancels it. Anchor = first recorded live KEEP callback;
cancel effect = that time +10 ms. Classify that order's subsequent recorded
fills at/after the effect, valued at each fill's **+1 s mid**. Earlier fills
are separate; they cannot be prevented by that cancellation.

## Mutually exclusive economic classes

G is gross positive value of the suffix; L is the magnitude of its gross
negative value. Units below are cents. Meaningful summed value uses 1e-6 c.
Incomplete suffixes are not labeled pure or mixed based on observed portions.

| Class | CURRENT orders | CURRENT suffix net, c | FRONT orders |
|---|---:|---:|---:|
| Purely bad observed suffix | **1,114** | **-122,122.64** | 1,241 |
| Good only | 711 | +58,278.41 | 1,370 |
| Mixed, gains outweigh losses | 68 | +2,025.88 | 121 |
| Mixed, losses outweigh gains | 84 | -6,889.27 | 238 |
| Mixed near tie | 0 | 0 | 2 |
| Near-zero valued suffix | 21 | 0 | 18 |
| No suffix fills | 3,416 | 0 | 2,877 |
| No eligible KEEP callback | 400 | not evaluated at this anchor | 419 |
| Incomplete suffix valuation | 18 | partial values only | 17 |
| **All generations** | **5,832** | | **6,303** |

CURRENT has 1,998 generations with complete marked suffix fills; FRONT has
2,990. Do not divide a selected row count by all filled orders and call it a
decision success rate. The no-opportunity class can contain fills: its future
cannot be evaluated at an absent KEEP callback, not because its value is zero.

Post-run descriptive arithmetic from the frozen classes: pure-bad suffixes
carry **92.85%** of CURRENT's observed gross negative suffix value
(122,122.64 / 131,521.00 c), and **85.02%** for FRONT. This is a hindsight
accounting fraction, not a forecast hit rate or a realizable avoided-loss
estimate. Denominator includes the observed part of incomplete suffixes;
their missing values are not imputed to zero.

## Sequence matters: a mixed order is not necessarily a cost of keeping queue

These are cross-tabs within the mixed classes, not additional disjoint
populations to add to the table above:

| Observed mixed sequence after the anchor | CURRENT | FRONT |
|---|---:|---:|
| Bad then good, later gain compensates loss | **11** | 17 |
| Bad then good, later gain does not compensate | 11 | 11 |
| Bad then good, net tie | 0 | 1 |
| Good then bad, with a feasible intervening cancel effect | **124** | 316 |
| Interleaved/simultaneous | 6 | 16 |

Every good-then-bad row in this sample has an actual recorded KEEP callback
whose effect is after its good fills and at/before its bad fills. Thus these
124 CURRENT cases do not require sacrificing the recorded earlier good
fills to remove the later bad suffix, under the local permanent-cancel test.
This is stronger than labeling every mixed order “necessary bad exposure.”

The 11 compensated bad-then-good CURRENT cases are candidates for a real
retention trade-off: a permanent cancel avoiding the early bad group also
removes the later good group. They do **not** prove accepting that loss was
needed to obtain front priority. A cancel/re-entry path remains untested.

## Limits that matter to the next step

- These labels describe **recorded continuations**. Among CURRENT's 1,114
  pure-bad rows, 662 orders were later closed as removed and 452 as filled.
  Removing an order truncates observation of what might have happened had it
  remained live. “No good fill in this suffix” is not “no later good ever.”
- Value is +1 s, not permanent loss or settlement P&L. The pending combined
  development audit must add the declared same-cut 5/30 s and END sensitivity
  with explicit/common cohorts before claiming a loss cannot recover.
- Classification is hindsight. No evidence here establishes that these groups
  are distinguishable before execution. No predictor was fitted or evaluated.
- A first-anchor compensated suffix can still contain a removable bad tail;
  this is why timing/sequence is a separate axis. These classes alone are not
  KEEP/PULL rules. Sunk prefix losses do not belong in the later choice.
- Queue observations are saved activation/first-review records, not an
  invented continuous trajectory. A new gid at the same price is a new
  order. Hypothetical FRONT and CURRENT paths are not matched per-order proof
  of queue causality. All other orders/inventory feedback remain fixed here.

## Verification and integration

`classify.py` and declaration/input hashes were committed before reading fill
outcomes. Synthetic controls cover every category, missing marks, effect
boundaries and feasible sequence separation. All input hashes, market/order
membership, one-row identity, prefix/suffix accounting and fill counts pass.
The **38 arm-market totals** reproduce accepted +1 s and END values. Full
CURRENT +1 s remains -78,485.90 c and FRONT -61,051.63 c; the classification
does not change these results. Independent class predicates refuse a wrong
pure-bad label and a missing-mark case disguised as pure-bad.

`readback.py` rereads the exported generation/class/sequence/reconciliation
files and independently verifies all grouped values and counts. Its extra
fractions/end-cause breakouts are explicitly post-run descriptive arithmetic,
not a changed declaration or reading rule. Outputs include a deterministic
casebook in `outputs/examples.csv`, chosen by hash, not by profitable examples.

Compute through PF_SLOT=codex/run_light.sh: prepare 0.25 CPU s, run 2.20 s,
readback 0.56 s = **3.01 CPU s**, peak 158 MiB. Count this against the pending
audit's existing 0.05h allowance and cumulative 2.50h cap; no extra budget.
No raw tape, historical replay, fit, new day, development-source edit or push.

**Development integration:** the uncommitted headroom draft inspected during
this turn classified only oracle cuts (`generation_row` / `cut_kind`) and
collapsed retained generations to `not_cut`. That is insufficient for this
USER requirement: it hides compensated exposure and good-only orders.
Incorporate/reproduce these all-order first-anchor classes, preserve the
separate oracle-cut analysis, and carry the queue/truncation/horizon limits
into the same combined receipt. Do not create an extra fix-only round.

**Next priority is unchanged:** finish the feasible-cancel headroom audit
with these classes, then ask whether timely decision features can distinguish
pure/uncompensated exposure from valuable exposure. These results motivate
that question; they do not yet answer it or authorize a new model/replay.
