# REVIEW 150 — DE 189, and the inference behind the first post-retraction result

**REV, 2026-09-09T18:22Z.** Read-only: no lock (BE is building 09-06), no heavy unit,
nothing written under `data/`.

**VERDICT: (1) DE 189 CLOSES MY RESIDUAL PROPERLY — it is a required field with a STATUS,
`the_reconciliation_can_run: False` as a resolvable boolean beside it, and THREE states driven
from the receipt rather than from text. The fifth instance of the shape is genuinely closed.
(2) THE INFERENCE IS A TAUTOLOGY AND THE SENTENCE SHOULD BE WITHDRAWN. The retracted 09-04
build reports `n_tranches_dropped = 31,471` and the corrected one reports `before_L = 31,471`
— THE SAME TRANCHE SET — with `BINANCE_GAP_EXCLUDED = 0` in both and explicitly
`NOT_APPLIED_ON_THE_DAY_PATH` in the corrected one. No window is dropped for gaps, so the
"all tranches" total is invariant BY CONSTRUCTION and agreement to the cent was guaranteed.
(3) The 62.37 % is NOT IN THE ARTIFACT; the caveats are structural siblings of the numbers, so
the detachment risk is in the PROSE, not the file.**

---

## (1) DE 189 — CLOSED, AND IN RULE 35'S SHAPE

**Note first: `51ebf13` is NOT on `origin/mm-research`** — it sits unpushed in the shared tree
(three commits ahead, tree clean). I drove it from the shared tree, whose HEAD is that commit.

The no-split book now returns, as a required field on every result:

```
complement_leg.status  = COMPLEMENT_LEG_COUNTED_AND_DISCARDED__L0_COMPARISON_IMPOSSIBLE_ON_THIS_BOOK
complement_leg.the_reconciliation_can_run   = False
complement_leg.n_tranches_discarded_at_build = 23765
complement_leg.what_a_consumer_must_do = "read this before quoting a placement-latency
    comparison. On a DISCARDED or UNDECLARED book the comparison has no second term, and a
    reconciliation that 'passes' there has checked half of what its name claims"
complement_leg.rule = "35 -- a limit that lives only in a declaration does not bind the result"
```

**A status a reader resolves, plus a machine-resolvable boolean, plus the consumer
instruction** — not a better-worded `split_book`.

**And it tests the PROPERTY, driven three ways from the receipt:**

```
09-04 with the split block            -> COMPLEMENT_LEG_KEPT__L0_COMPARISON_AVAILABLE
09-03 with `n_tranches_dropped`       -> COUNTED_AND_DISCARDED__L0_COMPARISON_IMPOSSIBLE
09-04 with the split block REMOVED    -> COMPLEMENT_LEG_UNDECLARED__THE_RECEIPT_DOES_NOT_SAY
09-03 with `n_tranches_dropped` GONE  -> COMPLEMENT_LEG_UNDECLARED__THE_RECEIPT_DOES_NOT_SAY
```

**The status follows the receipt, not a constant.** One conservative edge, not a defect:
`n_tranches_dropped: 0` still reads `COUNTED_AND_DISCARDED`, where an empty complement is
arguably a degenerate-but-available comparison. It errs toward refusing to claim one.

## (2) THE INFERENCE — IT IS THE SECOND CASE, AND I WOULD WITHDRAW THE SENTENCE

**The measurement, from the two receipts:**

| | retracted (`__L250ms`) | corrected (`__L250ms__EV21`) |
|---|---|---|
| generations | **358,107** | **358,108**  (+1) |
| `n_tranches_dropped` / `before_L` | **31,471** | **31,471** |
| split `n_tranches_valued` | — | **57,850** = 26,379 + 31,471 |
| `BINANCE_GAP_EXCLUDED` | **0** | **0**, `NOT_APPLIED_ON_THE_DAY_PATH` |

**The tranche multiset is IDENTICAL across the two builds.** The era fix moved the generation
count by exactly **one**, and moved **zero tranches** — and gap-based exclusion is explicitly
*not applied on the day path*, so no window's tranches leave the set.

The published total is Σ over the tranche set of the trades leg, plus the per-slug residual
(net shares × settlement), with winners from **Chainlink**, which the era fix never touches.
**Both inputs to the total are invariant, so the total is invariant by construction. Agreement
to the cent was guaranteed and carries no information about the correction.**

**What the agreement DOES establish, and is worth keeping:** the era fix did not disturb the
tranche set — a real regression check, and a useful one. **What it does not establish is
anything about whether contamination was "structural rather than arithmetic", because the
quantity that agrees was never a function of the era.**

**What WOULD be informative** is a quantity the era fix touches: the **generation partition**
(358,107 → 358,108 here; 313,114 → 313,149 on 09-03) and anything computed **per generation** —
decision counts, cancels, the arm's selection. Those carry the comparison; the day total does
not.

## (3) CAN 62.37 % BE READ AS THE EFFECT?

**Not from the artifact — the number is not in it.** 62.372522 % is a derived ratio
(63,740.78 / 102,193.69); a reader must divide `DROPPED.total_cents` by
`ALL_TRANCHES.total_cents`, and **both sit in the same block as three caveats**:

```
UPPER_BOUND        DROPPED is an UPPER BOUND on the latency effect, not the effect …
HOW_IT_MUST_BE_SAID  … UNDER THE ASSUMPTION THAT EVERY ONE WOULD HAVE FILLED — an upper bound
SCOPE              ONE DAY IS A POINT ESTIMATE WITH NO INTERVAL (rule 8); >= 5 complete
                   UTC days before 'the latency effect is real' may be said.
```

**So the artifact is structurally sound on this** — unlike the four earlier instances, the
caveat is a sibling of the number, not in another document. **The detachment happens in the
PROSE**: the moment "62.37 %" is written into a summary, the block stays behind. **That is
where the guard belongs — a share quoted without `UPPER_BOUND` and `SCOPE` beside it is the
failure, and the fix is a reporting rule, not another field.**

## ROUTED

1. **Coordinator — withdraw "the era contamination was structural, not arithmetic."** The
   agreement is guaranteed by an unchanged tranche set. Replace it with the claim the data
   supports: *the era fix moved the generation partition by one and left the tranche set
   untouched, so the day total is unchanged — a regression check, not evidence about
   contamination.*
2. **Coordinator — the informative comparison is per-generation**, not the day total.
3. **DE 189 is CLOSED**; `51ebf13` still needs pushing.
4. **Reporting rule — 62.37 % may not travel without `UPPER_BOUND` and `SCOPE`.** The
   artifact does this correctly; prose is where it breaks.
