# REVIEW 178 — day one: your numbers reproduce, your instinct on (2) is right and the mechanism is not the one you named, and the framing drifts in one place

**REV 135, 2026-09-11T05:54:50Z** (clock read separately). Read-only: no lock, no heavy
unit, nothing written under `data/`. Tip `68479dd`. **Every number below is recomputed from
`fwd/de_settle_ckpt_2026-09-07_*.jsonl` and `fwd/de_settle_result_20260907_*.json`, not
taken from your pane.**

## 0. YOUR FIGURES REPRODUCE EXACTLY

Book `0815cad74f53f118…` (`be_daybook_20260907_btc__L250ms__FWD1.pkl`), 500 draws,
baseline −4,437.7396c, `be_module` = params v29's. CONDVALUE `observed_D −11,017.712`,
`p 0.31736`; HAZARD `observed_D +5,256.177`, `p 0.65868`. `interval: None` — correct at
G=1 under rule 8. **I reproduced both p-values from the checkpoints to the count: 158 and
329 draws at-or-beyond.** Nothing in your reading is a misread field.

## 1. CHALLENGE (2) — YOU ARE RIGHT THAT SOMETHING IS OFF, AND THE MECHANISM IS *ZERO*, NOT THE NULL MEAN

The declared statistic is, at `de_settlement_control_aggregate.pooled`:

```python
n_ge = sum(1 for v in null if abs(v) >= abs(D_arm))
```

**Two-sided on |D| around ZERO — not around the null mean.** That single fact explains the
whole apparent inconsistency, and it is not the symmetry you guessed.

| CONDVALUE_X_SKEW | value |
|---|---|
| observed D | **−11,017.7** |
| null min / q25 / median / q75 / max | −10,923.5 / 3,606.6 / 7,569.1 / 12,286.8 / **24,659.1** |
| null mean / sd | 7,888.2 / 6,430.1 |
| draws ≤ 0 | **55 / 500** |
| **declared** `#{\|D_null\| ≥ \|D_arm\|}` | **158 → p = 0.3174** |
| one-sided LEFT `#{D_null ≤ D_arm}` | **0 → p = 0.0020** |
| location-centred (NOT the declared test) | **0 → p = 0.0020** |
| **observed's percentile within its own null** | **0.0 %** |

**The 158 draws are almost all large POSITIVE ones.** The null sits at +7,888 with 445 of
500 draws above zero and a tail to +24,659; the arm is far to the LEFT. An absolute-value
test around zero cannot see direction, so it scores a record-low excursion as ordinary
because the null routinely produces equally large excursions **the other way**.

**So: the p-value is internally consistent — I reproduced it — and it is measuring
MAGNITUDE where the reading assumes POSITION.** Those coincide only when the null is
centred at zero, and this null is centred 7,888c away from it.

**THE RULING, AND IT CUTS AGAINST THE ARM SO SAY IT PLAINLY: the declared construction
STANDS.** It was committed before any draw (`b72e329`), and rule 11 forbids changing a
statistic because the number looks wrong. Two things follow, both of which must be carried
rather than fixed:

1. **Disclose the property, do not repair it.** *"The declared two-sided statistic is on
   |D| about zero. The null is centred at +7,888c, so a large negative arm excursion is
   scored against the null's large positive ones. CONDVALUE's day sits below all 500
   draws (0th percentile) and the declared statistic returns 0.3174."* Both sentences are
   true; the first is what the test uses.
2. **It cannot manufacture a pass, which is why it survives.** The construction under-detects
   arm *harm*, never arm *skill* — failing to see a bad day is not a false positive for the
   thing being tested. Under my own conservative/anti-conservative rule this is the
   survivable direction: it makes the arms look better than they are, and correcting it
   after seeing the data would be choosing after seeing even though the correction would
   hurt them. **You may not quote the 0.0020.** It is not the declared statistic.

## 2. CHALLENGE (1) — THE NULL MEAN IS A POOR COMPARATOR, AND THE RIGHT ONE IS HARDER ON CONDVALUE AND EASIER ON HAZARD

A mean summarising a distribution that spans −10,924 → +24,659 with sd 6,430 carries little.
**Use the percentile.** It changes both readings, in opposite directions:

- **CONDVALUE: 0.0th percentile — below every one of 500 matched-random draws.** That is
  *stronger* than "underperformed the null mean", not weaker.
- **HAZARD: 36.4th percentile** (null min −9,415.7, mean +7,940.5, max +25,948.6; one-sided
  left p 0.3653). That is **"below the null's centre", not "underperformed"**. Lumping it
  with CONDVALUE overstates it.

**Your sentence "BOTH ARMS UNDERPERFORMED MATCHED RANDOM" is right about CONDVALUE and too
strong about HAZARD.** The defensible form: *on 09-07, CONDVALUE's D was below all 500
matched-random draws; HAZARD's sat near the null's 36th percentile.*

And the descriptive fact underneath is worth stating because it is the one a reader will
remember: **on 09-07, cancelling at RANDOM beat not cancelling at all by ~7,900c on average,
while CONDVALUE cancelled 26,264 generations and finished 11,018c BELOW the no-cancel
baseline.** That is a real and legible statement about this day.

## 3. CHALLENGE (3) — ONE DAY LICENSES "ON THIS DAY", AND I HAVE TO CORRECT MYSELF UPWARD

"Both arms underperformed random" is licensed **for 09-07 only**, in the percentile form
above. G = 1; the cluster unit is the UTC day; the estimand pools seven.

**But I am correcting my own instinct, not just yours.** I came to this expecting to say
"day one is nearly a coin flip". **It is not.** Under a null in which the arm behaves like a
matched-random canceller, a negative day has probability ≈ 55/500 = **11 %**, not 50 % —
this null is 89 % positive. CONDVALUE did not merely land negative; it landed below 500 of
500 draws. **Day one is genuinely extreme evidence against CONDVALUE.** Saying otherwise to
soften it would be the same error in the other direction.

What that does *not* license is a verdict, for the reason the design already recorded: the
per-day effect is not consistent — the freeze's own pre-registration names 09-05's
−4,941c sign flip — and one cluster from a wide day-to-day distribution is what the seven-day
design exists to average.

## 4. HOLDING YOU TO THE RULE — ONE DRIFT, AND IT IS ONE PHRASE

Your framing is closer to the line than it needs to be in exactly one place:

- **"reproducing step 2's verdict" — DROP IT.** Step 2 ran on 09-04/05/06 **under the
  pre-correction policy that has since been retracted** (`03dbc1e`). A different population
  and a different decision rule cannot be "reproduced" by this day. *"Consistent with"*
  overstates it too at G=1. **This is the phrase that carries a reader to NO_EFFECT.**
- **"CONDVALUE's negative day appears to end it" — TRUE, but say WHAT it ends.** At G=7 the
  tolerance is 0, so one negative day makes **conjunct (a) unattainable**. That is futility
  of the test, not absence of effect — and it is exactly DE's own field:
  *"A test that cannot attain its own threshold has measured nothing about the arms — it has
  measured the calendar."* The user took N=7 knowing 7-for-7 was required (amendment 9).
- **The rest of your framing holds** and is more careful than it needed to be.

**The sentence stands as written before any number existed: NOT_ESTABLISHED_AT_THIS_POWER,
never NO_EFFECT — and day one is not the test.** The asymmetry cuts both ways exactly as you
say: a failure being stronger evidence against the family does not make one day a failure of
the test, and CONDVALUE's futility on the sign gate is a fact about the calendar the design
chose.

## 5. WHAT I WOULD GIVE THE USER

> **Day one (09-07) is in and it is bad for CONDVALUE.** Its settlement delta was −11,018c
> against a no-cancel baseline of −4,438c, and it sits **below all 500 matched-random draws**
> — on this day, cancelling at random beat it, and beat not cancelling at all by ~7,900c.
> **HAZARD was +5,256c, near the null's 36th percentile — unremarkable, neither skill nor
> harm.** **The declared two-sided statistic returns p = 0.3174 and 0.6587**, because it
> compares |D| about zero while this null is centred at +7,888c; the construction was fixed
> before any draw and is not being changed. **At N=7 the design tolerates zero negative days,
> so CONDVALUE can no longer clear the sign gate** — that is the calendar the test chose, not
> a measurement of the arm. **One day is one cluster of seven. The verdict is
> NOT_ESTABLISHED_AT_THIS_POWER, never NO_EFFECT.**

## 6. SCOPE

Recomputed from the two 500-draw checkpoints and the two result artifacts: both declared
p-values to the exact count, the null order statistics, the negative-draw counts, the
one-sided-left counts, the location-centred contrast, and both percentiles. The declared
construction read at `de_settlement_control_aggregate.pooled`. **Not done:** DE's formal
fields, which are still owed and which I have not pre-empted; anything about days two to
seven, which do not exist.
