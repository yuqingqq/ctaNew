# REVIEW 226 — **THE RESULT STANDS.** Both arms are futile at day two of seven, the arithmetic reproduces, and the forward test ends in the pre-written fail sentence — with a multiplicity caveat that would have bitten even a perfect run

**REV, 2026-09-11T15:57Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Drives on scratch and in my own worktree (restored).

## 1. FIRST, A CORRECTION TO MY OWN REVIEW 224

I wrote in REVIEW 224 §2, and repeated in REVIEW 225, that **CONDVALUE "remains live and the
bar is unanimity: all five remaining days must also be negative"**. **That was wrong, and
wrong in the direction that overstates what was left.** I read the day-cluster gate as
sign-*consistency* in either direction, so I treated an all-negative run as the extreme that
passes. The declared design requires **positive** days: `tolerance_negative_days = 0`, and a
non-positive day is a minority day counted against the arm. Under the design as declared,
**CONDVALUE was already dead at day one**, before I ever called it live.

My HAZARD conclusion was right and right for the right reason. My CONDVALUE conclusion was a
misreading of the alternative, and the emit had it correct. Recorded here beside the ruling so
the next reader of 224/225 finds the correction, not the claim.

## 2. THE FUTILITY ARITHMETIC — RECOMPUTED, NOT READ

```
two-sided exact sign test, G = 7, minority = the non-positive days already seen
  CONDVALUE  n_neg 2  ->  2*(C(7,0)+C(7,1)+C(7,2))/2^7 = 2*29/128 = 0.453125
  HAZARD     n_neg 1  ->  2*(C(7,0)+C(7,1))    /2^7 = 2* 8/128 = 0.125
  unanimity  n_neg 0  ->  2*C(7,0)/2^7                        = 0.015625
threshold 0.025 = Holm's first step at alpha 0.05, m = 2
```

**Both reproduce the record exactly**, and both exceed the threshold: **FUTILE, both arms**,
with five days still unscored. The record's own `why` lines say it in the same terms.

**And the per-arm signs are right this time.** `per_day_D_by_arm` carries:

```
CONDVALUE_X_SKEW        2026-09-07  -14645.078818   2026-09-08  -49303.579891
HAZARD_OVER_SKEWED_REF  2026-09-07   +4925.363903   2026-09-08  -23977.998804
```

HAZARD's `negative_or_zero_days` is `["2026-09-08"]` alone — **the day-keyed collapse I filed
in REVIEW 225 is fixed**, the 09-07 positive is correctly excluded, and the shared per-day map
DE found while fixing it is the reason the old block could not have been right. `ANY_ARM_ALREADY_DEAD`
and `EVERY_ARM_ALREADY_DEAD` are both `true`; `floor_at_the_G_ACHIEVED_SO_FAR` is `0.5`
(2/2²) against `0.015625` at the declared G.

## 3. (a) — CONFIRMED, BOTH ARMS, EXACTLY

```
CONDVALUE_X_SKEW  landed -14645.078818000005  rebuilt -14645.078818000005   identical
HAZARD_OVER_SKEWED_REF     +4925.363903000005          +4925.363903000005   identical
book 887a97eb41e9 (a rebuild)   oracle 172a93073ead (a later ledger)   n_draws 500   DESCENDANT both
```

Both arms reproduce to every printed digit on a freeze-built book and a later oracle state. p
moves on both (0.181637→0.169661, 0.728543→0.706587) and the reason is structural:
`seed_for(book_sha, arm)` makes the null a function of the book. *(The combined (a) record
still does not exist in `fwd_a_0907_rebuild/`; I compared the two cells against day one's
landed combined record.)*

## 4. RULING: **THE RESULT STANDS**

The record is day two's result. Criteria (1)–(4) were met and licensed in REVIEW 225; the two
objections I raised there are answered: the tally is now keyed on the **arm**, and the futility
block is computed per arm with its cause selected by the computation rather than typed.

**Two residuals, neither of which touches the verdict:**

- **`STOP_ADVICE: null` while `EVERY_ARM_ALREADY_DEAD: true`.** The field that exists to say
  *stop* is empty at the moment every arm is dead. It costs nothing to fill and it is the one
  place a reader looks.
- **The unconditional per-window table is still 43 rows with 0 non-null `gap_seconds`** —
  REVIEW 190's day-one defect, now on day two. It does not bear on the result; it bears on the
  table's claim to be a record.

## 5. THE OUTCOME, IN MY WORDS

**Under the design as declared and frozen before any forward day was valued, the forward test
is over and neither arm passes.**

> **`NOT_ESTABLISHED_AT_THIS_POWER`, both arms.** `CONDVALUE_X_SKEW` and
> `HAZARD_OVER_SKEWED_REF` cannot reach the declared threshold on the remaining five days:
> their best attainable day-cluster p values are **0.453125** and **0.125** against a Holm
> step-one threshold of **0.025**. The test stops for futility at G = 2 of 7.

**What that sentence does and does not say.** It says the test could not establish an effect at
the power the design bought. **It does not say the arms have no effect** — futility is a
statement about the instrument's reach, not about the world. Stopping now cannot inflate
anything: as the record puts it, futility stopping only ever reduces the chance of declaring
success.

**Separately, and descriptively:** on the two days scored, both arms **lost** against their
zero-cancel baselines — CONDVALUE −14,645.08 and −49,303.58 cents, HAZARD +4,925.36 then
−23,978.00. That is what the numbers say; it is not what the test tested, and it carries none
of the test's protection against selection.

## 6. THE MULTIPLICITY CAVEAT — AND IT IS SHARPER THAN A CAVEAT

The two arms were carried forward from a screen of **69 candidates** (REVIEW 131). Holm's
`m = 2` covers the two arms carried, **not the 69 looked at**. Price the design's own best case
against the screen:

```
the design's floor at G = 7 (unanimity, two-sided)   0.015625
x m = 2   (Holm, as declared)                        0.031250   <  0.05   would have passed
x 69      (the candidates actually screened)         1.078125   >  1      cannot pass at all
```

**Even a perfect run — seven days, every one positive, both arms — could not have survived a
correction over the selection that produced the arms.** The design's minimum attainable p is
larger than 1/69. So the forward test was, by construction, incapable of delivering evidence
that outlives its own screen, whatever the days had shown.

That is not a criticism of stopping now, and it does not change today's verdict. **It means
the fail sentence is the weaker of the two things that were true**: the test failed on the
days, and the test could not have succeeded on the multiplicity. Both belong in whatever the
programme writes down as the outcome, and the second one was knowable — and was written down —
before any day was valued.

## SCOPE

Closed over: the superseding record read key by key, including the whole `futility` block,
`per_day_D_by_arm`, the flags and the floors; the futility arithmetic recomputed from
first principles for both arms; both (a) cells compared to day one's landed cells; the
multiplicity arithmetic computed. **Not closed over:** (a)'s combined record, which does not
exist; the 43 null window rows; conjunct (b), the matched-random null, which futility on (a)
makes moot for these two arms but which is unexamined here.

## ROUTED

1. **DE — `STOP_ADVICE` is null while every arm is dead** (§4).
2. **DE — the gap table is still 43 rows of nulls** (§4), REVIEW 190 unfixed on a second day.
3. **Coordinator — the outcome sentence is §5, and §6 belongs with it.** The multiplicity
   result was computed before the forward test began; it should be reported beside the fail,
   not after it.
4. **Me — REVIEW 224 §2 and REVIEW 225 carried a wrong CONDVALUE conclusion** (§1), corrected
   here.
