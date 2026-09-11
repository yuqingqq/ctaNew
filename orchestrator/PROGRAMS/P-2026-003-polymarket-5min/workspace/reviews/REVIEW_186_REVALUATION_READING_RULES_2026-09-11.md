# REVIEW 186 — reading rules for the re-valuation's ΔD, written with ZERO cells visible

**REV 143, 2026-09-11T07:10:33Z** (clock read separately). Read-only. **Written before BE's
rebuilt book exists and before any cell of DE 252's table exists.** Every boundary below is
derived from quantities that existed before the rebuild: |D| = 11,017.71c, 143.8 altered
gap-seconds, 27 windows, 287 supplied windows, 86,100 s of replay time.

**THE TWO REFERENCE LEVELS, from which every rule is derived:**

```
interval-scoped uniform   143.8s / 86,100s x |D|      =    18.40c
window-scoped   uniform   27x300 / 86,100s x |D|      = 1,036.50c
|D| (day one, CONDVALUE)                              = 11,017.71c
```

---

## RULE 1 — THE SCOPING VERDICT, AS BANDS OVER |ΔD|

| band | predicate | verdict |
|---|---|---|
| **INTERVAL_SCOPED_CONSISTENT** | `abs(dD) <= 184c` | k_int ≤ 10×, k_win ≤ 0.18×. The fix altered replay **input during gap intervals**. |
| **AMBIGUOUS_SCOPE** | `184c < abs(dD) < 518c` | Both mechanisms reachable. **Assign no scope. Report both k's.** |
| **WINDOW_SCOPED_CONSISTENT** | `abs(dD) >= 518c` | k_win ≥ 0.5×. A per-window boolean gated whole windows. |

**Why 184c:** 10 × the interval reference. Q-DA-58's measured fill concentration is **8.2×**,
so up to ~10× is *inside what this programme has measured*; above it the interval mechanism
needs a concentration nothing here supports.
**Why 518c:** half the window-scoped uniform. A window mechanism that alters all 27 windows
sits near 1× under uniformity; a factor-2 shortfall is ordinary (partial, not total,
alteration). Below 0.5× both mechanisms are strained.

**THE TWO NUMBERS YOU ASKED FOR:**
- **200c → AMBIGUOUS_SCOPE** (k_int 10.9×, k_win 0.19×). Just over the line, leaning
  interval. **Do not call it interval-scoped;** report `k_int=10.9, k_win=0.19` and say the
  band.
- **800c → WINDOW_SCOPED_CONSISTENT** (k_int 43.5×, k_win 0.77×). 43× concentration on 143.8
  seconds is not supported by any measurement here; 0.77× of the window uniform is ordinary.

**The verdict is about the MECHANISM, never about the arm.** `INTERVAL_SCOPED_CONSISTENT` is
not "small so ignore it" and `WINDOW_SCOPED_CONSISTENT` is not "large so worry" — they name
which of my two REVIEW 184 ends the data sits at, and that is all.

## RULE 2 — WHAT EACH FIRING PATTERN SAYS

Publish **Δcents per gap-second per window** beside every row; it is the discriminator.

| pattern | reading |
|---|---|
| **Aggregate fires, no single window** | Spread across many windows, each < 110c. **The era fix working as described**, systematically. Consistent with window-scoped. Least alarming. |
| **A single window fires, aggregate does not** | **OFFSETTING MOVES — the pattern the per-window tripwire exists for.** An era fix should not produce large opposite-signed moves. **Investigate before accepting any new D.** |
| **Both fire, 20:45 dominates** | 20:45 holds 36.2 s / 6 gaps = **25 % of all gap time**, so concentration *there* is the expected interval-scoped shape. **Benign-looking and also the shape a rescue would take** — read it with the sign, never alone. |
| **Both fire, flat across 27** | Window-scoped and systematic. Legitimate, and it means day one's original D was materially wrong. |

**The ratio test that separates the third row from a finding:** if 20:45's **Δcents per
gap-second** is comparable to the other 26, its size is explained by its gap time. **If its
RATIO is an outlier, the size is not explained by gap time and that is a finding**, whatever
the aggregate does.

## RULE 3 — `SIGN_CHANGE_HALT`: A RESCUE IS NOT CREDIBLE AT ANY ΔD, AND I SAY SO NOW

A sign change needs `dD >= +11,017.71c` — **599× the interval reference, 10.63× the
window-scoped uniform.**

**The decisive pre-declared fact: at 8.2×, the highest concentration this programme has ever
measured, the window-scoped mechanism reaches 8,499c = 77 % of |D|. THAT IS NOT A SIGN
CHANGE.** A sign change requires exceeding every concentration ever measured here, in a
correction that changes **gap accounting only** — not the policy, not theta, not the models.

> **RULING, TYPED BEFORE THE NUMBER: a sign change on this rebuild is a finding about the
> REBUILD, not a corrected result. The new D is NOT adopted. Day one's value becomes
> UNRESOLVED, not positive.** Day one's −11,018 sat **below all 500 matched-random draws**;
> a gap-accounting correction that moves a 0th-percentile observation across zero is more
> likely to have changed something it does not claim to change than to have revealed the day.

**AND THE ESCAPE, so this is falsifiable and not a refusal to believe data:** the halt is
released, and the new D adopted, **only** on all four — (a) the 27-row table sums to ΔD with
`abs(residual) < 1c`; (b) **no** window's Δcents-per-gap-second is an outlier against the
other 26; (c) a diff of the two books shows **identical generation sets, fill sets and score
streams outside the 27 windows**; (d) the two books' `producing_code` closures differ **only**
by the era fix. **Absent any one of those, the sign change indicts the fix.**

*(Direction note: this rule disbelieves a result that would HELP the arm. That is deliberate —
the conservative direction is the one that survives, and I am applying it against the outcome
I would otherwise be accused of wanting.)*

## RULE 4 — THE RESIDUAL

`residual = (sum of the 27 rows) − dD`.

| band | verdict |
|---|---|
| `abs(residual) < 1e-6 c` | **ROUNDING.** Accept. Float64 over 27 terms of magnitude ≤1e4 gives ~1e-11c; anything under 1e-6c is arithmetic. |
| `1e-6 c <= abs(residual) < 1 c` | **UNEXPLAINED_SMALL.** Not floating point. Report; do not halt. Most likely a quantisation or display artifact. |
| `abs(residual) >= 1 c` | **FINDING — HALT.** |

**Why ≥1c halts, and it is the strongest reading of this field: a contribution landed in no
row means ΔD includes a change OUTSIDE the 27 gap-bearing windows.** That does not merely
fail bookkeeping — **it invalidates both reference levels, and therefore Rules 1, 2 and 3
with them**, because every band above is computed on the assumption that only the 27 changed.
**A residual ≥1c retires this entire filing until the extra contribution is located.**

## RULE 5 — WHAT NONE OF IT LICENSES

- **The tripwire is about 09-07's GAP TREATMENT. It is not about either arm's skill.** No
  value of ΔD, in any band, is evidence for or against CONDVALUE or HAZARD.
- **A clean rebuild does not make −11,018 more credible as evidence about the arm** — it makes
  it more credible as a **measurement**. Those are different claims.
- **A dirty rebuild does not rescue the arm; it impeaches the DAY.**
- **One day. One cluster of seven.** Nothing here touches conjunct (a)'s futility, the N=7
  floor (7-for-7 required), the 69-candidate selection history, or the second-attempt problem.
- **Uniformity still carries the rule-11 defence** (DA 231), not this table. A tripwire
  measures magnitude; it does not make a correction applied to one seen day into a correction
  applied uniformly.

## THE CEILING — THE STRONGEST SENTENCE EACH OUTCOME PERMITS, BOTH WRITTEN NOW

**CLEAN REBUILD** (`abs(residual) < 1c`, no `SIGN_CHANGE_HALT`, no unexplained single-window
ratio outlier):

> *The era correction moved 09-07's settlement delta by **[ΔD]c** — **[k_int]×** the
> interval-scoped reference and **[k_win]×** the window-scoped one, band **[BAND]** — with the
> 27-row table summing to ΔD at residual **[r]**. **Day one's D stands as measured at
> [new D].** The day's gap accounting was **[not] materially wrong**; this establishes nothing
> about either arm's skill, and 09-07 remains one cluster of seven with conjunct (a) already
> unattainable for CONDVALUE.*

**DIRTY REBUILD** (any halt, or a residual ≥1c, or an unexplained ratio outlier):

> *The era correction moved 09-07's settlement delta by **[ΔD]c**, which **[flipped the sign /
> left a residual of [r] / left window [w] an unexplained ratio outlier]**. **That is a finding
> about the correction, not a corrected result.** The new D is **NOT adopted**; day one's
> value is **UNRESOLVED** pending a diff of the two books' generation, fill and score sets
> outside the 27 windows. **Nothing about either arm's skill is established or withdrawn by
> it**, and the original −11,018c stands as the last value produced by a pipeline whose
> provenance is established (REVIEW 180).*

---

**Filed before the book. If the book beats this filing to disk, this rule set is void and must
be re-declared by someone who has not seen the table** — a rule written after seeing is worse
than a late rule, and I will say so rather than backdate.
