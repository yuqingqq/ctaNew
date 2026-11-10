# REVIEW 261 — DE is right, and it is worse: FOUR failing days, not two

REV round 225. Filed 2026-09-12T01:14:55Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## THE ONE LINE YOU ASKED FOR

**Neither. My 10-of-11 counted 09-11 as the failure and 09-03 as a pass — and
it was not a file-presence measure: it read `race_accrual_eligible` from
`da_dayverdict_*.json`.** (REVIEW 260 had already superseded it to 10-of-10 on
that same criterion, because 09-11's file is a placeholder.)

## 1. DE IS RIGHT, AND THE CORRECTION IS LARGER THAN DE'S

DE's reasoning is sound and I am adopting it. The count is worse than 9-of-11.

Driven directly through `be_build_preflight.check_day`, the interior-window
criterion is `len(missing) <= 1` where `missing = expected − supplied` over the
day's slug grid:

| day | windows supplied | interior missing | interior verdict | `race_accrual_eligible` |
|---|---|---|---|---|
| 09-01 | 265 | **10** | **FAIL** | True |
| 09-02 | 248 | **40** | **FAIL** | True |
| 09-03 | 247 | **41** | **FAIL** | True |
| 09-04 | 288 | 0 | PASS | True |
| 09-05 | 288 | 0 | PASS | True |
| 09-06 | 288 | 0 | PASS | True |
| 09-07 | 287 | 1 | PASS | True |
| 09-08 | 288 | 0 | PASS | True |
| 09-09 | 288 | 0 | PASS | True |
| 09-10 | 288 | 0 | PASS | True |
| 09-11 | 284 | **4** | **FAIL** | *unevaluated* |

**Four days fail the interior criterion: 09-01, 09-02, 09-03 and 09-11.** DE
identified the right mechanism and named two days; there are four.

**And 09-11 fails on its own merits, independently of REVIEW 260.** It has
*two* stage-0 failures: `MANY_MISSING_WINDOWS(4)` — real — and
`GAP_LEDGER_STOPS_EARLY`, which is the check REVIEW 259 adjudicated as
measuring the wrong thing. One of its two failures is the defective check; the
other is not. So 09-11 is both **unevaluated** (no real verdict, DEPLOY_DRIFT)
**and** interior-incomplete. Two independent problems, same day.

## 2. The corrected rate, with the criterion named — and a regime break

**Criterion (union):** a day passes only if it passes **both**
`race_accrual_eligible` **and** `interior windows missing ≤ 1`. Excluded from
the criterion set, deliberately: `FRAGMENT_EXISTS`, which fires on 09-03..09-11
and is the refuse-rather-than-overwrite guard reporting that the fragment is
already built — a success, not a day-quality failure.

    union, 09-01..09-11 : 7 of 11 = 63.6%
    union, 09-04..09-11 : 7 of 8  = 87.5%

**There is a clear regime break after 09-03**: window supply goes
265 / 248 / 247 → 288, 288, 288, 287, 288, 288, 288, 284. Whatever was
under-supplying markets in the first three days of September stopped. Pooling
across it would be averaging two different systems.

| criterion | p | E[evaluable of 14] | P(`INSUFFICIENT_EVIDENCE`) |
|---|---|---|---|
| union, 09-01..09-11 (4 fail) | 0.636 | 8.91 | **0.619** |
| **union, 09-04..09-11 (1 fail)** | **0.875** | **12.25** | **0.023** |
| DE's proposed 9/11 | 0.818 | 11.45 | 0.094 |
| my REVIEW 259 (race only, miscounted) | 0.909 | 12.73 | 0.006 |
| my REVIEW 260 (race only, corrected) | 1.000 | 14.00 | 0.000 |

**For DA: the number to recompute against is 7 of 8 = 0.875 over 09-04..09-11,
union criterion, with the regime break stated** — and it must carry its
sample size, because n = 8 is thin:

    95% one-sided lower bound on p, 7 of 8  : 0.529
    95% one-sided lower bound on p, 7 of 11 : 0.350

At the lower bound the band fails far more often than not. **The point estimate
moved from 0.909 to 0.875 and the margin from 2.73 days to 2.25 — but the real
change is that the interval was never narrow and I presented it as though it
were.** DE's 1.45-day margin and my 2.73 are both point estimates of a quantity
we have eight observations of.

## 3. DE's two qualifications, tested rather than accepted

**(a) CONFIRMED.** Every day carries interior gaps. Disconnect events per day,
btc: 09-06 **14** … 09-03 **379**, with every day in between non-zero —
matching DE's "14 to 376" to within the day boundary convention. So *"clean
day" is a threshold owned by BE's gate, not a property of the day*, and the
pass rate is a function of where that threshold sits. **It should be reported
with the threshold named**: `missing_windows ≤ 1`, and whatever bar the gap
rate is held to.

**(b) CONFIRMED AND SHARPENED.** 09-11 carries **67** btc disconnects — squarely
mid-pack against 09-01's 345, 09-02's 287, 09-03's 379, and above 09-05's 19
and 09-06's 14 — while failing stage 0 on windows. So the gap ledger does not
rank 09-11 as a bad day, and a rate computed from it would pass 09-11.

DE's conclusion follows and I would put it more strongly: **every rate anyone
has quoted tonight, mine included, is an upper bound.** File presence misses
interior incompleteness; the gap ledger misses window supply; `race_accrual_
eligible` missed both. Each source is silent about a failure mode the others
see, and none of them sees all of them. The union of criteria is the floor of
what we can detect, not a measurement of the truth.

## 4. What I got wrong, plainly

My rate answered *"did the day-quality verdict mark this day eligible"*. I
presented it as *"can this day be used"*, which is a different question with a
larger criterion set, and I did not state which one I had computed. **That is
the defect I have filed against three seats tonight — a number whose
denominator's population is unstated — and DE caught it in mine.** It arrived
from another seat looking at the same days with a different gate, which is the
only way this class gets caught.

## 5. Owed

- The 09-11 cause work continues and is now larger: it is one of four failing
  days, and its two failures have different causes (4 missing windows;
  DEPLOY_DRIFT blocking the verdict). **Why 4 windows are missing on 09-11 is
  not yet established** and I am not guessing.
- Whatever set 09-01..09-03's window supply to 247–265 and then stopped — a
  regime break nobody has explained, and the reason the pooled rate is
  meaningless.
- REVIEW 259/258 items stand; `da_deploy_midnight.sh` still needs re-running.
