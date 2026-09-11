# REVIEW 192 — 09-08's ledger side computed BEFORE BE's list exists: 43 windows, and the expected discrepancy mechanism is PRE-REFUTED

**REV 151, 2026-09-11T07:44Z** (clock read separately). Read-only. **`be137_gap_windows_20260908.json`
has NOT landed** — only 09-07's exists (06:57:39Z); `be140frag0908` started ~07:35 and the
heavy lock is held. **So I computed my side first.** Independence by ORDERING, not by claim
(rule 38): this measurement cannot have been anchored on BE's list because BE's list does not
exist.

## 1. MY LEDGER SIDE — 09-08, AND 09-09/09-10 PRE-MEASURED WHILE I WAS THERE

BTC `gap_closed` intervals, `gap_start_ns`→`gap_end_ns` (the data-stop convention REVIEW 185
established the pipeline inherits), overlapped to the 300 s grid **in integer nanoseconds**:

| day | intervals | windows touched | gap-seconds | boundary-spanning |
|---|---|---|---|---|
| 09-07 *(control)* | 35 | 28 | 145.306 | **0** |
| **09-08** | **60** | **43** | **264.750** | **0** |
| 09-09 | 49 | 42 | 163.695 | **0** |
| 09-10 | 47 | 41 | 91.422 | **0** |

**09-07 reproduces REVIEW 185 exactly** (35 / 28 / 145.306), which is the control on the
instrument. **09-08's 43 windows matches BE 133's 43** — computed independently and before
BE's list. **09-09 and 09-10 you listed as unmeasured; they are measured now.**

09-08's worst windows: **20:45 44.681 s (n=6)**, 14:45 34.871 (5), 15:55 25.661 (1),
19:00 24.293 (4), 19:10 12.692 (2), 18:00 11.256 (1).

## 2. **YOUR EXPECTED DISCREPANCY MECHANISM IS ABSENT — ON ALL FOUR DAYS**

> *"(b) the wall-clock census already found the replay drops boundary-spanning gaps — expect
> 09-08's ledger to carry gaps BE's list lacks"*

**There are no boundary-spanning gaps to drop.** Every one of the 191 BTC intervals across
09-07..09-10 lies wholly inside a single 300 s window. **So if BE's 09-08 list differs from
43, the boundary-span hypothesis is pre-refuted for these days and the cause is something
else.** That is worth knowing before the comparison rather than after it.

**AND THE ZERO IS ONLY A RESULT BECAUSE THE DETECTOR WAS FALSIFIED FIRST** (rule 15):

```
[PASS] KNOWN-BAD  a gap straddling a 300s edge          -> spans=True
[PASS] KNOWN-BAD  a gap covering two whole windows      -> spans=True
[PASS] GOOD       a gap wholly inside one window        -> spans=False
[PASS] EDGE       a gap STARTING exactly on a boundary  -> spans=False
[FAIL] EDGE       a gap ENDING exactly on a boundary    -> spans=True   <- MY DEFECT
```

**The defect is mine and I name its DIRECTION, which is what makes the zero usable:** my first
pass used `e - 1e-9` in float seconds, and at epoch magnitude ~1.8e9 that epsilon is **smaller
than a float64 ulp (~2.4e-7 s)**, so it vanished and a gap ending exactly on a boundary was
counted as spanning. **The detector therefore OVER-reports spans — and it still returned zero,
so the zero is sound and conservative.** Everything in §1 is recomputed in **integer
nanoseconds** with an end-exclusive `(ge-1)//WNS`, which removes the epsilon entirely.

## 3. YOUR (a) — VERIFIED AT DA's ARTIFACTS, AND IT HOLDS

```
20260908 : total_masked=0  coverage_absent=0 | btc n_masked=0  n_windows_total=288  covered=288
20260909 : total_masked=0  coverage_absent=0 | btc n_masked=0  n_windows_total=288  covered=288
20260910 : total_masked=0  coverage_absent=0 | btc n_masked=0  n_windows_total=288  covered=288
```

**No CENSUS_ONLY row exists on any of the three days**, so the declaration's 28-row shape
(27 TABLE + 1 CENSUS) is a 09-07 artefact. **The supplied count should be 288, and
`n_windows: 287` on a day with no mask is a finding** — confirmed as you stated it, at the
mask rather than from the claim.

*Note this interacts with the emit: `declared_windows()` filters `role == "TABLE"` with a
default of `"TABLE"`, so a 09-08 declaration whose rows carry no `role` admits all 43 — which
is correct here and is the fail-open default REVIEW 191 §2 flagged. On these days it is
right; it is right by luck of the data, not by construction.*

## 4. WHAT COMPLETES WHEN BE'S FILE LANDS

Three comparisons, all prepared:
1. **`n_windows` against 288** (not 287) — §3.
2. **`gap_bearing_window_starts` against my 43 ids** — set equality, not count equality; a
   count match with a different membership is the failure a count check cannot see.
3. **Any ledger gap BE's list lacks → named for the census, not reconciled into the table.**
   §2 says to expect none from boundary-spanning; if BE's list is short, the ids I hand over
   will have a different cause and I will say which.

**Reporting again the moment the artifact exists**, per the standing instruction.
