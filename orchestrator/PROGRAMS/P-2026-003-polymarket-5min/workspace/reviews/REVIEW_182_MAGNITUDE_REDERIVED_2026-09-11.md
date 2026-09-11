# REVIEW 182 — the magnitude re-derived on the right set and the right mechanism: 19% is the reference, 5.3x-with-total-reversal is the rescue, and the mask flip is bounded at ~3.5%

**REV 139, 2026-09-11T06:58:14Z** (clock read separately). Read-only: no lock, no heavy
unit, nothing written under `data/`. **Written before the rebuilt book exists** — every
number is a scenario against a stated denominator, not a measurement.

---

## 0. WHAT I GOT WRONG, IN ONE LINE EACH

- **Wrong set.** I used the 12 masked windows. **BTC's masked share is 1 window**, not 12;
  the 12 was seven coins over 2,016 coin-windows. REVIEW 181 caught the population error;
  DA 230 caught that it was the wrong quantity entirely.
- **Wrong mechanism, and this is the one that matters.** I assumed **row removal**. DA 230
  establishes the fix changes **replay content in RETAINED rows** — the gap list removes
  nothing (`BINANCE_GAP_EXCLUDED_STATUS = NOT_APPLIED_ON_THE_DAY_PATH`).
- **The consequence is directional.** Removal can only SUBTRACT a window's contribution.
  **A content change can REPLACE it — so the swing is up to TWICE the contribution, from −x
  to +x.** My bound was not merely mis-scaled; it was the wrong shape.

---

## 1. THE RE-DERIVATION — 27 of 287 BTC WINDOWS (9.41 %), CONTENT CHANGE

`D = −11,017.71c`. Uniform-share deficit in the 27 windows = **1,036.5c**.

| concentration in the 27 | deficit there | full-flip swing | % of \|D\| | erases D? |
|---|---|---|---|---|
| **1.0× (no concentration)** | 1,037c | **2,073c** | **18.8 %** | no |
| 2.0× | 2,073c | 4,146c | 38 % | no |
| 3.0× | 3,110c | 6,219c | 56 % | no |
| **5.31× — the break-even** | 5,504c | 11,018c | **100 %** | **exactly** |
| **8.2× (the Q-DA-58 mapping)** | 8,484c | 16,967c | **154 %** | **YES** |
| 10.6× | 10,987c | 21,974c | 199 % | YES |

**THE ANSWER TO "PLAUSIBLE, LIKELY, OR NEAR-CERTAIN": PLAUSIBLE. NOT LIKELY. NOT NEAR-CERTAIN.**
Erasure needs a **conjunction**, and the two halves are not equally available:

1. **Concentration ≥ 5.31×.** This half is *supported*: Q-DA-58 measured the worst 10 % of
   fills carrying **77 %** of drift — 8.2× on a 9.4 % base — and gap-bearing windows are
   mechanistically the right candidates (bursty tape, adverse selection, stale information
   at the cancel decision). **I would call this half likely.**
2. **A near-TOTAL reversal of those windows' net contribution.** This half is *not*
   supported. The era fix changes which era's coverage the replay resolves; it perturbs what
   the arm sees, it does not reverse the arm's economics. At the Q-DA-58 concentration of
   8.2×, **the flip fraction still has to reach 0.65** to erase the deficit — two thirds of
   the affected windows' contribution must not merely move but change sign. **A partial
   perturbation is the ordinary outcome; a two-thirds reversal is not.**

**So: materially more plausible than under my removal framing — a rescue is now inside the
range the programme has measured for the first factor — and still requiring a second factor
for which there is no evidence.** I would tell the user "possible, and we have pre-declared
how we will report it", not "likely".

## 2. THE TRIPWIRE — 25 % IS RETIRED. HERE IS THE DERIVED REPLACEMENT, TYPED NOW.

**25 % was very nearly right BY ACCIDENT** — it sits just above the 18.8 % no-concentration
level — but it was derived from the wrong set and the wrong mechanism, so its agreement is a
coincidence and it should not be kept on the strength of it. Being right for the wrong reason
is still wrong.

**DECLARED NOW, BEFORE THE REBUILT BOOK EXISTS:**

1. **UNCONDITIONAL REPORT — no threshold.** The rebuild publishes, per arm: **ΔD**, and the
   **per-window arm-vs-baseline contribution of all 27 gap-bearing windows and of the 1 BTC
   masked window, listed separately.** A threshold decides what is *surprising*; it must not
   decide what is *published*.
2. **`CONCENTRATION_FINDING` at |ΔD| > 18.8 % of |D| (= 2 × 27/287, i.e. 2,073c).** Above
   this level the 27 windows **demonstrably carry more than their uniform share** — that is
   an arithmetic fact about those windows, true regardless of what D becomes, and it is the
   finding. **Type the number, not the percentage: 2,073c.**
3. **`SIGN_CHANGE_HALT` — if the rebuild flips D's sign, the round STOPS for a ruling.** A
   correction that reverses a seen day's result is a finding **about the correction**, not a
   new result, and it must not be absorbed into a superseding D. This is the tier my 25 %
   did not have and it is the one that matters.

**Why three tiers rather than one:** under content change the tripwire and the rescue are
*very different* thresholds — 18.8 % fires at 1.3× concentration, while erasure needs 5.31×
**and** a two-thirds reversal. A single threshold conflates "the windows are concentrated"
with "the answer changed", and those need different responses.

## 3. THE MASK FLIP, BOUNDED SEPARATELY AS ASKED

**BTC's masked share is 1 window of 287 = 0.35 %.** If the fixed era flips it to unmasked:

| concentration | ΔD contribution | % of \|D\| |
|---|---|---|
| 1× (uniform) | 38c | **0.35 %** |
| 3× | 115c | 1.05 % |
| 10× | 384c | **3.48 %** |
| 20× | 768c | 6.97 % |

**So the mask flip alone cannot plausibly move D by more than ~3.5 %, and at uniform share
it is a third of one percent.** It is an order of magnitude below the gap-window effect and
**must be reported as its own line, never pooled with the 27** — otherwise a 3 % mask effect
and a 30 % content effect arrive as one number and neither is readable.

## 4. THE FRAME STILL GOVERNS, AND IT IS UNCHANGED BY ANY OF THIS

The fix's rule predates the data; **the decision to apply it to 09-07 came after −11,018**;
uniformity is the defence, now declared for the mask too (DA 231). **Nothing in §1–§3 changes
that** — a larger, differently-shaped magnitude makes the uniformity declaration *more* load-
bearing, not less, because the correction can now move the number by a lot. **The tripwire is
not a substitute for the uniformity declaration; it is what makes the uniformity declaration
checkable after the fact.**

## 5. SCOPE — AND THE LIMIT THAT RUNS AGAINST EVERY NUMBER ABOVE

**I have no per-window P&L.** Every row of §1 and §3 is a scenario against a stated
denominator, not a measurement; the rebuild produces the actual ΔD and my job here was to say
in advance what its ranges mean. **Two assumptions are load-bearing and neither is
established:** that only the 27 gap-bearing windows change content (a wider era effect would
raise every bound), and that the Q-DA-58 concentration measured on a fill-level cash ranking
transfers to a window-level gap ranking — **a different ranking variable, and I am mapping
across it deliberately and saying so.** If DA or BE can produce the per-window contributions
from the EXISTING book before the rebuild lands, §1's table stops being scenarios and becomes
one number — **that is a ten-minute query and it would be worth more than this entire
filing.**
