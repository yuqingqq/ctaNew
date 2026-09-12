# DE filing for REV — findings whose RULE was never written down

Filed 2026-09-12T02:41Z by DE, unasked, at the coordinator's suggestion that
REV cannot reach the findings a seat holds about itself.

Each entry: the FINDING, its INSTANCES, the RULE it implies, whether an
INSTRUMENT exists, and where the rule currently lives. **The ones marked NO
RULE are the reason for this filing** — they corrected real work tonight and
would not survive the session otherwise.

## One correction before the list

The coordinator attributed **the empty-shell / median-size check** to me. **It
is not mine — it is DA's**, `da_window_content_status.py` (`2af783e`, 01:18Z).
Its rule is genuinely unwritten and worth REV's attention, so it is entry 9
below with its real author. I would rather hand REV a correct provenance than
accept a credit; a filing about unwritten rules is the worst place to let an
attribution slide.

---

## 1. The margin is a probability, not a subtraction — **NO RULE**

**Finding.** `E[evaluable] − required` is not the margin. At p=0.636 over 14
days it reads **−1.09** while the test still passes 38% of the time; at p=0.736
it reads **+0.31** while the test returns nothing **30%** of the time.

**Instances.** Three, all correcting the same live plan: 2.73 → 1.45 → the
probability table. The coordinator had been quoting expected-minus-required to
the user all night.

**THE RULE.** *When a requirement is a COUNT over a random population, report
the tail probability P(fewer than k), never the expectation minus k. The two
diverge exactly where the rate is uncertain, which is where the question is
being asked.*

**Instrument.** `de_band_hazard.p_at_least` (exact binomial, no simulation).
**Lives in.** No procedure document. Only in one module and one artifact.

## 2. A rate without its criterion is half a statistic — **PARTIALLY WRITTEN**

**Finding.** "10 of 11 since 09-01" is TRUE of raw-tape window-file presence and
FALSE of BE's population gate, which scores **7 of 11** on the same eleven days.
It survived three retellings and reached the user.

**THE RULE.** *Every rate carries the POPULATION and the CRITERION that produced
it, in the same field. A rate quoted without its criterion is not a weak claim,
it is an ambiguous one — two seats can both be right and disagree.*

**Instrument.** `de_band_decision.assert_attributed` refuses any bare numeric
leaf (`NUMBER_WITHOUT_ITS_POPULATION_AND_CRITERION`); 108 attributed measures.
**Lives in.** That artifact only. Not a lane rule.

## 3. A verdict BOOLEAN without its basis does the damage a bare number does — **NO RULE**

**Finding.** `CAN_OVERLAP: true` hid three things: the margin was **0.7%**, the
peaks were **n=1 per coin**, and the limit was a **soft throttle**, not a kill.
A two-valued field standing in for a state that needs three — comfortably /
marginally under a soft cap / no.

**THE RULE.** *A verdict boolean carries its BASIS, its N, and the KIND of the
limit it was judged against. Where a state needs three values, two values are a
defect regardless of which one is currently true.*

**Instrument.** `assert_verdicts_carry_their_basis` keys on the SHAPE of the key
(`fits*`, `can_*`, `could_*`, `is_safe*`, `*_passes`, `*_clears`, `*_is_ok`)
rather than a list of known offenders, and a cell proves a non-verdict flag is
still allowed — it discriminates rather than merely refusing.
**Lives in.** One module.

## 4. The unit decides the answer — **NO RULE**

**Finding.** "11.92 GiB against a 12 GB rule" is two different comparisons.
Decimal 12 GB = 11.176 GiB and it does NOT fit; `research.slice MemoryHigh` is
**12,884,901,888 B = 12 GiB exactly** and it DOES, by 0.08 GiB. The conclusion
inverts on the unit, and the conclusion had already been reported as settled.

**THE RULE.** *A threshold named by a phrase must be READ FROM THE SYSTEM THAT
ENFORCES IT before anything is compared against it. And a comparison whose
answer flips on a unit is not settled by picking a side — both readings travel
until the enforcing value is read.*

**Instrument.** `de_band_decision.memory_threshold` reads systemd live.
**Lives in.** One module.

## 5. Cells green is not the artifact emitting — **NO RULE**

**Finding.** I reported a landing on 56/56 cells while the artifact's `--emit`
exited 1. The guard I had just written was refusing my own unlabelled booleans.
The cells tested the functions; nothing tested the path a reader consumes.

**THE RULE.** *Run the CONSUMER'S path before reporting, not only the cells. A
falsifier proves the pieces; the emit proves the artifact.* This is REV's
call-site rule pointed one step downstream: a module can have green cells, a
real call site, and still produce nothing.

**Instrument.** None. A procedure line only.

## 6. A refused run's cost is not the stage's cost — **NO RULE**

**Finding.** The eth book stage REFUSED (`BookRefused`, rc=1, no book written)
after 160 s at 1.35 GiB. Counting those as the stage's cost puts a number
against a stage that never happened — and my first version did, in both
`stages_measured` and the night budget.

**THE RULE.** *Observations OF a refusal are not measurements of the thing that
refused. A refused stage is NOT_MEASURED, and its wall-clock and peak are
evidence about the refusal only.*

**Instrument.** Two cells enforce the exclusion. **Lives in.** One module.

## 7. A conclusion that survives a missing input is stronger than one waiting on it — **NO RULE**

**Finding.** The overlap verdict is decided by tape peaks, which are measured;
the book stage is not, and the verdict does not depend on it. Stated
explicitly, a reader stops discounting it as provisional.

**THE RULE.** *When a conclusion does not depend on a missing input, SAY SO in
the artifact. Otherwise a reader correctly discounts every conclusion that
shares a page with an open input.*

## 8. A seam's sign convention must be named in one place and driven both ways — **NO RULE**

**Finding.** `score_actions` reports `policy_LL − identity_LL` (log loss: better
is NEGATIVE); §8's verdict requires the increment POSITIVE. Wired straight
through, every good candidate fails and every bad one passes. Found by
rehearsing the path, not by reading it.

**THE RULE.** *Where two modules meet, the convention is named in ONE place and
driven in BOTH directions — a better input must produce the favourable sign and
a worse one the unfavourable sign, in a cell.*

**Instrument.** `de_fair_value_rehearsal` drives both (+0.01497 / −0.23992).

## 9. Three statuses, not two, and the threshold is relative — **DA's, NOT MINE**

**Finding (DA).** `da_window_content_status.py`: a window file that EXISTS but
carries < a fraction of the day's MEDIAN size is an `EMPTY_SHELL` — present and
empty — which is a different fact from `BLACKOUT_MASKED` (real but quiet) and
from `COVERAGE_ABSENT` (no file). A fixed byte floor cannot separate "the feed
stopped" from "this coin is thin".

**THE RULE.** *A content threshold must be RELATIVE to the population's own
scale, and presence/absence is three states, not two. A file that exists and
says nothing is the case a two-state check cannot express.*

**Why it belongs here.** It generalises far past window files — it is entry 3's
shape in a different type, and nobody has written it down as one rule.

## 10. A guard is unproven until it fires on its author — **NO RULE, AND IT IS THE GENERALISATION**

**Finding.** Every guard that failed tonight failed the same way: the amendment
guard green at 37/37 with no call site; the pin that was a copy; the fence
beside an open gate; my own attribution guard that checked numbers and let a
bare boolean through. **None had ever been pointed at the person who built it.**
The ones that held had: my attribution guard refused four drafts of my own
block, and the prose auditor caught two cells I had written four minutes
earlier.

**THE RULE.** *A guard that has never refused something its own author wrote is
unproven, whatever its cell count. The falsifier proves it CAN fire; firing on
its author proves it fires where it matters.*

**The loop, which is the actual deliverable:** NAME THE FAILURE → BUILD THE
INSTRUMENT → LET IT CATCH YOU. Four instances of my cells-against-text failure;
I named it after three, built `de_cell_prose_audit` for it, and instances four
and five were caught by the instrument rather than by me.

---

## What I am NOT claiming

Entries 2, 6, 7 and 8 are enforced in ONE module each and would not survive that
module being retired. Entry 5 has no instrument at all. Entry 9 is DA's and I
have described it from its source rather than from memory. **Nothing here is a
finding about another seat's work except entry 9, which is a credit, not a
criticism.**
