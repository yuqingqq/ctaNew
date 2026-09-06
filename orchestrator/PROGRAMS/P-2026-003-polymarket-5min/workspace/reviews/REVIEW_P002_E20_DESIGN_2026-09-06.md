# REVIEW — P-2026-002 E2.0 declaration: APPROVED for the ADA smoke. Δ_rs isolates the period and is silent on the population, and `SETTLED_ALIVE` is labelled at the death bar, not the gate

**Filed** 2026-09-06T04:33Z (clock read before composing) · reviewer seat
(pm-codex) · tip `02c5400` · **filed under P-2026-003's review dir because that is
this seat's filing surface; the subject is P-2026-002 E2.0** · no code fixed · **no
data touched** · nothing sealed opened.

**ROUTING — CHECKED**, every citation resolved at the file and line named.

## VERDICT

**APPROVED for the ADA smoke to run.** This is the most carefully pre-registered
declaration I have reviewed in either programme: the threshold is cited to a
pre-existing plan, the reproduction control refuses on mismatch, the falsifiers
plant the very phenomenon under test, and the rule-5 boundary is handled better
than the question I was asked assumed.

**Three things v2 must carry before the smoke's RESULT is read** (none blocks the
run):

1. **Δ_rs is declared on "the SAME sweep events" but the two mids have different
   existence conditions.** The true mid has no validity window; the proxy requires
   two prints within 10 s. **Nothing says Δ_rs is computed on the intersection.**
2. **`SETTLED_ALIVE` is labelled at 1.8 bps, which is the DEATH bar. The plan's own
   gate is 2.3.** A cell at 2.0 would read ALIVE while failing §1.5 gate 1.
3. **The settle is a POINT against a threshold** while an interval is declared and
   G = 16 makes one claimable. The decision rule does not use it.

---

## 1. The true mid — **matches the amendment, and the clock choice is the best thing in the document**

`the_true_mid.definition`: *"m_true(u) = (best_bid + best_ask)/2 taken from the LAST
bookTicker message with exchange transact_time T <= u; for m(t−) the search is
STRICTLY BEFORE the sweep's own T."* **That is the definition verbatim, plus the
strict-inequality refinement for the pre-sweep mid, which the amendment does not
spell out and which is the half that would otherwise leak the sweep into its own
baseline.**

**The clock is exchange `T`, not `recv_ns`, and the reasoning is exactly right:**
*"recv_ns is OUR stamp … it carries ~74 ms of measured one-way latency plus clock
offset straight into the markout. Exchange T is the stamp the event itself carries
(CLAUDE.md rule 3)."* **Rule 3 applied to the quantity that is most sensitive to
it.**

**And `validity_rule: NONE` is correct and correctly explained** — *"the book is
continuous: a bookTicker message stands until the next one. The 10 s validity
window belongs to the PROXY mid, which is built from prints and goes stale."*
Quote age is **reported (p50/p99 per symbol-day) and not filtered**, because
*"filtering on quote age would select on activity"* — the right call, and the
staleness is then checkable rather than assumed.

**The proxy mid is stated** (EXPERIMENT_PLAN §1.2, two-sided last-print mid, both
legs within 10,000 ms) **and recomputed rather than taken from E1**, so *"Δ_rs
isolates the MID and not the period."*

### **FINDING 1 — it isolates the period and is silent on the population**

`leg_i_VOID.quantity` says both legs are computed *"on the SAME admissible days
from the SAME sweep events."* **But the two mids do not exist on the same sweeps.**
The true mid exists for every sweep with any prior bookTicker; the proxy requires a
taker-buy **and** a taker-sell print within 10 s. On a quiet stretch the proxy is
undefined and the true mid is not.

**If Δ_rs is not restricted to the intersection, it mixes a mid difference with a
population difference — which is the one thing it must not do.** The declaration
anticipates the general hazard and closes the *period* version of it; the
*population* version is not addressed.

**v2 must say which**: Δ_rs over sweeps where **both** mids are defined, with the
count and share of sweeps dropped for want of a proxy reported as a status (rule 4).
That number is also interesting in itself — it measures how much of E1's window the
proxy could not see.

## 2. Population — sound, and the exclusions are statuses

* **cluster unit: UTC day** (rule 8) ✓; **`min_complete_days: 14`,
  `refuses_below_min: true`** ✓
* **day admission**: 24 hour-files for **both** bookTicker and trade, **and**
  intra-day gap fraction < 0.05 (share of the day's seconds with no bookTicker),
  **both computed at run time** ✓
* **`the_admissible_set_is_an_OUTPUT`** — *"declared as an output, never as an
  expectation, so no belief about which days should qualify can become a filter."*
  The structural count is given as-of 2026-09-06T04:20Z (16 complete days;
  08-19 a 12-hour partial, 08-26 a 23-hour partial after a reboot, 09-06 in
  progress) **and the gap-fraction leg is explicitly recorded as NOT YET
  EVALUATED.** ✓
* **exclusions are counted statuses, not drops**: the `identity_check`
  (`|mean es − mean Λ − mean MO| > 0.2 bps` ⇒ `proxy_incoherent`) excludes the
  symbol-day **and** *"the excluded fraction is reported with every table (rule
  4)"* ✓
* **the event unit is the SWEEP, not the print**, collapsed on
  `(transact_time, is_buyer_maker)` with a qty-weighted price — because *"the
  collected `trade` stream is per-match and FINER than E1's aggTrades, so without
  this collapse the event populations would not be comparable."* **That is a
  comparability defect caught before it could be made.**

**G is stated as an output and the thresholds are expressed as fractions of it**
(τ*'s `ceil(0.774194·G)`, gate 3's 70%), with G = 16 as the worked example —
so the rules do not pre-suppose the count they are applied to. ✓

## 3. The gate quantity — both reported, and the two legs use different weightings **on purpose**

`primary`: notional-weighted rs(τ*) on true mids. **`also_reported`: eq-weighted
"so Δ_rs is comparable to what E1 gated on"**, per side, Λ(τ) and es, and
size-bucketed by notional quintile. ✓

**Which settles and which voids is cleanly separated:**

| leg | quantity | threshold | consequence |
|---|---|---|---|
| i — VOID | Δ_rs(τ*) = rs_proxy − rs_true, **EQ-weighted** | > +1.0 bps | voids E1's ADA pass |
| ii — SETTLE | rs(τ*) on true mids, **NOTIONAL-weighted** | < 1.8 bps | kills the ADA cell |

**and the outcome table has five cells including `UNDECIDABLE` as a status**
(*"either leg has no admissible estimate — a STATUS, never resolved by
assumption"*). ✓

`why_notional` is measured, not asserted: *"E1's own audit measured the gap at
eq − notional ≈ 2.8 bps on ADA, twenty times ADA's 0.14 bps margin over fee."*
Consistent with the HANDOFF's +2.44 → −0.32 (gap 2.76).

**τ\* is fixed ex ante "to forbid horizon shopping"**, as a fraction of G with the
plan's "≥ 24 of 31" re-expressed as 0.774194 — ceil(0.774194 × 16) = 13. ✓

## 4. The thresholds — **+1.0 bps verified pre-registered; 1.8 verified, but the LABEL on it is wrong**

**+1.0 bps is not chosen here.** `EXPERIMENT_PLAN.md:316–319`: *"Pre-registered
consequence: Δrs > +1.0 bps on any E1-B passer voids its pass"*, and the gate table
at `:456` repeats it. **Declared long before this declaration, cited correctly.** ✓

**1.8 bps is the VIP0+BNB maker fee** (`EXPERIMENT_PLAN.md:34`) and the amendment
(`E1_CODE_REVIEW.md:399–401`) says *"if notional-weighted rs(τ*) **< fee at VIP0**
… the ADA cell dies regardless of eq numbers."* **Correctly sourced.** ✓

### **FINDING 2 — but the plan's own gate is 2.3, and the outcome table labels 1.8 as ALIVE**

`EXPERIMENT_PLAN.md:183`: *"day-clustered mean **rs(τ*) ≥ fee_maker + c_safe**
(VIP0+BNB: **≥ 2.3 bps**)"*, and `E1_RESULTS.md:15` calls 2.3 *"the VIP0 hurdle"*.

So **1.8 is the death bar and 2.3 is the pass bar**, and the declaration's table
reads:

```
SETTLED_ALIVE : Delta_rs <= 1.0 AND notional rs_true >= 1.8
```

**A cell at 2.0 bps would be labelled `SETTLED_ALIVE` while failing §1.5 gate 1.**
The declaration does compute the full gate — *"the full section 1.5 gate is computed
for every symbol whatever the verdict"* — but **the label is what a reader
resolves.** Rename to `NOT_KILLED` or `ALIVE_PENDING_GATE_1`; the 1.8–2.3 band is a
real band where a cell is neither dead nor passing. On ADA's expected −0.32 this
never binds, which is exactly why it would go unnoticed.

### **FINDING 3 — the settle is a POINT, and an interval is computed but not used**

`statistics.interval` declares a stationary block bootstrap of the ratio-of-sums
(B = 2000, seed 20260906, 30-min bins, 4h blocks), and `interval_floor` correctly
says no interval below G = 5. **At G = 16 an interval is claimable — and the
decision rule compares a point to 1.8.** If the interval straddles the threshold,
SETTLED_ALIVE vs SETTLED_DEAD is decided by a point estimate. **v2 should say
whether the interval is decision-bearing or reported beside the point**, before the
number exists.

## 5. The E1 reproduction control — present, exact, **and it refuses**

`e1_reproduction_control` pins the numbers to reproduce, and **all four match
`E1_RESULTS.md:22–23` verbatim**: `rs_eq_bps 2.443`, `rs_notional_bps −0.322`,
`days_eq_positive 31`, `days_notional_positive 7`, over 31 days at τ* = 30 s on the
07-18..08-17 window, tolerance 0.05 bps.

**And the refusal is explicit**, in `falsifiers.known_bads`: *"the runner must
REFUSE if the E1 reproduction control misses E1's published ADA numbers by more than
0.05 bps."* ✓

The reasoning is the right one: *"If the proxy leg is a reimplementation that does
not match E1's, Δ_rs is a difference between two CODEBASES and the voiding rule
means nothing."*

## 6. Falsifiers — both directions, and one of them plants the phenomenon under test

**Positive controls:** a book moving in the maker's favour by a known bps must
return that number **on both sides**; a clean day must be ADMITTED; the identity
`mean es − mean Λ == mean MO` must hold **exactly** on a synthetic known mid path;
and — the best one — *"a synthetic population of many small POSITIVE events and few
large NEGATIVE ones must give **eq > 0 AND notional < 0** — the H1 shape itself, so
the weighting code is shown to **discriminate** rather than merely to run."*
**That is the fat-tick artifact planted as a control.**

**Known-bads:** an against-the-maker book must return rs negative **at the known
magnitude**; an injected gap must EXCLUDE with a named status and be counted; **a
sweep whose t+τ falls past the end of available book must be EXCLUDED and counted,
never valued at the last known mid**; reversing `is_buyer_maker` must FLIP the sign;
**the runner must REFUSE if this declaration's sha256 differs from the one it was
built against**; refuse below 14 admissible days; refuse on the reproduction miss.

**One case from the round's brief is not in the list:** *a sweep with **no standing
quote before it***. The falsifiers cover the far end (t+τ past the book) and
gap-excluded days, not the absence of `m(t−)` at a file or day boundary. A
first-sweep-of-file case should be planted and required to exclude-and-count.

## 7. Resources and the smoke

**ADA alone, all admissible days, RSS and wall recorded, before any other symbol** —
and the sizing is measured, not guessed: ADA bookTicker **41 MB/day gz** against BTC
**480 MB/day**, ~12×. *"No wall-clock estimate is offered before the smoke measures
one. The smoke IS the estimate."* ✓ Cap is the rule-20 wrapper at one CPU / 8G, and
*"a symbol that exceeds the cap REFUSES, the cap does not rise."* ✓

**Small note:** the smoke publishes ADA's economics, and **ADA is the cell this step
exists to settle.** Nothing seals it. The exposure is much milder than P-003's
multi-day case — each symbol is its own cell and every threshold is pre-declared —
but it is worth one sentence saying the other fifteen are unaffected by having seen
it.

## 8. Liveness and the rule-5 boundary — **handled better than as an admissibility floor**

**The boundary is stated with its exact ns** (1787579334881534478 /
2026-08-24T13:48:54Z) in `why_not_recv_ns`, **and the declaration explains why it
does not bind: the gate quantities do not use `recv_ns` at all.**

**And it does not stop there.** `robustness_splits_declared_now.post_era_boundary`
recomputes **every gate** on days entirely at or after the boundary, *"declared
BEFORE the run so it cannot become a rescue: the gate quantities do not use
recv_ns, and this split is the check on that claim"* — with 11 expected
post-boundary days, above the G ≥ 5 floor.

**I would push back on one premise of the question.** Making rule 5 the
*admissibility floor* here would be wrong: it is a statement about `recv_ns`
reliability, and this design reads exchange `T`, which the schema carries
(`recv_ns,E,T,u,bid,…`) and which post-parse stamping does not touch. A floor would
discard 08-20..08-24 for a reason that does not apply and cut G from 16 to 11 for
nothing. **Declaring the split as a check on the claim is the stronger move, and it
is what was done.**

**Collector liveness is not checked, and does not need to be** — E2.0 reads
historical tape, and a collector that died mid-window is caught twice by the day
predicate (missing hour-files, then gap fraction ≥ 0.05). **v2 should say that**,
rather than leave the absence to be read as an oversight.

---

## Verdict

**APPROVED for the ADA smoke.** v2 before the result is interpreted:

1. **State that Δ_rs is computed on the intersection** where both mids exist, and
   report the sweeps dropped for want of a proxy as a counted status.
2. **Rename `SETTLED_ALIVE`** — 1.8 is the death bar, 2.3 is §1.5 gate 1.
3. **Say whether the interval is decision-bearing** at G ≥ 5.
4. Plant the **no-standing-quote-before-the-sweep** falsifier.
5. One sentence each on the smoke's economics and on why collector liveness is not
   a precondition.

**Nothing here weakens the design.** The reproduction control, the sha refusal, the
H1-shape positive control and the pre-declared post-boundary split are four
instruments I have asked other seats for and not been given.

---

## CONTEXT

Far below the 80% reset threshold.
