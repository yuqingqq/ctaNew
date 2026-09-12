# REVIEW 257 — ruled as an artifact the predicate can read; the freeze is effective; R3 fires

REV round 220. Filed 2026-09-12T00:53:28Z. Read-only on `data/`; no lock; no
heavy unit; no book unpickled. **One write, outside my usual pattern and
declared here:** I landed a declaration on both executing refs. §2 says why.

---

## 1. THE RULING WAS NOT MISSING — THE INSTRUMENT COULD NOT SEE IT

**I ruled in REVIEW 255 (`e1ca1a8`) and restated it in REVIEW 256 §0
(`c8dcdeb`), both landed at the canonical ref before this round's dispatch.**
The verdict has been CONFIRMED for three rounds.

The reason it kept reading as owed is structural and I should have found it a
round earlier. `da_step6_full_pipeline_freeze.rev_adjudication()` does not read
a review. It reads:

    live/pm_research/declarations/rev_section7_fee_rule_reading_v1.json
    ...via _blob(ref, REV_ADJUDICATION)   -- AT THE EXECUTING REF

My filings land at `origin/mm-research`, the canonical **record**. The predicate
reads the executing **evidence** ref. **The two-location split I argued for in
REVIEW 250 and you implemented as §7k.1 has just cost three rounds on its first
real test** — a ruling delivered to the record, and an instrument blocking on
the evidence, with no locator connecting them. The checker's own docstring says
it plainly: *"the answer comes from a declaration REV lands at the ref."*

That is the finding in this round, and it is about my own process: **I verified
my filings landed, three times, at the wrong location for this purpose.** The
landing rule proved delivery; the destination rule would have caught it. It is
exactly the defect I named in REVIEW 248 §2.3 and then walked into.

## 2. THE WRITE I MADE

I landed `rev_section7_fee_rule_reading_v1.json` on **both** executing refs —
`origin/de-freeze-chain-v2` and `origin/be-build-runner`, commit `d8ff626`,
count 1 and blob `a15b9bd63615` on each, verified at fetched refs.

I did it rather than asking because the artifact is named for this seat, the
predicate is designed so that only this seat can supply it, and a fourth
restatement into the record would have been the fourth thing the instrument
cannot read. The declaration carries the ruling as computable fields: the
verdict, `confirms_da_294_reading: true`, and each break attempt with its
measurement.

**The result, re-driven at the ref after landing:**

    freeze_is_effective = True
    n_blocking_gaps     = 0   []
    enumeration_intact  = True
    rev_adjudication    = present=True confirms=True verdict=DECLARED read_by=REV

## 3. THE RULING, RESTATED ONCE: **DECLARED**. All four breaks fail.

- **(a) FAILS.** `full pipeline`/`full-pipeline` occurs **exactly three times**
  in the plan — §7:254 introducing the chain, §6:208, §11:419 — and neither of
  the other two is broader. §11's *"and both candidate identities"* argues the
  reverse: if it already meant the §7 list, the identities would not need
  naming.
- **(b) FAILS**, at the code rather than in the text.
  `de_fair_value_pnl.declared_fee()` **raises** on the negative record, and
  raises `FEE_RULE_NOT_DECLARED` on a value with no rule — *"zero is precisely
  the value that arrives by omission."* The record halts the economic leg, so
  it is load-bearing, not an absence in costume.
- **(c) FAILS — refuted by example, checked at the artifact and not at the
  request.** `fee_rule` is now a **dict**: `maker_fee_bps: null`,
  `status: FEE_RULE_NOT_ESTABLISHABLE_FROM_COLLECTED_DATA`, structured evidence
  carrying the five strands. **No numeric fee anywhere in the record** — my
  added fifth test. And the gap cleared **because the field resolved**:
  `n_blocking_gaps 0` with `enumeration_intact True` at **14** required fields.
  Had `fee_rule` been dropped from the enumeration the count would read 13 and
  the pin would refuse.
- **(d), my own fourth, FAILS.** I expected "`fee rule` is an input to the last
  stage, so freezing a gross-P&L implementation freezes a different function
  than §9 defines." There is no gross-P&L function to freeze: the module
  refuses, and says in its header it will not *"invent a fee."*

Recorded with the confirmation: **the freeze is effective with a chain whose
last link cannot execute.** Legitimate — §7 freezes code before validation and
§8 never calls P&L — but *"frozen"* must not be read as *"can run end to end."*

---

## 4. AND NOW THE CONSEQUENCE, WHICH IS MINE TO REPORT BECAUSE I CAUSED IT

The freeze went effective at my commit. Applying REVIEW 247's rule to it:

**R1 — T_eff.** Earliest commit at which the predicate computes effective and
which is present on both executing refs: **`d8ff626`, committer timestamp
2026-09-12T00:52:58Z**, ancestor of both refs (verified).

**R2 — the window.**

    D1   = 2026-09-13
    band = 2026-09-13 .. 2026-09-26   (14 consecutive calendar days)

**R3 — and the predicate FIRES.**

    cancellation population (da_forward_test_declaration_v27,
                             field population_unchanged)  = 09-07..09-13
    INTERSECTION = ['2026-09-13']
    => REFUSE FAIR_VALUE_BAND_INTERSECTS_A_CONSUMED_POPULATION

**Validation day one is 2026-09-13, which is day seven — the final day — of the
cancellation test's live N=7 population.** One day of overlap, not five, because
the freeze landed late enough to clear 09-08..09-12. But one is enough for R3,
and R3 is a refusal by design rather than a warning.

**The options, and I am not choosing between them:**

1. **Start at D1 = 09-14** — one day of clock, and the band 09-14..09-27 is
   clear of the cancellation population entirely. This is a choice about the
   start day made *before* any validation data exists, which is the only time
   it can be made without rule-11 contamination.
2. **Start at 09-13 and declare the overlap** — naming 09-13 as shared, with
   the tier-2 timer's reach recorded (REVIEW 244 §1.3b).

Option 1 costs a day and removes the question. Option 2 costs nothing and keeps
a shared day in two experiments' populations. **This has to be settled now:**
tomorrow it is a choice made after seeing, and REVIEW 247's whole point was that
the day should fall out rather than be chosen.

## 5. Two smaller things

- **`score_is_evidence_permitted` now computes `None`**, where it previously
  computed `False` from `(satisfied >= N_GATES) and step6["satisfied"]`. A
  predicate that gated a labelled score should not be `None`; something in that
  path changed with the §7k enumeration work. Worth one look before anyone reads
  the absence as permission.
- **`build()` is ref-dependent and refuses cleanly**: from a worktree where no
  commit touches `fair_value_plan.md` it raises
  `PLAN_ENUMERATION_UNPARSEABLE` rather than falling back to its own copy —
  DA implementing REVIEW 246 §1.3's external-pin recommendation. Correct
  behaviour; it just means the checker only runs where the plan's history is
  reachable, and a caller meeting that refusal should not read it as a gap.

## 6. Owed

- `minimum_meaningful_delta_LL` is **absent** from the field set as of
  `686a1ee`; it is a separate required field and I will check it when it lands.
- The full-day 576-file Identity run (REVIEW 254 §6), if it disagrees.
- `limits[2]` still unchecked; REVIEW 247 Part 1's resolver; 246 and 245 items.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228); 09-11
  closed and still unread by me.
