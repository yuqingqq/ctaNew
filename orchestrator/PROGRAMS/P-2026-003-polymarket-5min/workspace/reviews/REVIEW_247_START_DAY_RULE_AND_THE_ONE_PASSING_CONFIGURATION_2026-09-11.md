# REVIEW 247 — the start-day rule, and a test with one passing configuration

REV round 210. Filed 2026-09-11T23:36:12Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled. Fetched at 23:31Z. Both
executing refs `5daf76f`.

Current freeze state, read at the artifact: `freeze_is_effective: false`,
`n_blocking_gaps: 2` — `chain_link_not_implemented:pnl` and `fee_rule`.
Not one gap, two; both are economic-only by REVIEW 245's split, so **the
predictive half is already unblocked** and the start-day question is live now,
not later.

---

# PART 1 — THE START-DAY RULE

## 1.1 The defect that makes this urgent: there is no instant to be after

§8 says validation starts with the first complete UTC day **strictly after the
full pipeline freeze**. Searched in visible-failure form across the lane at the
ref for `effective_at|effective_utc|went_effective|effective_instant|
effective_since`: the only two hits are in `skew_bound.py` and
`SKEW_BOUND_RESULTS.md`, an unrelated module. **Nothing anywhere records when
the freeze became effective**, and `da_step6_full_pipeline_freeze_v1.json`
carries no timestamp field at all — only `declared_at_ref` and `ref_head`.

So today the sentence has no referent. Four different instants are all
defensible after the fact — the declaration's build time, the commit that closed
the last gap, the first checker run returning effective, the push that landed it
— and they can give **different start days**. That is the shape in which a day
gets assumed.

## 1.2 The rule, stated so the day falls out

**R1 — the effective instant is a commit timestamp, not a clock read.**

    T_eff = the COMMITTER timestamp (UTC) of the earliest commit C such that,
            with the lane checked out at C:
              freeze_is_effective  is True
              enumeration_intact   is True
              n_blocking_gaps      == 0
            AND C is present on BOTH executing refs
              (origin/de-freeze-chain-v2 AND origin/be-build-runner).

A commit timestamp is in the artifact, is the same for every reader, and cannot
drift with when someone happens to run a checker. "Present on both executing
refs" is the landing rule this programme already uses everywhere else — a count
at a fetched ref, never a sha in prose.

**R2 — the start day falls out of T_eff by arithmetic, with no choice in it.**

    D1 = the least UTC date D such that midnight(D) > T_eff
       = date(T_eff) + 1 day        (strictly; equality at exactly 00:00:00Z
                                     still yields the next day, since the day
                                     containing T_eff is not complete-after-it)
    band = D1 .. D1 + 13            (14 consecutive calendar days, §8)
    population = the first 10 evaluable complete BTC+ETH days inside the band

**R3 — the separation is a refusal, not a note.** The cancellation test's
population is `N=7, 09-07..09-13`, read at
`da_forward_test_declaration_v27.json` field `population_unchanged`. Therefore:

    if band ∩ {2026-09-07 .. 2026-09-13} ≠ ∅:
        REFUSE  FAIR_VALUE_BAND_INTERSECTS_A_CONSUMED_POPULATION
        naming the intersecting days

A refusal, not a warning, and computed from the other lane's declaration rather
than from a constant — so when the cancellation population changes, the guard
follows it. Under any freeze dated 2026-09-11 or 2026-09-12 this predicate
**fires** (D1 = 09-12 or 09-13, both inside 09-07..09-13). It first stops firing
at T_eff ≥ 2026-09-13T00:00:00Z, giving D1 = 09-14 and band 09-14..09-27.

**R4 — where it is recorded.** One new declaration,
`da_validation_window_v1.json`, written **once**, at the moment R1 first
resolves, carrying: `T_eff` and the commit C it came from; both executing-ref
heads at C; `D1`; the 14-day band; the R3 predicate's result with the
intersection it computed; and the digest of the freeze declaration that made it
effective. Superseded only under rule 13, never edited.

**R5 — the rule is checkable before it is needed.** The resolver ships with a
falsifier now, driven against synthetic `T_eff` values: a freeze at
23:59:59Z gives the next day; one at exactly 00:00:00Z gives the next day and
not that day; a band intersecting 09-07..09-13 refuses by name; a band clear of
it resolves. Written before any real `T_eff` exists, so no value can be chosen
after seeing one.

## 1.3 Why this satisfies rule 11 and a note would not

Rule 11 voids a test whose parameters are picked after seeing data. A start day
chosen by a human after the freeze is exactly that, however honestly chosen,
because the choice is unfalsifiable from the artifacts. Under R1–R5 the day is a
function of one commit timestamp, every input is a fetched ref or a landed
declaration, and a wrong day is detectable by re-running the resolver. The
separation from 09-07..09-13 is enforced by a predicate that stops the lane
rather than by a sentence someone must remember to read — which is the
difference the dispatch asked for.

**One thing I cannot make structural and will name instead:** R3 keeps the
fair-value band out of the cancellation population, but the tier-2 timer reaches
every closed day regardless of lane (REVIEW 244 §1.3b), so `D1` and its band
will be processed by `pm-evaluation-pipeline` as they close. That is
candidate-blind and cannot select or remove a day, but it should be a recorded
field in `da_validation_window_v1.json` rather than something discovered later.

---

# PART 2 — MY OWN OBJECT: the test that had one passing configuration

I chose to re-derive my own day-two ruling, because it is the assertion of mine
with the longest reach and the one most likely to rot quietly as days accrue.
MEM 399 noted day four moved HAZARD's cap `0.125 → 0.453125`, "the first cap
movement since the verdict was fixed", which is exactly the kind of movement
that invites a re-reading.

## 2.1 The four days, read at the newest record for each

| day | CONDVALUE_X_SKEW `observed_D_cents` | HAZARD_OVER_SKEWED_REF |
|---|---|---|
| 09-07 | **−14,645.08** | +4,925.36 |
| 09-08 | **−49,303.58** | **−23,998.00** |
| 09-09 | +10,295.84 | +10,836.31 |
| 09-10 | +1,741.26 | **−6,575.49** |

## 2.2 The arithmetic, recomputed from scratch rather than read

Two-sided exact sign test, G = 7 declared days, Holm threshold α/m = 0.05/2 =
0.025. Best attainable p given the non-positive days already spent:

| non-positive days | best case | two-sided p | verdict |
|---|---|---|---|
| 0 | 7 of 7 | 2·1/128 = **0.015625** | **PASSES** |
| 1 | 6 of 7 | 2·8/128 = **0.125** | cannot reach 0.025 |
| 2 | 5 of 7 | 2·29/128 = **0.453125** | cannot reach 0.025 |
| 3 | 4 of 7 | 2·64/128 = 1.0 | cannot reach 0.025 |

My numbers match the records exactly: CONDVALUE cap 0.453125 with 2 negatives,
HAZARD 0.125 at day two with 1 negative and 0.453125 at day four with 2.

## 2.3 What this actually means, and it is stronger than I said

**The test had exactly ONE passing configuration: seven positive days out of
seven.** The p-value ladder has no rung between 0.015625 and 0.125, so a single
non-positive day on an arm ends that arm irrecoverably. This is what
`tolerance_negative_days = 0` encodes, and it is what I corrected myself to at
REVIEW 226 — but stated as a p-ladder it is sharper than "the bar is unanimity":
there was never a near-miss available. An arm either ran the table or it was
over.

Three consequences.

1. **My day-two ruling was correct for BOTH arms, not only CONDVALUE.**
   CONDVALUE died on 09-07 (day one). HAZARD died on 09-08 (day two, cap 0.125,
   already 5× the threshold). At the moment I ruled RESULT STANDS, both arms
   were already unrecoverable. That is a stronger statement than I made at the
   time and it holds at the artifacts.
2. **MEM 399's cap movement is real and immaterial.** 0.125 → 0.453125 is a
   move from futile to more futile; the cap crossed 0.025 at day two and cannot
   come back. Nothing about the verdict moved, which MEM also says — I am
   confirming the arithmetic behind that sentence rather than the sentence.
3. **The design carried one degree of freedom.** N=7 with Holm m=2 admits a
   single passing outcome. Whatever else the forward test could have told us, it
   could not distinguish "slightly good" from "bad" — only "flawless" from
   "everything else." Worth recording as a property of the design, not of the
   result.

## 2.4 The part that transfers, and it is good news for the lane I have been auditing

§8 chose G=10, and on this axis it is strictly better:

| | G=7, Holm m=2 | G=10, Holm m=2 |
|---|---|---|
| 0 non-positive | 0.015625 **pass** | 0.001953125 **pass** |
| 1 non-positive | 0.125 | 0.021484375 **pass** |
| 2 non-positive | 0.453125 | 0.109375 |
| passing configurations | **1** | **2** |

So the fair-value plan tolerates exactly one bad day where the cancellation test
tolerated none. That is not a large margin, and it is worth saying plainly to
whoever reads the §8 result: **on the day a second non-positive portfolio-day
appears, that candidate is finished**, and the futility statement can be made
then rather than at day fourteen. §8 already has the machinery for this in its
`INSUFFICIENT_EVIDENCE` clause; what it does not have is the futility rung
stated in advance, which the cancellation lane learned to emit and this lane
should inherit.

## 2.5 One small thing found on the way

The 09-09 record (`_v7`) names `2026-09-10` among HAZARD's non-positive days.
That is legitimate — v7 is a late re-emit and the futility block is a running
statistic over all scored days — but it means **a day record's futility block is
not point-in-time for that day**. Anyone reading the records as a time series
will misdate the cap movements. One field (`futility_as_of_day`) fixes it.

## 3. Owed

- Part 1 is a specification, not an implementation: it needs a seat to build the
  resolver and land `da_validation_window_v1.json`, and I will verify the
  resolver's falsifier before any real `T_eff` exists.
- REVIEW 246's items remain open: the manifest's open day and the derived-tree
  read, the clause-content pin and the external pin source, and REVIEW 245's
  probe-error path.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228) — days
  three and four are now landed and read above; the round-boundary landing sweep
  as counts at fetched refs.
