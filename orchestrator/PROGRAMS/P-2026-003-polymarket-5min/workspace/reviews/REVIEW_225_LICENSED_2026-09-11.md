# REVIEW 225 — **LICENSED.** All four evidence criteria are met at the artifact. The result does **not** stand yet: the running tally is keyed on the day, not the arm, and it says no arm is dead while HAZARD is

**REV, 2026-09-11T15:54Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Drives on scratch and in my own worktree (restored).

Read: `fwd_v2/p003_de_forward_value_20260908.json` (15:52:37Z) and the two 09-07 rebuild cells
in `fwd_a_0907_rebuild/`. **The superseding re-emit (DE 346) does not exist at my read**, so
the verdict below is on the 15:52 record; criteria (1)–(4) are unchanged by a re-emit and I
rule them now.

## LICENSING: **LICENSED**

The four criteria I pre-declared in REVIEW 213 and restated every round since, each read at
the artifact:

| # | criterion | at the record |
|---|---|---|
| 1 | `cells[<arm>].book_receipt.admitted_by` = `DESCENDANT` both arms | **both DESCENDANT**, and a top-level `admitted_by` map repeats it. `builder_commit b34ed9fdd1e32fe2` = the freeze commit |
| 2 | one oracle sha, both arms | **`455b4132ec1f994c…`, 46,276 records**, in the top-level `winner_source`, in `winner_source_sha256_field`, and identical in both cells |
| 3 | the three unnamed members by digest | **3/3 `identical`, compared at 64 hex, both arms**, each with `recorded_from: the receipt's builder_commit b34ed9fdd1e3` |
| 4 | stage 0's verdict recorded | **`structured: true`** — not the fallback |

**On (4) specifically, since the round asked whether the fallback would satisfy it:** it did
not have to. The block carries `verdict: POPULATION_FREEZE_HOLDS`, **`n_files 67, n_verified
67`** — which closes the `62 of 67` gap I filed in REVIEW 222 §3 — the class split
(`n_PIPELINE 52`, `n_INSTRUMENT 15`, `INSTRUMENT_DRIFTED []`), `roots_are_the_declared_defaults:
true`, **the verifier's own path and sha256** (`74930d284a5cbec2`), the declaration it verified
(`da_population_freeze_v17.json`), and `time 2026-09-11T14:19:02Z` — **before** both arms
finished (15:00 and 15:43). A reader can check the gate ran, on what, with what, and when.

*Had it been `structured: false` with verbatim log lines, my answer would have been: it
satisfies the criterion's purpose and not its form — a reader could tell the gate ran and what
it said, but nothing computable would assert it, and the text is unconstrained. I would have
licensed on it and routed the structured version. That question is moot.*

**One disclosed weakness, and DE disclosed it rather than me finding it:** the source is
`/tmp/stage0_freeze_20260908.json`, and the block says so — *"the launcher wrote the gate's
report to /tmp, which is not run-scoped; future launches write it into the launch record before
the offer"*. The rows are copied into the record, so the **evidence** travels; the **source**
does not. Naming it is the right handling and the fix is already stated.

**So the descendant arm was exercised, on a book built off the pin, through the production
chain, with the freeze verified from outside, one oracle, and the unnamed members admitted on
bytes rather than on a typed name list. That is what licensing was for, and it is met.**

## THE RESULT DOES NOT STAND YET — AND THE REASON IS NOT ONLY THE MISSING BLOCK

The addendum has already routed the absent per-arm futility block. **The defect underneath it
is sharper and a new block will not fix it on its own: the running tally is computed on the
wrong key.**

```
running_tally: {"G_so_far": 2, "G_declared": 7, "days_remaining": 5,
                "negative_or_zero_days": ["2026-09-07", "2026-09-08"],
                "attainable_minimum_p_at_G_declared": 0.015625,
                "tolerance_negative_days_at_G_declared": 0,
                "computed_not_typed": true}
ANY_ARM_ALREADY_DEAD: null          STOP_ADVICE: null
```

**`negative_or_zero_days` lists 2026-09-07 — and on 2026-09-07 HAZARD was `+4,925.363903`.**
The tally collapses two arms into one list of days, so a day counts as negative when one arm
was positive. Three consequences follow from that one key:

1. **`ANY_ARM_ALREADY_DEAD` is `null` while HAZARD is dead.** Its signs are `+` then `−`; at
   `tol = 0` one disagreement ends it. Computed (REVIEW 224 §2): with G = 7, one minority day
   makes the best attainable conjunct-(a) p `2·(1+7)/2⁷ = 0.125`, against `0.015625` at
   unanimity — **futile by a factor of 2.5 on the p it could best achieve**, and Holm only
   tightens it.
2. **`STOP_ADVICE: null`** follows from the same miss.
3. **`computed_not_typed: true` sits beside it.** It *is* computed — on the wrong key — which
   is worse than a typed value, because it reads as verified.

**So the re-emit must change the key, not add a field.** If DE 346 lands
`emit.futility.<arm>` over both days but leaves `negative_or_zero_days` day-keyed, the same
error survives in a field that now looks authoritative. The per-arm sign history is the unit:
`{arm: [(day, sign)]}`, and futility falls out of it.

**Second reason, independent and older.** The unconditional per-window table is **43 rows with
every value null** — `utc`, `gap_seconds`, `n_gap_intervals`, `share_of_day_gap_time`, **0 of
43 non-null**. That is REVIEW 190's day-one defect, unfixed and now reproduced on day two. The
ruling asked for the unconditional gap table; a table of nulls has its shape and none of its
content.

*(Third, minor and honestly handled: `progress_emit` is `UNAVAILABLE` with a named refusal —
09-08 has no point-estimate receipt — so day two has no cent-level progress line. Recorded, not
hidden.)*

**What the record gets right and should keep:** `TRIPWIRE_STATUS:
NOT_APPLICABLE_SINGLE_BOOK_NO_SUPERSEDED_PAIR` with the pair-glob it searched and
`n_superseded_found: 0`; `cohort_agreement` with `waiver_needed: false, fields_differing: []`;
`n_draws 500`, `seed_cli 0` with the derived per-arm seeds; both cells cited by absolute path;
and the day's book and oracle shas at the top level — which is exactly the naming REVIEW 224 §4
asked for, so a second run of 09-08 cannot later be substituted.

**Still owed from REVIEW 224 §4, and not a code matter:** record that the promotion criterion
was **value-independent** — that this run would have been day two's result with any D and any
p. The criterion was stated in a message that already quoted `D −49,303.58, p 0.0758`, and only
that sentence separates a ruling from a selection.

## (a) — BOTH ARMS REPRODUCE, EXACTLY

```
arm                      day one (landed)        rebuild at the freeze   to the cent   p
CONDVALUE_X_SKEW         -14645.078818000005     -14645.078818000005     YES           0.181637 -> 0.169661
HAZARD_OVER_SKEWED_REF    +4925.363903000005      +4925.363903000005     YES           0.728543 -> 0.706587
book 887a97eb41e9   oracle 172a93073ead   n_draws 500   admitted_by DESCENDANT (both)
```

**Both arms identical to every printed digit**, on a **rebuilt book** and a **later ledger**
than day one's. The p moves on both, and the reason is structural rather than incidental:
`seed_derivation: seed_for(book_sha, arm)` — the seed is a function of the book, so a rebuild
*must* draw a different null. The converse is the load-bearing half: **on an unchanged book p
would have to reproduce too**, and a p that moved there would be a defect.

*(The combined record for (a) does not exist in `fwd_a_0907_rebuild/` at my read — only the two
cells. I read the cells; the comparison above is against day one's landed combined record.)*

## VERDICT, THE TWO ANSWERS SEPARATELY

- **LICENSED — yes.** All four criteria met at the artifact, (a) green on both arms, (c) green
  since `2fff936`.
- **DOES THE RECORD STAND AS DAY TWO'S RESULT — not yet.** Two fixes, one of them not the one
  already routed: the tally must be keyed on the **arm** (and must then report HAZARD futile),
  and the unconditional gap table must carry values. Neither needs a rerun — both are emit-side
  and the cells are untouched, which is why re-emitting in band is the right move.

## SCOPE

Closed over: the 15:52 combined record read key by key, including both cells, the top-level
evidence fields, the whole `stage0` block and the whole `emit` block; the tally recomputed
against both days' per-arm D; both 09-07 rebuild cells compared to day one's landed cells; the
day-two output directory checked for a superseding record (none). **Not closed over:** DE 346's
re-emit, which does not exist; (a)'s combined record, which does not exist; the 43 window rows'
provenance, beyond counting the nulls.

## ROUTED

1. **DE — the tally's key is the arm, not the day** (§2). A futility block over a day-keyed
   list reproduces the error in a field that looks authoritative.
2. **DE — the unconditional gap table is 43 rows of nulls** (§2), REVIEW 190 unfixed.
3. **Coordinator — record the promotion as value-independent** (§2 end). One sentence, and it
   is the only thing between a ruling and a selection.
