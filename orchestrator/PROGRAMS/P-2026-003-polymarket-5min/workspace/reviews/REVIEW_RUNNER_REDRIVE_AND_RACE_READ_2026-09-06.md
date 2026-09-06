# REVIEW — the runner is APPROVED for the 09-03 smoke; the race-read declaration is APPROVED to open with two items for v2

**Filed** 2026-09-06T04:41Z (clock read before composing) · reviewer seat
(pm-codex) · tip `46030b1` · no code fixed · no write under `data/` · nothing
sealed opened · **no data touched.**

**ROUTING — CHECKED**, everything driven by me under the rule-20 wrapper.

---

# (A) THE RUNNER — **APPROVED for the 09-03 smoke once BE's book exists**

All three of my `efba2b6` conditions are closed, and I drove each one.

## A.0 Both selftests reproduce my count against the coordinator's

```
de_multiday_design_declaration --selftest   PASS -- 53 checks
de_multiday_gate1_runner       --selftest   PASS -- 42 checks (42 run + 0 skipped)
```

**And before I restored my symlink, both REFUSED — correctly, and on the two
defects they were built to catch:**

```
DesignRefused : "the ledger root resolves to /home/yuqing/ctaNew-wt-rev/data, not
                 the declared /home/yuqing/ctaNew/data. A day set derived from the
                 wrong tree is not a smaller day set, it is a different question --
                 and the failure mode is an EMPTY answer that looks like a result."
RunnerRefused : "3 pinned model file(s) do not match their declared digest ...
                 this refuses the RUN, not a day."
```

**That is my §C finding and my R6 finding, both closed at the mechanism and both
firing against a real environment defect rather than a fixture.** The design
module's refusal reproduces my own wording of the hazard.

## A.1 The R7 assertion is now relative to the ruled set — **and the falsifiers are the two the round named**

`de_multiday_design_declaration.py:1255–1275`:

* **FALSIFIER A (grow):** planting `2026-09-06` as a seventh qualifying day —
  *"which is what 00:06Z on 09-07 will do"* — leaves every relation true
  (`ga == gq and gb <= ga`, plus the read-state predicate on Set B).
  The comment names the defect it replaces: *"The old assertion asserted 6, 6 and 3
  and would have gone RED on schedule."*
* **FALSIFIER B (shrink):** removing a ruled day from the qualifying set must
  refuse.

**The assertions are now set relations, not counts.** My blocker is closed, and it
is closed the right way — the relation is checked, the arithmetic is left to the
ledger.

## A.2 The cascade digest is now BOUND to the draws — **driven in five directions**

The mechanism is (b): the runner **verifies a provenance block on draws it
receives**, and it **recomputes** rather than compares. `verify_draw_provenance`
(`:240–278`) states the gap it closes verbatim — *"the digest says which cascade
EXISTS, not which one produced these draws"* — and checks four fields, with the
seed **recomputed** as `seed_for(book_digest, arm)` so a forged block cannot simply
declare one.

Driven:

| provenance | result |
|---|---|
| all four bind | **ADMITTED** |
| wrong `module_sha256` | **REFUSED** |
| wrong `book_digest` | **REFUSED** |
| wrong `seed` | **REFUSED** |
| wrong `arm` | **REFUSED** |
| **no provenance block at all** | **REFUSED** — *"A verified module that never touches the numbers verifies nothing."* |

**Five mutations, five refusals, and the positive control admits.** My item 2 is
closed.

## A.3 `--fixture-run` is proven data-free, and the offline accounting is computed

Driven from my worktree:

```
status FIXTURE_RUN_NO_DATA · G 3 · battery PASS
no_path_under_data_was_opened = true
expected_checks_in_the_source 42 · n_checks_run 36 · n_checks_skipped_offline 6
run_plus_skipped_equals_source_expected = true
skipped_offline: ["R6 positive control (reads the pinned model files)", ...]
```

**36 + 6 = 42, computed and asserted rather than reconciled by hand** — the same
shape as the receipt-count fix, applied to a two-mode battery where the two modes
would otherwise disagree on the total. And the skip list **names** what it skips, so
a shrinking battery cannot hide behind "offline".

**The non-vacuity check exists in the source** (`:886–896` — *"the instrument is not
vacuous — it DID observe the parameter file being read, so a zero above is a
measurement rather than a silent no-op"*), and it is **one of the six checks skipped
offline**, which is correct: it cannot run in the mode that has nothing to observe.

**One small gap:** the receipt carries the boolean `no_path_under_data_was_opened`
but **not the witness** — no count of paths observed, no list. A reader of the
receipt alone cannot tell the instrument could have fired; that evidence lives in
the source and in the online mode. **One field (`n_paths_observed`, and the list)
would make the receipt self-evidencing.** Not a blocker.

## A.4 Verdict on the runner

**APPROVED for the 09-03 smoke once BE's book exists.** My three conditions are
closed and driven. The two items I would still like — the witness field above, and
R2's consumed-hour overlap number from the design round — are reporting
improvements, not gates.

---

# (B) THE RACE-READ DECLARATION — **APPROVED to open**, with two items for v2

## B.1 The statistic is inherited, not new — and the one thing that moved, moved toward the ruled unit

**The estimand is entirely pre-existing, each element citing its authority:** NET
CENTS against the frozen per-coin **incumbent** (not a base rate, rule 9); the unit
is the **ACTION**, de-duplicated (rule 2, *"measured 1.99 rows/fill, max 23"*); each
tranche valued **at its own time and level** (rule 3); latency **L = 50 ms** in the
estimand (rule 7); pairing **BY_THRESHOLD** per R-497(F)(4). **Nothing here is
chosen now.**

**What changed from the interim is the CLUSTER UNIT, and it moved to the ruled
one:** *"the interim read had G = 1 and had to disclose the WINDOW as a weaker
substitute. Here the ruled unit is available and is the unit used — which is the one
thing five days bought."* `weaker_than_ruled: false`.

**That is not a choice after seeing.** The ruled unit was always the UTC day; the
interim used a substitute because G = 1 made the ruled unit unavailable. Moving to
it is returning to the declaration, not leaving it — **and the direction of the
change is toward the stricter unit, not the more favourable one.**

**And the multiplicity is READ, not typed:** `m = 2`, `read_not_typed: true`,
`source: be_freeze_audit.rule12_conjuncts()['d_multiplicity_at_freeze']`,
`recorded_in_the_frozen_bytes: true`. ✓

## B.2 The cluster treatment and the floor are stated up front

`cluster_disclosure`: ruled unit UTC day, unit used UTC day, G = 5, and the
permutation floor **computed here, not quoted** — 2^5 = 32 assignments, smallest
achievable one-sided p **0.03125**, **0.0625 at m = 2**, `clears_0_05: false`,
smallest clearing G = 6. And `intervals_claimable` refuses an interval **not because
of G** but because R-529(A) makes the read directional — *"an interval would imply
an inferential claim the ruling forbids."* ✓

R-529(A) is carried verbatim and up front, and the `NOT_reported` list forbids any
adjusted p presented as clearing a bar, any interval, and any winner statement. ✓

## B.3 The re-reads are DISCLOSED and deliberately not weighted — **and that is where the floor should have two numbers**

`two_of_the_five_are_re_reads: [20260901, 20260902]`, and the declaration is
explicit: *"Their contribution to a 5-cluster sign test is not a fresh draw in the
sense 09-03..09-05 are, and a reader counting five independent days would be
over-counting. It is DISCLOSED; this declaration does not resolve how to weight it —
that is the coordinator's and the USER's."*

**Refusing to decide the weight is right.** But **the floor is computed at G = 5
only** — 0.0625 — while the same block says five independent days would be an
over-count. **v2 should compute the floor at both readings**: at G = 5 it is 0.0625;
on the pessimistic reading where only the three first-openings are fresh, 2³ = 8
gives a smallest one-sided p of 0.125 and **0.25 at m = 2**. **Reporting the
optimistic floor with a prose caveat puts the better number in the field a reader
resolves and the worse one in a sentence.** Both, computed, and let the USER weigh.

## B.4 What is opened is derived from the WRITER, and what stays sealed is justified

`what_is_opened.OPENED_BY_THE_READ`: `per_coin_scores` and `report` from
`be_forward_day_SEALED_scores_<DAY>.json`, and — the part that matters —
*"source_of_this_field_list: be_forward_day.seal(), **read at the WRITER rather
than by opening a sealed file**"*. **The field list was obtained without consuming
anything.** ✓

`STAYS_SEALED`: the `SEALED_feed_<DAY>.jsonl` is not opened, because *"the read
needs the per-action scores, not the feed, and opening more than the estimand needs
is consumption without purpose (rule 11)."* ✓ **Rule 11 applied to the scope of a
read, not just to its timing** — which is the stricter reading and the right one.

Each day pins its receipt sha256; 09-03's is `9aec99cff1cfc9e6`, **which I verified
independently two rounds ago.** ✓

`opened_and_sealed_use_distinct_outdir_ROOTS ... not merely distinct filenames
within one outdir (R496-R6)` ✓

### **v2 item — what the read WRITES is not stated**

The declaration says precisely what is **read**. It does not say **what artifact the
read emits, to which path, or that the sealed files are left byte-identical
afterwards.** For an act whose whole safety rests on "opening is not modifying",
**that is the one sentence I would want before it happens** — and it is checkable
after the fact only if the pre-digests are recorded. The per-day receipt digests are
pinned; **the SEALED_scores digests are not.** v2 should pin them and require a
post-read re-verification.

## B.5 Rule 11 is in force with the escape closed

`may_not_be_chosen_on_what_is_seen` names parameter, threshold, horizon, budget,
candidate, winner criterion, **the day set** and **G**. And:

> *"the race is finished after this: there is no sixth day to add afterwards that
> would rescue significance — adding days AFTER seeing this read is selection on the
> outcome. **If G = 6 is ever wanted it must be declared BEFORE this read is
> opened.**"*

**That closes the one escape a directional-but-not-significant result invites**, and
it closes it in advance. It also interacts correctly with the Gate-1 design round,
where G = 6 is exactly what was being weighed.

## B.6 Does the read touch the Gate-1 books or the arms? — **No, by construction; but the declaration does not say so**

The read opens `be_forward_day_SEALED_scores_<DAY>.json` — BE's forward scorer's
output. The Gate-1 objects are the day **books** (`asm` + reference, which BE has
not yet built) and the arms' pinned thetas. **Different files, different producer,
no shared path**, and `AND_THE_SECOND_LIMIT_WHICH_IS_INDEPENDENT` states the
adjacent fact: *"the prior race cannot validate the CHANGED pipeline, because the
pipeline the race scores is not the one V2 changed."*

**But the declaration never asserts the separation directly.** It is true and I
checked it; it should be a field, because the two lines of work now run in parallel
on the same days and a reader will ask.

## B.7 Verdict on the race read

**APPROVED for the coordinator/USER to open.** v2 (or an addendum before the read)
should carry:

1. **The permutation floor at BOTH readings** — G = 5 → 0.0625, and the
   three-fresh-day reading → 0.25 at m = 2 (§B.3).
2. **What the read writes, where, and the SEALED_scores digests pinned** for a
   post-read byte-identity check (§B.4).
3. *(one line)* **the explicit statement that no Gate-1 book or arm artifact is on
   the read's path** (§B.6).

None of the three changes what the read may conclude. The first is the only one I
would want fixed **before** the numbers exist, because a floor is the kind of
figure that gets quoted from the field rather than the caveat.

---

## CONTEXT

Far below the 80% reset threshold.
