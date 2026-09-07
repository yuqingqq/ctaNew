# REVIEW 91 — PART A: **GO E1 MAY PROCEED**, and GO #8 re-cleared · PART B: BE 93, DA 122

**Reviewer (pm-codex), 2026-09-07T07:2xZ. Read at `92c2d56` in `~/ctaNew-wt-rev`; Part A
established at `df2158b` and unaffected by `5ecb3d7`..`92c2d56` (DA 122, which touches no DE
module). Read-only: no heavy unit, no lock, nothing written under `data/`, never `--open`,
and **no feed opened, hashed or read** — including in BE 93's verification, which needed
none. CHECKED = I went to the artifact or ran the code; AGREED = I read the same summary.**

---

# PART A — **GO E1 MAY PROCEED**

> **GO E1 MAY PROCEED at runner `f1f5947254e94b64…` / early-read `fbd01eb1765c9f4b…` /
> params v17 `81b2c2910b3c4799…` / exit maps v8 `bbc8bacfddef8585…`.** All four digests
> recomputed at the tip and equal to the dispatch (**CHECKED**). REV 90's §A0 is closed at
> the artifact, driven by me, and DE closed more of it than I asked.
>
> **No hold on this launch.** Two items are routed, neither gating: §C1 (`de_early_read`
> ships no shared-falsifier cell — a detection gap, not a correctness one) and §C2
> (`PARAMS_REL` records its digest but does not assert it).

## §A1 The §A0 defect, re-driven exactly as I drove it before

```
seal(arm, 4, 4)   sealed False   seal_status 'UNSEALED -- 4 of 4 days complete under the bar this call was given'
seal(arm, 3, 6)   sealed True    seal_status "SEALED -- 3 of 6 days complete. Every economic field is ABSENT…"
```

(**CHECKED**, my own call on a 09-06-shaped arm-day.) The literal is gone; **both branches
compute**; the sealed branch's string is the one the four landed days carry, unchanged. The
economics are carried at the ruled bar (`D_E0`, `Z`, `p_location`, `null_mean`, `null_sd`,
`null_draws_summary`) with the three counts, and absent at the six-day bar — so the change
touched the unsealed branch only, which is what it was supposed to touch.

**The falsifier I asked for exists and is the right shape.** DE reads `N of M` **out of the
string** rather than comparing it to a sentence someone typed — *"a cell that matched the new
wording would go green on the next rewording and prove nothing"* — and the known-bad asserts
the OLD literal FAILS the same predicate (`"UNSEALED_ALL_DAYS_COMPLETE"` contains no `N of M`),
so **the cell measures a delta from the behaviour that was landed**, which is rule 20's
own-baseline clause. A third cell checks the sealed branch field by field against what the
four landed days emitted (**CHECKED**, read at the diff; the whole battery runs green below).

**§A0(3), the two `G`s — closed, and DE avoided the trap I would have set.** The bare
`"G": params["G"]` is gone from the day-run receipt's top level (the three remaining
`"G": params["G"]` sites are other objects: `days_complete_when_this_day_started` at :1297,
the aggregate at :2454, and a different payload at :10790 — **CHECKED**). In its place one
object naming which G is which, with `design_G_from_params`, `params_file`,
`n_days_complete_at_this_emit` and `the_bar_this_run_sealed_against`. **DE kept the version
out of the key name deliberately** — *"a key called `design_G_from_params_v15` is a literal
that must track a moving thing, and `PARAMS_REL` is the thing that moves"* — which is my own
recurring finding applied against my own suggestion. It is right and I withdraw the shape I
proposed.

**And DE closed the residue I had NOT asked for.** `"4 of 4"` is true of the bar and says
nothing about the six, so `de_early_read` adds a wrapper-level `seal_standing` whose line is
**computed**, not typed. I evaluated the exact expression the heavy path runs at emit —
after ~70 minutes of work, where a KeyError would cost the whole run (R-747's class):

```
EMIT-TIME EXPRESSION OK -> ruled_days_in_the_population = 6
"UNSEALED under the USER's ruling R-754: 4 of 6 ruled days, read early on the user's
 instruction. NOT all days complete, NOT a validation, no interval."
```

(**CHECKED**, against the real ruling; `the_six_day_population.days` resolves.) That sentence
is the one the user needs beside the table, and it computes both numbers.

**Batteries at the tip, all run by me:** runner **350 checks / 0 disarmed / 0 skipped**, rc 0
(30.7 s, 867,892 KiB); `de_early_read` **16**; design declaration **117** under `-m`;
`de_receipt_correction` **29** — the last because it is the receipt-schema consumer and the
receipt's schema changed. The fixture run emits `clean true, n_skipped 134, n_disarmed 0`.
Rehearsal on the real 09-03: **READY, G 4, EXPLORATORY, NONE_BELOW_FIVE_DAYS, full pair
true** (**CHECKED**).

**Scope of what I did NOT do:** I did not emit a real day-run receipt. `--synthetic-day` is
gated by a declared fixture-name list and a launch-form guard, and I did not force it. The
receipt's new shape therefore rests on three things I did establish: the emit dict read at
the diff, the grep showing no bare `G` survives at that level, and the 350-check battery
whose DE 125 cell drives both seal branches on a 09-06-shaped arm-day.

*One observation, explicitly not a hold:* one bare `G` remains in the artifact —
`day_run.days_complete_when_this_day_started.G` = 6 — but it sits inside a block whose name
states what it counts, beside its own `n_days_complete`, so it is legible. §A0(3)'s defect was
the *smaller* G sitting outside the block with nothing saying which was which; that is fixed.

## §A2 **GO #8's clearance — RE-ISSUED** against the runner's current bytes

REV 89 cleared `ad15ddf125be8dec…`; the runner has since moved twice, to `520f33f4…`
(DE 121's early-read parameter) and now **`f1f5947254e94b64220c77bd…`** (DE 125). Tonight's
09-07 day run executes from these bytes, so the clearance must be re-issued or refused.

**RE-ISSUED, on four things I checked rather than assumed:**

1. **The sealed path is behaviourally unchanged.** For a day run outside the ruling
   `early_read` is `None`, so `seal(r, n_days_complete, params["G"])` is called exactly as
   before; I drove `seal(arm, 3, 6)` and got the string the four landed days carry.
2. **The receipt's schema change breaks no consumer.** DE claims a sweep found none; I ran my
   own. Every hit for a top-level `G` read is the **race-read** family's G
   (`be_race_reader`, `be_race_read_declaration*`, `da_race_read_verify`) or
   `de_early_read`'s own wrapper G — **nothing reads a day-run receipt's top-level `G`**
   (**CHECKED**). `de_receipt_correction`, the receipt-schema consumer, passes 29.
3. **The change is disclosed in the artifact rather than discovered later.**
   `no_bare_G_key_here` states that the four landed receipts keep the old bare `G: 6` and
   receipts from here on carry the block instead — a schema change between days, said so at
   the place a reader meets it (rule 4).
4. **The battery grew with the change** (347 → 350) and passes, with both new cells red-first.

**The condition stands, unchanged from R-747's cost:** nothing refreshes, checks out, edits
or lands from `wt-de` while a day-run or `deEARLY<D>` unit is running.

---

# PART B

## §B1 BE 93 — does `--verify` convert v1/v2 from scratch-built to repo-verified?

**YES for the content, and by a stronger route than my §B4 proposed.** I asked for a
re-derive from *the feeds and their producing receipts*. BE built it to prefer the
**producing receipt** and to fall back to the file's bytes only if no receipt block exists,
reporting **which source answered** per day — because *"'verified' from a producer's
attestation and 'verified' by re-hashing are not the same claim and a reader must be able to
tell them apart"* (**CHECKED** at `rederive_day`, which returns on the receipt and never
parses a feed).

That ordering is better than what I asked for, and the reason matters: **re-hashing a feed
would prove only that the file matches the pin TODAY. The producing receipt proves the
PRODUCER recorded those bytes at emit** — it reaches back to the moment of production, which
is the moment the pin is about.

The result, from my own run of the battery (its cells compute the real verification):

```
v1 {'VERIFIED_FROM_RECEIPT': 3, 'CORROBORATED_ABSENT': 2}
v2 {'VERIFIED_FROM_RECEIPT': 4, 'CORROBORATED_ABSENT': 2}
   -- every present day answered by its PRODUCING RECEIPT, no file needed hashing
```

(**CHECKED**; 14 checks, rc 0, including the shared `declaration_chain --falsify` cell.) The
09-06 row that the second read actually depends on re-derives on `path`, `sha256`
`0dfa62f3…` and `bytes` 236,128,877.

**And here is the part of my §B4 that this settles more completely than I expected.** I wrote
that the content was re-derivable but the **timing** claim — *pinned before any read* — was
what a scratch producer could not support. That framing was half right and I sharpen it now:
the worry behind "pinned before" is a **degree of freedom** — that a pin's value could have
been chosen after seeing the data. Re-derivation from the producing receipt **removes the
degree of freedom entirely**: the pin's value is forced by the producer's own record, and any
other value is a `MISMATCH`. What no artifact can establish is whether someone *read* the feed
between the producer's emit and the pin's write — but that no longer affects the pin's
correctness, only its authorship. **So: v1 and v2 are repo-verified in the sense that
matters, and rule 12's precedent (a scratch-dir builder voided a freeze) does not bite here,
because a pin's claim is reconstructible from a repo-produced artifact and a fitted
candidate's is not.**

**Two things I want on the record as the real strength of this instrument, neither of which I
asked for:**

- **An absent pin is a CLAIM and is falsified in both directions.** `exists: false` is not
  skipped: `CORROBORATED_ABSENT` requires **both** sources to agree there is nothing, and the
  same pin reads `MISMATCH` when a receipt does produce a feed. *"Skipping `exists: false`
  days would have made two of the five vacuous"* — precisely the vacuity rule 16 names.
- **`NO_SOURCE` is refused, never silently passed** — *"a check that cannot run is not a check
  that ran" (R-649)*. That is the distinction between "nothing to re-derive from" and "it
  re-derived and agreed", and it is a status rather than a pass.

### What `CORROBORATED_ABSENT` proves for 09-01/02 — and what it does not

**It proves:** two independent repo-produced sources agree there is nothing — the producing
receipt carries no `feed` block, and no file is at the pinned path. The pin's `exists: false`
is therefore tested, not asserted, and the test can fail (the battery drives the failing
direction on a fixture).

**It does not prove that no feed ever existed.** Absence today is consistent with a feed
having been produced and lost, and the pins themselves say as much: **09-01 and 09-02 carry
NO feed digest but DO carry `v2_scores_sha256`** (`aca22317…`, `7522786d…`) — so something
was produced for those days and the feed is what did not survive (**CHECKED** at the pins).
That matches the record: R-549 / RESULTS have them as READ-BUT-UNRECOVERABLE — consumed under
rule 11, with only seal-relocation receipts left.

**So `CORROBORATED_ABSENT` corroborates *unrecoverable*, not *never produced*,** which is
exactly the claim the first read's `G = 3` rests on: *consumed and unavailable are different
facts and both hold*. **For the SECOND read it is irrelevant** — 09-01/02 are not in v6's
READABLE set. The two days it constrains are the first read's, which is closed.

**And my REV 90 §B3 minor item is closed in passing:** the stale-scoped `all_five_present`
now has `all_pinned_days_present` beside it, derived, with the emitter recomputing rather than
inheriting the parent's `true` — and v2 is not edited (**CHECKED** at the battery's cells).

## §B2 DA 122 — the importer cell, and it caught one I had missed

**The rule is a cell now, bound to the resolution surface exactly as §B5 asked**, and DA added
a conjunct I did not think of: **the drive must be inside a BATTERY function**, because the
module that defines the shared helper carries the subprocess call in the helper's own body —
*"that is the helper, not a cell that runs when its battery runs"* — and a docstring
mentioning `--falsify` is prose, not a drive. The set is computed from the AST.

My own run of the census:

```
chain_resolution_surface: n_in_the_surface 13 | n_missing_the_cell 3
  missing: be_race_read_declaration_v3.py , be_rule22.py , de_early_read.py
  verdict: REFUSED_A_RESOLVER_SHIPS_NO_FALSIFIER_CELL
outside the surface (2): be_heavy_peaks.py , build_state_tape_v2.py
  -- "imports the module but references no resolution symbol -- a helper import is not a resolution"
```

(**CHECKED**.) The two `plain_create_mode` importers are correctly excluded — my refinement,
implemented. **Of the three flagged, `be_rule22.py` is one I did not find**: it *has* a drive,
at line 178, but inside `shared_falsifier` rather than a battery, so a presence test would
have passed it and the new conjunct catches it. That is the cell doing better than the
reviewer who asked for it.

**`n_scanned` beside `n_judged` — closed, and better than asked.** Not two numbers but a
classified breakdown that sums:

```
n_literals 358 | n_scanned_naming_a_non_head 50 | naming_a_non_head (judged) 4
the_scanned_set_by_class: ADMITTED_HASHED_AGAINST_A_RECORDED_DIGEST 2 | MARKED_AND_NOT_A_PIN 2
                          NOT_A_PIN__IN_A_SUPERSESSION_FIELD 7 | NOT_A_PIN__NO_OPEN 37
                          REFUSED_UNMARKED_NON_HEAD 2                       (2+2+7+37+2 = 50)
```

The 46 that are not judged are **classified**, not merely counted — so a reader of the summary
can no longer mistake 4 for the whole scan, which was the defect.

**`e2_a_episodes.py` classified — and the classification is a category I did not offer.** I
gave DA two boxes (reader of history / stale consumer). The right answer was a third: **a
KNOWN-BAD DRIVE that names the real superseded versions on purpose, to prove the chain-head
check FIRES.** Under the ruled predicate that makes it a reader of history, and it now cites
by the pair — the census reports `the_digest_predicate.admissible: true`, `n_digests 2`,
provenance field `recorded_by`, compared at lines 592/593/601 (**CHECKED**). The census still
reports it as `MARKED_AND_NOT_A_PIN` **and** as admitted by the digest predicate; that is two
independent facts side by side, not a contradiction, and reporting both is right.

`da_nonhead_census` battery: **30 checks**, rc 0.

---

# §C NEW, AND HOLDS

**No holds on GO E1 or GO #8.** H1 (REV 89) still stands against the six-day Gate-1 read only,
and does not reach either launch.

**§C1 — `de_early_read.py` ships no shared-falsifier cell** (DA's new census flags it, and I
confirm at the code: it imports `declaration_chain` and references `resolve_head`, with no
drive). **Not a GO E1 blocker**, and I want the reason on record rather than the verdict
alone: this is a *detection-coverage* gap, not a correctness one — the shared module's own
falsifier passes (15 cells, 0 failures) and ten other importers drive it, so a regression
there would be caught; what is missing is that it would not be caught *by this module's
battery*. One cell. Routed to DE with §C2.

**§C2 — the programme's most load-bearing non-head literal asserts nothing.** The census's two
`REFUSED_UNMARKED_NON_HEAD` entries are:

| where | names | head | why it is there |
|---|---|---|---|
| `be_race_reader.py:2420` | `be_race_read_declaration_v1.json` | v6 | a scratch fixture in `mkdtemp` — **but `flows_into_an_open: true`, so the census is right to refuse it and my REV 90 §6.3 "nothing to do" was too quick.** Mark it (an identifier the census's own vocabulary recognises) and the refusal clears honestly |
| `de_multiday_gate1_runner.py:66` | `de_multiday_gate1_params_v15.json` | v17 | **`PARAMS_REL`, and it is deliberate and correct**: the computation must stay v15's so the sealed six-day path is byte-identical outside the ruling (R-757). It carries **no digest** |

The second is the one worth a round. The early read's whole claim is *the computation is the
sealed runs'*, and that rests on v15's bytes. Today: `load_params` **records** the digest
(`record_input_digest("params", PARAMS_REL)`) and `de_early_read` **records** it again
(`computation_params: {path, sha256}` hashed at emit) — so a mismatch is *detectable after
the fact*, and v15's immutability is enforced by a checker someone runs. **Nothing asserts it
at the read.** The comparison target exists: the sealed receipts declare the params by pair
and DA 117 verified `92858fc7f9493f8e…` against the digest the receipt declares. **One line:
`de_early_read` compares its `computation_params.sha256` against the digest the day's sealed
receipt records, and refuses by name on a mismatch.** That closes the loop where the predicate
belongs, and it makes the census's refusal go away for the right reason rather than by
marking.

**Routed:**

| # | to | finding | kind |
|---|---|---|---|
| 1 | DE | `de_early_read` ships no shared-falsifier cell (§C1) | routed |
| 2 | DE | assert `computation_params.sha256` against the sealed receipt's recorded params digest, at the read (§C2) | routed |
| 3 | BE | `be_race_reader.py:2420` — mark the scratch fixture so the census's refusal clears for the right reason; **my REV 90 call of "nothing to do" was wrong** (§C2) | routed, mine |
| 4 | BE | `be_race_read_declaration_v3.py` and `be_rule22.py` — the falsifier cell; `be_rule22`'s drive is outside a battery | routed |

**Closed by this round, no action:** REV 90 §A0 in all three parts (both branches compute; one
G object; the falsifier reads the counts out of the string); §B3's `all_five_present`; §B4
(the pins re-derive from the producing receipts); §B5 in all three parts (the importer cell
bound to the resolution surface, the E2-A classification, scanned-beside-judged).

**Noted, not reviewed (REV 92, per the dispatch):** the checker's `FORKED_BY_EDIT_AND_SUPERSEDED`,
commit names and `pre_edit_digest_pinned_by`. R-762 reports v7 reads `AND_SUPERSEDED` and that
**nothing pins its pre-edit bytes** — which answers the question my §B7 left explicitly
unmeasured. I have not verified it; **AGREED**, and REV 92 checks it.

# §D WHAT I DID NOT ESTABLISH, AND MY OWN ERRORS

- **Not established:** a real day-run receipt's emitted shape (see §A1's scope); DE's
  `--synthetic-day` path; the sizing figures; MEM 250's sweep; the checker's new statuses.
- **AGREED, not established:** that BE's v1/v2 were written by *scratch scripts* specifically
  (I established that no emitter existed before BE 92 and that both versions now re-derive).
- **A third probe error of the same class, caught before it became a claim.** Reading the
  census's marked entries I probed `m.get('hashed')` — the key is `the_digest_predicate` —
  and got `admissible: None`, which I was one step from reporting as "DA classified it in the
  code but the census still refuses it." The full entry says `admissible: true`. Three rounds
  running, the same cure: **assert the key exists before reading what it says.** I am putting
  it here rather than in a footnote because the failure mode is stable and mine: I probe a
  nested structure by guessing a plausible key name, and a `.get()` on a wrong key returns
  `None`, which reads exactly like a real negative.
