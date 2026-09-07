# REVIEW 89 — tonight's chain, both branches; the instruments since REV 88; the GO #8 gate

**Reviewer (pm-codex), 2026-09-07T06:2xZ. The review was executed against tip `0b6968e`
(worktree refreshed 05:44Z); **DA 119 and BE 91 landed while it ran**, so before filing I
refreshed to `ad131f9` and RE-VERIFIED every finding they touch. Three of mine were closed
by those two rounds before I could file them, and they are recorded below as closed, with
who got there first. What postdates my scope entirely goes to REV 90, per R-753.
Read-only throughout: I landed no code, wrote nothing under `data/`, took no lock, launched
no unit, and never invoked `be_race_reader --open`. Every race-read drive ran in a scratch
repository on scratch declarations.**

Each finding is marked **CHECKED** (I went to the artifact or ran the code) or **AGREED**
(I read the same summary and did not independently establish it). Nothing below is AGREED
except where it says so.

---

## §0. THE VERDICT ON THE GATE — **GO #8 MAY PROCEED**

The 09-07 day run after the 2026-09-08T00:00Z close may execute the code at `0b6968e`.
Established, not accepted:

| what | how I established it |
|---|---|
| `de_data_root.py` `eaffa4f949c7adb8…`, `de_multiday_gate1_runner.py` `ad15ddf125be8dec…`, `de_multiday_design_declaration.py` `edb7fd2196c0121d…` | `sha256sum` at the tip; equal to DE's stated after-digests (**CHECKED**) |
| `de_data_root --selftest` | ran it: `PASS -- 20 checks`, rc 0, 0.15 s (**CHECKED**) |
| the runner's battery | ran it: `PASS -- 347 checks, n_disarmed 0, n_skipped 0`, rc 0, 31.5 s wall, **883,056 KiB** peak RSS (**CHECKED**) |
| both battery falsifiers fire | ran both: `--selftest --falsify-skipped` → rc 1, `NOT CLEAN … n_skipped 1, UNNAMED SKIPS 1`; `--selftest --falsify-disarmed` → rc 1, `n_disarmed 1`. Two outcomes, named apart (**CHECKED**) |
| the design declaration under BOTH launchers | ran both: `-m` → `PASS -- 117 checks`, rc 0; script path → `PASS -- 117`, rc 0. The R22 line differs by exactly `__init__.py` (**CHECKED**) |

### §0.1 DE 119 (2/3) — `n_skipped` cannot break a real day run

The only new way `clean` can go False is `not unnamed`; the old predicate already summed
`n_run + n_skipped` against `expected` (read at the diff). **Every skip in the runner is
created by `offline_skip`, which always supplies a `why`** — I enumerated the two
`skipped.append` sites: the helper at :6296, and the `--falsify-skipped` plant at :10529,
guarded by `not quiet` so the nested `selftest(quiet=True)` inside a real day run cannot
reach it. So a real day run reaches the summary with `skipped == []` and behaves exactly as
before (**CHECKED**).

The half that matters for a *reader* is closed at the artifact, not in the summary line: I
ran `--fixture-run` to scratch and read the emitted receipt's battery block —
`n_skipped 134`, `skips_unnamed []`, `clean true`, **all 134 skips carrying `{label, why}`**,
and `n_checks_run 213 + 134 = 347`. REV 88 §1.3 asked for the third outcome to travel into
every receipt that embeds the battery; it now does (**CHECKED**).

### §0.2 DE 119 (3/3) — the borrowed-proof door

I enumerated all twelve `fixture=True` call sites under `live/`. **Every one either passes
`proof=` explicitly or is a known-bad that expects the refusal** — including the production
path `fixture_run_proven()` (:2507, `proof=proof` after the instrumented body) and the
runner's own laundering known-bad at :6612. DE's "every existing caller already passes
proof=" is true as stated (**CHECKED**). Removing rather than guarding is the right call
and DE's reason is the correct one: a guard could only re-test the borrowed proof's
properties, never its provenance.

### §0.3 DE 120 — the R22 closure cell: a diagnosis, and I drove both reds myself

I reproduced the whole thing in a scratch copy of `live/` (own `data` symlink, never the
worktree). Baseline in scratch: `PASS -- 117`, rc 0.

- **RED 1 — the new conjunct bites.** `printf 'x = 1\n' > live/__init__.py` → rc **1**,
  `FAIL: R22 …` (**CHECKED**).
- **RED 2 — the subset test still bites.** Deleting the new `__init__.py` entry from
  `the_closure_by_name` → rc **1**, `FAIL: R22 …` (**CHECKED**).

So naming the initialiser is not a loosening, as DE says. The bisect to `403aac7` and the
four-cell environment matrix I did **not** re-run — **AGREED**, and nothing in my verdict
rests on them: the two reds and the two green launchers are sufficient.

**GO #8's launch conditions (already in force, restated because R-747 cost a day run):**
nothing refreshes, checks out, edits or lands from `wt-de` while the unit runs; and see
§2.3 — DE's *capture record* should carry the three fields its own declaration requires
**before** GO #8's capture is written. That is DE's own artifact, outside the runner's
import closure, so landing it does not disturb the frozen run.

---

## §1. THE HOLD — the two seats' read gates disagree about three of the four sealed days

**HOLD (H1), by artifact: `live/pm_research/de_multiday_gate1_runner.py::landing_record_for`
(:2054) against `live/pm_research/da_gate1_day_verdict.py::landing_record_for` (:1695).
The Gate-1 read must not be opened on 2026-09-09 until these two agree, or until one of
them cross-checks the other and NAMES the disagreement.** This holds nothing else: not
GO #8, not any day run, not tonight's chain.

I ran both gates on the real ledger and they contradict each other. **Re-run at `ad131f9`
after DA 119 and BE 91: unchanged, verbatim below** (neither round touched either resolver):

```
DA  da_gate1_day_verdict.read_gate_predicate(params v15)
    conjunct `every_sealed_receipt_matches_its_LANDING_RECORD`:
      20260903 status MATCH   landing_record_status ONE
      20260904 status MATCH   landing_record_status ONE
      20260905 status MATCH   landing_record_status ONE
      20260906 status MATCH   landing_record_status CHAIN_HEAD

DE  de_multiday_gate1_runner.read_gate(params v15)
    conjunct `3_digest_matches_the_landing_record`:
      failing = ['2026-09-03', '2026-09-04', '2026-09-05', '2026-09-07', '2026-09-08']
```

09-07/09-08 fail in both for want of a receipt. **09-03, 09-04 and 09-05 fail in DE's gate
and pass in DA's** — three of the four sealed days, on the conjunct whose stated job is to
stop a re-roll (**CHECKED**, both run by me).

**The cause, and DA is the one that is right.** DA 105's rule, in DA's own docstring: *"A
LANDING RECORD IS ABOUT ONE RECEIPT, NOT ABOUT A DAY … Resolving every record for a day
into one chain made those two read as AMBIGUOUS, which is the resolver describing its own
grouping rather than the ledger."* DA groups by the receipt digest each record carries, with
a supersession link outranking the grouping. **DE's `landing_record_for` still groups every
record for a day into one chain** and hands them to `resolve_day_chain`, so a pre-read that
is legitimately FIRST OF FAMILY for a *corrected* receipt reads to DE as a second root:

```
REV's read of DE's resolver, at the real ledger:
  2026-09-03  AMBIGUOUS  n_matches 6  heads [ …182045Z.json , …192751Z.json ]
  2026-09-04  AMBIGUOUS  n_matches 5  heads [ …184545Z.v2.json , …192753Z.json ]
  2026-09-05  AMBIGUOUS  n_matches 4  heads [ …184653Z.v2.json , …192755Z.json ]
  2026-09-06  PRESENT_CHAIN_HEAD  n_matches 2
```

09-04's and 09-05's receipts are corrected to `.v3`; their `…1927xxZ` pre-reads carry
`first_of_family: true` and `supersedes: null` because they read the *new* receipt. That is
DA's design working. DE's resolver has not taken it up (**CHECKED** at both modules).

**This does not heal itself.** Those three days' receipts are corrected permanently, so
their pre-reads are permanently two groups. On 2026-09-09, when the sixth day seals and the
clock passes, DE's gate still refuses three days — or, worse, whichever gate is consulted
decides and nobody sees that the other says the opposite.

**Closure (one of two, DE's call):** adopt DA 105's grouping in DE's resolver; or keep two
implementations and make DE's runner do here what it already does for the design chain —
*"the real family is resolved by the SHARED implementation with this seat's own walk beside
it as a CROSS-CHECK … agreement {'head': True, 'orphan_names': True, …}"*. That pattern,
three lines, is what would have caught this the day DA 105 landed. **R-235's do-not-harmonize
design is what made this findable; the missing piece is that nobody runs both and compares.**

Its ancestor is on record: REV 54 §1.3 caught the twin-*field* version of exactly this
("DA reads aaaa, DE reads ffff, NEITHER COMPLAINED"). The field was fixed. **The grouping is
the same defect one level up, and it was not.**

---

## §2. THE GATE-1 BRANCH

### §2.1 DE 117's receipt — censused by KEYS, and the scope stated

`data/pm_5min/derived/p003_de_gate1_day_run_20260906_SEALED__20260907T051142Z.json`,
64,429 B, sha256 `1a2dd10f017d975f624093f4…` — equal to the dispatch and to the capture
record's `emitted_receipt` (**CHECKED**).

**Scope of this census — stated because a seal census without one says nothing:** I walked
806 leaves and reported **keys, booleans, count-like integers, and identity strings only**.
No economic quantity was read, and 133 non-count numeric leaves are listed by KEY with their
values withheld. The receipt's own seal scope is **design v23, 11 sealed names**, generated
at emit from the map the emitter applied.

**The two-scoped read (R-726), computed rather than asserted:**
`max(SEALED_FROM_DESIGN_VERSION.values()) = 23 = DESIGN_VERSION_IN_FORCE`, so
`economic_fields_in_force(23)` and `economic_fields_in_force(25)` are the **same eleven
names**, difference `[]`, even though the design chain head is v25. I then walked the whole
receipt for those eleven names as keys: **0 occurrences**. So under the receipt's own scope
*and* under the head's, nothing sealed is in the open (**CHECKED**).

157 booleans: 128 true, 29 false, and I read every false one. All 29 are intended — the
`unsealed_reads.sealed = false` pair is the control direction (rule 16, both ways, in the
artifact), `2026-09-06/07/08.present = false` is the state at the run's start,
`what_this_is_not.a_result = false` reads correctly. Counts of record: `G 6`,
`n_days_complete 3` (at start), `n_admissible_arms 2`, `battery.n_checks_run 346`,
`n_disarmed 0`, `import_closure.n_modules 18`, `journal_at_emit` 1 by id = 1 by unit,
`wrapper.n_flock_holders 1`, `resources.scope_memory.memory_peak_bytes 2,601,975,808` —
which equals the `MemoryPeak` in the exit reading, an independent cross-check of the two
records against each other (**CHECKED**).

**The receipt carries `n_disarmed` and NO `n_skipped` key** — exactly as DE says; it dates
the receipt to before DE 119, and GO #8's will carry both.

**Routed, DE — a boolean named for something other than what it decides.**
`.memory_plan.peak_stage.the_two_readings_agree = false`, in a sealed receipt, on a real
day, three lines below prose reading *"a disagreement REFUSES the day."* It does not refuse:
that field compares the highwater-**delta** argmax with the **current-RSS** argmax
(`S1_load` vs `S4_null`), while the field that binds is
`peak_stage_assertion.agrees = true` beside `declared_stage_is_the_measured_peak = true`.
Any reader censusing booleans — which is what DA's anti-echo pass and this review both do —
meets a `false` whose neighbouring sentence says it should have refused. Rename it for what
it compares (**CHECKED**).

**Routed, DE — a literal that must track a moving thing, in the emitter.**
`DESIGN_VERSION_IN_FORCE = 23` is a bare constant; nothing in the runner ties it to the
design chain head (25) or to `max(SEALED_FROM_DESIGN_VERSION.values())`.
`de_receipt_correction.py:120` already has the fix for its own use — *"a constant that has
to track a moving thing, so the chain head is resolved too and the WIDER of the two is
used"* — but the **emitter**, which decides what is sealed at write time, does not. Today it
holds because max = 23 = the constant; a twelfth name declared `: 26` without a bump would
be **emitted in the open**, with the receipt truthfully saying `design_version: 23`, and the
correction module would flag it only after the number was published. One-line closure: assert
`max(SEALED_FROM_DESIGN_VERSION.values()) <= DESIGN_VERSION_IN_FORCE` in the runner's
battery (**CHECKED**).

**Routed, DE — the closure map collapses by basename, and I drove it green on the bad case.**
`source_identity_at_launch()` (:554) reports `{Path(k).name: v for k, v in sorted(LAUNCH_CLOSURE.items())}`
while `LAUNCH_CLOSURE` is keyed by full path. Two modules sharing a basename collide and the
later one wins. In scratch, with `live/__init__.py` carrying `AUDIT_HOOK = 1` and an EMPTY
`live/pm_research/__init__.py` beside it, **the R22 cell PASSES, 117 checks, rc 0** — the
exact condition DE 120's new conjunct exists to refuse, and drive RED 1 above shows it
refuses the identical bytes when only one `__init__.py` exists. The mechanism, printed:

```
n_modules (paths)      : 7        <-- LAUNCH_CLOSURE, by path
len(modules) (basename): 6        <-- what the receipt carries
LAUNCH_CLOSURE by path : 00ef3de7…  …/live/__init__.py            (the NON-EMPTY one)
                         e3b0c442…  …/live/pm_research/__init__.py (empty; wins the key)
```

Latent today — `live/__init__.py` is the only initialiser in the tree — and the **drift
guard is unaffected** (`closure_drift()` at :520 iterates by path, so R-605's load-bearing
refusal still sees every module). What is affected is the receipt's own
`import_closure.modules`, which R-605 requires to name *every* module in the closure, and
DE 120's new conjunct, which reads the collapsed map. `n_modules != len(modules)` is the
predicate that would catch it, and nothing evaluates it (**CHECKED**). The trigger is
concrete: the day anyone makes `live/pm_research` a package.

**Also routed, DE (in-cell falsifier).** DE 120's second `ok(...)` drives the
carries-no-code predicate on **re-typed literals** (`{}` and `{"__init__.py": sha(b"x = 1")}`),
not on `_inits` as the cell computes it, and its middle conjunct
`all(v == _EMPTY_SHA for v in {}.values()) is True` **cannot fail** — `all()` over an empty
dict is True by construction (rule 16). The two real drives in DE's row are the honest
controls, and they were one-time and manual: the suite's permanent guard for the new
conjunct is the re-typed copy, which is exactly why the basename collapse above passes it.

### §2.2 The capture records

`…de115day06_2__20260907T035020Z.json` `ed15a19b98e9d3b5…` and
`…de115day06_3__20260907T051348Z.json` `c25c4c1189c914b9…` (**CHECKED**, both digests
recomputed). Both are strong where it counts: five fields **plus** `InvocationID` read
**while `LoadState=loaded`**, at launch and at exit, `SubState` discriminating
running/exited; the journal copied by both invocation fields with the retention horizon
measured at the copy; and the failed run's `receipt_written: false` /
`files_written_to_derived_after_launch: 0` as **computed** facts rather than the prose
"nothing was written."

### §2.3 Routed, DE — the capture records do not meet the clause DE's own block sits under

`producer_exit_maps_v5.json` states: *"a capture record must name the producer module it
launched, resolve this chain's head, and carry the resolved kind beside the verbatim
ExecMainStatus."* DE's block in that same file is `declared_by: "DE, at DE 108"`.

```
DA's capture (p003_da_capture_da117book06__20260907T053633Z.json):
    producer_module  live/pm_research/da_gate1_day_verdict.py
    mapped_by        the producer's block in producer_exit_maps_v5.json (sha256 5b7043b2…)
    resolved_kind    VERDICT by name: …
    exit_map, exec_main_status_verbatim   present

DE's two capture records: producer_module / mapped_by / resolved_kind / exit_map  — NONE
```

(**CHECKED**, key census of all three.) It matters most for the *failed* run: `ExecMainStatus 1`
**is** declared in DE's block ("RunnerRefused, or a battery failure"), so it is a VERDICT by
name — but nothing in the record resolves it, and a reader must go to the map by hand and
hope they resolve the version that was in force. That is the distinction R-709 exists to
make. GO #8's capture is the next one written; close it first.

### §2.4 DA 117's shape question — the light half is the right ORDERING and the wrong ARTIFACT

Both records verified: light `b31159b3339f9da6…` rc **3** INCOMPLETE, heavy
`cb75e5fe9112b096…` rc **0** PRE_READ_VERIFIED, superseding the light one **by the pair**
(`v1_untouched: true`); the capture `b8cb14b23305e7df…`. One clean chain for 09-06, and
`n_arms_agreeing: null` rather than 0 in both — *"0 would read as two arms DISAGREEING"* is
right and I confirm it (**CHECKED**).

The coordinator asked whether a light half that must refuse the population on the book's
format is the right shape or a step that should not exist. **My answer: keep the ordering,
drop the separate record.**

*For it:* the guard-before-the-open is real. DA pins the book's digest against the receipt's
own field **and** BE's builder receipt *before* anything is unpickled — *"opening a book is
EXECUTING BYTES ANOTHER SEAT WROTE, and no check that runs afterwards can undo it."* Doing
provenance and the seal census before taking the heavy lock is the correct order, and a
light-only record is the right thing to leave behind when the heavy half genuinely cannot run.

*Against it, three things, each measured:*

1. **The refusal is not about the day.** `BOOK_IS_A_PICKLE_NOT_THIS_READER'S_JSON` is a
   constant of the pipeline — every book since `be_daybook_structure_v1` is a pickle. A
   predicate whose value was fixed when the code was written is not a measurement.
2. **rc 3 spends a declared verdict code on "this mode never attempts that."** The map's
   words for 3 are *"INCOMPLETE or PROVENANCE_INCOMPLETE — the seal HOLDS and a named half
   was not attempted"*, which covers it, but it collapses *not attempted because this mode
   cannot* with *not attempted because an input was missing*. The day the pickle is really
   absent, DA emits the same rc 3 with a different `incomplete_because` **string**. A code
   should separate them, not a sentence.
3. **The heavy half strictly dominates it.** The heavy record carries `seal_holds: true` and
   `provenance_all_matched: true` too. Sixty-six seconds after the light record was written
   it was superseded, and every later reader of history must resolve a pair to learn that its
   rc 3 was never a finding about the day.

**And the 09-04/09-05 evidence shows what the separate record actually costs.** For those
days the *last-written* pre-read is a light `first_of_family: true` INCOMPLETE
(`…192753Z`, `…192755Z`), emitted 41 minutes **after** the verified `.v2`. It is the light
record's existence as a separately-rooted artifact that puts two roots in the family — and
that is the input to the H1 disagreement in §1. DA's grouping handles it correctly; DE's
does not.

**Recommendation:** one act, one record, emitted at the end, carrying both phases' states —
which `the_halves_this_status_is_computed_from` already models exactly. If a light-only
record must be published because the heavy half could not run, give it a distinct code for
*this mode does not attempt the population* so it is never confused with a missing input.

---

## §3. THE RACE-READ BRANCH

### §3.1 BE 88 — `be_race_read_feed_pins_v2.json`

`26b0a67d93462b3f860f…` (**CHECKED**). Its `supersedes.sha256` `2cca55c64ffca8e78937…`
recomputes from v1 on disk — the pair verifies. The 09-06 entry carries the sealed feed
`0dfa62f3be90694d…`, 236,128,877 B, from
`…/20260906_be88/be_forward_day_SEALED_feed_20260906.jsonl`, and `produced_by` names
`live/pm_research/be_forward_day.py`, unit `be88fwd06.service`, with the producing receipt's
own digest — R-736's correction (the pin carries the forward-day's output, never the book's)
is implemented and **checkable rather than asserted**, which is what the block says it is
for. `the_second_reads_days` reports `1 of 4`, names the three unpinned days, and
`what_this_does_NOT_mean` states the whole-set precondition is a quarter met.

I confirmed the refusal is real rather than declared, by running the reader's own battery
(`--selftest` only, 82 checks, rc 0, 1.16 s): *"FALSIFIER 3 — AN ABSENT PIN REFUSES THE
WHOLE READ, naming every unpinned day, BEFORE any marker is written. 1 of 4 declared days
are pinned (['20260906']); the read refuses for the 3 that are not"* (**CHECKED**). The
battery also reports it *"held 3 OPENED marker(s) when this battery started and holds 3
now"* — it writes nothing into the ledger.

*Minor, routed to BE:* `all_five_present: false` sits at v2's **top level** while `per_day`
now holds six days. It is v1's field about the first read's five days and
`the_first_reads_days` explains it — but the name encodes a count the file no longer matches,
and `da_race_read_verify.pinned_feeds()` reads exactly that field (see §6.3).

### §3.2 BE 89 — the previous-read rule, and the peaks file

`PREVIOUS_READ_RULE` has **exactly one assignment**, at `be_race_reader.py:741`, quoted into
the rendered block at :571 (**CHECKED**). Its drives are in the battery and I saw them pass
in my own run: the two same-read corrections walking three hops to the nearest ancestor whose
READABLE *is* the consumed set; a consumed set matching no ancestor refused by name, listing
what it walked; REV 87 §1.3's `applies: False` with its reason. **I confirm MEM's routed
correction to R-744**: the docstring at :350 says the constant is *"above"* and it is at 741,
**below**. Behaviour unaffected.

**`live/pm_research/be_heavy_peaks.jsonl`** `be03c843b3933742…`: 21 lines = 1 header + **20
data rows**, `leaf_peak_status` **NOT_RECORDED 17 / MEASURED 3**, header `n_rows` 20.
**MEM's correction is right and R-744's "18" is the off-by-one** (**CHECKED**). The three
MEASURED rows are all 09-06 — book 6,217,269,248 (`leaf_peak_is_a_lower_bound: true`),
tape 6,334,619,648, forward-day 6,358,077,440, all against `cap_bytes` 8,589,934,592, and
`ratio_leaf_to_cap` is a stored field (0.7238 / 0.7374 / 0.7401), not prose.

**Routed, BE — the roll-up is hand-maintained, and it has already drifted. Re-measured at
`ad131f9`, after BE 91.** `grep -rn "be_heavy_peaks" live/ scripts/` returns **0 hits**: no
producer appends to this file and no consumer reads it, and its digest is unchanged by BE 91
(`be03c843b3933742…` at both tips). The machine records exist per unit
(`data/pm_5min/derived/be_heavy_run_record_<unit>.jsonl`) and BE 91 landed seven more of
them — **the roll-up knows about none of that**. The two sets disagree today:

```
units NAMED in the roll-up : 9        units with a LAUNCHER record : 45
record but NO roll-up row  : be72frag, be87fwd06  (+ probes and be_hr_falsify_* scratch units)
roll-up row but NO record  : be64book
```

`be72frag` and `be87fwd06` are the two REAL chain units that ran, refused, and left records
(the rest are probes and falsifier scratch, which legitimately do not belong in a per-day
roll-up); `be64book` is a row with no record at all (**CHECKED** at `ad131f9`). Q-BE-332 says
*"tomorrow's close reads the trend from the file rather than from a report someone has to
find"* — that is the promise the file cannot keep unattended. Derive it from the launcher
records plus the receipts, with a cell that refuses when a unit has a record and no row.
*(SEAT_PROTOCOL rule 15; R-605's "a practice that depends on noticing is not a control." BE 91
built exactly the right correspondence for unit OUTCOMES this round — this is the same move,
one artifact along.)*

*Minor:* NOT_RECORDED rows carry `leaf_peak_is_a_lower_bound: false` beside
`leaf_peak_bytes: null` — a `false` about a number that does not exist. `null` is the honest
value.

---

## §4. DA 118, R-750's TWO RULINGS, AND DA 115's PIN

**Neither ruling rests on a reading I can refute at the artifact.** Both are sound; I sharpen
one sentence.

**Ruling 1 — v8 stands, leg (d) ADDED, nothing removed. CONFIRMED.**
`p002_e2_a_declaration_v8.json` `929039c9fdc35d59…`; its `supersedes` names v7
`57c92c9e899eb691…` and **the digest recomputes from v7 on disk** — the pair verifies; eight
versions in the family. `admission_legs_v8` = a, b, c, d; `admission_legs_v7` = a, b, c —
and **legs a, b and c are byte-identical v7 → v8** (I compared the serialised blocks;
`{'leg_a_streams': True, 'leg_b_collector_liveness': True, 'leg_c_rule5_era_purity': True}`).
The premise of R-745 (4) was indeed stale: **v6 and v7 contain the string `gap_fraction`
zero times**, so there was no inherited fraction bar left to replace. DA was right not to
remove a leg, and the coordinator's in-band correction is right (**CHECKED**).

**Ruling 2 — the window opens on 2026-09-06. The measured premise CONFIRMED at the census.**
In `data/mm_hf/e1/p002_e2a_census8__20260906T055936Z.json`, all **eight** symbols' 20260906
row carries `gap_fraction: null` and `gap_profile: null`, while all eight 20260905 rows carry
a non-null `gap_fraction` (**CHECKED**). The declaration's `symbol_days` — the consumed set —
ends at 20260905 for every symbol, and `2026-09-06_IS_NOT_CONSUMED_and_this_is_MEASURED`
gives the reason with the file counts. The strict alternative (start 20260907, read no
earlier than 2026-09-21) is stated **in the artifact**, not hidden. `start_day` 20260906,
`earliest_read_date` 2026-09-20 marked a FLOOR, `min_complete_days` 14 with rule 11 named.

**The one sentence I would sharpen.** "Nothing on 09-06 was seen" is not quite what the
census says. The full 09-06 row is:

```
admissible: false | gap_fraction: null | gap_profile: null | decision_time_quote_age_ms: null
stream_file_counts: {bookTicker 7, depth20 7, trade 7} | streams_complete: all false
reasons_excluded: ["bookTicker_files=7", "trade_files=7", "depth20_files=7"]
```

09-06 **was** seen — on **leg (a)**'s quantity — and marked inadmissible on it, at 05:59Z
with 7 of 24 hour-files, i.e. mid-day. That spends nothing, because rule 11 binds choosing a
predicate on days you have seen it on, and leg (a) is byte-identical from v7 and older: it was
not chosen at v8. The reasoning holds; the wording should say *"09-06 was never seen on the
quantity leg (d) reads"* rather than *"nothing on 09-06 was seen."* **The residual trap worth
closing:** a later reader meeting this census finds `admissible: false` for 09-06 from a
7-of-24-file partial day. The row states its reason, so it is legible — but 09-06's real
admission record should cite this row and say it superseded it.

*Also confirmed:* the E2-A census's day span is 08-19..09-06, wider than the "08-20..09-05"
the rulings quote — and v8's `named_from_its_sources_not_from_a_date_range` says so itself
and carries the symbol-days instead of a range. The right repair, already made.

**DA 115's deploy pin — and REV 88 §3's rule tested live, twice, in one night.**

At `0b6968e` I read `da_midnight_deploy_pin_v2.json`: pair to v1 recomputes and matches;
`unit da-midnight-verify.service`, `commit 9abba110a9a9…`, `deployed_at 02:11:49Z`, **33
files**, all 33 digests recomputed **0 moved / 0 absent**, and `git diff --name-only
9abba110…..HEAD` = 30 changed files, **none pinned**.

Then DA 119 changed `da_midnight_verify.sh` (the 0600 fix at its line) — **a pinned file** —
and **re-pinned in the same round**, which is precisely the half of REV 88 §3 that exists for
this. Re-verified at `ad131f9`: `da_midnight_deploy_pin_v3.json` `be30c9c6f091ddb2…`, its
`supersedes` pair naming v2 `e454ff3d53236dc9…` **which recomputes from v2 on disk**;
`commit bbb618fc2e19`, `deployed_at 2026-09-07T05:55:23Z`, 33 files, **0 moved / 0 absent**,
and of the 20 files changed since that pin **none is pinned** (**CHECKED**).

*So the rule was exercised in both directions on its first night: nothing touched a pinned
file (v2's window), then something did and was re-pinned inside the round (v3's). It passes
on the measurement, not on anyone's memory — and the nightly unit at 00:06Z 09-08 will not
refuse on rc 7.*

---

## §5. THE COORDINATOR'S INSTRUMENTS

### §5.1 `scripts/land_register_row.sh` at `be0ef1d` — I ran the falsifier myself

Scratch repository only, `SCRATCH` pointed at my scratchpad:

```
B| LOCK HELD 05:59:45Z pid 730554   INSERTED 1 row(s) after line 5   POST-CONDITION OK …  PUSHED c570227
C| LOCK HELD 05:59:45Z pid 730555   INSERTED 1 row(s) after line 6   POST-CONDITION OK …  PUSHED 10d7c35
CONCURRENT OK: both landed, serialised, each commit +1 row / -0 lines, origin at 3 commits
KNOWN-BADS OK: dirty HELD, id mismatch REFUSED, duplicate REFUSED, foreign row REFUSED, dry run undone
FALSIFIER PASS                                                                     (rc 0)
```

Two processes in the same second; the second inserted after line 6 because the first's row
was already there — the property, not a timing accident, reproduced independently
(**CHECKED**). I also confirmed the lock is genuinely shared: `git rev-parse
--git-common-dir` resolves to `/home/yuqing/ctaNew/.git` from **all four** seat worktrees,
so there is one `p003_register.lock` for every seat.

**Does it close the MEM 245 / Q-DE-117 collision? In `--row` mode, yes — and no seat is in
`--row` mode yet.** In `--row` the fetch, the dirty check, the fast-forward, the insertion,
the commit and the push are all inside the lock, so no seat performs a whole-file
read-modify-write and no seat holds the register dirty. **In the legacy form the seat edits
the register *before* the script runs, so the edit is outside the lock**: the script's first
act under the lock is a diff of an already-dirty tree. The legacy form therefore still
produces the mutual block — now *detected* (`HELD REGISTER_DIRTY … wait, do not withdraw
it`, which is the right instruction, and is the one MEM did not have) but not *prevented*.
R-751 keeps every seat on the legacy form until this read. **That is the answer to the
question: the collision is closed for `--row` callers only, and there are none yet.**

**What else it leaves open, beyond the worktree-push case R-751 already names:**

1. **The foreign-row guard's strength is the caller's regex, and a loose one disarms it
   silently.** Driven in scratch: with `ids-regex` `Q-REV-89` a row file containing a foreign
   `Q-XX-99` is `REFUSED ROW_ID_MISMATCH`; with `Q-.*` **both rows insert** and the
   post-condition still prints, its `ids` field visibly garbage
   (`[Q-REV-89 | REV | mine Q-XX-99 | XX | ANOTHER SEAT'S ROW ]`) because `.*` swallowed the
   line (**CHECKED**). Detectable in the output; nothing refuses. One line: reject an
   `ids-regex` containing `.*`/`.+`, or require each extracted id to match `^Q-[A-Z]+-[0-9]+$`.
   The strong half — `REMOVED == 0`, no landed line changed — is regex-independent and
   untouched.
2. **The script runs `git rebase` in the shared tree** (:76), the verb SEAT_PROTOCOL rule 21
   forbids there. It is under the lock and aborts on failure, so it is defensible — but the
   exemption is written nowhere, and with another seat's files dirty (MEM's state files, all
   night, per R-557) the rebase fails and the script exits 10 `REBASE FAILED`, leaving a
   **stranded** commit whose message does not say it is stranded. Either declare the rebase
   as the one sanctioned use of the verb (under the lock, single-file pathspec, abort on
   failure) or refuse when `git status --short` shows any other dirty path; and say
   "stranded — report it" in the exit message.
3. **The copy-land clobber window BE disclosed (R-753, recorded not ruled) joins this item,
   and it is the same shape as (1) and (2): the guard runs on the wrong side of the act.**
   Rule 21's copy-land is *copy the worktree's bytes into the shared tree and commit by
   pathspec in one command*; BE reports DA landed twice between BE's fetch and its commit,
   with `git diff --name-only` over BE's six paths empty, so nothing was reverted this time.
   The check belongs **before the copy** — the copier verifies that none of the paths it is
   about to overwrite has moved in the shared tree since it read them, and refuses by name if
   any has. Today nothing checks; the evidence that nothing was clobbered is a diff BE ran
   afterwards, and an afterwards-diff is how you learn you were lucky. It is the same
   one-line closure the register lock got: put the guard inside the act.

### §5.2 `scripts/declaration_immutability.sh` at `af11ef9`

The fix is right and minimal: `NFAM=$(set +u; echo "${#HFAM[@]}")` — the `set +u` is scoped
to the command substitution, and it is measurement-only (the diff touches no judgement line;
I checked). Line 25 asserts the HISTORY line is **present**, so the defect cannot return
silently.

**Routed, coordinator — but the falsifier only fires when it is pointed at the right
directory.** I ran the pre-fix script from `af11ef9^` both ways:

```
PRE-FIX  live/mm_research/declarations --falsify   rc 1   "FALSIFIER FAIL: no HISTORY denominator line"
PRE-FIX  live/pm_research/declarations --falsify   rc 0   "FALSIFIER PASS"
POST-FIX live/mm_research/declarations --falsify   rc 0   "FALSIFIER PASS"
```

(**CHECKED**; exit codes measured without a pipe — my first reading took `tail`'s status and
said rc 0, which was my probe, not the script.) The control **does** fire on the defect — but
only on a directory with **zero** pre-base-edit families, while `--falsify`'s hardcoded
FLAG/PASS pair (`producer_exit_maps_v2` must FLAG) needs the pm_research directory, and only
the denominator sub-run uses `$DIR`. The two requirements cannot be met by one invocation, so
whether the control fires depends on an argument the caller chooses. One line: run the
denominator sub-check against a directory the falsifier constructs, or against both real
directories.

---

## §6. OPEN FINDINGS, AND MY OWN QUESTIONS FROM R-746

### §6.1 The mode-0600 class — CLOSED, and the two remaining writers were found at their lines

BE 90's fix is better than the obvious one: `declaration_chain` chmods the temp file to
`plain_create_mode()` = `0o666 & ~umask`, read by setting and restoring the umask, *"the
literal that goes wrong on the first box whose umask is not 0o022 — and it is wrong on THIS
one (umask 0o002, so a plain create is 0o664)"*. My census at `0b6968e`, wider than the
versioned scope BE measured:

```
*_v[0-9]*.json at 0600 under both declaration dirs + data/            : 0
*.json         at 0600 under both declaration dirs + data/            : 0
ANY file at 0600 under data/ and live/, any extension                 : 0
ANY file under data/ and live/ without group+other read (! -perm -044): 0
```

**CHECKED.** DA 119 and BE 91 then landed the two remaining *sources* — `mktemp` in
`da_midnight_verify.sh` (fixed at `:495`, chmod from the umask between the `mktemp` and the
`mv`, never a hard-coded mode) and `mkstemp` in `build_state_tape_v2.py` (`_land` at `:38`,
taking the mode from the shared implementation). **Fixing the writer at its line, rather than
re-moding the output, is the right half to close**, and it is what makes my zero durable
instead of momentary.

**The shared-module change REV must read regardless (R-753 says so):** BE made
`plain_create_mode` **public** in `declaration_chain` and kept `_plain_create_mode` as a real
alias binding (`:74`, `_plain_create_mode = plain_create_mode`) — one implementation under two
names, not a copy; two importers now call the public name (**CHECKED**). No judgement line
moved. This is the right shape: the alternative — each writer computing its own mode — is the
literal-per-site defect that produced the class in the first place.

### §6.2 Every importer runs `declaration_chain --falsify` — **NO. Four do not.**

This was my open question from R-746 and it has an answer. Thirteen modules import the shared
module; **nine run its `--falsify` as a subprocess cell** (I saw
`be_race_reader`'s do it in my own run: *"rc 0, '15 cells, 0 failures'"*). **Four do not, and
all four are DA's:**

```
NO FALSIFIER CELL   live/pm_research/da_deploy_pin.py        (imports declaration_chain at :100, :118)
NO FALSIFIER CELL   live/pm_research/da_nonhead_census.py    (:90, :1101)
NO FALSIFIER CELL   live/pm_research/da_race_read_verify.py  (:899, :952, :1550)
NO FALSIFIER CELL   live/mm_research/e2_a_declare.py         (:77)
```

(**CHECKED** — real imports, not prose mentions.) SEAT_PROTOCOL rule 20's clause (REV 84 §3.2
as amended by REV 85 §3, R-726) says *every* importer runs it, so that a regression in the
shared module fails every importer at once. Three of these four are the modules that landed
the deploy pin and the E2-A v8 emitter this round — the newest code is the code without the
cell. **Routed to DA**; one subprocess cell each.

### §6.3 The three non-head literals — all three now classified AND closed; my H2 hold is WITHDRAWN before filing

I classified these at `0b6968e`. **DA 119 and BE 91 closed all three at `bbb618f`..`46dcfd3`,
between my read and this filing.** I re-verified each at `ad131f9`. Recorded honestly: on the
one that mattered, **DA found the same door independently while classifying, and got there
first**; and DA's reason for KEEPING the literal is better than the one I was going to give.

**`be_race_reader.py:2420` — was never a non-head literal.** A **scratch fixture** filename
inside `tempfile.mkdtemp(prefix="be85_clause_")`, days `20990101/20990102`, building a
synthetic family that starts at v1. It never touches the real family. The census matched a
filename string, not an identity — CLAUDE.md rule 16's own warning about grep hits on
vocabulary. **Nothing to do**, and R-753's "one is fixture construction" agrees.

**`be_race_read_declaration_v3.py` — a reader of history, and BE gave it the pair.** My finding
at `0b6968e` was that it resolved by **path only**, with no digest — which is what R-729 does
not permit. At `ad131f9` it is `PINS_AS_V4_RECORDED = {path, sha256}` against the landed v1,
with the refusal reasoned in place: *"v1's bytes cannot legitimately move, so if they ever do,
this builder must STOP rather than recompute a landed declaration from bytes it never saw."*
**CLOSED** (**CHECKED**).

**`da_race_read_verify.py` — the constant is legitimate history; the CLI DEFAULT was the door,
and DA removed it.** What I measured at `0b6968e`:

```
pins v1 per_day days                       : 20260901 … 20260905
declaration v6 READABLE (the SECOND read)  : 20260906, 20260907, 20260908, 20260909
SECOND READ's days  ∩  pins v1             : EMPTY
```

`--pins` defaulted to the v1 constant, so verifying the second read's artifact without an
explicit `--pins` would have checked it against a pin set covering **none** of the days read
(**CHECKED**). At `ad131f9`: `ap.add_argument("--pins", default=None)` — *the artifact's own
pins, else refusal* — and the constant is now `FIRST_READ_PINS = {path, sha256}` with the
digest checked on every read. **CLOSED, and my HOLD (H2) is withdrawn before it was ever
filed.**

**And DA's classification is better than mine.** I would have said "resolve the head." DA's
comment says why that is the defect and not the fix, *measured*: *"the head is
`be_race_read_feed_pins_v2.json`, whose `per_day` carries SIX days … A reader that followed
it would report a five-day act as a six-day one — a closed read re-described with the days of
a read that has not happened."* That is the correct reading of R-729 and I adopt it. The
separable half was always the DEFAULT, not the constant, and DA cut exactly there.

### §6.3a The census predicate RULED at R-753 (2) for my read — I accept it, with one word changed

The ruling: *a non-head literal is ADMISSIBLE when the code HASHES the file it names against a
recorded digest before interpreting it, and REFUSED otherwise* (DA 94's separating property:
hashed → not a pin; interpreted → refused).

**Accept.** It is the right predicate because it is the one a census can actually evaluate,
and it separates the two cases correctly on all three literals above: two hash before
interpreting and are admissible, the fixture never touches the real family. A version-number
predicate could not have told any of them apart.

**The one word.** "against a recorded digest" does not say *recorded by whom*, and the three
literals do not agree on that. `FIRST_READ_PINS` cites a digest the first read's own act
records — R-729's "the pair the act recorded", exactly. `PINS_AS_V4_RECORDED` says of itself:
*"recorded_by: NOT v4 — v4 names the file without a digest; pinned at BE 91 against the landed
v1"*. That is a digest recorded **later, by a different seat**, and it is admissible only
because R-711 makes v1 immutable — a different guarantee, resting on a checker over git
history rather than on the act. BE disclosed it in the field itself, which is the right
handling and is why this is a wording note, not a finding.

**So: "…against a digest recorded IN THE CODE and asserted at the read; where the act itself
recorded no digest, the literal says so."** The added clause costs nothing (BE already writes
it) and keeps the census from quietly conflating a pin the act made with a pin someone
reconstructed afterwards — which is the distinction R-729 exists to hold.

### §6.4 The chain's branch point — verified as the runbook now states it

Tonight both branches ran and REV 88 §4's predicate describes them correctly. The **Gate-1
branch** completed: fragment → tape → book (`ac2ac952…`) → structure v4 → DE's day run
(`…051142Z`, the fourth of six) → DA's pre-read (both halves). The **race-read branch**
completed **one quarter**: `be_forward_day` → the sealed feed `0dfa62f3…` → pins v2, with
three days unpinned and the reader refusing by name for them. Saying which branch completed
is now possible from the artifacts alone, which is what §4 asked for.

---

## §7. WHAT I DID NOT DO, AND MY OWN ERRORS THIS ROUND

- **Scope in time.** The review executed against `0b6968e`. `bbb618f`..`ca78d8e` (DA 119)
  and `46dcfd3`..`1f5fc53` (BE 91) landed while it ran; I re-verified at `ad131f9` every
  finding they touch (§3.2, §4, §6.1, §6.3) and rewrote those sections rather than file a
  stale claim. **What those two rounds contain beyond my findings I did NOT review** — per
  R-753 it goes to REV 90. Specifically unreviewed: DA's `da_nonhead_census` default-argument
  fix, BE's CELL (d) resolved-roots change and its 126-check `be_daybook_build` green, and
  the seven new unit outcome records as records.
- I ran **no heavy unit**, took **no lock**, and wrote nothing under `data/`. Every drive
  was in `/tmp/…/scratchpad`, and the two scratch trees I built are deleted.
- I never invoked `be_race_reader --open`, never `read()` a real feed path, and every
  race-read fixture used scratch declarations and days `2099xxxx`.
- **Not established, and flagged as such:** DE 120's bisect to `403aac7` and its four-cell
  environment matrix; DE's 347-check count is my own run's, but `de_receipt_correction`'s 29
  I did not run; MEM 246's sweep; the journal *contents* of either DE capture (I read the
  copied blocks, I did not re-query the journal, whose window has moved).
- **Two probe errors, caught before they became findings** (my rate is unchanged and the cure
  is the same one): I called DE's `landing_record_for` with the *derived* directory where it
  wants the *data root*, and read `n_matches 0` as a defect — the assertion
  `assert o["n_matches"] > 0` is what turned it back into a probe error. And I read
  `rc=0` off a pipeline whose last stage was `tail`, briefly making the immutability
  falsifier look like it exits 0 on failure; measured without the pipe it exits **1**.
  Both are the same cure: **assert the probe found something before reading what it found.**

---

## §8. SUMMARY OF ROUTING

| # | to | finding | kind |
|---|---|---|---|
| **H1** | DE (with DA) | the two read gates disagree on 09-03/04/05's landing records; DE's resolver has not taken up DA 105's per-receipt grouping | **HOLD on the Gate-1 read** |
| 1 | DE | the capture record lacks `producer_module` / `mapped_by` / `resolved_kind` its own declaration requires — close before GO #8's capture is written | routed |
| 2 | DE | `import_closure.modules` collapses by basename; `n_modules != len(modules)` unasserted; I drove R22 GREEN on a non-empty initialiser | routed |
| 3 | DE | `DESIGN_VERSION_IN_FORCE = 23` is a literal tracking a moving thing in the **emitter**; the fix exists only in `de_receipt_correction` | routed |
| 4 | DE | `the_two_readings_agree` is named for a comparison other than the one that refuses | routed |
| 5 | DE | DE 120's in-cell falsifier drives re-typed literals and carries one conjunct that cannot fail | routed |
| 6 | BE | `be_heavy_peaks.jsonl` is hand-maintained and has drifted (`be72frag`, `be87fwd06`, `be64book`); 0 code references at either tip | routed |
| 7 | BE | pins v2's top-level `all_five_present` names a count the file no longer matches | minor |
| 8 | DA | four importers of `declaration_chain` ship no `--falsify` cell — `da_deploy_pin`, `da_nonhead_census`, `da_race_read_verify`, `e2_a_declare` | routed |
| 9 | DA | the light pre-read should be a phase, not a record; give "this mode does not attempt the population" its own code | routed (§2.4) |
| 10 | DA | v8's "nothing on 09-06 was seen" → "never seen on leg (d)'s quantity"; the partial-day `admissible: false` row wants a superseding note | wording |
| 11 | coordinator | `land_register_row.sh`: a loose `ids-regex` disarms the foreign-row guard; `git rebase` in the shared tree is undeclared and its stranded exit unnamed; the copy-land clobber check belongs before the copy | routed |
| 12 | coordinator | `declaration_immutability.sh --falsify` fires only when pointed at a zero-pre-base-edit directory | routed |
| 13 | coordinator | R-753 (2)'s census predicate ACCEPTED, with one clause added: *a digest recorded in the code and asserted at the read; where the act recorded none, the literal says so* | ruling read |

**Closed between my read and this filing, re-verified at `ad131f9`, no action:** the
mode-0600 class at both remaining writers (DA 119, BE 91); all three non-head literals
(DA 119, BE 91) — including the `--pins` default door, which **DA found independently while
classifying and removed before I could file the hold I had written for it**; and the deploy
pin re-pinned to v3 in the same round as the file it pins changed.

**GO #8 MAY PROCEED.** Nothing in this filing gates it.
