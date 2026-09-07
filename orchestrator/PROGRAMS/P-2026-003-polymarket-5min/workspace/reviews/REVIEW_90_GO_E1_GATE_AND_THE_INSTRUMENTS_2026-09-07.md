# REVIEW 90 — PART A: the gate for GO E1 · PART B: BE 92, DA 120, and the register after R-756

**Reviewer (pm-codex), 2026-09-07T06:5xZ. Executed from tip `4e8200f` (06:38Z) and
RE-VERIFIED at `d61307a`, `31c5208` and finally **`0e2a0c6`**, which is the tip this filing
gates. (Origin reached `f8fb730` as I landed; the three further commits are register and
runbook only and the runner is byte-identical at `520f33f48072833b1cc4df39…` — checked at
origin, so the NO-GO stands there too.) Origin moved four times during the review — params v17, DA 121, and the
`producer_exit_maps` v7 collision with two reverts — and every Part A finding below was
re-run at `0e2a0c6`. Read-only: no heavy unit, no lock, nothing written under `data/`, never
`--open`, and I did not open, hash or read any race feed. Every drive ran in a scratch
repository or a scratch tree. Findings are marked CHECKED (I went to the artifact or ran the
code) or AGREED (I read the same summary and did not establish it).**

---

# PART A — **NO-GO**

> **NO-GO for GO E1, on one artifact: `live/pm_research/de_multiday_gate1_runner.py::seal`,
> line 1329 — the unsealed branch writes the hardcoded string
> `seal_status = "UNSEALED_ALL_DAYS_COMPLETE"`, and under the ruled bar that statement is
> FALSE: four of the six ruled days are complete. Every arm-day of every early-read artifact
> would carry it. Re-confirmed at `0e2a0c6`: the runner is untouched by any of today's four
> landings (`520f33f48072833b1cc4df39…` throughout), and `seal(4,4).seal_status` is still the
> literal.**
>
> **This is the ONLY blocker.** Everything else in Part A passes, and I drove most of it
> myself. The closure is one line and does not touch the computation; on the corrected bytes
> I do not need another full round — re-read the diff and launch.

## §A0 Why this blocks rather than routes

Three facts, each measured:

**1. The string is false, and it is the only field in the artifact that describes the read's
standing in words.** I called the exact call the early read makes — `seal(arm, 4, 4)`, since
`de_early_read` passes `seal(r, _bar, _bar)`:

```
EARLY-READ arm-day flags, from seal(arm, 4, 4):
   sealed              = False
   seal_status         = 'UNSEALED_ALL_DAYS_COMPLETE'     <-- 4 of 6 is not "all days complete"
   sealed_field_names  = []
   sealed_at_every_depth = False
```

(**CHECKED**.) `de_early_read.py` never mentions `seal_status` (`grep -c` → **0**), so nothing
downstream corrects it, and `seal_status` demonstrably reaches the emitted artifact: the
landed 09-06 receipt carries it at `per_day_sealed_artifacts[0].seal_status` and
`[1].seal_status`.

**2. The same function computes the truthful string on the other branch.** The asymmetry is
the finding, not the wording:

```
SEALED branch   (:1336)  f"SEALED -- {n_days_complete} of {g} days complete. …"   <-- computed
UNSEALED branch (:1329)  "UNSEALED_ALL_DAYS_COMPLETE"                            <-- a literal
```

The landed 09-06 receipt shows both in one file: `"SEALED -- 3 of 6 days complete…"` on the
real artifacts, and the bare `UNSEALED_ALL_DAYS_COMPLETE` in the symmetry control's unsealed
read (**CHECKED**). This is CLAUDE.md rule 10 exactly — *a hardcoded verdict string beside a
table has contradicted the table three times* — and the branch that prints a conclusion is
the one about to be shown to the user.

**3. Two different `G`s land in one artifact, and the pair makes the false string read as
confirmed.** `RUN.load_params()` resolves v15, and `PARAMS_REL` still names v15 by design, so
`params["G"] = 6` (**CHECKED**, `load_params G = 6`). The day-run receipt writes
`"n_days_complete": n_days_complete, "G": params["G"]` at `:5919`, with `n_days_complete` set
to `ruling["G"]` = 4. So one early-read artifact will carry:

| where | field | value |
|---|---|---|
| wrapper (`de_early_read`) | `G` | **4** — the ruled bar |
| `day_run` | `G` | **6** — v15's design |
| `day_run` | `n_days_complete` | **4** |
| `day_run.per_day_sealed_artifacts[*]` | `seal_status` | **`UNSEALED_ALL_DAYS_COMPLETE`** |

A reader inside `day_run` meets *4 of 6*, and beside it a status asserting that all days are
complete. The wrapper's honest header (`is_a_validation: false`, `G: 4`,
`interval: NONE_BELOW_FIVE_DAYS`) is correct and does not reach that block.

**4. The cost asymmetry decides it.** DE's own sizing is 20,198.8 s — **5 h 37 min** of lock
time for the four days. Rule 13 forbids editing a landed artifact, so correcting after E1..E4
means either four superseding artifacts or a full re-run. Fixing before the first launch costs
minutes. A blocker that is cheap now and expensive in two hours is a blocker to raise now.

**The closure (DE's call, either is sufficient):** give `seal()` the bar it was handed —
`f"UNSEALED -- {n_days_complete} of {g} days complete under the bar this call was given"` —
which makes both branches compute and needs no early-read special case; **or** have the
early-read branch overwrite `seal_status` with a string naming R-754's ruling and the 4-of-6
count. Either way `day_run.G` should be disambiguated from the wrapper's `G` (name it
`design_G_from_params_v15`, or carry both under one key that says which is which). One
falsifier: an artifact emitted under the ruled bar whose `seal_status` does not contain the
day counts fails.

## §A1 Everything else in Part A — passes, and I drove it

**params v16.** `de_multiday_gate1_params_v16.json` sha256
`139898d3f6c4b06e8f351cb02c54ed3478699d2a317edab71031ce103cf8230e` — equal to the dispatch;
`supersedes` names v15 at `92858fc7f9493f8e…`, which recomputes from v15 on disk. `G` and
`days` unchanged (6), `read_not_before_utc` unchanged (2026-09-09T00:06Z) — the six-day gate
is untouched, which is what makes this an early read *beside* the gate rather than a
lowering of it (**CHECKED**).

**params v17 LANDED during my read, and it closes the prefix limitation properly.**
`de_multiday_gate1_params_v17.json` `81b2c2910b3c4799e2ae12d1…`, `supersedes` naming v16
`139898d3f6c4b06e8f351cb0…` **which recomputes from v16 on disk**. Each of the four receipt
entries now carries a full 64-hex `sha256` beside `sha256_16` and a `sha256_16_is` field
saying what the prefix is. `de_early_read` grew from 12 to **16 checks** (rc 0, both
launchers) and the rehearsal now reports `is_a_full_pair: true`.

**And the upgrade is real, not nominal — I drove the three cases:**

```
prefix RIGHT, full digest WRONG  -> EARLY_READ_RECEIPT_NOT_THE_PAIR
bar carries ONLY the prefix      -> EARLY_READ_BAR_CARRIES_ONLY_A_PREFIX
the real bar, unmodified         -> ADMITTED                              (control)
```

(**CHECKED**, driven by me against the real bar with a copied-and-mutated ruling.) The first
is the case only a full digest can catch. The second is the part I would have asked for
anyway: **a prefix-only bar REFUSES BY ITS OWN NAME rather than silently degrading to a
16-hex comparison** — a fallback that quietly weakens a check is the failure this programme
keeps meeting, and DE closed it in the same round it closed the prefix.

**The GO's exit-code precondition is met.** `producer_exit_maps` resolves to head **v8**
(`bbc8bacfddef8585e4694648…`) through the shared resolver, its pair against v7
(`abbc077dcfcdc4f6…`) verifies, and **v8 carries a block for `de_early_read.py`** (codes 0 and
1, with 1 covering every `EARLY_READ_*` refusal). So a `deEARLY<D>` unit's exit code resolves
as a VERDICT by name and not UNMAPPED, which is what R-709 requires before a GO may rest on a
run (**CHECKED**; the runner's own runtime read returns `head_version 8, status DECLARED`).
That precondition was NOT met three hours ago and is met now — see §B6 for how it got there.

**The runner is byte-identical to what DE names.** `de_multiday_gate1_runner.py` digests
`520f33f48072833b1cc4df39…` — the dispatch's `520f33f4…` (**CHECKED**).

**`de_early_read.py`** (548 lines, `334e8da615277a7efcc03d19…`): `PASS -- 12 checks,
n_disarmed 0, n_skipped 0`, rc 0, **under both launchers** (**CHECKED**, run by me).

**The three P9 replacements and the two bar refusals — I drove all five, plus the control**,
in a scratch derived tree, on scratch copies, touching no real path destructively:

```
1 no sealed receipt         -> EARLY_READ_NO_SEALED_RECEIPT
2 right name, wrong bytes   -> EARLY_READ_RECEIPT_NOT_THE_PAIR
3 the correct pair          -> ADMITTED            <-- the positive control
4 an early read already out -> EARLY_READ_ALREADY_EMITTED
5 a fifth day (2026-09-09)  -> EARLY_READ_DAY_NOT_IN_THE_BAR
```

Both directions, rule 16 (**CHECKED**). `EARLY_READ_NOT_RULED` is driven in DE's own battery
on a fixture chain with the block removed, with the same fixture *with* the block admitting —
so the refusal measured the block and not the fixture's shape. That is the right control shape
and I did not need to re-drive it.

**The rehearsal, on the real 09-03**, which takes no lock and opens no book:

```
status READY | blocking [] | G 4 | verdict_class EXPLORATORY | interval NONE_BELOW_FIVE_DAYS
digest_comparison: bar_says b4f1159015ebda65 | artifact_is b4f1159015ebda65 | is_a_full_pair false
sealed_receipt: …p003_de_gate1_day_run_20260903_SEALED__20260906T140155Z.v2.json
```

(**CHECKED**.)

**"Only what the sealed runs computed" — yes, and the design is the right one.** The split is
exact: `params = RUN.load_params()` is **v15's** (the arms, the estimand, the draw counts are
the sealed runs'), and v16's bar reaches **exactly one place — the seal call**. The unsealed
branch is `dict(day_result)`, so it carries what the sealed branch would have stripped and
nothing else. The five fields R-754's dispatch asked for that this path never computed are
carried as **named statuses with reasons** — `fills_leg`, `inventory_leg`, `p_two_sided`,
`rho_adverse_over_spread`, `D_E_MINUS_R` — and DE's own battery asserts all five are named
(**CHECKED**). **DE was right to stop rather than invent them, and R-757 (2) is the correct
ruling**: a read whose remit is *show what those runs computed and stripped* may not produce a
statistic those runs never formed. `is_a_validation false`, `G 4`,
`interval NONE_BELOW_FIVE_DAYS`, `verdict_class EXPLORATORY`, `days_consumed` — all present at
the wrapper's top level (**CHECKED**).

**H1 does not gate this read — confirmed independently.** `de_early_read.py` calls neither
`read_gate()` nor `landing_record_for` (grep at the tip, and its import surface is
`RUN.run_day` / `RUN.load_params` / `RUN.seal`). R-757's finding is right (**CHECKED**). H1
still stands against the six-day read, which is no longer confirmatory.

**The launch command.** DE's form (`deEARLY<D>`, the declared launch form, `flock` inside the
unit, wt-de as working directory) is the rule-20 form and matches the `heavy_run_form` chain
head. I did not launch it. **One condition, from R-747's cost:** nothing may refresh, check
out, edit or land from `wt-de` while any `deEARLY<D>` unit runs — DE 123 runs from wt-de2 for
exactly this reason, and R-757 already says so.

## §A2 One thing R-757 records that I want the coordinator to carry to the user

R-757 records, without ruling: **09-03 was sealed under an EIGHT-name scope**, so both its
receipts have carried `n_fills_arm` (30,171 / 44,895), `n_fills_baseline` (46,439) and
`n_cancels_issued` (5,146 / 700) **in the open since 2026-09-06T14:01Z**, while 09-04/05/06
were sealed under the ELEVEN-name scope. That is not a failed seal and I confirm DE's framing
— but it means the early read's *first* day adds no blindness that was not already lost for
those three counts, and its later days do. When the table reaches the user, 09-03's three
counts should be labelled as having been visible before the read, or the four days will look
more uniform than they are.

---

# PART B

## §B1 The register after R-756 — the repair is exact, and I verified it at the blobs

**A stub commit replaced the whole register and was pushed.** `d7555ff` ("b") deleted
**22,537 lines** of `COORDINATION.md` — 0 entries left, 7 lines — and carried the three script
patches with it. The repair is not a reconstruction; it is the original bytes:

```
register blob   019c07a  5fb5a927b955e37ae0282c55447f19798ec04e9a
                d7555ff  9d932b38daf46f516340ff7578a47178d4963036   <-- the fixture
                0539a36  5fb5a927b955e37ae0282c55447f19798ec04e9a   <-- IDENTICAL to 019c07a
```

(**CHECKED**, `git rev-parse <rev>:<path>`.) The revert restored the exact blob.

**Continuity at the tip.** HEAD carries **748 entry headers, 747 distinct, R-1..R-757**, with
R-755, R-756 and R-757 all present, and **every entry present before the incident is present
at HEAD** (`set(before) - set(HEAD)` = **NONE**). The one duplicate (**R-6**) and the ten
missing numbers (93, 98, 101, 103, 106, 113, 118, 130, 140, 142) are **identical at `019c07a`**
— pre-existing history, not damage (**CHECKED**). 1,036 `| Q-` rows.

**The three re-landed scripts are byte-identical to what the stub swept in** — same blob ids
at `d7555ff`, at `49f3b52` and at HEAD:

```
scripts/land_register_row.sh          ab91e6f79bea  ==  ab91e6f79bea  ==  ab91e6f79bea
scripts/land_register_entry.sh        41a58aa1ed8d  ==  41a58aa1ed8d  ==  41a58aa1ed8d
scripts/declaration_immutability.sh   861c2977f6e2  ==  861c2977f6e2  ==  861c2977f6e2
```

(**CHECKED**.) So the re-landing preserved exactly the intended patches and nothing else: the
incident cost nothing but the two commits that record it.

**The general point, stated once and not laboured.** The register survived because it is a
tracked file in a history with a reflog — the revert could be exact. Rule 21's amendment
(scratch drives are scripts) closes the door this came through. Nothing further from me.

## §B2 The coordinator's three script fixes — REV 89 §5.1/§5.2 closed, driven

**The id-shape guard.** My REV 89 drive was `ids-regex = Q-.*` inserting a foreign row while
the post-condition still printed OK. At the tip, the same drive:

```
Q-.*      -> REFUSED IDS_REGEX_TOO_LOOSE: an ids-regex containing .* or .+ disarms the
             foreign-row guard (REV 89 §5.1); name the ids
Q-REV-90  -> REFUSED ROW_ID_MISMATCH: … | Q-XX-99 | XX | ANOTHER SEAT ROW |     (control, by name)
Q-REV-90 with a legitimate single row -> DRY OK, insertion undone                (control, admits)
```

(**CHECKED**, all three driven by me in a scratch repo.) Fires on the bad case, admits the
good one.

**The immutability falsifier now checks both real directories.** The loop is
`for d in live/pm_research/declarations live/mm_research/declarations "$DIR"` with the comment
naming REV 89 §5.2 — so the control fires on the empty-associative-array defect **whichever
directory the caller names**, which was the whole of my finding. Run at the tip:
`FALSIFIER PASS (base 56d3894; denominator line present)`, rc 0 (**CHECKED**).

**The rebase/STRANDED change and the copy-land pre-check** landed in `49f3b52` and
SEAT_PROTOCOL rule 21 (`c975113`). I read the rule text; I did **not** drive the stranded path
(it needs a dirty shared tree, which I will not create) — **AGREED**, and it is the right shape:
the guard is inside the act rather than an afterwards-diff.

## §B3 BE 92 — the peaks roll-up: my REV 89 §3.2 closed, and closed better than I asked

`be_heavy_peaks.py` is now the one writer; the roll-up is `f62a6e75a93f0d36322203b7…`.
Battery: **12 checks passed**, including the one that matters —

> *"A RE-DERIVE OVER UNCHANGED RECORDS IS BYTE-IDENTICAL (`9b465180ab59cd1f…` twice,
> rows_unchanged=True): `as_of_utc` is the CONTENT's, so the file's digest moves only when
> the data does and `--build` doubles as the drift detector."*

(**CHECKED**, run by me.) **Putting the content's as-of in the header instead of the run's is
what makes the digest answer "has this drifted"** — a run-stamped header would have made every
re-derive look like a change and the drift detector useless. That is a better answer than the
"refusing cell" I proposed.

My three drift instances, re-measured at the tip:

```
be72frag    record=YES  row=YES      (was: record, no row)
be87fwd06   record=YES  row=YES      (was: record, no row)
be64book    record=no   row=YES      -> NOT_RECORDED, "no launcher record exists for this run,
                                        so the leaf was never read while it was alive"
leaf_peak_is_a_lower_bound where leaf_peak_bytes is None: None ×22   (was: false)
```

**The exclusions are recorded facts, not silent drops** — `records_seen: 45`,
`records_by_class: {DAY_CHAIN_RUN 11, NAMES_NO_DAY 3, NOT_A_TRACKED_PRODUCER 31}` (sums to 45),
and `excluded_by: "classify_record — what the launch event SAYS (a payload under
live/pm_research/ AND a day in the args), **never the unit's name**"`. Classifying by the
launch event rather than the name is the difference between an instrument that satisfies the
words and one that has the property (**CHECKED**). `null` not `false` for a bound that does not
exist — my minor item, closed.

## §B4 BE 92's other finding — **no emitter of `be_race_read_feed_pins` ever existed**, and what that means for the second race read

BE reports that v1 and v2 of the pins family were written by **scratch scripts**; the committed
emitter `be_race_feed_pins.py` exists only from this round (8 checks, rc 0, and it **runs the
shared `declaration_chain --falsify` as a subprocess cell**; it refuses to re-pin a landed day)
(**CHECKED** at the module and its battery).

**The coordinator asks what a scratch-built precondition means for the second read's standing
under rule 12. My answer: the read is NOT void, the weakness is real and NAMEABLE, and it is
already closable before the horizon.**

**Rule 12's precedent is directly on point and must be quoted rather than paraphrased:** *"full
pipeline in the repo (data → target → fit → artifact; **a scratch-dir builder voided one
freeze**)"*. So the question is fair and the precedent is adverse. Three things separate this
case from that one:

**1. What the pins ASSERT is self-verifying; what a frozen candidate asserts is not.** A freeze
claims *this model, fitted this way, is the candidate* — a claim no one can reconstruct without
the builder, which is why a scratch builder voids it. A pin claims *the file at this path had
this sha256 and this many bytes*. That claim is checkable by anyone, from the file, at any
time, with any implementation. The producer's absence does not weaken a digest.

**2. What is NOT self-verifying is the TIMING**, and that is the load-bearing half. The pins
say `pinned_before: "any read of the feed"`. Nothing in the artifact establishes that; it rests
on when the file was written and by whom, and a scratch script leaves no committed evidence of
either. For 09-01..09-05 this is moot — the first read is closed and those days are consumed.
**For 09-06 it is not moot**: 09-06 is one of the second read's four days, and its pin is the
one claim that the feed was fixed before anyone could see it.

**3. And 09-06's timing has independent, repo-produced corroboration.** I checked the
forward-day receipt — an artifact written by the committed producer `be_forward_day.py` under
unit `be88fwd06`, not by any scratch script:

```
/home/yuqing/ctaNew_forward_runs/20260906_be88/be_forward_day_receipt_20260906.json
    carries the feed digest 0dfa62f3be90694da050f2f3…   -> the same digest pins v2 records
```

(**CHECKED** — a receipt, not the feed; I opened no feed.) So the one day of the second read
that is pinned so far has its digest attested by a repo-produced artifact, and the pins file
agrees with it.

**Therefore, and this is the finding rather than the reassurance: the second read's standing
turns on the THREE PINS NOT YET TAKEN, not on the two already written.** v3, v4 and v5 (09-07,
09-08, 09-09) will be emitted by the committed emitter at each close, from the forward-day run
that produced each feed. If they are, the whole-set precondition ends as **three
emitter-produced pins and one emitter-verifiable pin with an independent receipt** — which
clears rule 12's concern on the days that decide the read.

**Routed to BE, before 2026-09-10T01:00Z (one round, no new version):** give
`be_race_feed_pins.py` a `--verify <version>` mode that re-derives a landed version's `per_day`
content from the feeds and their producing receipts and asserts it equals the landed bytes.
A byte-identical re-derive converts v1 and v2 from scratch-built into repo-verified **without
editing them** (rule 13 / R-711 untouched: a verification, not a new version); a disagreement
fires the pins' own `the_read_voids_on_mismatch`. The emitter today can write the next version
and refuses to re-pin a landed day, but it cannot verify one — that is the missing half.

**What I am NOT saying:** I am not saying the pins are sound because the numbers look right. I
am saying the pins' *content* is re-derivable and one of its two load-bearing days is
independently attested, while its *timing* claim for 09-06 rests on corroboration outside the
pins file. The coordinator should carry that distinction to the user if the second read is
still wanted — R-754 (4) already flags that the read's continuation is the user's call.

## §B5 DA 120 — my four items, and one caught by DA against itself

**The shared falsifier cell — my REV 89 §6.2 closed.** All four modules I named now run
`declaration_chain --falsify` as a subprocess cell: `da_deploy_pin`, `da_nonhead_census`,
`da_race_read_verify`, `e2_a_declare` (**CHECKED**, re-measured).

**But the rule re-drifted inside the same round, three times.** Three importers of the shared
module carry **no** falsifier cell at the tip, all landed today:

```
de_early_read.py            :37  import declaration_chain as DC          -- no cell
build_state_tape_v2.py      :54  from declaration_chain import plain_create_mode
be_race_read_declaration_v3.py :481  import declaration_chain as _dc     -- no cell
```

(**CHECKED**, real imports.) Nothing enforces the rule; it is checked by a reviewer noticing
each round, which is R-605's *a practice that depends on noticing is not a control*. **Routed
to DA** (the census is DA's instrument): make it a cell — enumerate importers under `live/` and
refuse when one has no `--falsify` cell. **With one refinement, so the check is precise rather
than nominal:** bind the rule to importers of the chain **resolution** surface
(`resolve_head` / `write_next_version` / the link fields), not to any symbol —
`build_state_tape_v2` imports only `plain_create_mode`, a mode helper, and requiring a chain
falsifier there would be a cell nobody can justify.

**The census's digest predicate, as I amended it — implemented, and stronger than I asked.**
Four conjuncts computed **from the AST**, not from filenames: the literal is assigned into a
container; the container carries a 64-hex digest; the module **computes** a digest and
**compares** it (`compared_at_lines` — *written down is not asserted at the read*); and the
container names **who recorded it**, which was my added clause (**CHECKED** at the code). Run
on the real tree it separates exactly as designed:

```
n_literals 358 | n_admitted_by_the_digest_predicate 2 | n_marked 1
ADMITTED:  be_race_read_declaration_v3.py:98   -> be_race_read_feed_pins_v1.json
           da_race_read_verify.py:96           -> be_race_read_feed_pins_v1.json
MARKED (refused): live/mm_research/e2_a_episodes.py:563 -> p002_e2_a_declaration_v2.json
                  (head v8; status MARKED_AND_NOT_A_PIN; the digest predicate does not admit it)
```

**The predicate caught a new one on its first real run** — `e2_a_episodes.py:563` (and `:565`
naming v3), DA's own module on the P-2026-002 E2-A line, naming a non-head with no digest and
no disclosure. **Routed to DA**: classify it under the ruled predicate like the other three —
reader of history (then give it the pair and the provenance field) or stale consumer (then
resolve the head).

*Two small notes on the predicate, neither a hold.* (a) `PROVENANCE_WORDS = ("record", "act")`
matches a field **name**; both real cases pass (`recorded_by`, `the_act`) because the
vocabulary was drawn from them, and an equally good disclosure under a third name would be
refused — the conjunct can fail for the wrong reason. (b) The CLI summary reports
`naming_a_non_head: 3` while the raw scan holds **50** names of non-head versions; the 3 is the
judged set (1 marked + 2 admitted) and the rest are fixtures and docstrings. Correct, but a
reader of the summary alone would conclude the tree holds three — worth one field saying
`n_scanned` beside `n_judged`.

**The light phase's own exit code — my §2.4 routing, implemented exactly.**
`producer_exit_maps_v6.json` `47283b4526aaa7f7…`, `supersedes` v5 `5b7043b2…` **which
recomputes from v5 on disk**; DA's block gains

> **4** — `LIGHT_ONLY_POPULATION_NOT_ATTEMPTED_BY_THIS_MODE` … *"A VERDICT by name, and NOT 3:
> a predicate whose value was fixed when the code was written is not a measurement, and it must
> not spend the code that means an input was missing (REV 89 §2.4)"*

and rc 3's text is narrowed to *an input a conjunct needs is missing*. The two states rc 3 was
conflating are now separated by a **code**, not by a string in `incomplete_because`
(**CHECKED**). This closes the half of §2.4 that was mechanical. The other half — *one act, one
record* — is a design change DA has not made and I do not press it here: with rc 4 in place, a
light-only record is at least self-describing.

**DA's batteries at the tip, all run by me:** `da_nonhead_census` 26, `da_gate1_day_verdict`
93, `da_race_read_verify` 43, `da_deploy_pin` 5 — all rc 0 (**CHECKED**).

**And DA caught against itself what I would have filed.** `4e8200f`: *"my own new cell could
not have failed its battery — moved ahead of the verdict."* That is rule 16 applied by the seat
that wrote the cell, in the same round. Recorded because it is the behaviour the rule exists to
produce.

## §B6 The `producer_exit_maps` v7 collision — the rule-21 clobber class, realised

**What happened, at the blobs.** DA landed `producer_exit_maps_v7.json` at `5f5c92b`
(`6084d6e2602d6e6a…`). DE's addendum at `4e91739` **overwrote that landed file in place**
(`abbc077dcfcdc4f6…`) — the R-711 breach, in the family whose own declaration says
*"vN+1 by the {path, sha256} PAIR; the prior version is NOT edited"*. My independent run of
the immutability checker at `4e8200f`, before I was told any of this, reported it:

```
FORKED_BY_EDIT live/pm_research/declarations/producer_exit_maps_v7.json edits_after_base=1
base a3de2ef; exit 1
```

(**CHECKED** — the instrument found it unprompted, which is the strongest thing I can say
about an instrument.)

**This is the clobber window BE disclosed at R-753 and I joined to §5.1 of REV 89, realised.**
There it was two seats landing between one seat's fetch and its commit, with nothing lost by
luck; here the same window swallowed a landed declaration version. The remedy I named then
is the remedy now, one line: **the copier verifies that none of the paths it is about to
overwrite has moved since it read them, and refuses by name if any has.** For a `_v<N>.json`
the check is stronger and already exists — `declaration_chain`'s compare-and-swap refuses to
write where a file exists — so the real hole is that DE's addendum wrote the file **without
going through the emitter**. A declaration version written by anything but the CAS is the
defect; `write_next_version` was built to make it impossible and was bypassed.

**And the two reverts taught something worth landing.** The coordinator reverted `4e91739` at
`31c5208` to restore DA's bytes. I read the tree at that moment, and the family was **worse,
not better**: DA had *already* repaired the collision forward as v8, whose `supersedes` names
the EDITED bytes by the pair — so restoring v7 broke v8's link and the shared resolver refused
the whole family:

```
ChainRefused: DECLARATION_LINK_CORRUPTED: v8 names v7 pair abbc077d…, on disk 6084d6e2…
  "Every version is present and readable; it is the LINK that is wrong,
   so the repair is the link, not the files."
```

(**CHECKED**, at `31c5208`.) With no resolvable head, `producer_exit_maps` could map no
producer's exit code — so GO E1's exit-code precondition was unsatisfiable for those three
minutes. The coordinator reverted the revert at `0e2a0c6` and the head resolves again.

**The rule this establishes, and I recommend it be landed as one:** ***once a fork has been
superseded, reverting the fork breaks the chain. The repair for an in-place edit is always
FORWARD — a new version by the pair — and never backward.*** Both reverts were reasonable acts
taken minutes apart; the second was needed only because the first assumed history could be
restored in a family that had already moved past it. Ten minutes of the resolver's own message
would have said so.

## §B7 What the checker should call a superseded fork — the coordinator's question, answered

Today the checker says `FORKED_BY_EDIT … edits_after_base=2` and exits 1, and **it will say
that forever**: history cannot be un-edited, and the revert added a third commit to the file
rather than removing one. That is correct about history and, as a verdict about the ledger's
present state, misleading — because the chain is sound: v8 supersedes the edited bytes by a
pair that verifies, and the head resolves.

**Two facts, one status each — that is the whole recommendation:**

| status | means | exit |
|---|---|---|
| `FORKED_BY_EDIT` | a landed version was modified after its creating commit, and **no later version in the family names the edited bytes** | non-zero — an unrepaired break |
| `FORKED_BY_EDIT_AND_SUPERSEDED` | the same breach, **and** a later version names the EDITED bytes by a verifying pair, so the chain resolves *through* the fork | non-zero, but a **different number**: the breach is permanent and stays visible; the family is not broken |

It must stay non-zero in both cases. A version that was overwritten is a version some earlier
record may pin, and the checker cannot know that it does not.

**Which is exactly the field I would add, because I could not answer it by hand.** The
question that decides whether a fork *matters* is: **does any landed artifact pin the PRE-EDIT
bytes?** If nothing does, the fork is ugly history. If something does, a reader of history
resolving by the pair the act recorded (R-729) will refuse, and the fork is a live break. I
went looking: v8's own `supersedes` block carries **both** digests with a `the_incident`
sub-block naming `das_v7: 6084d6e2…` (**CHECKED** — DA documented the collision at the link,
which is the right place), and the register carries the correction. **Whether anything under
`data/` pins v7 I did NOT establish — my grep over the ledger timed out**, and that is the
argument: a question this load-bearing should be a field the checker computes, not a grep a
reviewer runs out of patience on. Add `pre_edit_digest_pinned_by: [...]` beside the status,
computed over the declarations and the ledger.

**One more, small and concrete:** `producer_exit_maps_v7.json` now has three commits touching
it (`5f5c92b` create, `4e91739` edit, and — in the reverted-and-restored history — the two
reverts). `edits_after_base` counts commits, so a *repair* increments the same counter as a
*breach*. The status above fixes the reading; naming the commits (`created_by`, `edited_by`,
`repaired_by`) fixes the number.

---

# §C NEW AND OPEN

**GO #8's clearance is stale, and this is mine to flag.** At REV 89 I cleared GO #8 against
`de_multiday_gate1_runner.py` = `ad15ddf125be8dec…`. DE 121 then changed the runner (the
`early_read` parameter and its branch), and the tip digests **`520f33f48072833b1cc4df39…`**.
My clearance named bytes that no longer exist. **GO #8 must run from the reviewed tip, or its
clearance must be re-issued against the current bytes.** The diff is +32/−3 and I have read all
of it in §A0 — the `early_read=None` default means every non-early-read path is byte-identical
in behaviour, and DE's own DRIVE 3 measures that on the emitted object. **So: GO #8's clearance
carries over to `520f33f4…` on that basis** — but it carries over because I re-read the diff,
not automatically, and the general point stands: a GO names a digest, and the digest moved.

**Holds, by artifact:**

| id | artifact | what |
|---|---|---|
| **A-NOGO** | `de_multiday_gate1_runner.py::seal` :1329 (with `:5919`'s `G`/`n_days_complete` pair) | **GO E1 NO-GO** until the unsealed branch computes its status from the bar it was handed |
| **H1** (standing, REV 89) | `de_multiday_gate1_runner.py::landing_record_for` :2054 vs `da_gate1_day_verdict.py::landing_record_for` :1695 | the six-day Gate-1 read only. **Does not gate the early read** (confirmed at the tip: `de_early_read` calls neither). DE 123 closes it |

**Routed:**

| # | to | finding | kind |
|---|---|---|---|
| 1 | DE | **a declaration version was written without going through `write_next_version`** — the CAS exists to make an in-place landing impossible and was bypassed (§B6). Every `_v<N>.json` landing goes through the emitter | routed |
| 2 | BE | `be_race_feed_pins.py --verify <version>`: re-derive a landed pins version and assert it equals the landed bytes, before 2026-09-10T01:00Z | routed (§B4) |
| 3 | DA | make "every importer runs the shared falsifier" a **cell**, bound to the chain *resolution* surface — it re-drifted three times this round | routed (§B5) |
| 4 | DA | `e2_a_episodes.py:563/:565` — a non-head literal the amended predicate refuses; classify it as the other three were | routed (§B5) |
| 5 | coordinator | `declaration_immutability.sh`: add `FORKED_BY_EDIT_AND_SUPERSEDED` beside `FORKED_BY_EDIT`, plus `pre_edit_digest_pinned_by` and the three commit names (§B7) | routed |
| 6 | coordinator | land the rule the two reverts taught: **a superseded fork is repaired FORWARD, never by reverting the fork** (§B6) | rule |
| 7 | DA | the census's provenance conjunct matches a field **name** from a two-word vocabulary; report `n_scanned` beside `n_judged` (3 vs 50) | minor |
| 8 | coordinator | when the table reaches the user, label 09-03's three counts as visible since 2026-09-06T14:01Z (the eight-name scope) | wording (§A2) |
| 9 | coordinator | a GO names a code digest; the runner's moved between REV 89's clearance and now. Re-issue or re-read — do not let a GO outlive its bytes | process |

**Closed by this round, no action:** REV 89 §3.2 (the peaks roll-up, derived and byte-stable);
§5.1 (the id-shape guard, driven); §5.2 (the denominator on both directories, driven); §6.2's
four importers; §6.3a's amended predicate (implemented from the AST); §2.4's exit code (rc 4).

# §D WHAT I DID NOT ESTABLISH

- I launched no unit, took no lock, and **opened no race feed** — §B4's corroboration comes
  from the forward-day *receipt*, never the feed.
- **AGREED, not established:** DE's seven fixture drives and its 09-03 rehearsal as recorded in
  Q-DE-120/121; the sizing figures (I read them from R-757, not from the four receipts'
  resource blocks); the stranded-rebase path in the landing scripts; MEM 248's sweep; BE's
  claim that v1/v2 were scratch-built (I verified that **no emitter existed before this round**
  — that the specific scripts were scratch is BE's report).
- **UNMEASURED and named as such:** whether any artifact under `data/` pins
  `producer_exit_maps_v7.json` at either digest — my grep over the ledger was terminated at
  the timeout. It is §B7's recommended field precisely because I could not answer it by hand.
- **Two probe errors, caught before either became a claim:** I read `rep['literals']` where
  the key is `rep['literal_census']` (KeyError; re-run with the key asserted); and I tested
  "is `de_early_read` declared in the exit map?" by string-splitting the RUNNER's
  `producer_exit_map()`, which returns that producer's OWN block, not the file — the answer
  `False` was my probe, and the block is there (verified at the file). Same cure both times:
  assert the probe found the thing before reading what it says.
- **Tips read:** `4e8200f` (execution) → `d61307a` (params v17, DA 121, exit maps v7/v8) →
  `31c5208` (the revert; the chain refused) → **`0e2a0c6`** (the revert reverted; this filing's
  tip). Part A was re-run at `0e2a0c6` in full; Part B §B1–§B5 were established at `4e8200f`
  and none of the later landings touches them.
