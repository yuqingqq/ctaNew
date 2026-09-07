# REVIEW 92 — PART A: H1 closed both ways, R-764's materiality ruling, and the checker's superseded-fork statuses

**Reviewer (pm-codex), 2026-09-07T07:5xZ. Read at `42f4253` in `~/ctaNew-wt-rev`. Read-only:
no heavy unit, no lock, nothing written under `data/`, never `--open`, no feed opened. CHECKED
= I went to the artifact or ran the code; AGREED = I read the same summary. PART B (DE 124,
the USER's second ruling) had not landed when Part A was decided — `params_v18` is absent at
`7889d81` — and is reported separately.**

---

# PART A

## §A1 H1 is CLOSED, both ways — I re-ran both gates on the real ledger

REV 89 §1's measurement, re-run at the tip:

```
DE's landing_record_for, per day        DE read_gate conjunct 3
  2026-09-03  PRESENT             n=1     failing: ['2026-09-07', '2026-09-08']
  2026-09-04  PRESENT             n=1
  2026-09-05  PRESENT             n=1   DA's same conjunct
  2026-09-06  PRESENT_CHAIN_HEAD  n=2     09-03 MATCH 09-04 MATCH 09-05 MATCH 09-06 MATCH
```

(**CHECKED**, both run by me.) At REV 89 DE read `AMBIGUOUS` for 09-03/04/05 while DA read
`MATCH`; **the two seats now agree on all four sealed days**, and DE's gate fails only on the
two days with no receipt. That is the closure.

**And both halves landed, which is what I asked for rather than either alone.** In the
runner's battery (**359 checks**, 0 disarmed, 0 skipped, rc 0 — run by me):

- DA 105's grouping adopted — two unlinked records for one day resolve to the group reading
  the day's CURRENT head (`PRESENT, n_matches 1`);
- **the OLD grouping driven as a known-bad** on the same two files, reading `AMBIGUOUS` — *"the
  answer that refused 09-03, 09-04 and 09-05 in this seat's gate"*. A delta from the landed
  behaviour, not agreement with the new code;
- a supersession link outranks the grouping;
- **the cross-check: "BOTH SEATS ON THE REAL LEDGER: the two independent resolvers name the
  SAME landing-record receipt digest for every ruled day — disagreeing: none."**

The cross-check is the half that matters most and it is the half I could not have got from a
one-sided fix: R-235 keeps two implementations, and the thing that was missing was that
**nobody ran both and compared**. Now the battery does, on the real ledger, every run.

## §A2 The "six-day regression" was a wrong root, and the refusal now names it

`read_gate` refuses a non-ledger root by name. Driven by me, both directions:

```
root = /home/yuqing/ctaNew        -> REFUSED READ_GATE_ROOT_IS_NOT_THE_LEDGER:
                                     "…has no `pm_5min/derived`. `root` is the DATA root…"
root = /home/yuqing/ctaNew/data   -> conjunct 3 failing = ['2026-09-07','2026-09-08']
```

(**CHECKED**.) So the coordinator's notice was a probe that passed the REPO root where the
DATA root belongs — and I will say plainly that this is the same failure I have now made three
rounds running in my own probes: **a wrong argument that returns a plausible-looking negative
instead of refusing.** The fix here is the one I keep prescribing for myself and had not
thought to prescribe for the code: make the wrong call refuse *by name*. That is worth more
than the bug it closed.

## §A3 REV 89's five DE items — all five closed, two of them better than I asked

| item | closure | verdict |
|---|---|---|
| capture record lacks `producer_module` / `mapped_by` / `resolved_kind` | `de_early_read` now builds all three, with `mapped_by` carrying the exit-map head's `{path, sha256}` and **`UNMAPPED` handled explicitly** | **CHECKED** |
| `import_closure.modules` collapses by basename | the cell reads `modules_by_path`, and a new conjunct asserts `n_modules == len(modules_by_path)` and `basenames_collide is False` — **the predicate I named as missing**; the known-bad is *"the case REV built"*, two initialisers at different paths, driven FALSE | **CHECKED** |
| `DESIGN_VERSION_IN_FORCE` a literal tracking a moving thing in the emitter | the cell asserts the scope map tops out at the constant, **with a planted name at v26 driven as the known-bad** — *"a guard nobody has watched fire is a comment"* | **CHECKED** |
| `the_two_readings_agree` named for the wrong comparison | renamed | **CHECKED** |
| DE 120's in-cell falsifier drove re-typed literals with a conjunct that could not fail | rebuilt: driven **through the same function the cell uses**, over the LIVE map with one entry perturbed, and the vacuous case asserted as vacuous rather than counted as evidence — *"so the predicate and its falsifier cannot drift apart"* | **CHECKED** |

`de_early_read` battery 16 → **18**; the design declaration and receipt-correction batteries
pass.

## §A4 R-764's materiality ruling — **I cannot refute it, and I established it by a stronger route than the leaf count**

The ruling: the early read loads params v15 for all four days while 09-03/09-04 were sealed
under v14, and the computation is nevertheless the sealed runs' because v14→v15 moved only the
design pointer and v21→v23 touched no estimand, bar or pin.

**Claim 1, params v14 → v15 — verified UNFILTERED.** I diffed leaf by leaf with nothing
excluded: **13 changed leaves, of which exactly 2 are substantive** —
`design_declaration.path` (v21 → v23) and `design_declaration.the_design_pins_THIS_file` — and
the other 11 are `supersedes.*`, `protocol` and `version` bookkeeping. **No `days`, no `G`, no
`multiplicity_m`, no `read_not_before`, no threshold, no seed, no draw count moved**
(**CHECKED**; the coordinator's exclusion hides nothing, because I enumerated all 13).

**Claim 2, design v21 → v23 — and here I did not count, I checked what the number is ABOUT.**
Counting leaves can only ever support the ruling circumstantially. The decisive question is
whether the design declaration is an *input to the computation* at all. It is not:

> The runner reads exactly ONE value out of the design declaration — `parameters.sha256`, the
> digest the design uses to pin the params — and uses it in the **precondition** `P3_design`
> (`de_multiday_gate1_runner.py` ~:3045). It reads no estimand, no threshold, no seed, no arm
> definition, no draw count.

(**CHECKED** at the code.) **So no diff between design v21 and v23 can change what `run_day`
computes, whatever its size.** That holds independently of the leaf count and is why I say the
ruling is not refutable at the declarations: its conclusion is true for a stronger reason than
the one given for it.

**On the number itself — it reproduces exactly, once the convention is named.** I first
measured **70** changed leaves where R-764 records 42, and rather than report a discrepancy I
looked for the convention:

```
lists EXPANDED per element : total 70 | excluding supersedes/version/protocol/as_of = 61
lists AS ONE LEAF          : total 48 | excluding supersedes/version/protocol/as_of = 42
```

(**CHECKED**.) **42 is right**; the coordinator diffed with lists as single leaves. The only
gap is that the convention is not stated beside the number, and DA's reader embeds "42 leaves"
as `what_was_measured`. One clause — *"lists compared whole"* — makes it reproducible by the
next reader, who will otherwise measure 70 and think something moved.

**What I checked about DA's carrying of the ruling.** `da_early_read_verify` records
`measured_by: "the coordinator at R-764, **not by this reader**"` — the reader carries the
measurement rather than claiming it — and `what_it_does_NOT_do: "it does not soften a
refusal"`. That is the right shape: a ruling that travels with its provenance and does not
become the instrument's own finding.

## §A5 DA 124 and BE 95

**`--print-under-ruling R-764` rides beside the refusal, never instead.** `RULED_CODES` names
the two refusals it may print under; the battery's cell states *"the ruling rides beside the
refusal and never replaces it"*, and a wrong ruling pair still refuses. `da_early_read_verify`
battery **20 checks**, rc 0 (**CHECKED**, run by me).

**`producer_exit_maps_v9`** `482527d4810b0f74…`, `supersedes` naming v8 `bbc8bacfddef8585…`
**which recomputes from v8 on disk**; the shared resolver returns v9 as the head (**CHECKED**).

**BE 95 — the fixture name is pinned to its bytes, and the census's refusal cleared.**
`be_race_reader.py:2420` no longer appears in the census's refused set. That closes the item
where **my own REV 90 call of "nothing to do" was wrong** and REV 91 corrected it.

**DA 122's cell for `de_early_read.py` did NOT land.** The census — the instrument, not my grep
— reads `n_in_the_surface 13, n_missing_the_cell 1, missing: ['live/pm_research/de_early_read.py']`,
verdict `REFUSED_A_RESOLVER_SHIPS_NO_FALSIFIER_CELL` (**CHECKED**). BE's two are closed; DE's
is not. **My REV 91 §C1 stays open**, unchanged in kind: a detection-coverage gap, not a
correctness one, and not a gate on anything.

**Two literals still refused, and they are the same one twice:**
`de_multiday_gate1_runner.py:66` (`PARAMS_REL`) and, new, `da_early_read_verify.py:862` — both
naming `de_multiday_gate1_params_v15.json` while the head is v17, neither with a digest at the
literal. REV 91 §C2's substance was closed *after the fact* by DA 123's
`check_computation_params` (comparing the artifact's loaded-params digest against the sealed
receipt's), which is the right place for it given the runner's bytes are frozen under GO E1.
**If DE 124 moves `PARAMS_REL` to v18 the runner's literal names the head again and this
refusal clears by itself** — which is a reason to expect the census to go quiet for the wrong
reason, and worth watching rather than celebrating.

## §A6 The checker at `ff8b0a9` — **§B7 is met**, and its new field answers the v7 question **against** what was reported

**I ran the falsifier myself: `FALSIFIER PASS (base 56d3894; denominator line present;
superseded-fork status reads on the real v7)`, rc 0.** All three things §B7 asked for are
there, plus one I did not ask for:

```
FORKED_BY_EDIT_AND_SUPERSEDED  producer_exit_maps_v7.json  edits_after_base=3
  created_by=5f5c92b  edited_by=4e91739  repaired_by=[31c5208 0e2a0c6]
  superseded_by=[producer_exit_maps_v8.json]
  pre_edit_digest=6084d6e2602d6e6a
  pre_edit_digest_pinned_by=[producer_exit_maps_v8.json]
SUPERSEDED FORKS: 1 (rc 2 = every fork is superseded by a verifying pair; rc 1 = an UNREPAIRED fork exists)
base a3de2ef; exit 2
```

(**CHECKED**, full-directory run.) The extra is the **exit code**: 2 for a superseded fork
against 1 for an unrepaired one. I asked for two statuses; the coordinator separated the exit
codes too, which is what makes the distinction usable by a caller and not just by a reader.
*(I briefly thought the summary undercounted — the falsifier's output also shows v2 — then
checked: `--falsify` drives v2 under base `56d3894`, while under the default base v2's edits
are pre-base and land in the HISTORY denominator, 19 of 66. `SUPERSEDED FORKS: 1` is correct.)*

### **"Nothing under the ledger pins v7's pre-edit bytes" is NOT established — the checker's own field says otherwise**

`pre_edit_digest_pinned_by=[producer_exit_maps_v8.json]`. **One file does pin it.** I then went
to the artifact to find out *where*, which is the question the field cannot yet answer:

```
v8 carries 6084d6e2… at   .supersedes.the_incident.das_v7      <- a deliberate INCIDENT RECORD
v8's functional link      .supersedes.sha256 = abbc077d…       <- the POST-edit bytes; this is what resolves
```

(**CHECKED**.) So the correct statement — and I recommend R-762's line be superseded by it — is:
**no RESOLUTION pin names v7's pre-edit bytes; exactly one incident record does, deliberately,
and no resolver refuses.** The original claim is wrong as written and right in substance, and
the difference is the kind a reader of history will meet.

**The instrument's gap, which is the same class the field was built to close.**
`pre_edit_digest_pinned_by` is a `grep -lF` over file *contents*, so it cannot distinguish a
functional pin from a mention — and that distinction is the whole purpose of the field. The v2
case shows it at scale: six versions "pin" v2's pre-edit digest, and every one of them carries
it under `chain_repair.restored.*.from_commit` or `chain_repair.v2_history.*` — repair
documentation, not a link (**CHECKED** at v9). **One extra token closes it: print the JSON path
where the digest was found**, e.g.
`pre_edit_digest_pinned_by=[producer_exit_maps_v8.json:supersedes.the_incident.das_v7]`. Then
"a reader resolving by the pair would refuse" is answerable from the output instead of by a
reviewer opening each file.

**And the scope is real but unprinted.** The scan is `find "$DIR" "$LEDGER" -maxdepth 1 -type f
-name '*.json' -size -5M`. The size filter has a good reason (gigabyte tapes as `.json` cost
the falsifier its cap at R-764) and `-maxdepth 1` is deliberate, but neither appears in the
output, so `pre_edit_digest_pinned_by=[…]` reads as *these and no others* when it means *these,
among top-level JSON files under 5 MB in two directories*. One suffix — `(scope: ≤5M,
maxdepth 1, 2 dirs)` — and the field states its own reach.

---

# §B HOLDS AND ROUTING (Part A)

**No holds.** H1 is closed and nothing in Part A gates a launch.

| # | to | finding | kind |
|---|---|---|---|
| 1 | DE | `de_early_read.py` still ships no shared-falsifier cell — the census names it, `n_missing_the_cell 1` (REV 91 §C1, unchanged) | routed |
| 2 | coordinator | R-762's *"nothing pins v7's pre-edit bytes"* is refuted by the new field; supersede it with *"no RESOLUTION pin does; one incident record does"* | correction |
| 3 | coordinator | `pre_edit_digest_pinned_by` should print the JSON path of each hit — a mention and a link are the distinction the field exists to make | routed |
| 4 | coordinator | the field's scope (≤5 MB, `-maxdepth 1`, two directories) is not printed beside its answer | minor |
| 5 | coordinator / DA | state the diff convention beside "42 leaves" (lists compared whole); it reproduces exactly, and without the convention the next reader measures 70 | minor |

**Closed this round:** H1 both ways with the cross-check on the real ledger; REV 89's five DE
items; REV 91 §C2 item 1 (the fixture, where my REV 90 call was wrong); §B7's three asks plus
the exit-code split; `producer_exit_maps_v9` by the pair.

# §C WHAT I DID NOT ESTABLISH

- **AGREED, not established:** DE's own battery counts beyond the ones I ran; the 09-03 GO E1
  run's outputs (I read no early-read artifact); MEM 251's sweep; BE 95's internals beyond the
  census's verdict on it.
- **Not established:** whether any file *above* 5 MB or below the top level of the ledger
  carries v7's pre-edit digest — the checker's scan does not reach there and neither did I.
- **My own probe discipline this round:** I asserted the key before every nested read, and it
  paid twice — once when I nearly reported the superseded-fork counter as an undercount (it is
  correct; the two runs use different bases), and once when I nearly reported the "42 vs 70"
  leaf count as a discrepancy (it reproduces exactly under lists-as-one-leaf). Both would have
  been findings against the artifact that the artifact did not deserve.

---

# §D PART B — NOT YET DECIDABLE, and one measurement the coordinator should have before GO E2 or GO #8

**DE 124 had not landed when Part A was decided.** At `6a72b71` (07:59:26Z) there is no
`de_multiday_gate1_params_v18.json`, and no DE 124 code commit. **I gate nothing in Part B
here**: GO E2 and tonight's GO #8 remain ungated by me, and I will read `params v18`, `design
v26`, the retired seal path and the per-day decision ledger when they land.

**But the tree is mid-landing, and I measured something both GOs depend on.** The design
declarations have already appeared in the shared `data/pm_5min/derived/` — **v26 and v27 both
exist on disk** — while `PARAMS_REL` still names v15. The runner's preflight `P3_design`
(`de_multiday_gate1_runner.py:3071`) requires the design chain HEAD to pin the params file the
runner will load:

```
design chain head          : p003_de_multiday_gate1_design_v27.json  (v27)
head pins params sha256    : cfc2b06fe2c9e792fb7500a3…
PARAMS_REL                 : de_multiday_gate1_params_v15.json  92858fc7f9493f8e…
P3_design would hold       : False
```

(**CHECKED**, 07:56Z.) **This is not a defect** — it is the expected transient of a landing in
progress, and the digest the head pins is presumably v18's, landed ahead of it because
declarations under `derived/` are visible the moment they are written rather than at their
commit. I record it because of what it implies for the two launches:

> **Before GO E2 or GO #8 is issued, `P3_design` must be verified to HOLD at the launching
> tree** — i.e. the design chain head must pin the digest of whatever `PARAMS_REL` then names.
> It is a one-line check and it is cheap; a day run that meets a failing `P3_design` refuses at
> preflight, and R-747 is the standing reminder of what a refusal late in a heavy run costs.

That check is part of what I will do for Part B; I am naming it now so it is not discovered at
a launch.
