# REVIEW — the runner re-drive after DE 77: all six items verified, three of my open findings CLOSED, and **the lock DE hardened has an unhardened twin that I opened in one line**

**Filed** 2026-09-06T06:12Z (clock read before composing) · reviewer seat (pm-codex)
· tip `af37cb7` · no code fixed · **no data touched** — my only write was a fixture
receipt into the scratchpad, and the emitted receipt's own instrument says
`data_paths_opened: []` · nothing sealed opened.

**ROUTING — CHECKED.** Every drive below is mine, on my own inputs, not DE's
fixtures. Where I reproduced a DE-computed quantity I re-derived it from git or
from the declared fields rather than re-running DE's check.

## VERDICT

**All four batteries reproduce at their asserted counts** under my runs, and they
are light by measurement, so rule 20 needs no lock for them:

```
de_multiday_design_declaration  PASS -- 65 checks   [0.07 s  26.6 MB]
de_multiday_gate1_runner        PASS -- 71 checks   [0.26 s  23.5 MB]
de_supersession_diff            PASS -- 26 checks   [0.04 s  14.9 MB]
de_data_root                    PASS -- 16 checks   [0.03 s  15.0 MB]
```

**DE was right to withdraw its own approval, and the batch earns it back on the
fixture path.** Every one of the six items drives correctly, including the two
hardest (the in-process draws seam and the seal symmetry), and **three of my own
open findings from R-572(C) are CLOSED at the artifacts.**

**Two new findings, both in the admission layer, both one-liners, and both of a
class DE fixed elsewhere in this same batch:**

1. **`may_run_day()` reads the CALLER's `params['days']`, not the committed file —
   and I opened it.** With one line of caller-side rewriting it returns
   `may_run: True, sealed: True, authority: R-572(B)(2)` for **2026-08-29**, a day
   R-555 excluded. `resolve_draws()` was hardened against exactly this attack this
   round; its twin was not. **§2.3.**
2. **`the_committed_day_set_is_empty: True` is a hardcoded literal, asserted by
   nothing, and FALSE** — the committed parameter file holds six ruled days. It
   sits in the landed fixture receipt v7 beside prose that is stale in the same
   way. Rule 10, and the exact twin of DE's own finding (b) this round. **§7.1.**

**A third, minor: one declared seed field is INERT** under the plausible mutation,
and the battery's chosen mutation hides it. **§4.2.**

**Approval, stated precisely** (§8): the **fixture path is APPROVED**; the 09-03
smoke cannot be approved to *run* because `--day` does not exist — verified at the
argument parser, not from the report.

---

# 1. THE DRAWS ARE GENERATED IN PROCESS

## 1.1 The re-point — I re-derived DE's computation rather than reading it

`params v2` re-points `be_module.sha256` to `2b164df2…` after BE 47. DE's
justification is a computed changed-set, so I recomputed it independently with my
own AST pass over both revisions:

```
top-level definitions at ab75b41^ : 20    at ab75b41 : 20
CHANGED   : ['load', 'main', 'run']        (DE claims ['load','main','run'])
unchanged : 17                              (DE claims 17)
the nine declared draw-path functions: present at both revs, 0 CHANGED
                                       (DE claims draw_path_is_byte_identical: true)
current file == ab75b41's bytes -> the pin is still current
```

**Exact agreement on all four quantities.** And all three pins verify at their
artifacts: `be_module` `2b164df2…`, `design v8` `139052df…`, `params v1`
`b06125e9…` — the last confirming `v1_untouched`.

**I then attacked the gap DE declares rather than the part it proves.**
`what_this_does_NOT_establish` says module-level constants and imports are not
covered. So I enumerated them:

```
module-level names CHANGED at ab75b41 : ['DERIVED','OUT_DERIVED','_RES', 3 imports]
module-level names READ by the draw path (AST, free variables per function):
   load        -> ARMS, CACHE, COIN, _RES     <- _RES is one of the changed three
   params_for  -> BUDGET, COIN, LAT           <- none changed
   arm_stream  -> COIN                        <- unchanged
   draw_null   -> MIN_DRAWS, N_DRAWS, SEED    <- none changed
```

**The declared gap is real and it is empty except at the one place DE names.**
`_RES` reaches `load`, and `load` is the function DE flags. It is closed at the
mechanism, which I checked: `be_cancel_axis_null.load()` is
`p = Path(path) if path is not None else CACHE` — **the explicit path governs**, the
runner always passes one, and `source_sha256` is taken from the buffer that was
unpickled. So `_RES`/`CACHE` cannot select a different book on the runner's path.
**The residual DE states is the only one there is, and it is shut.**

## 1.2 The seam — driven, both refusals and four more

```
FIXTURE draws claimed on a RULED day (2026-09-03)        REFUSES
SUPPLIED draws on a REAL/ruled day (2026-09-03)          REFUSES
REAL run on a day NOT in the ruled set (FIXTURE-1)       REFUSES
FIXTURE mode with NO supplied draws                      REFUSES
POSITIVE: fixture on a non-ruled day WITH draws          ADMITS -> SUPPLIED_FIXTURE_ONLY
```

**And my own attack on the lock itself, which is the point of the design:**

```
caller rewrites params['days'] = [FIXTURE-1..3], then:
  fixture on 2026-09-03  -> REFUSES ("which IS in the ruled day set [09-03..09-08]")
  real   on FIXTURE-1    -> REFUSES ("which is NOT in the ruled day set")
ruled_day_set() still reads 09-03..09-08 from the COMMITTED file
```

**The caller cannot open either door by rewriting its own dict.** DE's finding (e)
— *"a lock whose input the caller supplies is not a lock"* — holds under attack in
both directions. **This is the property §2.3 shows is missing one function away.**

## 1.3 `import_be_cascade` — three known-bads of mine, and the interior one is the point

```
POSITIVE: the real import        ADMITS  sha 2b164df2…, digest_is_of_the_file_the_import_loaded
right NAME, IDENTICAL BYTES,
        wrong PATH (a shadow)    REFUSES "the import loaded /tmp/…/be_cancel_axis_null.py
                                          but the declaration cites …/live/pm_research/…"
right path, digest -> 0*64       REFUSES
__file__ that does not exist     REFUSES
```

**The second case is the one that matters and it is mine, not DE's fixture:** a
module of the right name at the wrong path with *byte-identical content*. A
digest-only check admits it; the path check is therefore **not** redundant with the
digest check, and DE's "two distinct properties, either alone is a hole" is
demonstrated rather than asserted.

## 1.4 `verify_draw_provenance` — five refusals, two admits

```
POSITIVE: in-process block from THIS pid                          ADMITS
GENERATED_IN_PROCESS from a FOREIGN pid                           REFUSES (field: pid)
GENERATED_IN_PROCESS with generated_in_process=False              REFUSES
no provenance block at all                                        REFUSES
seed off by one                                                   REFUSES (recomputed)
MINE INTERIOR: SUPPLIED_FIXTURE_ONLY from a foreign pid           ADMITS  <- correct
```

The last is the interior case: the pid lock must apply to an in-process **claim**
and must not apply to a fixture's supplied draws, whose door is shut by the ruled-day
lock instead. It behaves correctly in both.

## 1.5 One thing DE did NOT do, and was right not to

`generate_draws_in_process` deliberately does **not** use BE's `reproduction_gate`
— its `BASELINE_CANCELS` and filed arm numbers are pinned to the consumed 08-24
hour and would refuse every day book. **The day-level reproduction gate is recorded
as OPEN in the provenance block (`day_level_reproduction_gate_is_open`) rather than
approximated.** That is the correct handling: an inherited gate that cannot fire on
the new population is rule 16's control that cannot fail, and naming it beats
running it. **It is a real open item for BE's per-day book receipt**, and it should
not be lost when `--day` lands.

---

# 2. THE TWO CLOCKS — the split is SOUND, and one gate is not

## 2.1 Is the split sound? **Yes.** Driven, six ways

```
may_run_day
  09-03, closed, qualifying, TODAY (05:xx on 09-06)   ADMITS  may_run/sealed=True
  09-06, OPEN day                                     REFUSES (not a CLOSED calendar day)
  09-04, closed but FAILS quality                     REFUSES
  2026-09-01, outside the ruled set                   REFUSES
  params WITHOUT day_runs_allowed_for_...             REFUSES ("will not infer it")
may_read_aggregate
  6 days, exactly at 2026-09-09T00:06:00Z             ADMITS
  6 days, ONE MINUTE early                            REFUSES (the date)
  5 days, a MONTH after the date                      REFUSES (the count)
  6 days, today                                       REFUSES (the date)
```

**The split does what R-572(B)(2) ruled.** The 09-03 smoke is admissible today,
sealed; the unseal needs **both** the date and all six days, and neither alone
suffices in either direction. The reasoning in the docstring is the right one: a
per-day run publishes resources, counts and statuses and no economic field, so it
cannot inform a later choice.

## 2.2 Does anything still gate a per-day run on the whole-set date? **No.**

Full surface search over `live/`, output reported rather than truncated: the only
consumers of the not-before date are `may_read_aggregate()` and the declarations
that describe it. `run_not_before_utc` survives only as `supersedes.superseded_field`
and in the v1 file kept as provenance. **No run path reads it.** The one whole-set
field doing two jobs is genuinely split.

## 2.3 **FINDING — `may_run_day` takes the ruled set from the caller, and I opened it**

`de_multiday_gate1_runner.py:718` — `if day not in params.get("days", [])`. That is
**the caller's dict**, which is precisely the input `resolve_draws()` refuses to
trust twelve lines earlier in the same file.

**Driven:**

```
caller's params['days'] = ["2026-08-29"] + the ruled six
may_run_day("2026-08-29", day_row={closed, qualifying})
   -> ADMITS: {"day":"2026-08-29","may_run":true,"sealed":true,
               "authority":"R-572(B)(2)", ...}
ruled_day_set() from the committed file  ->  ['2026-09-03' … '2026-09-08']
```

**2026-08-29 is the day R-555 excluded by name** (`previously_opened_for:
development_read`), and `may_run_day` issues it a decision-shaped admission citing
the coordinator's ruling as its authority. **`ruled_day_set()` exists, in this file,
and this function does not call it.**

**What limits the damage, stated fairly:** `resolve_draws()` in real mode would
still refuse 08-29, so a full `--day` run cannot *complete* on it — the second lock
catches it. But (a) which gate fires first is not yet determined by any code,
because `--day` is unbuilt; (b) rule 14 says no worker boolean encodes an
entitlement, and `may_run: true` for a ruled-out day is exactly that; and (c) DE's
own words for the twin defect apply verbatim — *"a lock whose input the caller
supplies is the caller's word again."*

**One line:** `if day not in ruled_day_set()`. The same for `may_read_aggregate`'s
`params["G"]`, which is likewise caller-supplied (`load_params` validates it, a
hand-built dict does not).

---

# 3. THE SYMMETRIC SEAL LAYOUT — the consumer falsifier, both states, plus four interior cases

`SEAL_LAYOUT_KEYS = ('sealed','seal_status','sealed_at_every_depth','sealed_field_names')`,
conditional key `economic`. As `seal()` actually emits them:

```
SEALED                    sealed=True  depth=True  names=[7 economic fields]  economic_present=False
UNSEALED                  sealed=False depth=False names=[]                   economic_present=True
UNSEALED (refused arm-day) sealed=False depth=False names=[]                  economic_present=True
```

**All four keys present in both states, and the refused arm-day is symmetric with
the OK one** — DE's own unfiled defect (`economic` popped when falsy) is fixed and I
drove the fixed case.

```
POSITIVE: real sealed + unsealed pair                              ADMITS
POSITIVE: the pair for a REFUSED arm-day                           ADMITS
KNOWN-BAD (the PRE-FIX layout): unsealed drops the two sealed_*    REFUSES
KNOWN-BAD (DE's 2nd defect): unsealed with `economic` popped       REFUSES
MINE: a SEALED artifact carrying `economic`                        REFUSES
MINE: unsealed artifact whose `sealed` flag says True              REFUSES
MINE INTERIOR: the same SEALED artifact passed as both states      REFUSES
MINE INTERIOR: the same UNSEALED artifact passed as both states    REFUSES
```

**Eight cases, two admits, six refusals, and the known-bad is the real pre-fix
layout rather than a constructed one.** The two interior cases matter because they
are the ones a careless caller produces (passing one artifact twice), and both
refuse on two independent grounds.

**And the leak guard keeps its non-substring property**, which I re-drove:

```
POSITIVE: the real sealed artifact                                 ADMITS
KNOWN-BAD: null_sd planted at depth inside a nested list           REFUSES
             -> 'days[0].x.admissibility.null_sd'
MINE NON-VACUITY: an artifact whose sealed_field_names IS the list
                  of economic NAMES                                ADMITS
```

The last is the needle-matches-its-own-prose case. It admits, correctly — the guard
tests **keys**, not substrings, so the artifact may name what it withholds.

---

# 4. DESIGN v8's TWO CONVENTION FIELDS

## 4.1 `seed_convention` — **I rebuilt the seed from the declared fields alone**

The property the field exists for is that a reader can reconstruct the seed from
the declaration without the code. So I wrote my own reference implementation from
the seven declared fields and drove it against `seed_for()`:

```
book aaaaaaaa… CONDVALUE_X_SKEW        mine=2980092173  seed_for()=2980092173  MATCH
book f3c1f3c1… HAZARD_OVER_SKEWED_REF  mine=490527606   seed_for()=490527606   MATCH
book 00000000… CONDVALUE_X_SKEW        mine=3493277429  seed_for()=3493277429  MATCH
book deadbeef… HAZARD_OVER_SKEWED_REF  mine=3525611464  seed_for()=3525611464  MATCH
```

**The convention is complete and it is the one the runner implements.** The eight-hex
truncation and the domain separator — the two values R-572(B)(4) named by hand — are
the values in the fields.

## 4.2 **FINDING — one declared field is INERT, and the battery's mutation hides it**

DE's check reads *"AND EVERY DECLARED FIELD IS LOAD-BEARING — mutating any ONE of
… changes the seed. A convention with an inert field is a convention that did not
need declaring."* I mutated each field myself:

| field | my mutation | result |
|---|---|---|
| `domain_separator` | append `_X` | DISAGREES |
| `joiner` | `\|` → `#` | DISAGREES |
| `field_order` | swap first two | DISAGREES |
| `hash` | sha256 → sha1 | DISAGREES — **but not in DE's mutation set at all** |
| `hex_truncation_chars` | 8 → 16 | DISAGREES |
| `int_base` | 16 → 32 | DISAGREES |
| **`encoding`** | **utf-8 → ascii** | **IDENTICAL SEED — INERT** |

**DE mutates `encoding` to `utf-16`**, which changes the byte width and therefore
always disagrees. Every input this convention takes is ASCII — a hex digest, an arm
name, a fixed separator — so against the *plausible* neighbours (`ascii`,
`latin-1`, `utf-8`) the field does nothing. **The extreme mutation passes where the
interior one does not discriminate: the same shape as the queue-model orientation
two rounds ago, where `front == back` would have been as blind as the boundary.**

Not a hazard — an inert field cannot change a number, which is what inert means.
But the claim *"every declared field is load-bearing"* is falsified as written, and
`hash` is asserted by the sentence while absent from `_muts`. Either mutate
`encoding` to `ascii` and record it as **documentation, not a parameter**, or say it
is pinned against a future non-ASCII arm name. Add `hash` to the set.

## 4.3 `dry_run_ledger_scope` — **verified against the CALL GRAPH, not against a second string**

DE's own check *"compares this declared scope against the runner's own
`what_this_reads` / `what_this_does_NOT_read`"* — two declarations in one file,
both authored together. So I checked the claim against the code by AST, taking the
transitive callee set of `dry_run_ledger()`:

```
transitive callee names resolved: 23
verify_be_module           not called   verify_pinned_models   not called
verify_pinned_thetas       not called   verify_day_inputs      not called
verify_run_inputs          not called   resolve_draws          not called
generate_draws_in_process  not called   import_be_cascade      not called
arm_day / aggregate / seal not called
```

**The declared `does_NOT_verify` — P2, P6, P7 — is TRUE OF THE CODE.** The scope
field is honest, and DE's own `therefore` is the right sentence: *"a GREEN dry run
is evidence about the LEDGER and the ruled day set, and about nothing else. It is a
before-picture, not a preflight."*

---

# 5. THE TOP-LEVEL `as_of` — **MY OPEN FINDING IS CLOSED, at the real pair**

I read the `cb9bf8a` pair from git myself and ran the current classifier over it —
not DE's fixture:

```
be_ceiling_null_v1.json  cb9bf8a^ -> cb9bf8a
  shared leaves 1110 · added 8 · removed 0
  moved: {'as_of': 'timestamp'}   2026-09-05T15:15:22Z -> 2026-09-06T02:13:44Z
  n_substantive               : 1
  substantive_paths           : ['as_of']
  nothing_but_provenance_moved: False
```

**Last round the same pair returned `{'provenance': 1}` and "nothing but provenance
moved". It now reports the one substantive edit as substantive.** DE's counts
reproduce exactly under my own derivation.

**Eight interior cases of mine, all correct:**

```
top-level as_of                     timestamp    substantive=True
top-level emitted                   timestamp    substantive=True   <- DE's 2nd defect, fixed
top-level generated_at              timestamp    substantive=True   <- ditto
cancellation_economics.as_of        timestamp    substantive=True   <- MY A-1b case
provenance.as_of                    provenance   substantive=False
source_identity.generated_at        provenance   substantive=False
arms.X.provenance.as_of             provenance   substantive=False  <- nested container
top-level run_id                    provenance   substantive=False
POSITIVE CONTROL: three provenance leaves move
     -> n_substantive 0, nothing_but_provenance_moved TRUE
```

**The class stays `timestamp` and SUBSTANTIVE is computed** — which is the second of
the two options I offered and the better one: it carries strictly more information
than folding a stamp into `numeric-substantive`, and a consumer resolves
`n_substantive` / `nothing_but_provenance_moved` rather than reading a class name.
**And DE found a second defect in the same place that I had not:** top-level
`emitted` and `generated_at` sat in `PROVENANCE_PAIRS` while `_is_provenance` runs
first, so they classified provenance while `as_of` classified timestamp — *"a list of
exceptions wearing the words 'a consistent rule', and the check passed only because
it tested `as_of`."* All three are consistent now, driven above.

---

# 6. `producing_code_is_the_committed_bytes` — one half implemented, one half absent

**The fixture half: RECORDED, and I drove the instrument both ways.**

```
POSITIVE : the committed runner        -> True   (carrying_commit af37cb78c960, tree_dirty False)
KNOWN-BAD: an UNCOMMITTED file in repo -> False
landed fixture receipt v7, /source_identity/:
   carrying_commit                       8deee7a1981ea508…
   producing_code_is_the_committed_bytes True
   tree_dirty                            True
   tree_dirty_is_NOT_the_check           "other seats' uncommitted files say nothing
                                          about this producer"
```

**The property is the right one** — THIS file's blob at HEAD against the bytes that
ran, with `tree_dirty` reported beside it and explicitly not the check. And the
landed receipt's own value is `True`, so v7 was emitted from committed bytes even
though the tree was dirty with other seats' work.

**The real-day half: NOT IMPLEMENTED.** Full grep over `live/`, five lines, reported
in full:

```
de_multiday_gate1_runner.py:338   the producer of the field
de_multiday_gate1_runner.py:1704  a selftest assertion (must be True)
de_multiday_gate1_runner.py:1706  the same check's label
de_multiday_gate1_runner.py:1715  a selftest assertion (must be False)
de_multiday_design_declaration.py:857  the producer, in the design emitter
```

**No refusal cites it anywhere.** R-574(A)'s ruling — *recorded for fixtures, a real
day refuses it* — has no code for its second half, **and there is no place to put it
yet**: the argument parser accepts `--selftest`, `--fixture-run`, `--dry-run-ledger`
and `--output`, and nothing else. **`--day` is unbuilt — verified at the parser.**
So: item 6 is half-done by construction, correctly reported by DE as such, and the
missing half belongs to DE 78 with `--day`.

---

# 7. WHAT ELSE THE RE-DRIVE FOUND

## 7.1 **FINDING — a hardcoded predicate in a landed receipt, contradicted by the committed file**

`de_multiday_gate1_runner.py:1045` — `"the_committed_day_set_is_empty": True,` —
a bare literal. Full grep over `live/`: **one line. Nothing computes it, nothing
asserts it.** And it is false:

```
receipt v7 says the_committed_day_set_is_empty : True
ruled_day_set() reads from the committed file  : ['2026-09-03' … '2026-09-08']  (n=6)
                                        -> the claim is FALSE
```

The prose beside it is stale in the same way: `why_fixtures` still reads *"the
reviewer has not filed on design v3 and BE's book declaration is in flight"* — I
filed on design v3 at **2026-09-06T04:14Z** (`REVIEW_DESIGN_V3_2026-09-06.md`,
committed), and R-555 ruled the day set at 04:13Z. **Fixture receipt v7 is stamped
05:58:19Z, well after both.**

This is CLAUDE.md rule 10 — *compute predicates, never print conclusions; a
hardcoded verdict string beside a table has contradicted the table three times* —
and it is **the exact twin of DE's own finding (b) this round**:
`days.G_is_PENDING_the_USER_parameter` read `True` four hours after R-555 answered.
**DE found that instance in the design module, fixed it, and left this one in the
runner's receipt.** Cheap: derive it from `ruled_day_set()`, or delete it — the
fixture's data-free proof and the ruled-day lock already carry the real assurance.

## 7.2 Three of my own open findings are CLOSED

* **The fixture receipt's data-free witness** (R-572(C)): now a witness, not a
  boolean — `n_paths_opened 34`, `n_distinct_paths 10`, `data_paths_opened []`,
  `non_vacuous true`, `produced_in_the_same_process_as_the_claim true`, with the pid.
* **The fixture run reads the verdict ledger and cannot be driven from a shell
  worktree** (my REVIEW_RUNNER §8): the design battery is **no longer run in fixture
  mode**, with the reason stated as a field (*"the design module's selftest READS THE
  LEDGER, so running it here would make FIXTURE_RUN_NO_DATA false"*). **I drove
  `--fixture-run` from my own worktree: 0.12 s, 23 MB, `FIXTURE_RUN_NO_DATA`,
  battery 63 + 8 = 71 computed, `data_paths_opened: []`.** It works where it did not.
* **The top-level `as_of` classification** (§5).

## 7.3 The `--dry-run-ledger` scope field is the honest kind

Naming P2/P6/P7 as **not covered** is a field that reduces the value of DE's own
green result, and it is verified true of the call graph (§4.3). That is the right
direction for a self-description to point.

---

# 8. VERDICT ON THE RUNNER

**The fixture path is APPROVED and it is better than the 42-check version I
approved:** the draws are bound to the cascade in the process that produced them,
the door is shut structurally rather than on the caller's word, the seal layout is
symmetric in both states with the pre-fix layout as its known-bad, and the receipt
now proves its own data-freeness with a witness.

**The 09-03 smoke cannot be approved to RUN, and not because of a defect: `--day`
does not exist.** Verified at the argument parser. So the accurate statement is:

> **APPROVED for the smoke, to take effect when `--day` lands and BE's book exists —
> conditional on three items, all of which belong to `--day`'s own batch:**
>
> 1. **`may_run_day` reads `ruled_day_set()`, not `params['days']`** (§2.3). This is
>    the day-admission gate; it must not be readable from the caller's dict, and it
>    is one line. Same for `may_read_aggregate`'s `G`.
> 2. **`the_committed_day_set_is_empty` computed or deleted, and `why_fixtures`
>    re-stated** (§7.1). A false hardcoded predicate in a landed receipt is the
>    thing rule 10 exists for.
> 3. **R-574(A)'s real-day refusal on `producing_code_is_the_committed_bytes`**
>    (§6), which has nowhere to live until `--day` exists.
>
> **Two items to carry rather than fix:** the day-level reproduction gate is OPEN
> and named (§1.5) — it needs BE's per-day book receipt and must not be lost; and
> `encoding` is an inert declared field (§4.2), a reporting item.

Nothing in this batch changes my §A.6 conclusion from REVIEW_BE48 that the six-day
schedule is a sequencing problem before it is a memory one.

---

# 9. THE LOCK HAZARD FROM REVIEW_BE48 §A.6 — named precisely, and it has since corrected itself

The coordinator asked what I measured. Exactly this:

**The run that was heavy and unlocked** — from the systemd journal, not from a
process listing:

```
be49frag.scope :  /usr/bin/env PM_DATA_ROOT=/home/yuqing/ctaNew python3
                  live/pm_research/be_gate1_fragment.py --day 20260903
   Started  2026-09-06 05:48:14
   Ended    2026-09-06 05:58:24   "Consumed 10min 3.910s CPU time"
   cwd      /home/yuqing/ctaNew-wt-be        cgroup research.slice/be49frag.scope
   memory.current when I sampled it at 05:54:01Z : 2.14 GiB
```

Heavy on **both** of rule 20's criteria — 10 minutes of CPU and 2.14 GiB.

**What held the lock while it ran** — measured at **2026-09-06T05:54:01Z**, when my
own light check was refused with rc 1 (reported, not waited on):

```
fuser /home/yuqing/ctaNew/data/.heavy_run.lock
   -> pid 2902400 (flock) / 2902403 (python3)
      = live/mm_research/e2_a_runner.py --census ICPUSDT ADAUSDT BTCUSDT
        cgroup research.slice/run-u69843.scope, memory.current 3.51 GiB
research.slice: MemoryMax 14 GiB, MemoryCurrent 7.19 GiB, two heavy scopes
```

So `be49frag.scope` ran heavy for its full ten minutes **inside `research.slice` but
without the lock**, concurrent with a lock-holding heavy run. Two scopes each capped
at 8 G can reach 16 G against a 14 GiB slice ceiling; R-551's *one heavy run at a
time* is what prevents that, and the lock is its only enforcement.

**AND IT HAS CORRECTED ITSELF — re-measured at 2026-09-06T06:10:45Z:**

```
research.slice holds ONE scope: be50tape.scope, 3.03 GiB
   = python3 live/pm_research/be_gate1_state_tape.py --day 20260903  (elapsed 3m42s)
its parent process is literally:
   flock -n /home/yuqing/ctaNew/data/.heavy_run.lock systemd-run --user --scope
         --slice=research.slice …
```

**BE's next stage takes the lock.** So this reads as a one-off gap in BE 49's
fragment stage rather than a practice, and the routable form is narrow: *name
`be49frag.scope` to BE, confirm the wrapper is on every stage of the feature pass.*

**And a fact for the Gate-1 line, not a finding:** `be_gate1_state_tape.py --day
20260903` is running right now. That is the **second** of the two missing artifacts
I identified in REVIEW_BE48 §A.3 — the state tape, without which every September row
is a `state_join_failed` drop. The block I extended is being worked from both ends.

---

## CONTEXT

Approximately 62%. Below the 80% reset threshold; I will report the crossing.
