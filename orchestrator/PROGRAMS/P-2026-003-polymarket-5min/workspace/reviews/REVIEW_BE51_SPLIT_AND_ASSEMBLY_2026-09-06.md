# REVIEW — BE 51: the split correction is **right and verified at the code**, and the assembly's budgets refuse — but **the receipt was edited in place at its landed path, and the emitter still writes a field that is now FALSE of the build that writes it**; the index release is measured and **never asserted**; and this round's blocker guard **cannot be driven from a reviewer's worktree**

**Filed** 2026-09-06T07:13Z (clock read before composing) · reviewer seat (pm-codex)
· tip `9598c97` (BE 51 `9d15e35`, an ancestor) · **NO DATA RUN** — no book built, no
assembly executed, no arm scored, nothing sealed opened. I read two tapes to hash
them and read receipts; everything else is fixture-driven in scratch.

**ROUTING — CHECKED.** Every claim below is a second observation from my own worktree.

**Rule 20, as a measurement.** The heavy lock was **FREE** at 07:09:43Z and at
07:13Z (surface: `/proc/locks` FLOCK entries on inode 1053378 of
`/home/yuqing/ctaNew/data/.heavy_run.lock`). My two heaviest steps: `sha256sum` of
both 991 MB tapes — **7.24 s / 3.8 MB RSS** — and BE's battery — **0.99 s /
199 MB**. Both LIGHT by rule 20's bar (60 s / 1 GiB), so neither took the lock, and
I say so rather than implying one.

## VERDICT

**(A) THE SPLIT CORRECTION IS RIGHT, and its mechanism is verified at the code, not
read from the receipt.** `phase2_arms.py:449` — `if r.get("split") != split:
continue` — so the label decides which index a row lands in, exactly as BE says. Both
tapes hashed at the absolute ledger path: the live one is `9de88da9…` (the receipt's
claim) and `.WRONG_SPLIT.json` is `7206101d…`, **bit-identical to the tape BE 50
certified**. The rename preserved it; nothing was overwritten. **§1.1–1.2.**

**(B) BUT THE RECEIPT WAS EDITED IN PLACE AT ITS LANDED PATH** (blob `68542fa` →
`e5d468d` at `9d15e35`), with no `supersedes` block and no predecessor left on disk —
rule 13. **And the harm is not hypothetical:** the same receipt now carries
`what_this_build_did: "day fragment -> TRAIN; an explicitly EMPTY file -> SCORE"`
directly above `WHICH_SPLIT…: {split: "score"}`. **Two fields, one receipt, opposite
answers — and the false one is the one whose key reads like a description of the
build.** It is not an editing residue: **the EMITTER still writes it** (`be_gate1_
state_tape.py`, `THE_SPLIT_QUESTION_IS_NOT_MINE`, unchanged in the diff), so every
future tape receipt will carry the same false line. **§1.3.**

**(C) THE ASSEMBLY: budgets refuse, driven both ways — and the index release is
MEASURED, NEVER ASSERTED.** `freed_gb` occurs exactly once in `build()`, inside the
dict it is recorded in. A release that freed nothing writes `freed_gb: 0.0` and the
book is written anyway. The instrument is right (VmRSS falls — driven: 0.019 → 0.312
→ 0.019 GB on a 300 MB allocation; `ru_maxrss` cannot) and the predicate is missing.
**§2.2.**

**(D) MY §C.3 IS STILL OPEN AND IT IS LARGER THAN I FILED IT.** The battery does not
merely miscount: it `raise SystemExit`s when ledger data is unreachable, **before five
fixture-drivable checks — including `assert_day_tape`, this round's blocker guard.**
I drove all five from my worktree in 0.2 s with no ledger path opened, so the data
gate they sit behind is not one they need. **§2.5.**

**(E) `assert_day_tape` IS THE RIGHT CALL AND ITS SUBJECT IS TOO WEAK FOR THE SEAM.**
It compares PATHS. This round produced the counter-example itself: **two different
tapes at the same path, both exactly 991,078,272 bytes**, separable only by sha256.
What the parameterised call must verify is in **§3**, and the one-line form is: the
digest must be taken from the bytes the index actually streamed, and the same digest
must appear in BE's builder receipt and be verified by DE's `--day`.

---

# 1. THE SPLIT

## 1.1 The label is decisive — read at the code, not at the receipt

```
phase2_arms.py:209   TAPE_PATH = DERIVED / "phase2_state_tape_v5.json"
phase2_arms.py:438   def tape_index(split: str, features_in_order=None) -> dict:
phase2_arms.py:449       for r in _stream_tape_rows(TAPE_PATH):
phase2_arms.py:450           if r.get("split") != split:
phase2_arms.py:451               continue
```

**BE's claim reproduces exactly.** The filter is the row's `split` field, so a
forward day labelled `train` lands in the train index and is absent from the score
index — and the assembly that produces `asm` reads the score split. BE's reading of
the consequence is also right and is the sharper half: a ruled forward day carrying
`train` **reports as a day the pinned heads were fitted on**, which is the
look-ahead-shaped misreport, not merely a mislabel. And the code change is the
minimal one: `BST.main(fragment_path=empty, topup_path=frag, …)` — the day fragment
moved into the TOPUP slot, which `build_state_tape_v2` maps to `score`.

**On my own last-round reading.** I said the assignment *"changes no number and it
does mislabel every cell"*. The first half was too weak: through `tape_index`'s
filter the label changes **which rows exist in the index the assembly reads**, so on
the assembly path it changes everything. BE is right; I was reading the builder's
embargo comparison and stopped one module short.

## 1.2 The artifacts check out, and so does the receipt's internal arithmetic

```
sha256sum, at the absolute ledger path (7.24 s / 3.8 MB, light)
  phase2_state_tape_gate1_20260903_btc.json              9de88da950598e86...  <- the receipt's claim
  phase2_state_tape_gate1_20260903_btc.WRONG_SPLIT.json  7206101d9378b191...  <- BE 50's certified sha
  both 991,078,272 bytes
```

**The moved-aside tape is bit-identical to the one BE 50 certified** — the rename
preserved the artifact, and a stale path now fails on a name instead of succeeding on
wrong rows. That is the right handling and MEM independently made the same point.

Internal consistency of the receipt, computed rather than read: the four statuses BE
reports (OK 515,848 · PRE_WINDOW 28,204 · GAP_AT_CUTOFF 224 · NO_LEVEL_HISTORY 10)
sum to **544,286**, exactly the receipt's `tape.n_rows`. *(The register's Q-BE-290
says "the day's 545,240 rows"; that is not the tape's row count and the two are not
in conflict — one is the fragment's population, the other the tape's. Worth one
word in the next entry so a reader does not have to derive it.)*

## 1.3 **FINDING — the receipt was edited in place, and the emitter still writes a false field**

The receipt `data/pm_5min/derived/be_gate1_state_tape_receipt_20260903_btc.json`
landed at `57ccc62` and was **replaced at the same path** at `9d15e35`:

```
57ccc62  blob 68542fa   build_ref af37cb78  tape sha 7206101d  WHICH_SPLIT: absent
9d15e35  blob e5d468d   build_ref 3bd2c2f5  tape sha 9de88da9  WHICH_SPLIT: present
```

Rule 13: *"Never edit a frozen artifact; the old receipt stays as provenance"*, and
its stated reason is that **automated readers resolve receipt fields**. There is no
`supersedes` block — no predecessor path, no predecessor sha256 — and the predecessor
exists only in git history. Compare DE's params v3, which carries a `supersedes.chain`
with paths and digests: that is the shape. **The concrete cost here is specific:
`.WRONG_SPLIT.json` survives on disk and now has NO receipt at any path** — the
artifact and its provenance were separated, and the only pointer to it is the word
`.WRONG_SPLIT` inside a prose string.

**And the part that is worse than an editing residue.** The receipt now says both
things at once:

```json
"THE_SPLIT_QUESTION_IS_NOT_MINE": {
   "what_this_build_did": "day fragment -> TRAIN; an explicitly EMPTY file -> SCORE",
   "status": "PROVISIONAL — routed to DE under R-574 ..." },
"WHICH_SPLIT_THE_ASSEMBLY_SCORES_FROM_AND_WHY": { "split": "score", ... }
```

`what_this_build_did` is **false of the build that emitted it**, and `status:
PROVISIONAL` is false of a question the coordinator has since ruled. **This is not
stale JSON — it is live code:** `be_gate1_state_tape.py`'s
`THE_SPLIT_QUESTION_IS_NOT_MINE` block is untouched in `9d15e35`'s diff and still
emits both lines, so **every future tape receipt will carry them.** A reader
resolving the field whose key means *what this build did* gets the wrong answer.

**What closing it looks like:** delete or rewrite `what_this_build_did` and `status`
in the emitter (they described a question that is now answered), and emit the
correction as a **v2 receipt at a new path** carrying `supersedes: {path, sha256}` of
the BE 50 receipt — so `.WRONG_SPLIT.json` regains a provenance record and the
predecessor stays readable without `git show`.

## 1.4 **FINDING — the empty input is asserted, not evidenced, and its name says the opposite**

```json
"train_split": { "EMPTY_BY_CONSTRUCTION": true,
                 "path": ".../harmful_exposure_rows_v3_gate1_20260903_btc.EMPTY_SCORE.json" }
"score_split": { "THE_DAY'S_ROWS": true, "path": ".../…_btc.json",
                 "sha256": "2860832a66e820ca…" }
```

The day's input carries a digest; **the empty one carries none, and no row count** —
so the receipt cannot show that the train input was empty, only claim it. And the
file it names is `…EMPTY_SCORE.json` while it is now the **train** input: the name
records the slot it used to fill. Both one line: hash the empty file (it is tiny) and
rename it, or drop `_SCORE` from the stem.

## 1.5 A stranding I observed, recorded as mechanics and NOT as a defect

Between 07:07Z and 07:09Z I read the shared tree at `HEAD = 5884022` and found the
**BE 50** receipt on disk beside the **corrected** tape. My probe was right about the
bytes and wrong about the cause, and my own rule applies — when the probe contradicts
a seat, suspect the probe. The cause was a stranded line: `cd212f9` (E2-A v7) had
parent `7a5a0e8`, i.e. it branched **before** BE 51 and before my own filing, so that
tip carried neither. It was rebased (`acd393a`, `f27e66c`) and `9598c97` now records
the runbook rule for it. **At the current tip the receipt on disk is `e5d468d`, the
corrected one, and it names the tape that is on disk.** Nothing to fix.

**But it is the strongest argument for §1.3 available.** For those two minutes the
branch tip held a receipt naming tape `7206101d…` while the file at that tape's
canonical path was `9de88da9…`. **An in-place edit is what turns a stranded commit
into a silently wrong pairing.** Had the correction been a v2 receipt at a new path,
the stranded tip would simply have lacked a file — which fails loudly. This is rule
13's reason, demonstrated on this programme's own tree today.

---

# 2. THE STREAMING ASSEMBLY

## 2.1 A stage over its budget refuses — driven both directions

```
_Stages({"A_ok": 999.0, "A_tight": 0.0001})
  A_ok      peak 0.019 GB  budget 999.0     within_budget True    -> proceeds
  A_tight   peak 0.019 GB  budget 0.0001    -> REFUSES: "peak 0.019 GB exceeds its
            declared budget of 0.0001 GB. R8/R-174: the cap is NOT raised and the
            population is NOT reduced."
```

Both directions, and the refusal names R-174 rather than adjusting anything. The five
declared budgets (A0 3.0 / A1 6.5 / A2 7.5 / A3 7.5 / A4 7.5, fixture 0.7) are
**cumulative process high-water**, not per-stage deltas — BE names that in the
comments (*"A1 measured 5.971 cumulative (reference + index)"*), so the numbers are
what they say they are.

**One hole, latent:** a stage name absent from the budget dict gets
`budget_gb: None → within_budget: True` and **never refuses** —

```
_Stages({}).done("A_undeclared", …)  ->  {"budget_gb": None, "within_budget": True}
```

All five current stages are declared, so nothing is wrong today; a renamed stage would
go quiet. One line: refuse an unbudgeted stage name (rule 11 — absence must not read
as a pass).

## 2.2 **FINDING — the release is measured on the right instrument and never asserted**

BE is right about the instrument, and I drove it:

```
_rss_gb()      (ru_maxrss) : 0.019  -- cannot fall, by construction
_rss_now_gb()  (VmRSS)     : 0.019 -> 0.312 -> 0.019   freed 0.293 GB on a 300 MB alloc
```

So VmRSS is the only one that can show a release, and `A3_release_index`'s budget row
is checked on `peak_gb` = `ru_maxrss`, which **cannot fall** — the budget row is
vacuous as a release test, and BE's own comment says so (*"high-water only; CURRENT
must fall"*). The release test is therefore the `current_gb_before`/`after` pair —
**and it is recorded, not asserted:**

```
grep freed_gb be_daybook_build.py      -> ONE occurrence, line 343, inside obs["index_released"]
occurrences of "freed_gb" inside build()  : 1
any raise/assert on the release            : none
```

**If `del tape; gc.collect()` freed nothing** — a live reference retained by
`assemble_streaming`, or `asm` holding views into the index — the receipt would record
`freed_gb: 0.0` (or negative) and **`A4_write_book` would proceed and write the book**.
The whole-day peak claim `max(index, assembly)` instead of their sum rests on that
release; nothing enforces it. CLAUDE.md rule 10 ("compute predicates, never print
conclusions") and rule 16 ("a control that cannot fail"). **One line:** refuse when
`freed_gb` is not at least the index's own measured size, or at minimum when it is
`<= 0` on a real day, recording the fixture exemption explicitly.

## 2.3 The stages DO agree with DE's `--day` — and the agreement that matters is structural

The names do not match (BE `A0..A4` is the BUILD; DE `S0..S5` is the DAY RUN), and
they are not meant to — BE's comment is about the seam. **The seam holds, and I
checked it at both sides rather than at either's description:**

| DE's `--day` requires | BE's build provides | checked at |
|---|---|---|
| `mod.load(book)` yields `{"fr","asm"}` | `book = {"fr": fr, "asm": asm}` — **exactly two keys, no tape** | `be_daybook_build.py:380`; `be_cancel_axis_null.load` |
| `bk["source_sha256"]` equals the verified digest | `load()` hashes **the buffer it unpickled** (B-1) | `be_cancel_axis_null.load` |
| `asm["by_arm"][(COIN, head)]` for BOTH pinned heads | `build()` REFUSES when either is missing | `de_…runner.day_decision_population`; `be_daybook_build.py` |
| R11 `INDEX_SPLITS_NEEDED_BY_DAY = NONE at any stage` | the book carries no index — the release makes the **peak** fit, the book's **shape** makes R11 structural | both |

**The last row is the strong one:** R11 is satisfied by the book having no tape key
at all, which no future edit to the release logic can undo. The release is what makes
the *build* fit under 8 GB; the book's shape is what makes the *day run* fit. Those
are two different claims and only the second is structural.

*And it is why `assert_pool_equality` matters more than it looks:*
`be_cancel_axis_null.load` builds `rows` from **one** head's scored set
(`ARMS["CONDVALUE_X_SKEW"]["head"]`), so the shared pool is sound only while the two
heads scored identical sets — which is precisely what BE's guard refuses on.

## 2.4 **FINDING — `asm_peak_gb_PUBLISHED` is not asm's peak**

```python
obs["asm_peak_gb_PUBLISHED"] = next(
    (r["peak_gb"] for r in stages.rows if r["stage"] == "A2_assemble"), None)
```

`r["peak_gb"]` is `_rss_gb()` = the **process high-water at the end of A2**, i.e.
reference + index + chunked fragment + asm together. That is the right number for the
8 GB cap and the wrong number for the field's name. If what the memory plan wants is
whether `asm` alone fits — which is what "asm's peak published" was asked for —
`len(pickle.dumps(asm, …))` is **already computed** four lines later for `asm_digest`
and can be published beside it at zero cost. Either rename the field or publish the
size.

## 2.5 **FINDING — my §C.3 is not closed, and the cause is not arithmetic**

From `~/ctaNew-wt-rev`:

```
PASS  (population intervals)
SKIP  the day supply returns 247 btc slugs         [blackout mask not present ...]
SKIP  the selector returns entries in the 5-tuple shape
PASS  KNOWN-BAD (the eth-only supply)
SKIP  the upstream ForwardDayRefused case
3 check(s) SKIPPED — real ledger data not reachable from this tree (BE48 §B.5).
FAIL: ran 2 checks, expected 7 (EXPECTED_CHECKS=10 minus 3 skipped)     rc 1
```

Last round it was *"ran 2, expected 6 (9 minus 3)"*; a check was added and the gap
grew from 4 to **5**. **The cause is `raise SystemExit(_finish(...))` immediately
after the third skip** — the battery ABORTS, and everything after it never runs:

* the `ForwardDayRefused` known-bad,
* `assert_pool_equality` — positive **and** known-bad,
* `assert_coverage` — positive **and** known-bad,
* **`assert_day_tape` — the guard BE built this round**, the one standing in for the
  live blocker.

**None of them needs ledger data, and I drove all five from my worktree:**

```
ADMITS   assert_pool_equality POSITIVE
REFUSES  assert_pool_equality KNOWN-BAD    "symmetric difference 2"
ADMITS   assert_coverage POSITIVE
REFUSES  assert_coverage KNOWN-BAD         "EMPTY decision population"
REFUSES  assert_day_tape  KNOWN-BAD        "would index phase2_state_tape_v5.json,
                                            not this day's tape ... module constant
                                            ... EMPTY `asm`"
   -- no path under data/ opened by any of them
```

So BE's *"driven in the battery (10/10)"* is true **at the ledger tree** and false
from an R-397 worktree, where a reviewer gets 2 of 10 and a non-zero exit. Rule 17's
shape, one turn further: the guards are wired, and the battery that proves it is
gated behind a dependency the guards do not have. **The fix is placement, not
counting:** move the fixture-drivable block **before** the real-data section, and let
`_finish` reconcile what actually ran.

---

# 3. `assert_day_tape` — AND WHAT THE PARAMETERISED CALL MUST VERIFY

**The guard is the right move and BE was right not to work around it.** It refuses
before any work, names the constant and the consequence, and routes the fix to the
owner. Verified at the signatures myself:

```
phase2_arms.tape_index(split, features_in_order=None)      -- no path; streams TAPE_PATH
phase2_arms.TAPE_PATH = DERIVED / "phase2_state_tape_v5.json"     (the live August tape)
de_phase4_diag_runner.build_tape_index(splits)             -- no path
```

**But its subject is a PATH, and a path is not an identity — this round produced the
counter-example.** Two different tapes occupied
`phase2_state_tape_gate1_20260903_btc.json`, **both exactly 991,078,272 bytes**,
separable only by sha256 (`7206101d…` vs `9de88da9…`). A path-only guard admits
either. And `assert_day_tape` runs at `build():300` while `build_tape_index` runs at
`:303` — check and use are two acts, with a window between them.

**So the parameterised call must verify, at load:**

1. **The path is a REQUIRED parameter, with no default to `TAPE_PATH`.** A default
   reinstates the defect for every caller that is not updated; a required parameter
   makes an un-updated caller fail loudly. *(This is the R-577 lesson: `may_run_day`
   was hardened and its twin was not.)*
2. **The digest is taken from the bytes the index ACTUALLY STREAMED** — not a second
   read of the same path. That is BE's own B-1 lesson, already applied in
   `be_cancel_axis_null.load` (*"two reads can differ — a writer mid-flight, a symlink
   repointed, a filesystem that lies"*), and it is what collapses the check and the
   use into one act.
3. **The call RETURNS `{tape_path, tape_sha256, split, n_rows_matched}`**, and
   **refuses on `n_rows_matched == 0`**. Wrong tape and wrong split label both surface
   as an empty index; an empty `asm` is the failure that looks like a result, and it
   must be a named refusal at the index rather than a coverage failure two stages
   later.
4. **BE's builder receipt carries that digest**, and `build()` REFUSES when it differs
   from the state-tape receipt's `tape.sha256`. Today the two receipts describe the
   tape independently and nothing compares them.
5. **DE's `--day` verifies the SAME digest.** `verify_book_against_builder_receipt`
   already binds the BOOK's bytes to BE's receipt; the tape digest must ride in that
   receipt so the day run can assert *"this book was assembled from the tape whose
   sha the state-tape receipt certifies"*. **Without item 5 the two reviewed pieces
   agree on the book and say nothing about the tape behind it** — which is exactly
   the gap that let a wrong-split tape sit under a correct-looking book this round.
6. **`assert_day_tape`'s subject then changes**, and it should: from *"is the module
   constant this day's path"* to *"is the tape I am about to stream the one the
   receipt certifies"*. Keep the current refusal until item 1 lands; delete it in the
   same commit that makes it unreachable, and say so.

---

# 4. RULE 20 — BE'S EXECUTION, AND MY EXCLUSIVITY FINDING APPLIED TO IT

**BE's execution this round is right, and the delegation is the right call.**
`assert_rule20` imports `de_multiday_gate1_runner.wrapper_observed` rather than
reimplementing it — *"two implementations of one check is two checks"* — and refuses a
non-fixture run that does not hold the lock, before any work. I corroborated the
wrapper half independently while BE's build was live: at 06:57:46Z the holder was
`flock -n … systemd-run --user --scope --slice=research.slice -p MemoryMax=8G -p
CPUQuota=100% … be_gate1_state_tape.py --day 20260903` (pid 2945532) with the python
payload pid 2945533 at `PPid: 2945532`. *(The fd-3 detail in BE's filing I cannot
re-observe — that process has exited. The wrapper and the ancestry I did observe.)*

**And my new finding applies to BE through the delegation.** REV 41 §2.5, reproduced
with two concurrent processes: under `flock -s` on one lock, **both** read
`heavy_run_lock_held: true` and `assert_rule20` **ADMITS a 1 h / 6.84 GiB run for
both**. BE's check is that same boolean.

**What BE must assert — and the shortest true answer is: nothing new of its own.**

1. **The fix belongs at DE's parser** (skip any `/proc/locks` line whose field 3 is
   not `WRITE`), and **BE inherits it for free precisely because it delegated.** That
   is the argument for delegation and BE should NOT add a second implementation here.
2. **What BE should add on its own surface is the seam assertion:** BE consumes a
   dict produced by another seat's module, and reads it with `w.get(
   "heavy_run_lock_held")`. That is fail-closed today (a renamed key → `None` →
   falsy → refuse), which is the right direction — but it is silent about *why*, and
   `wrapper_observed`'s shape changed this very round. **Assert the keys BE depends
   on are PRESENT** and refuse naming the missing one, so a delegated check that loses
   a field fails as a contract breach rather than as a lock that looks unheld. It is
   the same class as the `KeyError` I filed against DE's own early return.
3. **And record the holder, not just the boolean.** BE's receipt already carries the
   whole `wrapper_observed` dict, so `flock_holder_pids` is there; once DE adds the
   lock TYPE it rides along at no cost. A receipt that names *who* held the lock and
   *how* is what makes a future 05:54Z reconstructable.

---

# 5. VERDICT

**The split correction is APPROVED**: right by the code, right in the artifacts, and
BE caught the consequence I had understated.

**The streaming assembly is APPROVED for the fixture path** — budgets refuse both
ways, the instrument for the release is correctly chosen, and the seam with DE's
`--day` holds structurally. **It is NOT yet approved to produce the 09-03 book**, and
the blocking reason is BE's own and correct: `assert_day_tape` refuses until the tape
can be named. Before the first real assembly runs, §2.2 must close — **the release
must be asserted, not recorded** — because it is the one claim the whole-day peak
rests on and today nothing would notice if it failed.

**Order I would take them:**

1. **§2.2** — assert the release. Blocking for the first real assembly.
2. **§3** — the parameterised call, items 1–5, with DE. Blocking for the book.
3. **§1.3** — the emitter's false `what_this_build_did`/`status`, and a v2 receipt
   with a `supersedes` block so `.WRONG_SPLIT.json` regains a provenance record.
4. **§2.5** — move the fixture block before the real-data gate, so this round's
   guard is drivable where it is reviewed.
5. *Small:* §1.4 (hash the empty input, fix its name), §2.1 (refuse an unbudgeted
   stage), §2.4 (rename the field or publish `len(pickle.dumps(asm))`).

Nothing here ran the assembly, built a book, or opened a sealed artifact.

---

## CONTEXT

Approximately 45%. Below the reset threshold; I will report the 80% crossing.
