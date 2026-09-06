# REVIEW — **the 09-04 book STANDS**, with two disclosed defects that cannot certify anything false; **BE's worktree finding is a HAZARD and it is not the environment variable — DA's own resolver reads the shell with `PM_DATA_ROOT` correctly set**; and **DE 92's lock-dependent check is closed: no battery verdict now depends on the ambient lock. GO.**

**Filed** 2026-09-06T11:31Z (clock read before composing) · reviewer seat (pm-codex)
· tip `6e32506` (BE 59 `50f30d9`; DE 92 `6f0370f`/`6e32506`; R-619 `4d2b40a` — verified ancestors)
· **LIGHT AND LOCK-FREE.** The heavy lock is held by pid 3282334 throughout; **I did not take
it**, and my one lock drive used a **scratch lock file in `/tmp`**, never the ledger's.

**Rule 20.** Heaviest steps: runner battery **25.63 s / 851 MB**, design **1.06 s / 206 MB**,
the book digest **0.21 s / 3.7 MB** (streamed). Batteries at the tip under my run:
**runner 235, design 103** — 0 failures. Both rehearsals **READY, `blocking: []`**.

**ROUTING — CHECKED unless a line says AGREED.**

---

# A. THE 09-04 BOOK

## A.1 The artifact and its digest — reproduced

```
be_daybook_20260904_btc.pkl   9193206c2fa338784bac8dbcb2c66c0f1c90e8c747b3a434a73ec829cffb5858
                              340,969,199 bytes
```

Matches the receipt's `sha256`, its `readback_sha256`, and `readback_matches: true` with
`digest_is_of_the_buffer_as_written: true`.

## A.2 The provenance chain — the bytes resolve, the POINTER does not

Both pinned inputs exist and **both digests match exactly**:

```
harmful_exposure_rows_v3_gate1_20260904_btc.json  839e5377b6666e59  711,128,106 B
phase2_state_tape_gate1_20260904_btc.json         3727de6533712057  1,165,058,495 B
```

**BE's own finding, confirmed:** `inputs_pinned.tape.receipt` names
`be_gate1_state_tape_receipt_20260904_btc.v3.json`, and only `.json` and `.v2.json` exist for
09-04 — **the head is `.v2`**. The literal was *true of 09-03*, where a `.v3` does exist, and
an f-string carried it to a day where it does not. BE's recurring class, found by BE.

**The chain still closes, by content:** the `.v2` receipt pins
`tape.sha256 = 3727de65…` — the same bytes the book names. So the pointer is wrong and
**nothing false is certified**: a verifier following it lands on an absent file and fails
honestly, rather than resolving to a different artifact. That is the good direction of a bad
pointer.

**A third shape, and I would route it:** BE's supersession blocks are **three different
shapes across three receipts** — 09-03 `.v2` carries `artifact` + `v1_tape_sha256` (no
`path`, no top-level `sha256`); 09-03 `.v3` carries `v1`/`v2`/`what_changed` and **no digest
at all**; 09-04 `.v2` carries `artifact` + `sha256`. **R-608's pair `{path, sha256}` is
honoured by none of them.** Under DE's resolver the 09-04 `.v2` is half a link and the 09-03
`.v3` is no link at all; DA's resolves the first through its `artifact` fallback. **Nothing
resolves BE's tape-receipt chain today**, so this is latent — but R-608 has now been read as
a programme rule by two seats, and if it is one, BE's three shapes are three instances.
Either BE 60 adopts the pair, or the ruling says it is scoped to the day chain.

## A.3 The population, its statuses, and set equality as R1's precondition

| | 09-03 | 09-04 |
|---|---:|---:|
| windows / slugs | 247 | **288** |
| reference generations | 313,114 | **358,107** |
| ADMITTED | 247 | **288** |
| TERMINAL_MARK_OK | 247 | **288** |
| TERMINAL_MARK_ENDED_IN_GAP | 63 | **78** |
| TRANCHE_KEPT | 46,439 | **57,850** |
| BINANCE_GAP_EXCLUDED / NO_REPLAY / RECONCILIATION_FAILED | 0 | **0** |
| covered / uncovered | 297,379 / 15,735 | **338,444 / 19,663** |
| coverage | 0.9497 | **0.94509183** |

Recomputed by me: **338,444 + 19,663 = 358,107 exactly**, and
338,444 / 358,107 = 0.9450918300954743 — the receipt's figure to the last digit. Every
exclusion is a **named status with a count**, and the three failure statuses are zero.

**Set equality is ASSERTED, not inferred.** Both heads report `n_scored_keys: 338,444`, and
the receipt carries `sets_are_equal: true` **with** `set_equality_asserted: true` and
`n_shared_keys: 338,444`. Equal counts over different sets is the failure R1's precondition
exists to exclude, and this receipt distinguishes them — which is what I asked for on 09-03.

## A.4 The resources — every stage inside its budget, and one number that moved the right way for the wrong-looking reason

| stage | peak GB | budget GB | wall s |
|---|---:|---:|---:|
| A0_reference | 2.307 | 3.0 | 664.4 |
| A1_index | 3.701 | 6.5 | 140.4 |
| A2_assemble | **4.913** | 7.5 | 1560.8 |
| A3_release_index | 4.913 | 7.5 | 1.7 |
| A4_write_book | 4.913 | 7.5 | 6.8 |

`peak_gb` = `peak_rss_gb` = `asm_peak_gb_PUBLISHED` = **4.913**, consistent across all three
fields. Wall 2,379.8 s against a stage sum of 2,374.1 s — 5.7 s unattributed, which is
overhead, not a gap. `ROUND_49_BUDGET_WITHDRAWN` carries the withdrawal beside the budgets.

**One thing worth naming rather than passing over: 09-04 carries 14 % MORE data than 09-03
and peaked LOWER** — 358,107 generations and 638,602 tape rows against 313,114 and 544,286,
at **4.913 GB against 5.317**. A resource number that moves against the data is either a real
improvement or a changed measurement, and a reader should be told which. The receipt does not
say. My reading, from the stage table: 09-04's index stage peaks at 3.701 and the assembly
adds 1.2 GB on top, where 09-03's assembly reached 5.317 — consistent with the chunking
(`n_chunks 48`, `chunk_windows 6`) holding the assembly's working set flat while the day grew.
**AGREED as a reading, not driven** — I did not re-run either assembly.

## A.5 `reasons_account_for_the_count: false` — the book STANDS, and the reason it stands is checkable

```
UNCOVERED_GENERATIONS.count       19,663      (reference generations not covered)
UNCOVERED_GENERATIONS.reasons_sum 29,465      = 103 + 22 + 29,340
   gap_at_cutoff_excluded 103 | no_level_history_excluded 22 | pre_window_excluded 29,340
reasons_account_for_the_count     false
identical_across_heads            true
```

Two populations, exactly as BE says: the count is **reference generations**, the dominant
reason is **fragment rows**. **A predicate that cannot pass on real data.**

**Does the book stand? YES — and not on my judgement.** I grepped every consumer:

```
live/pm_research/be_daybook_build.py:137   (where it is computed)
live/pm_research/be_daybook_build.py:962   (the battery's positive control)
live/pm_research/be_daybook_build.py:970   (the battery's known-bad)
```

**Nothing downstream reads it.** It gates no assembly, no scoring, no day. The numbers it
comments on are internally consistent and independently checked (§A.3). So the false
predicate is a **diagnostic that does not explain what it is filed under**, and the book is
sound for the 09-04 smoke with that disclosed.

**Two conditions on that verdict, and they are not decoration.**

1. **Nobody quotes `by_reason` as an account of the uncovered generations.** It is a
   fragment-row account sitting under a key that says `UNCOVERED_GENERATIONS`, with
   `count: 19,663` beside it. A reader will take 29,340 `pre_window_excluded` as a reason for
   uncovered generations. It is not one. Relabel it or make the predicate compare like with
   like.
2. **BE's battery asserts the predicate TRUE** (line 962: `ok(_ev[...]["reasons_account_for_
   the_count"] and ... == 10 and ...)`), and the known-bad at 970 constructs a mismatch by
   hand. So both directions are driven **in the fixture** — and the passing direction is the
   one that **cannot occur on real data**, because in a fixture the two populations coincide.

**That is a new shape for rule 16's list**, and it is worth the register's words: not a
control that cannot fail, and not one that cannot fire — **a control with both directions
driven whose PASSING case is unreachable in production.** The battery will stay green on
every real day while the field it asserts is false on every one of them.

## A.6 BE's worktree finding — a HAZARD, and it is **not** `PM_DATA_ROOT`

Confirmed at the machine: **all four seat worktrees have a materialised `data/`**, not the
symlink R-553 makes mandatory, and the 09-04 book is invisible under a worktree path.

**But the premise needs correcting, and the correction makes it worse, not better.** Driven:

```
DE's resolver, from wt-rev, PM_DATA_ROOT UNSET
   -> /home/yuqing/ctaNew/data   is_canonical: True   branch 3_canonical
      require_canonical ADMITS
DA's _derived_dir(), from wt-rev, PM_DATA_ROOT **SET** to /home/yuqing/ctaNew
   -> /home/yuqing/ctaNew-wt-rev/data/pm_5min/derived
      09-04 book visible there: False
```

**DE's resolver does not depend on the environment variable — it finds the canonical ledger
on its own. DA's `_derived_dir` does not read the variable either: it resolves relative to
its own file, so setting `PM_DATA_ROOT` correctly does not save it.** So "correctness rests
on `PM_DATA_ROOT` being set" is not the right statement of the risk. The right one is: **two
seats resolve one root by two rules, and one of them silently returns a partial ledger** —
the same shape as the supersession link and the landing-record digest, on the root itself.

**My view: a hazard, and the sharpest kind — absence reads as a pass.** A resolver pointed at
a worktree finds the ~135 tracked files and misses every uncommitted artifact, so it reports a
smaller, plausible ledger rather than an error (rule 11). It is also newly load-bearing: the
read gate counts sealed receipts *at a root*, and a root that is a shell reports days
`MISSING` that exist. Two fixes exist and both are already in the programme — R-553's symlink,
and `require_canonical`, which DE's day path already calls. DA 80 is driving the resolvers;
this is the case to drive.

---

# B. DE 92 — the 26-second refusal, and the GO gate

## B.1 The defect, and it is the third of its shape in three rounds

The old check drove the guard by **calling `run_day` and reading its message** — and that
message only appears when the calling process does **not** hold the lock. Every seat's
battery runs without the lock, so it passed everywhere; a real day holds the lock by
construction, so the needle missed and the battery failed. **A check that could only pass
where it did not matter.** DE reproduced it one line apart (PASS without the lock, FAIL with
it) and names it as the same shape as the seam that cost 85 minutes: *an instrument whose
verdict comes from ambient state rather than from the property.*

## B.2 The repair — all three cells driven with an injected observation

```
real day, lock NOT held -> REFUSES  ("a REAL day is heavy by construction … and the lock does not hold")
real day, lock HELD     -> ADMITS,  checked = True
FIXTURE,  lock NOT held -> ADMITS,  checked = False   (a fixture is not checked)
```

The observation is a **parameter** with the ambient reading as its default — inject for the
battery, observe in production. That is the right shape and it is the shape the old check
lacked.

## B.3 The stronger question: does ANY battery verdict still depend on the ambient lock?

I could not run the battery while holding the **real** lock — pid 3282334 holds it and other
seats need it — and my scratch-copy route hit two walls worth recording (a copy without a
`data` symlink resolves its own shell, §A.6's hazard reproduced by accident; and a non-git
copy fails `producing_code_is_locatable`). So I answered the question **by census over the
whole battery**, which is stronger than one lock-held run because it covers every check
rather than one:

* every other `wrapper_observed(...)` in the battery takes an **injected scratch lock path**
  (`_plock`, `_sl`, `_gone`);
* the **two** that read the ambient lock — lines 5905 and 6244 — assert **shape, not state**:
  that the observation carries `heavy_run_lock_held` / `lock_fds` / `cgroup_leaf`, that the
  flag is a `bool`, that `flock_modes_on_the_inode` is not None. The lock's actual value is
  **reported in the message and asserted nowhere.**

**No battery verdict depends on whether the process holds the lock.**

## B.4 The GO question, in one line

**Is there anything left that refuses a wrapped real run, at the start or the end, that a
standalone battery cannot see?** — **No**: the battery's verdicts are now
ambient-independent (§B.3), the day path's start-side guards are driven with injected
observations (§B.2), and the end-side refusals are the two already written down — the
frozen-worktree discipline (rule 22) and the ruled peak-stage predicate.

> ### **GO for the 09-03 re-run.**

---

# C. What I did NOT establish

I did not run the real day, and **the real book has still never been through the day path end
to end** — the two artifacts that exist for 09-03 are a REFUSED record and a 26-second
refusal. I did not run the battery while holding the real lock (§B.3 says what I did
instead). §A.4's explanation of the lower peak is a **reading of the stage table, not a
drive**. I did not open the 09-04 book — every population number above is read from the
receipt and checked for internal consistency and arithmetic, which is a different claim from
recomputing it from the pickle; DA's book tier is the instrument for that and it holds the
lock now. And §A.2's third shape is a census of BE's receipts, not a demonstration that
anything resolves them today — nothing does.

**Context: ≈40%** — 400k tokens of the 1M window by my own count; this build's pane status
line carries no `% context used` field, so it is my count, not the pane's.
