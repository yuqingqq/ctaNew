# DE_PROCEDURE.md — the DE seat's procedure file

Written 2026-09-08 at a seat reset, by the DE seat that ran GO E1–E4, GO R1/R2,
GO #8 and the first placement-latency point estimates. Until now DE was the only
seat with no procedure file: everything below lived in one context and would have
died with it. **Maintain this file.** When you learn something the hard way, add
it here in the same round — not "later".

Read this with `SEAT_PROTOCOL.md` (rules 20–22 in full) and `COORDINATOR_RUNBOOK.md`
§7d. This file is the DE-specific half those two do not carry.

---

## 1. THE LAUNCH FORM

Every heavy run is a **transient systemd SERVICE**, never a `--scope`:

```
systemd-run --user --unit=<NAME> --slice=research.slice \
  -p MemoryMax=8G -p CPUQuota=100% -p RemainAfterExit=yes \
  --setenv=PM_DATA_ROOT=/home/yuqing/ctaNew \
  --working-directory=<WORKTREE> \
  -- flock -n -E 75 /home/yuqing/ctaNew/data/.heavy_run.lock \
     /home/yuqing/pricer-sol/venv/bin/python3 <script> <args>
```

* **`--scope` is forbidden.** A scope registers processes the CALLER forks, so the
  run dies when the harness stops the launching shell's background task. The 09-03
  re-run was killed at 35 minutes that way with nothing written (R-628).
* **`-p RemainAfterExit=yes`** keeps the unit LOADED after exit so the five fields
  and the `InvocationID` can still be read. A succeeded transient unit is otherwise
  collected and `systemctl show` returns DEFAULTS (`LoadState=not-found`,
  status 0, success) — which reads as a clean run whether or not one happened.
* **`flock -n -E 75` INSIDE the unit.** 75 is the declared conflict code, distinct
  from the payload's own exit 1. A held lock exits 75 and the payload never starts:
  read `ExecMainStatus`, never assume the payload ran.
* **`PM_DATA_ROOT` is the REPO root** (`/home/yuqing/ctaNew`), not the data dir.
* **Capture the five fields + `InvocationID` WHILE THE UNIT IS LOADED**:
  `LoadState`, `ActiveState`, `SubState`, `ExecMainStatus`, `Result` + `InvocationID`.
  A reading with an empty `InvocationID` is **VOID** — not "success". Copy them into
  the capture record at the moment they are read, then stop the unit so the name is
  free (a loaded name, exited or failed, makes the next `systemd-run` FAIL).

### The sampler is a SEPARATE transient unit
The leaf peak (`VmHWM`) must be sampled **while the run is alive** — it cannot be
recovered afterwards. Run the sampler as its own tiny transient unit
(`-p MemoryMax=128M -p CPUQuota=10%`), never as a shell background job: the unit
survives the pane, and it did on every run where the in-pane watcher failed.
Select the leaf with `pgrep -f "<script> <distinctive arg>"` **filtered on
`comm == python3`** — otherwise you sample the `flock` wrapper and report its
2 MB RSS as the run's peak.

### The in-pane background watcher is DEAD (R-821)
A `while … sleep; done` background Bash that waits for the unit to leave `running`
**did not fire** for R1, R2 or GO #8. Do not rely on it and do not tell the
coordinator a run is being watched. **Poll the unit directly** when prompted, or
wait for the coordinator to tell you it exited. The coordinator's own monitor is
armed independently.

---

## 2. THE WORKTREE MAP (rule 22)

| worktree | what it is for |
|---|---|
| `/home/yuqing/ctaNew` | the SHARED tree. Landings only: `add -f` + `commit -- <paths>` + `push`. Never `checkout`, `reset`, `stash`, `clean`, `rebase` here. |
| `/home/yuqing/ctaNew-wt-de` | **THE RUN WORKTREE. FROZEN.** Runs execute from it and its HEAD does not move until the coordinator's GO names the refresh. As of this writing it is at `5020f96` — the bytes REVIEW 103 cleared, which GO #8 ran. |
| `/home/yuqing/ctaNew-wt-de2` | the working tree: edits, batteries, landings, and the first point-estimate runs. A result run here requires a clean committed HEAD and freezes the worktree for its duration, exactly like wt-de. Refresh it to the tip BEFORE editing. |
| `/home/yuqing/ctaNew-wt-e3` | the composition worktree: build and push composition heads here. |
| `/home/yuqing/ctaNew-wt-rr` | the re-run worktree, created at a composition head so wt-de can stay frozen for a different GO. |

* **NEVER touch wt-de while a run executes from it** (rule 22: a heavy run's code is
  frozen until its receipt lands). A landing to the runner mid-run would name code
  that did not execute.
* Refresh **only** with `bash scripts/wt_refresh.sh <wt> [ref]`. It drops the `data/`
  symlink, checks out, re-sweeps skip-worktree, re-links and verifies.
* **READ ITS WHOLE OUTPUT.** It REFUSES (exit 3) when a modified file differs from
  the ref, and the refusal prints BEFORE the status lines. Reading only `tail -1`
  is how I landed on a stale base and overwrote the user's own commit `849bef2`.
* A seat worktree's status is clean when it shows exactly `?? data` (the ledger
  symlink, R-625).
* **CORRECTION (DE 151): the point-estimate runs execute from wt-de2, not wt-de.**
  §2 said wt-de2 was for edits and batteries only; the artifacts say otherwise —
  every point-estimate run's `launch_form_at_runtime.cgroup_leaf` is `dePE…` and
  its `before_work.residency` paths are under `/home/yuqing/ctaNew-wt-de2`. That
  path does not trip `BE_CASCADE_DIFFERS` because its `be_module_citation.scope`
  is **THE ENTRY POINT ONLY** — no `be_cascade` in the params it resolves.
* **AND A SECOND CORRECTION, to §3: DE 131's temporary phase4 copy is for the DAY
  PATH, never the battery.** Pinning phase4 to v19's `ee4034c1…` in wt-de2 makes
  the battery FAIL at the cascade known-bad — `refuses(lambda: verify_be_module(P),
  … "BE_CASCADE_DIFFERS")` needs the module to be OFF the pin, and the positive
  control beside it uses `_with_current_cascade(P)`, which re-reads the digest from
  disk and passes either way. **The battery is green in wt-de2's NORMAL state**
  (phase4 at `52e76689…`). Copy the pinned bytes in only to drive `--synthetic-day`
  or a real day, and restore them after.
* **CHECK THE SHARED TREE'S STATUS BEFORE STARTING A ROUND (DE 152).** A dirty
  shared tree means someone else is mid-round in it. Two implementations of DE 151
  were written in parallel — one here, one in the shared tree — because neither
  side looked. `git -C /home/yuqing/ctaNew status --short` is the whole check.

---

## 3. THE COMPOSITION DISCIPLINE

A composition is a commit on `origin/mm-research-e3-composition` that a GO runs
from. **That branch ref is the ONLY thing that survives** — heads never pushed to
it (`5a34e722`, `9233b34`) are unreachable garbage.

Heads to date, each = the previous plus exactly what is named:

| head | added |
|---|---|
| `6c3a121` | `fe76d83` + DE 132/133's two files (the early-read ledger anchor) |
| `5020f96` | + DE 134's runner (day-path ledger anchor; the post-emit census fix) |
| `a54dcc2` | + the settlement estimator, inline-valued null, rule-11 guard, verified winner, `trades_cash_flow_cents` rename, the supersession rule, DA 130's reader |
| `850c166` | + REVIEW 104B §6's four, R-810's evidence-identity gate, R-811's swept L, ledger schema v3 |
| `b48af21` | + the USER's `849bef2`, with one cell made conditional (coordinator ruling, later WITHDRAWN) |
| `33e8584` | same, with REVIEW 106 §4a's mutated-params cell instead of the conditional — **current** |

### phase4 must stay at v19's pin
`de_phase4_diag_runner.py` is `be_cascade` module 4 in params v19, pinned at
**`ee4034c15c274982…`**. The launch's own preflight (`import_be_cascade` →
`verify_be_module`) verifies the cascade against the **frozen** params, so a
composition carrying a moved phase4 **aborts the run after taking the lock**
(`BE_CASCADE_DIFFERS`). mm-research's phase4 has been off that pin since DE 127;
that is why compositions are built from a pinned base and NOT from the tip.

**To run the runner's battery in wt-de2** (where phase4 is off the pin): copy
`git show 6c3a121:live/pm_research/de_phase4_diag_runner.py` in, run, restore.
This is DE 131's method and **REV 99 §A1 approved it** — "the bytes verified
against are the bytes the runs execute".

### Verify a composition BEFORE pushing, in that worktree
1. parents/merge-base: the base is the head you intended;
2. `git diff --name-only <base> HEAD` — the **exact** path list, nothing else;
3. every replaced path byte-equal to the landed blob it came from;
4. **ten cascade pins, 0 mismatched** (walk v19's `be_cascade.modules`);
5. batteries: runner, early read, decision ledger — all green **there**;
6. `--synthetic-day FIXTURE-DAY-1 --output <tmp>` → **rc 0** (drive it AT the
   composition; the tip refuses on the stale pin by design);
7. `rehearse` for every day the GO will run, with its supersession target;
8. rule 11: 09-03..06 admit as `DESIGN_DATA`, 09-07 refuses.

---

## 4. THE RUNNER'S STAGE MAP

Measured on two real days (the receipts carry per-stage `elapsed_s`):

| stage | 09-03 | 09-04 |
|---|---:|---:|
| S0_verify | 0.2 s | 0.3 s |
| S0b_battery | 56.2 s | 56.2 s |
| S1_load | 6.0 s | 6.9 s |
| S2_population | 0.5 s | 0.6 s |
| S3_baseline | 12.2 s | 12.8 s |
| **S4_null** | **9,467.4 s** | **10,433.9 s** |
| S5_seal | 0.0 s | 0.0 s |

**S4_null is 99.2 % of a run** and buys ONLY Z and p. Everything else is ~75 s.
That single fact is why the point-estimate mode exists and why the parallel null
is worth building.

---

## 5. REFUSAL NAMES DE OWNS

**Cascade / provenance**: `BE_CASCADE_DIFFERS` · `POST_EMIT_CENSUS_HAS_NO_BAR`.

**Ledger**: `DECISION_LEDGER_HAS_NO_ANCHOR` · `_ABSENT` · `_DIGEST_MISMATCH` ·
`_NO_INVENTORY_FIELDS` · `_NO_SIGN_CONVENTION` · `_SETTLEMENT_SCALARS_ABSENT_FOR_ARM` ·
`_SETTLEMENT_VALUE_ABSENT_FOR_ARM` · `_SETTLEMENT_NULL_NOT_REDERIVABLE` ·
`LEDGER_SCHEMA_UNKNOWN` · `DECISION_LEDGER_FULL_RUN_HAS_NO_NULL_DRAWS` ·
`DECISION_LEDGER_POINT_ESTIMATE_HAS_NULL_DRAWS` ·
`DECISION_LEDGER_MIXED_RUN_MODES` ·
`DECISION_LEDGER_NULL_VALUES_FIELD_ABSENT`.

**Valuation**: `ABSOLUTES_DO_NOT_RECONCILE` · `SETTLEMENT_LEGS_DO_NOT_RECONCILE` ·
`SETTLEMENT_EXCESS_DOES_NOT_RECONCILE` · `SETTLEMENT_NULL_DEGENERATE` (a STATUS,
never an abort — a degenerate null must not kill a day that already computed D_E0).

**Rule 11**: `SETTLEMENT_DAY_NOT_ADMISSIBLE` ·
`SETTLEMENT_DECLARATION_NAMES_A_CONSUMED_DAY`.

**Winner / verification**: `SETTLEMENT_WINNER_SOURCE_ABSENT` · `_MISSING_FOR_SLUG` ·
`_AMBIGUOUS` · `_RECORDS_DISAGREE` · `SETTLEMENT_WINNER_DISAGREES_WITH_CHAINLINK` ·
`SETTLEMENT_CHAINLINK_UNAVAILABLE` · `SETTLEMENT_CHAINLINK_NO_FILES_READ` ·
`SETTLEMENT_VERIFICATION_EMPTY` · `_SLUG_SET_MISMATCH` · `_PROVENANCE_INCOMPLETE` ·
`_STATUS_UNKNOWN` · `_SOURCE_MALFORMED` · `SETTLEMENT_WINNERS_NOT_VERIFIED` ·
`SETTLEMENT_CONVENTION_NOT_THE_PINNED_ONE`.

**Placement latency**: `SETTLEMENT_BOOK_DECLARES_NO_PLACEMENT_LATENCY` ·
`SETTLEMENT_BOOK_PLACEMENT_LATENCY_AMBIGUOUS` ·
`SETTLEMENT_BOOK_RECEIPT_NOT_PARSED` ·
`SETTLEMENT_PLACEMENT_LATENCY_DISAGREES_IN_THE_DOCUMENT` ·
`SETTLEMENT_PLACEMENT_LATENCY_ABSENT_FROM_THE_DOCUMENT`.

**Early read**: `EARLY_READ_NO_SEALED_RECEIPT` · `_RECEIPT_NOT_THE_PAIR` ·
`_ALREADY_EMITTED` · `_BAR_CARRIES_ONLY_A_PREFIX` · `_BAR_DIGEST_MALFORMED` ·
`_BAR_PREFIX_DISAGREES_WITH_ITS_OWN_DIGEST` · `_DAY_NOT_IN_THE_BAR` ·
`_WROTE_NO_DECISION_LEDGER` · `_SUPERSEDES_TARGET_ABSENT` · `_SUPERSEDES_DIGEST_MISMATCH` ·
`_SUPERSEDES_NOT_THE_HEAD` · `_SUPERSEDES_DIFFERENT_PLACEMENT_LATENCY`.

**Score routing (DE 164, gate item 4)**: `SCORE_BELONGS_TO_ANOTHER_GENERATION`
(the assembled score at a row's `(slug, side, t)` was built for a DIFFERENT
generation -- the abutting-boundary case DA 143 drove) ·
`SCORE_ROW_NAMES_NO_GENERATION` (the row carries no `gen`, so the comparison
would be SKIPPED, and a skipped check reads as a passed one). On the policy
side the disagreement is not a refusal but a COUNTED status --
`crossings_for_another_generation` / `reduce_crossings_for_another_generation`
in `harmful_stateful_policy` -- because a misrouted crossing is an exclusion,
not a broken run.

**Ledger write (DE 164, REV 115/116)**:
`DECISION_LEDGER_NULL_PARALLEL_LIST_LENGTH` ·
`DECISION_LEDGER_NULL_DRAW_PAIRING_UNRECORDED` ·
`DECISION_LEDGER_NULL_DRAW_PAIRING_BROKEN` ·
`DECISION_LEDGER_DIGEST_NOT_SUPPLIED`.

**Point estimate** (the family lives in `de_point_estimate_day.py`):
`POINT_ESTIMATE_RUN_HAS_NO_TEST_STATISTIC` · `POINT_ESTIMATE_RESULT_CONTRACT_VIOLATION` ·
`POINT_ESTIMATE_DRIVER_CHANGED_DURING_RUN` · `POINT_ESTIMATE_DRIVER_NOT_COMMITTED` ·
`POINT_ESTIMATE_OUTPUT_EXISTS` · `POINT_ESTIMATE_OUTPUT_NOT_DIRECTORY` ·
`POINT_ESTIMATE_PLACEMENT_LATENCY_ABSENT` · `POINT_ESTIMATE_PRIOR_UNREADABLE` ·
`POINT_ESTIMATE_PRIOR_IDENTITY_MISMATCH` · `POINT_ESTIMATE_PRIOR_LATENCY_MISMATCH` ·
`POINT_ESTIMATE_SUPERSEDES_NOT_A_POINT_ESTIMATE_RUN` ·
`POINT_ESTIMATE_SUPERSEDES_TARGET_ABSENT` · `POINT_ESTIMATE_SUPERSEDES_DIGEST_MISMATCH` ·
`POINT_ESTIMATE_SUPERSEDES_DIFFERENT_PLACEMENT_LATENCY`.

**Note the two SUPERSEDES families and do not confuse them.** The EARLY READ family
above (`EARLY_READ_SUPERSEDES_*`) validates a target the CALLER names; the point-estimate
family DISCOVERS its priors from the `(day, L_place_ms)` glob, so `_TARGET_ABSENT` and
`_DIGEST_MISMATCH` there are about a discovered pair going stale between resolution and
the write, not about a bad argument.

**AND THE LATENCY IS GUARDED TWICE, ON PURPOSE — DE 153, and the DE 152 merge lost one
of them by treating the two as one name.** `POINT_ESTIMATE_PRIOR_LATENCY_MISMATCH` says
**A FILE IS MISLABELLED**: it sits in the `L<tag>ms` family and declares another L.
`POINT_ESTIMATE_SUPERSEDES_DIFFERENT_PLACEMENT_LATENCY` says **THE RELATIONSHIP IS
WRONG**: a run at one latency is about to publish another latency's artifact as its
PREDECESSOR (R-811, R-828 §(2) — siblings, not successors). **The second is the one that
protects the published link, and it was measured missing:** discovery never globbed a
foreign-L file and a mislabelled one refused, but the path that FORMS the link compared
no latency at all and ADMITTED an L = 0 target for an L = 250 run. The guarantee had come
to rest on a FILENAME. It now rests on the target's own declared L, re-read from disk at
the write. **The lesson, which is the general one: a guarantee that survives only because
of a naming convention has not survived — name it and assert it.**

### RED BY DESIGN in the shared tree
mm-research's `de_phase4_diag_runner.py` is off v19's cascade pin, so **the runner's
battery and the day path refuse `BE_CASCADE_DIFFERS` there**. This has been true
since DE 127 and is EXPECTED, not a regression. It is green where phase4 is pinned
(any composition head, wt-de, wt-rr) and in wt-de2 under DE 131's temporary-copy
method. A red battery in the shared tree is never by itself a reason to stop —
but check it is THIS red and not another (REV 101 found a real red hiding behind it:
the by-design red aborted the battery at check 7, 177 checks before a genuine
failure that nobody could see).

---

## 6. THE LEDGER WRITER'S SHAPE

Row kinds, in write order: `HEADER` · `ARM_SCALARS` · `NULL_DRAW` · `FILL` ·
`DECISION` · `SETTLEMENT_SCALARS` · `SETTLEMENT_SLUG`.

* `HEADER` carries `schema_version`, `day`, the ruling, `inventory_fields_from_BE_96`,
  `settlement_rows_present` (computed from what is about to be written, never
  promised), `settlement_row_kinds`, `moments_ddof`, `buy_side` (the sign convention
  travels with the file — a reader that guessed "B" would value every fill backwards
  the day that constant changed) and `arms`.
* `ARM_SCALARS` carries the day's scalars and DE 131's `absolute` block.
* **`NULL_DRAW` carries `value` (the 5-second draw) AND `settle_value` (the
  settlement draw).** Before DE 143 only the 5-second draws were persisted, so every
  settlement Z and p was READ and not re-derivable — and a good-faith re-derivation
  from the 5-second rows is **2.2×–5.4× MORE EXTREME**, i.e. the error ran toward
  OVERSTATING significance. `null_values` and `null_settle_values` are paired by
  index; in point-estimate mode **`null_values` is `None`, not `[]`** (an empty list
  would read as "500 draws that all came out zero"), so every consumer must write
  `a["null_values"] or []`.
* `SETTLEMENT_SLUG` rows are per slug, per book (ARM and BASELINE), per arm — a day
  of 288 slugs and two arms writes 1,152 of them.

**`SCHEMA_VERSION = 4`, `KNOWN_SCHEMA_VERSIONS = (1, 2, 3, 4)`.** A reader that does
not know a file's version **REFUSES** it (`LEDGER_SCHEMA_UNKNOWN`) rather than
skipping the rows it cannot name and returning a complete-looking result — DA 131
measured that a conforming v2 reader silently dropped the settlement rows and
reported the 5-second D_E0 as the day's number. v2 = no settlement rows; v3 =
settlement rows; v4 = settlement rows AND the settlement draws.

**`moments_ddof: 0`.** The receipts' `null_sd` is the POPULATION sd
(`statistics.pstdev`). Measured on 09-04: `5339.1612146584675` (ddof 0) against
`5344.508397986252` (ddof 1). A re-deriver assuming ddof 1 gets a third number, so
the file says which it is and the falsifier pins it.

---

## 7. THE POINT-ESTIMATE CONTRACT

`run_day(..., point_estimate=True)` **skips S4 entirely** — not a shorter null, no
null. The draw count is never lowered (R-174); it is not taken at all.

The builder is tracked at `live/pm_research/de_point_estimate_day.py`; do not copy
it to scratch. From the clean, frozen run worktree, launch it through §1's service
and lock form:

```
systemd-run --user --unit=<NAME> --slice=research.slice \
  -p MemoryMax=8G -p CPUQuota=100% -p RemainAfterExit=yes \
  --setenv=PM_DATA_ROOT=/home/yuqing/ctaNew \
  --working-directory=<CLEAN-FROZEN-WORKTREE> \
  -- flock -n -E 75 /home/yuqing/ctaNew/data/.heavy_run.lock \
     /home/yuqing/pricer-sol/venv/bin/python3 \
     live/pm_research/de_point_estimate_day.py \
     <YYYY-MM-DD> <BOOK.pkl> \
     --output-dir /home/yuqing/ctaNew/data/pm_5min/derived
```

The driver takes `placement_latency` from the completed `day_run`, uses that same
object at the top level, checks every serialized `L_place_ms` before an atomic
publish, and names the prior `(day, L_place_ms)` artifact by path and recomputed
digest. If a family has an earlier abandoned sibling, `also_supersedes` names it
too, so the correction leaves one head. Contradictory priors remain unedited and
are superseded in-band.

* Every statistic field — `Z`, `p_location`, `null_mean`, `null_sd`,
  `null_draws_summary.n`, and the same five in `economic_settlement` — carries the
  string **`NULL_NOT_DRAWN_POINT_ESTIMATE_RUN`**. **Never a null, never a zero.**
* `test_statistic_from(receipt, field)` **REFUSES**
  `POINT_ESTIMATE_RUN_HAS_NO_TEST_STATISTIC`. DA's reader must refuse on the same
  name; it is filed in the register for DA to adopt, not assumed agreed.
* The receipt carries `run_mode: "POINT_ESTIMATE"` (a full run carries `"FULL"`).
  A full run at the same `(day, L)` is a **DIFFERENT artifact**, not a successor:
  neither supersedes the other.
* The arm-day status is **`OK_POINT_ESTIMATE`**, not `OK` — see §8.
* The estimates are the SAME numbers as a full run: same replays, same arithmetic,
  only the control absent. The falsifier proves it digit for digit.

Measured cost: **~2 minutes a day** against ~2 h 45 m for a full run (09-03 125.8 s, 09-04 129.4 s peak 3.110 GB, 09-05 124.7 s, 09-06 125.3 s peak 2.615 GB, 09-05 at L = 0 118.6 s peak 2.527 GB, 09-07 114.1 s).

**ONE PROCESS PER DAY.** A batch that ran six days in one process FAILED on day two: `_peak_rss_mb()` is `ru_maxrss`, non-decreasing for the life of the PROCESS, so day two's battery inherited day one's high-water and the peak-stage cell measured history rather than a property. Each day is its own transient unit, with `flock` the DIRECT parent of python3 — the run reads `/proc/<ppid>/exe` and refuses if its parent is not flock, so no shell may sit between them (use `-p StandardOutput=append:` for a log, never a `bash -c` wrapper).

---

## 8. THE DEFECT CLASS THAT COST THREE PATCHES IN ONE NIGHT

**A status compared by EQUALITY where membership was meant.**

Introducing `OK_POINT_ESTIMATE` broke three sites that all read
`r.get("status") == "OK"`, each failing silently or late:

1. the settlement branch in `run_day` — an `OK_POINT_ESTIMATE` arm-day emitted **no
   `economic_settlement` block at all**, so the first 09-05 L=250 artifact was
   written with the ruled endpoint missing;
2. the `elif` that fills `settlement_not_valued_reason` — same gate, same miss;
3. the ledger's `settlement` conditional — the settlement rows were dropped.

All three now test membership: `in ("OK", "OK_POINT_ESTIMATE")`.

**This is the THIRD instance of this shape in two days.** The others: `arm_day`
returning `"OK"` / `"DEGENERATE_ARM_DAY_REFUSED_TOO_FEW_DECISIONS"`, and the
seal/census reading `a.get("sealed")` and `_economic_keys_in`. **A new DE greps for
this FIRST when adding any run mode or status:**

```
grep -n '\["status"\] == "\|\.get("status") == "\|status == "OK"' live/pm_research/*.py
```

and checks each hit against every status value that can now occur. Related sites
that compare strings and must be swept the same way: `seal()`'s `_all_sealed`,
`_economic_keys_in`, DA's readers, and any `settlement_rows_status` consumer.

**A sibling of the same class**, also paid for tonight: matching a key by SUBSTRING
where a leaf NAME was meant. The placement-latency walk matched any key containing
`placement_latency`, so BE's L=250 book — which carries both
`placement_latency_ms: 250.0` and `TRANCHE_BEFORE_PLACEMENT_LATENCY: 23947` (a
COUNT) — read as AMBIGUOUS and refused. Match the leaf names you mean.

**And a third**: `builder_receipt_for` derived the receipt from the DAY, so an
`…__L250ms.pkl` book resolved the **L = 0** receipt — it would have verified a
250 ms book against the 0 ms book's digest and stamped `L_place_ms = 0.0` on a
250 ms run. The book's OWN receipt name is tried first now.

**A FOURTH, found by the USER at the artifacts (DE 151), and it is the same family
one level down: a value computed from the WRONG OBJECT, returning a
legitimate-looking default.** `run_day` passed `builder_receipt_for(...)` — which
returns a **`Path`** — into `placement_latency_from_the_book(builder_receipt: dict)`.
A Path is neither a dict nor a list, so the walk visited nothing, `found` stayed
empty, and the function returned its documented `L_place_ms: 0.0` with the source
string `"THE BOOK'S BUILDER RECEIPT DECLARES NONE"`. **Every L = 250 artifact
therefore said 250.0 at the top level and 0.0 under `day_run`, naming the same book
and the same digest.** No number moved — the latency is applied by BE at BOOK BUILD
time and `_plat` is computed at the emit, after `S5_seal`, with exactly two
references in the whole runner — and the six re-emits proved it: every economic
field reproduced byte for byte while the nested value moved 0.0 → 250.0.

**The three lessons, because none of them is "read more carefully":**
1. **A reader handed an input it cannot parse must REFUSE, not answer from its
   default** (`SETTLEMENT_BOOK_RECEIPT_NOT_PARSED`).
2. **A document must assert its own agreement before its bytes exist** —
   `assert_one_placement_latency` at the emit, on the serialised form AND on the
   bytes read back; **absence refuses too**, and so do `bool`, NaN and negatives,
   because `True` is an `int` in Python and would otherwise read as 1.0 ms.
3. **The type is the check.** Two call sites, one passing a `Path` and one a
   `dict`, disagreed for rounds and no battery could see it, because both returned
   a well-formed block.

**And the same class in a second module, same round:** `de_decision_ledger.recompute()`
called `statistics.fmean(vals)` as its FIRST arithmetic, so a POINT-ESTIMATE
ledger raised `StatisticsError` and **nothing** in the file could be read. **No null
is a STATUS, never a crash and never a zero** — and the ledger now DECLARES its
`run_mode` in the header rather than letting a reader infer it from an empty
container, with write-time refusals for a full run with no draws, a point estimate
carrying draws, mixed modes across arms, and a missing `null_values` field.

---

## 9. LANDING, AND THE GUARD THAT MUST REFUSE

Rule 21: copy the worktree's exact bytes into the shared tree and commit **by
pathspec in the same act**. Before the copy, check the paths have not moved:

```
git -C /home/yuqing/ctaNew diff --name-only <worktree BASE>..HEAD -- <paths>
```

**This check must EXIT NON-ZERO, not print.** Twice in one session a guard I wrote
printed a warning and an `&&` chain sailed past it — once **overwriting the user's
own commit `849bef2`**, dropping `chainlink_stream_provenance_check` from the runner
and carrying in unreviewed work. The repair is always FORWARD (a superseding commit
that restores the lost bytes and states the mistake), never a revert and never a
history rewrite.

Note the baseline that matters: compare against **the worktree's base**, and be aware
that a path legitimately moves when YOU landed it from the same worktree earlier —
in that case refresh the worktree and rebuild rather than forcing past the guard.

A landing script with this shape lived only in scratch during the reset; **write it
into `scripts/` so it survives** (open item).

---

## 10. SURFACES DE RELIES ON (other seats')

* **BE** — `be_cancel_axis_null`: `load`, `replay`, `flagged_stream`, `arm_stream`,
  `draw_flags`, `draw_null`, `MIN_DRAWS`. `harmful_stateful_policy`: `SIDES[0]` is
  the BUY side (the ledger's sign convention), and `_SlugReplay` is constructed
  fresh per slug. The daybook **and its own builder receipt**
  (`be_daybook_receipt_<day>_btc[__L250ms].json`), which since BE 101 declares
  `placement_latency_ms`.
* **DA** — `da_early_read_verify.resolve_early_read_head` (the early-read family's
  head, resolved by the `{path, sha256}` PAIR with a 64-hex digest, never by stamp)
  and `head_standing`. **DA prints every economic value**; DE quotes none.
* **Shared** — `declaration_chain.resolve_head` (resolve heads through it, NEVER
  `ls | tail`: `params_v9` sorts last lexically). `exp_m6_settlement.load_streams` /
  `read_at` — **never edit that module**: its digest is recorded as `reader_module`
  in landed artifacts, so a fork invalidates their provenance. Wrap it instead.

---

## 11. OPEN CONDITIONS DE OWES (as of 2026-09-08 03:5xZ)

1. **09-05 at L = 0** in point-estimate mode as the paired control for the 250 ms
   run, and the remaining days at 250 as BE 104's books land. **Take the lock
   BETWEEN BE's builds, never during; never launch while a `be*` unit runs.**
2. **DE 144, the parallel null.** Correctness condition is absolute: draw all flag
   sets **serially** first (preserving the RNG stream exactly — `draw_flags` is the
   only RNG consumer and BE's module says so), then replay in parallel, and
   reproduce 09-04's landed D_E0, D_E_settle, null mean, null sd, Z and p **exactly,
   every digit, both arms**. My earlier implementation was removed in the `9316a81`
   repair and exists nowhere — rebuild from scratch. Fork AFTER the book is loaded
   (copy-on-write); measure real per-worker memory before choosing a worker count;
   `imap` preserves order, which is what keeps the reduction identical.
3. **DE 142** — `stream_provenance.files` does not reach the artifact, so the user's
   `chainlink_stream_provenance_check` cannot be re-run from an artifact alone. One
   additive line, plus a falsifier that re-runs the check FROM THE EMITTED ARTIFACT.
4. **DE 149's owed measurement** — per-worker memory for a per-slug replay on a real
   book (independence itself is established; see the register).
5. The **register row** for DE 146–150 is not filed.
6. The day-run receipt's filename carries the legacy `_SEALED_` glob while every
   field in it says UNSEALED. It is a naming lie on an unsealed receipt.
7. **`p003_de_point_estimate_day_20260905_L250ms__20260908T034458Z.json`** (untracked,
   57,752 B) is the ABANDONED first 09-05 attempt with no settlement block. Never
   read it; delete or leave untracked.
8. Write the landing guard (§9) into `scripts/`.

**CLOSED at DE 164:** item 8 -- the landing guard is `scripts/de_land.sh`.
It EXITS NON-ZERO (3 = dirty outside the pathspec, 4 = a path moved in the
shared tree since the worktree's base, 5 = copy/commit failed, 6 = STRANDED)
and it runs BEFORE the copy. Both guards are driven with a positive control;
drive them again if you change it, because a guard that only prints is how
`849bef2` was overwritten. It skips the copy when source and destination are
the same inode (a seat worktree's `data` is a symlink to the shared tree's).

**AND FOUR THINGS DE 164 LEARNED THE HARD WAY:**
1. **A FLAKY CELL IS A REAL DEFECT UNTIL PROVEN OTHERWISE.** The DE48 cell
   failed one run in four and looked like scheduling noise. It was not: the
   heartbeat daemon thread could append AFTER `TERMINAL`, so the log's last
   line was a heartbeat and a reader would read a DEAD RUN AS ALIVE -- the
   exact failure DE48 exists to prevent. Reproduced DETERMINISTICALLY by
   calling `stage()` after `terminal()` on the HEAD bytes at the unit; no
   load needed once you know where to look. **Load-dependence is a SYMPTOM
   of a race, and the race is in the thing being measured more often than in
   the measurement.** The cell now WAITS FOR THE CONDITION (never a fixed
   sleep) and the log SEALS at `TERMINAL`.
2. **A BATTERY RED IN THIS TREE IS NOT NECESSARILY YOURS.** Establish the
   baseline by running the module at HEAD's bytes in a scratch copy
   (`git show <tip>:<path> > scratch/...`) BEFORE spending a round on it.
   Two of the three reds I met were pre-existing: `be_cancel_axis_null` and
   `da_elementwise*` on the stale `de_section81_cache_12.pkl`
   (`ASSEMBLY_PREDATES_CAUSAL_SCORING`, BE's to rebuild), and
   **`de_lane4_real_parity` GATE 3, red since BE 96 (`c707eb8`,
   2026-09-07T08:23:42Z)** -- the machine's `FILL_CHARGED` gained
   `inventory_before/after/unit` and the INDEPENDENT builder never did, so
   the LANE4 parity anchor has not compared anything for two days. Do NOT
   "fix" it by copying the machine's fields into the independent builder:
   that is R-235's do-not-harmonize hazard, and the builder is written from
   the DECLARED semantics on purpose.
3. **THE DESIGN EMITTER NEEDS `PM_DATA_ROOT=/home/yuqing/ctaNew`.** Without
   it the emission's frozen `data_root` / `worktree_data_shell_trap` blocks
   resolve to the seat worktree and `correction_census` refuses the version
   as a FROZEN BLOCK CHANGE. The battery goes green the moment the env var
   is set; nothing about the code is wrong.
4. **THE PIN CRANK IS FOUR EDITS, NOT ONE.** params v<N+1> (both sites: the
   `be_cascade.modules` digests AND `be_module`), the design module's
   `VERSION` **and** `PARAMS_REL` **and** its `V<N>_DECLARATION` chain entry
   (the battery refuses if the chain is one short), and
   `de_multiday_gate1_runner.PARAMS_REL` -- which is the one that bites: the
   runner resolves params by that literal, so leaving it behind makes the
   run refuse `BE_CASCADE_DIFFERS` against the OLD version's digests while
   the new version sits on disk agreeing with everything.

**CLOSED at DE 151/152:** the driver is no longer scratch-only
(`live/pm_research/de_point_estimate_day.py`, and it now REFUSES to emit unless its
own bytes are the bytes HEAD holds — `POINT_ESTIMATE_DRIVER_NOT_COMMITTED`); the
placement-latency contradiction and the ledger's point-estimate crash are fixed with
falsifiers in both directions; all six point-estimate artifacts are re-emitted
superseding the old by pair; and the point-estimate family now RESOLVES its own
priors (`also_supersedes` absorbs an abandoned sibling), which was the head-resolution
gap this file had only flagged. **STILL OPEN:** items 2 (DE 144's parallel null),
3 (DE 142's `stream_provenance.files`), 4 (DE 149's per-worker memory), 6 (the
`_SEALED_` glob on an unsealed receipt's filename) and 8. **AND ONE NEW:** `D_E0_role`
and `settlement_rows_status` — the fields that say "diagnostic, not ruled endpoint" —
are read by no module but `de_decision_ledger.py`.

---

## 12. HABITS THAT PAID FOR THEMSELVES

* **Verify at the artifact the claim names** — not a proxy, not memory, not a
  report. Every number in a DE report should have been read from a file in that
  round.
* **Red-first.** A known-bad establishes its own baseline INSIDE the cell and
  asserts a delta; a battery whose verdict changes when its cells are reordered is
  measuring history, not a property (REV 83 §5 — and one of my cells failed exactly
  that way when a neighbouring cell started allocating).
* **A cell must own its fixture.** Three of my cells in three rounds inherited a
  fixture another cell left behind; one broke a supersession chain that DA's
  resolver caught by name.
* **Compute predicates, never print conclusions.** A printed reason that contradicts
  the computed verdict beside it has now happened twice.
* **Say what is NOT established.** Every report should carry its own limits: what
  was measured, what was inferred, and what nobody has checked.

---

## 13. WHAT 2026-09-09/10 COST, AND THE SIX THINGS THAT WOULD HAVE SAVED IT

Written by DE 179–193, the seat that ran the first two corrected point
estimates. Every item below is something that went wrong ONCE and will go
wrong again for the next seat if it is not here.

### 13.1 `RemainAfterExit=yes` MAKES `is-active` A LIE
A transient unit with `RemainAfterExit=yes` stays **`ActiveState=active`
after it EXITS SUCCESSFULLY** (`SubState=exited`). So

    until ! systemctl --user is-active --quiet $U; do sleep 5; done

**never terminates on a successful run.** It cost me a peak-RSS sampler that
spun instead of sampling — the 09-04 peak is `null` in its record for that
reason — and it made me report a finished run as "past ten minutes, checking
whether it hung" when it had already succeeded. **Wait on SubState:**

    until [ "$(systemctl --user show $U -p SubState --value)" != "running" ]; do sleep 2; done

A FAILED unit goes `ActiveState=failed`, which is why this trap only bites on
the runs that worked.

### 13.2 A RUN RECORD MUST NAME ONLY ARTIFACTS NEWER THAN ITSELF
The record took `ls -t <pattern> | head -1`. On a run that exited **75** and
never started, that named an artifact from **2026-09-08** — a record claiming
an artifact its own run did not produce. Use `find -newer <the record>`, and
write the LAUNCH row first so it is the timestamp to compare against. The two
bad rows were corrected IN BAND (a `CORRECTION` row), never edited.

### 13.3 THE SCORING-PATH WAIVER: WHAT IT IS AND HOW TO USE IT
Books go stale the moment `de_multiday_gate1_runner.py` moves, because that
module is in every book receipt's REFUSABLE set — reached at **exactly one
attribute, `ruled_day_set`, which reads `PARAMS_REL`**. Since DE 179 the
refusal carries its own evidence and a waiver exists (USER ruling, DE 180):

    --waive-scoring-path        # on de_multiday_gate1_runner.py AND on
                                # de_point_estimate_day.py

**ASKING IS NOT GRANTING.** `de_scoring_path_delta.waiver_available` decides on
five computed conditions. **Measure it BEFORE you launch** (~4 s, no lock):
build the `differ` list from the receipt's REFUSABLE set against disk, call
`delta(...)`, and check `INTERSECTION == 0` and every `byte_identity_on_the_path`
`identical`. If the intersection is NOT empty the answer is a REBUILD, not a
flag.

### 13.4 `de_land.sh` REFUSES WHEN *YOU* MOVED THE PATH — AND THE TIP MOVES UNDER YOU
Exit 4 ("these paths moved in the shared tree since <base>") fires when you
landed from the same worktree earlier in the round. §9 already says refresh
and rebuild rather than force. **What §9 did not say:** after refreshing,
**re-verify your base at the NEW tip**, because it can move again between two
of your own commands — mine moved `67d4eab → 9d7c6c0` mid-sequence. Diff your
saved file against the tip's version of each path and confirm the only delta
is yours before copying anything in. Exit 6 (STRANDED) means committed, not
pushed: **report it, never rebase here.**

### 13.5 NEVER EDIT SOURCE BY INDEX SLICE
`s[:start] + new + s[end:]` between two comment anchors **deleted five working
falsifier cells** because the region held more than I meant. CLAUDE.md already
forbids it; this is the DE instance. The battery's **check-count assertion is
what caught it** (21 against 25) — which is the reason a battery asserts its
own size instead of running whatever it finds. Anchor to exact strings and
`assert old in s` before writing.

### 13.6 A HEAVY RUN CAN BE OOM-KILLED AS A BYSTANDER AT 852 MB
BE's 09-03 rebuild died `Result=oom-kill` while its own unit peaked at
**852 MB against an 11.87 GB `MemoryMax`**. It was **`systemd-oomd`**, acting on
**`research.slice` pressure** (`ManagedOOMMemoryPressure=kill`, limit 3.44 GB;
slice `MemoryHigh` 12.88 GB / `Max` 15.03 GB; `memory.events` already showed
`oom_kill 15`). The slice holds **GBs of page cache** from 300 MB pickles with
nothing running. **So a unit cap does not protect you and a small footprint does
not either.** Before launching a heavy run, check the SLICE, not just the unit —
and know the scheduled consumer:

**`pm-evaluation-pipeline.service` — `OnCalendar=*-*-* 03,09,15,21:50 UTC`,
holds ~10 GB for about an hour, `MemoryMax=16 GiB` inside a 14 GB slice.**
A cap larger than its own slice cannot throttle itself; it can only get a
bystander killed. **Do not have a heavy run in flight across :50 on those
hours.**

### 13.7 THE POINT-ESTIMATE LAUNCH FORM THAT WORKS
Kept here so the next seat does not rebuild it. One day per unit, `flock` the
DIRECT parent of `python3`, a LAUNCH row before and an EXIT row after, peak
sampled in-line against SubState (13.1), and the artifact resolved by
`-newer` (13.2). The working script lived at
`scratchpad/de179/run_day.sh`; its shape is: LAUNCH row → `systemd-run
--unit=<TAG> --slice=research.slice -p MemoryMax=8G -p RemainAfterExit=yes
-p StandardOutput=append:<log> --working-directory=<FROZEN WT> -- flock -n -E 75
<lock> python3 live/pm_research/de_point_estimate_day.py <DAY> <BOOK>
--output-dir <derived> --waive-scoring-path` → sample → EXIT row → stop +
reset-failed so the unit name is free.

### 13.8 STANDING CONSTRAINTS A NEW DE MUST NOT DISCOVER THE HARD WAY
* **Rule 34a (USER, `abd4b07`): WRITING DOES NOT CONSUME A DAY, READING DOES.**
  The Tier-2 timer writes `data/pm_5min/tier2/**/day=2026-09-08/` and later on
  its own schedule. **Their existence is not permission.** Do not read,
  summarise, aggregate or quote them. Build and score **09-03..09-07 only**.
* **Rule 37: the coordinator holds the lock trigger.** When told to stand by,
  do not poll and do not eat `Exit 75`s — a tight retry loop is contention.
  (DE 184 asked for a retry to be ARMED; DE 192 withdrew that. Read the
  latest dispatch, not the habit.)
* **09-03's EV21 book has NO complement leg** — 23,765 tranches were counted
  and DISCARDED at build time — so its reconciliation reports
  `RECONCILIATION_UNAVAILABLE_BOOK_CARRIES_NO_SPLIT` and the latency question
  stays unanswered for that day until it is rebuilt with the split.

---

## 14. THE STEP-2 VERDICT HANDOFF — executable cold by a cleared seat

Written by DE 217 at 97 % context, while `deSettle212` was still drawing.
**The run does not need me. The VERDICT needs this section**, because the
numbers alone do not say what they mean.

### 14.1 THE ONE LINE THAT MUST SURVIVE

> **"If it comes back `NO_SETTLEMENT_SKILL_OVER_MATCHED_RANDOM`, that is
> the FIRST LINE of the report, not a caveat at the end."**

That outcome is not a disappointment to be softened. It is plan v2 step 3
firing, and step 3 is **the branch that protects the validation set**: it
stops us spending untouched days on these frozen arms. Report it first,
plainly, and without hedging.

### 14.2 WHAT TO RUN

```
cd /home/yuqing/ctaNew-wt-arms/live/pm_research      # the ARMS PIN
PM_DATA_ROOT=/home/yuqing/ctaNew \
  /home/yuqing/pricer-sol/venv/bin/python3 de_settlement_control_aggregate.py
```
It reads `data/pm_5min/derived/settle/` (results + checkpoints) and
`data/pm_5min/derived/` (book receipts). It REFUSES rather than reports on
a bad input; every refusal below is a real answer, not an obstacle.

### 14.3 THE PROVENANCE THAT MAKES THE VERDICT FINAL RATHER THAN PROVISIONAL

| fact | ref |
|---|---|
| declaration, **committed BEFORE any draw** (rule 6) | `b72e329` — `de_settlement_control_declaration_v1.json` |
| matched count named **BY FIELD PATH**, not phrase | `a52baec` — v2 |
| aggregator + **GREEN falsifier** | `84b249e` |
| the arms pin the run executes from | `~/ctaNew-wt-arms`, `adbebf9` |

**The falsifier is the part that matters most**: the failure verdict is
**reachable and fires** on a synthetic set where neither arm beats matched
random, both-win passes, mixed resolves by the declared rule, and Holm is
driven step-down on `p=0.026` against `0.025` — the exact shape the
asymmetry PRIMARY produced. *A verdict function that has never produced a
failure verdict is the same class of instrument as a null that has never
fired* (REV 165).

### 14.4 THE FIVE THINGS THAT TRAVEL WITH THE NUMBER, ALWAYS

1. **PRIMARY is 09-04/05/06. 09-03 is COMPANION-ONLY and is NEVER
   PROMOTED** — fixed before the draws, because it was already excluded
   before the asymmetry draws and cannot be promoted after being seen to
   change a verdict. `aggregate` REFUSES
   `SETTLEMENT_CONTROL_COMPANION_DAY_PROMOTED_TO_PRIMARY` if it ever
   appears in PRIMARY.
2. **Each cell contributes EXACTLY its declared 500** — the emit refuses
   `..._CELL_N_DIFFERS_FROM_THE_DECLARED_N` on anything else. The floor is
   a minimum, not a licence to report a different n than declared.
3. **The four publication elements TOGETHER** (plan step 2's own clause):
   the zero-model-cancel baseline, every control distribution, coverage
   and exclusion counts **including 09-03's `n_present` 287 / `n_masked`
   40 / `mask_identity_hash` `7d82f393…`**, and settlement finality.
   Absence REFUSES.
4. **EFFECTIVE INDEPENDENT UNITS ARE FOUR (three complete), NOT EIGHT.**
   The two arms within a day share the day, the book, the reference path
   and a **bitwise identical** baseline — two policies read off ONE
   realisation. Rule 8 forbids an interval below five complete days.
5. **THE MATCHING BIAS RUNS TOWARD FINDING THE ARM SKILFUL.**
   Distinct-generation matching UNDER-matches raw exposure (max 23 cancels
   on one reference generation), so the control is the weaker de-leverer
   and **any arm advantage is an UPPER BOUND** on the part that is not
   exposure. And per REV 162: **never** say "invariant to de-levering" or
   "exactly zero" — the control removes the FIRST-ORDER effect only.

Also: these are **CONSUMED days (rule 11)**, so a pass is DEVELOPMENT
evidence and **never** validation.

### 14.5 TWO CLEARANCES ALREADY ESTABLISHED — do not re-litigate them

* **The `MIN_DRAWS` collision cannot have touched this run.** Seven modules
  define `MIN_DRAWS = 200`; step 2 reads NONE of them. It has its own local
  `FLOOR = 200`, takes `n_draws` as a REQUIRED keyword (500, passed
  explicitly), and calls only `build_pool_from_rows`, `draw_one` and
  `flags_for` — none of which reference it. `draw_many` does, and step 2
  does not call it, **because it drives its own loop to checkpoint every
  draw**. Neither the draw count, the floor, nor the refusal threshold
  comes from a colliding symbol.
* **None of the nine off-path guards in `de_multiday_gate1_runner.py` would
  change what step 2 reports**, and the reason is structural: **step 2 never
  calls `run_day`**. Its whole runner surface is four calls —
  `load_params`, `import_be_cascade`, `winner_source`,
  `settlement_legs_by_slug`. It uses the runner as a LIBRARY. And
  `verify_run_inputs` **PASSES at the arms pin** (driven: `cited_not_copied`
  true, models and thetas verified): its refusal is a property of the TIP,
  where two cascade modules differ against v29 — the same two the pin
  restores.

### 14.6 THE ONE GAP I DID NOT CLOSE — hand it on

Step 2 calls `R.winner_source(required_slugs=slugs)` **without**
`verification=`, so the Chainlink cross-check does not run inside step 2 and
**DE 204's settlement-source fallback disclosure is not computed per cell.**
No number moves — the venue winners are the same object the day artifacts
valued — but **09-03's unadjudicable window (`btc-updown-5m-1788469500`,
drift bound 2.859 USD against margin 1.154 USD) is not disclosed by the
per-cell results.** The fix belongs in the AGGREGATOR's publication block,
in `wt-de2`, never in the running worktree. Until it lands, say so when
quoting any 09-03 companion figure.

### 14.7 THE POST-VERDICT BRANCH IS DECIDED IN ADVANCE — do not wait for anyone

Authorised by the coordinator on the USER's delegation (DE 218), **before**
the draws landed. Whichever fires, execute it; neither is a fresh decision.

**IF `NO_SETTLEMENT_SKILL_OVER_MATCHED_RANDOM` — plan step 3 fires, and it
fires CHEAPLY.** Report the failure as the FIRST LINE. Publish the four
elements together. **Then STOP**: no freeze, no forward test, no redesign.
Do **not** spend untouched days on these frozen arms; `QR_SKEW_ONLY`
remains the reference. A redesign uses CONSUMED data only and starts a NEW
freeze and a NEW validation clock — a fresh decision, not a continuation of
this one. **This outcome protects five validation days and is not a
disappointment.**

**IF EITHER ARM BEATS MATCHED RANDOM — REPORT THE PASS, PUBLISH EVERYTHING,
AND DO NOT FREEZE.** Plan step 4 commits builder, scorer, thresholds,
settlement convention, null construction and decision rule, and **freezing
is the one step that cannot be taken back**. REV 166 established that **at
least 53 verdict-bearing guards have only ever been seen returning one
value — 23 of the 34 refusals sit in `de_multiday_gate1_runner` and
`de_point_estimate_day`, the two modules producing these numbers, and
TWELVE bear directly on what the freeze commits.** REV is driving them.
**A freeze resting on guards nobody has seen fire is a freeze resting on
nothing.** The freeze waits on REV's twelve — hours, not days, and it buys
the difference between a freeze and a freeze that means something.

## 15. EVERY DRIVE NAMES ITS TREE (USER/DA 272 ruling, 2026-09-11T19:0xZ)

**The shared tree `/home/yuqing/ctaNew` is DECLARED NON-EXECUTING for both
lanes.** It sat 204 commits behind `origin/mm-research` while carrying
PRE-FREEZE copies of frozen closure modules — `de_forward_evaluator`
`7c137ddc` where the freeze declares `6290bb25`,
`de_settlement_control_aggregate`, `de_revaluation_emit` — and 8 of 36
cells fail when the day-record falsifier is driven there.

**A green result from the shared tree is not evidence.** Neither is a
red one: `SUPERSESSION_CHAIN_FORKED` was reported ABSENT twice in one day
by two readers, and both greps were against that tree while the refusal
had been in `live/pm_research/de_day_record.py` on both chain refs since
18:48Z. The coordinator recorded the same wrong-tree error at 12:52Z; DA
hit it from the other side over `de_combine_day_cells`.

**So every drive states the tree it ran in, in the same line as its
result**, and a claim about code — present, absent, green, red — names
either a worktree path or a ref:

    driven from /home/yuqing/ctaNew-wt-de2   36/36
    git grep -n <name> origin/de-freeze-chain-v2 -- live/

`git grep <ref>` needs no checkout and cannot be stale, which makes it the
cheaper habit and the one to reach for first.

**A corollary that cost a round:** a falsifier run in only one tree tests
one tree's data. Two of my cells indexed a key present on a single status
branch and raised `KeyError` instead of FAILING — an exception where a
verdict belongs — and that surfaced only when the same file was driven on
an mm-research worktree where those branches take a different path.

**`origin/mm-research` is the USER's fork and the user decides what lands
there.** DE does not push to it. One DE commit (`a95bd524`, the day-record
module and the day-slice definition) was pushed at 19:05Z before this rule
existed and is the user's to keep or revert.

## 16. WHEN A TIGHTENING LANDS, RE-DRIVE WHAT TESTS IT (REVIEW 205, DE 372)

**Four instances in three rounds, and every one was invisible in the
module that changed, because the module that changed was green:**

| tightening | what broke |
|---|---|
| my gate 4 required a canonical population | the FIXTURES of gates 5 and 6, which call `build_actions` |
| DA's gate 4 tightening | DA's own gate-4 probe |
| my `action_keys_sha256` became required | DA's gate-6 probe, which constructs `ReplayInputs` |
| — and that probe crashing | an unprobed LEDGER row defaulted to SATISFIED |

The last one is the reason this is a rule and not a habit: **a tightening
made a published ledger read BETTER than before.**

**The rule:** when a tightening lands, re-drive the falsifiers of
everything that TESTS the tightened thing — not only the thing itself —
at a named tree (§15), before reporting the tightening.

**The rule as an instrument**, because a rule nobody runs is prose beside
a table: `live/pm_research/de_dependents_sweep.py <module.py> [--tree D]`
finds every file importing the module, drives each `--falsify`, and
refuses `DEPENDENT_FALSIFIER_IS_RED_AFTER_THE_TIGHTENING` naming them. A
dependent with no falsifier is REPORTED, never counted green.

**Two traps it fell into first, both worth keeping in mind:**

- **A stale `.pyc` defeated its own fixture.** `VALUE = 1` and `VALUE = 2`
  are the same SIZE, and CPython reuses cached bytecode when mtime and
  size match — so the "tightened" module behaved like the old one and the
  dependent stayed green. The sweep sets `PYTHONDONTWRITEBYTECODE`.
- **The verdict is the EXIT CODE, never a count parsed from output.** A
  module that DRIVES other modules echoes THEIR totals; taking the last
  `N/M cells pass` line read a nested module's count and attributed it to
  the wrong subject — the exact defect the sweep exists to catch,
  committed by the sweep.

**`$?` AFTER A PIPELINE IS THE LAST COMMAND'S, AND I READ IT WRONG TWICE
IN ONE DAY.** Reporting the ledger's exit code I wrote
`python … | grep …; echo "rc=$?"` and published **rc=0 in both states**.
Measured properly — redirect to a file, then read `$?` — the broken state
is **rc=1** and the fixed state **rc=0**, so a harness reading the exit
code was never lied to. The same mistake cost an inert stage-0 gate at
16:41Z (`grc=$?` after `gate | head | tr`). **Never read `$?` through a
pipe: redirect, then read.**

**And a `[FAIL]` grep matches vocabulary, not identity.** Two of DA's
PASS labels contain the literal text `"[PASS]/[FAIL] lines"` — the NAME of
a counting method — so `grep -c '\[FAIL\]'` returns 2 on a run with zero
failures. The sweep's own parser anchors at line start
(`^\s*\[(PASS|FAIL)\]`) and returns 0 on that same output, which is why
the instrument was right while my ad-hoc grep beside it was not.

**And the sweep's first real run found something a direct run hid:**
`da_fair_value_ledger.py` is green from the repo with a full environment
and **red from `/tmp` with a minimal one**, its own control naming the
cause — `A PATH KNOWN TO BE LANDED COUNTS > 0 — the control that catches a
cwd bug` → `{'de-freeze-chain-v2': 0, 'be-build-runner': 0}`. I nearly
"corrected" the sweep to match the greener answer. **When two drives
disagree, the one with the narrower environment is usually the honest
one.**

## CURRENT POSITION — DE, 2026-09-11T12:51Z (written at 97% context)

**Read R-906 (coordinator) after compaction; it carries the coordinator's view.**

### Where the work stands
Forward test, N=7, population 09-07..09-13. Day one (09-07) is VALUED and
combined under a measured waiver; 09-08 onward are not yet valued.

### The commit family (all on `origin/de-freeze-chain-v2`, most also on
`origin/be-build-runner`)
- `f309602` read-once: the run reads the settlement oracle ONCE before the
  first arm; both arms consume it. Licensed by falsifier (a), which PASSED:
  09-07 reproduces `CONDVALUE −14645.078818` / `HAZARD +4925.363903` to the
  cent, 23 s, peak 3.07 GiB, artifact
  `fwd_v2/p003_de_readonce_falsifier_a_20260907_rehearsal.json`.
- `c853e2d` hunk B (`ruled_day_set` resolves the freeze chain) + hunk C
  (build rule declared, not a literal).
- `92e4b7c` REVIEW 169: per-slug `SETTLEMENT_WINNER_MISSING_FOR_SLUG`
  restored on the read-once path; declaration resolved by identity, no glob.
- `21678a1` REVIEW 170: `DECLARATION_IDENTITY_UNPINNED` a real refusal;
  `ruled_day_set` refuses `RULED_SET_UNRESOLVED` instead of falling back.
- `eb923d3` hunk D: untouched-days guard reads DA's attestation
  (`ATTESTATION.per_day[<day>].previously_opened_for`) by declared identity.
- `3dbb107` **MISTAKE — edited `be_score_neutrality.py`**, the certificate's
  producer. Reverted in `5efb8f0`; the file is `a455191d6bceec7e` again and
  the certificate is VALID.
- `5efb8f0` `resolve_declaration_pins()` in the RUNNER (agree-or-refuse,
  `DECLARATION_PIN_CONFLICT`), replacing the edit above.
- `1b65b14` stage-0 row `comparator_is_the_certified_producer`.

### The stage-0 refusal — DIAGNOSED, not a blocker
`de_preflight_matrix` run from **wt-de2** reports for 09-08:
- `verify_run_inputs` → `REFUSED RUN: 6 pinned model file(s) do not match
  their declared digest … 'why': 'ABSENT'`
- `book_receipt` → `REFUSED BOOK_BUILT_BY_DIFFERENT_SCORING_CODE at the
  settlement-control path for 2026-09-08`

Run from **wt-deval** — the tree the chain actually executes from — only
`book_receipt` remains, and that one is EXPECTED: the matrix points at the
old `…__FWD1` 09-08 book built at `7ed5a90`, while (6) is about BE's
rebuilt `dbb11e4` book. **The model-digest refusal is wt-de2-local (the
model files are absent there).** Lesson: run the matrix from the tree the
chain runs from; a refusal read in the wrong tree is not a refusal.

### Pending, in order
1. **(c)** the growing-ledger fixture for read-once.
2. **REV's four admitting-arm cells** on the production path, each naming
   `admitted_by`: descendant+5 digests → `DESCENDANT`; exact pin → `EXACT`;
   non-descendant → refuse; one digest changed → refuse by digest name;
   declaration sha moved → refuse by name.
3. **(6)** the descendant-arm end-to-end on BE's rebuilt 09-08 book at
   `data/pm_5min/derived/rebuild_identity/` (builder_commit `dbb11e4`),
   POINT_ESTIMATE via the production chain → cells → combined
   `..._rehearsal.json` + emit; the record must say `admitted_by: DESCENDANT`.
4. Re-arm `deCHAIN0908` for 500 draws, then `deCHAIN0909`/`0910` on books.

### Standing rules (hard)
- **NEVER edit `be_score_neutrality.py`** — the neutrality certificate is
  pinned to its `producer.sha256`; editing it voids the certificate. I did
  this once at `3dbb107`.
- Stage 0 carries `COMPARATOR_ON_DISK_IS_NOT_THE_CERTIFIED_PRODUCER`; a
  worktree can hold edited bytes long after a commit is reverted — a
  fast-forward does NOT overwrite a locally modified file.
- **Instruments: new files only.** Never edit a pinned/valuation-path module
  without a rule-13 supersession and its falsifier.
- **stop → re-arm, never in place.** A script under a running unit is frozen
  bytes; list executing units before any fast-forward.
- Land from a tree no unit executes from; `wt-deval` is the chain's tree.
