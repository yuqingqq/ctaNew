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

**Point estimate**: `POINT_ESTIMATE_RUN_HAS_NO_TEST_STATISTIC` ·
`POINT_ESTIMATE_RESULT_CONTRACT_VIOLATION`.

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

Measured cost: **~2 minutes a day** against ~2 h 45 m for a full run.

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
