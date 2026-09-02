# Review — DA round 10 (HELD batch at worktree commit `3a89e6c`)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned commit executed: `3a89e6c`** ("HELD: DA round 10 batch (RR12-1 split, R-420 log, CO-R4,
R-411 constants, v2 wiring)"), detached in `~/ctaNew-wt-rev`.
**Request of record:** `REQUEST_DA_ROUND_10_2026-09-02.md`.
**Composed 2026-09-02T13:28:09Z.** One filing, per R-377.

**Limits observed, and checked rather than asserted.** Read-only under `data/`. A full
`ls -la --time-style=full-iso` of `data/pm_5min/derived/` (184 entries) taken before the first
command and re-taken after every write-capable step — **identical every time, including
`.da_midnight_verify.log`**. Every verifier/mask/preflight run carried `DA_MIDNIGHT_OUTDIR` +
`DA_MIDNIGHT_LOG` pointed at scratch, or a scratch `PM_DATA_ROOT`, or both. `DA_MIDNIGHT_MODE`
was never set. The real `da_midnight_verify.sh` was never run: the admission matrix was driven
against a **copy** whose two canonical defaults are replaced by a standin directory (diff to the
artifact: exactly those two lines) with a stub verifier, so no configuration of that harness can
reach `derived/`. `systemctl --user cat` only. `COORDINATION.md` never written. Worktree clean
at `3a89e6c` after.

Scope confirmed: **nine files, +657/−48**, exactly as listed; no other file differs from
`b75c9fe`.

---

## Verdict

### HOLD. Nothing here bears on tonight — the installed unit runs v1 either way — and the batch should land as v1 + fixes in DA round 11.

Three of the five findings are in the batch's own headline fix. **RR12-1's split is real and I
verified it at an emitted artifact** — a mask produced from this worktree records
`carrying_commit 3a89e6c`, the worktree's HEAD, which is precisely what R-420 §2 said it could
not do. But: the six provenance checks it was written to restore **still skip silently** (I
measured 238 vs 244, rc 0 both ways, with only the log's presence differing); the one git
question left hardcoded misreports provenance from any non-canonical tree; and the batch's own
new import **breaks a suite under `python3 -m`**, the CO-2 class this programme already named.

---

## 1. RR12-1 — the split works where it was aimed, and its own test is not the property

**The three branches, resolved at the artifact:**

| context | branch | `DATA_ROOT` |
|---|---|---|
| a tree with no `data/` at all | 3 | `/home/yuqing/ctaNew` (canonical) |
| **`~/ctaNew-wt-rev`** | **2** | `/home/yuqing/ctaNew-wt-rev` |
| `PM_DATA_ROOT` = an empty temp dir | 1 | that dir, unvalidated |

**The request's premise for this worktree is out of date, and it matters.** `~/ctaNew-wt-rev`
now carries `data/pm_5min/raw` (a symlink added at 10:04Z) and is the **only** seat worktree
that does — `-da`, `-de`, `-be` have none. So the resolution here takes **branch 2**, not
branch 3, and `DATA_ROOT` becomes a tree that carries the tape while its
`data/pm_5min/derived/` holds only the ~62 tracked receipts. That third layout — tape present,
artifacts absent — is what the resolver's single test cannot see, and it is where the six checks
go back to skipping (DA10-R1). It is also why `da_cross_venue_forensics` fails here and passes
under a complete data root: the predicate is `data/pm_5min/raw` alone, while consumers of
`DATA_ROOT` also read `data/mm_hf/…`.

**Branch 1 with an empty root — both answers, depending on the entry point.** `scan_day` and
`judge_day` **REFUSE by name** (*"no raw directory for 20260901 — an absent day is not a clean
day"*), which is right. `all_days()` returns `[]` and `load_gaps()` returns an empty mapping —
a clean report over nothing — and `pm_tape_density --selftest` passes **7/7 against an empty
data root**, so the module's own suite cannot tell the difference.

**The `REPO = DATA_ROOT` rebinding is contained, and the request's worry about CO-8 does not
reproduce.** Inside the batch every git question asks `CODE_ROOT` (`da_blackout_mask:134/152/795`)
— with one exception, DA10-R4. Outside it, nothing inherits the rebound name:
`v5_boundary_preflight.py:42` defines `REPO = Path(__file__).resolve().parents[2]` **itself** and
does not import `pm_tape_density`, so `v41_boundary_preflight.py:53` (`REPO = P.REPO`) keeps a
CODE root, which is what its `git -C str(REPO) show` needs. Measured: from `~/ctaNew-wt-rev`,
`v41.REPO` = the worktree; from a scratch tree, the scratch tree — the code tree of the running
file in both. Same for `o1_boundary_preflight.py:37` and `harmful_candidate_manifest.py:34`
(own definitions). **No importer asks a code question of the data root.**

## 2. The roots are emitted by one producer of three

`code_root` / `data_root` appear in `da_blackout_mask.py:259-262` and nowhere else. Verified at
artifacts, not by grep: a **real verdict** produced into scratch (33 top-level keys, `day_token
20260901`, `all_pass true`) carries neither, and a deep walk finds only
`blackout_mask_artifact.carrying_commit` and `day_bar_v2_governing.commits`; the preflight's
REFUSED emission carries neither. See DA10-R2.

## 3. The six checks — still silent, and the count asserts nothing

They are the log-echo block at `da_forward_day_verify.py:2690-2731`: two `LOG-ECHO PROVENANCE`
+ two `LOG-ECHO KNOWN-BAD` (one pair per day) and two standalone `LOG-ECHO KNOWN-BAD`s — six,
gated on `if _lg_p.exists():`, which the batch did not change; only the path did.

| run | checks | rc |
|---|---|---|
| from `~/ctaNew-wt-rev` (data root = worktree, no log) | **238** | **0** |
| same code, `PM_DATA_ROOT` = a complete scratch data root | **244** | 0 |
| same scratch root with **only `.da_midnight_verify.log` removed** | **238** | **0** |

The third row isolates it: the six that vanish are exactly the log-gated block, and their absence
is silent in every configuration. There is **no expected-count assertion anywhere in the module**
— `:4430` prints `f"…{checks} checks passed"` and returns 0. See DA10-R1.

## 4. CO-R4 — closed

`RC_ALL_PASSED 0 / RC_PREDICATE_DID_NOT_PASS 1 / RC_REFUSED 3`. Driven in a subprocess against
an empty derived root: **rc 3**, stdout a JSON object with `classification REFUSED`,
`exit_code 3`, `day 20260902`, and a `refusal` naming the path it looked in. Distinct from rc 1.

**The rc-3 question, answered:** there is no collision in any single channel. The launcher never
invokes the preflight (grep: no `preflight` token in `da_midnight_verify.sh`), and the two run
under different units — `da-midnight-verify.service` runs the launcher, the coordinator's
transient unit runs the preflight. The only reader who meets both is a human reading rc 3 in two
places, where it means *"could not `cd`"* from one and *"REFUSED"* from the other. Both are
"nothing was verified", so the meanings do not conflict — but the preflight's rc block would be
worth one line naming the launcher's 3/5/6 so the two tables are read together. Not a finding.

## 5. R-411 constants — verbatim, and nothing new

| R-424 §4 | the code |
|---|---|
| "≥ 144 of 288 windows" | `G_MIN_COMPLEMENT_WINDOWS = 144`, `WINDOWS_PER_DAY = 288` (`:82/:90`) |
| "every good window is scored regardless" | `counts_toward_G_scope` says *"G-COUNTING ONLY — every good window is scored regardless"*; **nothing reads `counts_toward_G` as a condition** anywhere in the batch — it is emitted (`:432`) and consumed by BE, not here |
| "per UNMASKED hour … calendar-24h reported beside it" | `P1_GOVERNING_DENOMINATOR = "per_unmasked_hour"` (`:87`), both figures emitted |
| the ruling cited | `"R-424 §4 (USER, 2026-09-02), applying R-411(i)"` / `(ii)` |

No constant appears that R-424 does not contain. The 143/144 edge is a selftest check (`:764`).

## 6. The v2 wiring

`EFFECTIVE_FROM_DAY 20260903`, `FROZEN_BY_USER True`, `CONTENT_DARK_GOVERNS True`;
`governs('20260902') False`, `('20260903') True`, `('20260904') True`; `V2_TRAILING_DAYS // 2 ==
3` **computed**. The verifier's `content_liveness_v2_for('20260902')` returns `status
CONTENT_DARK, governs False, frozen_by_user True, effective_from_day 20260903` — reported,
governing nothing. `da_content_liveness_v2_check.py` is **not** among the nine files, so it is
unchanged as claimed. The two effective days live in different modules — v1's `20260902`
(`da_content_liveness_rule.py:63`, R-419) and v2's `20260903` — and no check compares one to the
other; every governance question goes through `governs(day_token)`, never a date restated.

## 7. The launcher and the unit

**The matrix, driven against the canonical-unreachable copy:**

| configuration | rc | what happened |
|---|---|---|
| pair set, no `VERIFY_BIN` (default = own dir) | 0 | ran, log written to scratch |
| pair set, `VERIFY_BIN` = a **DIFFERENT** file | **0 — ADMITTED** | ran the different binary |
| pair set, `VERIFY_BIN` = same file, other spelling | 0 | admitted as a pin (`readlink -f`) |
| `LOG` only, different binary | **5** | pair guard, **before any write** |
| `OUTDIR` only | **5** | pair guard |
| neither override, different binary | **6** | admission guard; the standin stayed empty |

**The request's expected reproduction is inverted, and the code is right.** With the rehearsal
pair set, a different binary is **admitted** — the guard's own condition is
`{ -z LOG || -z OUTDIR }`, i.e. it fires only *outside* full isolation. Given the order (pair →
admission → binary), the substitution guard is reachable **only in a named canonical run**
(cgroup identity or `DA_MIDNIGHT_MODE=production`) — which is the case its comment describes. I
did not execute that path (never production); the ordering is read from the source and the three
refusals above bound it.

**The hardcoded `cd /home/yuqing/ctaNew/live/pm_research || exit 3` (`:40`)** is inert for
resolution: `$V`, `$OUTDIR`, `$LOG` and `$tmp` are absolute, `sys.path[0]` for the verifier is
`dirname($V)`, and the verifier's own roots are `__file__`- and `DATA_ROOT`-derived. So a
worktree run executes the worktree's verifier from the canonical directory, and **the record
says so**: the log carries `verifier:`, `verifier_sha256:`, `script_tree:` and
`script_tree_commit:` (`git -C "$SELFDIR"`). A worktree run here would record
`script_tree_commit 3a89e6c…` and execute the worktree's verifier (sha `e1925ee9…`); if it
inherited the unit's pin it would execute the canonical one (sha `cd052102…`) and the log would
name that file and its sha. The divergence is visible rather than silent — that is the fix
working.

**The installed unit is not this file.** `systemctl --user cat da-midnight-verify.service` differs
from the repo's unit at this commit by exactly the **seven added lines** — the RR12-1 comment and
`Environment=DA_MIDNIGHT_VERIFY_BIN=…`. The installed unit carries
`Environment=DA_MIDNIGHT_MODE=production` and
`ExecStart=/home/yuqing/ctaNew/live/pm_research/da_midnight_verify.sh`. So tonight: the canonical
script, `VERIFY_BIN` unset → its own script-relative default → the canonical verifier, admitted
via BOTH legs. v1 throughout, as the request says. When the batch lands the unit needs a
reinstall for the pin to exist; its absence is not a hazard (the default resolves to the same
file for the canonical `ExecStart`).

## 8. `open_decisions` — three of R-424's four ruled

`ruled` = `R-411(i)`, `R-411(ii)`, `R-408(3)`, each citing R-424; `still_open` =
`freeze_disposition`. R-424 §7 records **four** ruled — `R-408(2)` (the Phase-2 winner: does not
advance, no race admission) is in neither list. The code says "Three of these" deliberately
(`:356`), so it is a scoping choice, not a slip: the three are the ones that came from this
instrument's own escalations. `still_open` matches R-424 exactly (one open). Whether the block
should mirror all four is the coordinator's call; I record the difference rather than rule it.

**Positive control, driven:** an `ESCALATION_reviewer_probe` key injected into a verdict's
`blackout_mask_and_complement.complement_quality` comes back through `open_decisions`
**verbatim** — *"a producer-written escalation, carried verbatim"*.

## 9. Counts, launchers, and nothing written

| suite | `-m` | by path |
|---|---|---|
| `da_forward_day_verify` | **244** (complete data root) / 238 (worktree) | same |
| `da_blackout_mask` | **30** (complete root) / **rc 1 FAIL** from `~/ctaNew-wt-rev` | same |
| `da_governed_verdict_preflight` | **34** | 34 |
| `da_verdict_check` | 21 | 21 |
| `da_cross_venue_forensics` | **24** (complete root) / FAIL from the worktree (no `data/mm_hf`) | same |
| `pm_tape_density` | 7 | 7 |
| `da_hf_pm_alignment` | **ModuleNotFoundError** | **53** |

DA's reported counts (verify 244, mask 30, preflight 34) reproduce **only** where the data root
resolves to a complete tree. `git worktree list` unchanged after the mask suite (which adds and
removes a temporary worktree). `derived/` identical throughout.

---

## Findings

### DA10-R1 — MEDIUM — the six provenance checks still skip silently, and no count asserts otherwise

`da_forward_day_verify.py:2690`. The batch moved the log path from `__file__` to `REPO`
(=`DATA_ROOT`) and kept `if _lg_p.exists():`. Where the data root is a tree carrying the tape but
not the untracked artifacts — `~/ctaNew-wt-rev` today, and any worktree someone links `raw/` into
— the log is absent again and the six checks vanish with **rc 0**: 238 here, 244 with the log,
238 again when only the log is removed from an otherwise complete root.

The comment at `:2686-2689` states the defect exactly ("235 in the main tree, 229 here, with
nothing saying why") and the artifact still has it, one number over. The request describes the
fix as "the count asserted over checks that RAN" — **there is no such assertion**: `:4430` prints
the count and returns 0; the module has no `EXPECTED_CHECKS` and no expected-total anywhere.

Root cause, and why it will recur: `_resolve_data_root()` tests for `data/pm_5min/raw` — "does
this tree carry the tape" — while the checks read `data/pm_5min/derived/…` and
`da_cross_venue_forensics` reads `data/mm_hf/…`. A tree can pass the test and still lack what a
consumer needs, and nothing records which branch answered.

**Closure:** an expected-total assertion (the `EXPECTED_CHECKS` idiom DE's modules use, which
fires by name — I drove it in three DE rounds), or a printed SKIP status naming the absent input
so absence is a status and not a smaller number (rule 11, "quiet and empty are different"). The
resolver should additionally record its branch in the emission (DA10-R2), so a 238 is
self-explaining.

### DA10-R2 — LOW-MEDIUM — only the mask says which roots resolved

`code_root` / `data_root` are emitted by `da_blackout_mask.py:259-262` alone. A real verdict
artifact produced at this commit carries neither (33 top-level keys; the only provenance found
anywhere in it is `blackout_mask_artifact.carrying_commit` and `day_bar_v2_governing.commits`),
and the preflight's REFUSED JSON carries neither. The verdict is the governing artifact and the
preflight is what the coordinator reads at 00:14Z; both should be able to answer "which tree did
this come from" without the reader knowing how the process was launched. **Closure:** the same
two keys in the verdict envelope and in the preflight emission.

### DA10-R3 — MEDIUM — this batch breaks `da_hf_pm_alignment` under `python3 -m`

The batch adds `import pm_tape_density as _TDROOT` to `da_hf_pm_alignment.py` (it was not there
at `b75c9fe`), and that module is the **only one of the six touched files with no
`sys.path.insert(…parent)`** — `da_blackout_mask` has three, `da_forward_day_verify` two,
`da_cross_venue_forensics` and `da_verdict_check` one each.

```
python3 -m live.pm_research.da_hf_pm_alignment --selftest  -> ModuleNotFoundError: pm_tape_density
python3 live/pm_research/da_hf_pm_alignment.py --selftest  -> 53 checks passed
```

This is CO-2 exactly — "a suite that passes only because of how it was started" — and the module
is not in `v5_deploy_gates.py`, so no gate would catch it. **Closure:** the one line the other
five already carry, and the module added to the gates list so a launcher break cannot sit
uninvoked (R-370's rule).

### DA10-R4 — LOW-MEDIUM — the one git question left hardcoded misreports provenance

`da_forward_day_verify.py:1832-1836`, `_is_tracked()`:
`subprocess.run(["git", "-C", "/home/yuqing/ctaNew", "ls-files", "--error-unmatch", str(path)])`
— a hardcoded tree, asked about a path built from `DATA_ROOT`. Measured on a genuinely tracked
file (`data/pm_5min/derived/annotations/phase2_four_arm_v2.da_caveat_field.json`):

```
_is_tracked(/home/yuqing/ctaNew/…)        -> True    (the truth)
_is_tracked(/home/yuqing/ctaNew-wt-rev/…) -> False   (same file, same repo, other worktree)
```

RR9-3(b) replaced a false sentence with "a measured fact"; from any non-canonical data root the
measured fact is wrong, in the direction that says provenance does not exist when it does.
Tonight is unaffected (the unit runs canonical), but a worktree rehearsal disagrees with
production on a provenance field, which is what rehearsals exist to catch. **Closure:** ask the
tree the path lives in — `git -C str(DATA_ROOT)` — or resolve the path's own repository.

### DA10-R5 — LOW-MEDIUM — the RR12-1 control fails from a tree that carries the tape

`da_blackout_mask.py:856-857`. The control creates a temporary worktree, produces a mask from it,
and asserts `_prod["data_root"] == str(DATA_ROOT)`. The child worktree has no `raw/`, so **its**
resolution takes branch 3 (canonical); when the parent's own root is not canonical the two differ
and the control fails. From `~/ctaNew-wt-rev` the mask suite exits **rc 1** at
*"RR12-1 CONTROL: and it NAMES both roots"*. With `PM_DATA_ROOT` set (parent and child inherit
it) it passes 30.

It fails loudly, so nothing is mis-reported — but the assertion encodes the environment it was
written in rather than the property it names, and "both launchers rc 0" holds only where the data
root is canonical. **Closure:** assert what the child's own resolution *should* give (canonical,
since the temp worktree carries no tape) or simply `child.code_root != child.data_root`.

---

## Executed evidence

At `3a89e6c`, 2026-09-02T13:10–13:28Z, `~/ctaNew-wt-rev`:

| check | result |
|---|---|
| scope | nine files, **+657/−48**; no other file differs from `b75c9fe` |
| resolution branches | 3 → canonical; **2 → the worktree** (raw/ symlinked here, the only seat worktree that has it); 1 → the env path, unvalidated |
| branch 1, empty root | `scan_day`/`judge_day` **REFUSE by name**; `all_days()` → `[]`; suite passes 7/7 |
| `REPO` rebinding | contained; `v5`/`v41`/`o1`/`harmful_candidate_manifest` define their own; **no importer inherits it** |
| RR12-1's core fix | a mask built from this worktree records `carrying_commit 3a89e6c` — the worktree's HEAD |
| **the six checks** | **238 / 244 / 238** (worktree, complete root, complete root minus the log) — rc 0 every time |
| count assertion | none exists; `:4430` prints and returns 0 |
| CO-R4 | rc **3**, JSON on stdout, `classification REFUSED`, `exit_code 3`, day named |
| rc-3 collision | none in any channel — the launcher never invokes the preflight |
| R-411 constants | 144 / 288 / `per_unmasked_hour`, rulings cited, verbatim to R-424 §4; `counts_toward_G` gates nothing |
| v2 | `governs` False/True/True for 09-02/03/04; `V2_TRAILING_DAYS//2 == 3`; v2 checker not in the batch; the two effective days never compared |
| launcher matrix | 0/0/0 admitted (incl. a DIFFERENT binary under full isolation), **5/5/6** refused, standin never written |
| installed unit | differs from the repo's by exactly the seven pin lines; carries `MODE=production` + the canonical `ExecStart` |
| `open_decisions` | three ruled (R-411(i), R-411(ii), R-408(3)) + `freeze_disposition`; `ESCALATION_` key surfaced **verbatim** |
| suites | verify 244/238, mask 30 (**rc 1 from this worktree**), preflight 34, verdict_check 21, forensics 24 (fails without `data/mm_hf`), tape 7, **hf_pm_alignment `-m` FAILS / path 53** |
| `derived/` | **identical** before and after every step, 184 entries, `.da_midnight_verify.log` mtime unmoved |
| `git worktree list` | unchanged; worktree clean at `3a89e6c` |

---

## Disposition

- **HELD.** DA10-R3 (one line) and DA10-R1 (an expected total, or a named skip) are the two I
  would want before this lands; DA10-R4 and DA10-R5 are small and local; DA10-R2 is a two-key
  addition to two emissions. **v1 + fixes in DA round 11.**
- **Tonight is unchanged and I verified why:** the installed unit carries neither the pin nor
  this code, runs `/home/yuqing/ctaNew/live/pm_research/da_midnight_verify.sh` with
  `DA_MIDNIGHT_MODE=production`, and nothing in this batch is on that path. A zero-byte
  `preflight_20260902.json` still means REFUSED.
- **What the batch got right, said plainly:** the split's core defect is fixed and provable at an
  artifact (`carrying_commit` = the running tree), the pair/admission guards refuse before any
  write, CO-R4 is closed with a real subprocess, the R-411 constants are verbatim with their
  ruling, and the v2 wiring governs nothing until 09-03.
