# Review — DA round 11 (DA10-R1..R5 + R-434 §2, on the round-10 batch)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `e292439`** ("HELD: DA round 11 — DA10-R1..R5 + R-434 §2 on top of the
round 10 batch"), parent `3a89e6c`, detached in `~/ctaNew-wt-rev`.
**Request of record:** `REQUEST_DA_ROUND_11_2026-09-02.md`.
**Composed 2026-09-02T14:03:58Z.** One filing, per R-377.

**Constraints observed and checked.** Read-only under `data/`. A full `ls -la
--time-style=full-iso` of `data/pm_5min/derived/` (184 entries) taken before the first command
and after the last — **identical**. Every verifier / mask / preflight run carried
`DA_MIDNIGHT_OUTDIR` + `DA_MIDNIGHT_LOG` pointed at scratch and, where a complete tape was
needed, `PM_DATA_ROOT` pointed at the write-safe scratch data root built for round 10.
`DA_MIDNIGHT_MODE` never set; the real launcher never run; no timer, service or unit touched
(`systemctl --user cat` / `list-timers` only). `COORDINATION.md` never written.

Scope confirmed: **six files, +194/−10**, exactly as listed.

---

## Verdict

### RELEASE for `e292439` as the content of Q-DA-209. All five DA10 findings close, and the two I file back are one level down from the fixes themselves.

The skip machinery is the real thing: `ran + skipped == 247` is asserted in every layout, the
six LOG-ECHO checks name themselves and their absent input, and both falsifiers behave — the
old silent gate now passes where it was correct and **fails exactly where the defect was**.
What I file back: the same skip idiom in `pm_tape_density` **counts a SKIP as a pass**, and the
new gate added so "a launcher break cannot sit uninvoked" runs the launcher that was already
working.

---

## 1. DA10-R1 — the arithmetic, measured, and DA's pane is wrong by three

| layout | ran | skipped | total | rc |
|---|---|---|---|---|
| the worktree (branch 2 — tape symlinked, `derived/` tracked-only) | **241** | 6 | **247** | 0 |
| a complete scratch data root | **247** | 0 | **247** | 0 |
| the same root **minus only the log** | **241** | 6 | **247** | 0 |

**The reconciliation asked for:** DA's pane reports "238 ran + 6 named SKIPs". The `ran` figure
is **241**, not 238 — the pane carried round 10's number across the three checks this round
adds, and 238 + 6 = 244 is the round-10 total, not this one. 241 + 6 = 247, and the module's own
printed line says so. The other half of the pane ("244+3 ran, 0 skips") is right, written as a
sum. **The code is self-consistent and asserted; only the pane's transcription is off** — worth
correcting in the Q-DA-209 row rather than a finding against the artifact.

Each skip names the check and the path: `SKIP LOG-ECHO PROVENANCE (20260901): absent input
…/derived/.da_midnight_verify.log` ×6 (rule 4: a status, not a silent drop).

**Both falsifiers, driven:**

| mutant | result |
|---|---|
| one check deleted, complete root | **rc 1** — *"246 ran + 0 skipped = 246, expected 247. A check that neither ran nor named itself as a skip has VANISHED"* |
| the silent `if _lg_p.exists():` gate restored, root **with** the log | **rc 0, 247** — it was correct there |
| the same, root **without** the log | **rc 1** — *"241 ran + 0 skipped = 241, expected 247"* |

That is the discrimination DA claims: the pre-fix code passes where it was right and fails
exactly where it was wrong. A falsifier that fires on the general shape would have been weaker.

**Is rc 0 on a skip the right reading?** For the selftest, yes — and the reason is checkable
rather than a matter of taste: the summary line carries the accounting, and `v5_deploy_gates`
prints that line as the gate's summary. From my own gates run:

```
PASS  DA day-verifier selftest  (11s)  da_forward_day_verify selftests: 247 checks passed (0 skipped; ran+skipped=247)
```

so a skipping layout is visible to a gate reader without changing rc. What is **not** expressible
today is a policy — "a governed run must have zero skips" — because the runner decides on rc
alone. **Recommendation, not a finding:** an opt-in strict mode (`--selftest --require-no-skips`
→ rc 1 if any skip) that the canonical-tree gate uses, leaving rehearsals at rc 0. That keeps
rule 4's status and gives the governed context a decidable answer.

## 2. DA10-R2 — `roots` in every emission, and the branch is one of the three

`pm_tape_density.data_root_provenance()` returns `code_root`, `data_root`, `data_root_branch`,
plus `branches`, `resolver_predicate` and a `predicate_caveat` naming exactly the trap I found in
round 10 (a tree can satisfy the predicate and still lack `derived/` or `mm_hf/`).

| launch | branch recorded |
|---|---|
| the worktree carrying `raw/` | `2_code_tree_carries_the_tape` |
| `PM_DATA_ROOT` set | `1_env_PM_DATA_ROOT` |
| a tree with no `data/` at all | `3_canonical` |

| emission | carries `roots`? |
|---|---|
| the **verdict** envelope (produced into scratch) | **yes** — all six keys |
| the preflight, **admitted** shape | **yes** |
| the preflight, **REFUSED / rc 3** | **yes** — branch and both roots present |
| the **mask** producer block | **yes** — `data_root_branch` beside the pair |

One note for whoever reads tonight's artifact: the canonical tree *carries* the tape, so a
production run records **branch 2**, not `3_canonical`. `3_canonical` appears only from a tree
without a tape. That is correct and slightly counter-intuitive; the `branches` string in the
emission is what saves the reader.

## 3. DA10-R3 — closed, and the gate roster measured

`sys.path.insert(0, str(Path(__file__).resolve().parent))` at `:80`; **both** launchers give
`da_hf_pm_alignment selftests: 53 checks passed`. The module is now a gate: the list is 21 → 22
tuples, **15** of which pass `--selftest` (the coordinator's 14 → 15).

**The gate roster, and DA's "21 of 22":** I ran `v5_deploy_gates.py` at this tip (worktree, scratch
data root, rehearsal pair) and got **ALL 22 GATES PASS** — including `v4 behaviour
(git-extracted) … 10/10 pass` and `v4_1 boundary gate … 176 checks passed`. **I could not
reproduce any failure.** The gate whose outcome depends on the tree's layout is the v41 pair, not
the v4-behaviour one (see item 9), so I would not describe the difference as a load flake without
seeing which gate DA's run actually reported. Either way it is **not a regression from this
batch**: every gate that touches the six changed files passes here.

And a finding on the gate itself — **DA11-R2**.

## 4. DA10-R4 — `_is_tracked` asks the tree the path lives in

`:1840`, `git -C str(Path(path).resolve().parent)`. Driven from the worktree (data root =
worktree):

| input | answer |
|---|---|
| a tracked file by its canonical path | **True** |
| **the same tracked file under the worktree** | **True** (it was **False** at `3a89e6c`) |
| an untracked file (`.da_midnight_verify.log`) | **False** |

The answer no longer depends on where the process was launched. Closed.

## 5. DA10-R5 — the control asserts the property

The mask suite now passes **30** from a non-canonical parent (the worktree, branch 2) where it
exited **rc 1** at `3a89e6c`, and **30** under a scratch `PM_DATA_ROOT` (branch 1, which the child
inherits). The final form asserts `code_root` is the throwaway worktree, `data_root != code_root`,
and the branch is **not** `2_code_tree_carries_the_tape` — a tree with no tape cannot take branch
2 — with the comment recording that two earlier forms encoded the environment.

*Observation, not a finding:* the branch conjunct is a `!=`, so a **missing** `data_root_branch`
key satisfies it too (`_prod.get(...) != "2_…"` is True for `None`). Asserting membership in
`("1_env_PM_DATA_ROOT", "3_canonical")` costs nothing and cannot be satisfied by absence — worth
folding in whenever this line is next touched.

## 6. R-434 §2 — the mirror is complete

`ruled` now carries **all four** of R-424 §7: `R-411(i)`, `R-411(ii)`, `R-408(3)`, `R-408(2)`,
**each citing R-424** (checked, not assumed). R-408(2)'s text: *"the Phase-2 winner — RULED at
R-424 §3: the composed candidate DOES NOT ADVANCE, Q1_arrival is the surviving component of
record, and there is **NO race admission**."* `still_open` = `freeze_disposition` alone. The
`ESCALATION_` positive control still returns a producer-written key **verbatim**.

No new number: `G_MIN_COMPLEMENT_WINDOWS 144`, `WINDOWS_PER_DAY 288`,
`P1_GOVERNING_DENOMINATOR per_unmasked_hour`, `V2_TRAILING_DAYS // 2 == 3` still computed, and
`governs` **False / True / True** for 09-02 / 09-03 / 09-04.

## 7. `pm_tape_density` 7 → 9 — the branch is asserted, and the empty root is named

The resolved branch is asserted by name (*"the data root resolved by a NAMED branch
(1_env_PM_DATA_ROOT)"*), and an empty root prints
`SKIP resolved-root-carries-days: absent input …/raw (branch 1_env_PM_DATA_ROOT) -- an EMPTY data
root is a status, not a clean pass` instead of passing 7/7 on the synthetic tape. The carried
item from round 10 is closed at the behaviour.

The accounting around it is not — **DA11-R1**.

## 8. Nothing moved for tonight

| | |
|---|---|
| the shared tree's six files | **byte-identical to `b75c9fe`** (blob hashes) — the v1 path is untouched |
| `derived/` | listing **identical** before and after, 184 entries, full-iso mtimes |
| `git worktree list` | identical across the mask suite (which adds and removes one); no stale entries |
| installed unit | `ExecStart=/home/yuqing/ctaNew/live/pm_research/da_midnight_verify.sh`, `Environment=DA_MIDNIGHT_MODE=production`; timer next elapse **Thu 2026-09-03 00:06:00 UTC** |
| `~/ctaNew-wt-da` | clean at `e292439`; my own worktree clean at `e292439` after every mutant |

## 9. CO-8 — the classification confirmed, with a cleaner demonstration

**Confirmed: a resolver question on the coordinator's surface, not a DA regression.**
`v41_boundary_preflight` takes `REPO = P.REPO` from `v5_boundary_preflight:42`, which is
`Path(__file__).resolve().parents[2]` — its own code tree, unaffected by this batch's resolver
(I established the same in round 10, where the request's worry that v41 had inherited the DATA
root did not reproduce). From a bare worktree that tree holds no untracked ledgers, so the two
gates fail; nothing in DA's six files touches it.

**The measurement, sharper than the symlink experiment:** in **this** worktree, which already
carries per-file `data/pm_5min/*` symlinks, `v41_boundary_preflight` passes **176 checks with
nothing added** — `REPO/data/pm_5min/collector_provenance.jsonl` resolves. That isolates the
cause to the data links alone, with no edit to the tree under measurement.

Is DA's symlinking faithful? **As a diagnostic, yes** — it reproduces the passing condition and
identifies the input. Two caveats worth recording with it: it changes the tree being measured
(the failure returns when the links are removed, as DA saw), and it demonstrates the cause
without touching the defect, which lives in `v5_boundary_preflight`'s root and is the
coordinator's to fix. Nothing here needs to move before tonight.

---

## Findings

### DA11-R1 — LOW-MEDIUM — `pm_tape_density` counts a SKIP as a pass, and the gate line cannot tell

`live/pm_research/pm_tape_density.py:443`: after printing the SKIP the code runs
`checks.append(True)`, and the summary is `f"…{len(checks)} checks passed"`. So:

```
complete root : pm_tape_density selftests: 9 checks passed
EMPTY root    : pm_tape_density selftests: 9 checks passed     (+ one SKIP line above it)
```

The two summaries are byte-identical, and the summary is exactly what `v5_deploy_gates` captures
as the gate's one line. From my own gates run, the contrast inside a single invocation:

```
PASS  DA day-verifier selftest  …  247 checks passed (0 skipped; ran+skipped=247)
PASS  tape density              …  pm_tape_density selftests: 9 checks passed
```

The verifier tells a reader what did not run; the tape module does not, and its own SKIP message
says *"an EMPTY data root is a status, not a clean pass"* while the next line makes it one. This
is DA10-R1's defect one module over, in the same batch that closed it — a check that did not run
counted among those that did, with rc 0 and an unchanged summary.

**Closure:** the shape the verifier already has — a `skipped` list, `len(checks)` and
`len(skipped)` printed separately, and `ran + skipped == EXPECTED_CHECKS` asserted so the total
cannot drift. Roughly the six lines of `_selftests()`'s helper, reused.

### DA11-R2 — LOW-MEDIUM — the new gate runs the launcher that was already working

DA10-R3 was an `-m`-only break: at `3a89e6c`, `python3 live/pm_research/da_hf_pm_alignment.py
--selftest` passed **53** while `python3 -m live.pm_research.da_hf_pm_alignment --selftest` died
with `ModuleNotFoundError`. The fix adds the module to `v5_deploy_gates.py` so that "a launcher
break cannot sit uninvoked (R-370)" — but the gate is written as
`[PY, str(HERE / "da_hf_pm_alignment.py"), "--selftest"]`, i.e. **by path**.

Measured across the whole roster at this tip: **22 gates, 21 launched by path, exactly one
(`tier1 normalisation`) with `-m`.** So had this gate existed at `3a89e6c` it would have passed,
and the break that motivated it would still have sat uninvoked. The gate closes the "nobody runs
this file" half of R-370 and not the half DA10-R3 was about.

**Closure:** run each `--selftest` gate under **both** launchers — the two-launcher discipline
every filing in this programme applies by hand — or, minimally, add the `-m` spelling for the
modules that support it. The runner already treats a missing script as a failure, so a module
that cannot be imported as a package member would fail loudly.

---

## Executed evidence

At `e292439`, 2026-09-02T13:56–14:03Z, `~/ctaNew-wt-rev`:

| check | result |
|---|---|
| scope | six files, **+194/−10** |
| three layouts | **241+6**, **247+0**, **241+6** — all totalling **247**, rc 0 each |
| DA's pane arithmetic | `ran` is **241**, not 238; 238 was round 10's figure |
| falsifier: one check deleted | rc 1, *"246 ran + 0 skipped … expected 247"* |
| falsifier: the silent gate restored | **green with the log, red without it** |
| gate summary line | carries `(0 skipped; ran+skipped=247)` |
| `roots` | present in the verdict, both preflight shapes (incl. **rc 3 REFUSED**) and the mask |
| branch values | `1_env_PM_DATA_ROOT` / `2_code_tree_carries_the_tape` / `3_canonical`, one per launch |
| `da_hf_pm_alignment` | **53 / 53** under both launchers; `sys.path.insert` at `:80` |
| gate roster | **22 gates, ALL PASS**; 15 with `--selftest`; **21 by path, 1 with `-m`** — DA11-R2 |
| `_is_tracked` | True / **True** / False (canonical, worktree leg, untracked) |
| mask suite | **30** from the worktree (rc 1 at `3a89e6c`) and **30** under scratch `PM_DATA_ROOT` |
| `open_decisions` | four ruled, all citing R-424; R-408(2) says "NO race admission"; `still_open` = freeze_disposition; `ESCALATION_` verbatim |
| constants | 144 / 288 / `per_unmasked_hour`; `V2_TRAILING_DAYS//2 == 3`; governs F/T/T |
| `pm_tape_density` | **9** checks; branch asserted; empty root prints a named SKIP — **counted as a pass**, DA11-R1 |
| shared tree | the six files **byte-identical to `b75c9fe`** |
| `derived/` | identical before and after; `git worktree list` identical across the mask suite; no stale entries |
| unit / timer | untouched; next elapse Thu 2026-09-03 00:06:00 UTC |
| worktrees | `~/ctaNew-wt-da` and `~/ctaNew-wt-rev` both clean at `e292439` |

---

## Disposition

- **RELEASE** for `e292439` as the content of Q-DA-209. DA10-R1, R2, R3, R4, R5 and R-434 §2 all
  close at the artifact; the skip machinery is asserted in every layout and its falsifiers
  discriminate rather than merely fail. **No hold** — DA11-R1 and DA11-R2 are both one edit and
  neither touches tonight.
- **FILED:** **DA11-R1** (a SKIP counted as a pass in `pm_tape_density`, invisible in the gate
  line) and **DA11-R2** (the new gate exercises only the path launcher, not the `-m` one that
  broke).
- **CORRECTION for the Q-DA-209 row, not a finding:** the pane's "238 ran + 6 skips" should read
  **241 ran + 6 skipped = 247**.
- **Item 9:** classification confirmed — a resolver question on the coordinator's surface (CO-8),
  not a DA regression; and in a worktree that already carries the data links `v41` passes 176
  untouched, which demonstrates the cause without editing the tree under measurement.
- **Recommendation on rc (item 1):** keep rc 0 for a skip; add an opt-in strict mode so a
  governed run can *require* zero skips, since the gates runner decides on rc alone.
