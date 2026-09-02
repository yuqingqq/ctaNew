# Review — DA round 12 (DA11-R1, DA11-R2, the branch conjunct, the opt-in strict flag)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned commit executed: `636a455`** ("HELD: DA round 12 — DA11-R1, DA11-R2, the branch…"),
parent `e292439`, detached in `~/ctaNew-wt-rev` (the object is reachable from the shared store;
the named fetch is unnecessary and fails).
**Request of record:** `REQUEST_DA_ROUND_12_2026-09-02.md`.
**Composed 2026-09-02T14:50:20Z.** One filing, per R-377.

**Constraints observed and checked.** Read-only under `data/`; the `derived/` listing (184
entries) is **identical** before and after. Every verifier / mask / preflight run carried
`DA_MIDNIGHT_OUTDIR` + `DA_MIDNIGHT_LOG` at scratch and, where a complete tape was needed, the
scratch `PM_DATA_ROOT` from round 10. `DA_MIDNIGHT_MODE` never set; the real launcher never run;
no timer, service or unit touched. Every rc I quote comes from a captured `returncode`, never
through a pipe.

Scope confirmed: four files, **+81/−10** (`da_blackout_mask` +8/−2, `da_forward_day_verify`
+21/−2, `pm_tape_density` +28/−5, `v5_deploy_gates` +24/−1).

---

## Verdict

### RELEASE for `636a455` as the content of Q-DA-209. Both findings close, and the closures are the shape the programme asks for — derived, not transcribed; a status, not a silence.

One finding back, and it is DA11-R2's own shape one level down: the twin derivation runs in **one
direction only**, and the single `--selftest` gate it excludes is excluded **silently** — and that
gate is the one whose other launcher is broken.

---

## 1. DA11-R1 — the skip shape, and the summary a gate reader sees

| root | summary line |
|---|---|
| complete | `pm_tape_density selftests: 9 checks passed (0 skipped; ran+skipped=9)` |
| empty | `pm_tape_density selftests: 8 checks passed (1 skipped; ran+skipped=9)` preceded by `SKIP resolved-root-carries-days: absent input …/raw (branch 1_env_PM_DATA_ROOT)` |

**Distinguishable to the gate reader — which was the whole finding.** `v5_deploy_gates` captures
the last non-empty line as a gate's summary, and the two layouts no longer print the same
sentence. Before this round both read "9 checks passed".

**The count falsifier fires:** deleting a whole `ok(...)` (AST-exact, lines 447-450) gives
`AssertionError: pm_tape_density selftest FAILED: 8 ran + 0 skipped = 8, expected 9. A check that
neither ran nor named itself as a skip has VANISHED` — the verifier's own words, in the second
module. DA11-R1 is closed in the idiom it asked for.

## 2. DA11-R2 — the twins are derived, and one gate is excluded in silence

**Measured on the roster at this tip:** 22 gates → **14 twins** → **36**, matching DA's count. The
derivation predicate (`len(argv) == 3`, `argv[0] == PY`, `--selftest` present, `argv[1]` a `.py`
under `HERE`) is a property of the gate's shape, not a second list — which is the right closure:
a transcribed roster would go stale at the next gate.

**Which `--selftest` gates get no twin — the question the item asks.** Exactly one:

| gate | argv | why no twin |
|---|---|---|
| `tier1 normalisation` | `['-m', 'live.pm_research.tier1_pipeline', '--selftest']` | `len(argv) == 4` and `argv[1]` is `-m`, not a `.py` path |

The other seven twin-less gates are the non-`--selftest` ones (`v5 heartbeat behaviour`, `v5
deadline falsifier`, `chain equivalence`, `chain differential fuzz`, `preflight mutation audit`,
`v4 behaviour (git-extracted)`, `v4_1 mutation audit`) — scripts with their own semantics, not
module selftests, and rightly untwinned.

That one exclusion is **DA12-R1**, and it is not academic: `tier1_pipeline` **fails by path**.

**The falsifier, driven.** With `sys.path.insert(0, str(Path(__file__).resolve().parent))` removed
from `da_hf_pm_alignment.py` (the `3a89e6c` shape that started all of this):

```
path gate : rc=0  da_hf_pm_alignment selftests: 53 checks passed
[-m] twin : rc=1  ModuleNotFoundError: No module named 'pm_tape_density'
```

The twin catches exactly the break the path gate cannot see. And `--falsify` still works on the
larger roster: **37 gates, 1 FAIL — the injected canary — "falsifier fired: the runner DOES report
a red gate"**.

## 3. The branch conjunct — `!=` became membership, and the missing key now fails

`da_blackout_mask.py:874`: `_prod.get("data_root_branch") in ("1_env_PM_DATA_ROOT",
"3_canonical")`. My round-11 observation was that a **missing** key satisfied the old `!=` form;
driven here, deleting `"data_root_branch": TD.DATA_ROOT_BRANCH` from the child's emission now
takes the suite **red at the RR12-1 control**. The mask suite is **30** both from the worktree
(branch 2) and under the scratch `PM_DATA_ROOT` (branch 1) — the control still encodes no
environment.

## 4. `--require-no-skips` — three cells, and what the semantics will and will not carry

| cell | rc | line |
|---|---|---|
| complete root **+ flag** | **0** | `247 checks passed (0 skipped; ran+skipped=247)` |
| no log, **no flag** | **0** | `241 checks passed (6 SKIPPED, named above; ran+skipped=247)` |
| no log **+ flag** | **1** | `REQUIRE-NO-SKIPS: 6 check(s) did not run — LOG-ECHO PROVENANCE (2026…` — all six named, and the six SKIP lines are above it |

**Nothing passes the flag:** it appears nowhere in `v5_deploy_gates.py`, `da_midnight_verify.sh`
or the unit files, and the installed unit is unchanged. Nothing moves for tonight.

**On DA's policy note — the semantics support it, with one boundary worth stating and not
crossing.** The flag is opt-in, the failure names the checks that did not run, and rc 1 is
distinguishable from every other outcome, so "the caller that promised the input demands zero
skips" is expressible exactly as DA describes. The boundary: this flag governs the **selftest**,
an instrument check, not the verdict path. Passing it inside the 00:06Z run would couple the
night's verdict rc to the instrument's inputs; the natural home is the gate runner on the
canonical tree, where the launcher log is an input the tree really does promise. That is the
coordinator's call after tonight, and I state the shape rather than wire it.

## 5. The `v4 behaviour` gate — I cannot make it fail

Reconciled as a layout fact, as the request frames it. In my two full roster runs today — the
22-gate run at `e292439` (round 11) and the 37-gate `--falsify` run here — `v4 behaviour
(git-extracted)` passed both times (`10/10 pass`), under the worktree layout with the scratch
`PM_DATA_ROOT`. I have no red to record and no way to force one, so nothing here is a finding
against the batch.

## 6. The measurement defect, and whether the batch repeats it

`v5_deploy_gates.run_one` reads `p.returncode` directly (`:115`, `return p.returncode == 0, …`)
with `capture_output=True` — no pipe, and the twins go through the same function. The
`--require-no-skips` path returns an int from `_selftests` to `main` and out through
`SystemExit`; nothing in the batch reads an exit code through a shell pipeline. The class the
programme has recorded twice does not recur here.

## 7. Nothing moved for tonight

| | |
|---|---|
| the shared tree's four files | **byte-identical to `b75c9fe`** — the v1 path is untouched |
| `derived/` | listing identical, 184 entries |
| installed unit | `ExecStart=…/da_midnight_verify.sh`, `Environment=DA_MIDNIGHT_MODE=production`; timer next elapse **Thu 2026-09-03 00:06:00 UTC** |
| `~/ctaNew-wt-da` | clean at `636a455`; my worktree clean after every mutant |

## 8. Counts, the matrix, and rule 10/14

Verifier **247**, tape density **9**, gates **36** (37 with the canary). The module matrix under
both launchers on the complete root, all identical:

| module | `-m` | path |
|---|---|---|
| `pm_tape_density` | 9 | 9 |
| `da_hf_pm_alignment` | 53 | 53 |
| `da_forward_day_verify` | 247 | 247 |
| `da_blackout_mask` | 30 | 30 |
| `da_governed_verdict_preflight` | 34 | 34 |
| `da_content_liveness_v2_check` | 19 | 19 |
| `da_verdict_check` | 21 | 21 |
| `da_cross_venue_forensics` | 24 | 24 |

The only constant the diff adds is `EXPECTED_CHECKS = 9` — a suite count in the idiom the
verifier and DE's modules already use, not a governing number. No decision-shaped field; state
files untouched.

---

## Findings

### DA12-R1 — LOW-MEDIUM — the twin derivation runs one way, and its single exclusion is silent

`_module_launch_twins` (`:128-137`) derives a `[-m]` twin **from a path gate**. It has no inverse,
so a gate already written in the `-m` form gets no path twin, and nothing in the run says so. The
roster has exactly one such gate — `tier1 normalisation` — and the comment introducing the
derivation (`:119`) states the invariant as *"EVERY `--selftest` GATE RUNS UNDER BOTH LAUNCHERS"*,
which is false for it.

Measured, and this is why it matters rather than being a tidiness point:

```
python3 -m live.pm_research.tier1_pipeline --selftest   -> rc 0
python3 live/pm_research/tier1_pipeline.py --selftest   -> rc 1
   ModuleNotFoundError: No module named 'live'
   (tier1_pipeline.py:55 — from live.pm_research.coverage_ledger import …)
```

So the one gate the shape excludes is the one whose other launcher is broken, and the roster
reports 36 of 36 with that fact invisible. This is DA11-R2's own shape one level down: a
mechanism added so a launcher break "cannot sit uninvoked" quietly skips the module that would
fail.

**What I am not asking for.** `tier1_pipeline` may be package-only by design — it imports
`live.pm_research.coverage_ledger` absolutely — and making it path-launchable is a separate
question the coordinator or DA should decide on its merits.

**Closure, in this batch's own idiom.** Derive in both directions (an `-m` gate gets a path twin),
and where a twin cannot be derived — or is derived and known to fail — **print the exclusion as a
named status** with its reason, the way this same round made an absent input a named SKIP. An
unnamed exclusion and an absent check are the same failure wearing different clothes: `1 gate
excluded from twinning: tier1 normalisation (already `-m`; path launch fails: package-absolute
import)` would say in one line what took me a measurement to find.

---

## Executed evidence

At `636a455`, 2026-09-02T14:44–14:50Z:

| check | result |
|---|---|
| scope | four files, +81/−10 |
| tape density, complete vs empty root | `9 (0 skipped)` vs **`8 (1 skipped)`** with the skip named — the summaries differ |
| tape density, a check deleted | red: *"8 ran + 0 skipped = 8, expected 9 … has VANISHED"* |
| roster | 22 gates → **14 twins** → **36**; `--falsify` → **37, 1 FAIL (the canary), falsifier fired** |
| twin-less `--selftest` gates | exactly one: **`tier1 normalisation`** — DA12-R1 |
| `tier1_pipeline` | `-m` rc **0**, path rc **1** (`No module named 'live'`) |
| the `3a89e6c`-shape break | path gate **PASS 53**, `[-m]` twin **FAIL** by name |
| branch conjunct | membership at `:874`; deleting the key from the child's emission → **red** |
| mask suite | **30** from the worktree and under scratch `PM_DATA_ROOT` |
| `--require-no-skips` | rc **0 / 0 / 1**; the failing cell names all six; nothing in the roster, launcher or unit passes it |
| exit codes | `p.returncode` at `:115`, no pipe anywhere in the batch |
| shared tree | the four files **byte-identical to `b75c9fe`**; `derived/` identical; unit unchanged, next elapse Thu 00:06:00 UTC |
| module matrix | eight modules, both launchers, identical (9 / 53 / 247 / 30 / 34 / 19 / 21 / 24) |
| worktrees | `~/ctaNew-wt-da` clean at `636a455`; mine clean after every mutant |

---

## Disposition

- **RELEASE** for `636a455` as the content of Q-DA-209 (landing after the 00:14Z read on 09-03).
  DA11-R1 closes with the summary line a gate reader can tell apart and a count assertion that
  fires by name; DA11-R2 closes with twins **derived** from the roster's own shape rather than
  transcribed; the branch conjunct and the opt-in flag are both what I asked for and neither
  reaches tonight. **No hold.**
- **FILED:** **DA12-R1** (LOW-MEDIUM — the derivation is one-directional and its one exclusion is
  unnamed; the excluded gate's other launcher fails).
- **Item 4, stated not wired:** the flag's semantics support the nightly policy DA describes; the
  boundary is that it governs the selftest, so the gate runner on the canonical tree is its
  natural caller rather than the 00:06Z verdict run. The coordinator's call after tonight.
- **Item 5:** I could not make `v4 behaviour (git-extracted)` fail — it passed in both of my full
  roster runs today; nothing recorded against the batch.
