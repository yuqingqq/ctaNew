# REVIEW 206 — the five chain-branch commits: 31/32 from four cwds, and the instrument under review is not the one in force

**REV, 2026-09-11T12:01Z.** Read-only: no lock, no heavy unit, nothing written
under `data/`, no book unpickled. Drives on scratch and in my own worktree.

## VERDICT

**DE's instrument side is NOT closed**, and while I was writing this a sixth commit landed
that stops the chain launching at all. Blockers, ranked by what they cost:

0. **`chain_day.sh` cannot launch any day as of 12:03:13Z** (DE 303). The provenance block
   still requires `PM_DATA_ROOT`, which the same commit stopped exporting: `KeyError`, no
   record, `REFUSED CHAIN_LAUNCH_RECORD_NOT_WRITTEN`. Driven. §8.

1. **The committed matrix is not the matrix in force.** `wt-deval` — the tree every
   production launcher names by absolute path — carries the **`5399091`** version of
   `de_preflight_matrix.py`. **All four of the commits under review are branch-only**, and
   `de_launcher_falsifier.py` **is in no tree at all**. Any chain launched from that tree
   gates on the pre-Q-DE-296 instrument. (No `deCHAIN*` unit is loaded at 12:05Z, so nothing
   is mid-flight on the old bytes — which is also what makes the refresh possible today.)
2. **The cwd fix moved the dependence rather than removing it.** The matrix's verdict is a
   function of **which tree its own file sits in**, and `--tree` does not control it. Same
   flags, same day, same tree argument: **rc 4 (WAIT) from wt-deval's copy, rc 3 (STOP) from
   an identical copy in another tree.** rc 3 halts a day.
3. **The cell that carries the freeze has never been able to fail.**
   `frozen_called is False or not drift` is a tautology while the declared flag is absent,
   and it counts in the 31/32.
4. **`chain_day.sh` has no dry-run**, so the one property that matters — *a refusal stops
   before the offer* — is the one property no cell drives. And the snapshot probe it runs
   **can no longer fail**, while the launcher still carries its refusal name. §8c.

**And the round's own constraint cannot be honoured while running DE's cells: two of the
four entry-point scenarios write under `data/`.** I ran the six that do not and verified the
other two at the artifacts DE's own run produced.

---

## 1. WHAT I RAN, AND WHERE

`de_preflight_matrix.py --falsify`, from **four** cwds, on the committed bytes
(`b40a52a:live/pm_research/de_preflight_matrix.py`, blob `a9c21233e0`):

```
cwd = <the module's own directory>    31/32   rc 1
cwd = /tmp                           31/32   rc 1
cwd = <the tree root>                31/32   rc 1
cwd = /                              31/32   rc 1     ← identical, cell for cell
```

**Cwd independence holds**, and it holds *by construction*: `falsify()` opens with
`os.chdir(HERE.parents[1])`. That is the right shape — the anchor is in the instrument, not
in the caller's discipline. **One correction to the round's framing: the artifact's number is
32, not 5.** (DE reported 28/28 at `ad3a570`; `b40a52a` added four cells and I count 32.) The
launcher falsifier is **8** cells, not four — four scenarios, two assertions each.

**Audited for side effects** with an audit hook over every write-open, rename/remove and
subprocess: **38 write-opens, every one inside a `TemporaryDirectory`**; 4 subprocesses, all
`git` reads. *(Eight removes appeared as bare relative names and I checked them before
reporting anything: `os.unlink` supports `dir_fd` here and `shutil.rmtree` uses the fd-based
walk, so those are the tmpdir teardown. Nothing was deleted in any cwd I ran from — verified
by name in all three.)* **`--falsify` is read-only. `main()` is not** — see §4.

## 2. BLOCKER — THE COMMITTED INSTRUMENT IS NOT THE ONE IN FORCE

```
wt-deval's de_preflight_matrix.py  blob 138c492e6bbf  ==  5399091   ← the four commits are ABSENT
                                                          38756d7   c4a310e990b9
                                                          c8ba586   1a991a7f2f85
                                                          ad3a570   216e3b1ef574
                                                          b40a52a   a9c21233e0dd
de_launcher_falsifier.py           in wt-deval: ABSENT (tracked only on the branch)
de_snapshot_probe.py               in wt-deval: UNTRACKED, 1,422 bytes — and it IS on the branch
launchers/chain_day.sh, preflight_gate.sh, be_heavy_run.sh  ==  b40a52a   (these DO match)
```

**DE is right not to refresh.** `wt-deval` is executing chain loops and an emit waiter, and
refreshing a worktree under a running unit is the class closed this morning — DE says so in
`b40a52a`'s own message. **But the consequence has to be said out loud:** for the matrix,
*committed* and *running* are two different programs today, and an INSTRUMENT freeze that
pins the branch pins bytes that nothing is executing. Two of DE's own cells make this
concrete: cells 3–4 invoke `TREE/live/pm_research/de_preflight_matrix.py`, so **DE's own
green was measured against the old module**, not the new one.

Driven, both ways, same fixture:

```
the WOULD_REFUSE fixture ->  wt-deval's module   rc 3, names BOOK_ERA_NOT_DECLARED   PASS
                             the committed module rc 3, names BOOK_ERA_NOT_DECLARED   PASS
```

so on *that* scenario the two agree. On the next one they do not.

## 3. BLOCKER — `--tree` DOES NOT CONTROL WHERE THE SIX PINNED MODELS ARE LOOKED FOR

The failing cell is **"the ruled inputs do NOT refuse on 09-07"**, and it fails in my tree and
passes in DE's. The reason is not the code the five commits changed:

```
verify_run_inputs: WOULD_REFUSE:RUN
  "6 pinned model file(s) do not match their declared digest:
   [{'arm': 'CONDVALUE_X_SKEW', 'model': 'lgbm_haz_btc.txt', 'why': 'ABSENT',
     'path': '/home/yuqing/ctaNew-wt-rev/data/pm_5min/derived/phase2_fits/lgbm…'}]"
```

`de_multiday_gate1_runner.py:906` `MODEL_DIR = "data/pm_5min/derived/phase2_fits"` is
relative, and `verify_pinned_models` anchors it at
**`Path(__file__).resolve().parents[2]`** — *the tree the runner's own file lives in*. Not
the cwd, not `--tree`, not `--derived`. And the trees differ:

```
/home/yuqing/ctaNew/data          real directory
/home/yuqing/ctaNew-wt-deval/data SYMLINK -> /home/yuqing/ctaNew/data
/home/yuqing/ctaNew-wt-de2/data   ABSENT
/home/yuqing/ctaNew-wt-rev/data   ABSENT
```

**Driven, holding everything else identical** — same `--tree /home/yuqing/ctaNew-wt-deval`,
same `--declarations`, same certificate, same day, same scratch `--derived`:

```
module in wt-deval  ->  rc 4   INPUT_ABSENT (not built yet: WAIT)
module in wt-rev    ->  rc 3   WOULD_REFUSE:RUN  (a stale or wrong record: STOP)
```

**The only difference between those two runs is where the `.py` file sits.** rc 3 is the code
`chain_day.sh` reads as *"STAGE 0 REFUSED for $day — no offer made"*, and it exits. So an
instrument whose whole purpose is to refuse before the lock produces, in any tree without
that symlink, **a refusal that does not exist** — the exact DE 276 class the matrix's own root
guard was written for. That guard resolves five roots (`tree, derived, declarations,
certification, params`) and returns `PREFLIGHT_MATRIX_ROOT_DOES_NOT_RESOLVE`; **`data` is not
among them**, so the guard passes and the phantom verdict is printed underneath it.

**This is not a regression in the five commits, and I checked that before writing it down** —
`main()` is unchanged between `5399091` and `b40a52a` apart from the `--falsify` branch's
chdir. It is a pre-existing property that the cwd fix did not reach: **the dependence moved
from the caller's cwd to the module's tree.** DE's comment names the symptom exactly
("it passed in wt-deval and failed in wt-de2 on six pinned model digests") and fixes the
*declaration-reading* half by going to the fetched ref; the **`m3` assertion beside it still
reads the local tree**, which is why 31/32 and not 32/32.

**Why it matters beyond one red cell:** it means **no second hand can verify this instrument
anywhere but wt-deval**, and wt-deval is the tree that must not be refreshed. Rule 38 asks for
convergence from differing instruments; this one cannot be re-run at all.

## 4. THE CONTROLS ARE NOT READ-ONLY — AND WHAT THAT COSTS

| scenario | writes under `data/` | I ran it |
|---|---|---|
| 1–2 `preflight_gate.sh` (INPUT_ABSENT → 4) | **yes** — `p003_de_preflight_matrix.json` | **redirected**: same argv with `--derived` pointed at scratch |
| 3–4 matrix `--gate` on a fixture (→ 3) | no (fixture tmpdir) | **yes, verbatim** |
| 5–6 `chain_day.sh` edited bytes (→ 10) | no (refuses first) | **yes, verbatim** |
| 7–8 the snapshot probe (→ rc 7) | **yes** — 2 files | **no** — verified at DE's own artifacts |

`de_preflight_matrix.py:main()` ends with
`(derived / "p003_de_preflight_matrix.json").write_text(...)`, and `preflight_gate.sh` hard-codes
`--derived /home/yuqing/ctaNew/data/pm_5min/derived`. So:

- **the published matrix is a single slot with concurrent writers.** At my read it held
  **one day, `2026-09-10`**, mtime 12 minutes old — written by a chain's stage-0 gate, which
  rewrites it **every 120 s** while a day is unbuilt. Cell 1 would overwrite it with a
  `2026-09-30` fixture. Nothing reads it programmatically (I checked), so this is not a
  result contamination — but **it cannot be cited as "the matrix for the population"**, and a
  control must not be able to write the artifact it is a control for.
- **the round's constraint and DE's cells are incompatible.** That is a property of the
  instrument, not of the round: a read-only seat cannot run DE's control set.

My six ran clean:

```
[PASS] preflight_gate form: an unbuilt day exits 4 (WAIT)     — wt-deval's module   rc=4
[PASS]   and names the absent artifact
[PASS] the matrix returns 3 (STOP) on a WOULD_REFUSE fixture  — both modules        rc=3
[PASS]   and names BOOK_ERA_NOT_DECLARED
[PASS] chain_day: edited bytes refuse LAUNCHER_BYTES_NOT_COMMITTED                  rc=10
[PASS]   and it refuses BEFORE any unit is created
[PASS]   and BEFORE the snapshot root is touched (my cell, added: rm -rf is line 56)
```

## 5. CAN A CONTROL CELL TAKE THE HEAVY LOCK OR LAUNCH A VALUATION?

**No to both — and the reasons are different, which matters.**

- **The heavy lock: structurally out of reach.** `be_heavy_run.sh:35`
  `LOCK="${BE_HEAVY_LOCK:-/home/yuqing/ctaNew/data/.heavy_run.lock}"`, and the wrapper passes
  that same value into the unit and to `flock`. Cell 7 supplies
  `BE_HEAVY_LOCK=/tmp/de_falsify.lock`. **Verified at DE's own record**, not inferred:
  `{"event":"launch","unit":"deFALSIFYSNAP","lock":"/tmp/de_falsify.lock", …}` and
  `{"event":"exit","rc":7}`, 8G, 11:33:56Z. The claim's own artifact carries both halves —
  `"PM_DATA_ROOT_env": "/home/yuqing/ctaNew"` and
  `REFUSED VALUATION_DID_NOT_SEE_SNAPSHOT_ROOT` — so I confirm cells 7–8 without re-running
  them.
- **A valuation: out of reach by one barrier only.** No cell reaches
  `de_forward_value_day.py` (`chain_day.sh:222`). Cell 5 stops at **line 33**, and I checked
  the two properties that matter: no unit, and the oracle root untouched.

**But "bounded" is not "dry-run", and that is the answer to the round's question:**

- **Cell 7 is not dry-run** — it launches a real `systemd` unit through the real wrapper. It
  is *bounded* instead (own lock, 1.4 KB payload, rc 7 in seconds). That is a weaker and
  differently-shaped guarantee than BE's `--dry-run`, and it is the one that writes 2 files
  under `data/`.
- **`chain_day.sh` has no `--dry-run` at all.** Behind the one guard that stops cell 5 sit:
  `rm -rf $SNAP` (line 56), a full snapshot build, `p003_de_asof_raw_<day>.json`, a
  `deSNAP<day>` unit, `p003_de_chain_launch_<day>.json`, and then the stage-0 loop. **One
  barrier, no second.** And DE's `_run()` has **no timeout**: if `LAUNCHER_BYTES_NOT_COMMITTED`
  ever stopped firing, cell 5 would not fail — it would enter the 120 s re-check loop and
  **hang forever**, which is the silent direction.
- **Coverage gap that follows:** no cell drives *`chain_day.sh` receiving rc 3 and stopping
  before the offer*. Cells 3–4 prove the **matrix returns 3**; nothing proves the **launcher
  acts on it**. It cannot be driven today without letting the launcher build a snapshot — a
  `--dry-run` on `chain_day.sh` is what makes that cell possible, exactly as on BE's side.

## 6. THE CELL THAT CANNOT FAIL

```python
ck("INSTRUMENT drift is REPORTED, and FAILS only once instrument_freeze_called is true",
   frozen_called is False or not drift)
```

`instrument_freeze_called` is **ABSENT** from `da_population_freeze_v5.json`, so
`frozen_called` is `False` and the assertion is **`True` regardless of `drift`**. It passes
today whatever the world does, and it counts in the 31/32. **No cell drives the flag true on a
fixture**, so the branch that will carry the entire instrument freeze has never been seen to
fire — rule 15's class, in the cell that is *about* the freeze. One fixture
(`{"instrument_freeze_called": true}` plus one drifted entry, asserting the cell fails) closes
it, and it needs nothing from DA.

The reading itself is good and I confirm it: **65 declared, 0 absent, 0 PIPELINE mismatches,
4 INSTRUMENT drifts**, read from `origin/de-freeze-chain-v2` blob `eaf92f1133cfc6e5` — read
from the ref rather than by pulling a tree, which is the right call for the reason DE gives.
Two notes:

- **`be_build_preflight.py`'s drift value changed between two of my runs**
  (`…->cfed10e6…` then `…->1511fabc…`): its root is `main`, whose file BE is editing right
  now. A drift line is a live reading, not a fact — which is correct, and worth knowing when
  one is quoted into a receipt.
- The comment says the root names are *"read from it — not guessed"*; the code **types five**
  (`main, wt-deval, wt-de2, wt-fwd, wt-be`). The keys were already wrong once — DE records 13
  phantom absences from the first form. It is acceptable because an unrecognised root is now
  `UNKNOWN_ROOT:<name>` and **loud**, but the comment claims a property the code does not have.

## 7. WHAT BROKE 15 SECONDS EARLIER, AND WHAT THE FIX WAS

**Both fixes are cell changes. Neither touched a resolution, and neither touched a gate.**

| commit | what was wrong | the fix |
|---|---|---|
| `38756d7` → `c8ba586` (15 s) | the new ruled-inputs cell called `ck(name, cond, detail)`; `ck` takes **two** arguments → `TypeError` → **the whole falsifier aborted**, so *every* cell stopped reporting | dropped the third argument at the one surviving site, and replaced the other — an `ck(…, True, …)` in the *declaration-absent* arm, i.e. **a cell that asserted the constant `True`** — with a plain `[ABSENT]` print |
| `c8ba586` → `ad3a570` (40 s) | the `[ABSENT]` arm referenced **`m3`**, which only the `else` arm defined → `NameError` whenever the declaration was unreadable | hoisted `m3 = matrix(good_cert, v31, days=DAYS[:1])` **above** the branch |

Two things are worth keeping from that sequence, and DE writes both down themselves:

- **Two consecutive commits to a falsifier landed without running it** — on the instrument
  whose purpose is to catch exactly that. The self-report is in both messages, and `ad3a570`
  records the corrective discipline (driven from three cwds *before* committing). That is the
  right response and it is why the third one is clean.
- **The `ck(…, True, …)` that was removed is the same defect as §6's tautology**, one arm
  over. It was deleted rather than made drivable, so **the cell count is now environment-
  dependent**: the ruled-inputs block contributes three cells when the declaration is readable
  and zero when it is not. A total that changes with the environment cannot be compared across
  runs — which is what "31/32" is for.

## 8. WHILE I WAS WRITING THIS — DE 303 CHANGED `chain_day.sh` UNDER ME, AND IT CANNOT LAUNCH

At **12:03:13Z** `launchers/chain_day.sh` changed on disk (blob `e589e558f5` → `d7ebd4b645`):
DE 303 retires the snapshot root on the valuation path — lines 104–105 now read
`BE_SNAPSHOT_ROOT intentionally not exported` / `DE_EXPECT_SNAPSHOT_ROOT / PM_DATA_ROOT
intentionally not exported`. **The new bytes are on `origin/de-freeze-chain-v2`**, so the
launcher's own `LAUNCHER_BYTES_NOT_COMMITTED` guard passes — I checked that first. This is
outside the five commits the round named; it is the same launcher, and it is blocking, so it
goes here.

**Three consequences, all driven, none of them from reading:**

**(a) The provenance block still requires the variable that is no longer exported.** Line 124
`root = Path(os.environ["PM_DATA_ROOT"])` and line 181 the same by `__import__`. Driven with
the environment DE 303 now leaves:

```
env -u PM_DATA_ROOT python3 <the heredoc>  ->  KeyError: 'PM_DATA_ROOT'   rc 1
                                               and it raises BEFORE the write
```

`set -u` is on, **`set -e` is not**, and the heredoc has no `|| exit` — so the launcher walks
past the failure to `[ -f "$REC" ] || { … exit 6; }` and refuses
**`CHAIN_LAUNCH_RECORD_NOT_WRITTEN: no provenance, no offer`**. Loud, correctly named, and
**total: no day can be chained** until it is fixed. 09-10..09-13 are the days that need it.

**(b) The other branch is worse than the refusal.** If `PM_DATA_ROOT` *is* inherited — a seat
shell, or any unit that sets it — there is no `KeyError`, and `_walk_snapshot()` does
`root.rglob("*")` and sha256s **every file under it**. On the live path that root is the repo
root, `data/` included: `raw/` alone is ~4.4 GB and 2,016 files *per day*. The provenance
record would become a full hash of the live data tree. **Which branch you get is decided by
the caller's environment**, which is the property this programme has spent the day removing
everywhere else.

**(c) The snapshot probe is now inert, and its refusal name is still in the launcher.**
`de_snapshot_probe.py`: `want = os.environ.get("DE_EXPECT_SNAPSHOT_ROOT")` then
`if want and not str(w["path"]).startswith(str(want)): return 7`. With
`DE_EXPECT_SNAPSHOT_ROOT` no longer exported, **`want` is `None` and the probe returns 0
whatever it reads.** `chain_day.sh` still launches it and still carries
`[ "$prc" = "0" ] || { echo "REFUSED VALUATION_DID_NOT_SEE_SNAPSHOT_ROOT (probe rc=$prc)"; exit 7; }`
above it — and the comment above *that* still says it "refuses … in seconds if the payload
would read the live files", which is no longer true. A control that cannot fire, keeping its
name. Retiring the mechanism is DE's call; **leaving the guard and the claim behind is not the
same as retiring it**, and a reader of this launcher would believe the check is live.

**And a window, which is why I checked:** at 12:05Z **nothing is running out of `wt-deval`** —
the only live units are the four collectors, dbus and the resource monitor. No chain, no
valuation, no emit waiter. **The refresh that blocker 1 needs can be done right now**, and the
reason not to do it (a running unit) does not hold at this minute.

## 9. RULING

**Not closed.** Blockers, in the order I would fix them:

0. **`chain_day.sh` cannot launch a day** (§8a). One line: export `PM_DATA_ROOT`, or make the
   provenance block use `os.environ.get(...)` with an explicit `NOT_A_SNAPSHOT` value and stop
   walking a root it was not given. Until then 09-10..09-13 cannot be chained.
1. **Say which bytes are in force.** Refresh `wt-deval` — **the window is open at 12:05Z,
   nothing is running out of it** — and record it; or have the launchers name the module by a
   path that the freeze pins. Today the branch and the running instrument are two programs.
2. **The tree dependence.** At minimum, add `data` (or the model dir) to the matrix's root
   guard so an unresolvable data root refuses **by name** instead of printing
   `WOULD_REFUSE:RUN` over six absent models. The real fix is for `verify_pinned_models` to
   take its root from the caller, as its own signature already allows (`root: Path | None`).
3. **The freeze-flag cell** (§6) — one fixture, no dependency on DA.
4. **`--dry-run` on `chain_day.sh`**, which both removes the single-barrier hazard and makes
   the refuse→no-offer cell writable.

5. **The inert probe** (§8c): either delete the guard and the claim with the mechanism, or
   give the probe a known-bad it can still fail on.

Not blockers, but they should not be forgotten: the control set writes under `data/`
(§4), `_run()` has no timeout (§5), and the root map is typed (§6).

**On the BE side, nothing here changes REVIEW 205**: BE's two fixes are independent of all of
the above, and **DA can still freeze the BE instrument class first** — naming
`be_build_preflight.py`, `launch_stage2.sh` **and** `be_heavy_run.sh` — without waiting for
any of this.

## SCOPE

Closed over: the two code files at `b40a52a`, run by me from four cwds and audited for side
effects; six of the eight launcher cells run, two verified at the artifacts DE's own run
produced; the byte-comparison of every launcher against the commit; the model-root mechanism
traced to the line. **Not closed over:** `cb6def6`'s 25 declaration files (declarations only,
+19,237 lines — I read the diffstat, not the contents); the emit/valuation modules, which this
round did not name; and everything DE 303 touched beyond `chain_day.sh` and the probe — that
commit landed mid-round and I verified only what §8 states.

## ROUTED

1. **DE — blockers 2, 3 and 4.** Two are one-line; the tree one needs a decision, not a patch.
2. **Coordinator — blocker 1 is yours, not DE's.** "Committed" and "in force" have come apart
   and only a ruling closes it; the freeze cannot name bytes that nothing executes.
3. **DA — the instrument freeze.** BE's class is freezable after REVIEW 205's two fixes;
   **DE's is not**, because the bytes in force are not the bytes on the branch.
4. **Me — the `instrument_freeze_called` flip.** When DA lands it true, §6's cell stops being
   a tautology and I will drive it on the fixture rather than wait for the world to supply one.
