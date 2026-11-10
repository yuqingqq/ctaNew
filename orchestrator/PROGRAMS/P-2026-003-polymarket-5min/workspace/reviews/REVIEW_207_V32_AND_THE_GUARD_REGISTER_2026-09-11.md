# REVIEW 207 — the params supersession unblocks nothing, and the guard register is a snapshot no predicate keeps honest

**REV, 2026-09-11T12:15Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Drives on scratch.

## VERDICT

**(1) The supersession is clean as bookkeeping and it does not unblock 09-09..09-13.**
The bytes are identical on every ref, both sha links check out, and the only semantic change
is the day list — **but `de_multiday_gate1_params_v32.json` cannot be loaded at all**
(`expected_G` stayed 6 while `days` became 11), and the function the supersession was written
for, `ruled_day_set()`, **reads `de_multiday_gate1_params_v29.json` and returns the same six
days it always did.** Live consequence, driven: stage 0 now returns **3 = STOP** for every day.

**(2) The guard register is the right shape and its predicate is genuinely computed** — it
refuses right now, 31 of 167 rows unexercised, which is the user's directive made checkable.
**It cannot be satisfied vacuously** (explicitly closed, with a cell). **It cannot refuse an
unregistered refusal site** — I added one and the gate's verdict did not move. And **the
verifier's own falsifier cannot run today**: it aborts at cell 1 on a real PIPELINE drift.

---

# PART 1 — `v31 → v32` AND THE FREEZE CHAIN

## 1.1 Byte identity: confirmed, and one absence worth naming

| file | shared tree | `f309602` | chain tip | `bbb6184` | be-build-decl tip | mm-research |
|---|---|---|---|---|---|---|
| `de_multiday_gate1_params_v32.json` | `ae27c648…` | same | same | same | same | same |
| `de_arm_freeze_v11_amendment.json` | `fe072cd9…` | same | same | same | same | same |
| `de_multiday_gate1_params_v31.json` | **absent** | `c74a3335…` | same | **absent** | **absent** | `c74a3335…` |

**Both artifacts are byte-identical everywhere I can reach them.** (The chain branch has moved
past the `f309602` the round names — tip is `dbd8532`, Q-DE-305 — and the blobs are the same at
both.) **v31 is absent from `be-build-decl` and from the shared tree's working directory**,
which matters in §1.7: the preflight matrix's default `--params` is a v31 literal.

## 1.2 The diff: one semantic change, six bookkeeping ones

DA 234's "a single hunk" is the label **inside the file** (`THE_SINGLE_HUNK`), and it is right
about the field and wrong as a count: the textual diff is **6 hunks** and the structural diff
is **7 paths**. I compared the parsed documents key by key rather than reading the diff:

| path | v31 | v32 | what it is |
|---|---|---|---|
| `/days` | 6 days, 09-03..09-08 | 11 days, 09-03..09-13 | **the semantic change** |
| `/version`, `/protocol` | 31, `…_V31` | 32, `…_V32` | identity |
| `/SUPERSEDES/file`, `/rule`, `/sha256` | points at v29 | points at v31 | rule 13 chain |
| `/WHY_THIS_SUPERSEDES_V31` | — | added | prose + the claim block |

**No other key moved** — no threshold, theta, multiplicity, alpha, matching unit, `be_cascade`
digest or declaration ref. That part of the claim is exact, and it is exact *because* the
comparison is over every key, not over the hunks.

**The day-list claims, computed rather than read** (the file states them; rule 10 says compute):

```
every_existing_day_kept   claimed true   computed True   (and order-preserving: v31 is a prefix)
added                     claimed 09-09..09-13           computed ['2026-09-09'…'2026-09-13']
removed                   claimed []                     computed []
n                         6 -> 11, no duplicates
```

## 1.3 Both sha links check out

```
sha256(v31 bytes)                      = dd58223c6e3654a2…
v32.SUPERSEDES.sha256                  = dd58223c6e3654a2…   MATCH
sha256(v32 bytes)                      = bce61adf3353f032…
v11_amendment.frozen_parameters.params.sha256 = bce61adf3353f032…   MATCH
```

`de_arm_freeze_v11_amendment.json` pins v32 by digest, cites rule 13, and states
`NO_PIN_MOVES: the build pin stays 7ed5a9015f75`. As a supersession record it is correct.

## 1.4 BLOCKER — v32 CANNOT BE LOADED

`de_multiday_gate1_runner.load_params` derives `G = len(days)` and cross-checks it against the
file's own `expected_G` (R-555). **`expected_G` is one of the keys that did not change.**

```
v31: expected_G=6   len(days)=6     -> LOADS,  G=6
v32: expected_G=6   len(days)=11    -> RunnerRefused:
     "REFUSED: the ruled set has 11 days against the declared expected_G 6. R-555 fixes G at 6;
      a set that shrank is a day chosen after the fact, not a smaller test."
```

Driven, both files, through the real `load_params`. **The file's own
`what_did_not_change: "NOTHING else"` is the defect, stated as the assurance**: `expected_G`
is exactly what had to move with `days`, and a supersession that changes the derived quantity
without its declared cross-check produces a file that refuses itself.

Two notes on the refusal, both worth fixing with it:

- **The message is written for the opposite direction.** The condition is `G != expected_G`;
  the text explains only the shrink case ("a set that shrank is a day chosen after the fact").
  Here the set **grew**, and the sentence reads as an accusation of the wrong thing.
- **Whether G may become 11 at all is a ruling, not a patch.** R-555 fixes G at 6 and the
  user's N=6 ruling sits behind it. `expected_G: 11` would satisfy the runner; whether the
  ruled set *may* be 11 days is the coordinator's to say, and I am not treating it as clerical.

## 1.5 BLOCKER — `ruled_day_set()` READS v29, AND RETURNS SIX DAYS

Called, not read, from the production tree:

```
PARAMS_REL       = live/pm_research/declarations/de_multiday_gate1_params_v29.json
ruled_day_set()  = ['2026-09-03' … '2026-09-08']        n = 6
  2026-09-09 in the ruled set: False
  2026-09-10 in the ruled set: False
  2026-09-13 in the ruled set: False
```

`PARAMS_REL` is a module constant and it is **byte-identical on every ref** — chain branch,
mm-research, be-build-decl, and wt-deval's working copy all carry
`704995c07748`. `ruled_day_set()` takes no argument by design ("a lock whose input the caller
supplies is not a lock"), so **no amendment, pin or CLI flag can redirect it.**

**And here is why the fix landed on the wrong file.** v32's own stated reason is:

> *"`de_multiday_gate1_runner.ruled_day_set()` reads the params file's `days` and refuses any
> day outside it; the list ended 2026-09-08 (DE 299)"*

That sentence is **true of the file that is read and true of the file that was edited** —
`v29.days` and `v31.days` are the *same six days*. The symptom was correctly diagnosed and
then attributed to v31, which nothing reads, because v31 showed the identical list. **Two
files agreeing is what made the wrong one look like the cause.**

So there are now three params files in play on one path: **v29** (what the lock reads),
**v31** (what `chain_day.sh` passes as `--params` and what the matrix defaults to), and
**v32** (what the freeze pins). None of the three agree.

## 1.6 The resolver walks to v32 — on the chain branch only

```
from /home/yuqing/ctaNew-wt-deval :  path = …/de_multiday_gate1_params_v32.json
                                     pin  = {"path": "…_v32.json", "sha256": "bce61adf…"}
                                     days = 11
from /home/yuqing/ctaNew (AUTHORITY):path = …/de_multiday_gate1_params_v29.json
                                     pin  = {… "sha256": "066b52f2…"}
                                     days = 6
```

The authority path's answer is not a bad walk — **there is no walk.** The shared tree's
`be_score_neutrality.py` (`75e2bd0c…`, identical to `origin/mm-research`) has
**no `resolve_frozen_params_pin` and no `FREEZE_AMENDMENT_GLOB`**; `frozen_params` reads
`de_arm_freeze_v1.json` directly, which pins v29. The chain-branch copy (`46897831…`) has the
resolver.

**So "byte-identical across the three refs" is true of the declarations and false of the code
that reads them.** `de_arm_freeze_v11_amendment.json` sits on the authority path, correct and
inert, next to a module that cannot see it. (The shared tree is also missing amendments v8,
v9, v10 and v31 itself — the resolver would be walking a gapped chain there even if it
existed.)

## 1.7 What this does RIGHT NOW: stage 0 stops every day

Driven at 12:1xZ, from `wt-deval`, `--derived` redirected to scratch so nothing is written
under `data/`, day 2026-09-09, **no `--params` — exactly as `preflight_gate.sh` invokes it**:

```
  params        …/declarations/de_multiday_gate1_params_v31.json  OK      <- the default literal
  frozen_params WOULD_REFUSE:UNNAMED
  detail        "REFUSED: the ruled set has 11 days against the declared expected_G 6 …"
  WOULD_REFUSE by day: {'2026-09-09': ['frozen_params']}
  rc = 3
```

The path is: the matrix asks `de_settlement_control_run.frozen_params` → `BEN.frozen_params()`
→ the resolver → **v32** → `load_params(v32)` → `RunnerRefused`. `chain_day.sh` reads rc 3 as
*"STAGE 0 REFUSED for $day — no offer made"* and exits. **Every day, not just 09-09.**

**And the refusal loses its name on the way out.** `de_preflight_matrix._refuse()` extracts the
token with `text.split("REFUSED ", 1)` — it requires a **space** after `REFUSED`. The runner
writes `REFUSED:` with a colon, so the status is `WOULD_REFUSE:UNNAMED` and the reason lives
only in `detail`. The one refusal that would tell a reader *which file is superseded* is the
one that prints unnamed.

# PART 2 — `da_guard_register_v1.json` (DA 260)

## 2.1 The shape is right

167 rows, `BUILD_BUILDER` 22 / `VALUATION` 145, one row per refusal site, each carrying exactly
what the round specified:

```
lane, file, site ("be_daybook_build.py:625"), line, at_digest ("2d31a80b5ae5c2c6"), ref ("7ed5a9015f75"),
refusal ("SUPPLY_DOES_NOT_NAME_ITS_DAY"), in_function, condition ("sday is None"), inputs_read,
exercised_before_real_run, rehearsal {name, command}
```

`SOURCE_OF_THE_ROWS` says they were enumerated **by the operation** — every `raise` whose
message carries a `REFUSED <NAME>`, from the AST at the pinned digest — and DA writes down that
DE 303(2) and BE 164(1) had not landed theirs, so these "must be reconciled with theirs, not
assumed to agree." That is the right posture and it is the right shape.

## 2.2 The predicate is computed, and it REFUSES right now

`da_population_freeze_verify.guard_gate()` filters rows by lane, computes
`bad = [r for r in rows if not r["exercised_before_real_run"]]`, and raises naming the sites.
Driven, all three ways:

```
--guard-gate BUILD_BUILDER : REFUSED GUARD_ROW_NOT_EXERCISED_BEFORE_A_REAL_RUN  2 of 22
--guard-gate VALUATION     : REFUSED …                                         29 of 145
--guard-gate <ALL>         : REFUSED …                                         31 of 167
```

**This is the artifact working.** "Clear the issues first" is now a predicate that says no, with
31 named sites, instead of a sentence everyone agrees with.

## 2.3 It cannot be satisfied vacuously — closed explicitly, and driven

```python
if not rows:
    raise FreezeRefused(f"REFUSED {GUARD_GATE}: no rows for lane(s) {lanes} -- an empty "
                        f"lane cannot exonerate a run (the aggregate-only trap)")
```

Zero rows **refuses**; it does not pass. There is a cell for it
(*"an EMPTY register REFUSES rather than passing vacuously"*) and a cell for the admit
direction (every row flipped true on a scratch copy → `EVERY_GUARD_ROW…`). This is the one
place in today's instruments where the vacuity question was asked before I asked it.

## 2.4 BLOCKER — it cannot refuse an unregistered refusal site

**Driven, exactly as the round asked.** I copied a registered module to scratch and added one
new site:

```python
raise NeutralityRefused("REFUSED REV168_UNREGISTERED_SITE: a refusal nobody declared")
```

```
AST enumeration of the scratch copy   BEFORE: 10 sites     AFTER: 11 sites   (delta +1)
the enumeration SEES it by name       True
the register                          still 167 rows; no row names REV168_UNREGISTERED_SITE
guard_gate(BUILD_BUILDER)             identical verdict, identical counts, identical sites
```

**The gate reads only the register.** `da_population_freeze_verify.py` contains no AST work at
all, and the string `da_guard_register` appears in exactly one place in the entire tree —
`GUARD_DECL` in that verifier. The AST enumerator (`p003_refusal_exercise_check.py`) never
mentions the register. **Nothing joins the code to the rows**, so the register is a snapshot
that goes stale silently, and a new guard is a guard nobody is required to exercise.

**It is already stale, by measurement.** The register records
`the_exercise_instrument.POPULATION = 217` at 12:04:52Z. I ran the same instrument at 12:15Z:

```
POPULATION 218    n_exercised 178    never_exercised 40
```

**One site landed in eleven minutes**, and no predicate anywhere noticed. (The register's 167
rows are also a *subset* of that population by design — build and valuation paths only — but
nothing computes which sites are legitimately out of scope, so "167 of 218" is an unexplained
gap rather than a scoped one.)

**The missing piece is the join, not the enumeration.** `scan()` already produces
`site → refusal → file:line`; the rows already carry `site` and `at_digest`. A completeness
predicate — *every enumerated site on a registered lane has a row at the current digest, and
every row still exists in the code* — is a short function, and it is what turns the register
from a document into an instrument.

## 2.5 BLOCKER — the verifier's own falsifier aborts before any guard cell runs

```
$ da_population_freeze_verify.py --falsify
Traceback … line 149, in falsify
    ck("the real freeze VERIFIES", verify()["status"] == "POPULATION_FREEZE_HOLDS")
FreezeRefused: REFUSED POPULATION_FREEZE_FILE_DRIFTED: 2 of 67 declared file(s) changed on disk
  -- live/pm_research/de_forward_value_day.py, live/pm_research/de_settlement_control_run.py
  First: {'path': 'live/pm_research/de_forward_value_day.py', 'root': 'wt-deval',
          'declared': 'c397537121a29b44', 'on_disk': 'd024925c268e84c3', 'CLASS': 'PIPELINE'}
```

Two things, and the second is the instrument defect:

1. **The population freeze does not hold.** Two **PIPELINE** files drifted in `wt-deval` —
   DE 303/305 edited them after DA declared them. PIPELINE drift is the class that must fail,
   and it is failing correctly.
2. **`verify()` is called outside a `try` as cell 1's input**, so a legitimate refusal — the
   thing the verifier exists to detect — raises through the falsifier and **none of the other
   cells run**, including all four guard-gate cells I would otherwise be reporting on. This is
   the same shape as Q-DE-296's `TypeError`: one cell's failure stops the instrument from
   reporting on itself, **and here it happens precisely when there is something to report.**
   Cell 1 should be `try/except FreezeRefused` and record the refusal as its result.

# RULING

**(1) Not usable.** The supersession is correctly *recorded* and blocks the programme in three
independent places. In order:

1. **`expected_G`** — v32 refuses itself. Needs a ruling (may G be 11?) before a patch.
2. **`PARAMS_REL`** — `ruled_day_set()` reads v29; no supersession can reach it. Either the
   constant moves with the freeze, or `ruled_day_set()` resolves through the freeze chain like
   everything else. This is the one that actually gates 09-09..09-13.
3. **The authority path has no resolver** — `be_score_neutrality.py` on `mm-research` predates
   `resolve_frozen_params_pin`, so the freeze there silently resolves to v29. Landing the
   chain-branch module on `mm-research` closes it; until then "the authority path" and "the
   path that can read the freeze" are different trees.

Also: the matrix's default `--params` is a **v31 literal** that is absent from two of the trees
it may run in (§1.1), and `_refuse()` drops the refusal name whenever a message writes
`REFUSED:` instead of `REFUSED ` (§1.7).

**(2) The right artifact, two blockers.** Shape correct, predicate computed, vacuity closed,
and it refuses today with 31 named sites. It needs:

4. **A completeness join** (§2.4) — without it the register cannot notice a new guard, and it
   is already one site behind.
5. **Cell 1 wrapped** (§2.5) — the verifier must be able to report when its subject is drifted.

## SCOPE

Closed over: both declarations at every ref that carries them, compared key by key and by
digest; `load_params`, `ruled_day_set` and `frozen_params` **called**, not read, from the trees
that can run them; the matrix's gate driven end-to-end with `--derived` redirected to scratch;
the register's structure and its verifier read in full, its gate driven on three lanes, its
vacuity case driven, and the unregistered-site falsifier driven on a scratch copy.
**Not closed over:** the 167 rows' individual truth (I checked the shape and the predicate, not
that each `condition` matches its site — DA says DE 303/BE 164 rows are still landing and must
be reconciled); the 31 unexercised sites themselves; and whether G may become 11, which is a
ruling.

## ROUTED

1. **Coordinator — `expected_G` is yours.** R-555 fixes G at 6 and the user's N=6 ruling sits
   behind it. Nobody should patch v32 to `expected_G: 11` on clerical grounds.
2. **DE — `PARAMS_REL`.** This is the one that unblocks the days, and it is not in v32.
3. **DA — the completeness join and cell 1** (§2.4, §2.5), and the two drifted PIPELINE files
   in the population freeze (§2.5).
4. **Whoever lands the chain-branch `be_score_neutrality.py` on `mm-research`** — until then
   the authority path resolves the freeze to v29 and says nothing.
