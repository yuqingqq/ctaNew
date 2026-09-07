# REVIEW 93 — PART A: **NO-GO** for GO E2 and GO #8, on one artifact · PART B: the checker, the E1 artifact, DE's disclosure

**Reviewer (pm-codex), 2026-09-07T08:4xZ. Read at `d7c4faf` in `~/ctaNew-wt-rev` (design v26's
landing commit `4700030` included). Read-only: no heavy unit, no lock, nothing written under
`data/`, never `--open`, no feed opened. CHECKED = I went to the artifact or ran the code;
AGREED = I read the same summary.**

---

# PART A — **NO-GO** for GO E2 **and** GO #8

> **NO-GO on one artifact: `live/pm_research/declarations/de_multiday_gate1_params_v18.json`,
> field `be_module.sha256` = `93332a45faf714fe…`, against
> `live/pm_research/be_cancel_axis_null.py` which is `5607bfbfe1b4ef89…` on disk. BE 96
> (`c707eb8`) changed the cascade module DE's null control cites, and the citation was not
> re-pointed.** The runner refuses by name:
>
> ```
> RunnerRefused: REFUSED: BE's cascade module digest differs -- declared 93332a45faf714fe,
> found 5607bfbfe1b4ef89. A null run through a DIFFERENT cascade is not a control for this
> arm, and the citation must be re-pointed deliberately.
> ```
>
> **This is the guard working, on exactly the case it was built for** — two seats landing into
> one dependency inside one window. **Everything else in Part A passes**, including the
> gate condition I named at REV 92 §D. The closure is one paired landing and I have measured
> its justification for DE (§A6).

## §A1 What I ran, and where it failed

```
harmful_stateful_policy --selftest    90 checks OK        (BE 96's own; run by me)
de_decision_ledger      --selftest     6 checks PASS
de_early_read           --selftest    18 checks PASS
de_multiday_design_declaration        132 checks PASS
de_multiday_gate1_runner --selftest   *** rc 1 *** -- REFUSED at check 7, verify_be_module
```

(**CHECKED**, all run by me at the tip.) The runner's battery aborts seven checks in. **And
the guard is not battery-only** — `verify_run_inputs` (:887), *"Everything that must hold
BEFORE the first day is touched"*, calls it, and so does the module-load path at :1070, which
digests **the file the import actually loaded**. So a real day run refuses at preflight rather
than running through an unverified cascade. That is the right failure mode; it still consumes
a GO and takes the lock, and R-747 is the standing reminder of what discovering this at a
launch costs.

**Which module, and who moved it:** `params v18.be_module` names
`live/pm_research/be_cancel_axis_null.py` at `93332a45…`; the file digests `5607bfbfe1b4ef89…`;
`git log` on that path gives `c707eb8 — BE 96` as the commit that last changed it
(**CHECKED**). BE 96 touched three files — `be_cancel_axis_null.py` (+70),
`de_phase4_diag_runner.py` (+51) and `harmful_stateful_policy.py` (+83) — and params v18,
landed by DE 124 in the same window, carries the pre-BE-96 digest.

## §A2 Everything else in Part A passes, and REV 92 §D's condition is met

| what | measured | verdict |
|---|---|---|
| params **v18** `cfc2b06fe2c9e792…` from v17 `81b2c2910b3c4799…` | `supersedes.sha256` **recomputes from v17 on disk** | **CHECKED** |
| design **v26** `7de8906e607a66d4…` from v25 `b95ac59cf46d941d…` | pair **recomputes**; chain resolves, head = v26 | **CHECKED** |
| `PARAMS_REL` → v18 | `de_multiday_gate1_params_v18.json` | **CHECKED** |
| **`P3_design` — the condition I named at REV 92 §D** | head v26 pins `cfc2b06fe2c9e792…`; `PARAMS_REL` digests `cfc2b06fe2c9e792…` → **HOLDS** | **CHECKED** |
| design v26 tracked | `4700030`, blob == disk `7de8906e…` | **CHECKED** |

At REV 92 I measured `P3_design` failing (head v27 pinned a params file that did not exist
yet) and asked that it be verified at the launching tree. It holds now, and v27–v30 are
withdrawn.

## §A3 The seal is retired correctly — all three branches compute, and the sealed days stay reproducible

`seal(day_result, n_days_complete, g, *, unsealed_by_ruling=False)`. Driven by me:

```
seal(arm, 1, 6)                          sealed=True   "SEALED -- 1 of 6 days complete. …"
seal(arm, 4, 4)                          sealed=False  "UNSEALED -- 4 of 4 days complete under the bar this call was given"
seal(arm, 1, 6, unsealed_by_ruling=True) sealed=False  "UNSEALED_BY_USER_RULING R-765 -- 1 of 6 days complete"
                                                        + R5_retired_from = "params v18 / design v26"
```

(**CHECKED**.) **No branch prints a literal**; each computes from the arguments it was handed —
REV 90 §A0's property, now held across three branches instead of two. The ruling branch is
checked FIRST and returns unconditionally, so it cannot return a sealed artifact; and it stamps
*why* it is unsealed into the artifact, which is the provenance a later reader needs.

**The retired path is still drivable and the four sealed days remain reproducible**: with the
default flag, `seal(arm, 3, 6)` still produces the sealed string the landed days carry
(**CHECKED**). Retiring the seal did not delete the ability to reproduce what was sealed.

**The ruling is carried by the DECLARATION, not by a code constant.** `run_day` branches on
`params.get("user_ruled_unsealed_emission")` — a v18 block naming `ruled_by: THE USER`,
`recorded_at_utc: 2026-09-07T07:47:01Z`, `landed_at: R-765`, `authority: SEAT_PROTOCOL rule 14`
(**CHECKED**). That is the right home for it: rule 14's authority lives where a reader resolves
it, not in a boolean somebody typed.

*One asymmetry, observed and not a hold:* the early-read branch guards its own outcome
(`EARLY_READ_STILL_SEALED` refuses if the artifact came back sealed); the R-765 branch has no
such inverse guard. It does not need one — `unsealed_by_ruling` short-circuits before any
condition — but the two branches now differ in whether they check their own result, and the
early-read guard exists because that assumption was once wrong.

## §A4 The decision ledger

`de_decision_ledger.py --selftest` **6 checks PASS**, and the cell I most wanted to see is
there: the size claim is **stated, not discovered** — *"a REAL arm-day is ~600 draws + ~30–45k
fills and ~16–19k decisions per arm, so a two-arm day scales to roughly 2 MB gzipped — the
estimate is stated here rather than discovered on the first real day"* (**CHECKED**). The
runner writes the ledger **before** the receipt so the receipt can name it by digest, and *"a
day whose ledger cannot be written REFUSES"* — the R-765 ruling is *store the numbers*, and a
receipt promising a ledger that does not exist would be the failure mode. `inventory` is
carried as `ABSENT_UNTIL_BE_96` → and BE 96 is what supplies it, which is the same landing that
moved the cited cascade.

## §A5 The six walls — **none loosened**, and two are tightenings. One residual.

I drove the ones I doubted rather than reading them.

**(1) The bounded withdrawal.** The premise is that the withdrawn drafts never landed, and it
checks out: `git log --all` on the five paths gives **v27, v28, v29, v30 — zero commits each**
(**CHECKED**). Deleting a file that was never committed does not touch a landed version, so
R-711 is not engaged. v26 has exactly one commit and it is the version that stands.

**(2) The cadence table** classifies the eight keys from the family's **own history**, counted
over consecutive landed versions and never typed — and a key measured NEVER falls into the
FROZEN set **compared byte for byte**. That is *stricter* than the previous state, which
required all eight to change every version and so failed its own positive control on a constant
(`also_supersedes_is`). **A tightening, not a loosening.**

**(3) The MEASUREMENT permission (R7/R15/R22).** This is the one worth doubting: it lets a
declared block change if its measurement re-derives the new value. I drove it myself, with
cases the battery does not run:

```
1 unchanged (control)             PERMITTED
2 drop one real day + add a fake  REFUSED DERIVED_DAY_SET_SHRANK
3 drop one real day               REFUSED DERIVED_DAY_SET_SHRANK
4 invented closure entry          REFUSED DERIVED_CLOSURE_NOT_CAPTURED
```

(**CHECKED**.) `DERIVED_DAY_SET_SHRANK` is the safeguard that matters — it refuses the
direction that would quietly narrow the population, which a naive "must equal what the
measurement says" would admit. **Not a loosening**: a byte-freeze says *equal the old bytes*
and had made the family unbumpable from the first verdict landed after v25; the derivation
constraint says *equal what the measurement re-derives now*, and refuses four distinct wrong
directions by name. That is the correct check for a derived block, and it is strictly more
informative than the freeze it replaced.

**(4) The three predicated permissions — and the residual.** The withdrawal record and the pin
predicates behave. On `user_ruled_*` I drove the R-id four ways:

```
landed_at "R-765"  (real)      PERMITTED
landed_at "R-99999" (no entry) PERMITTED     <-- the residual
landed_at "banana"             REFUSED USER_RULING_BLOCK_INCOMPLETE
landed_at "" / missing         REFUSED USER_RULING_BLOCK_INCOMPLETE
```

(**CHECKED**.) So the predicate is **stronger than "a field is present"** — it validates the
id's *form* — and the ruling as written ("only with ruler/verbatim/time/R-id") is implemented
exactly. **The residual is existence:** `R-99999` is not in the register (`grep -c "^### R-99999 "`
→ 0) and is admitted. The cell's own justification argues one step further than the check
reaches — *"A block claiming rule 14's authority and naming no register entry is this seat's
opinion wearing it"* — and a block naming an entry that does not exist is equally that. One
line: resolve `^### <id> ` in the register at the tip. **Not a loosening, and not a hold**; the
landed block names R-765, which exists.

**(5)–(6)** The merge guard's narrowing rides on the cadence table above (a SOME key's content
stays governed by the guard; a NEVER key becomes frozen), and the head-version cell now names
PRE/POST emission instead of asserting `head == VERSION`, which was satisfiable only after the
write and so made the emit run its battery against a state it had not reached. Both restate
the property. **AGREED** on the emission-state cell; I read it but did not drive it.

## §A6 The closure, with its justification already measured

The pin lives in **params v18, which is landed and immutable** (R-711), so the fix is **params
v19 re-pointing `be_module.sha256`, and design v27 pinning v19's digest** (P3_design requires
the head to pin whatever `PARAMS_REL` names) — a paired landing.

**And DE already has a documented method for this act, in v18 itself.** `be_module_repoint`
records the last re-point (`2b164df2…` → `93332a45…`, `changed_by_commit 68b34bd`) along two
axes — the ten named draw-path functions and the nine pinned constants — with a self-correction
worth quoting because it is the reason the method is trustworthy: *"DE 77's justification
compared top-level DEFINITIONS only. THIS change moved NO definition at all — it would have
passed that check in silence. `ARMS`, `SEED`, `N_DRAWS` and `COIN` are module-level
ASSIGNMENTS, so a refitted theta or a changed seed is exactly the class of change that method
could not see."*

**I applied that method to BE 96's change so DE does not have to start from nothing:**

```
draw-path functions CHANGED : NONE
pinned constants CHANGED    : NONE
everything else that moved  : assign:EXPECTED_CHECKS (19 -> 23), def:selftest
```

(**CHECKED**, AST comparison across `c707eb8`.) **BE 96's change to the cited module is
battery-only** — it adds four checks and a cell driving the inventory path. So a re-point is
routine under DE's own method. **I am not pre-approving it**: the refusal's words are *"must be
re-pointed deliberately"*, and re-pointing a control's citation is precisely the act that must
not be automatic. I am handing over the measurement.

**One thing the re-point should widen, and it is the sharper half of this finding.** The
citation pins **one file**, and the file BE 96 changed *behaviourally* is a different one:
`harmful_stateful_policy.py` (+83, `_charge_fill` now computing and carrying
`inventory_before`/`inventory_after`). Neither it nor `de_phase4_diag_runner.py` is named
anywhere in params v18 (**CHECKED**). **The guard fired on the file that changed cosmetically
and would NOT have fired had only the behavioural file changed.** The runtime closure digest
does reach them — they enter the closure lazily at the cascade's first `load()` — so a real run
is covered; but the *declared citation*, whose refusal says *"a null run through a DIFFERENT
cascade is not a control"*, names one third of the cascade. **Routed: `be_module` should be a
list of the cascade's modules by pair, or state in the artifact why one file stands for three.**

---

# PART B

## §B1 The checker at `773f857` — **my REV 92 §A6 #3 is closed, and the v7 question is now ESTABLISHED by the instrument**

```
FORKED_BY_EDIT_AND_SUPERSEDED  producer_exit_maps_v7.json  edits_after_base=3
  created_by=5f5c92b  edited_by=4e91739  repaired_by=[31c5208 0e2a0c6]
  superseded_by=[producer_exit_maps_v8.json]  pre_edit_digest=6084d6e2602d6e6a
  pre_edit_digest_pinned_by=[producer_exit_maps_v8.json:supersedes.the_incident.das_v7]
```

(**CHECKED**, full-directory run, `SUPERSEDED FORKS: 1`, exit 2.) **The JSON path is printed**,
so a MENTION and a LINK now read differently in the output itself. At REV 92 I had to open v8
by hand to learn that its hit was an incident record rather than a resolution pin; **that is now
readable off the field**, and the statement I recommended is established by the instrument:
*no resolution pin names v7's pre-edit bytes; exactly one incident record does.*

The `IMMUT_SKIP_PIN_CENSUS` switch for the falsifier's denominator sub-runs is the right shape —
the sub-runs exist to prove the HISTORY line prints, and re-running the pin census inside each
of them was pure cost. Its default is off, so the census still runs in every real invocation.
I also note the implementation's own self-catch, recorded in the script: *"`python3 -` reads its
PROGRAM from stdin, so the hit list travels by file, not by pipe — the first version lost it"*.

**§A6 #4 is NOT closed** and stays minor: the field's scope (`-size -5M`, `-maxdepth 1`, two
directories) still does not appear beside its answer, so `pre_edit_digest_pinned_by=[…]` still
reads as *these and no others*.

## §B2 The E1 artifact — not yet on disk

`p003_de_early_read_day_20260903__*.json` does not exist at 08:4xZ (expected ~08:55Z). **I have
censused nothing**, and the key-only census is owed to the next round. Nothing here rests on it.

## §B3 DE's disclosed pre-existing FAIL in `de_phase4_diag_runner --selftest`

**Reproduced at the tip** (**CHECKED**):

```
[de_phase4_diag_runner] FAIL: DE46/DE56: THE R-499 ADMISSION STILL HOLDS -- `phase2_arms.py`
reads BLOCKING with ['<module top-level>', 'assert_tape_for_day', 'tape_index'] undeclared,
through USER_ADMISSIONS and its run
```

**Is it BE 96's?** I could not execute the pre-BE-96 state — the module needs real repo history
and refuses out of a `git archive` extraction (*"the fit bytes of `phase2_arms.py` are not
retrievable … with no left-hand side there is no drift to describe"*, which is itself the right
refusal). So I bounded it instead: **BE 96 did not touch `phase2_arms.py` at all, and its diff
to `de_phase4_diag_runner.py` contains ZERO mentions of `R-499`, `USER_ADMISSIONS`,
`phase2_arms`, `assert_tape_for_day` or `tape_index`** (**CHECKED**). The failing cell is
outside BE 96's blast radius, so **DE's "pre-existing" disclosure is well-supported** — by
scope, not by execution, and I say which.

**What it does and does not gate.** Nothing in the GO path runs this module's selftest: the day
run is gated by the *runner's* battery, which is where §A1's refusal lives. So this FAIL is a
disclosed surface that **nothing currently checks at a launch** — which is the reason to route
it rather than to leave it as a note. It is not a second NO-GO.

---

# §C HOLDS AND ROUTING

| id | artifact | what |
|---|---|---|
| **NO-GO** | `de_multiday_gate1_params_v18.json` → `be_module.sha256` `93332a45faf714fe…` vs `be_cancel_axis_null.py` `5607bfbfe1b4ef89…` | **GO E2 and GO #8 both blocked.** Closure: params v19 re-pointing, plus design v27 pinning v19 (P3_design). Justification measured in §A6 — the change is battery-only on both of DE's own axes |

| # | to | finding | kind |
|---|---|---|---|
| 1 | DE | `be_module` pins ONE file while the cascade is three; the behavioural change (BE 96) was in `harmful_stateful_policy.py`, which is named nowhere in params v18. Pin the cascade by pair, or say why one file stands for three | routed (§A6) |
| 2 | DE | the `user_ruled_*` predicate validates the R-id's FORM but not its EXISTENCE (`R-99999` admitted; no such entry) — the cell's own justification argues one step further than it reaches | routed (§A5) |
| 3 | DE | `de_phase4_diag_runner --selftest` FAILs at the tip; disclosed and outside BE 96's blast radius, but nothing in the GO path runs it | routed (§B3) |
| 4 | DE | `de_early_read.py` still ships no shared-falsifier cell (REV 91 §C1, third round open) | routed |
| 5 | coordinator | the checker's scope (≤5 MB, `-maxdepth 1`, two dirs) still unprinted beside its answer | minor |

**Closed this round:** REV 92 §A6 #3 (the JSON path, and with it the v7 question); REV 92 §D's
`P3_design` condition; REV 90 §A0's property extended cleanly to a third seal branch.

# §D WHAT I DID NOT ESTABLISH

- **Not established:** the E1 artifact (not on disk); whether the `de_phase4_diag_runner` FAIL
  predates BE 96 **by execution** (bounded by scope instead, and I say so); the emission-state
  cell (read, not driven).
- **AGREED:** DE's decision-ledger size estimate for a real day (its own cell states it as an
  estimate, which is the right form); MEM 253–255.
- **Method note:** I applied v18's own `be_module_repoint` axes to BE 96's change by AST
  comparison rather than by reading the diff, because the method's own recorded lesson is that
  a definitions-only comparison *"would have passed that check in silence"* for a changed seed
  or a refitted theta. Comparing module-level assignments as well is what makes the answer
  worth anything.
