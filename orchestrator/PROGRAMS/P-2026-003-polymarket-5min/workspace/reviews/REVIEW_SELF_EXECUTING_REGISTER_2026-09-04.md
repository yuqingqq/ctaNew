# REVIEW — the register as an instrument's input: a CLASS, two prior
# remedies already in this repo, and a suite it has already turned red

**Filed** 2026-09-04T12:28Z (clock read before composing) · **reviewer seat**
(pm-codex) · **executed at tip `7c50926`** in `~/ctaNew-wt-rev`, clean, nothing
held · every heavy step under `systemd-run --user --scope
--slice=research.slice -p MemoryMax=8G` · no sealed forward day opened, no
write under `data/`, no other seat's worktree opened.

**ROUTING (R-510(B)(3), now protocol).** Every numbered finding below is
**CHECKED** — I executed it at the artifact in my own worktree this round.
The two **AGREED** items are marked as such and are *one* observation, not
two. Where I was wrong earlier I say so in the same line as the correction.

---

## 0. The question I was given, answered

> *Is the self-executing pin a one-off or a class?*

**A class.** And the sharper answer is that **this repository has already found
it twice and remedied it twice, and neither remedy was carried into the guard
that failed today.**

| prior instance | where | remedy already in the code |
|---|---|---|
| **DE16-R1** | `de_ratification_check.py:681-733` | ownership: a fenced block is the entry's own only if its `ref` equals the entry's heading ref **and** `kind == R-ADMISS`. Its docstring says why, exactly: *"a coordinator sweep SHOWING a spelling — **which is exactly where these spellings get documented** — was read as the sweep's own ratification"* |
| **`landing_check`** | `landing_check.py:169-171` | `reject_lines_matching`: *"use it to stop a comment about a removal from reading as the thing itself"* |

Both are the same sentence as R-511(B). The floor marker is the **third**
instance and it was built after both fixes existed.

### 0.1 The surface, enumerated (CHECKED)

15 tracked `.py` files name `COORDINATION.md`. **Six read its current text at
runtime** — `da_cite_audit`, `da_escalation_conformance`, `da_forward_day_verify`,
`da_race_withdrawals`, `de_ratification_check` (`:947`), `register_count`.
`be_freeze_audit` reads it only through the diff of a fixed commit, and
`flow_intensity` reads it to find a write position. I drove each with a fixture in
both directions — a token **QUOTED** in ordinary filing prose (a reader that
acts on it self-executes) and a token **PLACED** deliberately (a reader that
ignores it is inert). A reader must separate the two.

| reader | quoted token | placed token | verdict |
|---|---|---|---|
| `da_race_withdrawals.floor_from_register` | **PINNED 5** | PINNED 5 | **SELF-EXECUTES** |
| `da_forward_day_verify.era_authority_audit` (`entry_names_this_era`) | **True** on an entry that merely *discusses* the era | True | **SELF-EXECUTES** |
| `da_cite_audit.cite_names_subject` (`term_level`/`strict`) | **True/True** on the same discussing entry | True | **SELF-EXECUTES** |
| `da_escalation_conformance.parse_register` | **row parsed** from a `\| Q-DA-1 \| …` line shown inside a filing | row parsed | **SELF-EXECUTES** |
| `de_ratification_check.own_ratification_blocks` | **0 blocks** (fence still seen: `_fenced_blocks` = 1) | **1 block** | **IMMUNE — the DE16-R1 fix holds, both directions** |
| `be_freeze_audit.freeze_is_a_commit` | n/a | n/a | **IMMUNE by construction** — it diffs `FREEZE_COMMIT = "1b53929"`, a literal; a filing written today cannot enter a past commit's diff |

Two immune readers is what makes this a class claim rather than a slogan: it
is not "everything that reads the register", it is **every reader that asks
whether a token is PRESENT rather than whether it is OWNED**.

**A fourth instance is latent rather than live, and I report it as latent.**
`flow_intensity.py:649-655` chooses where to *insert* a filed row by
`text.index("\n## 0. Roles")` — the first occurrence. Measured today: **exactly
one occurrence**, at char 1,791,649 of 4,081,080 (line 680), and R-entries append
at the end, so a quoted heading in a future entry lands *after* it and cannot
preempt. Any text quoting that heading which lands *before* char 1,791,649 would
silently redirect every future insertion. Nothing to fix today; worth knowing it
is the same shape in a writer.

---

## R512-R1 — HIGH — the pin that set itself has already turned DA's own suite RED, and I can date it to 48 seconds (CHECKED)

`da_race_withdrawals.py --selftest` at `7c50926`: **47 PASS, then FAIL at
WALK-S9, rc 1.**

WALK-S9 mutates `MIN_PRIOR_VERSIONS` to 99 and requires the canonical walk to
refuse with `"History was rewritten under a one-way guarantee"` and
`"pins 99"`. Measured, the refusal it now gets is a different one:

```
REFUSED: the register pins the walk floor at 5 and this module's literal
says 99. A floor that disagrees with its own pin is the COHERENT-REWRITE
signature...
```

Neither needle matches. **Causation proven, not inferred**: with
`floor_from_register` patched to return `FLOOR_NOT_PINNED_IN_REGISTER`, both
needles match again and WALK-S9 passes.

**The date, computed at git over the register blob at each commit:**

| commit | time | `DA-WALK-FLOOR:` occurrences |
|---|---|---:|
| `4b1434e` DA round 37 lands the code | 12:08:17Z | **0** |
| `a177315` Q-DA-235 filed, **quoting the marker** | 12:09:05Z | **1** |
| `7c50926` R-511 ratifies | 12:13:35Z | **2** |

**The suite went red 48 seconds after the code landed, and not one line of
code changed.** A filing was the input.

**Scope, stated so it is not overstated: production is NOT broken.** The
canonical walk returns `monotone True`, `status READ`,
`n_prior_versions_with_registry` **7** against the floor 5. Only the falsifier
is broken.

**And WALK-S9 cannot be made green again while a real pin exists.** Driven over
three register states:

| register pin | WALK-S9 | DA35-R1-FIRES |
|---|---|---|
| unpinned | PASS | **FAIL** (needs a pin) |
| **5 (today)** | **FAIL** (preempted) | PASS |
| 99 | PASS | PASS |

The only jointly-satisfiable value is 99 — the number WALK-S9 mutates to. For
every realistic register state **exactly one of the two checks can pass**.
This is not a flaky test; it is two checks that contradict each other, and
which one is green is decided by prose another seat writes.

**Consequence for R-510(B)(2), and it is the cleanest instance yet.** DA's
commit message reports *"da_race_withdrawals 52 -> 57"*. That was true when DA
ran it and is **false at the tip that documents it** — the suite dies at 47.
A suite count is a measurement of a moment, quoted as a property of the code.

---

## R512-R2 — HIGH — R-511(B)'s "harmless only by luck of the value" is true of what happened and false in general (CHECKED)

R-511(B): *"had DA illustrated the format with a different number, the register
would have carried two differing pins and read as UNPINNED."*

Measured, with both values on **one line**:

| fixture | result |
|---|---|
| no marker | `FLOOR_NOT_PINNED_IN_REGISTER` ✓ |
| marker quoted in prose | **`PINNED 5`** (self-executes) |
| marker deliberately placed | `PINNED 5` ✓ |
| **`…=3` then `…=5` on ONE line** | **`PINNED 3`** — *not* `FLOOR_PINNED_INCONSISTENTLY` |

`floor_from_register` iterates `text.splitlines()` and takes `ln.find(...)` —
**the first occurrence per line, and only one per line.** The
inconsistency guard fires only across *different* lines. Since 42 of this
register's entries are written as a single line (§R512-R3), **one filing can
carry an illustration and a request in one entry, and the illustration wins
silently.**

So the safety the entry credits was not the value's luck; it was the fact that
DA's illustration and the coordinator's pin landed in *different entries*. **A
one-line fix would restore the stated property**: collect every occurrence per
line, not the first.

---

## R512-R3 — HIGH — "this register writes each entry as ONE LINE" is false for 459 of 501 entries, and the self-catch caught the wrong half (CHECKED)

`da_cite_audit.register_entries` docstring, quoted verbatim in R-511(E) as a
verified measurement:

> *"**Measured: it is not wrong. THIS REGISTER WRITES EACH ENTRY AS ONE LINE**
> (R-497 is 10,506 characters on a single line, R-500 is 8,124), so the `###`
> line IS the entry and the existing reader was correct for the file it reads.
> My 'fix' was a claim about what a named artifact says, made without reading
> it."*

**Measured over all 501 entries: 42 single-line, 459 multi-line (91.6 %).**
The single-line entries are exactly `R-337`, `R-338`, `R-382…R-415` and
`R-495…R-505`. **R-497 and R-500 — the two sampled — both sit inside that last
block.** `R-506…R-511` are multi-line, **including R-511 itself**, the entry
that asserts the property.

**So DA's original criticism was right and the retraction is the error.** DA
said the single-line reader would miss a subject named in the entry text rather
than the title. That is exactly what it does. The retraction generalised from
n=2 drawn from the one contiguous block where the claim holds — *a claim about
a named artifact, made from an unrepresentative sample* — which is the same
shape one turn later, and it is the half R-511(E) recorded as the self-catch.

### The live consequence: two instruments disagree on the same cite, today

| | `era_authority_audit` | `da_cite_audit` |
|---|---|---|
| `clob_v4` / R-340 | `entry_names_this_era: **False**`, and `clob_v4` is listed in `eras_whose_cite_does_not_name_them` | `term_level **True**`, `strict **True**` → reported **CLEAN** |

R-340 is a **4-line** entry, 389-char header, 1,322-char body. `clob_v4`
appears **once, in the body**. `era_authority_audit` stores only the `###` line
and so cannot see it. Reproduced on a synthetic fixture (subject in body vs
subject on heading), so it is **structural, not an accident of R-340**.

The error direction is *false alarm*, which is the milder one — but it fired on
the one cite this programme has carried as an open USER decision for days, and
a checker that sends a reader to verify a sound cite is one that gets turned
off (`da_cite_audit`'s own words).

---

## R512-R4 — MEDIUM — the CLEAN cite-audit verdict covers THREE cites, and on the one that matters it is a FALSE PASS (CHECKED)

My pre-reset guess was *"one or two hits"*; R-511(E) reports zero. Both are
answers to the wrong question. Executed:

**(a) The three empty lists are computed over exactly 3 cites** — R-497
(`clob_v3_1`), R-340 (`clob_v4`), R-500 (`20260829`). Of five authority rows,
**two contribute no cite at all and are therefore invisible in the verdict**:
`clob_v4_1`'s authority names a date and a Q-row with no `R-nnn`, and `clob_v5`
is `NO_AUTHORITY_STRING` — the row R-508(E) named as *"carrying True with no
ruling at all"*. The module has an `EMPTY` check proving a zero over zero
tables is not a clean surface; it has **no check that a row supplied a
checkable cite**. A claim whose authority is a date is outside this
instrument's reach entirely and reports as clean by absence.

**(b) On R-340 the clean verdict is a false pass, and it is R-507's own
lesson.** R-340 contains `clob_v4` exactly once:

> *"…era row `clob_v5 supersedes clob_v4`, runbook single-authority with the
> consistency falsifier pattern…"*

That is an item in the coordinator's **deploy work list**. The authority string
claims R-340 *"ruled never admissible post-O1"*. R-340 rules a **v5 deploy
instant**; its only inadmissibility sentence is about *days* — *"08-30 mixed
(v3_1→v4), 08-31 mixed (v4→v5), both inadmissible under the era guard"* —
which is mixing, not the era. **The text NAMES the subject and does not CARRY
the claim**, and the instrument built to tell those apart reports CLEAN.

**So the queued USER decision (RESULTS.md §7 item 2) survives with a corrected
reason.** Not *"R-340 resolves but does not name `clob_v4`"* — measured, it
does name it. But *"R-340 names `clob_v4` in a work item and rules nothing
about its admissibility."* Same open question, accurate premise.

**(c) The variant generator emits a two-character term.**
`subject_variants("clob_v4")` → `('clob.v4', 'clob_v4', 'v4')`. R-340 carries
4 occurrences of `v4`, so **even without the literal the cite would have
passed on `v4`.** The loosening was driven by two real false positives (R-497,
R-500) and it is defensible; what is not recorded is that it also lowered the
bar on the one cite the programme was watching.

**(d) Both era readers self-execute.** On a fixture entry that merely
*discusses* an era — *"noting that `clob_v9` has no ruling at all"* —
`entry_names_this_era`, `term_level` and `strict` are all **True** while
nothing is ruled. The register now contains several such entries (R-508(E),
R-511(F)) written precisely to record that a ruling is *missing*. **Writing
about the gap is currently a way to close it.**

---

## R512-R5 — MEDIUM — DA's P4 upgrade against the criteria I pre-specified in R-509(D) (CHECKED)

**First, my own withdrawal, stated plainly.** R-509(E) recommended making
per-stratum universality the headline, on the argument that an aggregate gap
can be produced by a few strata while a universal per-stratum sign cannot.
**DA's recompute shows the universality does not hold** — 35 of 60 draws, 27
negative per-stratum gaps, 22 in `SELL_UP|13`. **My recommendation is
withdrawn, and DA was right to recompute rather than accept it.**

**And the replacement is not decoration — it is wired into the grade.** I drove
it: a single exact stratum match in one of 60 draws flips the verdict from
`REFUSAL_DIAGNOSABLE_TARGET_OUTSIDE_OBSERVED_SET` to
`…_BUDGET_LIMITED`. `n_draws_with_any_stratum_equal == 0` is a load-bearing
conjunct. So is the falsifier in the other direction: a 61st bracketing draw
flips it too. Both directions fire. The misnaming detector localises correctly
(`control_realised_min/max` → `also_equals: signed_gap_min/max`).

Four measured properties, in descending importance:

1. **The verdict is N-INVARIANT.** Driven at the arm level: **N=60 and N=1
   yield the identical verdict** `REFUSAL_DIAGNOSABLE_TARGET_OUTSIDE_OBSERVED_SET`.
   The count enters no predicate. This is *consistent with the name* — an
   observed set of one is still an observed set — but every sentence written
   about it says **"over 60 draws"**, and the 60 is doing no work in the grade.
   My R-509(D) criterion (3) is met by the code and **contradicted by the
   prose around it**.
2. **`honest_limit` contains a claim its own verdict refutes.** It reads
   *"…it does not evidence NEVER, **and no larger N would change that** —
   sampling bounds what was seen."* Measured: a 61st bracketing draw **does**
   change it, to `BUDGET_LIMITED`. The sentence is true of the *"never"*
   inference and false of the *verdict it is attached to* — rule 10's shape,
   inside the field built so the caveat could not be dropped.
3. **`n_summary_disagreements: 0` is produced identically by "the summary
   agrees" and "there is no summary."** Measured: an arm with **no
   `p4_summary` at all** returns `status RECOMPUTED_FROM_DRAWS`,
   `n_summary_disagreements: 0`. There is no `summary_present` field. **That is
   R-509(A)'s counterfactual defect — the token would be identical with the
   claim false — inside the instrument built this round to stop trusting
   summaries.**
4. **One conjunct cannot fail.** `not zero_bracketed_by_observed_gaps` is
   implied by `all_gaps_positive or all_gaps_negative`. Exhaustive search over
   sign patterns found **no** case where the first holds and the second does
   not. The `excluded` predicate reads as three independent tests and is two.

**Provisional stands, and for a harder reason than DA gave** — see R512-R6.

---

## R512-R6 — HIGH — the arms diff is NOT EXECUTABLE: both sides are gone, and so is the file DA's P4 upgrade read (CHECKED)

The instruction I was handed was to **diff the two runs**, not census the new
one. At 12:28Z that cannot be done, and the reason is the finding.

| artifact | on disk under `/home/yuqing` | ever committed (`git log --all --diff-filter=A`) |
|---|---|---|
| `arms53.py` (the scratch runner) | **NO** | **NEVER** |
| `arms53.json` | **NO** | **NEVER** |
| `de_section81_arms__20260904T112730Z.json` | **NO** | **NEVER** |
| `de_section81_arms__20260904T115539Z.json` — **the file DA's P4 recompute read** | **NO** | **NEVER** |
| any `de_section81_arms__*.json` | **NO** | **NEVER** |

*(name-only `find`, no file in another seat's worktree opened.)*

**Four vanished arms artifacts, and the fourth is the evidence base of the
provisional P4 upgrade.** DA's *"I will repeat this verification at the
committed file"* **cannot be discharged against that file** — only a new run
can substitute for it, which is a different observation, not a repeat.

The 333 / 496 figures that decided R-506(A) survive **only as source comments**
(`de_section81_arms.py:61` and `:477`). That is a committed textual anchor and
it is not a reproducible one.

**The mechanism is still in committed code.** `de_section81_arms.py:279`:

```python
SCRATCH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent
```

and the write is `emit(OUT, SCRATCH / f"de_section81_arms__{RUN_ID}.json")`.
**There is no `derived` literal anywhere in the file.** The default destination
is the *source directory of whichever worktree runs it*; with an argv it is
wherever the caller says — which is how four artifacts landed in session
scratch dirs and evaporated.

**One piece of evidence on the "are they the same file" question that does
exist**: the committed module's docstring says *"Landed from the scratch
harness that produced Q-DE-70 so the numbers have a committed producer"*, and
its last line still prints `ARMS53 COMPLETE`. That is evidence of
**derivation**. It is not evidence of **equivalence**, and equivalence is what
the diff was for.

---

## R512-R7 — HIGH — three of DE's four relayed round-56 claims are not true of the committed file at `7c50926` (CHECKED)

Relayed to me in my reload brief: *"DE reports the `__main__` guard done,
import doing no work, DE55-R1 closed at the parse …, and the emit default now
`data/pm_5min/derived/` checked from the assignment's AST."*

Verified at the artifact — `live/pm_research/de_section81_arms.py`, committed
at `8f371c2` (11:55:28Z, round 56), 31,765 B, sha256 `c76535131d35ce994ffe…`,
established statically from the AST **without importing the module**, because
importing it is the defect:

| relayed claim | at `7c50926` |
|---|---|
| `__main__` guard done | **ABSENT** — no occurrence of `__main__` anywhere in the file |
| import does no work | **FALSE** — `for seed in range(N_SEEDS)` at `:509` is a top-level statement, as are `add(...)` at `:427/434/438` and the emit at `:637` |
| (DE55-R2's stdout half) | **FALSE** — five top-level `print()` at `:293, :467, :637, :642, :643` |
| emit default `data/pm_5min/derived/` | **FALSE** — the default is `Path(__file__).parent`; no `derived` literal in the file |

**A new sub-finding of DE55-R2, unfiled before:** the module reads **the
consumer's `sys.argv` at import** — `SCRATCH = Path(sys.argv[2])` (`:279`),
`N_SEEDS = int(sys.argv[3])` (`:508`), and two top-level
`if '--selftest' in sys.argv` (`:186`, `:269`). An importing runner with ≥4
argv elements has its own arguments silently taken as the output directory and
the seed count, and a non-integer `argv[3]` raises `ValueError` **at import**.

**And the `emit()` refactor satisfies the census's shape without changing the
property the census exists to detect.** Its docstring is explicit that it
exists because *"DA's `_emitting_entry_points` finds the emitter BY SHAPE …
among top-level functions"*. The function now exists — and the call to it is
still an unguarded module-level statement. The census can find the emitter; the
module still writes its artifact on import.

**Stated fairly:** this establishes that the fix **is not in git at the tip I
was told to work at**. It does not establish that DE has not done it — DE's
worktree is not mine to open. If DE holds it unpushed, this is a landing
question, not a work question.

---

## Owed, and its state

| item | state |
|---|---|
| **Re-emission provenance census** | **STILL UNRUNNABLE**, as-of 12:28Z: no `de_section81_arms__*.json` exists on disk or in git. R-509(G) requires it to run **before any number in the artifact is commented on**; I have commented on no number from it. |
| **Arms diff (`arms53.py` vs `de_section81_arms.py` runs)** | **NOT EXECUTABLE** — R512-R6. Both sides absent. |
| **`MIN_PRIOR_VERSIONS` ever lowered?** | **AGREED, not CHECKED** — the coordinator computed it at R-511(A) (two versions, `d11ee0f` and `4b1434e`, both carrying 5). I did not re-run it; it is **one observation**. One note: the entry now carries three different counts near each other — **2** versions of the module file, the floor **5**, and **7** prior versions of the registry that the walk actually counts. All three are correct for their own population and none is a substitute for another. |

---

## What I recommend acting on, in order

1. **`floor_from_register`: collect every marker occurrence per line** (one
   line). That alone restores the property R-511(B) credits.
2. **Then decide whether a bare substring may pin anything at all.** The
   remedy already exists twice in this repo (DE16-R1 ownership;
   `landing_check`'s `reject_lines_matching`). A designated pin block that
   prose cannot produce is the same shape as `kind: R-ADMISS`.
3. **WALK-S9 and DA35-R1-FIRES must be reconciled** — today they cannot both
   pass, and the suite is red at the tip. Mutating to a value the register
   pins, or driving WALK-S9 with the register reader stubbed, resolves it.
4. **`era_authority_audit` must read the entry, not the `###` line** — 459 of
   501 entries are multi-line, and the single-line premise is false. Until it
   does, its `clob_v4` alarm is a false positive, and the two cite instruments
   will keep disagreeing.
5. **`da_cite_audit` should report the rows it could not check** — a
   `NO_AUTHORITY_STRING` row and a row whose authority names no `R-nnn` are
   invisible in the three lists a reader takes as the verdict.
6. **`p4_evidence_audit` needs a `summary_present` field**, and
   `honest_limit`'s *"no larger N would change that"* should be scoped to the
   *"never"* inference, which is what it means and not what it says.
7. **The arms emit default belongs under `data/pm_5min/derived/`**, and until
   it is there the vanishing will continue — four artifacts, one of which is
   now the evidence base of a provisional upgrade.

## What I could not attack

I did not open DE's worktree, did not run `de_section81_arms.py` (importing it
runs the arms; measured pre-reset at 410 s to the feed stage, killed at 900 s),
and did not re-derive R-511(A). Everything above is static analysis, fixture
drives and execution of committed instruments in my own worktree at `7c50926`.
