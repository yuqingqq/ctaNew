# Review — DA round 17 (DA16-R1 closed: the identity conjunct gets a driven falsifier, the predicate stated once)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `e353119`** — HELD, the eighth commit on DA's chain; row Q-DA-213 at
`3a3f5d3`. My rounds 15 + 16 filing (RELEASE `3b7e10a`, CO-10 CONFIRMED CLOSED) stands; this is
the addendum's round-17 diff and its mutants.
**Request of record:** `REQUEST_DA_ROUND_17_2026-09-02.md`.
**Composed 2026-09-02T18:02:39Z.** One filing, per R-377. **This filing precedes tonight's landing dispatch**, so
per R-454 §4 the disposition below names `e353119`.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach e353119`
(`e353119a0dc391fc17edb64152976a06b1313425`); `~/ctaNew-wt-da` never entered. The main tree's
`be_forward_day.py` was never read, run or cited (standing_rule 9) — nothing in this round touches
it. `__pycache__` cleared before **every** execution; both launchers; both streams captured to
separate files — **this module's FAIL line goes to STDOUT**, so a stderr-only reading would have
called every red run green. The file restored **byte-identical** after each mutant
(md5 `9cc83a6974c69944e34ba4a70f6b8949` = the blob at `e353119`), `git status --short` **0 lines**.
Nothing under `data/`: the `derived/` listing is **identical** before and after, **173 entries**.
`DA_MIDNIGHT_MODE` never set; `da_midnight_verify.sh` never run in any mode; no unit, timer or
anchor touched; the full gate runner never invoked (`--selftest` only, R-449's scope caveat).
`git worktree list` **33 before and after every single run**, including the six mutants.

**Scope confirmed at the object.** One file, `da_blackout_mask.py`, **+57/−2, four hunks**, the
earliest changed line **888** against `def selftest` at **493** — every hunk inside the suite,
production untouched. `da_governed_verdict_preflight.py`, `v5_deploy_gates.py`,
`pm_tape_density.py` and `da_forward_day_verify.py` are **byte-identical to `3b7e10a`** (empty
diffs). The `!= _there` control (`:967-972`) is unchanged — the diff carries no `_there` deletion.

## 0. What executed at the tip

| suite | rc | checks |
|---|---|---|
| `python3 -m live.pm_research.da_blackout_mask --selftest` | 0 | **38** |
| `python3 live/pm_research/da_blackout_mask.py --selftest` | 0 | **38** |
| `da_governed_verdict_preflight --selftest` | 0 | 39 |
| `v5_deploy_gates --selftest` | 0 | 5 |

Zero stderr under each. Roster arithmetic recomputed from the module: **23 declared + 15 twins =
38**, with **8 excluded** — seven "not a `--selftest` module gate; no second launcher exists to
derive" and `tier1 normalisation` **by name** in `TWIN_EXCLUSIONS`. DA's four cells reproduce.

## 1. The predicate stated once, two call sites (item 11)

`_names_the_executing_tree` is defined at **`:892`** and called at exactly **two** sites —
`:951` (CO-10 CONTROL) and `:1023` (DA16-R1 FALSIFIER, negated). There is **no second spelling**:
every `carrying_commit` comparison in the module is either inside that function (`:903-904`), the
separately-held `!= _there` control (`:967`), or the falsifier's own fixture check (`:1018`). The
round-16 ruling's control is intact and unchanged.

## 2. The four mutants, reproduced — and two of my own (items 11–12)

Each applied to my worktree copy, cache cleared, both streams captured, file restored:

| mutant | rc | red at | traceback | worktrees |
|---|---|---|---|---|
| identity conjunct dropped, producer intact | 1 | **`DA16-R1 FALSIFIER`** | none | 33 → 33 |
| producer answers `HEAD~2`, control intact | 1 | `CO-10 CONTROL` | none | 33 → 33 |
| producer answers `HEAD~1` | 1 | `CO-10 CONTROL` | none | 33 → 33 |
| identity dropped **and** producer `HEAD~2` | 1 | **`DA16-R1 FALSIFIER`** | none | 33 → 33 |
| **mine:** the fixture's text anchor no longer matches the producer | 1 | `DA16-R1 fixture: and it really does answer HEAD~2` | none | 33 → 33 |
| **mine:** the `!= _here` conjunct dropped | **0** | — **GREEN, 38 checks** | none | 33 → 33 |

Row 1 is the finding closed: at `3b7e10a` that same edit was **silent**, and here it is red by
name. **DA16-R1: CLOSED.** I contest nothing.

**Ruling on the text anchor (item 12): sufficient as it stands, and I drove the case rather than
arguing it.** The concern is real in kind — `'"rev-parse", "HEAD"]'` occurs four times (`:134` the
producer, `:797` the parent's own `_here` read, `:886` the suite's `_child_head` read, `:1012` the
fixture literal), and `.replace(…, 1)` takes the first only because the producer precedes the
suite in file order. So I made the first occurrence stop matching the producer (respelled `:134`
behaviour-preservingly) and ran the suite: the child's producer then answers `HEAD`, and the
fixture check at `:1018` goes **red by name** — *"and it really does answer HEAD~2 … checked, so
the falsifier below is refusing the thing it names"*. That is the difference from CO-11's second
half, where a mis-hit produced an unparsable copy and a traceback with nothing asserting the
arrangement: here the arrangement is **asserted before the falsifier fires**, in both directions
(`:1014` it ran at all, `:1018` it really answers the fourth commit), so every way the edit can
miss — wrong hit, no hit, broken copy — is a named red. A structural locator (the `_head_commit`
FunctionDef's span, BE's `_paste_into_reader` idiom) would be tidier and buys no verdict that the
two fixture checks do not already produce. **No change asked for.**

## 3. The conjunct that is now the one without a driver (item 11, and my finding)

Dropping `and prod.get("carrying_commit") != _here` from the predicate leaves the suite **green at
38** (row 6 above). It is not a hole: the precondition at `:905-912` asserts `_child_head not in
(_here, _there)`, so `carrying_commit == _child_head` already **implies** `!= _here` — the
conjunct cannot change a verdict while that precondition holds, in either the assertion or the
falsifier direction. What it does do is carry a claim the code does not:

> Two conjuncts on two DIFFERENT commits: the first is what a producer answering some other commit
> fails, **the second is what a producer answering the parent's fails**.

A producer answering the parent's HEAD fails the **first** conjunct — the case the message assigns
to the second is unreachable. By this round's own doctrine ("a conjunct whose removal nothing
noticed is what a driven falsifier fixes, and a further inequality does not") the `!= _here`
conjunct is now precisely that conjunct. Filed **DA17-R1 (LOW)**; two one-line closures in §Findings.

## 4. Discipline and the landing (item 13)

34 → **38** both launchers; preflight 39; gates 5; roster 23 + 15 = 38 with 8 excluded;
`derived/` identical at 173 entries; `git worktree list` 33 before and after every run — the
suite's throwaway child worktree is created and removed each time, and its own
`ok(not _wt_path.exists())` (`:1062`) asserts it; no stray tree in `/tmp`; `DA_MIDNIGHT_MODE`
never set; `da_midnight_verify.sh` never run; no unit, timer or anchor touched; register not
written.

One ordering question I checked rather than assumed: the falsifier leaves the child's copy mutated
(an uncommitted `HEAD~2` edit), and the DA10-R5 control that follows (`:1030-1055`) reads `_prod`
— the DIRTY arrangement's emission, captured **before** that edit — so nothing downstream reads a
producer that was deliberately made to lie.

## Findings

| id | severity | where | one line |
|---|---|---|---|
| DA17-R1 | LOW | `:904`, `:952-959` | the `!= _here` conjunct is silent when removed, and the control's message assigns it work the precondition makes unreachable |

**DA17-R1.** Measured: the suite is green at 38 with the conjunct gone. It is redundant rather than
wrong — `_child_head ≠ _here` is asserted at `:905-912` before the control runs — so I am not
asking for it to be deleted; belt-and-braces is cheap and it would matter if that precondition were
ever weakened. What should change is one of two things, either a single line: **(a)** make the
message say what the predicate does (the identity conjunct refuses *any* commit that is not the
child's, the parent's included, and the second conjunct is a guard against the precondition being
lost), or **(b)** give it its own driver the way this round gave one to identity — a producer
answering the PARENT's HEAD, asserted `not _names_the_executing_tree(...)`, which is CO-10's
original shape and would make the removal red. (b) is the closure that matches this round's
argument; (a) is the honest minimum under rule 10.

## Corrections of my own

One, caught before it reached this filing and worth recording because it is the exact trap MEM
named in R-449 §3(b): my first read of the roster reported **2 twins**, because
`_launch_twins(GATES)` returns `(twins, excluded)` and I measured `len()` of the tuple. Unpacked,
it is **15 twins and 8 excluded**, and 23 + 15 = 38 as DA reports. The wrong number was mine, not
DA's; I state it so the next reader of my numbers knows which reading is the checked one.

## Disposition

**RELEASE `e353119`.** The scope is the suite only and the other four files are byte-identical to
`3b7e10a`; the predicate is stated once and read from two call sites; the identity conjunct now has
a driver that is red by name under both the bare drop and the layered mutant; the wrong-commit
arrangement is an uncommitted edit for the stated reason and its two fixture checks make a missed
or mis-placed anchor red rather than silent — which I drove. Both launchers 38, zero tracebacks, no
worktree left behind, nothing under `data/`. **DA16-R1: CONFIRMED CLOSED.** One LOW finding, a
message and a driverless conjunct, for DA's next round — nothing that should hold tonight's
landing.
