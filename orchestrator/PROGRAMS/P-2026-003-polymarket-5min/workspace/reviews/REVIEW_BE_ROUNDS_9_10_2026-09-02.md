# Review — BE rounds 9 + 10 at `ff60d0a` (the receipt names the tree that executed; the check that was a function of the branch)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `ff60d0a`** (row Q-BE-235 `5b4eb06`), over round 9's `90638c3` (Q-BE-234).
Verified at the blob: `be_forward_day.py` sha16 **`871f5633d9855f74`**, **3,138 lines**,
`AUDIT_CASES` at `:1509` holding **26** tuples.
**Request of record:** `REQUEST_BE_ROUNDS_9_10_2026-09-02.md`. **Composed 2026-09-02T21:02:12Z.** One filing, per R-377.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach ff60d0a` (`data/pm_5min`
mirrored per entry) and in one scratch worktree of my own for item 2's reproduction;
`~/ctaNew-wt-de` and `~/ctaNew-wt-be` never read; the main tree's `be_forward_day.py` and the DE
modules never read, run or counted. No `--forward-day` against any day. `__pycache__` cleared
before every execution; both launchers; streams separate; **nothing mutated** — `git status --short`
**0** in my worktree throughout. `derived/` **173 before and after**; nothing written under
`data/`; no unit, timer, scope-with-timer or anchor; `DA_MIDNIGHT_MODE` never set;
`da_midnight_verify.sh` never run; no plan file edited. Every time below is from `date -u`.

## 1. Counts, executed — CONFIRMED (item 1)

| run | start | rc | wall | `  PASS` lines | summary | `git worktree list` |
|---|---|---|---|---|---|---|
| `-m live.pm_research.be_forward_day --selftest` | 20:27:07Z | **0** | **641 s** | **121** | `121 checks OK [sha 871f5633d9855f74]` | 34 |
| `live/pm_research/be_forward_day.py --selftest` | — | **0** | **641 s** | **121** | same, same sha | 34 |
| `--no-such-flag` | — | **2** | — | — | usage | 34 |

Both under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`; battery
20:27:07Z → 20:48:29Z; **34 worktrees before and 34 after**; `derived/` 173 → 173. The two
launchers ran the **identical check sequence** (diff of the PASS-label stream: empty). The audit's
own line reproduces verbatim: *"BE5-R3 the shipped mutation audit runs GREEN: **26 cases, 0
survivors, each mutant dying AT ITS NAMED CHECK** against a copy"*, and the CO-12 end-to-end
falsifier (one edit, two names) passes with it. R-477 §1 confirmed.

## 2. BE9-C1 — **CONFIRMED closed**, with a ruling on what remains git-dependent (item 2)

**The mechanism.** The check reads the executing tree's HEAD and **HEAD~2** (`:2684-2690`), makes a
detached worktree there, asserts the clean worktree names **its own** HEAD (`:2704-2712`), then
**plants** `# BE7-R4 planted difference` into the copy (`:2721-2722`) and asserts the flag flips
while the commit is unchanged (`:2724-2730`). HEAD~2 rather than HEAD~1 because at HEAD~1 the
commit read from the worktree and `HEAD~1` of the executing tree are the same string, so the swap
mutant is an equivalent substitution — BE's comment (`:2685-2689`) says it survived the audit, and
case 14 of the table is now that exact mutant.

**RULING: the FLIP is a function of the code alone; the check's PRECONDITIONS are not.** The planted
marker cannot be a no-op, so the round-9 failure mode — running bytes equal to the committed bytes —
is closed by construction, and the case "HEAD~2's driver equals the running bytes" is now harmless.
What still depends on git state is the *setup*: `rev-parse HEAD~2` must resolve. In a tree with
fewer than three commits, or a shallow clone of depth < 3, `_git_in` (`:2680-2682`) returns
`stdout.strip()` **without checking the return code**, so `_prev` is `""`, `git worktree add`
fails, and the suite goes red at `:2701` with an unexplained `returncode` rather than a named
refusal. It fails closed, which is the right direction, and it fails **mute**, which is not.
Filed **BE10-R1**.

**RUN B's condition, reproduced in my own scratch worktree — not the main tree.** I found a commit
whose HEAD~2 carries the running bytes: `874a041` (HEAD~1 `b7edcc1`, HEAD~2 `e791f4f`), where
the driver blob is `81392a9e7880` at **all three** commits and the running file's sha16 is
`871f5633d9855f74` — the pinned bytes exactly, in the git state that broke round 9.

```
20:48:46Z  worktree add --detach <scratch> 874a041 ; data/pm_5min mirrored
           path launcher, MemoryMax=8G
20:59:27Z  rc=0  640 s  121 PASS  "121 checks OK [sha 871f5633d9855f74]"
```

**121 / 121 holds there.** The fix is real, and — the point for item 5 — the condition it had to be
proved against **did not require the shared main tree**.

## 3. BE9-C2 — **CONFIRMED closed**; the receipt does say which root (item 3)

`EXEC_TREE()` (`:39-48`) is `Path(__file__).resolve().parents[2]`, and `_provenance` emits a
`roots` block (`:118-127`) naming `code_and_anchors` (the executing tree), `data` (absolute, with
`data_is_absolute_because`) and `git_objects` (the shared store). The two PASS lines confirm it in
execution: `:106` *"code and ANCHORS resolve in the executing tree (/tmp/…/wt) while `data/` stays
absolute by design"*, `:110` *"the audit copy links the EXECUTING tree's entries (20 of them) while
`data` is linked from the absolute root"*.

**RULING: the design is right, and for a stronger reason than the one given.** "A bare worktree
carries no `data/`" is true but incidental; the load-bearing reason is that the tape and the market
ledger are **shared program state**, not code — a per-checkout data root would let two seats'
receipts name the same commit while disagreeing about the population. And the receipt does not make
a reader know the rule: the roots are named in the artifact. One refinement, **BE10-R4**: `roots`
does not say whether `EXEC_TREE()` **equals** `REPO`. From the main tree the two coincide, and a
reader cannot tell whether a path resolved by rule or by coincidence; one boolean
(`exec_tree_is_repo`) closes it.

## 4. BE9-C3 — **CONFIRMED closed**; every git write enumerated (item 4)

| line | invocation | effect |
|---|---|---|
| `:104` | `git rev-parse HEAD` / `status --porcelain` (`_provenance`) | read (status may refresh the index) |
| `:139` | `git show <ref>:<path>` | read |
| `:2663` | `git rev-parse --git-common-dir` | read |
| `:2668` | **`git worktree add --detach <tmp>/be-r10-c3-stale HEAD`** | **write** — plants the stale entry |
| `:2681` | `rev-parse HEAD / HEAD~1 / HEAD~2` (`_git_in`) | read |
| `:2697` | **`git worktree add --detach <tmp>/wt <HEAD~2>`** | **write** — its own scratch worktree |
| `:2789` | **`git worktree remove --force <wt>`** | **write** — removes its own |
| `:2794`, `:2802` | `git worktree list --porcelain` | read |
| `:2805` | **`shutil.rmtree(<git-common-dir>/worktrees/be-r10-c3-stale)`** | **write, not via git** — deletes its own admin entry |

**`git worktree prune` is never called** — it exists only as a mutation in `AUDIT_CASES` (case 17),
which must die. So the check's own claim at `:2790-2793` ("the only git writes this selftest
performs are the creation and removal of its own") is **true**, counting the direct `rmtree` as the
removal of its own.

**RULING: the `rmtree` at `:2805` is acceptable, and it is the right choice.** `git worktree
remove` refuses a stale entry and `prune` is precisely the shared-state destruction the check
exists to forbid, so a direct deletion is the only remaining route. Two properties make it safe and
both hold: the path is built from `git rev-parse --git-common-dir` plus **the entry name this
selftest planted**, so it cannot name another session's entry; and it is `ignore_errors=True`.

That second property is also where the one leak lives — **BE10-R2**: `_gcd` (`:2663-2665`) takes
the command's stdout without checking its return code. An empty result is not absolute, so
`(_me_tree / "").resolve()` yields the executing tree, the `rmtree` targets a path that does not
exist, `ignore_errors=True` swallows it, and **the planted stale entry survives the selftest** —
silently, in the one check whose subject is not leaving shared state behind. A named refusal on an
empty `--git-common-dir` closes it.

One ordering note, not a defect: the `rmtree` (`:2805`) precedes the `ok(...)` that asserts the
entry survived (`:2807`), which is correct because the assertion reads `_listed`, captured at
`:2802` — but a reader meets the deletion before the check it enables.

## 5. RUN B's tree, and the rule (item 5)

**(a) It was `/home/yuqing/ctaNew`, the shared main tree — and no checkout was made.** `d62d114`
is **MEM's row commit** ("Q-MEM-42: round-54 true-up filed") at **19:26:12Z**; `4112484` is **MEM
round 54** at 19:25:50Z. The main tree's `.git/logs/HEAD` holds 1,735 entries of which **6 are
checkouts, none in that window** — the two entries there are commits. So the tree was at `d62d114`
because that commit was made in it, and BE ran the selftest in the tree as it stood. No third tree
is needed to establish this and I read nothing of BE's.

**(b) Is that inside the discipline? No — not as a matter of course**, for three reasons that have
nothing to do with intent. (i) The selftest performs **git-admin writes on the shared repository**
(§4: two `worktree add`, one `worktree remove`, one direct `rmtree` under `.git/worktrees/`) —
the very class BE9-C3 exists to police, executed in the tree every seat shares. (ii) The main tree
is where other seats' WIP lands; the runner cannot know it is clean. (iii) It is the landing
surface: the reflog shows commits at 19:25:50Z and 19:26:12Z, inside the window BE ran in, so the
run competed for the index lock with another seat's commit. **What makes RUN B legitimate is that
it was declared** — Q-BE-235 names the tree, the HEAD, the HEAD~1 and why — **and left no residue**,
which the coordinator verified independently.

**(c) The rule I propose:**

> A seat's selftest runs in that seat's own worktree. Where a check's condition can only be produced
> in another tree — the shared main tree included — the run is **declared in the row before it is
> made** (tree, HEAD, the condition, and why no other tree produces it), performs no write outside
> its own git-admin entries, and is verified afterwards from a third tree by `git worktree list`
> and `git status --short`. Convenience is never the reason; a condition is. And a seat that finds
> itself needing the shared tree should first look for a commit that reproduces the condition in its
> own — which is usually there.

That last clause is not rhetorical: §2 above reproduces RUN B's exact condition at `874a041` in a
scratch worktree, 121 / 121. **The exception was avoidable**, and the rule can be strict at no cost.
This is distinct from standing_rule 9, which is about seats READING another seat's WIP; this is
about a seat WRITING git-admin state in a tree it does not own.

## 6. The 26-case audit — CONFIRMED (item 6)

26 tuples at `:1509`, matching the request's inventory. **The round-9 survivor is case 14**:
`_wt_head = _git_in(_wt, "rev-parse", "HEAD")` → `_git_in(_me_tree, "rev-parse", "HEAD~1")`,
named to *"BE7-R4 a CLEAN worktree reads clean and names ITS OWN HEAD"* — killable **only** because
the worktree is anchored at HEAD~2; at HEAD~1 it is an equivalent substitution. Case 13 is the
round-9 defect itself (the plant replaced by a plain copy) and case 17 the prune mutant. The audit
fails on a survivor **and** on a mutant dying at the wrong check: `:2967-2969` asserts
`survivors == []` **and** `baseline_green` **and** `all(died_at_named_check)`, and `:2986-2989`
is the CO-12 falsifier (one edit under two names, both `died_at` equal). The audit copy links the
**executing** tree's 20 entries (`:110`).

Worth recording: **every finding I filed in rounds 6 and 7 now has a case of its own** — BE7-R2
(#18), BE7-R3 (#19), BE6-R1 (#20, #21), BE6-R2 (#22), BE6-R4 (#23), BE6-R7 (#24), BE6-R6 (#25).
They are not merely fixed; each has a mutant that must die at a named check.

## 7. Q-BE-234, corrected in band (item 7)

Q-BE-234 claimed, at `90638c3` (`bc8e366b7a1d6aa7`), **117 checks = 117 PASS lines, rc 0, under
both launchers**, usage rc 2, audit **21 cases / 0 survivors / 386 s**. What Q-BE-235 corrects is
not the arithmetic but its **durability**: R-465 measured rc 1 after 93 checks at the same commit,
in a scratch worktree and in the main tree, because the flip check depended on the running bytes
differing from the bytes at the worktree's commit — true in BE's tree at 18:36Z, false from the
next commit onward.

So, as stated: the **117/117 and the 21/0/386 s stand as history**, at that as-of and in that tree,
and **do not stand as properties of the code** — which is precisely what Q-BE-235 says, and it
reproduces the failure before fixing it. The third claim, *"the launch parity is the file"*
(BE6-R2's closure), **does stand**: it is case 22 of the audit and the suite's own summary prints
the child's sha (`[sha 871f5633d9855f74]` in both of my runs).

## 8. What the coordinator missed — the class (item 8)

A parse scan over the 121 checks finds **no predicate that cannot go red**: every `ok(False, …)`
is a red-only label (the strongest form — it exists to fire when an expected refusal did not
happen), and the single `ok(True, …)` (`:2018`) sits in an `except` and is reached only when the
tampered-byte import actually raised. **One weakness there**, filed as **BE10-R3**: the handler is
`except (ForwardDayRefused, Exception)`, so **any** exception — a `NameError` in the fixture
included — reads as "the tampered byte refused". Narrow it, or assert the message.

Checks that read git state rather than the code: the BE7-R4 block, by design (it is about git
state), and its two silent-failure paths are BE10-R1 and R2. No receipt field lacks its population
(`roots` carries its reason; the audit reports cases/survivors/attribution). `main()` (`:3104`)
has no exception surface to miss: `run_forward_day` catches broadly and records the refusal in the
receipt, and `--selftest` deliberately lets an `AssertionError` out.

**One observation from my own run, which is evidence for §4-§5 rather than a finding:** my scratch
worktree's teardown left nothing (`be10_runb_wt` absent from `git worktree list` and from
`.git/worktrees/`), and the count read 35 at 20:59:27Z and 34 at 21:00:00Z — another session's
selftest held its own scratch entry for those seconds. The shared worktree list **fluctuates while
other seats run**, so "count unchanged" is only assertable at quiescence, which is why the request
says before and after, never during.

## Findings

| id | severity | where | one line |
|---|---|---|---|
| BE10-R2 | LOW-MEDIUM | `:2663-2665`, `:2805` | an empty `--git-common-dir` resolves to the executing tree, so the planted stale entry **leaks silently** — the one path where this selftest can leave shared state behind |
| BE10-R1 | LOW-MEDIUM | `:2680-2682`, `:2690` | `_git_in` ignores the return code, so fewer than three commits or a shallow clone fails the check **mutely**, two lines from where the cause is |
| BE10-R3 | LOW | `:2017-2020` | the tampered-byte known-bad accepts **any** exception as proof of the refusal |
| BE10-R4 | LOW | `:118-127` | `roots` does not say whether the executing tree **is** `REPO`, so a reader cannot tell rule from coincidence |

**BE9-C1, BE9-C2, BE9-C3: all three CONFIRMED closed**, each at the PASS lines and each with a
mutant in the audit that must die at it. None contested.

## Disposition

**RELEASE `ff60d0a` as round 11's base.** 121 / 121 under both launchers with the identical check
sequence, usage rc 2, the 26-case audit green with attribution asserted, 34 worktrees before and
after, `derived/` 173 → 173, nothing under `data/`. The round did what round 9 could not: the flip
is planted rather than borrowed from the branch, the anchor moved to HEAD~2 so the swap mutant is
killable, the roots are named in the receipt, and the only git writes are the selftest's own two
entries — which I enumerated rather than took on trust, and reproduced RUN B's condition for in a
worktree of my own.

**Nothing here must precede round 11.** The durable landing is a data-side act with no code edit;
none of BE10-R1..R4 touches the landing path — R1 and R2 are in the BE7-R4 block, R3 in a
materialisation fixture, R4 in a receipt field's completeness. Route them to **round 12** with
BE8-R1 and BE8-R2. If BE opens the file for any reason before the landing, R2 is the one to take
with it: it is the only finding whose failure mode is shared state left behind, and it is a
three-line refusal.
