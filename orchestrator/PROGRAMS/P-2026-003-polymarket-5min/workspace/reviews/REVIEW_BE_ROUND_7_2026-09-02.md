# Review — BE round 7 (BE5-R1 the successor chain; BE5-R2 the pinned allowlist; BE5-R3 the audit ships; CO-12 and CO-13 confirmed)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `fcafe9f`** (row Q-BE-232, landed inside MEM's `1b874f9`); one pathspec
commit, `be_forward_day.py` only, **+299/−6**; driver sha **`857819a76ca5c3a9`** = BE's; 2,580
lines.
**Request of record:** `REQUEST_BE_ROUND_7_2026-09-02.md`.
**Composed 2026-09-02T17:43:58Z.** One filing, per R-377. Round 7 alone, as dispatched.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach fcafe9f`
(`fcafe9f5ef8437319080dd32875bb2238cfd8f78`). **The main tree's `be_forward_day.py` was never
read, run or cited** (R-457 §2 / standing_rule 9): every measurement below is from my worktree's
blob or from a temp copy the module itself builds. `__pycache__` cleared before every execution;
both streams captured to separate files; the file **restored byte-identical** after every mutant
(sha still `857819a76ca5c3a9`) and `git status --short` **0 lines**; nothing written under
`data/` — the `derived/` listing is **identical** before and after, **173 entries**, newest
`2026-09-02 10:34:15`, so **0 entries newer than 16:00Z**; no unit, timer, worktree, anchor,
manifest or candidate touched; `DA_MIDNIGHT_MODE` never set; no re-freeze; no real-day run.

**Citations.** The BE6-R2 citation of record resolves exactly at this tip: `_launch_parity`
`:2481`, `return rc == 0 and child == expect` `:2483`, the paired `ok` `:2532-2534`. **Every line
number in the request resolves at `fcafe9f`** — I checked each one I used (`:970`, `:1040-1069`,
`:1456`, `:1469`, `:1484`, `:1546`, `:1548`, `:1566`, `:1589`/`:1591`, `:2013-2042`,
`:2375-2385`, `:2453`/`:2462`, `:79-87`); none is off.

**One fact from the session, recorded not reviewed.** While I executed, BE's round-8 work landed in
the main tree as `c54e48e` ("the audit attributes on the failure, not on the transcript") and the
main tree is clean again. Nothing in this filing reads or judges it; the CO-12 finding below is
about `fcafe9f`, where it stands.

## 0. What executed at the tip

| launcher | rc | `  PASS` lines | printed total | wall |
|---|---|---|---|---|
| `python3 -m live.pm_research.be_forward_day --selftest` | 0 | **101** | `102 checks OK` | 3 m 36 s |
| `python3 live/pm_research/be_forward_day.py --selftest` | 0 | **101** | `102 checks OK` | 3 m 37 s |
| `--no-such-flag` | **2** | — | usage line on stdout | — |

stderr under each: the two-line numpy-reload warning only (round 6's item 6, unchanged).
The shipped audit inside the suite: **10 cases, 0 survivors**.

## 1. BE5-R1 — the chain (item 1)

`_flush` `:1040-1069` collects `prior = [base, .1, …]`, takes `stands_after = prior[-1]`, and
publishes `supersedes_receipt` (path, sha256, is_base, n_prior) plus `prior_receipts`. The three
suite checks (`:2013-2042`) drive three runs and assert `.2 → .1 → base`, the chain
`[base, .1]` with each sha re-hashed from disk, and every earlier receipt byte-identical after the
third run. Both shipped audit cases kill (case 1 `prior[0]` → red at "the chain is a CHAIN";
case 2 `prior_receipts` dropped → red at "the whole chain is carried"), which I confirmed by
running the audit and by the suite's own green.

**The ruling asked for: `n_prior` is published and unasserted, and I drove it.** Through the
module's own harness (`mutation_audit(cases=…)`, no source edit):

```
"n_prior": len(prior)  ->  len(prior) - 1     rc=0    SURVIVOR
```

The suite stays **green** with a receipt whose `n_prior` contradicts its own `prior_receipts`.
Today the two are built from one list three lines apart so they cannot disagree in fact — this is
a drift surface, not a defect, and it is the same shape DE has been closing all week (a value a
reader is invited to reconcile, with nothing reconciling it). Filed **BE7-R2 (LOW)**; the closure
is one conjunct in a check that already exists.

## 2. BE5-R2 — the pin (item 2)

`DECISION_ALLOWLIST_PINNED` `:970`; the two checks `:2375-2385` assert set-equality **and**
`len(DECISION_ALLOWLIST) == 1`, and that every pinned path carries a non-empty reason. Both shipped
cases kill — driven through the audit, `rc 1` each.

**Ruling on `len(…) == 1`: it is load-bearing, the cost is the right one, and it is the one pin
nothing exercises or documents.** Measured, because the request's premise needed checking: case 4
does **not** widen the pin "to match" — it adds `population.gate` to the **tuple only**
(`:1422-1425`), so the set comparison kills it and the length literal is never reached. I built
the case that does reach it — a *legitimate* second exemption, added to the table **with a reason**
and to the tuple together:

```
DECISION_ALLOWLIST_PINNED = ("gates[].gate", "population.gate")
DECISION_ALLOWLIST = {"population.gate": ("a SECOND exemption, fully documented"), …}
   ->  rc=1
```

Only `len(DECISION_ALLOWLIST) == 1` can refuse that, so the literal **is** the third pin and it
does what BE intends: an exemption from a rule-14 post-condition costs three deliberate edits.
Two things follow. The cost is **not fully stated** — the comment (`:962-968`) names the tuple
("adding a path here fails the membership assertion until this tuple is changed too") and says
nothing about the literal, so the third edit is met as a surprise. And the literal has **no case of
its own** in the shipped table. Filed together as **BE7-R3 (LOW)**; my case above is the missing
one, ready to paste.

## 3. BE5-R3 — the audit ships (item 3)

Everything the item claims, driven:

| property | drive | result |
|---|---|---|
| green baseline required (`:1548-1554`) | usage edit applied to **my** copy, then `mutation_audit(cases=())` | **REFUSED**: "the UNMUTATED copy is not green (rc=1) … A harness whose baseline is red proves nothing about its mutants" |
| a non-locatable case is a survivor (`:1556-1563`) | a case anchored on `"    checks += 1"` (3 hits outside the table) | `applied: false`, `n_anchor_outside_table: 3`, counted as a **survivor** |
| the table's own span excluded (`:1456-1467`) | every shipped case applies with exactly one hit outside it | 10/10 applied |
| the copy, never this file | file sha after the whole battery | `857819a76ca5c3a9`, worktree clean |

**Ruling on the `.git` link — a note, not a guard, and the site in the request is not the one that
does it.** `be_forward_day`'s own git calls run with `cwd=str(REPO)` (`:79`), i.e. against the
**main tree with its own worktree**, so they are not the foreign-worktree case at all. The case is
real one import away: `ev_replay_seam._git` (`:841-845`) runs `git` with
`cwd=Path(__file__).resolve().parents[2]`, which in a copy is the **temp tree**, whose `.git` is a
symlink to the main repository. Measured in a tree built by `_audit_tree` itself:

- `git rev-parse HEAD` → rc 0, the main tree's HEAD;
- `git status --porcelain` → rc 0, **3,194 lines**, every real path reported `T` (typechange) —
  the copy's symlinks compared against the main index;
- `ev_replay_seam.produced_at()` from inside the copy → `produced_at_commit` = the main HEAD,
  `working_tree_dirty: true`, `dirty_paths` listing the temp tree's own entries.

So the provenance a copy would publish is wrong — but nothing publishes from an audit child, and
the path is **not reached** by this audit: `produced_at()` is called only from `ev_replay_seam`'s
`receipt()` (`:677-710`), which the driver's suite does not call. On the index: after a full
baseline child the main repository's `.git/index` was **byte-identical** (sha `e503382cdbc8f2fb`
before and after), and in a controlled repeat a deliberate foreign-worktree `git status` left the
content unchanged as well. I therefore have **no evidence of a write** and will not assert one; the
residual worth a sentence in the module is the **lock**, not the stat cache — four seats share this
repository, and a `status` that decides to refresh takes `.git/index.lock`. If you want it gone
for nothing: **do not link `.git` into the copy**. Nothing the audit exercises needs it
(`_git_blob` and `_provenance` both use `cwd=REPO`), and a copy that is not a repository would
answer `git_readable: false`, which is the honest answer.

## 4. CO-12 — confirmed, with the calibration the severity turns on (item 4)

Reproduced at the tip through `mutation_audit(cases=…)`, no source edit:

| case | rc | `died_at_named_check` | reading |
|---|---|---|---|
| the usage edit, `want` = `R5(1) KNOWN-BAD: a SECOND run into the same outdir` (a check that **passes** at stdout :40) | 1 | **True** | the mutant dies at `BE34-R4`, is credited to a check it never reached |
| the same edit, `want` = a string printed **nowhere** | 1 | **False** | the substring test is **not** vacuous |
| `n_prior − 1` (§1) | **0** | True | `died_at_named_check` is True on a mutant that did not die at all |

That third row is the sharpest statement of the defect: the flag is a property of the **transcript**,
not of the death — it can read True when `rc == 0`. The kills themselves are real (every shipped
case returns rc ≠ 0, and the middle row shows a `want` that never appears is refused), so what is
unproven is exactly what the row and the BE5-R3 message assert — *"each mutant dying AT ITS NAMED
CHECK"*. **Confirmed at LOW-MEDIUM**, as routed; I contest nothing. The closure you routed —
attribute on the `AssertionError: <label>` line on **stderr**, streams separate, plus a named
control both directions — is the right one, and my second row is half of that control already.

## 5. CO-13 — confirmed, and the arithmetic prose is now false (item 5)

**101 `  PASS` lines, `102 checks OK`, under both launchers.** At the source: the audit block's
`ok(…)` increments `checks` inside the closure (`:1590`) and `:2462` increments it again. One
assertion, two counts. **Confirmed at LOW.**

**Ruling on the launch-parity arithmetic: the predicate is untouched, the sentence is not.** The
predicate compares `child == at_entry` with `at_entry` captured before the spawn, and the audit
block runs after `_selftest_launch` returns, so nothing about the comparison changes — measured,
the launch check reads *"the child counted **99** = this parent's count on entry (**99**)"* and
passes. But the same message ends *"so its total is ours minus this one check"*, and the parent's
total is **102** — the child plus **three** (the launch check, the audit's assertion, and CO-13's
phantom increment). Two of those three are legitimate; the sentence is wrong either way, and it was
already wrong the moment the audit block landed, independently of CO-13. This is the prose I
accepted in round 6 on the ground that `at_entry` is captured; it has now drifted exactly there.
Filed **BE7-R1 (LOW)**.

## 6. The two textual corrections (item 6)

Both confirmed at the artifacts, no finding sought:

- Q-BE-232 reads **"85 → 102 checks"**. Round 6's tip was **95** (Q-BE-231, and my round-6 filing
  measured 95 under both launchers), so round 7 is **95 → 102**; 85 is round 5's figure.
- `:1387-1390` says *"The siblings are symlinked so the copy imports the same modules"*, while
  `:1503-1507` **copies** `.py` files and symlinks the rest — and the inline comment immediately
  above the code explains why symlinking them failed. The header comment contradicts both the code
  and its own later note.

## 7. Counts and discipline (item 7)

One pathspec commit of the one file (`git show --stat`), sha as stated, 102/102 under both
launchers with CO-13's caveat, usage rc 2, `derived/` identical at 173 entries with nothing newer
than 10:34Z, no unit/timer/anchor/manifest/candidate touched, my worktree clean and the file
byte-identical after every mutant.

## Findings

| id | severity | where | one line |
|---|---|---|---|
| BE7-R1 | LOW | `:2535-2540` | the launch message says the parent's total is "ours minus this one check"; it is the child **plus three** |
| BE7-R2 | LOW | `:1060`, `:2030-2037` | `n_prior` is published and asserted nowhere — `len(prior) − 1` is a **survivor** |
| BE7-R3 | LOW | `:2375-2377`, `:962-968`, `AUDIT_CASES` | the third pin (`len(…) == 1`) is load-bearing but has no case and is not documented |
| BE7-R4 | LOW-MEDIUM | `:79`, `:85-91` | `_provenance` names the **main tree's** commit and dirtiness whatever tree executed |

**BE7-R1.** Predicate correct, sentence false; the module's own rule is that a message computes
what it claims. Closure: interpolate the difference (`checks - at_entry` at the point of printing
is not available inside `_selftest_launch`, so the honest form is to drop the clause and keep
`child == at_entry`, which is what the check actually proves).

**BE7-R2.** Driven: `rc 0`, survivor. Closure: add `and r4["supersedes_receipt"]["n_prior"] ==
len(_chain)` to the chain check at `:2030-2037` — one conjunct, and the shipped audit gains a case
for free.

**BE7-R3.** Driven both ways: the shipped case 4 dies on the set comparison (it widens the tuple
only), and a *legitimate* second exemption — table **and** tuple, both with reasons — is refused
`rc 1` by the length literal alone. So the literal is the third pin; it deserves the clause in the
comment that the tuple already has, and the case I ran above in `AUDIT_CASES`.

**BE7-R4.** Measured from my worktree at `fcafe9f`: `_provenance()` returns
`carrying_commit c54e48ea7fae7315…` — the **main tree's** HEAD — with `working_tree_dirty` read
from that tree, while `driver_sha256_prefix` correctly names the file that ran
(`857819a76ca5c3a9`). The git helper hardcodes `cwd=str(REPO)` (`:79`). A receipt written from a
worktree therefore claims a commit it did not carry and inherits another tree's dirt — and BE
builds in `~/ctaNew-wt-be`, so this is the normal case, not the exotic one. This **predates the
round** (the code is untouched by `fcafe9f`) and I file it because the receipt is the artifact of
record and the programme has already ruled the point one seat over: DA's CO-10 closed exactly
"the artifact must name the tree that executed". Closure is one word — run those two git calls with
`cwd=Path(__file__).resolve().parents[2]`, the tree `_selftest_launch` already resolves; the object
store is shared, so `_git_blob`'s freeze reads are unaffected.

## Corrections of my own

None. One premise I checked rather than inherited: the request's item 3 attributes the
foreign-worktree git calls to the copy of *this* module; they come from `ev_replay_seam`, and this
module's own calls run in the main tree by construction (§3). And item 2's "the pin widened to
match" is not what case 4 does (§2). Neither changes the round's substance; both changed what I
measured.

## Disposition

**RELEASE `fcafe9f`.** The chain is a chain and driven three receipts deep; the exemption is
pinned, with the membership assertion and both shipped cases killing; the audit ships, refuses a
red baseline by name, counts a non-locatable case as a survivor, and never touches the source —
which I verified by sha after eight mutants of my own. **CO-12 confirmed (LOW-MEDIUM)** with the
extra evidence that `died_at_named_check` can read True at `rc == 0`; **CO-13 confirmed (LOW)**,
101 assertions against a printed 102. Four findings, all strengthenings: three are one-line
closures inside checks that already exist, and BE7-R4 is a pre-existing provenance bug the
programme has already ruled elsewhere. Route them with CO-12/CO-13 to BE's next round.
