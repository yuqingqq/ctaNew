# Review — BE round 6 (BE34-R1 one fixture two consumers; BE34-R3 the parity that could not fail; BE34-R4/R5; three corrections in band)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `5e9ed91`** (covering `faaabdc` and `6913445`); row Q-BE-231 at `32d1116`.
**Request of record:** `REQUEST_BE_ROUND_6_2026-09-02.md` (`4daf878`).
**Composed 2026-09-02T17:06:56Z.** One filing, per R-377.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach 5e9ed91`; `~/ctaNew-wt-be`
never entered. HEAD `5e9ed9106f07c883e0512dc80a9455d12c64ce13`, `git status --short` **0 lines**
after the battery, and the file restored **byte-identical** to the pin (md5 `918c9330…` = the pin's
blob). `__pycache__` cleared before **every** execution (R-446 §1); both streams captured to
separate files, never piped. Read-only under `data/`: the `derived/` listing (184 entries,
full-iso mtimes) **identical** before and after. No re-freeze, no anchor/manifest/candidate edited,
no real-day run into `fwd5/` or any canonical outdir; every drive wrote to a `TemporaryDirectory`.
The one mutant that spawns the shared tree (J9) ran with `PYTHONDONTWRITEBYTECODE=1`; I verified
the launcher's own `dict(os.environ, …)` propagates it (`sys.dont_write_bytecode True` in the
child, 0 `__pycache__` dirs created), so nothing of mine was written into `/home/yuqing/ctaNew`
— the 16:47:26–40Z pyc cluster there is not mine.

## 0. What executed at the tip

| launcher | rc | checks | launch check |
|---|---|---|---|
| `python3 -m live.pm_research.be_forward_day --selftest` | 0 | **95** | child **94** = parent's count on entry (94) |
| `python3 live/pm_research/be_forward_day.py --selftest` | 0 | **95** | tree named = `/home/yuqing/ctaNew-wt-rev` |

The round-5 red is gone: at that pin the same suite gave 84 from this worktree plus a red 85th.
**BE34-R3 is closed as filed**, and the parity check now also refuses a child spawned from
another tree reporting 4242.

## 1. BE34-R1 — the three mutants, and the value classes the fixture omits (item 1)

My three round-3/4 mutants are **red by name**, each driven at the tip with the cache cleared:

| mutant | rc | checks | died at |
|---|---|---|---|
| streamed score `+ 1e-9` | 1 | 65 | `BE34-R1 ONE fixture, TWO consumers: … EQUAL …` |
| every second streamed row dropped | 1 | 65 | same check |
| `if built["reconciliation_failures"]:` → `if False and …` | 1 | 70 | `BE34-R1 KNOWN-BAD: the CALLER refuses the whole DAY by name` |

**The ruling asked for — yes, there are classes the fixture omits; I drove all five.** Both
consumers were run on the same rows, through `_r1_installed`, at the tip:

| class | `build_and_score` | `score_rows` | agree? |
|---|---|---|---|
| 0 — the committed fixture | 11 scores | 11 scores | **yes** |
| A — a coin **absent from `frozen["fits"]`** | btc only | btc only | **yes** |
| A2 — **no** coin has a fit | `{}` | **REFUSES** ("zero actions scored") | different **in kind** |
| B — a window fails reconciliation | its rows unscored | **scores them** | **no**, by design |
| C — every row featureless | `{}` | **REFUSES** | different in kind |
| D — `row["t0"]` ≠ the t0 parsed from the slug | slug's t0 | row's t0 | **no** |
| E — a NaN feature, identical on both sides | `nan` | `nan` | **`==` says no** |

Reading, in order of what it costs:

- **A is the real day's dominant class and it is not in the fixture.** Both fixture coins carry
  fits, so the `else` branch at `:843-845` (actions counted, nothing scored) is never compared. The
  driver itself discloses `coins_supplied_without_a_fit`; on 09-01 that is 5 of 7 coins. Driven,
  the two consumers **agree**, so this is coverage, not divergence — one line (a third coin, no
  fit) makes the branch part of the comparison. Filed **BE6-R7 (LOW)**.
- **A2/C is the honest answer to "a way the two agree that a real day would not": they do not
  agree, they differ in kind** — the reference REFUSES, the replacement returns an empty book.
  That is not a defect, because the caller refuses one level up (`:1169-1173`) — **but nothing
  drives that refusal.** I drove it through the real chain: rc 1, receipt `REFUSED`, the message
  verbatim. Filed **BE6-R3 (LOW)**, and the same drive produced **BE6-R1** below.
- **B is by design** (the streaming pass excludes a `bad` window, `score_rows` knows nothing of
  `bad`), and the day refuses anyway — but it means the asserted equality holds **only in the
  clean regime**, which the fixture's comment does not say. Worth a clause, not a finding.
- **D** is inert while `score_rows` is dead code; the fixture makes the two t0 sources agree by
  construction. Worth knowing if `score_rows` is ever revived as the reference.
- **E is a property of the predicate**: `_bs["scores"] == _sr` is exact float equality, so two
  consumers computing the **same** NaN are declared unequal (measured: `False`, while the same
  object compares `True`). Nothing in the fixture can produce NaN today; if a NaN class is ever
  added, the check goes red on agreeing consumers. The `DISTINCT and small` check is the right
  instinct; a `math.isnan`-aware comparison (or an explicit no-NaN assertion) completes it.

## 2. The reconciliation failure fails the DAY, through the production caller (item 2)

Confirmed at the artifact. The drive (`:1960-1990`) substitutes `build_and_score` in module
globals and calls **`run_forward_day`** — the production entry, not a helper — then reads the
**receipt from disk** and asserts rc ≠ 0, `outcome REFUSED`, "failed reconciliation" + "fails the
DAY", `refused_at`, and that **no `reconciliation: PASS`** is recorded. The `sys.modules` snapshot
around it is real (see item 6).

**"Does the refusal name the window?"** At the consumer, yes — the check above it names the eth
t0s (1788000300 excluded, 1788000900 still scored). **In the artifact a real refusal leaves, no.**
`build_and_score` counts `recon_fail` and keeps no slug, so the receipt says "1 of 4 windows" and
an operator cannot tell which. Filed **BE6-R4 (LOW)** with the file's own idiom as the closure.

## 3. BE34-R3 — reproduced, and the predicate is a count, not the file (item 3)

**The requested reproduction.** Mutant J9 (`tree = REPO`, the pre-fix hardcoded root) applied in
my worktree: **rc 1, red by name** — *"the child counted **101** = this parent's count on entry
(**94**)"*. It is a no-op in the shared tree and a killer here, exactly as BE says.

**What that reproduction also shows.** `_launch_parity` is `rc == 0 and child == expect`: a
**count**. So I built two trees that are not this one — `live/pm_research/*.py` copied, `data` and
`orchestrator` symlinked — and pointed `tree` at each:

| divergent tree | md5 of its driver | child count | verdict |
|---|---|---|---|
| same checks, one comment appended | `a23b6777…` (parent `918c9330…`) | 94 | **PASS — survivor** |
| same, plus the printed total `{checks + 1}` | `3ea58e0e…` | 95 | red by name |

The second row is my own instrument's positive control (rule 15): it fires. The first is the
finding — **a child running a file the parent is not running passes whenever the counts agree**,
which is precisely the condition BE34-R3 was opened about. Filed **BE6-R2 (MEDIUM-LOW)**.

**Ruling on "rc and count in one check".** BE's arithmetic is right but it is about the *message*,
not the predicate: `at_entry` is **captured** before the spawn (`:2205`), so the comparison
`child == at_entry` is unaffected by how many checks follow; splitting into two `ok()` calls would
make the parent's total `at_entry + 2` and the prose "ours minus this one check" false. The
load-bearing reason is the one the comment gives — one condition, so weakening either half breaks
the other — and I accept the shape on that ground.

## 4. BE34-R4 and BE34-R5 (item 4)

Both hold, and I drove the mutants rather than reading them:

| mutant | rc | checks | died at |
|---|---|---|---|
| `main` ignores its parameter (`argv = list(sys.argv)`) | 1 | 73 | `BE34-R4 … the message matches the argv PASSED` |
| `why` says modules "execute at HEAD" | 1 | 34 | `R-421(2)/BE34-R5 the closure DECLARES its method` |
| `closure_method` renamed (J14 itself) | 1 | 34 | same check |

Two things worth recording. Under the argv mutant the **rc half PASSES** (`_rc_usage == 2 and
_rc_day == 2` still holds, because both calls fall through to usage) — the **message** check is
the half that kills it, as its comment claims. And **re-entry cannot recur**: `sys.argv` is
neutralised to `["be_forward_day.py"]` around the calls (`:2001-2006`), so a `main` that ignores
its parameter finds no `--selftest`; measured, the mutant run printed the suite's first PASS line
**once** — one suite, no nested spawn.

## 5. The three corrections, and the 5-vs-4 (item 5)

- **(a) verified at the artifact.** R-442 is committed `7b08679` at **2026-09-02T14:38:16Z**,
  entry stamped 14:37Z, ruling *"all six USER decisions ruled"*. Q-BE-230's as-of is **14:52Z** —
  later. So the disposition column was **stale when filed**, not overtaken, and BE's reading is
  correct. Carrying the correction as a **new row** (Q-BE-231) rather than an edit is rule 13's
  shape; Q-BE-230 is untouched at `32d1116`.
- **(b) confirmed.** `768465a` is one file, **1 insertion / 1 deletion**, and the word-diff shows
  it rewriting the mutation-count parenthetical **inside the landed Q-BE-230 row**
  (`50 → 49 → 47 mutants` becomes `run THREE times … 50: 5 survived; 49: 3 survived; 47 at the
  final`). BE names it as its own fault and the rule it restores (append-only) is the right one.
- **(c) reconciles by name, and is not checkable here.** The four passes are internally coherent —
  `mr5` 13 = 9 + 4, `mall` 50 = 44 + 6 **lines** (5 distinct, H14 twice), `mall2` 49 = 46 + 3,
  `mall3` 47 = 47 + 0 — and 44 + 6 = 50 is what exhibits the duplicate. But **no harness and no
  log is in the repo** (no `MUTS` list under `live/pm_research/`, no audit log in the workspace),
  so I can confirm the arithmetic and not the run. **Ruling: accept the reconciliation by name
  this round**; it becomes checkable when BE5-R3 ships the audit (round 7), and that is where it
  should be re-stated, not re-litigated here.

## 6. The `sys.modules` restores and the numpy reload (item 6)

The three restores (`:1521-1529`, `:1698-1736`, `:1967-1976`) share one shape: delete every module
**not** in the snapshot, then re-install the snapshot. That is wider than the hazard it exists for.

**Attributed, not inferred.** Running the suite under `-W error::UserWarning` turns the warning
into a traceback: the reload fires at **`:1568`**, the check that re-imports `harmful_exposure_rows`
after the **first** restore, down the chain HER → `policy_optimizer_queue_realistic` →
`policy_bounds_v1` → `flow_fill_development:31` → `numpy/__init__.py:531 _reload_guard()`. So the
snapshot at `:1521` was taken **before** numpy was first imported, and the restore evicted a
C-extension package that has nothing to do with the tmpdir.

**Severity: LOW, and the reason is ordering, not design.** Nothing in this suite carries a numpy
object across the boundary — the fixtures are lists of Python floats — so the documented hazard
(`isinstance` against a type object from the other numpy, dtype/warning registries) has nothing to
act on today. It is a latent trap in an instrument whose whole purpose is to be trusted when it
goes green. Filed **BE6-R6 (LOW)** with the narrow restore as the closure: evict only names whose
`__file__` lies under the tmpdir root, which keeps exactly the isolation the comment at `:1515`
asks for (the anchors imported from the run dir) and leaves the tree's modules alone.

## 7. The 63-mutant audit, and J15/J16 (item 7)

**The audit is at the previous bytes.** The sha it names, `dd3739d655c1620b`, is the file at
**`6913445`**; the released tip is `957a9d3cc38b3dde`. The +13 lines that closed J14 are precisely
what the audit could not have covered. BE's row is not misleading about this (it says J14 closed
at `5e9ed91`), and I drove two mutants of the new check red myself (item 4), so nothing is
unguarded — but the released bytes carry no audit. Filed **BE6-R5 (LOW)**. The totals themselves
are not reproducible here: no harness, no logs (item 5(c)).

**Ruling on J15/J16.** The regress is real: a falsifier's falsifier is unfalsifiable at some level,
and any guard added to kill J15/J16 has the same property one level up. **BE's stopping point is
the correct one, because it is empirical rather than asserted** — J10/J11 survived a 61-mutant
audit before the negative half existed and are killed after, which is a measured statement that
the half is load-bearing (rule 15's spirit: the instrument was seen to change an outcome). Two
caveats. First, the argument licenses stopping, not weakness: **BE6-R2 is not a mutation artifact**
— the same-count divergence is a real class the predicate admits, and closing it is not a step in
the regress. Second, process fault (iii) — `grep -c KILLED` reading 60/63 because a mutant's label
contains the word — is rule 10 in miniature: the shipped audit should **compute** its verdict
counts from verdict-initial lines, not grep for vocabulary (my own DE-round experience of a silent
regex mismatch is the same failure).

## 8. Discipline (item 8)

Three commits, **one file each**: `faaabdc` +238/−23, `6913445` +76/−15, `5e9ed91` +13, all
`live/pm_research/be_forward_day.py`. Nothing under `data/pm_5min/derived/` (184-entry listing
identical before/after — my count includes the `total` and `.` lines, the coordinator's 173 counts
files; same listing, different convention, no discrepancy to resolve). No re-freeze; no anchor,
manifest or candidate touched; `da_midnight_verify.sh` not run; no timer, service or unit touched;
`COORDINATION.md` not written by me. On the round-7-vs-8 labelling: agreed, no correction needed.

## Findings

| id | severity | where | one line |
|---|---|---|---|
| BE6-R1 | MEDIUM-LOW | `:1220`, `:1158`, `:1170` | `refused_at` names the last gate that **PASSED**, for both day-level refusals |
| BE6-R2 | MEDIUM-LOW | `:2192`, `:2207-2241` | the launch parity compares a **count**, so a byte-different tree with the same count passes |
| BE6-R3 | LOW | `:1169-1173` | the caller's zero-score refusal has **no falsifier**, with the harness two lines away |
| BE6-R4 | LOW | `:821-823`, `:1158-1163` | a refused day's receipt never says **which** window failed reconciliation |
| BE6-R5 | LOW | tip vs `6913445` | the released bytes carry **no** mutation audit (the 63 ran at `dd3739d6…`) |
| BE6-R6 | LOW | `:1521-1529` (+ two) | the restore evicts modules unrelated to the tmpdir; numpy is reloaded at `:1568` |
| BE6-R7 | LOW | `:1247-1249` | the fixture omits the real day's dominant class — a coin with **no** frozen fit |

**BE6-R1 — `refused_at` names a gate that passed.** `rec["refused_at"] = rec["gates"][-1]["gate"]`
is right when the exception comes from inside `gate()` (that entry is the REFUSED one) and wrong
for the two **bare raises** after a gate completed. Driven: substituting a `build_and_score` that
returns clean counters and `scores: {}` refuses the day correctly (rc 1) and the receipt carries
`('rows_and_scores_streamed','PASS'), ('reconciliation','PASS'),
('bridged_windows_equal_row_windows','PASS')` **and** `refused_at: bridged_windows_equal_row_windows`
— the same gate, PASS and blamed. The reconciliation case has the same shape and the selftest
**pins** it (`:1981` asserts `refused_at == "rows_and_scores_streamed"`), so today the
misattribution is asserted as expected behaviour. Error direction: never a pass that should be a
refusal; a reader — or an automated resolver, which is rule 13's stated reason receipts have
fields — is sent to a gate that passed. Closure: set `rec["refused_at"]` at each bare raise (or
route both checks through `gate()` so the REFUSED entry is appended), and have the selftest demand
the **check's** name.

**BE6-R2 — the parity is a count, not the file.** Reproduction in §3. The message already prints
the tree it spawned, which is how the divergence becomes visible to a human — but rule 10 is that
the predicate must compute what the message claims. Closure in the file's own idiom: have the
child print its module sha (`_sha_file` is already the file's way of naming bytes) and require
`child_sha == _sha_file(Path(__file__))` for the same-tree spawn; the stub, which prints none,
stays refused by the same predicate.

**BE6-R3 — a refusal without its falsifier.** `:1169-1173` fires correctly (driven: rc 1,
`outcome REFUSED`, the message verbatim), and it is the guard that stands between an empty book
and a "scored" day when the frozen artifact's fits do not cover the supply. Closure: the harness
the round already built —
`globals()["build_and_score"] = lambda sel, fr: {…, "scores": {}, "windows_with_rows": set()}` —
plus the receipt assertions, three lines beside the reconciliation drive.

**BE6-R4 — the count without the name.** Closure: carry the failing slugs bounded, the way
`assert_window_sets_agree` already returns `bridged_without_rows_examples[:8]`, and quote them in
the refusal.

**BE6-R5 — the audit's bytes.** Closure: re-run the audit at the released bytes (or, better, land
BE5-R3's shipped audit and let the suite run it), and state the sha the audit ran on in the row —
which BE already does, correctly, which is how this was checkable.

**BE6-R6 — the wide restore.** Mechanism, attribution and closure in §6.

**BE6-R7 — the fixture's missing class.** Closure: a third fixture coin present in the windows and
absent from `_R1_FROZEN["fits"]`, so the `else` branch at `:843-845` is inside the equality. Driven,
the consumers agree today; the point is that the check would not notice if they stopped.

## Corrections of my own

None this round. My round-5 statement — that the suite's launch check was red in this worktree
because the child ran the shared tree's file — is confirmed at the artifact by the J9 reproduction
(child 101 vs parent 94); what I add here is a precision I did not have then, not a retraction:
the predicate that catches it is a count, so the same divergence is invisible whenever the counts
agree (BE6-R2).

## Disposition

**RELEASE `5e9ed91`.** Everything the round claims, I reproduced: the two consumers are compared
on one fixture and three mutants of that comparison die by name; a reconciliation failure fails the
day through the real caller and its receipt records no PASS beside the refusal; the child is
spawned against the tree of `__file__` and the pre-fix hardcoded root is red here; usage returns 2
through `main(argv)` with re-entry structurally stopped; the disclosure is asserted by content and
J14 is dead. None of the seven findings is a wrong number — two are greens that can mislead
(BE6-R1, BE6-R2), four are guards or coverage missing beside guards that exist, one is a latent
process-state trap. Route them to BE round 7 alongside BE5-R1/R2/R3; BE6-R2 and BE6-R1 first.
