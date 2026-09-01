# Review — 011 batch + DE first batch + the `1aaac18` closure claims
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `a3e7fc834211cd848421c9dbf6ef1ec4ab37a477`.**
**Request of record:** `REQUEST_011_BATCH_AND_DE_BATCH_2026-09-01.md` (four scopes).
**Composed 2026-09-01T14:50:51Z.** One filing, per R-377.

**Method statement, because this seat changed model family today (R-375).** Everything
below was executed against committed bytes in a detached `git worktree` at the pinned
tip, with the production ledgers symlinked read-only; no seat conversation, filing or
self-report was used as evidence for any verdict. Where a claim is confirmed I say what
I ran and what came back; where I agree with a seat's own claim that is consistency, and
I have tried to make every agreement rest on an independent recomputation rather than on
reading their code and nodding at it. Nothing in the repository working tree was modified:
all mutants were applied inside the lab worktree and reverted, and the lab was verified
byte-clean afterwards.

**Live-mutation statement:** nothing was armed, restarted, stamped, or written to a
production ledger, artifact or tape. The only writes were to a scratch directory and to
`--dry-run`'s own `mkdtemp` output.

---

## Verdict

### HOLD — two findings. The 011 RESULT is verified and is not what the hold is about.

**RR3 is RELEASED. The DE batch is RELEASED with no findings. Scope 4 is ruled below.**

The hold is on **F-1** (the R-280 wiring is protected by a source-text guard only: the
exact defect this batch exists to close survives both the suite and the dry-run seam,
and the artifact it produces reproduces the exact Q-DA-197 contradiction while the run
exits 0) and on **V41-RR1**, whose two operator-path fixes remain uncovered at
`1aaac18` and whose required entry-point fixture is still absent.

I could reproduce no defect in the numbers. Q4's adjudication is arithmetically exact,
independently reproduced from the artifact's own per-window increments, and the
conclusion — *no Q4 cell clears the family-wise bar* — survives every alternative I
tested, including the frozen two-sided form the preregistration actually declares.

---

## Scope 1 — BE's 011 batch (Q-BE-218)

### What I reproduced independently, and it all holds

| claim | how I checked it | result |
|---|---|---|
| six Q4 increments, +278.6c..+3867.1c over 166 windows | re-ran the sign-flip null myself from `economics[budget].increment_by_window`, seed 20260828, 2000 draws, sorted keys | **all six p reproduce to the last digit** |
| best Q4 p 0.0200 → holm 0.1199; no Q4 cell survives | independent Holm step-down over all 24 p, monotonicity enforced | **0 cells disagree** |
| only the six Q1_arrival cells survive | recomputed `status == OK AND holm_p < 0.05` from the cells | **exact match, and the published LIST equals the per-cell FLAGS** |
| 24/24 declared cells, denominator not shrunk | `arms x heads x budgets` recomputed == emitted keys; `holm_denominator == 24` | **holds**, `holm_denominator_is_declared_not_evaluated: true` |
| tape/fragment/topup byte counts | `stat` on the three files | **3,170,987,711 / 1,241,115,096 / 705,063,901 — exact** |
| Q4 paired against the committed incumbent | `incumbent_net_cents +8466.4`, `net_cents +12333.5`, `paired_against_incumbent: true`, increment sum == the cell statistic | **exact**, 94 of 166 windows positive |
| R-306 as A1.4 records it | read the cell against the amendment text | conjunction + worse side present; the tie on p disclosed **as a tie**; the CI clause carried in its R-286 form with *"NO literal interval is claimed: rule 8 forbids one at G=0"* **in the cell** |
| Q-DA-197 F4 (the rule-10 string) | searched the artifact | old `min |calibration slope deviation|` string **gone**; `Q3_cell_rule` now states min-slope / worse side |

### The fixes are red-first where the code is the guard. Five mutants, five kills

Each mutant restores exactly the pre-fix behaviour and nothing else; each was applied in
the lab and reverted.

| mutant | restores | runner selftest |
|---|---|---|
| survivor predicate → Holm alone | Q-DA-197 F2 | **KILLED** (positive control: 24 survivors where 12 are expected) |
| `n_stat = n_actions` | F1 | **KILLED** (Q2 cell states 177,674 behind a 17,604-row AUC) |
| `head = None` (git unreadable) | F5 | **KILLED** (the UNAVAILABLE placeholder must not read as a ref) |
| `run_as_of() -> ""` | F6 | **KILLED** |
| `working_tree_dirty: False` | the dirty flag | **KILLED by an INDEPENDENT read** — this control does not trust the emitter |
| packed block infers widths from the model | blocker (b) | **KILLED** |
| width disagreement unchecked | blocker (b) | **KILLED** |

Blocker (b) — `compact_design` nulling PM/FN/ST — is properly closed: the widths are
recorded, the absence refuses, and a *disagreeing* width refuses too, which is the case
that would still have returned the right COUNT of numbers.

### F-1 — HIGH — the R-280 wiring guard is a source-text check, and the defect it names survives it

`main()`'s call site is protected by three string-presence assertions over its own
source (`"load_verified_incumbent(" in _mn`, `"apply_incumbent(" in _mn`,
`"incumbent=inc_pred" in _mn`). A source guard can see a deleted line. It cannot see a
call whose result never reaches the consumer.

**Executed known-bad (MUTANT A).** One line inserted immediately before the consumer,
every guarded string left intact:

```python
inc_pred = None  # MUTANT A: unwire the incumbent at runtime
rep["economics"] = q4_economics(ap, EVAL[coin]["kept"], incumbent=inc_pred)
```

- `phase2_iter011_run.py --selftest` → **GREEN, 0 failing**;
- `phase2_iter011_run.py --dry-run` → **exit 0**;
- the dry log still prints `[btc/incumbent] INCUMBENT_REWEIGHTED_ONLY applied to 400 rows`,
  because that line is emitted before the unwiring;
- and the emitted artifact reproduces **the exact Q-DA-197 contradiction**:
  `incumbent_null_applicability.comparable.Q4_combined_ev = true`, all six Q4 cells
  `NO_INCUMBENT_COUNTERPART`, and a `q4_incumbent` block still asserting the incumbent
  was loaded and applied.

The `--dry-run` harness is the right instrument and it does run `main()`'s own path — it
is how BE found the packing defect. But it **asserts nothing about its own output**: it
is exempt from the output guard by declared mode and exits 0 on absence. So the harness
that exists to prove the wiring cannot fail when the wiring is cut. That is rule 17's
first half satisfied and its second half — *"an artifact-level guard that refuses output
produced without it"* — still missing, one batch after the same shape was diagnosed.

**Required closure**, and it is cheap because the predicate is already computable at
emission:

1. An artifact-level guard: **if `incumbent_null_applicability.comparable[head]` is true,
   every cell of that head must carry `status == OK`, `paired_against_incumbent: true`
   and a non-null `incumbent_net_cents`, or the run REFUSES.** I verified both directions
   for you: it FIRES on MUTANT A's artifact and ADMITS the real one (rule 16).
2. `--dry-run` must assert its own family before exiting 0 — at minimum that Q4 carries an
   increment — so the seam can fail as well as run.

Nothing in the shipped artifact is wrong because of this. The finding is that the artifact
being right is not currently protected by anything that can fail.

### F-2 — MEDIUM — the adjudicated Q4 null is one-sided; the frozen preregistration says two-sided, and it has never been amended

`ITER011_CONDITIONAL_VALUE_PREREGISTRATION.md` §5(2), frozen: *"null = window-level
sign-flip permutation of per-window paired differences, ≥1000 permutations, **two-sided
p**."* `sign_flip_null` adjudicates `p_value = (ge_one + 1)/(n_perm + 1)` with
`"sided": "one", "alternative": "greater"`.

The substitution is well reasoned (a two-sided test scores `|sum|`, so a candidate losing
by 120c earns the p of one winning by 120c), it is register-recorded (R-286 routed it,
R-288 recorded the closure), and the artifact discloses it in prose in 16 places. Two
things are nevertheless open:

- **the frozen document still says two-sided.** A reader resolving this artifact against
  the preregistration it names finds a contradiction, and rule 4 puts amendments of a
  frozen design with the USER, not in code.
- **R-288's compensating disclosure is not in this artifact.** R-288 records that
  `p_two_sided` "stays as a REPORTED diagnostic so nothing citing it breaks". The string
  `p_two_sided` occurs **zero** times in the emitted artifact, so the frozen-form p cannot
  be recovered from it.

**Impact, measured rather than asserted.** I recomputed the two-sided form for all six Q4
cells and re-ran Holm over the full family with them substituted in:

| | one-sided (emitted) | two-sided (frozen form) |
|---|---|---|
| best Q4 p | 0.019990 | 0.049975 |
| best Q4 holm | 0.1199 | **0.2999** |
| survivors | the six Q1 cells | **the same six Q1 cells** |

**No verdict in this artifact changes.** This is a provenance defect, not a result defect,
and it should be closed as one: a DRAFT-FOR-USER-FREEZE amendment (A2) recording the
one-sided adjudication with R-286/R-288 as its cause, and `p_two_sided` emitted beside
`p_value`. It matters prospectively — on a future run a one-sided p can pass where the
frozen two-sided p does not, and then the discrepancy is load-bearing.

### F-3 — MEDIUM — the six surviving cells sit exactly on a 500-draw permutation floor, and they are the only cells carrying no floor disclosure

Every Q1/Q2/Q3 cell has `p = 1/501 = 0.001996007984…` **exactly** — 0 of 500 matched-random
draws beat the observed. The survivors clear the bar at `holm = 24 × p = 0.047904` against
0.05. Had **one** draw of the 500 beaten the observed, `p = 2/501` → `holm = 0.0958` and
**nothing in the family survives.**

The artifact says this — but only where it does not decide anything. Measured over the
cells: **6 of 6 Q3 cells** carry *"At the PERMUTATION FLOOR … the null is resolution-limited
and more permutations are needed"*; **0 of 6 Q1 cells** carry any floor or resolution
language, and Q1 is the head that survives. The Q1 detail string is one line: *"single
coin, single head: adjudicated on its own matched-random null (prereg §5.1), 1 p available"*.

Note also the asymmetry in resolution: the head that FAILED (Q4) was measured at **2000**
draws, the head that SURVIVED at **500**. A1.6 pins `n_perm = 2000`; at that resolution the
floor would be `holm = 24/2001 = 0.0120` with real headroom instead of 0.0479 against 0.05.

**Required closure:** carry the floor disclosure on every cell whose p equals `1/(n+1)` —
above all on a surviving one — with the one-draw margin stated; and raise the matched-random
draws to the pinned 2000 before this family is cited as evidence of anything.

### F-4 — MEDIUM — the survivor conjunct blocks Q3 for a reason Q3's frozen gate never had

Prereg §3's gates are not uniform. Q1: *"beats the matched-random null AND beats the
incumbent hazard head"*. Q2: *"same, on the fill-conditional population"*. **Q3:
*"calibration slope CI excludes 0 for each, reported separately"* — no incumbent term at
all.**

The six Q3 cells are adjudicated on their own matched-random null, both sides at the floor,
and are then published `survives… : false` because their status is `NO_INCUMBENT_COUNTERPART`
— a status whose stated reason is *"the incumbent has no magnitude head"*. Under the F2
conjunct (a USER ruling, which I am not disputing), a head whose declared gate never
mentions the incumbent can never survive.

The information is recoverable — `cells_passing_holm_but_not_OK` lists all twelve — but the
artifact's headline predicate now answers a question Q3's frozen gate does not ask, and a
reader taking "survivors" as "heads that passed their declared gate" reads Q3 as failed.

**This is a design question, not a defect to fix in code, and it is not mine to rule.**
Routed to the coordinator for the USER: should `NO_INCUMBENT_COUNTERPART` attach to a head
whose frozen gate carries no incumbent term? Meanwhile the cheap mitigation is to report
each head's OWN declared-gate outcome as a separate field from the joint-reading flag.

### F-5 — LOW — `n_actions` carries WINDOWS in the six Q4 cells

F1's fix makes `n_actions` carry the statistic's own n. For Q4 that n is **166 windows**,
while the other eighteen cells carry actions (177,674 / 17,604 / 7,988). `statistic_n_basis`
discloses it honestly, but the field name asserts a unit the value does not have, and A1.5
says *"n for every reported head is the ACTION count"*. A reader comparing `n_actions`
across cells compares windows against actions. Add a `statistic_n_unit`, or stop calling
the field `n_actions` when it is not one.

### F-6 — LOW — 24 declared cells carry 12 distinct results

Q1, Q2 and Q3 statistics are budget-invariant: each is one number replicated across the
three budgets (verified: 2 distinct `(statistic, p)` pairs per head over 6 cells). Only Q4
varies by budget (6 distinct). So "six of 24 cells survive" is **two AUCs, each counted
three times**. This is the frozen design and it is conservative for Holm — a larger
denominator makes survival harder — but the headline reads as more evidential breadth than
exists. State the distinct-result count beside the cell count.

### F-7 — LOW — `working_tree_dirty` reads False when git could not be read, and names no paths

`"working_tree_dirty": bool(dirty)` with `dirty = None` when `git status` fails — an
unreadable status is reported as **clean**, with the disclosure in a separate field
(`working_tree_status_read`). Rule 11 applies to the flag itself. Separately, the flag is
unqualified: this run recorded `working_tree_dirty: true` and a reader cannot tell whether
the dirt touched the producing code. It did not, and I had to use git to find that out —
see Scope 4.

### The ruling the request asked for: does a ref-of-adjacent-commit with matching content identity satisfy R-306's committed-producer rule?

**Yes, here, and I established it by execution rather than by accepting the framing.**

`fit_code_ref = f421bba…` is a MEM commit ("MEM round 5"), which reads alarming, and the
same artifact says `working_tree_dirty: true`. So I hashed the producing set myself:

- all **12** lattice files hash exactly to the artifact's own `fit_code_files` map at
  `f421bba`, `4438961`, `a3e7fc8` and `HEAD`, and their combined hash recomputes to the
  declared `ad535550d366347d`;
- the **runner** (`129718604943682a`) and the **library** (`3fc3a0229a5e358b`) hash-match at
  all four refs as well.

Every byte that produced this artifact is present at the named commit. R-306's rule is met
in substance and not merely in reference. The residual is F-7: the artifact asserts a dirty
tree without naming what was dirty, so the reassurance requires git rather than the receipt.

---

## Scope 2 — DE's first batch (Q-DE-16/17/18) — RELEASED, no findings

| check | result |
|---|---|
| `de_lane4_real_parity.py --selftest` | **35 checks, exit 0**, incl. the matched pair (no-repost PASSES / reposting FAILS `no_fill_after_effective`) |
| the parity receipt | `status: VERIFICATION_ONLY_NO_ECONOMICS_READ`; **nine gates pass**, `failing_slugs: []`, `n_failing_windows: 0` on all nine |
| population, and that no gate passed on an empty set | 471/471 windows admitted, 826,238 generations, 35,083 cancels; the policy acted in every window |
| exclusions are statuses, not drops | `ZERO_LENGTH: 6` and every other status explicitly 0, both at generation and window level |
| no economics leaked | keys `economics` / `pnl` / `net_cents` / `per_cancel` **absent** from the receipt; the only economic-claim string in either results doc is the prohibition itself |
| registry proposal not applied | `contracts/contracts.yaml` blob is **byte-identical at `f421bba` and `a3e7fc8`** (`c8e1151…`) |
| `de_registry_amendment_check.py --selftest` | **26 checks, exit 0**, with two known-bads that FIRE (withdrawn amendment C leaves a consumed type with no producer; an "additive" block redefining an existing type is REFUSED) — and it reports the ownerless ActionSpace field rather than silently fixing it |
| `de_phase4_protocol_check.py` | `consistent`, exit 0; `--selftest` **10 checks**, incl. a doctored-Holm-floor known-bad |
| arms that cannot run | carried as named statuses (`NO_NEUTRAL_REFERENCE`, `NO_RELEASED_PREDICTOR`, `NO_CONTRACT_IDENTITY_FOR_AN_ACTING_CONTROL`), not as absences |

One recommendation, not a finding: **the parity receipt carries content hashes and no commit
ref** (`code_identity`, taken at import — the disclosure itself is good). That is the same
class as Q-DA-197's F5 on the 011 side, which BE has now closed. Adding `carrying_commit`
to the DE receipt would make its binding survive exactly the accident that happened in
Scope 4. I verified all five files hash-match `a3e7fc8` today, but the receipt cannot say so
by itself.

---

## Scope 3 — the `1aaac18` closure claims

### V41-RR3 — **HOLD RELEASED**

The fixture now has the property that makes the scenario real: the fake socket increments
`msg_by_coin[coin]` on every `recv`, i.e. **during** the attempt, not before it. All three
mutations ROUND2 required were executed at the tip against the patched collector:

| mutation | result |
|---|---|
| production diagnostics rewired to the old coin-global delta | **KILLED** — 2 checks fail |
| backoff rewired to the old coin-global condition | **KILLED** — the call-site check fails |
| attempt-local producer (`conn.record`) removed | **KILLED** |

Patch applies clean at the tip; patched `collect_pm.py --selftest` is 40 checks, exit 0.

**One caveat, filed as a caveat and not as a blocker.** The `_market_wiring_probe` is
wall-clock bounded (`WINDOW_S = 1`, real sleeps). I observed **one** failure of
`V41-RR3 WIRING: a socket that DELIVERED 3 then died reports conn_msgs=3…` at 13:57Z that
I could not reproduce in roughly sixty further runs: 0 failures in 11 sequential runs
unloaded, 0 in 40 at four-way parallelism, 0 in 12 under a 12% CPU quota. A control whose verdict depends on a one-second wall
clock can report a false red on a loaded box — and the same dependence means its green is
timing-conditional. Make the probe's completion deterministic (drive the loop on an event,
or assert only after confirming the socket delivered all `deliver` messages) before this
check is relied on as a gate. Separately, `collect_pm.py --selftest` carries a
load-sensitive timing control (*"same gzip ON the loop stalls it"*): **5 failures in 40
runs** at four-way parallelism and **2 in 12** under a 12% CPU quota, against 0 in 11
sequential unloaded runs. Baselines for this suite must be taken unloaded; a red there is
not a regression.

### V41-RR1 — **HOLD NOT RELEASED**

The emitter and the selection fix are real. Selection now follows the earliest actual start
and binds `recv_ns`; my control mutant (revert selection to prefer `in_window`) is **caught
at check 149**, so the lab fires in both directions. The file is byte-identical at
`1aaac18` and the pinned tip, so this verdict is at the tip.

**What is not closed — two claimed path fixes are covered by nothing:**

| executed mutant | restores | v4_1 selftest |
|---|---|---|
| `--post-recovery` scan reverted to `ep` | residual 1 exactly — discovery names an EARLY pid the emitter cannot retrieve, `target_start=None` | **164/164 GREEN** |
| the abort branch's scan reverted to `ep` | residual 3's operator path — an EARLY start never enters `target_starts`, so fail-closed abort emits over a span that ran | **164/164 GREEN** |

And ROUND2's fifth required item is still absent: **the suite never calls `main()`.** The
only references to it in the file are its own `def` and `sys.exit(main())`; the section
labelled "the ENTRY-POINT path" calls `recovery_pid_candidates()`,
`make_recovery_bundle()` and `day_era_admission()` directly. The new check added for
residual 1 is `ok(P.EARLY_SCAN_LOOKBACK_S > 0, …)` — an assertion about a **constant**,
which cannot see which population `main()` actually scans. That is the same shape as F-1
in Scope 1: a text-or-constant proxy standing in for a wiring test.

**Required closure (unchanged from ROUND2, plus the mutants above as its acceptance test):**
drive `main()` in the suite for early-only, early+in-window, same-day late, cross-midnight
late, and the no-start abort, and require each of the two mutants above to fail it.

---

## Scope 4 — the shared-index process finding (Q-DE-19) — ruled

**(a) Does the mixed commit contaminate either batch's provenance for freeze purposes? No,
and this is established by execution, not by reasoning about intent.**

- DE's receipt names five files by content hash; **all five hash-match `a3e7fc8` exactly**
  (`da_replay_parity_battery`, `de_lane4_real_parity`, `harmful_exposure_rows`,
  `harmful_stateful_policy`, `policy_optimizer_queue_realistic`).
- BE's producing set — 12 lattice files + runner + library — hash-matches `f421bba`,
  `4438961`, `a3e7fc8` and `HEAD`, and the combined lattice hash recomputes to the declared
  value.

Both batches are fully reconstructable from committed bytes. The defect is the commit
**message's attribution**, and it is corrected in band by `0b2d57e` without rewriting shared
history — which is rule 3 applied correctly rather than a workaround.

Two consequences worth recording. First, the shared worktree is the most likely reason the
011 artifact carries `working_tree_dirty: true` — DE's 13 paths were staged in the same tree
while BE's run was launching. Nothing produced by that run was dirty (I checked), but the
receipt cannot say so, which is precisely why F-7 asks for the dirty paths to be named.
Second, DE's receipt survives this only because I could re-hash today; a receipt carrying its
own `carrying_commit` would have survived it by itself.

**(b) Mechanism recommendation — recommendation only; the decision returns to the
coordinator/USER.**

**Per-seat `git worktree`.** It is the only one of the three options that removes the shared
index rather than asking four seats to remember something: each seat gets its own working
tree and its own index against one shared repository, and a concurrent `git commit -a` in
another seat cannot see, stage or sweep its files. `git commit -- <paths>` and staging
discipline are corrections that work until the first time a seat forgets, and the failure is
silent and only visible afterwards in someone else's commit message. I ran this entire
review out of two throwaway worktrees; the cost is one command per seat and a symlink for
`data/`.

Second-best, if worktrees are refused: every seat commits with an explicit pathspec **and**
every result-bearing receipt carries its own `carrying_commit`, so a mis-attributed message
can never break the artifact→commit binding — the binding stops depending on the commit
being tidy.

---

## Executed evidence

At the pinned tip `a3e7fc8`, as of 2026-09-01T14:50Z:

| check | result |
|---|---|
| `v5_deploy_gates.py` at the tip | **ALL 16 GATES PASS**, exit 0 (the known-flaky `v4 behaviour` gate passed, 10/10) |
| `phase2_iter011_run.py --selftest` at the tip | **GREEN, 184 checks, 0 failing** |
| `phase2_iter011_run.py --dry-run` at the tip | exit 0, `main()`'s own path, incumbent applied to 400 synthetic rows |
| `v41_boundary_preflight.py --selftest` at the tip | **164 checks passed** |
| patched `collect_pm.py --selftest` (F3 patch applied in the lab) | **40 checks**, exit 0 |
| `de_lane4_real_parity.py --selftest` | 35 checks, exit 0 |
| `de_registry_amendment_check.py --selftest` / `--report` | 26 checks, exit 0 / consistent |
| `de_phase4_protocol_check.py` / `--selftest` | consistent / 10 checks, exit 0 |
| `da_iter011_contract_verify.py --selftest` | **18/21, exit 1 — refuses, as BE disclosed** |
| Q4 p reproduced from the artifact's own increments | **6 of 6 exact** |
| Holm recomputed over 24 cells | **0 disagreements** |
| survivor rule recomputed | **exact**, list == flags |
| producing-code hashes vs `f421bba`/`4438961`/`a3e7fc8`/`HEAD` | **14 of 14 files match at every ref** |
| DE receipt code identity vs `a3e7fc8` | **5 of 5 match** |
| `contracts.yaml` at `f421bba` vs `a3e7fc8` | identical blob |
| tape / fragment / topup bytes vs disk | exact |
| mutants executed | **14** — 11 killed a suite; the **3 survivors** are exactly the two findings (MUTANT A for F-1; the two CLI-scan reverts for RR1) |
| lab worktrees after the review | clean; the repository working tree was never modified |

**On DA's verifier.** `da_iter011_contract_verify.py --selftest` now fails 3 of 21 against
the new artifact, exactly as BE disclosed. Two of the three are discrimination controls that
correctly no longer fire because the defects they detect are fixed. The third is
`holm_p_equals_declared_denominator_times_cell_p`, a positive control that encoded **flat
Bonferroni** as the expected property — valid only while every p was tied at the floor, which
Q-DA-197 itself flagged as the case that could not distinguish the two procedures. The p have
now untied and BE's generator does the correct step-down: I recomputed all 24 adjustments
independently and they match the artifact exactly, and the runner's own suite carries a
check that at least one cell is adjusted by a smaller multiplier than Bonferroni's (18 of 24
here). **The generator is right and the control is stale.** DA's instrument needs its
positive control re-anchored to a step-down; independent contract-level re-verification of
this artifact is owed and this filing is not a substitute for it — my read is independent of
BE, but it is one read.

---

## Disposition

- **HOLD** on **F-1** (artifact-level guard + a dry-run that can fail) and on **V41-RR1**
  (the two uncovered path fixes and the missing `main()` fixture). Both have their
  acceptance tests written above as executed mutants: the fix is closed when those mutants
  fail a suite.
- **HOLD RELEASED: V41-RR3.**
- **DE's first batch: released, no findings** — one recommendation (`carrying_commit`).
- **F-2 through F-7** are filed against the 011 batch and do not, by themselves, hold it;
  F-2 and F-4 need a USER ruling rather than a code change, and F-3 should be closed before
  this family is cited as evidence anywhere.
- **Nothing here disputes the result.** Q4's increment over the committed incumbent is
  positive in all six cells and clears no family-wise bar under either the emitted
  one-sided null or the frozen two-sided one. The decision metric does not survive, and
  that finding is now independently reproduced.
