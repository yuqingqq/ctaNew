# Review — DE rounds 33 + 34 as ONE filing at `47a2ba6` (the run path, the head-scoring module, the protocol check)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `47a2ba6`** (row Q-DE-52 `68f63aa`) on top of `6d04833` (Q-DE-51
`2b72d02`). Shas verified at the blob: runner **`def7053dcc8291a7`** (1,271 lines),
`de_head_scoring.py` **`a074c150a1f2155d`** (335), protocol check **`bd9755c90296b171`** (286).
**Request of record:** `REQUEST_DE_ROUNDS_33_34_2026-09-02.md`. **Composed 2026-09-02T19:09:49Z.**
One filing, per R-377.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach 47a2ba6` (`data/pm_5min`
mirrored per entry); `~/ctaNew-wt-de` and `~/ctaNew-wt-be` never read — round 35 received nothing
from here; the main tree's `be_forward_day.py` never read, run or counted (standing_rule 9, BE
round 10 open). `__pycache__` cleared before every execution; both launchers; streams captured
separately. **The declared OUTDIR was never passed to `--run`** — my two drives used scratch paths
and `data/pm_5min/derived/phase4_diag_r459` **remains absent**. `derived/` **173 before and
after**; nothing written under `data/`; no plan file edited; no unit, timer, scope or anchor;
`DA_MIDNIGHT_MODE` never set; `da_midnight_verify.sh` never run; `git worktree list` unchanged;
worktree clean.

## 1. Counts — CONFIRMED (item 1)

| module | `-m` | by path | `  PASS` lines | stderr |
|---|---|---|---|---|
| `de_head_scoring` | 21, rc 0 | 21, rc 0 | **21** | 0 |
| `de_phase4_diag_runner` | 58, rc 0 | 58, rc 0 | **58** | 0 |
| `de_phase4_protocol_check` | 22, rc 0 | 22, rc 0 | **22** | 0 |

PASS lines = summary = rc 0 under both launchers, R-468 §1's figures reproduce.

## 2. The claimed closures — present, driven, or not (item 2)

| claim | line | verdict |
|---|---|---|
| DE33-C2 thresholds from `causal_thresholds` | `de_head_scoring:161-186` | **driven** — and closed **for both heads**, not only LGBM: `linear_d_btc.json` carries `causal_thresholds` keyed `5%/10%/15%` (0.6356 / 0.4353 / 0.3199), the LGBM map likewise |
| DE33-C5 `ARM_SPEC` + two guards | `:82-96`, `:581-587`, `:589-595` | **present and load-bearing**; no live `th.get(arm, 0.5)` anywhere (the only hits are a comment `:81` and a refusal message `:1115`) |
| DE33-C8 outdir refusal under the CLI | `validate_outdir:335-363`, `run():1171`, `main():1255-1263` | **driven twice**, with scratch paths only: an existing scratch dir → **rc 2**, and a non-existent scratch dir → **rc 2**, both *"is not the declared output directory"*, **zero tracebacks, nothing created**, declared OUTDIR still absent |
| DE33-C9 `TRANCHE_NO_MARKOUT` counted | `:236`, `:284-288` | **present** — and untested; see DE34-C2 |
| DE32-R3 the ladder as sets | protocol check `:226-244`, known-bad `:264` | **closed** — three-way set equality (addendum, runner, frozen §4) plus an explicit both-directions check; the direction that passed green at `e52d183` is now asserted |
| DE33-C1 (the claimed half) | `de_head_scoring:65-90`, `:113-158` | **driven** — the incumbent loads at **60 normalisers / 61 hazard weights** and the head under test at **106 features**, each refusing at 1 / w−1 / w+1 by name; `verify_fit_code` runs before either |

## 3. DE34-C1 — **CONFIRMED**, and the consequence is wider than a traceback (item 3)

`_head_scorer` (`:1228-1243`) is round 33's stub, unchanged, and `score_events_for` (`:1217-1225`)
still passes it as the scorer — so `run()` (`:1191`) scores through it. The new module's scoring
API has **no call site**: of `de_head_scoring`, the runner uses only `thresholds` (`:164`) and
`verify_fit_code` (`:1150`, inside the suite); `load_incumbent`, `score_incumbent`, `load_lgbm`
and `score_lgbm` are called nowhere in the runner. `LightGBMError` is not in `main()`'s except
tuple (`:1260-1261`). **The consequence R-468 states holds**: a `--run` against the declared OUTDIR
feeds for ~29 minutes and then tracebacks at the first LGBM cell, because the stub passes
`[[row["t"]]]` — width 1 against a head fitted on 106.

**And the half that would not traceback is worse.** The stub's incumbent branch (`:1238-1243`) reads
`linear_d_{coin}.json`, then returns `float(sum(coefs.values()) * 0.0 + 0.5)` — **a constant 0.5**,
the fit multiplied by zero. So the incumbent arm would score every generation identically: with
`theta_for` at 0.4353 (btc, 10%) it cancels **everything**, and no refusal fires anywhere. A cell
ordering that reached the incumbent first would produce a complete, plausible, meaningless arm. The
constant is also the one `ARM_SPEC`'s second guard exists to forbid, reached by a different door.

**On the closure.** A named preflight refusal before the feed is the right shape and I would make it
stronger in one respect: preflight should not merely check that a head *loads*, it should **score
one real row through the same code path the run will use** (`de_head_scoring.score_lgbm` /
`score_incumbent` on a real-width vector taken from the first generation), because that is the only
check that fails when the scorer is a stub. A load-only preflight passes on this tip's code: the
booster loads fine; it is the *call* that is wrong.

## 4. DE34-C2 — **CONFIRMED**, and the predicate is doubly inert (item 4)

`:1146` reads `ok("TRANCHE_NO_MARKOUT" in build_reference.__doc__ or True, …)`. Two defects, not
one: the `or True` makes it unfailable, and **even without it** the condition tests a **docstring**,
not the counting the label claims ("tranches with no markout are COUNTED under their own status").
**No check anywhere exercises the status**: `TRANCHE_NO_MARKOUT` occurs at `:236` (initialised),
`:286` (incremented) and `:1146` only — R-468's reading is right, and `TRANCHE_KEPT` is in the same
position. Closure in the module's own idiom: build a two-generation fixture where one tranche has
`markout_cents_per_share = None`, call the reference builder, and assert the two counters.

## 5. DE34-C3 — **CONFIRMED**; what the protocol check should assert (item 5)

The diff `6d04833 → 47a2ba6` removes exactly three horizon checks —
`ok("FILL_HORIZON_S = 1.0 s" in text …)`, `ok("estimand_horizon_s" in text …)` and
`ok(_RUN.FILL_HORIZON_S == 1.0 …)` — and adds one net check, so `EXPECTED_CHECKS` 24 → 22 is
arithmetically right (24 + 1 − 3). **Nothing in the file discloses the removal**; the only surviving
"24" is the unrelated iteration-011 multiplicity arithmetic (`:68`, `:128`, `:166`).

**Ruling, given EST-R2** (verified in R-468 §2: the cell is the generation-level over-the-hold
number `DRAFT:68` prescribes, and the receipt's `fill_horizon_s = 1.0` is what misstates it). The
protocol check should assert the **binding**, never the value:

1. a cross-module pin — `_RUN.FILL_HORIZON_S == phase4_generation_tables.FILL_HORIZON_S` — which
   introduces no number and restores the property the removed check had;
2. **the receipt's declared horizon equals the horizon the addendum declares**. That check **fails
   today**, correctly, because the addendum declares none — which is the point: it names the open
   item instead of hiding it, and it goes green the moment addendum v2 says what the number is.

Removing the horizon checks silently is the opposite move: it makes the protocol check quiet about
the one estimand question the round is open on.

## 6. DE34-C4 — **CONFIRMED**, measured, and the pin design (item 6)

Computed myself from the manifest (`fit_code_ref e12e2c70c133a003…`, **12** `fit_code_files`):
**two have moved** — `harmful_exposure_rows.py` (`c2e40100ddf3f7a1` → `1bbd8e7525fc27ac`) and
`phase2_arms.py` (`3249dfc61c31b8d2` → `ab19f5c639333bdc`); the other ten are byte-identical,
**including both files `PINNED_CODE` names** (`:54`). So the pin passes because it names two files
that did not move, and says nothing about the two that did — one of which,
`harmful_exposure_rows.py`, is **the feed the run uses** (`select_v2_era`, `replay_with_recorder`,
`generation_table`, and the markout arithmetic that defines a tranche).

**Is the present selection a verification of anything the run will call?** As a **fit-provenance**
pin, yes and coherently: those two files are the arithmetic that fitted the heads, and naming them
in the docstring with their shas is honest. As a **wiring** verification, no: `de_head_scoring`
imports neither module — it re-implements the incumbent's arithmetic in-file (`:113-131`) and
applies the LGBM artifact through the booster — so the pin verifies bytes the applying path does
not execute.

**Recommended design:** compute the pin instead of listing it — take the run path's own import
closure ∩ `fit_code_files` and verify each, so the set cannot drift from what executes; report the
files that moved as **named statuses with their reason** (rule 4) rather than by omission; and where
a moved file is admissible because the function the run calls is byte-identical (R-468 measured
`_feature_pass`'s def bytes identical by AST), make **that** the declared exemption, per function,
with the AST equality computed — never implied by a selection that happens to match. A whole-file
pin at `e12e2c7` for files nothing imports would be pageantry; a function-level pin only for the
functions the run calls is the honest version.

## 7. What the coordinator missed — read for the class (item 7)

I ran an AST scan over all three modules for checks whose predicate cannot go red, and read for the
other four classes the request names.

**(a) Predicates that cannot go red — three more beyond `:1146`:**

- **`:1085` (DE34-R1, MEDIUM).** `ok(MAX_CANCELS_PER_MINUTE == float("inf") and "the frozen
  protocol names no rate limit" in (cell_params.__doc__ or "") + open(__file__).read(), …)` — the
  second conjunct **greps the file's own source for a sentence**, so it can only fail if someone
  deletes the comment; and **the sentence it pins is false**: `DRAFT:71` names
  `max_cancels_per_minute` as declared per cell with the identity `requested = passed + suppressed`
  reported (my EST-R4, verified in R-468 §2). The suite now defends an incorrect claim about the
  frozen document.
- **`:1043` (DE34-R2, LOW-MEDIUM).** `ok("build_reference" in globals() and "select_v2_era" in
  build_reference.__doc__, …)` under the label "the reference is built from
  `harmful_exposure_rows`' OWN pieces" — a **docstring** stands in for the behaviour. The module's
  own AST idiom (used at `:958-970`) would assert the calls resolve to `HER.*`.
- **`de_head_scoring:271` (DE34-R3, LOW).** `ok(True, f"and the two synthetic rows score …")` —
  disclosed honestly in its own message ("reported rather than asserted different, because a tree
  ensemble may legitimately map two synthetic rows to one leaf"), but it still spends one of the 21
  on a check that cannot fail. Attach the two numbers to the width/range check that can.

**(b) A constant that is a policy input in disguise — DE34-R4 (MEDIUM).** `:648`
`res = arm_result(reference, ctrl_scores, c, theta=0.5)`: the control's threshold is the file's own
0.5, the exact class `:589-595` refuses for every treated arm ("a policy constant is an input").
It is **inert today** — the control's scores are all 1.0 so any theta below 1 cancels the same set,
and `theta_repost = 0.25` never fires because the control emits no later event — and it becomes
**live** the moment EST-R5's fix gives the control per-generation scores and repost parity. Bind it
to the treated arm's own theta when that lands.

**(c) A docstring over absent code.** Round 32's instance is closed (`tranche_table` is really
called, `:1068`), but the class recurs in (a): `:1043` and `:1085` are both prose standing where a
predicate belongs, and `:1146` is prose *and* unfailable.

**(d) Refusal classes `main()` does not catch — DE34-R5 (MEDIUM).** The tuple at `:1260-1261`
enumerates the five in-project classes and none of the collaborators': `lightgbm`'s
`LightGBMError` (**live today** — DE34-C1's traceback), `harmful_stateful_policy.InvalidParameter`
(reachable whenever a bound threshold makes `theta_repost < theta_cancel` false — not with today's
0.32–0.71, but it is a fit-supplied number), and `phase4_generation_tables.UndeclaredEstimand`
(not reachable while every call passes `declare_cap=True`). A refusal that reaches the CLI as a
traceback is DE33-C8 one door over.

**(e) `validate_receipt` runs BEFORE anything is written — verified.** `:1210` validates,
`:1211` `out.mkdir(parents=True, exist_ok=False)`, `:1212-1213` writes; the directory itself is
created only after the receipt validates, so a receipt missing a binding field leaves no directory
behind. This one is right and worth recording as such.

**(f) One more source-text check — DE34-R6 (LOW-MEDIUM).** `:961-970` asserts substrings inside
`run_cell`'s source (`"res = arm_result("`, `'vals.append(res["cost_adjusted_value_cents"])'`,
`"harm_by_slug" not in`). It is AST-scoped to the function, which is better than a file grep, but it
is still one-directional: a reformat across two lines turns it **red** for no reason, and the same
regression under a renamed variable passes **green**.

## 8. My open findings at this tip (item 8)

| finding | status at `47a2ba6` |
|---|---|
| **DE32-R1** (the budget never reaches the replay) | **CLOSED — and for both heads**, not only LGBM: `theta_for` → `HS.thresholds` (`:160-168`, `de_head_scoring:161-186`), and the incumbent's `causal_thresholds` are budget-keyed |
| **DE32-R3** (the grid binding one-directional) | **CLOSED** (protocol check `:226-244`), driven |
| **DE32-R4** (arm identity is the caller's key) | **CLOSED** — `ARM_SPEC` `:82-96` and the two guards `:581-595`; `spec` carried into the receipt `:599` |
| **DE32-R2** (the degenerate 200-draw null) | **CLOSED AS FILED** — the pool is the real generations keyed `slug\|side\|gen` (`:629-631`) and each draw is valued by a replay (`:648-650`), asserted at the source (`:961-970`). Its residual survives in DE31-R2 |
| **DE31-R2** (a forced draw passes silently; no freedom reported) | **OPEN** — the null block calls `MRC.draw` + `refuse_if_not_random` and reports `n` and quantiles; nothing measures or reports per-stratum freedom, so a degenerate real pool would still read as an interval |
| **DE31-R1** (rho reachability at `gen_start + L`, frozen Cap 1 says `t + L`) | **OPEN in the estimator, INERT at this call site** — the runner's score events are stamped `t = g["t0"]` (`:1219-1221`), so the decision row's `t` *is* the generation start here and the two references coincide. It becomes live the moment a score event is stamped anywhere but generation start |
| **DE32-R5** (row/docstring ahead of code) | the round-32 instance is closed; the class recurs — DE34-R1/R2 and C2 |

## Findings

| id | severity | where | one line |
|---|---|---|---|
| DE34-R1 | **MEDIUM** | `:1085` | a check greps the file's own source for a sentence about the frozen protocol — and the sentence is false (`DRAFT:71`) |
| DE34-R4 | **MEDIUM** | `:648` | the control's `theta=0.5` is the file's own policy constant, the class `:589-595` refuses for every other arm |
| DE34-R5 | **MEDIUM** | `:1260-1261` | `main()` catches five in-project refusal classes and none of the collaborators' — `LightGBMError` is live today |
| DE34-R7 | **MEDIUM** | `de_head_scoring:54`, `:65-90` | the pin verifies two files the applying path never imports, and is silent about the two that moved — one of them the feed |
| DE34-R2 | LOW-MEDIUM | `:1043` | a docstring stands in for the behaviour the label claims |
| DE34-R6 | LOW-MEDIUM | `:961-970` | a source-substring check: red on a reformat, green on a renamed regression |
| DE34-R3 | LOW | `de_head_scoring:271` | `ok(True, …)` spends a check on a report |

DE34-C1 **CONFIRMED** (with the constant-0.5 incumbent half above), DE34-C2 **CONFIRMED**,
DE34-C3 **CONFIRMED**, DE34-C4 **CONFIRMED**. None contested.

## Disposition

**RELEASE `47a2ba6` as the base of round 35 — and not as a runnable tip.** What rounds 33 and 34
built is sound where it is finished: the outdir refusals fire by name under the CLI with nothing
created, the arms table and its two guards close the caller-key and defaulted-theta classes, the
thresholds come from the fits for both heads, the ladder is one set in three places, and
`de_head_scoring` is a good module — it refuses a wrong-width vector by name on both real heads,
which is exactly the failure round 33 discovered by traceback after a 29-minute feed.

The reason this is a base and not a producer is DE34-C1: **the module that refuses is not the module
that runs**. Before any `--run` against the declared OUTDIR, round 35 should wire
`de_head_scoring`'s two scorers into `score_events_for`, add the preflight that scores one real row
through the same path, and put `LightGBMError` in `main()`'s tuple (DE34-R5). DE34-R1 and R4 are
one-line changes that should travel with it; DE34-R7 wants a decision about the pin's shape before
the wiring, not after. My EST-R1/R2 remain the estimand questions the USER rules on; nothing in this
round changes them.
