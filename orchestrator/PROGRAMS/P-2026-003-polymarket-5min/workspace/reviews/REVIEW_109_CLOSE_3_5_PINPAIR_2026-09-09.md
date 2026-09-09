# REVIEW 109 — the close of (3) and (5), the second pin pair, and the stale assembly cache

**REV, 2026-09-09T05:37Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`. Everything below ran in a scratch clone at the tip.

**THE ANSWER THAT GATES EVERY CORRECTED RUN: YES, A DAY CAN NOW RUN.** The in-run battery
that REVIEW 107 said no day could pass returns **rc 0** at the tip, and the full battery is
**PASS 402 / 0 disarmed / 0 skipped**. All ten cascade pins and the entry pin match the
tip's files — **0 mismatches**.

**VERDICTS.** **(3) CLOSED — and I drove it end to end through the REAL scorer, not a
patched one.** **(5) CLOSED — the count is the one I named, from the source I named, and it
FIRES on a real partial generation whose first crossing differs.** **The pin pair is a PURE
RE-POINT** — but the pair to gate is **v23 + design v31**, not v22 + v30: v22 moved one of
two pin sites for the same file and would still have refused. **The stale assembly cache is
NOT safe: the guard covers one of eight score-readers; seven others read it unguarded.**

---

## 1. THE PIN PAIR — PURE, EXACT, AND IT IS v23, NOT v22

**Purity.** Leaf-by-leaf over the whole params JSON, outside the two provenance blocks
(`be_module_repoint`, `supersedes`), the second pair changed **nothing but digests, the
design pointer and the version**:

| | leaf differences outside provenance |
|---|---|
| v21 → v22 | `be_cascade.modules[0].sha256`, `modules[4].sha256`, `design_declaration.path`, `…the_design_pins_THIS_file`, `version` |
| v22 → v23 | `be_module.sha256`, `design_declaration.path`, `…the_design_pins_THIS_file`, `version` |

**No estimand, bar, threshold, arm, horizon, day set, alpha, multiplicity, read gate or
settlement-endpoint field moved** — they are absent from the diff, which is the proof.

**Exactness.** `70b54a4` changed exactly one file, `de_phase4_diag_runner.py` = cascade
module **[4]**. `272dfb3` changed exactly one file, `be_cancel_axis_null.py` = cascade
module **[0]** *and* the `be_module` entry pin. The pair re-points exactly those and nothing
else, and every pin verifies against the tip:

```
[0..9] all OK    be_module OK    mismatches: 0
```

**THE PAIR TO GATE IS v23 + DESIGN v31.** `be_cancel_axis_null.py` is pinned in TWO places —
`be_cascade.modules[0]` and `be_module` — and v22 moved only the first. Driven:

```
params v22: ten-module check -> REFUSED   entry pin -> REFUSED
params v23: ten-module check -> PASSES    entry pin -> PASSES
```

**A day under v22 would have refused in exactly the place REVIEW 107 named.** One detail
worth carrying: v22's refusal reads *"BE's cascade module digest differs"* and **does not
carry the `BE_CASCADE_DIFFERS` name** — the entry-pin check has its own unnamed message, so
a reader grepping for the named refusal would have concluded the tree was clean. Give the
entry-pin refusal a name of its own.

## 2. (3) THE NON-FINITE SCORE — CLOSED, driven through the real scorer

The predicate landed where I asked for it — at the consumer, in `generation_scores`, where
the score enters the decision — and **not** in `score_incumbent_condvalue`, which still
returns `nan` for `[1e308] * n` and `inf` for a single huge component. That is the right
placement: the scorer is a pure function and the decision is where an unrepresentable
number becomes a silent verdict.

**My own case, driven against the closed code.** DE's cell proves the predicate fires by
monkeypatching the scorer to return `nan`; that shows the predicate works but not that the
real path can reach it. So I drove the real one. The route is a **near-zero `norm_sd`**, and
here is a correction to my own REVIEW 107: I named `linear_btc.json`, the LGBM z-scale that
(2) pinned — but `score_incumbent_condvalue` normalises with the **incumbent's own**
`norm_mu`/`norm_sd`, which come from **`linear_d_btc.json`**. Two files, one letter apart,
different jobs — the trap DE's own (2) comment flags. Both are in the digest set, so both
are covered; the mechanism is the second file.

```
generation_scores(..., head="incumbent_linear_d") with the incumbent's norm_sd at 1e-320
  -> DiagRefused: NON_FINITE_SCORE: incumbent_linear_d scored row
     ('s1','BUY_UP',0,100.0) as nan from FINITE features
```

Every input finite, the real scorer, the real function, refusing by name and naming the row.

**The converse, as asked.** Where the non-finiteness originates in the FEATURES, the feature
guard still fires first: composing raw inputs at `1e308` refuses `NON_FINITE_FEATURE`, while
`1e300` composes finite and scores finite (`2.2e301`). So the two guards are complementary
and neither is redundant: the feature guard catches non-finite inputs before the clamp, the
score guard catches finite inputs whose product overflows. **I found no third route.** The
LGBM head does not overflow the same way — `score_lgbm_condvalue` on a `1e308` vector
returns a finite `0.673`, because a booster saturates — but the guard sits at the consumer,
so it covers that head too.

## 3. (5) THE PARTIAL-ROW COUNT — CLOSED, and it FIRES

`rows_in_by_generation(split_of)` is exactly the computation I named: a group-by over the
tape index, keyed `(slug, side, gen, t_start)`, taken **before the feature pass drops
anything**, from a parameter `generation_scores` was already passed. The comparison is made
at the **union** of the merged per-row scores, which is the caveat I flagged and the place
DE's own docstring always pointed to. Nothing came from BE.

**And it fires on a real partial generation — driven, with the decision consequence
attached.** Three rows in the tape index for one generation; the EARLY row dropped from
`kept`:

| | scores kept | `PARTIAL_ROWS` | dropped | status |
|---|---|---|---|---|
| complete (3 of 3) | t=100 → 4.1546, t=103 → 0.5862, t=106 → 1.9187 | **0** | — | `COMPUTED_AGAINST_THE_TAPE_INDEX` |
| partial (2 of 3) | t=103 → 0.5862, t=106 → 1.9187 | **1** | **1** | `COMPUTED_AGAINST_THE_TAPE_INDEX` |

and at `theta = 3.036638`, through the policy engine:

```
COMPLETE -> 1 cancel at t_request 100.0        PARTIAL -> 0 cancels
```

**The dropped early row was the crossing row, and losing it removed the cancellation
entirely.** That is exactly why a maximum was insensitive and a first crossing is not — and
it is now REPORTED rather than silent. The status is a computed string, not a promise, and
a missing index still yields `NOT_COMPUTABLE…`, so absence never reads as a pass.

**One leftover, non-blocking, same class as REVIEW 107's (1a):** `_rows_expected` is now
dead code that still `return 0` under a docstring saying *"rows-in per generation exists
nowhere … That function is not DE's surface"* — the claim this round refuted, two hundred
lines from the function that refutes it. Delete it or point it at `rows_in_by_generation`.

## 4. THE STALE ASSEMBLY CACHE — REBUILDABLE, AND NOT CURRENTLY SAFE

**What it is.** `data/pm_5min/derived/de_section81_cache_12.pkl`, 28 MB, **dated Sep 4
12:00** — before DE 155. I opened it: `asm["by_arm"][("btc","incumbent_linear_d")]` holds
**29,813 scores, each a bare float keyed at the generation start**, e.g.
`('btc-updown-5m-1787579400','BUY_UP', 0.100146715) -> 0.0710557…`. **That is precisely the
pre-DE-155 shape the guard names**, so BE's refusal is correct and BE was right that it is
not caused by BE's own change.

**Is it rebuildable, and by whom?** Yes — it is the section-81 assembly, DE's artifact
(`de_section81_arms.py` writes the same structure under a scratch name). Rebuilding means
re-running that assembly under the corrected scoring: a heavy DE run, not something BE can
do from its side. **BE is blocked on a DE artifact**, and `be_cancel_axis_null.py` — the
module BE just changed, which is cascade module [0] *and* the entry pin — is one of the
readers, so BE cannot complete its own battery until the rebuild lands.

**Does anything depend on it that would silently use pre-causal scores? YES.** Thirteen
modules reference the file. The `ASSEMBLY_PREDATES_CAUSAL_SCORING` guard exists in **exactly
one** of them (`de_phase4_diag_runner._head_scorer`). Of the other twelve:

- **Seven read `asm["by_arm"]` — the score half — with no guard at all**:
  `be_cancel_axis_null.py`, `be_daybook_builder_declaration.py`,
  `be_generation_count_derivation.py`, `da_elementwise.py`, `da_elementwise_hz.py`,
  `da_elem_grid.py`, `da_de53_exclusion.py`.
- **Five read only `fr`** (reference, rows, population, statuses) and are unaffected, because
  the pre-causal defect is in the scores, not the reference:
  `be_ceiling_null.py`, `da_baseline_concentration.py`, `da_ceiling_attainable_701.py`,
  `da_rebate_ceiling.py`, `de_section81_mid_census.py`.

**Said precisely, because "reads it" is not "is wrong":** the cached floats are the OLD
statistic — one maximum per generation — which is the statistic theta was fitted on, so a
consumer that counts generations or reads a score distribution is internally consistent.
The harm is for anything that presents a result as being under the corrected causal rule, or
that makes a timing or cancel decision from those floats. **The cheap, complete fix is to
move the shape check to the LOAD site** — one predicate in whatever opens this pickle — so
every reader gets the refusal instead of one path having it and seven not. A guard that
covers one of eight readers is not what makes a stale cache safe.

## 5. ROUTED

1. **DE — rebuild `de_section81_cache_12.pkl` under the corrected scoring**; BE's battery
   and seven modules wait on it. Put the shape check at the load site while rebuilding.
2. **DE — name the entry-pin refusal** (today it is the only cascade refusal without a name,
   and that is how v22 read clean).
3. **DE — delete or re-point `_rows_expected`**, whose docstring still states the claim (5)
   refuted.
4. **Coordinator — the pair of record is params v23 + design v31.** v22 + v30 is superseded
   and would not have run a day.
