# RESULTS — P-2026-003 Polymarket crypto 5-min

Consolidated 2026-09-03T03:23Z, substantially rewritten 2026-09-03T08:29Z,
reliability-corrected 2026-09-04T09:59Z, and **extended 2026-09-04T13:48Z with
§0, the first absolute economics this programme has produced**. **Single writer: the
coordinator.** This
file is the compact, artifact-anchored answer to "what has been tested and what
came out of it". `STATUS.yml` and `workspace/HANDOFF.md` remain the running
state; `COORDINATION.md` remains the append-only register. Read this first, then
`HANDOFF.md` for the live detail.

Primary result numbers below are retained from the named artifacts used in the
consolidation. The 2026-09-04 reliability classification, code-path evidence and
independent checks are recorded separately in
`RESULT_RELIABILITY_AUDIT_2026-09-04.md`. Where a previous document disagrees,
this file says so explicitly and the disagreement is a **correction**, not a
restatement.

---

## 0. 2026-09-04 — the first absolute economics, and what they say

**This section supersedes §1 wherever they disagree.** Until today this
programme had **no absolute economic number at all** — every result was
candidate-versus-incumbent, model against model, with no no-cancel baseline
anywhere. §1's "profitability withdrawn" was correct and is now superseded by a
measurement rather than by a restatement.

**All figures: btc, ONE CONTIGUOUS HOUR (2026-08-24 13:50–14:50Z, 12 windows,
4,315 fills), development evidence, point estimate and NO interval** — 12
windows is below the 5-complete-day cluster floor. `is_a_validation` false.
Artifact `de_section81_arms__20260904T131*Z.json`, emitted from a clean tree on
the branch with 7/7 identity files matching; the earlier `125340Z` emission named
a carrying commit **not on the branch**, and **26 of 26 economic quantities are
bit-identical across the two — the defect was the record, not the result.**

| leg | baseline (0 cancels) | CONDVALUE (333) | HAZARD (48) |
|---|---:|---:|---:|
| spread capture | +10,566.95c | — | — |
| adverse selection | −1,968.19c | — | — |
| **fills leg (maker P&L)** | **+8,598.76c** | **−953.92c** | **−12.56c** |
| **inventory leg** | **+8,587.54c** | **+3,348c** | **+650c** |
| **both legs** | | **+2,394.40c** | **+637.83c** |

### What is established

- **Market-making pays on this hour.** Reproduced by **four independent
  producers**, one of which recomputes the whole decomposition from the
  reference alone without any replay (Δ 7.3e-12).
- **The ranker selects correctly on both axes.** Per **lost fill** (not per
  cancel), removed fills carry **3.01× the average adverse at 0.832× the average
  spread**; HAZARD is sharper at **3.81× / 0.76×**.
- **The ranker avoids the biggest winners.** Of the 43 largest-P&L fills,
  **CONDVALUE declined 1 where chance says 14.35; HAZARD declined 0** — and 0
  under the extreme-|P&L| ranking too.
- **No path effect.** Every retained fill is **bit-identical in both replays**,
  across both heads, three budgets and three latency rungs (6 non-vacuous cells);
  the strongest removes 82.6% of the book and re-prices nothing. So the delta
  **is** the declined fills' value. CLAUDE.md pitfall 4 does not fire here.
- **The two arms differ 10.95×**, splitting as **per-fill cost 5.643× and
  cascade 1.940×**. A cancel removes **4.32 and 2.23 fills**, against **1.12 for
  a random cancel**. Direction: **cheap fills first, few fills second.**

### What is NOT established, and these are the load-bearing caveats

- **That the book has harvestable structure at all.** The "top 1% carry 113%"
  concentration **sits inside a no-tail Gaussian null's 90% band** (observed
  1.1295, null median 0.6765, band [0.427, 2.060]; **19% of Gaussian books
  exceed 1**). It is a statement about **dispersion**, not tail-dependence.
- **"99% of this book clears break-even fourfold" is an artifact of which tail
  was removed.** Body r is **1.1077** removing the winners, **0.2504** removing
  the extremes, **0.1863** whole-book. **Same book, three answers.** Selecting
  the top k by outcome and evaluating the remainder conditions on the outcome.
- **The inventory reversal runs through a mechanism its author refuted.** The
  prediction was pre-registered: **P1 (baseline leg negative) was REFUTED by its
  own falsifier** — the leg is +8,587.54c — and **P4 was derived from P1**, so
  a right sign came from a wrong model. The real route is a **directional bet**:
  CONDVALUE's terminal net **flips** (+146.74 → −147.28) rather than shrinking.
- **The inventory leg has a cluster unit of 12, not 4,315.** Three of twelve
  windows carry **81.4%**; one gap-ended window carries **28%** of the baseline
  leg. The conclusion survives both terminal-mark views (+773c / +4,543c) but the
  **level moves 28% on one window**.
- **Neither arm is distinguishable from random cancellation** on the fills leg
  (CONDVALUE z = −0.20, p = 0.43; HAZARD z = +0.26, p = 0.60) — though the two
  **point estimates are a factor of seven apart** and the z-statistic hides it.

### The specification, and the statistic that replaced it

An overlay pays iff **r = adverse/spread ≥ σ/α**: **27.60% for CONDVALUE,
19.88% for HAZARD**, against this book's **18.63%**. **Both lose; HAZARD by
6.7% of relative adverse.**

**But `r` is REFUTED as a survey statistic.** Three books holding N, total
spread and total P&L **exactly** — so `r` is identical at 18.6259% — have
overlay ceilings of **0.00%, 10.61% and 21.66%**, and α ceilings of 1.00, 10.01
and 100.35. Whether an overlay can pay is a property of the **joint distribution
of (spread, adverse) across fills**, which a ratio of totals discards. **σ/α is a
per-book quantity, not a constant.**

**The replacement is `V_oracle`** — the sum of |P&L| over fills whose P&L is
negative. Model-free, exactly the ceiling **any** overlay could reach, one filter
and one sum. **A survey where `V_oracle` ≈ 0 closes the overlay line because NO
POLICY COULD HAVE PAID**, which is a different and better reason than "this
policy did not". That survey is running across the admissible development window
(the hf_ws_v2 boundary 2026-08-24T13:48:54Z to the freeze epoch), touching **no
sealed day**. The measured hour is **one cell** of it.

---

## 1. Bottom line — credible negative diagnostic; profitability withdrawn

**The candidate has not demonstrated an improvement over the incumbent.** The
narrow result is reproducible: on the two pre-declared economic read days,
**09-01 and 09-02**, BTC is negative at all three equal-action-count operating
points. The 08-29 read is development evidence and does not upgrade that
two-day result to validation.

`MATCHED_VOLUME` is the repository label, but it matches the **number of cancel
actions**, not shares or notional. It is also a **model-vs-model comparison**:
the candidate and incumbent are both linear harmful-flow predictors over the
same 54 Polymarket plus six fine-flow inputs. The candidate is the frozen
`PM_PLUS_FINE` fit; the incumbent is the per-coin,
generation-reweighted `INCUMBENT_REWEIGHTED_ONLY` fit. It is not a no-prediction
or no-cancel benchmark.

| population | 5% | 10% | 15% |
|---|---:|---:|---:|
| 09-01 | **-789.12c** | **-2,016.71c** | **-1,476.01c** |
| 09-02 | **-227.60c** | **-1,237.84c** | **-2,975.36c** |
| pooled | **-1,012.68c** | **-3,038.75c** | **-3,949.76c** |

A negative value is candidate minus incumbent five-second gross cancel value:
for example, pooled 10% means the candidate captured **$30.39 less** avoided
adverse markout than the equal-count incumbent. It is not realised account P&L.
The declaration ordering was correct and an independent recomputation matched
all nine cells exactly.

**Inference remains preliminary.** Only two pre-declared days have been read;
no day-cluster interval is claimable. The reported p-values use window-level
sign flips, which the declaration itself labels weaker/optimistic relative to
the ruled UTC-day cluster. A high one-sided p for `candidate > incumbent` means
the candidate failed to show a win; it is not proof that the candidate is
permanently worse. In addition, the incumbent's equal-count cutoff is selected
retrospectively from the full evaluated population. This isolates descriptive
ranking quality but is not an executable operating point.

**BY_THRESHOLD remains a misleading headline.** It reads strongly positive on
BTC because one candidate-calibrated theta makes the candidate cancel about
three times as often. Its exact decomposition assigns more than the entire
positive increment to **volume**, with the equal-count **quality** term negative
throughout. It is not evidence that the candidate ranks cancellations better.

### Profitability correction — all prior dollar/return claims withdrawn

The following previously reported labels are **not reliable and must not be
quoted as profitability**:

| withdrawn label | previously reported value |
|---|---:|
| filled notional | $226,594 ($75,531/day) |
| no-cancel baseline P&L | $1,801.29 ($600.43/day) |
| return on filled notional | 0.7949% |
| best cancel overlay | +$620.58 (+34.5%); $807/day combined |

The scratch `prof.py` calculation retains only the first row for each
`(slug, side, gen)` action, then treats `preventable_shares` as all fill shares.
But the producer explicitly defines `preventable_shares` as only the tranches
inside the one-second action horizon and at or after the 50 ms latency cutoff;
earlier fills are `stale_shares`, and later fills are outside the population.
The result is neither total filled notional nor a whole-book no-cancel baseline,
and the baseline and overlay can use different rows of a multi-row action.
Fees, realised exits/settlement, quote size, inventory and capital are also
absent. Therefore **P003 currently has no reliable profitability estimate.**

There is also a durability gap: `matched_volume()` has no committed caller, and
the canonical `be_read_cells.compute()` still emits only `BY_THRESHOLD` and
`BY_COUNT`; the interim and profitability reports were assembled by scratch
scripts. The arithmetic is reproducible today, but the result path is not yet a
stable in-repo pipeline. Full audit: `RESULT_RELIABILITY_AUDIT_2026-09-04.md`.

**Current race state: G = 3 of 5.** R-503 re-verdicted 09-03 on its covered
complement (287/288 windows, with 15:20Z named and counted), so 09-01, 09-02 and
09-03 accrue. This rule was applied after the 09-03 coverage failure was seen;
that provenance is stamped in the artifact. 09-03 has not become a third
economic read, and its superseding verdict lost the scheduled-unit attribution
prefix—an open provenance issue. See §3.

---

## 2. Iteration 011 — the conditional signed-value family

**Artifact:** `data/pm_5min/derived/iter011_conditional_value_v1__coin_btc.json`
(188,119 B, `as_of` 2026-09-02T05:21:34Z, written 05:35:04Z). BTC only.
**Class:** `development_evidence.is_a_validation = false` — both fitting and
evaluation populations are development. It selects; it never validates
(prereg 4).

| head | statistic | lgbm | linear | p | Holm | survives |
|---|---|---|---|---|---|---|
| **Q1_arrival** | AUC, 311,640 rows | **0.8303** | **0.7733** | 0.001996 | 0.0479 | **yes** |
| Q2_sign | AUC, 33,622 actions | 0.6003 | 0.5824 | 0.001996 | 0.0479 | no — `NO_INCUMBENT_COUNTERPART` |
| **Q3_magnitudes** | calibration slope, 15,912 rows | **0.6888** | **0.6437** | 0.001996 | 0.0479 | **yes** |
| **Q4_combined_ev** | **net cents vs incumbent**, 166 windows | +3,867.1 / +2,818.2 / +2,472.6 | +3,277.5 / +278.6 / +1,565.4 | 0.01999 best | **0.1199 best** | **no** |

**Q1 is a real increment.** Its gate — *"beats the matched-random null AND beats
the incumbent hazard head"* — has **both conjuncts computed and true**.
Incumbent hazard head AUC **0.7139077** (`incumbent_auc`, 2,000 permutations,
166 units), so the increments are **+0.1164** (lgbm) and **+0.0594** (linear),
on 166/166 windows with zero exclusions.

**Four qualifications, all read off the artifact, none of them optional:**

1. **Every surviving p is a floor, not a measurement.** All 18 non-Q4 cells
   carry the *identical* p = 0.001996 = **1/501** with
   `at_permutation_floor: true`, 500 draws. Holm 0.0479 sits 0.0021 under the
   0.05 line. **Stated precisely** (MEM round 72's correction to this file,
   verified by computation): if the *whole tied family* draws the other way
   (p → 2/501), Holm goes to **0.0958** and the entire surviving set collapses;
   if a **single** cell moves, it sorts behind the still-tied cells at adjusted
   p 0.0279 and the leading Holm stays **0.0479**. The fragility is the family's,
   not one draw's — the earlier wording implied the latter.
2. **"12 of 24 survive" overstates the evidence, and the artifact says so
   itself.** `distinct_results`: 24 declared cells → **12 distinct**, and
   **4 distinct surviving results** (Q1×2 arms, Q3×2 arms). Budgets select
   cancellations, not predictions, so Q1/Q2/Q3 carry one statistic replicated
   across three budgets. *Read the survivor count as distinct results, never as
   independent evidence.*
3. **The p-values are optimistic by declaration.** `cluster_disclosure`:
   `G_complete_utc_days: 0`, ruled unit **UTC day**, unit actually used
   **window**, `weaker_than_ruled: true`, `intervals_claimable: false` — units
   within a day are not independent. *Evidence, never a significance
   certificate.*
4. **Q3 passes a weaker gate than Q1.** Q3's frozen gate carries **no incumbent
   term** (`carries_incumbent_term` false for its conjunct set), so it passes a
   gate that never required beating anything. That is not Q1's achievement.

**Q4, the decision metric, in full.** All six point increments are positive
(+278.6 … +3,867.1 net cents; best cell REPORTED-not-adjudicated as candidate
+11,743.9c vs incumbent +8,466.4c). None survives: best one-sided p 0.01999
over 2,000 sign-flip permutations — **not** at the floor (floor 1/2001), so this
one is a real measurement — giving Holm 0.1199 over the family of 24. Status is
`GATE_PARTIALLY_EVALUATED`, and `passed` reads **null, not false**: the
structured conjuncts record `increment_beats_incumbent: false` and
`matched_random: null` — a conjunct nobody computed reads null (R-397 ruling 2).
The frozen design declares a **two-sided** p (0.04998, reported); the adjudicated
p is one-sided per R-286/R-288, and amendment A2 is a DRAFT awaiting the USER,
because only the USER amends a frozen design.

**One tension left open, deliberately.** That cell's prose `detail` says *"The
incumbent counterpart EXISTS (comparable=True) and was NOT COMPUTED, so only the
matched-random conjunct was evaluated"* — which is the **opposite way round**
from its own structured fields (`gate_conjuncts_unevaluated: ["matched_random"]`,
and an incumbent-increment p that plainly was computed). This is the rule-10
shape (prose beside a table). It is recorded here as an **observed tension to be
adjudicated**, not as a proven error, and nothing in this file is read off that
prose.

### Corrections to earlier documents (in-band, rule 13)

| where | said | the artifact says |
|---|---|---|
| `HANDOFF.md` §Current model state | `cells_by_status` = **18 OK + 6 NO_INCUMBENT_COUNTERPART** | **12 OK + 6 NO_INCUMBENT_COUNTERPART + 6 GATE_PARTIALLY_EVALUATED** (denominator 24). The six Q4 cells were counted as OK; they are not |
| coordinator's own report to the USER, 2026-09-03 ~03:08Z | "12 of 24 cells survive" quoted without the multiplicity disclosure | 12 cells = **4 distinct surviving results**, all at the 1/501 floor |

---

## 3. Forward race — the only path from development evidence to validation

**Bar:** ≥5 complete UTC days, each FINISHED ∧ AFTER ∧ ADMISSIBLE ∧ HEALTHY.
**State: G = 3 of 5** after the R-503 superseding verdict for 09-03
(`659ed66`, 2026-09-04).

Freeze epoch `1787897340` = **2026-08-28T06:09:00Z**. Every day below is read
from its own `da_dayverdict_<day>.json`, `verdict_split` and `era_admission`.

| day | current race disposition | note |
|---|---|---|
| **08-29** | no — **withdrawn** | Passes the four conjuncts, but R-500 deliberately excludes it from G; its read is development evidence on `clob_v3_1` |
| 08-30 / 08-31 | no | mixed-era and/or quality failures; neither is decision-bearing |
| **09-01** | **ACCRUED** | first race day; subsequently opened under the pre-declared interim |
| **09-02** | **ACCRUED** | first governed verdict; subsequently opened under the pre-declared interim |
| **09-03** | **ACCRUED under R-503** | 287/288 covered; 15:20Z is named and counted as accounted loss; BTC P1 95.61 s/hr against 120 |
| 09-04 → | open | two more accruing days are required; earliest G=5 is the **2026-09-06T00:06Z** verdict if 09-04 and 09-05 accrue |

**R-503 changed the coverage treatment, not the data.** A closed day may now be
judged on its covered complement when at least the already-ruled 144-of-288
floor is met. Missing windows are named and counted as accounted loss. For
09-03 the complement is 287/288, so its previously unevaluable quality becomes
evaluable and passes. The rule was adopted after this missing-window failure
was observed; the artifact records that `prompted_by` provenance rather than
presenting the change as neutral housekeeping.

**Economic-read state is different from race accrual.** 08-29 was opened as a
development read. 09-01 and 09-02 were then opened under the pre-declared
interim and are consumed. 09-03's accrual does not by itself add another
economic observation: it still must be scored, sealed, and governed by a later
declared read before its economics can be opened.

**09-02, at `data/pm_5min/derived/da_dayverdict_20260902.json`** (43,449 B,
sha256 `6f283262df463957…`, `as_of` 2026-09-03T00:06:01.399Z, written by the
scheduled unit — `ExecMainStatus=0`, `Result=success`, 00:06:00→00:06:06 UTC):

| conjunct | value |
|---|---|
| FINISHED | `day_closed_calendar` true (the tape selector reads false; its predicate lags the boundary by up to one window — disclosed in the row) |
| AFTER | `post_freeze_pass` true, 288/288 every coin |
| ADMISSIBLE | `clob_v4_1`, era-pure, no boundary inside the day — an interlock, not a quality grade |
| HEALTHY | `day_quality_pass` true under the governing `day_bar_v2`: btc P1 **73.71** s/hr (bar 120) · P2 **0.00 %** (bar 5 %) · P3 **219.7** s (bar 900); eth **1.85** · **0.00 %** · **15.5** s |

**Reported beside it and NOT governing:** btc `windows_gap_affected` **50.3 %**
coin-level (145/288 windows, 287 gap intervals, 1,769 s lost) against eth
**1.7 %**; the count bar `gap_rate_under_bar` fails (304 gaps, 12.67/hr, 8 hours
over the hourly bar); `tape_density` UNMEASURED (its receipt covers 13 days, not
this one).

**Content liveness governs for the first time and reads THIN.**
`content_liveness_rule` `governs: true`, `frozen_by_user: true` (R-386, module
`7196676840304f30`, effective from 20260902): status **CONTENT_THIN, 7 of 7
coins thin, 0 unjudgeable** (btc L1 0.138, longest thin run 40 windows; hype
0.055 passes L1). It **does not veto HEALTHY**
(`content_thin_vetoes_HEALTHY: false`, ruled by R-409): disclosed and masked,
not inadmissible. The blackout mask artifact is `WRITTEN`, 7 coins, **251 masked
windows**.

**09-01 and 09-02 were scored and read under the interim declaration.** They are
now consumed and cannot be reused as untouched forward validation. The 09-01
receipt records **610,064 BTC + 441,409 ETH actions** with masking applied at
supply (141 windows across seven coins), so exclusions were counted before rows
were built. 09-03 remains an accrued race day, not an opened economic result.

**Two unexplained outages remain on the record** (09-01): 00:00–01:05Z (65 min)
and 22:45–23:35Z (50 min) at 0.01–2.2 % of median window content, on all seven
coins, with **no gap rows** — invisible to every duration bar. Two independent
instruments (collector-log msgs/s and raw gzip-trailer bytes) agree to one
minute. This is the class the content-liveness rule exists for.

---

## 4. The Phase-4 diagnostic the USER scheduled (R-459) — RAN, AND DIED

**Both original blockers were cleared on 2026-09-03**: the USER declared the
population split (R-496, MECHANICS on both splits, labelled per cell) and DE
built the producer half that had never been dispatched. The USER then admitted
the `_stream_tape_rows` fit-vs-tip drift (R-499) — **conditionally**:
`tape_rows_array_closed()` is evaluated at run time on the actual tape, and the
run refuses if it returns False, the ruling notwithstanding.

**Launched 07:01:35Z. Died 07:09:18Z on a `MemoryError`.** The conditional
admission worked exactly as designed — the first progress line records
`admitted_by USER`, `recorded_at R-499`, `condition_holds true`, with the
evidence read off 3,170,987,711 bytes of real tape.

**The cause, measured at full scale** (not scaled from a slice, which is what
the original price did wrong): `tape_index[score]` 1.42 GB, `tape_index[train]`
3.90 GB, `fragment json.loads` **8.33 GB** — resident *before* the per-window
pass does any work — then ~3.55 GB accumulated across 1,125,289 rows. 8.33 +
3.55 crosses the 12 GB cap.

**The worse half was not the crash.** The progress log held one line —
`preflight_passed` — and then silence, indistinguishable from a healthy run;
the traceback went to a session scratch dir. It was found by reading the process
table. DE round 48 fixed that first: a terminal record on **every** exit path
(success, exception, signal, atexit fallback), stderr `dup2`'d into the outdir,
a 30 s heartbeat, all installed *before* the first expensive stage and asserted
from the parse — with a falsifier that **SIGTERMs a live run** and asserts the
log's last line says so.

**The fix bounds rather than enlarges, and no cap increase was requested**
(coordinator ruling: a memory cap is a safety property, not a budget to spend
down). Chunked assembly, partition by split, `_BN_CACHE` cleared between chunks
as a runtime call on the module rather than an edit to it: ~9.6 GB → ~6–7 GB.
The relaunch is gated on proving chunked-and-partitioned equivalent to whole on
real data, tolerance declared before looking.

Two blockers, one of each kind:

1. **A producer step nobody had been dispatched to build** (coordinator's
   omission, stated as such): rounds 33–42 hardened the runner's instruments —
   necessary work, since round 33 would have fed for ~29 minutes and then
   crashed on a stub scorer feeding the booster one column against 106 — but the
   expensive half was never assigned. **Dispatched 03:13:51Z as DE round 43**;
   see §6 for what that immediately turned up.
2. **A declaration that is the USER's** (rule 14): the §3 population
   (08-24/08-25) spans **both** fit splits — 1,125,289 `train` rows and 638,917
   `score` rows — so every cell would score generations the heads were fitted
   on. Either the run is declared a MECHANICS diagnostic on a consumed
   population (splits labelled per cell), or it is restricted to the `score`
   split (smaller population, §3's counts change). **No seat may choose.**

**Cost, as far as it is known:** the feed is MEASURED at ~28.6 min for the §3
population, both coins. The feature assembly is **UNMEASURED** — a tape index
over `phase2_state_tape_v5.json` (3,170,987,711 B) and `_feature_pass` over
`harmful_exposure_rows_v3_eraB.json` (1,241,115,096 B, 1,135,943 rows). One
`arm_result` is unmeasured on real data with a floor of 0.007 s; 200 draws is
**800 replays**, plus rejected attempts.

---

## 4a. The forward decision-metric path — built 2026-09-03, **not yet released**

Built across BE rounds 14–21 after the finding in §1. What it now does, and
every item below was landed only after an adversarial review drove the previous
version:

- **The estimand is fenced.** `increment()` — the decision metric under the
  USER's by-threshold ruling (R-497 (F)(4)) — no longer accepts a bare theta. It
  takes the object the fence returns plus a budget key.
- **The fence fetches its own evidence.** The declaration names
  `verification_ref {path, sha256}`; the fence opens and rehashes it, and an
  **inline verification block is REFUSED** — supplying the evidence is the act
  being forbidden. This went beyond the reviewer's specified fix, which BE
  judged would have *"satisfied the letter and not the principle."*
- **The numbers are bound to the bytes.** `derive_days_from_rows` asks the rows
  artifact which days it contains; `verify_declaration_by_recomputation`
  re-derives the quantile map restricted to the declared days —
  `all_coins_reproduce` **True**, `max_abs_difference` **0.0**, over 1,135,930
  rows.
- **`RETROSPECTIVE_TOPK` is refused, not offered.** `evaluate_policy` silently
  fell back to a threshold *read off the data being scored*. That fallback sat
  directly on the path this programme was about to run.
- **Reconciled against a known answer on already-consumed data**: iteration
  011's Q4, **36/36 predicates**, both permutation p-values bit-for-bit, Holm
  reproducing across the declared family. Re-run independently by the
  coordinator and by the reviewer; the cell digest is stable across two
  `PYTHONHASHSEED`s.
- **The declared family is 18, enumerated not multiplied** — superseding the
  coordinator's "doubles" (R-498). `require_declared_count` refuses a different
  count *or the same count with a substituted cell*.

**The gate is shut.** Three release reviews; the first two found the fences
real, tested both ways, and **off the path**. **No forward day may be scored
until it opens** — which is also why the 08-29 read the USER preserved has not
happened.

---

## 4b. The recurring failure of this codebase, named with its instances

**Five zero-consumer / zero-reachability findings in one day, each found by a
different route and none by a green suite.** `SEAT_PROTOCOL` 17 already names
the class (*suite-green is not pipeline-wired*); what is new is the frequency.

| # | the fence | how it failed |
|---|---|---|
| 1 | `require_operating_point` / `require_arm_identity` | **every** executable call inside `selftest()`; the decision metric passed through neither |
| 2 | six evaluator functions (I11-2) | falsifier-proven, zero call sites in the runner |
| 3 | `assert_frozen_contract` | one call site, inside `anchor_drift_root`, wrapped in `try/except Exception: pass`. The binding it guards **already fails** (`eb8733da…` vs `03762753…`); survivable only because the drift is metadata-only — *benign by luck, not by check* |
| 4 | the R-486 `governs` stamping | **both** production call sites deletable with 254 checks still passing |
| 5 | `counts_toward_race` | written and never read; the field the race is counted by still reads `True` for the withdrawn day — binding against **edits**, not against **counting** |

Two adjacent shapes found the same day, both worth naming: a control that
**hand-injects what production drops** (`dict(_f, coin=…, verification=…)`) so
it passes on a shape production cannot produce; and a check that was
**structurally incapable of passing** — the token taken over one object while
its expectation was rebuilt from another, so False in the honest case and True
in none. This programme has long named controls that cannot *fail*; that is the
mirror image.

---

## 5. What else has been settled, and what it cost to settle

| question | answer | where |
|---|---|---|
| What do these binaries settle on? | **Chainlink**, never Binance — verified in `data/pm_5min/markets.jsonl`. The exact settlement statistic is contested; the repo's own reconstruction favours **S60 endpoints (99.8 %)** over a TWAP-vs-open reading. **No settled form is asserted.** | R-253, Q-DA-142/146 |
| Is CLAUDE.md rule 9's parenthetical right? | **No — it is FALSE and must not be cited.** Rule 9 still binds this program through a different door: the PM book (`Identity`) already prices the event, so skill is reported incremental to `Identity`, never to a base rate | — |
| Sub-second Binance data | reliable **only** from 2026-08-24 13:48:54 UTC (`recv_ns >= 1787579334881534478`). Earlier tape is usable for ≥1 s bars only | `data/mm_hf/collector_runs.jsonl` |
| Fair-price Identity | built (typed). The **challenger protocol is not freeze-ready and no challenger has been scored** | `STATUS.yml: hazard-fair-price` |
| Skew semantics | `QR_SKEW_ONLY` user-frozen | `STATUS.yml: hazard-skew` |
| Seven-arm integrated replay | contracts, parity stubs and inert trajectories only — bit-identical parity against a real seven-arm replay, lifecycle economics and the integrated candidate freeze are all **open** | `STATUS.yml: hazard-integrated-replay` |

**The review machinery, measured:** 494 register entries, 579 filed seat rows,
57 adversarial review filings, **85 distinct numbered findings**. Two from the
last 24 hours show the shape of what it catches:

- **BE12-S1** — a selftest "positive control on a real emission" ran
  `run_forward_day("20260902")` under a comment asserting that day refuses in
  ~0 s. When the scheduled unit wrote the 09-02 verdict at 00:06Z, gate 1 began
  to PASS and the control silently became a **full closed-day scoring run inside
  the selftest** (measured: 14 min, ~16 GB, killed; wrote only to its own temp
  outdir). Its subject was the calendar, not the code. Now pinned to `21000101`
  and proved unscorable **before** the driver is called.
- **DA20-R2** — the R-486 `governs` stamping was suite-green but
  **unfalsifiable**: deleting *either* production call site left 254 checks
  passing. Rule 17's shape. Fixed in DA's rebuilt held chain (unpushed).

---

## 6. What the seats hold right now (all halted 2026-09-03 03:18Z)

| seat | worktree | held | state |
|---|---|---|---|
| BE | `~/ctaNew-wt-be` | nothing unpushed; clean | round 12 landed (`f47ceb7` code, `669ef72` row Q-BE-237). Coordinator verified 129/129 checks, rc 0, driver sha `0d688474a715e899` |
| DA | `~/ctaNew-wt-da` | **2 unpushed commits** `3c49cb7` (round 20 code) → `a36db71` (row Q-DA-216); clean | rebuilt HELD chain, ready to land after the 09-04 00:06Z run |
| DE | `~/ctaNew-wt-de` | **1 unpushed WIP commit `0d03902`** (+248/−27, one file), suite **RED** by design | round 43, §6a below |
| MEM | main tree | nothing unpushed | round 71 swept (`d9b85ee`); reports nothing lost |
| reviewer | `~/ctaNew-wt-rev` | nothing unpushed | DA-20 filing landed (`cc4cfb9`); its context had reached 100 % and auto-compacted during the stop |

### 6a. What DE round 43 turned up in four minutes — the producer half bites

Wiring the expensive half moved `phase2_arms.py` from **1 reached entry to 5**,
and the runner's own code pin went **BLOCKING** by name on `_stream_tape_rows`.
Measured from the blobs: that function **changed between the fit and the tip** —
sha `f0741bc4b170fabc` → `f0b3bccfb8ec5b88` at commit `2e1204f` ("BE: T2
fail-open readers", 2026-08-29). The diff is confined to one branch: EOF without
the rows array's closing `]` used to return and now raises; **the accepting path
is byte-for-byte unchanged**. The tape's last bytes are `...}}]}` — its rows
array is closed — so the new refusal branch cannot fire for this input, and DE
added `tape_rows_array_closed()` as the predicate behind that claim rather than
asserting it.

**Open question DE did not rule, and should not have:** whether declaring
`_stream_tape_rows` is a seat's call or the USER's. It needs no number and no
policy choice — only a computable statement about code and about this tape's
last bytes — but the judgement is unreviewed.

This is the value of running the producer: **one four-minute wiring attempt
surfaced a fit-vs-tip code drift that no instrument round had found in ten
rounds.**

---

## 7. Open USER decisions — **one**, plus one queued

**Seven rulings were taken on 2026-09-03, followed by R-503 on 2026-09-04.**
Recorded here because several changed what earlier sections of this file said.

| ruling | where | consequence |
|---|---|---|
| 09-02 **accrues** | R-496 | G 1 → 2 |
| addendum v2 **adopted as a package** | R-496 | split declared; unblocked the diagnostic's declaration half |
| era: **quality is the bar, not collector version** | R-497 | `clob_v3_1` admitted; the invariant repair found **two more** unruled entries (`clob_v5` had `True` and no ruling at all; `clob_v4`'s cite does not name it) |
| operating point: **declare a grid, report all, select none** | R-497 | runs on `FROZEN_FROM_TRAIN_QUANTILE` |
| futility check: **configurable** | R-497 | parameterised, with a coordinator-added guard: a run refuses unless its config sits in a commit that provably predates the read |
| pairing: **both, threshold primary** | R-497 | family 12 → **18**, enumerated (R-498) |
| `_stream_tape_rows` drift: **ADMIT** | R-499 | conditionally — the condition is re-evaluated at run time and the run refuses if it fails |
| **08-29 withdrawn from the race, kept readable** | R-500 | **spends the cleanest post-freeze day.** G stayed 2 at that ruling; the current G=3 comes from 09-03, not from reversing this withdrawal |
| uncovered windows judged as accounted loss on the covered complement | R-503 | 09-03 changes from unevaluable to accrued at 287/288; missing 15:20Z window named; G 2 → 3. The rule change records that 09-03 prompted it |

**On the 08-29 withdrawal.** The day is *admissible and deliberately not
entered* — two separable facts. The verdict stops asserting `era_admissible:
false` (made false by R-497) and carries the true one instead. **The withdrawal
is recorded before any read and is binding after it**: re-admitting a day whose
economics have been seen is selection on the outcome. The reviewer attacked
removal, re-citation and day-substitution on a real git history; each refuses
**by name**, and the guard is proved non-vacuous. The later DA25 repair wired
`counts_toward_race` into the verdict checker and preflight, so the withdrawal
now binds both the registry and the count: 08-29 remains eligible on the four
conjuncts but does not accrue.

| # | still open | status |
|---|---|---|
| 1 | **the Phase-2 winner** | the race decides it. Not before G=5 |
| 2 | the `clob_v4` cite — its R-340 resolves but does not name `clob_v4` | **queued**, packaged by DA with both answers and what the runner would do under each |

---

## 8. Sister program P-2026-002 (HF market making) — a gate opened today

Read `orchestrator/PROGRAMS/P-2026-002-hf-market-making/workspace/HANDOFF.md`.
E1 is complete and the verdict is **overlay-only**:

- **E1-B, standalone Binance MM: no real pass.** Majors negative pre-fee
  (BTC −0.19, ETH −0.23 bps at 30 s). ADA's +2.44 screen pass is an H1
  fat-tick artifact — notional-weighted it is **−0.32 bps**; the same flip
  appears on every wide-tick name.
- **E1-A, passive-execution overlay for the XS book: PASS, audit-robust.**
  T_p=600 s: touch 3.45 [3.11, 3.79] / sweep 6.26 [5.76, 6.75] against an 8 bps
  capstone; stale-shadow 7.20; excl-ICP 6.15; per-symbol max 7.53. Still
  maker-optimistic (H1, no queue position).

**The E2.0 / E2-A gate is now MET and unworked.** It required 14 days of L2;
measured 2026-09-03 03:18Z: `data/mm_hf/raw/depth20/BTCUSDT` holds **351 hourly
files, 2026-08-19 12:00 → 2026-09-03 03:00Z**, and the Hyperliquid side
(`hl_raw/l2Book/BTC`) holds **350**, 16 symbols each, 44 GB total. Both
collectors are live. E2.0 is the true-mid recompute (notional-weighted) that
voids or settles ADA; E2-A resolves the overlay bracket with real books.
**Nothing has been dispatched against it** — this program has been calendar-
blocked since 2026-08-19 and the calendar has now moved.

---

## 9. How to reload this program's context

0. **If you are the coordinator**: `workspace/COORDINATOR_RUNBOOK.md` — the
   cold-start order, the seat→pane map (re-derived, never hardcoded), dispatch
   and register mechanics, the verification-battery pattern, what dies when a
   coordinator session is cleared (the `/loop` wakeup and the commit monitor —
   nothing in git or `data/`), and the standing prohibitions.
1. `workspace/SEAT_PROTOCOL.md` — who may write what. **One writer per state
   file.** `STATUS.yml` and `HANDOFF.md` are MEM's; `COORDINATION.md` R-entries
   are append-only; this file is the coordinator's.
2. `workspace/COORDINATION.md` — the register. Read the **last five R-entries**
   and the Q-filing table, not the whole file.
3. This file, then `workspace/HANDOFF.md` (11.8k lines — read the dated entries
   at the top and the sections you need).
4. `STATUS.yml` — 15 tasks with statuses, 317 flags, 10 standing rules.
5. `live/pm_research/plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN.md` §10 — the
   governing TODO. The stateful cancel×skew worksheet's 47/113 checkbox count is
   **not** project completion.

**The reliability rules in `CLAUDE.md` are not style.** Each was bought with
dissolved work: rule 11 (choosing after seeing voids the test), rule 12 (a
freeze is a commit), rule 15 (every checker ships a falsifier), rule 16 (verify
at the artifact a claim names), rule 10 (compute predicates, never print
conclusions) — this consolidation applied rule 10 and rule 16 to the program's
own documents and found two errors in them (§2).
