# Review — DE rounds 31 + 32 (the Phase-4 diagnostic DECLARED; three instruments; the runner's shell)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `e52d183`** (covering `d94aa1e`); rows Q-DE-49 (`f76d6a5`), Q-DE-50
(`03c5a23`). Ruling of record **R-459**.
**Request of record:** `REQUEST_DE_ROUNDS_31_32_2026-09-02.md`. ONE filing, per R-377.

**THE FIRST CHECK, DONE FIRST.** `DE_PHASE4_PROTOCOL_DRAFT.md` hashes
**`ab07fd71c9fc2bff…`** — the value the addendum binds and the request names. Its last commit is
`cdb16d4` (R-397), before this round: **the frozen document is UNEDITED**, by sha and by history.
The addendum hashes `35e8aba1381cfa4e`, the runner `cdf9541756fc5f7b` — both as stated.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach e52d183`; `~/ctaNew-wt-de`
never read (committed blobs only); the main tree's `be_forward_day.py` never read, run or cited
(standing_rule 9). `__pycache__` cleared before every execution; both launchers; streams captured
separately; every mutated file restored **byte-identical** (shas above, re-verified after the
battery) and `git status --short` **0 lines**. Nothing under `data/`: `derived/` **173** before and
after, and `data/pm_5min/derived/phase4_diag_r459/` **absent** throughout. Synthetic inputs only —
the §3 population was never replayed. No unit, timer or anchor; `DA_MIDNIGHT_MODE` never set;
`da_midnight_verify.sh` never run; register read-only.

## 0. Counts

| module | `-m` | by path |
|---|---|---|
| `de_rho_estimator` | 21, rc 0 | 21, rc 0 |
| `de_score_stream` | 24, rc 0 | 24, rc 0 |
| `de_matched_random_control` | 21, rc 0 | 21, rc 0 |
| `de_phase4_diag_runner` | 35, rc 0 | 35, rc 0 |
| `de_phase4_protocol_check` | 21, rc 0 | 21, rc 0 |
| `de_ratification_check` | 184, rc 0 | 184, rc 0 |

Zero stderr under every one. `--run` → **rc 2**, `error: unrecognized arguments: --run`.

## 1. The declaration against the ruling and the frozen document (item 1)

Read line by line against R-459 §2/§4 and the frozen §3/§4/§5/§6. **Nothing widens, narrows or
renames what the ruling lifted**, and I found no cell the addendum declares that the frozen grid
does not permit, nor the reverse:

| declared | frozen / ruling | agrees |
|---|---|---|
| 471 windows, btc 234 / eth 237, 08-24 + 08-25, 3 excluded for Binance discontinuity, `clob_v3_1` | `DRAFT:78-80` verbatim | yes |
| CONSUMED, `is_a_validation=false`, G = 0, no interval, `DIAGNOSTIC_NEVER_EVIDENCE` | `DRAFT:82-88`, R-459 §2 | yes |
| nine latency rungs, **not** a selection axis | `DRAFT:101` | yes |
| budgets 5/10/15 %, selection axis, **all reported, none selected** | `DRAFT:103`, R-459 §2 (as corrected in R-461) | yes |
| PRIMARY: btc, 250 ms, 10 %, reset-cost-at-start False, reduce off, conjunction over both repost-fill arms and both protection modes | `DRAFT:118-120` | yes |
| null: matched on action count, side, hour; compared on the DECISION metric, never a proxy; N ≥ 200 | `DRAFT:148-152` | yes |
| null at PRIMARY (200 draws), eth conditionally, every other cell a labelled point estimate | R-459 §4 asks the addendum to name exactly this | yes |
| heads: `Q1_arrival` of `composed_lgbm` vs `linear_d_{coin}` (btc `18701008c2bd18c6`), manifest-bound | R-459 §2 | yes |

Two things I checked because they could have been selection in disguise. The eth null is
conditional on **wall-clock** ("if the PRIMARY completes inside its window"), not on any number the
run produces — rule 11 is untouched. And the §d table's "explicitly NOT run" list closes the space:
a later receipt carrying one of those cells would be a cell nobody declared, which is the
falsifiable form. **No DE31 finding on item 1.**

## 2. The arm name, and what `run_cell` actually composes (item 2)

The addendum defines `CONDVALUE_OVER_SKEWED_REF` as a conditional-value head over the frozen
skew-ON reference, **no interaction, no fair-price term**. `harmful_stateful_policy`'s parameter
set (`_REQUIRED_PARAMS` + `_OPTIONAL_PARAMS`, `:167-179`) has **no interaction and no fair-price
parameter at all**, and `run_cell`'s `params` sets `predictor_enabled = arm != "QR_SKEW_ONLY"` and
nothing else arm-dependent. So the composition that would be replayed **is** the declared one, and
`X_SKEW`'s interaction is not merely absent but inexpressible. Answer to item 2: **yes.**

What that same reading exposes is filed as **DE32-R4**: the arm's identity is carried by the
caller's dict key. `run_cell` iterates `scores_by_arm` without checking its keys against `ARMS`,
gives **every** arm identical params, and never records the reference's provenance — so
`QR_CANCEL_HOLD_X_SKEW` and both `CONDVALUE_OVER_SKEWED_REF/*` arms are one configuration under
three score streams, and the addendum's careful name resolution is enforced nowhere in code.

## 3. `de_rho_estimator` — driven (item 3)

(a) **The estimand matches the frozen `:150`.** `adverse = −size·sign·(mid_markout − px)`,
`spread = size·|px − mid_at_fill|`, **ratio of sums** over the same fills — a value ratio, not a
mean of ratios. Driven against hand arithmetic: two fills (BUY_UP 10 @50, mid 49.5 → markout 48;
SELL_UP 10 @49, mid 49.5 → markout 50.5) give adverse **35.0**, spread **10.0**, rho **3.5** —
exact.
(c) `spread == 0` → `rho None` with `rho_undefined_reason`, never `inf`; the `harm/sac` proxy is
carried under `rho_captured_over_sacrificed_PROXY` beside a field naming it a different quantity —
**never merged**.
(d) Falsifiers, all driven by me and all refusing **by name**: `latency_ms=None` and `-1`
("latency enters the estimand … cannot be defaulted"), an unknown side, a fill missing a field. A
fill inside the window is **received and charged** (status `IN_LATENCY_WINDOW`, counted, rho 4.0).
`SIDES` **is** the policy's object (`RHO.SIDES is HSP.SIDES` → True, `('BUY_UP','SELL_UP')`) — a
restatement could drift; this cannot.

(b) **The reachability reference is not the frozen document's**, and this is **DE31-R1**.
`classify` (`:98`) computes `reach_ns = gen_start_ns + L`. The frozen Cap 1 (`DRAFT:36-39`) defines
the window as **`t + L`** — the decision row's time. Since generation start precedes the decision
row's `t`, `gen_start + L ≤ t + L`, so the estimator labels **REACHABLE** fills that arrive before
`t + L`, which the protocol says the policy could not have prevented. rho itself is unaffected
(both statuses are in `_COUNTED`, which is right), so this is a **labelling** defect — but §e(4) of
the addendum promises those statuses and their counts as the reported exclusions, and
`adverse_by_status` splits the loss on that boundary. Driven: a fill at `gen_start + 100 ms` with
L = 250 ms reads `IN_LATENCY_WINDOW`; the same fill referenced to a decision row 200 ms after
generation start would too, but one at `gen_start + 300 ms` with `t = gen_start + 100 ms` reads
REACHABLE here and in-window under Cap 1.

## 4. `de_score_stream` — driven (item 4)

All four, by name: a **cross-coin** head (`linear_d_btc.json` verified, then used to score eth) →
*"the verified files … belong to another coin. A cross-coin fit LOADS PERFECTLY and scores
nonsense"*; **moved bytes** (manifest sha altered) → refused before load; **`verified={}`** →
*"the manifest check is not optional"*; a **NaN** score → refused **at the row** with its index.
The adapter's own output is fed to `harmful_stateful_policy.validate_scores` and accepted (rule 16,
the check that caught the `("BUY","SELL")` restatement). The **permutation control** driven on 200
synthetic events with an informative scorer: real lift **1.0000**, permuted max **0.1800**,
**0/200** permutations reach it.

## 5. `de_matched_random_control` — the per-stratum question, measured (item 5)

Round 32's fix is right in kind: with freedom present, a draw identical to the treated arm is
**REFUSED** by name (*"cancelled exactly the treated arm's own 5 actions while its strata had room
to differ"*). And the case the item asks about is real: a pool whose strata **all** have one member
forces the draw to reproduce the treated arm, and `refuse_if_not_random` **passes**. That is the
correct refusal behaviour — refusing there would refuse the only legal draw — but the pass is
**silent**, and a null with no freedom yields `q50 = q95 = max` while reading as an interval. Filed
as **DE31-R2**, and it is not hypothetical: see DE32-R2, where the runner's own shipped fixture is
exactly that pool.

## 6. `de_phase4_protocol_check` — the grid binding, driven both ways (item 6)

| mutant | protocol check |
|---|---|
| a **200 ms** rung added to `LATENCY_RUNGS_MS` (code only) | **rc 1**, red at *"THE RUNNER'S LATENCY AXIS IS THE ADDENDUM'S"* |
| a **25 ms** rung added (its digits appear inside "250") | **rc 1**, red — the pinned `== 9` closes the substring hole I expected to find |
| a **400 ms** rung added to the **addendum** only | **rc 0, GREEN at 21** |

So the binding catches code-widening (including the substring case I tried to sneak past it) and
**does not catch declaration-widening**: an addendum that declares ten rungs while the runner
implements nine passes. Filed as **DE32-R3**. The frozen sha binding, the R-459 naming, the three
required declarations and the unrelated-sha non-substring check all hold.

## 7. The runner — DE32-C1..C5, each at the blob (item 7)

| finding | verdict | what I measured |
|---|---|---|
| **DE32-C1** no `--run` | **CONFIRMED** | `main` (`:578-586`) parses `--selftest` only; `--run` → argparse **rc 2** |
| **DE32-C2** the feed is named, never invoked | **CONFIRMED** | `tranche_table` occurs **once** in the runner — in the docstring at `:25-26`, present tense (*"The feed is `phase4_generation_tables.tranche_table(..., declare_cap=True)`"*); only `FILL_HORIZON_S` is imported (`:61`); the only other file mentioning `tranche_table` is its own definition module — it still has **no consumer** |
| **DE32-C3** `RHO` imported, never called | **CONFIRMED** | the single reference is `RHO.EXPECTED_CHECKS` at `:567`; `run_cell` emits `cost_adjusted_value_cents`, `n_cancels`, `net_diff_cents` and **no `rho`/`retention_share`**, both of which `evaluate_predicates` reads with `.get` — the selftest plants `rho=0.8`/`1.2` at `:513-515` |
| **DE32-C4** the null is synthetic and valued on the forbidden proxy | **CONFIRMED, and worse than stated** | `pool` uses `SIDES[i % 2]`, `hour = i % 24`; `treated` is **not** the top decile by harm — it is the first `⌈n/10⌉` positive-harm slugs in **lexicographic slug order** (`sorted(harm_by_slug)`); each draw is valued `sum(harm_by_slug[...])`, a **harm sum**, which `evaluate_predicates` then compares against `net_diff_cents` in **cents** (`beats_null_q95`, `:265-268`) — two different units |
| **DE32-C5** undeclared thetas, single-valued conjunction axes | **CONFIRMED, and see DE32-R1** | `theta_cancel` 0.8 / `theta_repost` 0.3 as `c.get` defaults, `repost_dwell_s` **hardcoded** 2.0 (`:288-290`); `protection_mode` defaults to `PROTECTION_MODES[1]` and `repost_fill_model` to `REPOST_FILL_MODELS[0]` (`:293-297`) where PRIMARY is the **conjunction over both** (`DRAFT:118-120`) |

**The refusals are sound** — driven: a rung off the frozen ladder, a budget off the frozen three,
and `enable_reduce` ON each **REFUSE by name**, while the PRIMARY cell is accepted (the positive
control). `OUTDIR` is absent and stays absent.

## 8. The estimand's cap — ruled (item 8)

**The frozen document is NOT silent, so the premise's second branch does not arise.** Cap 2
(`DRAFT:41-45`) declares it in the protocol itself: *"the one-second horizon is part of what the
cell means (R-165(2) item 5) … capped at `FILL_HORIZON_S = 1.0 s`, so any cell built on them
estimates 'value preventable WITHIN ONE SECOND of the decision row', not 'value preventable'"* —
the same words the receipt carries at `:205-208`. So the cap is **declared before everything**, the
receipt **restates** it, and rule 6 is satisfied: this is not a limit introduced after the
declaration.

Two residuals, both worth one line each and neither a rule-6 breach. The **addendum** never
mentions the cap although it restates the population, the grid, the arms, the null and the
predicates from the same frozen document — a reader of the declaration alone would not know the
estimand is one-second-capped, and the addendum is what the run is measured against. And the cap's
**enforcement** is `tranche_table`'s refusal to emit without `declare_cap=True`; the runner imports
the constant and never calls the emitter (DE32-C2), so at this tip the receipt would assert a cap
that nothing on its path enforces. Both close in the round that wires the feed.

## 9. IR-R4 as narrowed — which half is closed (item 9)

**Neither, at `e52d183`.** DE reports the "no production consumer" half closed by the runner; the
runner names `tranche_table` in one docstring line and never calls it, and no other module
consumes it, so `tranche_table` has exactly the number of production consumers it had before this
round: **zero**. The "built at run time from the archive, refused if absent" half is untouched by
construction (there is no run path). The addendum's own §f wording — *"Until the runner supplies
that table this diagnostic cannot run, and no fixture stands in for it"* — is still the accurate
statement; the row's narrowing is ahead of the artifact. Recorded as part of **DE32-R5** below
rather than as a separate finding, since it is the same fact as C2 read from the row's side.

## 10. Counts and discipline (item 10)

All six counts reproduced on both launchers, zero tracebacks, zero stderr; `--run` rc 2; the three
shas above unchanged after the battery; `derived/` 173 before and after with `phase4_diag_r459/`
absent; the two plan files unedited; no unit, timer, anchor or scope touched; register read-only.

## Findings

| id | severity | where | one line |
|---|---|---|---|
| DE32-R1 | **MEDIUM** | `:288`, `:142-146` | the **budget axis never reaches the replay** — `theta_cancel` is the constant 0.8 while the bound thresholds artifact is keyed by budget |
| DE32-R2 | **MEDIUM** | `:313-326`, `:509-512` | the shipped **200-draw null is degenerate**: 20 one-member strata, so all 200 draws are identical |
| DE32-R3 | LOW-MEDIUM | `de_phase4_protocol_check:219-225` | the grid binding is **one-directional**: a rung added to the addendum only passes |
| DE31-R1 | LOW-MEDIUM | `de_rho_estimator:98` | reachability is referenced to `gen_start + L`, the frozen Cap 1 says **`t + L`** |
| DE32-R4 | LOW | `:283-300` | **arm identity is the caller's dict key**: every arm gets identical params, keys are unchecked, the reference's provenance is unrecorded |
| DE31-R2 | LOW | `de_matched_random_control:120` | a **forced** draw passes silently — no freedom is reported, so a degenerate null reads as an interval |
| DE32-R5 | LOW | row Q-DE-50 | the row's IR-R4 narrowing and the runner's docstring both describe a feed the code does not call |

**DE32-R1 — the budget axis does not reach the replay.** `validate_cell` refuses a budget off the
frozen three and the budget is carried into the receipt key (`:249`), but `run_cell`'s `params`
has no budget at all: the only traffic control the policy exposes is `max_cancels_per_minute`,
which the runner sets to `inf`, and `theta_cancel` is `c.get("theta_cancel", 0.8)`. The artifact
the addendum binds — `phase2_fits/lgbm_thresholds_btc.json` — is keyed **exactly** by the three
budgets: `{"5%": 0.7075, "10%": 0.3245, "15%": 0.1793}`. So as it stands the three budget cells are
the **same replay**, at a threshold that is none of the three, and the receipt would label them
5/10/15 %. This is DE32-C5's undeclared default read from the other end, and it is the half that
turns a labelling defect into three identical cells. Closure: select `theta_cancel` from the
head's thresholds by the cell's budget, and refuse a cell whose budget has no threshold in the fit.

**DE32-R2 — the null that cannot vary.** Measured on the runner's own selftest fixture (20 slugs,
`side = SIDES[i % 2]`, `hour = i % 24`): **20 strata, maximum stratum size 1**, and over 200 seeds
`MRC.draw` returns **one distinct draw** and **one distinct value** (200.0), so
`q50 == q95 == max`. The check at `:509-512` says *"THE NULL RUNS 200 DRAWS at a declared cell —
the protocol's minimum (§6)"*: 200 draws that cannot differ are one draw with a count. With C4's
harm-sum valuation this is also the quantile `beats_null_q95` compares against a cents figure.
Closure: strata with real (side, hour) from the population, and a freedom check — DE31-R2's —
that refuses or labels a null whose draws are forced.

**DE32-R3 — the binding's missing direction.** Driven above. Closure: parse the rung list out of
the addendum's §b cell and assert **set equality** with `LATENCY_RUNGS_MS` **and** with the frozen
§4 ladder, rather than counting the runner's rungs that appear in the text.

**DE31-R1 — the reference point.** Reasoning and drive in §3. Closure: carry the decision row's `t`
per fill (the score stream already emits it) and compute `reach_ns = t + latency_ms·1e6`; keep both
statuses counted, since the charging rule is what makes the estimator honest.

**DE32-R4 — arm identity.** Closure: a `name → params` table with the conjunction axes expanded,
and a refusal for any key not in `ARMS`; record the reference replay's provenance in the receipt
beside the arm names.

**DE31-R2 — freedom unreported.** Closure: return the per-stratum alternatives count from `draw`,
and have the null label itself degenerate when the total is zero.

**DE32-R5 — the row and the docstring are ahead of the code.** The docstring's *"The feed **is**
`tranche_table(..., declare_cap=True)`"* and the row's "no production consumer half closed" both
describe a call that does not exist at this tip. This is BE's `:1388` shape (a comment contradicting
its own module) and the cheapest closure is the honest tense: *will be*, until the run round lands.

## Disposition

**RELEASE `e52d183` (covering `d94aa1e`) as a DECLARATION AND THREE INSTRUMENTS — and the runner is
not released as a producer.** The declaration is exact against R-459 and the frozen document, which
is unedited by sha and by history; it was written before any cell exists; and the three instruments
each ship falsifiers that fire by name, which I drove rather than read. The runner refuses every
undeclared cell, writes nothing, and cannot be run at all — which is why the seven findings below
cost nothing today and would each corrupt cells tomorrow.

**Before any cell is produced**, the record should show: DE32-C1 (a run path), C2 (the feed
invoked, with `declare_cap=True` rather than an imported constant), C3 (rho and retention computed
per cell), C4 + DE32-R2 (a null on real strata, valued on the decision metric, with its freedom
reported), C5 + DE32-R1 (thresholds from the fits by budget, and the conjunction axes expanded).
DE31-R1 and DE32-R3/R4 are one-line closures that should travel with them. None of this holds up
tonight's other landings; the run is a later round, which is the order the addendum itself sets.
