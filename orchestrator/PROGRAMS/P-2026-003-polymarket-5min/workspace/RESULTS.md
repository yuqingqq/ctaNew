# RESULTS — P-2026-003 Polymarket crypto 5-min

Consolidated 2026-09-03T03:23Z, substantially rewritten 2026-09-03T08:29Z,
reliability-corrected 2026-09-04T09:59Z, extended 2026-09-04T13:48Z with
§0, **reconciled through R-531 at 2026-09-04T14:25Z**, and annotated for the
user-directed v2 build at 2026-09-04T15:27:56Z, pipeline-status updated at
2026-09-04T16:44:23Z, and prospectively resumed for Gate 1c at
2026-09-05T00:58:42Z, returned to a fixed Gate-1c support stop at
2026-09-05T01:10:57Z, then prospectively resumed for Gate 1d at
2026-09-05T05:03:29Z, and advanced to the declared Gate-1e accounting audit at
2026-09-05T05:14:39Z, which reached its fixed refusal at
2026-09-05T05:28:23Z; the continued Gate-1f input audit refused at
2026-09-05T05:50:05Z, with bounded regression reconciliation completed at
2026-09-05T09:54:58Z. **Current state: the lifecycle gross ledger is complete,
but Gate 1 refused because owned-order per-fill maker fees are unavailable;
the v2 route is stopped at 1/7. No strategy-net or matched decision statistic
exists, no gross value is promoted as its substitute, and no broad survey
result exists. Single writer remains the coordinator; the later
user instruction is not yet a register entry.** This
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

> **PLANNING/BUILD UPDATE — NOT A RESULT.** The prospective governing plan is
> now `live/pm_research/plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN_V2.md`.
> It moves canonical action identity and a >=200-draw side/hour/action-count
> matched control ahead of broad integration, requires stateful cascade
> economics before promotion, and defers fair price. Two Gate-0 instruments
> were added and passed synthetic batteries only: the exact-tranche action
> ledger and static action-bundle control. No artifact below was changed,
> re-scored or upgraded by this work.
>
> **15:50Z continuation, still not a result:** the opt-in reference producer
> now retains missing-markout tranche identities, the canonical earliest-row
> action adapter is built, and the composed Gate-0 runner passes its synthetic
> seams. No real-data receipt or economic conclusion was created by these
> additions.
>
> **16:06:44Z Gate-0 pipeline smoke, still not an economic result:** under a
> one-CPU/3 GiB cap, the one-window BTC consumed-development receipt
> `p003_v2_gate0_smoke__20260904T160623Z.json` (sha256
> `d63ccc59bae1e733a2fa4c840ce2b1c2bdc33494ce989df55f3816f4fd906be7`)
> reconciled 3,557 canonical actions to 3,557 reference generations and 458
> exact fill identities. All six identity/partition/matched-null predicates
> were true; the static null contained 200 count/side/hour-matched draws.
> Maximum in-process RSS was 347,080 KiB and wall time 11.70 s. The 355 treated
> actions are a deterministic hash-ranked wiring probe declared without fill
> values, not a predictor or benchmark. The static screen is not
> cascade-feasible and no economic values from it are interpreted here.
>
> The first capped attempt reached its 3 GiB job limit while the historical
> selector built a global Binance gap index and was manually stopped; it wrote
> no artifact. The successful receipt supersedes that attempt with an exact
> interval-local selector reading three adjacent hourly files. This is retained
> as a resource-path correction, not omitted as an unsuccessful result.
>
> **16:25:59Z Gate-1 control refusal, also not an economic result:** the acting
> iid-permutation adapter passes 18 synthetic checks and its smoke wrapper
> seven, including the required stateful mismatch falsifier. On the same
> one-window consumed-development population, only 1 of 4,000 proposals
> matched treatment's realised `CANCEL_ISSUED` counts by maker side/UTC hour;
> 200 accepted draws were required, so the module refused and published no
> smaller null or economic comparison. Failure receipt
> `p003_v2_gate1_smoke__20260904T162106Z.json`, sha256
> `ede26d60fdb425e9d760adca48e24191620c2fb15a5fe70028e124e758b1ebc9`.
> Runtime was 4m44s on one CPU with a 250.5 MiB peak under 3 GiB. The result is
> a control-design finding: uniformly redistributing high-score events changes
> hold/repost suppression and therefore the action count. The plan now forbids
> raising that rejection budget and routes to a constrained exact-fiber switch
> null with explicit support/mixing diagnostics. Gate 1 remains uncleared.
>
> **16:39:29Z constrained-null refusal, still not an economic result:** the
> replacement module passes 13 synthetic checks and its wrapper seven in both
> invocation modes. Its capped real receipt
> `p003_v2_gate1_switch_smoke__20260904T163438Z.json` (sha256
> `cdff1a14de7ecff3351dc90da224e2d44bad43d13ef67e66934b859d777a36a9`)
> found a traversable exact-count component: 2,443 of 5,000 symmetric proposals
> moved, all four chains left treatment, and 400 retained samples contained 399
> distinct states with every stateful/count/score/source identity true. It
> nevertheless failed its predeclared mixing bar: ESS for distance from
> treatment was 10.53 versus 100 required. Every chain's last 20 samples also
> sat much farther from treatment than its first 20 (means 237–251 versus
> 114–129), directly showing inadequate burn-in. The aggregate null comparison
> is absent. Individual audit-trail state values remain explicitly partial and
> are not interpreted. Runtime was 249.89 s on one CPU; receipt max RSS was
> 333,856 KiB (systemd cgroup peak 250.8 MiB), both below 3 GiB.
>
> The plan's stop applies: Gate 1 is not cleared, overall progress remains 1/7,
> and Gates 2–6 do not start. Extending attempts/burn-in/thinning or changing
> the switch kernel after viewing this consumed path is prohibited. Resumption
> needs a new prospective control estimand and user ruling, not another retry.
>
> **Post-receipt code verification:** all ten v2 batteries pass under one
> CPU/1 GiB and the parent diagnostic suite passes 223/223. The parent suite
> initially exposed a missing static reached-function edge caused by the
> injectable selector seam; the direct historical call was restored without
> changing the smoke's injected branch. Consequently the current working tree
> differs from each successful smoke receipt at exactly one recorded identity
> file, `de_phase4_diag_runner.py`. The receipts retain their own source hashes,
> remain uncommitted/unfrozen, and were not rerun after the stop.
>
> **2026-09-05T00:58:42Z prospective continuation — not a result:** the user
> authorised a genuinely different Gate-1c estimand. The plan now fixes a
> sequential hard quota on actual state-machine cancels after uniform random
> score proposals: under-quota proposals refuse; above-quota proposals suppress
> later cancel crossings using a fixed between-threshold score, without
> force-cancels or economic outcomes in selection. The fixed gate is 200
> accepted draws in at most 1,000 proposals, with synthetic work capped at one
> CPU/1 GiB and one consumed-window smoke at one CPU/3 GiB. It has not yet been
> built or run. The iid and switch failures remain consumed; their budgets and
> kernels are unchanged. Gate 2 remains unauthorised.
>
> **2026-09-05T01:10:57Z Gate-1c support refusal — not an economic result:**
> the new controller passes 14 synthetic checks and its wrapper eight. The
> fixed one-window receipt
> `p003_v2_gate1_quota_smoke__20260905T010921Z.json` (sha256
> `e10dec7167a1b61a17c87b3ff0d19cd6c11692a6280035181e9cf5f1985a2ab8f`)
> set the treated actual-action quota at BUY_UP 150 / SELL_UP 110 in UTC hour
> 13. Only 16 of 1,000 independent proposals reached both counts versus 200
> required; all 984 rejections were `UNDER_QUOTA`, and only 16 distinct realised
> action sets existed versus the fixed 50 minimum. All accepted mechanics,
> source and exact-quota identities were true. The matched-null status is
> `ABSENT_REFUSED_SUPPORT_GATE` and its aggregate partial metric is null; the 16
> retained draws are audit records, not a smaller null. Runtime was 82.81 s,
> process max RSS 337,028 KiB, under one CPU/3 GiB. No profitability, harm or
> benchmark-performance inference follows. The budget/rule cannot be changed
> on this consumed window, Gate 1 remains refused, and Gates 2–6 do not start.
> Final consolidated verification: all 12 v2 module/wrapper batteries and the
> 223-check parent diagnostic suite pass under one CPU/1 GiB. Exact identity
> comparison finds 11/12 Gate-1c receipt files byte-identical to the current
> tree; only the governing plan differs because the prospectively hashed
> declaration was extended after execution with this result/status. No named
> source-code file changed after the receipt.
>
> **2026-09-05T05:03:29Z prospective continuation — not a result:** the user
> authorised a different Gate-1d estimand. Before output, the plan fixed full
> enumeration of within-side/hour cyclic rotations of the complete clustered
> treated score sequence, exact actual-issued-count conditioning, assignment
> deduplication, a minimum 200 distinct joint phases, and a 200-draw uniform
> without-replacement full replay. It uses no quota suppression, force-cancel,
> proposal limit or economic outcome in selection. Synthetic work is capped at
> one CPU/1 GiB and the only real one-window smoke at one CPU/3 GiB with a
> ten-minute ceiling. It has not yet been built or run; Gate 2 remains off.
>
> **2026-09-05T05:13:04Z Gate-1d support result — not an economic result:**
> receipt `p003_v2_gate1_cyclic_smoke__20260905T051116Z.json`, sha256
> `8a97102cc11f5f8c94f1545deb0df75a82d6bb44a6970fd5fc4faaf723074650`.
> Complete enumeration covered 1,891 BUY and 1,666 SELL offsets; 18 and 40
> exact-count phases produced 720 joint assignments. The fixed 200 uniform
> without-replacement full replays were distinct and every source, action-count,
> score-clustering, separability and stateful identity was true. Runtime 99.85
> s, process max RSS 338,448 KiB under one CPU/3 GiB. Economics remain
> `INCOMPLETE_NOT_STRATEGY_NET`; no partial value is interpreted.
>
> **05:14:39Z Gate-1e declaration — not a result:** pin that receipt and its
> 200 phases; reconcile baseline/treatment/control received fills, five-second
> P&L, spread/adverse, rho, terminal inventory, cancel/hold/repost/reset paths
> and required fees. A missing per-fill maker-fee ledger or other required term
> makes strategy net and the aggregate decision null unavailable and refuses
> Gate 1. Gate 2 remains off.
>
> **05:28:23Z Gate-1e result — accounting complete, Gate 1 refused:** receipt
> `p003_v2_gate1_economics_smoke__20260905T052605Z.json`, sha256
> `e78fe495846cf22e834b63e04aea445cf1616563cb932a11f304d3a7ba2abd42`.
> The 5,869-row / 3,557-action population, Gate-1d receipt SHA, complete
> 720-phase support, exact 200 offsets, all score-assignment hashes and all
> realised-action identities reproduced. QR_SKEW_ONLY, treatment and 200
> controls passed every gross fill/decomposition/terminal/rho/lifecycle/rate
> identity. All 202 maker-fee ledgers are
> `UNAVAILABLE_NO_PER_FILL_MAKER_FEE`; all 202 fee-adjusted strategy nets, the
> treatment decision value and matched decision null are therefore null.
> Public taker/trade fees were not substituted, and owned-order ack/fill
> causality remains explicitly unobservable. Runtime 22.98 s, process max RSS
> 338,556 KiB, one CPU/3 GiB, swap off. This is an identification refusal—not
> a negative P&L result—and triggers the declared stop. Gate 2 and Gates 3–6
> did not start; overall progress is terminal at 1/7.
> Gate-1e checkpoint regression closure: all 16 then-existing v2
> module/wrapper batteries and the 223-check parent diagnostic suite passed
> under one CPU/1 GiB with swap off.
>
> **05:50:05Z Gate-1f acquisition result — no economic result:** corrected
> receipt `p003_v2_gate1f_owned_source_audit__20260905T054941Z.json`, sha256
> `c99109943de37d37d2fc8358628640214d489752e96bb8ca4f86e144bf197f47`,
> supersedes `...T054848Z.json` (`bf3d01fa...b292022`), whose path census missed
> Tier-1's distiller directory; every gate field/refusal was unchanged. The
> fixed external owned-execution manifest is absent. Public raw date directories
> span August 19–September 5 and Tier-1 public trades August 20–September 2,
> but no public source binds client order, venue ack, maker fill and exact fee.
> The contract module passes 11 checks under one CPU/512 MiB. Decision metric
> null, Gate 1 refused, Gate 2 off. More public tape cannot close this join.
> Latest bounded regression at 2026-09-05T09:54:58Z: all 17 current v2
> batteries pass (182 checks total), and the parent suite passes 223/223, under
> one CPU/1 GiB with swap off. No result or gate changed.

---

## 0. 2026-09-04 — the first absolute economics, and what they say

**This section supersedes §1 wherever they disagree.** Until today this
programme had **no absolute economic number at all** — every result was
candidate-versus-incumbent, model against model, with no no-cancel baseline
anywhere. §1's "profitability withdrawn" was correct and is now superseded by a
measurement rather than by a restatement.

> **REGISTERED RESULT STATE (R-531, 2026-09-04T14:25Z): HALTED.** All heavy work stopped
> on the USER's instruction; the collectors continued running and the work was
> preserved. The only absolute-economics evidence remains the one BTC
> development hour below. The proposed broad `V_oracle` survey **did not run**:
> `survey.py` raised `AttributeError` because it treated `ref[slug]` as a dict
> when it is a list, and exited rc 1. There is no ~40-cell or 138-cell result.
> The declared survey span contains **G = 0 complete UTC days**, so it could not
> have produced decision-bearing day-cluster evidence even if the script had
> completed. Do not describe that survey as running, completed, or
> decision-bearing.
>
> **Later scope update:** at 15:27:56Z the user authorised the prospective v2
> plan and lightweight build. Heavy result-bearing work remains off. This
> changes the next plan, not any number or reliability classification here.

> **READ THIS BEFORE THE NUMBERS.** This section states its current frame with
> confidence. **Two previous frames were stated with the same confidence today
> and both were withdrawn within hours:** *"cancelling does not pay"* →
> *"the book has too little adverse selection to harvest"* → *"the lever is not
> exhausted, the rankers do not work"*. **Five coordinator rulings were reversed
> in about two hours** (R-513→R-515, R-516(C)→R-518, R-520/521's three
> withdrawals, R-523→R-528, R-528→R-529). Every reversal was caught by a seat
> checking at an artifact — which is the system working — but **the rate of
> correction is itself a fact about how much weight this section can bear**, and
> a reader who takes the current frame as settled is making the mistake the
> previous two frames' readers would have made.
>
> **Two further objections are OPEN and unanswered, raised independently by two
> seats:** the **701% ceiling has NO NULL** — `Σ|P&L|` over losing fills is
> large for *any* noisy book, and the comparison that matters is against what a
> **random or naive policy captures**, not against the net; and **`V_oracle` may
> have `r`'s disease** — it counts every negative fill as capturable, is
> unattainable under the cascade, and nobody has tried to construct two books
> with identical `V_oracle` and different **attainable** value. **Until those are
> answered, treat the ceiling figures as PENDING A NULL.**
>
> **And the whole of this section is ONE HOUR** — twelve contiguous 300 s
> windows of one coin. Every ratio, the cascade decomposition, the break-even
> and `V_oracle` have **n = 1 in the only unit that clusters**.

**All figures: btc, ONE CONTIGUOUS HOUR (2026-08-24 13:50–14:50Z, 12 windows,
4,315 fills), development evidence, point estimate and NO interval** — 12
windows is below the 5-complete-day cluster floor. `is_a_validation` false.
Base economics artifact
`data/pm_5min/derived/de_section81_arms__20260904T134055Z.json`; ceiling
extension
`data/pm_5min/derived/de_section81_arms__20260904T135755Z.json`. Both were
emitted from clean trees and their carrying commits are ancestors of the branch;
the base artifact has 0 of 7 code-identity differences at its carrying commit.
The earlier `125340Z` emission named a carrying commit **not on the branch**, and
the shared economic quantities are bit-identical — the defect was the record,
not the result.

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
- **"99% of this book clears break-even fourfold" is WITHDRAWN — the bound
  FLIPS.** De-tailing must be symmetric, by |P&L|; ranking by signed net removes
  only the *good* outliers, which is selection. Same algebra, opposite
  direction: **winner-ranked r_ex ≥ 110.58%, extreme-ranked r_ex ≤ 26.76%** —
  and **26.76% is BELOW CONDVALUE's 27.60% break-even**. Body r is 1.1077
  removing the winners, 0.2504 removing the extremes, 0.1863 whole-book: **same
  book, three answers.** That the two rankings disagree **1.13 against 0.10 on
  the same 43-fill budget** is itself a measurement — the extreme set contains
  large **losers** roughly offsetting the large winners.
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

**The replacement proposed that day was `V_oracle`** — the sum of |P&L| over
fills whose P&L is negative. It is model-free and is an unattainable
perfect-foresight ceiling for an overlay that can only decline fills. **R-531
qualifies it pending two missing checks:** a null against random or naive-policy
capture, and a test that separates equal `V_oracle` books with different
attainable value under the cascade. The planned development-window survey
**never ran**: `survey.py` failed with `AttributeError`, rc 1, before producing a
~40-cell or 138-cell result. The measured hour below is therefore the entire
observed `V_oracle` population, not one completed cell of a broader survey.

---

### The ceiling, measured at last — and it says the lever is NOT exhausted

**`V_oracle` for the measured hour = 60,303.76c against a baseline net of
8,598.76c — 701% of the book — at `oracle_f` 0.4802 (2,072 of 4,315 fills are
losers).** An overlay can only decline, so the most it could ever add is
achieved by declining every losing fill and keeping every winning one.

| arm | fills-leg delta | **share of the ceiling captured** |
|---|---:|---:|
| CONDVALUE | −953.92c | **−1.58%** |
| HAZARD | −12.56c | **−0.02%** |

**Both arms captured a NEGATIVE fraction. They are not "far from the best
possible" — they are on the other side of zero from it.** 48% of fills lose
money and they total **seven times the book's whole net**: the information is
there in enormous quantity and neither ranker finds any of it. **The ceiling
costs declining nearly half the book, which is why `oracle_f` is quoted with it
and never without.** `ceiling_capture` returns a **signed** fraction for exactly
this reason — `|delta|/V_oracle` would have hidden the sign.

**The inventory leg's ceiling is the finding rather than the number.** Pooled
`V_oracle` 302,934c against a leg net of 8,587.54c — **a ratio of 35×**, so
that net is the RESIDUE of ~300,000c of offsetting per-fill contributions, and
CONDVALUE's +3,348c reversal is **1.1% of that gross**. A near-cancellation of
huge two-sided terms moves enormously on small changes in which fills are
received — **independent support for distrusting the P4 reversal, from the
opposite direction to DE's own pre-registered doubt.** Per window (the correct
unit) ceilings run 109%–11,883% of each window's net.

### The specification a modeller can aim at — the actionable form of the ceiling

*"−1.58% of the ceiling"* is true and hard to act on. Translated into what a
ranker must **do**: the book's P&L is **bimodal at roughly ±30c around a mean of
+2c** — losers average **−29.104c**, winners **+30.719c**, book **+1.993c**. A
declined set that is `q` losers has mean `q·(−29.104) + (1−q)·(+30.719)`.
Inverting, at CONDVALUE's own budget of 1,440 fills:

| | loser-rate `q` | **lift over the 48.02% base** | capture |
|---|---:|---:|---:|
| random | 48.02% | — | −4.76% |
| **CONDVALUE (observed)** | 50.24% | **+2.2 pp** | −1.58% |
| **HAZARD (observed)** | 51.15% | **+3.1 pp** | −0.02% |
| **break-even** | **51.35%** | **+3.3 pp** | **0%** |
| +10% of ceiling | 58.35% | **+10.3 pp** | +10% |
| +25% | 68.85% | +20.8 pp | +25% |
| +50% | 86.35% | +38.3 pp | +50% |
| perfect ranker at this budget | 100% | +52.0 pp | +69.50% |

**HAZARD is 0.2 points of loser-rate lift short of break-even; a 10% capture
needs +10.3 points, about three times its current discrimination.** This is the
first statement in the programme of what the model must **achieve** rather than
what it failed to achieve, and it is computed from the ceiling rather than from
a comparison to another model.

### The forward race is DIRECTIONAL, not significance-bearing — USER ruling, 2026-09-04

**The ceiling of a clustered permutation test is its floor: G clusters admit
2^G sign assignments, so the smallest achievable one-sided p is 1/2^G.** Read at
the artifacts: cluster unit **UTC day** (ruled), bar **G = 5**, multiplicity
**2** (executed: `race_multiplicity_at_freeze: 2`, members `PM_PLUS_FINE` and
`PM_FINE_EXTENDED`, `recorded_in_the_frozen_bytes: true`). **At G = 5 with
m = 2 the best possible adjusted p is 0.0625 > 0.05. The smallest G that clears
is 6.**

**THE USER RULES: the race stands at G = 5 and establishes DIRECTION AND
CONSISTENCY, never a Holm-clearing verdict.** The recorded multiplicity of 2
stands and is not made decorative. Every future statement of a race result must
say so up front.

**Why nobody computed it:** this programme applied permutation-floor reasoning
intensively **to the number of DRAWS** (1/501; n raised 500 → 2,000 because 500
gave only 1.04× headroom) and **never to the number of CLUSTERS**. *"Draws are
free and each cluster costs a calendar day. The cheap resource was tuned to
4.17× headroom; the expensive one was set to a round number and never checked."*

### On ceilings generally — a corrected claim, with its as-of

An earlier version of this section said **no value ceiling had ever been
computed in either programme. That is FALSE as stated and is withdrawn.** An
independent AST search for the *structural signature* of a ceiling — **as-of
2026-09-04T13:54:40Z over 189 files / 3,421 functions** — found **three, all in
P-2026-003's own code**: `state_gate_v1.bound_over_bins → bound_cents` (landed
2026-08-23), `adverse_move_fast.py:234 → oracle_upper_bound_cents_per_decision`
— whose name contains two of the original search words — and
`policy_bounds_v1.py::bound_table`, "the 16-bin all-gates bound", an upper bound
for the TIME-GATE lever, found by DE on its own surface (186 files, 3,378
functions, as-of 2026-09-04T14:03:33Z). The first search's
**method was sound and its SURFACE was the defect** — `live/pm_research/` was
never enumerated, and the conclusion was generalised to "either programme"
anyway.

**The claim also shipped inside a DOCSTRING and the arms emission before it was
checked** — *"where it becomes citable"* — which is worse than a wrong message
because it is harder to correct. **A negative existence claim carries a SURFACE
and an AS-OF or it is not a claim.**

**What survives and is citable:** no value ceiling in `live/mm_research/`
(11 files) or the registers, **and none anywhere for the CANCELLATION-OVERLAY
lever** — the claim that bears on the survey. `skew_bound.py` is **not** a value
ceiling either. And `markout_cents_per_share` (`harmful_exposure_rows.py:309-313`)
**is** the per-fill P&L and has existed since **2026-08-25 08:21**, so
`V_oracle` was computable on day one of this dataset.

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

**Current race state: G = 4 of 5** (09-04 accrued at the 00:06:01Z scheduled unit, `counts_toward_race: True`; the race is DIRECTIONAL, not significance-bearing — R-529(A)). R-503 re-verdicted 09-03 on its covered
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
**State: G = 4 of 5** — 09-04 accrued 2026-09-05T00:06:01Z (R-532(G)); 09-03 by the R-503 superseding verdict
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
4. `STATUS.yml` — current task/flag state; do not rely on historical
   counts quoted elsewhere.
5. `live/pm_research/plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN_V2.md` —
   the prospective governing gates. The v1 plan and stateful cancel×skew
   worksheet remain provenance; neither checkbox count is project completion.

**The reliability rules in `CLAUDE.md` are not style.** Each was bought with
dissolved work: rule 11 (choosing after seeing voids the test), rule 12 (a
freeze is a commit), rule 15 (every checker ships a falsifier), rule 16 (verify
at the artifact a claim names), rule 10 (compute predicates, never print
conclusions) — this consolidation applied rule 10 and rule 16 to the program's
own documents and found two errors in them (§2).
