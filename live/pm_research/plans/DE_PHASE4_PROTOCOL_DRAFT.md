# Phase-4 protocol — latency × queue-reset-cost × budget — DRAFT-FOR-USER-FREEZE

**STATUS: FROZEN — IN FORCE. USER ruling 2026-09-02 ("we can proceed the five decisions according to the recommendation", R-397). Declared BEFORE any cell is read (rule 11); no Phase-4 cell existed at freeze time (verified by de_phase4_protocol_check.py at the reviewed tip). Grids remain gated on a Phase-2 winner; the freeze arms the protocol, it does not start a run. Frozen by coordinator commit on the USER ruling.**
Nothing in this document has been computed on a Phase-4 cell; no Phase-4 cell
exists. The point of freezing it now is that every quantity below — the
primary cell, the null, the minimum sample, the multiplicity, the gate — is
chosen while the numbers are still invisible.

**Drafted by:** DE (R-379 TASK 3). **Frozen by:** the USER. **Amended by:** the
USER only (SEAT_PROTOCOL rule 4); a seat drafts, nobody amends a design after
seeing it.

**EXECUTION IS SEPARATELY GATED.** `HANDOFF.md` Immediate order item 4 holds
fair-price, skew and replay work to BUILD/FREEZE-ONLY: *no PnL, capacity,
promotion or forward verdict is claimable.* This protocol may be frozen under
that hold; **it may not be run until the hold is lifted by the USER**, and the
freeze is what makes running it later legitimate rather than retrospective.

**Scope:** `STATEFUL_HARMFUL_CANCEL_TODO.md` §7 (Phase 4), executed through
`harmful_stateful_policy.py` on generation-level tranche tables. **The score is
UNCHANGED**: no refit, no threshold search, no feature change. Phase 4 measures
a frozen score's economics across an operating grid; it does not select a
model.

---

## 1. The estimand, and the two caps that are part of it

**Estimand.** For one declared cell, the **cost-adjusted value of the stateful
cancel/hold/repost policy relative to the QR_SKEW_ONLY no-cancel shadow**, in
cents, on the decision unit: *avoided conditional harm − sacrificed favourable
fill value − lost spread capture − queue-reset/repost cost + marginal
inventory-risk benefit − action/traffic cost* (TODO §6's ledger, computed and
never printed).

**Cap 1 — latency is inside the estimand, not beside it.** A cell at latency
`L` values only tranches at or after `t + L`; tranches inside the window are
CHARGED as stale, never counted as prevented. Rule 7. `L` is a cell coordinate
and appears in every cell's own label.

**Cap 2 — the one-second horizon is part of what the cell means (R-165(2)
item 5).** The per-row latency labels are capped at `FILL_HORIZON_S = 1.0 s`,
so any cell built on them estimates *"value preventable WITHIN ONE SECOND of
the decision row"*, not *"value preventable"*. `phase4_generation_tables.
tranche_table` **refuses to emit without `declare_cap=True`**, and every
Phase-4 receipt must carry `estimand_horizon_s` beside the number. A cell that
loses the cap has changed its estimand without saying so.

**The unit is the GENERATION, not the row** (rule 2, R-165(2) item 5).
Generation-level tranche tables are the feed; a generation's value is the
value at its FIRST crossing, never the sum over its rows (measured 1.99
rows/fill, max 23).

---

## 2. The declared parameters, each with the ruling that fixed it

Every one of these is an INPUT with no default that encodes a policy choice;
`harmful_stateful_policy.validate_params` refuses an undeclared one and
refuses an inert declaration.

| # | parameter | value / treatment | authority |
|---|---|---|---|
| 1 | **queue-reset cost semantics** | **PRIMARY = policy-induced charging only** (`charge_reset_cost_at_generation_start = False`). The decision metric is DIFFERENTIAL against the reference trajectory, so reference-coincident generation starts are common-mode and charging them would tax the policy for events it did not cause. The charge-every-repost reading runs as a **declared ablation cell**, and **every receipt NAMES which cell it reports.** | R-165(2)(1) |
| 2 | **post-repost fill model** | **BOTH arms are a MANDATORY BRACKET**: `REFERENCE_FILLS` (queue-optimistic) and `NO_FILLS_UNTIL_NEXT_GENERATION` (pessimistic). A Phase-4 conclusion must hold under BOTH. **A sign flip across the bracket is a FAIL, never a choice.** | R-165(2)(2), P-2026-002 queue-bracket rule |
| 3 | **below-clock anchoring** | accepted AS BUILT: a pure function of the score stream, eligibility ≥ hold start. Not a Phase-4 degree of freedom. | R-165(2)(3) |
| 4 | **reduce-operations budget sharing** | `reduces_share_cancel_budget` is a **DECLARED VENUE-SEMANTICS PARAMETER of this protocol**, not a property of the machine: reduce operations keep SEPARATE counters, and whether they consume the cancel rate budget is priced here. **Both values are run**, and neither is a default. The reduce lane itself stays an explicit ablation (`enable_reduce=False` in the primary). | R-165(2)(4) |
| 5 | **feed** | **generation-level tranche tables** (`harmful_exposure_rows.generation_table` shape), never per-row latency labels. Any cell that consumes the per-row labels DECLARES the 1 s cap in its estimand. | R-165(2)(5) |
| 6 | **repost-landing charge ambiguity** | `charge_reset_cost_at_generation_start` is the parameter the machine refuses to guess. The skew freeze §7 Q3 records it as **a live obligation, cited not resolved**, and states that **it must be ruled before any lifecycle-economics number is claimed.** Parameter 1 above IS that ruling for Phase 4's primary cell; the ablation preserves the alternative. | skew freeze §7 Q3 (coordinator, correctable) |
| 7 | **protection mode** | both `REDUCING_SIDE_PROTECTION` and `ALL_ORDERS_OVERRIDE` are REQUIRED cells (TODO §6). They are a **conjunction, not a selection**: the verdict requires the same sign under both. | TODO §6 |
| 8 | **rate limit** | `max_cancels_per_minute` declared per cell; requested / effective(passed) / suppressed counted separately and reported as the identity `requested = passed + suppressed`. | TODO §6, LANE4 B1.4 |

---

## 3. The population, and what it can and cannot support

**Development population (available now):** the v3.4 consumed fragment —
**471 windows, btc 234 / eth 237, days 2026-08-24 and 2026-08-25, 3 windows
excluded for Binance discontinuity**, era `clob_v3_1`, selection reproduced by
execution on 2026-09-01 (`de_lane4_real_parity.run`).

**These days are CONSUMED (rule 11).** Every Phase-4 cell computed on them is
**DEVELOPMENT EVIDENCE**: `is_a_validation = false`, **G = 0 complete UTC
days**, no interval claimable, and no forward verdict. This is the iteration-011
lesson applied in advance rather than discovered in the artifact: the receipt
must **compute** `is_a_validation`, not assert it.

**Validation** is a separate act on **≥5 complete, later, untouched UTC days**
after a freeze, scored without refitting. Nothing in this protocol licenses it.

**Cluster unit is the UTC day** (rule 8). Below G=5 complete days: point
estimate, no interval, said out loud. Window-clustered intervals may be
reported as diagnostics and may never carry the verdict.

---

## 4. The grid, and which axes are SELECTION axes

| axis | rungs | selection axis? |
|---|---|---|
| latency `L` | 5, 10, 20, 30, 50, 75, 100, 150, 250 ms | **NO** — the operative rung is fixed by a deployment ack-latency measurement, *"the smallest ladder rung ≥ the measured upper bound"*, and where reconciliation cannot resolve adjacent rungs **the coarser rung applies** (OP_PLANE_PLAN §5.1). Replay results are computed per rung precisely so that substitution is legal. **Nobody may pick a rung by looking at the results.** |
| queue-reset cost `c` | reported as a **DERIVED BREAK-EVEN**, plus a declared sensitivity curve over {0.00, 0.01, 0.05, 0.10, 0.25, 0.40, 0.75, 1.50} cents/cancellation | **NO, by construction** — reporting the cost at which cost-adjusted value crosses zero answers the actual question ("how cheap must the queue reset be?") with ONE number per cell instead of eight tests. The bracket spans the reduced-fine point estimates in TODO §7 (BTC ≈0.29/0.38/0.32c, ETH ≈0.011/0.068/0.072c per cancellation at 5/10/15% budgets) with rungs on both sides |
| budget `b` | 5%, 10%, 15% | **YES** — someone chooses a budget |
| repost fill model | 2 | NO — mandatory bracket (conjunction) |
| protection mode | 2 | NO — both required (conjunction) |
| reset-cost semantics | 2 | NO — one PRIMARY, one named ablation |
| reduce config | off / on+shared budget / on+separate budget | NO — off is PRIMARY, the two on-cells are named ablations |
| coin | btc, eth | adjudicated separately |

**Size of the space: 9 × 8 × 3 × 2 × 2 × 2 × 3 × 2 = 10,368 cells.** That is the
size of the SPACE and it is **not** the candidate count — the distinction
LANE4 B4 pays for three times. §7 computes the count.

---

## 5. The PRIMARY cell, declared now, before any cell exists

> **PRIMARY = coin `btc`; latency rung `250 ms`; budget `10%`;
> `charge_reset_cost_at_generation_start = False`; `enable_reduce = False`;
> conjunction over BOTH repost-fill arms and BOTH protection modes.**

Each coordinate is fixed by a rule stated before the data, not by taste:

- **250 ms — the slowest declared rung**, by the conservative-degradation rule
  (§4): the venue ack bound is **not observable at this venue** (`OP-Latency
  Budget` leg 4), `tau_operative` is UNMEASURED until an Actuator exists
  (R-55), so the honest substitute for an unmeasured bound is the coarsest
  rung on the ladder. **NOTE the tension, disclosed rather than resolved:**
  the DE ladder in TODO §7 tops at 250 ms while OP's τ ladder tops at 1000 ms
  and R-8 kills the lever above 1000 ms. Whether the Phase-4 ladder should be
  extended to meet OP's is **an open question for the USER (§12 Q2)**.
- **10% — the middle rung of a three-rung grid**, chosen a priori as neither
  extreme.
- **btc primary, eth reported** — the R-306 precedent (btc-only adjudication,
  eth reported) rather than a new convention.
- **Conjunctions, not selections** — a bracket and a required-cell pair make
  the test HARDER and therefore do not multiply (§7).

Every other cell in §4's space is a **declared sensitivity surface, reported
without an inferential claim**. Reading a verdict off a sensitivity cell is
choosing after seeing and voids the test.

---

## 6. The null, and the minimum sample — computed, not asserted

**Null design (rule 6, declared before the result).** Matched random
cancellation on identical opportunities: draws are matched on **action count,
side and hour**, and the comparison is on the **DECISION metric** —
cost-adjusted value in cents, and `rho = adverse / spread` — **never on a
proxy such as harm share** (rule 7). The control's action count is
*determined* by the treated arm and is never a caller-chosen number (LANE4
B1.1); the draw orders its pool into a total order **before** the RNG touches
it, and a budget that exceeds the eligible pool **REFUSES** rather than
clamping.

**Minimum sample, derived from the family size rather than habit.** With a
permutation/draw null, the smallest attainable p is `1/(N+1)`. For a
Holm-corrected family of `m` cells to be *able* to clear α = 0.05 at all:

`m / (N + 1) < 0.05`  ⟹  `N ≥ 20m`

| adjudicated family | m | minimum draws | at N = 200 |
|---|---|---|---|
| **PRIMARY as declared (btc only)** | **1** | **20** → the rule-6 floor of **200 binds** | Holm-adjusted floor **0.00498**, clears |
| btc + eth both adjudicated | 2 | 40 → 200 binds | 0.00995, clears |
| *if the USER rules the latency ladder a selection axis* | 9 × 3 × 2 = **54** | **1,080** | **0.269 — NO CELL COULD SURVIVE, whatever the effect** |

**The third row is the point of computing this in advance.** Iteration 011
discovered exactly this shape inside its own artifact — every surviving p at
the `1/501` floor and Holm at `24 × 0.001996 = 0.0479`, a family with no
headroom. **DECLARED HERE: N = 200 draws minimum; and if §12 Q1 is ruled such
that the ladder or the budget grid becomes a selection axis, N rises to
1,080 in the same ruling or the Phase-4 economic claim is not made.**

---

## 7. Multiplicity accounting

**Two different multiplicities, and conflating them is a third way to be
wrong.**

1. **Phase-4's own cell multiplicity** = the number of cells at which a
   POSITIVE VERDICT may be claimed. Under §5 that is **1** (btc primary), or
   **2** if the USER adjudicates eth. Brackets and required-cell pairs are
   conjunctions and do not multiply; ablations and the sensitivity surface
   carry no claim and do not multiply. **Recorded in the receipt as the
   derivation, not as an integer.**
2. **The forward race's candidate multiplicity** (rule 12, recorded at freeze
   time) is a DIFFERENT number and is **not restated here**. It is owned by
   `da_replay_parity_battery.candidate_multiplicity`, which **computes** it
   from `consumes_predictor` and `roles` — both of which are **still owed** by
   the seat that owns the arm implementations (LANE4 B4.2/B5.1). Until those
   declarations exist, **no candidate count is the record**, and a Phase-4
   receipt may not print one.

**Anti-transcription obligation.** The receipt carries the INPUTS and the
DERIVATION, and a check asserts that the printed steps evaluate to the printed
answer (LANE4 B4.3's own defect: a "full space" of 21 that no reader could
reproduce).

---

## 8. Reporting — every row of TODO §7, plus the disclosures

For **every** cell (primary, ablation and sensitivity alike):

- gross avoided harm, favourable-fill sacrifice, cost-adjusted value;
- cancellations per minute and per generation;
- effective / stale / unresolved / zero-value cancellations;
- hold duration, repost count, queue resets;
- fill and share retention, spread capture;
- retained-book adverse-cost / spread-capture ratio (`rho`);
- terminal inventory, peak inventory, reducing/increasing split;
- **complete maker P&L, post-fill markout and inventory loss** — `net_cancel_cents`
  is NOT strategy P&L and may not be labelled as such;
- comparison against `QR_SKEW_ONLY`, `QR_CANCEL_HOLD_X_SKEW` and matched
  random on identical opportunities;
- marginal module deltas: hazard → conditional value; cancel → cancel × skew;
  `Identity` → fair-price challenger.

And with every table, without exception:

- **`n` AND `as-of`** for every quoted population (rule 8) — the tape grows
  during measurement;
- **exclusions as COUNTED STATUSES** (rule 4), never silent drops — the
  window- and generation-level status counts from the reference build;
- **windows-affected beside P1/P2/P3** where a forward receipt is involved
  (HANDOFF item 0h);
- **the lattice drift declared** (item 0f) and the collector era NAMED
  (item 0g);
- **which reset-cost cell is being reported** (R-165(2)(1));
- `estimand_horizon_s` and the latency rung, in the cell's own label;
- **`is_a_validation` COMPUTED** from the population, never asserted.

**No verdict strings.** Every claimed property is a computed predicate beside
its number (rule 10). A hardcoded verdict has contradicted its own table three
times in this programme.

---

## 9. The gate, and the parts of it that are not yet declared

**TODO §7's Phase-4 gate:** *positive cost-adjusted value at a material
retention level, with inventory and traffic no worse than their declared
limits, on both the point estimate and the matched-random comparison. ETH may
be rejected separately if its very small per-cancel margin cannot survive the
cost grid.*

**One term of the surrounding gate IS declared and is carried unchanged:**
`HARMFUL_FILL_HAZARD_TOXICITY_PLAN` states that *"strategy viability requires
reaching `rho < 1` at material retention after declared costs"*, so **`rho < 1`
is the declared bar** and this protocol does not restate or move it.

**Three of the sentence's other terms have no declared value anywhere in the
programme** — the same phrase, "material retention" and "the declared inventory
and traffic limits", appears in two plans and is defined in neither. They are
proposed here so they are frozen before the numbers exist, and each is flagged
as needing the USER's word:

| term | PROPOSED | why this and not something else |
|---|---|---|
| "material retention level" | `retention_share_fraction ≥ 0.50` in the primary cell | the zero-repost degeneracy (LANE4 B1.3): an arm that cancels once and never reposts wins on every activity-normalised metric while trading nothing. A retention floor is what makes the win non-degenerate |
| "traffic no worse than its declared limit" | cancels per generation ≤ 1 (structural) **and** cancels/minute ≤ the declared `max_cancels_per_minute` of the cell, with `requested = passed + suppressed` reported | the structural half is already guaranteed by the machine; the rate half is the venue-facing one |
| "inventory no worse than its declared limit" | per-slug `peak_abs_net` no greater than QR_SKEW_ONLY's own per-slug peak, and terminal net reported per slug | inventory is PER-SLUG (R-184 (vii)); a cross-slug aggregate is reporting-only and no decision may read it |

**Q3 in §12 asks the USER to rule these.** Until then the gate has a hole, and
a Phase-4 run would be scored against a bar nobody set.

---

## 10. What voids this protocol

1. **Choosing after seeing** any coordinate of §5, any threshold in §9, or the
   null design in §6 (rule 11).
2. **Reading a verdict from a sensitivity or ablation cell.**
3. **A sign flip across the mandatory bracket** reported as a choice rather
   than a FAIL (R-165(2)(2)).
4. **Any economic claim on the consumed fragment** presented as validation
   (§3).
5. **Running before the item-4 hold is lifted.**
6. **A receipt that prints a candidate multiplicity** while `consumes_predictor`
   and `roles` are still owed (§7).
7. **Executing on a red suite.** Numbers may not come from an instrument that
   has not shown it can fire (rule 15).

---

## 11. Red-first obligations on the Phase-4 harness itself

Before any cell is read, the harness ships — each with a known-bad that FAILS
pre-fix and a positive control that ADMITS the good case (rules 15/16):

1. **The parity battery green on the REAL reference** — the disabled-predictor
   and infinite-threshold arms bit-identical to QR_SKEW_ONLY, cancel-and-hold
   equivalence against an independently-built trajectory, one cancel per
   generation, latency-window fills charged stale (`de_lane4_real_parity`).
   A Phase-4 economics run on a harness whose parity was never checked at
   scale would inherit any coupling invisibly.
2. **A vacuum refusal**: zero admitted windows, or zero cancels issued, must
   REFUSE rather than report passing gates.
3. **A perturbation control**: one extra event must break parity, or the
   anchor is decorative.
4. **A cap-declaration refusal**: `tranche_table` without `declare_cap=True`
   must refuse, naming what the cell would actually be estimating.
5. **A determinism control across `PYTHONHASHSEED`**: two runs byte-identical.
   A fixed RNG seed over a process-dependent iteration order is an independent
   draw, not a reproduction (LANE4 falsifier 2).
6. **A matched-control assertion**: the control's action count, side and hour
   profile equal the treated arm's, asserted rather than assumed, and the
   control must select DIFFERENT generations than the treatment.

---

## 12. Open questions — for the USER, not resolved here

**Q1 — is the latency ladder a SELECTION axis?** §4 treats it as not one,
because OP's design fixes the operative rung by an external measurement. If
the USER rules otherwise, §6's minimum sample rises from 200 to **1,080** in
the same ruling, or the economic claim is not made. *Recommendation: not a
selection axis, with the rung frozen at 250 ms per §5.*

**Q2 — should the Phase-4 latency ladder be extended to meet OP's τ ladder?**
DE's tops at 250 ms; OP's tops at 1000 ms and R-8 kills the lever above it.
A ladder that stops at 250 ms cannot express the rung a deployment might
actually land on. *No recommendation: extending it adds rungs to a grid that
has not been run, and the coordinator owns the ladder.*

**Q3 — the three undeclared gate terms in §9** (material retention, traffic
limit, inventory limit). *Recommendation: freeze the proposed values, which
are conservative and each motivated by a named failure mode; any of them is
one line to override.*

**Q4 — does eth get adjudicated, or only reported?** §5 follows the R-306
btc-only precedent. If eth is adjudicated the primary family is 2, which §6
already prices. *Recommendation: btc adjudicated, eth reported, consistent
with 011.*

**Q5 — Phase 4 runs inside a replay seam with no registry owner.** See
`DE_REGISTRY_AMENDMENT_PROPOSAL.md`: `EV-Replay` has no module record and the
`harmful_stateful_policy` dialect is not in the EV_REPLAY_PLAN census. This
does not block the protocol, but a Phase-4 receipt would be the first
result-bearing artifact produced by an unowned seam. *Recommendation: rule the
registry amendment before Phase 4 is executed, not before it is frozen.*
