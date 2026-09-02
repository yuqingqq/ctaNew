# Phase-4 diagnostic addendum — the scheduled execution, declared before any cell is read

**This is an ADDENDUM. `DE_PHASE4_PROTOCOL_DRAFT.md` is FROZEN and is not
edited by it (rule 13).** The frozen document is bound here by its bytes:

| bound artefact | sha256 |
|---|---|
| `live/pm_research/plans/DE_PHASE4_PROTOCOL_DRAFT.md` | `ab07fd71c9fc2bffcd2fc3736e65a9ce43a8bd7ce0d94f361d2acea2f12c3c75` |

**Authority.** The protocol was frozen under R-397 with execution separately
gated by HANDOFF Immediate-order item 4 (*"may not be run until the hold is
lifted by the USER"*). **R-459 records the USER's seventh ruling — "Yes
schedule this test" — which lifts that hold for THIS execution only**: one
diagnostic execution of the frozen protocol on its own §3 population, with
`Q1_arrival` of `composed_lgbm` as the head under test against the incumbent
head, across the full latency axis. Nothing here widens the grid, names a
Phase-2 winner, or admits anything to the forward race.

**Written before any cell is read (rule 6).** No Phase-4 cell exists at the
time of writing. Everything below — the population, the cells, the arms, the
null, the predicates and the reading — is fixed while the numbers are still
invisible. The run is a later round; this round builds and declares, and the
reviewer reads the declaration first.

---

## a. Population, and what every output says about itself

The frozen §3 population **exactly**: the v3.4 consumed fragment — **471
windows, btc 234 / eth 237**, days **2026-08-24 and 2026-08-25**, 3 windows
excluded for Binance discontinuity, era `clob_v3_1`.

These days are **CONSUMED** (rule 11), and they are also iteration 011's own
development window (`phase2_fits/fit_slugs.json`). Therefore every output of
this execution carries, computed and not asserted:

- `is_a_validation = false`
- `G = 0` complete UTC days
- `DIAGNOSTIC_NEVER_EVIDENCE` — the shape `be_fragment_diagnostic_v1.json`
  already uses
- **no interval claimable** beyond the null's own draws at the cells where the
  null actually ran; every other cell is a **point estimate, labelled**

It is **not** a Phase-2 winner, **not** a race admission, and the
multiplicity of the forward race is **unchanged**. The forward race's frozen
arm-A file (`harmful_reduced_fine_candidate_v1.json`, `1b53929`) is not the
object of this execution and is not touched.

## b. Cells — the frozen grid, nothing outside it

**PRIMARY cell settings, as frozen (§5):**
`charge_reset_cost_at_generation_start = False`, `enable_reduce = False`,
**conjunction over BOTH repost-fill arms and BOTH protection modes**.

| axis | what this execution does | selection axis? |
|---|---|---|
| latency `L` | **swept, all nine rungs 5, 10, 20, 30, 50, 75, 100, 150, 250 ms** | **NO** (§4) — the sweep is a REPORT; nobody picks a rung by looking at the results |
| budget `b` | **5% / 10% / 15%, all three reported** | **YES** — and **none is selected here**; a budget is someone's choice, made elsewhere |
| coin | **btc and eth**, adjudicated separately | — |
| repost fill model | both (conjunction) | NO |
| protection mode | both (conjunction) | NO |
| queue-reset cost `c` | the **derived break-even** plus the frozen sensitivity curve {0.00, 0.01, 0.05, 0.10, 0.25, 0.40, 0.75, 1.50} c/cancellation | NO, by construction |
| reduce config | **off** (PRIMARY) | NO |
| reset-cost semantics | **PRIMARY only** (`charge_..._at_generation_start = False`) | NO |

**No new rung, no new number, no cell outside the frozen grid.** The
sensitivity curve is reported as frozen and carries no inferential claim
(§5's own words).

## c. Arms, and the arm-name resolution

| arm | role | source of record |
|---|---|---|
| `QR_SKEW_ONLY` | reference | frozen lane, RUNNABLE |
| `QR_CANCEL_HOLD_X_SKEW` | incumbent policy | frozen lane, RUNNABLE |
| `CONDVALUE_OVER_SKEWED_REF` **(new name — see below)** with the **incumbent head** | comparison arm | `phase2_fits/linear_d_{coin}.json`, btc sha `18701008c2bd18c6` (R-398), verified before load |
| `CONDVALUE_OVER_SKEWED_REF` with the **head under test** | the object of the ruling | `Q1_arrival` of `composed_lgbm`: `lgbm_haz_{coin}.txt` + `lgbm_thresholds_{coin}.json`, bound by `fit_manifest.json` |
| `RANDOM_MATCHED` | the null | `de_matched_random_control.py`, contract identity declared there and in §d |

**Unrunnable, and they stay unrunnable:** `HAZARD_ONLY_NEUTRAL` and
`CONDVALUE_NEUTRAL` — **NO_NEUTRAL_REFERENCE**: the frozen reference is
skew-ON, so a neutral-placement arm has no reference to be run against, and
naming one would be inventing a population. `CONDVALUE_X_SKEW_X_FAIRPRICE`
stays out: **no challenger was ever scored**.

**THE ARM-NAME COLLISION, RESOLVED BY NAMING (DE's ASK A1/A2;
`de_lane4_real_parity.py:118-141`).** The composition that exists on the
frozen lane is *"a conditional-value predictor over the frozen SKEWED
reference, with NO interaction"*. `CONDVALUE_X_SKEW` asserts an interaction
that does not exist — DA's loader refuses `interaction=False` under it —
and `CONDVALUE_NEUTRAL` decomposes to `("condvalue",)` and would omit the
skew that was in force in the placement it inherited. **A frozen name is not
reused for a composition it does not describe.** This execution therefore
names a NEW arm:

> **`CONDVALUE_OVER_SKEWED_REF`** — a conditional-value head applied over
> the frozen skew-ON reference, with no interaction term and no fair-price
> component. It is a name for what runs, not a claim about what wins.

The seven-arm vocabulary is not amended by this addendum; the new name
belongs to this diagnostic until a coordinator ruling says otherwise.

## d. The null — design, sample, and where it runs

**Design, as frozen (§6):** matched random cancellation on identical
opportunities, **matched on action count, side and hour**, compared on the
**DECISION metric** — cost-adjusted value in cents and `rho = adverse /
spread` — never on a proxy such as harm share.

**Contract identity of the acting control** (`de_matched_random_control.py`):
an ACTING arm that cancels generations chosen uniformly at random within
(side, hour) strata, where the per-stratum count is **determined by the
treated arm** and never caller-chosen; the pool is put in a total order
**before** the RNG touches it; a stratum whose eligible pool is short of the
demand **REFUSES rather than clamping**; and a draw identical to the treated
arm's own actions is **refused by identity**.

**N = 200 draws minimum**, as the frozen §6 declares.

**Where the null runs, and where it does not — declared now, with the
arithmetic.** One 471-window single-arm replay measured **1,339.6 s** in
LANE4. On `research.slice` (`MemoryMax` 18.4 G, `CPUQuota=1200%` — twelve
cores) 200 draws at one cell is of order **6 hours** wall-clock, so:

| cell | null | reason |
|---|---|---|
| **PRIMARY (btc, 250 ms, 10%)** | **200 draws** | the protocol's minimum, and the only cell the family is adjudicated at |
| eth at 250 ms, 10% | **200 draws** if the PRIMARY completes inside its window; otherwise **point estimate, labelled** | both coins are adjudicated separately (§4); the second family member costs another ~6 h |
| every other latency rung × budget × coin | **POINT ESTIMATE, no interval, labelled** | 9 × 3 × 2 = 54 cells at ~6 h each is ~13 days of slice time; the ladder is not a selection axis, so a point estimate is what it is for |

**Explicitly NOT run:** the null at the 54-cell sweep; the reduce-on
ablations; the `charge_reset_cost_at_generation_start = True` ablation; any
cell for `CONDVALUE_X_SKEW_X_FAIRPRICE`, `HAZARD_ONLY_NEUTRAL` or
`CONDVALUE_NEUTRAL`. Any of these appearing in a later receipt would be a
cell nobody declared.

## e. The predicates the run computes, and the reading fixed in advance

**Computed in code, never printed as a conclusion (rule 10):**

1. per coin and budget: `rho(L)` at every rung, `rho_min = min over L of
   rho(L)`, and the predicate **`rho_min < 1`** evaluated in code;
2. `net(Q1 head) − net(incumbent head)` per cell, compared against the
   null's quantiles **at the cells where the draws actually ran**; elsewhere
   the difference is reported as a point estimate and labelled as one;
3. **retention share beside every rho** — a rho computed on a population the
   policy has emptied is not the same number as a rho at full retention;
4. **every exclusion a status with a count** (rule 4), including the rho
   estimator's own `IN_LATENCY_WINDOW` / `NO_MID_AT_FILL` /
   `NO_MID_AT_MARKOUT` / `NON_FINITE` / `ZERO_SIZE`;
5. `rho = adverse / spread` (`de_rho_estimator.py`) **and** the existing
   `rho_captured_over_sacrificed` proxy (`harmful_action_eval.py:192`),
   reported side by side under their own names and never conflated.

**THE READING, FIXED BEFORE THE RESULT:**

- **`rho ≥ 1` at every rung including 5 ms, with the full composition ⇒ the
  route CLOSES.** This is taken in-sample, which is the flattering
  direction, so a failure here is conclusive: if the composition cannot pay
  for itself on the days it was fitted on, at a latency nobody can achieve,
  it will not pay for itself later.
- **`rho < 1` somewhere with material retention ⇒ NOT validation.** It is a
  reason to finish the integration and let untouched days decide. No cell of
  this execution may be quoted as evidence for a forward claim.

## f. Instruments, and one declared limit

| instrument | file | selftest |
|---|---|---|
| `rho = adverse / spread` over received fills | `de_rho_estimator.py` | 21 checks |
| score-stream adapter, manifest-bound | `de_score_stream.py` | 24 checks |
| acting matched-random control | `de_matched_random_control.py` | 20 checks |

**DECLARED LIMIT — IR-R4 (stated, not worked around).** There is no
generation-tranche artifact with a production consumer (`tranche_table` has
none). The score-stream adapter therefore does **not** read features from a
file: it takes the feature table the runner builds and turns head + table
into score events. Until the runner supplies that table this diagnostic
cannot run, and no fixture stands in for it. This is a limit of the
diagnostic, recorded here rather than closed by a substitute population.

**The runner emits economics by design.** LANE4's `_receipt_cell` refuses
every economics key (`de_lane4_real_parity.py:8-14`) because LANE4 is a
verification harness; the diagnostic runner is a **separate entry point**,
never a switch on the parity harness, and it writes to a NEW directory
`data/pm_5min/derived/phase4_diag_r459/`. It touches no frozen anchor and no
`fwd*` directory.

---

**Nothing in this addendum has been run.** It is the declaration the run is
measured against; if the run's receipt and this document disagree, the
document was written first and the disagreement is the finding.
