# PM_STRUCTURE_REVIEW_LOOP — extensibility review of PM_ARCHITECTURE.md

Charter (user, 2026-08-20): **review the STRUCTURE — not the model — until it
is genuinely easy to plug in new SOTA theories, new mechanisms, new venues.**
Convergence: two consecutive iterations with zero MUST-FIX.

This loop asks a different question from its predecessors. Not "is the model
right" (sketch loop) or "is the data complete" (data loop), but:
**when the next theory or mechanism arrives, how much has to be rewritten?**

## The empirical test this loop must apply

This program has a REAL change log from 2026-08-19/20. Any proposed structure
must be replayed against it and scored: for each change, how many modules
would have been touched, and would the change have been LOCAL?

| # | Actual change that happened | Ideal blast radius |
|---|---|---|
| 1 | binary-GLFT improvisation → published prediction-market HJB (arXiv 2607.17991) | 1 module |
| 2 | fair value source: Binance-synthetic → stream-anchored | 1 module |
| 3 | σ: static → rolling HAR → variogram (target itself was wrong) | 1 module |
| 4 | quoting: continuous-δ closed form → discrete per-level EV | 1 module |
| 5 | v(t): sum → min-structure | 1 module |
| 6 | pull: τ_min line → (\|d\|,r) surface → participation region + size | 1–2 |
| 7 | Q_max: variance cap → loss-given-adverse cap + portfolio aggregate | 1–2 |
| 8 | pair: net q → (q_up,q_down) joint conditions | 2 (inventory, pairs) |
| 9 | rewards: PnL line → constraint w/ shadow price → principal–agent + contest | 1–2 |
| 10 | siting US → EU → suspended | 0 model modules (config only) |
| 11 | latency 120 → 471 → 1700 ms | 0 (one param, one owner) |
| 12 | scope: PnL-first → mechanism-first | 0 (experiment registry only) |
| 13 | whole components ADDED late (flow, propagator, book microprice, short-horizon alpha, cross-coin, portfolio risk) | new module, no edits to existing |

A structure that would have required broad edits for these is not adaptable,
whatever its diagram looks like.

## Iteration protocol
Three lenses, then triage → amend `PM_ARCHITECTURE.md` → re-review.

| Lens | Question it owns |
|---|---|
| **A Extensibility** | plug-in points; the change-log replay above; what a NEW theory costs |
| **B Contracts** | are boundaries in the right places; are interfaces complete, minimal, non-leaky; does the dependency rule actually hold |
| **C Portability** | does it survive a new venue / new horizon / new mechanism; is it over-fitted to Polymarket-5min |

| iter | date | MUST-FIX | status |
|---|---|---|---|
| 1 | 2026-08-20 | A:3 B:7 C:3 | **v2 rewritten** (not amended) |

## Iteration 1 outcome
v1 scored **LOCAL 5 / SPREADING 3 / STRUCTURAL 5** on its own change log and
failed **all six** forward plug-in tests. Convergent findings across independent
lenses (strong signal): SP-Venue/Instrument specs (A+C), objective ownership /
de-economised constraints (A+B), merge fill+adverse-selection (B, implied by A),
specs as versioned data (C, supported by A's REQUIRES).

Root causes fixed in v2:
1. **Decision plane was the closed-form SOLUTION's shape, not the PROBLEM's** —
   forbade 4 adopted theories; no module owned the objective (so rewards had
   nowhere to land and gate-before-economics was structurally available).
   → Objective / Constraints / Solver / Actuator; `feasible()` returns a size
   and a **shadow price**, never a bool.
2. **`as_of(t)` was a 1.7 s look-ahead sold as a guarantee** — at r=2 s the peek
   (≈0.08 bps) exceeds the modelled risk (0.007 bps). → `Known[V]` +
   `view(now)`; no event-time filter API exists.
3. **Venue facts entered 5×** (R-ONCE violated in the exact form it forbids)
   → SP-Venue/SP-Instrument as versioned data; `w_declared` ≠ `w_hat`.
4. **BeliefProcess** replaces `(p̂, confidence)`: without `constituents`, the
   pivotal experiment (plan-X2) is not expressible against the production module.
5. R-ONCE/R-KNOW/R-NULL/R-VERSION/R-REQUIRES each given an ENFORCEMENT POINT.

Iteration 2 re-runs the same three lenses against v2 — the change-log replay is
the acceptance test (target: no STRUCTURAL, ≤1 SPREADING).

| 2 | 2026-08-20 | A:5 B:6 C:5 | **v3 rewritten (smaller than v2)** |

## Iteration 2 outcome
Score **LOCAL 7 / SPREADING 4 / STRUCTURAL 2** (v1: 5/3/5) — improved, but
**FAILED** the acceptance target. Portability interface-changes 4 → 1 hard.
v2 also introduced two bugs of its own.

Root causes fixed in v3:
1. **`t_known` had no source** — `recv_ns` exists on no history we own, so the
   natural impl `t_known := t_event` passes every v2 check and returns the
   1.7 s peek TYPE-LAUNDERED (worse than v1: v1's bug was legible).
   → `t_known_prov{OBSERVED|IMPUTED|ASSUMED}` + `t_known_err`; replay refuses
   inside the error. Composition is **MAX**, not min.
2. **`Σ terms` objective was a theory commitment** excluding 3 adopted theories
   (multiplicative M-1, CARA M-8, occupancy-integral M-5); and `Constraints`
   was asked for a shadow price `∂V/∂x` it cannot compute.
   → `DE-ValuationScheme` (value form + constraint handling + solve as ONE
   choice); constraints declare form, the scheme prices duals.
3. **v2's `L_adv` forbade pair-harvest** — charged a riskless paired position
   at full notional. → inventory as `(m, u)`; only `|u|` bears risk.
4. **R-ONCE had no PnL enforcement and v2's own term list double-counted twice**
   (rewards as term AND constraint; adverse_selection beside `E[markout|fill]`).
   → explicit term ledger, one owner per effect.
5. **Specs versioned by event time** (`valid_from/to`) — the same bug R-KNOW
   exists to prevent. → content hash + `observed_at`.
6. Regressions/omissions restored: EV-Settlement (1,292 resolutions, no
   consumer), iteration-1 MF-6 (scope + keyed identity), field-level nullity
   with bias direction, pair-aware `rest()` (the PM book is UNIFIED across the
   token pair), `fee_schedule` size argument, three-way `X2` naming fault.
7. CUT per lens C: specs demand-driven not prerequisite; `VarianceBudget`
   machinery; "layer" → YAML + loader; unreachable enum branches;
   solver impls marked BUILT vs NAMED-SEAM. **v3 is 188 lines vs v2's 221.**

Iteration 3 acceptance target unchanged: 0 STRUCTURAL, ≤1 SPREADING.


| 3 | 2026-08-20 | user review: 9 | **v4 rewritten** |

## Iteration 3 (user review of v3) — 9 MUST-FIX, all accepted
v3 fixed the NAMING of several contracts without making them USABLE. v4 gives
them signatures. Fixes, in the user's numbering:

1. **Spec versioning needs BOTH time axes** — hash+observed_at (knowledge) AND
   valid_from/to (validity), with `FieldState = Resolved|Disputed|Unknown`.
   `Disputed` is live: Gamma vs CLOB registry disagree on the rewards band now.
2. **Identity/cardinality did not exist** — added `VenueId·InstrumentId·
   TokenId·RiskFactorId·PortfolioId`; `SP-Params` keyed by **(name, scope)**,
   scope in the KEY; `DE-Allocator` owns portfolio budget. Cross-coin,
   multi-horizon and second-venue become CONFIGURATION, not structural change.
3. **Knowledge-time bypasses closed** — `coverage` bound to the view's `now`;
   `evaluate()` takes a `StateView`; `SelfState.envelope()` returns the
   submitted/acked/filled exposure bracket; composition defined for error,
   provenance, staleness (not just MAX for t_known); typed `Unavailable`.
4. **DecisionScheme is now a contract** — typed `DecisionProblem`; and the two
   conflated axes split: valuation/control form (GLFT|PerLevelEV|HJBQVI|CARA-CE)
   × scope coupling (PerToken|JointPair|PortfolioJoint), with declared
   compatibility. A new control theory no longer reimplements pair coupling.
5. **One action vocabulary** — `evaluate(action_set, …) -> Uncertain[ActionOutcome]
   | Unavailable` replaces `rest()`, so cancellation queue loss, taker slippage,
   conversion cost, completion risk and batch/AMM actions are priceable;
   `queue_bracket`'s hard-coded two-scenario form is replaced by `Uncertain[T]`.
6. **PnL partition declared** — `spread + transient_AS + permanent_AS + snipe +
   own_impact = markout(τ)`, each with unit/conditioning/sign/coverage. v3
   assigned owners without defining the split — the double-count it claimed to fix.
7. **L_adv was sign-reversed** — `|u|(1−p̂)` charges 0.02/share where ~0.98/share
   is lost. Now `unpaired_cost_basis(q_up,q_down,m)`. **Third recurrence of the
   picking-up-pennies error class, this time inside its own fix.**
8. **Environment seam** — `Environment{clock, feeds, venue, rng, artifacts}`
   with Live/Replay/Sim impls; deterministic tie-breaking, warm-state snapshot,
   restart parity, artifact resolution by `artifact_id`+`fit_data_through`.
   Resolves the "EV drives the stack" dependency inversion.
9. **Program control owned** — `Gate{...}` schema + precondition DAG; STOP as a
   first-class gate with `on_fail=HALT_PROGRAM` and a named owner;
   `EV-Settlement` now read by `DE-Allocator` so redemption delay and capital
   lockup enter decisions.

Next: re-run the 13-change replay + plug-in tests against v4 (target unchanged:
0 STRUCTURAL, ≤1 SPREADING).

| 4 | 2026-08-20 | review: 8 | **v4 NOT CONVERGED; issues recorded** |

## Iteration 4 review of v4 — 8 MUST-FIX

Full review: `PM_STRUCT_ITER4_REVIEW.md`.

Provisional score: **LOCAL 9 / SPREADING 2 / STRUCTURAL 2**. The remaining
structural causes are the conflation of HJBQVI control with CARA utility, and
the absence of an owner and payout functional for rewards competition.

The other MUST-FIXes close the full-Environment escape hatch, make parameter
scope composite, add the missing variance partition, separate markout from the
full objective ledger, define shared algebraic types/module manifests, and put
provenance on individual spec fields.

| 5 | 2026-08-20 | review: 9 | **v5 NOT CONVERGED; issues recorded** |

## Iteration 5 review of v5 — 9 MUST-FIX

Full review: `PM_STRUCT_ITER5_REVIEW.md`.

Provisional score as written: **LOCAL 9 / SPREADING 3 / STRUCTURAL 1**. After
restoring `ParamId` to the parameter key: **LOCAL 10 / SPREADING 3 /
STRUCTURAL 0**.

v5 closes the v4 utility/control and incentive-ownership gaps. Remaining work
restores parameter identity, makes coupling hierarchical, keeps utility outside
the additive wealth ledger, supplies covariance/falsification for variance,
adds dependence to `Uncertain`, makes competition state knowledge-time safe,
packages incentive theory as one extension, replaces factor-weighted loss with
scenario risk, and defines the OP health/halt path.

| 6 | 2026-08-20 | review: 4 | **v6 NOT CONVERGED; issues recorded** |

## Iteration 6 review of v6 — 4 MUST-FIX

Full review: `PM_STRUCT_ITER6_REVIEW.md`.

Strict score as written: **LOCAL 11 / SPREADING 0 / STRUCTURAL 2**. The two
structural rows are portfolio scenario risk, whose scenario model is unnamed
and ownerless, and incentive theory, whose knowledge/uncertainty wrapper and
instrument scope are discarded at `DecisionProblem` while its cash-flow seam
incorrectly targets utility.

The OP improvement is not yet executable: the port manifest carries no
`HaltState`, and the claimed cancel command has no edge to `DE-Actuator`.
Remaining work preserves the competition wrapper, routes incentive cash into
wealth, types and owns adverse scenarios, and completes the halt/cancel path.

| 7 | 2026-08-20 | review: 5 | **v7 NOT CONVERGED; issues recorded** |

## Iteration 7 review of v7 — 5 MUST-FIX

Full review: `PM_STRUCT_ITER7_REVIEW.md`.

Strict replay score remains **LOCAL 11 / SPREADING 0 / STRUCTURAL 2**.
V7 correctly preserves the competition wrapper, routes incentive cash into
wealth and gives the kill path both constraint and actuator edges. Scenario
risk and portfolio competition nevertheless remain structural because their
claimed fields and joint types are absent from the canonical
`DecisionProblem`.

Additional boundary regressions must be removed before convergence: dynamic
coupling is described but not carried by the problem; OP reads EV despite the
EV-is-read-by-none invariant; and restoring the old spec block duplicates
instrument-scoped rewards under `SP-Venue`.

| 8 | 2026-08-20 | review: 6 | **v8 NOT CONVERGED; issues recorded** |

## Iteration 8 review of v8 — 6 MUST-FIX

Full review: `PM_STRUCT_ITER8_REVIEW.md`.

V8 moves the v7 fixes into canonical blocks and correctly separates scenario
names, removes duplicate venue rewards, excludes EV from runtime health, and
adds risk/coupling fields to `DecisionProblem`. The strict replay is still at
best **LOCAL 11 / SPREADING 0 / STRUCTURAL 2**: risk scenarios remain neither
PIT-safe nor position-independent, and the scalar competition producer cannot
supply the new joint consumer.

The deeper audit also found that `contract_inventory.py` passes an invalid
baseline and misses the exact bare-to-wrapped type narrowing it is intended to
prevent. The halt path lacks canonical telemetry/ack semantics, the SOTA axes
remain closed unions, and belief/contracts/static coupling do not yet form one
immutable point-in-time decision snapshot.

| 9 | 2026-08-20 | review: 5 | **v9 NOT CONVERGED; issues recorded** |

## Iteration 9 review of v9 — 5 MUST-FIX

Full review: `PM_STRUCT_ITER9_REVIEW.md`.

V9 correctly names the joint competition aggregator, improves the
knowledge-time decision snapshot, opens the utility/solver protocols, fixes
identity/parameter namespace issues and replaces the v8 markdown scanner with
an owner-qualified type diff whose core regression tests pass.

The strict replay nevertheless remains at best **LOCAL 11 / SPREADING 0 /
STRUCTURAL 2**. The estimated risk distribution is still typed over scenario
sets and its claimed sole limit owner is absent from canonical YAML. The
competition mismatch is fixed, but the one-registration incentive extension is
also absent from the purported source of truth.

The deeper audit found that `contracts.yaml` is neither complete nor
reference-closed despite its canonical claim, while `contract_check.py`
silently ignores risk-owner notes, compatibility validation, registries and
contract version. The halt graph still has no cancel-command producer or
observable heartbeat path, and the explanatory document has drifted from the
new YAML in several behavior-bearing places.

| 10 | 2026-08-20 | review: 6 | **v10 NOT CONVERGED; issues recorded** |

## Iteration 10 review of v10 — 6 MUST-FIX

Full review: `PM_STRUCT_ITER10_REVIEW.md`.

V10 corrects the estimated distribution support type, names risk and incentive
contracts, introduces static/dynamic coupling and pinned outcomes, completes
the intended heartbeat/cancel producer graph, and broadens the structural
checker to notes, registries, validation, rule bodies and version monotonicity.

The strict replay nevertheless remains at best **LOCAL 11 / SPREADING 0 /
STRUCTURAL 2**. Scenario limits are not resolved or wired into a PIT-safe
constraint/allocator path, while the new `IncentiveModel` is not selected or
connected to module and cash-flow consumers.

The adversarial audit also proves that the target invariant accepts undefined
protocol/variant/module types, missing registries, duplicate ownership and
conflicting port manifests. The canonical source remains incomplete, coupling
config duplicates per-decision state, pinned outcomes lose their evaluator,
and cancellation variants lose typed payload and reconciliation semantics.

| 11 | 2026-08-20 | review: 5 | **v11 NOT CONVERGED; issues recorded** |

## Iteration 11 review of v11 — 5 MUST-FIX

Full review: `PM_STRUCT_ITER11_REVIEW.md`.

V11 fixes the concrete v10 type gaps: coupling config/state are separated,
pinned outcomes are callable, unavailable/unwrap policies are parameterised,
scenario limits and allocator consumers are named, incentive selection is
pinned, cancellation payloads are typed, and the checker covers more reference
locations with new self-tests.

The replay improves provisionally to **LOCAL 11 / SPREADING 1 / STRUCTURAL 1**.
Scenario risk now has owners and consumers but requires coordinated type and
single-copy corrections. Incentive theory remains structural because the
selected extension's resolved contracts and joint competition never enter the
outcome/payout evaluation path.

The executable audit also shows that a path-only migration entry still masks a
different coupling narrowing, rules/ports are not enforced, runtime producers
can disappear through `config_supplied`, and an unchanged HEAD-to-worktree
comparison fails on all historical allowlist entries. Decision snapshot
equalities and real ModuleManifest wiring remain absent.

| 12 | 2026-08-20 | review: 5 | **requested v12 is v11.1; NOT CONVERGED** |

## Iteration 12 review — repository labels the target v11.1

Full review: `PM_STRUCT_ITER12_REVIEW.md`.

Commit `ad5cbc0` closes M11-1 by replacing the path-only allowlist with
target-ref, exact old/new, version-bound migration records. The previous
coupling A→C exploit is rejected and unchanged HEAD-to-worktree comparison
passes.

This is a checker patch, not an architecture iteration:
`PM_ARCHITECTURE.md` and `contracts.yaml` have zero changed lines, the
contract remains version 11, and the human document remains titled v10.
Documentation improvement is therefore zero, not merely marginal.

The replay stays **LOCAL 11 / SPREADING 1 / STRUCTURAL 1**. Semantic
rules/producers/ports, scenario type consistency, incentive-to-outcome wiring,
decision snapshot equality and ModuleManifest coverage remain unresolved.
The gate also still accepts same-version additions and duplicate keys in
`contracts.yaml`.
