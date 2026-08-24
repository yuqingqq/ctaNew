# EV_GATES_REVIEW_LOOP — charter

Self-paced adversarial review loop over `plans/EV_GATES_PLAN.md`. Started
2026-08-23 by **BE**, on transfer of ownership under ruling **R-18**.

## Why this loop exists

`EV_GATES_PLAN.md` was written by the **coordinator** as build item B4, because
the EV plane had no session. R-18 records that as the wrong fix — the right one
was to get an owner assigned — and rules that **the coordinator does not author
module plans**: workers design, build and measure; the coordinator writes
decision rules and ratifies. A module plan is a DESIGN artifact and therefore
worker output.

The harm is demonstrated rather than hypothetical: OPS examined a single row of
the coordinator's other plan and found a MUST-FIX-class defect that made a
refutation convertible into a pass by fiat — inside the rule written to prevent
exactly that. Both coordinator-authored documents are **single-pass and
unreviewed**, while the DE plans went through **ten** adversarial iterations.

**BE now owns `EV_GATES_PLAN.md` fully and may revise, restructure or REFUTE
it.** BE is the natural owner because BE lives inside gates more than any plane:
84 `route_a_v1` gates, 126 `route_a_v2`, `G-FF1`…`G-FF4`, and the
`PASS` / `INSUFFICIENT_EVIDENCE` / `MODEL_REFUTED` vocabulary came out of the
sigma review rounds BE ran.

**Treat the document as a draft by a non-specialist.** Deference to it is the
failure mode this loop exists to correct.

## What is NOT in scope

`STOP-MM-VIABLE` stays defined as written with **THE USER** as its owner. BE
owns the gate machinery around it. **BE does not own the decision to fire it,
and neither does the coordinator.** A finding may attack whether the gate is
well-formed, evaluable, or correctly wired; it may not propose changing who
fires it.

## Objects under review

- `plans/EV_GATES_PLAN.md` — the gate registry, the precondition DAG, the
  verdict vocabulary, and `STOP`.

## Goal

The plan is (1) **complete** — covers every content `EV-Gates` needs, with an
owner for each; (2) **typed** — every construct it describes is expressible in
`contracts/contracts.yaml` v22, and it obeys the architecture's own rules
(plane ordering, R-SSOT, R-HALT, R-VERSION, R-6 parameter classes); (3) **sound**
— survives adversarial concrete cases drawn from the programme's real gates,
real verdict words and real receipts.

## Method

Three independent review agents probe with **distinct lenses** and return
findings. **BE verifies each finding against the files before acting**, applies
confirmed fixes, and records the iteration here. **Reviewers never edit.** BE
never grades its own homework as a reviewer.

Lenses used: **completeness** · **type integration / architecture conformance** ·
**adversarial concrete cases**.

## Finding classes

`MUST-FIX` — a defect that would propagate into code, or contradicts a measured
fact or an architecture rule · `SHOULD-FIX` — a gap or ambiguity that invites a
future defect · `NOTE` — recorded, no change.

**Every finding needs a concrete failure case, not a preference.**

## Binding constraints on fixes

- The plan stays **DESIGN**. No fix may record as settled what a measurement
  could settle, invent a measured number, or add a PnL/capacity claim.
- `FLOW_MODEL_STATE.md` wins on facts; `contracts/contracts.yaml` v22 wins on
  types; `PM_ARCHITECTURE.md` explains and does not define.
- No fix may create an `EV → OP → DE` edge, or an ownerless / two-owner
  quantity.
- **A threshold is never re-cut here.** Under R-6 a frozen verdict bar is
  Class D and moves only before its measurement runs. This loop may find that a
  bar is *wrong*; the remedy is to say so, not to move it.
- BE may **refute** the plan. If the design is unsalvageable the honest output
  is a recorded refutation and a replacement, not a patched draft.

## Stop rule

The loop ends when an iteration produces **zero confirmed MUST-FIX findings
twice in a row** (loop-until-dry, K=2), or the user stops it. Verdict per
iteration: `DEFECTS_FOUND_AND_APPLIED` | `CLEAN` | `BLOCKED(reason)`.

---

## Iteration log

> **LOOP STATUS: OPEN — stop counter 0 of 2.** Computed, not asserted:
> `python3 live/pm_research/check_loop_log.py live/pm_research/EV_GATES_REVIEW_LOOP.md`
>
> **`CLOSED` on an iteration heading means THAT ITERATION is closed. It does NOT
> mean the loop is closed.** The loop closes only on **two consecutive iterations
> with zero confirmed MUST-FIX**. Iterations 1-4 returned **20, 62, 53, 40**.
> BE let one word do two jobs and a reader took "iteration 4 CLOSED" for "the
> loop has closed" — the name is not the definition, in BE's own log.


### Iteration 1 — 2026-08-23 — **CLOSED** — verdict: `REFUTED_IN_SUBSTANCE`

3 of 3 lenses reported · 20 confirmed MUST-FIX · applied as
`EV_GATES_PLAN.md` **Revision 1** (rewrite, not patch).
Working notes from during the iteration are kept below the summary.

**Outcome: the plan is SALVAGEABLE IN STRUCTURE, REFUTED IN SUBSTANCE.** Roughly
forty lines survive — §6's ownership split, §3's non-runtime halt path, §5.4's
change-detector-vs-conformance point. Everything decision-bearing fails. The
charter permits refutation and this is the honest use of it: the remedy is a
bottom-up rewrite seeded from the artifacts that already work, not a patch.

**Convergence across independent lenses is the strongest signal here.** Two
lenses independently found the false premise; two independently found `STOP`
unevaluable; two independently found `Gate.threshold` unable to hold the bars.
Each is counted once.

#### THE DECISIVE FINDING — routed to the coordinator as BE-4, NOT fixed by BE

**`STOP-MM-VIABLE`'s threshold is SIGN-BLIND, and the evidence that kills the
programme already SATISFIES it.** Metric is verbatim the Layer-1 estimand;
threshold is *"the interval must exclude zero on at least one verdict coin"*;
measured btc `h=5` is **−0.532 ¢ [−0.797, −0.287]** — excludes zero, on a verdict
coin. **Threshold met → `PASS` → `on_pass` = "proceed to the DE build."** The
plan cites this number eleven lines later as proof STOP is "closer to firing than
any other gate". `+0.532 [+0.287, +0.797]` and `−0.532 [−0.797, −0.287]` are the
same verdict under it. It is also the construction §5.2 forbids everywhere else.

BE does **not** fix this: R-18 gives `STOP` to the USER. Filed as BE-4 with the
R-6 argument that the amendment window is open **only until STOP is evaluated
once**, because (c) "invalidate every verdict under the old bar" is free only
while no verdict exists.

#### Also confirmed, verified against files (iteration-1 MUST-FIX)

Beyond M1–M12 recorded below, the adversarial lens added:

| # | defect | verification |
|---|---|---|
| M13 | §5.2's check **rejects the programme's only clean `PASS`** | `gff1_side_v3.json` carries `verdict: PASS`, `threshold: 0.99`, `wilson95: [0.9936, 1.0]` and **no `tolerance`, no `ci_hi_abs`** — it is a one-sided *superiority* test, not an equivalence test |
| M14 | §5.3 **rejects the 210 sigma gates §8 promises to admit** | of 31 receipts only **6** carry `days_sampled`; **23 carry neither** it nor a provenance block — including `sigma_route_a_v1.json`, the sole receipt for all 84 `route_a_v1` gates (verified: no `provenance` key) |
| M15 | the three-verdict vocabulary is an **equivalence grammar**; half the registry is directional | `ci_hi_abs` is an ABSOLUTE bound and destroys the sign. `route_a_v2` gate 3 is explicitly one-sided (`upper bound < 0`), `LAYER2` §3 is `POSITIVE`/`NEGATIVE` by direction of exclusion, `G-FF3` is a sign test at `threshold: 0.0` |
| M16 | §3 **suppresses a refutation that is robust across the whole upstream interval** | `ww_v1` returned `DEAD` 8/8 coin-days, and the flip point is btc markout −0.180 ¢ against a measured CI of [−0.797, −0.287] — the *most favourable* endpoint is 1.6× beyond it. "Not evaluated" and "insufficient evidence" are different states |
| M17 | §3 breaks `on_fail`-as-**reroute** on a family the coordinator never authored | `G-FF2.on_fail` is *"drop queue modelling; regress fill on level and displayed depth"* — a route, not a stop. §3 would freeze `G-FF3`/`G-FF4` instead of following it, making `G-FF4`'s conclusion *"MM is not deployable"* unreachable exactly when true |
| M18 | §5.1's witness check is **degenerate on all three real gates it can be tested against** | both instances the plan itself cites (the algebraic identity, the 60–1000× denominator) had witnesses that were nameable and unconstructible; R-14 already ruled §5.1 insufficient and required an MDE |
| M19 | `SKEW_SUFFICIENT` → `PASS` **launders an upper bound into a validation** | `PLACEMENT_SKEW_RESULTS.md` states it *"is an upper bound, not an achievable result"* — the exact harm the plan correctly refuses to do to `RETAIN` three sentences later |
| M20 | `UNIDENTIFIABLE` → `INSUFFICIENT_EVIDENCE` **inverts the operational instruction** | `UNIDENTIFIABLE` means more data cannot help (`FLOW_MODEL_STATE` §3: *"do not schedule work against these"*); `INSUFFICIENT_EVIDENCE` means collect more. `VOID` and `DAY_BLOCK_UNAVAILABLE` are admissibility refusals, a third state again |

**§7's own falsifiers fire on three of four bullets**, on evidence that predates
this review: the dialect map loses information in four places where one was said
to be decisive; `STOP` is already unevaluable; and the triviality edge
(`DA coverage ──▶ every gate above`) is drawn in the plan's own spine.

#### Direction for Revision 1 — rewrite, keeping three sections

Seed the registry bottom-up from what already works rather than re-deriving it:
`BE_FLOWANDFILLS_PLAN.md`'s populated `gates:` block as the first real instances ·
**per-protocol declared verdict mappings** instead of a central map that goes
stale whenever a worker freezes a bar · **executable vacuity controls** instead of
declared witnesses (four protocols already ship them) · a **four-state**
vocabulary with a signed `bound_kind` · and a `STOP` whose preconditions are
**disjunctive**, so a kill-gate can still fire when the evidence is worst. That
last property is not a refinement — it is the entire purpose of the gate, and
this draft removes it.

---

#### Iteration 1, working notes — findings as verified at the 2-of-3 mark

**Verification status: every claim below was checked by BE against the named
file before being recorded. Unverified reviewer claims are not listed.**

#### CONFIRMED MUST-FIX — the two that change the plan's shape

**M1. The plan's central premise is FALSE.** §1: *"`Gate.preconditions` exists
in v22 and nothing populates it, so the DAG is a type with no instances."*
It is populated — in **BE's own plan**. `plans/BE_FLOWANDFILLS_PLAN.md`
declares `preconditions: [G-FF1]` (line 953), `[G-FF1, G-FF2]` (961),
`[G-FF2]` (972), with `inference_method`, `review_date` and per-gate `on_fail`.
A three-edge sub-DAG already exists on disk, and §3's "illustrative spine" omits
all three — so the plan's graph is **less complete than the graph it says does
not exist**. R-5 repeats the false claim, so it will propagate.

**M2. The flagship gate can never fire, and the plan claims the opposite.**
`CANCEL_POLICY_PROTOCOL.md:23` (frozen): *"`cancel_v1` will therefore never
report"*; R-11 confirms *"its only outstanding precondition — `cancel_v1` — will
never report, because the family it would test is dead."* `cancel_v1` is a
declared precondition of `STOP-MM-VIABLE` (§4). Under the plan's OWN §3 rule —
*a gate whose preconditions have not passed is not evaluated, verdict
`INSUFFICIENT_EVIDENCE` by construction* — **`STOP` is pinned at
`INSUFFICIENT_EVIDENCE` permanently and its `review_date` ("first review when
`cancel_v1` reports") never arrives.** §4 asserts `STOP` "is closer to firing
than any other gate in the registry"; it is the one gate the DAG can never let
fire. This is §5.1's own anti-pattern — *a gate that cannot fire is not a gate* —
committed by the gate written to showcase it, and §7's fourth falsifier is
already satisfied.

*Scope note:* fixing this changes the **machinery** (the DAG needs a
precondition state for `DISCHARGED_BY_REFUTATION` — question settled negatively,
will not be measured — distinct from `NOT_YET_EVALUATED`). It does **not** touch
who fires `STOP`. The USER owns that and this loop does not.

#### CONFIRMED MUST-FIX — types (every one grep-verified in v22)

| # | defect | evidence |
|---|---|---|
| M3 | `Gate.threshold: float` cannot hold the plan's own bars | `contracts.yaml:393`. `STOP`'s bar is a predicate; `ww_v1` is 4 numbers + a quantifier; `cancel_v1` is a 4-clause conjunction |
| M4 | `GateEvidence` has **no gate identifier**, no scope, no weighting arm | `contracts.yaml:847-856`. 210 sigma gates emit evidence nothing can attribute. Also blocks R-DUAL (per-fill AND share-weighted) and `R-FLOW.per_coin` |
| M5 | `Gate.id: str` but `preconditions: list[GateId]` | `contracts.yaml:390` vs `:397`; `GateId` is only an opaque external at `:1893`. The DAG's edges point at a type the nodes do not have. Sibling `AdmissibilityRule.id: AdmissibilityRuleId` gets this right (`:704`) |
| M6 | `Gate` has no `version`, `status`, or `provenance` | `contracts.yaml:388-405`. R-VERSION (bitemporal), R-10 (address must change with content) and R-3-amended R-PROV are all unevaluable on the one object whose purpose is to gate. `AdmissibilityRule` has `version: int` + `status: DRAFT\|FROZEN\|RETIRED` |
| M7 | `on_fail: str`, and `HALT_PROGRAM` appears **nowhere** in v22 | `contracts.yaml:399`; 0 grep hits. Sibling `AdmissibilityRule.on_fail` IS an enum (`:717`). §3's architectural claim is right in prose with no type-level barrier |
| M8 | Three of §5's four checks name fields that exist in no type | `days_sampled`: **0 hits** in contracts.yaml. No failing-witness field, no `code_hash`/conformance ref on `Gate` |
| M9 | `EV-Gates` has no module record and the registry no container type | **0 hits** for `EV-Gates` in contracts.yaml. No `GateRegistry`. Meanwhile sigma evidence already lives inline on `ReducedFormFit.mean_gate`/`.var_gate` (`:838-839`) — the registry it claims to own has no address |
| M10 | Census misses whole frozen-bar families | `edge_l1_v1`, `policy_v1`, `layer2_v1` (frozen by R-14), `queue_type_v1`, both skew bars, BE retention gates, U1-U11, DA admissibility. §1 files `HORIZON_DEPENDENT`/`SKEW_ROBUST` as "ad-hoc" when they are verdicts of **frozen** protocols |
| M11 | `RETAIN` excluded but its replacement type never provided | `RETAIN`: **0 hits** in contracts.yaml; `FlowModelStatus` (`:1307`) does not contain it. The Hawkes retention decision ends up owned by no type and no module — R-SSOT gives it **zero** owners, not one |
| M12 | §5.1 is already extended by a ruling the plan does not carry | R-14 Amendment 2: §5.1 *"must also declare what it takes to FIRE"*; `layer2_v1` shipped an a-priori MDE under it. The plan still says only "FAILS" |

**A contract-vs-architecture conflict found in passing, not the plan's fault but
now BE's to route:** `PM_ARCHITECTURE.md` §1 says health sources are the
telemetry ports of DA/BE/DE and OP-LatencyBudget — **"NEVER EV"** — but v22's
module records give `EV-Markout`, `EV-Calibration` and `EV-Orchestrate` all
`produces: [... HealthEvent]`. The types already contradict the rule for the EV
plane. Recorded for the coordinator; not fixable inside this plan.

#### Where the plan is CORRECT (verified, recorded so the loop is not one-sided)

- §3's refusal to reintroduce `EV → OP → DE` matches `PM_ARCHITECTURE` §1 and
  the port map `EV-*: [read_all]` (`contracts.yaml:1856`). Reasoning sound; only
  the type-level backing is missing.
- The sigma counts are right: 7 symbols × 6 horizons × 2 gates = **84**;
  `SIGMA_ROUTE_A_V2_PROTOCOL` states **126** verbatim. "210" is consistent.
- §5.2 is the one anti-pattern check that maps onto real fields
  (`tolerance`, `ci_hi_abs`, `p_value: float | NullPin` all exist).
- Choosing `GateEvidence.verdict` as the canonical vocabulary is the right pick.
- §4's Layer-1 input numbers are accurate against the source.

#### BE's own two, from before the reviewers reported

Both independently re-confirmed above as M3 and M4.

---

#### Iteration 1, working notes — BE's own pre-reviewer checks

Three independent reviewers dispatched (completeness / type-integration /
adversarial-cases). Findings pending; nothing applied yet.

**Two defects BE found while verifying the plan's own load-bearing claims,
before the reviewers reported** — recorded here so they are not later credited
to a lens that did not find them:

1. `Gate.threshold` is typed `float` in v22 (`contracts.yaml:393`), but the
   plan's flagship gate `STOP-MM-VIABLE` (§4) declares
   `threshold = "the interval must exclude zero on at least one verdict coin"`.
   That is a predicate, not a float. The plan's own headline gate is not
   representable in the type it claims not to be inventing.
2. `GateEvidence` (`contracts.yaml:847-856`) carries `test`, `conditioning`,
   `multiplicity`, `verdict`, `effect_size`, `ci_hi_abs`, `tolerance`,
   `p_value` — and **no gate identifier**. With 210 sigma gates each emitting
   evidence, nothing attributes a verdict to the gate that produced it.

Both to be re-checked against whatever the reviewers return, and neither is
applied yet.
### Iteration 2 — 2026-08-23 — **CLOSED** — verdict: `REFUTED_IN_MACHINERY`

3 of 3 lenses · **62 MUST-FIX raised across the three** (with overlap) on the
revision BE wrote. Stop counter **0 of 2**.

**The verdict changed class, and that is the progress.** Iteration 1 was
`REFUTED_IN_SUBSTANCE` — false premises, checks that rejected the corpus's real
work. Iteration 2 is `REFUTED_IN_MACHINERY`: *"the structure is now right and
the mechanisms are not; the fixes are local rather than another rewrite."* The
§0.1 refutation table verified line-for-line, the census arithmetic that can be
checked checks, and three ideas — invariance-over-interval, executable controls,
per-protocol mappings — are the right shape.

**The single cause of nearly every new defect:** BE wrote the machinery without
walking it against the receipts first. Every new construct fails on the first
real artifact it meets.

#### TWO FIXES APPLIED IMMEDIATELY — one is a safety defect BE created

**1. The disjunction and the sign-blind bar MULTIPLY. Now gated on BE-4.**
Revision 1 said disjunctive `STOP` preconditions were *"applied here"* (line 247)
while §4.1 routed the sign-blind bar as unfixed (line 225). Compounded, a leg
becomes any (protocol, coin, day) cell whose interval excludes zero on a verdict
coin — and under a sign-blind bar a **POSITIVE** cell satisfies the threshold
identically to a negative one. That raises the independent chances of `STOP`
reading `PASS` → *"proceed to the DE build"* from about **2 to about 24**. BE
stated both defects in adjacent sections and did not notice they multiply.
**Disjunction is now explicitly NOT APPLIED and conditional on BE-4 landing**,
and a leg is defined as the protocol's own **frozen verdict unit** — otherwise a
single `layer2_v1` cell (`NEGATIVE −2.37 [−3.81, −1.08]`) would fire what the
frozen R-14 coin bar declined at `UNDETERMINED`, which R-17 applied unsoftened.
Bar-softening by another route. Machinery only; who fires `STOP` is unchanged.

**2. BE's renumbering broke a citation inside a FROZEN document.** Verified:
`LAYER2_PROTOCOL.md:105`, frozen under R-14, cites *"EV_GATES_PLAN §5.1 extended
from failing witness to firing witness"* — and in Revision 1 §5.1 is "Not a
runtime edge" while that content is at §6.2. **This is R-10's own defect —
address must change with content — committed by the plan that imports R-10.** BE
cannot edit a frozen protocol to repoint it, so a stable-anchor map is now at the
top of the plan and future revisions cite `EVG-VACUITY` etc. rather than numbers.

#### Confirmed and NOT yet applied — for Revision 2

- **`bound_arm` refuses `PASS` on every family that has ever earned one** —
  `G-FF1` (Wilson **lower** bound), all 84 `route_a_v1` (upper bound within
  tolerance), all 126 `route_a_v2` (one-sided upper `< 0`) — and contradicts a
  live v22 rule at `contracts.yaml:616`. It also gets the polarity backwards:
  `SKEW_ROBUST` is the **conservative** lower-bound arm and would be refused,
  while `ww_v1`'s upper arm *strengthens* `DEAD`. The distinction wanted is
  *an estimand that is only a bound* vs *a bound used as a test statistic*.
- **The four states cannot hold U6**, which is simultaneously a decisive
  rejection of one null, a miss on its declared bar, and a structural
  non-identification of the part that matters — three facts, one slot. Indicated:
  a **tuple** `(null_disposition, bar_disposition, identification_disposition)`.
- **`resolvable_by` is answered BOTH ways by `LAYER2_PROTOCOL` §3's own two
  paragraphs** (*"resolution is calendar"* vs *"≈140 era days **or** ≈6 days at
  full coverage"*), and R-19's day-stratified sampler makes a `SAMPLE` stamp
  decay into `CALENDAR` with no re-measurement — the drift-under-a-frozen-label
  defect R-20 exists to stop.
- **`NOT_IDENTIFIABLE` has a measured false-positive record**: *"terminal
  mechanism is unidentifiable"* sits in `FLOW_MODEL_STATE` §4 **Withdrawn**,
  because the two hypotheses *"were never observationally equivalent — nobody had
  tested amplitude"*. A non-identification claim must name the contrast that
  would overturn it.
- **The reroute example dies on its own artifact**: `G-FF2.on_fail` drops queue
  modelling, and `G-FF4.metric` is `QueueBracket.sign_agreement` — following the
  reroute makes G-FF4's metric *undefined*, not reachable.
- **§7's R-20 bullet creates a FIFTH copy of the anchor** and the receipt copy
  holds the **derived** `f*_low`, not the inputs R-20 actually requires.
- **The meta-finding stands**: no contract delta ships from this plane again
  without a `contract_check.py` run pasted into this log.

Revision 2 is local repair, not a third rewrite. Iteration 3 follows it.

#### working notes, superseded by the CLOSED iteration-2 entry above

**Second lens (contract delta): 20 MUST-FIX + 10 SHOULD-FIX.** Running total for
iteration 2 is **40 MUST-FIX across two lenses**, on the revision BE wrote.

#### THE META-FINDING, and it is BE's own defect

> *"Re-derive the non-additive list mechanically by running
> `contract_check.py <git-ref> WORKTREE` against a draft v23 rather than by
> inspection."*

**The repo contains a contract checker and BE did not run it.**
`live/pm_research/contracts/contract_check.py` exists, takes
`<base-ref> [<ref>|WORKTREE]`, and its `--selftest` prints **14** PASS results
here (BE first wrote "8/8" from memory — corrected by counting). It detects
exactly the class of error §9 shipped: `unresolved reference`, `consumes X with
no declared producer`, `duplicate declaration (local type AND prelude.external)`,
and unused/over-recorded migrations.

BE hand-authored a contract delta **by inspection** while the tool that validates
deltas sat in the same directory. That is this corpus's own lesson — *a SHA-256
is a change-detector, not a conformance checker* — turned on BE: §9 asserted a
conformance nobody checked. **Binding for Revision 2: no contract delta ships
from this plane again without a checker run pasted into the loop log.**

#### Three HARD STOPS no migration record can waive — all verified by BE

| defect | verification |
|---|---|
| **`EraDayCount` does not exist** | `grep -c EraDayCount contracts.yaml` → **0**. BE invented a type in §9 (`review_date: Date \| EraDayCount`). Trips `unresolved reference` |
| **`GateEvidence` has no declared producer** | it exists only inline on `ReducedFormFit.mean_gate`/`.var_gate` (`:838-839`); **no module** lists it under `produces`. `EV-Gates.consumes: [GateEvidence]` trips `consumes … with no declared producer` |
| **`GateId` moving out of `prelude.external`** | it sits at `:1893` under `prelude.external`. Structuring it is an unrecorded **removal**; leaving it in both places trips the checker's only R-SSOT-named error |

#### Two defects where BE made the plan WORSE than v22

- **`on_fail` is a NARROWING, not a widening — and it deletes the route text
  §3.3 depends on.** v22 has `on_fail: str`. §9's enum keeps only `HALT_PROGRAM`
  intact; the three real routes — `G-FF2` *"drop queue modelling; regress fill on
  level and displayed depth"*, `G-FF3` *"exclude strata where |zeta| exceeds
  gross capture"*, `G-FF4` *"the module is Unavailable … MM is not deployable"* —
  all collapse to `REROUTE`, which records **that** an edge exists and not
  **where it goes**. §3.3 quotes G-FF2's route verbatim as its worked case, so
  the rule becomes unexecutable on its own example, on 3 of the 4 registry-seed
  gates.
- **`bound_arm` refuses the programme's only clean `PASS` — R0-7 returning under
  a new field name.** `G-FF1` passes **by a Wilson lower bound**, so
  `bound_arm = LOWER`, and §2.4 says *"`PASS` is refused on a bound arm."* The
  distinction §2.4 wanted is *an estimand that is only a bound* vs *a bound used
  as a test statistic*; one field cannot carry both.

#### Two additions reintroduce defect classes v16 already removed

`bound_kind` beside the `GateBar` variant tag is the `estimand_route` defect
(v15's enum-on-a-record, replaced by a real union in v16). `ci_hi_abs` retained
beside signed `ci_lo`/`ci_hi` is the `density`/`g_prime` defect — and it has a
live wrong answer: for btc `−0.532 [−0.797, −0.287]`, `ci_hi_abs` is 0.797 by
definition and 0.287 to a producer who reads the name, which against a 0.5 ¢
tolerance gives `MODEL_REFUTED` and `PASS` from the same row.

#### And one that inverts the architecture's first rule

Making registry-facing fields mandatory on `GateEvidence` silently obligates
**BE-Uncertainty** — the only existing producer of `GateEvidence` values, via
`ReducedFormFit` — to emit a `gate: GateId` that must already exist in EV's
registry. `PM_ARCHITECTURE` §1: *"EV reads all planes and is read by none."*
That makes BE read EV. Fix direction: split the in-band record from a
registry-facing wrapper produced by EV-Gates only.

Third lens pending. Revision 2 waits for it.

#### working notes, superseded by the CLOSED iteration-2 entry above

**The completeness lens returned 20 MUST-FIX + 8 SHOULD-FIX on Revision 1.** The
rewrite did not converge. Recorded before the other two lenses report, because
the pattern it names is structural and BE should not defend the revision it just
wrote.

**The pattern, in the reviewer's words and confirmed by BE:** *"§§2–7 grew a
great deal of new machinery and §9 did not grow with it."* §3 is the largest
section in the plan and §9 declares **not one field** for it — no `preconditions`,
no `NOT_EVALUATED`, no `blocked_by`, no carrier for the four precondition states
§3.4 introduces. BE rewrote the prose faster than the type delta, which is the
same class of defect as Revision 0 asserting a construct that no type supports.

**Two new rules misfire on their own motivating cases. Both verified by BE:**

- **§2.4's ingest refusal rejects the receipt §2.2 uses to justify the fourth
  state.** `QUEUE_AND_TYPE_PROTOCOL.md` declares C1's outcome space as
  `MATERIAL | IMMATERIAL | PARTIAL | VOID` (lines 43–52). The receipt returned
  **`UNIDENTIFIABLE`** — `QUEUE_AND_TYPE_RESULTS.md:22` says verbatim *"a branch
  the protocol does not contain."* So the verdict string has no mapping in its
  pinned `spec_hash`, and Revision 1's own rule refuses it first.
- **§2.3 is a central map of two verdict words, one section after §2.4 forbids
  central maps** — and applied globally it would delete `layer2_v1`'s eight
  measured within-day cells, because `DAY_BLOCK_UNAVAILABLE` means *"report the
  within-day interval"* in V5 and *"refuse"* in `BE_BELIEF_RESULTS.md`.

**And the per-protocol `verdict_map` has no legal path.** BE verified: **zero of
seven** frozen protocols declare one (`SIGMA_ROUTE_A`, `SIGMA_ROUTE_A_V2`,
`EDGE_LAYER1`, `CANCEL_POLICY`, `LAYER2`, `POLICY_COMPARISON`, `QUEUE_AND_TYPE`
— 0 hits each). Adding one edits the text the `spec_hash` pins, which §6.4
forbids. So on day one every receipt in the corpus is refused and the only
sanctioned remedy is closed. §2.4 also never states the map's format and §9
declares no field to hold it.

**Census still incomplete**, and the reviewer's diagnosis is precise: §1 claims
to have been *"walked from the protocol files"* but was walked from the
iteration-1 finding list, so families nobody complained about are still absent —
BE's **own** BE-Belief go/no-go gate, the EV-Replay harness gates, inventory-walk,
the terminal-mechanism bar, the one-book identity, and **`STOP` itself**, which
§4 calls the flagship. `EV-Markout` is listed as a `STOP` precondition and is a
**module**, not a gate, so it can never be registered.

Stop counter: **0 of 2**, and iteration 2 cannot be clean. Two lenses pending;
Revision 2 waits for all three rather than being written twice.

#### working notes, superseded by the CLOSED iteration-2 entry above

Three fresh reviewers dispatched against **Revision 1** (completeness /
contract-delta / adversarial). Each was briefed that Revision 1 is a REWRITE and
that **new machinery is where new defects live** — the four-state vocabulary,
disjunctive kill-gate preconditions, per-protocol verdict maps, executable
vacuity controls, and the §9 contract delta are the specific attack surface.
Each was told the programme's base rate (a sibling plan returned 60 findings /
31 MUST-FIX) and that **deference is the failure mode**.

Stop counter: **0 of 2**. Iteration 1 was not clean, so the earliest the loop can
end is iteration 3, and only if both 2 and 3 return zero MUST-FIX.

### Iteration 3 — 2026-08-23 — **CLOSED** — verdict: `CONVERGING_NOT_VERIFIED`

3 of 3 lenses · **53 MUST-FIX** (16 completeness · 21 contract-delta · 16
adversarial) · applied as **Revision 3**. Stop counter **0 of 2**.

**Six iteration-2 findings verified CLOSED** by the adversarial lens: the
`ci_arm`/`estimand_is_idealised` split, the `gff1_side_v3` admission,
provenance-as-state, the frozen-citation repoint, the disjunction withdrawal, and
`NOT_IDENTIFIABLE` as a state distinct from `INSUFFICIENT_EVIDENCE`.

**The defect that matters most was BE's, inside the fix for the same defect.**
§2.4 — written as the repair for *"BE wrote the machinery without walking it
against the receipts"* — claimed *"all 84 `route_a_v1` pass … and all 126
`route_a_v2` gate-3 cells."* Verified: `sigma_route_a_v1.json` holds **84
verdicts, all `INSUFFICIENT_EVIDENCE`, zero `PASS`**; `route_a_v2` has **no
receipt** and reads `PRE-REGISTERED / NOT YET EXECUTED`; and §6.3 of the same
plan calls `gff1_side_v3.json` *"the programme's only clean `PASS`"*. §2.4
counted 211, §6.3 counted 1.

**Three breaks in the chain from R-25 to `FIRE_SIDE`, all verified, all
propagated by BE before checking:** `STOP`'s own metric has **never been
computed** (zero `stop_*` receipts); `edge_l1_v1` is **not STOP's estimand** (no
fee, never cancels, and its protocol forbids combining layers into one PnL);
and the reading is **horizon-dependent**, holding at 1 of 4. Escalated as BE-6
and BE-7.

**§9 was found not to PARSE**, to carry no `fields:` keys (25 unexplained
removals), and BE's binding remediation command was unrunnable (`22` is not a git
ref). All four mechanically verified and fixed; §9 now parses, both blocks.

**BE-7:** the category-error premise the coordinator asked BE to carry is refuted
in its general form by `ww_v1` — an upper-bound test over *"parameter-free
maximal supersets"* that answers an existential — and R-25 depends on that very
property. The narrow form (opportunity cost is in no receipt) survives and is
what BE kept.

**Not clean. Completeness 16 MUST-FIX, contract-delta 21 MUST-FIX.** Stop counter
stays **0 of 2**.

**The pattern iteration 3 names, and it is the sibling of iteration 2's.**
Iteration 2: *BE wrote the machinery without walking it against the receipts.*
Iteration 3: **BE wrote the ruling into the PROSE without walking it into the
TYPES.** Revision 2 added `DISCHARGED_AS_MOOT`, `on_verdict`, `FIRE_SIDE` and
`PASS_SIDE` — and §9 declared **not one of the four**.

#### Four defects BE verified mechanically and fixed immediately

| # | defect | how verified |
|---|---|---|
| 1 | **§9's YAML did not parse.** `list[GateBar]` inside a flow sequence → `yaml.parser.ParserError`. `contract_check.load()` has no try/except, so it would have died on an uncaught traceback before running one check | ran `yaml.safe_load` on the block |
| 2 | **No `fields:` key** on `Gate`/`GateEvidence`/`GateRegistry`, so `flatten()` emits zero `field:Gate.*` and the diff reads as **25 unexplained REMOVALS** — incl. `Gate.preconditions`, the DAG's only edge field | parsed and inspected keys |
| 3 | **BE's own binding remediation command was UNRUNNABLE.** `contract_check.py 22 WORKTREE` → `fatal: invalid object name '22'`. `22` is a contract version; the tool takes a **git ref**. It had already propagated into this log | ran it |
| 4 | **BE claimed `--selftest` is "8/8 green"; it prints 14.** Stated from memory **inside the banner whose whole point is that hand-inspection is the defect** | counted the output |

The correct command, verified: `contract_check.py 2f6a156 WORKTREE` (the commit
holding `version: 22`). §9 now parses, both blocks, every type with `fields:`,
`modules:` hoisted out of the type namespace.

#### §9 rewritten to close the type findings

Parameterised states are **real unions** with payloads instead of enums with
loose siblings — the `estimand_route` defect v16 removed, which Revision 2
reinstated three times. `Contradicted(by, claimed, measured)` now carries the 4×
over-report that a flat enum could not hold; `NotIdentifiable(reason,
overturning_contrast)` forces a non-identification to name what would refute it.
`gff1_side_v3.json` and `sigma_route_a_v1.json` are both **constructible** —
`NullPin` on the six fields that were rejecting them, which is R-NULL, the
corpus's own mechanism, available all along. `bound_kind` deleted (duplicated the
union tag), `favourable_arm` moved to `Gate`, `Gate.version` deleted (two
counters for one gate was R-10's defect inside the field added to satisfy R-10),
`FIRE_SIDE`/`PASS_SIDE`/`on_verdict` declared, `Precondition` typed with all five
states including `DischargedAsMoot(closed_by)`.

**Recorded as still open rather than papered over:** the BE-reads-EV inversion,
`note:GateEvidence` being normative and now false, `SideConventionEvidence`'s
third verdict enum, and the migration records — which must come **from a checker
run**, not inspection, and that run needs a candidate v23 file that does not yet
exist.

### Iteration 4 — 2026-08-23 — **CLOSED** — verdict: `REFUTED_IN_MECHANISM`

3 of 3 lenses · **40 MUST-FIX** (12 mechanism · 12 vocabulary · 16
over-admission). Stop counter **0 of 2**.

#### THE CHECKER RUN — the loop's binding rule, finally discharged

```
contract_check.invariants(v22 + §9 patch)  ->  5 ERRORS
contract_check.invariants(v22 baseline)    ->  0
```

**BE's banner claimed this run was blocked** pending a candidate v23 file. **It
is twelve lines.** BE declared a check too expensive to run, having never tried
it, in the banner whose subject is that hand-inspection is the defect — the
second instance of that shape in one section, after claiming `--selftest` was
"8/8 green" when it prints 14.

All five fixed and **re-run: 0 errors**. They were: `registry: true` (a boolean
invented where v22's `registry:` names a type) ×2; `duplicate declaration (local
type AND prelude.external)` for **`GateId` and `Provenance`** — the checker's
only R-SSOT-named error, which iteration 2 named **by string**, shipped twice;
and `GateEvidence` with no declared producer.

**Honest boundary:** `invariants()` is one of three checks. The structural diff
and the migration records are still undone — the patch removes 12 `Gate` fields
and type-changes ~10 more.

#### The finding that invalidates the whole M13 repair chain

**`gff1_side_v3.json` is not a `GateEvidence`. It is a `SideConventionEvidence`**
— its fields match `contracts.yaml:607-616` field for field, and §9's own module
block already lists that type in `EV-Gates.consumes`. It was **never rejected by
`GateEvidence`, because `GateEvidence` never described it.** BE widened six
fields to admit a receipt that belongs to a different type — and five of those
six (`test`, `conditioning`, `multiplicity`, `effect_size`, `tolerance`) are
**non-null in 84/84 `route_a` cells**, so nothing on disk needed them widened.
`GateEvidence` is `ReducedFormFit.mean_gate`/`var_gate`: **BE loosened the
PRICING gate to fix a problem that did not exist**, re-opening the v15
null-comparison defect its own note was written to close.

#### The live consequence, and it lands on `STOP`

§6.4 names `edge_layer1_v1.json` as *the* `CONTRADICTED` exemplar — *"flagged
loudest"*. §4.5 then reads **that same receipt**, cell by cell, to `FIRE_SIDE` on
the programme-ending gate, and says nothing about population. Nothing in the plan
ties `provenance_state` to `decision_eligible`. Flagged loudest on one page,
consumed unflagged on another.

#### Also confirmed

`§6.3` has one operative rule left (*not p-value-only*) and an all-`NullPin`
`PASS` clears it — on `Conjunction`, the bar shape `STOP` itself uses, because
`bound_kind` was deleted and 4 of 7 bar variants now select **no branch**.
`NullPin` carries `bias_direction` in the type; the plan writes it **zero times**
and reads it never, and route_a's nulls have **no legal declarer**. `ci_arm` can
be flipped to turn `gff1_side_v1.json`'s real 226-of-500 `INSUFFICIENT_EVIDENCE`
into a `Pass()` on a `HALT_PROGRAM` gate, dropping a guard v22 hardcoded.
`provenance_state` is keyed on a **field name** (31 − 8 = 23, reproducing BE's
own count), so the corpus's two richest-provenance artifacts sort to `Missing`
while two receipts claiming a day the era cannot contain also sort to `Missing`.

#### working notes, superseded by the CLOSED iteration-4 entry above

**The vocabulary lens answers the coordinator's first question: ZERO of the four
frozen directional tests survive a round trip through Revision 3's grammar.**
12 MUST-FIX. Running total for iteration 4: **24 MUST-FIX**.

| test | result |
|---|---|
| `route_a_v2` gate 3 | EXPRESSIBLE **WITH LOSS** — loses its kind, its unit-bearing threshold, its max-cell simultaneity and Bonferroni family, and has no interval slot |
| `LAYER2` §3 cell bar | EXPRESSIBLE **WITH LOSS** — `ExcludesZero` holds ONE side, but the bar names two, so `POSITIVE` (`CARRY_RESCUES`, the outcome that would SAVE the programme) maps to `MODEL_REFUTED` |
| `LAYER2` §3 roll-up | **NOT EXPRESSIBLE** |
| `G-FF3` | **NOT EXPRESSIBLE** |
| `STOP-MM-VIABLE` | **NOT EXPRESSIBLE** |

**The roll-up failure is the worst kind.** The only near-neighbour variant is
`AtLeastK(k: int, …)` — and a hardcoded `k` is **exactly what R-14 Amendment 1
froze a rule to forbid**: *"the day rule must survive a running collector; `DAYS`
went stale four times in three days, so no verdict hardcodes a day count."*
`k=3` means 75 % at four days and 43 % at seven. BE wrote the one construction
the corpus had already outlawed by name.

**`G-FF3` cannot be registered at all**: `favourable_arm` is non-optional and
G-FF3 declares no favourable side — its question *is* the sign. Registration
would force EV-Gates to invent a direction the frozen gate never declared, which
§7 forbids in its own words.

#### BE verified the two structural claims against its own §9

- **`Gate` deletes 12 v22 fields**: `spec_hash`, `threshold`, `unit`, `question`,
  `metric`, `owner`, `inference_method`, `frozen_at`, `on_pass`, `data_prereq`,
  `strata_hash`, `artifact_hash`. **`spec_hash` is the one §2.4's entire
  enforcement rests on** — *"no declared mapping in its **pinned `spec_hash`**"* —
  and §7 says EV-Gates records `bar_ref`, `frozen_at`, `spec_hash`, **none of
  which is in the type**.
- **§2.1 names five carrier fields; §9 supplies two.** `tolerance` and
  `ci_hi_abs` are present; **`threshold`, `ci_lo` and `ci_hi` are absent.** So
  `gff1_side_v3.json`'s literal keys `{threshold: 0.99, wilson95: [...]}` have
  **no home in the type** — the **fifth** rejection of that receipt, inside the
  revision whose §9.1 announces it *"is constructible"*.

#### The pattern, restated by the lens and accepted

Iteration 2: machinery not walked against the receipts. Iteration 3: rulings
written into prose, not into types. Iteration 4: **BE wrote the types from §2's
ARGUMENT rather than from the four bars they must hold.** Each revision fixes
what the last review pointed at and reproduces the generative error one level
further out.

One lens pending. Revision 4 waits.

#### working notes, superseded by the CLOSED iteration-4 entry above

**The coordinator predicted the trap and the trap is still set.** They asked
whether `on_verdict` — *assemble and schedule, do not execute* — is *"actually
wired to anything, or prose in a plan with no mechanism behind it,"* noting
*"that last one is the trap my original had."* Mechanism lens: **12 MUST-FIX**,
and the answer is **nothing is wired**.

#### Three things BE verified mechanically

| # | finding | verification |
|---|---|---|
| 1 | **0 of 20 behavioural commitments have a rule or check.** The plan contains `rules:` **0 times** and `checks:` **0 times**; v22 carries 19 `checks:` entries. The corpus's own idiom for making a rule executable was available for four iterations and BE used it zero times | `grep -c` both files |
| 2 | **BE redeclared TWO types that already sit in `prelude.external`** — `GateId` (`:35`) and `Provenance` (`:16`) — while §9 declares both locally (`:675`, `:744`). That is `duplicate declaration (local type AND prelude.external)`, the checker's **only R-SSOT-named error**, which iteration 2 named **by string**. BE shipped two instances of it. §9's `Provenance` note even *quotes* the prelude declaration and then redeclares it | grepped both |
| 3 | **The corpus's ONLY executing gate logic is a file this plan has never mentioned.** `evaluation_pipeline.py:221-223` hard-codes `if result.verdict != "PASS": raise`, the 500-fill `VOID` floor, and the Wilson lower bound — a real DAG edge blocking Tier-2 on `G-FF1`, implemented outside any registry. The plan mentions `evaluation_pipeline` **zero times**, and its `FIRE_SIDE`/`PASS_SIDE` strings would **raise** there | grepped the code and the plan |

**Finding 3 is the one that reframes the loop.** BE has written a gate-registry
plan through four iterations without reconciling it against the one place in the
repo where a gate verdict actually gates something.

#### The pattern, one level up again

- Iteration 2: *BE wrote the machinery without walking it against the receipts.*
- Iteration 3: *BE wrote the ruling into the prose without walking it into the types.*
- Iteration 4: **BE walked the rulings into the types and did not walk the types
  into the checker, the modules, or a single check line.**

`on_verdict` specifically: typed as `FailRoute`, whose five variants contain none
that can hold *"assemble and schedule"* — and which **does** contain
`HaltProgram()`, the one thing §4.1 says the gate must not do, so the type admits
the forbidden value. It sits on a record **no module consumes**. Its "schedule"
half needs a clock port `R-ENV` forbids and only `OP-Monitor` has — the module
§5.1 spends a section forbidding EV-Gates to touch. Its "assemble" half has no
bundle type, no recipient, no completeness criterion.

Two lenses pending. Revision 4 waits for all three.

### Revision 2 — 2026-08-23 — R-24 / R-25 applied, plus three iteration-2 repairs

Not an iteration; the changes between iteration 2 closing and iteration 3 opening.

**R-24 — `STOP` AMENDED.** Verdict is now directional and symmetric:
`FIRE_SIDE` when **both** verdict coins exclude zero **from below**, `PASS_SIDE`
when both exclude **from above**, `INSUFFICIENT_EVIDENCE` otherwise. `on_verdict`
is **assemble the evidence and schedule the owner's decision** — the gate
computes and presents, it does not execute. The coordinator tightened their own
R-23 draft from *at least one coin* to **both coins in both directions**,
catching the same carelessness as the sign-blindness one notch smaller.

**The reasoning §4.2 now carries, and it is the most important idea in the
plan:** `STOP` asks whether there is **any** configuration in which a maker is
paid, and **no measurement can answer the "any"** — the receipts answer *in the
configurations we tested, no*. A statistical threshold therefore always answers a
**narrower question than the gate asks**, so treating its output as the decision
is a **category error**. Opportunity cost is the deciding input and no receipt
contains it. Generalised in the plan as a rule for kill-gates: a gate whose
question is broader than its metric **may present and may not execute**.

**R-25 — `cancel_v1` DISCHARGED AS MOOT** by `ww_v1`'s `DEAD` verdict. The
falsifier was answered; the §2 grid was never built and never needs to be.
Recorded explicitly in §4.4 so nobody later reads the absent receipt as missing
evidence. **Consequence: `STOP`'s preconditions are all in, and today's evidence
reads `FIRE_SIDE`** — which under `on_verdict` assembles the owner's decision.

**The disjunction machinery is therefore MOOT for `STOP`** and is not applied:
the precondition set resolved, so nothing needed relaxing. The discipline it
carried is retained for any future kill-gate — a leg is the protocol's own
**frozen verdict unit**, never a cell.

**BE UNDERSTATED M16 and the stronger number is now carried.** BE published the
`ww_v1` robustness margin as **1.6×**, computed with `R = 0.219`. The coordinator
verified independently using `R`'s point estimate (0.153) and `R`'s CI upper
(0.184): flip points **0.116 ¢** and **0.145 ¢** against a most-favourable
endpoint of **0.287 ¢**, i.e. **2.0–2.5×**. BE re-derived and confirms. Same
formula, correct inputs, stronger conclusion — recorded as a correction rather
than silently swapped, with both inputs named so the difference is auditable.

**Three iteration-2 repairs applied:**

1. **The `bound_arm` rule is replaced.** It refused `PASS` on every family that
   has ever earned one — `G-FF1`'s Wilson lower bound, all 84 `route_a_v1`, all
   126 `route_a_v2` gate-3 — and had the polarity backwards, refusing
   `SKEW_ROBUST`'s conservative arm while `ww_v1`'s upper arm *strengthens*
   `DEAD`. Split into **`ci_arm`** (which endpoint is the statistic — no effect
   on `PASS`) and **`estimand_is_idealised`** (generous by construction, under
   an assumption the arm cannot test — refuses `PASS`), plus `favourable_arm`.
2. **§6.3 now pins a worked admission**: `gff1_side_v3.json`'s exact shape
   (`threshold` + `wilson95`, no `tolerance`, no `ci_hi_abs`) is **admissible as
   written**, and any formulation that rejects it is wrong by construction.
   Revision 0 rejected it; Revision 1 rejected it again under a new field name.
3. **§6.4 no longer refuses on provenance.** Revision 1's refusal rejected
   `sigma_route_a_v1.json` — the sole receipt for all 84 cells — while §10
   promised to admit them. Registration now records a **state**:
   `ENUMERATED | MISSING | CONTRADICTED(by)`. `CONTRADICTED` is what the real
   failure needed: the defect was never missing provenance, it was a **4×
   over-report** that a present-vs-absent check passes cleanly.

**§9 is now banner-marked NOT CHECKER-VALIDATED**, listing the three hard stops
and the incomplete non-additive list, per the binding rule below.

