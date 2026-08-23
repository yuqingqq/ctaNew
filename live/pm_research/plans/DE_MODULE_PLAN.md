# DE plane — module plan: ActionSpace · Constraints · DecisionScheme · Allocator · Actuator

Plan only. No code, no measurement, no fitted quantity. Status: **DESIGN**, not
decision-eligible. **Revision 6, 2026-08-23** — applies `DE_PLAN_REVIEW_LOOP.md`
iteration 5: the ≤|net| cap's rationale rebuilt a second time — after two
successive false derivations from `L_adv` arithmetic (iteration 4's "falls at
ANY price pair" ignored that `L_adv` is dollar basis, not shares, and the flip
can RAISE it when the flipped side is expensive), the cap now stands
DEFINITIONAL to the REDUCING-ONLY state and the predicate is shown unable to
substitute in either direction; the §2 table's oracle-door scope reworded to
size-bearing verbs. **Now Revision 7** — iteration 7 header correction:
**Revisions 6–1 predate version control of this file and are NOT preserved**
(the "in git history" claim was false — the file was untracked); committed
per-iteration from this revision onward. Cross-session ledger:
`orchestrator/PROGRAMS/P-2026-003-polymarket-5min/workspace/COORDINATION.md`.

**Division of labour:**
[`DE_PLACEMENT_POLICY_PLAN.md`](DE_PLACEMENT_POLICY_PLAN.md) (current revision
per its own status line — cross-references here are revision-free by design,
iteration 3) is the **policy content**. This document is the **module structure**: the five DE
modules of `PM_ARCHITECTURE.md` §1, their contracts under §6/§8/§9, what the
measurements already fix, and what blocks the rest. Precedence:
`FLOW_MODEL_STATE.md` wins on facts; **`contracts/contracts.yaml` wins on
types** (the architecture prose defers to it — iteration 1 caught this plan
copying stale prose over the YAML); `PM_ARCHITECTURE.md` wins on rules.

---

## 0. The plane, and what exists

```
DE  DECISION   ActionSpace ❌ · Constraints ❌ · DecisionScheme ❌
               · Allocator ❌ · Actuator ❌
```

| module | owns | pinned by measurement | blocked by |
|---|---|---|---|
| ActionSpace | the typed action menu | one-book identity · tick regimes · fee schedule · fill rates per placement | venue mechanics facts (§1.3) |
| Constraints | feasibility oracle, HARD limits, the "new risk" predicate | terminal collapse · side-aware risk · verdict coins | `ScenarioLossLimit` / `κ_$` values (SP-Params, choices) |
| DecisionScheme | utility × solver × coupling | coupling (identity, exact) · the rule policy (skew measured, cancel specified) | optimizing solvers on BE/sigma clocks · utility on the edge sign |
| Allocator | capital/risk split, capital ops (MINT/MERGE), size budget | per-window settlement · cash-at-risk envelopes · pairing rates | cross-window correlation (§6.1) · SP-Params |
| Actuator | sole venue writer, lifecycle, rate budget, reconciliation | nothing — **the tape ends here** | venue access (deployment-gated) |

---

## 1. `DE-ActionSpace` — the typed menu

### 1.1 One exposure, the contract's six verbs

The state is **one signed Up-equivalent exposure per `(coin, window)`** — the
one-book identity is exact (1,081,800 checks, worst deviation 0.00000), so
listing Up- and Down-side actions separately would double-count every action.

**The verbs are the v22 contract's, not invented ones** (iteration 1: Revision
1 wrote `PLACE/NONE/SPLIT` against an existing
`ActionSpace.verbs: QUOTE|CANCEL|MINT|MERGE|CROSS|WAIT` — a name-vs-definition
instance inside a plan warning about name-vs-definition):

```
QUOTE(side ∈ {BID_UP, ASK_UP}, placement ∈ {JOIN, FRONT_ON_FORMATION}, size)
CANCEL(order_ref)
CROSS(side, size)                      # taker; fee 0.07·p(1−p) $/share, measured
MINT(size) | MERGE(size)               # $1 ↔ complete set; capital ops (§4)
WAIT
```

`ASK_UP ≡ BID_DOWN` identically; the complement never appears as a verb.
**What v22's `Action{verb, token?, level?, size?}` cannot yet carry:** an
`order_ref` (so `CANCEL` is inexpressible) and a `placement` field (so the
level-policy cannot be expressed except by abusing `level`). Those two fields
are the real ActionSpace contract gap — §6.2.

### 1.2 Placement is a level-policy, and `FRONT_ON_FORMATION` is conditional by nature

The btc/eth tick is 1 cent and the modal book is a 1-tick spread (median
exactly 1 tick, p90 2 ticks — improvement is possible in roughly the top decile
of quotes and nowhere else; the separate 99.9 % figure is conditional on the
tails-only fine-tick regime and shows the 1-cent spread is a *constraint*) —
so there is usually no price to choose, only where in the queue to stand
(iteration 3 aligned this with the policy plan's modal-population phrasing). At a static 1-tick book, price improvement would cross or lock
and a same-price queue jump is impossible, so **fronting is only executable at
genuine level re-formation**. The placement values are therefore `JOIN` and
`FRONT_ON_FORMATION` (front at the next formation event, rest `JOIN`
meanwhile) — the measured `SKEW_LB` arm, which is also the arm the composed
policy quotes. The symmetric-front idealisation (`front=True` everywhere,
`q_ahead=0`) exists only as a replay bound, never as an action. A price-offset
extension for the 0.001 tail tick is expressible later; nothing scheduled
needs it.

### 1.3 Venue facts this menu assumes, none verified

Tested at the Actuator boundary, not discovered there: minimum order size vs
the research `QUOTE_SIZE = 5` shares; whether resting `ASK_UP` requires holding
Up tokens (`MINT` first — capital, §4); supported time-in-force; whether
`CANCEL` is free and unthrottled; **whether the venue rejects self-matches**
(the scheme's cancel-before-cross rule, policy plan §4.7, does not depend on
it, but the fact belongs here); **what happens to orders resting at
resolution** (auto-cancel, convert, orphan); **settlement latency** — when
proceeds of a carried residual return as quotable capital. Any of these failing
reshapes the menu or the Allocator's rolling problem — falsifier §7.1.

---

## 2. `DE-Constraints` — the feasibility oracle, and the one definition of "new risk"

An **oracle, not a precomputed set** (`ConstraintSet.feasible(ActionSet,
PortfolioState) → FeasibleSet`): feasibility is conditional on the candidate
actions.

**THE PREDICATE (iteration 1 — one phrase had two meanings):** an action is
**new risk** iff it increases **contingent `L_adv`** — unpaired cost basis at
risk of the position *plus the worst-case fill of every resting quote*. The
halt edge and the `r<60` rule use this same predicate. Two consequences, both
deliberate: two-sided quoting reserves headroom for both quotes filling (the
honest reading for a maker — a feasibility check that prices only the held
position cannot prevent breach by fills on quotes it already approved); and
retraction of already-resting quotes is never the oracle's job — it can only
refuse, so retraction belongs to the scheme (terminal) or the halt path.

| constraint | source | class |
|---|---|---|
| `HALTED ⇒ FeasibleSet = ∅` — refuse all **size-bearing** verbs (iteration 5: "venue-write" wrongly implied `CANCEL`, which stays feasible), **not** the new-risk predicate (a reducing `CROSS` is not new risk under the predicate, yet must not trade on untrusted state — iteration 2); size-less `CANCEL`/`WAIT` remain feasible per §2a and are moot, the book being already retracted (iteration 4); `DEGRADED ⇒ REDUCING-ONLY`. **"∅" has a carrier only under §2a's default-DENY pin** (iteration 3) | OP halt port, latched, fail-closed | architecture §1 halt edge — **whose label ("no new risk") states the WEAKER rule; recorded divergence, §6.2, for coordinator reconciliation** (iteration 3) |
| per-scenario loss cap on contingent `L_adv`: `loss(s) ≤ ScenarioLossLimit(s.id)` — **evaluates scenario PnL directly**, never summed signed loadings | SP-Params keyed by scenario | architecture §8 |
| per-market contingent `L_adv ≤ κ_$` | SP-Params | architecture §8 |
| no new risk at `r < 60` | measured terminal collapse | policy plan §7 |
| eligibility: btc/eth only | `verdict_coins` | policy plan §0 |

**Breach response is a defined state, not an exception (iteration 1).** A burst
can fill the same re-posted quote many times between decisions, jumping `net`
past a cap with no infeasible action ever proposed. On breach of any cap the
scheme enters REDUCING-ONLY and the breach is a `HealthEvent` (§3.5).

**REDUCING-ONLY, defined once (iteration 2 — three statements had drifted):**
adding side cancelled; feasible = reducing quotes **and reducing `CROSS`, both
sized ≤ `|net|`** — **an explicit HARD rule of this state, owned by the
oracle's REDUCING-ONLY branch, and DEFINITIONAL: the state means never past
flat. Iterations 3–5 settled that the new-risk predicate cannot substitute
for the cap in EITHER direction — whether a flip past flat raises or lowers
contingent `L_adv` depends on size and the price pair
(`(s−|net|)·basis_new` vs `|net|·basis_old`; `L_adv` is dollar cost basis,
not share magnitude), so the predicate sometimes admits a flip (net +10 Up
at ~0.50, reduce sized 18 → flip to 8 ≈ $4.00 < $5.00) and sometimes
over-refuses one (net +10 Up at basis 0.10, reduce sized 18 at ~0.10 → flip
to 8 at ≈0.90 ≈ $7.20 > $1.00). Two successive derivations from `L_adv`
arithmetic were false (iterations 3 and 4); none is needed — the cap covers
both verbs because past-flat is past-flat, maker or taker. "The oracle will
refuse it" remains withdrawn** — plus capital ops (exposure-zero, permitted *within
this state*; under HALTED they are blocked like everything else, and their
only door is the Actuator — `CapitalOpCommand` bypasses the oracle and
DE-Allocator has no `halt_in`), `CANCEL`, `WAIT`. **Scope:** a per-market
`κ_$` breach or `r<60` enters it for that `(coin, window)` only; a
scenario-cap breach or `DEGRADED` enters it globally. One state, three
entries — cap-breach, `r<60`, `DEGRADED` — one permitted-action set.

**While HALTED (iteration 1 — the disposition was undefined):** `cancel_all`
has fired via the halt port (architecture: the halt port routes both the
constraint edge *and* the priority `cancel_all` — a reviewer's claim that
nothing triggers it was wrong and is logged); **all venue actions are blocked,
including risk-reducing `CROSS`** — the reason we halted means the state
cannot be trusted to trade on, even to reduce. A latched halt therefore
converts an operational fault into a carried residual bounded by realized
`L_adv`. **That is the designed, fail-closed degradation**; the operator reset
path is the only exit. If `r≈60` passes while halted, the terminal reduction is
forgone by design (policy plan §7). **Enforcement is named twice by design
(iteration 2 — the rule previously had no owner):** the oracle returns the
empty feasible set, and the Actuator, reading `halt_in = HALTED`, refuses every
venue write except `cancel_all` (§5) — a size-bearing solver decision already
in flight at latch time dies at either door (a `CANCEL`-only decision passes
the oracle by §2a and is harmlessly absorbed: the book is already retracted —
iteration 4 scoped this sentence).

### 2a. `FeasibleSet` semantics — pinned, because without the pin the halt door inverts (iteration 3)

v22's `FeasibleSet{max_size: dict[str,float], binding}` carries no action list
and no status field, so "∅" needs a convention — and the natural cap-dict
reading (missing key = uncapped) makes an *empty* set fully **permissive**,
silently inverting the oracle door of the halt rule and leaving
DEGRADED/REDUCING-ONLY with **no** enforcer (the Actuator's refuse-all covers
HALTED only). The pin, to land as contract *notes* (§6.2):

- **Key domain:** `"<verb>:<side>"` (e.g. `"QUOTE:BID_UP"`, `"CROSS:ASK_UP"`),
  within the one-instrument problem — **one `DecisionProblem` per
  `(coin, window)` is now a stated rule** (§3.3), which is what lets the keys
  drop the instrument. Side-keying is required because REDUCING-ONLY must
  forbid one side of the same verb while capping the other at `|net|` —
  verb-only keys cannot express the state at all.
- **Default-DENY:** a missing key means size 0. `HALTED ⇒ max_size = {}` is
  then literally the empty feasible set; DEGRADED/REDUCING-ONLY is the map
  with only reducing-side keys present (capped at `min(|net|, budget)`), and
  the oracle itself enforces the state — the Actuator stays the second door,
  not the only one.
- **Scope of the pin:** it governs `QUOTE` and `CROSS` — the size-bearing
  verbs the scheme can emit. Capital ops (`MINT`/`MERGE`) never appear in the
  map at all: they are Allocator-issued via `CapitalOpCommand`, route around
  the oracle, and their only HALTED door is the Actuator (iteration 4 dropped
  them from this sentence — a key domain of `"<verb>:<side>"` cannot express
  side-less verbs, and a dead-letter scope invites a wrong §6.2 note).
  `CANCEL` and `WAIT` are size-less and **always feasible in every state** —
  cancelling or doing nothing can never add risk, and REDUCING-ONLY depends on
  `CANCEL` being available. Under HALTED the question is moot: `cancel_all`
  has already retracted the book and the Actuator door blocks every venue
  write regardless.

**Removed from Revision 1 (iteration 1):** the "rate-budget feasibility" row.
DE-Constraints' contracted inputs are `DecisionProblem`/`HaltState`/
`ScenarioLossConstraint` — there is no port that could carry Actuator budget
state, so the row was a gate that cannot fire. Rate-budget pressure routes as
Actuator telemetry → OP → `DEGRADED HaltState`, which Constraints already
reads.

**SSOT note (iteration 1; classed under Ruling R-6, 2026-08-23):** every
constant in this table is an SP-owned bitemporal handle (`ParamId` / venue
fact); numbers in these plans are exposition; no module inlines them. Their
**character** (`SP_PLANE_PLAN.md` §4) bounds who may move them:
**Class A** configuration (`κ_$`, `CapitalBudget`, the `γ` ladder,
`refuse_k`) — coordinator-tunable; sweeps report the RANGE, never a best
point. **Class B** load-bearing (`quote_size_pin`, cancel-by deadline) —
changeable, but a change invalidates every measurement conditioned on the old
value, stated BEFORE the change. **Class C** measured (`r_terminal` — the
`r=60` handle, still with its withdrawn-grid provenance caveat —
`tau_operative`, `verdict_coins`, fee/tick/settlement) — published by a
worker, adopted by the coordinator, never chosen. **Class D** frozen verdict
bars (R-1's `ww_v1` rule) — after the measurement runs the bar is EVIDENCE,
not configuration; the standing instruction to refuse and record any post-run
move applies. The §5 replay set is OPERATIVE (R-6: no live orders, so no
second configuration exists behind it); every replay receipt states the set
it ran under.

---

## 3. `DE-DecisionScheme` — three axes, reconciled to the v22 config

`DecisionSchemeConfig{ utility, solver, unwrap_policy, unavailable_policy,
coupling_mode, incentive_model }` — the YAML's fields, not the prose's
(Revision 1 quoted `coupling_ref` and omitted `incentive_model`; the YAML
wins). Validation is n-ary.

### 3.1 Solver — `RulePolicy_v1` registers; the optimizers stay blocked for measured reasons

The composed policy (where to rest · when to leave · when to cross · when to
stop) registers as **`RulePolicy_v1`** beside the named seams `ClosedFormGLFT`,
`PerLevel`, `HJBQVI`. Every optimizing solver needs a fill-intensity model and
a fair price: `BE-FlowAndFills` requires a `VALIDATED` artifact (10 forward
days; none exists) and Route A is `PRICING HOLD`. `RulePolicy_v1` consumes no
belief, competition, outcome or incentive input — **its manifest is exactly
(`view`, `self`, `actions`, `portfolio`, `risk_scenarios`, `constraints`,
`horizon`), and that list is load-bearing: n-ary validation checks it against
§3.4, and it cannot silently grow** (iteration 2 rewrote the opening clause,
which contradicted the manifest it introduced). Size discipline arrives through
`constraints`: the scheme sizes each `QUOTE` within `FeasibleSet.max_size`
(§4) — no budget field enters `DecisionProblem` and the manifest does not
grow. `Decision.duals` is declared empty for a rule policy (duals become
load-bearing with optimizing solvers and incentive obligations); `rationale`
is consumed by EV-Attribution.

### 3.2 Utility — explicitly none, not silently misdeclared

Choosing a utility is a risk-appetite decision on top of an undetermined edge
sign. But `utility` is a required config field and `RulePolicy_v1` evaluates no
utility — declaring `risk_neutral` would be exactly the silent-misdeclaration
class n-ary validation exists to reject (iteration 1). So the registry gains
**`utility_none`**: a registered `UtilityFunctional` whose evaluation is a
typed refusal, R-COMPAT-valid *only* with solvers whose manifest declares no
utility consumption. The mean-variance `γ`-ladder of the policy plan stays a
reporting device, not a config value. `RiskNeutral/CARA/PathFunctional` remain
seams for the optimizer era.

### 3.3 Coupling — Dynamic, because the universe churns by construction

```
window_i{Up, Down}  ATOMIC        # exact: 1.08 M checks, 0 violations
window_i ↔ window_j SHARED_RISK   # same coin; correlation UNMEASURED → §6.1
```

Markets resolve every five minutes, so the coupling is
`coupling_mode: Dynamic` — `ResolvedCoupling.Dynamic(Known[CouplingGraph])`
carried per `DecisionProblem` (Revision 1 said `PER_DECISION` in prose while
quoting the Static ref form — reconciled). **The composition root constructs
the per-decision graph** from the live window universe; that duty, plus
`spec_snapshot` pinning and `horizon` semantics (`horizon` = time to this
market's resolution, the `r` every rule is indexed on) are composition-root
obligations named here so they are not discovered at build time. **Stated as a
rule (iteration 3): ONE `DecisionProblem` per `(coin, window)`** — it is what
`horizon = r` already implied and what §2a's instrument-free keys require. In
the rule-policy era the graph has no consumer inside the solver (the manifest
excludes `coupling`); it exists for the Allocator's scope and the optimizer
era, and its omission from the solver is manifest-licensed like any other
unconsumed input (§3.4).

### 3.4 Declared policies — legal variants only, omission licensed by manifest

v22's `UnavailableAction` is `Halt | RefuseAction | FallBack(FallbackPolicy)` —
Revision 1's `TOLERATE_UNUSED` was not a legal value (iteration 1). The
declaration:

```
unavailable_policy.by_input = { risk_scenarios: Halt }     # consumed, required
unwrap_policy.by_input      = {}                           # nothing unwrapped
```

Inputs the solver does not consume (`belief`, `competition`, `outcomes`,
`incentives`, **`coupling`** — iteration 3 completed the list) carry **no
entry**: their absence is licensed by the §3.1 manifest. The validation rule,
stated precisely (iteration 3): `by_input` must carry an entry for every
*consumed* input **whose type has an `Unavailable` arm** (today: exactly
`risk_scenarios`); consumed always-`Known` inputs (`view`, `self`, `actions`,
`portfolio`, `constraints`, `horizon`) need no entry and can never trigger the
policy; an omitted key on a consumed can-be-Unavailable input is a wiring
error, not a default.

**One contract change here is NOT additive (iteration 1):**
`DecisionProblem.belief` is `Known[BeliefProcess]` with no `Unavailable` arm,
so a problem cannot even be constructed while BE-Belief is a named seam.
Widening it to `Known[BeliefProcess] | Unavailable` is a change to an existing
field — a **migration record** (operation, key, old, new, version), and §6.2
lists it as such. `incentive_model` gets the mirror of §3.2:
**`incentive_none`**, a registered null `IncentiveModel` (empty contributions),
replaced by a real registration when `ρ`/rewards facts exist.

### 3.5 Telemetry — fail-closed needs sources, and Revision 1 gave DE none (iteration 1)

In v22 the **explicit** `DE-Constraints`/`DE-Allocator` records omit
`telemetry_out` (recordless modules ride a `DE-*` wildcard default that
includes it — iteration 2 corrected the overstated premise; the consequence
stands for the explicit records). A scheme refusal or a constraints-input
outage that never reaches OP-Monitor means no halt latches and the resting
book stays in the market — fail-closed silently not applying. **Each
DE module except ActionSpace declares `telemetry_out`, with named
`HealthEvent` sources:** DecisionScheme — solver refusal, `Halt`-policy
trigger, decision latency; Constraints — required input unavailable, cap
breach detected; Allocator — budget breach, capital-op failure; Actuator —
rate budget, venue errors, reconciliation divergence (§5). Port additions are
in §6.2.

---

## 4. `DE-Allocator` — capital, size, and the capital-op channel

**Unit: the `(coin, window)` instrument.** Settlement is per window (99.8 % on
1,465); the Allocator's problem is rolling: budget across the live window set
under the per-scenario caps.

**Ownership chain for capital and size (iteration 1 — both were unowned;
iteration 2 — Revision 2's chain had no contract carrier: no module consumed
`CapitalBudget` and the scheme's manifest excluded any budget input). The
wireable chain uses types v22 already has:**

```
SP-Params      total capital, κ_$, ScenarioLossLimit, size pin  (choices, bitemporal)
DE-Allocator   CapitalBudget (v22: by_instrument Money — unchanged)
DE-Constraints consumes CapitalBudget (ADDITIVE module-record change, §6.2)
               and folds it into the oracle alongside the caps: the implied
               per-verb size limit lands in FeasibleSet.max_size — a field
               v22 ALREADY HAS
DecisionScheme sizes each QUOTE within FeasibleSet.max_size (manifest unchanged)
Action.size    carries it
```

**Transport constraint, recorded as a gate note:** every measured basis — fill
rates, the 9.4× ratio, the skew cuts, the envelopes — is conditional on 5-share
quotes. Live size starts pinned to the measured support; any larger size
re-opens fill-rate/skew transport and must re-measure before promotion.

**Capital ops have one issuer and one executor (iteration 1 — MERGE had two
writers and no channel).** `m = min(q_up, q_down)` is paired, riskless, redeems
$1; `MERGE` recycles that collateral and changes exposure by exactly zero;
`MINT` pre-funds ask-side inventory if the venue requires it (§1.3, unverified).
**The Allocator is the sole issuer of `MINT`/`MERGE` (R-SSOT); they flow to the
Actuator on a typed `CapitalOpCommand` channel (new ports, §6.2) — not through
`Decision`, keeping the scheme's menu free of capital ops it should never
reason about.** The scheme's §1.1 menu retains the verbs only because
`ActionSpace.verbs` is one enum; the scheme never emits them.

**Settlement booking:** the carried residual settles at `S60(T)`; settlement
FACTS are DA (`DA-Settlement`, read downward), the cash books to the
WealthLedger per architecture §7, and the Allocator sees it return as
`PortfolioState.capital` — subject to the unverified settlement-latency fact
(§1.3). **The measured cash-at-risk envelopes (btc p95 $8.11–9.14 skewed) are
conditional on the no-MINT-first branch** (iteration 1); if minting is
required, capital locked is ~$1/share of ask-side inventory and the envelope is
not the capital requirement.

Build-order anchor (architecture §11.7): **`L_adv` + `DE-Allocator` + STOP gate
before any order.**

---

## 5. `DE-Actuator` — the interface where the tape ends

Sole venue writer. Owns order lifecycle, rate budget, actuation-level debounce,
**and reconciliation**. Ports (corrected to the v22 map, iteration 1): `venue`,
`halt_in`, `cancel_cmd_in`, `telemetry_out` — plus `capital_cmd_in` (§4, new).
`cancel_all` bypasses the solver but not the Actuator; idempotent,
`retry_until_ack`. **While `halt_in` reads `HALTED`, the Actuator refuses every
venue write except `cancel_all` — the second enforcer of §2's halt rule
(iteration 2), so an in-flight solver decision cannot slip through at latch
time.**

**Reconciliation is an Actuator duty, and it is what makes gaps and restarts
survivable (iteration 1).** `SelfState` is accounting from our own observed
fills; during a collector gap the venue keeps filling our resting quotes
unobserved, so post-gap `net`/`cost_basis` is silently wrong unless
reconciled. On **every recovery, restart, and halt-reset: reconcile open
orders and fills against the venue before any quoting resumes** —
reconcile-before-quote, no exceptions. Divergence between reconciled and
internal state is a `HealthEvent`. The replay convention (kill state at gaps)
is the harness analogue, not a substitute.

**The τ-rung seam, with its observation mechanism named (iteration 1):** venue
acks are NOT observed (contract: confirmation by open-order reconciliation), so
deployment measures an **upper bound** on cancel latency, owned by
`OP-LatencyBudget`. The operative rung is the smallest ladder rung ≥ that
bound — conservative by construction — stored as an SP-Params value with
provenance, consumed by the Actuator. If the bound exceeds the ladder's top
rung (1000 ms), the cancellation lever is dead at deployment regardless of
replay results (falsifier §7.3). If reconciliation cannot resolve adjacent
rungs, the coarser rung applies — the seam degrades conservative, never
optimistic.

**Debounce vs policy hysteresis (iteration 1 — two owners, one phenomenon):**
the flat band's entry/exit hysteresis is a **policy** parameter, in the replay
grid, because the band generates the flips. Actuator debounce exists only to
protect the venue session and **must be identity in any replay comparison** —
if actuation damps flips, the executed policy is no longer the measured
policy.

## 5b. Lifecycle — states neither plan owned (iteration 1)

```
STARTUP/RESTART   reconcile-before-quote (§5); resume from SelfState only after
                  venue agreement; never assume the book we left is the book
                  that exists
RUNNING           the composed policy, per window
DEGRADED          REDUCING-ONLY, global scope (§2's canonical definition —
                  one permitted-action set, three entries)
HALTED            cancel_all fired; nothing else, including reducing CROSS (§2);
                  carry; operator reset is the only exit
TERMINAL          per window: r≈60 adding-side retraction (scheme's action),
                  r<60 reducing-only, r→0 carry (policy plan §7)
RESOLUTION        cancel-by deadline (SP param) ahead of T unless the venue
                  auto-cancels (§1.3, unverified); residual settles; proceeds
                  return per settlement latency; next window's quoting is
                  independent of the previous window's unresolved settlement
                  only if capital headroom says so (Allocator)
SHUTDOWN          cancel_all + reconcile + persist SelfState
```

---

## 6. Build order — demand-driven, three stages

### 6.1 Now — demanded by the policy plan's §8 replays

- **ActionSpace vocabulary and Constraints content as replay parameters** — the
  replay policies are ActionSpace words; writing them as §1's typed menu keeps
  the harness dialect and the contract from diverging. **Parity is a promotion
  rule, not a vocabulary aspiration (iteration 1): any candidate promoted on
  forward days must run as the registered `RulePolicy_v1` under EV-Replay —
  a harness transcription can develop, it cannot promote.**
- **The cross-window correlation measurement** (fills §3.3's SHARED_RISK edge):
  same-coin adjacent windows, correlation of residual moves and of simulated
  `net` under the standard two-sided replay. Retires the DA plan's falsifier #2
  in whichever direction it lands and decides §4's character.

### 6.2 When code starts — the reconciled contract-change list (iteration 1)

Split honestly by kind — Revision 1's list was wrong in both directions
(`CANCEL` already exists in `ActionSpace.verbs`; the real gaps were unlisted):

| kind | change |
|---|---|
| **contract edit + migration record** | widen `DecisionProblem.belief` → `Known[BeliefProcess] \| Unavailable` (§3.4 — NOT additive) |
| contract addition (additive) | `Action.order_ref`; `Action.placement ∈ {JOIN, FRONT_ON_FORMATION}`; `CapitalOpCommand` + `capital_cmd_out/in` ports (§4); `DE-Constraints` consumes `CapitalBudget` (module record — closes the size chain, §4); `telemetry_out` made explicit on the Constraints/Allocator records with named `HealthEvent` sources (§3.5); **`FeasibleSet.max_size` notes: side-keyed key domain + default-DENY (§2a — without it "∅" has no carrier and an empty set is permissive)** |
| architecture-prose reconciliation (coordinator) | relabel the §1 halt edge at DE-Constraints from "no new risk" to refuse-all/`FeasibleSet = ∅` — the current label states the weaker rule iteration 2 removed, and a future reconciliation citing it would revert the fix (iteration 3) |
| registry entry (no contract edit) | `RulePolicy_v1` (solver, with consumed-inputs manifest); `utility_none`; `incentive_none`; R-COMPAT rows for the triple |
| pure config (no contract edit) | the §3.3 coupling graph (`CouplingGraph` is already `config_supplied`); `unavailable_policy`/`unwrap_policy` values |

Version bump, structural checker, migration discipline as always.

### 6.3 Waiting, with the thing each waits on

| item | waits on |
|---|---|
| optimizing solvers (GLFT/PerLevel/HJBQVI) | `VALIDATED` FlowAndFills (10 forward days) + Route A gates |
| a real utility choice | an edge sign (~25–30× data at settlement; cancel grid at Layer 1) |
| `ScenarioLossLimit`, `κ_$`, `γ`, total capital, cancel-by deadline | SP-Params — **the operative set exists (R-6)**; further movement per class: A freely (report ranges), B with invalidation stated first |
| Allocator beyond accounting | §6.1's correlation result |
| Actuator | venue access (deployment-gated) |
| a real `IncentiveModel` | `ρ`/rewards facts; enters as ONE registration, never a four-file edit |

---

## 7. What would falsify this structure

1. **Venue mechanics contradict the menu** (§1.3): minting-first, min size
   above 5 shares, throttled/priced cancels, self-match handling, resolution
   auto-cancel behaviour, settlement latency. Each reshapes ActionSpace or the
   Allocator's rolling problem; none is testable passively.
2. **Cross-window correlation dominates** (§6.1): the `(coin, window)` unit
   stops being the risk unit and the Allocator is a real portfolio optimizer on
   the critical path much earlier than §6.3 assumes.
3. **Measured cancel-latency bound exceeds the τ ladder** — the cancellation
   lever dies at deployment even if every replay cell is positive; only the
   skew and terminal levers survive.
4. **Competition reflexivity**: every fill rate and warning window is measured
   on a tape without us in it. `BE-Competition` is `Unavailable` — tolerated
   for research by §3.4's manifest licensing, but any live gate must re-verify
   the fill-rate asymmetry the skew mechanism runs on.
5. **Contingent-`L_adv` feasibility is unworkably tight** (§2): if pricing
   worst-case fill of both resting quotes leaves no feasible two-sided book
   under plausible caps, the constraint design forces one-sided quoting and
   the composed policy's economics change shape.

## 8. What this plan deliberately does not do

- **Write policy content** — the placement plan (current revision per its
  status line).
- **Choose a utility, set any SP-Params value, or estimate PnL/capacity** —
  mechanism-first stands.
- **Own the strategy spec** — `SP-Strategy` is a SPECS-plane slot; DE consumes
  its declared config bitemporally.
- **Touch the venue or its docs as if verified** — every §1.3 fact is an
  assumption until the Actuator boundary tests it.
- **Freeze contracts** — §6.2 names the changes by kind; the contracts file
  and checker remain the source of truth when they land.
