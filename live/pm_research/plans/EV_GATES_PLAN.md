# EV-Gates — model plan: the gate registry, the DAG, and STOP

**Revision 3** — 2026-08-23. Status: **DESIGN**, not decision-eligible.
Owner: **BE**, transferred under ruling **R-18**. Review loop:
`EV_GATES_REVIEW_LOOP.md` (iteration 1 `REFUTED_IN_SUBSTANCE`, iteration 2
`REFUTED_IN_MACHINERY`).

**Revision 2 applies R-24 (STOP amended, §4) and R-25 (`cancel_v1` discharged as
moot, §4.4), makes the verdict grammar DIRECTIONAL (§2), and repairs the two
checks that rejected the corpus's own clean work (§6.3, §6.4).**

> `FLOW_MODEL_STATE.md` wins on facts · `contracts/contracts.yaml` v22 wins on
> types · `PM_ARCHITECTURE.md` explains and does not define.

> **STABLE ANCHORS — read before citing a section number.** Revision 1's
> renumbering broke a live citation inside a **frozen** document:
> `LAYER2_PROTOCOL.md:105` (frozen under R-14) cites *"EV_GATES_PLAN §5.1
> extended from failing witness to firing witness"*, and in Revision 1 §5.1 is
> "Not a runtime edge" while that content moved to **§6.2**. BE cannot edit a
> frozen protocol to repoint it, so the mapping is recorded here instead:
>
> | cited as | Revision 0 | now |
> |---|---|---|
> | vacuity / witness check | §5.1 | **§6.2** |
> | `p_value`-only refusal | §5.2 | **§6.3** |
> | population / `days_sampled` | §5.3 | **§6.4** |
> | change-detector vs conformance | §5.4 | **§5.4** (unchanged) |
>
> This is R-10's own defect — *an amendment that changes content must change the
> address* — committed by the plan that imports R-10 in §9. Future revisions
> cite `EVG-VACUITY`, `EVG-PVALUE`, `EVG-POPULATION`, `EVG-CONFORMANCE` rather
> than numbers.

**Revision 1 is a REWRITE, not a patch.** The coordinator-authored draft
(Revision 0, single-pass, in git history) was reviewed by three independent
lenses and returned **20 confirmed MUST-FIX**. Three sections survived and are
carried forward largely unchanged — the ownership split (§7), the non-runtime
halt path (§5.4), and change-detector-vs-conformance (§6.4). Everything
decision-bearing was rebuilt. §0.1 records what was refuted, because a rewrite
that hides what it replaced is how a defect comes back.

---

## 0. The decision in one line

**`EV-Gates` owns the enumeration of pre-registered verdict cells, the
precondition semantics, the verdict *grammar*, and `STOP`'s wiring — and it owns
none of the thresholds and none of the mappings.** It is a bookkeeper with teeth.

**And it cannot be built against contracts v22.** Every substantive commitment
below is unrepresentable in the current types. Revision 0's claim that it "does
not invent types, it says who owns them" was false. This plan is therefore
**half module plan, half contract-change proposal**, and §9 states the v23 delta
explicitly rather than leaving an implementer to discover it.

### 0.1 What Revision 0 got wrong — the refutation, kept

| # | Revision 0 claim | verified reality |
|---|---|---|
| R0-1 | "`Gate.preconditions` … nothing populates it, so the DAG is a type with no instances" | **False.** `plans/BE_FLOWANDFILLS_PLAN.md` lines 953/961/972 declare `preconditions: [G-FF1]`, `[G-FF1, G-FF2]`, `[G-FF2]`. A three-edge sub-DAG exists; R0's spine omitted all three |
| R0-2 | `STOP` "is closer to firing than any other gate" | **Inverted.** `cancel_v1` is a declared precondition and `CANCEL_POLICY_PROTOCOL.md:23` says it *"will therefore never report"*. Under R0's own §3, `STOP` is pinned `INSUFFICIENT_EVIDENCE` forever |
| R0-3 | `STOP` threshold "interval must exclude zero on ≥1 verdict coin" | **Sign-blind.** btc `h=5` = −0.532 ¢ [−0.797, −0.287] **satisfies it**, so the evidence that kills the programme reads `PASS` → "proceed to the DE build". Routed to the USER as **BE-4**; not fixed here |
| R0-4 | one verdict vocabulary, mandatory central map | Covers **under a third** of the corpus's verdict words, and a central map goes stale whenever a worker freezes a bar |
| R0-5 | three verdicts suffice | The three are an **equivalence grammar**. `ci_hi_abs` is an absolute bound and destroys the sign, while `route_a_v2` gate 3, `LAYER2` §3 and `G-FF3` are all directional |
| R0-6 | §5.1 failing-witness check | **Degenerate on all three real gates it can be tested against**, and R-14 had already ruled it insufficient |
| R0-7 | §5.2 "tolerance and ci_hi_abs non-null on every PASS" | **Rejects the programme's only clean `PASS`** — `gff1_side_v3.json` is a one-sided superiority test with neither field |
| R0-8 | §5.3 `days_sampled` mandatory | **Rejects the 210 sigma gates §8 promises to admit**: 23 of 31 receipts carry no provenance block, including `sigma_route_a_v1.json` |
| R0-9 | census ≈ 220 gates | Missed ≥6 frozen families and had **no declared unit of enumeration** |
| R0-10 | `RETAIN` "needs its own type" | The type was never provided, leaving the Hawkes retention decision with **zero** owners (R-SSOT requires exactly one) |

---

## 1. The census — by a declared unit, walked from the protocol files

**Unit of enumeration, declared:** *one registry entry = one **pre-registered
verdict cell** — the smallest (protocol, scope) pair that receives its own
verdict.* Revision 0 counted `route_a` at cell granularity and `ww_v1` at family
granularity, which made "roughly 220" an artifact of the counting convention.

| family | protocol | cells | note |
|---|---|---:|---|
| `route_a_v1` | `SIGMA_ROUTE_A_PROTOCOL.md` | 84 | 7 sym × 6 hz × 2 gates |
| `route_a_v2` | `SIGMA_ROUTE_A_V2_PROTOCOL.md` | 126 | 42 fits × 3 gates |
| `G-FF1…G-FF4` | `BE_FLOWANDFILLS_PLAN.md` §gates | 4 | **already carry `preconditions`** — the registry seed |
| `edge_l1_v1` | `EDGE_LAYER1_PROTOCOL.md` | 8 | 2 verdict coins × 4 horizons |
| `ww_v1` | `CANCEL_POLICY_PROTOCOL.md` §1.4 | 8 + 1 | 8 coin-days + family verdict |
| `cancel_v1` | same, §2.3 | — | **never instantiated**; family closed by R-11 |
| `layer2_v1` | `LAYER2_PROTOCOL.md` §3 | 8 + 2 | frozen by R-14, measured under R-17 |
| `policy_v1` | `POLICY_COMPARISON_PROTOCOL.md` | 6 | incl. an equivalence bound |
| `queue_type_v1` | `QUEUE_AND_TYPE_PROTOCOL.md` | C1 + C2 | + a 1 % reconciliation gate |
| skew | `PLACEMENT_SKEW_RESULTS.md`, `SKEW_BOUND_RESULTS.md` | 2 × per-coin | two 4-way bars |
| BE retention | `BE_FLOWANDFILLS_MODEL_PLAN.md` §4.3/6.1/6.2 | per layer × coin | promotion + Hawkes retention |
| `U1…U11` | `FLOW_UNCERTAINTY_LOOP.md` | 11 | each with a pre-stated `CLEARED if` |
| DA admissibility | `CLOB_ADMISSIBILITY_PROTOCOL.md`, `PM_MEASUREMENT_PREREG.md`, leak canary | — | **not DAG nodes — see §5.3** |
| `G1`/`G2` | `PROGRAM.md` | 2 | sketch only, never pre-registered |

**The registry TRANSCRIBES `BE_FLOWANDFILLS_PLAN.md`'s `gates:` block under
BE's ownership; it does not import it.** Revision 3 said *"it is imported, not
re-derived"*, and that was wrong on two counts, both found while running R-57's
condition 4 against this plan's own removal set:

- **That file is SUPERSEDED (2026-08-21), retained as the fill-model audit
  trail.** An audit trail is not a live source. Importing from one makes the
  registry track a document nobody may edit and nobody maintains — wrong since
  the day it was superseded, independent of any contract version.
- **The block is plan-local YAML, not `Gate` instances.** `G-FF1…G-FF4` do not
  appear in `contracts.yaml`; its `question:`/`threshold:`/`unit:` keys are
  **name collisions in schema position**, not `Gate.question` citations.

So the four G-FF gates are **re-declared here, in current contract vocabulary,
owned by BE-FlowAndFills**, citing the audit trail as provenance. Their content —
the three `preconditions` edges, `inference_method`, `review_date`, per-gate
`on_fail` — is preserved verbatim in meaning; only the vocabulary moves.

> **Why this correction exists.** BE grepped the twelve fields Q-BE-7 removes,
> found seven of them in that block *in schema position*, and filed `Q-BE-15`
> claiming R-57 condition 4 bit. It did not. BE applied the "schema position"
> half of the test and never asked **whose schema** — the same false-positive
> class R-59 identifies in three coordinator instruments. The removal was always
> safe; an audit trail referencing its era's vocabulary is not a dangling
> reference, it is what an audit trail is. The defect was here all along.

**No total is quoted.** A single number invites the counting-convention error
Revision 0 made. The registry reports counts per family, per unit.

---

## 2. The verdict grammar — four states and a bound kind

Revision 0's three states are an **equivalence** grammar. They cannot carry a
sign, and half this registry is directional.

### 2.1 Bound kind, declared per gate at registration

```
TWO_SIDED_EQUIVALENCE   PASS iff |effect| CI bound lies INSIDE tolerance
                        carrier: tolerance + ci_hi_abs      (route_a calibration)
ONE_SIDED_SUPERIORITY   PASS iff the SIGNED bound clears the threshold
                        carrier: threshold + ci_lo or ci_hi (G-FF1, route_a_v2 g3)
SIGN                    PASS iff the interval excludes zero from a NAMED side
                        carrier: signed ci_lo, ci_hi        (LAYER2 §3, G-FF3)
```

**Every gate declares which null it tests and which direction is favourable.**
Revision 0 left this to the reader, so `REJECTS_UNIT_POISSON_CLUSTERING` — the
outcome that *licenses* the Hawkes layer — mapped to `MODEL_REFUTED` by letter
and `PASS` by intent.

### 2.2 The four states

```
PASS                 the declared bound clears, in the declared direction
MODEL_REFUTED        the declared bound is violated, interval excluding it
INSUFFICIENT_EVIDENCE(resolvable_by = CALENDAR | SAMPLE)
                     neither, AND more of the named resource would resolve it
NOT_IDENTIFIABLE(reason)
                     neither, and MORE DATA CANNOT HELP
```

**`NOT_IDENTIFIABLE` is the state Revision 0 lacked, and its absence inverted an
operational instruction.** `queue_type_v1` C1 returned `UNIDENTIFIABLE`:
*"crediting all of it reproduces `FRONT`, crediting none reproduces
`BACK_DISPLAYED`; the interior is reachable only under an assumption, not a
bound"* — and `FLOW_MODEL_STATE` §3 files it under *"do not schedule work against
these"*. Mapping it to `INSUFFICIENT_EVIDENCE` tells a reader to collect data
that cannot change the answer.

### 2.3 Admissibility is a FLAG, not a verdict

`VOID` (< 500 fills, five protocols) and `DAY_BLOCK_UNAVAILABLE` (*"a refusal,
not a small number"*) say **the test was never admissible**. They are recorded as
`admissible: false` + `reason` **beside** the verdict, never as one. A cell that
is inadmissible has no verdict at all.

### 2.4 Mappings are declared PER PROTOCOL, never centrally

Revision 0 wrote a central map covering under a third of the vocabulary, and a
central map goes stale every time a worker freezes a bar.

**Rule:** each protocol declares its own `verdict_map` in the protocol, inside
the text the `spec_hash` pins. **`EV-Gates` refuses to ingest a receipt whose
verdict string has no declared mapping in its pinned `spec_hash`.** The refusal
is the enforcement; nobody maintains a list.

Two mapping rules are binding, from real laundering found on disk:

- **An IDEALISED estimand may not map to `PASS` — but a CI arm is not an
  idealisation.** `SKEW_SUFFICIENT` is *"an upper bound, not an achievable
  result"* by its own results doc, because it is computed under a re-post
  idealisation the arm cannot test — btc gets *"124 free re-fronts per window"*
  under it and zero under the lower bound. Mapping that to `PASS` promotes an
  idealisation into a validation.

  **Revision 1 encoded this wrongly and it is corrected here.** It said
  *"receipts carrying `bound_arm: UPPER|LOWER|POINT` are refused `PASS`"*, which
  refuses `PASS` on the one family that **has** earned it and on every family that
  **could**: `G-FF1` passes by a Wilson **lower** bound (`contracts.yaml:616`
  makes `EV-Markout` refuse without it); `route_a_v1`'s 84 cells and
  `route_a_v2`'s 126 would pass by an **upper** bound inside tolerance and a
  one-sided upper bound `< 0` respectively.

  **Corrected 2026-08-23 — BE's first version of this paragraph claimed all 210
  had passed. They have not.** `sigma_route_a_v1.json` carries **84 verdicts, all
  `INSUFFICIENT_EVIDENCE`, zero `PASS`** (`n_oos_days: 1` against a 10-day
  requirement); `route_a_v2` is `PRE-REGISTERED / NOT YET EXECUTED` with **no
  receipt on disk**. §6.3 of this same document says `gff1_side_v3.json` is *"the
  programme's only clean `PASS`"* — so §2.4 counted 211 and §6.3 counted 1, in
  one plan. The argument does not need them: `G-FF1` alone is sufficient, and the
  other two matter prospectively. **BE wrote this paragraph as the FIX for
  "wrote the machinery without walking it against the receipts", and did not walk
  it against the receipts.** It also had the polarity backwards: `SKEW_ROBUST` is the
  **conservative lower-bound** arm and would have been refused, while `ww_v1`'s
  upper arm makes `DEAD` *harder* — it strengthens its verdict.

  **Two independent fields, because one cannot carry both:**

  | field | meaning | effect on `PASS` |
  |---|---|---|
  | `ci_arm: UPPER \| LOWER` | which CI endpoint is the test statistic | **none** — this is normal one-sided inference |
  | `estimand_is_idealised: UPPER \| LOWER \| NO` | the estimand is generous by construction, under an assumption the arm cannot test | **refuses `PASS`** when the idealisation flatters the verdict |
  | `favourable_arm` | which direction the gate's declared null makes favourable | decides whether an idealisation flatters |

  `estimand_is_idealised` is declared by the protocol, not inferred from the
  shape of a number.
- **A scoped clearance keeps its scope.** `CLEARED-ABSENT-IN-TRADE` establishes
  only absence of a *per-trade, in-transaction* rebate; every `ρ`-dependent
  estimand stays `Unavailable`. Verdicts carry the protocol's own scope-limit
  string as a mandatory non-empty field, and a `verdict_coins` scope, since every
  protocol here is *btc/eth verdict, rest descriptive*.

### 2.5 `RETAIN` — excluded, and its owner named

`RETAIN` is a **model-selection** outcome, not a gate verdict, and must not be
laundered into `PASS`. Revision 0 was right and stopped one step short: it
deferred the replacement type to nobody, leaving the Hawkes retention decision
with zero owners.

**Owner: BE. Type: `ModelSelectionOutcome{ layer_ref, decision:
RETAIN|DELETE|CENSORED|UNRESOLVED, evidence, spec_hash }`**, read by nobody at
decision time. Declared in §9's delta.

Recorded honestly: C2's `RETAIN` interval shows *"fit stability across window
resamples, not sampling uncertainty"* — so no inferential verdict should attach
to it under any vocabulary.

---

## 3. Precondition semantics — the part Revision 0 got backwards

Revision 0: *"a gate whose preconditions have not passed is not evaluated, and
its verdict is `INSUFFICIENT_EVIDENCE` by construction."* Three defects.

### 3.1 `NOT_EVALUATED` is not `INSUFFICIENT_EVIDENCE`

The first is a **scheduling fact**, the second a **claim about evidence**.
Conflating them lets the registry assert something false about measurements that
exist. `NOT_EVALUATED(blocked_by: GateId)` is its own state and carries no
inferential content.

### 3.2 Blocking applies to ESTIMATES, not to REFUTATIONS

**Rule:** a downstream gate is blocked only if its metric is a function of an
unvalidated upstream **point estimate**. A gate whose verdict is invariant across
the upstream's entire interval **is evaluated**, with the upstream interval
carried as declared scope.

Worked case: `ww_v1`'s upstream (Layer-1) is `HORIZON_DEPENDENT`, so Revision 0
would suppress it. But `f* = |markout|/(spread+|markout|)`, and btc's flip point
is |markout| = 0.180 ¢ against a measured CI of [−0.797, −0.287] — the *most
favourable* endpoint is **1.6× beyond** it (eth 4.6×). The refutation is more
robust the worse the upstream is, and Revision 0 discarded it.

### 3.3 A non-passing precondition is followed via `on_fail`, not frozen

`Gate.on_fail` already carries a **route**. `G-FF2.on_fail` is *"drop queue
modelling; regress fill on level and displayed depth"* — the model changes shape
and continues. Revision 0 had no concept of that and would have frozen `G-FF3`
and `G-FF4`, making `G-FF4`'s conclusion *"MM is not deployable"* unreachable
exactly when true.

**Rule:** resolve a non-passing precondition by following its `on_fail` edge.
Blocking is reserved for `on_fail = HALT_PROGRAM`.

### 3.4 Precondition states

```
PASSED                    satisfied
NOT_YET_EVALUATED         scheduling; may resolve later
DISCHARGED_BY_REFUTATION  the question is SETTLED, negatively, and will not be
                          measured — e.g. cancel_v1, whose family R-11 closed
INADMISSIBLE              never evaluable on this data
```

`DISCHARGED_BY_REFUTATION` is what Revision 0 lacked, and its absence is why
`STOP` was pinned forever.

---

## 4. `STOP-MM-VIABLE` — AMENDED under R-24, preconditions closed under R-25

**Owner: THE USER.** Not BE's and not the coordinator's — R-18. BE owns the
machinery; BE does not own the decision, and neither does the coordinator.

### 4.0 R-24 was PROSE-ONLY. It now has a mechanism — R-36 landing evidence

R-36 re-opened R-24 as **prose-only and not in force**, because BE had reported
it applied when it was a sentence: `on_verdict` typed `FailRoute`, zero checks,
zero code. The landing evidence R-36 named was *"a check that FAILS when a
sign-blind threshold is supplied."*

**It exists: `live/pm_research/ev_gates.py`.**

| element | location | what it does |
|---|---|---|
| `assert_directional()` | `ev_gates.py:106` | feeds a rule an input and its **mirror through zero**; refuses any rule answering the same for both |
| `stop_verdict_original_SIGN_BLIND()` | `ev_gates.py:75` | the bar R-24 replaced, kept **only as the failing witness** |
| `stop_verdict_r24()` | `ev_gates.py:54` | the amended rule: both coins, both directions |
| selftest | 11 checks | `python3 live/pm_research/ev_gates.py --selftest` |

**Demonstrated, not asserted.** On the measured btc/eth `h=5` cells the old bar
returns `PASS` — and returns `PASS` again on their reflection `btc [+0.287,
+0.797] · eth [+0.759, +1.726]`. Same verdict for "the maker is destroyed" and
"the maker is paid". The amended rule returns `FIRE_SIDE` and `PASS_SIDE`.

**The design choice matters more than the code.** Sign-blindness is **DETECTED
by mirroring the evidence, not DECLARED by the author.** A `favourable_arm` field
asserting one's own directionality is exactly what failed everywhere else here —
*the name is not the definition*, and an author who has not noticed the defect
will not declare it. The mirror test does not care what the author believed.

This is also the first instance in this plan of §6.2's own requirement — a check
with a concrete input under which it fails, and the failing input is on disk.

### 4.1 The amended gate

```
id             STOP-MM-VIABLE
question       Is there ANY configuration in which a maker on these markets
               is paid?
owner          the user (human)
metric         net edge per fill, after fee, against book mid at a fixed short
               horizon, under a STATED cancellation policy
inference      day-clustered WHERE CLUSTERS PERMIT, block bootstrap,
               notional-weighted AND per-fill

verdict        FIRE_SIDE              both verdict coins exclude zero FROM BELOW
               PASS_SIDE              both verdict coins exclude zero FROM ABOVE
               INSUFFICIENT_EVIDENCE  otherwise

on_verdict     ASSEMBLE THE EVIDENCE AND SCHEDULE THE OWNER'S DECISION.
               The gate COMPUTES AND PRESENTS. It does not execute.

preconditions  ww_v1 (DEAD) · cancel_v1 (DISCHARGED_AS_MOOT, R-25) ·
               edge_l1_v1 · EV-Markout · G-FF1
review_date    calendar or era-day count — never an event a closed family
               would have to produce
```

**Both coins in both directions.** The coordinator tightened their own R-23
proposal, which said *at least one coin* for the pass side: that was loose in the
same careless way as the sign-blindness, one notch smaller. Recorded because the
symmetry is the point — a bar that is strict in one direction and permissive in
the other is a bar with a thumb on it.

**Transcription corrected 2026-08-23.** BE first copied this line without
R-24's qualifier *"where clusters permit"*. That is not cosmetic: the only
evidence `STOP` has is `edge_l1_v1`, whose sample is **one UTC day**, and
`EDGE_LAYER1_PROTOCOL` states *"day-clustered are not computable and must not be
claimed"* while frozen V5 requires such a fit to report
`DAY_BLOCK_UNAVAILABLE`. Without the qualifier the gate declared an inference its
own evidence cannot produce. The ruling had it right; BE dropped it.

### 4.2 Why `on_verdict` presents rather than executes — carry this reasoning

**`STOP`'s owner is human because opportunity cost is not in any receipt.** That
is the form of the argument that survives, and it is sufficient.

> **The stronger form BE first wrote — *"no measurement can answer the ANY"* —
> is REFUTED by an artifact in `STOP`'s own precondition list.**
> `CANCEL_POLICY_PROTOCOL.md` builds `ww_v1`'s envelope events as
> *"parameter-free **maximal supersets** of the trigger family — no
> trigger-parameter choice can shape them"* (:56) and states the test *"is an
> **UPPER-bound test**, so failing it **kills the family** rather than merely
> disappointing it"* (:96). That is exactly a measurement answering an
> existential: bound the family from above, refute the bound. **§4.4 then relies
> on it** — `cancel_v1` is discharged because *"the falsifier was answered"* —
> so the strong form and §4.4 cannot both stand.
>
> **And the generalisation proved too much.** *"A gate whose question is broader
> than its metric may present and may not execute"* disqualifies every gate here:
> `G-FF1` asks "is the side convention right?" and measures 600 sampled
> transactions — yet its `on_fail` is `HALT_PROGRAM`. Under the general rule
> §3.3's *"blocking is reserved for `on_fail = HALT_PROGRAM`"* would have **zero
> instances**. Escalated to the coordinator, who authored the reasoning, as
> **BE-7**.

The narrow claim still does the work `STOP` needs: The receipts answer a different question:
*in the configurations we tested, no.* A statistical threshold will therefore
**always answer a narrower question than the gate asks**, and treating its output
as the decision is a **category error**.

**Opportunity cost is the deciding input, and no receipt contains it.** Whether
to stop a programme depends on what else the effort would buy — which is not in
this tape, this protocol, or any gate in this registry.

This is why the amended `on_verdict` assembles and schedules rather than halts.
It is also the general rule for kill-gates in this registry: a gate whose
question is broader than its metric may present, and may not execute.

### 4.3 What the amendment satisfied, so the record is auditable

R-6 admits a Class-D amendment only if it (a) precedes the run, (b) is motivated
by information that is not the result, and (c) invalidates every verdict computed
under the old bar. All three held, and **only** at that moment:

- **(a)** `STOP` had never been evaluated.
- **(b)** the motivation is BE's finding that the vocabulary is an **equivalence
  grammar that cannot express sign** — a defect in the **instrument**, not in an
  outcome, visible without looking at any result.
- **(c)** nothing to invalidate: no verdict existed under the old bar.

The old bar was sign-blind: btc `h=5` at **−0.532 ¢ [−0.797, −0.287]** satisfied
*"the interval must exclude zero on at least one verdict coin"*, so the evidence
that the maker is destroyed read `PASS` → *"proceed to the DE build"*.

### 4.4 `cancel_v1` is DISCHARGED AS MOOT — R-25

`cancel_v1` was listed as a precondition while its family was already closed,
which made `STOP` unevaluable: **a gate that cannot fire, fifth logged instance,
inside the gate that exists to end the programme.**

**Ruled: `cancel_v1` is discharged as moot by `ww_v1`'s `DEAD` verdict.** The
falsifier was answered; the §2 grid was never built and never needs to be. A
precondition satisfied by *"the question it guarded was closed upstream"* is
**discharged, not outstanding**.

**Recorded explicitly so nobody later reads the absent receipt as missing
evidence.** There is no `cancel_v1` receipt and there should not be one. An
implementer looking for it is looking for something the programme correctly did
not produce.

### 4.5 Consequence — preconditions are all in, and today's reading

With `cancel_v1` discharged, **`STOP`'s preconditions are ALL IN.**

> **`STOP`'s OWN METRIC HAS NEVER BEEN COMPUTED.** There is no `stop_*` receipt
> on disk — verified, zero. Everything below reads the **state of the
> preconditions**, not a verdict of this gate. `edge_l1_v1` is also not STOP's
> estimand: STOP's metric says *"net edge per fill, **after fee** … under a
> STATED cancellation policy"*, while `edge_l1_v1` subtracts no fee and its maker
> never cancels — and its own protocol says *"a positive Layer-1 markout … does
> not mean market making is profitable … do not combine the two layers into a
> single PnL figure."* Both receipts read here are stamped
> `RESEARCH_ONLY_NOT_DECISION_ELIGIBLE`.
>
> So §4.5 reports **preconditions**, and the sentence "today's evidence reads
> `FIRE_SIDE`" was BE overstating a precondition reading as a gate verdict.

**The reading is `FIRE_SIDE` at `h=5` and `INSUFFICIENT_EVIDENCE` at every other
horizon. It is NOT determinate, and BE reported it as determinate before
checking.** Applying the amended bar cell by cell to `edge_layer1_v1.json`:

| `h` | btc | eth | verdict |
|---:|---|---|---|
| **5** | `[−0.797, −0.287]` ✓ | `[−1.726, −0.759]` ✓ | **`FIRE_SIDE`** |
| 15 | `[−0.765, −0.178]` ✓ | `[−1.284, **+0.089**]` ✗ | `INSUFFICIENT_EVIDENCE` |
| 30 | `[−1.047, −0.216]` ✓ | `[−1.393, **+0.059**]` ✗ | `INSUFFICIENT_EVIDENCE` |
| 60 | `[−0.834, **+0.633**]` ✗ | `[−2.479, −0.807]` ✓ | `INSUFFICIENT_EVIDENCE` |

`FIRE_SIDE` holds at **one of four** horizons. The metric says *"a fixed short
horizon"* and does not say which, and `EDGE_LAYER1_PROTOCOL` warns in advance
that **selecting one after seeing the results is tuning**. Two further free
parameters sit in the same sentence: the Layer-1 estimand is **gross** (no fee
subtracted) while the metric says *"after fee"*, and its policy is a resting
two-sided quote that **never cancels** while the metric says *"under a STATED
cancellation policy"*.

**Escalated as BE-6.** Pinning `h`, the fee treatment and the cancellation policy
is a parameter choice on the user's gate, not BE's to make. Until they are pinned
by value with an R-6 class and an owner each, **`STOP` has no determinate
reading** and §4.5 must not be cited as though it does.

Per `on_verdict`, that **assembles the owner's decision; it does not execute
one.** The gate has done its job by producing a determinate reading and handing
it over.

**The disjunctive-precondition machinery of Revision 1 is now MOOT for `STOP`**
and is not applied: the precondition set resolved, so nothing needed relaxing.
The discipline it carried is retained for any future kill-gate — **a leg is the
protocol's own frozen verdict unit** (`ww_v1` family verdict, `layer2_v1` coin
verdict, `edge_l1_v1` coin verdict), never a cell, because a disjunction over
cells would fire what a frozen coin bar declined and that is bar-softening by
another route.

### 4.6 The honest reading of the inputs

Layer 1: a maker who never cancels loses on both verdict coins. `ww_v1`: no
cancellation policy on this tape recovers it, **DEAD 8/8 coin-days**, and the
refutation is robust across the **entire** upstream interval — the flip point is
btc `|markout|` **0.116 ¢** at `R`'s point estimate and **0.145 ¢** at `R`'s CI
upper, against a measured CI whose most favourable endpoint is **0.287 ¢**, i.e.
**2.0–2.5× beyond**. Layer 2: negative on 8/8 cells with the frozen bar reading
`UNDETERMINED`.

*Correction, recorded rather than silently swapped:* BE first published this
margin as **1.6×**, computed with `R = 0.219`. The coordinator verified it
independently using `R`'s point estimate (0.153) and `R`'s CI upper (0.184),
giving 2.47× and 1.98×. Same formula, correct inputs, stronger conclusion. BE
under-claimed, which is the right direction to err.

---

## 5. What the DAG is not

### 5.1 Not a runtime edge

`on_fail = HALT_PROGRAM` is a **programme-control path operated by a human
owner**, not a wire into `DE`. Architecture v7 listed EV among OP's health
sources, creating `EV → OP → DE`; that edge was removed deliberately and this
plan does not reintroduce it.

**Checkable, not merely argued:** `EV-Gates.produces` **excludes** `HealthEvent`;
its port manifest is `read_all` only (`contracts.yaml:1856`, `EV-*: [read_all]`);
`HALT_PROGRAM` is **not** a `HaltState` and has no `reset_authority`.

**Contract conflict, surfaced not resolved (coordinator's to route):** v22 gives
`EV-Markout`, `EV-Calibration` and `EV-Orchestrate` all
`produces: [… HealthEvent]`, while `PM_ARCHITECTURE` §1 says health sources are
**"NEVER EV"**. The types already contradict the rule for the EV plane.

### 5.2 Not a place where thresholds live

§7.

### 5.3 Admissibility is a predicate, not a node

Revision 0's spine drew `DA coverage / admissibility ──▶ every gate above`, which
satisfies its own §8 triviality falsifier on the page it is written on.
Admissibility is a **predicate on the evidence** (`admissible: false` + reason,
§2.3). The edge is deleted.

### 5.4 A SHA-256 is a change-detector, not a conformance checker

Carried from Revision 0 unchanged — committed code conformed to no frozen
protocol while the snapshot verified clean. `spec_hash` pins the protocol,
`code_hash` pins the implementation, and a **separate declared conformance test**
asserts the code implements it. Registering the hash alone does not satisfy this.

---

## 6. Registration — when, and what is checked

### 6.1 The trigger, which Revision 0 never defined

**Registration is bound to the FREEZE event.** A protocol is frozen and
registered in the same act; an unregistered gate's receipt is refused at ingest.
The `G1`/`G2` failure Revision 0 cites — sketched, marked "pre-register properly",
never done, while eleven experiments ran — is a **missing trigger**, not a
missing check.

### 6.2 Executable vacuity controls, not declared witnesses

Revision 0 required *"a concrete input under which it FAILS"*. That check is
**degenerate on all three real gates it can be tested against**: both instances
it cites had witnesses that were nameable and unconstructible, and R-14 had
already ruled it insufficient and required a minimum detectable effect.

**Replacement, which four protocols in this corpus already ship:**

1. **A passing control RUN**, not a prose witness — a synthetic fixture on which
   a vacuous probe demonstrably fails. Precedents: `CANCEL_POLICY_PROTOCOL` §1.5
   (synthetic episode `W = Δ` exactly, plus a shuffle control), `LAYER2_PROTOCOL`
   §4 (identity control, known-winner fixture, winner-shuffle),
   `EDGE_LAYER1_PROTOCOL` rule 6, `queue_and_type.py`'s 34 self-tests.
2. **An a-priori MDE at the declared `n`** — R-14 Amendment 2. `layer2_v1` shipped
   MDE ≈ 2.4 ¢/cell against a ~0.2 ¢ question, which is *why* it returned
   `UNDETERMINED` informatively rather than accidentally.
3. **Both witnesses named**: what makes it fail **and what makes it fire**.

A prose witness is unfalsifiable by construction; a control that must fail on a
vacuous probe is not.

### 6.3 `PASS` requires a bound, never a p-value

Carried from Revision 0, **conditioned on `bound_kind`** — because as written it
rejected the programme's only clean `PASS`:

- `TWO_SIDED_EQUIVALENCE` → `tolerance` and `ci_hi_abs` **finite and
  non-negative**
- `ONE_SIDED_SUPERIORITY` / `SIGN` → `threshold` and the relevant **signed** bound

"Non-null" is the wrong predicate: `ci_hi_abs = NaN` is non-null, and every
ordered comparison against NaN is false, so the same NaN yields `PASS` or
`MODEL_REFUTED` depending on which way the comparison is written. **NaN is
refused before the comparison, not by it.** The rule that survives intact:
*a `p_value`-only verdict is rejected.*

**Worked admission, so this check can be tested rather than believed.** The
programme's only clean `PASS` is `gff1_side_v3.json`, whose fields are
`{threshold: 0.99, wilson95: [0.9936, 1.0], agreement: 1.0, verdict: PASS}` —
**no `tolerance`, no `ci_hi_abs`**, because `GFF1_PROTOCOL` defines `PASS` as
*"Wilson **lower** bound ≥ 0.99"*. Under `bound_kind: ONE_SIDED_SUPERIORITY` with
`ci_arm: LOWER` that receipt is **ADMISSIBLE as written**, and any future
formulation of this check that rejects it is wrong by construction. Revision 0's
version rejected it; Revision 1's `bound_arm` rule rejected it again under a new
name. Third time it is pinned to a named artifact.

### 6.4 Population — by declared sampling UNIT

Revision 0 mandated `days_sampled`, which appears **nowhere** in v22 and would
reject the 210 sigma gates §8 promises to admit. It is also the wrong quantity
for Route A, whose day unit is the **OOS test day**, not the CLOB window slug —
two day universes under one field name.

**Rule:** the receipt declares its **sampling unit** and, where it has one, the
**enumerated sample in that unit** — `window_slug_day` for replay probes,
`oos_test_day` for Route A. The unit is named per protocol; there is no single
hardcoded field name, because `days_sampled` means CLOB window slugs and Route A
never touches the CLOB tape.

**Revision 1 made this a refusal and the refusal was wrong.** It said a receipt
without an enumeration is *refused* — which rejects `sigma_route_a_v1.json`, the
sole receipt for all **84** `route_a_v1` cells, and the same for the 126, i.e.
exactly the gates §10 promises to admit. Two sections of the same plan gave
opposite answers on the same artifacts. **Registration never refuses on
provenance.** It records a state, visibly:

| provenance state | meaning | admits? |
|---|---|---|
| `ENUMERATED` | unit declared, sample enumerated | yes |
| `MISSING` | pre-R-6 receipt, no provenance block | **yes, flagged** |
| `CONTRADICTED(by)` | a provenance block is present and **measurably wrong** | **yes, flagged loudest** |

`CONTRADICTED` is the state the 2026-08-23 failure actually needed and Revision 1
lacked: the defect was never *missing* provenance. `edge_layer1_v1.json` and
`skew_bound_v1.json` both carry `{source_days: [4 days], n_days: 4}` on a sample
§1f measures as **one** day — a **4× over-report** that any "mandatory non-empty"
check passes cleanly. A flag that only distinguishes present from absent cannot
see it.

Never silently exempt, and **never upgraded by editing a frozen protocol.**

### 6.5 Exhaustive outcome space

Promoted to a registration check from the programme's own META-RULE, added after
its second instance. A live third instance: `policy_v1`'s `TRADE_OFF_CONFIRMED`
requires `Δfill > 0` with CI excluding zero, which `action_fill` satisfies **by
algebraic identity** (`front = min(size, cum) ≥ back = min(size, max(0, cum −
queue_ahead))`). Registration must reject a branch that carries no evidence.

---

## 7. What `EV-Gates` does not own

- **Thresholds.** Frozen per protocol by the coordinator. EV-Gates records
  `bar_ref`, `frozen_at`, `spec_hash`; it never chooses or adjusts one.
  **And under R-20, a frozen bar's INPUTS are anchored by value too** — a bar
  that is a function of Class-C measured values silently moves when those are
  re-published, so the registry stores the input snapshot alongside the bar. The
  obligation is to *surface* a would-have-moved, symmetrically, in either
  direction; never to propagate it.
- **Mappings.** Declared per protocol (§2.4).
- **Running the tests.** The probes own their measurements.
- **Halting anything at runtime.** §5.1.
- **`STOP`'s verdict, and `STOP`'s bar.** The user's.

---

## 8. What would falsify this design

- **The four states are still too few.** If a protocol's verdict cannot be
  expressed without discarding something a reader needs, the grammar is wrong
  again. Revision 0 failed this on four words where it said one would be
  decisive; Revision 1 must be held to the same test.
- **Per-protocol mappings drift.** If two protocols map the same word to
  different states, the refusal-at-ingest rule has moved the inconsistency rather
  than removed it.
- **The DAG is trivial.** If in practice every gate's real preconditions are
  "all of DA", §5.3's predicate is doing all the work and the edge list is
  decoration.
- **Registration stays a formality.** If a vacuity control is ever satisfied by a
  fixture nobody could fail, §6.2 has degenerated into what it replaced.
- **`STOP` never becomes evaluable** even under disjunctive preconditions.

---

## 9. The contracts v23 delta this plan requires

Cross-plane contract changes are coordinator-gated, so this is a **proposal**.

> **✅ CHECKER RUN — 2026-08-23. Candidate v23 returns 0 invariant errors.**
>
> ```
> contract_check.invariants(v22 + §9 patch)   ->  0 errors
> contract_check.invariants(v22 baseline)     ->  0 errors
> ```
>
> **BE's previous banner said this run was blocked** — *"doing so requires
> drafting a candidate v23 `contracts.yaml`, which is the next step and is not
> done."* **That was false, and it is what let five errors through.** Merging both
> §9 blocks onto v22 and calling `contract_check.invariants()` is **twelve
> lines**. BE declared a check too expensive to run, having never tried it, in
> the banner whose entire subject is that hand-inspection is the defect. Second
> instance of that exact shape in one section — the first was claiming
> `--selftest` was "8/8 green" when it prints 14.
>
> **The five errors the run found, all now fixed:**
> `unresolved registry GateVerdictLedger` and `GateRegistry` — BE wrote
> `registry: true`, inventing a boolean where v22's `registry:` names a **type**;
> `duplicate declaration (local type AND prelude.external)` for **both** `GateId`
> and `Provenance` — the checker's only R-SSOT-named error, which iteration 2 had
> already named **by string**, shipped twice; and `EV-Gates consumes GateEvidence
> with no declared producer`.
>
> **Still not validated, and this is the honest boundary.** `invariants()` is one
> of three checks. The **structural diff** (`flatten`/`diff`) and the
> **migration records** are not done: the patch removes 12 `Gate` fields and
> type-changes ~10 more, every one of which needs a `migrations.yaml` record, and
> those must be derived from a `diff` run rather than by inspection. Command:
> `python3 live/pm_research/contracts/contract_check.py 2f6a156 WORKTREE`
> (`2f6a156` holds `version: 22`).

**Prelude removals — part of this patch.** `GateId` and `Provenance` are
declared in v22's `prelude.external`. Structuring them locally without removing
them there produces `duplicate declaration (local type AND prelude.external)` —
the checker's only R-SSOT-named error. Both removals need a `migrations.yaml`
record.

```yaml
prelude:
  external:
    remove: [GateId, Provenance]
```

**Notation:** this is a **PATCH**, not a replacement. Only the keys shown change;
every unlisted v22 field survives.

> **⚠ AMENDED 2026-08-23 — "only the keys shown change" WAS AMBIGUOUS FOR
> LIST-VALUED KEYS, and the ambiguity cost a real production.** DE's assembly read
> `produces: [GateEvidence]` as *replacing* the key's whole list and drafted a
> migration record authorising `BE-Uncertainty.produces:
> ['dict[InstrumentId, Known[Uncertain[PathLaw]] | Unavailable]'] → ['GateEvidence']`
> — **deleting the uncertainty module's entire path-law output.** That is not what
> this section asks for, and its own comment says so: *"already true in v22 via
> `ReducedFormFit.mean_gate/var_gate`; v22 simply never declared it."* The intent
> was always to **declare a production that already existed.**
>
> **BINDING RULE, so no assembler has to infer it again:** for the list-valued
> module attributes `produces` / `consumes` / `requires`, **a patch UNIONS with
> the v22 value.** Elements are added, never replaced, and **a removal must be
> written explicitly** as `- !remove <element>`. Nothing in §9 uses `!remove`, so
> **§9 removes no module element at all** and every module line here is
> **additive**.
>
> Spelled out so it needs no inference:
>
> ```yaml
> BE-Uncertainty:
>   produces:                                             # UNION, not replace
>   - dict[InstrumentId, Known[Uncertain[PathLaw]] | Unavailable]   # v22, SURVIVES
>   - GateEvidence                                        # declared, already true
> ```
>
> **Consequence for the batch: the 20th migration record must NOT be filed.** A
> migration record *authorises* a change; it does not *validate* one. Filing it
> would have used the migration mechanism to launder a defect past the checker —
> the checker would have gone green precisely because the record made the deletion
> legal. **The non-additive count stays at 19.** Unions are block-style single-quoted variants,
matching `CancelAllStatus` (`contracts.yaml:382`), because a flow sequence splits
on the commas inside parentheses.

```yaml
types:

  GateId:
    fields:
      protocol: str
      name: str
      version: int

  GateBar:
    kind: union
    variants:
    - 'Scalar(value: float, unit: str)'
    - 'TwoSided(interval: Interval)'
    - 'ExcludesZero(side: enum:POSITIVE|NEGATIVE)'
    - 'Conjunction(parts: list[GateBar])'
    - 'Disjunction(parts: list[GateBar])'
    - 'AtLeastK(k: int, parts: list[GateBar])'
    - 'PerScope(by_scope: dict[ScopeKey, GateBar])'

  PreconditionState:
    kind: union
    variants:
    - 'Passed()'
    - 'NotYetEvaluated(blocked_by: GateId)'
    - 'DischargedByRefutation(by: GateId)'
    - 'DischargedAsMoot(closed_by: GateId)'
    - 'Inadmissible(reason: str)'

  Precondition:
    fields:
      gate: GateId
      state: PreconditionState

  ProvenanceState:
    kind: union
    variants:
    - 'Enumerated(unit: str, sample: list[str])'
    - 'Missing()'
    - 'Contradicted(by: ImmutableId, claimed: str, measured: str)'

  VerdictState:
    kind: union
    variants:
    - 'Pass()'
    - 'ModelRefuted()'
    - 'InsufficientEvidence(resolvable_by: ResourceRequired)'
    - 'NotIdentifiable(reason: str, overturning_contrast: str)'
    - 'FireSide()'
    - 'PassSide()'

  ResourceRequired:
    fields:
      kind: enum:DAYS|WINDOWS|EVENTS|INSTRUMENT|SPECIFICATION
      quantity: float | NullPin
      at_declared_n: int | NullPin

  FailRoute:
    kind: union
    variants:
    - 'HaltProgram()'
    - 'Reroute(to: str, target: GateId | NullPin)'
    - 'BlockDownstream()'
    - 'ReportOnly()'
    - 'PublishConclusion(text: str)'

  ProvenancedValue:
    fields:
      value: float
      artifact_id: ImmutableId
      provenance: Provenance

  Provenance:
    kind: union
    variants:
    - 'Measured()'
    - 'Declared()'
    - 'Chosen()'
    - 'Assumed()'
    - 'Imputed()'
    notes: >-
      Carries ruling R-3, which is RULED and PENDING A CONTRACT EDIT. v22
      declares Provenance in prelude.external with NO members, so R-PROV is
      keyed on a value the contract never defines and no checker can evaluate
      it. This delta is the contract move R-3 was waiting for.

  Gate:
    fields:
      id: GateId
      status: enum:DRAFT|FROZEN|SUPERSEDED|RETIRED
      superseded_by: GateId?
      bar: GateBar
      bar_provenance: Provenance
      bar_input_snapshot: dict[str, ProvenancedValue]
      bound_kind_derived_from: bar
      null_hypothesis: str
      favourable_arm: enum:UPPER|LOWER
      preconditions: list[Precondition]
      on_fail: FailRoute
      on_verdict: FailRoute
      code_hash: Hash
      conformance_ref: ImmutableId
      vacuity_control_ref: ImmutableId
      firing_witness_ref: ImmutableId
      mde_at_declared_n: float | NullPin
      verdict_map: dict[str, VerdictState]
      outcome_space: list[str]
      review_date: Date

  GateEvidence:
    fields:
      gate: GateId
      scope: ScopeKey
      coin: str?
      per_fill: Interval | NullPin
      share_weighted: Interval | NullPin
      ci_arm: enum:UPPER|LOWER
      estimand_is_idealised: enum:UPPER|LOWER|NO
      decision_eligible: bool
      admissible: bool
      inadmissible_reason: str?
      provenance_state: ProvenanceState
      inference_actual: str
      evaluated_at: Timestamp
      code_hash: Hash
      status: enum:CURRENT|SUPERSEDED|WITHDRAWN|INVALIDATED
      supersedes: ImmutableId?
      withdrawn_reason: str?
      verdict: VerdictState?
      tolerance: float
      ci_hi_abs: float | Unavailable
      effect_size: float
      test: str
      conditioning: str
      multiplicity: str
      sampling_unit: str | NullPin
      p_value: float | NullPin

  GateVerdictLedger:
    fields:
      entries: dict[GateId, list[GateEvidence]]

  GateRegistry:
    fields:
      entries: dict[GateId, Gate]

  ModelSelectionOutcome:
    fields:
      layer_ref: ImmutableId
      decision: enum:RETAIN|DELETE|CENSORED|UNRESOLVED
      evidence: ImmutableId
      spec_hash: Hash
```

**Module records** — hoisted out of the type namespace, because in Revision 2
`modules:` sat among the type names and would have been flattened as
`type:modules`, so no module record was ever created and §5.1's "checkable"
claim had nothing to check:

```yaml
modules:
  BE-Uncertainty:
    produces:
    - GateEvidence      # already true in v22 via ReducedFormFit.mean_gate/var_gate;
                        # v22 simply never declared it, which is why the checker
                        # reports EV-Gates consuming a type nothing produces

  EV-Gates:
    consumes:
    - GateEvidence
    - SideConventionEvidence
    produces:
    - Gate
    - GateRegistry
    - GateVerdictLedger
    ports:
    - read_all
    - artifact_resolver
```

### 9.1 What this fixes, and what is still open

**Fixed here**, each closing a named iteration-2 or iteration-3 finding:

- **Parameterised states are real unions**, not enums with loose siblings:
  `Contradicted(by, claimed, measured)` — the payload IS the 4× over-report and
  Revision 2's flat enum could not hold it; `NotIdentifiable(reason,
  overturning_contrast)` — a non-identification must name what would refute it,
  because *"terminal mechanism is unidentifiable"* is in the **Withdrawn**
  table; `InsufficientEvidence(resolvable_by)` carries a resource record, since
  `LAYER2_PROTOCOL` §3 answers CALENDAR **and** SAMPLE in two paragraphs.
- **`gff1_side_v3.json` is constructible.** `tolerance`, `ci_hi_abs`,
  `effect_size`, `test`, `conditioning`, `multiplicity` all take `NullPin`
  (R-NULL, the corpus's own mechanism). Revision 2 fixed the prose and left the
  **type** rejecting it — the fourth rejection of the same receipt.
- **`sigma_route_a_v1.json` is constructible**: `sampling_unit: str | NullPin`
  and `provenance_state: Missing()`.
- **`bound_kind` is deleted** — it duplicated the `GateBar` variant tag, the
  `estimand_route` defect v16 removed. **`ci_hi_abs` is kept only as the
  equivalence arm** beside `per_fill`/`share_weighted` intervals, which also
  gives R-DUAL both arms on one record instead of a discriminator that lets one
  substitute for the other.
- **`favourable_arm` moved to `Gate`**, beside `null_hypothesis` — on
  `GateEvidence` it gave `route_a_v1` 84 chances to contradict its own gate.
- **`FIRE_SIDE`/`PASS_SIDE` are in the vocabulary**, so `STOP` no longer needs a
  second one, and `on_verdict` is declared.
- **`Gate.version` deleted** — `GateId.version` is the single counter; two was
  R-10's own defect inside the field added to satisfy R-10.

### 9.2 Reconciliation with the ONLY executing gate logic — resolved under R-33

`live/pm_research/evaluation_pipeline.py:221-223` is the **one place in this repo
where a gate verdict gates anything**: it hard-codes
`if result.verdict != "PASS": raise`, an `n_validated_tx < 500` floor and a
`wilson_lo < threshold` comparison, and it blocks Tier-2 derivation on `G-FF1`.
A real DAG edge, implemented outside any registry, which four iterations of this
plan never mentioned.

**BE filed this as a question (`Q-BE-6`) and then withdrew it under R-33**,
because it was a gate that returns its input. The running code is not in conflict
with any frozen document; **this plan was in conflict with working code**, and
the plan is BE's to fix. So:

- **The registry does not replace that check and does not require it to change.**
  It is the reference implementation of `ONE_SIDED_SUPERIORITY` plus a `VOID`
  floor, written before the vocabulary existed and consistent with it.
- **`gff1_side_v3.json` is a `SideConventionEvidence`, not a `GateEvidence`** —
  its fields match `contracts.yaml:607-616` field for field, and §9's module
  block already lists that type in `EV-Gates.consumes`. It was never rejected by
  `GateEvidence` **because `GateEvidence` never described it**, which means the
  six-field widening BE made to admit it was a **category error**.
- **That widening is REVERTED.** Five of the six fields (`test`, `conditioning`,
  `multiplicity`, `effect_size`, `tolerance`) are **non-null in 84/84 `route_a`
  cells**, so no receipt on disk needed them. `GateEvidence` is
  `ReducedFormFit.mean_gate`/`var_gate` — **BE had loosened the PRICING gate to
  fix a problem that did not exist**, re-opening the v15 null-comparison defect
  its own note exists to close. Only `ci_hi_abs` keeps a widened form, and as
  `float | Unavailable` rather than `NullPin`, because route_a's null is a
  **refusal to compute** (`bootstrap_draws: 0`), which is what `Unavailable`
  means and is not an assumption.
- **Any future `EV-Gates` implementation must reproduce that check's behaviour
  on `G-FF1`, not supersede it** — and a `FIRE_SIDE`/`PASS_SIDE` string reaching
  it would `raise`, which is correct: those are roll-up verdicts and this edge
  reads a cell verdict.

**Still open and NOT fixed here** — recorded rather than papered over:

- **The BE-reads-EV inversion.** `ReducedFormFit.mean_gate/var_gate:
  GateEvidence` means `BE-Uncertainty` must emit a `gate: GateId` that only
  exists once EV registered it, and `PM_ARCHITECTURE` §1 says *"EV reads all
  planes and is read by none."* The fix direction — split the in-band record
  from a registry-facing wrapper — is a larger change than this section should
  make unreviewed.
- **`note:GateEvidence`** is normative, is now false (it describes three states
  and tolerance-only `PASS`), and needs replacement text.
- **`SideConventionEvidence.verdict`** stays a third, three-state enum at
  EV-Gates' own input port.
- **The migration records** are not written. They must be **derived from a
  checker run**, not from inspection — which is the whole lesson — and the run
  needs a candidate v23 file that does not yet exist.

## 10. What this plan deliberately does not do

- Does not retro-register the 210 sigma gates by hand — they enter by
  `spec_hash` with `provenance: MISSING` flagged (§6.4).
- Does not define `EV-Attribution`; deferred until a book exists.
- Does not re-open any frozen threshold, and does not move `STOP`'s bar.
- Does not author the per-protocol verdict maps. Workers declare them; EV-Gates
  refuses receipts that lack one.
