# SP plane — model plan: Venue · Instrument · Strategy · Scenarios · Params

Plan only. No code, no measurement, no fitted quantity. Status: **DESIGN**, not
decision-eligible. **Revision 20** — reviewed and revised by the **DA** plane on
ownership transfer under ruling **R-18**; review record in
[`SP_PLAN_REVIEW_LOOP.md`](../SP_PLAN_REVIEW_LOOP.md). Revision 3 applied
**R-20**; **Revision 4 repairs iteration 2's findings, most of which Rev 2/3
introduced** — recorded in the loop file, not here.

Revision 1 was written by the coordinator as build item B1 because the SP plane
had no owning session. R-18 records that as the wrong fix and transfers the
document to DA, which owns the measured facts that populate `SP-Venue` and
`SP-Instrument`. Rev 1 was single-pass and unreviewed; iteration 1 returned **60 findings, 31
MUST-FIX class**, across three independent lenses that converged on five of them
— so this revision is substantial rather than cosmetic.

> For current programme state read [`FLOW_MODEL_STATE.md`](../FLOW_MODEL_STATE.md).
> For types, `contracts/contracts.yaml` v22 wins. For rules,
> `PM_ARCHITECTURE.md` wins. This document is a plan.

**Ownership, precisely.** The **design** of this plane is DA's. The **values**
in `CHOSEN` rows are the coordinator's under R-6 Class A/B. This document may
not set, move or re-tune a `CHOSEN` value, and it does not. Where the review
found the **Class A/B/C/D taxonomy itself** defective, that is a decision rule
and therefore the coordinator's to re-cut — those findings are escalated in
**§10**, not fixed here.

---

## 0. The decision in one line

**`SP` is the plane where CHOICES live, as distinct from measurements.** A risk
limit has no truth value to measure; a fee schedule has nothing to choose. That
distinction is the plane's reason to exist and it survived review.

What did **not** survive is the inference Rev 1 drew from it — that the plane is
therefore coordinator-owned. Authoring a plane's design is worker work whoever
the plane serves; only the choices are the coordinator's. R-18.

## 1. Why the plane needed a plan

`SP` sits at the root of `SP ← DA ← BE ← DE`, and it was the only plane with no
plan while **five documents name `SP-Params` as the owner of a number**. An
ownerless quantity at programme scale is the defect class the architecture
review has been killing one type at a time.

**What this document unblocks, stated accurately.** Rev 1 claimed to unblock
`DE-ActionSpace` and `DE-Constraints` (build item B2). Only the second is true.
`DE_MODULE_PLAN.md` §1.3 **used to list** — and, as of `DE_MODULE_PLAN.md:94-98`, no longer lists `min_size`, which it now records as MEASURED — ActionSpace's blocker as **venue mechanics
facts** — minimum order size against the 5-share quote, whether resting
`ASK_UP` requires holding Up tokens, time-in-force support, whether `CANCEL` is
free and unthrottled, self-match handling, the disposition of orders resting at
resolution, settlement latency. **`min_size` IS knowable and is measured — see §4**; the rest are not knowable passively. They are `SP-Venue` **fields** — listed in §4 for the consumer index but **not register entries**, since the variant is not expressible on `ParamValue` — which is what unblocks a *builder*; they
do not become facts by being listed.

## 2. The five records, and the line between them

Each of `SP-Venue`, `SP-Instrument`, `SP-Strategy` and `SP-Scenarios` is a
`SpecRecord`: bitemporal (`observed_at` + validity interval), with provenance
**per field**, because records reconciled from different authorities cannot
carry one record-level source.

**`SP-Params` is NOT a `SpecRecord`** and Rev 1 was wrong to say it was.
`contracts.yaml` types it as `entries: dict[(ParamId, ScopeKey), ParamValue]`,
and `ParamValue` carries `valid_for: ScopeKey` — a *scope*, not a time interval.
It has no `valid_from`/`valid_to`. **So R-VERSION is unsatisfiable for every row
in §4 as the contract stands**: moving `kappa_usd` overwrites the old value with
nothing recording that the old one held until then, and every replay pinned by
`DecisionProblem.spec_snapshot` before that date silently resolves the new one.
Listed as a required contract change in §7.

| record | owns | character |
|---|---|---|
| `SP-Venue` | matching, `tick_rule`, `min_size`, rate limits, `fee_schedule`, `rebate_schedule`, capability flags | **facts about the venue** — measured, documented, or **`Unknown`** |
| `SP-Instrument` | `settlement{…}`, `T`, payoff, `complement` — **`incentive_contract` RESTORED at iteration 11 (§10.33, self-resolved under R-33).** Rev 14 removed it here; `PM_ARCHITECTURE.md` — which this plan's header says **wins on rules** — never did, and `SP-Instrument` is a `SpecRecord`, so this field is the only place `FieldState.Disputed` can live (`contracts.yaml:1268-1272`). Removing it is what made §10.20's *"the rewards band sits on no `SpecRecord`"* true. Restored; §10.20 and §10.33 both close. The architecture's Records section carries `SP-Instrument{ settlement{…}, T, payoff, complement, incentive_contract }` with the note *"the resolved band/rate/eligibility live in `SP-Instrument.incentive_contract`"*. This plan's own header says **"for rules, `PM_ARCHITECTURE.md` wins"** — so the architecture already rules the one-carrier question the other way, and §10.20 currently asks the coordinator to spend a ruling restoring a field only SP ever removed. Iteration 3's named harm: an escalation naming a hole that does not exist. *(Tail of this cell is the pre-restoration note, retained for the record:)* Removed from this cell**: §2 rules one carrier (`IncentiveModel.contract_spec`) and this table contradicted it | **facts about the contract**, instrument-scoped and time-varying |
| `SP-Strategy` | `utility`, `solver`, `coupling`, `constraints`, `action_space`, `impls`, `nulls`, **`LossFunctional`**, **`unwrap_policy`**, **`unavailable_policy`** | **structural choices** — which plugins, which shape |
| `SP-Scenarios` | `ScenarioId -> AdverseScenario{outcome_map, scope, provenance, completeness, on_incomplete}` | **declared adverse worlds** — a modelling choice |
| `SP-Params` | `(ParamId, ScopeKey) -> ParamValue` | **scalars**, whatever their character — see §4 |

**None of the four record types exists in v22.** They appear only in
`PM_ARCHITECTURE.md` prose, which that file's own rule forbids from introducing
a type. §7 lists them as required additions; without them a builder writing
`SP-Venue{…}` has no canonical definition and the contract checker cannot
detect a later narrowing.

**The fee family, corrected.** Rev 1 wrote
`fee_schedule = (PIECEWISE_MINPQ, {rate, size_rounding})`. That is wrong twice. **AND IT IS STILL STANDING IN `PM_ARCHITECTURE.md` (§10.34)** — its Declarative-rules section carries that exact string verbatim, and §2 of this plan says the four record types *"appear only in `PM_ARCHITECTURE.md` prose"*, so **that string IS the definition a builder reads**. SP corrected its own copy and attributed the error to Rev 1 rather than to the source, leaving a **2× cost error** (3.50 ¢/share against a measured 1.75) in the file it says wins on rules — which would turn the measured ~225 bps crossing cost into ~400 bps. Not DA's file to edit; routed.
`MINPQ` reads as `min(p, 1−p)`, which at `rate = 0.07`, `p = 0.5` yields
**3.50 ¢/share against a measured 1.75 ¢** — double, turning the measured
~225 bps crossing cost into ~400 bps. The measured schedule is the *product*
`0.07·p(1−p)`. And the declared params have no slot for **incidence**, so the
taker-only fact cannot be stored and a maker leg computes a non-zero fee against
744/754 observed zero legs. The family is `(PRODUCT_PQ, {rate, incidence,
size_rounding})`; no fee family is defined in v22, so it too is a §7 item.

**Disputed needs a named handler.** The rewards band is the live `Disputed`
case — Gamma and the CLOB registry disagree, both present in `markets.jsonl`.
Rev 1 cited it as motivation and then left it with **three** claimed carriers
(`SP-Instrument.incentive_contract`, an `SP-Params.rewards_band` precedent in
`BE_BELIEF_PLAN`, and `IncentiveModel.contract_spec`, a registered plugin).
A `Disputed` `FieldState` can only be written on one record; write it on the
wrong one and a reader sees `Resolved` with no sign the authorities disagree.
**SUPERSEDED AT ITERATION 12 — DO NOT BUILD FROM THE NEXT TWO PARAGRAPHS.**
The carrier is **`SP-Instrument.fields["incentive_contract"]`**, restored in §2's
record table, which is a `SpecRecord` field and therefore the only place a
`FieldState.Disputed` can be written. `IncentiveModel.contract_spec` returns
`Known[IncentiveContract] | Unavailable` with **no `Disputed` arm**, so building
to it forces a **silent selection** between Gamma and the CLOB registry — the
wiring error the paragraph below names, delivered by the paragraph below.
§10.20/§10.33 are closed and Q-DA-20 is withdrawn. *Superseded text follows.*

**One carrier: `IncentiveModel.contract_spec`**, with `SP-Params` holding the
dispute *handler* (§4) — because a decision consuming a `Disputed` field must
name a declared handler, and silent selection is a wiring error.

*(ALSO SUPERSEDED AT ITERATION 12: limb (iii) below — "§2's own record table
removed `incentive_contract`" — is **no longer true**; the field was restored at
iteration 11, which is what dissolved §10.20. Limbs (i) and (ii) still hold of
`IncentiveModel.contract_spec` and are why that carrier is not used.)*

**THAT RESOLUTION CANNOT BE BUILT AS WRITTEN, and it fails in exactly the way
the paragraph above names (§10.20, Q-DA-20).** Three facts, each read off the
contract rather than inferred: (i) `contract_spec` is
`(InstrumentId) -> Known[IncentiveContract] | Unavailable`
(`contracts.yaml:457`) — **there is no `Disputed` arm**; (ii) `FieldState` occurs
in v22 **only** inside `SpecRecord.fields` (`contracts.yaml:1268-1269`), so a
`Disputed` `FieldState` can only be written on a `SpecRecord`; (iii) §2's own
record table **removed** `incentive_contract` from `SP-Instrument` in favour of
this carrier. Net: the rewards band sits on **no `SpecRecord`**, therefore has
**no `FieldState`**, therefore **cannot be `Disputed` anywhere** — the resolver
must return `Known` (a silent selection between two disagreeing authorities,
which this paragraph calls a wiring error) or `Unavailable` (which halts on a
band that is merely disputed, not missing). **A reader then sees exactly what
sentence three warns about: a resolved value with no sign the authorities
disagree.** DA does not pick between adding a `Disputed` arm to `contract_spec`,
restoring an `SP-Instrument` field, or giving `IncentiveContract` its own
`SpecRecord` — all three are contract changes. Also unspecified: the *type* of
the dispute handler (`SourceId`? enum? `PluginRef`?).

## 3. R-PROV is unenforceable — diagnosis confirmed, resolution corrected

**The diagnosis stands, and was verified rather than taken on trust.**
`Provenance` is a v22 prelude `external` primitive with no enumerated members,
and `R-PROV` has a `body` and **no `checks:` list**, unlike every enforceable
rule. A rule keyed on values the contract never defines cannot fire. *Fourth
logged instance.*

**Why careless enumeration is worse than silence** — unchanged from Rev 1, and
correct: `κ_$` is a **choice** with no fact of the matter that MUST gate;
`D̂(source)` is an **assumption** standing in for an unknown fact that must NOT.
Collapse them and either every hard risk limit becomes illegal to enforce, or
`R-PROV` goes vacuous.

**Rev 1's proposed resolution had three defects, all found in review:**

1. **No member fits an observed value.** `provenance` is carried by *fifteen*
   non-parameter types. A `FlowArrival` read off the wire is not estimated with
   a fit window, not from documentation, has a truth value, is not a stand-in,
   and is directly observed — so it is unlabellable, or mislabelled `MEASURED`
   and thereby made decision-gating with no artifact to point at. **An
   `OBSERVED` member is required.**
2. **Axis collision.** `Known` already carries
   `t_known_prov: OBSERVED|IMPUTED|ASSUMED` on the *knowledge-time* axis, and
   R-IMPUTE keys three checks on those words. Reusing `ASSUMED`/`IMPUTED` on the
   *value* axis leaves a reader unable to tell which axis fired.
3. **The clause "with teeth" has no teeth.** "`CHOSEN` may gate and nothing but
   the coordinator may set it" is keyed on `ParamValue.owner: ModuleId`, and
   **the coordinator is not a module.** The field is either invalid or filled
   with a downstream module, at which point the check passes vacuously — the
   *gate that cannot fire* defect reproduced inside its own fix.

**DA PROPOSAL — supersedes R-3's enumeration, held in the R-35 contract batch, NOT ratified** (R-3 ruled a *five*-member enum, no `OBSERVED`, no authority axis): enumerate
`OBSERVED · MEASURED · DECLARED · CHOSEN · ASSUMED · IMPUTED`, distinguish the
two axes by name, add an authority axis (e.g.
`ParamValue.set_by: COORDINATOR | MODULE(ModuleId)`), and state the intended
member for each of the fifteen carriers — not `ParamValue` alone. §7.

## 4. The parameter register

**Every non-CHOSEN row names both a publisher and an adopter.** Rev 1 spelled
that out for one row and left the others with a single name or a dash, which is
the ownerless-quantity defect §1 opens by naming.

**Character and class are different axes and are recorded separately.** The
*character* column is the `Provenance` axis that `R-PROV` keys on (§3); the R-6
*class* is who may move the value. Rev 3 overwrote character with class on five
rows, which both erased the `R-PROV` axis and made those rows read as
"non-CHOSEN", silently breaching the publisher/adopter rule below. Class now
lives in the status column beside its ruling.

**Rows carry the ruling that last moved them.** A row with no ruling stamp is
unreviewed — Rev 1's `tau_ladder` row stayed stale precisely because R-8's split
landed in three other files and this one had nowhere to record it.

**How a row becomes a key — DA design, no ruling required, and it was missing
until Rev 13.** `ParamId.namespace` is **mandatory** (`contracts.yaml:247-249`,
`namespace: ModuleOrPluginId`, not optional), so until this paragraph existed
**no row in this table was writable as a key** and two implementers would have
built two disjoint key spaces. The convention:

- **REGISTER ENTRY OR RECORD FIELD — decided by the CHARACTER column, which is
  where the marking actually lives.** Rev 13 said "status column" and no row's
  status column carries the marking, so read literally every row including
  `min_size` defaulted to a register entry. A row is a **record field, not a
  register entry**, when its character cell names a record: `min_size`,
  `matching/T/payoff/complement`, `settlement_latency`, `fee_schedule` and
  `rebate_schedule`, `tick_rule`, the `settlement` spec, and rate limits /
  capability flags / resolution disposition. Those carry a `FieldState` on a
  `SpecRecord` and have **no `ParamId` at all**. This matters more than
  bookkeeping: `ParamValue.value` is `any` and would **swallow an `Unknown`**
  that a `FieldState` expresses, so routing an absent venue fact through the
  register makes it read as a present one — the defect the rate-limits row
  exists to name.
- **`namespace = "SP-Params"`** for every row that is a register entry. **Caveat
  (§10.21):** `ParamId.namespace` is `ModuleOrPluginId`; v22's `modules:` block
  registers 25 ids and **none is `SP-*`** — `SP-Params` is a *type*
  (`contracts.yaml:422`), not a module. `ModuleOrPluginId` is `external` so
  nothing rejects the string and the key space is at least consistent, but the
  field exists to name an owning module and this one names none. §7 item 1 asks
  for "an SP spec-resolver module/port" without giving it an id; that id and
  this namespace should be the same string, and neither is chosen here.
- **`name`** is the key-column string with backticks and any parenthetical
  alias stripped: `` `kappa_usd (κ_$)` `` → `kappa_usd`;
  `` `tau_ladder_rungs {0,50,100,500}` `` → `tau_ladder_rungs` (the braces are
  the VALUE, not the name); `` `tau_decision_rung = 250 ms` `` →
  `tau_decision_rung`. `ScenarioLossLimit(<scenario_id>)` is the one row keyed
  by a **contract type** (`ScenarioLossLimit`, `contracts.yaml:426` — Rev 13
  cited `:436`, which is `ScenarioLossConstraint`) rather than a name; it is
  `name = "ScenarioLossLimit"` scoped per scenario.
- **ONE ROW IS ONE `name`, so a row carrying several parameters is not yet
  keyable (§10.22).** Three do: `belief_a`/`belief_b` (two names or one pair?),
  STOP's `h` / fee treatment / cancellation policy (**three**, and the last two
  are prose from which no name is derivable), and the expected-port registry ·
  `LANE_PROGRESS` grace · `period` bundle. This is the shape iteration 2 fixed
  on `tau_ladder`, where two quantities shared one key and were therefore one
  entry. Splitting them assigns names and classes, so it is escalated rather
  than done here. **`period` is registered TWICE** — its own row and inside the
  bundle — which under this naming rule yields two different names for one
  quantity with nothing asserting their equality; §6's strictest-alias rule has
  no two names to bind until the split is ruled.
- **`ScopeKey` is written at the BROADEST key that is true of the value, and
  subset-order resolution supplies the default.** Rev 13 got this backwards: it
  required "exactly the fields the scope column names and no others", which
  forbids `ScopeKey{}` and therefore demands one write **per 5-minute market**
  (≈2,000/day) for nine instrument-scoped rows whose values are constant across
  every market — and contradicts §9's "the register grows when a consumer
  demands a row". `ScopeKey` resolves in **subset order**
  (`contracts.yaml:261`), so an entry at `ScopeKey{}` already answers every
  `{instrument: I}` query: **that is the contract's own defaulting mechanism and
  the convention should use it, not ban it.** The scope column names the axis on
  which a value **may** vary, not a requirement to enumerate that axis. So
  `quote_size_pin` is written once at `ScopeKey{}`. **DO NOT NARROW IT — §5a
  clause (g) / §10.23.** Rev 15 ended this sentence "and narrowed only if some
  market ever needs a different pin", which licences — in the section a builder
  actually reads — the exact write clause (g) forbids, on the exact value the
  exploit uses: adding `(quote_size_pin, ScopeKey{instrument: I})` shadows the
  Class-D entry with no clause firing. **DA is self-bound against it**, and any
  narrowing of a published Class-C or Class-D row needs a ruling first. §6's
  strictest-alias rule does **not** cover this — it binds two *names*, not two
  *keys of one name*. The genuine constraint is the one *equally-
  specific matches FAIL LOUD* creates: never write the same `name` at two keys
  of equal specificity. `ScenarioLossLimit` is the exception — §2 pins it to
  `ScopeKey.scenario` **only**, so it is enumerated per scenario by rule, not by
  this convention.

**FIVE rows cannot be keyed at all, for two different reasons, both escalated rather than guessed (§10.17 for scope, §10.22 for multi-name).** *(Corrected at iteration 13. Iteration 12 found the count saying FIVE against a body naming FOUR, added the missing row, and incremented the header to SIX — preserving the off-by-one it was fixing, the second consecutive iteration to repair this figure and leave it wrong. The body names FIVE on the scope axis: `staleness_deadline` and per-port `period` (port), plus `refuse_k`, `knowledge_lag` and `our_feed_lag` (feed). §10.22's multi-name rows are a SEPARATE count, not added here. The omitted row was `our_feed_lag` — the row **R-55 itself created** — which is `feed`-scoped and therefore blocked by the identical finding. Q-DA-28 asks the coordinator to restate this count under one rule for `external` opaque ids; restating it from the old enumeration would have left the newest row unwritable after the ruling meant to clear it.)*
*Per port:* `staleness_deadline` and per-port `period` are scoped per port,
`port` is a `TelemetryPortId`, and **`ScopeKey` has no port field** — its eight
members are `venue · factor · instrument · horizon · feed · region · portfolio ·
scenario` (`contracts.yaml:251-260`). *Per feed:* worse, and Rev 13 missed it —
**no `FeedId` VALUE is named anywhere in this programme.** `FeedId` is an
`external` opaque id; this plan, `PM_ARCHITECTURE.md` and `OP_PLANE_PLAN.md`
name the type and never an instance. So `refuse_k` (**Class D — GUARD**) and
`knowledge_lag` (**Class D — VERDICT**) — two of the most restricted rows in the
register — **cannot be written either**, and that is a different defect from the
port rows. **`our_feed_lag` (R-55) makes THREE on this axis** — and its scope is
wrong besides: R-49 measured btc p90 at **282-342 ms against 67-92 ms**
elsewhere, so one `feed`-keyed value understates the **verdict coin** roughly
fourfold, in the optimistic direction. Scope is DA design, but the row cannot be
written at all until the `FeedId` question is ruled, so both wait on Q-DA-21. Note also that §10.17's candidate resolution *"`TelemetryPortId` is a
`FeedId`"* would make `ScopeKey.feed` mean **port** on the OPS rows and **data
feed** on these two: one member, two meanings, under resolution that FAILS LOUD
on equally-specific matches. All contract surface; none of it DA's to decide.

| ParamId | scope | character | publisher → adopter | status / ruling |
|---|---|---|---|---|
| `total_capital` | portfolio | CHOSEN | — → coordinator | **operative set §5**. Money, notional only |
| `quote_size_pin` | instrument | CHOSEN (was B) | — → coordinator | **CLASS D — VERDICT.** **the ONLY size row** (§6). Every Class-D verdict in the programme is conditioned on it · R-20 generalisation |
| `kappa_usd` (`κ_$`) | instrument | CHOSEN | — → coordinator | **operative set §5** |
| `ScenarioLossLimit(s)` | **scenario** | CHOSEN | — → coordinator | **RESOLVED — R-35.** Scenario-scoped; the architecture's §8 keys the cap per scenario and evaluates per-scenario loss directly, so signed loadings cannot cancel hard exposure. Rev 1's "$200 portfolio-wide" was the coordinator's own shorthand contradicting the register in the same document, and `de_constraints.py` implemented the shorthand. **With one declared scenario the arithmetic is unchanged today; the STRUCTURE fails when a second scenario exists and a portfolio ceiling silently permits a per-scenario breach.** Superseded text: Registered scenario-scoped, but §5's own binding arithmetic and the shipped `de_constraints.py` both aggregate one $200 ceiling **across markets** — a portfolio cap. `ScopeKey.scenario` has no wildcard either, so `{scenario: *}` is not writable and omitting it *is* the portfolio key this row forbids |
| `gamma_ladder` | portfolio | CHOSEN, diagnostic only | — → coordinator | **NOT supplied via `PluginRef.config`** — §9 adopts DE's `utility_none` registration and keeps the ladder a **reporting device, not a config value**; the `PluginRef.config` wiring is the exact routing DE's n-ary check exists to reject, and the shipped set holds it as a plain member (`ev_replay.py:51`). Diagnostic **only while no verdict rests on one rung** — nothing detects the moment one does (§10.18) |
| `refuse_k` | feed | CHOSEN | — (frozen at 1.0) | **CLASS D — GUARD.** **R-20.** The no-peek coefficient of R-REFUSE. At 0 the guard collapses to `t_known ≤ now` |
| `knowledge_lag` | feed | CHOSEN | — → coordinator (a declared convention; **no DA cadence measurement stands behind it**) | **CLASS D — VERDICT.** 250 ms. **`ww_v1` turns on it**: `R(τ)` counts fills with `W > lag + τ` · R-20 |
| `primary_horizon` (`h`) | instrument | CHOSEN | — → coordinator | **CLASS D — VERDICT.** `h=5` is the verdict rung. **8/8 negative in point estimate, 5/8 with the interval excluding zero.** **Read the second number only:** the verdict is `R(250,h) < f*_low(h)` and `f*_low` reads the CI ENDPOINT, never the point estimate — so "8/8 negative" is satisfied at eth h=15 (point estimate -0.586) precisely where the verdict flips, because its CI spans zero. The robustness that matters is the 5/8, not the 8/8. **THREE cells span zero — btc h=60, eth h=15, eth h=30** — and the **specific** measured defence — 1,611 btc fills discarded, terminal minute invisible by construction — is **btc h=60 only**; R-35 separately notes *all* longer horizons are population-shifted, a general caveat rather than a measured defence. DE's R-34 adjudication assessed the eth route independently and called it **real**, which is why §10.9's live route runs through them · R-8 (falsifier), not R-20 |
| `queue_bound_arm` | instrument | CHOSEN | — → coordinator | **CLASS D — VERDICT.** **`{FRONT, BACK_DISPLAYED}`** (`bounds.join` is a RECEIPT KEY whose arm is `BACK_DISPLAYED`). The **frozen** R-1 bar uses **one** arm, `BACK_DISPLAYED`; "requires BOTH" belongs to the *proposed* `cancel_v1` bar in a family R-11 closed. the arm sets **which bound the upper-bound test reads**; R-1 froze `BACK_DISPLAYED` as the generous arm. **`f*_low` on the FRONT population is UNMEASURED, so no FRONT verdict exists.** The frozen bar is calibrated on `JOIN_BBO` (`edge_layer1_v1.json`, btc n=10,294; 10,387 is that receipt's JOIN arm in `warning_window_v1.json`); the FRONT `R` figures come from a 5.4x larger, structurally different population (n=56,053) with no Layer-1 markout of its own. Rev 5 claimed "both arms are DEAD" by comparing FRONT's `R` against JOIN's bar — a statistic against a bar from another population. Withdrawn |
| `our_feed_lag` | feed | **MEASURED** | OPS publishes → coordinator adopts → DA writes the entry (R-37) | **NEW ROW, created by R-55 (iteration 10).** R-55: *"`our_feed_lag` becomes a NEW Class-C row — genuinely measured, 18.9 M rows, 8 coin-days"*, after OPS **refused** to have a composite of measurement-plus-assumption recorded as Class C. **CLASS C — adopt, never choose.** Distinct from `tau_operative`, which stays UNMEASURED until an Actuator exists; this is a **lower bound** on achievable τ, not τ. Rev 15 carried no row for a value a ruling had created — the ownerless-quantity defect §1 opens by naming, arriving through the ledger rather than through a deletion |
| `collector_era` | venue | MEASURED | DA publishes → coordinator adopts | **CLASS-D CONSEQUENCE.** Denominator of the Layer-2 proportion bar ("75 % of era days, min 4"). An era change must state which bars it makes un-evaluable · R-20 |
| `tau_ladder_rungs` `{0,50,100,500}` | venue | CHOSEN | — → coordinator | resolution only. **CONTRADICTED — ESCALATED §10.29 / Q-DA-32 (iteration 10).**
`WW_EBX_PROTOCOL.md` §1, **FROZEN under R-51**, reads *"the verdict rungs are
`τ = 500 ms` (primary) and `1000 ms` (beside)"*, and R-54 issued `DEAD_4CH` on
**8/8 cells at those rungs**. So a verdict now turns on 500 while this row says
none does and leaves it in a freely-movable CHOSEN set. **R-8's own
generalisation** — a row a documented falsifier turns on is Class D *whatever its
nominal class* — makes 500 Class D. The loop's founding defect one rung up, and
SP is the stale side. DA does not move a class: escalated. **250 is NOT in this set** · **R-1** (which froze the 250 rung), narrowing R-8. **R-8's own text says `{0,50,100,250,500}` stay Class A while citing R-1's freeze of 250 — the ledger is internally inconsistent; erratum requested, §10.12** |
| `tau_decision_rung` = **250 ms** | venue | CHOSEN | — | **CLASS D — VERDICT.** **R-1 froze it.** The three-way `ww_v1` verdict is computed on `R(τ=250 ms)`. Rev 3 wrongly left it interior and said "no verdict turns on them" |
| `tau_kill_bound_ms` = **1000** | venue | CHOSEN | — | **CLASS D — GUARD.** **R-8**, frozen, stated as an **absolute** 1000 ms, not "the top rung". **1000 ms REMAINS a reported ladder rung** — the receipt stamps six rungs and R-9 verifies measured vs shuffled `R` at every one; de-linking it from the ladder would stop `R(1000)` being produced |
| `tau_operative` | venue | **MEASURED at deployment, UPPER BOUND** — **BUT TWO CONSUMERS READ A RUNG, NOT A BOUND (§10.31).** `OP_PLANE_PLAN` §5.1 and `DE_MODULE_PLAN` §5 both say *"the operative rung is the smallest ladder rung ≥ the measured upper bound … stored as an SP-Params value with provenance, consumed by the Actuator"*. This row registers the **bound**; an Actuator reading "the SP-Params value" gets the raw bound instead of the coarser rung it should round up to — **optimistic, the one direction OPS says the seam must never degrade.** Two rows are needed, or one row plus a stated rounding rule. Also: R-55 holds `tau_operative` **UNMEASURED** until an Actuator exists, so this row's "MEASURED at deployment" is a future tense that reads as present | `OP-LatencyBudget` publishes → coordinator adopts | venue ack is **not observed**; value is a reconciliation bound, conservative by construction |
| `D_hat(source)` | — | **see `ImputationRule`** | DA | **ROW DELETED — AND A CONSUMER STILL READS THE DELETED ROW (§10.32).** `MEASUREMENT_PLAN` §1.2 Class B computes `t_known := t_event + D̂(source)` with **`D̂` from `SP-Params.at(t_event)`**, and R-6 §4a's ratified table lists `D̂` among the Class-C SP-Params rows. So a builder implementing IMPUTED knowledge time calls `SP-Params.at()` and resolves nothing — or restores the row and recreates the two-owner defect this deletion prevents. The deletion is right on the contract (`ImputationRule` is a live v22 type consumed by `DA-Ingest`) but it silently removed a row a **ruling** ratified and a sibling still reads. Needs saying in one place, not resolving in two. `D̂` *is* `ImputationRule.{delay, error, bias_direction}`, DA-owned. Two owners of one name otherwise · §10.1 |
| `belief_a` | instrument × horizon | **MEASURED** | **BE publishes → coordinator adopts → DA writes the entry** (R-37: three roles; EV does neither) | **ROW RESTORED — Rev 4's deletion was wrong three ways.** The publisher is **BE**, not EV (so there was no `EV → DE` edge here; the real one **was** `de_constraints.py`, **removed under R-32, landing-verified under R-37**). `ReducedFormFit` has no `a`/`b` and no `artifact_id`, so the redirect target could not hold them. And R-6's Class-C table plus `BE_BELIEF_PLAN` both place them in SP-Params — deleting the row created an **ownerless** quantity. BE's own two-object split (`â` measured; the deployment pin `a := 0` chosen) is open and must not be quoted as one parameter · §10.13 |
| `belief_b` | instrument × horizon | **MEASURED** | **BE publishes → coordinator adopts → DA writes the entry** (R-37) | **SPLIT FROM `belief_a` at iteration 10.** One row is one `ParamId.name` (§4), so a two-name row resolved to a single key covering two quantities — the `tau_ladder` shape iteration 2 fixed. Splitting is pure naming and is DA design. **No class is stamped beyond MEASURED**, because the ledger records that `belief_a` is TWO objects and **only one of them is Class C** — the measured fit here, the deployment pin `a := 0` separately (Q-DA-26) · §10.13 |
| `verdict_coins` | portfolio | **measured inputs + CHOSEN thresholds** | BE publishes → **coordinator ratifies under R-ADMISS** | **ESCALATED §10.5** — it is a quantifier domain, not a value |
| `r_terminal` (`r=60`) | instrument | **MEASURED**, used as a handle | BE publishes → coordinator adopts | terminal collapse located at `r<60`. **The `+0.422` step is EXCLUDED from the verdict as circular** — do not cite it as the evidence |
| STOP's `h`, fee treatment, cancellation policy | instrument | CHOSEN | **THE USER** → — | **ANSWERED — R-35, to the user.** Coordinator recommends pinning **h=5** with the ladder beside it (h=5 is where every leg of the evidence was measured; longer horizons are population-shifted by construction). The user owns STOP, so this is a recommendation. *Seat resolved: the user.* **`FIRE_SIDE` is R-24 vocabulary. R-36 re-opened R-24 as PROSE-ONLY; `R-42` LIFTED IT — R-24 IS NOW IN FORCE**, the machinery being `ev_gates.py` (`assert_directional()`, 23 selftests as of 2026-08-23; R-42 recorded 17 at ruling time and the file has grown since), demonstrated by the coordinator against the real h=5 cells and their reflection through zero: the ORIGINAL threshold returns PASS on both, the AMENDED one FIRES on the measured cells and PASSes on the mirror. `FIRE_SIDE` is citable. `EV_GATES_PLAN` BE-6: until pinned by value with a class and an owner each, `STOP` has no determinate reading, and it reads `FIRE_SIDE` at one of four horizons. **Residual, narrower than Rev 12's "seat conflict":** the seat is RULED (the user, R-35). What is still open is only **whether STOP's `h` and `primary_horizon` are the same quantity** — if they are, one is owned by the user and the other by the coordinator, and that needs saying |
| `cancel_by_deadline` | instrument | CHOSEN | — → coordinator | **DEFERRED — cancellation family CLOSED (R-11).** Real dependency is the resolution-disposition venue fact below, not `ww_v1` |
| `rewards_band_dispute_handler` | instrument | CHOSEN | — → coordinator | **NEW ROW.** Which authority wins when Gamma and the CLOB registry disagree |
| `reset_authority` | portfolio | CHOSEN, load-bearing | OPS proposes → coordinator | **NEW ROW.** Changing who may clear a latch changes what every past halt meant |
| `staleness_deadline` (per port) | feed | **CHOSEN multiple on a MEASURED cadence** | OPS publishes cadence → coordinator sets multiple | **NEW ROW.** §8's "genuinely both" falsifier has **already fired** here |
| `min_size` | venue | **`SP-Venue` field, NOT a register entry** | **DA publishes → coordinator adopts** | **`FieldState.Resolved(5, markets.jsonl, OBSERVED)`** — provenance is `OBSERVED`, **not** `MEASURED`: read off the wire, not estimated with a fit window (§3 defect 1). **Caveat, and it is worse than "unratified":** `Provenance` is used as a type on sixteen carriers in v22 and has **NO MEMBERS AT ALL** (`contracts.yaml:185,267,335,416,486,701`; R-3's enumeration is one of the six never-executed directives, `COORDINATOR_REVIEW_LOOP.md:110` RIGHT-BUT-UNENFORCEABLE). So `OBSERVED` is not *excluded* from a ratified enum — nothing is enumerated, and no checker can reject any string. `OBSERVED` is written here because **R-IMPUTE's** checks already use it as a member of `Known.t_known_prov` (`contracts.yaml:72-79` for the rule and its three checks, `:180` for the enum) — the closest ratified usage. *(Rev 13 attributed those checks to **R-PROV**, which contradicts §3's own diagnosis that `R-PROV` has a `body` and no `checks:` list. Real citation, accurate quote, wrong rule — the class this document keeps catching elsewhere.)* §7 item 3 asks for the enum. `orderMinSize = 5` on **every row that carries the field**; the stable figure is the **44 early rows predating it** (`markets.jsonl` is live append-only, so absolute counts drift between reads), and `flow_uncertainty.py:35` already reads it. Rev 5 recorded this as `Unknown`, which was wrong. **Consequence, load-bearing for §6: `min_size` EQUALS `quote_size_pin`, so the pin has ZERO DOWNWARD HEADROOM** — it can be raised, never lowered |
| `matching`, `T`, `payoff`, `complement` | venue / instrument | **`SP-Venue`/`SP-Instrument` fields, NOT register entries** | DA → coordinator | **MISSING IN REV 7 — added.** §2 declares these SP-owned and §4 carried none. **"horizon" is overloaded three ways and needs one name** — `primary_horizon` = 5 s here; `ScopeKey.horizon` is the decision horizon; `MEASUREMENT_PLAN` uses `SP-Instrument.horizon` for the 300 s market window that derives `phase()`. **These are NOT the same quantity**: `BE_FLOWANDFILLS` computes `horizon_effective = min(horizon, T - t)`, which degenerates if `T ≡ horizon`. A builder resolving "no quoting after T" against a 5 s value halts every market 5 s in (`MEASUREMENT_PLAN` derives `StateView.phase()` from `SP-Instrument.horizon`; `BE_FLOWANDFILLS` computes `min(horizon, T-t)`) — **one name needed**. `matching` is what `queue_bound_arm`'s arms mean |
| `settlement_latency` | instrument | **`SP-Instrument` field** (§1's venue-mechanics list says "`SP-Venue` fields" — **this one is the exception**, corrected here rather than in two places) | DA → coordinator | **`FieldState.Unknown(reason, sources_tried)` — MISSING IN REV 7.** §1 promised it a home and §4 had none. `DE-Allocator` blocks on it (when carried-residual proceeds return as quotable capital) and `EV-Replay` makes it the boundary where replay purity lapses. The `settlement` row below is the FORMULA, not the latency |
| per-port `period` | feed | **MEASURED** | **OPS publishes → coordinator adopts** | OP §8a: "observed, not chosen". Needs its OWN key — it is the measured half of `staleness_deadline` |
| expected-port registry · `LANE_PROGRESS` grace · per-port `period` | feed | **CHOSEN** (registry, grace) + **MEASURED** (`period`) | OPS proposes → coordinator | **NOT `SP-Venue` facts** — chosen operational configuration, so they do not belong on a record §2 defines as venue facts. **MISSING IN REV 7** — three of the nine rows OPS hands to SP. The port registry is the **quantifier domain of the staleness guard**: drop a port and its monitor stops firing with no value moving. **CLASS DISPUTED, ESCALATED §10.16 / Q-DA-17:** SP reads **Class D** (§10.4 applied to a GUARD, under **R-20's un-numbered guard generalisation** — *"Class D covers any value on which a verdict or a guard turns"* — NOT R-20's numbered clause 2, which is the candidate-bar rule); `OP_PLANE_PLAN.md` §8a reads **Class A**. DA does not resolve it here — a class assignment needs a ruling. `period` is the MEASURED half of `staleness_deadline` and **needs its own key**; bundled here only because OPS hands the three together |
| rate limits, capability flags, resolution disposition | venue | **`SP-Venue` fields, NOT register entries** | DA → coordinator | **`FieldState.Unknown(reason, sources_tried)`** — genuinely not passively knowable. A `FieldState` variant, expressible on `SpecRecord.fields` and **not** on `ParamValue`, whose `value: any` would swallow it |
| `fee_schedule`, `rebate_schedule` | venue | `SP-Venue` | **DA** publishes → coordinator adopts | fee MEASURED — anchor: `FLOW_MODEL_STATE.md` §1, taker-fee row; **rebate `Unavailable`** — zero observed fee is not evidence of a rebate |
| `tick_rule` | **instrument**, state-dependent | `SP-Venue`, `(BANDED, …)` | **DA** publishes → coordinator adopts | measured on **btc only**; `Unknown` elsewhere. **Two frozen estimands are denominated in ticks**, so the tick state freezes with those bars · R-20 |
| `settlement` spec | instrument | `SP-Instrument` | **DA** publishes → coordinator adopts | anchor: `FLOW_MODEL_STATE.md` §1, settlement row |

**Not an SP row:** the half-spread `h` in the dump backstop. It is a *market
observable*: the measured **half-spread is 0.50 ¢** (`h = 0.005`), against a 1-tick median **spread** (the p90 spread is 2 ticks, so the p90 half-spread is 1.0 c, not 0.5) — Rev 3 wrote the spread figures into
a half-spread row, a 2x error worth ~22 % in `N*`. It is a `DA`/`StateView`
read, not an `SP-Venue` constant. The DE plan's sentence deferring it to
SP-Venue is wrong and is flagged to DE.

**The rule the table encodes:** a worker may *publish a measurement that a
parameter must equal*; only the coordinator may *choose* one — **and every
adoption is an explicit act, not an automatic inheritance.**

## 5. The operative parameter set

**One set, marked OPERATIVE** (user-ratified 2026-08-23, R-6). Rev 1 called it
"replay defaults — deliberately not deployment values" in §5 and "the operative
set" in §5a; two readings let an adverse verdict be dismissed as replay-only and
a favourable one cited as operative. There is no second configuration behind it.

```
SET NAME: SP_PLANE_PLAN_s5_operative_R6       # the DEPLOYED name -- see note (iii)
total_capital         = $1,000 USDC notional
quote_size_pin        = 5 shares              # the only size row, §6
kappa_usd             = $50 USDC per market, against L_adv (a cost basis)
ScenarioLossLimit(<scenario_id>)  = $200 USDC  # PER DECLARED SCENARIO (R-35)
refuse_k              = 1.0                   # CLASS D - GUARD (R-20; §10.2 CLOSED)
gamma_ladder          = {0, 1e-3, 1e-2, 1e-1} # sensitivity axis, not a preference
knowledge_lag         = 250 ms
primary_horizon       = 5 s
```

**Three honesty notes on this block.** (i) **R-6 ratified SIX entries**
(`capital_budget`, `max_quote_size`, `κ_$`, `ScenarioLossLimit`, `refuse_k`, the
γ ladder). `knowledge_lag` and `primary_horizon` were added by DA and are
**recorded, NOT ratified**; `total_capital` and `quote_size_pin` are renames of
the first two. Adding to a ratified set is not DA's to do — flagged for
ratification or removal. (ii) **The set does not pin every value a frozen
verdict turns on.** `queue_bound_arm`, `tau_decision_rung`, `tau_ladder_rungs`,
`tau_kill_bound_ms` and `verdict_coins` are §4 rows — three of them Class D —
with no line here, so two runs stamped with the same set name can differ on
values that decide the verdict. The receipt already pins more than this block
does. (iii) **The name is CONFIRMED by R-37** (DA pre-resolved it under R-33 while the coordinator still had it open, so DA's "never a choice" characterisation was wrong). `ev_replay.py`
stamps `SP_PLANE_PLAN_s5_operative_R6` and every receipt **that stamps a set name** carries it — measured **2026-08-23: 4 of 34 files** in `data/pm_5min/derived/` — `policy_bounds_v1.json` (R-50), `ww_ebx_v1.json` (R-54), `state_gate_v1.json` (R-53) and `ev_replay_v1_smoke.json`. Rev 13 measured 1 of 31; three protocol receipts have landed since, so R-37's rename-cost reasoning now covers four provenance stamps, not one, so Rev 13's "every receipt on disk" was literally false; R-37's own "every published receipt" is quoted correctly and is the claim that matters.
R-37's reasoning: the shipped name is on **every** published receipt, so renaming would invalidate their provenance to gain only tidiness, and it is the more informative name (it cites both plan section and ruling). R-37 names the general pattern — *whatever ships first with a citation becomes the standard by default* — and marks this the **benign** case, corrosive only where no standing rule exists.

**Units are now explicit per line** because the three limits are not
commensurable as Rev 1 wrote them: a notional budget, a cap on `L_adv` (a cost
basis), and a cap on scenario PnL.

**Rev 1's "exercises both branches" claim is withdrawn.** It said `κ_$ = $50`
against `$1,000` "makes the per-market cap bind before the portfolio cap on a
single market and after it across four". Four markets at $50 total $200 — a
fifth of a $1,000 budget, which binds at twenty markets, not four. Read against
the $200 scenario limit instead, four markets tie it exactly, so the branch
taken depends on `≤` versus `<`. **Neither reading reaches the configuration
claimed**, and DE would have built its feasibility oracle against a set that
never exercises the portfolio branch. Re-deriving the binding counts is a
coordinator choice — **§10.14**.

**The set is incomplete until at least one `AdverseScenario` is declared.**
`SP-Scenarios` is empty, so `RiskScenarios` is `Unavailable`, and
`DecisionSchemeConfig.unavailable_policy` halts `RulePolicy_v1` on it at the first decision (**not** the module manifest, which carries no unavailable-input behaviour) — the branches §5
exists to exercise are never entered. **`LossFunctional` is NOT ownerless** —
§2 assigns it to `SP-Strategy` as a **peer field of** `constraints`, not a
member of it, and it is a live v22 type (`contracts.yaml:432-435`,
`loss: (JointOutcome, PortfolioState, ActionSet) -> Wealth`) consumed at
`ScenarioLossConstraint.loss_fn`. Rev 12 and earlier asserted an ownerless
quantity that §2 already owns, and sent a builder to the wrong field.

**`params.at(t)` is required, not optional.** Without it `params.get(name)`
resolves *today's* value inside yesterday's replay — a look-ahead in parameter
space. It cannot be built until `ParamValue` carries a validity interval (§2).

## 5a. Tuning authority — R-6's taxonomy, **under escalation**

R-6 defines four classes: **A** configuration (free) · **B** load-bearing
(change forces a re-run) · **C** measured (adopt, never choose) · **D** frozen
verdict bars (changeable only before the measurement runs), with a **five-clause**
amendment mechanism for D — clauses (a)(b)(c) from R-6 itself, **(d) from R-38**
and **(e) from R-40**, all five set out below and all five binding.

**R-6 has two columns and Rev 12 reproduced only one.** This section is cited as
an authority for the taxonomy — `OP_PLANE_PLAN.md` §8a treats it as
interchangeable with the ledger's own §4a — so dropping the obligation column
made the cited copy **the permissive one**. Restored verbatim in substance from
R-6 (`COORDINATION.md` §4a):

| class | what it permits | **what it OBLIGES of the worker** |
|---|---|---|
| **A** | free to change | sweep it when asked; **report the RANGE, not the best point** |
| **B** | change forces a re-run | tell the coordinator what a change would invalidate, **before it is made** |
| **C** | adopt, never choose | publish the measurement; the coordinator adopts it |
| **D** | frozen | **REFUSE a post-hoc change and escalate** |

And R-6's closing duty, which is the only clause in the taxonomy addressed to a
worker under top-down pressure, and therefore the one this section least
tolerates losing: *if the coordinator ever asks a plane to move a Class-C or
Class-D value after a result is visible, **the plane should refuse and say so in
`COORDINATION.md`***. §10.6's withdrawal rests on the Class-B obligation above,
which a reader of Rev 12's §5a could not have verified.

**R-20 amends it in two ways, and both are applied in §4.** *(Numbering caution, and it is the source of a leak iteration 8 repaired downstream while leaving this copy: the items below labelled **(1)** and **(2)** are THIS SECTION'S numbering, not R-20's. R-20's own numbered clause 2 is the candidate-bar rule; what §5a calls (2) — the guard generalisation — is **un-numbered** in the ledger. Q-DA-17's Class-D derivation rests on that distinction, so cite it as "R-20's guard generalisation", never as "R-20 clause 2".)*

**(a) BEFORE THE MEASUREMENT RUNS** — an amendment made after the result is
visible is not an amendment, it is a retrofit. **(b) MOTIVATED BY INFORMATION
THAT IS NOT THE RESULT** — a methodological reason that would have applied had
the number come out the other way; R-38 concedes this clause is judgment-laden
and is the weakest of the five. **(c) EXPLICITLY INVALIDATING EVERY VERDICT
COMPUTED UNDER THE OLD BAR** — and note R-38's finding that (c) is a **reward,
not a cost** wherever the standing verdict is a refutation the amender wants
gone, which is why (d) exists. Clauses (a)-(c) are R-6's own three; Rev 13
asserted all five were "set out below" while (a) and (b) appeared nowhere in the
document — the same half-missing shape Rev 13 was written to repair.

**(1) Freezing a bar SNAPSHOTS ITS INPUTS BY VALUE.** When a Class-D bar is
frozen, every Class-C value it consumes is recorded into the freeze record with
its **numeric value, `artifact_id` and provenance**, and the bar is defined
against that snapshot rather than a live reference. A later Class-C
re-publication does **not** move a frozen bar — it produces a **CANDIDATE** bar
requiring an explicit Class-D amendment under **all five clauses** (a)-(e). This keeps
both properties at once: Class C stays adopt-never-choose, so the coordinator
still cannot pick the markout; and Class D becomes genuinely frozen, so the bar
cannot drift underneath it.

*The retroactive anchor for R-1 is recorded in the ledger:* btc spread
**+0.642**, `|markout|_lo` **0.287** → `f*_low` **30.9 %**; eth **+0.778**,
**0.759** → **49.4 %**. R-11's DEAD/DEAD is anchored to those values. **If a
re-publication would flip a verdict, that is a FINDING to surface, not a silent
correction.**

**(3) R-38 — CLAUSE (D): an amendment may not, by itself, change a verdict.**
Invalidating verdicts computed under the old bar renders them **UNDETERMINED**,
never the opposite verdict; re-establishing one requires **re-running the
measurement** under the new bar at the original evidentiary standard. This makes
clause (c) cost-bearing in **both** directions — erasing a refutation buys an
**obligation**, not a result. *Consequence, stated so it is not argued later:*
R-11's DEAD/DEAD and R-17's Layer-2 negative rest on bars frozen before their
data was read and anchored by value under R-20; an amendment to either vacates
the verdict to UNDETERMINED **and obliges a re-run**, and **no amendment reaches
the evidence.**

**(3a) R-43 — CLAUSE (D)'s STANDARD IS PINNED BY THE R-20 SNAPSHOT.** Clause
(d) obliges a re-run *"at the original evidentiary standard"*, and DA escalated
that a standard which is not itself pinned can be reinterpreted downward by the
same amender who moved the bar (Q-DA-15). **R-43: it was already pinned, and is
now stated.** R-20 clause 1 snapshots a frozen bar's inputs **by value**, and
**the evidentiary standard is part of that snapshot** — sample rule, weighting,
day-clustering, CI method, *as recorded at freeze*. Re-running "at the original
standard" means **against the snapshot**, not against whatever the standard has
since become. R-43's reasoning generalises: *an unstated pin is
indistinguishable from no pin*.

**Consequence for §10.10, and it raises that item's price.** The freeze record
must now carry the *method*, not only the numbers — R-20 clause 1 as written
names "numeric value, `artifact_id` and provenance", which does not reach sample
rule or CI method. §10.10 already escalates that **the freeze record has no
contract carrier at all**; R-43 enlarges what that missing carrier must hold.
Recorded here rather than actioned: DA does not add fields to a record type it
does not own.

**(4) R-40 — CLAUSE (E): a vacated verdict carries its provenance
PERMANENTLY.** Clause (d) closes amend-your-way-to-a-pass but leaves a residual:
an amender who vacates and never completes the owed re-run holds the verdict at
UNDETERMINED indefinitely — four-fifths of the erasure for the price of an
unpaid IOU. So a vacated verdict reads **"VACATED — was DEAD under bar X;
re-run owed"**, and the vacating amendment **files a §0a register row** for
that re-run. Limbo becomes loud, and R-28's append-only philosophy reaches
verdicts. *(DA corroborated this residual independently at §10.)*

**Two mechanism gaps in clause (e), named rather than papered over (§10.19,
Q-DA-19).** (i) *"Auto-files" describes no mechanism* — §0a rows are "appended
by the **asking plane**" (`COORDINATION.md:20-21`), the vacating party is
normally the coordinator, who is not a plane, and there is no amendment-derived
row type. Until one exists the filer is **whichever plane owns the vacated
verdict's page**, and this sentence is that assignment; "auto" is dropped
because a duty nothing performs is worse than a duty someone owes. (ii) *No type
carries a verdict state.* `ParamValue` (`contracts.yaml:413-421`) holds
`value / provenance / owner / valid_for / measured_at / fit_data_through /
artifact_id` and there is no verdict record and no freeze record (§10.10, open),
so **"VACATED — was DEAD under bar X" has nowhere to live but prose**. The same
holds for the A/B/C/D class itself: it appears in **no** contract type and in
**no** `rules:` key, so every Class-D freeze in §4 is prose-enforced and **no
checker can detect a Class-D value moving** — iteration 2 caught exactly that
string going stale in a receipt (`"quote_size_pin": "CLASS B"` after R-20 made
it D). §7 asks for a `Provenance` enum and an authority axis; it does not ask
for a class field, and it should.

**(g) DA PROPOSAL — SCOPE IS A MOVEMENT. No ruling stands behind this
clause; it is not in force (Q-DA-25).** Clauses (a)-(f) all govern changing a
value. **None governs changing its SCOPE**, and Rev 14 opened that door itself:
§4 now writes a row at the broadest true key and lets subset-order resolution
default the rest — correctly, since the alternative demanded ~2,000 writes a day
— but nothing then governs **adding a NARROWER entry**. Write
`(quote_size_pin, ScopeKey{instrument: I}) = 20`: the Class-D entry at
`ScopeKey{}` never moved, no value was amended, so no clause fires; §6's
strictest-alias rule binds two *names*, not two *keys of one name*; and R-20
clause 1 snapshots *"numeric value, `artifact_id` and provenance"* — **not
scope** — so the freeze record cannot see it either. Every consumer for market
`I` resolves 20 while the register still reads 5. The proposal: **for a Class-C
or Class-D `name`, ADDING, REMOVING or RE-SCOPING any entry is a movement of
that value under clauses (a)-(f), and R-20 clause 1's snapshot must include
scope.** DA cannot adopt it — it amends the amendment mechanism.
**INTERIM SELF-BINDING, which DA can do and does:** until this is ruled, DA
writes no entry at any key narrower than the one a Class-C or Class-D row is
already published at, and will refuse such a write if asked.

**(f) DA PROPOSAL — DIRECTIONAL TRIGGER. No ruling stands behind this clause;
it is not in force (Q-DA-18).** Clauses (a)-(e) all fire on **invalidation**;
none fires on **direction**. A two-step loosening therefore costs one re-run:
amend a snapshotted input so the bar moves toward the standing verdict without
flipping it (clause (d) yields UNDETERMINED, clause (e) files the row), discharge
the re-run at the same answer, and then let the Class-C re-publication that R-20
itself ordered cross the now-nearer bar. Every step is legal and the refuting
margin is silently consumed. The proposal: **an amendment that reduces the
refuting region must re-run under BOTH bars and report both arms, triggering on
the region rather than on whether the verdict happens to flip.** DA cannot adopt
this — it is an amendment to the amendment mechanism, and R-38 clause (d) came
from the coordinator ruling against R-6's own clause (c), not from a plane.

**(2) [§5a's numbering; UN-NUMBERED in R-20] Class D covers any value on which a VERDICT OR A GUARD turns**, not
verdict bars alone. Applied in §4: `refuse_k` (the no-peek coefficient),
`knowledge_lag`, `primary_horizon`, `queue_bound_arm` and `quote_size_pin` each
carry a named verdict or guard, and `collector_era` and `tick_rule` carry
Class-D consequences.

**The remaining taxonomy defects stand escalated in §10**, not edited in place.
DA has changed no class assignment except where a *ruling* moved it — R-8's τ
split and R-20's two amendments.

## 6. The size pin, and why it is not free

`quote_size_pin = 5 shares` is formally CHOSEN, but **every measured number in
this programme is conditional on it** — the join-touch fill brackets, the
Layer-1 markout decomposition, the inventory walks, the skew bounds, the
cross-window correlation, the `ww_v1` verdict. Changing it does not re-tune a
policy; it **invalidates the measurements the policy was built from**.

**Rev 1 defeated its own constraint by aliasing.** It placed `max_quote_size`
inside `capital_budget` — Class A, freely changeable, no re-run obligation —
gave `quote_size_pin` a separate Class-B row, and then set them equal in §5.
The same number sat in two rows with two different classes, and the free one was
the one wired into the size chain. Raising the "budget" would have moved the pin
with none of Class B's cost.

**Fixed:** one `ParamId`, instrument scope. **Under R-20 the pin is now
CLASS D — VERDICT**, not Class B: every Class-D verdict in the programme is
conditioned on it. **AND A FROZEN PROTOCOL CARRIES THE OLD READING WHILE CITING
THIS SECTION AS ITS AUTHORITY (§10.30 / Q-DA-32).**
`POLICY_BOUNDS_PROTOCOL.md` §3 — frozen under R-45, *after* R-20 — reads
*"doubles as the **Class-B robustness probe** of the 5-share pin that every
published number is conditioned on (**R-6/SP §6**)"*, and
`DE_PLACEMENT_POLICY_PLAN` propagates it as *"the Class-B answer"*. One value,
two classes, on the row every Class-D verdict is conditioned on — and **the
permissive reading is the one inside the frozen document**, mis-citing this
section as its source. Class B licenses a change that costs a re-run; Class D
obliges refusal. SP is correct; under R-28 the protocol takes an **annotation
beside**, never an edit, and that annotation is not DA's to write on a document
DE owns. The A-vs-B framing above is the history, not the current
class. `total_capital`
carries notional only and cannot express shares. **And the pin has a MEASURED FLOOR, not merely a class.** `min_size` is 5 on every market carrying the field — **equal to the pin** — so `quote_size_pin` can be
raised but **never lowered**. §6's "not free" is therefore stronger than a
re-run obligation: downward movement is infeasible at the venue, not just
expensive.

**Standing rule: an aliased
value takes the strictest class of its aliases** — and the alias is currently
duplicated as two independent literals in code — **and the guard SP §6 demanded now EXISTS**, `ev_replay.py:277-285`, asserting *"the two SP_OPERATIVE literals are identical (SP §6 guard)"*, landed under R-33; Rev 13 still stated the obligation as owed — **and Rev 14
then discharged the duty and re-stated it as owed in the same paragraph**, which
is the failure mode this document names, committed inside the repair for it. The
duty is DISCHARGED; nothing further is owed here.

## 7. Contract changes this plan requires

Rev 1 listed **one** and called it "self-ratified here". Self-ratification is
the conflict R-18 exists to end: **DA proposes, the coordinator ratifies.**
None of these is applied.

1. **Four record types** — `SP-Venue`, `SP-Instrument`, `SP-Strategy`,
   `SP-Scenarios` — plus an SP spec-resolver module/port. Without them
   `DecisionProblem.spec_snapshot` and `ResolvedContracts.spec_hash` have no
   producer.
2. **`ParamValue.valid_from`/`valid_to`** — R-VERSION is otherwise
   unsatisfiable and replay snapshots are not reproducible across a param
   revision (§2).
3. **The `Provenance` enum with `OBSERVED`, de-collided axis names, and an
   authority axis** (§3).
4. **A fee family** `(PRODUCT_PQ, {rate, incidence, size_rounding})` — none is
   defined in v22 (§2).
5. **`DE-Constraints.consumes: CapitalBudget`** — Rev 1 said "no further change
   needed"; v22 has `CapitalBudget` *produced by* `DE-Allocator`, absent from
   `DE-Constraints.consumes`, typed `{by_instrument: dict[InstrumentId, Money]}`
   with no size field. The DE plan already lists this change.
6. **A human-seat owner type** — `EV-Gates` gives `STOP-MM-VIABLE`
   `owner = the user`, and `ParamValue.owner` is a `ModuleId`; neither the user
   nor the coordinator can be serialised into the register as it stands.

## 8. What would falsify this design

- **The choice/measurement line does not survive contact.** A parameter that is
  genuinely both — a choice whose admissible *range* is measured — makes the
  enum too coarse and demands a `(character, constraint)` pair. **This has
  already fired**: `staleness_deadline` is a chosen multiple on a measured
  cadence. Rev 1 nominated `quote_size_pin` as the row to watch; the rows that
  are actually both are `quote_size_pin` — **restored: Rev 1 nominated it, later
revisions recorded the nomination as refuted, and §6's own `min_size` finding
vindicates it. `min_size = 5` measured at the venue makes the pin's admissible
range `[5, ∞)`, which is exactly "a choice whose admissible RANGE is measured"** —
plus `staleness_deadline`, `verdict_coins` and
  `tau_operative`.
- **The operative set selects the answer.** If a DE verdict flips between
  `SP_PLANE_PLAN_s5_operative_R6` and a plausible alternative, the set is not
  shape-only.
  **The alternatives must be enumerated before the sensitivity run**, or the
  test cannot bite — whoever picks "plausible" after seeing the flip decides the
  outcome.
- **`r_terminal` is an artefact.** The `r ≈ 60` handle may be an artefact of the
  confounded 60 s structure; its headline evidence is already quarantined as
  circular. A re-measurement trigger is owed.
- **`verdict_coins` is revised.** Expected first revision if a thin coin's micro
  share crosses the R-DUAL bar — **restored here because `OP_PLANE_PLAN` quotes
  this bullet as SP's own text and Rev 4 had dropped it.** Note the membership
  rule is frozen in `FLOW_MODEL_PROTOCOL_V4/V5` with a stated
  `restriction_reason`, so a revision is a protocol amendment, not an adoption.
- **Bitemporality is unused.** Currently untestable — the machinery does not
  exist on `ParamValue` (§2).
- **`SP-Scenarios` cannot express the real adverse world.** If the binding risk
  is a *path* rather than a terminal outcome map, the declared form is
  inadequate and `BE-ScenarioProvider` is promoted from deferred to demanded.

## 9. What this plan deliberately does not do

- **Does not choose a utility.** But "utility stays empty" is not expressible:
  `DecisionSchemeConfig.utility` is a non-optional `PluginRef`. The honest
  statement is that `DE_MODULE_PLAN` §3.2 registers **`utility_none`** — valid only with
  solvers declaring no utility consumption — and keeps the γ ladder a
  **reporting device, not a config value**. **SP adopts that.** An earlier
  revision named `cara` with γ through `PluginRef.config`: the exact wiring DE's
  n-ary check exists to reject.
- **Does not set deployment capital or risk limits.** §5 is the operative set;
  deployment is a future re-freeze that explicitly invalidates the verdicts
  computed under it.
- **Does not populate records speculatively.** SP is populated on demand; the
  register grows when a consumer demands a row.
- **Restates measured facts only where R-20 requires a snapshot, and names the
  source.** Rev 1 claimed not to restate and then restated five; Rev 4 repeated
  the claim while §4 still carried `+0.422`, the verdict coins, 250 ms, 1000 ms,
  0.50 ¢ and the R(250) figures. The honest rule: **a number appears here only
  if a frozen bar's snapshot requires it**, and it carries its anchor.

- *(SUPERSEDED by the bullet above — retained for the record, and note how it
  came to look live: an earlier repair split the marker "Previously worded as:"
  off this bullet to fix a fused line, which removed the only thing marking it
  as history. A formatting fix silently promoted a withdrawn claim.)*
  **Does not restate measured facts.** Rev 1 said this and then restated five —
  the fee, the tick regime, the settlement spec, `+0.422`, and the verdict
  coins — becoming the drifted second copy it named as the failure mode. It had
  already happened: the `r_terminal` circularity caveat was present in the DE
  plan and absent here. **Evidence columns now carry anchors into
  `FLOW_MODEL_STATE.md`, not numbers.**

## 10. Escalated to the coordinator — taxonomy defects, not design defects

Under **R-18** the coordinator reserved re-cutting the class taxonomy to
themselves and invited DA to say so with evidence where it is wrong. *(That
instruction reached DA in the R-18 dispatch message; it is **not** in the ledger
text of R-18, so treat this as DA's paraphrase of an instruction rather than a
quotable ruling — an earlier revision presented it as a direct quotation, which
a reader could not verify.)* It is wrong in the places below. Each is evidenced;
none is edited here.

**STATUS after iteration 10 — read this before ruling on anything below.**
Iteration 3 audited these escalations against the authorities and found five of
them defective. DA has corrected them here rather than leaving them in front of
a ruling.

| item | status |
|---|---|
| 10.2, 10.3 | **RULED, CLOSED** by R-20 (cores verified sound) |
| 10.4, 10.5, 10.10 | **OPEN, verified sound** |
| **10.9** | **UPHELD IN PART — R-38; clause (d) adopted** — see the correction notice; DA's earlier "hole in R-20" framing was **wrong**. **RE-OPENED at iteration 7:** the subsection's closing directional requirement was never ruled, cited an authority that evaluates something else, and was not in §5a. Relocated to §5a clause (f) as a **PROPOSAL**; Q-DA-18 |
| 10.1, 10.7 | **OPEN, NARROWED** — real cores, wrong framings, now trimmed |
| **10.6, 10.8, 10.11** | **WITHDRAWN by DA** |
| **10.12** | **ANSWERED** in DE's coordinator review (now CLOSED, R-41): strictest-alias reading confirmed, **and the R-8 annotation IS owed** — the ask no longer stands |
| **10.13** | **RULED — R-37.** BE publishes, the SP owner (DA) writes the entry, EV does neither |
| **10.14** | **PARTLY DISCHARGED** — the κ_$ binding counts re-derived; the "exercises both branches" claim withdrawn |
| **10.15** | **RULED (R-35), CLOSED.** Scenario-scoped; line cite re-aimed after the R-35 conformance moved the code |
| **10.16** | **NEW at iteration 7 — OPEN.** Expected-port registry: Class **D** in §4, Class **A** in `OP_PLANE_PLAN.md` §8a. Class conflict, not DA's to resolve |
| **10.17** | **NEW at iteration 7 — OPEN.** `ScopeKey` has no port member, so two §4 rows cannot be written as keys at all |
| **10.18** | **NEW at iteration 7 — OPEN.** Conditional classes have no detector and no ordering rule |
| **10.19** | **NEW at iteration 7 — OPEN.** Clause (e) has no carrier; the A/B/C/D class has no machine surface anywhere |
| **10.21** | **NEW at iteration 8 — OPEN.** `SP-Params` is not a `ModuleOrPluginId`; no `SP-*` module id exists in v22 |
| **10.22** | **NEW at iteration 8 — OPEN.** Three rows carry several parameters, so they cannot be keyed; `period` is registered twice |
| **10.23** | **NEW at iteration 9 — OPEN, Q-DA-25.** Scope is not a movement: a narrower `ScopeKey` entry shadows a Class-D row and no clause fires. **Opened by Rev 14's own fix.** DA self-bound in the interim |
| **10.24** | **NEW at iteration 9 — OPEN, Q-DA-26.** `BE_BELIEF_PLAN`'s deployment pin `a := 0` is an unregistered CHOSEN value hiding inside an item stamped RULED |
| **10.25** | **NEW at iteration 9 — OPEN, Q-DA-27.** R-8's owed annotation is untracked and every channel that would surface it is closed |
| **10.26** | **NEW at iteration 9 — OPEN, Q-DA-28.** "Does a value of this id type exist?" was asked on 1 of `ScopeKey`'s 8 axes and answered two contradictory ways |
| **10.27** | **WITHDRAWN IN SUBSTANCE at iteration 11.** Central claim FALSE — `maker_sides` and `primary_style` are pinned BY VALUE in `FLOW_MODEL_PROTOCOL_V4/V5`, three lines below `verdict_coins`. Residue is register completeness, a SHOULD not an ask | 
| **10.28** | **NEW at iteration 9 — OPEN, Q-DA-30.** `SP-Strategy`'s ten structural fields have no taxonomy, and DE already classes one of them as free config |
| **10.29** | **NEW at iteration 10 — OPEN, Q-DA-32.** `τ = 500 ms` is a frozen VERDICT rung in `WW_EBX_PROTOCOL` and a free CHOSEN grid value in §4 |
| **10.30** | **NEW at iteration 10 — OPEN, Q-DA-32.** A FROZEN protocol carries the pre-R-20 Class-B reading of `quote_size_pin` and cites SP §6 as its authority |
| **10.31** | **NEW at iteration 10 — OPEN, Q-DA-35.** `tau_operative` registers a BOUND; `OP_PLANE_PLAN` §5.1 and `DE_MODULE_PLAN` §5 both read a RUNG |
| **10.32** | **NEW at iteration 10 — OPEN, Q-DA-35.** `D̂` deleted here, still read by `MEASUREMENT_PLAN` §1.2 and ratified as Class C by R-6 §4a |
| **10.20, 10.33** | **SELF-RESOLVED BY DA under R-33 clause 3 at iteration 11 — NO RULING OWED.** The architecture already mandates the carrier that makes `Disputed` expressible; §2's field is restored and both items close |
| **10.34** | **NEW at iteration 10 — OPEN, Q-DA-35.** A 2× fee error (`PIECEWISE_MINPQ`) is still live in the rules authority SP says wins |

**DA withdrew three escalations and corrected a fourth. The corrections matter
more than the items:** an escalation that names a hole which does not exist
wastes a ruling, and in 10.9's case would have induced a **narrowing of R-20 —
loosening a good rule for no reason**.

**10.1 The four classes do not cover the SIX-member enum** (`OBSERVED` was added in Rev 2), **and `CHOSEN` now maps to A, B or D with no stated rule.** `ASSUMED` and `IMPUTED` have no class.
**Note the narrowing:** since §4 records character and class as *orthogonal*
axes, `CHOSEN` mapping to A, B or D is the expected consequence of that
orthogonality, not a defect. The live defect is only the unclassed arm — and
R-3's already-ruled "`ASSUMED` and `IMPUTED` may not gate" may discharge it. The live
casualty is `D̂(source)`: §4 of Rev 1 called it `ASSUMED`/"may not gate", §5a put
it in **Class C — measured**, whose text says the coordinator adopts these
because choosing one would be inventing a fact. Since
`t_known := t_event + D̂(source)`, the Class-C reading lets an assumption dressed
as a fact set knowledge time for every record. **A fifth arm is needed:
revisable, never gating.** The same error is in R-6's own table.

**10.2 — RULED (R-20), CLOSED. Moved Class A → Class D.** R-REFUSE:
`admitted(k, now) ⇔ k.t_known + refuse_k · k.t_known_err ≤ now`. Set it to 0 —
permitted "freely, at any time" — and the guard collapses to `t_known ≤ now`,
admitting every value at nominal knowledge time regardless of its error bar.
That is the look-ahead the knowledge-time layer exists to block, and Class A
carries no re-run obligation, so every markout and calibration row afterwards
sits on a different admission population with no invalidation record. **This is
the τ-ladder defect one level deeper.** Class B at minimum; arguably D once a
verdict conditioned on it is visible.

**10.3 — RULED (R-20), CLOSED. Bars now snapshot their inputs by value; a
Class-C re-publication produces a CANDIDATE bar, not a moved one. Retained below
as the evidence the ruling rests on.** `f*` is a *function* of Class-C values:
`f*_low = |markout|_lo / (spread + |markout|_lo)`. Neither markout nor spread
capture is a §4 row, and Class C **obliges** adoption. Arithmetic, verified:

| `|markout|_lo` | `f*_low` | vs measured `R(250) = 15.3 %` |
|---:|---:|---|
| 0.287 (published) | 30.9 % | DEAD |
| 0.150 | 18.9 % | DEAD |
| **0.110** | **14.6 %** | **NOT DEAD** |

A re-sample that moves the markout lower bound to ~0.11 flips btc from DEAD to
INDETERMINATE, reopening the §2 grid and vacating R-11's family closure —
**with no Class-D amendment ever written, because the bar's text never moved.**
Proposed: a bar's inputs freeze *with* the bar, and a Class-C re-publication of
a bar input is a Class-D event requiring clauses (a)–(c).

**Related, narrower — and re-attributed.** The indexical wording is
**`DE_MODULE_PLAN.md:456-458`, inside §5 `DE-Actuator`** — *not* §7.3, which
does not exist (DE's §7 has no subsections; the only "7.3" in that file is DE's
own dangling self-pointer). Note the phrase wraps across lines 456/457, so
`grep "ladder's top rung"` returns a FALSE ZERO — read the lines. And DE carries
a **second** indexical copy at §7 item 3 ("exceeds the τ ladder"), so naming one
address invites the half-fix this document logs as its recurring defect. **Still
live, still un-routed, and parked inside an item stamped CLOSED** — under R-33
this is a cross-plane flag DA routes and DE fixes. **CORRECTED at iteration 10:
Rev 15 said "it is now routed" and NOTHING WAS ROUTED** — the string existed only
in this file; no §0a row, no entry in any cross-plane loop, and DE's two copies
plus a **third in `OP_PLANE_PLAN.md` §8b** still carry the wording and the dead
§7.3 address. R-33 says resolve without asking **and record what you did**;
claiming a routing is not routing. Now genuinely filed as **Q-DA-33**. It says
"if the bound exceeds *the ladder's
top rung* (1000 ms)", so adding a 2000 ms rung re-satisfies **that sentence**.
**R-8's own text is absolute** ("if the deployment-measured ack bound exceeds
1000 ms"), as §4 records — so this is a DE wording flag, **not** an R-8 defect,
and must not be folded into §10.12's erratum request. §4 now states the
bar as an absolute 1000 ms; the DE plan's wording needs the same fix.

**10.4 Quantifier domains are not values, and the taxonomy only classifies
values.** A hard constraint is defeatable by editing its domain with no
parameter change at all: drop the binding scenario from `SP-Scenarios` and the
scenario cap has nothing to bind on; add a verdict coin and "DEAD only if BOTH
verdict coins are DEAD" quantifies over a coin never measured; ship a collector
change and `collector_era` resets, making a "minimum 4 era days" floor
unreachable for four days. **Every set appearing as a quantifier domain in a
frozen rule needs a register row and must freeze with the rule.**

**10.5 `verdict_coins` is misclassified on its face.** Class C says the
coordinator may not choose it — but membership rests on measured inputs **plus
chosen thresholds**, and "which rows enter or leave a lane" is an R-ADMISS
coordinator ruling. It is simultaneously a measurement DA must publish and a
selection decision the coordinator must ratify. Proposed: freeze the set per
protocol, and treat membership changes as R-ADMISS decisions with both arms
reported, never as an automatically adopted measurement.

**10.6 — WITHDRAWN by DA after iteration 3.** Its only live instance is dead: the escalation rests on `quote_size_pin` being Class B, and this same document records it as **CLASS D** twice (§4, §6). The other R-6 Class-B row, `cancel_by_deadline`, is DEFERRED. It also misquotes the rule it attacks — "a re-run; the old results do not carry over" appears in no authority; R-6's actual Class-B row already obliges the worker to say what a change would invalidate **before** making it, which was this item's own proposed fix. Retained for the record only. *Original text:* Class B costs only "a re-run; the
old results do not carry over" — no written amendment, no named invalidation, no
owner, no deadline. Every Class-D verdict in the programme is conditioned on
`quote_size_pin = 5`. Moving the pin vacates them all as "does not carry over" —
**vacated but never withdrawn** — which is a cheaper way to undo a refutation
than any Class-D amendment. Proposed: a Class-B change touching evidence under a
frozen bar satisfies clauses (a)–(c) and names the vacated verdicts *before* the
change.

**10.8 — WITHDRAWN by DA**, superseded by 10.10, which diagnoses the same gap correctly (it is enumeration, not class). Retained for the record only. *Original text:* the snapshot clause names only Class-C inputs. R-20 records "every **Class-C** value it consumes". But a bar depends
on inputs of every class: `f*` consumes measured markout and spread (C), while
`R(τ)` is computed on a fill set defined by `quote_size_pin`, `knowledge_lag`,
`primary_horizon` and `queue_bound_arm`. R-20's *other* clause pulls those into
Class D, so the combination closes the hole — but by two rules meeting rather
than by one saying it. **Proposed: the snapshot covers every input the bar
consumes regardless of class**, which is simpler and cannot be defeated by a
future row whose class nobody re-examined. DA has written the register as if
this holds (§4) because R-20's generalisation implies it; say so or correct it.

**10.9 — CORRECTED after iteration 3. The defect is real; DA's diagnosis of its
CAUSE and its cheapest ROUTE were both wrong.**

> **Correction 1 — this is NOT a hole in R-20, and DA's earlier framing would
> have caused harm.** R-8 had *already* generalised Class D to any row on which
> a documented **falsifier** turns; R-20's addition was **guards**
> ("guards were the gap"). `primary_horizon` carries a falsifier, so it was
> Class D under **R-8**, not R-20. And Class D is the **most restrictive**
> class — under A the same move is free, under B it costs only a re-run — so
> R-20 **raised** the price of this exploit. Telling the coordinator that R-20
> opened it invites a narrowing of R-20: a loosening for no reason. **The hole
> is in R-6 clause (c) and is class-independent.**
>
> **Correction 2 — the cheapest route is not h=60, and it needs no re-run at
> all.** `warning_window_v1.json` publishes R(250) at h=5, **15 and 30 — there
> is no h=60 arm**. eth's Layer-1 CI spans zero at **both h=15 and h=30**, so
> either zeroes eth's `f*_low`, breaks "DEAD only if BOTH verdict coins are
> DEAD", and vacates R-11 — using figures that already exist. **DA reported clause (a)
> as therefore "vacuous"; iteration 5 showed that was WRONG** — see the DRAFT
> caveat below. The structural defect in clause (c) is unaffected. Only btc h=60 has a
> recorded population-artefact defence; eth h=15/30 have none.

**The defect, restated correctly: Class D's clause (c) is a REWARD, not a cost,
because every frozen verdict in this programme is a REFUTATION.** R-6's original
three-clause test ends with *"explicitly invalidating every verdict computed under the old
bar"*. That is written for the case where the threatened verdict is a **pass**.
Here every Class-D verdict is a **DEAD**, so clause (c) is precisely what a
captor wants — the mechanism's cost clause is the objective.

**The amendment path, with the corrected routes.** Verified arithmetic:

| route | coin | `f*_low` | published `R(250)` | outcome | re-run needed? |
|---|---|---:|---:|---|---|
| h=5 (frozen) | btc / eth | 30.9 / 49.4 % | 15.3 / 14.3 % | DEAD / DEAD | — |
| **h=15** | **eth** | **0 %** | **15.0 %** | **NOT DEAD** | **no — but see the DRAFT caveat** |
| **h=30** | **eth** | **0 %** | **16.1 %** | **NOT DEAD** | **no — but see the DRAFT caveat** |
| h=60 | btc | 0 % | *no ww arm exists* | not DEAD | yes |

**DRAFT CAVEAT — iteration 5, and it weakens DA's own sharpening.**
`warning_window_v1.json` carries `protocol: ww_v1_DRAFT_PENDING_FREEZE`,
`status: RESEARCH_ONLY_NOT_DECISION_ELIGIBLE_BRANCH_NOT_EVALUATED`;
`edge_layer1_v1.json` carries `status: RESEARCH_ONLY_NOT_DECISION_ELIGIBLE`.
**DA's first framing said clause (a) was vacuous because the figures are
published. That was wrong. DA's SECOND framing — that the DRAFT status makes
them inadmissible — was ALSO wrong, and worse.** Both legs of the frozen R-1 bar — both from **one** receipt, `edge_layer1_v1.json`, not two; `warning_window_v1.json` supplies the verdict side, not the bar
come from these same two receipts (`CANCEL_POLICY_PROTOCOL.md:157`: "the six
numbers were checked against `derived/edge_layer1_v1.json`"). If a
`RESEARCH_ONLY` label disqualified figures, it would vacate **R-1's own bar and
R-11's DEAD/DEAD wholesale** — a cheaper amendment route than the one this
escalation exists to close. **The correct distinction: admissibility comes from
the FREEZE ACT, not the receipt label.** R-1 explicitly admitted the h=5 six by
freezing them; the h=15/30 arms have never been through any adoption act. So
clause (a) is not vacuous — not because the receipt is draft, but because those
arms are unadopted. **The
structural defect stands: clause (c) still rewards the amender.**

Amend `h` 5 s -> 15 s and the lever is **eth**, not btc: (a) contested — the
figures exist but are decision-ineligible; (b) a horizon argument, self-attested and
unfalsifiable; (c) invalidates R-11 — **which is the goal**. **btc stays DEAD at
every published horizon** (`f*_low` 30.9 / 21.8 / 25.4 % at h=5/15/30 against
`R(250)` ≈ 15.3 / 15.9 / 15.9 %), so the family falls through eth alone: **eth** stops being DEAD, "DEAD only if BOTH verdict coins are DEAD" fails, and the closed family reopens. **Nothing stands in the way on the eth routes.** The
population-artefact record exists only for btc h=60 — the route this escalation
originally and wrongly named; eth h=15/30 have no such defence.

**UPHELD — R-38, ruling against R-6's own clause (c).** The coordinator's finding: clause (c) was built as a deterrent, but deterrence only works when the standing verdict is one the amender wants to KEEP — and every frozen verdict here is a **refutation**, so an amender who wants one gone satisfies (c) *eagerly*. **Clause (d) as adopted: an amendment may not, by itself, change a verdict.** Invalidation renders a verdict **UNDETERMINED**, never the opposite verdict, and re-establishing one requires **re-running the measurement** under the new bar at the original evidentiary standard. You cannot amend from DEAD to alive — only to "not yet determined", and then you owe a measurement. DA's earlier "make Class D DIRECTIONAL" phrasing is superseded. **DA PROPOSAL, NO RULING STANDS BEHIND IT — relocated to §5a clause (f) and
filed as Q-DA-18.** Rev 12 stated the directional requirement here in mandatory
voice ("must re-run under both bars"), one sentence after conceding that DA's
directional phrasing was superseded, in a subsection the status table marks
CLOSED — and cited "the R-ADMISS both-arms rule" as its authority, which
**evaluates something else**: `contracts.yaml:93` is
`two_arms: on_fail == EXCLUDE_UNIT => required_gap_arm == BOTH`, unit exclusion
from a coverage population, not bar reporting. A rule whose text reads as a
mandate while its cited computable form never reaches an amendment is this
programme's recurring defect class, committed inside the clause written to close
a capture path. The proposal itself survives and is stated as a proposal at §5a
clause (f); the "do not reopen without genuinely new data" bar is **DE's protocol
text** (`CANCEL_POLICY_PROTOCOL.md:23`), **not R-11's**, so no ruling stands
behind that either. Tightening a bar and loosening one should not cost the same.

**10.10 The snapshot gap is ENUMERATION, not class — 10.8 was the wrong
diagnosis and DA withdraws it.** The frozen `ww_v1` bar also consumes the
queue-bound arm, the micro-class arm, the 500-fill VOID floor, `verdict_coins`
and the era/day set. **None is in R-20's recorded anchor**, and the receipt
(`warning_window_v1_dayseries.json`) already stamps more than the governing
document does. Proposed: the snapshot enumerates **every input the bar consumes,
of any class or type**, fixed at freeze time — and R-20's freeze record needs a
**contract carrier**, which §7 does not list. It currently exists as three
hand-maintained copies with no single source of truth.

**10.11 — WITHDRAWN by DA after iteration 3. The premise is false.** DA claimed the only written criterion was the ~35 % micro-share bar with "no document saying why sol is excluded". `FLOW_MODEL_PROTOCOL_V4.yaml:147-149` **freezes** `verdict_coins: [btc, eth]` with `restriction_reason: BRACKET_WIDER_THAN_ANY_SUPPORTABLE_CONCLUSION`, restricted 2026-08-21 by decision — a different rule from the micro-share bar, which governs whether the raw *count layer* is admissible. sol is `descriptive_only` by frozen rule and cannot enter a family verdict, so the attack does not run. Retained for the record only. *Original text:* The
only written rule is the ~35 % R-DUAL micro-share bar, and **sol sits at
29.7 %** — below it, yet excluded from the set with no document saying why.
Publishing `{btc, eth, sol}` therefore *follows* the rule, and Class C obliges
adoption; sol's `ww_v1` join-arm population is n=459 against the frozen 500-fill
floor, so sol is VOID, VOID is not DEAD, and the family verdict cannot be DEAD.
No value moves and no amendment is written. The four thresholds that determine
the domain have no register rows.

**10.12 R-8 CONTRADICTS ITSELF on the τ=250 rung — erratum requested.** R-8
generalises Class D to falsifier-bearing rows and cites *"R-1 already froze the
τ=250 ms decision rung"* as its precedent — while its own adopted text lists
`{0, 50, 100, **250**, 500}` as remaining **Class A**. Both cannot hold: the
frozen `ww_v1` verdict is computed on `R(τ=250 ms)`. §4 resolves it in favour of
R-1, but a worker document should not silently resolve a ruling's internal
contradiction. **ROUTED — R-35** to DE's `COORDINATOR_REVIEW_LOOP`, and **ANSWERED there; the ask no longer stands — but the OBLIGATION IT CREATED IS
OWED AND UNTRACKED (Q-DA-27).** That loop resolved Q-DA-6 by the strictest-alias reading (`COORDINATOR_REVIEW_LOOP.md:163-168`) and has since **CLOSED at its stop rule** (R-41). It answered the second half **differently from this row's guess**: the operational risk dissolves, but **R-8's ledger text remains internally inconsistent and DOES owe an annotation beside it under R-28** — the contradiction is *not* dissolved without an erratum. Rev 12 left this row soliciting a ruling that already exists. If one IS needed, R-28 makes it an **annotation beside** R-8, never an edit. `DE_MODULE_PLAN` and
`OP_PLANE_PLAN` both carry the Class-A version. **FALSE — CORRECTED at iteration 10.** `OP_PLANE_PLAN` §5.2 reads *"Interior rungs `{0, 50, 100, 500}` remain Class A … **`τ = 250 ms` is NOT among them**: it is the R-1-frozen Class-D decision rung"*, and its §8a table stamps `tau_decision_rung = 250 ms` as **D — VERDICT, frozen by R-1**; OPS logs that as its own fix. **`DE_MODULE_PLAN` is the ONLY carrier.** Q-DA-27's erratum request rests on a two-document spread that is one document — narrowed accordingly.

**10.13 `belief_a`/`belief_b` — the edge question needs a ruling, not a
deletion.** Rev 4 deleted the row claiming an `EV → DE` edge; the publisher is
**BE**, and R-6's Class-C table plus `BE_BELIEF_PLAN` both place the values in
`SP-Params`, so the deletion created an ownerless quantity. Restored in §4.
**RULED — R-37: BE fits and publishes; the SP owner (DA) writes the entry on adoption; EV does neither.** The deciding consideration is **direction, not authorship** — an `EV → SP` write is an edge out of a plane read by no one. `MEASUREMENT_PLAN` §3.3 updated. Also open: BE's two-object split (`â`
measured, the deployment pin `a := 0` chosen) must not collapse into one
parameter, and `provenance = fitted` is a member of no ruled enum. **AND THE PIN
IS AN UNREGISTERED CHOSEN VALUE, sitting inside an item the status table stamps
RULED (Q-DA-26).** `BE_BELIEF_PLAN.md` **§4.3, line 523** (Rev 19 cited `:377`, which is blank —
the file grew 146 lines) — *"### 4.3 Deployment policy: pin
a = 0"* — is a live deployment choice with published arms, and it has **no §4
row, no class, no owner line**. §4's `belief_a`/`belief_b` row points a reader
here for it, and this item reads closed, so nothing surfaces it: the
ownerless-quantity defect §1 opens by naming. Under R-37 the SP owner writes the
entry, so the ROW is DA's to add — but the VALUE is CHOSEN, and DA adds no row
for a value whose class it cannot set.

**10.14 The κ_$ / budget binding counts need re-deriving, and no item owned it.**
§5 withdrew Rev 1's "exercises both branches" claim — four markets at $50 is a
fifth of a $1,000 budget, and reads as an exact tie against the $200 scenario
limit — but the pointer sent the reader to an unrelated escalation twice.
**Partly discharged:** `de_constraints.py:20-24` already records the withdrawal
— **but READ ON BEFORE CITING IT: `:24-30` still asserts the oracle "aggregates
one $200 ceiling across open markets ... pending the coordinator's ruling on the
scope contradiction. No semantics change here until that ruling."** R-35 RULED
and the code was conformed (`:203-209` computes the per-scenario minimum; `:169`
says "SCENARIO-KEYED per Ruling R-35"), so the module docstring now contradicts
both the ruling and the function beneath it. A coordinator opening this anchor
to check "partly discharged" reads that their own R-35 is still pending. **Not
DA-owned — an R-33 clause-1 conformance break for DE (*CORRECTED at iteration
10: Rev 15 wrote "routed to DE" and no routing existed outside this sentence;
filed as **Q-DA-33***) — the ruling wins,
the docstring is stale.** The withdrawal record itself and its selftest was RELABELED "scenario-cap branch"; only the re-derivation remains, downstream of §10.15. **Requested: re-derive the counts, or state that the operative
set is not required to exercise both branches and fix the selftest's rationale.**

**10.15 — RULED (R-35), CLOSED.** Scenario-scoped; the architecture wins; DE
keys it by scenario. The coordinator identified the root cause as their own §5
shorthand rather than the register. Retained below as the evidence.

*Original:* the $200 limit is registered scenario-scoped and used portfolio-wide
by this document AND by the shipped oracle. The binding text is **`contracts.yaml` `AdverseScenario.notes`** — "Loss limit
lives ONLY in SP-Params keyed by `ScopeKey.scenario`". *(An earlier revision
quoted this as if from §4; §4's row now points back here, so the three arms are
the CONTRACT, §5's arithmetic, and running code — not the register.)* But §5's binding arithmetic reads
four markets against the $200 *in aggregate*, and `de_constraints.py:132-137` **as it stood before the R-35 conformance**
implements exactly that — `head_s = slimit - here - open_markets_l_adv`, one
ceiling summed across open markets, its selftest labelled "portfolio branch". **That describes the SUPERSEDED code and must not be checked against the file today:** `:203-209` is now `head_s = min(max(0.0, slimit - here - other) for other in scenario_losses.values())` — the per-scenario MINIMUM, not one summed ceiling — and the label is now `"scenario-cap branch"`, exactly as §10.14 records four rows above. Rev 13 re-aimed the line cite onto the conformed code, turning a historical claim into a false current one; a coordinator opening it would have judged either the escalation wrong or R-35's conformance unlanded.
Meanwhile the genuinely portfolio-scoped row (`capital_budget`) is read by
nothing. And `{scenario: *}` is not writable: `ScopeKey.scenario` is
`ScenarioId?` with no wildcard, so omitting it *is* the portfolio key the row
forbids. **Three-way disagreement between the register, the plan's own
arithmetic, and running code.** Needs a ruling before §10.14's re-derivation,
because the answer changes what is being re-derived.

**10.7 — NARROWED after iteration 3.** Its first limb attacked a sentence that
**exists nowhere in R-6 or the ledger** — DA wrote it and then argued with it.
R-6's actual Class-A cell reads "freely, any time; sweep it when asked, report
the range, not the best point", and the surviving substance (a Class-A parameter
sits inside a replay's data-generating process) was already actioned by R-20 for
`refuse_k`. **Only the second limb stands:** Class D binds "before the
measurement runs", which does not bind a NEW protocol testing the same claim —
charter `ww_v2` and its bar is freely cuttable. Proposed: bind the freeze to the
**claim**, not the protocol id. *Withdrawn limb, for the record:* Class A's
justification — "changing them changes what
the system *does*, never what is *true*" — is false for any parameter inside a
replay, because the parameter is inside the data-generating process of the
measurement. And Class D binds "only before the measurement runs", which does
not bind a **new protocol testing the same claim**: charter `ww_v2` and its bar
is freely cuttable. Proposed: bind the freeze to the *claim*, not the protocol
id.

**10.16 The expected-port registry has TWO CLASSES in two plans.** §4 reads it
**Class D** (§10.4 applied to a GUARD, under R-20's **un-numbered guard
generalisation**, not its numbered clause 2 which is the candidate-bar rule: the registry is the
*quantifier domain* of the staleness guard, so dropping a port silences a
monitor with no value moving). `OP_PLANE_PLAN.md` §8a reads it **Class A**
("a configuration fact: which modules must run"). *(Cited by section, not line:
Rev 13's `:587` was correct when written and OPS has since edited that file,
moving the row to 600 — and, re-measured at iteration 13, to **607**. The note about drift had itself drifted, which is the argument for section-and-quote citation making itself twice. A line number into a page another plane is actively
editing is unstable by construction — the inverse of R-10, where an address that
is not content-derived drifts when the content moves under it. Cross-plane
citations in this document now name sections and quote text.)* Both cannot hold, and the **A**
reading leaves available precisely the capture the **D** reading exists to stop.
Compounding it, Rev 12 recorded the D stamp in a **sixth cell of a five-column
table**, so the class was deleted at render and the row read as unreviewed under
§4's own rule — found independently by all three reviewers in iteration 7. Row
repaired; **the class conflict is escalated, not resolved**, because §5a states
that DA changes no class assignment absent a ruling. Note the D derivation rests
on §10.4, which is itself open.

**10.17 `ScopeKey` cannot express a per-port scope.** `staleness_deadline` and
per-port `period` are scoped per port; `port` is a `TelemetryPortId`; `ScopeKey`
(`contracts.yaml:251-260`) has eight members and **none of them is a port**.
Either `TelemetryPortId` is a `FeedId` — which should be stated, since
`ScopeKey` resolution FAILS LOUD on equally-specific matches — or `ScopeKey`
needs a ninth member. Both are contract changes. Until one lands, **two rows of
§4 cannot be written as keys at all**, which is a strictly worse condition than
the missing-`namespace` defect this iteration also repaired.

**10.18 Conditional classes have no detector and no ordering rule.** Two rows
carry a class that is a function of what other documents currently do:
`gamma_ladder` ("diagnostic only while no verdict rests on one rung") and
`LANE_PROGRESS` grace (`OP_PLANE_PLAN.md` §8a: **A** while nothing gates on it,
**D** if ever read as an admissibility threshold). R-8 supplies the
classification rule; nothing supplies a **detector** — there is no consumer→row
index — and nothing addresses **order**. A value may be moved freely while it is
Class A and a verdict built on it afterwards: at every instant the rules held,
and the result is a verdict conditioned on a hand-picked value with no re-run,
no freeze record and no amendment. Prospective, not live (γ feeds
`utility_none`), which is why it is an escalation and not a defect.

**10.19 Clause (e) has no carrier, and the A/B/C/D class has no machine surface
at all.** Detailed at §5a. Short form: "VACATED — was DEAD under bar X" has no
type to live on (`ParamValue` carries no verdict state; no freeze record
exists — §10.10); and **class appears in no contract type and in no `rules:`
key**, so every Class-D freeze in §4 is prose-enforced and no checker can detect
a Class-D value moving. Iteration 2 already caught that string going stale in a
published receipt. §7 should ask for a class field; it currently does not.

**10.20 The `Disputed` carrier cannot carry `Disputed`.** Detailed at §2. The
one named carrier returns `Known | Unavailable`, `FieldState` lives only on
`SpecRecord`, and §2 removed the field that would have held it — so the live
`Disputed` case in this programme (the rewards band, Gamma vs the CLOB registry)
can only be recorded by silently selecting one authority. Three candidate
contract changes, none of them DA's to pick.

**10.21 `SP-Params` is not a `ModuleOrPluginId`.** `ParamId.namespace` is typed
`ModuleOrPluginId` (`contracts.yaml:249`); v22's `modules:` block registers 25
ids — `BE-*`, `DA-*`, `DE-*`, `EV-*`, `OP-Monitor` — and **none is `SP-*`**.
`SP-Params` is a *type* (`contracts.yaml:422`), not a module. `ModuleOrPluginId`
is `external`, so nothing rejects the string and the key space is at least
self-consistent — which is why §4 adopts it as the working convention rather
than blocking on it — but the field exists to name an owning module and this
value names none. §7 item 1 asks for "an SP spec-resolver module/port" without
giving it an id. **That id and this namespace should be the same string**, and
DA does not choose module ids.

**10.22 Three register rows carry more than one parameter, so they cannot be
keyed.** `belief_a`/`belief_b`; STOP's `h` / fee treatment / cancellation policy
(three, and the last two are prose from which no `name` is derivable); and the
expected-port registry · `LANE_PROGRESS` grace · `period` bundle. One row is one
`name`, so each bundle currently resolves to a single key covering several
quantities — the shape iteration 2 fixed on `tau_ladder`, where two values
sharing one key **were** one entry. Splitting them assigns names, classes and
owners, so it is escalated. Two consequences worth naming: **`period` is
registered twice** (its own row and inside the bundle), yielding two names for
one quantity with nothing asserting their equality; and if STOP's `h` **is**
`primary_horizon` — the residual §4's STOP row leaves open — the two names give
two independent entries with **two different owners**, and §6's strictest-alias
rule has no pair of names to bind them across.

**10.23 Scope is not a movement — and Rev 14's own fix opened the door.**
Detailed at §5a clause (g). Short form: adding `(name, ScopeKey{instrument: I})`
beside a Class-D entry at `ScopeKey{}` changes what every consumer for market `I`
resolves, while the frozen row never moves, no clause fires, and R-20 clause 1's
snapshot cannot see it because it records value/`artifact_id`/provenance and not
scope. Proposed as clause (g); **DA self-binds in the interim** and will refuse
such a write.

**10.24 An unregistered CHOSEN value inside an item stamped RULED.**
`BE_BELIEF_PLAN.md` §4.3 (line 523, not the `:377` Rev 19 cited)'s deployment pin `a := 0` has no §4 row, no class and no
owner. Detailed at §10.13.

**10.25 R-8's owed annotation is untracked, with every surfacing channel closed.**
DE's coordinator review resolved Q-DA-6 by the strictest-alias reading and found
that **R-8's ledger text still owes an annotation-beside under R-28**. That loop
has since CLOSED, Q-DA-6 is closed, §10.12 now reads "the ask no longer stands",
and the erratum directive is one of the six never-executed items. Meanwhile R-8's
text still reads *"Interior rungs `{0, 50, 100, 250, 500}` remain **Class A**"* —
the permissive reading of a frozen verdict rung — and `DE_MODULE_PLAN.md:461`
carries the same. Nothing tracks the annotation; nothing will surface it again.

**10.26 The "does an instance of this id type exist?" question was asked once.**
§4 establishes that **no `FeedId` value is named anywhere** and concludes two
Class-D rows are unwritable. §10.21 meets the identical premise on
`ModuleOrPluginId` and concludes the opposite — the string is fine because
nothing rejects it. Same defect, two incompatible resolutions. The question was
never asked for the other six axes: `VenueId`, `PortfolioId`, `RegionId` and
`RiskFactorId` are all `external` prelude primitives and **no instance of any is
named anywhere in `pm_research/`**, and `InstrumentId` is a struct whose
mandatory `venue: VenueId` inherits the blockage. On the §4 reasoning the
unwritable-row count is most of the register; on the §10.21 reasoning it is zero.
**One rule, applied to all eight axes, and the count re-stated under it.**

**10.27 Two of the frozen research action's components are unregistered.**
`BE_FLOWANDFILLS_MODEL_PLAN.md:359-361` defines
`A = (coin, slug, start_time, horizon, maker_side, level_up, size_shares,
queue_rule)` — the object every measured number in this programme is conditional
on. §4 registers `horizon`, `size_shares` and `queue_rule` as Class D.
**WITHDRAWN IN SUBSTANCE at iteration 11 — the central claim was FALSE, and
DA had already withdrawn §10.11 for the identical error in the identical file.**
Rev 17 said `level_up` and `maker_side` have *"no snapshot obligation"*. They are
**pinned BY VALUE in the frozen protocol**: `FLOW_MODEL_PROTOCOL_V4.yaml:151-152`
carries `primary_style: JOIN_TOUCH` and `maker_sides: [BUY_UP, SELL_UP]`, and
`V5.yaml:337-338` the same — **three lines below `verdict_coins`**, which is the
freeze this plan cites in §8 and used to withdraw §10.11. `level_up` is a `float`
(`contracts.yaml:1435`) determined by `primary_style`, not a free parameter. The
supporting evidence was mismatched too: the JOIN n=10,387 vs FRONT n=56,053
argument is about `queue_rule`, which §4 **does** register as `queue_bound_arm`,
Class D. **Surviving residue, and it is a SHOULD not a coordinator ask:** neither
has a §4 row, which §9's populate-on-demand rule already governs; no verdict is
at risk because the values are snapshotted. Superseded text:
`level_up` and `maker_side` have no row, no class, no owner, no snapshot
obligation. The plan already knows placement decides which population the bar
is computed on — `JOIN_BBO` n=10,387 against FRONT n=56,053, 5.4x — and §10.10,
whose whole purpose is enumerating every input the bar consumes, omits it.
Related: `ScopeKey` has no **side** member either, a question asked for `port`
(§10.17) and never for side.

**10.28 `SP-Strategy`'s ten structural fields have no taxonomy.**
§2 assigns `SP-Strategy` ten fields as "structural choices". **FOUR** —
`coupling`, `action_space`, `impls`, `nulls` — occur exactly once in this
document, the §2 table cell, never classed, deferred or escalated. *(Rev 15 said
"six … occur exactly once"; measured word-boundary counts give four, because
`solver` and `unwrap_policy` occur a second time **inside this very paragraph**,
which also escalates them — both halves of that sentence were refuted by the
sentence stating them, and Q-DA-30 carried the figure stamped "verified". The
substance — six fields have no taxonomy — is unaffected.)* R-6's own Class-D examples are protocols and rules, so
the taxonomy is not scalar-only by construction. The consequence is live:
**`DE_MODULE_PLAN` §6.2 lists `unavailable_policy`/`unwrap_policy` under "pure
config (no contract edit)"** — free — while SP §5 says `unavailable_policy` is
what **halts `RulePolicy_v1` at the first decision**, and under R-20's guard
generalisation a value a halt turns on is Class D. Two plans, one value, two
classes, and unlike §10.16 nobody has noticed. `solver` is worse: moving
`RulePolicy_v1` invalidates every replay exactly as `quote_size_pin` does, with
no row and no re-run obligation. Either bring `SP-Strategy` into §4 with classes
and owners, or state that structural choices are outside the taxonomy and say
who owns them — the current silence hands DE a free hand over a halt.

**10.29 `τ = 500 ms` is a frozen VERDICT rung and a free CHOSEN grid value at
once.** Detailed at §4. `WW_EBX_PROTOCOL.md` §1 (FROZEN, R-51) makes 500 the
**primary verdict rung** and R-54 issued `DEAD_4CH` on 8/8 cells there, while
§4's `tau_ladder_rungs` row calls the interior rungs "resolution only" and leaves
them CHOSEN. R-8's generalisation settles the direction; the class move is the
coordinator's. `OP_PLANE_PLAN.md` §8a carries the same stale cell, so the fix is
two documents. **The loop's founding defect one rung up:** a lever survives if
the rung it is measured at stays freely movable.

**10.30 A FROZEN protocol carries the pre-R-20 class of `quote_size_pin` and
cites SP §6 as its authority.** Detailed at §6. `POLICY_BOUNDS_PROTOCOL.md` §3
(frozen R-45, after R-20) calls it a **Class-B** robustness probe;
`DE_PLACEMENT_POLICY_PLAN` propagates "the Class-B answer". SP is correct and the
permissive reading sits in the frozen document. Under R-28 the remedy is an
**annotation beside**, not an edit, and it is not DA's to write.

**10.31 `tau_operative` registers a BOUND; two consumers read a RUNG.** Detailed
at §4. `OP_PLANE_PLAN` §5.1 and `DE_MODULE_PLAN` §5 both define the operative
value as *the smallest ladder rung ≥ the measured bound*, stored in SP-Params and
consumed by the Actuator. SP registers the bound itself, so an Actuator reading
"the SP-Params value" gets the raw bound rather than the rung — **optimistic, the
one direction OPS says the seam must never degrade.** Two rows, or one row plus a
stated rounding rule; either is a register change, not DA's to pick alone.

**10.32 `D̂` has two carriers across two DA-owned documents, and R-6 ratified the
one SP deleted.** Detailed at §4. `MEASUREMENT_PLAN` §1.2 reads `D̂` from
`SP-Params.at(t_event)`; R-6 §4a lists `D̂` among Class-C SP-Params rows; SP
deleted the row in favour of `ImputationRule`. SP is right on the contract and
wrong to have resolved a ratified row unilaterally. One answer, one place.

**10.33 + 10.20 — SELF-RESOLVED BY DA UNDER R-33 CLAUSE 3 (stale ask in your
own plan) at iteration 11. NO RULING IS OWED; the ask is withdrawn.** Three
verified facts dissolve it: (a) `PM_ARCHITECTURE.md` declares
`SP-Instrument{ settlement{…}, T, payoff, complement, **incentive_contract** }`;
(b) this plan's header says *"for rules, `PM_ARCHITECTURE.md` wins"*; (c) §2
states each of `SP-Venue`/`SP-Instrument`/… **is a `SpecRecord`**, and
`FieldState.Disputed` lives only on a `SpecRecord` (`contracts.yaml:1268-1272`).
So the carrier the architecture already mandates is exactly the one that makes
`Disputed` expressible, and it needs **no contract change beyond §7 item 1**,
already finalised in `CONTRACTS_BATCH_v23` §2. §10.20's premise — *"the rewards
band sits on no `SpecRecord`"* — held only because **SP §2 removed the field
unilaterally**. Both items close; the register row is corrected. What remains is
DA's own edit: restore `incentive_contract` to §2's `SP-Instrument` cell, which
§2 now does. *Original framing retained below.*

**10.33 §10.20 escalates restoring a field `PM_ARCHITECTURE` never removed.**
Detailed at §2. The architecture still declares
`SP-Instrument{…, incentive_contract}`, and this plan's header says the
architecture wins on rules. §10.20 asks the coordinator to choose among three
contract changes, one of which is restoring that field — a ruling spent on a hole
that may not exist. Re-scope §10.20 before ruling it.

**10.34 A 2× fee error is still live in the rules authority.**
`PM_ARCHITECTURE.md` carries `fee_schedule = (PIECEWISE_MINPQ, {rate,
size_rounding})` verbatim, which SP §2 refutes by arithmetic (3.50 ¢/share
against a measured 1.75, turning ~225 bps into ~400 bps). SP corrected its own
copy and blamed Rev 1 rather than the source. Since §2 says the record types
exist *only* in that file's prose, the uncorrected string is the definition a
builder reads. Routed, not editable by DA.
