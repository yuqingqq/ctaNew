# Registry-amendment PROPOSAL — EV-Replay, DE-ActionSpace, OP-LatencyBudget

**STATUS: PROPOSAL. Nothing here is applied.** `contracts/contracts.yaml`
wins on types (EV_REPLAY_PLAN precedence banner); this file is a draft that
lands only by coordinator/USER act after review. **The DE seat does not edit
`contracts.yaml`.**

**Drafted by:** DE (seat staffed 2026-09-01, R-379). **Raised by:** the
coordinator's R-379 audit finding — *EV-Replay has ZERO presence in
contracts.yaml v24, no module entry and no types, while Phase 4 runs inside
that seam.* **Verified at the artifact, not accepted from the dispatch**
(rule 1): every claim below is a predicate in
`de_registry_amendment_check.py`, which reads THIS FILE's YAML blocks and the
real `contracts.yaml`, and ships falsifiers in both directions (rule 15/16).

**As-of 2026-09-01T13:38:54Z; repo HEAD `d929031`; `contracts.yaml`
version 24.**

---

## 0. What the audit got right, and the one thing it did not see

**Right, and reproduced by execution:** `EV-Replay` is absent from `modules`;
no type carrying the replay vocabulary exists (`RunRecord`, `ReplayReceipt`,
`ReplayWindowSpec` — none); `DE-ActionSpace` is absent from `modules` while
the type `ActionSpace` exists (`contracts.yaml:1201`) and the code exists
(`de_actionspace.py`); `OP-LatencyBudget` is absent from `modules`.

**Not silent, one file over.** EV-Replay module records are **named debt with
a named trigger** in `CONTRACTS_BATCH_v25.md` §3: *"enters by ruling when the
plugin path first demands a module record."* So the correct finding is not
*forgotten*; it is that **the declared trigger names the wrong demand.** The
plugin path is still unbuilt (EV_REPLAY_PLAN §2 marks it NOT IN v1 twice),
so on the trigger as written the record would never enter — while Phase 4's
economics already run inside the seam through a replay dialect
(`harmful_stateful_policy.py`) that the plan's own census does not count.
**The proposal below therefore restates the trigger; restating it is a
ruling, not a repair, and it is asked for, not taken.**

**Two stale carriers found while verifying this, both citation-only:**

| carrier | says | truth |
|---|---|---|
| `plans/EV_REPLAY_PLAN.md` §5 (Rev 10) | the candidate is in `CONTRACTS_BATCH_v24.md` §3 | that file was renamed to `CONTRACTS_BATCH_v25.md` at `459c3b1` (R-121). Rev 10's own stale-carrier repair re-pointed at "the v24 accumulator", which had already been retargeted |
| `de_actionspace.py` (`contract_conformance` comment) | the v23→v24 change record is drafted in `CONTRACTS_BATCH_v24.md` | same rename; **`459c3b1` edited this very file and left the citation behind** |

`de_actionspace.py` is DE's, and its comment is corrected in this batch. The
plan file is not mine to edit; it is filed for its owner. **Neither correction
restarts any clock (rule 3): both are citations, and no number moves.**

---

## 1. EV-Replay — the module entry and the types it needs

### 1.1 Every type is derived from committed code, never from the prose

The plan describes `RunRecord` in §2's box; the **code** is what this
amendment records. Source of record, cited by SYMBOL rather than by line so the citation
cannot drift: `ev_replay.RunRecord`, `ev_replay.ReplayEnv.run`,
`ev_replay.ReplayEnv.receipt`, `ev_replay.record_hash`.
Where the plan and the code differ, the code wins here — a registry that
records an aspiration cannot be checked against a run.

**AMENDMENT A — additive types.** Verbatim-ready for `contracts.yaml:types`:

```yaml
# DE-AMENDMENT-A
ReplayFill:
  fields:
    t: Timestamp
    maker_side: str
    level: float
    size: float
    mid_at_fill: float
    aggressor_micro: bool
  notes: one maker fill as the environment emits it (ev_replay.py:82-98); RAW,
    carrying no markout and no evaluation field
UnavailableInterval:
  fields:
    t0: Timestamp
    t1: Timestamp
  notes: a collector-gap interval during which state was killed and resting
    quotes retracted; first-class in RunRecord because an absence must be a
    status, never a silent omission
ReplayWindowSpec:
  fields:
    slug: str
    inputs_hash: Hash
  notes: 'the environment takes an EXPLICIT window list and stamps it, never
    chooses (EV_REPLAY_PLAN section 2: selection is an R-ADMISS decision the
    coordinator ratifies). inputs_hash covers the gap list and token ids, which
    shape records and previously moved run_hash unattributably.'
RunRecord:
  fields:
    slug: str
    coin: str
    arm: str
    queue_bound: enum:FRONT|BACK_DISPLAYED
    fills: list[ReplayFill]
    mid_t: list[Timestamp]
    mid_v: list[float]
    unavailable_iv: list[UnavailableInterval]
    diagnostics: dict[str, int]
    record_hash: Hash
  notes: 'RAW events of one window replay, carrying NO evaluation field by
    design (EV_REPLAY_PLAN section 0 rule 1: markout, calibration, gate
    verdicts and attribution are computed after a run returns, never inside
    the loop and never visible to the solver). record_hash covers the FULL
    record including the mid path and interval endpoints the receipt body does
    not serialize.'
ReplayReceipt:
  fields:
    protocol: str
    status: str
    engine_hash: Hash
    state_lag_s: Duration
    quote_size_shares: float
    seed: int
    collector_era: str
    sp_parameter_set: dict[str, any]
    windows: list[ReplayWindowSpec]
    records: list[RunRecord]
    gates: dict[str, any]
    provenance: Provenance
    run_hash: Hash
  notes: 'the stamped identity of one replay run. Gate outcomes live IN the
    receipt, not in stdout, so a PASS cell is checkable at the artifact rather
    than by entailment from fail-loud ordering. status is
    RESEARCH_ONLY_NOT_DECISION_ELIGIBLE in v1 and no field here is
    decision-eligible: models estimate, policy decides.'
GenerationTrancheTable:
  fields:
    unit: enum:GENERATION
    latency_ms: int
    estimand_horizon_s: Duration
    estimand_note: str
    markout_s: Duration
    n_generations: int
    n_rows_consumed: int
    rows_per_generation: float
    generations: dict[str, GenerationTranche]
  notes: 'the Phase-4 feed (R-165(2) item 5): generation-level tranche tables,
    NOT per-row latency labels, because a generation is the cancellable unit
    and several rows share one outcome (measured 1.99 rows/fill, max 23).
    estimand_horizon_s is REQUIRED and carried, so a downstream cell cannot
    lose the 1 s cap it inherited; the emitter refuses to produce this record
    without an explicit declaration of the cap.'
GenerationTranche:
  fields:
    n_rows: int
    preventable_value_cents: float
    preventable_shares: float
    stale_shares: float
    t_start_min: Timestamp
    t_start_max: Timestamp
    coin: str
    day: Date
  notes: one cancellable generation's value AT ITS FIRST CROSSING, never the
    sum over its rows; summing would count one outcome once per row
```

**Not proposed, deliberately.** The parity canon's trajectory/event vocabulary
(`replay_traj_canon_v1`, `Event`, `Trajectory` in
`da_replay_parity_battery.py`) is **DA's**, and B3.5 sets the precedent that
membership may be drafted while spelling belongs to its owner. Naming it here
would be DE spelling DA's types into the registry without DA in the room.
It is named as **owed**, in §4.

### 1.2 The module entry

**AMENDMENT B — additive module record.** Verbatim-ready for
`contracts.yaml:modules`:

```yaml
# DE-AMENDMENT-B
EV-Replay:
  consumes:
  - ReplayWindowSpec
  produces:
  - RunRecord
  - ReplayReceipt
  ports:
  - read_all
  notes: 'offline replay ENVIRONMENT (architecture section 9): Live/Replay/Sim
    are implementations behind the same ports, and the replay clock plus tape
    port go to the replay runner alone. EV reads all planes and is read by
    none, so no output of this module may enter a policy loop: evaluation runs
    as a separate pass over a completed RunRecord. Window selection is NOT this
    module job and is supplied, never chosen. The registered-plugin path and
    artifact resolution are NOT IN v1; a DecisionProblem consume edge enters
    only when that path is built, and adding it earlier would record a seam
    that does not exist.'
```

**And one line of `config_supplied`,** because the invariant that every
consumed type has a declared producer is exactly the property this seam
should satisfy or explain:

```yaml
# DE-AMENDMENT-C
config_supplied:
- ReplayWindowSpec
```

`ReplayWindowSpec` is supplied rather than produced **as a statement of the
sampler rule**, not as a convenience: making a module produce it would create
a registry-level licence for an environment to choose its own windows, which
is the sampler defect the plan names (`FLOW_MODEL_STATE.md` §1f) with a
contract behind it.

### 1.3 The trigger, restated — and this is the ask, not the act

| | trigger of record (`CONTRACTS_BATCH_v25.md` §3) | proposed |
|---|---|---|
| condition | the plugin path first demands a module record | **a replay seam carries a result-bearing number** |
| status today | not fired; plugin path NOT IN v1 | **fired** — Phase 4's latency x cost x budget grids run through `harmful_stateful_policy.py` and `phase4_generation_tables.py`, neither of which has a module record |

The reason to prefer the second is the one R-379 states: a quantity with no
owner is this programme's oldest defect class, and the plugin path is a
*mechanism* trigger where the defect is about *results*. **A ruling is
required. The DE seat does not rule on the accumulator's own trigger.**

---

## 2. DE-ActionSpace — reconciliation, and a type with no owner

### 2.1 The name is not the definition, a sixth time

`ActionSpace` (`contracts.yaml:1201`) is a **verb menu**:
`verbs: list[enum:QUOTE|CANCEL|MINT|MERGE|CROSS|WAIT]`, `derived_from:
VenueCapabilities`. `de_actionspace.enumerate_actions` returns a
**`list[Action]`** — concrete, sized, side-keyed, order-ref-bearing actions,
i.e. an `ActionSet`. **The module named DE-ActionSpace does not produce the
type named ActionSpace.** Recording it as though it did would be the
name-is-not-the-definition defect (five instances on record) with a registry
entry to make it durable.

### 2.2 Two types in v24 have no producer at all — found while checking this

Measured over the whole `modules` block: **no module produces `ActionSpace`
and no module produces `VenueCapabilities`.** `DecisionProblem.actions:
ActionSpace` therefore names a field nothing in the registry can fill, and
`ActionSpace.derived_from: VenueCapabilities` names an input nothing derives
it from. `contract_check.invariants()` cannot see this: its producer check
runs over `modules[*].consumes`, and `DecisionProblem` is `config_supplied`,
so a field of a config-supplied type is never asked who fills it.

**Scoped, because the raw set is large and mostly benign.** The same
detector returns **20** entries over v24, and most are nested value types — a
`Position` inside a config-supplied `PortfolioState` arrives with the record
that contains it and needs no producer of its own. **Membership alone is not
the finding.** What singles `ActionSpace` out is a conjunction, checked as
one: *no producer* **and** *built code* (`de_actionspace.py`) **and** *no
module record*. That is an unwired seam, not a nested value — R-379's own
defect class, one type over from where the audit pointed.

**Reported, not fixed:** whether the venue-capability axis is built, folded
into SP, or struck is a design decision with an owner who is not DE. The
amendment deliberately leaves the gap open, and the check asserts that it is
still open after the amendment applies — a proposal that quietly closed a
decision it only had standing to report would be the worse outcome.

### 2.3 The proposed entry, recording only what the code does

**AMENDMENT D — additive module record.**

```yaml
# DE-AMENDMENT-D
DE-ActionSpace:
  consumes:
  - HaltState
  - PortfolioState
  - FeasibleSet
  produces:
  - ActionSet
  ports:
  - state_view
  - telemetry_out
  notes: 'the action MENU enumerator: given halt state, position, resting
    quotes and the feasibility oracle side-keyed map, it emits the FINITE menu
    of venue-EXPRESSIBLE actions the solver may choose among. The oracle says
    what is feasible; this module says what is expressible, and the menu is
    the intersection. It owns ONE fact: the venue floor orderMinSize = 5, so
    an action the oracle caps below the floor is feasible and INEXPRESSIBLE
    and is omitted as a typed fact, never truncated silently. Menu
    construction is DEFAULT-DENY end to end. It does NOT produce the type
    ActionSpace, which is a verb menu derived from VenueCapabilities: the
    module emits concrete Actions, and conflating the two would put the
    name-is-not-the-definition defect into the registry.'
```

**AMENDMENT E — NON-ADDITIVE, needs a migration record and a ruling.**
`ActionSet` is currently in `config_supplied`; with D applied it has a real
producer, and leaving it in both places states that the same object is
simultaneously supplied and produced.

```yaml
# DE-AMENDMENT-E
- from_version: 24
  to_version: 25
  operation: remove
  key: config_supplied:ActionSet
  old: "config_supplied"
  new: "produced by DE-ActionSpace"
  reason: 'DE-ActionSpace (amendment D) produces the action menu from
    committed code (de_actionspace.enumerate_actions). A type that is both
    config-supplied and produced has two authorities and no way to tell which
    one a consumer read.'
```

This one is **held, not urged**: `BE-FlowAndFills.consumes: ActionSet` today
resolves through `config_supplied`, and moving the authority changes what a
BE consumer is entitled to assume. It belongs in the accumulator's §1 with
the other non-additive records, behind BE's confirmation.

---

## 3. OP-LatencyBudget — DEFERRED WITH A NAMED TRIGGER, not absent

`OP-LatencyBudget` has an owning plan (`OP_PLANE_PLAN.md` §5), a decomposition
into four legs, and a frozen Class-D bar (R-8: an ack bound above 1000 ms
kills the cancellation lever at deployment independent of any replay result).
It has **no code, and legs 2-4 are unmeasured** — leg 4 (`confirm`) is *not
observable at this venue at all*, which is why `OrderRecord.t_acked` is
nullable.

**No module record is proposed, and that is a positive statement.** Per R-55,
`tau_operative` stays UNMEASURED until an Actuator exists; per `SP_PLANE_PLAN`
§10.31 the parameter needs **two rows or one row plus a stated rounding rule**
(a bound and the ladder rung above it), because two consumers today read a
rung where the registry offers a bound — and the failure is optimistic, the
one direction OPS says this seam must never degrade. Recording a single
`LatencyBudget` type now would freeze the ambiguous shape.

> **DEFERRED WITH TRIGGER.** `OP-LatencyBudget` enters the registry when
> **a DE-Actuator exists and a deployment produces an ack-latency upper
> bound** — the same instant `tau_operative` stops being unmeasured. Recorded
> so absence is not later mistaken for review (the accumulator's own §3
> discipline).

---

## 4. What this proposal does NOT do

- It does not edit `contracts.yaml` or `migrations.yaml`. Every record above
  is verbatim-ready and unapplied.
- It does not rule on the EV-Replay trigger (§1.3), on amendment E (§2.3), or
  on the ownerless `ActionSpace`/`VenueCapabilities` axis (§2.2).
- It does not name DA's trajectory canon into the registry (§1.1). **Owed:**
  the parity vocabulary that `harmful_stateful_policy` and
  `da_replay_parity_battery` already exchange has no registry presence either;
  it needs DA's spelling and a joint record.
- It introduces **no number**. Every value cited exists in committed code or
  in a frozen plan.
- It is not a claim that EV-Replay's harness is validated. `ev_replay.py`
  carries `status: RESEARCH_ONLY_NOT_DECISION_ELIGIBLE`, and §4.1 of its plan
  records that golden-window parity **structurally cannot fail in the v1
  form**. A module record makes the seam ownable; it does not make it green.

## 5. The check behind every claim here (rule 15)

`de_registry_amendment_check.py --selftest` evaluates each claim as a
predicate, reads the YAML blocks **out of this file** rather than holding its
own copy (so the doc and the check cannot drift), and ships both directions:

- **positive controls** — the unamended `contracts.yaml` passes
  `contract_check.invariants()`; the amended document also passes; the amended
  document REMOVES the two producer gaps it claims to remove.
- **known-bads that must FAIL** — an amendment referencing an undeclared type
  must raise an unresolved-reference error; a module record consuming a type
  with no producer and no `config_supplied` entry must be caught; a doctored
  copy of this file with a block deleted must make the check refuse rather
  than pass on the blocks that remain.
