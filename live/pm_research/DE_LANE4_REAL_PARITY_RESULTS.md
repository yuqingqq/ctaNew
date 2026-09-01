# Lane-4 seven-arm parity, run against the REAL QR_SKEW_ONLY shadow

**GENERATED from the receipt by `de_lane4_results_doc.py` — do not hand-edit.** Every number below is read from
`data/pm_5min/derived/de_lane4_real_parity_v1.json`; the selftest regenerates and compares byte-for-byte, so the document and the artifact cannot drift.

**Protocol** `de_lane4_real_parity_v1` · **status** `VERIFICATION_ONLY_NO_ECONOMICS_READ` · **as-of** 2026-09-01T14:06:39Z · **elapsed** 1339.6 s

**THIS IS VERIFICATION, NOT SCORING.** No economics may be read from it: the receipt structurally excludes the economics block, the fills block and the per-cancel records, and the standing hold (HANDOFF item 4) forbids PnL, capacity, promotion and forward verdicts.

## 1. What ran, and on what

| | |
|---|---|
| population | `v3_4_consumed_fragment`, era `clob_v3_1`, coins btc, eth |
| windows selected | **471** |
| windows excluded for Binance discontinuity (before selection) | 3 |
| UTC days | 2026-08-24, 2026-08-25 (n=2) |
| windows ADMITTED | **471** (100.0% of selected) |
| window exclusions, as counted statuses | none |
| generations ADMITTED | **826,238** |
| generation exclusions, as counted statuses | {'ZERO_LENGTH': 6} |
| stub score events | 1,652,476 |
| cancels issued (active-stub cell) | **35,083** |
| windows where the policy ACTED | 471 |
| fills charged STALE inside the latency window | 616 |
| aggregate gate digest | `b65b74e82010000987e70865a4ad6f88…` |

**Every exclusion above is a counted status, never a silent drop (rule 4), and every population carries its n and its as-of (rule 8).** The battery REFUSES a vacuum in both directions: zero admitted windows, and zero cancels issued — because every lifecycle gate would then pass on an empty set.

## 2. The gates

| gate | pass | failing windows |
|---|---|---|
| `gate_cancel_and_hold_equivalent` | **PASS** | 0 |
| `gate_disabled_bit_identical` | **PASS** | 0 |
| `gate_inf_equals_disabled` | **PASS** | 0 |
| `gate_infinite_threshold_bit_identical` | **PASS** | 0 |
| `gate_invariants` | **PASS** | 0 |
| `gate_lifecycle_closed` | **PASS** | 0 |
| `gate_one_cancel_per_generation` | **PASS** | 0 |
| `gate_rate_identity` | **PASS** | 0 |
| `gate_stale_fills_inside_latency_window` | **PASS** | 0 |

**ALL GATES PASS: True.**

The two anchors the LANE4 spec calls bit-identical are bit-identical at real-data scale: a disabled predictor and an infinite cancel threshold each reproduce the QR_SKEW_ONLY passthrough event for event, and the two equal each other (so score evaluation is provably side-effect-free). Cancel-and-hold equivalence is checked against an **independently constructed** trajectory — written from the declared semantics, not by calling the machine's own event builders.

## 3. Arm runnability — reported, never dropped

| arm | status |
|---|---|
| `CONDVALUE_NEUTRAL` | `NO_NEUTRAL_REFERENCE` |
| `CONDVALUE_X_SKEW` | `NO_RELEASED_PREDICTOR` |
| `CONDVALUE_X_SKEW_X_FAIRPRICE` | `NO_RELEASED_PREDICTOR` |
| `HAZARD_ONLY_NEUTRAL` | `NO_NEUTRAL_REFERENCE` |
| `QR_CANCEL_HOLD_X_SKEW` | `RUNNABLE` |
| `QR_SKEW_ONLY` | `RUNNABLE` |
| `RANDOM_MATCHED` | `NO_CONTRACT_IDENTITY_FOR_AN_ACTING_CONTROL` |

**2 of 7 arms are runnable on the frozen reference**, and the reasons are missing INPUTS, not a missing predictor. See §5.

## 4. The contract leg — DE's own exporter through DA's own loader

Run on one window (`btc-updown-5m-1787579400`): the contract's value here is its REFUSAL surface, which one window exercises as well as 471 do.

- inert arms admitted: **7 of 7**, refusals: none
- `inactive_predictors_agree`: **True** — every submission with the predictor off is bit-identical, whatever its arm
- `pass`: **True**
- DA canon `replay_traj_canon_v1`; DA's `ARMS` tuple matches DE's exactly: **True**

**The repost-axis diagnostic, run as a matched pair** (two cells differing only in whether the policy reposts):

| cell | reposts | `no_fill_after_effective` | `pass` |
|---|---|---|---|
| reposting | 58 | **True** | False |
| no_repost_permanent_hold | 0 | **True** | True |

**Diagnosis holds: False** — see finding C2.

**Both readings of a STALE cancel, neither sound** (finding C1):

| reading | requested | effective | suppressed | identity holds |
|---|---|---|---|---|
| `AS_SUPPRESSED` | 111 | 58 | 53 | True |
| `DROP` | 111 | 58 | 0 | False |

## 5. Declared parameters, and the code that produced this

```json
{
  "active": {
    "cancel_effective_latency_ms": 50.0,
    "charge_reset_cost_at_generation_start": false,
    "enable_reduce": false,
    "max_cancels_per_minute": "inf",
    "predictor_enabled": true,
    "protection_mode": "ALL_ORDERS_OVERRIDE",
    "queue_reset_cost_cents": 0.1,
    "repost_dwell_s": 0.5,
    "repost_fill_model": "REFERENCE_FILLS",
    "theta_cancel": 0.9,
    "theta_repost": 0.1
  },
  "permanent_hold_theta_repost": "-inf",
  "stub": {
    "early_frac": 0.25,
    "late_frac": 0.75,
    "late_score": 0.0,
    "note": "sha256-derived STUB, not a model; no predictor is released for this use",
    "salt": "de_lane4_real_parity_v1"
  }
}
```

**Code identity, taken AT IMPORT** (a long run outlives edits to its own source; hashing at receipt-write time would stamp the receipt with code that did not produce it):

| file | sha256[:16] |
|---|---|
| `da_replay_parity_battery.py` | `66a175fb7ca4ec06` |
| `de_lane4_real_parity.py` | `942d74e927d389ef` |
| `harmful_exposure_rows.py` | `1bbd8e7525fc27ac` |
| `harmful_stateful_policy.py` | `83315157d7fd338d` |
| `policy_optimizer_queue_realistic.py` | `ae13af2931bc33b8` |

## 2. Two defects in the state machine, both found by real data, both fixed red-first

`harmful_stateful_policy.py` was **gated green at the synthetic level** (R-165:
78 checks, all seven TODO §6 parity gates with both falsifier arms). Real data
found two defects its fixtures could not produce, because both need a shape
that only a real tape has: **consecutive generations that abut exactly**, so
that `GEN_START` of the next generation outranks `GEN_END` of the current one
at the same instant.

**Neither defect moves a committed number.** `de_lane4_real_parity.py` — written
today — is the ONLY importer of the module in the repository (verified by
search); `be_trajectory_export.py` cites it in comments and imports nothing.
Phase 3 was explicitly parked as "full-tape integration is Phase-4 work".

### D1 — the guard fired on a case that was correct

`btc-updown-5m-1787580000/BUY_UP/gen 449`: its only tranche lands at
`198.186235413`, which is **exactly its own `t1` and gen 450's `t0`**. The side
was HELD when 449 started (`GEN_START_MISSED_HELD`, no policy record) and had
just become repost-eligible; gen 450's start is processed first, reposts, and
clears `held` and `was_eligible`. The fill of 449 then arrived at a side that
was no longer held, on a generation that was never joined, and
`_on_fill` raised `RuntimeError: ... machine bug` — **on a fill that was
genuinely missed-while-held**.

The licence predicate was **side-scoped and one event stale by construction**;
the fact it needed is **generation-scoped**. Fix: `_SideRun.missed_gens`
records the generations whose start was missed while held, and the guard reads
that. **The guard is not weakened** — a fill on a generation that was neither
joined nor missed still raises, driven at the unit because the replay cannot
legitimately reach it.

**Red-first, confirmed against the committed pre-fix code** (`git show HEAD:`):
the three-generation fixture in selftest group P **RAISES pre-fix** and values
the fill as `missed_while_held` after.

### D2 — the machine quoted during a hold, and every invariant passed

The module header declares that derived effectiveness times "settle lazily at
the next processed event". **`_on_gen_start` was the one handler that read
`run.held` without settling.** Because consecutive generations abut and
`GEN_START` outranks `GEN_END`, whenever no fill or score of that side falls
between a cancel's effectiveness and its generation's end, **nothing had
settled it** — so the side PLACED on the next generation during what should
have been a hold, and every subsequent fill of that generation was CHARGED to
a policy that should have had no order there.

Measured on the committed pre-fix code, on a two-generation fixture with a
cancel effective at `t=2.0` inside a generation ending at `t=5.0`:

| | pre-fix | post-fix |
|---|---|---|
| `PLACE` on the next generation | **yes, at t=5.0** | no |
| `gen_starts_missed_held` | **0** | 1 |
| fill at t=7.0 (markout −9c) | **CHARGED as received** | `FILL_MISSED_HELD` |
| `received_shares` | **1.0** | 0.0 |
| trajectory time-ordered | **no** — `PLACE` at t=5.0 preceded `CANCEL_EFFECTIVE` at t=2.0 | yes |
| `check_invariants` | **all True** | all True |

**The last row is the finding inside the finding.** The trajectory was
internally consistent and economically wrong, and the invariant battery could
not tell. This is the R-249 class — a control that cannot fail — reached from a
new direction: not a control that was written badly, but a correct control
whose predicate space did not contain the defect.

Fix: `self._settle(run.pending, t)` at the top of `_on_gen_start`. Selftest
group Q carries the known-bad **and** a positive control — the same fixture with
the cancel landing AFTER its generation's end resolves STALE, holds nothing,
and the next generation is placed and charged normally, **identically pre- and
post-fix** (verified against `HEAD`), so the fix changes only the case it
targets.

Suite: **83 → 89 checks, green.**
## 3. Four findings against the parity CONTRACT — the first real trajectory falsifying it, as designed

LANE4 B1.8 said it plainly: *"No BE trajectory has been checked through it —
the interface is designed and tested against round-tripped stub output, and its
first real use is the first thing that can falsify the contract."* This is that
first use. **All four findings are against `da_replay_parity_battery.py`, which
is DA's instrument: they are FILED, not fixed.** Each is established by
execution and carries a check in `de_lane4_real_parity.py`'s suite.

### C1 — the outcome space has no term for a STALE cancel

The machine resolves a cancel **three** ways: `EFFECTIVE`, `STALE` (admitted by
the limiter, but effectiveness landed at or after the reference generation's own
end, so nothing was removed) and rate-`SUPPRESSED`. The contract's identity is
`requested = effective + suppressed`, and its `KINDS` tuple has no
`CANCEL_STALE`.

Both available readings are unsound, so the exporter **refuses to pick one**:
`stale_cancel_reading` has no default, `DROP` breaks the identity, and
`AS_SUPPRESSED` records a limiter refusal that never happened. Both are run and
both are reported.

**On real data this is not a corner case — it is half the cancels.** On the
sampled window: **111 requested, 58 effective, 53 STALE, 0 rate-suppressed**,
and of the 58 that bound, **51 prevented nothing** (`zero_value`). So under
`DROP` the contract's identity fails by 53; under `AS_SUPPRESSED` it holds only
by attributing 53 limiter refusals to a limiter that refused nothing (`passed`
111 of 111). A third term is not a tidiness question.

### C2 — LATENT: a same-generation repost is invisible to `no_fill_after_effective`

DA's `gen` field carries the **reference** generation; the machine's unit is the
**policy** generation (`1` vs `1.r1`), because a repost is a new order with a
fresh queue position and a fresh cancel entitlement. When a hold releases onto
the SAME reference generation whose cancel just bound, a subsequent fill of that
generation reads, to the contract, as *a fill after the cancel bound*.

**Constructible, and shown as a matched pair on a fixture** — two cells
differing only in whether the policy reposts: the no-repost cell passes
`no_fill_after_effective`, the reposting cell fails it (selftest, both
directions).

**AND IT DID NOT FIRE ON REAL DATA — stated because the receipt says so, not
because it is convenient.** On the sampled window the diagnostic reports
`diagnosis_holds: false`: the reposting cell PASSED. Measured over three real
windows to find out why:

| | |
|---|---|
| reposts | 147 |
| of those, onto the SAME reference generation whose cancel had bound | **8** |
| of those 8, generations taking a further charged fill | **0** |

So the shape exists on real data and the collision does not, for a measurable
reason: the hold must outlast neither more nor less than the generation itself.
Median generation lifetime on the sampled window is **0.054 s** (p90 0.473 s,
max 18.2 s) against a declared `repost_dwell_s` of **0.5 s**, so **only 9.3% of
generations outlive the dwell** — the reference generation is almost always dead
before the side becomes eligible, and the repost lands on a later one.

**This is a LATENT trap, not a cleared one, and Phase 4's own grid walks toward
it:** shorten the dwell, lengthen the latency rung, or move to a coin with
longer-lived generations, and the same trajectory that passes today starts
failing a predicate that is about the contract's identity axis rather than
about the policy. It is B2/B3's own lesson (*"keying on a subset of identity is
a defect per axis, not a defect once"*) one axis further down.

### C3 — the composition that actually exists has no name

LANE4 §3 names arms 3 and 4 by their **placement** ("fill-hazard-only cancel,
NEUTRAL PLACEMENT"). The frozen reference is `QR_SKEW_ONLY` — **skew is ON**,
verified at the engine: every `queue_realistic` cell has `skew: True`, and the
neutral cells that do exist (`JOIN_ONLY`, `FRONT_ONLY`, `CANCEL_ONLY`) are all
non-queue-realistic. BE's exporter reads the same name as a claim about the
**predictor** and declares 011 exportable as `CONDVALUE_NEUTRAL` because no
skew/inventory state reaches the model.

Both readings are defensible. The run that actually exists — a conditional-value
predictor scoring over a skewed reference, with **no interaction** between them
— is:

- **not `CONDVALUE_X_SKEW`**: that name asserts an interaction, and the loader
  **REFUSES** `interaction=False` under it (verified by execution);
- **not honestly `CONDVALUE_NEUTRAL`**: that name decomposes to
  `components: ("condvalue",)`, so the submission omits the skew that was in
  force in the placement it inherited — **and the loader ACCEPTS it** (verified).

The only name the contract accepts is the one that drops a component that was
really there: precisely the mislabel BE's own comment says the contract
structurally cannot catch. **ASK (§6 A1).**

### C4 — the contract cannot name a declared stub, and cannot name an acting control

`PREDICTORS = ("none", "composed_linear", "composed_lgbm")`, and
`predictor_active=True` with `predictor="none"` refuses. Two consequences:

1. **A declared STUB scorer is unrepresentable** — yet LANE4 §1 requires the
   battery to be *"built and proved BEFORE any predictor exists"* with "every
   arm a typed stub". The instrument's own design intent cannot be submitted
   through the instrument's own contract.
2. **`RANDOM_MATCHED` acts without a predictor.** A matched-random control must
   declare `predictor_active=False`, which then makes it violate the
   inert-agreement clause — correctly, because it acted. The inert axis
   conflates *"no predictor"* with *"no action"*.

Neither blocks this battery (its acting cells are checked natively), and both
block the seven-arm family from being populated with a control. **ASK (§6 A2).**

### C5 — the export through DA's kind set is a PROJECTION, and is labelled one

DA names seven kinds; the machine emits seventeen. `GEN_END`, `HOLD_START`,
`REPOST`, `FILL_PREVENTED`, `FILL_MISSED_HELD` and the rest have no counterpart
and are dropped. A digest taken through the exporter is therefore a digest over
a projection — B1.6's own objection to ignoring an undeclared FIELD, one level
up at the KIND. **Disclosed, not silently taken:** the load-bearing
full-fidelity comparison is native (`hsp.bit_identical` over all kinds and all
fields); the contract leg exists to exercise the REFUSAL surface.
## 6. ASKS — routed, not resolved (SEAT_PROTOCOL rule 13)

**A1 — the arm-name collision (C3). Which reading of "NEUTRAL" governs?**
LANE4 §3 names the arm by its PLACEMENT; BE's exporter reads it as a claim
about the PREDICTOR. Under the first reading, arms 3 and 4 are unrunnable on
the frozen lane and 011's own submission is mislabelled; under the second they
are runnable and the skew that shaped the placement goes unrecorded. **DE's
recommendation: neither name is used until the vocabulary gains a name for
"condvalue over a skewed reference, WITHOUT interaction"** — the composition
that actually exists. A one-line ruling either way unblocks the family; a
resolution by resemblance is the move B2.8 exists to prevent.

**A2 — can the seven-arm family be populated at all today, and with what
control?** Arms 3/4 need a neutral-placement reference that the frozen lane
does not contain (the only neutral cells are non-queue-realistic, so borrowing
one compares two placement engines). Arms 5/6 need a released predictor, and
BE has already declared 5 and 6 ABSENT pending Phase-3 skew wiring. Arm 7 is a
control that ACTS without a predictor and is therefore unrepresentable in the
contract's inert axis. **Two of seven arms are runnable today.** Whether the
family waits for the skew wiring, or is re-scoped, is a coordinator call.

**A3 — the contract needs three things before a real trajectory can be scored
through it, all DA's** (filed, not touched): a `CANCEL_STALE` kind or a third
term in the outcome identity (C1); a policy-generation axis, or lifecycle
checks keyed on one (C2); and a name for a declared stub predictor plus a
representation for an acting control (C4). **None blocks this battery** — its
acting cells are checked natively — and all three block the seven-arm
submission path.
