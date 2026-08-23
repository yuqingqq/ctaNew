# CONTRACTS_BATCH_v23 — the single §2.2 submission (R-35 held-as-one-batch; DE consolidates)

**Status: DRAFT — CONSOLIDATION IN PROGRESS.** Assembled by DE per R-35/R-43
from the planes' finalized deltas; **BE's delta is PENDING with its blocker
named**, so this document is not yet ready for ratification. Each entry cites
its source-of-truth delta — this file consolidates STRUCTURE (dedupe,
additive-vs-migration split, cross-plane consumers) and does not restate full
rationales.

**Ratification landing evidence (per R-36, named in advance):**
`contracts.yaml` v22 → v23 diff matching this document exactly; migration
records for every §1 item bound to (operation, key, old, new, version);
`contract_check.py` selftests green; the structural diff clean. "Applied"
names those four artifacts or it is not applied.

---

## §1 — NON-ADDITIVE (migrations; called out separately per R-35)

| # | change | source | status |
|---|---|---|---|
| M-1 | widen `DecisionProblem.belief` → `Known[BeliefProcess] \| Unavailable` | DE (Q-DE-6; module plan §6.2 — a problem cannot be constructed while BE-Belief is a named seam) | FINALIZED |
| M-2 | `Gate` field removals (12) + type changes (~10); `GateId`/`Provenance` removed from `prelude.external` | BE (Q-BE-7, §9 delta) | **PENDING — NOT READY: structural diff and migration records undone (BE's own filing)** |

## §2 — ADDITIVE, deduped across planes

**DE (Q-DE-6; source: `DE_MODULE_PLAN.md` §6.2) — FINALIZED:**
- `Action.order_ref` (CANCEL is currently inexpressible)
- `Action.placement ∈ {JOIN, FRONT_ON_FORMATION}`
- `CapitalOpCommand` + `capital_cmd_out/in` ports (Allocator sole issuer;
  Actuator executor)
- `DE-Constraints` consumes `CapitalBudget` — **co-sponsored by DA** (also in
  SP §7); one entry, closes the size chain via `FeasibleSet.max_size`
- `FeasibleSet.max_size` notes: side-keyed `"VERB:SIDE"` key domain +
  DEFAULT-DENY (without the pin, ∅ has no carrier and an empty set is
  permissive)
- `telemetry_out` explicit on `DE-Constraints`/`DE-Allocator` records **AND**
  `HealthEvent` added to their `produces`; `HeartbeatPulse` emission for the
  four acting DE modules — **consumer: OPS** (their OP-2 names this as the
  dependency that un-HALTs the UNKNOWN modules); also resolves the ports-map
  self-inconsistency (`_representation` vs OP-Monitor's map-only ports)

**OPS (Q-OPS-2; source: `ops/proposals/OP_CONTRACT_DELTA_v23.md`) —
FINALIZED, verified against v22 by DE (CancelAllStatus: exactly three
occurrences — rule, type, producer; consumed by nothing):**
- `OP-Monitor.consumes` gains `CancelAllStatus` — a conformance repair, not a
  design change: makes `R-HALT`'s `Unconfirmed ⇒ HALTED` evaluable. **Until
  ratified, R-HALT must not be cited as active protection** (binding on DE's
  carry analysis, acknowledged).

**DA (Q-DA-8; source: SP §7, six changes, `ParamId.namespace` withdrawn) —
FINALIZED:**
- four SP record types (the register's machine-readable home — also the
  named end-state for the R-32 inline duplication and its SP §6 guard)
- `ParamValue` validity interval (`params.at(t)` — without it a replay reads
  today's value inside yesterday's decision)
- `Provenance` enum + authority axis — **carries R-3 (currently
  RIGHT-BUT-UNENFORCEABLE); co-sponsored by BE** whose delta also lists it;
  one entry
- the fee family correction; the human-seat owner record

**Registry/config riding alongside (no contract edit; listed so the batch is
the complete wiring picture):** `RulePolicy_v1` registration with
consumed-inputs manifest **+ the R-42 revelation selftest** (sentinel
non-manifest fields; ships WITH the wiring); `utility_none`; `incentive_none`.

## §3 — KNOWN FOLLOW-ONS, deliberately NOT in this batch

Contract-shaped open register rows that are NOT smuggled in ahead of their
rulings: Q-DA-19 (a class field / machine surface for A–D + a carrier for
clause (e)'s vacate provenance), Q-DA-20 (a `Disputed` arm), Q-DA-9's
operative-set scope. Each enters a future batch after its ruling.

## §4 — Consolidator's checklist to READY

1. BE finalizes Q-BE-7 (structural diff + migration records) → M-2 moves to
   FINALIZED.
2. DE re-verifies every §2 entry against contracts v22 HEAD at submission
   time (the batch was assembled against v22; any interim contract motion
   invalidates the diff).
3. Submit as ONE §2.2 request; coordinator ratifies once; landing evidence
   as named above.

## §5 — RATIFICATION CONDITIONS, declared in advance per R-57 (the bar before the data, applied to a contract change)

§2 ADDITIVE: ratified on arrival, subject only to the §4 re-verification.
§1 NON-ADDITIVE: five conditions —
1. M-1 consumers each declare an `UnavailablePolicy` (a widened union with
   an undeclared handler is a silent None at the first fault);
2. M-2 removals each carry a `removals_allowlist` entry bound to
   (operation, key, old, new, version) — NEVER path-keyed (M11-1);
3. M-2 type changes run through `contract_check.py` (selftest green,
   HEAD→WORKTREE clean);
4. **NO FROZEN ARTIFACT REFERENCES A REMOVED FIELD** — grep before
   submission, zero hits or the removal comes out of the batch;
5. landing evidence per R-36: `version: 23`, checker selftest green,
   HEAD→WORKTREE clean, one `migrations.yaml` record per non-additive
   change.

**CONDITION 4 — EXECUTED 2026-08-23 ~17:50, and it did NOT bite.**
The twelve removed `Gate` fields were DERIVED (v22 fields minus the §9
patch's fields — exactly 12: `artifact_hash, data_prereq, frozen_at,
inference_method, metric, on_pass, owner, question, spec_hash,
strata_hash, threshold, unit`), then word-boundary-grepped across 15
frozen/protocol artifacts (all `*PROTOCOL*.md` + the frozen V5 plan — a
superset of R-57's eight):
- **7/12 unambiguous snake_case identifiers: ZERO hits anywhere**
  (`artifact_hash, data_prereq, frozen_at, inference_method, on_pass,
  spec_hash, strata_hash`).
- **3/12 prose-common words: 43 raw hits, every one classified**
  (question 24, threshold 10, unit 9) — all ordinary English ("the
  question this run asks", "no threshold sweeps", "inference unit is the
  UTC day"); NOT ONE references the `Gate` contract field. Full
  file:line lists preserved for audit.
- **Supplementary, same break-class**: the two prelude removals
  (`GateId`, `Provenance`) — ZERO hits in frozen artifacts (and both
  types SURVIVE as local/structured types in the patch, so a reference
  would re-resolve rather than orphan).

**Result: zero contract-field references in frozen artifacts; no removal
comes out of the batch on condition 4.**

**Checklist item 2 — VERIFIED 2026-08-23 ~17:15 (R-56 next-items pass):**
contracts.yaml HEAD still at version 22, no interim motion. Per entry:
`CancelAllStatus` exactly 3 occurrences (rule/type/producer — OPS's repair
still valid and still needed); `order_ref` 0, `CapitalOpCommand` 0 (DE
additives still absent, correctly pending); `CapitalBudget` present as a
type with no DE-Constraints consumer (the batch adds it); `HeartbeatPulse`
present (the batch adds DE emission); the `belief` widening NOT yet applied
(M-1 correctly pending). **READY still blocks on exactly one item: Q-BE-7
(no BE delta artifact exists on disk — verified by file search, not by the
register row alone).**
