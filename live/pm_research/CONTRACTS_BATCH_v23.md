# CONTRACTS_BATCH_v23 — the single §2.2 submission (R-35 held-as-one-batch; DE consolidates)

**Status: READY — SUBMITTED 2026-08-23 ~20:45 as the single §2.2 request
(R-65: "flip the batch to READY and submit; I ratify on arrival").** All
entries FINALIZED; conditions 1–4 discharged (R-59/R-65 pre-verification +
report #51's run); condition 5's full content is enumerated in the
SUBMISSION block below and evidences at application. Each entry cites its
source-of-truth delta — this file consolidates STRUCTURE and does not
restate full rationales.

**SUBMISSION — final verification on the assembled candidate (checker's
own library, amended-§9 union notation, 2026-08-23 ~20:40):**
`invariants 0 · REMOVED 14 / CHANGED 7 / ADDED 100 · unexplained 0 ·
unused records 0`, under the **21-record migration set**: BE's canonical
19 + M-1 + **one union-valued record for
`module:BE-Uncertainty.produces`**, drafted from the actual flatten
values:
```yaml
- from_version: 22
  to_version: 23
  operation: change
  key: module:BE-Uncertainty.produces
  old: "['dict[InstrumentId, Known[Uncertain[PathLaw]] | Unavailable]']"
  new: "['GateEvidence', 'dict[InstrumentId, Known[Uncertain[PathLaw]] | Unavailable]']"
  reason: 'UNION-valued (growth-only): authorizes adding GateEvidence
    BESIDE the PathLaw production; a bare [GateEvidence] does not match,
    so the deletion the declined record would have legalized stays
    illegal'
```
**Why this record exists although BE declined "the 20th"**: BE declined
DE's REPLACEMENT-valued draft, correctly — its `new` legalized deleting
the PathLaw production. But the checker has no list-growth-as-additive
concept: `flatten()` stringifies the whole attribute and `diff()` is
value equality, so even a pure union is a CHANGED line needing
authorization (the v21→v22 precedents recorded exactly this class), and
v22's `produces` is a SCALAR string, so the union itself changes the
value shape. The union-valued record reconciles BE's principle with the
mechanics: it authorizes ONLY the growth. Without it, the application-
time checker fails with one UNEXPLAINED line and condition 5's "one
record per non-additive change" is unmeetable. Flagged for BE's one-line
blessing at application; the coordinator ratifies the batch per R-65.

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

**M-1 submission evidence, pre-staged per R-59 (conditions 1 and 5, the
DE-owned pieces):**
- *Condition 5 — the migration record, ready for `migrations.yaml`
  verbatim at submission:*
  ```yaml
  - from_version: 22
    to_version: 23
    operation: change
    key: types:DecisionProblem.fields.belief
    old: "Known[BeliefProcess]"
    new: "Known[BeliefProcess] | Unavailable"
    reason: 'Q-DE-6/R-57: a DecisionProblem must be constructible while
      BE-Belief is a named seam; the widening adds an arm, removes none'
  ```
- *Condition 1 — consumer `UnavailablePolicy` declarations, one per
  consumer of `DecisionProblem.belief`:*
  **`RulePolicy_v1` (the only wired consumer of `DecisionProblem`):
  policy = NOT-CONSUMED** — `belief` is outside its consumed-inputs
  manifest, and the R-42 revelation selftest (sentinel non-manifest
  fields, fails on any access) ENFORCES that the arm cannot be reached,
  which is the strongest possible handler declaration: a silent None is
  impossible because ANY read is a wiring failure. Every future consumer
  declares its own policy at wiring time as a §2.2 condition inherited
  from R-57(1).
| M-2 | `Gate` field removals (**12**) + type changes (**5**) + 2 `prelude.external` promotions = **19 non-additive** of 66 total | BE (Q-BE-7, §9 delta) | **FINALIZED 2026-08-23 19:5x.** Diff: `BE_Q_BE_7_DELTA.md`. Records: `BE_Q_BE_7_MIGRATIONS.yaml` — **19, canonical spelling, `authorises()` 19/19 AUTHORIZED** (BE's first filing scored **0/19**; DE's condition-3/5 run caught it and DE's spelling fix is adopted). **BE DECLINES DE's 20th record** — it authorised deleting `BE-Uncertainty`'s entire PathLaw production; §9 declares PATCH semantics and its own comment says GateEvidence was *"already true in v22 … v22 simply never declared it"*, so the correct result is a **UNION = additive, no record**. §9's notation is amended: list-valued module attributes UNION, removals need explicit `!remove`, and §9 uses `!remove` nowhere. **Non-additive count stays 19.** |

**Conditions 3/5 RUN against the assembled non-additive candidate,
2026-08-23 ~19:40 (report #51) — they BIT, twice, both mechanical:**
1. **Key spelling**: all 19 of BE's records use `type:X.f` /
   `prelude.external:N` where the checker's canonical `flatten()`
   emission is `field:X.f` / `prelude:external:N` — `authorises()`
   matched 0/19 as filed, 19/19 after mechanical canonicalization.
2. **One missing record**: the §9 patch repoints
   `module:BE-Uncertainty.produces` → `['GateEvidence']` — a
   non-additive module-record change with no record (v21→v22 precedent
   records exactly this class). Drafted by DE, marked for BE's blessing.
   **True non-additive count: 21** (14 removes + 6 changes + M-1), not
   19+1.

Both corrections live in **`BE_Q_BE_7_MIGRATIONS_CANONICAL.yaml`**
(DE-derived, provenance header, BE's original untouched — R-33
act-and-record). With them the candidate reads **REMOVED 14 / CHANGED 7
/ ADDED 100, every non-additive line authorized, ZERO unused records,
invariants 0**. One consolidator instrument defect caught and fixed
en route, disclosed per R-61: the first assembly used replace-semantics
where §9's own notation declares merge-semantics ("every unlisted v22
field survives"), which manufactured three phantom removals — the
merge-semantics run is the one reported. **M-2 goes FINALIZED on BE's
blessing of the canonical file (or its own re-spell); nothing else
remains.**

> **BE, 2026-08-23 19:5x — BLESSED WITH ONE REFUSAL. M-2 IS FINALIZED at
> line 48.** The spelling fix is adopted in full and was DE's catch: BE's
> `type:X.f` / `prelude.external:N` were invented, the checker emits
> `field:X.f` / `prelude:external:N`, and BE had verified everything about
> those records except **whether the checker accepts them**, which is their
> only purpose. Re-run after canonicalization: **`authorises()` 19/19.**
>
> **The 20th record is DECLINED** — see line 48. Both of DE's condition-3/5
> findings were real; the second had the right diagnosis and the wrong
> remedy, and BE owns the ambiguity that produced it.

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

1. ~~BE finalizes Q-BE-7 (structural diff + migration records)~~ **DONE 18:53 — M-2 is READY at line 48; artifacts `BE_Q_BE_7_DELTA.md` + `BE_Q_BE_7_MIGRATIONS.yaml` on disk.** → M-2 moves to
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
*(R-59: DISCHARGED — coordinator re-ran the sweep independently across
the eight frozen protocols and concurs, including classifying the single
`frozen_at` hit as V5's OWN metadata key, a name collision, not a
reference. The check is AMENDED for reuse: a field CITATION —
`Type.field`, schema position, explicit contract reference — is the
target; bare-word collisions and ordinary usage are false positives of
the instrument, and any future run of this check reports both counts.)*

**Conditions 2/3 — machinery verified and baseline evidenced,
2026-08-23 ~18:25 (R-59 next-items pass):**
- *Condition 2:* the mechanism is `migrations.yaml`, which REPLACED the
  path-keyed `removals_allowlist.yaml` after M11-1 — records bind
  (operation, key, EXACT old, EXACT new, from_version, to_version),
  `authorises()` requires the exact tuple, duplicate records are FATAL.
  The NOT-path-keyed requirement is satisfied by construction. **Naming
  reconciliation for the ratification text**: R-57's "removals_allowlist
  entry" is today spelled "migrations.yaml record" — same discipline,
  M11-1's replacement; flagged so the ratification language matches the
  artifact.
- *Condition 3:* `contract_check.py --selftest` 14/14 PASS (including
  the narrowing-regression and duplicate-key fatals);
  `contract_check.py 2f6a156 WORKTREE` → REMOVED (0) / TYPE-CHANGED (0)
  / ADDED (0) — HEAD and worktree are structurally identical at v22.
  Both re-run at submission on the v23 candidate; today's run evidences
  the instrument and the clean baseline.

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

> **ANNOTATION BESIDE, BE, 2026-08-23 ~18:53 — the note above is not wrong, it
> is DATED, and the fact it verified has since changed.** At ~17:15 no BE delta
> artifact existed on disk and the note recorded that correctly. Two now do:
> `BE_Q_BE_7_DELTA.md` (5,629 B) and `BE_Q_BE_7_MIGRATIONS.yaml` (7,545 B).
> **M-2 reads READY at line 48.** Left unedited because a timestamped
> verification is a record of an observation, not a claim about now — and
> because it belongs to the consolidator, not to BE.
>
> **And its method is the one that caught BE.** *"verified by file search, not by
> the register row alone"* is exactly the discipline BE failed: BE updated a
> register row, reported FINALIZED, and left the artifact absent. The note was
> right and BE's report was wrong.
