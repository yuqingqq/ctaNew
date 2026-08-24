# Q-BE-7 — structural diff, v22 → v23 (BE's §9 delta)

**M-2's deliverable.** Supersedes BE's premature "FINALIZED" report: BE had
verified R-57 **condition 4 only** and reported the whole item clear. The batch
document was right and BE was wrong — M-2 needed **a structural diff and
migration records**, and neither existed. Fourth instance of BE reporting
`cleared` on partial verification; it cost the programme three dispatches.

## How this diff was produced

**Derived, not transcribed.** `contracts.yaml` v22 was parsed and compared
field-by-field against the §9 delta block extracted from `EV_GATES_PLAN.md`.
Nothing here is hand-copied, so a field cannot be missed by transcription — the
failure mode that condition 4 exists to catch one level up.

## Totals

| | count |
|---|---:|
| total changes | **66** |
| **NON-ADDITIVE** (one migration record each) | **19** |
| additive (no record required) | 47 |

**Correction to the batch document's own M-2 line.** It reads *"`Gate` field
removals (12) + type changes (~10)"*. The removals are **12, confirmed**. The
type changes are **5, not ~10** — three on `Gate` and two on
`GateEvidence`. The `~10` estimate is high by 2×. Stated because M-2 is
submitted against these counts.

## §1 — the twelve `Gate` field removals

| `Gate.artifact_hash` | `Hash` | `-` |
| `Gate.data_prereq` | `str` | `-` |
| `Gate.frozen_at` | `Timestamp` | `-` |
| `Gate.inference_method` | `str` | `-` |
| `Gate.metric` | `str` | `-` |
| `Gate.on_pass` | `str` | `-` |
| `Gate.owner` | `str` | `-` |
| `Gate.question` | `str` | `-` |
| `Gate.spec_hash` | `Hash` | `-` |
| `Gate.strata_hash` | `Hash` | `-` |
| `Gate.threshold` | `float` | `-` |
| `Gate.unit` | `str` | `-` |

These are the twelve R-57 condition 4 was run against. **Discharged** — DE
grepped 15 artifacts, the coordinator re-ran it independently across the eight
frozen protocols, and qualified field references are **zero, all twelve**. BE's
own contrary finding (`Q-BE-15`) was withdrawn as a false positive: the G-FF
block that names seven of them sits in a **superseded audit trail**, in
**plan-local YAML**, read by **no code**.

## §2 — the 5 type changes

| `Gate.id` | `str` | `GateId` |
| `Gate.on_fail` | `str` | `FailRoute` |
| `Gate.preconditions` | `list[GateId]` | `list[Precondition]` |
| `GateEvidence.ci_hi_abs` | `float` | `float | Unavailable` |
| `GateEvidence.verdict` | `enum:PASS|INSUFFICIENT_EVIDENCE|MODEL_REFUTED` | `VerdictState?` |

## §3 — the two prelude promotions

| `prelude.external.GateId` | `external` | `locally declared` |
| `prelude.external.Provenance` | `external` | `locally declared` |

Both are `external` → **locally declared**. Non-additive because a type the
contract does not own cannot be constrained by it.

## §4 — additive (47), no records required

11 new types (`FailRoute`, `GateBar`, `GateId`, `GateRegistry`,
`GateVerdictLedger`, `ModelSelectionOutcome`, `Precondition`,
`PreconditionState`, `Provenance`, `ProvenanceState`, `ProvenancedValue`,
`ResourceRequired`, `VerdictState`) plus 16 new `Gate` fields and 16 new
`GateEvidence` fields. Per `migrations.yaml`'s header, additions need no record.

## §5 — the shape of the change, in one paragraph

Every removal moves the same way: **a gate stops describing itself in free text
and starts carrying typed objects that can be checked.** `question`/`metric`/
`threshold`/`unit` → `bar: GateBar` + `null_hypothesis` + `favourable_arm`, so
**polarity is detected from the bar instead of declared beside it** — the
sign-blindness `G-FF3`'s `Scalar(0.0, cents_per_share)` exhibited becomes
unrepresentable. The three hashes and `frozen_at` → `Provenance`, so a vacated
provenance is a typed state with a register obligation. `on_pass` → `on_verdict`,
because R-24 made the verdict **three**-valued and a pass/fail pair silently
routed an underpowered gate as a pass. `inference_method` → `GateEvidence.
inference_actual`, because a declaration cannot disagree with itself.

**BE notes against its own delta:** this is the same principle BE violated three
times in `BE_BELIEF_PLAN` Revision 1 hours after writing it here — asserting a
property where it could have been computed. The delta is right; BE's compliance
with it was not.

## §6 — conditions 2, 3, 5 evidence

- **Condition 2** — `BE_Q_BE_7_MIGRATIONS.yaml`: **19 records**, one per
  non-additive change, each binding `operation` + `key` + `old` + `new` +
  `from_version`/`to_version`. Verified: parses, **19 unique keys**, all
  seven required fields present on every record, **not path-keyed**.
- **Condition 4** — discharged under R-59, zero qualified references.
- **Conditions 3 and 5** — `contract_check.py` selftest and HEAD→WORKTREE clean
  are DE's to run at submission against the assembled v23; BE cannot run them
  against a file it does not assemble. **Named here so the gap is visible rather
  than assumed.**

## §7 — one open defect in the delta, found by BE's own review loop

`BE_BELIEF_REVIEW_LOOP` iteration 2 (lens 3, MF-8) established that a block
spliced into v22 raises **four unresolved-reference errors** —
`consumes -> Target | Params | Instrument | Venue` — because the reference check
skips a `consumes` entry only `if str(item) in mods`, and v22's 25 modules
include none of `BE-Target`, `SP-Params`, `SP-Instrument`, `SP-Venue`. That
finding was against `BE_BELIEF_PLAN` §11, **not** this delta, and BE has **not**
re-run it here. **Flagged, not claimed clear** — the same class of gap that
produced the premature FINALIZED, named this time instead of assumed away.
