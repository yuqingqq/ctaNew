# Codex re-review — DA T1/T2 (`1775d81`) — 2026-08-29

**Exact reviewed commit:** `1775d817f33717985234123a08f20d9d6599fad7`

**Scope:** the committed DA closure of Batch-3 T1 (whole-stream per-row state
conformance) and T2 (EOF read as completion). No state tape was gated and no
fragment score was run.

## Verdict

**T1 HOLD MAINTAINED — THE FRAGMENT GATE MUST REMAIN PAUSED.** The exact
empty-state counterexample now refuses, but a partially populated state row is
still accepted. The newly reported per-row feature-count distribution exposes
the defect without making it affect the verdict.

**T2 ARRAY-CLOSE CASE CLOSED NARROWLY.** EOF before the rows array's `]` now
refuses and a complete array passes. A single-object tape cut after `]` but
before the outer `}` is still accepted, so the broader malformed-document
truncation class remains open. The separate `phase2_arms` and BE reader
siblings are outside this DA commit and remain open as already filed.

## What the commit correctly closes

- `da_state_tape_verify._stream_array` now refuses EOF before the rows array
  closes.
- The exact Batch-3 fixture—400 valid nested rows followed by an empty
  `state={}`—now reaches production `verify()` and refuses.
- A complete 401-row fixture passes the new scan and carries
  `per_row_feature_count` in the verdict.
- `harmful_rows_loader.stream_ok_rows` now refuses EOF before `]` rather than
  returning a shortened prefix.

These are real improvements. The residual below is a narrower version of T1,
not a denial that the named empty-row case moved.

## T1-R1 — “any declared feature” is not schema conformance

The whole-stream loop computes:

```python
_here = _declared & set(r)
if not _here:
    _nonconf += 1
```

Thus it rejects a row with **zero** declared fields and accepts a row with
**one**. The first 400 rows still supply the union used by
`no_undeclared_reduction`, so missing fields on later rows remain invisible to
the governing predicates.

Executed full-schema falsifier:

- rows 1–400: all 48 declared non-identity fields;
- row 401: `state_status="OK"` plus one numeric feature, every other declared
  feature absent;
- result: `verify()` **does not refuse**;
- reported distribution: `{2: 1, 48: 400}`;
- `state_status_present`: **PASS**, 401 `OK` rows.

The distribution contains the proof that the tape is ragged, but no predicate
requires its support to equal the declared carried set. A reported failure that
does not enter the verdict is not a gate.

**Closure:** derive the exact carried feature set from schema minus declared
reductions. For every row, require equality with that set after flattening, or
an explicitly declared status-specific schema. Refuse missing **and extra/wrong
layout** fields with the first identity and the missing/extra names. Make the
per-row count/set predicate load-bearing. Add positive controls for every
declared non-OK status rather than weakening the rule to accommodate them.

## T2-R1 — outer object closure is not checked

Executed parser cases:

```text
{"rows":[{"x":1}]}   -> accepted (complete control)
{"rows":[{"x":1}]    -> accepted (missing final object brace)
{"rows":[{"x":1}     -> REFUSED (missing array close; fixed case)
```

On `]`, `_stream_array` returns immediately and never checks the containing
object's remaining syntax. For a V5 tape whose metadata precedes `rows`, a
writer killed after the array but before `}` therefore produces invalid JSON
that the gate accepts as complete.

This case does not necessarily lose a row, so it is less decision-bearing than
the fixed missing-`]` case. It is still a false completeness claim and should
refuse at an artifact-verification boundary.

**Closure:** distinguish bare-array inputs from object-wrapped arrays; for the
latter, consume only whitespace and the required closing `}` after `]`, then
require EOF. Reject trailing garbage, missing object close and additional
undeclared structure. Exercise complete, missing-`]`, missing-`}`, and trailing
garbage cases.

## Execution record

- exact committed source imported from the clean review worktree;
- 401-row partial-schema falsifier executed against the real 48-field schema;
- state-status control confirmed the last row was counted as `OK`, not rejected
  for an unrelated absent-status reason;
- complete / missing-outer-close / missing-array-close parser triplet executed;
- no real state-tape gate and no fragment score run.

## Release condition

Do not use `1775d81` to certify the fragment tape. Close T1-R1, then re-run the
exact empty, partial, wrong-layout and complete controls through production
`verify()`. T2-R1 can ride the same repair, while the non-DA T2 readers and the
fragment provenance/empty-embargo findings remain independently required.
