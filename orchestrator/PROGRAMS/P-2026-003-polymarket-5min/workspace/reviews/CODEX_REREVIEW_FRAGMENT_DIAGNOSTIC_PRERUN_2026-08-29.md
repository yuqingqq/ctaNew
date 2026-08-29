# Codex re-review — fragment diagnostic pre-run — 2026-08-29

**Exact reviewed tip:** `09409da08c2154cfed6839bc3b98e1a56d2046d4`

**Result-bearing code reviewed:** `be_fragment_diagnostic.py` through
`73a0a9d`; `phase2_arms.py` through the FD-R7 rebind; DA verdict
`da_verdict_fragment_6fe1c2c4.json` from `b02fa84`; R-310 exclusion ruling
from `0cc0278`; and the T1/T2/encoder rulings through R-313.

**Timing and scope:** before the one permitted fragment score. I exercised the
committed selftest, the real gate-loader path, and synthetic falsifiers only. I
did **not** call `score_stage` on the real fragment exposure/tape, did not write
`be_fragment_diagnostic_v1.json`, and produced no model or strategy number.

## Verdict

**PRE-RUN HOLD MAINTAINED. DO NOT FIRE THE FRAGMENT SCORE.**

The R-310 exclusion rule is represented honestly and the exact current verdict
is accepted as 17 passed plus the two ruled diagnostic-only exclusions. Several
earlier defects are genuinely closed. The actual score path, however, has not
been exercised and cannot currently run: `_feature_pass` removes `status` from
every kept row while `score_stage` immediately requires that field. In addition,
the claimed exposure binding is only a reported hash, population mismatch is
reported rather than refused, and both the gate and local tape index retain
fail-open structural cases.

The independent DA hold already filed in
`CODEX_REREVIEW_DA_T1_T2_2026-08-29.md` also remains load-bearing. R-312
correctly requires a repaired gate, a fresh gate run, and a superseding verdict
before clearance. R-313 adds the independent consumer-side exact-field check to
that sequence.

## Closures verified

- **FD-R1, narrow part:** the score function now calls the gate loader before
  indexing/scoring. Against the real files it re-derived 19 applicable
  predicates, 17 pass, and exactly
  `both_splits_populated`/`embargo_respected` fail and carry the R-303/R-306
  citations. Tape content re-hashed to `a6e841e8644265fc`; the verdict re-hashed
  to `ce5240c2f136978e`.
- **FD-R2, split selection:** `score_stage` explicitly requests `split="score"`.
- **FD-R3:** candidate and incumbent are both passed their own frozen causal
  threshold maps; a partial map is refused by `evaluate_policy`.
- **FD-R4:** receipt construction now reads the nested `ev["budgets"]` maps.
- **FD-R5, CLI shell:** no arguments and unknown mode return 2; missing tape,
  existing output, and absent written reason refuse.
- **FD-R7, narrow array truncation:** the shared reader refuses EOF before the
  rows array's `]`, and the identity move/rebind is disclosed. The outer-object
  close residual remains open below.
- **FD-R8:** output language clearly says model diagnostic,
  `DIAGNOSTIC_NEVER_EVIDENCE`, not strategy performance and not evidence for
  `QR_CANCEL_HOLD_X_SKEW` versus `QR_SKEW_ONLY`.

The committed selftest is green at 32 checks. That is useful evidence for the
helpers above, but it never calls `score_stage`; green does not establish an
executable result path.

## PR3-FD1 — the score path deterministically refuses every non-empty real population

`phase2_arms._feature_pass` appends kept rows containing only:

```text
slug, day, t0, t_start, side, gen, latency, coin
```

It does not carry `status`. `score_stage` then immediately executes:

```python
assert_field_readable(kept, "status", str, "post-feature-pass rows")
```

I called the real `score_stage` function with controlled upstream seams and a
one-row block having exactly the committed `_feature_pass` output shape. It
refused:

```text
DiagnosticRefused: 1/1 sampled rows carry no 'status'
```

This is a hard pre-number failure. The R-311 statement that the real path was
exercised must be narrowed to **the real gate-loader path**; the real score path
was not exercised. The selftest's CLI cases deliberately stop before this call
and therefore cannot see it.

**Closure:** make the status assertion apply at a stage that actually carries
the field, or preserve the field with an explicitly reviewed identity/rebind.
Then add an end-to-end synthetic `score_stage`/CLI positive control that reaches
receipt cells, alongside threshold, split, population, and output known-bads.

## PR3-FD2 — the exact exposure input is not bound

`load_gate_verdict` says it provides three bindings, including that the tape was
built from the exact exposure being scored. The code only hashes the supplied
exposure and returns that hash in the receipt. It never compares it with a hash
in the tape, verdict, freeze, or a fixed consumer pin. The current tape header
contains no exposure path/hash and the DA verdict contains no exposure
path/hash.

Executed falsifier: one tape and one otherwise-acceptable verdict were supplied
with two different exposure files. Both were accepted:

```text
exposure A accepted -> 3e4bcd92097bdc77
exposure B accepted -> 4e8062c7c07414eb
```

The current real call reports `0a3f2e0b2cf7f788`, but reporting the bytes chosen
by the caller does not prove those are the bytes from which the certified tape
was built. Thus R-311's “exposure bound” claim is not implemented.

**Closure:** put the exposure content hash in a load-bearing producer artifact
(preferably tape header plus verdict), and require equality with the exact
`exposure_path` before feature construction. Include a wrong-exposure known-bad.

## PR3-FD3 — FD-R6 population reconciliation does not refuse mismatch

`reconcile_population` refuses non-zero `state_join_failed` and duplicate kept
action rows, but merely returns `reconciles: false` when
`len(kept) + sum(drops) != expected_rows`. `score_stage` never checks that
boolean and proceeds.

Executed callable falsifier:

```text
kept=1, drops=0, expected=2
-> ACCEPT, reconciles=false
```

The selftest checks only the matching positive control, so the claimed “full
reconciliation” closure is incomplete.

**Closure:** raise before scoring on any mismatch, invalid/negative expected
count, negative/non-integer drop, or undeclared drop status; add the 1-vs-2
known-bad through `score_stage`, not just the helper.

## PR3-FD4 — gate evidence can be structurally incomplete and still authorize

The exact committed verdict currently contains the expected 19 predicates and
a tape hash, but the consumer does not require either property:

- a verdict containing only one passing applicable predicate was accepted;
- a verdict with the ruled two failures but no `tape_sha256_prefix` was
  accepted;
- gate code identity, ledger pin identity, and verdict content identity are
  copied/reported or trusted from the local verdict rather than checked against
  an independently fixed expectation.

This is distinct from R-313's clarification that a **new passing** predicate may
be accepted. A present new passing predicate is harmless under the ruling; an
omitted governed predicate is not a pass.

**Closure:** require a versioned minimum predicate universe (allow additional
passing predicates), a non-empty well-formed subject hash equal to the tape,
and the expected gate/verdict provenance. Missing governed predicates and a
missing hash must refuse. Preserve R-310: only the two fixed exclusions may
fail; every other applicable failure refuses.

## PR3-FD5 — the local index still launders absent fields and duplicate keys

The consumer index still uses:

```python
state_status -> default "OK"
idx[key] = value
```

Executed falsifiers against `_index_tape`:

```text
missing state_status -> ACCEPT as OK with a numeric vector
duplicate tape key   -> ACCEPT, two rows collapse to index_n=1
```

R-313 also records the more damaging encoder behavior: a missing/`None` feature
can become numeric zero, including a missing guard becoming
“present-and-zero.” The current consumer has no exact 45-field check before
encoding. The later kept-row duplicate check cannot recover a tape row already
overwritten in the index.

**Closure:** in the diagnostic consumer, before encoding, require the exact
pinned 45-field state set, explicit declared `state_status`, declared status
value, and unique `(slug, side, gen, t_start)` key. Refuse missing/extra fields
with identity and names. Add complete, one-field, unguarded-None, missing-status,
and duplicate-key controls. The identity-file encoder repair may remain queued
only if these independent pre-score checks and the repaired DA gate are both
load-bearing for this diagnostic.

## PR3-FD6 — the current DA verdict remains under-certified

The exact v2 artifact's reported fact is encouraging:
`per_row_feature_count={48:472413}`. That does not close the checker defect:
the current T1 implementation accepts a one-field row and reports the ragged
distribution without turning it into a failing predicate. The parser also
accepts a document ending after the rows array `]` without the containing
object's `}`. Because the diagnostic calls the shared parser, that T2 residual
also remains in its dependency path.

R-312's sequence is therefore correct and is incorporated into this hold:
repair T1/T2 red-first, re-gate the exact v2 bytes, issue a superseding verdict,
and bind the harness to it. Do not treat the current uniform distribution as a
substitute for a checker that can refuse its known-bad.

## Executed review record

- exact current source/selftest: **32/32 green**;
- real gate-loader path only: **accepted**, 17 pass + 2 ruled exclusions;
- real fragment score: **not run**;
- synthetic actual `score_stage` seam: **refused on absent post-feature-pass
  `status`**;
- exposure-binding falsifier: **two different inputs both accepted**;
- population mismatch falsifier: **accepted with `reconciles=false`**;
- incomplete-predicate verdict: **accepted**;
- missing-subject-hash verdict: **accepted**;
- missing-status tape row: **accepted as `OK`**;
- duplicate tape identity: **silently overwritten**;
- DA partial-schema and missing-outer-close results: inherited from the exact
  executed T1/T2 re-review and ratified by R-312.

## Release conditions

1. Close DA T1-R1 and T2-R1, re-gate the exact tape, and commit a superseding
   verdict.
2. Add the R-313 exact 45-field/status/duplicate consumer checks before
   encoding.
3. Bind the exposure bytes and require complete gate evidence plus a mandatory
   tape subject hash.
4. Make population reconciliation fail closed.
5. Fix the post-feature-pass `status` seam and execute a synthetic end-to-end
   score/CLI positive control that reaches cells without using the real
   fragment data.
6. Re-request pre-run review. Until it explicitly releases this hold, keep the
   single fragment score dark.

Nothing in this filing changes the frozen model, multiplicity, race, or
strategy baseline. The eventual integrated strategy experiment still retains
`QR_CANCEL_HOLD_X_SKEW` as baseline and `QR_SKEW_ONLY` as comparator.

