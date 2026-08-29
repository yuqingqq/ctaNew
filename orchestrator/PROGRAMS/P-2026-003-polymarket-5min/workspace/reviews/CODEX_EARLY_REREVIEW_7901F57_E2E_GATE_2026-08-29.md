# Codex early re-review — `7901f57` end-to-end and gate closures — 2026-08-29

**Exact reviewed commit:** `7901f571740c9d9d2bd4323cfedbd0dcb2088e45`

**Scope:** the new synthetic `score_stage` positive control and the PR3-FD2,
FD3, FD4, and FD5 repairs in `be_fragment_diagnostic.py` and
`build_state_tape_v2.py`. No real fragment score was run and no result-bearing
number was produced.

## Verdict

**END-TO-END PATH ESTABLISHED; FRAGMENT SCORE HOLD MAINTAINED.**

The new control really calls production `score_stage`, the frozen candidate
and incumbent scorers, and `harmful_action_eval.evaluate_policy`. Its first
execution correctly exposed the impossible threshold-mode string in the prior
consumer. The repaired path reaches three non-zero candidate cells and both
evaluators report `CAUSAL_FROZEN_FROM_TRAIN`.

PR3-FD2, FD3, and FD5 are materially improved, but FD4 is not closed: the
consumer's three-name predicate universe is smaller than DA's current
load-bearing contract and does not require a governed predicate to be
applicable. Both bypasses execute successfully. The synthetic control also
does not yet prove that the incumbent scorer is live: all of its present
assertions pass with a forced all-zero incumbent score vector.

The current real tape predates the new input stamp and is correctly refused by
this consumer. It must be rebuilt and re-gated after the remaining contract
repairs; the real score remains dark.

## Executed evidence

At the exact commit:

```text
python3 live/pm_research/be_fragment_diagnostic.py --selftest
BE FRAGMENT DIAGNOSTIC SELFTEST GREEN: 0 failing (61 PASS lines)
elapsed 1.09 s; maximum RSS 219,604 KiB
```

The direct synthetic receipt contained three budgets:

```text
candidate net cents: 10%=210, 15%=280, 5%=165
incumbent net cents: 10%=460, 15%=505, 5%=305
both modes: CAUSAL_FROZEN_FROM_TRAIN
status: SYNTHETIC_SELFTEST_NOT_A_RESULT
```

These are test-fixture values, not research results.

### Closures verified

1. **The causal-mode assertion is real.** The old string `CAUSAL_FROZEN` does
   not occur in the evaluator. The consumer now requires the evaluator's real
   `CAUSAL_FROZEN_FROM_TRAIN` value and explicitly identifies
   `RETROSPECTIVE_TOPK` as a refusal.
2. **PR3-FD2 consumer comparison works.** A different exposure file from the
   tape's `input_sha256.score` stamp refuses. The builder now writes SHA-256
   stamps for both splits.
3. **FD3 works.** `score_stage` reads `recon["reconciles"]` and refuses a
   non-reconciling population rather than merely reporting the false field.
4. **FD5 works locally.** The index requires declared `state_status`, refuses
   duplicate identities, and enforces exact state-field-set equality.

## Finding 1 — FD4 predicate contract remains bypassable

`da_state_tape_verify.py` declares seven current `LOAD_BEARING` predicates.
The BE consumer requires only three names:

```text
DA load-bearing (7):
  absorption_within_bound
  both_splits_populated
  dataset_non_empty
  gap_count_matches_expected
  half_open_containment_landed
  no_rows_skipped_by_builder
  provenance_matches_expected

BE required (3):
  both_splits_populated
  dataset_non_empty
  embargo_respected
```

The five DA load-bearing names absent from the BE set are therefore optional.
On a valid synthetic tape/exposure binding, I supplied only the three BE names,
with the policy-excluded pair failing. `load_gate_verdict` returned **ACCEPT**:

```text
OMIT FIVE DA LOAD-BEARING: ACCEPT, n_applicable=3, n_failed=2
```

I then declared `dataset_non_empty` present but `applicable=false`. The
consumer again returned **ACCEPT**:

```text
DATASET_NON_EMPTY N/A: ACCEPT, n_applicable=2, n_failed=2
```

This happens because required-name presence is checked with a set, while
applicability is used only to remove entries from evaluation. It recreates the
checked-nothing class DA's own `LOAD_BEARING` contract was introduced to stop.
The consumer also treats a missing `applicable` field as false, while DA's
verdict writer/evaluator defaults it to true.

**Required closure:** version and require at least DA's full current
load-bearing set plus the governed embargo predicate; require each governed
name exactly once and in its permitted applicability state. For this fragment,
the expected failed set must remain exactly the two policy exclusions and all
other governed predicates must be applicable and pass. Add known-bads for an
omitted load-bearing name, a duplicate name, missing `applicable`, and an
impermissible N/A.

## Finding 2 — the end-to-end control does not falsify a dead incumbent arm

I replaced only `phase2_iter011_run.apply_incumbent` with a same-length vector
of zeros and called the real synthetic `score_stage`. Every assertion currently
made by the new end-to-end block still passed:

```text
three_cells=True
candidate_nonzero=True
both_causal=True
increment_not_none=True
synthetic_stamp=True
ALL_PASS=True
incumbent nets=[0.0, 0.0, 0.0]
```

`threshold_mode` proves which threshold path the evaluator took; it does not
prove that the model arm produced a meaningful score vector. A non-`None`
increment is also compatible with subtracting a dead zero arm.

**Required closure:** on this deliberately non-degenerate deterministic
fixture, assert finite, non-constant candidate and incumbent score vectors and
assert non-zero economics for both arms (or expose equivalent synthetic-only
diagnostics in the receipt). Keep the current three-cell and synthetic-stamp
checks.

## Finding 3 — the new builder stamp has a consume/hash race

`build_state_tape_v2.main` consumes both exposure inputs during the build, then
computes `input_sha256` only when constructing the output header. If an input
is replaced between its read and the late hash, the tape can contain rows from
old bytes while stamping the new bytes; the consumer will then accept the new
file as the source of rows it did not produce.

**Required closure:** hash each input before consumption, re-hash after
consumption, refuse if either changed, and stamp the verified pre/post-equal
digest. A committed immutable snapshot or equivalent open-file identity
contract would also close the race if it is enforced and carried.

## Current artifact and remaining hold

The on-disk `be_fragment_state_tape_v2.json` has builder ref `6fe1c2c4...` and
contains neither `input_sha256` nor `builder_sha256`. Executing the new gate
against it refuses as intended:

```text
REFUSED: be_fragment_state_tape_v2.json does not stamp the exposure input it
was built from (input_sha256.score)
```

After Findings 1–3 are repaired, rebuild the tape from an attributable commit,
write a fresh DA verdict over the rebuilt bytes, and re-run the pre-score
checks. The previously filed strict target-latency contract gap from
`CODEX_EARLY_REREVIEW_E7DE218_VALUATION_CONTRACT_2026-08-29.md` also remains
open: `rejoin_source_fields` still reconstructs the gate through
`latency or {}` without first refusing a partially malformed latency cell.

Until those closures execute, **do not run the single real fragment score**.
