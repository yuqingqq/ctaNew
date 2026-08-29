# Codex final pre-run review — fragment score — 2026-08-29

**Exact reviewed code tip:** `e9e03983fb60f6ecc6dda0aecadd85e1990c181c`

**Request:** `REQUEST_FRAGMENT_PRERUN_FINAL_2026-08-29.md` at `a92382a`

**Reviewed repairs:** `c62cdb3`; **v4 DA verdict:** `f337f8b`

**Scope boundary:** source, adversarial controls, immutable artifact bindings,
an independent whole-stream DA verification, and the production
`load_gate_verdict` seam. I did **not** call the real `score_stage`, did not read
a fragment diagnostic number, and did not create the result artifact.

## Verdict

**HOLD RELEASED — RELEASE THE ONE REAL FRAGMENT SCORE UNDER R-293.**

The three R-319 findings are closed in the production paths, the exact v4 tape
is content-bound to its committed clean builder and current exposure inputs,
DA's committed verdict reproduces independently over every row, and the real
consumer accepts that exact tape/exposure pair while re-deriving the ruled
18-pass/2-fail/20-applicable state. I found no remaining correctness blocker on
the path from `score_stage` through the phase-2 scorers and
`harmful_action_eval`.

This is a narrow release for **one** `DIAGNOSTIC_NEVER_EVIDENCE` run. It is not
release of the fragments for validation, race admission, candidate selection,
re-freezing, parameter changes, or strategy-performance claims.

## R-319 closures

### 1. Exclusions no longer act as waivers

`load_gate_verdict` now requires all nine governed names exactly once and pins
the two exclusions to the state DA declared in Q-DA-164:

- `both_splits_populated`: `applicable=true`, `pass=false`;
- `embargo_respected`: `applicable=true`, `pass=false`;
- every other governed predicate: applicable and passing;
- `not_applicable`: empty;
- recomputed failed set: exactly the ruled pair.

The production consumer refuses an exclusion made N/A, an exclusion made
passing, a missing governed predicate, a duplicate, or an extra failure. The
suite's exact ruled state is its positive control. This closes R-319 Finding 1
from both directions rather than merely checking that no unlisted name failed.

### 2. New predicates cannot disappear as N/A

Every non-governed predicate must be unique, applicable, and passing. Executed
known-bads for a new N/A predicate and a new failing predicate refuse; a new
applicable/pass predicate remains accepted. This preserves the intended open
minimum universe without allowing a newer gate check to become inert.

### 3. Valuation validation is before reconstruction

`rejoin_source_fields` now calls `assert_valuation_inputs` before its first
`hm.keptrow` call. The production-seam falsifier sends a scalar latency cell
through `rejoin_source_fields` and receives controlled `DiagnosticRefused`, not
the former raw `AttributeError`. The legitimate zero-fill/false-gate control
still passes, so the repair does not redefine a genuine no-fill as malformed.

## Exact v4 artifact chain

I recomputed these from the live files rather than trusting the request:

| Item | Recomputed result |
|---|---|
| v4 tape | 861,494,871 bytes; SHA-256 `14f77d413022a6a4ce5ac28c7c7746bef497084a215619a8119e2a234b30a5c9` |
| builder ref | `cf0bad555b4a656719769147228f13fd5c5fa378` |
| builder bytes | stamped and `cf0bad5:live/pm_research/build_state_tape_v2.py` both `f870310beb707d4508511210295edac43df3caa81f5646386856cf976da93a31` |
| score input | stamped and live both `0a3f2e0b2cf7f788b14205b45928966f53ee5a64961cde36686025fc95de0dd4` |
| empty train input | stamped and live both `92bf3bf4952109fd8d4023b4a60f3f4878dc61fec55f657e784c9b1d54eaa3c8` |
| ledger | stamped and live both `e1dcd4eb8a85a0b5b2f86ed0bf4f5d43ec40bf6b9ced713201b13240e639a2ae` |
| population | score 253 slugs; train 0; 472,413 emitted rows |
| status counts | OK 442,964; PRE_WINDOW 29,129; GAP_AT_CUTOFF 307; NO_LEVEL_HISTORY 13 |
| clock basis | decision time window-relative; decision epoch and label exit absolute |

The builder hashes both input files before consumption, hashes them again after
consumption, refuses any movement, and stamps only the equal result. The v4
header carries that R-318(3) protocol. The outer object terminates cleanly with
`]}` and DA's strict stream reader consumed it to EOF.

## Independent DA reproduction

I ran the exact `e9e0398` checker over the entire v4 tape with expected gap
count 307, full expected provenance, the pinned ledger, and its full expected
SHA. The terminal exit was 1 **because the exact two ruled predicates fail**;
that is the expected diagnostic gate shape, not an execution failure.

The independent result matched `da_verdict_fragment_v4.json`:

- 472,413 rows;
- exact per-row carried-field distribution `{48: 472413}`;
- 20 applicable predicates, 18 passing, 2 failing;
- failed set exactly `both_splits_populated` and `embargo_respected`;
- `not_applicable=[]`;
- GAP_AT_CUTOFF 307; g0 4/4 flagged; g1 232/232 unflagged;
- zero pre-emission skips and zero as-of violations;
- provenance, ledger, half-open containment, and exact-set checks passing.

The committed and live verdict bytes are identical at SHA-256
`0c1969d342a64e857efe8948c1399a4ce50e55c7594cd2f6248b6fc52aab501c`.
Its subject prefix is the recomputed v4 prefix. The committed verdict records
gate code `82e4d880b807cca8`; the same SHA is present at `c62cdb3` and at the
reviewed tip. The independent run therefore reported newer HEAD `e9e0398` but
identical checker bytes, not a code change hidden by the later register commits.

## Production consumer and scoring path

Calling the real `load_gate_verdict(v4, exposure)`—and nothing downstream—
accepted and reported:

- `all_pass_recomputed=false`, derived from the predicate table;
- 18 passing / 2 policy-excused / 20 applicable;
- governed load-bearing identity `c499e4efd214a89f`;
- v4 prefix `14f77d413022a6a4`;
- exposure prefix `0a3f2e0b2cf7f788` and `exposure_matches_tape_stamp=true`;
- verdict prefix `0c1969d342a64e85`;
- the pinned ledger and clock basis from the tape header.

The remaining result path was exercised with substituted data by the production
`score_stage` entry point. It reached three non-zero receipt cells and showed:

- both candidate and incumbent vectors finite, non-constant, and distinct;
- both arms in `CAUSAL_FROZEN_FROM_TRAIN`, never retrospective top-k;
- both arms evaluated on the identical kept rows;
- non-zero economics for each arm and a real increment at every budget;
- reconciliation is read as a refusal condition, not merely printed;
- canonical ordering is applied before scoring;
- candidate and incumbent frozen threshold keys agree at 5%, 10%, and 15%;
- the CLI default remains the declared 50 ms latency.

No score-path file changed after `c62cdb3`; subsequent commits add the DA
verdict and register entries. The frozen candidate verifies directly against
freeze receipt v3, and the incumbent artifact verifies against its manifest.
The measured fit identity remains the ruled post-FD-R7 value
`e27cab9e5f6ce8e5`; `be_fitcode_rebind_v1.json` remains byte-identical to the
artifact at `6fe1c2c` and records the independently reproduced parser-only
rebind.

## Executions and resource use

```text
python3 live/pm_research/be_fragment_diagnostic.py --selftest
BE FRAGMENT DIAGNOSTIC SELFTEST GREEN: 0 failing
96 PASS lines; max RSS 220,032 KiB

python3 live/pm_research/da_state_tape_verify.py --selftest
da_state_tape_verify selftests: 141 checks passed
max RSS 93,980 KiB

python3 live/pm_research/da_state_tape_verify.py verify ...v4...
472,413 rows; 18 PASS / 2 ruled FAIL / 0 N/A; expected exit 1
elapsed 2:00.38; max RSS 73,616 KiB

load_gate_verdict(v4, exposure)
ACCEPT; elapsed 2.32 s; max RSS 200,352 KiB
```

The canonical result path
`data/pm_5min/derived/be_fragment_diagnostic_v1.json` was absent after review.

## Conditions on the released run

Run exactly once with the reviewed v4 tape and exposure, a fresh canonical
output, a reason written before execution, and the R-148(3) resource ceiling.
Do not rebuild, rebind, change latency, change either model, change thresholds,
or substitute a verdict inside this release. A suitable pre-number reason is:

```text
R-293 single authorized DIAGNOSTIC_NEVER_EVIDENCE fragment read; final
pre-score hold released by CODEX_FINAL_PRERUN_FRAGMENT_SCORE_2026-08-29.md.
```

The artifact must retain R-293/R-294's frozen interpretation: positive is weak
comfort only; negative is ambiguous and non-actionable; every outcome leaves
the race, frozen candidate, admission rule, and multiplicity untouched. The
253-window, selected, incomplete-day population remains inadmissible, and the
known gmax-tie order sensitivity remains honestly mitigated by canonical sort
rather than represented as order-independence.
