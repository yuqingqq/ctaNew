# Codex early re-review — R-319 / v4 preflight — 2026-08-29

**Exact reviewed tip:** `95aeda4b5aaf6aa184419feecba5411ed9b894cb`

**Scoped commits:** DA `5c24621` + `1dd891f`; BE `6982025` +
`cf0bad5`; coordinator launch record `95aeda4`.

**Scope boundary:** source, synthetic controls, adversarial gate/rejoin fixtures,
and live v4 launch provenance/resource state. No fragment score was run and no
diagnostic number was read. The v4 build may finish; this filing does not
authorize its score.

## Verdict

**R-318 DEAD-ARM AND BUILDER-RACE REPAIRS VERIFIED. FD4 HOLD MAINTAINED. THE
SINGLE REAL FRAGMENT SCORE STAYS DARK.**

The dead-incumbent control now bites at the production score seam. The builder
now hashes each exposure input before consumption, hashes again afterward,
refuses any movement, and stamps the verified digest. DA's exact carried-set
and outer-object repairs are present, its eight-name load-bearing identity is
stable and order-independent, and both exact-tip suites pass.

The gate consumer still does not enforce the ruled predicate state. It treats
the two policy-excluded failures as if they were general applicability/pass
waivers. Executed verdicts with `both_splits_populated` made N/A, with that
predicate made passing, and with a new predicate made N/A are all accepted.
That contradicts both DA's always-applicable load-bearing contract and R-310's
binding to exactly the two named failures.

A narrower FD1 integration defect also remains: the strict valuation-input
validator is called after `hm.keptrow` reconstructs the gate. A malformed
scalar latency cell therefore raises raw `AttributeError` before the declared
validator can issue its controlled refusal. This is fail-loud rather than a
silent wrong number, and the current artifact's earlier whole-stream census is
clean, but the advertised pipeline wiring and its unit-only falsifier do not
agree.

## Exact-tip executions

```text
python3 live/pm_research/be_fragment_diagnostic.py --selftest
BE FRAGMENT DIAGNOSTIC SELFTEST GREEN: 0 failing
84 PASS lines; elapsed 1.12 s; max RSS 219,980 KiB

python3 live/pm_research/da_state_tape_verify.py --selftest
da_state_tape_verify selftests: 141 checks passed
elapsed 1.04 s; max RSS 93,788 KiB
```

The clean review worktree used the repository's exact `95aeda4` source and a
read-only link to the live ignored data ledger. No producing source was edited.

One full-stream regression over the known-good 1,764,206-row tape also returned
`all_pass=true` and exact distribution `{48: 1764206}`, with 73,564 KiB peak
RSS. I do **not** use that run as exact-ref provenance evidence: it started in
the primary worktree and the branch advanced while it ran, so its late
`gate_code_identity` named newer bytes than those loaded at process start. The
substantive result supports the exact-set derivation, but the fresh v4 gate
receipt remains the attributable full-stream proof.

## Closures verified

### 1. Dead-incumbent liveness — closed

Replacing `phase2_iter011_run.apply_incumbent` with a same-length vector of
zeros now refuses inside real `score_stage`:

```text
DiagnosticRefused: the incumbent score vector is CONSTANT
(1 distinct value over 240 rows)
```

The ordinary synthetic run carries 240 finite/distinct candidate scores, 240
finite/distinct incumbent scores, non-zero economics for both arms, and
different economics at all three budgets. This closes the prior control that
proved the threshold path but not the scoring arm.

### 2. Builder consume/hash race — code closure verified

`build_state_tape_v2.main` computes `_in_before` before either exposure is
loaded, computes `_in_after` after all rows are consumed, refuses any differing
split, and stamps the equal digest plus `input_hash_protocol`. The v4 artifact
must still be checked for those fields and their equality when it lands.

At inspection, the live v4 unit was bounded and attributable to the repaired
builder:

```text
unit                 be-frag-tape4-1788004676.service
BUILD_REF            cf0bad555b4a656719769147228f13fd5c5fa378
LEDGER_PATH           ledger_pin_fragment_v1.jsonl
builder sha256        f870310beb707d4508511210295edac43...
cf0bad5 blob sha256   f870310beb707d4508511210295edac43...  (equal)
MemoryMax             10G
MemoryPeak observed   4,965,154,816 bytes
```

The concurrent v3 unit is correctly classified rehearsal-only. Its
`BUILD_REF=38859f6...` does not identify the later source bytes it actually
loaded, so no v3 value or receipt may substitute for v4.

### 3. DA exact-set / outer-object machinery — source and controls verified

The consumer now requires exact equality to the header/schema-derived carried
set on every streamed row, emits `per_row_conformance_exact` as always
applicable and load-bearing, and refuses missing outer `}`, trailing bytes, and
an unclosed rows array. DA's load-bearing identity recomputes to
`c499e4efd214a89f` over eight sorted names; adding a name moves it and ordering
does not.

Final release still requires DA's fresh verdict over the exact v4 bytes. The
old or moving-worktree regression is not a substitute.

## Finding 1 — FD4 still confuses an excused failure with a waiver

DA states in code that `both_splits_populated` is **ALWAYS APPLICABLE**,
**LOAD_BEARING**, and non-waivable. R-310 permits its known failure for this
diagnostic population; it does not permit making the check disappear.

I built a verdict containing every governed name exactly once and correct
subject/exposure hashes, but marked both ruled names
`applicable=false, pass=false`. `load_gate_verdict` returned:

```text
BOTH_SPLITS N/A: ACCEPT
predicates_failed_and_EXCUSED_by_policy = {}
```

The consumer admits this because `illegal_na` excludes every name in
`DIAGNOSTIC_PREDICATE_EXCLUSIONS`, including the non-waivable split check.

It also accepts the other wrong direction:

```text
both_splits_populated applicable=true, pass=true
result: ACCEPT
excused failures: {embargo_respected only}
```

Thus the consumer does not require the failed set to be exactly the ruled pair.
The positive fixture happens to use the right shape, but there is no known-bad
for either incorrect state.

### Required closure

Require the governed state explicitly, not by category:

- `both_splits_populated`: exactly once, `applicable is True`, `pass is False`;
- `embargo_respected`: exactly once in the precise state DA emits for v4
  (expected from current DA code: applicable and failing on the empty split);
- every other governed predicate: exactly once, applicable and passing;
- the recomputed failed set: exactly the two policy-fixed exclusions.

Add known-bads for an exclusion that passes, an exclusion made N/A, and an
exclusion missing from the recomputed failed set.

## Finding 2 — a new N/A predicate is silently accepted

R-310 allows a newer **passing** predicate while treating any other failure as
a refusal. Adding this row to an otherwise correctly shaped verdict is
accepted:

```json
{"predicate":"brand_new_check","applicable":false,"pass":false}
```

That predicate is excluded from `applicable`, never reaches `unexcused`, and
is not in the fixed governed-name set. A new check can therefore be disabled
without being either passing or policy-excluded.

### Required closure

Every additional predicate must be unique, explicitly applicable, and passing,
unless a separately ruled N/A name/state is declared. Add a new-N/A known-bad;
retain the existing new-passing positive control.

## Finding 3 — FD1 validates after reconstruction

`rejoin_source_fields` currently executes:

```text
row[any_fill_ahead] = hm.keptrow(row)[any_fill_ahead]
...
assert_valuation_inputs(kept, 50)
```

despite its comment and contract saying validation occurs first. Through the
actual rejoin seam, `latency["50"] = 7` produces:

```text
AttributeError: 'int' object has no attribute 'get'
```

The suite's scalar-cell known-bad calls `assert_valuation_inputs` directly, so
it cannot detect this wiring order. Tolerated malformed shapes such as
`latency=None` do reach `DiagnosticRefused`; the gap is specifically between a
green validator unit and the integration that calls it.

### Required closure

Move `assert_valuation_inputs` before every `hm.keptrow` call, then add a seam
test that passes the malformed scalar cell through `rejoin_source_fields` and
asserts `DiagnosticRefused`. Keep the legitimate zero-fill structural control.

## v4 sequencing

Allow the bounded v4 build to finish. Then:

1. verify its builder/source/input/ledger stamps and that the pre/post-hash
   protocol is carried;
2. DA gates the exact v4 bytes and writes a fresh, subject-bound verdict;
3. repair Findings 1–3 and rerun both exact-tip suites plus the adversarial
   fixtures;
4. request the final pre-score re-review.

Until that review explicitly releases it, **do not run the one real fragment
score**. This hold does not alter the frozen candidate, the
`QR_CANCEL_HOLD_X_SKEW` baseline, or the required `QR_SKEW_ONLY` comparator.
