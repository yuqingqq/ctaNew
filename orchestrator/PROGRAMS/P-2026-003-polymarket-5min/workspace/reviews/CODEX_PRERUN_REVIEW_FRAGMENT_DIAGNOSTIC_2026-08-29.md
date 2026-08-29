# Codex pre-run review — fragment diagnostic machinery — 2026-08-29

**Reviewed branch tip:** `45ba8a078aceeda37828a8d57f6abf70d80081bc`

**Reviewed draft:** untracked `live/pm_research/be_fragment_diagnostic.py`,
SHA-256 `35a2bf70d9ec021883e2f85b0d1081f5a3f07c79d354d88aea5c790b249057c5`
(698 lines).

**Timing:** before any fragment score exists. I did not call `score_stage` on
the real fragment rows or state tape and produced no result number.

## Verdict

**STOP BEFORE DIAGNOSTIC SCORE.** The draft has useful fail-closed components,
but the result-bearing path is not wired to the state-tape gate, reads the
wrong ruled split, runs the incumbent retrospectively on the score population,
and consumes the evaluator's return shape incorrectly. Its green selftest does
not enter `score_stage`.

The fragment tape's separate provenance/empty-embargo stop in
`CODEX_PREFLIGHT_FRAGMENT_TAPE_2026-08-29.md` also remains in force. This filing
reviews the score machinery independently so both sets of defects can be
closed before paying for another heavy run.

## What is sound in the design

- The draft preserves the pre-registered interpretation
  `DIAGNOSTIC_NEVER_EVIDENCE`; neither a positive nor negative read may promote,
  replace, tune or reschedule the candidate.
- The exact LGBM model, value model, threshold file and scaler verify against
  freeze receipt v3; the scaler also verifies independently against the fit
  manifest.
- Candidate and incumbent score vectors are constructed on the same `kept`
  row list and their lengths must match.
- Candidate thresholds come from the frozen training artifact. The evaluator
  uses action/generation first crossing, a latency-aware value, and a
  side-by-absolute-hour matched-random control.
- The censoring receipt's committed cutoff and 253-window count are checked;
  selected/incomplete fragments are disclosed unconditionally.
- This model diagnostic does not use the broken cross-window trajectory clock;
  `harmful_action_eval` derives hour from `t0+t_start` and has no global
  rate-limit/hold state.

These are helper/capability properties. The real score composition below does
not yet preserve all of them.

## FD-R1 — the state-tape gate is not consumed

R-298's sequence makes DA gate verification a precondition to the score. The
draft has no gate-verdict path, no `all_pass` check, no gate-code identity, and
no comparison between a verdict subject hash and `tape_path`. The only DA file
read by `score_stage` is the **fragment censoring receipt**, which describes
which time fragments are inadmissible; it is not the state-tape verifier's
verdict.

Consequently, running DA's gate and obtaining a refusal would not stop this
harness. A caller can pass any parseable tape path directly to `_index_tape`
and proceed. This is rule 17 at the boundary: the checker may exist and run,
but its answer does not reach the consumer.

The tape builder also does not record the fragment exposure input hash, and
the score path does not assert a pre-registered state-tape hash or source-row
hash. A successful join shows compatible keys, not that the feature values
came from the ruled input bytes.

**Closure:** require the exact DA verdict as an input; recompute its verdict
from predicate contents; require `all_pass`; verify gate code and ledger pin;
bind its subject hash/bytes to `tape_path`; and bind the tape to the exact
fragment exposure input. A missing/mismatched verdict must refuse before
indexing.

## FD-R2 — the ruled score split is read as train

R-303 puts all 253 fragment windows in `split="score"` and makes train empty.
`score_stage` calls:

```python
_index_tape(tape_path)
```

but `_index_tape` defaults `split="train"`. An executed five-row fixture with
four train rows and one score row returned:

```text
_index_tape(path)                 -> 4 rows
_index_tape(path, split="score") -> 1 row
```

On the real ruled tape, the current call refuses with zero entries. That is a
loud failure, which is better than a wrong number, but the score path is not
runnable and the existing selftest does not exercise this call site.

**Closure:** pass `split="score"` explicitly from `score_stage`, record it in
the receipt, and add an end-to-end synthetic score-stage seam where train and
score contain deliberately different identities.

## FD-R3 — the incumbent threshold is selected retrospectively

The candidate is correctly evaluated with:

```python
theta_frozen=model["causal_thresholds"]
```

The incumbent call omits `theta_frozen`:

```python
HAE.evaluate_policy(kept, inc, latency_ms=L, budgets=budgets)
```

`evaluate_policy` therefore derives top-k cutoffs from the scored fragment and
labels the incumbent `RETROSPECTIVE_TOPK`. The committed incumbent artifact
already carries its own `causal_thresholds`; failing to pass them makes one arm
causal and the other seen-data-selected.

An executed synthetic call reports exactly:

```text
candidate threshold mode = CAUSAL_FROZEN_FROM_TRAIN
incumbent threshold mode = RETROSPECTIVE_TOPK
```

This is a decision-changing look-ahead violation, not reporting metadata.

**Closure:** pass `theta_frozen=inc_model["causal_thresholds"]`, require the
complete budget map, assert both arms and every budget report
`CAUSAL_FROZEN_FROM_TRAIN`, and include a falsifier where retrospective and
frozen thresholds select different generations/net.

## FD-R4 — the receipt loop reads the wrong evaluator level and crashes

`harmful_action_eval.evaluate_policy` returns metadata plus a nested
`budgets` mapping:

```text
top-level = budgets, latency_ms, n_actions, n_generations, n_rows,
            rows_per_action, threshold_mode, unit
budget cells = result["budgets"]["5%"], ...
```

The draft instead iterates `for b in sorted(ev_c)` and treats every top-level
value as a budget dictionary. Executed on a valid synthetic evaluator return,
the current loop reaches `latency_ms` and raises:

```text
AttributeError: 'int' object has no attribute 'get'
```

No fragment receipt can be produced through this path.

**Closure:** consume `ev_c["budgets"]` and `ev_i["budgets"]`; require exact,
equal declared budget keys; carry each arm's top-level action/row/mode metadata;
and add a seam that executes the same receipt-construction loop used by
production.

## FD-R5 — the green suite never enters the result-bearing path

The draft selftest is genuinely useful and is green at 17 checks. It covers
field/population vacuity, candidate/scaler identity, receipt cutoff, and indexer
equivalence. It never calls `score_stage`, so FD-R1 through FD-R4 are invisible.

There is no score CLI or writer. Executed:

```text
python3 be_fragment_diagnostic.py --score
usage: be_fragment_diagnostic.py --selftest | --build-rows
exit code 0
```

Thus an orchestration typo can look successful while doing no scoring—the
already-recorded false-success class.

**Closure:** add a `--score` mode requiring explicit tape and gate-verdict
paths, a fresh output, and a write reason; unknown modes must exit non-zero.
Run a synthetic end-to-end seam through the real CLI and assert the receipt
contents respond to candidate/incumbent thresholds.

## FD-R6 — partial population loss and index corruption can still pass

After `PA._feature_pass`, the only population assertion is `if not kept`.
Losing a non-total subset to `state_join_failed` or another unexpected drop is
accepted and merely printed. The score-stage population therefore is not
reconciled to the ruled exposure/state-tape population.

The local indexer compounds this:

- missing `state_status` defaults to `OK`;
- duplicate `(slug, side, gen, t_start)` keys silently overwrite earlier rows;
- the field-readability guard samples only the first 200 rows.

The state-tape gate should make malformed inputs refuse, but FD-R1 means this
consumer cannot rely on that yet. Even after wiring the gate, independent
consumer checks should refuse duplicates, absent status, and any unexpected
join loss.

**Closure:** reconcile exact source/status/index/kept counts; require
`state_join_failed=0` and explicitly rule every allowed exclusion; reject
duplicate keys and absent/undeclared statuses; report the full reconciliation
in the receipt.

## FD-R7 — T2 versus frozen identity is an unresolved design conflict

The diagnostic indexer calls `PA._stream_tape_rows`, one of the four Batch-3
T2 readers that accepts EOF before `]` as normal completion. That function
lives in `phase2_arms.py`, an identity-lattice file. Fixing it moves the
`3d0b6c8c6dfe9466` identity which this draft explicitly requires unchanged;
not fixing it violates the read-path stop; copying/bypassing it would leave the
production consumer broken and add another parser fork.

This needs an explicit pre-score resolution, not an implicit exception. The
clean direction is to commit the parser-only repair, prove its semantic
invariance on complete tapes plus refusal on truncation, and rebind/rederive
the affected artifact chain under the coordinator's citation-correction rules.
Do not use the diagnostic's zero-identity assertion to preserve a known
fail-open reader.

## FD-R8 — this is not the strategy baseline comparison

The draft compares `LGBM_PINNED` with the model incumbent and matched random in
an abstract first-crossing cancel evaluator. It does **not** run
`QR_CANCEL_HOLD_X_SKEW` or the required `QR_SKEW_ONLY` comparator, and it does
not simulate cancellation effectiveness, limiter suppression, hold/repost, or
skew interaction.

That is acceptable only as a narrowly labeled **model diagnostic**. It cannot
be reported as market-making strategy performance or as evidence that
cancel+hold×skew beats skew-only. The next result-bearing strategy iteration
must retain those two named arms, the queue-realistic exposure path and an
absolute trajectory clock.

## Executed review record

- source compiled in memory: PASS;
- draft selftest: 17/17 green;
- no-op CLI known-bad: `--score` prints usage, exits 0;
- split fixture: default index 4 train rows versus explicit score index 1;
- evaluator-shape fixture: current receipt loop raises `AttributeError`;
- threshold-mode fixture: candidate causal versus incumbent retrospective;
- static call/path audit: no state-tape gate-verdict consumer and no
  `QR_CANCEL_HOLD_X_SKEW`/`QR_SKEW_ONLY` strategy arms;
- **real fragment score not run**.

## Minimum safe sequence

1. Close the fragment tape provenance/empty-embargo filing and all four T2
   consumers, resolving the `phase2_arms` identity conflict explicitly.
2. Close FD-R1 through FD-R7 and commit the harness before any number exists.
3. Run the synthetic CLI seam, including known-bad gate, split, threshold,
   evaluator-shape, duplicate and partial-drop cases.
4. Request a pre-run re-review of the committed machinery.
5. Only after clear release, execute the single diagnostic run. Preserve its
   `DIAGNOSTIC_NEVER_EVIDENCE` interpretation.
6. Do not call the output strategy performance; the eventual integrated
   experiment keeps `QR_CANCEL_HOLD_X_SKEW` as baseline and `QR_SKEW_ONLY` as
   comparator.
