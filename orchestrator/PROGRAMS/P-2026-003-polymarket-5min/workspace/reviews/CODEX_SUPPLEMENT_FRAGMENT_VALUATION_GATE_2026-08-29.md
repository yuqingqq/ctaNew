# Codex supplement — fragment valuation gate projection — 2026-08-29

**Exact reviewed tip:** `22ac05363436d5b6d585c22753c4faa74650807f`

**Scope:** the interaction between
`be_fragment_diagnostic.score_stage`, `phase2_arms._feature_pass`, and
`harmful_action_eval.evaluate_policy`. No real fragment score was run and no
diagnostic result was produced.

## Verdict

**PRE-RUN HOLD MAINTAINED.** PR3-FD1 has a silent second half. The loud missing
`status` assertion currently stops the run. If that assertion were merely
removed, the same projection also lacks `any_fill_ahead`, so the evaluator
would value every selected action at zero and could publish a fabricated clean
negative result.

## Executed mechanism

`phase2_arms._feature_pass` projects a source row onto eight fields:

```text
slug, day, t0, t_start, side, gen, latency, coin
```

It omits both `status` and `any_fill_ahead`. The fragment harness passes these
projected rows directly to `harmful_action_eval.evaluate_policy`, whose value
function is:

```python
preventable_value_cents if row.get("any_fill_ahead") else 0.0
```

Executed one-action falsifier, holding the score, latency payload, frozen
threshold, and cancellation fixed:

```text
projected row, gate absent: net=0.0,   harm_avoided=0.0,   cancelled=1
same row, gate true:         net=123.0, harm_avoided=123.0, cancelled=1
```

The action still cancels in both cases. Only the absent dictionary key changes
the economic result. Therefore an all-zero receipt would look like model
failure while actually measuring a projection error.

## Historical-scope check

This specific defect does **not** by itself invalidate the existing phase-2
four-arm receipts. The artifact-producing `phase2_arms.stage_score` path does
not pass `sc["kept"]` directly. It first executes:

```python
srows = [harmful_hazard_model.keptrow(r) for r in sc["kept"]]
```

`keptrow` reconstructs `any_fill_ahead` from the canonical latency predicate
before evaluation. I verified that composition at the recorded score ref
`e12e2c70c133a0034336b6370f70cc3ab3aecc72` as well as the current source.
The pending fragment harness omits that composition, which is why its path is
affected while the prior phase-2 path is not.

## Closure

- Preserve/reconstruct both `status` and `any_fill_ahead` by an identity-exact
  source-row join; do not relax the current status assertion.
- Require each kept projected identity to match exactly one source exposure
  row; missing and duplicate source identities refuse.
- Refuse a kept row whose valuation gate is absent or not boolean.
- Add the executed missing-gate case as the end-to-end positive control's
  known-bad twin.
- Require the synthetic `score_stage`/CLI positive control to reach cells with
  a deliberately nonzero economic value. Reaching all-zero cells is not proof
  that the valuation path is wired.

These conditions supplement, rather than replace, the six release conditions
in `CODEX_REREVIEW_FRAGMENT_DIAGNOSTIC_PRERUN_2026-08-29.md` and the R-313
exact-field condition in
`CODEX_EARLY_REREVIEW_R313_CONSUMER_CHECK_2026-08-29.md`.
