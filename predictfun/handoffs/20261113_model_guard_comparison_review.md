# What the guard comparison isolates

Static source review as of 2026-11-13T02:15:29.597599+00:00. The cited files match the
recovery experiment's source declaration at
`78285fe02d00962abc5f7148e6001a8f39a0ddc6`. This review adds no model,
threshold, historical measurement or changed policy.

The portfolio comparison includes a difference in admission and reaction
timing, in addition to the learned prediction target and features.

- `pf_order_policy_engine.source()` passes the hook a callback to the frozen
  engine's `guard_state()` with its G1/G2 settings. The declared base rules
  contain G1 and G2.
- `pf_step4_ladder_v3.guard_state()` queries the last Binance observation at
  the modeled knowledge clock. G1 uses the trailing one-second movement;
  G2 uses the previously computed trailing 250 ms movement class. This
  callback is a deterministic movement rule, not another fitted forecast.
  The simulation's modeled knowledge clocks remain an execution assumption.
- In `pf_model_eval_policy.install.Hook.targets()`, the `causal_guard` branch
  applies that callback and its earlier-trained score/line before the code
  reaches the accepted-order age check. It can withhold an initial quote and
  react at ages below 100 ms. Its trained score/line chronology remains covered
  by the existing fit/audit receipts; this source review does not independently
  re-estimate or certify that entire training pipeline.
- STATIC10 and HISTORY10 first call the reconstructed base `default` targets,
  then score only an existing, nonpending order when `due()` permits it. That
  function requires at least 100 ms of order age and 100 ms between scores.
  The base `default` hook does not apply the G1/G2 movement callback as a
  withdrawal trigger. It retains its data, price, size and inventory checks.
- The frozen recovery experiment uses the same `due()` gate. Its added
  recovery logic changes entry after an explicit model pull; it does not add
  the guard's immediate movement veto to all initial quotes or replacements.

HISTORY10 versus STATIC10 is the comparison with the same post-placement
policy and scoring schedule for the temporal-feature change. HISTORY10 versus
the chronological guard is a comparison of complete policies: prediction,
admission and immediate movement responses differ. The latter remains a valid
incumbent strategy benchmark; it is not an isolated estimate of the effect of
adding temporal features or changing model capacity.

The [fill-timing accounting](20261113_model_cancel_timing_review.md) documents
negative realized fill-mark value before the learned value policies' earliest
possible cancellation effect. Taken together, the source and value reviews
explain why an improvement in offline prediction error need not fix the
quoting policy. They do not identify which alternative veto, model or price
would have avoided those fills, or prove that copying the movement guards
would improve the candidate. No rule is changed from this review.

The original primary/stress failure and the queued recovery experiment remain
unchanged. If the full recovery experiment fails, a later controlled study
can separate fresh-entry value and immediate safety responses from the
post-placement cancellation head. That would require its own declared
comparison; it is not included as an undeclared variant here.

Sources: [model policy](../research/pf_model_eval_policy.py),
[base targets](../research/pf_default_pull_hooks.py),
[callback wiring](../research/pf_order_policy_engine.py),
[frozen engine](../research/pf_step4_ladder_v3.py),
[base specification](../research/pf_order_policy_spec.py),
[recovery policy](../research/pf_recovery_experiment_policy.py).
Exact file identities are in
[the source-review receipt](../reviews/model_upgrade_20261112_v2/guard_comparison_source_review.json).
