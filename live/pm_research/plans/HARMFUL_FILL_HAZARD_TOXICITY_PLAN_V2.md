# Harmful-fill recovery plan v2 — control first, cascade before promotion

**Status:** V2 TERMINALLY STOPPED AT 1/7 / GATE 1D FINITE ACTING CONTROL GREEN /
GATE 1E LIFECYCLE LEDGER COMPLETE BUT GATE 1 REFUSED FOR UNAVAILABLE
OWNED-ORDER PER-FILL MAKER FEES / GATE 1F INPUT AUDIT REFUSED—NO OWNED
EXECUTION EXPORT / GATES 2–6 NOT STARTED / NOT FROZEN
**Authorised:** direct user instruction, 2026-09-04T15:27:56Z; coordinator
register citation pending
**Gate-1c authorisation:** direct user instruction to continue,
2026-09-05T00:58:42Z; this authorises only the different control estimand
declared below, not a larger or tuned rerun of either failed sampler
**Gate-1d authorisation:** direct user instruction to continue,
2026-09-05T05:03:29Z; this authorises only the finite cyclic-phase estimand
declared below, not any change to the three consumed Gate-1 failures
**Scope:** offline research only; no live cancellation path, venue adapter,
execution server or heavy data/model run is authorised by this document
**Supersedes prospectively:** HARMFUL_FILL_HAZARD_TOXICITY_PLAN.md
**Preserved provenance:** the v1 plan, its frozen declarations and every result
remain historical evidence. Nothing here re-scores or makes a seen day new.

This is the governing plan for new P-2026-003 work. It changes the order of
proof after the 2026-09-04 result review. It does not change the cost model,
rewrite a frozen artifact or promote a model.

## 1. Why the plan changed

The latest result-bearing surface supports only a narrow conclusion:

- the absolute economics come from one BTC development hour (12 windows,
  4,315 fills), with no day-cluster interval and no net-profitability claim;
- the broad survey never ran; survey.py failed on the real producer shape and
  there is no 40-cell, 138-cell or 600-cell survey result;
- the measured 701% V_oracle ratio is a tranche-level, perfect-foresight
  static ceiling without a declared random/naive null and without the
  cancellation cascade;
- both tested rankers had negative capture against that ceiling, but this is a
  development diagnostic, not independent validation;
- the earlier G=5 race is directional only. With recorded multiplicity 2,
  its best possible Holm-adjusted p-value is 0.0625; a significance-bearing
  race needs at least G=6 and a newly committed pipeline.

The old plan put broad seven-arm integration and fair-price work ahead of the
missing control/attainability proof. V2 reverses that dependency. First prove
that the exact action unit and matched control are well-defined; then put them
through the stateful cascade and full economics; only then spend more compute
on extra prediction modules.

## 2. Claims and names that are now fixed

1. **V_oracle is descriptive only.** It is the observed tranche filter from
   the static reference path. It is not an incumbent, model or achievable
   cancellation policy.
2. **The incumbent is the policy baseline.** QR_SKEW_ONLY is the unchanged
   neutral no-prediction reference; where the lifecycle protocol requires it,
   QR_CANCEL_HOLD_X_SKEW is also reported. “Incumbent” never means the oracle.
3. **One row is one action.** The canonical unit is one declared decision for
   one (slug, side, generation). Repeated score rows cannot multiply one
   economic outcome.
4. **Static action-bundle ceiling is not cascade-feasible.** Selecting whole
   generations at the same side/hour action budget is closer to an attainable
   policy than selecting individual fills, but it still assumes the later
   neutral path survives cancellation. Only stateful replay may carry the word
   “cascade”.
5. **Models estimate; gates decide.** No model artifact emits a promotability
   boolean.
6. **No repaired old survey.** A future breadth measurement is a newly
   declared v2 receipt after the statistic, null, producer shape and complete-
   day population are fixed. The failed ad-hoc script is not resurrected.

## 3. Resource envelope

Until the user explicitly widens the envelope:

- documentation, source inspection and synthetic selftests are allowed;
- no model fit, broad raw-tape replay, grid, cache rebuild or survey is run;
- no cache is deleted;
- every new real runner starts with a one-cell/one-window smoke receipt and a
  stated memory/CPU cap;
- widening from smoke to development cells requires the smoke artifact to
  reconcile and report all exclusion statuses.

## 4. Revised gated plan

### Gate 0 — canonical identities and a declared static null

Build and verify:

- a producer-side action/fill ledger before tranche identity is lost;
- exactly one action per (slug, side, generation) with an event-carried
  decision time;
- one ledger fill per reference tranche, distinguished by stable ordinal even
  when timestamps coincide;
- latency classification at decision_t + L;
- explicit statuses for pre-effective fills, missing markout, missing shares,
  source exclusions and unmatched reference generations;
- a uniformly random action-bundle null matched exactly on action count, maker
  side and UTC hour, with at least 200 draws fixed before values are read;
- a perfect-foresight whole-action ceiling at the identical stratum budgets.

Current implementation:

- de_action_economic_ledger.py — built; 14 lightweight synthetic checks pass.
- de_action_bundle_control.py — built; 12 lightweight synthetic checks pass.
- de_canonical_action_population.py — built; 11 lightweight synthetic checks
  pass, including the no-fallback exclusion control.
- de_v2_gate0_runner.py — built; nine end-to-end synthetic seam checks pass.
- de_phase4_diag_runner.build_reference has an opt-in v2 producer mode that
  retains missing-markout tranche identities with stable source ordinals; the
  historical/default reference shape remains filtered.

The capped real Gate-0 smoke completed at 2026-09-04T16:06:44Z:

- artifact `p003_v2_gate0_smoke__20260904T160623Z.json`, sha256
  `d63ccc59bae1e733a2fa4c840ce2b1c2bdc33494ce989df55f3816f4fd906be7`;
- one consumed-development BTC window,
  `btc-updown-5m-1787579400`, selected without outcome/value fallback;
- 5,869 source rows -> 3,557 canonical actions, all status `OK`, exactly equal
  to 3,557 neutral-reference generations;
- 458 exact fill identities: 367 `PRE_EFFECTIVE_FILL` and 91
  `PREVENTABLE_VALUED_FILL`; the fill statuses partition the population;
- 355 deterministic probe actions and 200 random action-bundle draws matched
  on count, maker side and UTC hour; all six computed identity/null predicates
  are true;
- 11.70 s wall time and 347,080 KiB maximum RSS under a one-CPU/3 GiB external
  cap.

This is a pipeline receipt, not an economic result. The deterministic 10%
hash-ranked probe is wiring only and was fixed without looking at fill value.
The receipt remains uncommitted/unfrozen and its static control is not
cascade-feasible.

The first smoke attempt is retained as a superseded resource finding. The
historical selector built a global Binance gap index before applying the
one-window limit, reached the 3 GiB job cap and was manually stopped without an
artifact. `de_v2_local_selector.py` replaced that smoke-only path with an exact
interval-local continuity check over three adjacent hourly files; the retry is
the successful receipt above. No economic observation came from the failed
attempt.

A receipt that reports missing markouts but does not opt into identity
retention is marked incomplete and the control refuses it. In v2 mode, the
identities remain present with `markout_cents_per_share` null and the affected
actions remain explicit excluded statuses.

**Gate-0 exit: CLEARED for pipeline identity, 2026-09-04T16:06:44Z.** The real,
consumed-development smoke receipt has exact action
and fill reconciliation, statuses partition both populations, the declared
null contains at least 200 matched draws and no result is described as net or
cascade-feasible.

### Gate 1 — acting matched control through the stateful cascade

Implement a runner adapter that turns each selected generation action into an
actual cancel request on an independent replay clock. The treated model and
each random draw must use the same neutral opportunity population and match
action count by side/hour. Equality between a forced random draw and treatment
is legal; it is arithmetic when a stratum has no slack.

Every arm reports:

- requested, passed, stale, unresolved and effective cancels;
- prevented and sacrificed fills/shares after latency;
- reposts, queue resets and reset cost;
- complete maker P&L decomposition without double-counting spread;
- terminal/peak inventory, inventory loss and settlement/mark status;
- fees and every still-unpriced term as an explicit status;
- net action value and rho = adverse / spread only when their denominators
  and all required terms are complete.

**Gate-1 exit:** the acting matched control completes a synthetic falsifier and
a one-cell consumed-development smoke replay with all accounting identities
green. A static action-bundle result cannot substitute for this gate.

**2026-09-04 acting-null checkpoint — ordinary rejection is ruled out.**
`de_v2_acting_matched_control.py` passes 18 synthetic checks, including a
positive planted-harm control and a stateful falsifier where equal selected
budgets realise unequal cancel counts. `de_v2_gate1_smoke.py` passes seven
synthetic wrapper checks. The capped real smoke then correctly refused:

- failure receipt `p003_v2_gate1_smoke__20260904T162106Z.json`, sha256
  `ede26d60fdb425e9d760adca48e24191620c2fb15a5fe70028e124e758b1ebc9`;
- only 1 of 4,000 independent uniform score permutations matched the treated
  realised cancel count by side/hour, versus 200 required; 3,999 were rejected
  under `realised_action_count_mismatch`;
- 4m44s on one CPU, 250.5 MiB peak under the 3 GiB cap;
- no partial null and no economic statistic were published.

This repeats the structural warning in the earlier 60-draw Phase-4 diagnostic:
uniformly scattering above-threshold events changes hold/repost suppression and
therefore usually changes action count. Increasing the rejection budget after
seeing 1/4,000 would not fix the estimand and is prohibited.

Gate 1 therefore remained open. Its next attempted sub-gate was a **constrained switch
null** over the exact realised-action-count fiber: begin at the feasible treated
assignment; propose symmetric high/low score swaps within side/hour; replay the
whole state machine; accept only states that retain the exact realised counts;
and retain rejection as a self-loop. Burn-in, thinning, chain count, sample
count, support/mixing checks and a degeneracy refusal must be fixed in code
before another real receipt. The stationary target is uniform only over the
connected component reached by the declared switch chain, so the receipt must
say that explicitly; it must not silently claim the original iid-permutation
null. If the chain cannot leave treatment or has inadequate effective support,
Gate 1 refuses and the route stops pending a genuinely different declared
control.

**2026-09-04 constrained-null result — support exists, mixing fails.** The
switch module passes 13 synthetic checks and its fixed smoke wrapper passes
seven in both script/module invocation modes. The real capped retry emitted
`p003_v2_gate1_switch_smoke__20260904T163438Z.json`, sha256
`cdff1a14de7ecff3351dc90da224e2d44bad43d13ef67e66934b859d777a36a9`, and
refused on its predeclared effective-sample-size bar:

- 5,000 symmetric proposals yielded 2,443 exact-fiber moves and 2,557
  realised-count-mismatch self-loops (48.86% move acceptance);
- all four chains left treatment; 400 post-burn/thinned samples contained 399
  distinct states and zero identity samples;
- every proposal/sample stateful, score-multiset, realised-count and source
  identity was true;
- distance diagnostic R-hat was approximately 1.0, but effective sample size
  was only 10.53 versus the fixed minimum 100;
- every chain was still moving away from treatment: mean switched high-score
  positions rose from 114–129 in its first 20 retained samples to 237–251 in
  its last 20. The 250-step burn-in was therefore inadequate, not merely a
  conservative ESS formula;
- 249.89 s wall / 250.50 s user CPU; receipt `ru_maxrss` 333,856 KiB, while the
  systemd job reported a 250.8 MiB cgroup peak; both are far below 3 GiB.

The aggregate matched-null metric is absent because the support/mixing gate is
red. Individual partial state values remain in the audit trail and carry the
explicit `INCOMPLETE_NOT_STRATEGY_NET` status; they are not interpreted.

Post-receipt verification at 2026-09-04T16:44:23Z: all ten v2 module
selftests pass under a one-CPU/1 GiB cap, and the parent
`de_phase4_diag_runner.py` suite passes 223/223 after restoring its static
reached-function pin. That compatibility edit makes the current tree differ
from each successful smoke receipt at exactly one named identity file,
`de_phase4_diag_runner.py`; the injected local-selector branch used by the
smokes is semantically unchanged. The receipts retain their own hashes and
remain uncommitted/unfrozen historical executions; no real rerun was made after
the gated stop.

The v2 stopping rule now applies. Extending burn-in/thinning or changing the
kernel after seeing this trajectory would adapt the null on consumed data.
Gate 1 is refused, Gates 2–6 do not start, and the long loop stops after docs
and verification. Resumption requires an explicit new control estimand and a
new prospective declaration—for example, a sequential random action-quota
policy rather than a score-permutation null—not a larger retry of either
failed sampler.

**2026-09-05 user continuation — Gate 1c prospective declaration.** The user
authorised the different control estimand at 00:58:42Z. Before implementation
or inspection of any Gate-1c output, the following design and refusal rules are
fixed:

- The treated target remains its actual state-machine `CANCEL_ISSUED` count in
  each `(maker_side, UTC_hour)` stratum. It is not requested crossings, static
  selected rows or a post-hoc economic subset.
- A proposal is an independent uniform permutation of the treated score
  multiset within side/hour over the same neutral canonical action keys. This
  proposal distribution is used only to randomise action timing; it is not the
  previously failed iid exact-count rejection null.
- Replay the proposal through the unchanged stateful cancel/hold/repost
  cascade. If its uncapped issued count is below treatment in any stratum,
  reject it as `UNDER_QUOTA`; a later action is never forced and a held/non-live
  order is never counted as an action.
- Otherwise, keep the proposal until the target-th issued cancel in each
  stratum and sequentially suppress later above-cancel crossings in that
  stratum. Suppression replaces only those later scores with the fixed midpoint
  `(theta_cancel + theta_repost) / 2`: it cannot issue a cancel and, like an
  above-cancel score, it remains above the repost threshold. This preserves the
  post-quota hold/repost-clock meaning without force-cancelling or consulting
  fill, markout or P&L values.
- Replay the capped stream independently. It must produce exactly the treated
  issued-action count in every side/hour stratum; canonical keys, decision
  times and source identity must remain unchanged. Any exact-count failure,
  prefix-causality failure or economics identity failure refuses the run.
- The resulting null is the **sequential random action-quota policy induced by
  uniform score proposals and a hard issued-cancel stop**. It is not uniform
  over the exact-count fiber and must never be labelled the iid-permutation or
  switch-chain null. Proposal scores after the quota are intentionally
  suppressed; pre-cap score-multiset identity and every suppression count must
  be reported separately.
- Gate-1c v1 is scoped to `protection_mode=ALL_ORDERS_OVERRIDE` and
  `enable_reduce=false`, matching the fixed smoke. It refuses inventory-coupled
  reducing-side protection rather than assuming the per-side prefix argument
  still holds; that interaction belongs in the later full Gate-1 integration.
- The fixed real-smoke requirement is 200 independent accepted draws from at
  most 1,000 proposals on the same consumed one-window development cell. No
  attempt-budget, threshold, seed rule or suppression rule may be changed after
  seeing the result. Accepted-state distinctness and under-quota rejection
  counts are mandatory diagnostics; at least 50 distinct realised action sets
  are required. `accepted < 200` or distinctness below 50 refuses without a
  partial null.
- Synthetic gates must include a planted-harm positive control, an under-quota
  known-bad refusal, an exact-quota stateful case, and a post-quota
  repost-clock equivalence check. The real smoke may start only after all pass.
- Synthetic work runs under one CPU/1 GiB. The real fixed smoke runs under one
  CPU/3 GiB and may read only the already proven three-file interval-local
  slice. No fit, broad replay, survey, grid, cache rebuild or Gate-2 work is
  authorised by this declaration.

Passing Gate 1c would establish a feasible acting matched-volume comparator,
not strategy profitability and not all of Gate 1. Gate 1 clears only when that
control is joined to a complete lifecycle economic ledger with the accounting
identities listed above. Failure of the fixed synthetic or real Gate-1c gate
returns the programme to a gated stop; the two earlier null receipts remain
consumed failures either way.

**2026-09-05 Gate-1c result — fixed support gate refused.** The new controller
passes 14 synthetic checks and its fixed wrapper passes eight under the
one-CPU/1 GiB cap. The single authorised real smoke then emitted
`p003_v2_gate1_quota_smoke__20260905T010921Z.json`, sha256
`e10dec7167a1b61a17c87b3ff0d19cd6c11692a6280035181e9cf5f1985a2ab8f`,
and refused without an aggregate null:

- the treated actual-action quota was 150 `BUY_UP` and 110 `SELL_UP` cancels
  in UTC hour 13 on 3,557 canonical actions (5,869 source rows);
- only 16 of the fixed 1,000 independent proposals reached both quotas, versus
  200 required; all 984 rejections were `UNDER_QUOTA`;
- those 16 audit-trail draws produced only 16 distinct realised action sets,
  below the declared 50 minimum, and 376 later high-score events were
  suppressed across them;
- every accepted draw exactly matched the issued-action quota and all accepted
  source, prefix, score-suppression and stateful identities were true. The
  treated replay was deterministic and source inputs were unchanged;
- the matched-null status is `ABSENT_REFUSED_SUPPORT_GATE`; its aggregate
  partial metric is null. The 16 retained draws are mechanics audit records,
  not a smaller null and not economic evidence;
- wall time was 82.81 s with 83.45 s user CPU and receipt `ru_maxrss` 337,028
  KiB under the one-CPU/3 GiB external cap. The transient-unit summary reported
  only 1.2 MiB peak, inconsistent with the process observation, so the receipt
  RSS is the conservative resource figure.

The construction therefore fails its prospective support gate. Its budget,
seed rule, thresholds and suppression rule must not be changed on this consumed
window. Gate 1 remains refused, overall v2 progress remains 1 of 7, Gates 2–6
do not start, and the long loop returns to a gated stop. This is a control-
feasibility result: it does not show that treatment is economically beneficial
or harmful. Final consolidated verification at 2026-09-05T01:16:00Z: all 12
v2 module/wrapper batteries and the 223-check parent diagnostic suite pass
under a one-CPU/1 GiB cap. Exact receipt-to-current-tree verification finds 11
of 12 named identity files byte-identical; the only drift is this plan itself,
because its prospectively hashed declaration was subsequently extended with
this result/status record. No named source-code file changed after the receipt.

**2026-09-05 user continuation — Gate 1d prospective declaration.** The user
authorised another genuinely different control estimand at 05:03:29Z. This
design responds to the measured mechanism in the earlier refusals: iid
permutation destroyed score clustering, while local exact-fiber switches did
not mix. It does not enlarge or tune either route. Before implementation or
inspection of any Gate-1d output, the following finite design is fixed:

- Use the same neutral canonical action keys, decision timestamps, treated
  scores, policy parameters and actual state-machine `CANCEL_ISSUED` targets by
  `(maker_side, UTC_hour)`. No fill, markout or P&L value enters phase selection.
- Inside each side/hour stratum, order canonical keys by window epoch, slug,
  decision time and generation. A control phase is one circular rotation of
  the complete treated score-value sequence over that ordered key sequence.
  Offset zero is treatment and remains in the finite support if it satisfies
  the same mechanical predicates.
- A rotation preserves the exact score multiset and the circular sequence of
  adjacent score transitions. It randomises only the phase of that clustered
  sequence against neutral opportunities; the receipt must not call this iid,
  a uniform action-set null or the prior switch-chain target.
- Gate-1d v1 is limited to the fixed one-window smoke,
  `protection_mode=ALL_ORDERS_OVERRIDE` and `enable_reduce=false`. With those
  restrictions each side is mechanically separable. Enumerate **every** cyclic
  offset for each side/hour on a fresh side-isolated replay and retain only
  offsets whose actual issued count equals treatment. No proposal cap,
  rejection-budget extrapolation, force-cancel or quota suppression is used.
- Deduplicate mechanically accepted offsets by their full rotated score
  assignment. The joint finite support is the Cartesian product of the unique
  accepted phase assignments across strata. It must contain at least 200
  distinct joint assignments; otherwise refuse with no smaller null.
- When support is sufficient, draw exactly 200 joint assignments uniformly
  **without replacement** from that finite Cartesian product using the fixed
  seed `20260905`. Replay every composed draw through the full stateful engine.
  Every draw must match actual side/hour action counts and pass source, score,
  decision-time, separability and stateful identities. A single mismatch
  refuses the control.
- Synthetic gates must include a planted-harm positive control with at least
  200 finite phases, a known-bad support-shortage refusal, a complete-enumeration
  count check, a nonzero phase that preserves clustering/multisets, and a
  deliberately inventory-coupled protection refusal.
- Synthetic work runs under one CPU/1 GiB. The single real enumeration/smoke
  runs under one CPU/3 GiB, swap disabled, with a ten-minute runtime ceiling,
  and may read only the proven interval-local three-file slice. No fit, broad
  replay, survey, grid, cache rebuild or Gate-2 work is authorised.

A green Gate 1d receipt would establish this finite clustered-phase acting
comparator only. It would not validate the predictor, establish profitability
or clear all of Gate 1; the complete lifecycle ledger would still be required.
Failure of synthetic identities, finite support, full composed replay or the
resource ceiling returns the programme to a gated stop. The three previous
control receipts remain consumed failures and are never pooled with Gate 1d.

**2026-09-05 Gate-1d result — finite acting support cleared.** The cyclic
module passes 13 synthetic checks and its fixed wrapper passes eight under one
CPU/1 GiB. The single authorised real receipt is
`p003_v2_gate1_cyclic_smoke__20260905T051116Z.json`, sha256
`8a97102cc11f5f8c94f1545deb0df75a82d6bb44a6970fd5fc4faaf723074650`:

- all 1,891 BUY_UP and 1,666 SELL_UP cyclic offsets were enumerated exactly
  once on the same 3,557 canonical actions (5,869 source rows);
- 18 unique BUY_UP phases matched the treated 150-action count and 40 unique
  SELL_UP phases matched the treated 110-action count, giving a complete
  Cartesian support of 720 distinct joint assignments versus 200 required;
- the fixed seed selected 200 uniformly without replacement, and all 200 full
  replays were distinct, matched actual side/hour action counts, preserved
  keys/times/score multiset/circular clustering and passed every stateful and
  separability identity;
- the neutral source statuses remained ADMITTED 1, gap/reconciliation/missing-
  terminal exclusions zero and 458 valued tranches;
- runtime was 99.85 s with 100.47 s user CPU and receipt `ru_maxrss` 338,448
  KiB under one CPU/3 GiB, swap disabled and the ten-minute ceiling.

This clears only the Gate-1 acting-comparator support seam. The receipt labels
its economics `INCOMPLETE_NOT_STRATEGY_NET`: maker fees, complete spread/adverse
components, terminal inventory value and owned-order causality are not yet
reconciled there. No partial economic value from the receipt is interpreted.
Overall gate progress therefore remains 1 of 7 and Gate 2 remains off.

**Gate 1e lifecycle-economic completeness declaration, 2026-09-05T05:14:39Z.**
Before building or inspecting any Gate-1e output, the following downstream
audit is fixed:

- Pin the Gate-1d receipt and sha above. Rebuild the same one-window neutral
  reference and deterministic hash-probe treatment, verify its canonical
  population/action counts, policy parameters, treated realised action IDs,
  720-support metadata and the exact 200 recorded phase/offset assignments.
  No phase is reselected and no Gate-1d economic value is an input.
- Full-replay QR_SKEW_ONLY, treatment and exactly those 200 control phases.
  For every arm compute received-fill identities, five-second maker P&L,
  spread/adverse decomposition on the identical mid-known denominator, rho,
  cancel requested/passed/effective/stale/unresolved counts, holds, reposts,
  queue resets, fill/share retention, per-slug terminal/peak inventory and the
  continuation from each fill's markout to the recorded window-terminal mark.
- Enforce the identities `spread + adverse = five-second maker P&L` on the
  decomposed subset and `five-second fills leg + terminal inventory leg =
  total-to-terminal` on identical fills. Every missing mid, markout, terminal
  mark, share or fee is a counted status, never zero-filled or silently dropped.
- Do not assume a zero maker fee or infer an owned maker fee from a public
  taker/trade fee field. If the replay cannot bind a per-fill maker fee ledger,
  set fee-adjusted strategy net and the aggregate decision-metric null to null
  and refuse Gate 1. Maker rebates/liquidity rewards remain excluded by scope.
- `owned_order_ack_fill_causality` remains a research-simulation assumption,
  not observed venue evidence. Carry it as an explicit limitation; never
  relabel public-market prints as owned fills.
- Gate 1e clears only if all source/action/replay/gross-accounting identities
  are green **and** every required monetary term, including maker fees, is
  priced on the same received-fill population. Otherwise emit a complete
  status ledger but `REFUSED_REQUIRED_ECONOMIC_TERMS_UNAVAILABLE`, no partial
  matched-null comparison, and stop before Gate 2.
- Synthetic gates require a hand-computed spread/adverse/terminal/fee identity,
  a missing-fee refusal, a missing-terminal refusal and a receipt/offset
  identity falsifier. Synthetic work is one CPU/1 GiB. The one fixed real audit
  is one CPU/3 GiB, swap disabled and at most five minutes, using only the
  proven interval-local slice and recorded phases.

Gate 1e is an accounting/status audit, not a licence to fill missing terms with
assumptions. A refusal caused by unavailable public-data terms is a valid
terminal research finding for this repository and stops the v2 route.

**2026-09-05 Gate-1e result — lifecycle ledger complete, Gate 1 refused.**
The accounting core passes 12 synthetic checks and its pinned wrapper 11 under
one CPU/1 GiB. The one authorised real receipt is
`p003_v2_gate1_economics_smoke__20260905T052605Z.json`, sha256
`e78fe495846cf22e834b63e04aea445cf1616563cb932a11f304d3a7ba2abd42`.

- The rebuilt population remained 5,869 source rows / 3,557 canonical actions.
  The Gate-1d receipt hash, 720-support geometry, exact 200 offsets, policy,
  treated action identities and all 200 score-assignment hashes reproduced.
  No named Gate-1d source-code file drifted; only this plan had the expected
  post-receipt documentary update.
- QR_SKEW_ONLY, treatment and all 200 controls were full-replayed. Every
  source, stateful, action, fill-identity, received-markout, spread/adverse,
  five-second-to-terminal, cancel-lifecycle, rate-limit, mid/markout/terminal,
  share and rho population predicate was true across all 202 arms.
- All 202 fee ledgers are explicitly
  `UNAVAILABLE_NO_PER_FILL_MAKER_FEE`. Public neutral market data does not
  identify an owned-order per-fill maker charge, and the audit did not replace
  it with zero or a public taker/trade fee. Accordingly all 202 fee-adjusted
  strategy nets are null, the treatment decision value is null and the matched
  decision null is absent. Gross audit fields are not interpreted.
- Owned-order acknowledgement/fill causality remains unobservable from this
  public-market counterfactual replay. That limitation is carried explicitly
  rather than promoted into venue evidence.
- Runtime was 22.98 s with 23.53 s user CPU and receipt `ru_maxrss` 338,556
  KiB under one CPU/3 GiB, swap disabled and the five-minute ceiling.

The receipt status is
`LIFECYCLE_LEDGER_COMPLETE_GATE1_REFUSED_REQUIRED_MAKER_FEES_UNAVAILABLE`.
This is the prospective stopping condition, not a negative P&L observation:
the required strategy-net estimand cannot be identified from the repository's
public neutral replay. Overall v2 progress is terminal at 1 of 7 gates. Gate 2
and Gates 3–6 do not start. Resuming them would require a new, reliable
owned-order maker-fee/ack/fill data source, a prospective amendment and fresh
unconsumed evaluation data; it is not authorised by this plan.
At the Gate-1e checkpoint, all 16 then-existing v2 module/wrapper batteries and
the 223-check parent diagnostic suite passed under one CPU/1 GiB with swap
disabled.

**2026-09-05T05:43:57Z user continuation — Gate-1f owned-execution input
admission declared before its audit receipt.** This continuation does not
revise Gate-1e, impute its missing fees, reopen its consumed economic output or
authorise Gate 2. It authorises only a bounded repository/source audit and an
offline input contract for data produced outside this research repository.

- The fixed candidate is
  `data/pm_5min/owned_execution/manifest.json`. Absence emits
  `REFUSED_NO_OWNED_EXECUTION_SOURCE`; the audit does not search arbitrary home,
  credential or account paths.
- The manifest must identify protocol, venue, a SHA-256 pseudonym of the owned
  account, the committed pipeline/freeze commit and timestamp, policy and fee-
  schedule hashes, complete UTC days, and byte hashes/counts for separate order
  and fill JSONL exports. The freeze commit must exist and contain the whole
  pipeline; all admitted days must be strictly later than its UTC date.
- Every order row must carry reference-generation identity, client and venue
  order IDs, market/asset/side, decision/submission/ack/terminal timestamps,
  requested price/size, acknowledgement and terminal statuses. Every fill row
  must join one acknowledged owned order and carry unique fill/trade identity,
  fill time/price/shares, venue-asserted `MAKER` role, exact fee amount and
  currency, fee rate, and fee-schedule identity. Zero is accepted only as an
  explicit finite fee bound to that schedule—not inferred from a missing or
  public trade field.
- Require complete export declarations for orders, acknowledgements, fills and
  fees, at least five complete post-freeze UTC days, at least 200 owned orders
  and at least 200 owned maker fills. Reject duplicate/orphan IDs, time
  inversions, public/counterfactual source modes, day/count/hash mismatches and
  missing fee values.
- Existing public surfaces are evidence only about the market: the collector's
  public market websocket, Tier-1 public trades and cached on-chain receipts do
  not bind an order to this research account. More public raw tape cannot meet
  the contract. No credential reader, order placer, cancellation client or live
  execution code may be added here.
- The contract battery must include a complete positive fixture plus known-bad
  public-source, orphan-fill, pre-ack fill, missing-fee, hash and pre-freeze-day
  refusals. Synthetic and real preflight runs are one CPU/512 MiB, swap off,
  with a 30-second ceiling. A missing/invalid source preserves the 1/7 stop and
  creates no economic statistic.

Gate-1f is an acquisition/admission seam, not an eighth strategy gate. Even a
green input would only permit a new prospective amendment and fresh evaluation;
it would not retroactively clear Gate-1e or make seen data untouched.

**2026-09-05T05:50:05Z Gate-1f result — no admissible owned-execution source.**
`de_v2_owned_execution_input.py` passes 11 synthetic checks under one CPU/512
MiB. Its positive control requires 200 exact owned maker fills over five
post-freeze complete UTC days and admits explicitly recorded, schedule-bound
zero fees; known-bads refuse public-source mode, orphan and pre-ack fills,
missing fee amounts, hash drift, freeze-day reuse and taker fills.

The corrected fixed repository receipt is
`p003_v2_gate1f_owned_source_audit__20260905T054941Z.json`, sha256
`c99109943de37d37d2fc8358628640214d489752e96bb8ca4f86e144bf197f47`.
It supersedes `...T054848Z.json` (sha256
`bf3d01fa61ee799860ec8bbc764645b0e034162f1611a54879b146d99b292022`),
whose Tier-1 date range was null because the path-only census omitted the
intermediate `distiller=...` directory. The correction changes no gate field
or conclusion.

- The fixed candidate
  `data/pm_5min/owned_execution/manifest.json` does not exist.
- The public collector uses the market websocket and records public
  `last_trade_price`/`fee_rate_bps`; Tier-1 carries public transaction hashes
  but no client-order identity. The prior uncertainty audit explicitly records
  venue-ack latency as unobserved without placing orders.
- Public raw directories span 2026-08-19 through the current 2026-09-05
  directory (18 dates); Tier-1 public-trade manifests span 2026-08-20 through
  2026-09-02 (13 dates). These are coverage facts, not complete/untouched-day
  admissions, and additional public tape cannot create owned-order causality.
- Receipt status is `REFUSED_NO_OWNED_EXECUTION_SOURCE`; decision metric is
  null, Gate 1 remains refused and Gate 2 remains unauthorised. Runtime was
  0.006 s with 19,760 KiB process RSS under one CPU/512 MiB, swap disabled.

The next admissible input must be produced outside this repository and contain
the manifest, policy/freeze and fee-schedule hashes plus complete owned order,
ack, terminal, maker-fill and exact fee records defined by the validator. This
repo must remain research-only; do not add venue credentials, signing, order
submission or cancellation code here.

Latest bounded regression closure at 2026-09-05T09:54:58Z: all 17 current v2
module/wrapper batteries pass (182 checks total), including Gate 1f's corrected
date-census and known-bad refusals, and the parent diagnostic suite passes
223/223. Both suites ran sequentially under one CPU/1 GiB with swap disabled.
This verification changes no gate or economic conclusion.

### Gate 2 — smallest decisive core ablation

Do not start with seven arms. Run the smallest set that answers whether the
prediction stack adds value over no prediction and matched action volume:

1. QR_SKEW_ONLY incumbent;
2. acting RANDOM_MATCHED at the treated action count by side/hour;
3. fill-hazard cancellation with neutral placement;
4. conditional-value cancel × frozen skew.

Use the pinned model family and frozen skew semantics. Use only consumed
development data. Report per UTC day, never only pooled, and label G<5 as a
point estimate without an interval.

**Gate-2 exit:** conditional value improves the decision metric over acting
matched random and the incumbent without worsening declared inventory and
traffic limits. Predictive AUC or harm-share lift alone cannot clear this gate.

If the core ranker fails, stop the route. Do not add fair price, tune the
threshold or broaden the grid on the same data.

### Gate 3 — lifecycle, latency and cost robustness

Only a Gate-2 survivor enters the declared latency and queue-reset-cost grid.
The primary verdict is full stateful net economics at material retention. The
grid is sensitivity analysis, not evidence of real cancel effectiveness;
public market data has no owned-order ACK/fill causality.

**Gate-3 exit:** a predeclared candidate remains better than matched random and
the incumbent across the required cells, reaches rho < 1 where defined and
has acceptable inventory/traffic. Any incomplete economic term prevents a net
claim.

### Gate 4 — optional fair-price increment

Fair price is deferred until the core ranker clears Gate 3. Identity remains
mandatory. PM microprice and at most one predeclared cross-venue residual may
then be tested incrementally, using the already typed point-in-time interface.
No fair-price challenger is necessary for a valid core result.

**Gate-4 exit:** the challenger improves the full decision metric over
Identity on a population not used to select its transform. Otherwise retain
Identity and close the lane.

### Gate 5 — commit/freeze

A freeze is a commit containing the entire data → target → fit → action →
stateful replay pipeline. The receipt records:

- builder and dependency hashes plus commit ref;
- feature/action schemas and all thresholds/horizons;
- candidate count and multiplicity;
- null design and number of draws;
- source/stamp boundaries, exact seen-day list and exclusions;
- the forward start, strictly after the commit.

The present v2 plan and Gate-0 modules are **not frozen** while uncommitted.

### Gate 6 — untouched forward days

Score the committed candidate unchanged on later, complete, healthy UTC days.
Seen days remain consumed forever for this line.

- G >= 5: direction and consistency only unless the exact declared test can
  attain its threshold;
- with multiplicity 2 under the clustered sign/permutation floor discussed in
  the current record, require at least G >= 6 for a possible Holm-clearing
  result;
- intervals and resampling use UTC day, never windows or fills, as the cluster
  unit.

Any candidate change starts a new forward clock. The prior G5 race cannot
validate the v2 pipeline.

## 5. Module map

| Module | State under v2 | Next proof |
|---|---|---|
| Neutral replay and PRED_STATE_V1 | built/reusable | provenance and reconciliation in new receipt |
| Conditional hazard/value heads | built development artifacts; not promoted | core acting ablation |
| Frozen skew semantics | built/reusable | parity inside new acting runner |
| Typed fair-price Identity and challengers | machinery built, challengers unscored | deferred to Gate 4 |
| EV/action/constraint interfaces | built/reusable | connect to canonical action ledger |
| Canonical action adapter | built; synthetic tests and capped real smoke pass | retain as Gate-1 input |
| Canonical exact-tranche action ledger | built; synthetic tests and capped real smoke pass | extend only through explicit economic statuses |
| Static matched action-bundle null | built; 200-draw real smoke reconciles | falsifier only; never call cascade-feasible |
| Interval-local one-window selector | built; real continuity receipt passes at ~339 MiB RSS | smoke-only, not a broad-run selector |
| Gate-0 end-to-end runner | **Gate 0 cleared** by capped real pipeline smoke | preserve receipt identity; no economic interpretation |
| Acting iid-permutation control in stateful replay | built; 18 synthetic checks pass; capped real smoke refused at 1/4,000 exact matches | retained as falsifier; do not raise attempt budget |
| Gate-1 smoke wrapper | built; seven synthetic checks pass; real refusal receipt preserved | reuse only after the constrained control clears synthetic gates |
| Constrained exact-fiber switch null | built; 13 synthetic checks pass; real retry found support but refused at ESS 10.53 < 100 | no adaptive rerun |
| Gate-1 constrained smoke wrapper | built; seven checks pass in both invocation modes; capped receipt preserved | stopped by mixing gate |
| Sequential random action-quota control | built; 14 synthetic checks pass; real smoke refused with 16/1,000 accepted and 16 distinct | no adaptive rerun; audit receipt preserved |
| Gate-1 action-quota smoke wrapper | built; eight synthetic checks pass; fixed real refusal receipt preserved | stopped by support gate |
| Finite cyclic-phase acting control | built; 13 synthetic checks pass; complete real enumeration yields 720 exact-count joint phases and 200 distinct full replays | preserved Gate-1d input; no economic interpretation |
| Gate-1d cyclic-phase smoke wrapper | built; eight synthetic checks and fixed real support receipt pass | immutable input reproduced by Gate 1e |
| Lifecycle economic completeness audit | built; 12 synthetic checks; all 202 real-arm gross identities pass but all maker-fee ledgers are unavailable | terminal Gate-1 refusal; never zero-fill fees |
| Gate-1e pinned wrapper | built; 11 synthetic checks; receipt reproduces all 200 phases/actions and refuses with null strategy net/null decision metric | terminal receipt preserved |
| Gate-1f owned-execution input contract | built; 11 synthetic checks; corrected repository audit refuses because the fixed external export is absent | await an offline authenticated export from outside this repo; no venue adapter here |
| Full lifecycle economic ledger | complete for public-replay gross terms; owned maker fee and ack/fill causality unavailable | blocked on a new reliable owned-order data source outside current public inputs |
| Core four-arm runner | deliberately not built as one reconciled receipt | blocked; Gate 2 did not start |
| Freeze/forward scorer | old race exists; invalid for v2 | blocked; no v2 candidate cleared Gate 1 |

## 6. Immediate implementation order

1. Keep this plan and the Gate-0 modules reviewable; do not call them
   frozen until committed.
2. **Built and Gate-0 smoke checked:** preserve missing-markout tranche identities in
   the opt-in v2 neutral reference without altering old artifacts.
3. **Built and Gate-0 smoke checked:** canonical one-action-per-generation adapter; its
   decision time is the earliest source row before status filtering.
4. **Completed but refused on real data:** the iid-permutation acting control
   passes synthetic falsifiers but produced only 1/200 required matched draws
   in its fixed 4,000-proposal real smoke budget. Do not widen that budget.
5. **Completed but refused:** the constrained exact-fiber switch null found a
   large feasible component but failed its predeclared mixing bar (ESS 10.53 <
   100). Do not extend or tune it on this consumed window.
6. **Completed and refused:** the sequential random action-quota controller
   passes 14 synthetic checks and its wrapper eight, but only 16/1,000 real
   proposals reached the quota and only 16 distinct states existed. Its
   matched-null metric is absent. Do not widen or modify it on this window.
7. **Gated stop:** do not integrate a 16-draw audit trail into the lifecycle
   ledger and do not build the four-arm Gate-2 receipt. Further work needs an
   explicit new user ruling and prospective design.
8. **User-resumed / declared before execution:** build Gate 1d's finite
   cyclic-phase control exactly as declared; pass its positive, finite-support,
   clustering and known-bad synthetic checks under one CPU/1 GiB.
9. Enumerate the one-window phase support and, only if it contains at least 200
   distinct joint assignments, replay the fixed 200 uniform without-replacement
   sample under one CPU/3 GiB and the ten-minute ceiling.
10. A Gate-1d refusal stops the loop. A pass advances only to completing the
    lifecycle ledger; Gate 2 remains unstarted until the whole Gate-1 exit is
    green.
11. **Gate 1d passed:** preserve its 720-phase finite support and fixed 200
    draws. Build/run Gate 1e exactly as declared; do not use its partial metric.
12. If any required monetary term is unavailable, emit the counted status and
    stop with no decision-metric null. Start Gate 2 only if Gate 1e genuinely
    clears the full Gate-1 exit.
13. **Completed / terminal stop:** Gate 1e reproduced all 200 fixed controls
    and all gross lifecycle identities, then refused because every arm lacks
    an owned-order per-fill maker-fee ledger. Preserve the null strategy-net
    and decision-metric fields. Do not build Gate 2 or reinterpret gross audit
    fields as the missing estimand.
14. **User-continuation acquisition audit completed/refused:** Gate 1f defines
    the external owned-execution export contract and confirms no candidate is
    present. Preserve the stop. Additional public tape, a public trade fee or
    another maker's on-chain tier cannot satisfy the owned-order join.

## 7. Stopping rule

Stop the harmful-fill route if either condition holds on the proper gate:

- the frozen core ranker cannot beat acting matched random and the incumbent
  on the full decision metric; or
- after full lifecycle costs it cannot approach break-even at material quote
  retention without violating inventory or traffic limits.

A negative result closes this feature/data design. It does not authorise
tuning on the same days, replacing the null, adding fair price to rescue the
headline, or reporting the static oracle as achievable value.
