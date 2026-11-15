# STOP EV model and timing diagnostic commit review

The completed model shows exploratory ranking signal, but its expected-value
interpretation and timing validity still need repairs before its output can
justify the keep/pull pipeline. The new timing diagnostic is useful as a
sensitivity analysis; its groups do not isolate E2-independent outcomes.

Reviewed commits: `ce6edfa7`, `490f7407`, `34734643`, `f3fb7e3b`, and the newly
arrived `0b7e30cd`. Current timing certificate integration `79e8f13d` remains
separate from the model's original dataset. Review only: no model, label,
policy or active experiment sources were changed.

## High priority findings

1. **The Huber head does not estimate conditional mean value.**
   `pf_stop_ev_model.py:153–167` fits Huber on filled values and multiplies its
   output by fill probability, calling the result EV. Huber estimates a robust
   location; asymmetric large losses can make its sign differ from the mean.
   The included mathematical control has 900 outcomes of +1 and 100 of −100:
   the mean is −9.1, but the Huber minimizer with delta 1 is +0.8889. A large-delta
   quadratic control recovers the mean. This proves the target mismatch, not
   how much of the historical miscalibration it caused. Use a mean-preserving
   target or explicitly calibrated mean/tail model before a zero-EV threshold.

   Saved predictions, five consumed test days / 137,703 actions, source as of
   2026-11-15T14:00:57Z, independently checked at 14:35:33Z: the top decile
   (13,771 actions) forecasts +0.4345 cents and realizes −0.5266 cents per order.
   The model keeps 9,727 actions with total local value −$55.41 and pulls 92.94%.
   Its local improvement over the order-only B2 model is $17.94 across the five
   days; the reported decision superiority test remains NOT-SHOWN. These are
   local neutral-path values, not portfolio P&L or quote uptime.

2. **The model has not adopted the repaired timing population.**
   `pf_stop_ev.py:560–583` removes untimed records from the source ledger and
   rejects only their future reach to the focal price; the model loads those
   labels at `pf_stop_ev_model.py:117`. Earlier queue depletion, partial fills,
   shared-volume effects and changed order births are still possible. The
   separate certificate handles these dependencies, but its merge did not
   automatically rebuild this model.

   Exact action/time joins on the same 137,703 test predictions find 81,630
   certified estimated-timing labels and 56,073 without a certificate. Values
   match exactly on the certified overlap. Of the report's 4,572 nominally
   strict-WS actions, 993 also lack the missing-timing certificate. None of
   these counts is a measured mislabel rate. Use the certified dataset for a
   newly declared fit and keep the broader user-approved dataset as sensitivity.

3. **The permutation gate was relaxed after observing its failures.**
   `pf_stop_ev_model.py:187–190,254–256` replaces the declared two-sided rule
   with a signed one-sided rule. Saved fit receipts still have `T5_pass=false`
   for Oct 31 and Nov 3; the report recomputes them under the new rule and
   publishes MODEL-READY. This is disclosed in prose, but it does not constitute
   a pass of the frozen check. Both versions use only 10 draws per fold,
   below CLAUDE's required 200. The modified function accepts a deliberately
   supplied IC of −1 on every draw while rejecting +1; its one-sided result
   cannot establish absence of predictive dependence. This is not evidence
   that actual leakage exists. Preserve the failed receipts, distinguish
   exploratory results, and preregister a suitable falsifier with sufficient
   draws before another claim. The 500-draw matched decision null is a separate
   test and does not repair this training check.

4. **The new E2SEP WS group can still depend on earlier E2 timing.**
   `pf_stop_ev_e2sep.py:114–128,193–197,246–254` classifies only trades reaching
   the focal order in its future avoidable window, then calls positive IC there
   SKILL-ON-WS. A synthetic replay reproduces the failure: at decision 150 ms,
   an E2 trade at 100 ms has drained queue ahead and a WS trade at 200 ms fills
   five shares. The group is WS-REACHED. Moving only that E2 trade −250 ms puts
   it before activation and changes the later local value from +5 cents to zero.
   The prefix certificate correctly rejects invariance. Apply the prefix check
   or explicitly name this group future-WS-reached, with prior E2 dependence
   unresolved; it cannot settle whether the model's skill is a timing artifact.

## Additional diagnostic and reproducibility issues

- `pf_stop_ev_e2sep.py:208–223` rebuilds neutral generations and eligibility,
  then recomputes E2-REACHED membership separately for every shift. The range
  of those group means mixes timing effects with population changes. An
  unchanged mean cannot establish action-level robustness. Report fixed
  identity/state comparisons where they exist, changed births/unmatched actions
  separately, and describe R950 as three common-offset scenario means. A shift
  applied uniformly to every E2 event is not a bound on independent timing errors.
- `pf_stop_ev_e2sep.py:180,204–206` records only `E.source_ids()`. That manifest
  omits `pf_stop_ev_e2sep.py` itself and is sampled after the day, so its mixed-source
  check cannot detect changing the shift/group builder between days. Pin the full
  dependency closure and check before/after each day; preserve existing outputs.
- `pf_stop_ev_model.py:222–229` writes predictions and fit metadata but never
  saves the full/B2 fitted LightGBM heads or hashes prepared inputs. The existing
  predictions can be audited; the fitted models cannot be loaded for stage 3
  or inference-parity checks. Persist them with feature order, parameters,
  data/source hashes, and an artifact reload comparison in the next version.
- E2SEP's zero-shift and prediction joins omit `decision_ms` and do not enforce
  one-to-one joins (`:168–174,245–247`). Its stated exact reproduction contract
  should refuse shifted decision times, duplicate keys and missing actions.
  No actual bad join in its historical outputs is asserted by this review.

## Evidence and next step

`reviews/stop_ev_model_commit_review_20261115/receipt_v2.json` verifies all five
prediction files against their prepared action labels and independently
reproduces the published IC, local gain and kept-value numbers. Hurdle products
and fill-probability bounds pass; duplicate, probability and label corruptions
are refused. Inputs were hashed before/after the review. The first review
attempt refused a string-versus-integer day-key join; its log is retained, and
v2 explicitly normalizes that key without changing action identities.

`e2sep_v1.json` records the synthetic prefix counterexample, zero-shift/E2-group
positive controls, and the missing source-manifest entry. Both checkers were
committed before execution and ran on the Codex slot, under 200 MiB measured
RSS. No raw tape, sealed dates, new fits or new historical nulls were read/run.

Priorities: integrate certified label provenance; repair the mean-value target;
freeze valid training checks and preserve fitted artifacts; qualify/repair E2SEP
group interpretation. Then evaluate the model and its policy separately. The
running E2SEP experiment remains owned by the other session and was not altered.
