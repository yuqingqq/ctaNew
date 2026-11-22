# Horizon and condition diagnostic — pre-output protocol

USER, 2026-11-22: inspect the conditions more carefully and assess whether
one second is too short. This is a saved-evidence diagnostic, not a new
candidate, tuning exercise or continuation of the stopped positive-pressure
replay. Source receipt target: **2063d31a91ab7e6124d7b8b7b7ec5df01ea66711**.
Current repository state: **26d50e3ee2451cef7944d8f48b9c64c80176b539**.

Freeze this protocol and script before computing new horizon/condition tables.
Do not change the existing frozen candidate's primary endpoint after seeing
these results. All evidence is from the same consumed Oct 24–27 markets.

1. Hash-bind the accepted pressure/ablation ledgers, saved fill mids, original
   audit state rows and frozen pressure edges to their committed receipts.
   BASELINE and FULL remain separate. Reconcile FULL at +1 s to the accepted
   value; no tape load, resimulation, label expansion or model fitting.
2. Report 0/50/100/1,000/5,000/30,000 ms markout and END. Markout is side-signed
   quantity times future mid minus execution price. END is a variable-horizon
   pre-expiry mark, not settlement or realized liquidation. Report each
   horizon's native availability and a common set of fills valid at every
   listed horizon. Show count, shares, positive/negative/net value, per share,
   mark age and day/side breakouts. Missing horizons remain missing.
3. On that common set, tabulate transitions from a negative +1 s markout to
   nonnegative +5/+30 s, with fill/share denominators and weighted value.
   These outcome-conditioned transitions describe recovery/persistence;
   they are not a training or decision population and do not define a
   profitable policy. Do not choose the most flattering horizon.
4. Deeper condition population: all saved neutral BASELINE `kind=live`
   snapshots, one per generation, including no-future-fill rows. Reconcile
   their +1 s window values to the accepted audit. Value only that generation's
   actual fills at/after the snapshot's cancel effect; preserve its structural
   and inventory continuation. A window with any missing required mid is
   censored for that horizon, not set to zero. Compare +1/+5/+30 s on the same
   complete-window cohort and show all censored/no-fill counts separately.
   This is an exploratory description of neutral-path states, not an estimate
   of how a newly guarded policy would perform.
5. Predeclared conditions only: joint current Binance pressure (non-adverse
   <=0 / positive below the old q5 boundary / q5) × side-adverse Predict
   order-flow imbalance over 250 ms (>0 or <=0); separate marginal tables
   for own best-depth depletion (<0), order distance from best, generation
   age (<0.5 / 0.5–2 / 2–10 / >=10 s), and inventory sign in the order's
   direction. Missing inputs are their own status. Use only snapshot inputs;
   no fill-time signal, future T1, venue-only queue or outcome-selected masks.
   Report per day/side, opportunity/no-fill counts, shares, positive/negative
   and net values. Five views are exploratory multiplicity, not five validated
   signals. No ranking, selected rule, model-skill claim or confidence interval.
6. Checks: known hand markouts for bid/ask, null horizon propagation, common
   cohort membership, unique fill/state keys, expected market population,
   identity refusal, exact +1 s reproduction and END sums. Cap this entire
   saved-data diagnostic at 0.01 core-hours and keep original ladder cumulative
   use <=2.0. Run only PF_SLOT=codex through run_light with absolute paths.

The development session retains the stopped broader-guard experiment and its
resource gate. This task does not authorize resuming it or spending above its
0.14-hour per-round ceiling. Results and a concrete interpretation/analysis
handoff will follow in a second review-only commit, preserving other work.
