# Other-session status and TARGET-TEST review

Checked 2026-11-14 05:41 UTC. TARGET-TEST finished at 05:39:03 UTC and its
results were committed as `adbcdb89` at 05:40:21 UTC. No failure was found in
its driver or stage logs. All nine scoring and nine evaluation stages exist.
The next recorded work is JOINT-LADDER (in progress), followed by ORDER-HARM;
there is no completed joint-ladder/order-harm result in this snapshot.

The report is as of 05:37:13 UTC, on nine consumed dates (2026-10-23..27,
10-31, 11-01..03), first-minute windows only. There are 5,184 window-sides
and 2,290,842 guard steps. All stage receipts report zero tau-zero parity
differences, no reference-fill differences, unchanged clocks and caught
mutants. Recomputed onset scores match exactly, with no unmatched onsets.
The report records zero fills missing their one-second value. Report/runtime
copies match their SHA256SUMS. Verification is of sources and receipts;
historical fills/model predictions were not independently rebuilt.

## Performance finding

These are simulated one-second fill marks per replay date, not realized
portfolio P&L. The primary subset is WS-timed trades, order age >=100 ms.

| Comparison | M0 | M50 | M250 |
|---|---:|---:|---:|
| Rebuilt FBF25 minus old FB25 | -$33.8 | -$15.1 | +$242.8 |
| Rebuilt FBF25 minus blanket guard LDG | +$74.8 | +$55.7 | -$490.4 |

The rebuilt model has not established improvement over the old one. Its
gain over the blanket guard reverses under M250 on all nine dates; including
estimated-time trades also reverses the M50 comparison (-$48.8/day). The
simple influence-distance rule loses to the blanket guard at every clock.
The saved R601 predicate is NOT-SHOWN, rather than WORSE, because M50 has
only six negative dates versus the declared seven-date threshold.

## Issues before the next experiment

1. Prediction coverage differs materially: OLD lacks 40 guard-step scores;
   NEW lacks 275,520; FULL lacks 277,568 (12.12% of guard steps). Missing
   scores force pulling. These are step counts, not numbers of cancelled
   orders. Most are masked-start states; 2,048 additional FULL states lack
   fair price. Compare policies with the same valid-information mask and
   fallback before attributing differences to model quality or horizon.
2. The three previously identified gaps remain in this completed run:
   original-best orders always pull, onset-trained predictions are applied
   at subsequent steps, and levels are simulated independently. The other
   session explicitly acknowledges them and plans joint inventory and
   per-order KEEP-versus-PULL labels. That is the appropriate next direction.
3. A joint engine must give each actual taker leg one shared quantity budget
   across our orders and include pending cancels in exposure. Uncapped parity
   with independent rungs is not sufficient if the reference duplicates fills.
   Check non-overlapping parity separately from overlap corrections.
4. Scaling inventory capacity with order size is a separate risk-budget
   experiment. Model comparisons must hold the shared capacity and sizing
   fixed; report any larger-size/larger-cap gain with its additional exposure.
5. The claim "horizon is not the problem" is too broad. The experiment did
   not establish superiority of its two-second oracle over its one-second
   oracle in this harness. It does not rule out different learnability,
   residual-horizon targets or the previously noted clean-target/oracle
   mismatch. Leave-one-day-out on consumed dates is still exploratory.

The research run is operationally healthy. The unresolved issue is whether
selective retention improves value under credible timing, shared execution
and inventory accounting. The current results do not establish that.

Receipt: `reviews/model_target_test_review_20261114/status.json`.
No other-session source, frozen result, worker or trading setting was changed.
