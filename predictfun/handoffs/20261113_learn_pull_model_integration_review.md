# Maker next-trade study: observed refills do not identify cancellations

Review as of **2026-11-13T17:39:41.719097+00:00**. Producer LEARN-PULL report as of **2026-11-13T16:48:15Z**, source state `81ae9c60`; review instrument frozen at `a8200982`. Nine consumed dates: Oct 23–27/31 and Nov 1–3. No frozen model, policy, threshold or replay changed.

The published descriptive calculations reconstruct exactly from the saved tables. Their interpretation as cancellation labels does not follow from the available observations. This matters directly for the five/50-share question: a small order that is fully filled can be classified as having been pulled, whereas a larger residual order may later fill again.

## What was checked

The review binds eight producer sources, three review sources and 39 published inputs. It checks all daily coverage metadata and all three receipt copies. It executes the exact producer observation, frame-presence and feature-time functions on synthetic counterexamples, including the producer’s positive selftest and all three planted-error controls. Changed artifact hashes and bad coverage metadata are refused.

The report reconstruction uses the published R554 classifier values in place of its fitting loop. It **does not retrain or independently verify the reported classifier AUC or feature importance**. Every other report calculation is reconstructed from the original tables; the resulting report text and reading JSON match exactly. The auxiliary job exited zero in 6.31 seconds, using 715 MiB peak RSS; no raw-tape replay was performed.

Saved populations, at the producer as-of time:

| Population | Rows | Interpretation |
|---|---:|---|
| Next-trade observations | 193,153 | Maker/price intervals selected after a stopping fill and resolved by a later trade |
| Top-five group | 78,294 | 21,765 have no later fill by that signer: 27.80%; the study’s guard signal fires on 59.84% |
| Field group | 114,859 | 46,022 have no later fill by that signer: 40.07%; guard signal fires on 57.89% |
| Big-move fill records | 345,527 | Event-associated fill records, not a deduplicated portfolio |
| Big-move events | 30,588 | Selected realized moves, not all decision opportunities |

The producer records 49,274 estimated-time ambiguous cases and 58,135 same-frame-stamp cases; missing assigned inventory and unpaired windows are zero. These counts do not form a complete action ledger: expired proof intervals, unresolved proofs and unavailable feature anchors are not all emitted as statuses. The observation rows are not neutral order-generation actions.

## Exact counterexamples and their implications

1. **A sole-maker stopping fill does not prove residual quantity.** In one possible FIFO book, maker 7 has five shares ahead of maker 9’s 45 shares. A five-share trade fully exhausts maker 7; the price still shows 45 shares. In another possible book, maker 7 starts with 50 shares, fills five, then cancels the remaining 45; maker 9 replaces that liquidity. Both can show the same aggregate depth and the same next trade filling maker 9. The exact extractor reports `stayed=0` in both cases. Full fill and cancellation are not identified separately.

2. **A later fill by the same signer does not prove the same order survived.** The exact extractor reports `stayed=1` for maker 7 filling at both times. That observation is compatible with retaining the residual and with cancelling and placing another order. Order hashes, original sizes and remaining sizes are absent from these rows. The historical frequency of either misclassification is unknown.

3. **The frame predicate is weaker than a front-at-best proof.** It tests whether the price appears with positive quantity anywhere in the displayed depth. The synthetic case passes even when another price is the best. Aggregate depth does not identify which maker owns the residual quantity.

4. **Features can follow the hidden action.** The exact feature selector uses the last grid step at or before the next trade minus 100 ms. With proof at 100 ms, cancellation at 200 ms and next trade at 1,000 ms, the selected feature time is 900 ms. Being earlier than the trade does not make it earlier than the cancellation being classified. The `gap` feature also depends on the future next-trade time and is unavailable at a fixed earlier decision time.

5. **The classifier is not a chronological forecast test.** Its training selector is every date except the test date, so earlier test dates use later training dates. The P feature also inherits the previously documented nested-fold problem. Reported AUC is descriptive classification evidence only; it is not evidence for a deployable pull rule.

## Implication for the model goal

The public observations support studying repeated maker participation and local price state. They do not show the maker’s cancellation rate, continuous quoting, retained order priority, or the value our own order would receive by staying. The study’s guard comparison is a signal on these maker-selected intervals, not our actual sequential order availability. Its next-trade marks weight volume at that price, not our remaining quantity or attainable fills.

The native size experiment remains the appropriate controlled test of the user’s mechanism: original quantity, residual quantity, order identity, partial fills and inventory exposure are explicit. It must distinguish KEEP, RESET and PULL without treating an exhausted five-share order as a discretionary withdrawal. Nothing in this review selects a new threshold or alters the ongoing candidate.

For any later imitation model, maker action labels would need order-level lifecycle evidence and features observed before that action. Without that evidence, next-trade membership should retain its descriptive name. Adding model capacity cannot repair an unidentified target.

## Evidence

- [Review receipt](../reviews/model_recovery_20261113/learn_pull_integration_review_v1.json), SHA `4adef0bec83de8e5358c52f67c764f0737856cf8e5e4eef6346b364393445842`.
- [Prior declaration](../reviews/model_recovery_20261113/learn_pull_integration_review_v1_declaration.json), SHA `0c0ebf6906ce8c599145fa6e8ff851948e1ebb4b674d622b9fef45cf45b8da72`.
- [Review instrument](../research/model_learn_pull_integration_review.py).

The original producer outputs are preserved. No live calibration, forward validation, improved-policy or top-maker-parity claim follows from this review.
