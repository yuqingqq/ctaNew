# Latest severe-risk and queue results: artifact review

Reviewed commits b08304d2/ab3d1e42 (calibration), 05cebee6/83f339df
(information), bcd77450/4de41980/afccb5be/12441034/5b6cda26 (queue).
Review source: 7054cd34, `research/pf_latest_results_review.py`.
Evidence: `reviews/latest_results_review_20261117/receipt_v1.json` and
`interpretation_v2.json`; the latter supersedes the interpretations identified
below, preserving all original outputs. As of 2026-11-17 07:29:36 UTC.

The reported acceptance predicates reproduce. Neither probability candidate
improves economic selection sufficiently; the additional information candidates
fail their declared comparison; the tested passive placements lose on the
saved eligible states. These results do not establish an information limit,
or identify severe-event ranking as the sole economic bottleneck.

## Findings requiring correction

1. **Material interpretation overreach: an information limit was not tested.**
   HANDOFF's 06:19 entry concludes that available causal data contain no further
   useful information. The actual test compares one fixed in-band model with
   five extra features. On Oct25-27, 22,440 decisions / 518 severe events,
   AUC is P 0.731189, M87 0.696538, M87+ 0.699781. M87 also has less training
   history and a different training population than P. Its failure cannot
   establish that the original inputs are fully exploited. AUC of an event
   indicator is not dollar-weighted loss discrimination. Existing P already
   carries ranking signal within the band. The test uses the **old independent
   severe head**, not the newly chained/calibrated severe probabilities
   (`pf_severe_info.py:95`). It therefore does not directly diagnose the new
   candidates' residual errors.

2. **Material estimand limit: queue priority is confounded with exposure.**
   FIRST versus LATE changes activation time and exposure duration. EARLY
   includes fills before the new-best birth and is selected using that future
   birth; it is not a deployable entry rule. The comparisons test those complete
   posting setups, not queue position alone. All values use a common endpoint,
   not a fixed horizon after each fill. Comparing their per-share value directly
   with makers' fill-plus-five-second markouts would be incorrect. Structural
   drift/cross cancels remain active, though learned pull rules are absent.
   IMPROVE also assumes the observed tape survives a changed displayed book.

3. **Concrete population-accounting defect in the queue test.**
   `pf_queue_place_test.py:240` discards the status when path construction fails;
   line247 skips the entire causal state on a failed five-second path. Across
   Oct23-27 the receipts count 151,228 causal states, but only 149,174 unique
   state identities have any saved rows: a gap of **2,054**. The saved artifact
   cannot reconstruct the reasons for each missing state. Sixty saved causal
   arm rows also lack two-second values without a per-horizon exclusion status.
   The report prints only the gone-before-FIRST exclusion, omitting other event
   exclusions. Restore an explicit candidate/state ledger with statuses for
   each arm and horizon before claiming a complete population comparison.
   This does not change the reproduced negative means for saved eligible rows;
   the missing population's effect has not been established. No tape rerun or
   candidate-source fix was performed in this review.

4. **Smaller reporting corrections.** The 0.731 risk AUC is for Oct25-27,
   n22,440 / 518 severe. Across all Oct24-27 low-risk rows it is **0.713170**,
   n30,080 / 675 severe. CAL improves value RMSE in **1/4** primary cells,
   not zero: Oct26 posts 32.593206 -> 32.586208 cents, n3,994. It remains a
   failed economic candidate. Low-probability O/E comparisons also change
   membership: BASE n14,201 /305 events, SMOOTH n13,846 /292, CAL n13,790 /284.
   These are each model's own predicted-risk bands, not the same fixed actions.

## What the results support

Calibration primary evaluation is Oct26-27, n17,859 neutral decisions, with
Oct23-24 training and Oct25 calibration. The chain guarantees
P(severe) <= P(any loss); saved heads and reconstructed maps reproduce the
reported outcomes. Amount and gain heads are held fixed.

| Comparison against BASE | SMOOTH | CAL |
|---|---:|---:|
| Severe log loss improves, day x kind cells | 3/4 | 3/4 |
| Severe count O/E closer to one | 4/4 | 4/4 |
| Own low-risk-band O/E, BASE 1.442 | 1.215 | 1.143 |
| Net-value RMSE improves | 1/4 | 1/4 |
| Net-value quintile calibration improves | 1/4 | 0/4 |
| Declared matched-participation test | Fail | Fail |

Probability estimation improves, but this is insufficient for better economic
selection. The experiments do not isolate event ranking from severity-amount
errors, gain errors, or their combination in predicted net value. Calibration
of two separate factors is also not necessarily rank preserving for their
product. The statement that severe ranking is *the* binding error is withdrawn
as an established conclusion.

Queue values below are cents per candidate order at the five-second endpoint,
including no-fill outcomes, on saved eligible Oct23-27 rows (30 shares, M50,
58 ms placement). They are neither settlement P&L nor per-fill five-second
markouts.

| Setup | n | Mean value | Filled orders |
|---|---:|---:|---:|
| FIRST, improved best | 58,236 | -8.79 | 38.5% |
| FIRST, depleted best | 53,240 | -16.52 | 44.2% |
| EARLY, depleted best, ex post | 22,300 | -18.31 | 61.7% |
| JOIN at best | 149,174 | -13.05 | 41.7% |
| REST1, one tick behind | 146,483 | -13.89 | 33.5% |
| IMPROVE, spread at least two ticks | 7,976 | -7.23 | 36.1% |

Matched IMPROVE minus JOIN is +5.613 cents on those same 7,976 states, positive
on four of five days; IMPROVE itself remains negative. EARLY minus FIRST is
-3.531 on 22,300 matched births. Different participation/exposure makes these
results unsuitable as evidence that queue priority itself is worthless.

## Next priority

First repair the missing-state/status ledger. Then use identical neutral
resting-order states to separate useful queue retention from avoidable adverse
fills: the same side, price, remaining size, horizon and actual cancellation
latency, with keep and pull branches. Report foregone gains and avoided ordinary
and severe losses, alongside value at matched participation, by day. An ex-post
oracle can bound loss avoidable at that latency, clearly marked non-deployable;
a declared causal score must then earn improvement over matched controls.
This distinguishes a timing/latency limit from failure to rank economic costs.
It does not justify another controller change or model promotion yet.

## Verification scope

Saved calibration heads were reloaded and metrics rebuilt. The original six
information-test fits were reproduced with the unchanged recipe in a separate
directory because original per-row scores/models were not saved; AUCs and both
predicates match exactly. All five queue parquet hashes, 548,871 eligible rows,
quantity bounds, young-fill decomposition and all published boolean readings
reconcile. Original receipts report 2,760 path-parity checks, zero failures,
and 2,746/2,746 planted no-shift detections; these tape checks were not rerun.
Valid and deliberately changed-result controls and original selftests pass.
Resource use: 44.62 s, 890 MiB peak RSS, Codex core8. No candidate or controller
changed. Dates are consumed development data; no confidence interval or
prospective validation claim. Later untouched days remain necessary.
