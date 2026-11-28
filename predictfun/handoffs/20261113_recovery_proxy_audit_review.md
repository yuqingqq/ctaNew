# Fixed recovery proxy: later-day forecast audit

The fixed recovery proxy has little demonstrated incremental forecast value.
The saved audit is as of **2026-11-13T07:23:08.921632+00:00** and reconstructs
all market features for **11,691** later neutral actions: 1,714 Oct25 diagnostic
calibration actions and **9,977** actions across six consumed test dates
(Oct26/27/31, Nov1/2/3). All short targets are eligible; no missing short labels.
The model was trained once on the earlier 3,741 Oct23/24 seed actions.

On the six test dates, pooled RMSE is **5.19342 c/action**, against **5.19597**
for the seed-only constant mean. The equal-day mean MSE improvement is
**0.03485 c²**, with the declared 1,000-draw day-bootstrap interval
**[-0.09024, +0.17810] c²**. The saved predicate for a positive interval is false.
These are different weightings: pooled RMSE weights actions, the interval
weights UTC days equally.

The **3,399** test actions with nonnegative estimates have predicted short value
summing to **+1,480.99 cents**, versus observed **-527.27 cents**. This is a
calibration concern around the recovery threshold. These overlapping local
values are not additive portfolio P&L. The proxy estimates retained original-
order short value on neutral resting states; this does not identify fresh-entry
value or calibration on actual recovery-wait states.

This fixed market-only proxy differs from the expanding, order-state-aware
short head in the [local action-model review](20261113_recovery_local_review.md).
No retuning, policy change or post-hoc replacement is made. Full replay remains
necessary to evaluate the frozen selective policy and re-entry rule together.

Evidence: [audit receipt](../reviews/model_recovery_20261113/proxy_audit_result.json),
SHA `d8df50c2be792712037e9ece531606901b7e1b7f7546b75b0ee2cd2bba2369fd`;
[completed runner](../reviews/model_recovery_20261113/proxy_audit_runner_finished.json)
as of 07:23:09.185022 UTC. The previously declared seven control groups passed,
all bound artifact hashes and three receipt copies match. This is consumed-
history exploratory evidence, not forward validation or live improvement.
