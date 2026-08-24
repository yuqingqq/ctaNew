# Optimization iteration 005 — queue-action harmful result

**Verdict: `REJECT` for BTC and ETH. Stateful policy comparison is also
non-promotable because a parent parity control failed. Research only.**

Protocol: `QUEUE_ACTION_HARMFUL_PROTOCOL.md`. Implementation:
`policy_optimizer_queue_action_harmful.py`. Receipt:
`data/pm_5min/derived/policy_optimizer_queue_action_harmful_v1.json`. Artifact
ID: `bc30eb68c4bf3722e74d4c8b0b7d974c5c4433633a455309a0c6c9d42877b022`.

## Model result

| coin | train/dev eligible rows | train/dev economic rows | weighted Brier skill | q>0.5 fraction | decisive failures |
|---|---:|---:|---:|---:|---|
| BTC | 47,006 / 19,221 | 540 / 157 | -5.11% | 64.56% | worse than prevalence; loses to train-selected always-cancel on both dev days |
| ETH | 64,000 / 28,324 | 332 / 62 | effectively 0% | 0.00% | degenerates to never-cancel; no positive daily value; loses to old v5 selection |

BTC's selected static gross value was positive on both development days and
better than old v5 on the same rows, but worse than cancelling every eligible
action. ETH emitted no cancels. The frozen model gate fails for both coins.

## Diagnostic policy replay

BTC appeared to improve the incumbent by +93.33 c and +24.78 c on August
23/24 while reducing cancel traffic and terminal inventory. ETH exactly
collapsed to `QR_SKEW_ONLY` and lost -205.20 c and -33.10 c to the incumbent.
These figures are diagnostic only and cannot be used for selection.

`QR_SKEW_ONLY` did not reproduce the frozen parent aggregate inside the
iteration-005 candidate replay. Isolation showed that a no-cancel arm is
conformant when signal timestamps are held fixed: cancel-effective events
belonging to the new candidate called `resync()` on every arm. A fill can change
skew intent without immediately resynchronizing placement, allowing the new
arm's timer to decide when the comparator change becomes effective. Iteration
006 subsequently reproduced the historical baseline pair exactly when every
cell was isolated, so the stored baseline itself was not changed.

This is a correctness failure, not evidence for or against the candidate.
Further optimization must use the iteration-006 independent-arm wrapper.

Verification: 7 new checks, 16 queue parent checks, deterministic action trace,
all-false candidate parity, model receipt hashes, and point-in-time finite
feature checks passed. The computed queue-parent parity check failed and is
retained in the receipt.
