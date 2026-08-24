# Nonlinear adverse value and pessimistic skew — development result

**Status: `DEVELOPMENT / INSUFFICIENT_EVIDENCE`; decision eligible: no.**
Research only. Incentives are excluded. No live order, cancel, venue, or
latency-measurement path was added.

## Nonlinear value model

Protocol: `BE_ADVERSE_MOVE_PLAN.md` section 10. Implementation:
`adverse_move_nonlinear.py`. Receipt:
`data/pm_5min/derived/adverse_move_nonlinear_development_v3.json`.
Artifact ID:
`b25b750f491e234bafb28a938ececed0986787d91386315c7b99fc5eb15bc0d8`.

- Population: 231,092 exact-event shadow-action rows, BTC/ETH, one window per
  coin/day; train 2026-08-20/21/22, already-seen development holdout
  2026-08-23/24.
- Target: incentive-free gross value of preventing the fill. Maker rebate and
  liquidity rewards are zero; spread capture remains inside observed markout.
- Comparison: pinned LightGBM classifier/regressor against the refitted
  logistic/ridge family on the identical target and split.
- Result: **0/52** nonlinear value cells have positive holdout R²; range
  **[-0.0961, -0.0041]**. The tree classifier beats logistic Brier skill in
  **2/10** coin/horizon cells. **No cell passes the frozen development gate.**

The narrow diagnostics are not hidden. BTC H=100 ms at L=10/25 ms improves on
ridge on both days and has +0.00740/+0.00865 c/decision aggregate selection gain
against the training-selected constant, but the constant comparison reverses
on 2026-08-23 and R² stays negative. ETH H=100/L=75 improves on both comparators
on both days, but realized cancellation value remains negative on both days and
R² is -0.0562. Complexity improved a few action signs; it did not learn value.

## Stateful skew

Implementation: `policy_optimizer_skew.py`. Receipt:
`data/pm_5min/derived/policy_optimizer_stageB_skew_v1.json`.
Artifact ID:
`372ea3c2a202bc6043398ae4b5f6f36f5a9540db27f3f6b0e9e17bb83c509c67`.

The runner maintains the actual replay inventory and uses `SKEW_LB`: only the
reducing side may front, only on genuine level formation; after a full lift it
rejoins behind displayed depth. It does not sum overlapping adverse rows.

- Population: 300 windows, five complete days, BTC/ETH, six frozen Stage-B
  cells.
- Controls: infinite-band skew equals JOIN exactly; deterministic replay;
  pessimistic re-post semantics pinned.
- Result: **60/60 coin-day cells negative; no promotion.** Skew loses less than
  symmetric FRONT in 60/60 cells, but beats JOIN in only 15/60.
- At r_cut=0, size=5, daily p95 cash-at-risk is $7.00-$12.68 BTC and
  $3.55-$4.49 ETH.

Skew remains valuable as an inventory-risk controller. It is not a P&L edge
and does not repair negative per-fill economics.

## Latency consequence

The user-specified sequence was model first, real latency second. The model
gate failed, so the real-latency experiment is deferred rather than used to
rescue an unqualified model. If a future frozen candidate passes, the required
external measurement is the complete chain:

```
event received -> features ready -> decision -> cancel submitted
               -> venue ACK -> order no longer fillable
```

Report p50/p90/p99 by load, coin, side and trigger. `tau_operative` is the
cancel-effective result, never inference time or ACK time alone. The live venue
path belongs outside this research-only repository.
