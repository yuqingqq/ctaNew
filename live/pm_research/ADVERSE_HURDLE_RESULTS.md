# Action-conditioned adverse-value hurdle — development result

**Status: `DEVELOPMENT / INSUFFICIENT_EVIDENCE`; decision eligible: no.**
Incentives are excluded. No live order, cancellation, venue, or execution path
was added.

Protocol and interpretation: `plans/BE_ADVERSE_MOVE_PLAN.md` section 12.
Implementation: `adverse_move_hurdle.py`. Local receipt:
`data/pm_5min/derived/adverse_move_hurdle_development_v4.json`. Artifact ID:
`64150f91d1292e169221f1f373fea905ce70b140c6b92e769572732d14433c70`.

## Experiment

The v4 model uses the same 231,092 exact-event, action-conditioned rows as v3:
BTC/ETH, one window per coin/day, 2026-08-20/21/22 train and the already-seen
2026-08-23/24 development holdout. All 52 valid horizon/latency cells use the
same pinned LightGBM capacity as v3, without early stopping, threshold tuning,
or holdout-selected hyperparameters.

The zero-inflated target is decomposed as:

```
P(fill remains preventable after L | x, action)
  * E(incentive-free gross cancel value | preventable fill, x, action)
```

This remains one action-bound expected-value model. It does not multiply an
independent price-direction forecast by a separately estimated fill rate. The
decision rule is fixed at `CANCEL iff expected gross value > 0`. Direct v3
LightGBM and refitted ridge are comparators on the identical target and split.

## Result

| test | result |
|---|---:|
| Preventable-fill Brier skill > 0 | 41/52 cells |
| Conditional-value holdout R² > 0 | 0/52 cells |
| Combined unconditional-value R² > 0 | 0/52 cells |
| Selection gain > training-selected constant | 5/52 cells |
| Selection gain > constant on both days | 1/52 cells |
| Aggregate gain > direct v3 tree | 30/52 cells |
| Full development gate | 0/52 cells |

Preventable-fill Brier skill ranges from -0.0618 to +0.2469. It is positive in
all 26 BTC cells and 15/26 ETH cells. The conditional-value R² range is
[-0.9293, -0.0042], and the combined R² range is [-0.3341, -0.0006]. The
nonpreventable target equals zero exactly in all 52 train/holdout audits.

The narrow action-selection hints do not pass robustness checks:

- BTC H=250 ms/L=10, 25 and 50 ms beats the direct tree on both development
  days and has tiny aggregate gains over the constant. Each constant comparison
  reverses sign between the two days; combined R² is -0.140 to -0.103.
- ETH H=100 ms/L=50 ms is the only cell that improves on the constant on both
  days. It still realizes -0.0103 c/decision, has combined R² -0.0084, and loses
  to the direct tree on one day.
- ETH H=1,000 ms/L=150 ms has the largest aggregate selection gain (+0.0252
  c/decision), but its fill Brier skill is negative, its constant comparison
  reverses by day, and combined R² is -0.0644.

## Interpretation

The fast feature layer can predict whether cancellation is still early enough
to prevent a fill, especially for BTC. The unresolved problem is the signed
economic value of the preventable fill: the model cannot reliably distinguish
damage avoided from profitable spread capture forfeited.

The hurdle structure improves policy selection over direct regression in many
cells, but does not beat the constant action robustly. It therefore does not
license dynamic cancel×skew replay, a cancellation layer, or real-latency work.
The next model experiment must target conditional signed value rather than
increase event frequency again.

## Verification

- 116 adverse/skew module self-test checks pass, including 14 new hurdle checks.
- 14 contract-check self-tests pass; canonical contract diff is empty.
- Python compilation and artifact provenance hashes pass.
