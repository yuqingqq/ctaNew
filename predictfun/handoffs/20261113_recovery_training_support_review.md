# Fixed recovery proxy: fast-return inputs excluded by training setup

As of 2026-11-13T03:54:16.600667+00:00, the
[training-support audit](../reviews/model_recovery_20261113/training_support_audit.json)
passes all five control groups and reconstructs the fixed model's available
feature set from its exact training data and unchanged LightGBM settings.
Source: `51d342e851cbb803d625784d69e7457f75596ea3`.

The population is all 3,741 eligible short-target seed actions: 1,939 on Oct23
and 1,802 on Oct24. Their data receipts are as of 03:03:26.601858 and
03:27:54.261506 UTC respectively. There are no excluded short-target actions.
This audit describes training support only; it does not score outcomes or
measure later forecasting performance.

Of 83 supplied features, 65 are available in the serialized model and 64
actually receive splits. Eighteen have no constructed bins: six constant
grid-lag age/missing indicators and twelve varying features. The latter include
all five 25–500 ms Binance returns, their side-signed versions, the Predict.fun
spread and the gap × return interaction. These return inputs are observed and
varying, rather than missing or constant:

| Raw return input | Negative actions | Zero actions | Positive actions | Missing | Constructed bins |
|---|---:|---:|---:|---:|---:|
| 25 ms | 31 | 3,685 | 25 | 0 | 0 |
| 50 ms | 54 | 3,641 | 46 | 0 | 0 |
| 100 ms | 99 | 3,557 | 85 | 0 | 0 |
| 250 ms | 183 | 3,404 | 154 | 0 | 0 |
| 500 ms | 357 | 3,062 | 322 | 0 | 0 |

Every row describes the same 3,741 training actions. The receipt contains the
full support counts, including missing values, for all 83 features.

The fixed configuration has `min_data_in_leaf=500`, `feature_pre_filter=true`
and 80% row bagging. Its 300 serialized trees contain four leaves in 61 trees
and five leaves in 239 trees, despite a maximum of 31. Reconstructing dataset
bins with the original settings reproduces all available/unavailable feature
identities exactly. Synthetic controls retain a dense binary feature and
exclude a rare binary feature and a constant feature under those same settings.

LightGBM 4.6.0 documents that its default feature filter removes features that
are unsplittable under the minimum-leaf setting; the leaf minimum itself is an
approximation based on Hessians. This review does not infer an exact filter
boundary from raw counts alone. [Official parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#feature_pre_filter).

The practical limitation is specific: this fixed recovery proxy has no direct
tree splits on the fast-return inputs listed above. It can still respond
through other features, including longer returns and book/flow history.
Specifically, the frozen `History.batch` defines `s_return_accel` as
`side_sign * (bn_r250 - 0.25 * bn_r1000)`. The same audit records 255 bins and
13 splits on that feature. Thus 250 ms return information still reaches the
model through this transformation; the direct-feature exclusion must not be
described as total absence of short-return information. This
does not establish underfitting, the value of those discarded inputs, or that
lowering the leaf minimum would improve forecasts or trading.

The frozen experiment continues unchanged. Its already declared
[later-day proxy forecast audit](20261113_recovery_proxy_audit_plan.md) remains
pending. No new model, parameter search, outcome-based selection, policy change
or promotion resulted from this descriptive check. The bounded run exited zero,
used 219 MiB peak RSS and took 1.01 seconds on the free auxiliary slot.
