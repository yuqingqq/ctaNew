# Fixed recovery proxy: training support audit

This descriptive audit follows inspection of the fitted model's metadata, which
marks some short Binance return features unused. It is not a new forecast test,
model candidate or parameter search. The recovery experiment remains frozen.

Before numerical execution, commit `pf_recovery_training_audit.py`. Through the
shared bounded wrapper, verify the exact seed data/model identities, inspect
every training feature's finite/missing/distinct/sign support, reconstruct
LightGBM dataset bins under the unchanged pinned parameters, compare their
availability with the serialized model, and count actual tree leaves/splits.
No model is trained and no outcomes are scored. Synthetic dense, rare, constant
and missing feature controls accompany incorrect-input and availability refusals.

The immutable `training_support_audit.json` binds source, both seed populations,
their as-of times, fit and model. This is training-data description only: no
null, interval, performance claim, threshold choice or changed policy. Whether
the proxy forecasts later values is addressed by the already declared
[forecast audit](20261113_recovery_proxy_audit_plan.md).

LightGBM's [version 4.6.0 parameter documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#feature_pre_filter)
describes the default filter for features unsplittable under `min_data_in_leaf`.
Its [model-text source](https://raw.githubusercontent.com/microsoft/LightGBM/v4.6.0/src/boosting/gbdt_model_text.cpp)
treats `none` feature metadata as unused. Neither fact alone establishes why a
particular input is unavailable or whether a different setting would help.

Run, only when a numeric slot is available:

```bash
python3 -m predictfun.research.pf_recovery_training_audit
```
