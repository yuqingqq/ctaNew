# Optimization iteration 007 — generation-deduplicated model result

**Verdict: `REJECT` for BTC and ETH. Research only.**

Protocol: `QUEUE_GENERATION_MODEL_PROTOCOL.md`. Implementation:
`policy_optimizer_queue_generation_model.py`. Receipt:
`data/pm_5min/derived/policy_optimizer_queue_generation_model_v1.json`.
Artifact ID: `3e78dd1bb8e9d616eb98ce1df42abeadfc00b5365c55508c05b894bc5147338f`.

| coin | generation rows train/dev | economic generations train/dev | q>0.5 fraction | model behavior |
|---|---:|---:|---:|---|
| BTC | 2,248 / 694 | 30 / 16 | 0% | constant never-cancel |
| ETH | 5,395 / 2,308 | 23 / 8 | 100% | constant always-cancel |

Both Brier skills are numerically zero and ROC AUC is 0.5. BTC reproduces
isolated `QR_SKEW_ONLY`; ETH is nearly skew-only because its constant signal
rarely gets a new eligible false-to-true edge. Neither passes the model or
stateful adoption gates. All five generation-selection, isolation,
determinism, all-false, and receipt controls pass.

The result rejects per-generation LightGBM on six training windows; it does
not show that generation conditioning is intrinsically useless. The decisive
constraint is sample support: only 53 economic training generations across
both coins. The next bounded test increases windows per coin/day without
changing the model or policy.
