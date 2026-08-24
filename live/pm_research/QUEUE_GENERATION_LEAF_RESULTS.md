# Optimization iteration 009 — generation-compatible leaf result

> **Superseded on 2026-08-24 for cancellation-effectiveness inference.** The
> leaf experiment predates the PM/HF data-correctness repair and its labels do
> not have positively measured cancel-before-fill timing.  Its negative model
> result remains a historical diagnostic, not a valid test of preventable
> cancellation.  See `DATA_CORRECTNESS_REPAIR_2026-08-24.md`.

**Verdict: `MODEL_GATE_FAIL` for BTC and ETH. No policy replay.**

Protocol: `QUEUE_GENERATION_LEAF_PROTOCOL.md`. Implementation:
`adverse_move_queue_generation_leaf.py`. Receipt:
`data/pm_5min/derived/adverse_move_queue_generation_leaf_v1.json`. Artifact ID:
`ad3b781fe5e02619dac919cbe1679624f73fb82b133ee5a673e46ad36e1ad7eb`.

Changing only `min_child_samples` from 200 to 20 made both models nonconstant,
so iteration 008's constant prediction was correctly attributed to an
unsplittable tree. It did not reveal a robust harmful-flow signal.

| coin | weighted Brier skill | weighted ROC AUC | q>0.5 fraction | selected gross value by dev day |
|---|---:|---:|---:|---:|
| BTC | -14.05% | 0.505 | 28.14% | -0.0011 / +0.0380 c per generation decision |
| ETH | -59.92% | 0.399 | 58.17% | -0.0398 / +0.0004 c per generation decision |

BTC beats old v5 selection on both days but loses to the train-selected
never-cancel constant on August 23 and has worse calibration than prevalence.
ETH fails calibration, value, constant, and old-v5 comparisons. Neither model
authorizes stateful policy replay.

All five sample, one-parameter, generation-identity, and model-receipt controls
pass. Further model-family, leaf, threshold, H/L, or sample-size selection on
August 23/24 is closed. The next statistically honest step is to accumulate new
complete UTC days and freeze a fresh train/forward split; the PM and HF
collectors remain running.
