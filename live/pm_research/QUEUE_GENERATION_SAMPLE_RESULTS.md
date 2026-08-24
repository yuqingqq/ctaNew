# Optimization iteration 008 — five-window generation sample result

> **Superseded on 2026-08-24 by the data-correctness rebuild.** The original
> result below used a trace that omitted warm-up trades, deduplicated queue
> resync states, performed epoch-scale float time arithmetic, and treated
> unresolved public-tape fills as preventable.  See
> `DATA_CORRECTNESS_REPAIR_2026-08-24.md`.  The rebuilt 50-window trace passes
> authoritative parity, but zero economic rows have positively measured cancel
> timing; both models remain rejected and are constant on the residual
> optimistic labels.  Corrected artifact ID:
> `3414fc563b673e7f010614b367b2191dee69d5f52cf6544d0f25162d45d887ba`.

**Verdict: `REJECT` for BTC and ETH. Research only.**

Protocol: `QUEUE_GENERATION_SAMPLE_PROTOCOL.md`. Implementation:
`policy_optimizer_queue_generation_sample.py`. Receipt:
`data/pm_5min/derived/policy_optimizer_queue_generation_sample_v1.json`.
Artifact ID: `2e748945748e4d9730c6fa4ec927036d89905a2c2d50732f24446fedf94089f6`.

The frozen 50-window sample completed with all six sample, generation,
isolation, determinism, all-false, and receipt controls passing.

| coin | generation train/dev rows | economic train/dev generations | fitted q>0.5 fraction | result |
|---|---:|---:|---:|---|
| BTC | 9,512 / 5,537 | 123 / 70 | 0% | constant never-cancel |
| ETH | 22,104 / 12,585 | 72 / 43 | 0% | constant never-cancel |

The fivefold data increase did not make the inherited LightGBM nontrivial.
The cause is now deterministic: its pinned `min_child_samples=200`, while each
coin has fewer than 200 total economic training generations. No split is
possible, so Brier skill is numerically zero and ROC AUC is 0.5.

On the larger visible sample, no-cancel skew beats the baseline on both BTC
development days but raises terminal inventory; ETH is mixed. These are seen
development results, not an adoption. The next model-only iteration freezes a
generation-compatible leaf floor before rerunning the same sample; it does not
tune thresholds or policy.
