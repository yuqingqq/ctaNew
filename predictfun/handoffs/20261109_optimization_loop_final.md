# Optimization loop: completed bounded research cycle

As-of **2026-11-09T18:11:31.114972+00:00**. Seven passes are complete, with eight receipt sets because exact-v3 paired parity was recorded as a separate supplement. The final audit verified **91 unique pinned artifacts**, every declared source at its committed revision, declaration ancestry/timing, output hashes, receipt links and summary arithmetic. The frozen v3 hash is unchanged. All research jobs are finished; no strategy was promoted or deployed.

The original plan at **4666fc52** required the inventory and duration/latency rounds, a published comparison and the next evidence needed. The loop completed those, then separately declared and completed equal-clock ordering, native A2 execution, native latency sensitivity, paired-engine validation and paired-fill retention integration. This closes that bounded investigation. It does **not** establish the wider objective of higher P&L with lower realized inventory variance.

## Main findings

On **220 consumed Nov8 windows**, first60s, with **279,059 factual decisions and13,077 neutral pull actions per arm/profile**, zero excluded windows:

| 50ms updates-first versus its reference | Fewer venue requests | More original-price quote time | More active share-time |
|---|---:|---:|---:|
| Faster assumed feedback | 5.35% | 2.94% | 3.97% |
| Base assumed feedback | 5.33% | 4.17% | 4.10% |
| Slower assumed feedback | 3.72% | 4.37% | 2.58% |

The native comparison was produced as-of **2026-11-09T17:15:06.600749+00:00**. This is one consumed partial day, not independent validation; there is no confidence interval. Faster/base/slower use assumed create/cancel/pending delays20/10/50,40/22/95,80/50/160ms. They are not measured latency quantiles or separate samples.

The50ms rule keeps **90.34–97.48%** of the100ms coverage gain across those assumptions, but active exposure and off-target time remain higher than the reference. Fewer cancelled-order reservations do not imply lower active inventory risk.

Inventory restrictions offer the opposite trade-off in the earlier planner: individually reducing retention keeps **4.86%** of unrestricted coverage gain; requiring room for all same-side reservations keeps **0.68%**. Their added non-reducing share-time and active inventory-range expansion are zero on the factual inventory path, while their aligned-time gains are only **0.129% /0.018%**. This is conditional exposure accounting, not counterfactual fill or variance validation.

These calculations were recomputed by the final audit from the verified receipts. The scope of the native and earlier-planner comparisons differs; they are not pooled into one economic score.

## Implementation delivered

- `LZ`: generated paired-order research engine matching pinned native f20fdb1 incremental sizing. Same-price growth keeps the front member; paired fills and pending members preserve logical order accounting.
- Fixed an LZ-only rejection-accounting mismatch: terminal rejection releases the unanswered-create slot immediately, avoiding an unnecessary250ms retry. This intentionally changes fixed-size rejection-stress paths. Ordinary fixed-size controls and no-token outputs pass; **blanket B4 fixed-size parity does not**.
- `LC50`: generated bounded retention with exact-deadline update priority, no off-guard replenishment, and cancellation-only prediction/fill probes. Late cancelled-order fills can preempt retention through CAP10; RO, stop, price and score limits also preempt it.
- B4/exact-v3 checks:238 synthetic native traces and51 Python checks. LC50 integration:228 traces and33 Python checks; all172 holds ended within50ms with no off-guard top-up. Each native oracle has positive and deliberately corrupted-request/state controls.

The native oracles consume Python effective desired quotes. They verify mapping, requests and fill lifecycle, not independent native model/guard calculation or actual live scheduling. Synthetic complete-function-output parity is narrower than historical output-file parity.

## Remaining gates — not satisfied by this loop

| Requirement | Current evidence | What would resolve it |
|---|---|---|
| Higher net trading value and lower realized inventory variance | Not established by quote-time proxies | Calibrated fills followed by economic comparisons |
| Calibrated filled and unfilled own-order paths | Existing exact-identity clock audit rejects the old timing assumptions | Own-fill ms match/status/confirmation clocks and receipt timestamps; reconcile both action populations |
| Independent validation | One consumed partial day only | At least5 complete untouched UTC days under a committed pipeline; keep Nov9–13 protected |
| Full B4 parity | Fixed-size rejection exception; historical file outputs untested | Explicit handling of the frozen bug in comparison semantics plus historical file-output parity |
| Deployed LC50 ordering and native guard equivalence | Synthetic scheduler/effective-want oracle only | Separate scheduler/policy integration evidence before a live port |

A4b capture support is recorded as implemented by the PORT session on native main79a05d3, but a new suitably captured run is not part of this cycle. No deployment or protected-data read is authorized by this closure. The frozen forward race and broader B1–B8 backlog remain under their existing procedures.

## Audit and reproduction

- Current index: `reviews/optimization_loop_20261109/findings_v8.json`, superseding v7 without modifying earlier receipts.
- Final audit: `reviews/optimization_loop_20261109/completion_audit/receipt.json`; checker `research/pf_optimization_loop_audit.py`, sourcea921b70c.
- Original and expanded declarations, immutable failed attempts, receipts and verifications remain in `reviews/optimization_loop_20261109/`. Mutable register: `plan.json`.
- Detailed comparisons and runnable validation commands: handoffs v2 (inventory/ordering), v4 (native latency), v6 (paired/full-v3), v7 (paired continuity). Numeric stages use `run_light.sh` serially. Existing frozen destinations refuse overwrite; reproduction needs fresh destinations.
- Research branch: `research/predict-opt-loop-20261109`; checkout `/tmp/cta-opt-loop-20261109`. No production code, frozen engine, dollar loss budget or protected data was changed.

All numeric and audit jobs completed18:09:51Z. The audit used21MiB peakRSS. The shared research slot is free; recheck process state before C1. There is no background optimization job still running.
