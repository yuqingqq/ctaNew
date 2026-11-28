# First model experiment — complete negative result

The first model experiment is complete. Its final integrity check passed at
2026-11-13T02:37:37.072135+00:00, covering all 18 full scenario-day runs,
five policies, 288 windows per policy/run and 7,625 saved inference samples.
The checker binds 1,011 artifacts. Its primary/stress, fill-age and combined
improvement predicates are all false. Integrity PASS does not mean that the
model improved the strategy. The broader better-model goal remains active.

The six consumed evaluation dates are Oct26, Oct27, Oct31, Nov1, Nov2 and Nov3:
1,728 windows per policy/scenario. Primary/stress values below come from the
corrected report as of 2026-11-12T22:40:51.901377+00:00; age500 values come from
the corrected report as of 2026-11-13T02:37:01.110290+00:00.

| Scenario | HISTORY10 compared with | Common / missing marked windows | Marked difference, USD [95% day interval] | Inventory-square change |
|---|---|---:|---:|---:|
| X | causal_guard | 1,517 / 211 | -645.80 [-767.61, -517.42] | +84.45% |
| stress | causal_guard | 1,538 / 190 | -1,153.97 [-1,292.14, -1,016.82] | +30.65% |
| age500 | causal_guard | 1,654 / 74 | -9.34 [-53.25, +24.63] | +219.06% |
| X | static10 | 1,537 / 191 | +28.08 [-59.10, +117.42] | +18.19% |
| stress | static10 | 1,572 / 156 | +17.27 [-104.93, +161.03] | +11.14% |
| age500 | static10 | 1,651 / 77 | +9.70 [-90.21, +107.81] | +64.09% |

Each marked-value comparison uses common windows for its two policies and
retains missing counts. Inventory-square exposure uses every window. Equal
five-share sizing and ten-share reserved-inventory limits did not produce
equal realized inventory risk. Intervals resample the six UTC days with the
declared 1,000 draws. These are small, consumed historical populations; they
do not provide untouched forward validation.

The fill-age sensitivity also fails the declared value/risk test. Its small
marked difference versus STATIC10 is uncertain and accompanies higher risk;
it does not rescue the failed primary/stress result. HISTORY10 has 684 fills
under age500, across all 1,728 windows, versus 13,168 in primary replay. The
sensitivity removes own-fill credit below 500 ms while subsequent queues,
inventory and decisions evolve. It is not a measured venue rule, a post-hoc
filter of the original path or a formal lower bound on P&L.

The complete quote-presence audit is as of
2026-11-13T02:37:22.045021+00:00. Every row below covers all 1,728 windows,
including windows with zero presence. These are actual simulated accepted-order
interval unions, not client-observed orders or guaranteed fillable liquidity.

| HISTORY10 scenario | Any quote, first 60 s | Both sides, first 60 s | Any quote, whole 240.1 s |
|---|---:|---:|---:|
| X | 25.82% | 2.83% | 9.70% |
| stress | 31.97% | 4.35% | 10.94% |
| age500 | 26.68% | 3.28% | 8.00% |

The first minute permits new risk; the rest is reduce-only. Whole-window
presence therefore answers a different question from entry-phase presence,
and neither is a full 300-second live-round uptime measurement. The candidate
still has sparse two-sided quoting. Primary/stress positive accepted lifetimes
have medians of 132/220 ms, respectively; the earlier repeated-pull problem
was not resolved by adding temporal features.

The evidence identifies several distinct issues. The local retained-action
region was miscalibrated despite positive forecasts. The
[fill-timing review](20261113_model_cancel_timing_review.md) also finds negative
realized fill value before the fitted policy's first possible learned cancel
effect. Finally, the [guard source review](20261113_model_guard_comparison_review.md)
shows that the incumbent can withhold before placement and apply immediate
trailing Binance movement responses, while the fitted value policies first
score an accepted order at 100 ms. The guard comparison evaluates complete
policies; it cannot attribute the entire difference to prediction quality.
HISTORY10 and STATIC10 share the scoring schedule and remain the controlled
temporal-feature comparison.

This does not establish that an alternative entry gate, faster threshold or
wider ladder would improve performance. A later cancel cannot alter an already
executed same-generation fill, but earlier decisions and re-entry can change
future exposure. Those distinctions are why the next experiment includes a
short avoidable-value component and a shared recovery rule in labels and replay.

The corrected empty-fill age checker is retained as explicit provenance. Its
version-3 declaration and controls accompany the superseding smoke receipt;
the failed original receipt remains preserved. The separately exercised
corrected entrypoint has exact values and bytes for all 22 artifacts across
five policies and eight zero-floor control windows, as of
2026-11-12T16:45:28.875122+00:00. Those source and artifact hashes were checked
again during this review. Settlement ties use the superseding correction;
nonsettlement metrics and predicates remain unchanged. The final integrity
receipt has a narrower remit than live calibration or an independent estimator.

The [frozen recovery experiment](20261113_model_recovery_plan.md) has started
its first full seed-day build after the original audit chain completed. Its
scientific source remains `78285fe02d00962abc5f7148e6001a8f39a0ddc6`; the original
negative experiment, timing breakdown and guard review are preserved. No new
recovery/value model has demonstrated a gain yet. The recovery estimator is
still a neutral resting-order proxy rather than identified fresh-entry value,
and the full repeated-policy replay must test its usefulness.

No protected payload, live trading change, top-maker parity claim or promotion.
This remains one-layer JOIN research. Live queue/fill/cancel calibration and at
least five later untouched complete dates are unresolved requirements for a
forward-validation claim.

Checklist scope: completion refers to the declared primary, combined execution
stress and full-path age500 runs. It does not mean that every diagnostic in the
shared findings checklist has been completed. This study does not supply an
isolated zero-shift run (C2), B1a-weighted P&L or the exact three age bands (C1),
cancel-pending value attribution (C3), or an age-by-trade-anchor table (C12).
The frozen comparison did not include those partitions; its separate age500
path is not a substitute for them. Partial live matching rates are not valid
per-fill probability weights. These omissions limit execution interpretation
and preclude treating the simulated dollar totals as calibrated live profit.

Sources: [final integrity check](../reviews/model_upgrade_20261112_v2/final_evidence_check.json),
[primary/stress corrected report](../reviews/model_upgrade_eval_20261112_v2/result_settlement_v3.json),
[age500 corrected report](../reviews/model_upgrade_age_20261112_v2/result_settlement_v3.json),
[complete presence](../reviews/model_upgrade_20261112_v2/presence_detail.json),
[age-checker correction](../reviews/model_upgrade_20261112_v2/age_check_v3_declaration.json),
[superseding smoke receipt](../reviews/model_upgrade_20261112_v2/age_smoke_early_v3.json),
[corrected-entrypoint parity](../reviews/model_upgrade_age_entry_parity_20261112/parity.json).
