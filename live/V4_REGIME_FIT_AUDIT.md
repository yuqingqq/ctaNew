# V4 Model Regime-Fit Audit

Scope: model-only residual-alpha edge, not bot PnL. Metric is 24h forward `alpha_vs_btc_realized` in bps, selected by current v4 two-book shape (`K_LONG=1`, `K_SHORT=2`). Positive spread means the model distribution is favorable before costs and overlays.

Regime classifier: fixed, PIT-observable BTC 30d return buckets. Stability is evaluated across calendar periods with at least 30 cycles in that bucket.

## Bucket Results

| bucket | class | cycles | spread | long | short edge | +cycle | stable periods | worst period |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bear_deep_lt_-20` | **CONDITIONAL** | 277 | 143.4 | -4.8 | 148.2 | 54% | 2 / 100% | 121.1 |
| `bear_mid_-20_-15` | **AVOID** | 256 | -63.8 | 20.5 | -84.4 | 50% | 4 / 50% | -481.4 |
| `bear_mild_-15_-10` | **CONDITIONAL** | 604 | 203.8 | 134.1 | 69.6 | 60% | 5 / 80% | -3.5 |
| `side_down_-10_-5` | **CONDITIONAL** | 742 | 50.2 | 49.6 | 0.7 | 53% | 5 / 80% | -30.3 |
| `side_flat_-5_+5` | **FRAGILE** | 2512 | 1.3 | -11.6 | 12.8 | 51% | 5 / 40% | -70.6 |
| `side_up_+5_+10` | **CONDITIONAL** | 860 | 117.0 | -4.7 | 121.7 | 54% | 5 / 80% | -100.3 |
| `bull_mild_+10_+15` | **FRAGILE** | 808 | -1.5 | -51.6 | 50.1 | 50% | 5 / 60% | -222.0 |
| `bull_hot_+15_+20` | **AVOID** | 378 | -38.1 | 21.8 | -59.9 | 49% | 4 / 0% | -118.6 |
| `bull_deep_gt_+20` | **AVOID** | 1050 | -47.1 | -0.1 | -47.0 | 49% | 3 / 0% | -301.9 |

## Period Texture

Macro regime edge by period:

| macro | period | cycles | spread | long | short edge |
|---|---:|---:|---:|---:|---:|
| `bear` | `2023` | 199 | 215.0 | 183.4 | 31.6 |
| `bear` | `2024` | 193 | 122.5 | 48.2 | 74.3 |
| `bear` | `2025H1` | 181 | 146.4 | 119.6 | 26.9 |
| `bear` | `2025H2` | 228 | 46.5 | 63.7 | -17.2 |
| `bear` | `2026` | 336 | 127.8 | 8.8 | 118.9 |
| `bull` | `2023` | 969 | 19.3 | 36.0 | -16.7 |
| `bull` | `2024` | 772 | -46.5 | -27.9 | -18.6 |
| `bull` | `2025H1` | 241 | -233.2 | -229.4 | -3.8 |
| `bull` | `2025H2` | 113 | 43.3 | 27.1 | 16.2 |
| `bull` | `2026` | 141 | 24.0 | 36.8 | -12.8 |
| `side` | `2023` | 1022 | 58.9 | 19.5 | 39.4 |
| `side` | `2024` | 1231 | -6.1 | -37.4 | 31.3 |
| `side` | `2025H1` | 532 | -39.7 | -100.4 | 60.7 |
| `side` | `2025H2` | 745 | 49.6 | 92.7 | -43.1 |
| `side` | `2026` | 584 | 124.3 | 24.3 | 100.0 |

Side sub-buckets by period, because side is the unstable distribution:

| side bucket | period | cycles | spread | long | short edge | +cycle |
|---|---:|---:|---:|---:|---:|---:|
| `side_down_-10_-5` | `2023` | 167 | 91.7 | 32.3 | 59.5 | 56% |
| `side_down_-10_-5` | `2024` | 276 | -30.3 | 23.0 | -53.2 | 47% |
| `side_down_-10_-5` | `2025H1` | 110 | 39.1 | 46.2 | -7.0 | 49% |
| `side_down_-10_-5` | `2025H2` | 142 | 110.0 | 71.1 | 38.9 | 57% |
| `side_down_-10_-5` | `2026` | 47 | 221.0 | 210.3 | 10.7 | 64% |
| `side_flat_-5_+5` | `2023` | 650 | 61.0 | 28.4 | 32.6 | 53% |
| `side_flat_-5_+5` | `2024` | 658 | -31.3 | -79.8 | 48.5 | 48% |
| `side_flat_-5_+5` | `2025H1` | 293 | -42.6 | -107.2 | 64.6 | 49% |
| `side_flat_-5_+5` | `2025H2` | 500 | -70.6 | 48.0 | -118.6 | 49% |
| `side_flat_-5_+5` | `2026` | 411 | 77.6 | 30.0 | 47.6 | 57% |
| `side_up_+5_+10` | `2023` | 205 | 25.4 | -19.2 | 44.6 | 53% |
| `side_up_+5_+10` | `2024` | 297 | 72.2 | 0.2 | 72.0 | 58% |
| `side_up_+5_+10` | `2025H1` | 129 | -100.3 | -210.0 | 109.7 | 40% |
| `side_up_+5_+10` | `2025H2` | 103 | 549.7 | 339.5 | 210.2 | 58% |
| `side_up_+5_+10` | `2026` | 126 | 240.4 | -63.8 | 304.2 | 55% |

## Interpretation

- **FARM** buckets are distributions where the residual mean-reversion model has positive edge across multiple periods. These are the model natural habitat.
- **CONDITIONAL** buckets have positive average edge but weaker period stability or one bad sub-period; trade only with reactive gate / kill-switch.
- **FRAGILE** buckets are not reliably separable by this BTC30 classifier alone; they need either side-specific conditioning or lower gross.
- **AVOID/REDESIGN** buckets are hostile to this residual mean-reversion model. Do not tune the same model harder there; use a different construction, usually momentum/crowding-aware.

## Action Map

1. Keep bear farming: bear buckets are the most consistent positive distribution, mostly driven by the short edge plus still-positive long edge.
2. Treat side as the main unresolved problem: side can be excellent or inverted by period. BTC30 side sub-buckets alone do not solve the sign problem.
3. Avoid deep/hot bull mean-reversion shorts: the model short edge flips negative when BTC30 is hot. Bull needs a separate momentum-long-only or crowding-aware design, not more residual-MR tuning.
4. Next optimization should be regime-specific: preserve the v4 residual model in FARM buckets; in FRAGILE side buckets test conservative long-leg de-gross/veto; in bull build a separate model/construction and require separate validation.

