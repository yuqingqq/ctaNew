# Maker3 in thin / wide Predict books vs calm books — 2026-11-25

**Answer.** The Predict book does thin out and widen toward the close: one-sided books go from ~0 % before 120 s to 76 % in the last 5 s, and spreads > 4 ¢ from 0.2–0.5 % to 35 %. Maker3 does **not** make its money there. Across all 934 of its markets (consumed Oct 24–27), its passive fills in calm books before 270 s earn **+$13,551** (+0.77 ¢/share, beating other top-10 makers at +0.18 and the field at −0.02). Its fills stamped ≥ 270 s lose on all four days in every book class: −$3,745 in thin/wide books (−4.52 ¢/share) and −$2,891 in calm books. Late in the window it does worse per share than the other top-10 makers and the field. Thin/wide books give Maker3 a larger instantaneous edge, but the adverse move after the fill is larger still.

At market level the picture is different. Maker3 earns in markets whose last minute is thin/wide (contested, near 50 ¢): +$9,145 over 276 markets, mostly completed round trips. That money is made **before** the late window, and +$8,173 of it comes from Oct 26. Markets whose last minute stays calm (mostly already decided) lose −$6,510 on all four days, almost all from inventory left at resolution.

The late-heavy markets seen on the dashboard (e.g. 3789484) are the minority: median Maker3 market has 2.8 % of its passive shares after 180 s; 71/934 have a majority there.

Saved data only; consumed days; descriptive attribution, no intervals (4 days < 5), no model / replay / validation.

## 1. Check of the earlier analysis

| Earlier packet | What it measured | Why it does not answer this question |
|---|---|---|
| [maker3_spread_risk_20261124](../maker3_spread_risk_20261124/report.md) | displayed spread at 3 clocks; FIFO pairs with *both* legs > 4 ¢ | 19 complete BOOK markets (17 active, 1.8 % of Maker3's 934); no depth / thinness; "both legs wide" is a strict pair criterion; wide-book shares swing 308.58 → 1,551.93 with the clock in 3789484 |
| [maker3_regime_spread_20261125](../maker3_regime_spread_20261125/report.md) | "volatile" = 10 s RMS of Predict 1 s mid changes; spread bands | same 19 markets; mid-movement volatility is not the USER's notion (thin book, wide BBO); its volatile-exit gain is two Oct 26 markets (without them −$69.59) |

Both are internally consistent (their checks pass) but their population and definitions cannot carry a general thin/wide-book claim. This packet redoes the question on every Maker3 market with book depth included.

## 2. Definitions (contract.md, fixed before economics)

- Population: every accepted fill leg in `econ/fills_valued` on Oct 24–27 (status ok, resolution defined), **2,587,302 Maker3 passive shares (119,269 legs) + 9,580 taker legs in 934 markets**. Comparison: the other nine ex-ante top-10 makers (by Oct 23 shares) and the field.
- Book clock **P = settlement stamp − 2.5 s**: before the match on every lifecycle row but 22 of 60,000 (window ≤ 2.78 s). Sensitivity clock U = the stamp (post-trade).
- Classes at the clock: CALM (spread ≤ 1 ¢, thinner-side touch > 43.7 sh = market-wide q25), THIN (≤ 1 ¢, touch ≤ 43.7), WIDE (> 1 ¢), ONE_SIDED, UNKNOWN. The touch includes any Maker3 order resting there.
- Values per share, YES space: edge = s·(mid_P − p); adverse = s·(V − mid_P); net to resolution = edge + adverse − fees; markouts from the stamp (1 s, 10 s, END = 298 s mid). A fill's net is exactly its marginal value against not doing that fill and holding everything else to resolution; reducing fills therefore carry the outcome they gave up.

## 3. The book thins and widens toward the close (outcome-blind)

| window_s | market_seconds | one_sided | spread_gt_1c | spread_gt_4c | thinner_touch_median_sh | two_sided_at_extreme_price |
|---|---:|---:|---:|---:|---:|---:|
| 000-060 | 69060 | 0.0% | 4.6% | 0.5% | 121 | 0.3% |
| 060-120 | 69060 | 0.0% | 3.3% | 0.2% | 153 | 5.5% |
| 120-180 | 69060 | 1.3% | 3.3% | 0.2% | 169 | 21.3% |
| 180-240 | 69060 | 11.4% | 4.1% | 0.3% | 160 | 37.2% |
| 240-270 | 34530 | 29.4% | 13.9% | 0.9% | 101 | 50.3% |
| 270-285 | 17265 | 46.3% | 26.1% | 3.7% | 62 | 58.9% |
| 285-295 | 11510 | 60.2% | 43.8% | 11.3% | 25 | 66.4% |
| 295-300 | 5755 | 75.9% | 60.6% | 35.2% | 12 | 74.7% |

One-sided books are pinned (the remaining side sits at 1 ¢ or 99 ¢): mostly decided markets. A thin/wide two-sided book near 50 ¢ is the contested case and is split out in §5.

## 4. Maker3's activity is front-loaded; more of its late fills meet thin/wide books

| window_s | share_of_m3_volume | in_thin_or_wide | in_one_sided | m3_share_of_all_maker_volume |
|---|---:|---:|---:|---:|
| 000-060 | 27.3% | 29.1% | 0.0% | 6.5% |
| 060-120 | 22.9% | 25.3% | 0.0% | 6.5% |
| 120-180 | 20.1% | 24.0% | 0.2% | 5.7% |
| 180-240 | 16.7% | 27.3% | 1.8% | 4.9% |
| 240-270 | 7.3% | 38.1% | 4.5% | 4.3% |
| 270-285 | 3.3% | 46.7% | 4.4% | 4.8% |
| 285-295 | 2.0% | 64.6% | 7.1% | 5.3% |
| 295-300 | 0.5% | 78.7% | 12.3% | 2.0% |

Maker3 is a steady ~5 % of maker volume and does not take over the late book; in the last 5 s its share falls to 2 %. 29.1 % of all its passive volume meets a thin or wide book at P; after 270 s, 55.5 %.

## 5. Economics: calm books pay, thin/wide books do not

| book_at_P | group | shares_k | net_usd | net_c | edge_c | adverse_c | m1s_c | m10s_c | end_c | net_c_if_book_at_U |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CALM | MAKER3 | 1,808.7 | +10,661 | +0.59 | +1.55 | -0.96 | +0.58 | +0.34 | +1.42 | +0.07 |
| CALM | OTHER_TOP10 | 9,604.3 | +15,521 | +0.16 | +1.59 | -1.43 | +0.33 | +0.17 | +1.16 | +0.28 |
| CALM | FIELD | 24,590.7 | -11,103 | -0.05 | +0.97 | -1.01 | +0.06 | +0.00 | +1.48 | +0.11 |
| THIN | MAKER3 | 565.0 | -2,976 | -0.53 | +2.09 | -2.61 | +0.66 | +0.29 | -0.32 | +0.66 |
| THIN | OTHER_TOP10 | 2,842.9 | -909 | -0.03 | +1.82 | -1.86 | +0.42 | +0.18 | -0.52 | -0.26 |
| THIN | FIELD | 6,457.1 | -8,531 | -0.13 | +1.60 | -1.74 | +0.13 | -0.20 | +0.86 | -0.71 |
| WIDE | MAKER3 | 186.6 | -1,537 | -0.82 | +3.96 | -4.78 | +0.56 | -0.10 | +3.45 | +0.22 |
| WIDE | OTHER_TOP10 | 761.0 | -355 | -0.05 | +2.63 | -2.68 | +0.63 | +0.29 | +0.24 | -0.55 |
| WIDE | FIELD | 2,887.3 | -9,461 | -0.33 | +3.68 | -4.01 | +0.73 | +0.34 | +0.17 | -0.13 |
| ONE_SIDED | MAKER3 | 26.6 | +46 | +0.17 |  |  | -0.04 | -1.13 | -2.49 | +1.22 |
| ONE_SIDED | OTHER_TOP10 | 60.0 | +827 | +1.38 |  |  | -0.36 | -1.30 | +6.00 | +1.36 |
| ONE_SIDED | FIELD | 1,393.9 | +8,269 | +0.59 |  |  | -0.76 | -1.62 | -1.50 | +0.35 |

Early vs late (day signs Oct 24/25/26/27; top3_market_share = top-3 markets' fraction of absolute per-market contributions):

| stamp | book_at_P | m3_shares_k | m3_markets | m3_net_usd | m3_net_c | other_top10_net_c | field_net_c | m3_day_signs | top3_market_share | m3_net_c_at_U |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| <270 | CALM | 1,751.4 | 934 | +13,551 | +0.77 | +0.18 | -0.02 | +++- | 6% | +0.20 |
| <270 | THIN_or_WIDE | 668.8 | 933 | -768 | -0.11 | -0.06 | -0.11 | --+- | 5% | +1.05 |
| <270 | ONE_SIDED | 17.5 | 69 | -87 | -0.50 | +1.26 | +0.83 | --+. | 43% | +1.10 |
| >=270 | CALM | 57.3 | 141 | -2,891 | -5.05 | -2.83 | -0.63 | ---- | 17% | -4.82 |
| >=270 | THIN_or_WIDE | 82.8 | 138 | -3,745 | -4.52 | +0.42 | -0.61 | ---- | 18% | -4.93 |
| >=270 | ONE_SIDED | 9.1 | 75 | +132 | +1.45 | +1.79 | +0.10 | +.+. | 56% | +1.37 |

Readings (computed in `summarize.py`):

- **Calm early fills are Maker3's profit centre:** +0.77 ¢/share, ahead of both peer groups, positive on 3/4 days (Oct 27, 71 markets, negative), not concentrated (top-3 markets 6 %).
- **Thin/wide books give more edge, more adverse selection:** edge +2.09 / +3.96 ¢ vs +1.55 calm, adverse −2.61 / −4.78 vs −0.96. Before 270 s the thin/wide net is ≈ field (−0.11 ¢) and its sign flips at the U clock (+1.05). It is not a stable reading. The U clock is post-trade and partly caused by the fill itself.
- **Late fills lose robustly:** ≥ 270 s, thin/wide −4.52 ¢ and calm −5.05 ¢, negative on all 4 days at both clocks, not concentrated (17–18 %). In thin/wide late books Maker3 is −4.52 ¢ against +0.42 for other top-10 makers and −0.61 for the field.

The late loss sits in **near-money (10–90 ¢) markets**:

| price | book_at_P | legs | markets | shares_k | net_usd | net_c | edge_c | adverse_c | m1s_c | m10s_c | end_c |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NEAR | CALM | 787 | 75 | 23.7 | -1,887 | -7.95 | +3.10 | -11.04 | +0.85 | -3.23 | -6.76 |
| NEAR | THIN | 1128 | 96 | 27.9 | -2,170 | -7.77 | +7.55 | -15.32 | +1.18 | -0.68 | +0.37 |
| NEAR | WIDE | 1384 | 92 | 34.3 | -1,142 | -3.33 | +8.57 | -11.90 | +1.55 | -1.28 | +0.03 |
| EXTREME | CALM | 874 | 108 | 33.5 | -1,004 | -2.99 | -1.32 | -1.68 | +0.88 | +1.89 | +2.12 |
| EXTREME | THIN | 219 | 68 | 6.8 | -7 | -0.10 | +0.41 | -0.52 | +1.20 | -0.20 | +0.72 |
| EXTREME | WIDE | 331 | 66 | 13.8 | -426 | -3.09 | +10.03 | -13.11 | +0.59 | +1.90 | -1.92 |
| EXTREME | ONE_SIDED | 197 | 75 | 9.1 | +132 | +1.45 |  |  | +1.62 | +3.30 | -4.21 |

The edge against the mid 2.5 s earlier is large (+7.6 / +8.6 ¢ in near-money thin/wide books) and the 1 s markout is positive. The 10 s markout turns negative, and the loss is realised by resolution: classic end-game adverse selection when the outcome is still open. The P mid can be up to 2.5 s stale near the close, so part of that "edge" is price movement before the match, not spread captured.

## 6. Market level: contested markets pay, decided-but-calm markets lose on inventory

Where Maker3's net (both roles) falls in the window:

| window_s | all_markets_net_usd | volatile_late_markets | calm_late_markets | pinned_late_markets |
|---|---:|---:|---:|---:|
| 000-060 | +3,156 | +3,438 | +677 | +779 |
| 060-120 | +7,480 | -385 | +507 | +5,006 |
| 120-180 | +4,896 | +5,258 | +619 | -880 |
| 180-240 | +183 | +653 | +45 | -497 |
| 240-270 | -4,851 | +2,639 | -4,908 | -1,385 |
| 270-285 | -3,670 | -831 | -2,414 | +12 |
| 285-295 | -2,609 | -1,809 | -758 | -75 |
| 295-300 | -93 | +183 | -278 | -3 |
| pre/post_stamp | -1 | -1 | +0 | -0 |

Markets by the class of most of their 240–299 s seconds:

| market_class | markets | decided_by_270s | net_usd | round_trips_usd | residual_usd | net_per_market | oct26_net_usd | day_signs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VOLATILE_LATE | 276 | 27% | +9,145 | +14,051 | -4,622 | +33.13 | +8,173 | +++- |
| CALM_LATE | 192 | 76% | -6,510 | +46 | -6,392 | -33.91 | -2,278 | ---- |
| PINNED_LATE | 367 | 100% | +2,958 | +2,052 | +1,170 | +8.06 | +1,372 | +-++ |
| MIXED | 99 | 81% | -1,102 | -330 | -655 | -11.13 | -765 | -+-- |

Maker3's total over the four days is +$4,491 (round trips +$15,818, residual −$10,499, fees −$828). Two separate things happen:

1. **Contested markets** (thin/wide late, mostly undecided at 270 s) carry the round-trip profit. That profit is earned before ~270 s (+$11,603); fills stamped ≥ 270 s in those markets lose −$2,458. Oct 26 is 89 % of the class total; Oct 27 (10 markets) is ≈ 0.
2. **Calm-late markets** (mostly decided, book still two-sided and tight) lose on all four days. The loss is residual inventory held into resolution (−$6,392) plus losing fills at 240–285 s.

Mean |position| is 215 shares at 240 s and 248 at END (median 41 → 33): Maker3 does not flatten into the close.

## 7. First evidence for the skew / take logic (uncontrolled; the next study's inputs)

Maker3's taker legs before 270 s (only 20 legs after 270 s):

| take_effect | legs | shares_k | net_usd | net_c | fee_c | edge_c | m1s_c | end_c |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ADD | 3905 | 73.5 | +1,024 | +1.39 | +0.53 | -2.69 | +1.41 | +8.22 |
| REDUCE | 5128 | 75.2 | -3,314 | -4.41 | +0.46 | -2.05 | +0.78 | -5.35 |
| FLIP | 274 | 6.5 | +167 | +2.55 | +0.56 | -2.77 | +1.50 | -1.22 |
| OPEN | 235 | 5.5 | +281 | +5.06 | +0.97 | -4.71 | +1.28 | +14.79 |

Position-adding takes behave like informed directional trades (+8.22 ¢/share to the END mark after paying 2.69 ¢ to cross). Reducing takes pay for de-risking (−4.41 ¢ to resolution). This is a descriptive split by realised effect, not an identified trigger.

Passive fills by position size (before 270 s): the reducing share is 49.5–52.4 % up to 1,000 shares and 43.8 % above 1,000. The edge on reducing fills is 0.0–0.5 ¢ smaller than on adding fills. Large positions do not make reducing fills dominate. This is compatible with mild price skew at most; taker-flow direction is not controlled here, so no skew claim.

## 8. What this means for our strategy (no rule adopted)

- Do not chase late thin/wide books: even Maker3 loses there, and more than its peers.
- Maker3's distinctive edge is in **calm books early in the window** (+0.77 ¢ vs field −0.02). This is consistent with the earlier queue / turnover findings.
- The largest drag is **inventory carried into resolution in markets that are decided while the book stays calm**. That is where an end-of-window inventory rule could matter, but it has not been tested here.

## Limits

Consumed Oct 24–27 only (development data); Oct 27 has 71 markets because Maker3 stopped at 05:55. Four days give point estimates and day signs only. Per-fill value to resolution is additive and exact as a "remove this fill" contrast with V exogenous. It is noisy for near-money late fills and says nothing about what the taker flow would have done otherwise. Book state at P is up to 2.5 s before the match; the U clock is post-trade. The touch includes Maker3's own resting size. No private quote, cancel or decision time is identified.

## Integrity

`grid.py` (outcome-blind stage 1) → `contract.md` (definitions before economics) → `build.py` → `summarize.py` (computed readings) → `check.py` → `make_tables.py` (every table above rendered from saved outputs). Checks: 13/13 pass ([checks.json](checks.json)):

- independent merge_asof book state on 4,000 legs, 0 mismatches;
- per-leg future-perturbation falsifier (907,876 later frames, 0 changes) plus a positive control (269/300 fire);
- 8 synthetic classifier cases;
- per-market reconciliation to `econ/maker_market` on all 934 markets (max 2.3e-13 USD);
- inventory re-derived by a loop on 40 markets (5,650 legs);
- headline readings recomputed from legs;
- three corruptions refused.

The first check run failed on the falsifier's own construction: a union of several legs' "future" frames perturbed other legs' P frames. It was rewritten to test one leg at a time; build outputs are unchanged. CPU: wrapped stages 42.5 s plus unwrapped exploratory probes; conservative charge 90 CPU s (0.025 core-h), claude slot.

## Next (USER direction 2026-11-25: establish skew and active-taking logic before broad testing)

One study on the same 934-market population, with the data checks first:

1. **Skew.** Compare Maker3's passive fill-side mix with a side-neutral null: the market's contemporaneous maker flow on Maker3's reducing side (±5 s), by |position|, time and book class. Add fill-price position (inside / at / behind touch) on reducing vs adding sides.
2. **Active taking.** Every market-second is a decision-time row (≈ 280 k). Features known at that time: position, time to end, book class, Predict mid move, own recent fills, BTC move. Target: take / no take, and its direction (add / reduce). Compare against matched non-take seconds and shift nulls (≥ 200 draws, declared first).
3. **BTC data first.** The BTC 1 s grid in `maker_trace_v2/btc` shows the mid unchanged in 73–93 % of seconds, with a 1-tick spread and ~46-tick jumps. It must be verified against the raw bookTicker before any BTC-conditioned take rule.
