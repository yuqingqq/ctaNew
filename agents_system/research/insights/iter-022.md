# iter-022 — short-horizon cross-sectional REVERSAL / lead-lag spillover (ONLINE-SOTA)

## STATUS: READY (transport-first pre-check PASSED on BOTH universes)

## The idea (online-SOTA led)
A coin's **trailing cross-sectional-relative return** negatively predicts its **next-4h
alpha-residual**: recent relative *out*performers reverse, relative *under*performers bounce.
This is the documented crypto **"seesaw" / cross-crypto lead-lag** effect — a *cross-asset,
short-horizon* return-predictability channel structurally distinct from the production
contemporaneous XS-rank mean-reversion `pred`.

**Mechanism (prior reason to be ERA-STABLE):** capital-rotation + slow information diffusion
under limited attention and limits-to-arbitrage — a *behavioral/microstructure* effect, not a
price-feature regime. This is exactly the kind of mechanism the iter-022 directive flagged as
plausibly stable across 2021-26 (vs the regime-conditional signals — funding/mom180 — that
flipped sign on EXT).

**Citations:**
- Jia, Wu, Yan, Yin (2023), *A Seesaw Effect in the Cryptocurrency Market: Understanding the
  Return Cross-Predictability of Cryptocurrencies*, J. Empirical Finance 74. SSRN 3465924.
  → the five largest coins NEGATIVELY predict next-period small-coin returns; "flight to hot /
  flee from cold coins"; LASSO long-short profitable net of realistic costs.
- Guo, Sang, Tu, Wang (2024), *Cross-cryptocurrency return predictability*, J. Econ. Dynamics &
  Control 163, 104863. SSRN 3974583. → lagged returns of OTHER coins predict a focal coin;
  spillover via slow information diffusion + limited attention; long-short profitable OOS net of cost.
- Hou (2007), *Industry Information Diffusion and the Lead-Lag Effect in Stock Returns*, RFS 20(4)
  → the equity analog (positive lead-lag) and the diffusion mechanism; the crypto sign is the inverse.

## Why this is GENUINELY NEW (not in the rejected ledger)
Every prior alpha/timing attempt was either (a) a *contemporaneous* XS-rank feature, (b) a
*coincident* regime/structure detector (DVOL i5, positioning i9, fast-price i10, Absorption-Ratio
i20), or (c) a *sign-unstable* feature that flipped on EXT (mom180 i15, alt-bear i7, funding i21).
This is a **lagged cross-asset return → forward residual** signal with a documented behavioral
mechanism — never tested here. `return_1d` exists as a *per-symbol Ridge input*, but the model is
trained per-symbol time-series and barely monetizes the **cross-sectional** reversal (pred IC only
+0.0042 vs the raw cross-sectional signal −0.0358 — an 8.5× under-exploitation gap, and orthogonal).

## TRANSPORT-FIRST pre-check (the decisive numbers)
Signal `rel_ret_1d` = trailing 1d return minus the per-cycle cross-sectional mean (PIT). XS Spearman
IC vs forward 4h alpha-residual, 4h-entry grid:

| universe | IC(rel → alpha_resid) | t | n cycles |
|---|---|---|---|
| **HL70 (production)** | **−0.0360** | −9.76 | 2458 |
| **EXT 2021-26** | **−0.0302** | −12.33 | 11573 |

**Sign CONSISTENT (negative) and significant on BOTH universes** → PASSES the transport-first
fail-fast gate (this is where funding/mom180/alt-bear all died).

**Era-stability (EXT by year) — sign negative & significant in EVERY year:**
2021 −0.027(t−4.6) · 2022 −0.029(t−5.2) · 2023 −0.026(t−4.8) · 2024 −0.024(t−4.2) ·
2025 −0.034(t−5.8) · **2026 −0.065(t−6.9)** — strongest in 2026, the era where funding/44-sym alpha decayed.

## Orthogonality to the production pred (HL70, full 5m grid, n=115,449)
| quantity | value |
|---|---|
| IC(rel → alpha_A) | **−0.0358 (t−65.9)** |
| IC(pred → alpha_A) | +0.0042 (t+11.2) |
| XS corr(rel, pred) | **−0.022** (orthogonal) |
| IC(rel **after removing pred**) | **−0.0350 (t−65.0)** |

The reversal IC survives almost intact after projecting out `pred` (only 0.0008 shared). It is a
near-orthogonal, ~8.5× stronger cross-sectional signal than the production predictor.

## Proposed change (ONE, structural / untuned)
In the **sideways** regime (where the book ranks by `pred` mean-reversion), replace the rank key with
an **equal-weight ensemble of the two orthogonal mean-reversion signals**:
`score = z_xs(pred) − z_xs(rel_ret_1d)` (negative sign per the seesaw; `z_xs` = per-cycle
cross-sectional z-score), long top-K / short bottom-K with the existing beta-neutral leg sizing.
**Equal weight (0.5/0.5) is UNTUNED** — a structural choice → G3 waived, and it sidesteps the
nested-OOS death that killed every tuned blend in the ledger (cost-margin swap, decay sleeves).
Bull regime unchanged (mom_30d); bear unchanged (FLAT). Everything else (K=5, 6 sleeves, 4.5bps) fixed.

`rel_ret_1d` is built PIT from the panel's existing `return_1d` (trailing) cross-sectionally
demeaned per cycle — no new data, no look-ahead.

## Pre-registered gates (ADOPT criteria)
- **G1 look-ahead**: `rel_ret_1d` uses only trailing `return_1d`, demeaned within-cycle at entry; PIT. Must PASS review.
- **G2 in-sample**: Calmar > 1.68 (HL70). Report Sharpe/maxDD/Calmar/totPnL.
- **G3 nested-OOS**: WAIVED if the 0.5/0.5 weight is fixed (untuned, structural). If any weight is
  selected, must pass nested-OOS ≥ 1.68. **Pre-register equal-weight to keep G3 waived.**
- **G4 matched placebo (≥p95)**: the ensemble re-rank must beat a matched-random re-rank
  (shuffle `rel_ret_1d` within cycle, ≥100 seeds). Expect PASS given t−65 orthogonal IC — but this
  is the real test that the IC monetizes through the held-book, not just exists.
- **G5 ≥6/9 folds** improvement (or LOFO shows lift not 1-2-fold concentrated).
- **G6 paired-CI** vs champion must not cross zero.
- **G7 universe**: must improve on **HL70** AND hold sign/direction on **EXT** + S44 (transport
  already shown at the IC layer; must survive at the PnL layer — the decisive gate).
- **G8 cost**: report @1/3/4.5bps. **WATCH-ITEM**: reversal can raise turnover (ranks on recent
  movers). The 6-sleeve held-book amortizes turnover, but Implementation MUST report GROSS PnL and
  turnover vs champion (standing rule from iter-019) — the win must not be a cost artifact, and must
  survive at 4.5bps.

## Expected failure modes (honest)
1. **Cost** — the most likely killer: reversal trades against recent movers → potentially higher
   turnover; if the gross-PnL lift is eaten at 4.5bps, REJECT (check GROSS first).
2. **G7 PnL-layer transport** — IC transports, but the held-book monetization may not (the iter-018
   divergence-cut had great HL70 IC-layer behavior that died on EXT PnL). EXT + S44 PnL is decisive.
3. **G4** — if a within-cycle shuffle of rel_ret_1d ranks ~as well, the effect is rank-churn not
   skill (unlikely at t−65 but must be shown).

## Pre-check scripts
- `research/convexity_portable_2026-05-20/scripts/iter022_leadlag_transport_precheck.py` (HL70+EXT IC)
- `research/convexity_portable_2026-05-20/scripts/iter022_orth_fast.py` (orthogonality to pred)
- `research/convexity_portable_2026-05-20/scripts/iter022_era_stability.py` (EXT IC by year)

## Sources
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3465924 (Seesaw effect)
- https://www.sciencedirect.com/science/article/abs/pii/S0165188924000551 (Guo et al 2024)
- https://academic.oup.com/rfs/article-abstract/20/4/1113/1615954 (Hou 2007)
