# Consolidated conclusion — crypto XS alpha research (as of 2026-08-03)

One honest page pulling together the multi-cycle research. Supersedes the (now-stale) "cost-gated at retail"
bottom-line in `CONSTRUCTION_COST_IMPROVEMENTS_2026-07-29.md` §Executive-summary. Detail lives in that doc +
`SIGNAL_LATENT_MAP_2026-07-28.md`. All scripts in `live/`; everything uncommitted, for review.

## The edge — what it is, validated
- **A beta-NEUTRAL structural low-vol premium.** In the deployable top-40 (majors), the persistently-volatile
  majors (SOL/DOGE/racy alts) are over-demanded by leveraged/speculative flow → underperform the calm majors
  (BTC/ETH) on a beta-adjusted basis. It is ~one factor (lottery siblings MAX/skew redundant); a secondary
  within-name reversal rides along.
- **Gross is real and significant.** Top-40 beta-neutral alpha **OOS Sharpe +2.2..+2.5, day-clustered CI excludes
  0, robust across N=20–50 AND across full-sample-vs-PIT-trailing-ADV universe** (`build_validate_conclusion.py`,
  `build_review_pit.py`). Mechanism (BETWEEN-rvol IC) significant BOTH eras. NOT beta-driven (net beta ~−0.1,
  return is 100%+ alpha; `build_top40_beta.py`).
- **Beta-hedge is the MECHANISM, not an overlay** — the high-vol=high-beta linkage hides the alpha in raw returns
  (raw low-vol book is short-beta and bleeds); targeting the BTC-residual + hedging beta is what unlocks it.

## The result AFTER cost — the honest number (`build_net_result.py`)
PIT top-40, beta-neutral, day-clustered CI:
- **OOS: gross Sharpe +2.5 → net ~+1.0 at ≤8bps (majors cost), but CI SPANS 0** (c6 +1.33 [−0.05,+2.70];
  c8 +0.93 [−0.39,+2.32]; c12 +0.15..+0.84; c24 ~0-to-negative). Cost eats ~half-to-two-thirds of the gross.
- **RECENT: net ≈ 0, insignificant** (gross itself insignificant; uninformative, NOT reliably negative).
- **⇒ After cost the edge is a POSITIVE-but-not-statistically-significant ~+1 OOS Sharpe at fee-competitive
  majors cost, ~0 at full retail (24bps).** Real gross edge; net profitability hinges on execution cost.

## Deployable configuration
`per-symbol Ridge (BTC-residual target) → beta-hedge → BIG-NAMES universe (top ~20–40 by TRAILING ADV) →
turnover-control (band and/or EWMA λ≈0.7) → liquidity filter`. (EWMA only helps net at HIGH cost; at majors-cost
it's a wash.)

## Useful mechanisms / levers (ranked)
1. **Trade big names only (trailing-ADV top-20–40)** — the single biggest lever: signal survives/cleaner in
   majors, cost collapses 24→~7bps, net −4.5 → ~+1 Sharpe. Cost was mostly long-tail slippage.
2. **Beta-hedge (it's the mechanism).** 3. **Band / liquidity filter / EWMA turnover-control** (validated
   cost/robustness). 4. **Intermediate momentum (14d skip-recent)** — a real, persistent, DIFFERENT-root
   (underreaction) orthogonal sleeve; thin, standalone-only, not a booster.

## What does NOT work (nulls, so we don't re-explore)
- MORE gross from price/vol transforms: dispersion/regime gating, volatility-management, parametric portfolio
  policies — all null (mechanisms don't fit a one-factor reversal/low-vol book).
- Orthogonal free data (positioning/OI/OB-flow): real both-era signal but SUB-COST; carry (funding): non-
  stationary/fails; same-root lottery siblings (MAX/skew): redundant. Amihud/liquidity: capacity-walled.

## Regime context (exploratory, do not hard-gate)
The edge "bets the excitement deflates": works in quiet markets, vulnerable when volatile majors run (froth
persists). RECENT weakness ≈ that. Regime-dependence is itself non-stationary (OOS liked froth, RECENT quiet).

## Honest limits + remaining leverage
- Net is marginal (CI spans 0); RECENT uninformative; small concentrated book (12–40 names); un-fixable
  delisted-name survivorship (majors less exposed).
- **The remaining leverage is EXECUTION COST (≤~8bps makes it plausibly net-positive) and possibly paid
  orthogonal data — NOT more free-data signal research, which is exhausted and understood.**

## Method lessons (for the next edge)
Borrow a paper's PRINCIPLE not its recipe (match mechanism to the book's structure/constraint); find new edges
from a DIFFERENT economic root + a SLOW construction that dodges the cost wall; validate with PIT universe +
day-clustered CIs + explicit look-ahead checks + within/between decomposition; always separate gross-significant
from net-marginal. This discipline corrected the headline reads 3× this cycle.

## Scripts (this cycle)
build_signal_decomp / build_leg_check / build_edge_why / build_top40_why / build_top40_dig / build_top40_beta /
build_top40_regime / build_liquidity_tiers / build_deployable_stack / build_validate_conclusion / build_review_pit /
build_net_result / build_momentum_ts / build_momentum_net / build_mom_pipeline / build_mom_validate / build_edge_hunt
(+ cycles 1–5: build_notrade_band / build_beta_neutral / build_combined / build_net_cost / build_vol_managed /
build_vol_weight / build_ppp / build_turnover_opt / build_ewma_hedge; orthogonal loop: orthogonal_harness / orth_*).
