# Handoff — 2026-05-03

## Summary for the next person

**The strategy is deployable.** Two validated configurations:

1. **h=288 K=5 ORIG25** (legacy production): multi-OOS Sharpe **+3.30**
   [+1.11, +5.42] at 4.5 bps/leg taker (HL VIP-0). Currently running on
   this dev server via HL-only kline feed (Binance FAPI is geo-blocked
   from here).
2. **h=48 K=7 ORIG25** (recommended new): multi-OOS Sharpe **+3.63**
   [+1.31, +6.14] at the same cost. Artifact in `models/v6_clean_h48_*`,
   pipeline verified end-to-end. Awaits move to a FAPI-accessible server.

Both use the **same v6_clean features** (28-column set). The only
differences between configs are HORIZON_BARS (288 vs 48), TOP_K (5 vs 7),
and the model artifact's training labels (24h vs 4h forward demeaned
return). Code is parameterized via env vars — no in-place edits needed
to switch.

### What's settled (don't redo)

- **rc=0.50 is optimal** at both h=288 and h=48 (plateau 0.50-0.70).
  Was 0.33; bumped 2026-05-03.
- **Universe is fixed at ORIG25** (the 25 perps that existed at v6_clean
  selection time). Adding the 14 newer perps reliably hurts on multi-OOS.
- **Feature set is fixed at v6_clean (28 cols).** 50+ feature
  modifications have been tested (DVOL, funding, horizon-matched 4h
  variants, lean trims, additive Stage 2 candidates). All hurt Sharpe.
  v6_clean is a tight local optimum.
- **Model architecture is fixed at LGBM ensemble (5 seeds).** Linear
  oracle confirms the model is at the data ceiling, not the architecture
  ceiling. NN/transformer would add complexity without alpha.
- **The alpha is multi-day cross-sectional reversion**, sampled at
  whatever cadence (h=48 or h=288). Long-window features (return_1d,
  dom_z_7d, corr_change_3d) ARE the signal, not stale momentum proxies.

### What's pending (operational)

1. **FAPI server migration**. Current dev server is geo-blocked from
   `fapi.binance.com`; bot runs on `--source hl` fallback. Move to a
   server with FAPI access (US/EU VPS), then deploy h=48 K=7 there.
   Runbook: `live/MIGRATION_FAPI.md`.
2. **Shadow-mode validation** (optional but recommended). Run h=48 K=7
   alongside live h=288 for 1 week, compare cycle-by-cycle PnL on real
   data before cutover.
3. **Stake 100+ HYPE for HL Bronze tier** (10% taker discount). Brings
   per-leg fee from 4.5 → 4.05 bps. Estimated additional Sharpe ~+0.2.

### What might still help (research, not engineering)

The h=48 vs h=288 split delivers the +0.3 Sharpe lift, but the
architectural ceiling is reached. Genuine future improvement requires
**different problem framing**, not more tuning of v6_clean:

1. **Stack h=48 + h=288 as parallel sleeves.** They sample the same
   alpha at different rates → highly correlated, modest diversification.
   Worth trying if dual deployment is cheap.
2. **L2 microstructure features** (order book imbalance, depth ratio,
   spread dynamics). Never tested. Requires Tardis-style L2 data feed.
3. **On-chain features** (whale flows, exchange in/outflows, stablecoin
   supply). Genuinely orthogonal to price-derived signals. Requires
   Glassnode-style data integration.
4. **Per-symbol options data** (BTC/ETH/SOL only — sparse coverage
   breaks the XS architecture). Could spawn a separate options-overlay
   strategy on those 3.

### What does NOT help (proven negative, don't re-test)

- Different horizons other than 48 or 288 (h=24/36/72/96/144 all give
  Sharpe < +1.5 with CIs touching zero — bimodal structure)
- Different K (K=2-10 sweep done at h=48; K=7 marginal best)
- Different universe (full 39, alt 25-curated, drop-one-out — all hurt)
- Different rc (0.33-1.00 sweep; 0.50 is peak at both horizons)
- Replacing 24h-window features with 4h variants (catastrophic -3.67 Sharpe)
- DVOL features, funding-rate features, regime-conditional MoE, ridge
  regression (all proven worse than baseline LGBM ensemble)

The path to a deployable strategy is below. **For deployment context,
read `docs/STATUS.md` first; for the historical research arc, read
`docs/METHODOLOGY_REVIEW.md`.**

---

## Historical: original Apr 30 framing (preserved)

The signal class (4h-horizon alpha-residual prediction from kline + aggTrade
features, traded as long-short cross-sectional or pair-trading) is **fully
characterized**:

- Real, statistically robust (rank IC +0.035 OOS)
- Magnitude ~5–10 bps gross per trade
- Below retail VIP-0 round-trip cost (12 bps naked / 24 bps hedged)

**[Note 2026-05-03: the cost claim was wrong — actual HL VIP-0 taker is
4.5 bps/leg, not 12. At realistic cost, the strategy is well above
breakeven, Sharpe +3.30 (h=288) to +3.63 (h=48). The "fully
characterized" assertion below was based on the over-cost framing.]**

**Don't try to push this signal harder via more features or models.** The
audits in `ml/research/alpha_*_audit.py` already enumerated what's available.
Marginal improvements (more features, deeper trees, ensemble tricks) have
diminishing returns and have been tried.

The path to a deployable strategy is one of three structural pivots, ranked
by ROI to attempt:

## Option A: Different horizon — TESTED, h=288 IS PREFERRED (Apr 30 corrected)

Earlier "1d rejected" verdict was wrong: it compared bps-per-cycle without
normalizing for cycle length. h=48 has 6 cycles/day; h=288 has 1. On a
per-year basis, h=288 wins at every realistic fee tier:

- VIP-0 retail: h=48 -164%/yr vs **h=288 -32%/yr**
- VIP-3 taker: h=48 -54%/yr vs **h=288 -2.3%/yr**
- VIP-3 + maker: h=48 +0.3%/yr vs **h=288 +12.4%/yr (Sharpe 0.42)**
- VIP-9 maker: h=48 +37%/yr (Sharpe 1.15) vs h=288 +22%/yr (only place
  h=48 wins, via more cycles per year — note: returns are arithmetic
  per-cycle bps × cycles/year, not compounded)

Implementation: `alpha_v4_xs_1d.py` at HORIZON=288 with non-overlapping
sampling (rebalance every h bars). β-neutral OOS captures alpha cleanly
(ret_BN = +7.41 bps/cycle vs equal-weight ret +4.50, alpha +9.78). See
`docs/METHODOLOGY_REVIEW.md` Apr 30 follow-up for full fee-sensitivity
tables.

**Why this could work**: at h=288 (1d), the residual signal benefits from:
- Slower signal decay (per-trade alpha 30-50 bps vs 5-10)
- Fewer trades → lower amortized cost
- Different feature regime (1d momentum + reversal patterns)

**What's needed**:
- Change `HORIZON = 48` → `HORIZON = 288` across `alpha_v3.py` / `alpha_v4_xs.py`
- Re-audit features against the 1d alpha target (`alpha_feature_audit.py` with `HORIZON = 288`)
- Adjust cost model: cost is per-trade not per-day, so slow-trading helps
- Walk-forward fold sizes need to grow (50d train won't have many h=288 examples)

**Estimated effort**: 1-2 days. Mostly parameter changes and one round of
audits. Most code already supports it.

## Option B: Lower-fee venue / maker execution — STILL THE DEPLOYMENT LEVER

**Idea**: VIP-3 fee tier (~0.025% taker per leg, ~5 bps RT) and/or
post-only maker orders (~50% fill rate in calm regimes) close the remaining
cost gap.

**With Apr 30 corrected accounting at h=288 OOS β-neutral** (turnover-aware,
total turnover/cycle = 1.34):

| Tier | Fee/leg | Net/cycle | Net/year | Ann. Sharpe |
|---|---|---|---|---|
| VIP-0 retail | 12 bps | -8.69 | -32% | -1.09 |
| VIP-3 taker | 6 bps | -0.64 | -2.3% | -0.08 |
| VIP-3 + maker | 3 bps | +3.38 | +12.4% | +0.42 |
| VIP-9 maker | 1 bps | +6.07 | +22.1% | +0.76 |

Cost saving per tier ≠ per-leg fee saving — saving is `Δfee/leg × turnover_sum`
where turnover_sum is ~1.34 at h=288 (~0.83 at h=48). So a 6 bps/leg fee cut
saves ~8 bps net/cycle at h=288 (not 12).

**Open task**: a real maker-fill simulator (require L2 / Tardis data) to
refine the 50% fill assumption and check queue-position economics. Sharpe
0.42 at VIP-3+maker is marginal; better fill modelling could move it
either direction.

**Bootstrap CI caveat (Apr 30)**: at h=288 OOS, only 90 cycles of data.
Block-bootstrap 95% CI on Sharpe at VIP-3+maker = [-4.9, +4.0] — point
estimate not statistically distinguishable from zero. The 1d > 4h
relative ordering is robust; absolute deployment economics are not.
Require wider OOS sample before deployment.

**Why this works**: alpha is real, just below cost line. Cost reduction is
direct.

**What's needed**:
- Either a higher-volume venue / VIP tier (probably not realistic short-term)
- Or a post-only execution simulation: model fill probability vs queue position,
  measure effective fee given a maker-tilt strategy
- Maker simulation requires L2 orderbook data (not free; Binance Vision is L1)

**Estimated effort**: maker simulation is 1-2 weeks if including data
collection and modeling.

## Apr 30 — Plan-driven signal improvements (v4 → v6)

The "feature ceiling" suggested in earlier sections turned out to be partly
about how features were structured, not whether more existed. After running
a structured plan:

1. **xs_rank features** (per-bar pctile rank within universe): biggest single
   win, +1.4 Sharpe at deployment tier. Address scale heterogeneity across symbols.
2. **Top-K reduction**: K=5 → K=7 was a free +0.7 Sharpe — within-quintile
   rank carries genuine signal that gets diluted at K=5.
3. **Kline-flow features** (obv_z_1d, vwap_*, mfi): +0.3 Sharpe, modest.
4. **Universe trim by IS-IC bottom-quartile**: +0.3 Sharpe, mostly tightens CI.
5. **Funding-rate features**: regressed despite strongest single-feature IC.
   Signal already captured by v6's basket-relative features.

**Best config: v6 (32 features) + K=7 + β-neutral + IS-trim** →
OOS Sharpe +3.94 with 95% CI [+0.37, +6.74] at VIP-3+maker.
That's 9× over the corrected v4 baseline.

The remaining lever in this codebase is true microstructure data (Phase 3
in the signal-quality plan): pulling aggTrades for top-N symbols and
computing TFI / VPIN / Kyle's λ. This is genuinely orthogonal to klines
and is the next experiment.

Validation gap: 90 OOS cycles is the binding constraint on certainty. CI on
the best Sharpe spans [+0.37, +6.74] — point estimate is high but the
window is narrow. Forward testing on data past 2026-04-28 or pulling
additional history is the rigorous next step before deployment.

## Option C: Add orderbook L2 features

**Idea**: microprice, depth imbalance, queue position add 5-10 bps IC
contribution typical in microstructure research. Capture pending aggressor
pressure that price/volume miss.

**Why this could work**: orthogonal information source. The kline+aggTrade
universe is heavily arbed; L2 has lower mining ceiling.

**What's needed**:
- Tardis subscription or own L2 collection (8-12 GB / symbol / month)
- Storage scaling: 25 symbols × 12 months × 10 GB ≈ 3 TB
- New feature module `features_ml/orderbook.py`
- Re-run audits and v4 with orderbook features added

**Estimated effort**: 2-4 weeks counting data infrastructure.

## Other notes for the next person

### Where to start reading
1. `docs/METHODOLOGY_REVIEW.md` — full audit trail, numbered issues, fix attempts, results.
2. `ml/research/alpha_v3.py` — read for the curated 17-feature alpha-tailored model.
3. `ml/research/alpha_v4_xs.py` — read for the cross-sectional pipeline.
4. `features_ml/cross_sectional.py` — basket construction and basket-relative features.

### Look-ahead bugs to watch for
The codebase had two real look-ahead bugs found during this research:
1. **Sharpe target normalization shift** — `rolling.shift(1)` should be
   `rolling.shift(horizon)` because forward returns at horizon h require
   prices h bars ahead. Fixed in `_make_alpha_label` of all `alpha_v*.py`.
2. **VPIN bucket sizing** — used `total_vol.iloc[-1]` (full dataset) to
   size buckets; now uses trailing 7d window per bar. Fixed in
   `features_ml/trade_flow.py::_vpin`.

When adding new features, sanity-check by computing IC on `fwd_ret` *one bar
shifted* — features that have suspicious +0.10 IC vs forward return often
have a hidden lookback that uses bar-t close in computing bar-t feature.

### Fold purging is non-trivial
`ml/cv.py::FoldSpec` + `split_features_by_fold` handles standard cases, but
the alpha-residual labels also have an `exit_time` column that must be
purged from train. The `_expanding_train` helpers in each `alpha_v*.py`
script handle this — copy the pattern, don't reinvent.

### Per-symbol idiosyncrasies
- BTC alpha (vs ETH ref) has highest OOS IC (+0.08 in v3) but worst trade
  P&L because un-hedged BTC eats the alpha via market direction noise.
- ETH alpha (vs BTC ref) is the only consistently profitable single-pair
  setup, but only naked at q=0.99 OOS (+8.49 bps net) — and that's regime-
  dependent, not robust.
- SOL alpha is broken at this horizon. Don't waste time trying to make
  SOL work alone.

### Don't get tricked by walk-forward
WF results are 8-12 bps higher than OOS holdout. The gap is partly
hyperparameter overfitting (q, h, regime cutoff chosen by inspecting all
WF folds), partly genuine distribution shift. Always verify on the 90d
holdout before drawing conclusions.

## Open questions

- Does longer horizon (1d/1w) actually deliver 30+ bps alpha on these symbols?
  (Untested; is Option A above.)
- Does adding funding-rate features help? (Free public data, untried.
  Funding extremes are documented mean-reversion signal in perp basis.)
- Does maker-tilt simulation produce realistic fill rates? (Untested.
  Would require L2 data.)

## Contact

Original research and methodology by yq during the P-2026-001 program.
Code is research-grade; production hardening (live trading integration,
risk limits, monitoring) is out of scope.
