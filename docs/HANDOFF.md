# Handoff — 2026-04-30

## Summary for the next person

The signal class (4h-horizon alpha-residual prediction from kline + aggTrade
features, traded as long-short cross-sectional or pair-trading) is **fully
characterized**:

- Real, statistically robust (rank IC +0.035 OOS)
- Magnitude ~5–10 bps gross per trade
- Below retail VIP-0 round-trip cost (12 bps naked / 24 bps hedged)

**Don't try to push this signal harder via more features or models.** The
audits in `ml/research/alpha_*_audit.py` already enumerated what's available.
Marginal improvements (more features, deeper trees, ensemble tricks) have
diminishing returns and have been tried.

The path to a deployable strategy is one of three structural pivots, ranked
by ROI to attempt:

## Option A: Different horizon (highest expected ROI from this codebase)

**Idea**: residual mean-reversion at 1-day or 1-week horizon is documented
at 30-50 bps in academic literature (Lou & Polk 2014, several others).
4h is too noisy / too efficient. Move the prediction horizon up.

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

## Option B: Lower-fee venue / maker execution

**Idea**: VIP-3 fee tier (~0.025% taker per leg, ~5 bps RT hedged) would
flip the strategy from -16 bps to +3-5 bps net OOS. Or, use post-only maker
orders (~50% fill rate in calm regimes) to roughly halve effective fees.

**Why this works**: alpha is real, just below cost line. Cost reduction is
direct.

**What's needed**:
- Either a higher-volume venue / VIP tier (probably not realistic short-term)
- Or a post-only execution simulation: model fill probability vs queue position,
  measure effective fee given a maker-tilt strategy
- Maker simulation requires L2 orderbook data (not free; Binance Vision is L1)

**Estimated effort**: maker simulation is 1-2 weeks if including data
collection and modeling.

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
