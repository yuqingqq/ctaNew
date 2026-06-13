# Research Roadmap — Phase 2 (data fixes + deeper optimization)

**Created:** 2026-05-20 (after Phase 1 matrix complete)

Phase 1 (X6–X14d) shipped a 45-cell apples-to-apples matrix on HL-50.
Phase 2 addresses three areas the user flagged:

1. **Fix data quality** — aggT 26 syms missing, crossX coverage gaps
2. **Optimize model params** — beyond α sweeps, more hyperparam scope
3. **More universe tests** — beyond 4 universes already tested

---

## 1. Data quality fixes

### aggT data: 26 of 51 syms have ZERO coverage

The panel's aggT features (`aggr_ratio_4h`, `tfi_4h`, `signed_volume_4h`,
`buy_count_4h`, `avg_trade_size_4h`) are computed from raw Binance Vision
aggTrades data. We have aggTrades for 25 syms (ADAUSDT, APTUSDT, ARBUSDT,
ATOMUSDT, AVAXUSDT, BCHUSDT, BNBUSDT, BTCUSDT, DOGEUSDT, DOTUSDT, ETHUSDT,
FILUSDT, INJUSDT, LINKUSDT, LTCUSDT, NEARUSDT, OPUSDT, RUNEUSDT, SEIUSDT,
SOLUSDT, SUIUSDT, TIAUSDT, UNIUSDT, WLDUSDT, XRPUSDT) but NOT for 26 syms
(AAVE, ASTER, AXS, BIO, ETC, ENA, GMX, ONDO, JUP, LDO, ORDI, JTO, HYPE,
ICP, HBAR, PENDLE, TAO, TON, TRB, VIRTUAL, plus 6 others).

**Decision tree:**

- **If X16 subset (25 syms with aggT) gives big lift** → download missing 26 syms (~30-50GB, 2-4 hours)
- **If X16 subset doesn't help** → coverage isn't the bottleneck; investigate model issue

**Status:** X16 running.

### crossX data: partial coverage (53-65%)

- 6-8 syms missing from OKX or Coinbase entirely (data doesn't exist)
- ~5% NaN from trailing-30d z warmup
- Forward-fill to 5m proven HARMFUL (X14b/X14d): injects auto-correlation, hurts Pool+symid Ridge

**Possible improvements:**

- **A. Collect more OKX/CB historical data** (extend before 2025-04 to give 30d warmup further back)
- **B. Per-sym coverage flag feature**: add binary `has_crossX` so model learns when to trust
- **C. Sparse Ridge with sample weighting**: weight rows by crossX coverage
- **D. Subset universe to syms with full coverage**

X16 F2 tests D. X18 (planned) tests B.

---

## 2. Model parameter optimization beyond α sweep

### Already tested
- Ridge α grid {0.01-100} (X6 baseline), {0.001-300} (X8 wider, X10 C1)
- ElasticNet L1 mix (X8b)
- LGBM regularization (X8e: early stop, adaptive leaf, higher reg, combined) — **all hurt**
- C1 normalized sym_id dummies (X8c, X10)

### Remaining to test (X19 planned)
- **Feature preprocessing variants**:
  - Standard winsor p1/p99 + z (current default)
  - Rank-transform (full per-bar cross-sectional rank)
  - Robust scaling (median / MAD)
  - Quantile transform to Gaussian
- **Fold scheme variants**:
  - Embargo length (currently 1 day; test 3, 7 days)
  - Sliding window (vs current expanding)
  - More folds (12 vs current 9)
- **Target horizon variants**:
  - h=48 (current 4h)
  - h=24 (2h)
  - h=96 (8h)
  - h=288 (24h)
- **Per-sym minimum row threshold** (currently 300; test 100, 1000, 3000)

### Quick wins likely:
- Rank-transform preprocessing may help Ridge with heavy tails
- Per-sym 1000-row threshold may filter out under-sampled syms that add noise

---

## 3. More universe tests beyond X11

### Already tested (X11)
- HL-50 baseline (+2.01)
- HL-50 minus top-10-vol (+0.61)
- $5M+ HL vol filter, 29 syms (+0.92)
- 51-panel WITH BTC (+0.00 collapse)

### Remaining (X20 planned)
- **N-stress series**: HL-30, HL-40, HL-60, HL-70, HL-100 (sweep universe size)
- **Symbol-tier subsets**: top-10-vol only, top-25-vol only, bottom-25-vol only
- **Category-based**: L1 only, DeFi only, memecoins excluded
- **Coverage-based**: syms with 100% aggT+crossX coverage (intersection ~20 syms)
- **Time-window stability**: train on first 6 months, test on last 7 months (single OOS)

### Key questions to answer
- Is +2.01 a universe artifact or does similar performance hold at HL-30 / HL-40?
- Do top-vol syms ALONE carry the alpha, or does diversity matter?
- How fast does performance degrade as universe shrinks below 40?

---

## Execution order (Phase 2)

1. ✅ **X15 diagnostic** — investigate why crossX/aggT work in Per-sym not Pool+symid (done via inline bash)
2. 🔄 **X16 subset-universe test** — decides if aggT data collection worth it
3. **X17 aggT download** — if X16 confirms helpful, download missing 26 syms from Binance Vision
4. **X18 has_crossX coverage flag** — coverage-indicator feature interaction
5. **X19 hyperparam sweep** — preprocessing × fold scheme × target horizon
6. **X20 N-stress + tier subsets** — universe variants

Each step follows the discipline: review → diagnose → update docs → dispatch next.

---

## Memory discipline (all jobs)

- Detach via `nohup`-style background launch (PPID=1)
- float32 panels + sparse matrices where possible
- Explicit `del + gc.collect()` between variants
- `log_mem()` before/after each variant
- Sequential execution (no parallel jobs > 12GB)
