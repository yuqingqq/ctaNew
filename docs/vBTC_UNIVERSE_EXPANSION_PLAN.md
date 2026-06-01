# vBTC universe expansion plan

Last updated: 2026-05-13

---

## 2026-05-13 update — Formal universe-construction methodology (proposed)

After the Phase UNI-111 / Phase 1 fixed-reference test results (see `vBTC_STRATEGY_STATUS.md` § Universe-portability investigation), the original "expand to all eligible Binance USDM perp symbols" expansion approach is closed (catastrophic V3.1 collapse on the 111-panel — see Phase E5b). The replacement approach is: **define a principled, point-in-time universe from the start, rather than treating the current 51 as fixed and asking whether expansion helps**.

The current 51-symbol set was a casual selection. This doc now proposes three formal methodologies for a principled replacement. All run deterministic algorithms at each refresh date so the universe is replicable by any researcher applying the same rule.

### Method 1 — Liquidity-rank with PIT filters (recommended)

**Metric**: at refresh time T, score each candidate symbol by
$$Q_1(s, T) = \log_{10}(\text{ADV}_{90d}(s, T))$$

**Hard PIT filters** (must all pass to be a candidate):
1. Listed on Binance USDM perp ≥ 365 days before T (full lookback window valid; eliminates lucky-window meme picks)
2. Trailing 90d median quote-volume ≥ $30M (capacity floor)
3. Listed on Hyperliquid at T (execution venue alignment)
4. Not stable / wrapped / synthetic / tokenized-equity
5. No 30+ day gap in trailing 90d (avoid delisted-then-relisted)

**Universe**: rank survivors by $Q_1$, take **top 30**. Refresh **annually on Jan 1**.

**Parameter derivation (first principles, locked before run):**
- **N = 30**: strategy structure is K=3 per side from top-15 by IC. For S/N at rank-15/16 to exceed 1 (vs current 51-panel S/N=0.82), need ~25-35 candidates. Choose middle of range.
- **ADV ≥ $30M**: K=3 sizing at 1% of ADV per leg → $300k per leg → ~$5M strategy AUM capacity at this floor.
- **History ≥ 365d**: required for 180d trailing IC window to operate on full data.
- **Annual refresh**: balances composition stability vs adaptation. Quarterly OK; monthly over-fits.

**Pros**: trivially defensible, zero discretion after parameter derivation, replicable.
**Cons**: top-heavy by ADV — likely BTC/ETH/SOL/BNB at top, no diversity-aware structure.

### Method 2 — Liquidity + cluster diversification

Same PIT filters and $Q_1$ as Method 1, plus:
1. Cluster eligible symbols by 90d 4h-return correlation, hierarchical Ward, **K = 6 clusters**
2. Within each cluster, rank by $Q_1$ and take top 5
3. If a cluster has fewer than 5 eligible names, fill from next-largest

**Parameter derivation:**
- **K = 6**: Phase G already established K=6 as natural for 111-panel (separation +0.216).
- **5 per cluster**: 30 / 6.

**Pros**: prevents top-30 from being dominated by one beta family.
**Cons**: cluster K is one more degree of freedom; Phase F/G showed clusterings don't work as alpha features, but using only for universe construction is a softer use.

### Method 3 — Greedy max-diversification

Start with highest-$Q_1$ symbol. Iteratively add the symbol $s$ that maximizes
$$\Delta_s = Q_1(s) - \lambda \cdot \max_{u \in U} \rho_{s,u}$$
where $\rho_{s,u}$ is 90d return correlation. Continue until 30 names.

**Parameter derivation**: $\lambda$ set so 50% of score weight comes from liquidity and 50% from diversification → $\lambda = \text{stdev}(Q_1) / \text{stdev}(\rho)$ on the full eligible set at T.

**Pros**: theoretically optimal under a stated objective (max diversification at given liquidity).
**Cons**: $\lambda$ choice is implicit subjectivity; greedy not globally optimal.

### Recommended choice

**Method 1** unless Method 1 produces a top-30 that's so concentrated (e.g., 10 BTC-followers) that a researcher would reject it. Measure first, decide second.

### Test plan after universe lock

1. Build universe at each refresh date PIT (2025-01-01 and 2026-01-01 for current backtest period)
2. Compare to current 51 (overlap, drops, adds)
3. Rebuild panel (same WINNER_21 features, target_A computed against the new basket — same construction as production, different basket)
4. Train + run V3.1 with IC ranking
5. Decision branch:
   - Sharpe ≥ +2.0 with drop-5 std < 0.5 → **principled universe alone fixes most of the overfit**. The current "universe-overfit" was actually "bad-universe-overfit".
   - Sharpe +1.0-+1.5 → principled universe helps but still needs pipeline fixes (Phase 1 / xs_rank / sym_id) for full portability.
   - Sharpe < +1.0 → bad sign: the strategy is structurally fragile to any universe change. Freeze-and-monitor becomes the only viable path.

---

## Original 2026-05-11 plan (preserved below — Phase E1-E5 executed and CLOSED, see Phase E5b in vBTC_STRATEGY_STATUS.md)

## Goal

Test the hypothesis that expanding the 51-symbol panel to the full Binance USDM
perpetual universe (with PIT-eligibility filters) lets the refill mechanism find
higher-quality candidates outside the current panel.

## Hard eligibility criteria (applied PIT at each decision point)

A symbol enters the candidate panel at time `t` only if all of:

| # | Criterion | Threshold | PIT-safe? |
|---|---|---|---|
| 1 | Listed on Binance USDM perpetual | yes | yes |
| 2 | First kline data exists by `t - 120d` | ≥ 120d history | yes |
| 3 | Trailing 30-day median daily quote volume | ≥ $50M USD | yes (computed from past 30d klines) |
| 4 | Funding rate history exists for ≥ 30 days | yes | yes |
| 5 | No multi-day data gap in trailing 90d | < 5% missing 5-min bars | yes |
| 6 | Symbol is currently trading (not delisted) | yes | yes |

Notes:
- Criterion #2 (120-day history) is stricter than the model's PIT min_history_days=60
  because we want enough data to compute trailing-30d volume and rolling 90-day IC
- Criterion #3 ($50M daily volume) is a soft default; sweep later if needed
- These are applied at the SYMBOL ENTRY level. At each backtest decision, the
  panel includes only symbols passing all criteria as of that point in time

## Execution plan (4 phases)

### Phase E1: Discovery + sourcing (~4-6h)
- Enumerate all Binance USDM perpetual symbols from `data.binance.vision`
- For each symbol not currently in our panel:
  - Download 5-min klines (daily partitions, 2025-03-27 onward to match panel start)
  - Download funding rate history
  - Save to `data/ml/test/parquet/klines/{SYMBOL}/5m/` (existing schema)
- Output: list of candidate symbols with metadata (listing date, current volume)

### Phase E2: PIT eligibility audit (~2h)
- For each candidate symbol, compute the date it first passes all criteria
- Build a PIT eligibility table: `symbol → first_eligible_date`
- Compare to existing 51-panel — confirm existing names remain eligible
- Identify ~50-100 new candidates that pass criteria at some point in OOS window

### Phase E3: Feature pipeline rebuild (~6-8h)
- Build expanded feature panel with all eligible symbols
- Recompute basket-relative features (`bk_ema_slope_4h`, `dom_level_vs_bk`,
  `dom_change_288b_vs_bk`, `corr_change_3d_vs_bk`, `idio_vol_1d_vs_bk_xs_rank`,
  `idio_ret_48b_vs_bk`) with the EXPANDED basket
- Recompute per-symbol features for new tokens (return_1d, atr_pct, etc.)
- Verify reconstruction: existing 51 symbols' features should change only because
  basket changed (sanity check correlation with old features)
- Output: `outputs/vBTC_features_expanded/panel_variants_with_funding.parquet`

### Phase E4: Retrain + evaluate (~1-2h)
- Retrain LGBM 5-seed ensemble with expanded panel (new `sym_id` cardinality)
- Same train/cal/test split protocol (`_multi_oos_splits`)
- Build new audit panel (`alpha_vBTC_build_audit_panel.py` adapted)
- Run Phase 2b v3 protocol:
  - PIT rolling-IC top-15 universe from expanded pool
  - K=4 long/short with filter_refill_90d_mean
  - 100-seed matched placebo
- Compare to current 51-panel results:
  - Sharpe / CI / max DD / total PnL
  - Does the top-15 universe include any non-51 tokens? Which ones?
  - Does the lift survive matched placebo?

## Pass conditions

The expansion is a meaningful improvement if BOTH:
1. **Sharpe lift**: > +0.3 over current 51-panel best (+1.16 → ≥ +1.46)
2. **Beats matched placebo p95**: real variant placebo rank > 95%

If only (1) without (2): expansion produces a backtest-better number but the
mechanism is still random-equivalent exposure restructuring. Doesn't change the
core no-signal conclusion.

If neither: expansion confirms current 51-panel is near-optimal at this model
strength, and the bottleneck is per-name signal quality (features/horizon), not
universe size.

## Risks and known unknowns

- **Survivorship bias**: if we filter to "currently trading" symbols, we exclude
  delisted ones. For our 2025-2026 window this is probably small but worth
  noting.
- **Volume threshold sensitivity**: $50M is a reasonable default but could be
  swept. Lower threshold (e.g., $10M) admits more small-cap tokens; higher
  ($100M) keeps universe closer to current.
- **Feature recomputation cost**: basket-relative features touch all rows; this
  is the slow step. Will use chunked processing.
- **sym_id cardinality**: if we go from 51 to ~150 sym_ids, the categorical
  feature explodes. LGBM handles this fine, but it changes the model's effective
  capacity slightly.

## Out of scope (intentional)

- Hyperliquid-native data sourcing. We test on Binance USDM since the existing
  panel uses Binance Vision archives. HL-native expansion is a separate followup.
- Spot-perp basis features. Could add later if expansion shows promise.
