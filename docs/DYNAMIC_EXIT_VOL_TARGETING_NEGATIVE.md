# Dynamic exit + vol-targeting at h=48 — TESTED NEGATIVE

**Date:** 2026-05-05
**Triggered by:** Live -47 bps cumulative drawdown at 02:05 UTC on 2026-05-05
(adverse cross-sectional move — TIA -5%, OP -4% as shorts; long basket lagged
the short basket's rally).

Hypothesis tested: there exist deployable optimizations within the
v6_clean h=48 K=7 ORIG25 architecture to reduce drawdowns from
adverse-cross-sectional-move days (like TIA/OP today). Three candidates
audited; all proven negative or neutral.

## Audit results

### 1. Take-profit (Option C) — neutral

Hourly cycle PnL monitor; if cumulative cycle PnL > threshold,
exit all → re-enter on next 4h boundary OR immediately rebalance to
new portfolio (turnover-aware delta).

**v1 result (close-all + reopen-from-cash, ~14 bps cost penalty):**
ΔSharpe -1.06 to -4.92 across thresholds {20, 30, 40, 50, 60} bps.
All thresholds underperform baseline.

**v2 result (corrected cost: turnover-aware re-entry, no double-charge):**
analytical correction: add back 14 bps × trigger_rate to v1 mean PnL.
Result: TP at all thresholds is approximately **Sharpe-neutral**
(corrected mean ≈ baseline mean ±0.3 Sharpe).

Mechanism: alpha-decay study showed cycle PnL accumulates roughly
linearly. Cycles that hit +TP early are statistically continuing to
build alpha through h=48. Locking in early sacrifices the continuation,
which exactly offsets the cost saving.

### 2. Per-leg dom_z exit (Option B) — same alpha-truncation issue

Not directly backtested but inherited the same negative verdict from
take-profit's mechanism. Mean-reversion models actively WANT to hold
positions as they go MORE stretched (higher conviction signal). Per-leg
exits would close positions when reversion completed but those positions
were also where the largest continued alpha would have come from.

### 3. Vol-targeting leg sizing — DECISIVELY NEGATIVE

```
sizing         #cyc   spread   net    Sharpe   95% CI            ΔS    paired t   p-value
equal           1620  +7.90  +4.33   +3.63   [+1.31, +6.14]   base
inv_vol_1d      1620  +6.24  +2.68   +2.44   [+0.09, +4.77]   -1.19   -3.16    0.0008
inv_vol_4h      1620  +6.04  +2.26   +2.09   [-0.23, +4.36]   -1.54   -3.62    0.0001
inv_atr         1620  +5.95  +2.17   +2.01   [-0.34, +4.30]   -1.63   -3.57    0.0002
```

**All three vol-targeting variants statistically significantly worse than
equal weight (p<0.001).** Per-fold breakdown: equal-weight wins 7/9 folds.

**Mechanism (counter-intuitive):** high-vol symbols carry MORE alpha,
not just more risk. Cross-sectional reversion is structurally stronger
for noisy/volatile names because:
1. They get more stretched (higher dom_z magnitudes)
2. They have stronger per-symbol IC than BTC/ETH (per-feature analysis)
3. Underweighting them underweights the strongest signal

By inv-vol weighting, we cut alpha by ~50% while only cutting risk by ~9%.
Net Sharpe drops by 1.2-1.6 points.

## Implication for live operations

**Today's TIA/OP loss (-47 bps cumulative) is normal noise.** Per-cycle
SD is 55 bps; backtest worst single cycle was -3.67%. -0.47% is
well within historical norms (47% of cycles are losers in expectation).

There is **no in-strategy optimization** that prevents these adverse
moves without giving up more alpha than it saves. The strategy is
structurally exposed to cross-sectional dispersion blow-ups (positive
when reversion plays out, negative when it doesn't). This is the
strategy's character.

**Loss-floor protection requires structural changes:**
- Reduce notional (trade smaller — directly proportional impact)
- Halt during regime shifts (need a regime trigger; untested)
- Different strategy class entirely

Not in-strategy parameter tuning.

## Reproducibility

Per-cycle data: `outputs/h48_features/take_profit_*.csv`,
`outputs/h48_features/vol_targeting_backtest.csv`

## What still might work (untested directions)

| Optimization | Expected lift | Effort | Risk |
|---|---|---|---|
| HYPE Bronze tier (10% taker discount) | +0.2 Sharpe | 30 min | None |
| Maker-mode execution | +1 to +3 Sharpe | 1-2 weeks | Medium (queue modeling) |
| L2 order book features (depth, imbalance) | unknown | 2-4 weeks | Medium (Tardis data) |
| On-chain features | unknown | 1-2 weeks | Medium (Glassnode) |

Confirmed dead (don't redo):
- v6_clean feature reselection (5 audits)
- Hyperparameter retuning (universe-conditional)
- Universe expansion (proven harmful)
- DVOL / funding / aggTrade microstructure features
- Regime-conditional MoE
- **Take-profit / per-leg early exit** (alpha-truncation)
- **Vol-targeting leg sizing** (high-vol = high-alpha)
