# OB Aggregate-Crowding Market-Timing Signal

**Status:** research lead — honestly characterized, **not deployed**. Recent-era persistence is
statistically unconfirmable on available data; the decisive next step is a live paper-forward test.

A defensive market-timing overlay derived from the free Binance USDM `bookDepth` archive. It fades
**aggregate order-book crowding**: when the whole market's resting book leans one-sided (everyone
crowded the same way), it bets on a reversal. Separate from the convexity cross-sectional book — this
is a single directional bet on the market factor, not a beta-neutral alpha book.

## Where it came from

Exhaustive testing established the free order book carries **no incremental cross-sectional alpha**
(6 axes tested, all dead at the incremental gate — see `../MEMORY`/`project_l2_bookdepth`). The one
durable signal is at the **aggregate / market-timing** level, and only there.

## The signal (exact)

Every 4h bar (daily is fine at this horizon), with capital `C`:

```
1. SIGNAL     agg_imb = mean over ~175 coins of  (bidN±1% − askN±1%)/(bidN±1% + askN±1%)
2. NORMALIZE  z       = (agg_imb − mean_90) / std_90          # trailing 90 bars = 15 days
3. RAW        raw     = −clip(z, −1.5, +1.5)                  # CONTRARIAN, strength-scaled, capped
4. SMOOTH     frac    = EWMA(raw, halflife = 42 bars = 7d)    # slow; ~1.6%/bar toward raw
5. TARGET     target_$ = frac × C
6. TRADE      order_$  = target_$ − current_$                 # rebalance in BTC perp or a basket
```

`z > 0` = book bid-heavy (crowd long) → **short**. `z < 0` = ask-heavy (capitulation) → **long**.
Enter / add / reduce / exit / flip are all the single step 6 producing different-signed orders.

Key design points (each empirically forced):
- **±1% depth level** — strongest edge; near-touch (±0.2%) is HFT noise, deep (±5%) is stale.
- **Strength SCALES the position** (not a filter) — the edge lives in the extreme imbalances.
- **~7-day horizon** — crowding unwinds over 1–2 weeks; longer holding gave lower turnover/DD and
  higher recent Sharpe, and survives trend-control (genuine OB, not market mean-reversion).
- **No stop-loss** — the ±1.5 cap is the risk limit; a stop would exit the pre-crash short before it pays.

## Trading flow (C = $100k, real 2025-10 episode)

Typical exposure is **light**: avg |position| ≈ 0.25·C, peaks ~0.6–0.9·C, never hits the 1.5·C cap at
the 7d horizon; ~51% short / 38% long / 12% flat; rebalance ≈ 0.05·C/day.

```
ENTER + ADD  book crowds bid-heavy → sell a few $k/day, short builds to −$58k (losing as market rallies)
HOLD         hold through the top (no stop)
PAYOFF       2025-10-10 market −25.9% → −$59k short earns +$14.8k in a day
REDUCE       book flips ask-heavy → buy back gradually ($16–20k/day)
EXIT/FLIP    target crosses 0 → cover last of short, flip slightly long on the capitulation
→ window: market −21%, strategy +$8.5k on $100k
```

## Performance (net 10bps, honest)

| horizon | OOS Sharpe | RECENT Sharpe | maxDD (OOS/REC) | turnover/day |
|---|---|---|---|---|
| 2d (initial) | +0.61 | +0.52 | −38% / −19% | 0.23 |
| **7–10d (spec)** | **+0.6** | **+1.1** (fragile) | **−18% / −8%** | **0.05** |

Positive both eras at every horizon 1d→10d (robust, not knife-edge). It is a **defensive** profile:
positive on down-days, slightly negative on up-days, ~half the market's vol; in the OOS bull, buy-hold's
risk-adjusted return *beats* it (it gives up upside). Value = crisis-hedge / gross-modulator on a
long book, **not** standalone alpha.

## What it catches / what it doesn't

- **Catches** crowding-driven (froth→dump) crashes: worst-decile market days avg **+1.0%/day** vs market
  −7.3%, positive on 61%. Spectacular when short going in (2025-10-10 −26% → +21%).
- **Misses** ~40% of extreme days — exogenous shocks on a non-crowded book — and can get caught long
  (2025-04-06: ask-heavy → long → −11%). The contrarian rule is structurally wrong when the imbalance is
  *informative* (news/continuation) rather than *crowding* (reversion); the coarse aggregate book can't
  tell them apart, and strength/price/depth/alignment filters do **not** robustly separate them.

## Honest caveats

- **Fading** over time (sub-period IC −0.23 in 2023 → −0.03 mid-2025).
- **Recent-era unconfirmable** — single market series, few independent multi-week windows; recent CIs
  cross zero at every horizon (the +1.1 Sharpe is ~27 independent 10d windows). OOS-confirmed only.
- **Separate strategy** — directional market timing, must transport through Binance-train / HL-execute
  on its own merits; not a convexity enhancement.
- Refinements that **failed** both-era testing (do not re-add): strength-*filter*, price-gate, depth
  structure, cross-level alignment. The robust signal is exactly the 6 steps above.

## Files

| file | role |
|---|---|
| `bookdepth_loader.py` | fetch `bookDepth` → 4h PIT imbalance features (imb02/1/2/3/5, liq1, …) |
| `bookdepth_market_timing.py` | `agg_ob()` — aggregate the market book-lean; base timing test |
| `bookdepth_timing_backtest.py` | hardened backtest (signal, cost, DD, regime) + chart |
| `bookdepth_timing_walkthrough.py` | day-by-day episode walkthrough |
| `bookdepth_crowding_{horizon,nonoverlap,trendcontrol}.py` | signal validation (horizon, overlap, trend controls) |
| `bookdepth_crowding_tradeability.py` | episode-stability + tradeability diagnostics |
| `bookdepth_timing_{events,strength,depth_price,alignment,combined,horizons,horizon_trendctl}.py` | refinement tests (what survived / what didn't) |

**Next step:** stand up a paper-forward runner (the 6 steps above, logging `target_$`/`order_$`/P&L daily)
— the only thing that resolves recent-era persistence.
