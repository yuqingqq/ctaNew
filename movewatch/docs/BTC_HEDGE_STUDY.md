# Can BTC hedge the post-alert fade? (declared before results, 2026-10-02)

**Question.** After a replay NEW alert, does fading the coin with a BTC hedge capture the coin's
move relative to the market, net of costs and funding? The fade is short the pumped coin and
long β·BTC; for dumps it is long the coin and short β·BTC. The earlier studies measured the fade
against the median liquid coin, which is not tradable. BTC is.

**Status of the data: consumed.**
- Apr 2–Jul 28 (holdout) and Aug 2–Sep 24 (discovery) were both read by the persistence study.
- The typical pump fade relative to the market is already known from them: median −7.5% /
  −8.7% at 72 h.
- So this study is **exploratory**. A passing result becomes a **candidate**, validated only as
  an amendment to `docs/FORWARD_TEST_v2.md` on forward data. Nothing is adopted from it.

## Events
- Rows of `data/replay/persistence/features_{holdout,discovery}.parquet`: replay NEW alerts with
  stocks excluded and one alert per coin per 72 h. Rows are actions.
- BTCUSDT events get status `SELF` (BTC cannot hedge itself).

## Trade
- **Entry and exits:**
  - Entry: alert minute t + 1, the next 1-min close (the live `FU_ENTRY` is +60 s).
  - Exits: entry + 240 / 1,440 / 4,320 min (`0_4`, `0_24`, `0_72`).
  - `4_72`: enter at t + 241, exit at entry + 4,320. This is the +4 h check's timing.
- **Variants** (P&L per unit of coin notional, simple returns):
  - `U`: coin leg only.
  - `H1`: coin + 1.0 × BTC.
  - `Hb`: coin + β̂ × BTC. β̂ is the OLS beta of the coin's 5-min log returns on BTC's over
    minutes [t − 1,440, t − 60), the live `beta_btc` definition (`movewatch.followup.beta_to`),
    clipped to [0, 3].
  - `M` (reference, gross only, not tradable): coin minus the median liquid coin
    (`research.common.mkt_median`).
- **Costs per side** (Binance USDM VIP-0):
  - Taker fee 5 bp on each leg.
  - Coin slippage by pre-alert 24 h volume: ≥ $100M 5 bp; $20–100M 10 bp; $5–20M 20 bp;
    < $5M 40 bp.
  - BTC slippage 1 bp.
  - A ×2 slippage sensitivity is reported.
- **Funding:** actual Binance funding (Vision monthly `fundingRate` archives) for each
  settlement in (entry, exit], on entry notional. Long pays when the rate is positive.
  Missing archive → status `NO_FUNDING`.
- **Common sample:** all variants of a population/horizon use the same rows (valid prices for
  coin and BTC, no frozen bars, β̂ defined, funding present). Each exclusion is a counted status.

## Null and statistics
- **Random baseline** (rule 9, removes "alts drift against BTC"):
  - 2,000 random (coin, minute) draws per window.
  - Coins with ≥ $5M/day at the minute, stocks and BTC excluded.
  - Same trade, fade side = short the coin (as for pumps), same costs and funding.
- **Per (window, population, horizon, variant):** n, mean net % with a 95% CI (ISO-week block
  bootstrap, 2,000 draws), median, hit rate, std, 5th percentile.
- **Hedge effect:** var(`Hb` gross) / var(`U` gross), and the coin–BTC correlation over the hold.
- **Populations:**
  - Primary: pumps.
  - Also: dumps; pumps by breadth (the live cutoffs 0.0051 / 0.0468); `4_72` pumps by
    post-alert volume (quiet < 0.94×, heavy > 2.26×, `vol_persist`).
- **Candidate predicate** (computed in the script, never printed by hand), for
  `Hb`, pumps, at one horizon:
  - mean net > 0 with its CI above 0, in **both** windows; and
  - mean net − random mean > 0 with the difference's CI above 0 (independent week-block
    bootstraps), in **both** windows.
  - Four horizons are examined; requiring both windows is the replication bar.
- Memory: default slot cap (4 GiB AS), `MALLOC_ARENA_MAX=2`, abort above 2.0 GiB RSS; panels
  are disk memmaps.

## Corrections before the first result (2026-10-02, committed before the run)
No hedge statistic had been computed when these were made. The first run stopped at the
reconciliation self-test.
1. **Trigger minute.** The alert `time` stamps the close of trigger minute t, so
   t = (time − panel start) − 1. With it, the recomputed 0→72 h market-adjusted return equals
   the persistence study's `x72h` exactly (spot check on 4 events; the full selftest runs on
   every row and aborts above 1% mismatch). The first run, with t off by one, mismatched
   2,729 / 2,729.
2. **Stocks.** The persistence study's filter was a 37-ticker static list, and ON Semiconductor
   (ONUSDT) was among its events.
   - Events and random draws now also drop any symbol whose base a venue labels as a stock or
     commodity (`movewatch.tradfi.fetch_reference`, snapshot `data/replay/btc_hedge/tradfi_ref.json`)
     and whose price is within 2× of that label. The windows are months old, while a same-ticker
     crypto sits orders of magnitude away.
   - Status `TRADFI_PRICE`, counted.

## Results (2026-10-02; script at 5ddbb87; `data/replay/btc_hedge/results.json`, `events.parquet`)

**Run quality:**
- Reconciliation vs the persistence study's `x72h`: 2,610 rows, 0 mismatches.
- Statuses:
  - `TRADFI_PRICE` 39 holdout / 80 discovery. One is a false positive: a holdout QNT event at
    $63.78 is Quant, caught by the 2× rule against OKX's QNT stock at $48.40. It is 1 row of
    1,029 and was not rerun.
  - `SELF` 1; `NO_PRICE_OR_STALE` ≤ 18; `NO_BETA` ≤ 1; `NO_FUNDING` 0.
- Peak RSS 0.72 GiB.
- Note: 80 of 967 events in the Aug–Sep window were stock or pre-IPO perps that the persistence
  study's static filter missed.

**Pumps, BTC-hedged fade (`Hb`, net): mean [95% CI], median, win rate, 5th percentile.**

| window | horizon | n | Hb net mean | median | win | p5 | random Hb | M gross (untradable) |
|---|---|---:|---|---:|---:|---:|---|---|
| Apr–Jul | 0→4 h | 1,039 | −0.59% [−1.51, +0.36] | +2.04% | 60% | −23.0% | −0.59% [−0.78, −0.40] | −0.00% |
| Apr–Jul | 0→24 h | 1,036 | −0.41% [−1.92, +1.07] | +4.69% | 64% | −41.0% | −0.26% | +0.17% |
| Apr–Jul | 0→72 h | 1,029 | +0.61% [−2.58, +3.29] | +7.86% | 68% | −50.9% | −0.26% [−1.32, +0.67] | +1.22% |
| Apr–Jul | 4→72 h | 1,028 | +0.52% [−1.95, +2.83] | +5.06% | 64% | −46.1% | −0.19% | +1.03% |
| Aug–Sep | 0→4 h | 458 | +0.91% [−0.42, +2.22] | +2.54% | 61% | −18.7% | −0.64% [−0.85, −0.40] | +1.52% |
| Aug–Sep | 0→24 h | 458 | +1.22% [−2.12, +4.52] | +6.01% | 69% | −34.4% | −1.17% | +1.92% |
| Aug–Sep | 0→72 h | 455 | −0.68% [−4.24, +3.00] | +6.49% | 68% | −53.4% | −1.79% [−4.04, +0.43] | +0.87% |
| Aug–Sep | 4→72 h | 453 | −2.99% [−5.87, +0.19] | +3.46% | 60% | −52.2% | −1.84% | −1.65% |

**Candidate predicate (computed): false at all four horizons.**

**What it says:**
- **BTC does not hedge these trades.**
  - Coin–BTC return correlation over the hold is 0.00–0.09.
  - var(`Hb` gross) / var(`U` gross) is 0.97–1.00, so the hedge removes ≤ 3% of the variance.
  - The median pre-alert beta is 0.60 / 0.72.
  - After an alert the coin moves on its own; the BTC leg adds cost, not protection.
- **The fade wins often and small, and loses rarely and big.**
  - Median +2% (4 h) to +8% (72 h), with 60–69% winners.
  - 7–13% of trades lose more than 20%, and the 5th percentile is −19% to −53%.
  - The mean is about zero.
- **Costs + funding take what is left.**
  - Mean drag per trade: 0.6–0.7% at 4 h; 1.6–1.7% at 72 h, of which 1.0–1.1% is funding paid by
    the short (pumped coins carry negative funding).
  - Even the untradable market-median hedge (`M` gross) leaves only −1.7% to +1.9% gross,
    inside that drag.
- **The random baseline is negative** (−0.2% to −1.8%): shorting a random alt against BTC loses
  its costs. Pump fades beat it by 1–3 pp at short horizons, but `Hb` itself is never above 0
  with its CI.
- **Subgroups are inconsistent across windows**, so there is nothing to carry forward:
  - Breadth thirds: market-wide pumps 0→4 h −0.47% / +1.53%.
  - Heavy post-alert volume 4→72 h: +4.17% [−2.71, +9.52] vs −5.19% [−10.98, +1.67]. The sign
    flips, as in the persistence study.
  - Dumps (long coin, short BTC): +0.13% / +1.58% at 4 h, CIs span 0.

**Consequence:** no amendment to `docs/FORWARD_TEST_v2.md`; the predicate passed nowhere. The
BTC fields stay recorded live (descriptive). Changing the hedge instrument cannot fix it,
because even the ideal market hedge sits inside the cost + funding drag. The left tail is the
problem: a different exit rule, such as a stop, would be a new hypothesis, would need a fresh
declaration, and could only be validated forward.
