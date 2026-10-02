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
