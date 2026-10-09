# Extreme-crash strategy: buy the hardest-hit coins when the median coin falls ≥10% in 15 minutes (declared before the confirmation data, 2026-10-09)

## The strategy
1. **Watch, every minute.**
   - Coins: Binance USDT perpetuals with ≥ $5M traded in the last 24 h, excluding stablecoins,
     index and stock perps. At least 50 coins must qualify.
   - For each coin, its 15-minute price change.
2. **Trigger.** At least half of those coins are down ≥ 10% in the last 15 minutes (the median
   coin is down ≥ 10%). Act on the first trigger only, then ignore triggers for 24 h.
3. **Buy, 1 minute after the trigger:** the fifth of coins that fell the most in those 15 minutes,
   equal amounts each. No leverage.
4. **Sell everything 4 hours after entry.** No stop loss.
5. **Size:** only money that can fall by half within minutes. On 2025-10-10 the basket fell 52%
   in 5 minutes before ending +6%.

## Where it comes from (consumed data; discovery only)
- `docs/FLUSH_SIGNAL.md` declared a 5% flush trade, and it failed on 2022-01 → 2025-08 (0 / 4
  cells).
- Its EXTREME tier was declared with this exact threshold, but as a descriptive tier:
  - 15 fires; E1-H +9.33% [+5.59, +12.77] at 4 h, stress 3; 13 / 15 wins;
  - the rehearsal year (2025-10 → 2026-09) was 2 / 2.
- Those windows are consumed for this question. The discovery record below re-runs them with
  this exact definition, descriptively.

## Discovery record: 2022-01 → 2025-08 (consumed; run before this freeze; `receipt_extreme_disc.json`)
Same code and definition as the test.
- The run reproduces the flush receipt exactly. All self-tests passed, including the new
  basket-path check (15 / 15). The falsifiers behaved. Peak RSS was 1.45 GiB.
- **Primary (EXTREME E1-H, 4 h, stress 3):**
  - mean +9.33% [+5.59, +12.77], p 0.0002, median +8.87%, 13 / 15 wins;
  - random-time null −0.67% (p_null 0.0005); best event left out +8.60%.
  - It would meet every condition, but this window found the strategy, so it is **not
    evidence**.
- **Plateau** (fires, mean): D8 (20, +7.56%), D12 (12, +12.69%), S25 (19, +8.31%),
  S75 (14, +10.22%).
- **Other cells:** E1-M +6.66%, E2-H +4.45%, E2-M +3.13%.
- **Worst basket drawdown during the hold:** usually 0 to −3%. Three exceptions: 2022-05-11
  −8.1%, 2023-06-10 −12.6%, 2025-02-03 −14.0%. The rehearsal year adds 2025-10-10 at −52.3%.

## Confirmation test: window C (UNTOUCHED for this question)
- **Data:** Binance USDM 1m from Binance Vision, all USDT perps including delisted ones,
  2020-01-01 → 2021-12-31 (`data/hist/usdm_1m_early`).
  - The download started before this commit; downloading reads no outcomes.
  - `build --window early` refuses until this file, `research/extreme_crash.py`, its tests and
    `research/flush_signal.py` are committed clean.
- **Triggers:** 2020-01-02 → 2021-12-30, from quarterly panels with ±2-day margins.
- **Read once.** `validate --window early` refuses a second run.
- **Not perfectly blind:**
  - the large crashes of this period (2020-03-12, 2021-05-19) are public knowledge;
  - other repo programs read 2020–2021 klines for cross-sectional alpha, a different question.
- **Thin universe.** 2020 had fewer liquid perps. Minutes with < 50 qualifying coins cannot
  trigger (status `THIN_UNIVERSE`, counted). The rule is not relaxed for this.

## Definition
- **Code.**
  - `research/flush_signal.py` is the frozen code (cb690cd), unchanged: detector, universe,
    LIQ, E1/E2 entries, H/M baskets, costs, self-tests, null pool, statistics.
  - `research/extreme_crash.py` configures it: windows, tiers, minimum, k.
- **Trigger:** EXTREME = share_10% ≥ 50%; 24 h refractory per tier.
- **Primary cell (k = 1): EXTREME, E1, H, 4 h, stress 3.**
  - E1: buy at the close of t + 1.
  - H: the fifth of U(t) with the lowest 15-minute return, equal weight.
  - Costs: 2 × (5 bp + 3 × slippage tier).
  - Funding: excluded.
- **Plateau variants of EXTREME:**
  - depth: share_8% ≥ 50% and share_12% ≥ 50%;
  - share: share_10% ≥ 25% and share_10% ≥ 75%.
  - A variant with ≥ 3 fires must have mean net > 0. A variant with < 3 fires is reported, not
    required.
- **Minimum:** ≥ 10 events and ≥ 5 weeks, otherwise `INSUFFICIENT`.
  - This is lower than the flush study's 15 because extreme crashes are rarer. It was set before
    any count from window C was seen.
- **Statistics** (as `docs/FLUSH_SIGNAL.md`):
  - event-level ISO-week block bootstrap, 4,000 draws;
  - one-sided p = (1 + draws with mean ≤ 0) ÷ (draws + 1);
  - random-time null: 300 pool minutes per UTC hour, ≥ 24 h from any WATCH minute;
    hour-matched; 2,000 draws.
- **VALIDATED**, computed; all of:
  1. mean net > 0, 95% CI excludes 0, and p ≤ 0.05;
  2. p_null ≤ 0.05;
  3. the plateau rule;
  4. mean net > 0 with the single best event left out.
- **SUGGESTIVE:** mean > 0 and CI excludes 0, but another condition fails. **FAILED** otherwise.
- **Descriptive** (decides nothing):
  - E1-M (buy the whole liquid market), E2-H, E2-M;
  - holds of 1 h and 24 h; stress 1, 5 and gross;
  - per-year means;
  - the event table with basket size and the **worst basket drawdown during the hold**;
  - FLUSH-tier E1-H for comparison.

## Self-tests and falsifiers (abort on failure)
- Inherited from `flush_signal.build`:
  - the detector equals a naive per-minute recompute at every trigger plus 200 random minutes per
    panel;
  - 3 baskets are recomputed from the per-symbol parquet files.
- New: each EXTREME basket's minute path, recomputed from the panel, must end at the recorded
  4 h gross return (1e-6).
- Falsifiers for the primary cell with k = 1 and minimum 10:
  - planted (hour-matched pool draws + 5%, 30 events) must be VALIDATED;
  - pool draws used as events must not be VALIDATED;
  - no events must be `INSUFFICIENT`.

## Forward record
- EXTREME fires after 2026-09-27 are the only fully out-of-sample evidence.
- They will be evaluated offline from Binance Vision data with this code, under a later
  declaration that adds a forward window.
- No change reaches the live service; forward test v2 is untouched.

## Memory
- Same structure as the flush study: quarterly float32 memmap panels, 5,000-minute chunks,
  `replay.mem` guard at 2.0 GiB, `research_slot.sh` at the default 4 GiB cap.
- Measured on 2022–2025: peak 1.34 GiB.

## Result: confirmation test, window C read once (2026-10-09 05:50Z, `data/replay/flush_signal/receipt_extreme_early.json`)
- **Build:**
  - 140 symbols with data; 0 download errors.
  - All self-tests passed (detector, parquet price reconciliation, 14 / 14 basket paths). The
    falsifiers behaved. Peak RSS 0.72 GiB.
  - 417,570 of 1,048,320 owned minutes were `THIN_UNIVERSE` (< 50 liquid coins), nearly all
    before 2020-11. The effective window is about Nov 2020 → Dec 2021.
- **Fires:** EXTREME 14 (12 weeks). Plateau variants: D8 16, D12 8, S25 15, S75 8; all required.

| primary: EXTREME, E1-H, 4 h, stress 3 | value |
|---|---|
| mean net [95% CI] | **+8.56% [+4.35, +12.40]** |
| one-sided p (bar 0.05) | 0.0002 |
| median, wins | +8.91%, 12 / 14 |
| random-time null mean, p_null | −0.45%, 0.0005 |
| best event left out | +7.75% (top event = 16% of the total) |
| plateau D8 / D12 / S25 / S75 | +5.96% / +10.49% / +6.91% / +8.63% |
| **verdict** | **VALIDATED** |

- **Descriptive:**
  - E1-M (whole market) +5.21% [+1.73, +8.37]; E2-H +3.16% [−0.68, +6.63]; E2-M +1.69%.
    Buying 1 minute after the trigger beats waiting for the turn again.
  - E1-H at stress 1 / 3 / 5: +8.89% / +8.56% / +8.23%.
  - E1-H by hold: 1 h +9.30%, 24 h +13.29%.
  - The FLUSH (5%) tier in this window: E1-H +3.08%.
- **Drawdown during the hold:** 4 of 14 baskets fell 17–40% before recovering:
  - 2021-05-19 −39.8% (ended −2.4%);
  - 2021-12-04 −19.8%;
  - 2021-04-18 −18.9%;
  - 2021-09-07 −17.3%.

### Across all periods (descriptive): 31 extreme crashes, 2020-11 → 2026-09

| period | crashes | mean, E1-H 4 h, stress 3 | wins |
|---|---|---|---|
| 2020–21 (test) | 14 | +8.56% | 12 |
| 2022–25 (discovery) | 15 | +9.33% | 13 |
| 2025–26 (rehearsal) | 2 | +4.89% | 2 |
| **all** | **31** | **+8.70% (median +8.11%)** | **27** |

- Worst trade −3.94%; best +19.57%.
- **Drawdown during the hold:** 7 of 31 baskets fell ≥ 10% at some point before the exit; the
  worst were −52.3% (2025-10-10) and −39.8% (2021-05-19).
- **No leverage, no stop.** A stop or 2× leverage turns those baskets into the realised losses.

### What it says
- **The strategy is validated on untouched data**, the first result in this project to pass a
  confirmation test. After the median liquid coin falls ≥ 10% in 15 minutes, buying the
  hardest-hit fifth for 4 hours earned about +8–9% per crash after 3× normal slippage, in two
  separate periods, with 27 of 31 trades positive.
- **What is still unknown:**
  - **Real fills.** These are the worst minutes to trade: thin books, exchange lag and outages
    (Binance had them on 2021-05-19 and 2025-10-10). The test assumed 3× normal slippage. The
    edge survives about 5% per side of extra cost, but real crash fills were never measured.
  - **Frequency:** about 4–7 crashes per year (2021 had 12), so returns come in rare bursts.
  - **Venue:** Binance only. Funding is excluded.
- **Forward:** new crashes after 2026-09-27 are the remaining check, to be added by a later
  declaration. Nothing reaches the live service.

## Execution stress test (declared before computing, 2026-10-09)
Real crash fills are the main unknown. Every window is consumed, so this test measures how
sensitive the result is to execution; it selects nothing.
- **Events:** all 31 EXTREME fires (2020–21 test 14, 2022–25 discovery 15, 2025–26 rehearsal 2).
- **Basket:** the hardest-hit fifth at t, as in the primary.
- **Costs:** 2 × (5 bp + 3 × slippage tier), on top of every stress.
- **Stresses:**
  - **Worst fill:** buy at the HIGH of entry minute t + 1 and sell at the LOW of exit minute
    t + 241. Both come from the per-symbol 1m parquet files; if the exit minute is missing, use
    the last low in [entry, exit].
  - **Latency:** buy at the close of t + 2, t + 3, t + 5 or t + 10; hold 4 h from entry.
    t + 10 is descriptive.
- **Statistics:** mean net with a 95% ISO-week block CI (`flush_signal.cell_stats`, minimum 10).
- **Self-test:** the t + 1 latency row must reproduce the recorded E1-H 4 h net (1e-6).
- **ROBUST**, computed: mean > 0 and CI excludes 0 for the worst fill and for latencies t + 2,
  t + 3 and t + 5, both on the 14 test events and on all 31. Otherwise **FRAGILE**, with the
  failing cases listed.
- Code: `research/extreme_crash_exec.py`. Receipt:
  `data/replay/flush_signal/exec_stress_extreme.json`.

### Result: execution stress (2026-10-09 05:55Z, `exec_stress_extreme.json`): **ROBUST**
- Self-test OK: the t + 1 row reproduces the recorded E1-H on all 31 events. Peak RSS 0.38 GiB.

Mean net, 4 h, stress 3 (95% CI; wins):

| entry | 2020–21 test (14) | all 31 |
|---|---|---|
| t + 1 (the rule) | +8.56% [+4.35, +12.40]; 12 | +8.70% [+6.05, +11.22]; 27 |
| t + 2 | +6.65% [+2.17, +10.70]; 11 | +7.75% [+4.88, +10.66]; 27 |
| t + 3 | +4.79% [+0.44, +8.66]; 10 | +7.27% [+3.84, +11.23]; 25 |
| t + 5 | +3.93% [+0.08, +7.41]; 11 | +8.87% [+2.96, +18.73]; 27 * |
| t + 10 (descriptive) | +2.96% [+0.05, +5.73]; 11 | +6.42% [+1.98, +13.46]; 24 |
| worst fill (high in, low out) | +5.39% [+1.10, +9.22]; 11 | +4.61% [+2.08, +7.00]; 24 |

\* Inflated by 2025-10-10: entering at t + 5 bought the wick bottom (+136%).

- **Speed matters.** In the test period each minute of delay costs about 1–2 points, and by
  t + 5 the edge is about half. Execution must be automated.
- **The worst fill inside the minute still leaves +4.6% to +5.4%.** The stress does not cover
  depth beyond the printed range or outages. On 2025-10-10 the worst fill was −8.2%.

## Live alert (deployed 2026-10-09 06:16Z, commit 3a82553)
- **Code:** `movewatch/watchers.py` `CrashWatcher` and `CrashDetector`; config `watchers.crash`.
  Alert only, never trades.
- **Data:** Binance USDM `!miniTicker@arr`. Closes are the last price before each minute boundary;
  the universe is 24 h quote volume ≥ $5M plus a trade in the last 15 minutes.
- **Exclusions:** stables, index perps, dated contracts and stock/commodity perps. Stock perps are
  caught by the static list, the engine's cluster verdict, or the engine's reference-price rule
  on the symbol's own price; 88 liquid ones are excluded today.
- **Message:** the share and median drop, the hardest-hit fifth with each coin's 15-minute drop,
  and the 4 h exit time. A follow-up 4 h later gives the basket's gross return since the
  1-minute entry.
- **Checks before deploying:**
  - 7 unit tests (`tests/test_crash_watcher.py`), including replaying the research panel through
    the live detector: it fires on 2024-03-05 19:56 exactly, with the research basket.
  - Two live rehearsals with the trigger forced: 245–302 liquid crypto perps; peak 58 MB
    standalone.
- **Forward test v2 untouched:** `watchers.py` and `watchers.*` are outside its fingerprint. The
  restarted run's FU_START carries 2375f81d463303ac, the freeze.
- **Limits:**
  - The 24 h refractory is not persisted across restarts.
  - After a restart or reconnect, the trigger is blind for 15 minutes (status `NO_HISTORY`).
  - Liquidity is the rolling 24 h volume, not the 24 h before the 15-minute block (a tiny
    difference).
