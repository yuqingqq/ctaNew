# Flush signal: after half the market drops ≥5% in 15 minutes, does buying pay? (declared before results, 2026-10-09)

**Question.** The signal search found that dumps during broad market moves reverse more over 4 h
(A1, `docs/SIGNAL_SEARCH.md`). A post-hoc look showed:
- the effect lives in market-wide liquidation flushes;
- for pumps, breadth was a proxy for alert speed.

This study restates the lead as a **market-level** signal, independent of the alert detector,
and tests it once on an untouched window. The question is whether, after a flush, buying the
market (M) or the hardest-hit coins (H) pays net of crash-stressed costs.

## Windows
- **Calibration and rehearsal** (consumed; descriptive only): 2025-10-01 → 2026-09-27.
  - The thresholds below were set on this year from how often each level occurs (minute-level
    breadth only; `flush_calib.py`, scratch).
  - No post-flush return was used to pick them. But the outcomes of six flush days in this year
    were seen in the same session, so this year cannot test the signal.
  - The full pipeline is rehearsed here, labelled `REHEARSAL`.
- **Test window H** (UNTOUCHED for this question): Binance USDM 1m, Binance Vision monthly
  archives, all USDT perps including delisted ones.
  - Data 2022-01-01 → 2025-08-31; triggers 2022-01-02 → 2025-08-30.
  - Downloaded into `data/hist/usdm_1m_hist`.
  - **Read once.** `build --window hist` refuses unless this file, the builder and its tests are
    committed clean. `validate --window hist` refuses a second run.
  - Not perfectly blind: large crashes in this period (LUNA, FTX, 2024-08-05, 2025-04-07) are
    public knowledge. Other repo programs read 2022–2025 klines for
    cross-sectional alpha, a different question. Flagged in the receipt.
- **Forward:** flushes after the freeze. Reading them is a later, separate declaration. The live
  service and forward test v2 are untouched (no code or config change reaches the service).

## Detector (computed every minute t, from closes up to t only)
- **Universe U(t):** USDT perps in the panel, excluding:
  - stables (`replay.STABLES`);
  - index perps (BTCDOM, DEFI, BLUEBIRD, FOOTBALL);
  - stocks and commodities (`common.is_tradfi`).
- **Each coin in U(t) needs:**
  - 24 h USD volume ≥ $5M (`LIQ`, the 24 h before the 15-min block start, as `replay.compute_blocks`);
  - finite closes at t and t − 15;
  - volume > 0 in the last 15 minutes.
- **Minimum size:** |U(t)| ≥ 50; otherwise the minute has status `THIN_UNIVERSE` and cannot
  trigger.
- **Share:** share_X(t) = fraction of U(t) with log(C[t] / C[t − 15]) ≤ log(1 − X).
- **Tiers** (the share threshold is 50% for all):

  | tier | condition |
  |---|---|
  | WATCH | share_3% ≥ 50% |
  | **FLUSH** | **share_5% ≥ 50%** (primary) |
  | EXTREME | share_10% ≥ 50% |

- **Plateau variants of FLUSH:**
  - depth: share_4% ≥ 50% and share_6% ≥ 50%;
  - share: share_5% ≥ 25% and share_5% ≥ 75%.
- **Fires:** each tier or variant fires at the first minute its condition holds, unless the same
  tier or variant fired in the previous 24 h.
  - Tiers are independent, so a WATCH never blocks a later FLUSH. On 2025-10-10 a WATCH fired at
    15:30 and the FLUSH at 20:58.
- **Population:** one row per FLUSH fire (rows are actions).

## Trades
- **Entries:**
  - **E1:** buy at the close of t + 1.
  - **E2 (wait for the turn):** find the first minute u in [t + 1, t + 60] where the median over
    U(u) of the 5-minute log return is > 0, and buy at the close of u + 1.
    - With no turn, buy at t + 61 (status `E2_TIMEOUT`, counted).
    - Every event keeps exactly one E2 trade.
- **Baskets**, membership fixed at the trigger minute t:
  - **M:** all of U(t), equal weight.
  - **H:** the fifth of U(t) with the lowest 15-minute return (ceil(n / 5) coins), equal weight.
- **Per-coin return:** C[exit] / C[entry] − 1.
  - No entry price: the coin is left out (status `NO_ENTRY_PRICE`).
  - No exit price: the last finite close in [entry, exit] (status `LAST_PRICE`).
- **Hold:** **4 h (primary)**; 1 h and 24 h descriptive.
- **Costs per coin:** 2 × (5 bp taker + stress × slippage tier by 24 h volume, 5/10/20/40 bp,
  `btc_hedge.TIERS`).
  - **Stress = 3 (primary)**; 1 and 5 descriptive.
- **Funding:** excluded. A 4 h hold spans at most one settlement. In flushes perps trade at a
  discount, so funding usually pays longs; leaving it out is conservative. Stated as a limitation.
- **Net** per trade = equal-weight mean over the basket of (return − cost).

## Cells and statistics
- **Cells:** {E1, E2} × {M, H}, hold 4 h, stress 3 → **k = 4**.
- **Inference unit:** the event.
  - 95% CI of the mean net: ISO-week block bootstrap, 4,000 draws.
  - One-sided p = (1 + draws with mean ≤ 0) ÷ (draws + 1).
  - Minimum: ≥ 15 events and ≥ 5 weeks, otherwise `INSUFFICIENT`.
- **Random-time null:**
  - Pool: 300 random minutes per UTC hour in the same window. A pool minute must be at least
    24 h away from any minute with share_3% ≥ 50%.
  - Each pool minute is a pseudo-trigger: same entry rule, same basket rule (H = that minute's
    15-minute losers), same hold and costs.
  - A null draw takes one pool minute per event with the same UTC hour; 2,000 draws.
  - p_null = (1 + draws with null mean ≥ observed mean) ÷ (draws + 1).
- **VALIDATED**, computed per cell; all of:
  1. mean net > 0, CI excludes 0, and p ≤ 0.05 / 4;
  2. p_null ≤ 0.05;
  3. mean net > 0 in each of the four plateau variants;
  4. mean net > 0 with the single best event left out.
- **SUGGESTIVE:** mean > 0 and CI excludes 0, but some other condition fails. **FAILED** otherwise.

## Descriptive (decides nothing)
- H − M with CI: the original "worst-hit bounce more" effect.
- The EXTREME and WATCH tiers.
- Holds of 1 h and 24 h; stress 1× and 5×.
- Per-year means; the event table; share of P&L from the top event; E2 timeouts and the other
  statuses.

## Self-tests (abort on failure)
1. **Detector reconciliation:** the vectorised minute series equals a naive per-minute recompute
   at every trigger minute and at 200 random minutes per panel.
2. **Price reconciliation:** basket returns for 3 events per window are recomputed from the
   per-symbol 1m parquet files, not the panel; they must agree to 1e-6.
3. **LIQ reconciliation** (rehearsal panels): the builder's LIQ equals `blocks_*_r15.npz` from
   the replay engine.
4. **Falsifiers for the instrument:**
   - planted (event nets = hour-matched null-pool draws + 5%, variants planted too) must be
     VALIDATED. At 3× stress a basket costs about 1–1.5% round trip, so +5% is a clear positive;
   - null-pool draws used as events must not be VALIDATED;
   - no events must be INSUFFICIENT.

## Memory
- Quarterly panels with ±2-day margins, as float32 memmaps; the detector runs in chunks of
  5,000 minutes.
- The process aborts above 2.0 GiB RSS (`replay.mem`) and runs through `research_slot.sh` at the
  default 4 GiB cap.

## Rehearsal (run and seen before this freeze; consumed year, descriptive)
- The full pipeline ran on 2025-10-01 → 2026-09-26 (`receipt_rehearsal.json`).
  - All self-tests passed: the detector matched the naive recompute, LIQ matched the replay
    blocks, and 3 baskets matched the parquet files.
  - The falsifiers behaved: planted VALIDATED, null-as-events FAILED, empty INSUFFICIENT.
  - Peak RSS was 1.55 GiB.
- 8 FLUSH fires, so every cell is INSUFFICIENT (< 15 events).
- E1 mean net at 4 h, stress 3: M +0.70%, H +1.52%. E2: M −1.71%, H −2.09%. Random-time pool:
  −0.97%.
- On 2025-10-10 the FLUSH fired at 20:58, before the cascade's bottom (~21:20). The E1 market
  basket lost 17% over 4 h.
- **Nothing in the design was changed after the rehearsal.** Re-tuning on one consumed event is
  what the freeze exists to prevent. Entry timing in cascades is the known weak point; the test
  measures it.

## Results: test window H, read once (2026-10-09 04:31Z, `data/replay/flush_signal/receipt_hist.json`)
- **Build:**
  - 545 symbols (541 after exclusions), 15 quarterly panels.
  - All self-tests OK: the detector matched the naive recompute on every trigger plus 200 random
    minutes per panel, and 3 baskets reconciled against the parquet files.
  - E2 timeouts 0; missing entry or exit prices 0; peak RSS 1.34 GiB.
- **Fires:** WATCH 156, **FLUSH 49 (41 weeks)**, EXTREME 15. Plateau variants: D4 84, D6 35,
  S25 73, S75 34.
- **Falsifiers:** planted VALIDATED, null-as-events FAILED, empty INSUFFICIENT, in all four cells.

| cell (4 h, stress 3) | mean net [95% CI] | p (bar 0.0125) | median | win | random-time null, p_null | best event left out | verdict |
|---|---|---|---|---|---|---|---|
| E1-M | +0.51% [−0.72, +1.78] | 0.21 | +0.58% | 55% | −0.80%, 0.0005 | +0.19% | FAILED |
| E1-H | +1.28% [−0.21, +2.87] | 0.048 | +1.53% | 63% | −0.73%, 0.0005 | +0.88% | FAILED |
| E2-M | −0.07% [−1.25, +0.94] | 0.54 | +0.63% | 59% | −0.80%, 0.0005 | −0.25% | FAILED |
| E2-H | +0.40% [−1.09, +1.85] | 0.30 | +0.77% | 59% | −0.73%, 0.0005 | −0.00% | FAILED |

- **0 / 4 VALIDATED.** Every cell fails on its CI.
- E1-H is the closest: it beats random times (whose mean is just the cost), and its plateau
  variants and best-event-left-out mean are all positive. But +1.3% per flush after 3× costs is
  not distinguishable from zero over 49 events.

### Descriptive (declared as descriptive; decides nothing)
- **Worst-hit beat the market** (H − M, E1): +0.77% [+0.33, +1.32]. The original A1 effect shows
  on untouched data, but it is small.
- **The bounce grows with severity.** E1-M mean, 4 h, stress 3:

  | trigger | mean |
  |---|---|
  | WATCH (−3%) | −0.03% |
  | FLUSH (−5%) | +0.51% |
  | FLUSH_D6 (−6%) | +1.94% |
  | EXTREME (−10%) | +6.66% |

- **EXTREME (n 15):**
  - E1-M +6.66% [+4.03, +9.09]; E1-H +9.33% [+5.59, +12.77]; E2-H +4.45% [+1.75, +7.16].
  - E1-H wins 13 of 15. With the best event left out: +8.60%. At stress 5: +8.93%.
  - It would take about 5% extra slippage per side to erase E1-H's gross +10.0%.
  - In the rehearsal year both EXTREME fires were positive for H (+5.1% on 2025-10-10 21:15,
    +4.7% on 2026-08-22).
- **Why FLUSH loses where EXTREME wins.** The FLUSH losers are first legs of multi-leg cascades;
  EXTREME fires later, deeper into the same cascade:

  | cascade | FLUSH (E1-M) | EXTREME fired | EXTREME (E1-M) |
  |---|---|---|---|
  | 2022-11-08 (FTX) | −9.1% | 84 min later | +11.1% |
  | 2025-02-02 | −14.5% | 195 min later | +1.5% |
  | 2024-04-12 | −7.2% | 67 min later | +4.3% |
  | 2024-12-09 | −3.6% | 14 min later | +14.3% |

- **E2 (wait for the turn)** is worse than E1 in every cell: the snap-back comes first.
- **Per year, E1-H:** 2022 +1.10% (n 14), 2023 +2.58% (17), 2024 +1.38% (15), 2025 −5.74% (3).

### What it says
- **As declared, the flush trade does not validate.** Buying the first minute half the market is
  down ≥ 5% in 15 minutes earns about +0.5% (market) to +1.3% (worst-hit) per flush after
  stressed costs. That is better than random times, but not distinguishable from zero.
- **The candidate is EXTREME**: the median coin down ≥ 10% in 15 minutes, about 4 per year.
  - Its threshold was fixed in this declaration before the data, but it was a descriptive tier,
    not a test cell. It is therefore NOT validated: k would have been 8, and n = 15 is exactly
    the minimum.
  - Crash-time execution is the real unknown. The margin is wide (about 5% per side for H), but
    so are spreads at those minutes.
- **Next, if pursued:** a forward-only declaration with EXTREME as the primary cell (E1, M and H,
  4 h). It would be read only after enough new EXTREME fires (≥ 10). At about 4 per year that
  takes years, so the realistic use is as a rare, pre-committed rule, not a validated edge.
