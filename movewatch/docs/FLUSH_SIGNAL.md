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
