# Short the heavy-volume pumps, with the tail cut (declared before results, 2026-10-09)

**Question.** The signal search validated post-alert volume: pumps whose trading stays heavy in
hours 1–4 fade more from +4 h to +72 h (A2, IC −0.20 on untouched Oct 2025–Mar 2026). Shorting
the heavy tercile won 66–69% of trades with a median of +6% to +8%. But the worst 5% lost 40–55%,
so the mean was about zero.
- Can a shorter hold or a stop-loss cut that tail enough to make the short pay?
- Is it confirmed on alerts that were never studied?

## Events (rows are actions)
- **Alerts:** replay NEW alerts with the ≤120 preset (floor ×1.5, z ×1.0, v ×1.0, liquidity
  ≥ $5M, alert logic v4), the same as the signal search.
- **Exclusions, as the signal search:**
  - stocks and commodities (static list, then `btc_hedge.stock_like` on the alert price);
  - BTC;
  - one alert per coin per 72 h, across directions;
  - trigger price reconciled against the panel (abort on mismatch);
  - frozen or delisted bars make a trade `STALE_OR_UNAVAILABLE`.
- **Population:** pumps (dir +1) with post-alert volume ratio
  `vol_persist = log10(vol[t+61, t+241) ÷ vol[t−179, t+1)) > 0.35259` (ratio > 2.25×).
  - This is the heavy-tercile cut frozen in `docs/SIGNAL_SEARCH_FREEZE.json` (A2). It is not
    re-fit.

## Trades (declared cells)
- **Entry:** short at the close of t + 241 (+4 h, once the volume ratio is known).
- **Exit:** at the close of t + 1441 (+24 h after the alert) or t + 4321 (+72 h).
- **Stop:** cover at the first minute close at or above entry × 1.25 (stop at +25%); the exit is
  that close, so gaps beyond 25% are paid in full.

| cell | exit | stop |
|---|---|---|
| S24 | +24 h | none |
| S72 | +72 h | none (the signal-search trade) |
| S24_stop25 | +24 h | +25% |
| S72_stop25 | +72 h | +25% |

- **Net** = −(exit ÷ entry − 1) − 2 × (5 bp + slippage tier by 24 h volume, 5/10/20/40 bp) +
  actual Binance funding received over the hold (short side).
  - Stress 1 is primary, as in the signal-search economics; stress 3 is descriptive.

## Windows
- **Screen** (consumed):
  - the signal-search windows S1 (Aug 2–Sep 24 2026) and S2 (Apr 2–Jul 28 2026);
  - V (Oct 2025–Mar 2026, with the seam de-dup).
  - **Reconciliation, abort on mismatch:** the builder rebuilds these rows and must match
    `data/replay/signal_search/rows_*.parquet` exactly, on the event set, `vol_persist` and the
    S72 net (= `signal_search.net(x4_72, fade)`).
- **Screen pass**, per cell:
  - mean net > 0 in S and in V separately;
  - pooled 95% CI excludes 0 (ISO-week block bootstrap with `cluster_weeks`, 4,000 draws).
- **Freeze:** the screen writes the carried cells and k to `docs/HEAVY_FADE_FREEZE.json`, which
  is committed before the confirmation rows are built.
- **Confirmation C** (UNTOUCHED for alert-level questions): replays `hfv_<panel start>` on the
  quarterly panels of 2020-01 → 2025-08.
  - Corpora: `usdm_1m_early` and `usdm_1m_hist`; the replays started before this file was
    committed; replays produce alerts only.
  - Signal range per panel: [panel start + 1 d 4 h, panel end − 3 d − 1 h). Gaps of about 5 h per
    quarter boundary, plus 2021-12-28 23:00 → 2022-01-02 04:00 at the seam between the two corpora.
  - One alert per coin per 72 h across the whole window.
  - `rows --window conf` refuses until the freeze file and this builder are committed clean, and
    the window is read once (`validate` refuses a second run).
  - Not perfectly blind: crash days in this period were read for the crash study, and pump
    alerts on those days are a small minority.
- **VALIDATED**, per carried cell; all of:
  1. mean net > 0, CI excludes 0, and one-sided p ≤ 0.05 / k;
  2. mean net > 0 with the single best event left out.
- **SUGGESTIVE:** mean > 0 and CI excludes 0, but another condition fails. **FAILED** otherwise.
  **INSUFFICIENT** below 100 events or 5 weeks.
- **Descriptive:**
  - stops at +15% and +40%; stress 3;
  - the same trade on all pumps and on the quiet tercile;
  - per year; median, win rate, 5th percentile, worst trade; stopped share.

## Self-tests
- The reconciliation above.
- Instrument falsifiers in validate, with the real week structure:
  - planted nets (mean +5%) must be VALIDATED;
  - zero-mean noise must not;
  - no events must be INSUFFICIENT.

## Memory
- One panel at a time, with float32 memmaps; `replay.mem` guard at 2.0 GiB.
- The replays measured 1.17 GiB peak on 4-month panels.

## Result: screen (2026-10-09; `docs/HEAVY_FADE_FREEZE.json`, k = 0)
- **Reconciliation exact:** 3,291 pumps; the event set equal; 0 `vol_persist` mismatches; 0 S72
  net mismatches against `signal_search.net`. 1,108 heavy pumps.
- **Cells** (stress 1, with funding; pooled S ∪ V):

  | cell | mean [95% CI] | median | wins | p5 | worst | mean S / V |
  |---|---|---|---|---|---|---|
  | S24 | +1.05% [−0.47, +2.50] | +4.49% | 63% | −34.1% | −183% | +0.83% / +1.23% |
  | S72 | +0.66% [−2.52, +3.43] | +7.15% | 68% | −45.9% | −675% | +2.00% / −0.45% |
  | S24_stop25 | +0.29% [−0.72, +1.29] | +3.80% | 60% | −28.1% | −41% | +0.40% / +0.20% |
  | S72_stop25 | +0.31% [−0.90, +1.49] | +4.51% | 59% | −29.3% | −41% | +0.38% / +0.26% |

- **Descriptive stops:**
  - 15%: S24 +0.65%, S72 +0.75%;
  - 40%: S24 +0.61%, S72 +0.98%.
  - Stopped share at 25%: 19% (24 h) and 29% (72 h).
- **No cell passes (k = 0).** Under this declaration the confirmation window is not read.

### What it says
- **A stop does not cut the tail usefully.** About a fifth to a quarter of heavy pumps rise
  another 25% within a day, and many then fade anyway. The stop locks in the loss and gives up the
  fade.
- The 24 h hold beats 72 h, but +1% per trade is not distinguishable from zero at n = 1,108.
- A follow-up with one exploratory filter (signal 3) is declared separately in
  `docs/HEAVY_FADE_V2.md`.
