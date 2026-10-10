# Short heavy-volume pumps whose futures trade at a discount by +4 h: v2 (declared before results, 2026-10-09)

Supersedes nothing in `docs/HEAVY_FADE.md`. That declaration screened four tail-cutting cells
and carried none (k = 0, `docs/HEAVY_FADE_FREEZE.json`). This is a separate declaration with
one cell.

**How the cell was chosen (stated plainly).** After v1's screen, an exploratory split of the same
consumed screen rows by signal 3 (`premium_h4`, the A3 tercile cuts frozen in
`docs/SIGNAL_SEARCH_FREEZE.json`) showed this for heavy pumps with a discount, covered at 24 h:
- +1.80% [+0.44, +3.16], n 533, median +5.2%, 69% wins;
- positive in S (+3.50%) and V (+0.64%).

The middle and premium groups were −1.36% and +1.37%, with CIs crossing 0. The cell was picked
after seeing these numbers, so they are not evidence. The only evidence is the untouched read
below.

## Cell (k = 1)
- **Population:** v1's heavy pumps (`vol_persist > 0.35259`, `docs/HEAVY_FADE.md`), with
  **`premium_h4 ≤ −0.000624`** (A3 cut 0: futures at a discount of at least 0.062% to the index,
  averaged over hour 3–4 after the alert).
  - `premium_h4` comes from `research.runners.first4h`, unchanged (Binance Vision
    premiumIndexKlines 1m).
  - Pumps with no premium data are excluded (status `PREMIUM_MISSING`, counted).
- **Trade:** short at the close of t + 241 and cover at the close of t + 1441 (24 h after the
  alert); no stop. Net at stress 1, plus funding received: v1's `S24`, computed by v1's builder.

## Window and rule
- **Confirmation:** the untouched replays of 2020-01 → 2025-08, the same rows as v1's conf
  (`python3 -m research.heavy_fade rows --window conf`, v1 builder, unchanged).
- **Read once:** `python3 -m research.heavy_fade_v2 validate` refuses unless this file, its
  script, its tests and v1's freeze are committed clean, and refuses a second run.
- **VALIDATED:** mean net > 0, 95% CI excludes 0 (`cluster_weeks` block bootstrap, 4,000 draws),
  one-sided p ≤ 0.05, and mean > 0 with the best event left out.
- **SUGGESTIVE:** mean > 0 and CI excludes 0, but another condition fails. **FAILED** otherwise.
  **INSUFFICIENT** below 100 events or 5 weeks.
- **Descriptive:**
  - S72 for the same cell; stops 15/25/40%; stress 3;
  - all heavy pumps; heavy ∩ middle; heavy ∩ premium;
  - per year; median, wins, 5th percentile, worst trade.

## Self-tests
- **premium_h4 reconciliation:** recomputed by this script for the screen-window heavy pumps, it
  must equal the signal-search rows' `premium_h4` (atol 1e-12) on every row where both exist.
- **Falsifiers:** v1's planted / noise / empty checks with k = 1.
- **Caveats:**
  - Binance Vision premium history may be thinner in 2020–21; the `PREMIUM_MISSING` count is
    reported.
  - The confirmation rows also carry v1's caveat: crash days were read for the crash study.

## Result: untouched read (2026-10-09 08:11Z, `data/replay/heavy_fade/receipt_v2_conf.json`): **FAILED**
- **Rows** (v1 builder; replays `hfv_*` for 2020-01 → 2025-08, 23 panels, peak 0.80 GiB):
  - 9,758 NEW alerts; 7,012 after the 72 h de-dup; 6,988 rows.
  - 0 trigger mismatches; BTC 21 and stock-price 3 excluded.
  - 4,823 pumps, 1,389 heavy.
  - Funding is missing for 14 of 3,792 coin-months; those trades are excluded.
- **premium_h4 reconciliation exact:** 1,080 / 1,080 screen rows, 0 mismatches.
  `PREMIUM_MISSING`: 18 heavy pumps in C. The falsifiers behaved (planted VALIDATED, noise
  FAILED, empty INSUFFICIENT).
- **Cell (heavy & discount, S24, stress 1):** n 548, mean **−0.63% [−2.28, +1.01]**, p 0.77,
  median +2.03%, 61% wins, p5 −29.5%, worst −175% → **FAILED**.

Descriptive:

| heavy pumps, S24 | n | mean [95% CI] | median | wins |
|---|---|---|---|---|
| all | 1,385 | −0.44% [−1.76, +0.62] | +2.49% | 61% |
| discount | 548 | −0.63% | +2.03% | 61% |
| middle | 393 | +1.02% [−0.32, +2.33] | +2.87% | 60% |
| premium | 426 | +0.18% | +2.75% | 62% |

- Holding 72 h instead (all heavy pumps): −0.73%, worst −909%.
- The discount cell with stops: −0.04% to −1.10%.
- Per year (discount, S24): 2021 −1.58%, 2022 −0.32%, 2023 +1.13%, 2024 +0.14%, 2025 −1.58%.

### Post-read descriptive: does the A2 ranking itself replicate? (the window is now read)
Spearman of `vol_persist` against the coin's 4 h → 72 h return, all pumps:

| period | pumps | Spearman | median 4→72 h: quiet / middle / heavy |
|---|---|---|---|
| screen (2025-10 → 2026-09) | 3,258 | −0.221 | −0.5% / −5.1% / −9.5% |
| C (2020 → 2025-08) | 4,806 | **−0.107** | **−0.9% / −2.0% / −4.5%** |

Shorting each tercile for 72 h in C: quiet −1.03%, middle −1.41%, heavy −0.73% per trade.

### What it says
- **The discount filter was noise.** It ranked worst of the three premium groups on untouched
  data.
- **Shorting heavy-volume pumps does not pay**, whether at 24 h, 72 h, with stops, or with the
  filter:
  - the median trade wins (+2% to +4%, about 61% of the time);
  - the runners (worst trades −175% to −900%) make the mean negative.
- **The post-alert volume signal is real as a ranking but small.** Heavy pumps give back about
  4 points more than quiet ones over 4 → 72 h in 2020–25, half the size seen in 2025–26.
- **Quiet pumps still drift down** (median −0.9%), so "buy the quiet pumps" has no basis either.
- **The short-the-pump line is closed on free data:** its left tail is structural.
- 2020-01 → 2025-08 is now consumed for alert-level pump questions.

### Post-read descriptive: what kind of coin runs? (both windows already read; `research/runner_profile.py`)
- **Data:** 8,114 pumps (2020-25 C, plus the 2025-26 screen).
- **Meme flag:** CoinGecko `meme-token` category fetched 2026-10-09, market-cap rank ≤ 1500: 171
  symbols (`docs/coingecko_meme_rank1500_2026-10-09.json`). It is a current snapshot, applied
  retroactively.
- **Runner:** coin return from +4 h to +72 h of +50% or more.

| group | share of pumps | runner rate | doubled | short S24, heavy pumps (2020-25 / 2025-26) |
|---|---|---|---|---|
| meme | 10% | 5.5% | 2.2% | −7.19% / −2.50% |
| non-meme | 90% | 2.7% | 0.6% | +0.14% [−0.66, +0.90] / +1.45% [+0.16, +2.76] |
| non-meme, ≥ $20M/day, listed ≥ 180 d | 32% | 2.2% | 0.4% | +0.38% [−0.96, +1.67] / +3.57% [+0.42, +6.57] |

- **Runners by liquidity at the alert:**
  - < $20M: 2.7%;
  - $20–100M: 2.8%;
  - ≥ $100M: 3.7%.
  - Runners are NOT concentrated in thin coins. The largest (MYX +835%, AIA, TNSR, RAVE, POWER)
    were liquid, non-meme, narrative small caps.
- **Runners by age on Binance futures:** < 30 d 4.5%; ≥ 1 y 2.3%.
- **Reading:**
  - Memes are the worst shorts.
  - Without them, the heavy-pump short is flat in 2020-25 and positive only in 2025-26.
  - The filters were chosen after seeing both windows, so nothing here is evidence. Only forward
    alerts can test it.

### Post-read descriptive: signal 2 by class of coin (2026-10-09 12:17Z; `research/volume_classes.py`)
**Setup:**
- Classes: the same 15 as signal 3 (`research/premium_classes.py`).
- Windows: S, V and C (2020–25).
- Trades (A2 cuts): FADE = short heavy pumps, 4 → 24 h; FOLLOW = buy quiet pumps, 4 → 72 h.
- Rule, stated before the run: the ranking is < 0 in all three windows, the trade's mean is > 0
  in all three (n ≥ 100), and C's CI excludes 0.
- Control: all pumps reproduce the ranking (−0.195 / −0.247 / −0.107) and the v2 receipt's C
  heavy S24 (−0.44%, n 1,385).

**Result: none of 30 tries works.**
- **The ranking is robust across classes.** It is negative in every class and window except
  liquidity ≥ $100M in S (+0.016). It is strongest in the thin, unpopular and old coins:
  - liquidity < $20M: −0.26 / −0.26 / −0.21;
  - popularity rank > 150: −0.26 / −0.30 / −0.18;
  - age ≥ 1 y: −0.22 / −0.34 / −0.14.
- **Near misses** (positive in all three windows, C's CI spans 0):
  - FADE:
    - liquidity < $20M: +1.31 / +0.80 / +0.67%;
    - popularity 51–150: +0.73 / +0.70 / +0.67%;
    - age ≥ 1 y: +0.77 / +0.50 / +0.59%;
    - non-meme: +1.64 / +1.30 / +0.14%.
  - FOLLOW:
    - liquidity < $20M: +2.40 / +4.52 / +1.30%;
    - age 90 d–1 y: +3.50 / +1.87 / +1.65%;
    - popularity > 150: +0.27 / +5.25 / +1.15%.
- **Post-hoc (picked after seeing): thin coins, both legs pooled.**
  - Stress 1: S +1.84% [+0.20, +3.46], V +2.55% [−0.14, +5.28], C +0.94% [−0.21, +1.98].
  - Stress 3: C +0.14%.
  - The 2021 fade leg was −4.1%.
  - This is a forward-only candidate at best, and slippage in thin coins decides it.

### Post-read descriptive: regimes and the book view (2026-10-09 13:24Z; `research/regime_split.py`, `research/fade_portfolio.py`)
**Regime at the alert.** Gauges were fixed before the run and use closes through the day before:
- alt bear: more than half of liquid alts below their price 90 days earlier;
- BTC bear: BTC below its price 90 days earlier;
- pump-heavy: trailing-90-day pump rows per liquid coin above the median, 6.9 a year.

**Short heavy pumps 24 h, pooled 2020–26:**

| gauge | bear (or pump-heavy) | bull (or pump-light) |
|---|---|---|
| alt | +0.73% [−0.31, +1.68] | −0.86% |
| BTC | +0.80% | −0.43% |
| pump intensity | +0.68% | −0.56% |
| alt bear & pump-heavy | +1.05% [−0.52, +2.43], n 890 | |

- Within 2020–25, the alt-bear periods gave **+0.08%** (n 749).
- The bear-period gain is V (+1.38%) and S (+0.99%).
- The pump-heavy regime existed only in 2021 (a bull) and in 2025–26. So "bear & pump-heavy" is
  essentially the last year: one occurrence, nothing to test it on.

**Book:** 1% of capital per trade, booked on the exit day, no compounding; heavy pumps; S24.

| book | 2020-01 → 2026-09 | 2020–25 (C) | V (Oct 25–Mar 26) | S (Apr–Sep 26) | 2021 | max DD | max open |
|---|---|---|---|---|---|---|---|
| all, stress 1 | +0.8%/yr, Sharpe 0.18 | −1.3%/yr | +15.9%/yr, Sharpe 2.7 | +7.2%/yr | −8.4% | −9.9% | 12 |
| non-meme, stress 1 | +2.5%/yr, Sharpe 0.71 | +0.3%/yr | +15.3%/yr, Sharpe 2.8 | +13.7%/yr, Sharpe 1.9 | −1.8% | −4.8% | 11 |
| all, stress 3 | −1.5%/yr | −2.6%/yr | +7.7%/yr | +0.1%/yr | −9.3% | −15.2% | 12 |
| non-meme, stress 3 | +0.4%/yr | −1.0%/yr | +7.8%/yr | +7.4%/yr | −2.7% | −6.6% | 11 |

- Rolling 90-day P&L is negative in 40–50% of windows.
- The aggregate is meaningful only in the last year: about +14%/yr per 1% risked per trade at
  stress 1, and about +7.5% at stress 3. Over 2020–25 it is flat to negative.
- P&L scales linearly with size, and so do drawdowns, the runner hit (worst 24 h trade −183% of
  its notional) and gross exposure (up to 11–12 positions open at once).

**Per-trade by year** (2026-10-09 13:40Z; `research/fade_portfolio.py --trades`; non-meme heavy short 24 h, stress 1)

| year | n | win rate | median win / loss | lose > 50% | losses ÷ wins | mean |
|---|---|---|---|---|---|---|
| 2021 | 211 | 58% | +6.3 / −6.4% | 1.9% | 1.16 | −0.78% |
| 2022 | 145 | 66% | +6.8 / −5.3% | 1.4% | 0.88 | +0.58% |
| 2023 | 204 | 61% | +6.1 / −5.2% | 1.5% | 0.87 | +0.60% |
| 2024 | 282 | 60% | +6.4 / −5.6% | 1.4% | 0.90 | +0.43% |
| 2025 | 729 | 64% | +7.8 / −7.8% | 2.2% | 0.91 | +0.57% |
| 2026 | 688 | 63% | +9.8 / −10.4% | 2.2% | 0.81 | +1.49% |

- **The win rate is stable** (58–66%).
- **Losers take back 81–91% of the gross wins every year except 2021**, when they took back 116%
  (and 180% with memes).
- **The recent gain is bigger moves, not more wins.** Median wins and losses both grew from
  about ±6% to about ±10%, with a slightly better ratio.
- So the edge is the thin remainder after losses, and a few extra runners erase it.

**Is every year after 2021 positive?** (2026-10-10 00:59Z; same script with a cluster-week bootstrap 95% CI per
year; non-meme heavy short 24 h; "2022–25" is pooled after 2021 was seen to lose, so post-hoc)

| year | mean, stress 1 | 95% CI | mean, stress 3 | 95% CI |
|---|---|---|---|---|
| 2021 | −0.78% | [−3.02, +1.23] | −1.21% | [−3.45, +0.80] |
| 2022 | +0.58% | [−1.32, +2.45] | +0.12% | [−1.79, +2.01] |
| 2023 | +0.60% | [−1.06, +2.24] | +0.14% | [−1.50, +1.77] |
| 2024 | +0.43% | [−1.15, +1.97] | −0.07% | [−1.64, +1.48] |
| 2025 | +0.57% | [−0.75, +1.83] | −0.05% | [−1.37, +1.21] |
| 2026 | +1.49% | [−0.10, +3.05] | +0.80% | [−0.78, +2.37] |
| 2022–25 pooled | +0.54% | [−0.30, +1.36] | −0.01% | [−0.85, +0.81] |
| all 2020–26 | +0.72% | [+0.02, +1.44] | +0.14% | [−0.56, +0.86] |

- At stress 1 every year except 2021 has a positive mean (2020 has only 12 trades). **No single
  year's CI excludes 0** (computed in the script, at both stresses).
- Tripling slippage costs about 0.5–0.6% per trade. That is the whole 2022–25 edge: pooled
  +0.54% becomes −0.01%. Only 2026 stays positive at stress 3.
- The all-years CI just excludes 0 at stress 1 (+0.02%), but the meme exclusion was chosen
  after seeing, and the bound is gone at stress 3.
- All heavy pumps (memes kept): 2022–25 pooled +0.48% [−0.35, +1.28] at stress 1, −0.06% at
  stress 3.
- Reading: the sign is consistent across years, but the size is about the same as the cost
  uncertainty. Before 2025 it is a slight edge before costs, not a tradable one.

**Stop-loss on the F1 trade** (2026-10-10 03:11Z; `research/fade_stops.py`; descriptive, all windows consumed;
non-meme heavy pumps, short +4 h → +24 h, funding included; a stop covers at the first minute
close at or above entry × (1 + stop), so a jump through the level fills beyond it)

| stop | mean, stress 1 | 95% CI | mean, stress 3 | win | p5 | worst | stopped |
|---|---|---|---|---|---|---|---|
| none | +0.72% | [+0.02, +1.44] | +0.14% | 62% | −28.5% | −175% | 0% |
| +15% | +0.48% | [−0.12, +1.08] | −0.10% | 57% | −17.8% | −40% | 26% |
| +25% | +0.23% | [−0.38, +0.86] | −0.35% | 60% | −27.2% | −48% | 15% |
| +40% | +0.39% | [−0.26, +1.07] | −0.18% | 62% | −41.3% | −62% | 7% |

- **Every stop lowers the mean**, in 2020–25 and in 2025–26 alike. Stopped pumps often fade
  anyway, so the stop locks in a loss the hold would have recovered.
- **What a stop buys is the tail.** The worst trade goes from −175% to −40% / −48% / −62%;
  overshoot past the level is 15–25 points.
- **+15% costs the least mean** (−0.24 points) and also tightens p5 (−28.5% → −17.8%). It is the
  only stop that improves the worst case per unit of mean: about 4× smaller worst loss for a
  third less mean. The level was picked after seeing this table, so it is a forward-only candidate.

**Entry timing for the F1 trade** (2026-10-10 03:14Z; `research/fade_entry_timing.py`; descriptive, consumed;
non-meme heavy pumps, every trade held 20 h, stress 1 + funding; E4 reproduces the research S24
net exactly: n 2,271, max difference 0)

| entry | trades | mean | 95% CI | median | worst | E4 on the same rows |
|---|---|---|---|---|---|---|
| **E4: +4 h (F1)** | 2,271 | **+0.72%** | [+0.02, +1.44] | +3.5% | −175% | |
| E6: +6 h | 2,271 | +0.47% | [−0.22, +1.16] | +3.2% | −205% | |
| E8: +8 h | 2,271 | +0.42% | [−0.42, +1.17] | +2.6% | −471% | |
| E12: +12 h | 2,271 | +0.41% | [−0.34, +1.11] | +2.2% | −286% | |
| E24: +24 h | 2,271 | −0.38% | [−1.03, +0.24] | +0.9% | −231% | |
| WEAK3: 3% below the +4 h price | 1,859 | +0.49% | [−0.41, +1.39] | +3.0% | −311% | +3.87% |
| SPIKE10: 10% above the +4 h price | 856 | −0.22% | [−1.87, +1.36] | +4.6% | −282% | −10.60% |
| OFFPEAK10: 10% off the post-alert high | 2,055 | +0.68% | [−0.17, +1.51] | +3.1% | −339% | +1.36% |

- **Entering at +4 h is the best of these, in 2020–25 and in 2025–26 alike.** Each delay gives
  up part of the fade (the median falls from +3.5% to +0.9% by +24 h), and the tail does not
  shrink: later runners make the worst trades worse.
- **Waiting for the fade to start (WEAK3, median trigger +5.5 h)** misses the first leg. The same
  trades entered at +4 h made +3.87%.
- **Selling the next leg up (SPIKE10)** picks the runners: the 38% of pumps that rise another
  10% lost −10.6% when shorted at +4 h. Shorting them at the spike instead still averages
  −0.2%.
- The signal is only known at +4 h (it needs hours 1–4 volume), so entering earlier is not an
  option with this rule.
