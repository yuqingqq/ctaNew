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
