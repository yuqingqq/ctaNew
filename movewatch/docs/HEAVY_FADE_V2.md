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
