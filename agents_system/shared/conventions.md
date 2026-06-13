# Coding & measurement conventions (all agents must follow)

## Point-in-time (PIT) — non-negotiable
- Rolling stats use **trailing** windows only. Z-scores / betas `.shift(1)` (or `.shift(horizon)`
  for h-bar forward labels). Never use the current or future bar in a feature.
- Labels (forward returns / alpha) are realized at `exit_time`; purge them from any training fold
  whose features precede `exit_time`. Walk-forward with 1-day embargo.
- Preprocessing (winsor, rank-transform, scaling) is **fit on train only**, applied to test.
- A sign-prediction / efficacy signal computed from realized IC must be lagged by the label horizon
  (HOLD) — the most recent realized alpha at decision time t is from cycle t−HOLD.
- IC > +0.10 on a new feature is a red flag for hidden look-ahead — investigate.

## Backtest measurement
- 4h cycles, disjoint-window annualization √(6·365). For horizon h bars: √(365·288/h).
- Held-book net PnL per cycle = Σ wᵢ·retᵢ − turnover·0.5·cost. Cumulative bps equity = cumsum.
- maxDD = min(equity − equity.cummax()). Calmar = ann_return_bps / |maxDD|.
- Placebo: ≥100 seeds, matched to the treatment's exposure (basket size / skip rate / flip rate).
- Bootstrap: block by fold for paired CIs.

## Code style
- Match existing `research/convexity_portable_2026-05-20/scripts/X*.py` style: self-contained
  script, `main()`, `flush=True` prints, parquet caches, absolute repo paths.
- Reuse existing helpers (`load_close`, `ann`, held-book loop, `X70` pipeline functions,
  `x6`/`x6b`) rather than reimplementing. Don't reinvent the engine.
- Keep new scripts runnable in <10 min where possible; cache expensive intermediates.
- Never modify the baseline-defining scripts (X116/X117) or the cached baseline preds.

## Data locations
- klines: `data/ml/test/parquet/klines/{sym}/5m/`
- panels: `outputs/vBTC_features/`
- preds & caches: `research/convexity_portable_2026-05-20/results/_cache/`
- HL70 preds: `.../_cache/x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet`

## Scope
Research code only. No live-trading/execution code. Free public data unless a paid feed is
explicitly approved by the human (e.g. Glassnode/Deribit for an orthogonal-data experiment).
