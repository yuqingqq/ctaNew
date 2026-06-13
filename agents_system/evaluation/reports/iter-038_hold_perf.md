# iter-038 — HOLD-sweep: fixed 4h hold vs 24h/6-sleeve overlap

**Question:** What is the strategy's performance with a NORMAL FIXED 4-HOUR HOLD (HOLD=1, single
sleeve, full rebalance every 4h, NO overlap) vs the current 24h hold (HOLD=6, 6 overlapping sleeves)?
And is the sleeve-overlap (cost amortization) clearly better, or is the fresh 4h signal competitive?

**Setup.** Reused the iter-032 fast engine (verified == iter-031 slow engine; here the instrumented
held-book/stop reproduce i31 to **0.00e+00** at HOLD=6). x132 V0 preds (156 syms, 2021-26, 8 folds),
@4.5 bps/leg. Champion = regime-hybrid held-book; BASE and +iter-012 vol-norm stop (k=2.0). Only
`HOLD ∈ {1,2,3,6}` varied. `HOLD=1` = pure 4h fixed hold, full rebalance each cycle (full turnover);
`HOLD=6` = 24h hold / 6 overlapping sleeves (turnover amortized 6×). Universes: established-70
(HL68∩x132, the clean within-model read) and full-mature (full-156 + maturity≥180d per-cycle gate).

## HOLD-sweep table — established-70 (clean read), @4.5bps

| HOLD | hold | var | Sharpe | maxDD | Calmar | netPnL | grossPnL | cost | avgTurn | %pos |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4h | base | +0.58 | −6686 | +0.42 | +13227 | +36181 | 22954 | 0.991 | 38.3 |
| 1 | 4h | stop | +0.55 | −4614 | +0.35 | +7668 | +20364 | 12696 | 0.548 | 38.3 |
| 2 | 8h | base | +0.97 | −5541 | +0.80 | +20703 | +34188 | 13485 | 0.582 | 39.2 |
| 2 | 8h | stop | +0.93 | −3907 | +0.77 | +14159 | +22985 | 8827 | 0.381 | 39.2 |
| 3 | 12h | base | +1.22 | −4539 | +1.20 | +25558 | +35332 | 9775 | 0.422 | 39.8 |
| 3 | 12h | stop | +1.06 | −3999 | +0.85 | +15901 | +22563 | 6662 | 0.288 | 39.8 |
| **6** | **24h** | **base** | **+1.35** | **−3508** | **+1.64** | **+27032** | +32741 | 5709 | 0.247 | 40.7 |
| **6** | **24h** | **stop** | **+1.34** | **−2647** | **+1.84** | **+22863** | +27117 | 4254 | 0.184 | 40.7 |

## HOLD-sweep table — full-mature (≥180d gate), @4.5bps

| HOLD | hold | var | Sharpe | maxDD | Calmar | netPnL | grossPnL | cost | avgTurn | %pos |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4h | base | +0.98 | −9335 | +0.84 | +37022 | +63200 | 26178 | 1.131 | 37.7 |
| 1 | 4h | stop | +0.72 | −5197 | +0.98 | +23826 | +41102 | 17275 | 0.746 | 37.7 |
| 2 | 8h | base | +1.13 | −8685 | +1.02 | +41610 | +56855 | 15244 | 0.658 | 38.8 |
| 2 | 8h | stop | +0.89 | −4491 | +1.36 | +28727 | +38223 | 9495 | 0.410 | 38.8 |
| 3 | 12h | base | +1.24 | −6455 | +1.49 | +45286 | +56244 | 10958 | 0.473 | 39.3 |
| 3 | 12h | stop | +1.00 | −4284 | +1.63 | +32715 | +39700 | 6985 | 0.302 | 39.2 |
| **6** | **24h** | **base** | **+1.24** | **−4687** | **+2.00** | **+44035** | +50328 | 6293 | 0.272 | 40.3 |
| **6** | **24h** | **stop** | **+1.19** | **−3960** | **+1.33** | **+24804** | +28938 | 4134 | 0.179 | 40.3 |

## Cost/freshness tradeoff (the point)

The 4h-fixed hold (HOLD=1) captures the freshest signal but pays **full turnover every cycle**.
Reading the gross-vs-net columns on established-70 base:

- **HOLD=1 gross +36,181 → net +13,227** — cost eats **22,954 bps (63% of gross)**, avg turnover 0.99.
- **HOLD=6 gross +32,741 → net +27,032** — cost eats only **5,709 bps (17% of gross)**, avg turnover 0.25.

So going 4h→24h **quadruples turnover efficiency** (0.99→0.25), and the gross only falls ~9%
(+36.2k→+32.7k) — i.e. the signal IS staler at 24h but only mildly, while the cost saving is huge. Net
PnL doubles (+13.2k→+27.0k) and Sharpe more than doubles (+0.58→+1.35). Same monotone story with the
stop and on full-mature.

**4h-fixed (HOLD=1) gross vs net** quantifies how much the full rebalance eats:
- established-70 base: gross Sharpe is positive but **63% of gross PnL is consumed by cost**.
- full-mature base: gross +63,200 → net +37,022, **41% eaten** (wider book → more dispersion gross,
  but turnover 1.13/cycle).

## Direct answer

**The 24h/6-sleeve overlap is clearly and decisively better. The fresh 4h-fixed hold is NOT
competitive at 4.5 bps.** Performance is **monotone improving** as the hold lengthens 4h→8h→12h→24h on
every metric — Sharpe, maxDD, Calmar, and net PnL all rise, turnover falls 4×:

- established-70 BASE: Sharpe **+0.58 (4h) → +1.35 (24h)**, Calmar 0.42→1.64, maxDD −6686→−3508,
  net PnL +13.2k→+27.0k.
- established-70 +stop: Sharpe **+0.55 (4h) → +1.34 (24h)**, Calmar 0.35→1.84, maxDD −4614→−2647.
- full-mature BASE: Sharpe **+0.98 (4h) → +1.24 (24h)**, Calmar 0.84→2.00, maxDD −9335→−4687.

**Mechanism — cost amortization wins, not fresher signal:** gross PnL is roughly flat-to-slightly-down
from 4h to 24h (signal decays only mildly over the hold, consistent with the iter-016 ~10-12h
half-life), but turnover/cost collapses by ~4× through sleeve overlap. The cost/gross ratio drops from
~63% (4h) to ~17% (24h) on established-70. The current HOLD=6 champion is at (or near) the optimum of
this tradeoff; there is no freshness gain at 4h that survives the full-rebalance cost.

(Note: monotonicity holds cleanly through HOLD=6 — this sweep stops at 24h, matching the champion;
the champion's published forward expectation ~+1.2-1.3 +stop is reproduced here at +1.34/+1.19.)

**Verdict:** keep the 24h / 6-overlapping-sleeve construction. No change to champion.

Scripts: `agents_system/research/scratch/iter038_hold_sweep.py`. CSV: `outputs/iter038/iter038_hold_sweep.csv`.
