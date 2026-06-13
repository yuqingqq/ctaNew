# iter-036 — WHY is >70 (wide) worse than established-70? COUNT vs COMPOSITION

Decomposes the iter-034 gap (established-70 +1.34 vs full-156 +1.03) into a COUNT effect (fewer
names better) vs a COMPOSITION effect (those specific 70 are special), and tests whether either is
ex-ante reproducible or just noise. All +stop @4.5bps, full 2021-26, same V0 x132 preds, folds 1-8.
Engine + cached struct_meta reused verbatim from iter-035 (no rebuild).

## References (match iter-034/035 exactly)
| panel | Sharpe | maxDD | Calmar | totPnL | fp |
|---|---|---|---|---|---|
| full-156 (naive) | +1.03 | −3960 | +1.16 | +21611 | 6/8 |
| established-70 (curated) | +1.34 | −2647 | +1.84 | +22863 | 7/8 |
| mature-wide (≥180d, nested-OOS, sz 35-142) | +1.20 | −2978 | +1.63 | +22864 | 7/8 |

## 1. COUNT within the MATURE pool — random-N draws (20 seeds/N, nested-OOS)
| N | Sh mean | p25 | p50 | p75 | p95 | max |
|---|---|---|---|---|---|---|
| 40 | **+0.89** | +0.71 | +0.89 | +1.06 | +1.33 | +1.47 |
| 70 | **+1.08** | +0.99 | +1.03 | +1.17 | +1.36 | +1.43 |
| 100 | **+1.15** | +1.09 | +1.15 | +1.23 | +1.28 | +1.28 |
| full (~140) | **+1.20** | (deterministic = mature-wide) |

Sharpe **RISES MONOTONICALLY with N** within the mature pool (+0.89 → +1.08 → +1.15 → +1.20).
Smaller mature panels are WORSE, not better. **There is NO count/dilution effect** — the >70 set is
not hurt by "too many mediocre names"; on the contrary, dropping mature names DESTROYS breadth/Sharpe.
The mediocre-mature names are NOT diluting alpha; the wider mature set has the best risk-adjusted
return of any random subset.

## 2. COMPOSITION — is the established-70 special?
- established-70 fixed Sharpe **+1.341**.
- random-70-from-mature: mean +1.082, p50 +1.034, **p95 +1.361**, max +1.432.
- **established-70 ranks p90** of random-70 (borderline "special"). BUT random-70 (mean +1.08) is
  itself WORSE than the full mature pool (+1.20) — i.e. *any* random 70-cut of the mature pool loses
  ~0.12 Sharpe vs trading all mature names. The established-70 is a slightly-above-average 70-cut
  that recovers (and marginally exceeds) the mature-wide point estimate.
- overlap: **65/68** established-70 names are in the last-fold mature pool (142 eligible) → the
  curated 70 is *almost entirely a subset of the mature pool*; its "specialness" is at most a mild
  above-median draw, not a distinct alpha source.

## 3. EX-ANTE structural capture (nested-OOS) — can a rule reproduce the curated-70?
| ex-ante rule | Sharpe | Calmar | totPnL | fp |
|---|---|---|---|---|
| top-40 by trailing cum-$vol | +0.81 | +0.79 | +11676 | 7/8 |
| top-70 by trailing cum-$vol | +1.02 | +1.11 | +15886 | 7/8 |
| top-100 by trailing cum-$vol | +1.17 | +1.27 | +18437 | 7/8 |
| **top-70 by listing-age (BEST)** | **+1.21** | +1.51 | +22753 | 7/8 |

- Best structural cap (top-70 by listing-age) = **+1.21 ≈ mature-wide +1.20**; it does NOT reach the
  curated-70 +1.34. It ranks **p75 of random-70-from-mature → does NOT beat random** (same as
  iter-035's p32 within-pool finding).
- $-volume capping is actively HARMFUL at small N (top-40 +0.81, top-70 +1.02) — confirms count
  effect from §1: any structural N-cut that shrinks the panel loses Sharpe.

## 4. NOISE — paired block-bootstrap CIs (per-fold resample, 2000 boots)
| pair | mean diff (bps/cycle) | 95% CI | verdict |
|---|---|---|---|
| best-exante − mature-wide | −0.011 | [−0.61, +0.59] | CROSSES 0 |
| established-70 − mature-wide | −0.000 | [−1.14, +1.07] | CROSSES 0 |
| established-70 − full-156 | +0.122 | [−1.57, +1.46] | CROSSES 0 |
| best-exante − full-156 | +0.111 | [−0.71, +1.03] | CROSSES 0 |

**Every gap crosses zero.** The established-70's +1.34 vs full-156's +1.03 (and vs mature-wide
+1.20) is statistically indistinguishable from noise. Confirms iter-035.

## VERDICT
- **NOT a count effect** — fewer mature names is *worse* (Sharpe rises monotonically with N within the
  mature pool). The >70 set is not diluted by mediocre names.
- **NOT a real composition effect** — the curated 70 ranks only p90 (borderline) of random-70, is 95%
  a subset of the mature pool, and its point-estimate edge over mature-wide is exactly 0.00 bps with a
  CI spanning [−1.14, +1.07]. It is a mildly-above-median mature subset, not a special alpha set.
- **NOT ex-ante reproducible beyond the maturity filter** — the best structural cap (age) only matches
  mature-wide (+1.21), fails the random-70 placebo (p75), and capping by $-volume hurts.
- The single mechanism behind everything: the iter-035 **maturity≥180d filter** lifts the naive
  full-156 (+1.03→+1.20) by dropping just-listed names with thin, unstable history. Beyond that floor,
  WHICH mature names you trade does not matter (random-pool ≈ curated ≈ structural-cap), and trading
  FEWER of them strictly hurts. The 70-vs-wide gap is within noise.

DEPLOYABLE STANDARD (unchanged from iter-035): trade the FULL mature pool (≥180d + hygiene + exec
liquidity floor + dedup), refreshed quarterly. Do NOT cap to a fixed N and do NOT try to reproduce the
curated-70 — neither adds honest Sharpe. The maturity filter is the entire transferable lever.
