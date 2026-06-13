# iter-020 — Absorption Ratio (correlation-eigenstructure fragility) as a LEADING DD signal

## STATUS: NO-CANDIDATE (honest dead-end; structurally new observable, empirically same wall)

## Online SOTA research (this is the construction/risk-model lead the directive pointed at)
Directive: explore portfolio-construction / weighting beyond rank-K-equal-weight (HRP, min-var,
covariance shrinkage, eigenvalue clipping, risk-model construction). Searched 2023-26 work:

1. **HRP / RL-Bayesian-HRP for crypto** — López de Prado HRP and 2024-25 extensions.
   - alexeygolev.blog HRP-for-crypto (2022-25 backtest): HRP variants do NOT beat BTC-hold once
     costs counted; HRP is a *long-only diversified allocator*, not a L/S alpha lever.
   - arXiv:2508.11856 (RL-Bayesian-HRP, US equities 2020-25 OOS).
   - VERDICT for our book: HRP weights by the *dendrogram of the covariance* → on a 25-active
     L/S book it collapses toward inverse-vol-on-clusters. We KNOW inverse-vol HURTS (alpha lives
     in volatile alts) and sector/cluster features HURT (iter-F/G in MEMORY). HRP = those two
     dead levers combined. Dead on arrival.

2. **Ledoit-Wolf / nonlinear (QIS) covariance shrinkage, NN covariance cleaning** —
   ledoit.net Goldilocks (RFS 2017); arXiv:2507.01918 (E2E NN covariance cleaning, US equities
   2000-24: lower realized vol, smaller maxDD, higher Sharpe vs SOTA).
   - VERDICT: min-variance / max-diversification on a shrunk covariance = a *variance-minimizer*.
     Our edge is a per-symbol z-scored alpha-residual; min-var sizing fights the alpha (down-
     weights the volatile alts that carry it = inverse-vol again). beta-neutral leg sizing already
     in the side book. Naive risk-weighting is the dead lever. Need a mechanism that's genuinely
     different.

3. **Eigenvalue clipping / market-mode removal / Absorption Ratio** — THE genuinely-new thread.
   - Detrended cross-correlations RMT, crypto (PMC12731959, 140 coins 2021-24): λ1 (market mode)
     of the correlation matrix swells in stressed windows ("exhausts a large part of the trace").
   - **Kritzman, Li, Page, Rigobon (2010) "Principal Components as a Measure of Systemic Risk" —
     the Absorption Ratio (AR)**: fraction of total cross-sectional variance absorbed by the top
     eigenvector(s) of the trailing return-correlation matrix (SSRN 1633027; portfoliooptimizer.io
     write-ups). Claimed to be a **LEADING indicator**: "significant increases in the AR are
     followed by significant stock market losses." HONESTY FLAG from the practitioner write-up:
     "AR tends to spike *during* crashes and remain elevated afterward, **with mixed evidence for
     pre-crash signals**." So leading-ness must be TESTED, not assumed.

## Why AR was worth a pre-check (mechanism for THIS book)
The iter-006 root cause of the −57% DD is a **correlated alt deleverage = the market mode of the
universe correlation matrix swelling** (BTC-only regime can't see it; the per-sym z-target removes
the market direction the loss rides on). The AR measures *exactly that* — eigenstructure
concentration, a CORRELATION-STRUCTURE signal, structurally DIFFERENT from every prior rejected
free observable, which were all *levels/trends*: implied-vol level (i5 DVOL), price level/trend
(i7), net-short trend (i8), positioning/OI level (i9), fast price (i10). None of those tested
correlation concentration. If anything free could LEAD this DD, AR is the strongest prior.

## Data pre-check (iter020_absorption_ratio_precheck.py — 157s, HL70 baseline engine)
Built a fully-PIT Absorption Ratio on the 4h universe log-returns (trailing corr matrix,
first-eigenvalue share AR1 and Kritzman N≈n/5 share AR_n5; z-score and 15-bar Δ variants; all
lagged to t−1). Tested two questions:

**(A) LEAD-LAG (the test that killed i5/i9/i10) — Spearman IC of lagged AR vs book PnL:**
| feature | vs PAST PnL | vs FWD-24h | vs FWD-1cyc | window |
|---|---|---|---|---|
| AR1 | **−0.105** | −0.092 | −0.030 | 180-bar |
| AR_n5 | −0.119 | −0.109 | −0.035 | 180-bar |
| AR1 | **−0.123** | −0.115 | −0.038 | 60-bar (faster) |
- **AR is COINCIDENT, not leading.** IC vs PAST PnL is stronger than IC vs FWD-24h in every variant
  and window; IC at the actual trade horizon (next cycle) is ~−0.03 ≈ noise. Same signature as DVOL
  (i5: past −0.259 > fut −0.228) and positioning (i9). The correlation concentration rises WITH /
  AFTER the deleverage, not before — confirming the literature's "mixed pre-crash" caveat for this
  market. The endogenous reflexive-liquidation wall (i10) applies to eigenstructure too.

**(B) PRE-CHECK-G4 / R4 — AR-fragility de-gross (top-tercile AR → gross 0.40) vs matched random
de-gross of equal %-time (200 seeds), left-tail / maxDD cap:**
| window | q=0.70 rank | q=0.80 rank |
|---|---|---|
| 180-bar | p30 | p78 |
| 60-bar | **p94** | p70 |
- ALL cells **< p95**. The one near-miss (60-bar q=0.70, p94) is unstable — the SAME cell is p30 at
  the 180-bar window. Rank swings p30↔p94 with an arbitrary window knob = "tune the window until a
  cell looks OK" overfit, not a robust tail-selector. A matched RANDOM de-gross caps the tail as
  well or better → same "run smaller" wall as i1/i9, already served (better, R6 3/3) by the adopted
  iter-012 vol-norm reactive stop.
- The Sharpe "lift" (+2.06/+2.18) is the familiar lower-exposure→higher-Sharpe artifact (random
  de-gross matches it), NOT tail skill.

## Verdict + lesson
**NO-CANDIDATE.** The Absorption Ratio is the most on-mechanism free observable tested in the whole
run (it measures the DD root cause directly via correlation eigenstructure), and it STILL fails:
(A) it's coincident at the 24h trade horizon (fails the alpha-track lead-lag premise) and (B) an
AR-fragility de-gross has no edge over matched-random de-gross (fails PRE-CHECK-G4/R4). 

LESSON (sharpens i5/i9/i10): the wall is not "we tested the wrong feature" — it's that the
correlated alt deleverage is an *endogenous reflexive cascade* whose every free trailing imprint
(price, vol, positioning, AND now correlation eigenstructure) is COINCIDENT. Even the
correlation-STRUCTURE signal, which by construction sees the market-mode that IS the loss, sees it
only as it happens. The construction/risk-model SOTA family (HRP, min-var, shrinkage) reduces to
the already-dead inverse-vol/sector levers on this L/S book; eigenvalue-clipping's fragility
signal (AR) is coincident. Champion UNCHANGED (baseline Calmar +1.68 + optional iter-012 stop).

## Scripts
- `research/convexity_portable_2026-05-20/scripts/iter020_absorption_ratio_precheck.py`
  (lead-lag + PRE-CHECK-G4/R4; AR_WIN env-tunable; reuses X117 held-book engine).

## Sources
- Kritzman, Li, Page, Rigobon (2010), Principal Components as a Measure of Systemic Risk:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1633027
- Absorption Ratio write-up (formula, percentile-exposure use):
  https://portfoliooptimizer.io/blog/the-absorption-ratio-measuring-financial-risk/
- Detrended cross-correlations & RMT, crypto market mode (PMC12731959):
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12731959/
- López de Prado HRP for crypto (cost-aware backtest):
  https://alexeygolev.blog/hierarchical-risk-parity-hrp-for-crypto-portfolio-optimisation/
- Ledoit-Wolf nonlinear (Goldilocks) shrinkage: http://www.ledoit.net/Goldilocks_RFS_2017.pdf
- E2E NN covariance cleaning (min-var, 2025): https://arxiv.org/html/2507.01918v2
- Momentum-weighting unhelpful vs equal-weight (conviction-weighting): ReSolve DAA Part 4,
  https://investresolve.com/dynamic-asset-allocation-for-practitioners-part-4-momentum-weighting/
