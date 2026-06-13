# iter-025 — spot-perp BASIS dislocation as a cross-sectional alpha (ONLINE-SOTA)

## STATUS: NO-CANDIDATE (failed the FIRST pre-check — univariate XS-IC ≈ 0)

## The idea (online-SOTA led; aimed at "fundamentally-different construction", not a price/funding overlay)
Instantaneous **spot-perp basis dislocation** — the perpetual trading rich/cheap vs its own
spot index — as a cross-sectional signal. Mechanism prior: a perp trading *rich* to spot
(positive basis) is over-bought by leveraged longs and tends to mean-revert / underperform
(ideal SHORT); a perp trading *cheap* is an ideal LONG. This is a **microstructure mispricing**,
structurally distinct from price-momentum (`mom30`), the XS mean-reversion `pred`, and the
iter-022/023 return-based tilts — and it is the *instantaneous* dislocation, distinct from
**funding** (iter-021), which is the time-integral / clamp of the basis.

**Citations (online research, May 2026):**
- "Perpetual Futures and Basis Risk: Evidence from Cryptocurrency", AEA 2026 program
  (aeaweb.org/conference/2026/program/paper/ByyFEfr4) — basis as arbitrage/mispricing signal.
- "Temporal Dynamics of Market Microstructure in Cryptocurrency Perpetual Futures",
  MDPI JRFM 14(5):103 2026 — spot-perp spread ≥20bps as arbitrage state (≈18.7% of obs);
  only ~40% of top spreads profitable after costs/reversal.
- "Revisiting the Bitcoin Basis" (CF Benchmarks 2025) — momentum/sentiment drivers of basis.
- "Fundamentals of Perpetual Futures", arXiv:2212.06888 — basis↔funding clamp identity.

## Data (free, already built — no new collection)
`outputs/vBTC_features_spot/spot_panel.parquet` (built earlier for vBTC): Binance spot+perp
basis features for **20 of the 70 HL70 symbols**, window **2025-07 → 2026-04** (5-min grid).
Features: `sp_basis_4h`, `sp_basis_z1d` (z-scored basis), `sp_retdiff_4h` (spot−perp 4h return
divergence), `sp_taker_imb_1d`. The 20-sym/partial-window coverage is itself the *favorable*
in-sample subset; a real build would need spot for all 70 + an EXT panel for G7 transport — which
we do NOT have. So a fail on this favorable subset is decisive.

## Pre-checks in fail-fast order (the decisive numbers)

### (1) Univariate XS-IC vs fwd 4h alpha-residual — **FAIL (signal ≈ zero)**
4h entry grid, 1,775 cycles, 20 symbols (Spearman, per cycle):
| signal | mean IC | t |
|---|---|---|
| sp_basis_4h | **+0.0023** | +0.34 |
| sp_basis_z1d | **+0.0030** | +0.54 |
| sp_retdiff_4h | **−0.0062** | −1.10 |
| sp_taker_imb_1d | **+0.0022** | +0.40 |
| pred (ref) | −0.0124 | −2.19 |

All four are statistically indistinguishable from zero (|t|≤1.1). For contrast, the signals that
*passed* the IC layer in this loop were 6–7× larger (rel_ret_1d −0.036 t−9.8; MAX −0.045 t−11.8).
Basis simply does **not forecast** the 4h cross-sectional alpha-residual.

### (2) Orthogonality to pred — trivially orthogonal (corr +0.001..+0.012)
…but orthogonal-to-zero is not a signal; this only confirms there is nothing to add.

### (3) DECISIVE construction-layer marginal (pred pools, 200 random seeds) — **FAIL (wash)**
Within the pred-conditioned long pool (LONG cheapest-perp K) / short pool (SHORT richest-perp K),
vs matched-random-K from the SAME pool:
| signal | LONG cheap-perp | rank | SHORT rich-perp | rank |
|---|---|---|---|---|
| sp_basis_4h | −2.11 bps | p17 | +0.67 bps | p90 |
| sp_basis_z1d | −1.20 bps | p62 | −0.96 bps | p14 |
| sp_retdiff_4h | −1.93 bps | p24 | +0.79 bps | p91 |

Best rank p90/p91, never ≥p95; sign-unstable across the three basis variants. No separable,
monetizable edge within the held-book pool — the same third-wall outcome as iter-022/023, but here
it is moot because the signal already died at the IC layer.

## Why it fails (mechanism — maps to a mapped wall, one layer earlier)
At the 4h horizon the spot-perp basis is a clamped, fast-mean-reverting microstructure quantity
whose only persistent cross-sectional content is the **funding premium it integrates into** — and
funding as an XS predictor was already walled in iter-021 (G7 sign-flip: HL70 IC +0.013 / EXT −0.011).
The *instantaneous* dislocation carries even less directional information than the integral: it
predicts the next-block convergence of basis itself, not the symbol's forward alpha-residual. So
basis collapses into the funding wall (regime/era-conditional, no stable XS sign on free data) and
fails the very first pre-check.

## Verdict
**NO-CANDIDATE — do not build.** Honest fail at the FIRST pre-check (univariate XS-IC ≈ 0, |t|≤1.1)
on the favorable 20-symbol in-sample subset, with the construction-layer test confirming a wash.
No engine change; champion unchanged. This also forecloses the "microstructure mispricing" angle on
free data: basis ⊂ funding wall.

## Map after iter-025
The online-research candidates surveyed this iteration all reduce to mapped walls:
- **spot-perp basis** (built + pre-checked) → funding wall (iter-021), IC≈0. ← tested, NO-CANDIDATE.
- **stablecoin exchange netflow / on-chain flows** (CryptoQuant/Glassnode) → PAID data (human key) AND
  a market-wide directional/leading signal = the DD-leading-coincident wall (iter-005/009/010).
- **return-dispersion factor timing** (Maio/Stivers) → a timing lever = the G4/iter-001/002 wall;
  crypto per-cycle IC predictability R²≈0.005 (iter-006) already shows regime timing is unforecastable.
- **lead-lag DTW (alts→BTC)** → wrong direction for a beta-neutral residual book; no XS application.

## Scripts
- `research/convexity_portable_2026-05-20/scripts/iter025_basis_precheck.py` (IC + orthogonality + construction-layer; the decisive fail)

## Sources
- https://www.aeaweb.org/conference/2026/program/paper/ByyFEfr4
- https://www.mdpi.com/2227-7072/14/5/103
- https://www.cfbenchmarks.com/blog/revisiting-the-bitcoin-basis-how-momentum-sentiment-impact-the-structural-drivers-of-basis-activity
- https://arxiv.org/pdf/2212.06888
- https://cryptoquant.com/asset/stablecoin/chart/exchange-flows/exchange-netflow-total (paid — for completeness)
