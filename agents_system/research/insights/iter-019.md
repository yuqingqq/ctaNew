# iter-019 — Research insight (ONLINE-SOTA-LED)

## Mandate
Lead with online SOTA; the in-house data-variant space is mapped (18 iters). Find a
GENUINELY NEW optimization on top of the FIXED baseline (HL70 regime-hybrid held-book +
adopted iter-012 vol-norm reactive stop). Champion alpha unchanged.

## Online research (2024-26 literature)

### Idea A — Meta-labeling (López de Prado) — PRE-CHECKED, REJECTED before build
- **Sources:** [Meta-Labeling, Wikipedia](https://en.wikipedia.org/wiki/Meta-Labeling);
  [Hudson & Thames, "Does Meta Labeling Add to Signal Efficacy?"](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/);
  [QuantConnect, "Why Meta-Labeling Is Not a Silver Bullet" (Baldisserri)](https://www.quantconnect.com/forum/discussion/14706/why-meta-labeling-is-not-a-silver-bullet/).
- **Mechanism hoped-for:** a secondary classifier learns P(a selected leg is a profitable
  trade) and filters/sizes legs. The primary (per-symbol Ridge `pred`, z-scored per symbol)
  is blind to cross-sectional/basket context and market state, so a meta-model could in
  principle add orthogonal info — especially on the long leg (known 47.8% hit vs short 57.4%).
- **The killer caveat (Baldisserri):** meta-labeling only helps if the secondary model sees
  information the primary does **not** ("can't squeeze the same orange twice"). For
  ML-on-identical-data it adds ~nothing.
- **Pre-check (`iter019_metalabel_precheck.py`, 18,800 legs):** built the real K=5/K=5 legs,
  labeled each by signed realized alpha-residual, tested candidate meta-features
  (pred rank-gap, XS pred dispersion, XS pred skew, n_eligible, |pred|, side) for
  **conditional IC after residualizing out the pred-decile mean** (the orange-squeeze test):

  | feature | raw IC | residIC (after pred) | residIC (after pred×side) |
  |---|---|---|---|
  | rank_gap | +0.008 | −0.008 | −0.009 |
  | xs_disp  | +0.017 | −0.001 | −0.007 |
  | xs_skew  | −0.021 | −0.012 | −0.006 |
  | n_elig   | −0.014 | +0.003 | +0.013 |
  | abs_pred | +0.011 | −0.008 | — |
  | **is_long** | **−0.067** | **−0.061** | (removed by construction) |

  The ONLY real signal is `is_long` (shorts beat longs) — that is the **already-known
  structural side asymmetry**, and acting on it = net-short / asymmetric-K, which iters 008
  and the vBTC ASYMK phase already REJECTED. Every cross-sectional meta-feature collapses to
  |residIC| ≤ 0.013 (noise). Confirmatory: long-leg fwd-alpha IC vs PIT trailing alt30 = −0.044
  with a NON-monotone quintile pattern (−1.8/+9.9/+9.3/−13.2/−3.5) — the long underperformance
  is NOT forward-separable, same coincident-not-leading alt-bear wall as iters 006/007/008/010.
  **Verdict: meta-labeling is dead on this book — the secondary model has no information the
  primary lacks.** (This is itself a clean SOTA-grounded confirmation of the i6/i16 wall.)

### Idea B — Transaction-cost-aware no-trade band — PRE-CHECKED, VIABLE → PROPOSED
- **Sources:** [Baldi-Lanfranchi, "Transaction-cost-aware Factors", FoFI 2024](http://wp.lancs.ac.uk/fofi2024/files/2024/04/FoFI-2024-163-Federico-Baldi-Lanfranchi.pdf);
  ["Cost-aware Portfolios in a Large Universe of Assets", arXiv:2412.11575 (2024)](https://arxiv.org/html/2412.11575).
  Finding: choosing **rebalancing speed** optimally (proportional turnover penalty / no-trade
  region) beats cost-agnostic rebalancing **even with a suboptimal alpha model** — i.e. the
  win is a pure cost-efficiency lever, not an alpha claim.
- **Mechanism for THIS book:** the held-book is 6 overlapping sleeves, K=5/side. Names near
  the rank cutoff churn in/out every cycle, generating small rebalances that pay cost with
  ~zero signal (iter-016 showed sleeves 2-6 hold *stale, mildly-anti-signal* positions). A
  per-symbol **no-trade band** — only execute a weight change if |target − held| ≥ δ — suppresses
  this rank-boundary churn. This is NOT a prediction (no G4 placebo wall) and NOT the rejected
  vBTC Phase-K pred-unit cost-margin swap (that gated *swaps by pred-lift*, a tuned alpha
  threshold that died nested-OOS); this gates the *realized net-weight change* δ, a structural
  turnover reducer.

- **Pre-check #1 — turnover anatomy (`iter019_notradeband_precheck.py`):** trade sizes cluster
  at ~0.033 (≈17% of a full leg = one sleeve-rotation step). **56% of trades are small churn,
  = 45% of total turnover** → a large cost pool with little signal content.

- **Pre-check #2 — band sweep @4.5bps (full leg = 0.200):**

  | band δ | Sharpe | totPnL | maxDD | turnover | grossPnL |
  |---|---|---|---|---|---|
  | 0.000 (base) | +1.93 | +10,472 | −5,674 | 800 | +12,272 |
  | 0.005 | +1.95 | +10,587 | −5,646 | 795 | +12,375 |
  | 0.010 | +1.95 | +10,600 | −5,627 | 792 | +12,381 |
  | **0.020** | **+1.97** | **+10,663** | **−5,512** | 790 | +12,441 |
  | 0.030 | +1.91 | +10,403 | −5,885 | 736 | +12,058 |
  | 0.050 | +2.31 | +12,501 | −4,585 | 491 | +13,605 |
  | 0.080 | +1.02 | +6,229 | −9,045 | 341 | +6,996 |

  Two regimes: (i) **small band δ≈0.005–0.02 is cost-only** — turnover down slightly, **gross
  PnL ~flat/up** (+12,272→+12,441), Sharpe/Calmar up modestly, maxDD down. Honest saving.
  (ii) **δ=0.05 looks spectacular (Sharpe +2.31) but gross PnL JUMPS to +13,605** → it is no
  longer cost-only, it DRIFTS the executed book (changes the bet). Treat as suspect.

- **Pre-check #3 — per-fold robustness (the discriminator):**
  - **δ=0.02: total +191 bps, folds_positive 6/7** (+6/+20/+4/+133/+44/+1/−18) — broadly
    distributed, NOT 1-2-fold concentrated. Survives the cheap robustness check.
  - δ=0.05: total +2029 bps but **3/7 folds**, entirely carried by folds 5+6 (+702,+1429),
    LOSES folds 1,2,3 (−200/−66/−173) — classic overfit signature; would die LOFO/nested-OOS.

## Conclusion
**ONE candidate proposed: a small transaction-cost-aware no-trade band (δ on net-weight
changes), chosen by nested-OOS, with the large-band region explicitly fenced as overfit.**
Genuinely new (cost-efficiency layer, never tried on this engine), 2024-SOTA-grounded, clean
mechanism, and survives the cheap per-fold pre-check. Modest expected lift (Calmar +1.68→~+1.71
at δ=0.02) — but it is the first honest, non-prediction improvement lever found since the
reactive stop. Meta-labeling (Idea A) honestly closed.
