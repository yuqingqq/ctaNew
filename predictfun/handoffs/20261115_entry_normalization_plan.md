# Normalization audit before the component check

User asked to review normalization before the planned fill/adversity/queue
component study. The component study has not launched. This is a read-only
audit, zero new historical model fits or candidates; labels, raw outcomes and
all frozen artifacts stay unchanged. No significance or economic-selection
claims, so no new permutation selection test is used.

Scope: the 325-input information study and the earlier CLEAN_TREE/CLEAN_RIDGE
preprocessor. They are different paths: the information study fits raw columns
with LightGBM; the cleanup uses economic transforms and train-only clipping,
imputation, scaling and column removal. A cleanup defect cannot directly explain
the raw study's failure.

Before seeing audit statistics, freeze these checks:

- All seven existing days: units, side aliases, price/spread bounds, positive
  sigma/liquidity denominators, unknown-reference masking, actual expiry versus
  research cutoff time, share/lot and bps/volatility transformations, bounded
  fill fractions, exact economic target decomposition. Float32 expiry checks
  allow 3e-5 seconds of rounding; cutoff checks allow 0.01 ms.
- Reconstruct each saved cleanup preprocessor from its original training data
  and require exact saved metadata/column equality. Test transformation must
  not modify fitted statistics. Folds are strictly chronological.
- Measure later missing/extreme flags discarded for being constant in training;
  measure columns discarded as training duplicates that diverge on later data.
  Keep action counts, inverse-sampling weighted fractions, affected columns and
  saved aliases. A constant training flag cannot teach a new response; retaining
  its status supports monitoring/fallback, not a guaranteed predictive fix.
- Compare unweighted fitting statistics with empirical quantiles weighted by
  the existing inclusion weights (2 initial / 32 later). Describe the difference
  rather than declaring every unweighted transform invalid. Report clipping
  counts and weighted mass; retain all labels and rows.
- Compare train/test marginal KS distances for public sizes in fixed lots versus
  ratios to the existing causal liquidity scale, and returns in bps versus
  known-volatility units. This measures distribution shift only; it is not a
  predictive-performance comparison or choice of a new candidate.
- Record how many actions occur before 25 seconds: the loader provides at most
  five seconds of pre-window books, so these actions cannot have a full trailing
  30-second liquidity history. The median is frame-weighted, not time-weighted.

Data: the existing 13,911 eligible label probes and five consumed evaluation
days (Oct 27/31, Nov 1/2/3). All source exclusions remain in receipts. Every
statistic carries its population and as-of. No confidence intervals, forward
claim, target change, model promotion or component test.

Synthetic controls: missing/extreme flag loss, accidental duplicate divergence,
weighted population shift, test-only transform immutability, unit conversion
versus state-dependent scaling; known-bad schemas/weights must refuse. Source
and artifact hashes are frozen before historical audit. One Codex light slot,
2 GiB self-guard; the older size supervisor resumes after success or failure.
