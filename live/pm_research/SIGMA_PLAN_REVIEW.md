# SIGMA_PLAN_REVIEW — implementation-readiness review

Object: `SIGMA_PLAN.md` at commit `80f823e` (`SIGMA_PLAN finalized:
feasibility before machinery`). Date: 2026-08-20. Cross-checked against the
current `BE_BELIEF_PLAN.md`, `MEASUREMENT_PLAN.md`, `SIGMA_DIAGNOSTICS.md`,
`contracts/contracts.yaml`, the MNAR collector repair at `15d8fc2`, and the
three existing blend experiments.

## Verdict

The plan is a large conceptual improvement over v1-v3: it asks what sigma is
for before choosing an estimator, separates the rolling-TWAP variogram from the
conditional settlement variance, fixes `w=60` while keeping a free fitted
`w_hat` diagnostic-only, moves cheap falsification ahead of machinery, and
requires a simple baseline before a multi-scale blend. Those decisions should
survive.

It is **not implementation-ready**. Six issues are load-bearing:

1. the estimand mixes price, relative-return and log-return units;
2. the consumer list omits BE-Belief's non-rare stream fallback and hardcodes a
   Gaussian link that the current belief contract forbids;
3. the anchor is scheduled for implementation before its feed semantics are
   verified, and its error variance is not identifiable from one noisy proxy;
4. the proposed book re-read does not survive the known MNAR CLOB loss merely
   because it is paired;
5. the live output is a forecast of future physical variance, not "realised
   sigma", while the proposed QLIKE labels are massively overlapping;
6. `c(r)` and H-3 are not yet well-defined or powered enough to be gates.

**Recommendation: HOLD estimator implementation.** Repair the Phase-0 spec,
freeze the typed output boundary, and run the deterministic/data-quality checks
first. Sigma remains a sensible next research focus, but only as a small
falsification-and-risk-plumbing track; it must not displace the queue/fill gate
that can still stop the programme.

## What should remain unchanged

- `w_declared = 60 s`; `w_hat_free` is diagnostic only and never changes the
  settlement kernel.
- Fit the physical-volatility process from the tape, never from Bernoulli
  winners. Outcome data is for calibration/falsification only.
- Keep the rolling-increment variogram used for estimation distinct from the
  conditional settlement-innovation law used by consumers.
- Keep the variance ledger explicit and audit it with standardised residuals;
  do not add `sigma_perp`, `kappa(r)`, anchor error and a calibration multiplier
  without named ownership and covariance semantics.
- Per-symbol level, a single-scale incumbent, tape-side walk-forward fitting,
  asymmetric uncertainty output, and a permanent H-3 stop rule are all good.
- The multi-scale blend is a challenger. If it does not beat the frozen simple
  baseline out of sample, do not ship it.

## MUST-FIX

### S1 — choose one unit space and make every equation type-correct

`SIGMA_PLAN.md:93-99` fits **log increments**, `:275-287` names `X` as the raw
settlement TWAP and writes `E[X_T]-X_0`, while the existing experiments use
`(E-X_0)/X_0`. The plan then reports `sigma_eff` in bps. A raw dollar numerator
cannot be divided by a bps denominator, and the H-3 inversion at `:185-188` has
the same defect.

Choose and name one model coordinate, for example:

```
x_t       = log(S_t / S_ref)              # dimensionless
e_r       = x_T - E_t[x_T]                # dimensionless
Sigma(r)  = Var_t(e_r)                    # return^2
sigma2    = variance rate per second       # return^2 / second
d         = (E_t[x_T] - x_0) / sqrt(Sigma(r))
```

If the anchor algebra must remain in arithmetic price space, declare the price
coordinate separately and define exactly where it is normalised; do not silently
replace `E[log X]` with `log E[X]`. `omega_P`, nuggets, strike error, `c(r)`,
standardised residuals and H-3 must all use the same coordinate and units.

Acceptance:

- every public variance field carries `unit_space` and `support`;
- dimensional tests reject raw-price/bps composition and cross-symbol pooling
  before normalisation;
- the reference probability calculation reproduces the chosen formula from one
  typed fixture at every horizon.

### S2 — freeze the real consumer contract before estimator machinery

The plan says sigma is not needed for the level, but the newer
`BE_BELIEF_PLAN.md:643-656` requires `sigma_eff` for the stream-forecast fallback
when the book is `Unavailable`; that condition is explicitly described as not
rare. The fallback is selected by book staleness/coverage, so its loss and
validation population differ from the main book path. It cannot be omitted from
the purpose statement.

The plan also defines `d_book = Phi^-1(p_book)` and downstream `phi(d)` even
though `contracts.yaml:396-400` says never hardcode Gaussian Phi and BE-Belief
adopts a logit recalibration. Dynamics must consume the selected `LinkFunction`
(`g_inv` and a typed derivative/density operation), or the architecture must
explicitly pin and version a Gaussian-only path law. A probability reparameterised
through probit is not evidence that the book follows Gaussian X-space dynamics.

Finally, `SIGMA_PLAN.md:626-629` says the architecture has no carrier but the
build order puts the contract fix last, after the variogram, blend, calibration
and link. That reverses the modularity goal.

Move the contract to Phase 0. At minimum it must identify:

```
symbol, as_of, fit_data_through, target_interval, horizon_domain,
unit_space, sigma2_rate, settlement_var(r), increment_var(h),
anchor_error_budget, calibration_curve, se_log_sigma,
w_declared, w_hat_free_diagnostic, link_id, coverage, provenance
```

Expose behavior through a typed `PathLaw`/uncertainty protocol rather than a
bag of scalar parameters. Define separate consumers for main-path dynamics,
fallback level, stand-down and H-3. The fallback must refuse when its own sigma
or target inputs are unavailable; it may not inherit book-sourced `d` after the
book has failed.

### S3 — verify and identify the anchor before implementing it

The current Phase 0 implements the anchor first and verifies S30/S60 semantics
second. If those streams are not synchronous trailing arithmetic means, the
helper implemented in step 1 is wrong by construction. Reverse those steps.

The statement that the `r=30 s` anchor is "exact" is too strong. The decomposition
of the known and future halves is exact; replacing latent spot with
`P_hat = 2*S30-S60` still assumes compatible feed windows, common event support,
synchronisation and a locally linear path. The extrapolation can amplify
asynchronous updates and noise.

`omega_P = Var(P_hat-P_t)` is also not identified by comparing `P_hat` with one
Binance perp and "subtracting the known basis". The latent Chainlink spot does
not exist as an observed truth, and time-varying basis plus Binance proxy error
remain in the residual. Define one of:

- an operational proxy-relative anchor error, with that proxy and its basis
  explicitly part of the estimand;
- a two-proxy/error-in-variables estimate with assumptions and sensitivity; or
- a conservative lower/upper error bracket propagated through the ledger.

Acceptance requires common-knowledge-time alignment of S30/S60, an age/skew
bound, semantics/reconstruction residuals by horizon and regime, and an explicit
covariance test. Independence of pre-`t` anchor error and post-`t` innovation is
a Brownian-model claim, not a type-system fact.

### S4 — pairing does not repair the MNAR book sample

Phase-0 step 5 says the corrected model-versus-book comparison "is paired, so it
survives the MNAR gap". It does not. Pairing makes the two forecasts share the
same **observed** rows; it does not recover the busy BTC intervals dropped by the
slow-consumer failure. If relative model/book performance changes with volatility,
staleness or activity, conditioning on observed rows biases the paired delta.

The newer belief plan independently finds that the snapshot-only top of book is
stale by more than the calibration effect and requires rebuilding it from
`price_change.best_bid/best_ask` plus snapshots. Therefore the old book series
cannot decide programme identity, even for a one-off descriptive re-read.

Required repair:

1. build the dense, knowledge-time top-of-book series;
2. stamp slow-consumer and other gap causes, with protected-span admissibility;
3. use only gap-complete legacy units or, preferably, post-repair collection;
4. report selection deltas by activity/volatility and keep the result
   `DESCRIPTIVE` until day-clustered inference is available;
5. compare against the walk-forward recalibrated book as well as the raw book.

The anchor comparison can still be paired on the settlement tape, but the
book-beat verdict cannot be said to "survive" missing book regimes.

### S5 — call the output a physical-volatility forecast and remove overlap power inflation

At decision time the next 300 s variance is unknown. A trailing estimator whose
weights minimise QLIKE against the **next** 300 s realised variation is an ex-ante
forecast of physical variance; the future tape quantity is its ex-post target.
Calling the live output "realised sigma" blurs the exact PIT boundary this plan
is trying to protect. "Not fitted to binary outcomes" and "not predictive" are
not synonyms.

Likewise, one-second labels for next-300-second variation overlap 299/300 of
their support. They provide many numerical rows, not thousands of independent
observations per day. The `63k ticks/day` language overstates level-fit power too:
non-overlapping 600 s increments yield only about 144 units per full day, and all
horizons share the same underlying path.

Specify:

- `forecast_as_of`, `target_start`, `target_end`, and `fit_data_through`;
- train-through-day `d-1`, day-block test folds and an embargo covering the
  longest label/support;
- either non-overlapping targets or overlap-aware sample weights and block
  inference;
- the effective sample size by horizon, not tick count;
- frozen QLIKE plus calibration and underprediction-tail diagnostics;
- a single-scale baseline evaluated on exactly the same folds.

No blend weight should be called per-symbol stable from the present two-day,
one-regime sample.

### S6 — make `c(r)` and H-3 identifiable gates

The finalisation predicts `c(30)` will breach by comparing the corrected-anchor
residual (about 2.6 bps) with the diffusion term (1.77 bps) **before measuring
the anchor-error term that Phase 0 explicitly adds**. That residual can be
evidence for `omega_P`, shape failure, or covariance; it is not yet evidence for
`c(r)`. Compute the ledger first.

Define the calibration estimand explicitly, for example:

```
c(r) = [ Var(e_r) - a(r)*omega_P^2 ] / [ sigma2_hat*k_law(r) ]
```

with the chosen unit space, covariance policy, non-negative residual behavior,
training cutoff and uncertainty. If instead `c(r)` calibrates total variance,
then it must multiply the entire ledger and may not simultaneously be described
as a diffusion-shape correction. The current prose and equation permit both
readings.

At two day clusters, a point estimate crossing `[0.8, 1.25]` is not an honest
go/no-go test. The plan itself says outcome CIs do not exist at this sample size.
Freeze the band and inference rule before the next read; until enough independent
days exist, a breach is a redesign diagnostic, not a programme gate. Also state
whether "20-30% SE" refers to `c`, `c-1`, variance or volatility; 20% relative
uncertainty in a variance multiplier does not become 5% merely because its mean
is near one.

H-3 needs domain rules. The inversion
`sigma_book = (E-K)/Phi^-1(mid)` is undefined near 0.5, can be negative when the
book and stream disagree in sign, explodes under tick quantisation, and currently
mixes units. Pre-register an admissible moneyness domain, use the selected link,
handle sign-conflict/censoring without conditioning away hard cases, and score
the final sigma forecast against future physical variation out of sample. The
recalibrated executable book is the required baseline.

## SHOULD-FIX

1. The exact discrete correction is only applied on the `r <= w` branch. Under
   the natural one-second endpoint convention, the `r > w` coefficient also has
   a roughly `+0.5 s` correction, and the `r=60` table does not match the stated
   discrete formula. Freeze the sampling convention and test continuity at
   `r=w`; small is not the same as exact.
2. Treat `t(4)` as an explicit conservative policy assumption, not "a standard
   crypto value". Name its owner and sensitivity range. The probability floor is
   a risk/decision policy and should not silently become evidence about the link.
3. PIT/body-CDF fitting must be walk-forward. A pooled empirical CDF checked on
   the same two days it was estimated from is descriptive fit, not calibration.
4. ZEC is a useful no-outcome control, not a statistically independent OOS
   asset: it shares the same date, crypto regime and estimator-selection process.
5. Replace "the one measured edge" with "a small forecast-calibration effect".
   The current Belief plan says it is a correctness module, that executable
   economics are presently indistinguishable from zero, and that selection
   destroys most of the midpoint gap.
6. Reconcile the purpose list: `sigma_perp` is called a surviving basis job in
   the executive section but is deferred to PM-E1 and explicitly excluded from
   the settlement ledger. That separation is valid, but the output/owner names
   must prevent a future consumer from adding it to settlement variance.

## Revised build order

### Phase 0A — definitions and deterministic checks

1. Freeze the consumer matrix, unit space, physical-forecast estimand, typed
   uncertainty contract and refusal behavior.
2. Verify S30/S60 window semantics, timestamp alignment and common-knowledge-time
   construction before writing the anchor helper.
3. Derive and unit-test the discrete kernels under the actual feed sampling
   convention, including continuity at `r=w`.
4. Implement the anchor only after 2 passes; define the proxy/bracket for anchor
   error and the covariance policy.

### Phase 0B — data admissibility and feasibility

5. Rebuild dense top of book; classify gaps and isolate post-MNAR-repair data.
6. Fit a frozen, per-symbol single-scale physical-vol baseline on tape folds.
7. Measure the complete variance ledger and then `c(r)` with block/day
   uncertainty. Do not use the present two-day point estimate as a gate.
8. Re-read fallback calibration and model/book scoring on admissible dense-book
   rows; label it descriptive until its day threshold is reached.

### Phase 1 — only after Phase 0 is coherent

9. Run the multi-scale QLIKE challenger on identical embargoed folds.
10. Add shrinkage only where the between/within-symbol evidence supports it.
11. Run walk-forward PIT/link diagnostics and propagate the frozen policy floor.
12. At the pre-registered data horizon, run H-3; a null permanently demotes sigma
    to risk plumbing.

## Acceptance summary

| item | minimum evidence before build/adoption |
|---|---|
| estimand | one unit space, typed dimensions, kernel fixtures pass |
| anchor | feed semantics + skew bounds + proxy-error bracket pass |
| data | dense book, cause-aware gaps, post-repair/admissible sample |
| baseline | per-symbol single scale, tape-side day-block OOS |
| `c(r)` | residual defined after ledger, frozen band + interval rule |
| blend | paired OOS QLIKE gain vs baseline, overlap-aware |
| fallback | separately scored on book-unavailable/stale population |
| H-3 | valid book-implied domain, recalibrated-book control, >=30 days |
| architecture | BE-Uncertainty module + typed PathLaw before estimator code |

## Final assessment

The plan found the right scientific question: the previous sigma work was
partly compensating for a mean/anchor error, and sophisticated variance machinery
should have to earn its existence. That is not marginal documentation progress;
it changes the research order and can save a full implementation cycle.

The next improvement is not more prose. It is a short executable Phase 0 with a
frozen typed estimand, honest data admissibility and one simple baseline. Until
those six MUST-FIX items close, building the variogram blend would recreate the
same failure pattern the plan correctly diagnoses: a statistically polished
estimator attached to an unstable target and an incomplete consumer contract.
