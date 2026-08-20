# SIGMA_PLAN_REVIEW_ITER2 — review of the sigma spec repair

Object: commit `cc1d0e7` (`sigma spec repair: all six MUST-FIX applied, and
the anchor ledger is closed-form`), based on tracking commit `46d3e90` and the
first review at `6bea435`. Date: 2026-08-20.

Reviewed artefacts:

- `SIGMA_PLAN.md` Revision 2;
- `sigma_kernels.py`;
- `contracts/contracts.yaml` v13 and its two v12→v13 migrations;
- the P-2026-003 status and handoff changes.

## Verdict

This is a meaningful repair and should not be reverted. It chooses a coherent
normalised arithmetic-return coordinate, fixes the mixed discrete/continuous
kernel at `r=w`, makes `c(r)` diagnostic rather than a two-day gate, restores the
MNAR qualification, names the live output as a physical-volatility forecast,
puts a simple incumbent before the blend, and adds an uncertainty boundary
before estimator machinery.

The commit's headline is nevertheless too strong. **The six prior items are not
all closed, and the anchor ledger is not a conditional variance ledger.** The
new kernel computes the unconditional mean-squared error of one chosen
extrapolator under an assumed one-second Brownian sampling model. It does not
compute the conditional mean and conditional variance given the information
actually available at decision time. Four contract/API defects and several
stale v1 claims then make the specification internally inconsistent.

**Decision: HOLD remains. Phase 0A steps 1–3 cannot be marked DONE.** Keep the
useful code as a model fixture, but do not build the estimator or treat the new
`2.44 bps` row, `c(30)=1.14`, or the `floor/ceiling` fields as pricing inputs.

## Executable audit

The shipped commands pass:

```
python3 live/pm_research/sigma_kernels.py --selftest                 PASS (13)
python3 live/pm_research/contracts/contract_check.py --selftest     PASS (13)
python3 live/pm_research/contracts/contract_check.py 46d3e90 HEAD   PASS
python3 -m py_compile live/pm_research/sigma_kernels.py              PASS
```

The v12→v13 contract diff has two authorised removals and 49 additions, with no
unexplained removal/change. The tests establish that the code matches its own
discrete-BM assumptions. They do not establish those assumptions against the
Chainlink streams or establish a conditional forecast distribution.

Adversarial probes expose missing refusal behavior:

```
k_law(-1)                  -> 0.0
anchor_error_coeff(-1)     -> 0.005662...
anchor_error_coeff(30.4)   -> silently rounds the known-window length
settlement_var(..., nugget=5) != ledger(...)["total"]
```

The contract promises refusal outside the horizon domain, but the only
executable public functions accept invalid and off-grid horizons.

## MUST-FIX

### M2-1 — `a(r)` is unconditional anchor MSE, not conditional settlement variance

Revision 2 defines

```
Sigma(r) = Var_t[x_T - E_hat_t[x_T]] = sigma²*k_law(r) + sigma²*a(r)
```

and calls the second line exact because `P_hat=2*S30-S60` is a fixed linear
functional of the latent path. Fixed linearity is enough to compute its
**unconditional MSE**. It is not enough to make `P_hat` the conditional mean
given the observed S30/S60 values.

Under the commit's own one-second discrete Brownian convention, restrict to a
translation-invariant endpoint estimate

```
P_t_hat(alpha) = S60 + alpha*(S30-S60).
```

The Brownian linear projection gives:

```
alpha*                         = 1.499167...
E[P_t | S30,S60]               ≈ 1.499*S30 - 0.499*S60
conditional residual variance  = 8.2590 sigma²
plan extrapolator               = 2.000*S30 - 1.000*S60
plan unconditional MSE          = 9.5139 sigma²
```

If the process start is treated as known, the unconstrained coefficients are
`1.701/-0.836` with residual variance `8.091 sigma²`; either way, `2/-1` is not
the Brownian conditional mean. It is a local-linear trend extrapolator imposed
on a path model whose trajectories have no local derivative.

Consequences:

- the forecast has a state-dependent conditional bias tied to `S30-S60`;
- that bias belongs in the numerator/mean model, not silently in a zero-mean
  variance line;
- `a(r)` combines conditional variance and squared conditional bias;
- the probability link is centered incorrectly even if its unconditional Brier
  improves on one rally day;
- the Monte-Carlo independence check creates the future draw independently by
  construction, so it cannot validate the real-data covariance assumption.

Choose explicitly between two legitimate specifications:

1. **Conditional model:** derive `E[P_t | information_t]` and
   `Var(P_t | information_t)` under a declared state model, then propagate both
   through the settlement kernel; or
2. **Operational forecast-error model:** retain `2*S30-S60` as a candidate
   anchor, call `a(r)` unconditional/state-conditional forecast MSE rather than
   `Var_t`, and estimate its bias, variance and covariance out of sample.

Do not call the ledger closed until the mean and variance are separated.

### M2-2 — the code freezes a one-second oracle convention before semantics are verified

`sigma_kernels.py:22-25` calls 60 equally spaced one-second samples the
"ACTUAL feed sampling convention" because publications arrive at roughly 1 Hz.
Those are different facts. The relay's publication cadence does not prove how
Chainlink constructs its internal 30 s or 60 s aggregate. EXP-M6 verifies that
the published S60 endpoint reproduces settlement; it does not prove a 60-point
rectangular kernel, synchronous S30/S60 support, or equal one-second weights.

The revised plan itself correctly leaves S30/S60 semantics as Phase-0A step 5
and says failure changes the helper. Therefore step 3 cannot simultaneously be
DONE with an "exact" discrete kernel and exact coefficient `9.5139`.

Required repair:

- represent the averaging kernel as a versioned weight schedule/sampling
  convention, not module constants `DT=1`, `W_DECLARED=60`;
- verify window endpoints, sample weights, update triggers and S30/S60 event-time
  alignment before choosing the discrete or continuous implementation;
- keep discrete-1s and continuous kernels as sensitivity fixtures until then;
- never infer internal samples from outward message frequency;
- only mark the kernel frozen after the semantics test passes.

### M2-3 — `floor/ceiling` is not an ordered bracket

The model-implied `9.5139 sigma²` is a useful reference MSE for the fixed
`2/-1` extrapolator under Brownian motion with synchronous noiseless averages.
It is not a distribution-free lower bound. A different path law, an optimal
conditional anchor, serial dependence, or negatively correlated feed error can
produce lower MSE. Calling it a floor requires assumptions that are neither
typed nor tested.

Likewise, the S30/S60-versus-Binance residual is not automatically an upper
bound on latent Chainlink anchor error. It equals a mixture of anchor error,
time-varying basis and proxy error. Covariance can increase **or decrease** its
variance. Subtracting a mean basis does not order the variances.

The scalar `omega_scale` is also insufficient. Asynchrony/noise on S30 and S60
has a 2×2 covariance matrix; its contribution changes with the horizon-specific
linear weights and generally is not a scalar multiple of the Brownian `a(r)`
curve.

Replace `floor/ceiling` with non-ordered model/proxy diagnostics until ordering
is justified, or construct a conservative bound from explicit bounds on both
proxy errors and their covariance. Propagate the S30/S60 error covariance
through the horizon weights. Consumers must not use `min/max` semantics on the
current fields.

This also invalidates the tracking claim that the bracket "supplies S3" now.
S3 remains open.

### M2-4 — the executable ledger has a hidden third line and no domain enforcement

`settlement_var` documents a closed two-line ledger but returns:

```
sigma²*k_law(r) + sigma²*a(r)*omega_scale + nugget
```

`ledger()` omits `nugget`. Thus two public functions disagree on total variance,
and a third component enters pricing without ownership, support, covariance or
double-count analysis. A variogram nugget can be observation noise, feed noise,
or small-scale process variance; those have different mappings into conditional
settlement uncertainty. It cannot be appended as a horizon-constant scalar by
default.

In addition:

- negative `r` is accepted;
- non-integer `r` is passed into a discrete kernel while the anchor path silently
  rounds `w-r`;
- `w<=0`, invalid fast/slow windows and out-of-grid horizons are not rejected;
- `k_law` and `anchor_error_coeff` therefore need not describe the same temporal
  support;
- the contract promises `Unavailable`, but the functions return floats.

Either remove `nugget` from `settlement_var`, or register it as a named third
component with a declared estimand and propagation law. Make one ledger function
the single source of truth, add exact domain/grid validation, and test refusal
paths as well as happy-path algebra.

### M2-5 — contracts v13 cannot enforce the consumer boundary it claims

The new carrier is a good direction, but several fields and module edges are
wrong or incomplete:

1. `BE-Uncertainty.consumes: RawEvent` bypasses the DA/StateView knowledge-time
   seam even though it also declares a `state_view` port. A BE module should
   consume knowledge-truncated normalised state, not raw landing-zone events.
2. `PathLaw.link: LinkFunction` duplicates `BeliefProcess.link`. BE-Uncertainty
   neither owns nor consumes the selected link, and no equality invariant ties
   the two copies. Use one immutable `LinkRef`/ID owned by the belief/spec plane.
3. The plan requires `g_inv` **and a derivative**, but `LinkFunction` exposes
   only `g` and `g_inv`. The dynamics consumer remains unimplementable.
4. The plan says every artefact carries `target_start` and `target_end`; PathLaw
   contains neither.
5. "Refuse after `fit_data_through`" is backwards. A walk-forward forecast is
   intentionally used after its fit cutoff. The required invariant is
   `fit_data_through < as_of`, with no training-data read beyond the cutoff.
6. The protocol accepts only `r`, so it cannot enforce `as_of`, fit cutoff,
   target interval or knowledge time.
7. `w_hat_free_diagnostic` is a `float`, not `Duration`; variance-returning
   methods are untyped `float`; `CalibrationCurve.band`, intervals and inference
   rule are free strings; `AnchorErrorBudget` does not say whether its scalars
   are bps², multiples of sigma², or horizon coefficients.
8. `HorizonDomain.basis_crossover` is required even though the basis study is
   explicitly deferred; there is no `NullPin|Unavailable` branch.

Add a typed forecast request/context carrying `as_of`, target interval and
knowledge cutoff; consume it through StateView. Make link identity single-copy,
add the derivative/density operation, fix temporal semantics and type the
variance/calibration quantities rather than relying on notes around floats and
strings.

### M2-6 — Revision 2 is an overlay, not yet one coherent canonical plan

The head block corrects many v1 decisions, but contradictory live text remains:

- `:269-275` says the book has beaten the model three times and candidate (a) is
  out, while `:178-184` and `:770-784` say the verdict is unadjudicated/MNAR;
- `:311-322` still uses `Phi^-1`, raw `E[X]-K`, and calls for "realised sigma";
- `:373-374` still calls the r=30 estimate exact with no linearity assumption;
- `:542`, `:584`, and `:591-592` still argue from 63k ticks and thousands of
  observations despite S5 saying that language was struck;
- `:574-578` retains the old `c*diffusion + weight*omega` equation rather than
  the new horizon coefficient/bracket representation;
- `:626` again says "realised sigma";
- `:669-670` repeats the backwards fit-cutoff refusal;
- old `Phi/phi` formulas survive in the consumer table and risk list after the
  selected-link correction.

The later correction paragraphs do not make the earlier statements harmless;
new implementers will reasonably follow equations and tables in their local
section. Rewrite the canonical sections in place, move historical v1 text to an
appendix or Git history, and run a contradiction scan. Only then can the status
say all six fixes are applied.

## Status against the first review

| prior item | iter-2 status |
|---|---|
| S1 unit space | **PARTIAL PASS** — good coordinate choice; old mixed-unit/H-3 text remains |
| S2 consumer contract | **PARTIAL** — module exists, but fallback/link/derivative and ownership boundary are incomplete |
| S3 anchor order/identification | **OPEN** — semantics unverified; MSE is not conditional variance; bracket is unordered |
| S4 MNAR pairing | **PASS** — correction is explicit and tracking no longer licenses the old sample |
| S5 forecast/overlap | **PARTIAL PASS** — correct validation section; missing target fields and stale power claims remain |
| S6 `c(r)`/H-3 | **PARTIAL PASS** — diagnostic status/domain rules improve; `c(30)=1.14` uses the wrong provisional residual and the contract is weakly typed |

## Additional corrections

1. The provisional `2.6 bps` used to print `c(30)=1.14` came from
   `SIGMA_DIAGNOSTICS.md:156-175`, where the anchor is a Binance-mid path
   level-anchored to Chainlink—not `2*S30-S60`. It cannot be used as the realised
   error of the new anchor. Retiring D2 as an invalid prediction is reasonable;
   advertising 1.14 as its replacement is not.
2. The independence Monte Carlo draws `fut` independently by construction. Add
   an analytic covariance identity under the model and an empirical lagged
   covariance/sensitivity estimate; do not present the present test as evidence
   about market data.
3. `anchor_error_coeff` accepts configurable `s_fast/s_slow` but hardcodes
   `P_hat=2*S_fast-S_slow`; that formula is correct only for 30/60. Either remove
   the false generality or derive coefficients from the supplied windows.
4. Test exact expected rational coefficients and every supported horizon, not
   only broad ranges such as `9<a_spot<10`.
5. The contract checker still does not validate the semantic claims above; its
   green result means structural references/migrations passed, not that v13 is a
   sound BE boundary.

## Revised next steps

1. **Unfreeze the "actual 1 s" claim.** Complete the S30/S60 semantics and
   alignment study first.
2. **Choose the anchor estimand.** Conditional Brownian projection versus an
   operational local-trend forecast; separate conditional mean, bias and
   variance.
3. **Replace the fake bracket.** Model/proxy diagnostics plus a covariance-aware
   bound or an explicitly non-ordered uncertainty set.
4. **Repair the kernel API and v13 carrier.** One ledger, no hidden nugget,
   domain refusal, StateView inputs, single-copy link, derivative and temporal
   request fields.
5. **Rewrite Revision 2 in place** so one canonical equation and one consumer
   matrix remain.
6. Re-run unit, contract and adversarial tests. Only then mark Phase 0A 1–3
   complete and proceed to the single-scale baseline.

## Final assessment

The commit made real progress and found a useful fact: the previous plan omitted
the forecast-anchor error induced by using smoothed streams as a spot proxy. The
mistake is promoting one model-implied unconditional MSE into an exact
conditional variance budget before the feed semantics, conditional mean and
proxy covariance are known.

Keep the arithmetic coordinate, the corrected discrete formula as a fixture,
the physical-forecast terminology, the MNAR correction, the diagnostic `c(r)`
policy and the contract-first ordering. Reopen the anchor ledger and v13
boundary. The correct next focus remains Phase 0A specification work, not
estimator implementation.
