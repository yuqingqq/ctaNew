# SIGMA_PLAN_REVIEW_ITER3 — review of Revision 3 and contracts v14

Object: commit `7474e49` (`sigma ITER2: the anchor is a conditional MEAN
problem, not a variance ledger`), based on the second review at `20b759e` and
the status correction at `cf300ff`. Date: 2026-08-20.

Reviewed artefacts:

- `SIGMA_PLAN.md` Revision 3;
- `sigma_kernels.py`;
- `contracts/contracts.yaml` v14 and the v13→v14 migrations;
- the P-2026-003 status and handoff changes.

## Verdict

Revision 3 is a substantial improvement. It reproduces the Brownian projection
calculation correctly, separates the `2/-1` trend extrapolator's squared bias
from model-implied conditional variance, removes the false ordered bracket,
reopens the sampling-convention gate, collapses the old overlays into one plan,
puts requests behind `StateView`, and removes the hidden nugget from the
executable ledger. These changes should be retained.

The repair is not implementation-ready. Its new empirical-anchor recommendation
and its structural variance ledger are two different statistical specifications,
but the plan combines them. The executable fixture then cannot represent the
empirically estimated anchor the plan requires: any supplied `alpha` is compared
with the Brownian `alpha_star`, labelled biased, and priced with the Brownian
conditional variance. Feed covariance has incompatible units across plan,
contract and code; refusal does not cover the unverified convention or invalid
variance inputs; and v14 carries timestamps without the invariants that make them
safe.

**Decision: PARTIAL, HOLD.** Do not implement the estimator, close Phase 0A, or
use the printed `sigma_eff` values. The conditional-mean insight passes; the
empirical/structural composition and executable carrier do not.

## Executable audit

The shipped checks pass:

```
python3 live/pm_research/sigma_kernels.py --selftest                 PASS (24)
python3 live/pm_research/contracts/contract_check.py --selftest     PASS (13)
python3 live/pm_research/contracts/contract_check.py cf300ff HEAD   PASS
python3 -m py_compile live/pm_research/sigma_kernels.py              PASS
```

The v13→v14 structural diff reports 9 authorised removals, 13 authorised
changes and 77 additions, with no unexplained structural change. That establishes
that the migrations match the YAML inventory. It does not establish the semantic
invariants described in prose.

Adversarial probes:

```
settlement_var(30, 1) under default UNVERIFIED convention -> numeric Fraction
ledger(60, 1, alpha=1.7)["bias_coeff_on_S30_minus_S60"] -> 0.200833...
ledger(60, 4, feed_cov=I).feed_var                     -> 9.9867
ledger(30, -1).total_var                               -> -4.6911
ledger(60, 1, feed_cov=[[0,100],[100,0]]).total_var   -> -120.905...
Unavailable.__slots__                                  -> ('reason',)
```

The first result violates the declared fail-closed policy. The third exposes a
unit mismatch: if `Omega=I` is in bps² as Revision 3 states, `u' Omega u` is
about 2.4967 bps² and must not be multiplied by `sigma²=4`. The next two show
that the new domain validation still permits impossible pricing variances. The
last does not implement the contract's `Unavailable{reason,since,cause}`.

## MUST-FIX

### M3-1 — choose reduced-form regression or a structural ledger; do not combine both

Revision 3 §2.2 says to regress observed settlement `x_T` directly on observed
`(S30,S60)` at each horizon and says the regression residual variance is
`Sigma(r)` itself. That is a legitimate reduced-form route, subject to the
statistical qualifications below. But §3.2, §4 and G3 separately construct

```
Sigma(r) = sigma² k_law(r) + sigma² v(r) + u' Omega u.
```

Those cannot both be inputs to the same estimate without a reconciliation rule.
The direct-regression residual already contains future innovation, uncertainty
about the latent current path, stream measurement error, their covariance and
model misspecification. Adding separately estimated `k`, `v` and `Omega` to that
residual double-counts them. Conversely, subtracting structural lines from the
residual to obtain `c(r)` requires those lines to be identified under a declared
latent measurement model.

Direct regression also does not make the identification problem disappear while
the plan still asks step 6 to estimate a separate 2x2 `Omega`. Regressing the
target on two noisy observed predictors identifies a reduced-form coefficient
and total residual law; it does not identify the predictors' latent measurement-
error covariance. Measurement error also changes the fitted coefficient, so
estimating `alpha` on noisy streams and then adding `u' Omega u` is generally not
a valid correction.

Finally, ordinary least squares estimates the best **linear projection**. It is
the conditional mean only if the conditional mean is linear (or the object is
explicitly defined as the linear projection), and its pooled residual variance
is conditional variance only under a suitable conditional variance model.
Otherwise it is another unconditional forecast MSE—the exact category Revision
3 correctly removed from the Brownian variance line.

Choose one canonical route:

1. **Reduced form:** define `m_r(S30,S60,state)` and its OOS conditional residual
   distribution as the pricing law. `k/v/Omega` remain diagnostics and are never
   added to the reduced-form residual; or
2. **Structural:** declare a latent state and stream measurement equation,
   identify `v` and `Omega` under it, and derive the conditional mean and variance
   together. The direct regression is then a challenger/diagnostic, not the same
   ledger.

Whichever route is selected must specify cross-fitting, per-horizon/per-symbol
scope, day blocking, residual conditional-mean tests and heteroskedasticity or a
refusal. Two day clusters can produce a descriptive coefficient, not a pricing-
ready conditional law.

### M3-2 — the fixture cannot express an empirically estimated anchor

The plan makes `alpha_provenance: ESTIMATED` a precondition for pricing and says
the Brownian `alpha_star` is diagnostic only. The executable fixture does the
opposite. `ledger(..., alpha=a)` always computes

```
bias_coeff = a - alpha_star_model
anchor_var = sigma² * v_Brownian(r).
```

For an empirical estimate `a=1.7` at `r=60`, it reports a known bias coefficient
of `0.200833...`. Applying the documented mean correction subtracts that term
and algebraically returns the center to the Brownian `alpha_star`. Thus no
empirical coefficient different from `2700/1801` can become the fixture's
zero-bias conditional mean.

The code carries neither `alpha_provenance` nor a fitted conditional-mean
reference and implements no `conditional_mean`. Its `alpha` argument is therefore
misleading: it changes feed weights and the reported bias, but it cannot replace
the model-implied projection used to define correctness.

The contract is ambiguous in the same place. `AnchorSpec.alpha`, `bias_coeff`
and `cond_var` are scalars even though the plan and fixture show that the anchor
is horizon-dependent (`alpha_star(30)=1.2496`, versus 1.4992 outside the window)
and recommend fitting it “at each r.” Either a `PathLaw` is a horizon family—in
which case those fields need horizon-indexed functions/maps—or it is a single-
request object, in which case the repeated `ForecastRequest` protocol and
duplicate instrument/as-of/target fields need strict equality invariants.

Required repair:

- separate `alpha_model` from `alpha_estimated` and the actually selected
  conditional-mean coefficient;
- define bias relative to the selected empirical estimand, not automatically
  relative to the Brownian fixture;
- implement `conditional_mean` beside the single ledger and test that an
  estimated coefficient remains the mean instead of being cancelled;
- give every anchor quantity explicit horizon/request scope and provenance;
- keep the model-vs-estimate gap as a diagnostic, not a pricing correction.

### M3-3 — `Omega` is neither identified nor unit-consistent

Revision 3 defines `Omega` in bps² and the contract attaches a `UnitSpace` to
`FeedErrorCov`. `sigma_kernels.py` instead documents its matrix as being “in
sigma² units” and multiplies `u' feed_cov u` by `sigma2` in `ledger`. With
`sigma2=4` and an identity covariance supplied in bps², the fixture returns
about `9.9867` instead of `2.4967` bps². Both interpretations cannot implement
the same carrier.

The code also accepts matrices that are not positive semidefinite, including one
that drives total settlement variance to `-120.9`. No symmetry, unit, finiteness,
PSD or provenance check exists. `FeedErrorCov` uses three bare floats while
`VarianceQuantity` is supposedly mandatory for variance crossing the boundary;
`sigma2_rate` also shares the same type as a terminal variance even though its
time dimension is bps²/second.

Before `Omega` enters a ledger:

- choose physical bps² or a dimensionless multiple of `sigma²`, type it once,
  and remove the other convention;
- validate symmetry, finiteness and PSD, with `Unavailable` on failure;
- type the time dimension of `sigma2_rate` separately from terminal bps²;
- state how `Omega` is identified. If the route is direct regression, it is part
  of the reduced-form residual unless an external measurement-error design
  identifies it separately;
- test exact unit fixtures with `sigma2 != 1`, which the current tests omit.

### M3-4 — sampling and refusal remain descriptive, not fail-closed

The repair correctly labels every fixture convention `UNVERIFIED`, but the main
public accessor still returns a numeric variance under the default unverified
convention. `settlement_var` discards `ledger["convention_status"]`, so its caller
cannot even observe the warning. This contradicts the file header, contract and
module note that no unverified convention may price.

The claimed “versioned weight schedule” is also not implemented in the fixture.
Its records contain window lengths and optional lag fields, and `_obs` always
constructs rectangular trailing means. It cannot represent arbitrary weights,
irregular support or update triggers. In the YAML, `weights_fast/slow` do not
carry their support timestamps/offsets, so the aligned and one-second-lagged
schedules can have identical lists of weights. `fast_slow_synchronous: false`
does not say which stream lags or by how much.

Domain checks improved for `r`, but not for the other inputs. Negative `sigma2`
is accepted and produces negative total variance; non-PSD covariance is accepted;
unknown conventions raise `KeyError`; invalid alpha/covariance shapes are not
typed refusals. The fixture's `Unavailable` has only `reason`, while v14 requires
`reason`, `since` and `cause`.

There is also a live contradiction inside the executable documentation:
`sigma_kernels.py:22` says the exact coefficient is `1799/1200`, while its own
test at `:347-351` says that fraction is wrong and verifies `2700/1801`.

Required repair:

- refuse numeric settlement output while convention status is not `VERIFIED`,
  or expose only an explicitly named sensitivity function that cannot satisfy
  the pricing protocol;
- represent a weight schedule as `(relative support time, weight)` pairs plus
  update/alignment semantics;
- validate every pricing input, including nonnegative finite rates and PSD
  covariance;
- implement the contract's refusal object or explicitly declare the fixture's
  separate type;
- add adversarial status/unit/PSD/provenance tests and fix the stale fraction.

### M3-5 — v14 carries request fields but does not enforce request consistency

Adding `ForecastRequest` is directionally right. The contract still has no
invariants tying it to the law being queried:

- request instrument equals `PathLaw.instrument`;
- request target interval equals the law target interval;
- request horizon is consistent with `as_of` and the target interval;
- `knowledge_cutoff <= request.as_of`;
- the request is within the law's coverage and horizon domain;
- the law's `fit_data_through` is no later than the request knowledge cutoff;
- `LinkRef(link_id,version)` resolves to the exact `BeliefProcess.link` used by
  the consumer.

R-WFWD checks only the two stored PathLaw fields. The checker records its check
strings but does not evaluate their inequalities; “0 invariant failures” means
references and migration structure are green, not that temporal safety holds.
The `no_future_train` string is not an executable invariant.

`LinkRef` removes the duplicated link object but does not by itself guarantee
assembly equality. `g_prime` and `density` duplicate the same derivative concept
without an equality/semantic rule, allowing two implementations to disagree.

Required repair: define executable construction/query invariants for all fields
above, make one derivative operation canonical, and add negative fixtures that
swap instrument, target, link version and knowledge cutoff. A typed timestamp
that is never compared is documentation, not look-ahead protection.

### M3-6 — the canonical rewrite still contains contradictory live guidance

The rewrite is much better than Revision 2's overlay, but the claimed clean
contradiction scan is not supported:

- §2.2 says direct regression is robust to the unverified convention and removes
  the semantics study from the anchor's critical path (`:149-159`), while §12
  says step 5 gates “the kernel and the anchor together” (`:556-562`, `:584-586`);
- §2.2 requires `alpha` to be estimated before pricing (`:139-147`), while §10
  says it “must be assumed per-symbol until measured otherwise” (`:510-511`);
- §2.2 says the regression residual is `Sigma(r)` itself, while the canonical
  ledger separately adds `k`, `v` and `Omega`;
- the status and handoff still state that `2*S30-S60` “fixes” the anchor even
  though Revision 3 says `alpha=2` is a biased trend extrapolator and its one-day
  Brier gain is not evidence for that coefficient;
- `HANDOFF.md:19` still calls contracts v13 the source of truth after v14 landed;
- the fixture's `1799/1200` versus `2700/1801` conflict survived the scan.

Replace each with one scoped statement. In particular, distinguish “the v1
lagging-S60 direction was wrong” from “the `alpha=2` anchor is correct”—only the
first is supported. Then rerun a mechanical contradiction search across the
plan, code header, status and handoff, not only the canonical plan body.

## Status against the iter-2 review

| prior item | iter-3 status |
|---|---|
| M2-1 conditional mean versus MSE | **PARTIAL PASS** — Brownian decomposition is corrected; the empirical alpha cannot become the executable mean and OLS residual variance is overclaimed |
| M2-2 sampling semantics | **PARTIAL PASS** — convention is unverified/reopened; neither code nor YAML fully represents its weighted temporal support, and pricing does not refuse |
| M2-3 unordered evidence/covariance | **PARTIAL** — false bracket removed and 2x2 shape added; `Omega` remains unidentified, unit-inconsistent and unvalidated |
| M2-4 one ledger/domain | **PARTIAL PASS** — hidden nugget and duplicate total fixed; invalid rates/covariances and unverified conventions still return numbers |
| M2-5 consumer contract | **PARTIAL** — StateView, derivative and request fields improve v13; request/link/time/horizon equality is unenforced and rate dimensions are incomplete |
| M2-6 canonical plan | **PARTIAL PASS** — rewrite is materially clearer; several load-bearing contradictions and stale tracking claims remain |

## Revised next steps

1. **Freeze one statistical estimand:** reduced-form conditional law or structural
   latent-state decomposition.
2. **Repair alpha ownership:** empirical versus model coefficients, horizon
   scope, conditional-mean implementation and provenance.
3. **Resolve the feed-error route:** identification, units, covariance support
   and whether it is already inside a reduced-form residual.
4. **Make the fixture genuinely fail closed:** unverified status, invalid rates,
   covariance PSD, request scope and typed refusals.
5. **Complete v14 invariants:** instrument/target/horizon/knowledge/link equality
   and executable negative tests.
6. **Reconcile the canonical documents and tracking**, then rerun shipped and
   adversarial checks. Only after these pass should Phase 0A move to the empirical
   anchor experiment.

## Final assessment

The commit correctly identifies the central conceptual lesson: forecast-anchor
bias belongs in the conditional mean, not in a variance budget. The next mistake
would be to treat a best linear regression and its pooled residual as a complete
conditional law while also adding a structural Brownian/feed ledger.

Keep the corrected projection algebra, non-ordered evidence, one-ledger API,
StateView boundary, request carrier and canonical rewrite. Hold implementation
until the empirical and structural routes are separated and the selected route
is representable end to end—from fitted alpha, through units and temporal
invariants, to a refusal-safe probability input.
