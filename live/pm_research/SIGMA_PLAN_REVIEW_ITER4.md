# SIGMA_PLAN_REVIEW_ITER4 — review of Revision 4 and contracts v15

Object: commit `be23ec1` (`sigma ITER3: pick ONE estimand route -- reduced
form prices, structural diagnoses`), based on the iteration-3 review at
`11e3c09`. Date: 2026-08-20.

Reviewed artefacts:

- `SIGMA_PLAN.md` Revision 4;
- `sigma_kernels.py`;
- `contracts/contracts.yaml` v15 and the v14→v15 migrations;
- the P-2026-003 status and handoff changes.

## Verdict

Revision 4 fixes the largest conceptual error in Revision 3. It explicitly
chooses the reduced-form conditional law as the pricing estimand, leaves the
structural `k_law/v/Omega` construction as a diagnostic, and forbids adding the
two. It also separates model and estimated anchors by horizon, makes feed-error
covariance physical bps² with PSD validation, types the variance rate, expands
`Unavailable`, evaluates several request comparisons, and removes the duplicate
link derivative. These changes should be retained.

The repair is still not implementation-ready. The canonical plan says Route A
does not depend on knowing the internal S30/S60 sampling kernel, but the only
pricing function refuses Route A unless that convention is `VERIFIED`. More
importantly, request/time invariants are a separate optional helper rather than
part of the pricing query, and the helper permits both future-issued laws and
reversed target intervals. The Route-A gate accepts negative residual variance
and non-finite diagnostic values; its scalar tests do not implement the promised
per-horizon/per-symbol evidence. Route B also puts selected-anchor squared bias
back into a field named `anchor_var`, and exposes `model_total_var` despite the
claim that no total is reachable.

**Decision: PARTIAL, HOLD.** The route decision passes. Do not fit or expose a
pricing estimator until the chosen route has one atomic, fail-closed request API
and its numeric/statistical gates are executable. Route-B results remain model
diagnostics and must not drive a pricing level.

## Executable audit

The shipped checks pass:

```text
python3 live/pm_research/sigma_kernels.py --selftest                 PASS (41)
python3 live/pm_research/contracts/contract_check.py --selftest     PASS (13)
python3 live/pm_research/contracts/contract_check.py 11e3c09 HEAD   PASS
python3 -m py_compile live/pm_research/sigma_kernels.py              PASS
```

The v14→v15 diff reports 7 authorised removals, 4 authorised changes and 42
additions. That establishes migration/inventory consistency, not the semantic
properties below.

Focused adversarial probes:

```text
verified Route-A law with resid_var = -1                -> -1
n_day_clusters = NaN, test p-values = NaN               -> 30
law.as_of = 2000 for request.as_of = 1000, and
target start = 2000 > end = 1060                        -> check_request True
model_var_diagnostic with empirical alpha changes
anchor_var from 5354821/648360 to 304589/36000          -> different
structural diagnostic exposes model_total_var           -> 9324001/324180
pricing_var arguments                                   -> (law, r, grid_only)
sigma2_rate = infinity                                  -> OverflowError
```

The first two are impossible pricing states that pass the advertised gate. The
third proves the temporal carrier is not fail-closed. The fourth contradicts the
plan's alpha-independent conditional-variance statement. The fifth makes the
“no total a pricer could reach for” self-test a key-name test rather than a type
boundary. The sixth shows request consistency cannot be enforced by the public
pricing accessor. The final result is an exception rather than the contract's
typed refusal.

## MUST-FIX

### M4-1 — Route A is still gated on Route B's sampling convention

The route table in §2.3, the risks in §9 and build step 5 all say the same thing:
Route A regresses the observed published streams and does **not** need their
internal sampling convention; convention verification gates Route B. This is a
sound and useful reason to choose Route A.

The executable and contract implement the opposite at the pricing boundary:

- `ReducedFormLaw` carries a mandatory `SamplingConvention`;
- `pricing_var` calls `_conv` and refuses unless its status is `VERIFIED`;
- `SamplingConvention` says no `PathLaw` with an unverified convention may
  price;
- `BE-Uncertainty` says it refuses whenever that status is unverified;
- the plan header and §3.3 describe the same refusal.

Thus Phase 0A step 5 still gates Phase 0A step 6 in code even though the route
decision says it does not. This is not a conservative extra check: it removes
the operational advantage used to select the reduced-form estimand and conflates
two different provenance questions.

Required repair:

- remove internal-kernel verification from Route-A pricing;
- give Route A its own observed-stream schema/provenance (stream identities,
  point-in-time reads, units and alignment at the published timestamps);
- retain `SamplingConvention` only on the structural decomposition;
- add a test where a well-formed fitted Route-A law prices while structural
  kernels still refuse under `UNVERIFIED`;
- make plan, contract, module notes, status and handoff state that rule once.

### M4-2 — request invariants are optional and temporally incomplete

`check_request` is directionally useful, but neither `pricing_var` nor
`conditional_mean` accepts a `ForecastRequest` or `LawHeader`; neither calls the
helper. A caller can obtain both pieces of a distribution without checking any
instrument, target, link, fit cutoff or knowledge-time invariant. Correctness
therefore depends on every future caller remembering an unrelated pre-call.

Even when called, the helper accepts a law issued in the future relative to the
request (`law.as_of > req.as_of`). It checks only the target end against
`as_of+horizon`: it never requires `target.start < target.end`, validates the
start's semantic relationship to the market window, or establishes a version
validity interval for the law. A reversed target can therefore pass. The
contract requires `Unavailable{reason,since,cause}`, but executable refusals
normally leave `since` and `cause` unset; the self-test checks slots, not refusal
provenance.

Required repair:

- expose one atomic query such as
  `pricing_distribution(path_law, request, observables)`;
- make it run request validation before both the conditional mean and variance,
  returning one typed result/refusal;
- require `law.as_of <= request.as_of` or define and enforce an explicit law
  version-validity interval;
- validate `target.start < target.end`, interval identity/width, horizon, fit
  cutoff, knowledge cutoff, coverage, instrument and exact link version;
- populate `since` and a machine-actionable cause on every boundary refusal;
- test that no public pricing path can bypass these checks.

### M4-3 — Route-A fail-closed validation is numerically unsafe

A synthetic law that passes the convention, status, cluster, cross-fit and
p-value gates returns `resid_var=-1` as a pricing variance. `NaN` cluster counts
and `NaN` p-values also pass because ordered comparisons with `NaN` are false.
P-values above one are not rejected; malformed horizon entries can raise rather
than refuse; and an infinite structural rate raises `OverflowError` while being
converted to a `Fraction`.

The same object stores one `n_day_clusters` and two scalar p-values for an entire
horizon map, although §2.3 requires the evidence per horizon and per symbol.
`alpha` is neither validated nor used by `pricing_var`; mean and variance can be
queried from inconsistent objects. `resid_var` is a bare executable number even
though the boundary contract says variances use `VarianceQuantity`.

Required repair:

- validate finiteness and domains before conversion: positive residual
  variance; finite rate; integer nonnegative sample/cluster counts; p-values in
  `[0,1]`; finite coefficients and covariance entries;
- return `Unavailable`, never a raw conversion/key/type exception, for malformed
  fitted artefacts;
- put coefficient, variance, effective sample size, cluster count and all gates
  inside each per-symbol/per-horizon fit entry;
- query mean and variance from the same validated fit/version atomically;
- add negative/zero/NaN/infinity/missing-field fixtures to the public query.

### M4-4 — non-rejection tests are not evidence of a conditional law

The current pricing policy equates `p >= 0.01` with a verified zero conditional
mean and constant conditional variance. Failure to reject is not equivalence,
especially at the minimum ten day clusters where these tests can have little
power. The gate does not name a test, conditioning basis, multiplicity policy,
effect-size tolerance or confidence interval. A high p-value can mean “not
enough data,” not “the pooled residual is a conditional variance.”

This matters because Route A is the only probability-level fallback law. A
misspecified mean or variance is not merely a diagnostic error at that boundary.

Required repair:

- pre-register the conditional-mean and heteroskedasticity procedures,
  conditioning variables and per-horizon multiplicity policy;
- gate on effect-size/confidence bounds against economically stated tolerances,
  not only non-rejection p-values;
- report OOS conditional calibration and proper variance/distribution scores;
- distinguish `INSUFFICIENT_EVIDENCE` from `MODEL_REFUTED`;
- treat ten day clusters as a minimum for attempting inference, not automatic
  evidence that the law is valid.

### M4-5 — Route B still mixes conditional variance with squared bias and is reachable

The plan correctly states that all anchors in the translation-invariant family
have the same conditional variance and differ only in known conditional bias.
But `model_cond_var(r, alpha)` explicitly returns conditional variance **plus
squared bias** whenever `alpha != alpha_star_model`. `model_var_diagnostic`
passes the selected (including empirical) alpha to that function and publishes
the result as `anchor_var`. Changing only the empirical alpha therefore changes
the alleged conditional variance. This reintroduces the exact variance/MSE
category error that Revision 3 removed, albeit on the diagnostic route.

The returned dict also contains `model_total_var`. The self-test proves only
that keys named exactly `total_var` and `sigma_eff` are absent, then claims no
total is reachable. Renaming a numeric total is not a semantic type boundary.
The contract's R-ROUTE rule constrains reduced-form composition but does not
make a structural `settlement_var` uninhabitable for a pricing consumer.

Required repair:

- compute structural conditional `v(r)` only at the model projection, independent
  of the selected empirical anchor;
- expose selected-anchor model gap, known bias and unconditional MSE as separate
  diagnostic fields, never inside `anchor_var`;
- return a distinct `DiagnosticVarianceDecomposition` type that cannot satisfy
  any pricing-law/probability-consumer protocol;
- encode both directions of R-ROUTE: reduced-form pricing contains no structural
  additions, and structural route cannot provide `settlement_var` to a pricer;
- replace key-name tests with consumer/type-level negative assembly tests.

### M4-6 — contract, units and operational role still contradict Revision 4

`PathLaw.estimand_route` is an enum, but the carrier is not a discriminated
union: it always carries reduced-form and structural fields. A Route-A object is
therefore forced to populate an anchor, rate, structural convention and related
state, while a structural object retains the same pricing-shaped protocol.
`increment_var` also has no declared reduced-form source, so its semantics under
the selected route are unclear.

The code now uses `(offset, weight)` schedules, but `SamplingConvention` still
declares `weights_fast/weights_slow: list[float]`. It cannot encode support or
which stream is offset; `fast_slow_synchronous: false` is not enough. The plan
and code call `k_law` and `v(r)` dimensionless, while multiplying a bps²/second
rate by them to obtain bps². Their exact values carry seconds (the plan itself
labels the seam value `20.5028 s`), so the contract needs a coefficient-duration
type. `CalibrationCurve` still describes `c(r)` as “the residual AFTER the
ledger” and asks whether it multiplies one line or the whole ledger, contradicting
Revision 4's definition `Sigma_A/model_total_B` that is added to neither.

Finally, “diagnostic only” is too broad for the stated consumers. The consumer
matrix has at least four controls that need variance shape, while §2.3 says the
decomposition is needed only by `c`, the ledger and H-3. If Route B supplies
shape to participation, quotable-horizon, pickoff or rewards controls, it is an
operational decision input even if it does not set the probability **level**.
If Route A supplies that shape, the plan and protocol must say how.

Required repair:

- make `PathLaw` a real discriminated union of a pricing
  `ReducedFormPathLaw` and non-pricing structural diagnostic type;
- define the source and request semantics of both terminal and increment
  variance under Route A;
- type kernel coefficients as durations, so rate × coefficient has bps² units;
- make the contract weight schedule carry `(relative_offset, weight)` plus
  alignment/update semantics;
- rewrite `CalibrationCurve` to the Revision-4 ratio definition;
- separate “not a probability-level input” from “not operational,” and map the
  dynamics consumers to their actual source.

## Tracking and canonical-document corrections

These do not independently block the estimator, but they make the current state
hard to audit:

- `SIGMA_PLAN.md` has two headings numbered `2.3`;
- the plan and handoff say the self-test has 40 checks; it runs 41;
- the status focus and sigma task still instruct the reader to choose the route
  and describe Revision 3 as current, although Revision 4 chose it;
- the handoff's read-first/current-verdict prose still says Revision 3 mixes the
  routes;
- the plan says the structural dict has no total while it exposes
  `model_total_var`;
- the status simultaneously says the convention gates Route B only and that
  `pricing_var` refuses Route A while it is unverified.

Update these in the same repair that resolves the underlying contradictions;
do not merely change the labels.

## Disposition of iteration-3 findings

| prior item | Revision-4 result | status after this review |
|---|---|---|
| M3-1 estimand route | Route A selected; no summation in the fixture | **substantive pass**, carrier/gating still partial |
| M3-2 empirical anchor | selected/model coefficients separated by horizon | **partial** — Route-B `anchor_var` still absorbs model-gap MSE |
| M3-3 Omega units/PSD | bps² once and PSD checks added | **partial** — coefficient time units and typed boundary remain wrong |
| M3-4 refusal/convention | structured refusal and schedules added in code | **partial** — Route A wrongly gated; NaN/negative/inf unsafe |
| M3-5 request invariants | comparisons and negative fixtures added | **partial** — optional/bypassable and incomplete time checks |
| M3-6 canonical scan | major prose rewritten | **partial** — live plan/contract/tracking contradictions remain |

## Acceptance sequence

1. Freeze the exact Route-A carrier and one atomic request-to-distribution API.
2. Remove structural convention semantics from Route A and validate the
   published-stream provenance it actually needs.
3. Make every input domain and temporal invariant fail closed; add adversarial
   tests at the public boundary.
4. Pre-register per-symbol/per-horizon statistical equivalence/calibration
   gates and encode their evidence in each fit.
5. Make Route B a separate diagnostic type; remove bias from its conditional
   variance and specify whether/how its shape drives controls.
6. Repair the YAML units, schedules, `CalibrationCurve`, route union and
   tracking text; run the checker from v15 to the next version.
7. Only then run Phase 0A step 6 on day-blocked data. Phase 0A step 5 may proceed
   independently for Route-B diagnostics; it must not block a valid Route-A
   fit.

Until those acceptance tests pass, estimator implementation and any
probability-level use remain on **HOLD**.
