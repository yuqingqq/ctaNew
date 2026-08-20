# SIGMA_PLAN_REVIEW_ITER5 — Revision 5 measurement-readiness review

Object: commit `58f6716` (`sigma ITER4: make the route choice hold at the
executable boundary`), based on the iteration-4 review at `ef6d1df`. Date:
2026-08-20.

Reviewed artefacts:

- `SIGMA_PLAN.md` Revision 5;
- `sigma_kernels.py`;
- `contracts/contracts.yaml` v16 and the v15→v16 migrations;
- the P-2026-003 status and handoff changes.

## Verdict

**YES: start measuring the model. Decision: MEASUREMENT GO, PRICING HOLD.**

Revision 5 closes the specification questions that blocked a meaningful
experiment. Route A is now unambiguously the empirical pricing estimand, its
residual is the whole terminal variance, Route B cannot satisfy the pricing
protocol, and internal S30/S60 kernel semantics no longer gate the observed-
stream regression. Mean and variance come from one fit and one atomic query;
the anchor bias no longer re-enters structural conditional variance; evidence
is scoped to each symbol/horizon; and non-rejection is no longer defined as a
pass.

There should be **no Revision 6 before Phase 0A step 6**. The next useful work is
to implement and run the real point-in-time Route-A fit. Phase 0A step 5 may run
in parallel for Route-B diagnostics and must not delay that fit.

“Measurement GO” does not mean a usable probability law exists today. The repo
contains a synthetic boundary fixture, not a fitted estimator, and the available
tape spans only two day clusters. A run now is therefore valuable as an
end-to-end descriptive result and pipeline validation, but it cannot return a
pricing `PASS`; the declared minimum is ten independent day clusters, with the
effect-size equivalence gates also passing OOS.

## What passed

All six iteration-4 acceptance directions are substantively resolved:

| item | Revision-5 result | readiness |
|---|---|---|
| M4-1 route coupling | Route A carries `StreamProvenance`, not `SamplingConvention` | **PASS** |
| M4-2 optional request checks | one atomic `pricing_distribution(law, request, observables)` | **PASS** |
| M4-3 numeric domains | real fit fields reject negative/zero/non-finite variance and invalid counts | **PASS for typed fit inputs** |
| M4-4 non-rejection gate | per-fit `GateEvidence` separates pass, insufficient evidence and refutation | **PASS in design** |
| M4-5 structural reachability | distinct diagnostic type; model conditional variance takes no empirical alpha | **PASS** |
| M4-6 route carrier/units/role | union, offset schedules, second-valued coefficients, route-agreement ratio, consumer map | **PASS** |

The shipped checks pass:

```text
python3 live/pm_research/sigma_kernels.py --selftest                 PASS (45)
python3 live/pm_research/contracts/contract_check.py --selftest     PASS (13)
python3 live/pm_research/contracts/contract_check.py ef6d1df HEAD   PASS
python3 -m py_compile live/pm_research/sigma_kernels.py              PASS
```

The v15→v16 checker reports 29 authorised removals, 11 authorised changes and
91 additions with no unexplained inventory change. As before, that checker does
not validate all YAML or runtime semantics; the cleanup items below demonstrate
the distinction.

## The measurement to run now

Implement Phase 0A step 6 as one frozen experiment, not another plan rewrite:

1. Build point-in-time rows per symbol and horizon `r ∈ {30,60,120,180,240,270}`
   from the observed published streams and observed settlement mark.
2. Fit the translation-invariant regression

   ```text
   x_T - S60 = alpha(r) * (S30 - S60) + residual
   ```

   per symbol and horizon. Do not add `k_law`, `v(r)` or `Omega`; they are
   already represented in the OOS residual.
3. Cross-fit by whole day, train only through the preceding day, and embargo at
   least the longest target support. Use non-overlapping labels or declared
   overlap-aware weights; never random-row CV.
4. Persist every OOS prediction and residual with instrument, target interval,
   decision/knowledge timestamps, fold, horizon and source-row provenance.
5. Report per symbol/horizon: fitted alpha, OOS residual variance, effective
   observations, day clusters, fold dispersion, conditional-mean evidence and
   conditional-variance evidence.
6. Freeze the exact gate procedures and tolerances **before reading their
   results**. Report effect estimates and block/day confidence bounds even when
   the verdict is `INSUFFICIENT_EVIDENCE`.
7. With the current two-day tape, label the output `DESCRIPTIVE`. Re-run the
   identical frozen analysis after at least ten day clusters; only then can a
   fit possibly emit `PASS`.

This experiment can refute the proposed linear/homoskedastic law. That would be
an empirical result requiring a richer Route-A mean or variance model, not
evidence that the route decision or structural ledger should be reopened.

## Cleanup to do alongside the measurement implementation

These are real defects, but none changes the estimand or blocks fitting the
data. Fix them in the estimator implementation commit rather than opening
another theory-review cycle.

### C5-1 — v16 has a duplicate YAML type and a stale superseded carrier

`contracts.yaml` defines `ReducedFormFit` twice. A strict unique-key parse fails:

```text
duplicate key 'ReducedFormFit' at contracts.yaml:798
```

The ordinary YAML loader silently keeps the second, old three-field definition:

```text
ReducedFormFit = {alpha, resid_var, n_effective}
```

It therefore discards `n_day_clusters`, `cross_fitted`, `mean_gate` and
`var_gate` from the machine-readable source of truth even though the fixture
requires them. The green checker does not catch duplicate mapping keys.

`ReducedFormLaw` is also retained with the v15 sampling convention, parent-level
cluster count and bare p-value gates, while the Revision-5 carrier is
`ReducedFormPathLaw`. Build step 6 still says to emit `AnchorSpec` and that old
`ReducedFormLaw`. That is stale migration debris, not an unresolved model
choice.

Cleanup acceptance:

- keep one `ReducedFormFit`, with all seven Revision-5 fields;
- delete the superseded `ReducedFormLaw` or redefine one unambiguous ownership
  relationship without its v15 convention/p-value fields;
- change build step 6 to emit `ReducedFormPathLaw` plus per-fit `GateEvidence`;
- make the contract checker reject duplicate YAML keys before inventorying.

### C5-2 — malformed public-boundary objects can still raise or lose refusal provenance

The typed, correctly constructed path is fail-closed. Arbitrary malformed input
is not yet a total function:

```text
pricing_distribution(law, object(), obs)       -> AttributeError
TargetInterval(None, 1060)                     -> TypeError during validation
NaN observable refusal                         -> since=None, cause=None
UNVERIFIED StreamProvenance refusal             -> since=None
```

This does not affect a controlled fitting experiment, but it contradicts the
strong claim that every boundary refusal is typed with `since` and `cause`.
Before any consumer binds to this API, validate request/interval/provenance
types before dereference or comparison and normalize all nested refusals at the
public boundary.

### C5-3 — validate the evidence payload, not only its confidence bound

`GateEvidence.validate` checks the confidence-bound/tolerance comparison but
does not validate `effect_size`. Both `effect_size=NaN` and
`effect_size=100` with a small declared `ci_hi_abs` currently pass. It also does
not record a confidence level or inference method as structured fields.

For the measurement script:

- require finite `effect_size` and `abs(effect_size) <= ci_hi_abs`;
- carry confidence level, block unit, fold definition and tolerance identifier;
- make the tolerance identifier resolve to the frozen pre-result protocol, not
  an arbitrary post-result string.

Again, this is experiment implementation hygiene. It does not justify delaying
the regression or revisiting Routes A/B.

## Decision boundary from here

- **GO now:** implement/run Phase 0A 6 on current data; run Phase 0A 5 in
  parallel if desired.
- **DESCRIPTIVE now:** two day clusters are enough to debug the pipeline and
  estimate provisional coefficients, not enough for pricing readiness.
- **PRICING HOLD:** until at least ten day clusters and every frozen OOS
  equivalence/calibration gate passes.
- **No more pre-measurement sigma review:** only reopen the statistical form if
  the measured OOS residuals refute it.

The programme has moved from “what is the model?” to “does the chosen empirical
model survive the tape?” That question must now be answered with data.
