r"""Route-A pricing boundary + Route-B diagnostics for the 5-min binaries.

STATUS: FIXTURE. Phase 0A is open and the estimator is on HOLD. No route-A law is
fitted, so `pricing_distribution` refuses everything today. That is the point:
the boundary is fail-closed by construction, not by remembering to check.

WHAT ITER4 CHANGED (M4-1..M4-5)
-------------------------------
M4-1  Route A is NO LONGER gated on the internal sampling convention. The plan
      said route A regresses published streams and does not need to know how
      Chainlink builds them -- and then the only pricing function refused unless
      that convention read VERIFIED. That contradiction removed the very reason
      route A was chosen. Route A now carries StreamProvenance (identities, PIT
      read rule, units, published-timestamp alignment); SamplingConvention is
      route B only.
M4-2  ONE ATOMIC QUERY. `pricing_distribution(law, request, observables)` runs
      every request/time invariant BEFORE computing either the mean or the
      variance and returns a single typed result. v4 exposed `pricing_var` and
      `conditional_mean` separately, neither of which called `check_request`, so
      correctness depended on every future caller remembering a pre-call.
M4-3  Numeric fail-closed. v4 priced `resid_var = -1` and passed NaN cluster
      counts and NaN p-values, because every ordered comparison with NaN is
      False. Infinity raised OverflowError instead of refusing.
M4-4  Non-rejection is not evidence. A p-value gate treats "not enough data" as
      "verified". Each fit now carries a GateEvidence verdict that distinguishes
      PASS / INSUFFICIENT_EVIDENCE / REFUTED, with an effect size and bound.
M4-5  Route B no longer puts squared bias in `anchor_var`: conditional variance
      is computed ONLY at the model projection, and the selected anchor's bias
      and unconditional MSE are separate diagnostic fields. The result is a
      distinct type that cannot satisfy the pricing protocol -- v4's "no total
      is reachable" self-test only checked two key NAMES.

    python3 sigma_kernels.py --selftest
"""
import math
from fractions import Fraction as F

HORIZON_GRID = (30, 60, 120, 180, 240, 270)
MIN_DAY_CLUSTERS = 10          # minimum to ATTEMPT inference, not evidence of validity


class Unavailable:
    """Typed refusal matching contracts Unavailable{reason, since, cause}."""
    __slots__ = ("reason", "since", "cause")

    def __init__(self, reason, since=None, cause=None):
        self.reason, self.since, self.cause = reason, since, cause

    def __repr__(self):
        c = f", cause={self.cause!r}" if self.cause is not None else ""
        return f"Unavailable({self.reason!r}, since={self.since!r}{c})"

    def __bool__(self):
        return False


# ----------------------------------------------------------- validation ------
def _num(x, name, since=None, cause=None):
    """Finite exact number or a refusal. Never raises (v4 raised OverflowError)."""
    if isinstance(x, bool) or isinstance(x, str) or x is None:
        return Unavailable(f"{name} is not a number: {x!r}", since, cause)
    try:
        if isinstance(x, float) and not math.isfinite(x):
            return Unavailable(f"{name} is not finite: {x!r}", since, cause)
        return F(x)
    except (TypeError, ValueError, OverflowError):
        return Unavailable(f"{name} is not convertible to an exact number: {x!r}",
                           since, cause)


def _pos(x, name, **kw):
    v = _num(x, name, **kw)
    if isinstance(v, Unavailable):
        return v
    return v if v > 0 else Unavailable(f"{name}={float(v)} is not strictly positive", **kw)


def _nonneg(x, name, **kw):
    v = _num(x, name, **kw)
    if isinstance(v, Unavailable):
        return v
    return v if v >= 0 else Unavailable(f"{name}={float(v)} is negative", **kw)


def _count(x, name, **kw):
    if isinstance(x, bool) or not isinstance(x, int):
        return Unavailable(f"{name} must be a plain integer, got {x!r}", **kw)
    return x if x >= 0 else Unavailable(f"{name}={x} is negative", **kw)


def _horizon(r, grid_only=True, **kw):
    if isinstance(r, bool) or not isinstance(r, (int, F)):
        return Unavailable(f"r must be an exact integer/Fraction, got {type(r).__name__}", **kw)
    if r != int(r):
        return Unavailable(f"r={r} is off the sample grid", **kw)
    r = int(r)
    if r <= 0:
        return Unavailable(f"r={r} is not a positive horizon", **kw)
    if grid_only and r not in HORIZON_GRID:
        return Unavailable(f"r={r} is not on HORIZON_GRID {HORIZON_GRID}", **kw)
    return r


# ============================ ROUTE A — the pricing boundary =================
class StreamProvenance:
    """What route A ACTUALLY needs: the published streams, not their internals.

    M4-1. Route A regresses observed S_fast/S_slow on observed x_T. It needs the
    stream identities, that reads are point-in-time at knowledge time, the units,
    and alignment AT THE PUBLISHED TIMESTAMPS. It does NOT need to know how the
    venue computes the aggregate internally -- that is SamplingConvention, and it
    gates route B only.
    """
    __slots__ = ("fast_id", "slow_id", "unit_space", "point_in_time",
                 "aligned_at_publication", "status")

    def __init__(self, fast_id, slow_id, unit_space, point_in_time,
                 aligned_at_publication, status):
        self.fast_id, self.slow_id = fast_id, slow_id
        self.unit_space, self.point_in_time = unit_space, point_in_time
        self.aligned_at_publication, self.status = aligned_at_publication, status

    def validate(self):
        if self.status != "VERIFIED":
            return Unavailable(f"stream provenance is {self.status}",
                               cause="STREAM_PROVENANCE_UNVERIFIED")
        if not self.point_in_time:
            return Unavailable("streams are not read point-in-time at knowledge time",
                               cause="NOT_PIT")
        if not self.aligned_at_publication:
            return Unavailable("stream reads are not aligned at publication timestamps",
                               cause="MISALIGNED")
        return True


class GateEvidence:
    """M4-4. Failure to reject is not equivalence.

    A verdict, an effect size and a bound -- not a bare p-value. INSUFFICIENT_
    EVIDENCE and MODEL_REFUTED are different states and both refuse; only PASS
    prices, and PASS requires the effect-size confidence bound to sit inside the
    pre-registered tolerance.
    """
    __slots__ = ("test", "conditioning", "verdict", "effect_size",
                 "ci_hi_abs", "tolerance", "multiplicity", "p_value")

    def __init__(self, test, conditioning, verdict, effect_size, ci_hi_abs,
                 tolerance, multiplicity, p_value=None):
        self.test, self.conditioning = test, conditioning
        self.verdict, self.effect_size = verdict, effect_size
        self.ci_hi_abs, self.tolerance = ci_hi_abs, tolerance
        self.multiplicity, self.p_value = multiplicity, p_value

    def validate(self, label):
        if self.verdict not in ("PASS", "INSUFFICIENT_EVIDENCE", "MODEL_REFUTED"):
            return Unavailable(f"{label}: unknown verdict {self.verdict!r}")
        if not self.test or not self.conditioning or not self.multiplicity:
            return Unavailable(f"{label}: test, conditioning basis and multiplicity "
                               "policy must be pre-registered",
                               cause="UNREGISTERED_PROCEDURE")
        if self.p_value is not None:
            p = _num(self.p_value, f"{label}.p_value")
            if isinstance(p, Unavailable):
                return p
            if not (0 <= p <= 1):
                return Unavailable(f"{label}: p_value {float(p)} outside [0,1]")
        if self.verdict != "PASS":
            return Unavailable(f"{label}: {self.verdict}", cause=self.verdict)
        tol = _pos(self.tolerance, f"{label}.tolerance")
        if isinstance(tol, Unavailable):
            return tol
        hi = _nonneg(self.ci_hi_abs, f"{label}.ci_hi_abs")
        if isinstance(hi, Unavailable):
            return hi
        if hi > tol:                       # equivalence, not non-rejection
            return Unavailable(
                f"{label}: PASS claimed but the |effect| confidence bound "
                f"{float(hi)} exceeds the tolerance {float(tol)}; that is "
                "non-rejection, not equivalence", cause="NOT_EQUIVALENT")
        return True


class ReducedFormFit:
    """One (symbol, horizon) fit. M4-3: evidence lives HERE, not on the parent."""
    __slots__ = ("alpha", "resid_var", "n_effective", "n_day_clusters",
                 "cross_fitted", "mean_gate", "var_gate")

    def __init__(self, alpha, resid_var, n_effective, n_day_clusters,
                 cross_fitted, mean_gate, var_gate):
        self.alpha, self.resid_var = alpha, resid_var
        self.n_effective, self.n_day_clusters = n_effective, n_day_clusters
        self.cross_fitted = cross_fitted
        self.mean_gate, self.var_gate = mean_gate, var_gate

    def validate(self, key):
        a = _num(self.alpha, f"{key}.alpha")
        if isinstance(a, Unavailable):
            return a
        v = _pos(self.resid_var, f"{key}.resid_var")      # v4 priced -1
        if isinstance(v, Unavailable):
            return v
        for nm, val in (("n_effective", self.n_effective),
                        ("n_day_clusters", self.n_day_clusters)):
            c = _count(val, f"{key}.{nm}")                # v4 passed NaN
            if isinstance(c, Unavailable):
                return c
        if self.n_day_clusters < MIN_DAY_CLUSTERS:
            return Unavailable(
                f"{key}: {self.n_day_clusters} day clusters < {MIN_DAY_CLUSTERS}; "
                "descriptive coefficient, not a pricing law",
                cause="INSUFFICIENT_EVIDENCE")
        if self.cross_fitted is not True:
            return Unavailable(f"{key}: not cross-fitted; in-sample residual "
                               "variance understates conditional variance",
                               cause="NOT_CROSS_FITTED")
        for label, g in ((f"{key}.mean_gate", self.mean_gate),
                         (f"{key}.var_gate", self.var_gate)):
            if not isinstance(g, GateEvidence):
                return Unavailable(f"{label} is missing", cause="NO_EVIDENCE")
            ok = g.validate(label)
            if isinstance(ok, Unavailable):
                return ok
        return True


class TargetInterval:
    __slots__ = ("start", "end")

    def __init__(self, start, end):
        self.start, self.end = start, end

    def __eq__(self, o):
        return isinstance(o, TargetInterval) and (self.start, self.end) == (o.start, o.end)

    def validate(self):
        if not (self.start < self.end):        # v4 accepted start > end
            return Unavailable(f"target interval is reversed or empty: "
                               f"[{self.start}, {self.end})", cause="BAD_INTERVAL")
        return True


class ForecastRequest:
    __slots__ = ("instrument", "as_of", "knowledge_cutoff", "target_interval",
                 "horizon", "link_ref")

    def __init__(self, instrument, as_of, knowledge_cutoff, target_interval,
                 horizon, link_ref):
        self.instrument, self.as_of = instrument, as_of
        self.knowledge_cutoff, self.target_interval = knowledge_cutoff, target_interval
        self.horizon, self.link_ref = horizon, link_ref


class AnchorObservables:
    __slots__ = ("s_fast", "s_slow", "t_known")

    def __init__(self, s_fast, s_slow, t_known):
        self.s_fast, self.s_slow, self.t_known = s_fast, s_slow, t_known


class ReducedFormPathLaw:
    """THE PRICING CARRIER. Route A only; no structural fields (M4-6).

    Deliberately has no `settlement_var`, no `increment_var` and no sampling
    convention. Terminal variance is reachable ONLY through
    `pricing_distribution`, which validates the request first.
    """
    __slots__ = ("instrument", "as_of", "fit_data_through", "target_interval",
                 "coverage_from", "coverage_to", "link_ref", "provenance",
                 "by_horizon", "status")

    def __init__(self, instrument, as_of, fit_data_through, target_interval,
                 coverage_from, coverage_to, link_ref, provenance, by_horizon,
                 status="FITTED"):
        self.instrument, self.as_of = instrument, as_of
        self.fit_data_through, self.target_interval = fit_data_through, target_interval
        self.coverage_from, self.coverage_to = coverage_from, coverage_to
        self.link_ref, self.provenance = link_ref, provenance
        self.by_horizon, self.status = by_horizon, status


class Distribution:
    """The single typed result of the atomic query."""
    __slots__ = ("mean", "var", "horizon", "instrument", "as_of", "provenance")

    def __init__(self, mean, var, horizon, instrument, as_of, provenance):
        self.mean, self.var, self.horizon = mean, var, horizon
        self.instrument, self.as_of, self.provenance = instrument, as_of, provenance


def check_request(law, req):
    """Every request/law invariant. Called BY pricing_distribution, not instead."""
    since = getattr(req, "as_of", None)
    if not isinstance(law, ReducedFormPathLaw):
        return Unavailable("pricing requires a ReducedFormPathLaw (route A); the "
                           "structural decomposition is a diagnostic and cannot "
                           "answer a pricing request", since, cause="WRONG_ROUTE")
    if law.status != "FITTED":
        return Unavailable(f"law status is {law.status!r}", since, cause="UNFITTED")
    for nm, iv in (("request", req.target_interval), ("law", law.target_interval)):
        ok = iv.validate() if isinstance(iv, TargetInterval) else Unavailable(
            f"{nm} target_interval is not a TargetInterval", since)
        if isinstance(ok, Unavailable):
            return ok
    if req.instrument != law.instrument:
        return Unavailable(f"instrument mismatch: {req.instrument!r} vs "
                           f"{law.instrument!r}", since, cause="INSTRUMENT_MISMATCH")
    if req.link_ref != law.link_ref:
        return Unavailable(f"link mismatch: {req.link_ref!r} vs {law.link_ref!r}",
                           since, cause="LINK_MISMATCH")
    if req.target_interval != law.target_interval:
        return Unavailable("target interval mismatch between request and law",
                           since, cause="TARGET_MISMATCH")
    if law.as_of > req.as_of:                   # v4 accepted a future-issued law
        return Unavailable(f"law is issued in the future relative to the request "
                           f"({law.as_of} > {req.as_of})", since,
                           cause="LAW_FROM_THE_FUTURE")
    if req.knowledge_cutoff > req.as_of:
        return Unavailable("knowledge_cutoff is after as_of", since,
                           cause="LOOKAHEAD")
    if law.fit_data_through > req.knowledge_cutoff:
        return Unavailable("law was fitted past the request's knowledge cutoff",
                           since, cause="LOOKAHEAD")
    if law.fit_data_through > law.as_of:
        return Unavailable("law fit_data_through is after its own as_of", since,
                           cause="LOOKAHEAD")
    if req.horizon not in law.by_horizon:
        return Unavailable(f"no fit at horizon {req.horizon}", since,
                           cause="NO_FIT_AT_HORIZON")
    if req.as_of + req.horizon != req.target_interval.end:
        return Unavailable("horizon inconsistent with as_of and target end", since,
                           cause="HORIZON_INCONSISTENT")
    if not (law.coverage_from <= req.as_of <= law.coverage_to):
        return Unavailable("as_of outside the law's coverage", since,
                           cause="OUT_OF_COVERAGE")
    return True


def pricing_distribution(law, request, observables):
    """THE ONLY pricing path. Atomic: validate, then mean AND variance together.

    Mean and variance come from the SAME validated fit, so they cannot be taken
    from inconsistent objects (M4-3). Returns Distribution or Unavailable.
    """
    ok = check_request(law, request)
    if isinstance(ok, Unavailable):
        return ok
    p = law.provenance.validate() if isinstance(law.provenance, StreamProvenance) \
        else Unavailable("law carries no StreamProvenance", request.as_of)
    if isinstance(p, Unavailable):
        return p
    if not isinstance(observables, AnchorObservables):
        return Unavailable("observables must be AnchorObservables", request.as_of)
    if observables.t_known > request.knowledge_cutoff:
        return Unavailable("observables are newer than the knowledge cutoff",
                           request.as_of, cause="LOOKAHEAD")
    fit = law.by_horizon[request.horizon]
    if not isinstance(fit, ReducedFormFit):
        return Unavailable(f"fit at {request.horizon} is not a ReducedFormFit",
                           request.as_of)
    v = fit.validate(f"fit[{request.horizon}]")
    if isinstance(v, Unavailable):
        return Unavailable(v.reason, request.as_of, v.cause)
    sf, ss = _num(observables.s_fast, "s_fast"), _num(observables.s_slow, "s_slow")
    for x in (sf, ss):
        if isinstance(x, Unavailable):
            return x
    alpha = F(fit.alpha)
    return Distribution(mean=ss + alpha * (sf - ss), var=F(fit.resid_var),
                        horizon=request.horizon, instrument=law.instrument,
                        as_of=request.as_of, provenance=law.provenance)


# ==================== ROUTE B — structural diagnostics only ==================
def _rect(k, lag=0):
    return tuple((lag + j, F(1, k)) for j in range(k))


SAMPLING_CONVENTIONS = {
    "disc1s_v0": dict(
        w=60, dt=1, status="UNVERIFIED", fast=_rect(30), slow=_rect(60),
        alignment="both right-aligned at t; synchronous support assumed",
        note="assumed from ~1 Hz publication cadence, which is a CADENCE and NOT "
             "A KERNEL. Gates ROUTE B ONLY -- route A never reads this.",
    ),
    "disc1s_lag_fast_1s": dict(
        w=60, dt=1, status="UNVERIFIED", fast=_rect(30, lag=1), slow=_rect(60),
        alignment="fast stream right-aligned one sample EARLIER than t",
        note="cheapest probe of the synchronous-support assumption.",
    ),
}
DEFAULT_CONVENTION = "disc1s_v0"


def _conv(name):
    c = SAMPLING_CONVENTIONS.get(name or DEFAULT_CONVENTION)
    return c if c is not None else Unavailable(f"unknown sampling convention {name!r}")


def _cov(a, b, dt=1):
    return sum(ai * bj * min(i, j) * dt for i, ai in a.items() for j, bj in b.items())


def _add(*terms):
    out = {}
    for coef, d in terms:
        for i, v in d.items():
            out[i] = out.get(i, F(0)) + coef * v
    return {i: v for i, v in out.items() if v != 0}


def _obs(c):
    w = c["w"]
    return ({w - int(o): x for o, x in c["fast"]}, {w - int(o): x for o, x in c["slow"]})


def mu_weights(r, w):
    if r >= w:
        return {w: F(1)}
    return _add((F(1), {i: F(1, w) for i in range(r + 1, w + 1)}), (F(r, w), {w: F(1)}))


def k_law(r, convention=None, grid_only=False):
    """Var(x_T - mu_t)/rate. UNITS ARE SECONDS (M4-6): rate[bps^2/s] * k[s] = bps^2."""
    c = _conv(convention)
    if isinstance(c, Unavailable):
        return c
    w = c["w"]
    r = _horizon(r, grid_only)
    if isinstance(r, Unavailable):
        return r
    if r <= w:
        return F(r * (r + 1) * (2 * r + 1), 6 * w * w)
    return F(r - w) + F((w + 1) * (2 * w + 1), 6 * w)


def alpha_star_model(r, convention=None):
    """The MODEL projection weight. A reference, never the definition of truth."""
    c = _conv(convention)
    if isinstance(c, Unavailable):
        return c
    w = c["w"]
    r = _horizon(r, False)
    if isinstance(r, Unavailable):
        return r
    fast, slow = _obs(c)
    d = _add((F(1), fast), (F(-1), slow))
    m = _add((F(1), mu_weights(r, w)), (F(-1), slow))
    return _cov(m, d, c["dt"]) / _cov(d, d, c["dt"])


def model_cond_var(r, convention=None):
    """CONDITIONAL variance at the MODEL PROJECTION ONLY. Seconds.

    M4-5: takes no alpha. v4 accepted one and returned cond-var PLUS squared bias
    whenever it differed from alpha*, then published that as `anchor_var` -- so an
    empirical alpha silently changed the "conditional variance". That is the exact
    variance/MSE conflation Revision 3 removed, reintroduced on route B.
    """
    c = _conv(convention)
    if isinstance(c, Unavailable):
        return c
    a = alpha_star_model(r, convention)
    if isinstance(a, Unavailable):
        return a
    fast, slow = _obs(c)
    err = _add((F(1), mu_weights(r, c["w"])), (F(-1), slow), (-a, fast), (a, slow))
    return _cov(err, err, c["dt"])


def selected_bias_coeff(r, alpha, convention=None):
    """(alpha - alpha*) — the KNOWN conditional bias, as a separate field."""
    a = alpha_star_model(r, convention)
    if isinstance(a, Unavailable):
        return a
    v = _num(alpha, "alpha")
    return v if isinstance(v, Unavailable) else v - a


def selected_uncond_mse(r, alpha, convention=None):
    """cond_var + bias^2. A separate diagnostic; never `anchor_var`."""
    c = _conv(convention)
    if isinstance(c, Unavailable):
        return c
    v = model_cond_var(r, convention)
    b = selected_bias_coeff(r, alpha, convention)
    for x in (v, b):
        if isinstance(x, Unavailable):
            return x
    fast, slow = _obs(c)
    d = _add((F(1), fast), (F(-1), slow))
    return v + b * b * _cov(d, d, c["dt"])


def feed_cov_validate(omega):
    """bps^2, symmetric, finite, PSD (M3-3/M4-3)."""
    if omega is None:
        return None
    try:
        vals = [_num(omega[i][j], f"Omega[{i}][{j}]") for i in (0, 1) for j in (0, 1)]
    except (TypeError, IndexError):
        return Unavailable(f"feed covariance is not a 2x2: {omega!r}")
    for v in vals:
        if isinstance(v, Unavailable):
            return v
    a, b, c2, d = vals
    if b != c2:
        return Unavailable(f"feed covariance is not symmetric: {float(b)} != {float(c2)}")
    if a < 0 or d < 0 or a * d - b * c2 < 0:
        return Unavailable(f"feed covariance is not PSD (diag {float(a)},{float(d)}; "
                           f"det {float(a * d - b * c2)})")
    return ((a, b), (c2, d))


class DiagnosticVarianceDecomposition:
    """Route B's result. A DISTINCT TYPE that cannot answer a pricing request.

    M4-5: v4 returned a dict and proved only that two key NAMES were absent. A
    dict with a renamed total is not a type boundary. `pricing_distribution`
    rejects this object on type, and it exposes no settlement_var/increment_var.
    """
    __slots__ = ("r", "diffusion", "cond_var_at_model", "feed", "model_total",
                 "alpha_star", "selected_alpha", "selected_bias_coeff",
                 "selected_uncond_mse", "convention", "convention_status")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    use = "DIAGNOSTIC_ONLY"


def model_var_diagnostic(r, sigma2_rate, selected_alpha=None, omega=None,
                         convention=None, grid_only=True):
    """Route-B decomposition. Never a pricing input; gated on the convention."""
    c = _conv(convention)
    if isinstance(c, Unavailable):
        return c
    rr = _horizon(r, grid_only)
    if isinstance(rr, Unavailable):
        return rr
    rate = _nonneg(sigma2_rate, "sigma2_rate")           # v4 raised on infinity
    if isinstance(rate, Unavailable):
        return rate
    om = feed_cov_validate(omega)
    if isinstance(om, Unavailable):
        return om
    a_star = alpha_star_model(rr, convention)
    cond = model_cond_var(rr, convention)                 # at the projection ONLY
    sel = a_star if selected_alpha is None else _num(selected_alpha, "selected_alpha")
    if isinstance(sel, Unavailable):
        return sel
    u = (sel, F(1) - sel)
    feed = F(0) if om is None else (u[0] * u[0] * om[0][0] + 2 * u[0] * u[1] * om[0][1]
                                    + u[1] * u[1] * om[1][1])
    return DiagnosticVarianceDecomposition(
        r=rr, diffusion=rate * k_law(rr, convention), cond_var_at_model=rate * cond,
        feed=feed, model_total=rate * k_law(rr, convention) + rate * cond + feed,
        alpha_star=a_star, selected_alpha=sel,
        selected_bias_coeff=selected_bias_coeff(rr, sel, convention),
        selected_uncond_mse=rate * selected_uncond_mse(rr, sel, convention),
        convention=convention or DEFAULT_CONVENTION, convention_status=c["status"])


# ------------------------------------------------------------- selftests -----
def _good_gate(kind):
    return GateEvidence(test=f"{kind}_test", conditioning="(S_fast-S_slow), |d|",
                        verdict="PASS", effect_size=0.01, ci_hi_abs=0.03,
                        tolerance=0.05, multiplicity="Holm across 6 horizons",
                        p_value=0.4)


def _good_law(**over):
    ti = TargetInterval(1000, 1060)
    fit = ReducedFormFit(alpha=F(3, 2), resid_var=F(30), n_effective=4000,
                         n_day_clusters=12, cross_fitted=True,
                         mean_gate=_good_gate("cond_mean"), var_gate=_good_gate("hetero"))
    kw = dict(instrument="BTC", as_of=900, fit_data_through=800, target_interval=ti,
              coverage_from=0, coverage_to=5000, link_ref=("logit", "v1"),
              provenance=StreamProvenance("chainlink:btc:s30", "chainlink:btc:s60",
                                          "NORM_ARITH_BPS", True, True, "VERIFIED"),
              by_horizon={60: fit})
    kw.update(over)
    return ReducedFormPathLaw(**kw), ti


def _req(ti, **over):
    kw = dict(instrument="BTC", as_of=1000, knowledge_cutoff=1000,
              target_interval=ti, horizon=60, link_ref=("logit", "v1"))
    kw.update(over)
    return ForecastRequest(**kw)


def selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + str(detail)) if detail != '' else ''}")

    law, ti = _good_law()
    obs = AnchorObservables(s_fast=10, s_slow=8, t_known=1000)

    print("M4-1: route A is NOT gated on the internal sampling convention:")
    d = pricing_distribution(law, _req(ti), obs)
    check("a well-formed route-A law PRICES while every convention is UNVERIFIED",
          isinstance(d, Distribution), f"mean={float(d.mean)} var={float(d.var)}"
          if isinstance(d, Distribution) else repr(d)[:70])
    check("...and the structural kernels still refuse to price at the same time",
          all(c["status"] == "UNVERIFIED" for c in SAMPLING_CONVENTIONS.values()))
    check("the pricing carrier has no sampling convention at all",
          "convention" not in ReducedFormPathLaw.__slots__
          and "provenance" in ReducedFormPathLaw.__slots__)
    bad_prov, _ = _good_law(provenance=StreamProvenance(
        "a", "b", "NORM_ARITH_BPS", True, True, "UNVERIFIED"))
    check("but UNVERIFIED STREAM provenance does refuse",
          isinstance(pricing_distribution(bad_prov, _req(ti), obs), Unavailable))
    npit, _ = _good_law(provenance=StreamProvenance(
        "a", "b", "NORM_ARITH_BPS", False, True, "VERIFIED"))
    check("non-point-in-time stream reads refuse",
          isinstance(pricing_distribution(npit, _req(ti), obs), Unavailable))

    print("M4-2: one atomic query; nothing can bypass the invariants:")
    check("mean and variance come from ONE call on ONE validated fit",
          isinstance(d, Distribution) and d.mean == F(11) and d.var == F(30))
    check("no public settlement_var/increment_var on the pricing carrier",
          not hasattr(law, "settlement_var") and not hasattr(law, "increment_var"))
    check("the only public pricing entry takes a request",
          pricing_distribution.__code__.co_varnames[:3] == ("law", "request", "observables"))
    fut, tif = _good_law(as_of=2000)
    check("refuses: law issued in the future (v4 returned True)",
          isinstance(pricing_distribution(fut, _req(tif), obs), Unavailable))
    rev = TargetInterval(2000, 1060)
    rlaw, _ = _good_law(target_interval=rev)
    check("refuses: reversed target interval (v4 returned True)",
          isinstance(pricing_distribution(rlaw, _req(rev, horizon=60), obs), Unavailable))
    for nm, kw in (("instrument", dict(instrument="ETH")),
                   ("link version", dict(link_ref=("logit", "v2"))),
                   ("knowledge cutoff after as_of", dict(knowledge_cutoff=1200)),
                   ("horizon with no fit", dict(horizon=30)),
                   ("horizon vs target end", dict(horizon=120, target_interval=ti))):
        check(f"refuses: {nm}",
              isinstance(pricing_distribution(law, _req(ti, **kw), obs), Unavailable))
    check("refuses: observables newer than the knowledge cutoff",
          isinstance(pricing_distribution(
              law, _req(ti), AnchorObservables(10, 8, t_known=1500)), Unavailable))
    r = pricing_distribution(law, _req(ti, instrument="ETH"), obs)
    check("refusals carry since AND a machine-actionable cause",
          r.since == 1000 and r.cause == "INSTRUMENT_MISMATCH", f"{r.cause}")

    print("M4-3: numeric fail-closed at the boundary:")
    for nm, fitkw in (("negative resid_var (v4 priced -1)", dict(resid_var=F(-1))),
                      ("zero resid_var", dict(resid_var=F(0))),
                      ("NaN day clusters (v4 passed)", dict(n_day_clusters=float("nan"))),
                      ("NaN in resid_var", dict(resid_var=float("nan"))),
                      ("infinite resid_var", dict(resid_var=float("inf"))),
                      ("too few day clusters", dict(n_day_clusters=2)),
                      ("not cross-fitted", dict(cross_fitted=False)),
                      ("float day clusters", dict(n_day_clusters=12.0))):
        base = dict(alpha=F(3, 2), resid_var=F(30), n_effective=4000,
                    n_day_clusters=12, cross_fitted=True,
                    mean_gate=_good_gate("m"), var_gate=_good_gate("h"))
        base.update(fitkw)
        bl, _ = _good_law(by_horizon={60: ReducedFormFit(**base)})
        check(f"refuses: {nm}",
              isinstance(pricing_distribution(bl, _req(ti), obs), Unavailable))
    check("infinite structural rate REFUSES (v4 raised OverflowError)",
          isinstance(model_var_diagnostic(60, float("inf")), Unavailable))
    check("NaN structural rate refuses",
          isinstance(model_var_diagnostic(60, float("nan")), Unavailable))
    check("evidence is per (symbol,horizon) fit, not one scalar on the parent",
          "n_day_clusters" in ReducedFormFit.__slots__
          and "n_day_clusters" not in ReducedFormPathLaw.__slots__)

    print("M4-4: non-rejection is not equivalence:")
    for nm, gk in (("INSUFFICIENT_EVIDENCE", dict(verdict="INSUFFICIENT_EVIDENCE")),
                   ("MODEL_REFUTED", dict(verdict="MODEL_REFUTED")),
                   ("PASS but CI bound exceeds tolerance",
                    dict(ci_hi_abs=0.5, tolerance=0.05)),
                   ("p-value above 1", dict(p_value=1.4)),
                   ("no pre-registered conditioning", dict(conditioning="")),
                   ("no multiplicity policy", dict(multiplicity=""))):
        g = _good_gate("m")
        for k, v in gk.items():
            setattr(g, k, v)
        bl, _ = _good_law(by_horizon={60: ReducedFormFit(
            F(3, 2), F(30), 4000, 12, True, g, _good_gate("h"))})
        check(f"refuses: {nm}",
              isinstance(pricing_distribution(bl, _req(ti), obs), Unavailable))
    g = _good_gate("m")
    g.verdict = "INSUFFICIENT_EVIDENCE"
    bl, _ = _good_law(by_horizon={60: ReducedFormFit(
        F(3, 2), F(30), 4000, 12, True, g, _good_gate("h"))})
    res = pricing_distribution(bl, _req(ti), obs)
    check("INSUFFICIENT_EVIDENCE is distinguishable from MODEL_REFUTED",
          res.cause == "INSUFFICIENT_EVIDENCE", res.cause)

    print("M4-5: route B keeps bias OUT of the conditional variance:")
    d0 = model_var_diagnostic(60, 1)
    d1 = model_var_diagnostic(60, 1, selected_alpha=F(17, 10))
    check("cond_var_at_model does NOT move with the selected alpha (v4 it did)",
          d0.cond_var_at_model == d1.cond_var_at_model,
          f"{float(d0.cond_var_at_model):.4f} both")
    check("model_cond_var takes no alpha argument at all",
          "alpha" not in model_cond_var.__code__.co_varnames[:2])
    check("the selected anchor's bias and MSE are SEPARATE fields",
          d1.selected_bias_coeff != 0 and d1.selected_uncond_mse > d1.cond_var_at_model,
          f"bias {float(d1.selected_bias_coeff):.6f}")
    check("route B returns a distinct TYPE, not a dict",
          isinstance(d0, DiagnosticVarianceDecomposition) and not isinstance(d0, dict))
    check("that type cannot answer a pricing request (type boundary, not key name)",
          isinstance(pricing_distribution(d0, _req(ti), obs), Unavailable)
          and pricing_distribution(d0, _req(ti), obs).cause == "WRONG_ROUTE")
    check("it exposes no settlement_var/increment_var a pricer could bind to",
          not hasattr(d0, "settlement_var") and not hasattr(d0, "increment_var"))

    print("kernel algebra (unchanged, exact):")
    check("k_law(30) == 1891/720", k_law(30) == F(1891, 720))
    check("alpha* == 2700/1801 exactly", alpha_star_model(180) == F(2700, 1801))
    check("k_law continuous at r=w", k_law(60) == F(61 * 121, 360))
    check("no stale wrong-fraction claim in the prose",
          "1799" not in open(__file__).read()[:open(__file__).read().index("def selftest")])

    print(f"\nroute A prices today? {isinstance(pricing_distribution(law, _req(ti), obs), Distribution)}"
          " for a SYNTHETIC law only — no real fit exists (Phase 0A 6 unrun).")
    print("route B under BTC rate 1.089^2, UNVERIFIED convention, diagnostics only:")
    print("   r   alpha*   diffusion  cond_var@model  model_total")
    for rr in HORIZON_GRID:
        D = model_var_diagnostic(rr, F(1089, 1000) ** 2)
        print(f" {rr:4d} {float(D.alpha_star):8.4f} {float(D.diffusion):11.3f}"
              f" {float(D.cond_var_at_model):15.3f} {float(D.model_total):12.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
