r"""Model fixture + runtime invariants for the 5-min binaries.

STATUS: FIXTURE. Phase 0A is open, the estimator is on HOLD, and the sampling
convention is UNVERIFIED. Nothing here may price. That is now ENFORCED rather
than asserted (M3-4): `pricing_var` refuses unless a reduced-form law has been
fitted AND its convention reads VERIFIED, and no API adds a reduced-form
residual to the structural lines.

THE ROUTE DECISION (ITER3 M3-1)
-------------------------------
Revision 3 recommended regressing observed x_T on observed (S30, S60) AND kept
the structural ledger sigma^2*k_law + sigma^2*v + u'Omega u. Those are two
estimators of the same quantity. The regression residual ALREADY contains future
innovation, latent-path uncertainty, stream error and their covariances, so
adding the structural lines double-counts them.

  ROUTE A  REDUCED FORM  -> the PRICING law.       pricing_var()
  ROUTE B  STRUCTURAL    -> DIAGNOSTICS ONLY.      model_var_diagnostic()

They are never summed. The consumer matrix decides: the only consumer needing a
LEVEL is the BE-Belief fallback, which needs Sigma(r), not its decomposition.
The decomposition is needed only by c(r), the k-ledger and H-3, none of which is
a gate.

The routes have DIFFERENT PREREQUISITES and DIFFERENT DELIVERABLES:

  A: needs no sampling semantics (a regression on published streams does not
     care how Chainlink builds them). Does NOT identify Omega -- Omega is inside
     its residual. Yields a pricing law only.
  B: needs the semantics (you must know the true covariance's shape to
     extrapolate it to lag 0). Identifies Omega as the lag-0 discontinuity of
     the bivariate cross-variogram -- i.e. the NUGGET. Yields the decomposition.

WHAT OLS DOES AND DOES NOT GIVE YOU (M3-1)
------------------------------------------
OLS gives the best LINEAR PROJECTION and a POOLED residual variance. That is the
conditional mean only if the conditional mean is linear, and the conditional
variance only under homoskedasticity. Otherwise the pooled residual is an
UNCONDITIONAL forecast MSE -- the same category error this file removed from the
Brownian variance line one revision ago, one level up. Under the Gaussian
fixture they coincide; the whole point of going empirical is not to rely on the
fixture. So Route A ships only with the residual-diagnostic gates in
`ReducedFormLaw`, and refuses without them.

    python3 sigma_kernels.py --selftest
"""
from fractions import Fraction as F

HORIZON_GRID = (30, 60, 120, 180, 240, 270)


class Unavailable:
    """Typed refusal matching contracts.yaml Unavailable{reason, since, cause}.

    Deliberately not a float: arithmetic on it raises rather than propagating a
    silently-wrong number.
    """
    __slots__ = ("reason", "since", "cause")

    def __init__(self, reason, since=None, cause=None):
        self.reason, self.since, self.cause = reason, since, cause

    def __repr__(self):
        c = f", cause={self.cause!r}" if self.cause is not None else ""
        return f"Unavailable({self.reason!r}, since={self.since!r}{c})"

    def __bool__(self):
        return False


# ------------------------------------------------ sampling conventions (M3-4) --
# A convention is a WEIGHT SCHEDULE: (offset_seconds_before_t, weight) pairs per
# stream, plus alignment semantics. v2 stored only window lengths and a lag flag
# and always built rectangular trailing means, so an aligned and a lagged
# schedule could carry identical weight lists. Offsets fix that.
def _rect(k, lag=0):
    """Rectangular trailing mean of k samples, right edge lagged by `lag` s."""
    return tuple((lag + j, F(1, k)) for j in range(k))


SAMPLING_CONVENTIONS = {
    "disc1s_v0": dict(
        w=60, dt=1, status="UNVERIFIED",
        fast=_rect(30), slow=_rect(60),
        alignment="both right-aligned at t; synchronous support assumed",
        note="assumed from ~1 Hz publication cadence, which is a CADENCE and "
             "NOT A KERNEL. EXP-M6 proves the published S60 endpoint reproduces "
             "settlement; it says nothing about how that endpoint is built.",
    ),
    "disc1s_lag_fast_1s": dict(
        w=60, dt=1, status="UNVERIFIED",
        fast=_rect(30, lag=1), slow=_rect(60),
        alignment="fast stream right-aligned one sample EARLIER than t",
        note="cheapest probe of the synchronous-support assumption.",
    ),
}
DEFAULT_CONVENTION = "disc1s_v0"


def _conv(name):
    name = name or DEFAULT_CONVENTION
    c = SAMPLING_CONVENTIONS.get(name)
    if c is None:
        return Unavailable(f"unknown sampling convention {name!r}")
    return c


# ------------------------------------------------------------ validation ------
def _check_r(r, grid_only):
    if isinstance(r, bool) or not isinstance(r, (int, F)):
        return Unavailable(f"r must be an exact integer/Fraction, got {type(r).__name__}")
    if r != int(r):
        return Unavailable(f"r={r} is off the sample grid")
    r = int(r)
    if r <= 0:
        return Unavailable(f"r={r} is not a positive horizon")
    if grid_only and r not in HORIZON_GRID:
        return Unavailable(f"r={r} is not on HORIZON_GRID {HORIZON_GRID}")
    return r


def _check_rate(sigma2):
    """M3-4: v2 accepted a negative rate and returned a negative variance."""
    try:
        v = F(sigma2)
    except (TypeError, ValueError):
        return Unavailable(f"sigma2_rate is not a number: {sigma2!r}")
    if v < 0:
        return Unavailable(f"sigma2_rate={float(v)} is negative")
    return v


def check_feed_cov(omega):
    """M3-3: symmetry, finiteness and PSD, in PHYSICAL bps^2. Refuses otherwise.

    UNITS ARE bps^2, matching SIGMA_PLAN section 3.2 and contracts FeedErrorCov.
    v2's code documented "sigma^2 units" and multiplied by sigma2 in the ledger,
    so with sigma2=4 an identity matrix contributed 9.9867 instead of 2.4967.
    One convention only; the sigma^2 multiplication is gone.
    """
    if omega is None:
        return None
    try:
        a, b = F(omega[0][0]), F(omega[0][1])
        c, d = F(omega[1][0]), F(omega[1][1])
    except (TypeError, ValueError, IndexError):
        return Unavailable(f"feed covariance is not a 2x2 of numbers: {omega!r}")
    if b != c:
        return Unavailable(f"feed covariance is not symmetric: {float(b)} != {float(c)}")
    if a < 0 or d < 0 or a * d - b * c < 0:          # 2x2 PSD test
        return Unavailable(
            f"feed covariance is not positive semidefinite "
            f"(diag {float(a)},{float(d)}; det {float(a * d - b * c)})")
    return ((a, b), (c, d))


# ------------------------------------------------- exact covariance algebra ---
# Grid index i counts SECONDS AFTER (t - w); i = w is decision time t.
# Cov(B_i, B_j) = min(i, j) * dt, in units of the variance rate.
def _cov(a, b, dt=1):
    return sum(ai * bj * min(i, j) * dt for i, ai in a.items() for j, bj in b.items())


def _add(*terms):
    out = {}
    for coef, d in terms:
        for i, v in d.items():
            out[i] = out.get(i, F(0)) + coef * v
    return {i: v for i, v in out.items() if v != 0}


def _schedule_weights(schedule, w):
    """(offset_before_t, weight) pairs -> grid weights."""
    return {w - int(off): wt for off, wt in schedule}


def _obs(c):
    w = c["w"]
    return _schedule_weights(c["fast"], w), _schedule_weights(c["slow"], w)


def mu_weights(r, w):
    """E[x_T | full path to t] as a path functional (total weight 1)."""
    if r >= w:
        return {w: F(1)}
    pre = {i: F(1, w) for i in range(r + 1, w + 1)}
    return _add((F(1), pre), (F(r, w), {w: F(1)}))


def k_law(r, convention=None, grid_only=False):
    """Var(x_T - mu_t)/rate, the post-t innovation. DIAGNOSTIC (route B)."""
    c = _conv(convention)
    if isinstance(c, Unavailable):
        return c
    w = c["w"]
    r = _check_r(r, grid_only)
    if isinstance(r, Unavailable):
        return r
    if r <= w:
        return F(r * (r + 1) * (2 * r + 1), 6 * w * w)
    return F(r - w) + F((w + 1) * (2 * w + 1), 6 * w)


def alpha_star_model(r, convention=None):
    """The MODEL-IMPLIED projection weight. A REFERENCE, never the definition
    of correctness (M3-2). Under disc1s_v0 and r > w this is 2700/1801."""
    c = _conv(convention)
    if isinstance(c, Unavailable):
        return c
    w = c["w"]
    r = _check_r(r, False)
    if isinstance(r, Unavailable):
        return r
    fast, slow = _obs(c)
    d = _add((F(1), fast), (F(-1), slow))
    m = _add((F(1), mu_weights(r, w)), (F(-1), slow))
    return _cov(m, d, c["dt"]) / _cov(d, d, c["dt"])


def model_cond_var(r, alpha=None, convention=None):
    """Var(mu_t - P_hat | fast, slow)/rate under the declared model.

    At alpha = alpha_star_model this is the conditional variance and is
    alpha-independent. At any other alpha it is that PLUS squared bias, i.e. an
    unconditional MSE -- which is why the caller must say which it wants.
    """
    c = _conv(convention)
    if isinstance(c, Unavailable):
        return c
    w = c["w"]
    a_star = alpha_star_model(r, convention)
    if isinstance(a_star, Unavailable):
        return a_star
    a = a_star if alpha is None else F(alpha)
    fast, slow = _obs(c)
    err = _add((F(1), mu_weights(r, w)), (F(-1), slow), (-a, fast), (a, slow))
    return _cov(err, err, c["dt"])


def feed_var(alpha, omega, convention=None):
    """u' Omega u with u = (alpha, 1-alpha), in bps^2. No rate multiplication."""
    om = check_feed_cov(omega)
    if isinstance(om, Unavailable):
        return om
    if om is None:
        return F(0)
    a = F(alpha)
    u = (a, F(1) - a)
    return (u[0] * u[0] * om[0][0] + 2 * u[0] * u[1] * om[0][1]
            + u[1] * u[1] * om[1][1])


# ------------------------------------------------------- the anchor (M3-2) ----
class AnchorSpec:
    """Horizon-scoped anchor. Separates the MODEL coefficient from the
    ESTIMATED one and defines bias against whichever is SELECTED.

    v2's fixture computed bias = alpha - alpha_star_model unconditionally, so an
    empirically fitted alpha was always labelled biased and the documented mean
    correction dragged the centre back to the Brownian projection. No empirical
    coefficient could ever become the zero-bias mean. Here `selected` says which
    estimand defines correctness, and bias is measured against THAT.
    """
    __slots__ = ("by_horizon", "selected_estimand", "convention")

    def __init__(self, by_horizon, selected_estimand, convention=None):
        # by_horizon: {r: {"model": F, "estimated": F | None}}
        # `selected_estimand` mirrors contracts AnchorSpec.selected: MODEL|ESTIMATED.
        assert selected_estimand in ("MODEL", "ESTIMATED")
        self.by_horizon = by_horizon
        self.selected_estimand, self.convention = selected_estimand, convention

    @classmethod
    def from_model(cls, convention=None, horizons=HORIZON_GRID):
        return cls({r: {"model": alpha_star_model(r, convention), "estimated": None}
                    for r in horizons}, "MODEL", convention)

    def selected(self, r):
        e = self.by_horizon.get(r)
        if e is None:
            return Unavailable(f"anchor has no entry for r={r}")
        if self.selected_estimand == "ESTIMATED":
            if e.get("estimated") is None:
                return Unavailable(f"selected=ESTIMATED but no estimate at r={r}")
            return e["estimated"]
        return e["model"]

    def bias_coeff(self, r):
        """Bias of the SELECTED anchor, as a multiple of (S_fast - S_slow).

        Zero by construction when the selected coefficient IS the estimand --
        which is the point: a fitted conditional mean is unbiased with respect to
        itself, and must not be 'corrected' toward the model.
        """
        return F(0) if isinstance(self.selected(r), F) else self.selected(r)

    def model_gap(self, r):
        """estimated - model. A DIAGNOSTIC on the process. Never a correction."""
        e = self.by_horizon.get(r, {})
        if e.get("estimated") is None:
            return Unavailable(f"no estimate at r={r} to compare with the model")
        return e["estimated"] - e["model"]


def conditional_mean(anchor, r, s_fast, s_slow):
    """E_t[x_T] under the SELECTED anchor: S_slow + alpha*(S_fast - S_slow).

    Implemented (M3-2 asked for it) so the mean is reachable without going
    through the variance, and so the selected coefficient is demonstrably the
    centre rather than something a bias term cancels back to the model.
    """
    a = anchor.selected(r)
    if isinstance(a, Unavailable):
        return a
    return F(s_slow) + a * (F(s_fast) - F(s_slow))


# ------------------------------------------- ROUTE B: diagnostics only --------
def model_var_diagnostic(r, sigma2_rate, anchor=None, omega=None,
                         convention=None, grid_only=True):
    """The STRUCTURAL decomposition. DIAGNOSTIC ONLY -- never a pricing input,
    and never added to a reduced-form residual (M3-1).

    sigma2_rate is bps^2 PER SECOND; k_law and the anchor coefficient are
    dimensionless; omega is bps^2. Result is bps^2.
    """
    c = _conv(convention)
    if isinstance(c, Unavailable):
        return c
    rr = _check_r(r, grid_only)
    if isinstance(rr, Unavailable):
        return rr
    rate = _check_rate(sigma2_rate)
    if isinstance(rate, Unavailable):
        return rate
    anchor = anchor or AnchorSpec.from_model(convention)
    a = anchor.selected(rr)
    if isinstance(a, Unavailable):
        return a
    fv = feed_var(a, omega, convention)
    if isinstance(fv, Unavailable):
        return fv
    diffusion = rate * k_law(rr, convention)
    anchor_v = rate * model_cond_var(rr, a, convention)
    return dict(
        use="DIAGNOSTIC_ONLY",
        r=rr, alpha=a, alpha_selected=anchor.selected_estimand,
        alpha_model=anchor.by_horizon[rr]["model"],
        diffusion_var=diffusion, anchor_var=anchor_v, feed_var=fv,
        model_total_var=diffusion + anchor_v + fv,
        convention=convention or DEFAULT_CONVENTION,
        convention_status=c["status"],
    )


# ------------------------------------------- ROUTE A: the pricing law ---------
class ReducedFormLaw:
    """Fitted conditional law of x_T on (S_fast, S_slow), per horizon.

    Its residual variance is the WHOLE of Sigma(r): future innovation, latent
    path uncertainty, stream error and their covariances are all inside it.
    Nothing structural is added to it, ever.

    OLS gives a linear projection and a POOLED residual. That is a conditional
    law only if the residual has conditional mean zero and is homoskedastic in
    the conditioning variables, so those are GATES, not footnotes.
    """
    __slots__ = ("by_horizon", "convention", "n_day_clusters", "cross_fitted",
                 "resid_mean_test_p", "hetero_test_p", "status")

    def __init__(self, by_horizon, convention, n_day_clusters, cross_fitted,
                 resid_mean_test_p, hetero_test_p, status="FITTED"):
        self.by_horizon = by_horizon        # {r: {"alpha": F, "resid_var": F}}
        self.convention, self.status = convention, status
        self.n_day_clusters, self.cross_fitted = n_day_clusters, cross_fitted
        self.resid_mean_test_p, self.hetero_test_p = resid_mean_test_p, hetero_test_p


MIN_DAY_CLUSTERS = 10


def pricing_var(law, r, grid_only=True):
    """Sigma(r) for pricing. Refuses unless every precondition holds (M3-4).

    This is the ONLY function that may feed a probability. It refuses under an
    unverified sampling convention, an unfitted law, too few day clusters, no
    cross-fitting, or failed residual diagnostics.
    """
    if not isinstance(law, ReducedFormLaw):
        return Unavailable("pricing requires a fitted ReducedFormLaw (route A); "
                           "the structural decomposition is diagnostic only")
    c = _conv(law.convention)
    if isinstance(c, Unavailable):
        return c
    if c["status"] != "VERIFIED":
        return Unavailable(f"sampling convention {law.convention!r} is "
                           f"{c['status']}; no unverified convention may price")
    if law.status != "FITTED":
        return Unavailable(f"law status is {law.status!r}")
    rr = _check_r(r, grid_only)
    if isinstance(rr, Unavailable):
        return rr
    if law.n_day_clusters < MIN_DAY_CLUSTERS:
        return Unavailable(f"{law.n_day_clusters} day clusters < {MIN_DAY_CLUSTERS}; "
                           "descriptive coefficient, not a pricing law")
    if not law.cross_fitted:
        return Unavailable("law is not cross-fitted; in-sample residual variance "
                           "understates conditional variance")
    if law.resid_mean_test_p is None or law.resid_mean_test_p < 0.01:
        return Unavailable("residual conditional-mean test failed: the linear "
                           "projection is not the conditional mean here")
    if law.hetero_test_p is None or law.hetero_test_p < 0.01:
        return Unavailable("heteroskedasticity test failed: the pooled residual "
                           "variance is an unconditional MSE, not Var_t")
    e = law.by_horizon.get(rr)
    if e is None:
        return Unavailable(f"law has no fitted entry for r={rr}")
    return e["resid_var"]


# --------------------------------------------- request invariants (M3-5) -----
class TargetInterval:
    __slots__ = ("start", "end")

    def __init__(self, start, end):
        self.start, self.end = start, end

    def __eq__(self, o):
        return isinstance(o, TargetInterval) and (self.start, self.end) == (o.start, o.end)


class ForecastRequest:
    __slots__ = ("instrument", "as_of", "knowledge_cutoff", "target_interval",
                 "horizon", "link_ref")

    def __init__(self, instrument, as_of, knowledge_cutoff, target_interval,
                 horizon, link_ref):
        self.instrument, self.as_of = instrument, as_of
        self.knowledge_cutoff, self.target_interval = knowledge_cutoff, target_interval
        self.horizon, self.link_ref = horizon, link_ref


class LawHeader:
    """The PathLaw fields the request must agree with."""
    __slots__ = ("instrument", "as_of", "fit_data_through", "target_interval",
                 "horizon_grid", "coverage_from", "coverage_to", "link_ref")

    def __init__(self, instrument, as_of, fit_data_through, target_interval,
                 horizon_grid, coverage_from, coverage_to, link_ref):
        self.instrument, self.as_of = instrument, as_of
        self.fit_data_through, self.target_interval = fit_data_through, target_interval
        self.horizon_grid = horizon_grid
        self.coverage_from, self.coverage_to = coverage_from, coverage_to
        self.link_ref = link_ref


def check_request(law, req):
    """Executable request/law invariants. R-WFWD and R-REQ.

    v14 added the timestamps and the checker reported them green -- but the
    checker only records check STRINGS, it never evaluates the inequalities. A
    typed timestamp that is never compared is documentation, not look-ahead
    protection (M3-5). These are the comparisons.
    """
    if req.instrument != law.instrument:
        return Unavailable(f"instrument mismatch: request {req.instrument!r} "
                           f"vs law {law.instrument!r}")
    if req.link_ref != law.link_ref:
        return Unavailable(f"link mismatch: request {req.link_ref!r} vs law "
                           f"{law.link_ref!r}; a law may not be read through a "
                           "different link than the belief uses")
    if req.target_interval != law.target_interval:
        return Unavailable("target interval mismatch between request and law")
    if req.knowledge_cutoff > req.as_of:
        return Unavailable("knowledge_cutoff is after as_of: look-ahead")
    if law.fit_data_through > req.knowledge_cutoff:
        return Unavailable("law was fitted on data past the request's knowledge "
                           "cutoff: look-ahead (R-WFWD no_future_train)")
    if law.fit_data_through > law.as_of:
        return Unavailable("law fit_data_through is after its own as_of "
                           "(R-WFWD cutoff_order)")
    if req.horizon not in law.horizon_grid:
        return Unavailable(f"horizon {req.horizon} outside the law's grid")
    if req.as_of + req.horizon != req.target_interval.end:
        return Unavailable("horizon inconsistent with as_of and target end")
    if not (law.coverage_from <= req.as_of <= law.coverage_to):
        return Unavailable("as_of outside the law's coverage")
    return True


# ------------------------------------------------------------- selftests -----
def selftest(sigma_btc=1.089):
    ok = True
    rate = F(1089, 1000) ** 2

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        d = f"  {detail}" if detail != "" else ""
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{d}")

    print("kernel (exact rationals):")
    check("k_law(30) == 1891/720", k_law(30) == F(1891, 720), str(k_law(30)))
    check("k_law continuous at r=w", k_law(60) == F(61 * 121, 360), str(k_law(60)))
    check("alpha_star_model(r>w) == 2700/1801 EXACTLY",
          alpha_star_model(180) == F(2700, 1801),
          f"{alpha_star_model(180)} = {float(alpha_star_model(180)):.7f}")
    # The v3 header asserted a fraction its own test contradicted. Guard the
    # PROSE (docstring + code above selftest); the literal necessarily appears
    # inside this guard, so the guard must not scan itself.
    _src = open(__file__).read()
    _prose = _src[:_src.index("def selftest")]
    check("no stale wrong-fraction claim in the module prose",
          "1799" not in _prose and "1799" not in (__doc__ or ""))

    print("M3-1: the two routes cannot be combined:")
    d = model_var_diagnostic(60, rate)
    check("structural result is tagged DIAGNOSTIC_ONLY", d["use"] == "DIAGNOSTIC_ONLY")
    check("structural result exposes no key a pricer would reach for",
          "total_var" not in d and "sigma_eff" not in d, str(sorted(d)[:4]))
    check("pricing_var REFUSES a structural dict",
          isinstance(pricing_var(d, 60), Unavailable))
    check("pricing_var refuses anything that is not a ReducedFormLaw",
          isinstance(pricing_var(None, 60), Unavailable)
          and isinstance(pricing_var(3.0, 60), Unavailable))

    print("M3-4: fail closed:")
    good = dict(by_horizon={60: {"alpha": F(3, 2), "resid_var": F(30)}},
                n_day_clusters=12, cross_fitted=True,
                resid_mean_test_p=0.4, hetero_test_p=0.3)
    unver = ReducedFormLaw(convention="disc1s_v0", **good)
    check("UNVERIFIED convention REFUSES to price",
          isinstance(pricing_var(unver, 60), Unavailable),
          repr(pricing_var(unver, 60))[:64])
    SAMPLING_CONVENTIONS["_test_verified"] = dict(
        SAMPLING_CONVENTIONS["disc1s_v0"], status="VERIFIED")
    ver = ReducedFormLaw(convention="_test_verified", **good)
    check("VERIFIED + fitted + gates passed does price", pricing_var(ver, 60) == F(30))
    for name, kw in (("too few day clusters", dict(n_day_clusters=2)),
                     ("not cross-fitted", dict(cross_fitted=False)),
                     ("residual mean test fails", dict(resid_mean_test_p=0.001)),
                     ("heteroskedastic", dict(hetero_test_p=0.001))):
        bad = ReducedFormLaw(convention="_test_verified", **{**good, **kw})
        check(f"refuses: {name}", isinstance(pricing_var(bad, 60), Unavailable))
    del SAMPLING_CONVENTIONS["_test_verified"]
    check("negative rate refuses", isinstance(model_var_diagnostic(30, -1), Unavailable),
          repr(model_var_diagnostic(30, -1))[:52])
    check("unknown convention refuses (v2 raised KeyError)",
          isinstance(model_var_diagnostic(30, 1, convention="nope"), Unavailable))
    check("off-grid / negative / float r refuse",
          all(isinstance(model_var_diagnostic(b, 1), Unavailable)
              for b in (-1, 0, 45, 30.4)))
    check("Unavailable matches the contract {reason, since, cause}",
          Unavailable.__slots__ == ("reason", "since", "cause"))

    print("M3-3: Omega units, PSD, and rate dimension:")
    I = ((F(1), F(0)), (F(0), F(1)))
    a_star = alpha_star_model(60)
    uIu = feed_var(a_star, I)
    check("Omega is bps^2 and is NOT multiplied by the rate",
          model_var_diagnostic(60, F(4), omega=I)["feed_var"] == uIu,
          f"{float(uIu):.4f} bps^2 (v2 gave {float(uIu) * 4:.4f} at rate 4)")
    check("exact unit fixture with rate != 1 (v2 had no such test)",
          model_var_diagnostic(60, F(4), omega=I)["diffusion_var"] == 4 * k_law(60))
    check("non-symmetric Omega refuses",
          isinstance(feed_var(2, ((F(1), F(1)), (F(2), F(1)))), Unavailable))
    check("non-PSD Omega refuses (v2 returned total_var = -120.9)",
          isinstance(model_var_diagnostic(60, 1, omega=((F(0), F(100)), (F(100), F(0)))),
                     Unavailable))
    check("negative diagonal refuses",
          isinstance(feed_var(2, ((F(-1), F(0)), (F(0), F(1)))), Unavailable))
    check("PSD boundary (det == 0) is accepted",
          not isinstance(feed_var(2, ((F(1), F(1)), (F(1), F(1)))), Unavailable))

    print("M3-2: an empirical alpha can be the mean:")
    emp = AnchorSpec({60: {"model": alpha_star_model(60), "estimated": F(17, 10)}},
                     "ESTIMATED")   # contracts AnchorSpec.selected
    check("selected() returns the ESTIMATE, not the model",
          emp.selected(60) == F(17, 10))
    check("bias of the selected anchor is ZERO (v2 reported 0.200833)",
          emp.bias_coeff(60) == 0)
    check("the model gap survives as a DIAGNOSTIC",
          abs(float(emp.model_gap(60)) - 0.200833) < 1e-5,
          f"{float(emp.model_gap(60)):.6f}")
    cm = conditional_mean(emp, 60, s_fast=10, s_slow=8)
    check("conditional_mean uses the estimate: 8 + 1.7*2 = 11.4",
          cm == F(114, 10), str(float(cm)))
    check("conditional_mean under the model anchor differs",
          conditional_mean(AnchorSpec.from_model(), 60, 10, 8) != cm)
    check("ESTIMATED provenance with no estimate REFUSES",
          isinstance(AnchorSpec({30: {"model": F(1), "estimated": None}},
                                "ESTIMATED").selected(30), Unavailable))
    check("field names mirror contracts AnchorSpec.selected (MODEL|ESTIMATED)",
          AnchorSpec.from_model().selected_estimand == "MODEL"
          and "alpha_selected" in model_var_diagnostic(60, 1))
    check("anchor is horizon-scoped, not a scalar",
          alpha_star_model(30) != alpha_star_model(180),
          f"alpha*(30)={float(alpha_star_model(30)):.4f} "
          f"vs alpha*(180)={float(alpha_star_model(180)):.4f}")

    print("M3-5: request invariants are EVALUATED, not just typed:")
    ti = TargetInterval(1000, 1060)
    law = LawHeader("BTC", as_of=900, fit_data_through=800, target_interval=ti,
                    horizon_grid=HORIZON_GRID, coverage_from=0, coverage_to=5000,
                    link_ref=("logit", "v1"))
    req = ForecastRequest("BTC", as_of=1000, knowledge_cutoff=1000,
                          target_interval=ti, horizon=60, link_ref=("logit", "v1"))
    check("a consistent request passes", check_request(law, req) is True)
    negatives = {
        "instrument swapped": dict(instrument="ETH"),
        "link version swapped": dict(link_ref=("logit", "v2")),
        "target interval swapped": dict(target_interval=TargetInterval(1000, 1120)),
        "knowledge cutoff after as_of": dict(knowledge_cutoff=1200),
        "horizon off the grid": dict(horizon=45),
        "horizon inconsistent with target end": dict(horizon=30),
    }
    for name, kw in negatives.items():
        base = dict(instrument=req.instrument, as_of=req.as_of,
                    knowledge_cutoff=req.knowledge_cutoff,
                    target_interval=req.target_interval, horizon=req.horizon,
                    link_ref=req.link_ref)
        check(f"refuses: {name}",
              isinstance(check_request(law, ForecastRequest(**{**base, **kw})),
                         Unavailable))
    late = LawHeader("BTC", 900, 1200, ti, HORIZON_GRID, 0, 5000, ("logit", "v1"))
    check("refuses: law fitted past the request's knowledge cutoff",
          isinstance(check_request(late, req), Unavailable))
    ahead = LawHeader("BTC", 700, 800, ti, HORIZON_GRID, 0, 5000, ("logit", "v1"))
    check("refuses: fit_data_through after the law's own as_of",
          isinstance(check_request(ahead, req), Unavailable))

    print("\nstructural DIAGNOSTIC under the model anchor "
          "(BTC rate 1.089^2 bps^2/s, UNVERIFIED, no feed error):")
    print("   r   alpha*   diffusion   anchor    model_total   NOT A PRICE")
    for r in HORIZON_GRID:
        D = model_var_diagnostic(r, rate)
        print(f" {r:4d} {float(D['alpha']):8.4f} {float(D['diffusion_var']):11.3f}"
              f" {float(D['anchor_var']):8.3f} {float(D['model_total_var']):13.3f}")
    print("  These are route-B diagnostics. Pricing goes through pricing_var(),")
    print("  which refuses today: no law is fitted and no convention is VERIFIED.")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
