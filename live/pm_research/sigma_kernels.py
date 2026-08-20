#!/usr/bin/env python3
r"""Model fixture for the 5-min binaries: settlement kernels and the anchor law.

STATUS: FIXTURE, NOT A FROZEN SPEC. Phase 0A steps 1-3 are REOPENED
(SIGMA_PLAN_REVIEW_ITER2.md). Everything here is exact arithmetic under a
DECLARED and CURRENTLY UNVERIFIED sampling convention. Nothing in this file may
be used as a pricing input until `SamplingConvention.status` reads VERIFIED.

What changed from the first version, and why (ITER2 M2-1)
---------------------------------------------------------
v1 computed the unconditional MSE of the fixed extrapolator P_hat = 2*S30 - S60
and entered it in the probability law as a zero-mean variance line. That is
wrong. 2/-1 is a local-linear-TREND extrapolator imposed on a path model whose
trajectories have no local derivative; it is not E[P_t | S30, S60]. Writing
P_hat(alpha) = S60 + alpha*(S30 - S60), the error splits as

    P_hat(alpha) - P_t = (alpha - alpha*)*(S30 - S60)  +  projection error
                          \_ conditional BIAS, known _/    \_ zero-mean _/
                             at decision time

so the first term belongs in the MEAN model and only the second is variance.
Under this convention alpha* = 1799/1200 = 1.4991667, NOT 2. At BTC's sigma the
bias the old spec buried in the variance has sd 1.22 bps against a total
sigma_eff(30) of ~2.4 bps -- i.e. HALF the standard deviation at r=30 was a
predictable, observable mean error. That is the same defect class as the
original S60-anchor bug this whole thread started from.

Consequences carried through here:
  - the conditional variance of the anchor error is alpha-INDEPENDENT (P_hat is
    a function of the conditioning variables), so the variance line is 8.2590,
    not 9.5139; the difference was squared bias;
  - alpha is a FREE PARAMETER to be estimated on the tape. alpha* is what the
    Brownian model implies, not what the market must do. Its distance from the
    estimate is a diagnostic on the process, exactly like w_hat_free;
  - 9.5139 was called a "floor". It is not: alpha* achieves 8.2590 within this
    same model (M2-3). Nothing here is a bound.

    python3 sigma_kernels.py --selftest
"""
from fractions import Fraction as F

# ------------------------------------------------------- sampling convention --
# M2-2: the averaging kernel is a VERSIONED WEIGHT SCHEDULE, not a constant.
# ~1 Hz publication cadence does NOT prove a 60-point rectangular internal
# kernel, synchronous S30/S60 support, or equal one-second weights. EXP-M6
# proves the published S60 endpoint reproduces settlement; it says nothing about
# how that endpoint is built. Until the semantics study (Phase 0A step 5) runs,
# every convention below is a SENSITIVITY FIXTURE.
SAMPLING_CONVENTIONS = {
    "disc1s_v0": dict(
        w=60, dt=1, s_fast=30, s_slow=60, status="UNVERIFIED",
        note="60 equally-weighted 1 s samples, S30/S60 synchronous and "
             "right-aligned at t. Assumed from ~1 Hz publication cadence, "
             "which is a cadence and not a kernel.",
    ),
    "disc1s_lag_s30_1s": dict(
        w=60, dt=1, s_fast=30, s_slow=60, status="UNVERIFIED", s_fast_lag=1,
        note="same, but S30 right-aligned one sample earlier -- the cheapest "
             "probe of the synchronous-support assumption.",
    ),
}
DEFAULT_CONVENTION = "disc1s_v0"
HORIZON_GRID = (30, 60, 120, 180, 240, 270)   # r = T - t, the decision horizons


class Unavailable:
    """Typed refusal. Deliberately NOT a float: arithmetic on it raises."""
    __slots__ = ("reason",)

    def __init__(self, reason):
        self.reason = reason

    def __repr__(self):
        return f"Unavailable({self.reason!r})"

    def __bool__(self):
        return False


def _conv(name):
    c = SAMPLING_CONVENTIONS[name or DEFAULT_CONVENTION]
    return c["w"], c


def _check_r(r, w, grid_only):
    """M2-4: exact domain validation. v1 accepted r=-1 and r=30.4 silently."""
    if isinstance(r, bool) or not isinstance(r, (int, F)):
        return Unavailable(f"r must be an exact integer/Fraction, got {type(r).__name__}")
    if r != int(r):
        return Unavailable(f"r={r} is off the {1}s sample grid")
    r = int(r)
    if r <= 0:
        return Unavailable(f"r={r} is not a positive horizon")
    if grid_only and r not in HORIZON_GRID:
        return Unavailable(f"r={r} is not on HORIZON_GRID {HORIZON_GRID}")
    return r


# ------------------------------------------------- exact covariance algebra --
# Grid index i = time (t - w + i); i = w is decision time t. B_0 is the level
# reference and cancels from every functional used here (all have total weight
# 1, so differences have total weight 0 and are translation-invariant).
# Cov(B_i, B_j) = min(i, j) * dt, in units of sigma^2.
def _cov(a, b, dt=1):
    return sum(ai * b.get(j, 0) * min(i, j) * dt
               for i, ai in a.items() for j in b if ai) * 1


def _cov_full(a, b, dt=1):
    return sum(ai * bj * min(i, j) * dt for i, ai in a.items() for j, bj in b.items())


def _add(*terms):
    out = {}
    for coef, d in terms:
        for i, v in d.items():
            out[i] = out.get(i, F(0)) + coef * v
    return {i: v for i, v in out.items() if v != 0}


def trailing_mean(k, w, lag=0):
    """Weights of the trailing k-second mean, right edge lagged by `lag`."""
    hi = w - lag
    return {i: F(1, k) for i in range(hi - k + 1, hi + 1)}


def mu_weights(r, w):
    """The IDEAL forecast E[x_T | full path to t], as a path functional.

    x_T averages the w samples at times t+r-w+1 .. t+r. Those at or before t are
    known; each later one has conditional mean B_t. Total weight is 1.
    """
    if r >= w:
        return {w: F(1)}
    pre = {i: F(1, w) for i in range(r + 1, w + 1)}       # w - r known samples
    return _add((F(1), pre), (F(r, w), {w: F(1)}))


# ------------------------------------------------------------ the two lines --
def k_law(r, convention=None, grid_only=False):
    """LINE 1: Var(x_T - mu_t)/sigma^2, the post-t innovation. Anchor-free.

        r <= w :  r(r+1)(2r+1) / (6 w^2)
        r >= w :  (r - w) + (w+1)(2w+1) / (6w)

    Continuous at r = w (both give (w+1)(2w+1)/(6w)). v1's table mixed this
    discrete branch with the continuous r - 2w/3 above r=w and so jumped 1.25%
    in sigma at the branch point.
    """
    w, c = _conv(convention)
    r = _check_r(r, w, grid_only)
    if isinstance(r, Unavailable):
        return r
    if r <= w:
        return F(r * (r + 1) * (2 * r + 1), 6 * w * w)
    return F(r - w) + F((w + 1) * (2 * w + 1), 6 * w)


def _obs(convention):
    w, c = _conv(convention)
    fast = trailing_mean(c["s_fast"], w, c.get("s_fast_lag", 0))
    slow = trailing_mean(c["s_slow"], w, c.get("s_slow_lag", 0))
    return fast, slow


def alpha_star(r, convention=None):
    """The conditional-projection weight at horizon r.

    Forecast family, translation-invariant (the BM level is not identified):
        P_hat(alpha) = S_slow + alpha*(S_fast - S_slow)
    alpha* = Cov(mu - S_slow, S_fast - S_slow) / Var(S_fast - S_slow).

    NOT a constant of nature -- it is what THIS path model implies. Estimate it
    on the tape; its distance from the estimate is a process diagnostic.
    """
    w, c = _conv(convention)
    r = _check_r(r, w, False)
    if isinstance(r, Unavailable):
        return r
    fast, slow = _obs(convention)
    d = _add((F(1), fast), (F(-1), slow))
    m = _add((F(1), mu_weights(r, w)), (F(-1), slow))
    return _cov_full(m, d, c["dt"]) / _cov_full(d, d, c["dt"])


def anchor_resid_coeff(r, convention=None):
    """LINE 2: the CONDITIONAL variance of the anchor error, /sigma^2.

    Var(mu_t - P_hat | S_fast, S_slow). Because P_hat is a function of the
    conditioning variables, this is ALPHA-INDEPENDENT -- every anchor in the
    family has the same conditional variance and they differ only in bias.
    """
    w, c = _conv(convention)
    a = alpha_star(r, convention)
    if isinstance(a, Unavailable):
        return a
    fast, slow = _obs(convention)
    err = _add((F(1), mu_weights(r, w)), (F(-1), slow),
               (-F(a), fast), (F(a), slow))
    return _cov_full(err, err, c["dt"])


def anchor_bias_coeff(r, alpha, convention=None):
    """Conditional BIAS of anchor `alpha`, as a multiple of the OBSERVABLE
    (S_fast - S_slow): bias_t = anchor_bias_coeff * (S30(t) - S60(t)).

    This is knowable at decision time. It belongs in the numerator of d, never
    in the variance. v1 buried it in the variance -- M2-1.
    """
    a = alpha_star(r, convention)
    if isinstance(a, Unavailable):
        return a
    return F(alpha) - a


def anchor_mse_coeff(r, alpha, convention=None):
    """UNCONDITIONAL MSE of anchor `alpha` = conditional variance + bias^2.

    This is what v1 called a(r) and entered as variance. Reported here only to
    show the decomposition; it is NOT the variance line.
    """
    w, c = _conv(convention)
    v = anchor_resid_coeff(r, convention)
    if isinstance(v, Unavailable):
        return v
    b = anchor_bias_coeff(r, alpha, convention)
    fast, slow = _obs(convention)
    d = _add((F(1), fast), (F(-1), slow))
    return v + b * b * _cov_full(d, d, c["dt"])


def feed_error_var(r, alpha, feed_cov, convention=None):
    """M2-3: propagate S_fast/S_slow measurement error through the horizon
    weights. feed_cov is the 2x2 [[v_ff, c_fs], [c_fs, v_ss]] in sigma^2 units.

    A SCALAR omega_scale cannot represent this: the contribution is
    u^T feed_cov u with u = (alpha, 1-alpha), which varies with the horizon
    weights and is generally not a multiple of the Brownian curve.
    """
    a = alpha if alpha is not None else alpha_star(r, convention)
    if isinstance(a, Unavailable):
        return a
    u = (F(a), F(1) - F(a))
    return (u[0] * u[0] * F(feed_cov[0][0]) + 2 * u[0] * u[1] * F(feed_cov[0][1])
            + u[1] * u[1] * F(feed_cov[1][1]))


# ------------------------------------ the ledger: ONE source of truth (M2-4) --
def ledger(r, sigma2, alpha=None, feed_cov=None, convention=None, grid_only=True):
    """The complete conditional settlement law at horizon r.

    Returns a dict; `total_var` is the ONLY variance any consumer may price
    with, and `settlement_var` below is a thin accessor onto this same dict, so
    the two can never disagree (v1 shipped two public functions that did).

    There is NO nugget argument. v1's `settlement_var` silently added one that
    `ledger` omitted. A variogram nugget may be observation noise, feed noise or
    small-scale process variance; those map differently into conditional
    settlement uncertainty and it cannot be appended as a horizon-constant
    scalar. It returns as a named component only after it has an estimand.
    """
    w, c = _conv(convention)
    rr = _check_r(r, w, grid_only)
    if isinstance(rr, Unavailable):
        return rr
    a = alpha_star(rr, convention) if alpha is None else F(alpha)
    diffusion = F(sigma2) * k_law(rr, convention)
    anchor = F(sigma2) * anchor_resid_coeff(rr, convention)
    feed = (feed_error_var(rr, a, feed_cov, convention) * F(sigma2)
            if feed_cov is not None else F(0))
    bias_c = anchor_bias_coeff(rr, a, convention)
    return dict(
        r=rr, alpha=a, alpha_star=alpha_star(rr, convention),
        diffusion_var=diffusion, anchor_var=anchor, feed_var=feed,
        total_var=diffusion + anchor + feed,
        bias_coeff_on_S30_minus_S60=bias_c,
        bias_is_zero_mean=(bias_c == 0),
        convention=convention or DEFAULT_CONVENTION,
        convention_status=c["status"],
    )


def settlement_var(r, sigma2, alpha=None, feed_cov=None, convention=None,
                   grid_only=True):
    """Sigma(r), from the single ledger. Refuses off-grid/invalid horizons."""
    L = ledger(r, sigma2, alpha, feed_cov, convention, grid_only)
    return L if isinstance(L, Unavailable) else L["total_var"]


def anchor_error_evidence(convention=None):
    """M2-3: NON-ORDERED reference points. None of these is a bound.

    v1 published a floor/ceiling. It was not an ordered bracket: the "floor"
    (the 2/-1 extrapolator's MSE) is beaten by alpha* inside this very model,
    and the Binance residual is a mixture of anchor error, time-varying basis
    and proxy error whose covariance can move it either way. Ordering requires
    assumptions that are neither typed nor tested.
    """
    out = {}
    for name, c in SAMPLING_CONVENTIONS.items():
        w = c["w"]
        out[name] = dict(
            status=c["status"],
            alpha_star=float(alpha_star(w + 1, name)),
            cond_var_at_alpha_star=float(anchor_resid_coeff(w + 1, name)),
            uncond_mse_at_alpha_2=float(anchor_mse_coeff(w + 1, 2, name)),
            is_a_bound=False,
        )
    return out


# ------------------------------------------------------------- selftests -----
def selftest(sigma_btc=1.089):
    w, ok = 60, True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")

    print("kernel (exact rationals, not ranges -- ITER2 correction 4):")
    check("k_law(30) == 1891/720 exactly", k_law(30) == F(1891, 720), str(k_law(30)))
    check("k_law continuous at r=w", k_law(60) == F((w + 1) * (2 * w + 1), 6 * w),
          f"{k_law(60)} = {float(k_law(60)):.4f}")
    check("r>w offset is w - 20.5028 exactly",
          k_law(120) - 120 == F((w + 1) * (2 * w + 1), 6 * w) - w,
          f"{float(k_law(120) - 120):+.4f}")
    for r in HORIZON_GRID:
        assert isinstance(k_law(r), F)

    print("domain refusal (v1 accepted all of these -- M2-4):")
    for bad, why in ((-1, "negative"), (0, "zero"), (30.4, "off-grid float"),
                     (45, "not on HORIZON_GRID")):
        got = settlement_var(bad, 1.0)
        check(f"settlement_var({bad}) refuses ({why})", isinstance(got, Unavailable),
              repr(got)[:58])
    check("k_law(-1) refuses", isinstance(k_law(-1), Unavailable))
    check("refusal is not a float and does not silently arithmetic",
          not isinstance(settlement_var(-1, 1.0), float))
    check("ledger() and settlement_var() cannot disagree (one source)",
          settlement_var(30, 1.0) == ledger(30, 1.0)["total_var"])
    check("no nugget backdoor in the public signature",
          "nugget" not in settlement_var.__code__.co_varnames)

    print("anchor: the M2-1 correction:")
    a_star = alpha_star(w + 1)
    # The exact value is 2700/1801, NOT the 1799/1200 you get by reading the
    # printed 1.499167 back as a decimal -- they differ in the 7th digit. This
    # is why ITER2 correction 4 asked for exact rationals rather than ranges.
    check("alpha* == 2700/1801 exactly, NOT 2", a_star == F(2700, 1801),
          f"{a_star} = {float(a_star):.7f}")
    v = anchor_resid_coeff(w + 1)
    check("conditional variance at alpha* = 8.2590", abs(float(v) - 8.2590) < 1e-3,
          f"{float(v):.4f}")
    check("conditional variance is ALPHA-INDEPENDENT (P_hat is F-measurable)",
          anchor_mse_coeff(w + 1, a_star) == v)
    m2 = anchor_mse_coeff(w + 1, 2)
    check("v1's 9.5139 is uncond MSE at alpha=2, = cond var + bias^2",
          abs(float(m2) - 9.5139) < 1e-3,
          f"{float(m2):.4f} = {float(v):.4f} + {float(m2 - v):.4f}")
    b = anchor_bias_coeff(w + 1, 2)
    fast, slow = _obs(None)
    d = _add((F(1), fast), (F(-1), slow))
    var_d = _cov_full(d, d)
    bias_sd = float(b) * float(var_d) ** 0.5
    check("the buried bias is ~50% of sigma_eff(30) -- not a rounding error",
          0.45 < bias_sd * sigma_btc / (float(settlement_var(30, sigma_btc ** 2)) ** 0.5) < 0.60,
          f"bias sd {bias_sd:.4f} sigma = {bias_sd * sigma_btc:.2f} bps vs "
          f"sigma_eff(30) {float(settlement_var(30, sigma_btc**2))**0.5:.2f} bps")
    check("alpha=2 is NOT zero-mean; alpha* is",
          not ledger(30, 1.0, alpha=2)["bias_is_zero_mean"]
          and ledger(30, 1.0)["bias_is_zero_mean"])
    check("alpha* is horizon-dependent inside the window",
          alpha_star(30) != alpha_star(w + 1),
          f"alpha*(30)={float(alpha_star(30)):.4f} vs alpha*(r>w)={float(a_star):.4f}")

    print("M2-3: nothing here is a bound:")
    ev = anchor_error_evidence()
    check("evidence is labelled non-ordered", all(not e["is_a_bound"] for e in ev.values()))
    check("v1's 'floor' 9.5139 is BEATEN inside the same model by alpha*",
          ev[DEFAULT_CONVENTION]["cond_var_at_alpha_star"]
          < ev[DEFAULT_CONVENTION]["uncond_mse_at_alpha_2"],
          f"{ev[DEFAULT_CONVENTION]['cond_var_at_alpha_star']:.4f}"
          f" < {ev[DEFAULT_CONVENTION]['uncond_mse_at_alpha_2']:.4f}")
    lag = ev["disc1s_lag_s30_1s"]
    check("a 1 s support shift moves alpha* -- semantics are load-bearing (M2-2)",
          abs(lag["alpha_star"] - float(a_star)) > 1e-3,
          f"alpha* {float(a_star):.4f} -> {lag['alpha_star']:.4f}")
    check("every convention is still UNVERIFIED",
          all(e["status"] == "UNVERIFIED" for e in ev.values()))

    print("M2-3: feed error needs a 2x2, not a scalar:")
    fc = [[F(1, 10), F(1, 20)], [F(1, 20), F(1, 10)]]
    f_hi = feed_error_var(w + 1, 2, fc)
    f_st = feed_error_var(w + 1, a_star, fc)
    check("feed contribution depends on alpha (so it is not a scalar rescale)",
          f_hi != f_st, f"{float(f_hi):.5f} vs {float(f_st):.5f}")
    # The covariance term is 2*alpha*(1-alpha)*c, whose SIGN flips with alpha
    # (1-alpha < 0 above alpha=1). So the same feed covariance raises the
    # variance for one anchor and lowers it for another: there is no ordering
    # to exploit, which is the whole of M2-3.
    neg = [[F(1, 10), F(-1, 20)], [F(-1, 20), F(1, 10)]]
    hi_dir = feed_error_var(w + 1, 2, neg) - f_hi
    lo_dir = feed_error_var(w + 1, F(1, 2), neg) - feed_error_var(w + 1, F(1, 2), fc)
    check("flipping the covariance sign moves alpha=2 and alpha=0.5 OPPOSITE ways",
          hi_dir * lo_dir < 0,
          f"alpha=2 {float(hi_dir):+.5f}, alpha=0.5 {float(lo_dir):+.5f}")
    check("feed error enters total_var only when supplied",
          ledger(30, 1.0)["feed_var"] == 0
          and ledger(30, 1.0, feed_cov=fc)["feed_var"] > 0)

    print("\nledger under the CONDITIONAL anchor (BTC sigma=1.089, UNVERIFIED "
          "convention, no feed error):")
    print("   r   alpha*   diffusion    anchor     Sigma  sigma_eff   v1 said")
    v1 = {30: 2.44, 60: 5.97, 120: 10.33, 180: 13.34, 240: 15.78, 270: 16.87}
    for r in HORIZON_GRID:
        L = ledger(r, sigma_btc ** 2)
        print(f" {r:4d} {float(L['alpha_star']):8.4f} {float(L['diffusion_var']):10.3f}"
              f" {float(L['anchor_var']):9.3f} {float(L['total_var']):9.3f}"
              f" {float(L['total_var']) ** 0.5:10.3f} {v1[r]:9.2f}")
    print("  v1's row used the alpha=2 MSE as variance, so it was inflated by the")
    print("  squared bias AND centred wrong. Both columns are fixtures, not")
    print("  pricing inputs: the sampling convention is UNVERIFIED and alpha must")
    print("  be ESTIMATED on the tape, not taken from this model.")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
