#!/usr/bin/env python3
"""Deterministic kernels and the closed variance ledger for the 5-min binaries.

Phase 0A step 3 of SIGMA_PLAN_REVIEW.md: derive and unit-test the kernels under
the ACTUAL feed sampling convention, including continuity at r = w, BEFORE any
estimator exists. Nothing here is fitted; every number is algebra plus a
Monte-Carlo selftest. No outcome data is touched.

UNIT SPACE (S1, frozen here and nowhere else)
--------------------------------------------
    x_t = 1e4 * (S_t - S_ref) / S_ref            model coordinate, bps
    S_ref = the window's strike X_0              per-window, known at t0
    sigma                                        bps / sqrt(second)
    Sigma(r), omega_P^2, nugget                  bps^2
    d = (E_t[x_T] - x_0) / sqrt(Sigma(r))        dimensionless

NORMALISED ARITHMETIC RETURNS, NOT LOG RETURNS -- see SIGMA_PLAN.md "R1
disagreement". The settlement mark is an ARITHMETIC mean of prices, so every
identity in this file (the TWAP kernel, the nowcast, the anchor decomposition)
is exact in price space and only approximate in log space.

CONVENTION (frozen): a w-second TWAP at time T is the mean of the w one-second
samples P_{T-w+1} .. P_T. The feed publishes at ~1 Hz (p50 955 ms), so w = 60
means 60 samples, not a continuous integral. Every kernel below is the discrete
one; the continuous form is kept only to report the error it would introduce.

    python3 sigma_kernels.py --selftest
"""
import numpy as np

W_DECLARED = 60          # seconds, EXP-M6, never fitted
DT = 1                   # feed sampling interval, seconds
GRID = (30, 60, 120, 180, 240, 270)   # the decision horizons r = T - t


# ---------------------------------------------------------------- kernels ----
def k_law(r, w=W_DECLARED):
    """Var_t[X_T] / sigma^2 for the DISCRETE trailing-mean settlement mark.

    Derivation. X_T = (1/w) sum_{j=1..w} P_{T-w+j}. Given F_t with r = T - t,
    the sample at T-w+j carries m_j = max(0, r-w+j) seconds of future
    innovation, and Var[sum_j B_{m_j}] = sum_{j,k} min(m_j, m_k). Evaluating:

        r <= w :  r(r+1)(2r+1) / (6 w^2)
        r >= w :  (r - w) + (w+1)(2w+1) / (6w)

    The two branches AGREE at r = w by construction -- both give
    (w+1)(2w+1)/(6w). The continuous law's r - 2w/3 does not: at w=60 the
    discrete constant is 20.5028 s against 20 s, so the r > w branch is
    r - 39.4972 and not r - 40 (SHOULD-FIX 1: the review's "roughly +0.5 s").
    """
    r = float(r)
    if r <= w:
        return r * (r + 1) * (2 * r + 1) / (6.0 * w * w)
    return (r - w) + (w + 1) * (2 * w + 1) / (6.0 * w)


def k_law_continuous(r, w=W_DECLARED):
    """The continuous-integral kernel, kept ONLY to report its error."""
    r = float(r)
    return r ** 3 / (3.0 * w * w) if r <= w else r - 2.0 * w / 3.0


def _cov_quadratic_form(weights):
    """Var[sum_i c_i B_i] / sigma^2 for BM sampled on the 1 s grid, B_0 = 0.

    weights maps grid index i (seconds since the start of the lookback) -> c_i.
    Cov(B_i, B_k) = min(i, k) * DT.
    """
    idx = np.array(sorted(weights), dtype=float)
    c = np.array([weights[int(i)] for i in idx], dtype=float)
    cov = np.minimum.outer(idx, idx) * DT
    return float(c @ cov @ c)


# ------------------------------------------------------- the anchor ledger ----
def _twap_weights(k, w=W_DECLARED):
    """Weights on the lookback grid [t-w .. t] for the trailing k-second TWAP."""
    return {w - k + j: 1.0 / k for j in range(1, k + 1)}


def anchor_error_coeff(r, w=W_DECLARED, s_fast=30, s_slow=60):
    """a(r) with  Var[E_hat_t[X_T] - E_t[X_T]] = a(r) * sigma^2.

    E_t[X_T] needs latent spot P_t (and, strictly inside the window, the
    trailing (w-r)-second TWAP). We observe neither; both are reconstructed
    from the S30/S60 pair:

        slope   = (S30 - S60) / (s_slow - s_fast) * 2      per second
        P_hat   = 2*S30 - S60                              spot nowcast
        S_k_hat = P_hat - (k/2) * slope                    any shorter TWAP

    The reconstruction error is ONE linear functional of the same latent path,
    so its variance is exact -- no independence assumption is needed BETWEEN the
    two reconstructed pieces (they are correlated; the plan's (r/w)^2 * omega_P^2
    silently assumed only one of them existed).

    At r = 30 the trailing half IS S30, observed, so a(30) = (r/w)^2 * a_spot
    exactly; at r >= w the whole mark is future and a(r) = a_spot, undamped.
    Those are the only two in-window points on the current GRID, which is why
    the plan's simpler form happened to be right there and nowhere else.
    """
    r = float(r)
    fast, slow = _twap_weights(s_fast, w), _twap_weights(s_slow, w)

    def lin(a, b, ca, cb):                       # ca*a + cb*b on the grid
        out = {i: 0.0 for i in range(w + 1)}
        for i, v in a.items():
            out[i] += ca * v
        for i, v in b.items():
            out[i] += cb * v
        return out

    p_hat = lin(fast, slow, 2.0, -1.0)           # 2*S30 - S60
    slope = lin(fast, slow, 2.0 / (slow_gap := (s_slow - s_fast)), -2.0 / slow_gap)

    def s_hat(k):                                # P_hat - (k/2)*slope
        return {i: p_hat[i] - (k / 2.0) * slope[i] for i in range(w + 1)}

    spot_true = {i: (1.0 if i == w else 0.0) for i in range(w + 1)}

    if r >= w:
        est, true = p_hat, spot_true
    else:
        k = int(round(w - r))                    # trailing part already fixed
        true_k = _twap_weights(k, w) if k > 0 else {}
        est = {i: (r / w) * p_hat[i] + ((w - r) / w) * s_hat(k)[i] for i in range(w + 1)}
        true = {i: (r / w) * spot_true[i]
                   + ((w - r) / w) * true_k.get(i, 0.0) for i in range(w + 1)}

    err = {i: est[i] - true[i] for i in range(w + 1)}
    return _cov_quadratic_form(err)


def omega_p_coeff(w=W_DECLARED, s_fast=30, s_slow=60):
    """omega_P^2 / sigma^2 -- the nowcast error the model itself implies.

    This is a FLOOR, not a measurement: it is what P_hat = 2*S30 - S60 costs
    with perfect, synchronous, noiseless feeds, purely from extrapolating. Real
    omega_P adds feed asynchrony, Chainlink aggregation and deviation-threshold
    staleness on top. Continuous-time value is 10 (i.e. sqrt(10) = 3.162 sigma).
    """
    return anchor_error_coeff(w + 1, w, s_fast, s_slow)


def legacy_anchor_coeff(w=W_DECLARED):
    """Same object for the OLD S60 anchor, which v1-v3 omitted entirely.

    Continuous value 20, i.e. sqrt(20) = 4.472 sigma. This term is the
    mechanical account of the k = 1.42 puzzle (SIGMA_PLAN.md section 7 source 2):
    an omitted variance line that outcome-MLE had to cover by inflating sigma.
    """
    err = _twap_weights(60, w).copy()
    err[w] = err.get(w, 0.0) - 1.0               # S60 - P_t
    return _cov_quadratic_form(err)


# ----------------------------------------------------------- the estimand ----
def settlement_var(r, sigma2, w=W_DECLARED, omega_scale=1.0, nugget=0.0):
    """Sigma(r) = Var_t[X_T - E_hat_t[X_T]], the CLOSED two-line ledger, bps^2.

    line 1  diffusion       sigma2 * k_law(r)           post-t innovation
    line 2  anchor error    sigma2 * a(r) * omega_scale pre-t reconstruction

    The two are independent because the first is a post-t innovation and the
    second a functional of the path up to t -- a Brownian-model claim, asserted
    here and checked in the selftest, NOT a type-system fact (S3).

    omega_scale >= 1 brackets the real anchor error: 1.0 is the model-implied
    floor above, and the empirical S30/S60-vs-Binance residual (basis
    contaminated) is the ceiling. Do NOT add sigma_perp or kappa(r) on top --
    R-ONCE, and this programme has double-counted variance three times.
    """
    return sigma2 * k_law(r, w) + sigma2 * anchor_error_coeff(r, w) * omega_scale + nugget


def ledger(r, sigma, w=W_DECLARED, omega_scale=1.0):
    """Per-horizon breakdown, for reporting rather than pricing."""
    s2 = sigma ** 2
    diff = s2 * k_law(r, w)
    anch = s2 * anchor_error_coeff(r, w) * omega_scale
    return dict(r=r, diffusion=diff, anchor=anch, total=diff + anch,
                sigma_eff=np.sqrt(diff + anch), anchor_frac=anch / (diff + anch))


# ------------------------------------------------------------- selftests ----
def selftest(sigma_btc=1.089):
    w, ok = W_DECLARED, True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")

    print("kernel:")
    lo, hi = k_law(w - 1e-9), k_law(w + 1e-9)
    check("k_law continuous at r = w", abs(lo - hi) < 1e-6,
          f"{lo:.6f} vs {hi:.6f}")
    check("branches meet at (w+1)(2w+1)/6w", abs(k_law(w) - (w + 1) * (2 * w + 1) / (6 * w)) < 1e-9,
          f"{k_law(w):.4f}")
    check("the continuous law is ALSO self-consistent at r = w (both give w/3)",
          abs(k_law_continuous(w - 1e-9) - k_law_continuous(w + 1e-9)) < 1e-6,
          f"{k_law_continuous(w):.4f}")
    mixed = k_law(w) - k_law_continuous(w)      # plan v1: discrete in, continuous out
    check("MIXING them (plan v1 section 3) jumps at r = w -- the real defect",
          abs(mixed) > 0.4, f"gap {mixed:+.4f} s of variance"
          f" = {100 * (np.sqrt(k_law(w) / k_law_continuous(w)) - 1):+.2f}% in sigma")
    check("r > w offset is w - 20.5028, not 2w/3 = 40",
          abs((k_law(120) - 120) + 39.4972) < 1e-3, f"{k_law(120) - 120:+.4f}")

    print("anchor ledger:")
    a_spot = omega_p_coeff()
    check("omega_P^2/sigma^2 near the continuous 10", 9.0 < a_spot < 10.0,
          f"{a_spot:.4f}  (omega_P = {np.sqrt(a_spot):.4f} sigma"
          f" = {sigma_btc * np.sqrt(a_spot):.2f} bps at BTC sigma)")
    a_old = legacy_anchor_coeff()
    check("old S60 anchor near the continuous 20", 19.0 < a_old < 20.0,
          f"{a_old:.4f}  ({np.sqrt(a_old):.4f} sigma)")
    check("nowcast halves the anchor variance", abs(a_spot / a_old - 0.5) < 0.02,
          f"ratio {a_spot / a_old:.4f}")
    check("r = 30 is the (r/w)^2-damped spot error (S30 is observed there)",
          abs(anchor_error_coeff(30) - 0.25 * a_spot) < 1e-9,
          f"{anchor_error_coeff(30):.4f} vs {0.25 * a_spot:.4f}")
    check("r > w enters UNDAMPED", abs(anchor_error_coeff(180) - a_spot) < 1e-9)
    interior = anchor_error_coeff(45)
    check("interior r needs the S_k reconstruction too, so a(r) > (r/w)^2*a_spot",
          interior > (45 / w) ** 2 * a_spot + 1e-6,
          f"a(45) = {interior:.4f} vs plan's damped form {(45 / w) ** 2 * a_spot:.4f}")

    print("monte carlo (independent check of the algebra):")
    rng = np.random.default_rng(7)
    n = 400_000
    B = np.concatenate([np.zeros((n, 1)),
                        np.cumsum(rng.normal(0, 1.0, (n, w)), axis=1)], axis=1)
    P, S60, S30 = B[:, w], B[:, 1:61].mean(axis=1), B[:, 31:61].mean(axis=1)
    Phat = 2 * S30 - S60
    mc = np.var(Phat - P)
    check("MC agrees with the closed form for omega_P", abs(mc / a_spot - 1) < 0.02,
          f"MC {mc:.4f} vs exact {a_spot:.4f}")
    fut = rng.normal(0, np.sqrt(180.0), n)       # post-t innovation, r = 180
    corr = np.corrcoef(Phat - P, fut)[0, 1]
    check("anchor error is independent of the future innovation", abs(corr) < 0.01,
          f"corr {corr:+.4f}")

    print("\nconsequence for D2 -- the predicted c(30) breach, recomputed:")
    realised = 2.6                               # bps, the figure D2 argued from
    bare = sigma_btc * np.sqrt(k_law(30))
    full = np.sqrt(settlement_var(30, sigma_btc ** 2))
    print(f"  diffusion only : sigma_eff(30) = {bare:.3f} bps -> c(30) = "
          f"{(realised / bare) ** 2:.2f}   (D2 said ~2.1, BREACH)")
    print(f"  closed ledger  : sigma_eff(30) = {full:.3f} bps -> c(30) = "
          f"{(realised / full) ** 2:.2f}   (band [0.80, 1.25])")
    print("  D2's breach was computed against an incomplete ledger. This does not")
    print("  prove c(r) is in band -- 2.6 bps is itself provisional -- it removes")
    print("  the prediction. Measure c(r) after the ledger, per S6.")

    print("\nledger by horizon (BTC sigma = 1.089 bps/sqrt(s), omega floor):")
    print("   r   diffusion    anchor     Sigma  sigma_eff   anchor%  plan v1")
    plan_v1 = {30: 1.77, 60: 4.87, 120: 9.74, 180: 12.88, 240: 15.40, 270: 16.52}
    for r in GRID:
        L = ledger(r, sigma_btc)
        print(f" {r:4d} {L['diffusion']:10.3f} {L['anchor']:9.3f} {L['total']:9.3f}"
              f" {L['sigma_eff']:10.3f} {100 * L['anchor_frac']:8.1f}% {plan_v1[r]:8.2f}")
    print("  anchor share RISES as expiry approaches (anchor/diffusion = 3*a/k ~ 1/r),")
    print("  so the (r/w)^2 damping does NOT make the nowcast free late in the")
    print("  window -- it is ~4% of variance at r=270 and ~48% at r=30.")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest() if "--selftest" in sys.argv else selftest())
