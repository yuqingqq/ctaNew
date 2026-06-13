"""Step 87: is the 24h sqbtcrel edge a FROZEN parameter-free structure that
generalizes, or only an in-sample RidgeCV fit (the K2/K3/W23 trap)?

Step 86 showed +2.29/+2.79 from IN-SAMPLE RidgeCV (alpha chosen within each
fold). That is exactly the signature this project has been burned by >=5x:
great in-fold CV, strongly negative under equal-weight (signed_equal -3.52),
historically fails honest nested-OOS. This test converts "the model can learn
it" into a falsifiable test by FREEZING the structure (no in-sample fitting)
and re-imposing the Step-84 rigor.

Structures (all on 24h volaug, hl42; net-of-cost annualized Sharpe + block
bootstrap CI + matched-basket placebo, identical to Step-84):
  twogrp_equal     : FROZEN zero-param  = mean(xsz SQ) + mean(xsz BTCREL)
  sigstable3       : FROZEN zero-param  = signed-equal over the step-86
                     sign-stable AND payoff-additive core:
                     +beta_to_btc_change_5d  (9/0 stable, LOO -14.9)
                     +return_1d_sq           (9/0 stable, LOO  -7.3)
                     -dom_btc_z_1d           (0/9 stable, LOO  -5.1)
  twogrp_ridge_nest: 2 group-mean inputs, RidgeCV alpha+coefs fit on
                     folds<k ONLY, applied to fold k (true nested-OOS)
  ref_insample     : sqbtcrel RidgeCV alpha-in-fold (= Step-86 ~+2.3, the
                     inflated reference)
  null_signed_eq   : signed_equal over the 9 sqbtcrel feats (Step-84 G4 null)

PRE-REGISTERED VERDICT (fixed before run):
  A frozen structure is a GENUINE edge iff
    (a) its net-Sharpe block-bootstrap CI excludes 0, AND
    (b) it beats the matched-basket placebo p95, AND
    (c) it is NOT an in-sample artifact: a frozen (zero-param) structure
        passes (a)+(b) on its own — i.e. we do NOT rely on the in-fold fit;
        and if ref_insample >> twogrp_ridge_nest >= frozen, that ordering is
        recorded as the K3 collapse signature.
  PASS  -> first genuine linear edge; deeper robustness/audit next, no
           backtest. FAIL -> 24h edge is an in-sample-fit artifact; the
           linear line closes honestly on direct evidence.
"""
from __future__ import annotations
import importlib.util, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


s76 = _imp("s76", "linear_model/scripts/76_minimal_orientation.py")
s78 = _imp("s78", "linear_model/scripts/78_nnls_poscoef_payoff.py")
s79 = _imp("s79", "linear_model/scripts/79_broader_universe_attrib.py")
s80b = _imp("s80b", "linear_model/scripts/80b_vol_interaction_payoff.py")
s84 = _imp("s84", "linear_model/scripts/84_proper_24h_rigor.py")
from ml.research.alpha_v4_xs_1d import _slice

OUT = REPO / "linear_model/results/step87_frozen_structure"
OUT.mkdir(parents=True, exist_ok=True)
OOS = s76.OOS
SQ, BTCREL = s80b.SQ, s80b.BTCREL
REF9 = SQ + BTCREL
ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0, 1e3, 1e4]
SIGSTABLE3 = {"beta_to_btc_change_5d": +1.0, "return_1d_sq": +1.0,
              "dom_btc_z_1d": -1.0}
N_PLACEBO = 150


def _grp_inputs(g):
    """Per-cycle two group-mean inputs: mean xsz(SQ), mean xsz(BTCREL)."""
    return np.column_stack([s78._xsz(g, [c for c in SQ if c in g.columns]).mean(1),
                            s78._xsz(g, [c for c in BTCREL if c in g.columns]).mean(1)])


def score(dec, folds, kind):
    rows = []
    s3 = [f for f in SIGSTABLE3 if f in dec.columns]
    s3w = np.array([SIGSTABLE3[f] for f in s3], float)
    for k in OOS:
        if k >= len(folds):
            continue
        tr = _slice(dec, folds[k])[0].dropna(subset=["alpha_beta"])
        if len(tr) < 500:
            continue
        coef = None
        if kind == "ref_insample":
            X, y = s84_train(tr, REF9)
            coef = RidgeCV(alphas=ALPHAS, scoring="r2",
                           fit_intercept=False).fit(X, y).coef_
        elif kind == "null_signed_eq":
            w = s76.fit_weights(tr, REF9, "alpha_beta")
            coef = np.array([np.sign(w[f]) or 1.0 for f in REF9], float)
        elif kind == "twogrp_ridge_nest":
            # nested: fit on folds < k only
            gx, gy = [], []
            for kk in range(1, k):
                trk = _slice(dec, folds[kk])[0].dropna(subset=["target_z"])
                for _, g in trk.groupby("open_time"):
                    if len(g) < 5:
                        continue
                    yy = g["target_z"].to_numpy(float)
                    if np.std(yy) > 1e-12:
                        gx.append(_grp_inputs(g))
                        gy.append(yy)
            if not gx:
                continue
            gcoef = RidgeCV(alphas=ALPHAS, scoring="r2",
                            fit_intercept=False).fit(np.vstack(gx),
                                                     np.concatenate(gy)).coef_
        te = dec[dec["fold"] == k].dropna(subset=["alpha_beta"]).copy()
        for t, g in te.groupby("open_time", sort=True):
            if len(g) < 5:
                continue
            y = g["alpha_beta"].to_numpy(float)
            if np.std(y) <= 1e-12:
                continue
            if kind == "twogrp_equal":
                sc = _grp_inputs(g) @ np.array([1.0, 1.0])
            elif kind == "sigstable3":
                sc = s78._xsz(g, s3) @ s3w
            elif kind == "twogrp_ridge_nest":
                sc = _grp_inputs(g) @ gcoef
            else:
                sc = s78._xsz(g, REF9) @ coef
            if np.std(sc) <= 1e-12:
                continue
            rows.append(pd.DataFrame({"open_time": t, "fold": k,
                                      "symbol": g["symbol"].to_numpy(),
                                      "score": sc, "y": y}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def s84_train(tr, feats):
    xs, ys = [], []
    for _, g in tr.dropna(subset=["target_z"]).groupby("open_time"):
        if len(g) < 5:
            continue
        z = s78._xsz(g, feats)
        y = g["target_z"].to_numpy(float)
        if np.isfinite(y).all() and np.std(y) > 1e-12:
            xs.append(z)
            ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


def metrics(df, placebo=False):
    if df.empty:
        return dict(net_sharpe=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    net_bps=np.nan, rho=np.nan, folds_pos=0, pl_p95=np.nan)
    net = s84.ls_series(df)
    sh, lo, hi, mu = s84.sharpe_ci(net["net"])
    d24 = df[df["open_time"].isin(set(sorted(df["open_time"].unique())[::s84.H24]))]
    rho = s79.payoff(d24)["decile_rho"] if not d24.empty else np.nan
    fp = sum(1 for _, g in net.groupby("fold") if g["net"].mean() > 0)
    p95 = np.nan
    if placebo:
        pls = [s84.sharpe_ci(s84.ls_series(df, np.random.default_rng(s))["net"])[0]
               for s in range(N_PLACEBO)]
        p95 = float(np.nanpercentile(pls, 95))
    return dict(net_sharpe=sh, ci_lo=lo, ci_hi=hi, net_bps=mu,
                rho=float(rho), folds_pos=fp, pl_p95=p95)


def main():
    print("=" * 100, flush=True)
    print("  STEP 87: frozen-structure nested-OOS test of the 24h sqbtcrel edge",
          flush=True)
    print("  PRE-REG PASS iff a FROZEN (zero-param) structure: CI excl 0 AND "
          ">placebo-p95; in-sample>>nested>=frozen = K3 collapse", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    _, _, dec, _, folds, _ = s84.build_dec()
    print(f"  dec: {dec['symbol'].nunique()} syms, "
          f"{dec['open_time'].nunique()} cyc", flush=True)

    rows = []
    for kind, frozen in [("twogrp_equal", True), ("sigstable3", True),
                         ("twogrp_ridge_nest", False),
                         ("ref_insample", False), ("null_signed_eq", False)]:
        df = score(dec, folds, kind)
        m = metrics(df, placebo=frozen)        # placebo only for the frozen ones
        m["structure"] = kind
        m["frozen"] = frozen
        rows.append(m)
        pls = (f" placebo_p95={m['pl_p95']:+.2f}" if frozen else "")
        print(f"  {kind:20s} {'FROZEN' if frozen else 'fitted'} "
              f"netSh={m['net_sharpe']:+.2f} CI[{m['ci_lo']:+.2f},"
              f"{m['ci_hi']:+.2f}] net={m['net_bps']:+.1f}bps ρ={m['rho']:+.3f} "
              f"f+={m['folds_pos']}/9{pls}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "summary.csv", index=False)
    R = {r["structure"]: r for r in rows}
    frozen_pass = [k for k in ("twogrp_equal", "sigstable3")
                   if R[k]["ci_lo"] > 0 and R[k]["net_sharpe"] > R[k]["pl_p95"]]
    insamp = R["ref_insample"]["net_sharpe"]
    nested = R["twogrp_ridge_nest"]["net_sharpe"]
    best_frozen = max(R["twogrp_equal"]["net_sharpe"],
                      R["sigstable3"]["net_sharpe"])
    k3_collapse = (insamp - nested > 1.0) and (nested - best_frozen < 0.5)
    print("\n" + "=" * 100, flush=True)
    if frozen_pass:
        v = (f"PASS — frozen zero-param structure(s) {frozen_pass} clear: "
             f"CI-excl-0 AND > placebo-p95 with NO in-sample fitting. First "
             f"genuine linear edge of the investigation — the 24h sqbtcrel "
             f"signal survives as a fixed parameter-free structure. Deeper "
             f"robustness/audit next, still NO backtest. "
             f"(in-sample {insamp:+.2f} / nested {nested:+.2f} / "
             f"best-frozen {best_frozen:+.2f})")
    else:
        tag = " K3-COLLAPSE confirmed (in-sample>>nested~=frozen)" if k3_collapse else ""
        v = (f"FAIL — no frozen zero-param structure clears CI-excl-0 + "
             f"placebo-p95. in-sample {insamp:+.2f} >> nested {nested:+.2f} "
             f">= best-frozen {best_frozen:+.2f}; signed_eq null "
             f"{R['null_signed_eq']['net_sharpe']:+.2f}.{tag} The 24h edge is "
             f"an IN-SAMPLE RidgeCV fit, not a generalizable structure — the "
             f"K2/K3/W23 pattern, now confirmed directly on the simplified "
             f"model. Linear beta-residual line closes honestly: no edge that "
             f"survives without in-sample weight fitting. Production LGBM "
             f"unaffected.")
    print(f"  VERDICT: {v}", flush=True)
    pd.DataFrame([{"frozen_pass": bool(frozen_pass), "k3_collapse": k3_collapse,
                   "verdict": v}]).to_csv(OUT / "verdict.csv", index=False)
    print(f"\nSaved under {OUT}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
