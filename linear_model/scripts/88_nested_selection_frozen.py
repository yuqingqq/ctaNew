"""Step 88: close the LAST loop on the Step-87 sigstable3 PASS.

Step 87 passed: a FROZEN zero-param signed composite (+beta_to_btc_change_5d
+return_1d_sq -dom_btc_z_1d) on 24h hit netSh +2.19, CI [+0.46,+4.52] excl 0,
> placebo p95, NO K3 collapse (in-samp +2.29 / nested-fit +3.21 / frozen
+2.19). The ONLY residual bias: that 3-feature/sign SELECTION used step-86
full-period hindsight; the hindsight-free twogrp_equal had CI crossing 0.

This test removes that bias: the subset AND signs are chosen PER FOLD from
strictly-past data only, then applied frozen (no weight fit) to fold k.

Pool (pre-registered) = the 9 sqbtcrel feats (SQ + BTCREL).
Per-fold k selection rule (pre-registered, parameter-free given the constants):
  past = OOS folds < k  (k==1 -> fold-0 train slice, still strictly past).
  For each pool feature compute its per-cycle IC vs alpha_beta on EACH past
  fold; SELECT it iff its per-fold IC has the SAME sign in ALL past folds
  AND |mean past IC| >= TAU (=0.005). Freeze sign = that consistent sign.
  score(fold k) = signed-equal over the selected/frozen subset (>=1 feat;
  if none qualify, that fold contributes no trades).

PRE-REGISTERED VERDICT (fixed before run). The nested-selected FROZEN
composite is a GENUINELY-VALIDATED edge iff ALL:
  V1 net-Sharpe block-bootstrap CI excludes 0;
  V2 beats matched-basket placebo p95 (150 seeds);
  V3 selection is stable, not churning: the modal selected subset (by
     frozen-sign signature) recurs in >= 6 of the 9 folds;
  V4 signal-dependent: shuffle-score netSh ~ placebo (|.|<placebo p95) AND
     negate-score netSh < 0.
PASS all -> FIRST genuinely-validated linear edge of the investigation
(deeper robustness/forward-test next, still NO backtest). ANY fail ->
the Step-87 sigstable3 PASS was structure-level hindsight selection;
record honestly. Context refs: sigstable3 (hindsight) and twogrp_equal.
"""
from __future__ import annotations
import importlib.util, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

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

OUT = REPO / "linear_model/results/step88_nested_selection"
OUT.mkdir(parents=True, exist_ok=True)
OOS = s76.OOS
POOL = s80b.SQ + s80b.BTCREL                 # pre-registered 9-feature pool
TAU = 0.005
SIGSTABLE3 = {"beta_to_btc_change_5d": +1.0, "return_1d_sq": +1.0,
              "dom_btc_z_1d": -1.0}
N_PLACEBO = 150


def fold_feat_ic(sl: pd.DataFrame, feats: list[str]) -> dict:
    """Per-feature mean per-cycle Spearman IC vs alpha_beta on slice `sl`."""
    out = {}
    for f in feats:
        ics = []
        for _, g in sl.dropna(subset=[f, "alpha_beta"]).groupby("open_time"):
            if len(g) < 5 or g[f].std() <= 1e-12 or g["alpha_beta"].std() <= 1e-12:
                continue
            ics.append(g[f].corr(g["alpha_beta"], method="spearman"))
        out[f] = float(np.mean(ics)) if ics else np.nan
    return out


def select_nested(dec, folds, k):
    """Subset + frozen signs from strictly-past folds only."""
    if k <= 1:
        past_slices = [_slice(dec, folds[0])[0]]          # fold-0 train (past)
    else:
        past_slices = [dec[dec["fold"] == j] for j in range(1, k)]
    per_fold = [fold_feat_ic(s, POOL) for s in past_slices if len(s)]
    if not per_fold:
        return {}
    sel = {}
    for f in POOL:
        vals = [pf[f] for pf in per_fold if pf.get(f) == pf.get(f)]  # drop nan
        if len(vals) < len(per_fold):
            continue
        signs = {np.sign(v) for v in vals if v != 0}
        if len(signs) == 1 and abs(np.mean(vals)) >= TAU:
            sel[f] = float(next(iter(signs)))
    return sel


def score_nested(dec, folds):
    rows, picks = [], []
    for k in OOS:
        if k >= len(folds):
            continue
        sel = select_nested(dec, folds, k)
        picks.append({"fold": k, "n_sel": len(sel),
                      "sig": "|".join(f"{'+' if s>0 else '-'}{f}"
                                      for f, s in sorted(sel.items()))})
        if not sel:
            continue
        feats = list(sel)
        w = np.array([sel[f] for f in feats], float)
        te = dec[dec["fold"] == k].dropna(subset=["alpha_beta"]).copy()
        for t, g in te.groupby("open_time", sort=True):
            if len(g) < 5:
                continue
            y = g["alpha_beta"].to_numpy(float)
            if np.std(y) <= 1e-12:
                continue
            sc = s78._xsz(g, feats) @ w
            if np.std(sc) <= 1e-12:
                continue
            rows.append(pd.DataFrame({"open_time": t, "fold": k,
                                      "symbol": g["symbol"].to_numpy(),
                                      "score": sc, "y": y}))
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
            pd.DataFrame(picks))


def score_fixed(dec, folds, weights: dict):
    rows = []
    feats = [f for f in weights if f in dec.columns]
    w = np.array([weights[f] for f in feats], float)
    for k in OOS:
        if k >= len(folds):
            continue
        te = dec[dec["fold"] == k].dropna(subset=["alpha_beta"]).copy()
        for t, g in te.groupby("open_time", sort=True):
            if len(g) < 5:
                continue
            y = g["alpha_beta"].to_numpy(float)
            if np.std(y) <= 1e-12:
                continue
            sc = s78._xsz(g, feats) @ w
            if np.std(sc) <= 1e-12:
                continue
            rows.append(pd.DataFrame({"open_time": t, "fold": k,
                                      "symbol": g["symbol"].to_numpy(),
                                      "score": sc, "y": y}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def m(df, placebo=False):
    if df.empty:
        return dict(netSh=np.nan, lo=np.nan, hi=np.nan, bps=np.nan,
                    rho=np.nan, fp=0, p95=np.nan)
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
    return dict(netSh=sh, lo=lo, hi=hi, bps=mu, rho=float(rho), fp=fp, p95=p95)


def main():
    print("=" * 100, flush=True)
    print("  STEP 88: nested per-fold feature/sign selection — close the "
          "hindsight loop", flush=True)
    print(f"  PRE-REG PASS: V1 CI-excl-0 V2 >placebo-p95 V3 modal subset >=6/9 "
          f"V4 shuffle~placebo & negate<0  (pool={len(POOL)} TAU={TAU})",
          flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    _, _, dec, _, folds, _ = s84.build_dec()
    print(f"  dec: {dec['symbol'].nunique()} syms, "
          f"{dec['open_time'].nunique()} cyc", flush=True)

    nd, picks = score_nested(dec, folds)
    picks.to_csv(OUT / "fold_selections.csv", index=False)
    print("\n  per-fold nested selection:", flush=True)
    for r in picks.itertuples():
        print(f"    fold {r.fold}: n={r.n_sel}  {r.sig}", flush=True)
    sig_counts = picks[picks.n_sel > 0]["sig"].value_counts()
    modal_n = int(sig_counts.iloc[0]) if len(sig_counts) else 0
    modal_sig = sig_counts.index[0] if len(sig_counts) else "(none)"

    mn = m(nd, placebo=True)
    sig3 = m(score_fixed(dec, folds, SIGSTABLE3))
    tge = m(score_fixed(dec, folds,
            {f: 1.0 for f in s80b.SQ} | {f: 1.0 for f in s80b.BTCREL}))
    # V4 signal-dependence on the nested score
    if not nd.empty:
        shuf = m(nd.assign(score=nd.groupby("open_time")["score"].transform(
            lambda s: np.random.default_rng(0).permutation(s.values))))
        neg = m(nd.assign(score=-nd["score"]))
    else:
        shuf = neg = dict(netSh=np.nan)

    print(f"\n  nested_selected  netSh={mn['netSh']:+.2f} "
          f"CI[{mn['lo']:+.2f},{mn['hi']:+.2f}] net={mn['bps']:+.1f}bps "
          f"ρ={mn['rho']:+.3f} f+={mn['fp']}/9 placebo_p95={mn['p95']:+.2f}",
          flush=True)
    print(f"  modal subset recurs {modal_n}/9 : {modal_sig}", flush=True)
    print(f"  shuffle netSh={shuf['netSh']:+.2f}  negate netSh={neg['netSh']:+.2f}",
          flush=True)
    print(f"  [ctx] sigstable3(hindsight)={sig3['netSh']:+.2f}  "
          f"twogrp_equal={tge['netSh']:+.2f}", flush=True)

    V1 = bool(mn["lo"] > 0)
    V2 = bool(mn["netSh"] > mn["p95"])
    V3 = bool(modal_n >= 6)
    V4 = bool(abs(shuf.get("netSh", 9)) < mn["p95"] and neg.get("netSh", 9) < 0)
    allp = V1 and V2 and V3 and V4
    rows = [dict(variant="nested_selected", **mn),
            dict(variant="sigstable3_hindsight", **sig3),
            dict(variant="twogrp_equal", **tge)]
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print("\n" + "=" * 100, flush=True)
    print(f"  GATES: V1 CI-excl-0={V1} | V2 >placebo-p95={V2} | "
          f"V3 modal>=6/9={V3} ({modal_n}/9) | V4 sig-dependent={V4}",
          flush=True)
    if allp:
        v = (f"PASS — nested per-fold selection (no hindsight) FROZEN composite "
             f"netSh {mn['netSh']:+.2f} CI[{mn['lo']:+.2f},{mn['hi']:+.2f}] "
             f"> placebo p95 {mn['p95']:+.2f}, stable modal subset {modal_n}/9, "
             f"signal-dependent. The Step-87 PASS was NOT hindsight — this is "
             f"the FIRST genuinely-validated linear edge of the investigation. "
             f"Next = deeper robustness + forward/paper plan, still NO "
             f"backtest. Production LGBM unaffected.")
    else:
        fails = [n for n, ok in [("V1", V1), ("V2", V2), ("V3", V3),
                 ("V4", V4)] if not ok]
        v = (f"FAIL ({','.join(fails)}) — nested-selected frozen composite "
             f"netSh {mn['netSh']:+.2f} CI[{mn['lo']:+.2f},{mn['hi']:+.2f}], "
             f"placebo p95 {mn['p95']:+.2f}, modal subset {modal_n}/9. The "
             f"Step-87 sigstable3 PASS leaned on step-86 full-period structure "
             f"selection; under honest nested selection it does not hold up. "
             f"Milder than K3 (weights frozen, no collapse) but still a "
             f"structure-level selection bias. 24h linear edge not validated. "
             f"Production LGBM unaffected.")
    print(f"  VERDICT: {v}", flush=True)
    pd.DataFrame([{"all_pass": allp, "verdict": v}]).to_csv(
        OUT / "verdict.csv", index=False)
    print(f"\nSaved under {OUT}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
