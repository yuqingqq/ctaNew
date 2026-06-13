"""C0-pre: the mandatory decisive precondition (per 3-agent review).

Kills the selection-endogeneity + vol-detector confounds before any C0-C2.
Convex-event label rebuilt on the FULL 51-name panel — EVERY (symbol,cycle)
in OOS folds, entered-or-not, both sides — using the BTC-beta-neutral
realized residual (alpha_vs_btc_realized) as the convex event:
  label_pos = residual fwd in top decile (big idiosyncratic UP)
  label_neg = bottom decile (big DOWN)
  label_abs = |residual| top decile (vol/convex EVENT, direction-agnostic)
Features = the 12 PIT signature cols, strictly prior. Classifier trained
OOS-symbol (5 disjoint groups, seed 20260519) — test rows are names the old
selector NEVER had to have entered. Plus a vol-orthogonalized variant: score
residualized on atr_pct (+ trailing realized vol) before AUC.

Pre-registered decisive read (miss rewrites diagnosis, not gate):
  A. full-panel OOS-symbol AUC(pos) <= 0.58  -> entered-only 0.68 was
     selector-echo; NO portable convexity property -> LINE CLOSED.
  B. |AUC(pos) - AUC(neg)| <= 0.04 AND ~ AUC(abs)  -> pure vol detector,
     not positive-convexity-specific -> LINE CLOSED (cost-only on a
     non-portable base = worthless per profitability review).
  C. vol-orthogonalized OOS-symbol AUC(pos) <= 0.55 -> signal IS just
     volatility -> LINE CLOSED.
  D. ONLY if AUC(pos) >= 0.62 AND AUC(pos)-AUC(neg) >= +0.06 AND
     vol-orthogonalized AUC(pos) >= 0.58 AND placebo ~0.5 -> a genuine,
     portable, vol-independent positive-convexity property -> rebuild plan,
     re-review. Anything else -> LINE CLOSED by measurement (honest).
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
import ml.research.alpha_vBTC_build_audit_panel as BA
from sklearn.linear_model import LogisticRegression

OUT = REPO / "research/convexity_mining_2026-05-19/results"; OUT.mkdir(parents=True, exist_ok=True)
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
SIG = ["atr_pct", "idio_vol_1d_vs_bk", "idio_vol_to_btc_1d", "idio_skew_1d",
       "idio_kurt_1d", "idio_max_abs_12b", "name_idio_share_1d",
       "name_factor_loading_1d", "funding_rate_z_7d", "return_1d",
       "dom_change_288b_vs_bk", "corr_to_btc_1d"]
RESID = "alpha_vs_btc_realized"   # BTC-beta-neutral realized residual (the convex event metric)
VOL = ["atr_pct"]                 # vol axis to orthogonalize against
SEED = 20260519


def _auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    t0 = time.time()
    cols = ["symbol", "open_time", RESID, "target_A"] + SIG
    p = pd.read_parquet(PANEL, columns=cols)
    p = p.dropna(subset=[RESID] + SIG).copy()
    # leak guard: features must not be ~the target
    for c in SIG:
        m = p["target_A"].notna() & p[c].notna()
        if m.sum() < 50000:
            continue
        ic = abs(np.corrcoef(p.loc[m, c].rank(), p.loc[m, "target_A"].rank())[0, 1])
        assert ic < 0.10, f"LEAK {c} rankIC {ic:.3f}"
    # convex-event labels on the FULL panel (per-cycle cross-sectional deciles)
    g = p.groupby("open_time")[RESID]
    hi = g.transform(lambda s: s >= s.quantile(0.90))
    lo = g.transform(lambda s: s <= s.quantile(0.10))
    ab = g.transform(lambda s: s.abs() >= s.abs().quantile(0.90))
    p["y_pos"] = hi.astype(int); p["y_neg"] = lo.astype(int); p["y_abs"] = ab.astype(int)

    syms = sorted(p["symbol"].unique())
    rng = np.random.RandomState(SEED); shf = syms.copy(); rng.shuffle(shf)
    gmap = {s: i % 5 for i, s in enumerate(shf)}
    p["grp"] = p["symbol"].map(gmap)

    X = p[SIG].to_numpy(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    iv = p[VOL].to_numpy(np.float64); iv = (iv - iv.mean(0)) / (iv.std(0) + 1e-9)

    def oos_sym_auc(yname, ortho=False):
        y = p[yname].to_numpy(); aucs = []
        for gg in range(5):
            tr = (p["grp"] != gg).to_numpy(); te = (p["grp"] == gg).to_numpy()
            if y[tr].sum() < 50 or y[te].sum() < 20: continue
            Xtr, Xte = X[tr], X[te]
            if ortho:
                # residualize signature score on vol: fit score, then remove
                # the part explained by atr_pct (OLS) on train, apply to test
                from numpy.linalg import lstsq
                clf = LogisticRegression(max_iter=400, C=0.5).fit(Xtr, y[tr])
                str_ = clf.decision_function(Xtr); ste = clf.decision_function(Xte)
                A = np.column_stack([np.ones(tr.sum()), iv[tr]])
                b, *_ = lstsq(A, str_, rcond=None)
                rte = ste - (b[0] + iv[te] @ b[1:])
                aucs.append(_auc(y[te], rte))
            else:
                clf = LogisticRegression(max_iter=400, C=0.5).fit(Xtr, y[tr])
                aucs.append(_auc(y[te], clf.decision_function(Xte)))
        return float(np.nanmean(aucs)) if aucs else np.nan

    res = {
      "n_rows": int(len(p)), "n_syms": len(syms),
      "auc_pos_full_oos_symbol": round(oos_sym_auc("y_pos"), 4),
      "auc_neg_full_oos_symbol": round(oos_sym_auc("y_neg"), 4),
      "auc_abs_full_oos_symbol": round(oos_sym_auc("y_abs"), 4),
      "auc_pos_vol_orthogonalized": round(oos_sym_auc("y_pos", ortho=True), 4),
      "auc_atr_pct_alone_pos": round(_auc(p["y_pos"], p["atr_pct"]), 4),
      "auc_atr_pct_alone_neg": round(_auc(p["y_neg"], p["atr_pct"]), 4),
    }
    # placebo: shuffle label, must -> ~0.5
    yp = p["y_pos"].to_numpy().copy(); rng.shuffle(yp)
    pa = []
    for gg in range(5):
        tr = (p["grp"] != gg).to_numpy(); te = (p["grp"] == gg).to_numpy()
        c = LogisticRegression(max_iter=300, C=0.5).fit(X[tr], yp[tr])
        pa.append(_auc(yp[te], c.decision_function(X[te])))
    res["auc_placebo"] = round(float(np.nanmean(pa)), 4)

    ap, an, aa = res["auc_pos_full_oos_symbol"], res["auc_neg_full_oos_symbol"], res["auc_abs_full_oos_symbol"]
    ao = res["auc_pos_vol_orthogonalized"]
    if ap <= 0.58:
        v = "A: LINE CLOSED — entered-only 0.68 was selector-echo (full-panel AUC<=0.58)"
    elif abs(ap - an) <= 0.04:
        v = "B: LINE CLOSED — pure volatility detector (AUC pos~=neg), not convexity-specific"
    elif ao <= 0.55:
        v = "C: LINE CLOSED — signal is just volatility (collapses when vol orthogonalized)"
    elif ap >= 0.62 and (ap - an) >= 0.06 and ao >= 0.58 and res["auc_placebo"] <= 0.55:
        v = "D: SURVIVES — genuine portable vol-independent positive-convexity; rebuild plan + re-review"
    else:
        v = "LINE CLOSED — fails decisive bar (ambiguous = conservative close)"
    res["VERDICT"] = v
    res["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "C0pre_decisive.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str), flush=True)
    print("C0PRE_DONE", flush=True)


if __name__ == "__main__":
    main()
