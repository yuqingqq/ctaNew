"""Step 82a: fold-level bucket-stability of the squared / non-monotone CARRIER.

The user's bucket->bps calibration (layer 2) is only well-posed if the
score->bucket->payoff map is STABLE across folds. Step 77 measured this for
the *shrunk composite* (found unstable, sign-flipping). It was NEVER measured
for the actual lead: the squared / U-shape (non-monotone) carrier that 80a
found is the lone essential group and 80b's best non-baseline
(ridge only_squared K3 +10.11 rho +0.552 8/9; sqbtcrel +6.97 7/9).

This is the PRE-GATE for Step 82b (the calibration). Carriers, on volaug/hl42
(same engine as 80a/80b), ridge_xsz + nnls_oriented:
  only_squared  = 5 *_sq features
  sqbtcrel      = squared(5) + btc_rel(4)

Per fold: decile->bps profile, per-fold decile monotonicity rho, which
buckets are profitable, cross-fold profile persistence.

PRE-REGISTERED PRE-GATE (fixed before run) — carrier is "calibration-ready" iff
  (a) per-fold decile rho has the SAME SIGN in >= 7/9 folds, AND
  (b) each of the full-sample top-3 buckets is positive in >= 6/9 folds, AND
  (c) mean consecutive-fold decile-profile Spearman >= +0.30.
PASS -> Step 82b nested-OOS calibration is well-posed (proceed).
FAIL -> the calibration would inherit instability; record as an informative
        negative measured on the RIGHT carrier (not the Step-77 proxy).
No backtest here.
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
s79 = _imp("s79", "linear_model/scripts/79_broader_universe_attrib.py")
s80b = _imp("s80b", "linear_model/scripts/80b_vol_interaction_payoff.py")
from ml.research.alpha_v4_xs_1d import _multi_oos_splits

OUT = REPO / "linear_model/results/step82a_carrier_bucket_stability"
OUT.mkdir(parents=True, exist_ok=True)
OOS = s76.OOS
SIGN_FOLDS, TOP3_POS_FOLDS, PROFILE_RHO = 7, 6, 0.30
CARRIERS = {"only_squared": s80b.SQ, "sqbtcrel": s80b.SQ + s80b.BTCREL}
MODELS = ["ridge_xsz", "nnls_oriented"]


def fold_decile_profile(df: pd.DataFrame, fold: int) -> np.ndarray:
    """Mean fwd-bps per score-decile within one fold (cycle-equal-weighted)."""
    g0 = df[df["fold"] == fold]
    per = []
    for _, g in g0.groupby("open_time"):
        if len(g) < 20:
            continue
        r = g["score"].rank(method="first")
        try:
            b = pd.qcut(r, 10, labels=False, duplicates="drop")
        except ValueError:
            continue
        per.append(g.assign(b=b).groupby("b")["y"].mean() * 1e4)
    if not per:
        return np.full(10, np.nan)
    M = pd.concat(per, axis=1).T.mean()
    return M.reindex(range(10)).to_numpy(float)


def analyze(df: pd.DataFrame) -> dict:
    folds = sorted(f for f in df["fold"].unique() if f in OOS)
    prof = {f: fold_decile_profile(df, f) for f in folds}
    P = np.vstack([prof[f] for f in folds])                 # folds x 10
    idx = np.arange(10)
    rhos = []
    for row in P:
        ok = ~np.isnan(row)
        rhos.append(pd.Series(row[ok]).corr(pd.Series(idx[ok]),
                    method="spearman") if ok.sum() >= 3 else np.nan)
    rhos = np.array(rhos, float)
    valid = rhos[~np.isnan(rhos)]
    sign_pos = int((valid > 0).sum())
    sign_neg = int((valid < 0).sum())
    same_sign = max(sign_pos, sign_neg)
    dom = +1 if sign_pos >= sign_neg else -1
    full = np.nanmean(P, axis=0)
    # top-3 buckets in the dominant direction
    order = np.argsort(-full) if dom > 0 else np.argsort(full)
    top3 = order[:3]
    top3_pos_folds = [int(np.nansum((P[:, b] * dom) > 0)) for b in top3]
    # consecutive-fold profile persistence
    pers = []
    for i in range(len(folds) - 1):
        a, b = P[i], P[i + 1]
        m = ~np.isnan(a) & ~np.isnan(b)
        if m.sum() >= 3:
            pers.append(pd.Series(a[m]).corr(pd.Series(b[m]),
                        method="spearman"))
    prof_rho = float(np.nanmean(pers)) if pers else np.nan
    gate = (same_sign >= SIGN_FOLDS
            and all(t >= TOP3_POS_FOLDS for t in top3_pos_folds)
            and (prof_rho == prof_rho and prof_rho >= PROFILE_RHO))
    return dict(n_folds=len(folds), dom_sign=dom, same_sign=same_sign,
                rho_per_fold=np.round(rhos, 3).tolist(),
                full_profile=np.round(full, 2).tolist(),
                top3_buckets=top3.tolist(),
                top3_pos_folds=top3_pos_folds,
                profile_persist_rho=prof_rho, gate_pass=bool(gate))


def main():
    print("=" * 100, flush=True)
    print("  STEP 82a: carrier bucket-stability pre-gate (volaug/hl42)", flush=True)
    print(f"  PRE-GATE: same-sign rho >={SIGN_FOLDS}/9 AND each top-3 bucket "
          f">={TOP3_POS_FOLDS}/9 pos AND profile-persist rho >=+{PROFILE_RHO}",
          flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()

    raw = pd.read_parquet(s80b.VOLAUG)
    raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
    hl = pd.read_csv(s80b.HL_MAP)
    folds = _multi_oos_splits(raw[raw["symbol"] != "BTCUSDT"])
    dec, fc, _ = s79.build_universe(raw, hl, folds, "hl42")
    dec = dec.merge(raw[["symbol", "open_time"] + s80b.VOL],
                    on=["symbol", "open_time"], how="left")
    for c in s80b.VOL:
        dec[c] = dec[c].astype("float32").fillna(0.0)
    s80b.add_interactions(dec)
    print(f"  hl42: {dec['symbol'].nunique()} syms, "
          f"{dec['open_time'].nunique()} cycles", flush=True)

    rows = []
    for cname, sub in CARRIERS.items():
        for mdl in MODELS:
            df = s80b.score(dec, sub, folds, mdl)
            if df.empty:
                print(f"  {cname}/{mdl}: no scores", flush=True)
                continue
            a = analyze(df)
            rows.append({"carrier": cname, "model": mdl, **a})
            print(f"\n  [{cname} / {mdl}]  dom={'+' if a['dom_sign']>0 else '-'}"
                  f"  same-sign={a['same_sign']}/9  "
                  f"profile-persist rho={a['profile_persist_rho']:+.3f}",
                  flush=True)
            print(f"    per-fold decile rho: {a['rho_per_fold']}", flush=True)
            print(f"    full decile bps    : {a['full_profile']}", flush=True)
            print(f"    top3 buckets {a['top3_buckets']} pos-folds "
                  f"{a['top3_pos_folds']} (need each >={TOP3_POS_FOLDS})",
                  flush=True)
            print(f"    PRE-GATE: {'PASS' if a['gate_pass'] else 'FAIL'}",
                  flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "summary.csv", index=False)
    anypass = bool(out["gate_pass"].any()) if len(out) else False
    print("\n" + "=" * 100, flush=True)
    if anypass:
        w = out[out["gate_pass"]].iloc[0]
        v = (f"PASS — {w['carrier']}/{w['model']} bucket map is fold-stable "
             f"(same-sign {w['same_sign']}/9, persist "
             f"{w['profile_persist_rho']:+.2f}). Step 82b nested-OOS "
             f"bucket->bps calibration on this carrier is WELL-POSED — "
             f"proceed (pre-registered threshold, random-bucket placebo).")
    else:
        bestp = out["profile_persist_rho"].max() if len(out) else np.nan
        v = (f"FAIL — no carrier bucket map is fold-stable (best "
             f"profile-persist rho {bestp:+.2f} < +{PROFILE_RHO}). The "
             f"bucket calibration would inherit the instability — measured "
             f"now on the RIGHT carrier (squared/non-monotone), not the "
             f"Step-77 shrunk-composite proxy. Step 82b NOT well-posed; "
             f"the non-monotone structure is real but its bucket map does "
             f"not persist OOS. Pivot to proper-24h (Step-76 bug) as the "
             f"remaining genuinely-untested direction.")
    print(f"  VERDICT: {v}", flush=True)
    pd.DataFrame([{"any_pass": anypass, "verdict": v}]).to_csv(
        OUT / "verdict.csv", index=False)
    print(f"\nSaved under {OUT}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
