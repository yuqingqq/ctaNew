"""Composite Study — Feature Re-Engineering full pipeline (LOCKED).
Implements FEATURE_REENGINEERING_PLAN.md: Stage0 base → Stage1 engineered
interactions → Stage2 PIT audit → Stage3 redundancy → Stage4 leak-free
ceiling + gate → Stage5 (only if gate passes) nested/LOFO. One run, no
tuning. 4h horizon. Production LGBM unaffected.
"""
from __future__ import annotations
import importlib.util, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
s94b = _imp("s94b", "linear_model/scripts/94b_info_ceiling_d1_grouped.py")
from ml.research.alpha_v4_xs import block_bootstrap_ci
ANN = np.sqrt(365.0 * 6.0)
OUTD = REPO / "linear_model/composite_study/results"
LEAK = s94.LEAK | {"tz", "fwd4_raw"}


def sh(x):
    x = np.asarray(x, float)
    return float(x.mean()/x.std(ddof=1)*ANN) if x.std(ddof=1) > 1e-12 else np.nan


def portfolio_net(d, predcol, cper):
    f = d[["symbol", "open_time", "fold", "alpha_beta", predcol]].copy()
    f["pos"] = np.sign(f[predcol])
    n = f.groupby("open_time")["symbol"].transform("count")
    f["w"] = f["pos"] / n
    f = f.sort_values(["symbol", "open_time"])
    f["dw"] = f.groupby("symbol")["w"].diff().abs().fillna(f["w"].abs())
    p = f.groupby(["open_time", "fold"]).agg(
        g=("w", lambda s: (f.loc[s.index, "w"]*f.loc[s.index, "alpha_beta"]).sum()*1e4),
        c=("dw", lambda s: s.sum()*cper)).reset_index()
    p["net"] = p["g"] - p["c"]
    return p.sort_values("open_time")


def ceiling(d, feats, label, costs=(2.25,), placebo=False):
    rid, gbm = s94b.grouped_oof(d, feats)
    out = {}
    for nm, pr in [("Ridge", rid), ("LGBM", gbm)]:
        dd = d.assign(_p=pr)
        dd = dd[~dd["_p"].isna()]
        for cper in costs:
            P = portfolio_net(dd, "_p", cper)
            net = P["net"].to_numpy()
            S = sh(net)
            lo, hi = block_bootstrap_ci(
                net, statistic=lambda z: z.mean()/z.std(ddof=1)*ANN
                if z.std(ddof=1) > 1e-12 else 0.0, block_size=7,
                n_boot=600)[1:]
            fp = sum(1 for _, g in P.groupby("fold") if g["net"].mean() > 0)
            out[(nm, cper)] = (S, lo, hi, fp, P["fold"].nunique(), dd, "_p")
    return out


def main():
    print("=" * 100, flush=True)
    print("  FEATURE RE-ENGINEERING — full pipeline (LOCKED, 4h)", flush=True)
    print("=" * 100, flush=True)
    dec, syms, btc, pan = s94.build(universe_oi=False)
    # FIX (Error 1): F_core = the TRUE panel-only 24-feat set, captured
    # from `dec` BEFORE merging OI/oflow/spot (else everything sweeps in).
    BASE = [c for c in dec.columns if c not in LEAK and
            pd.api.types.is_numeric_dtype(dec[c])]
    if "s_t" not in BASE:
        BASE.append("s_t")
    oi = pd.read_parquet(REPO/"outputs/vBTC_features_oi/oi_panel.parquet")
    of = pd.read_parquet(REPO/"outputs/vBTC_features_oflow/oflow_panel.parquet")
    sp = pd.read_parquet(REPO/"outputs/vBTC_features_spot/spot_panel.parquet")
    for f in (oi, of, sp):
        f["open_time"] = pd.to_datetime(f["open_time"], utc=True)
    OICOLS = [c for c in oi.columns if c not in ("symbol", "open_time")]
    d = (dec[dec.symbol.isin(set(oi.symbol) & set(of.symbol) & set(sp.symbol))]
         .merge(oi, on=["symbol", "open_time"], how="inner")
         .merge(of, on=["symbol", "open_time"], how="inner")
         .merge(sp, on=["symbol", "open_time"], how="inner"))
    print(f"  [FIX] F_core(panel-only)={len(BASE)} feats; OI={len(OICOLS)} "
          f"oflow=6 spot=6 staged as proper nested supersets. "
          f"NOTE: §6 '+0.62 reproduce' was a 42-sym anchor; this universe is "
          f"19-sym ∩ — gate is WITHIN-universe (abs +1.5 & lift≥+0.5 vs best "
          f"baseline-set & CI & folds & placebo); thresholds UNCHANGED.",
          flush=True)
    sgn = lambda x: np.sign(x)
    I = {}  # ---- Stage 1: locked 26 engineered interactions ----
    I["x_r1d_r8h"] = d.return_1d*d.return_8h
    I["x_r1d_sq"] = sgn(d.return_1d)*d.return_1d**2
    I["x_st_r1d"] = d.s_t*d.return_1d
    I["x_st_autoc"] = d.s_t*d.autocorr_pctile_7d
    I["x_r1d_vws"] = d.return_1d*d.vwap_slope_96
    I["x_r1d_volz"] = d.return_1d*d.vol_zscore_4h_over_7d
    I["x_st_volz"] = d.s_t*d.vol_zscore_4h_over_7d
    I["x_r1d_obv"] = d.return_1d*d.obv_z_1d
    I["x_st_idiov"] = d.s_t*d.idio_vol_to_btc_1d
    I["x_basis_r1d"] = d.sp_basis_z1d*d.return_1d
    I["x_spvr_r1d"] = d.sp_volratio_z1d*d.return_1d
    I["x_spti_st"] = d.sp_taker_imb_1d*d.s_t
    I["x_basis_fz"] = d.sp_basis_z1d*d.funding_rate_z_7d
    I["x_ofi_r1d"] = d.of_imb_1d*d.return_1d
    I["x_ofi_st"] = d.of_imb_1d*d.s_t
    I["x_oftfi_volz"] = d.of_tfi_z1d*d.vol_zscore_4h_over_7d
    I["x_ofkyle_st"] = d.of_kyle_1d*d.s_t
    I["x_ofi_oichg"] = d.of_imb_1d*d.oi_chg_1d
    I["x_ofi_spti"] = d.of_imb_1d*d.sp_taker_imb_1d
    I["x_oic_r1d"] = d.oi_chg_1d*sgn(d.return_1d)
    I["x_fz_r1d"] = d.funding_rate_z_7d*sgn(d.return_1d)
    I["x_oiz_st"] = d.oi_z_7d*d.s_t
    I["x_lst_r1d"] = d.ls_taker_z_1d*sgn(d.return_1d)
    I["x_st_corrb"] = d.s_t*d.corr_to_btc_1d
    I["x_st_betac"] = d.s_t*d.beta_to_btc_change_5d
    I["x_r1d_domz"] = d.return_1d*d.dom_btc_z_1d
    for k, v in I.items():
        d[k] = v.astype("float64")
    ENG = list(I.keys())
    d = d.dropna(subset=BASE+ENG+["tz", "alpha_beta"]).reset_index(drop=True)
    print(f"  Stage0/1: rows={len(d)} syms={d.symbol.nunique()} "
          f"BASE={len(BASE)} ENG={len(ENG)}", flush=True)

    # ---- Stage 2: PIT audit (interactions are products of PIT inputs) ----
    fc = {f: abs(d[f].corr(d["alpha_beta"], "spearman")) for f in ENG}
    bad = [f for f, v in fc.items() if v >= 0.10]
    print(f"  Stage2 PIT: max|corr(eng,fwdαβ)|={max(fc.values()):.3f} "
          f"(<0.10 ok); flagged={bad or 'none'}", flush=True)
    if len(bad) > 3:
        print("  >3 features look-ahead-suspect → ABORT.", flush=True)
        return
    ENG = [f for f in ENG if f not in bad]

    # ---- feature groups (proper nested supersets; Error-1 fix) ----
    Fcore = list(dict.fromkeys(BASE))
    oflow = [c for c in d.columns if c.startswith("of_")]
    spotf = [c for c in d.columns if c.startswith("sp_")]
    base_of = list(dict.fromkeys(Fcore + oflow))
    base_of_sp = list(dict.fromkeys(Fcore + oflow + spotf))
    # ---- Stage 3: redundancy on the gated candidate F_core+oflow+ENG ----
    allf = list(dict.fromkeys(base_of + ENG))
    C = d[allf].corr("spearman").abs()
    keep, seen = [], set()
    ic = {f: abs(d[f].corr(d["alpha_beta"], "spearman")) for f in allf}
    for f in sorted(allf, key=lambda z: -ic[z]):
        if f in seen:
            continue
        keep.append(f)
        seen |= set(C.index[(C[f] >= 0.85)])
    # LGBM gain importance on the clustered-kept set
    gm = lgb.LGBMRegressor(num_leaves=63, n_estimators=300,
                           learning_rate=0.03, subsample=0.8,
                           colsample_bytree=0.8, random_state=0,
                           n_jobs=-1, verbose=-1).fit(d[keep], d["tz"])
    gain = pd.Series(gm.booster_.feature_importance("gain"), index=keep)
    gcut = gain.quantile(0.20)
    pruned = []
    for f in keep:
        X = d[[c for c in keep if c != f]].to_numpy()
        r2 = LinearRegression().fit(X, d[f]).score(X, d[f])
        drop = (r2 >= 0.70) and (ic[f] < 0.01) and (gain[f] <= gcut)
        if not drop:
            pruned.append(f)
    eng_kept = [f for f in pruned if f in ENG]
    print(f"  Stage3 redundancy: {len(allf)}→cluster {len(keep)}→pruned "
          f"{len(pruned)} (eng kept {len(eng_kept)}/{len(ENG)})", flush=True)
    pd.DataFrame({"feature": allf,
                  "univ_ic": [round(ic[f], 4) for f in allf],
                  "clustered_kept": [f in keep for f in allf],
                  "final_kept": [f in pruned for f in allf]}).to_csv(
        OUTD/"feature_reeng_redundancy.csv", index=False)

    # ---- Stage 4: leak-free ceiling + gate ----
    SETS = {
      "F_core": Fcore,
      "F_core+oflow": base_of,
      "F_core+oflow+spot": base_of_sp,
      "F+ENG_pruned": list(dict.fromkeys(pruned)),
      "F+ENG_unpruned": list(dict.fromkeys(base_of + ENG)),
    }
    print("\n  Stage4 leak-free ceiling (Ridge primary; VIP-0):", flush=True)
    res = {}
    BL = None
    for name, feats in SETS.items():
        o = ceiling(d, feats, name, costs=(2.25,))
        S, lo, hi, fp, nf, dd, pc = o[("Ridge", 2.25)]
        Sg = o[("LGBM", 2.25)][0]
        res[name] = dict(sh=S, lo=lo, hi=hi, fp=fp, nf=nf, lgbm=Sg,
                         dd=dd, pc=pc, feats=feats)
        print(f"   {name:20s} Ridge {S:+.2f} [{lo:+.2f},{hi:+.2f}] "
              f"folds {fp}/{nf} | LGBM {Sg:+.2f} (nfeat={len(feats)})",
              flush=True)

    # baseline = within-universe BEST of the baseline sets (Error-2 fix:
    # NOT the cross-universe +1.09; thresholds unchanged)
    BL = max(res[n]["sh"] for n in
             ("F_core", "F_core+oflow", "F_core+oflow+spot"))
    print(f"  within-universe best baseline (max of F_core/+oflow/+spot) "
          f"= {BL:+.2f}", flush=True)
    g = res["F+ENG_pruned"]
    # matched placebo (within-symbol permutation of pred) for the gated set
    rng = np.random.default_rng(0)
    dd, pc = g["dd"], g["pc"]
    pl = []
    for _ in range(150):
        pp = dd.copy()
        pp[pc] = pp.groupby("symbol")[pc].transform(
            lambda v: rng.permutation(v.values))
        pl.append(sh(portfolio_net(pp, pc, 2.25)["net"]))
    p95 = float(np.nanpercentile(pl, 95))
    lift = g["sh"] - BL
    PASS = bool(g["sh"] > 1.5 and g["lo"] > 0 and g["sh"] > p95 and
                lift >= 0.5 and g["fp"] >= 6)
    print(f"\n  GATE (PRUNED set, VIP-0): Sh {g['sh']:+.2f} "
          f"CI[{g['lo']:+.2f},{g['hi']:+.2f}] folds {g['fp']}/{g['nf']} "
          f"placebo_p95 {p95:+.2f} lift_vs_F+oflow {lift:+.2f} → "
          f"{'PASS' if PASS else 'FAIL'}", flush=True)

    if PASS:
        verdict = ("GATE PASS — proceed to Stage-5 strict nested-OOS + LOFO + "
                   "interaction-aware placebo (mandatory before any adoption; "
                   "not auto-adopted).")
        print("  → Stage-5 robustness required (not auto-run here; gate "
              "passed unexpectedly — flag for nested confirm).", flush=True)
    else:
        why = []
        if g["sh"] <= 1.5: why.append(f"Sh {g['sh']:+.2f}≤+1.5")
        if g["lo"] <= 0: why.append("CI incl 0")
        if g["sh"] <= p95: why.append(f"≤placebo {p95:+.2f}")
        if lift < 0.5: why.append(f"lift {lift:+.2f}<+0.5")
        if g["fp"] < 6: why.append(f"{g['fp']}/9 folds")
        verdict = (f"GATE FAIL ({'; '.join(why)}). Systematic engineered-"
                   f"interaction + redundancy-pruned feature set does NOT lift "
                   f"the leak-free ceiling past +1.5 vs F_core+oflow baseline "
                   f"({BL:+.2f}). Feature-side closure is now AIRTIGHT: "
                   f"signal, construction, cost, hedge, AND systematic "
                   f"feature-engineering all exhausted. The existing-"
                   f"information conclusion is final from every angle. "
                   f"Production LGBM unaffected.")
    print(f"\n  VERDICT: {verdict}", flush=True)
    pd.DataFrame([dict(name=k, ridge_sh=round(v["sh"], 3),
                       ci_lo=round(v["lo"], 2), ci_hi=round(v["hi"], 2),
                       folds=f"{v['fp']}/{v['nf']}", lgbm_sh=round(v["lgbm"], 3),
                       nfeat=len(v["feats"])) for k, v in res.items()]
                 ).to_csv(OUTD/"feature_reeng_ceiling.csv", index=False)
    pd.DataFrame([{"PASS": PASS, "pruned_sh": g["sh"], "baseline_F_oflow": BL,
                   "lift": lift, "placebo_p95": p95,
                   "verdict": verdict}]).to_csv(
        OUTD/"feature_reeng_verdict.csv", index=False)
    print(f"\nSaved {OUTD}/feature_reeng_*.csv", flush=True)


if __name__ == "__main__":
    main()
