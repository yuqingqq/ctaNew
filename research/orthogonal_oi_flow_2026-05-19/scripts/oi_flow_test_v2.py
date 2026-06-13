"""OI/flow portable test — v2 CORRECTED (per ROUND2 results review RE-INITIATE).

Reuses the validated heavy compute of oi_flow_test.py VERBATIM (training,
sleeve construction, per-arm admissible-row pairing — all unchanged) and
fixes ONLY the aggregation, which had the B3-class duplicate-`time` cartesian
join (`a0.join(ar)` on a non-unique index across 5 disjoint groups).

Fixes:
  - run_arm tags each emitted per-group frame with its group id `g`
    (output-only; no training/eval logic touched).
  - Strict WITHIN-GROUP pairing by `g` (each group's `time` is unique).
  - mean-of-per-group portable Sharpe; per-group list (parity with
    ridge_a0_check.json); honest paired per-cycle diff = concat of per-group
    diffs (NO cartesian); honest n_eff = cycles/BLOCK.
  - PLAN-promised parity guards now implemented: per-group level
    block-bootstrap CI on A0 and ARM; LOFO single-group sign-flip on the
    lift; reported for ALL arms (resolves the Ridge-OI positive at parity
    with the bottleneck ridge_a0_check, not via a corrupted CI).
  - Persists per-group net frames so any re-analysis is free (no re-run).
Pre-registered gate UNCHANGED.
"""
from __future__ import annotations
import glob, json, os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "research/portable_alpha_2026-05-19/scripts"))
import phase_ah_sleeve as PA
import R1_baseline_frontier as R1
import R3c_portability_proper as R3c
import ml.research.alpha_vBTC_build_audit_panel as BA
from ml.research.alpha_v4_xs import block_bootstrap_ci
from sklearn.linear_model import Ridge
import importlib.util
_spec = importlib.util.spec_from_file_location("boif", REPO / "scripts/build_btc_oi_features.py")
BOIF = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(BOIF)

OUT = REPO / "research/orthogonal_oi_flow_2026-05-19/results"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
CACHE = REPO / "data/ml/cache"
G, SEED = 5, 20260519
BLK = R1.BLOCK


def _sh(x): return R1._sharpe(np.asarray(x, float))


# ---- helpers reproduced verbatim from oi_flow_test.py ---------------------
def build_oi_panel(panel):
    cached = {os.path.basename(f)[8:-8] for f in glob.glob(str(CACHE / "metrics_*.parquet"))}
    syms = sorted(s for s in panel["symbol"].unique() if s in cached)
    parts = []
    for s in syms:
        m = pd.read_parquet(CACHE / f"metrics_{s}.parquet")
        if len(m) < 50_000: continue
        if m.index.tz is None: m.index = m.index.tz_localize("UTC")
        grid = pd.DatetimeIndex(panel[panel.symbol == s]["open_time"]
                                .sort_values().drop_duplicates())
        f = BOIF.oi_features(m, grid).reset_index().rename(columns={"index": "open_time"})
        f["symbol"] = s
        parts.append(f)
    oip = pd.concat(parts, ignore_index=True)
    oip["open_time"] = pd.to_datetime(oip["open_time"], utc=True)
    return oip, BOIF.FEATS, sorted(oip["symbol"].unique())


def build_flow_panel(panel):
    files = sorted(glob.glob(str(CACHE / "flow_*.parquet")))
    FCOLS = ["signed_volume", "tfi", "aggressor_count_ratio", "avg_trade_size",
             "vwap_dev_bps", "large_trade_volume", "large_trade_count",
             "tfi_smooth", "signed_volume_z", "vpin"]
    parts, syms = [], []
    pan_syms = set(panel["symbol"].unique())
    for fp in files:
        s = os.path.basename(fp)[5:-8]
        if s not in pan_syms: continue
        d = pd.read_parquet(fp)
        cols = [c for c in FCOLS if c in d.columns]
        if not cols: continue
        d = d[cols].sort_index().shift(1)
        d = d.reset_index().rename(columns={d.index.name or "index": "open_time"})
        d.columns = ["open_time"] + [f"flw_{c}" for c in cols]
        d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
        d["symbol"] = s
        parts.append(d); syms.append(s)
    fp_ = pd.concat(parts, ignore_index=True)
    return fp_, [f"flw_{c}" for c in FCOLS], sorted(set(syms))


def leak_guard(panel, cols, tag):
    y = panel["target_A"]; m = y.notna(); rows = []; bad = []
    for c in cols:
        x = panel[c]; g = m & x.notna()
        ic = abs(np.corrcoef(x[g].rank(), y[g].rank())[0, 1]) if g.sum() > 30000 else 0.0
        rows.append((c, round(float(ic), 4)))
        if ic >= 0.10: bad.append((c, ic))
    (OUT / f"_leakguard_{tag}.txt").write_text(
        f"max|rankIC|={max(abs(r[1]) for r in rows):.4f}\n" +
        "\n".join(f"{c} {ic:+.4f}" for c, ic in rows))
    assert not bad, f"LEAK GUARD FAIL {tag}: {bad}"
    return max(abs(r[1]) for r in rows)


def coverage_group_auc(panel, addcols, gmap):
    nanflag = panel[addcols].isna().any(axis=1).astype(int).to_numpy()
    grp = panel["symbol"].map(gmap).to_numpy()
    aucs = []
    for gi in range(G):
        yb = (grp == gi).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb): continue
        order = np.argsort(nanflag)
        r = np.empty(len(nanflag)); r[order] = np.arange(1, len(nanflag) + 1)
        n1 = yb.sum(); n0 = len(yb) - n1
        auc = (r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        aucs.append(abs(auc - 0.5) + 0.5)
    return float(np.mean(aucs)) if aucs else 0.5


def fit_ridge(Xt, yt, Xte):
    mu, sd = np.nanmean(Xt, 0), np.nanstd(Xt, 0) + 1e-9
    Z = np.nan_to_num((Xt - mu) / sd); Zte = np.nan_to_num((Xte - mu) / sd)
    return Ridge(alpha=10.0).fit(Z, yt).predict(Zte)


def run_arm(panel, base_feats, add_feats, model, groups, fwd_resid, sigma, advc,
            listings, folds):
    """VERBATIM heavy compute from oi_flow_test.py; ONLY change: tag each
    emitted per-group frame with its group id `g` (output-only)."""
    feA0, feA = base_feats, base_feats + add_feats
    out = {"A0": [], "ARM": []}
    for gi, hold in enumerate(groups):
        hold = set(hold); tr_s = set(panel["symbol"].unique()) - hold
        panel["_tgd"] = R3c.disjoint_target(panel, tr_s).values
        preds = {"A0": [], "ARM": []}
        for fid in BA.OOS_FOLDS:
            if fid >= len(folds): continue
            tr, ca, te = BA._slice(panel, folds[fid])
            tr = tr[(tr.autocorr_pctile_7d >= BA.THRESHOLD) & tr.symbol.isin(tr_s)]
            ca = ca[(ca.autocorr_pctile_7d >= BA.THRESHOLD) & ca.symbol.isin(tr_s)]
            te = te[te.symbol.isin(hold)].copy()
            def adm(df): return df.dropna(subset=add_feats)
            tr, ca, te = adm(tr), adm(ca), adm(te)
            if len(tr) < 800 or len(ca) < 150 or len(te) < 40: continue
            yt = tr["_tgd"].to_numpy(np.float32); yc = ca["_tgd"].to_numpy(np.float32)
            mt, mc = ~np.isnan(yt), ~np.isnan(yc)
            if mt.sum() < 800 or mc.sum() < 150: continue
            bd = te[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
            bd["exit_time"] = (te["exit_time"].values if "exit_time" in te
                               else te["open_time"] + pd.Timedelta(minutes=R3c.HOR*5))
            bd["fold"] = fid
            for key, feat in (("A0", feA0), ("ARM", feA)):
                Xt = tr[feat].to_numpy(np.float32)[mt]
                Xc = ca[feat].to_numpy(np.float32)[mc]
                Xte = te[feat].to_numpy(np.float32)
                if model == "lgbm":
                    ps = []
                    for s in BA.SEEDS:
                        mdl = BA._train(Xt, yt[mt], Xc, yc[mc], seed=s)
                        ps.append(mdl.predict(Xte, num_iteration=mdl.best_iteration))
                    pred = np.mean(ps, axis=0)
                else:
                    pred = fit_ridge(Xt, yt[mt], Xte)
                d = bd.copy(); d["pred"] = pred
                preds[key].append(d)
        for key in ("A0", "ARM"):
            if not preds[key]: continue
            apd = pd.concat(preds[key], ignore_index=True).sort_values(["open_time", "symbol"])
            apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
            apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
            def elig(b, _h=hold):
                ts = pd.Timestamp(b, unit="ms", tz="UTC") - pd.Timedelta(days=PA.MIN_HISTORY_DAYS)
                return {s for s in _h if listings.get(s) and listings[s] <= ts}
            tt = sorted(apd[apd.fold.isin(PA.OOS_FOLDS)]["open_time"].unique())
            if len(tt) < 40: continue
            u = PA.build_rolling_ic_universe(apd, tt[::R1.HE], PA.TOP_N, elig)
            rec = PA.run_production_protocol_save_sleeves(apd, u)
            df, _, _ = R1.aggregate_capped(rec, fwd_resid, sigma, advc,
                                           cap_frac=1/3, sizing="equal",
                                           cost_mode="flat45")
            t = df[["time", "net_pnl_bps"]].rename(columns={"net_pnl_bps": f"n_{key}"})
            t["g"] = gi                                    # <-- the only change
            out[key].append(t)
    return out


def _bootci(arr, stat):
    try:
        _, lo, hi = block_bootstrap_ci(np.asarray(arr, float), statistic=stat,
                                       block_size=BLK, n_boot=2000)
        return round(float(lo), 3), round(float(hi), 3)
    except Exception:
        return None, None


def aggregate(o):
    """Strict within-group pairing by `g`. No cartesian."""
    a0 = {f["g"].iloc[0]: f for f in o["A0"]}
    ar = {f["g"].iloc[0]: f for f in o["ARM"]}
    common = sorted(set(a0) & set(ar))
    per_g, diffs, a0net, arnet = [], [], [], []
    for g in common:
        j = a0[g][["time", "n_A0"]].merge(ar[g][["time", "n_ARM"]], on="time", how="inner")
        if len(j) < 5: continue
        s0, s1 = _sh(j["n_A0"]), _sh(j["n_ARM"])
        per_g.append({"g": int(g), "A0": round(s0, 3), "ARM": round(s1, 3),
                      "lift": round(s1 - s0, 3), "n": int(len(j))})
        diffs.append((j["n_ARM"] - j["n_A0"]).to_numpy())
        a0net.append(j["n_A0"].to_numpy()); arnet.append(j["n_ARM"].to_numpy())
    if not per_g:
        return {"status": "insufficient"}
    lifts = np.array([p["lift"] for p in per_g])
    mean_lift = float(lifts.mean())
    pooled_diff = np.concatenate(diffs)            # honest: no cartesian
    a0_all, ar_all = np.concatenate(a0net), np.concatenate(arnet)
    n_eff = len(pooled_diff) / BLK
    dlo, dhi = _bootci(pooled_diff, lambda x: float(np.mean(x)))
    a0lo, a0hi = _bootci(a0_all, _sh)
    arlo, arhi = _bootci(ar_all, _sh)
    # LOFO single-group sign-flip on the lift
    flips = []
    for k in range(len(lifts)):
        ml = float(np.delete(lifts, k).mean())
        if np.sign(ml) != np.sign(mean_lift) and abs(mean_lift) > 1e-9:
            flips.append({"drop_g": per_g[k]["g"], "lift_wo": round(ml, 3)})
    mde = 2.49 * pooled_diff.std() / np.sqrt(max(n_eff, 1))
    npos = int((lifts > 0).sum())
    e = {"A0_sharpe_mean_of_group": round(float(np.mean([p["A0"] for p in per_g])), 3),
         "ARM_sharpe_mean_of_group": round(float(np.mean([p["ARM"] for p in per_g])), 3),
         "mean_lift": round(mean_lift, 3),
         "per_group": per_g, "groups_lift_positive": f"{npos}/{len(per_g)}",
         "A0_level_CI": [a0lo, a0hi], "ARM_level_CI": [arlo, arhi],
         "paired_diff_mean_bps": round(float(pooled_diff.mean()), 3),
         "paired_diff_CI_bps": [dlo, dhi],
         "diff_excludes_0": bool((dlo is not None and dlo > 0) or
                                 (dhi is not None and dhi < 0)),
         "lofo_sign_flips": flips,
         "n_eff_honest": round(n_eff, 1),
         "mde_bps_honest": round(float(mde), 3),
         "n_cyc_honest": int(len(pooled_diff))}
    # pre-registered gate (UNCHANGED): PASS iff mean_lift>=+0.5 AND CI excl 0
    # AND >=4/5 groups positive AND no LOFO sign-flip
    if (mean_lift >= 0.5 and e["diff_excludes_0"] and npos >= 4
            and not flips):
        v = "LEVER-REAL"
    elif mean_lift <= 0.2 or not e["diff_excludes_0"]:
        v = "no-portable-lift"
    else:
        v = "ambiguous"
    if abs(pooled_diff.mean()) < mde and not e["diff_excludes_0"]:
        v += " | underpowered (effect-size, not 'exhausted')"
    e["verdict"] = v
    return e


def main():
    t0 = time.time()
    panel = pd.read_parquet(PANEL)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    base = [f for f in BA.WINNER_21 if f in panel.columns and f != "sym_id"]
    oip, oicols, oi_syms = build_oi_panel(panel)
    flp, flcols, fl_syms = build_flow_panel(panel)
    panel = panel.merge(oip, on=["symbol", "open_time"], how="left")
    panel = panel.merge(flp, on=["symbol", "open_time"], how="left")
    rng = np.random.RandomState(SEED)
    syms = sorted(panel["symbol"].unique()); shf = syms.copy(); rng.shuffle(shf)
    groups = [shf[i::G] for i in range(G)]
    gmap = {s: gi for gi, gp in enumerate(groups) for s in gp}
    lk_oi = leak_guard(panel, oicols, "oi"); lk_fl = leak_guard(panel, flcols, "flow")
    auc_oi = coverage_group_auc(panel, oicols, gmap)
    auc_fl = coverage_group_auc(panel, flcols, gmap)
    print(f"OI {len(oi_syms)} flow {len(fl_syms)} | leak {lk_oi:.4f}/{lk_fl:.4f} "
          f"| covAUC {auc_oi:.3f}/{auc_fl:.3f}", flush=True)
    fwd_resid = R3c.build_beta_neutral_fwd()
    sig = pd.read_parquet(R3c.CW)
    sig = sig[[c for c in sig.columns if c.startswith("c_")]].rename(columns=lambda x: x[2:])
    sigma = sig.pct_change().rolling(288, min_periods=48).std().shift(1)
    sigma = sigma.clip(lower=sigma.quantile(0.20), axis=1)
    advc = {s: R1.COST_UNIT for s in syms}
    listings = PA.get_listings(); folds = BA._multi_oos_splits(panel)
    ARMS = {"A1_OI": oicols, "A2_FLOW": flcols, "A3_OIFLOW": oicols + flcols}
    res = {"oi_syms": len(oi_syms), "flow_syms": len(fl_syms),
           "leak_oi": lk_oi, "leak_flow": lk_fl,
           "cov_auc_oi": round(auc_oi, 3), "cov_auc_flow": round(auc_fl, 3),
           "BLOCK": BLK, "arms": {}}
    for aname, addf in ARMS.items():
        for mdl in ("lgbm", "ridge"):
            o = run_arm(panel, base, addf, mdl, groups, fwd_resid, sigma, advc,
                        listings, folds)
            # persist per-group frames (free re-analysis, no re-run)
            for key in ("A0", "ARM"):
                if o[key]:
                    pd.concat(o[key], ignore_index=True).to_parquet(
                        OUT / f"_pergroup_{aname}_{mdl}_{key}.parquet", index=False)
            e = aggregate(o)
            res["arms"][f"{aname}|{mdl}"] = e
            if e.get("status") == "insufficient":
                print(f"  {aname}|{mdl}: insufficient", flush=True); continue
            print(f"  {aname}|{mdl}: A0 {e['A0_sharpe_mean_of_group']:+.3f} "
                  f"ARM {e['ARM_sharpe_mean_of_group']:+.3f} lift "
                  f"{e['mean_lift']:+.3f} grp+ {e['groups_lift_positive']} "
                  f"diffCI {e['paired_diff_CI_bps']} LOFOflip {len(e['lofo_sign_flips'])} "
                  f"-> {e['verdict']}", flush=True)
    res["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "oi_flow_results_corrected.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nCORRECTED done [{res['elapsed_s']}s] -> oi_flow_results_corrected.json",
          flush=True)
    print("OIFLOW_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
